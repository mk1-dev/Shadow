"""Rebuild receiver evidence and labels using existing nuScenes/flow/BEV caches."""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadow_occ.annotation.data import digest
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--positive-fusion", choices=["best", "consensus", "intersection"], default="consensus")
    parser.add_argument("--negative-policy", choices=["endpoints", "endpoints+lit-receivers"], default="endpoints+lit-receivers")
    parser.add_argument("--split-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if source == output or output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output root, distinct from the source cache")
    output.mkdir(parents=True, exist_ok=True)
    shared_protocol_path = source/"train/bev_flow/bev_flow_protocol.json"
    if not shared_protocol_path.is_file():
        raise FileNotFoundError(shared_protocol_path)
    shared_protocol = json.loads(shared_protocol_path.read_text(encoding="utf-8"))
    config = shared_protocol["config"]
    provenance = {"shared_source_bev_protocol": {
        "path": str(shared_protocol_path), "sha256": digest(shared_protocol_path)}, "splits": {}}

    stop = Event()
    active_processes = {}
    process_lock = Lock()

    def run(script, *values, owner=None):
        if stop.is_set():
            raise RuntimeError("Rebuild stopped after a split worker failed")
        command = [sys.executable, str(ROOT/"scripts"/script), *map(str, values)]
        print("RUN " + " ".join(command), flush=True)
        process = subprocess.Popen(command, cwd=ROOT)
        with process_lock:
            if stop.is_set():
                process.terminate()
            else:
                active_processes[owner] = process
        try:
            returncode = process.wait()
        finally:
            with process_lock:
                if active_processes.get(owner) is process:
                    active_processes.pop(owner)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)

    def terminate_active_processes():
        stop.set()
        with process_lock:
            processes = list(active_processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def rebuild_split(split):
        origin, destination = source/split, output/split
        destination.mkdir()
        # Shared read-only inputs keep the rebuilt root compatible with the ablation runner.
        for name in ("pairs", "flow", "mature_bev"):
            path = origin/name
            if not path.is_dir():
                raise FileNotFoundError(path)
            (destination/name).symlink_to(path, target_is_directory=True)
        provenance["splits"][split] = {
            "pair_manifest_sha256": digest(origin/"pairs/pairs.jsonl"),
            "motion_protocol_sha256": digest(origin/"flow/analysis_protocol.json")}
        run("analyze_bev_flow_discrepancy.py", "--pairs", destination/"pairs/pairs.jsonl",
            "--analysis-root", destination/"flow", "--feature-root", destination/"mature_bev",
            "--minimum-bev-cosine", shared_protocol.get("minimum_bev_cosine", 0.3),
            "--extent-m", config["extent_m"], "--resolution-m", config["resolution_m"],
            "--light-threshold", config["light_threshold_L"], "--omit-receiver-features",
            "--output-root", destination/"bev_flow", owner=split)
        run("build_physics_shadow_labels.py", "--pairs", destination/"pairs/pairs.jsonl",
            "--analysis-root", destination/"flow", "--bev-flow-root", destination/"bev_flow",
            "--decision-policy", "factorized-v3", "--existence-mode", "static-dynamic",
            "--positive-fusion", args.positive_fusion, "--export-weak-labels",
            "--output-root", destination/"labels", owner=split)
        run("build_bev_flow_annotation_manifest.py", "--pairs", destination/"pairs/pairs.jsonl",
            "--labels-root", destination/"labels", "--negative-policy", args.negative_policy,
            "--output-root", destination/"teacher", owner=split)
    executor = ThreadPoolExecutor(max_workers=args.split_workers, thread_name_prefix="shadow-split")
    try:
        futures = {executor.submit(rebuild_split, split): split for split in ("train", "val")}
        for future in as_completed(futures):
            split = futures[future]
            future.result()
            print(f"COMPLETED split={split}", flush=True)
    except BaseException:
        terminate_active_processes()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    run("build_shadow_annotation_dataset.py", "--manifest", output/"train/teacher/canonical.jsonl",
        output/"val/teacher/canonical.jsonl", "--output-root", output/"data", "--seed", args.seed)
    run("check_shadow_annotation_data.py", "--data-root", output/"data")
    run("build_joint_shadow_annotation_dataset.py", "--region-data-root", output/"data", "--output-root", output/"joint")
    run("check_joint_shadow_annotation_data.py", "--data-root", output/"joint", "--require-both-classes")
    write_json(output/"rebuild_protocol.json", {**vars(args), "sources": provenance,
        "policy": "static/dynamic existence separate from motion-based ownership; no encoder or flow recomputation"})


if __name__ == "__main__":
    main()
