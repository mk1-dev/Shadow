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
    parser.add_argument("--reuse-analysis", action="store_true",
                        help="Reuse complete split analysis and retry only missing label stages")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--receiver-competition", choices=["on", "off", "score-only"], default="on")
    parser.add_argument("--audit-only", action="store_true", help="Compare explanations without exporting training data")
    parser.add_argument("--control-shift-m", type=float, default=3.0)
    parser.add_argument("--minimum-projection-gain", type=float, default=0.05)
    parser.add_argument("--minimum-explanation-margin", type=float, default=0.05)
    parser.add_argument("--appearance-prior-weight", type=float, default=0.2)
    parser.add_argument("--shadow-baseline-manifest")
    args = parser.parse_args()
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if source == output or output.exists() and any(output.iterdir()) and not args.reuse_analysis:
        raise FileExistsError("Use a new output root, distinct from the source cache")
    output.mkdir(parents=True, exist_ok=True)
    shared_protocol_path = source/"train/bev_flow/bev_flow_protocol.json"
    if not shared_protocol_path.is_file():
        raise FileNotFoundError(shared_protocol_path)
    shared_protocol = json.loads(shared_protocol_path.read_text(encoding="utf-8"))
    args.shadow_baseline_manifest = args.shadow_baseline_manifest or shared_protocol.get("shadow_baseline_manifest")
    baseline_args = ["--shadow-baseline-manifest", args.shadow_baseline_manifest] if args.shadow_baseline_manifest else []
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
        destination.mkdir(exist_ok=args.reuse_analysis)
        # Shared read-only inputs keep the rebuilt root compatible with the ablation runner.
        for name in ("pairs", "flow", "mature_bev"):
            path = origin/name
            if not path.is_dir():
                raise FileNotFoundError(path)
            linked = destination/name
            if not linked.exists():
                linked.symlink_to(path, target_is_directory=True)
        provenance["splits"][split] = {
            "pair_manifest_sha256": digest(origin/"pairs/pairs.jsonl"),
            "motion_protocol_sha256": digest(origin/"flow/analysis_protocol.json")}
        analysis = destination/"bev_flow"
        if args.reuse_analysis:
            for name in ("bev_flow_protocol.json", "windows.jsonl", "summary.json"):
                if not (analysis/name).is_file():
                    raise FileNotFoundError(analysis/name)
            print(f"REUSING analysis split={split}", flush=True)
        else:
            run("analyze_bev_flow_discrepancy.py", "--pairs", destination/"pairs/pairs.jsonl",
                "--analysis-root", destination/"flow", "--feature-root", destination/"mature_bev",
                "--minimum-bev-cosine", shared_protocol.get("minimum_bev_cosine", 0.3),
                "--extent-m", config["extent_m"], "--resolution-m", config["resolution_m"],
                "--light-threshold", config["light_threshold_L"], "--omit-receiver-features",
                "--output-root", analysis, *baseline_args, owner=split)
        def complete_labels(path):
            return all((path/name).is_file()
                for name in ("canonical.jsonl", "label_protocol.json", "summary.json"))

        def complete_teacher(path):
            return all((path/name).is_file() for name in ("canonical.jsonl", "protocol.json"))

        labels = destination/"labels"
        if not complete_labels(labels) and args.reuse_analysis and complete_labels(destination/"labels_retry"):
            labels = destination/"labels_retry"
        if not complete_labels(labels):
            if args.reuse_analysis:
                labels = destination/"labels_retry"
                if labels.exists() and any(labels.iterdir()):
                    raise FileExistsError(f"Retry output is incomplete and not empty: {labels}")
            run("build_physics_shadow_labels.py", "--pairs", destination/"pairs/pairs.jsonl",
                "--analysis-root", destination/"flow", "--bev-flow-root", analysis,
                "--decision-policy", "factorized-v3", "--existence-mode", "static-dynamic",
                "--positive-fusion", args.positive_fusion, "--export-weak-labels",
                "--receiver-competition", args.receiver_competition,
                "--control-shift-m", args.control_shift_m,
                "--minimum-projection-gain", args.minimum_projection_gain,
                "--minimum-explanation-margin", args.minimum_explanation_margin,
                "--appearance-prior-weight", args.appearance_prior_weight,
                "--output-root", labels, owner=split)
        else:
            print(f"REUSING labels split={split}", flush=True)
        label_protocol = json.loads((labels/"label_protocol.json").read_text(encoding="utf-8"))
        if label_protocol.get("receiver_competition", "off") != args.receiver_competition:
            raise ValueError("Cached label competition mode differs; choose a new output root")
        for key in ("control_shift_m", "minimum_projection_gain", "minimum_explanation_margin", "appearance_prior_weight"):
            if args.receiver_competition != "off" and label_protocol.get(key) != getattr(args, key):
                raise ValueError(f"Cached label setting differs: {key}; choose a new output root")
        if args.receiver_competition != "off" and not (destination/"competition/summary.json").exists():
            run("analyze_receiver_competition.py", "--labels-root", labels,
                "--output-root", destination/"competition", owner=split)
        if args.audit_only:
            return labels/"canonical.jsonl"
        teacher = destination/"teacher"
        if not complete_teacher(teacher) and args.reuse_analysis and complete_teacher(destination/"teacher_retry"):
            teacher = destination/"teacher_retry"
        if not complete_teacher(teacher):
            if args.reuse_analysis:
                teacher = destination/"teacher_retry"
                if teacher.exists() and any(teacher.iterdir()):
                    raise FileExistsError(f"Retry output is incomplete and not empty: {teacher}")
            run("build_bev_flow_annotation_manifest.py", "--pairs", destination/"pairs/pairs.jsonl",
                "--labels-root", labels, "--negative-policy", args.negative_policy,
                "--output-root", teacher, owner=split)
        else:
            print(f"REUSING teacher split={split}", flush=True)
        return teacher/"canonical.jsonl"
    executor = ThreadPoolExecutor(max_workers=args.split_workers, thread_name_prefix="shadow-split")
    try:
        futures = {executor.submit(rebuild_split, split): split for split in ("train", "val")}
        teachers = {}
        for future in as_completed(futures):
            split = futures[future]
            teachers[split] = future.result()
            print(f"COMPLETED split={split}", flush=True)
    except BaseException:
        terminate_active_processes()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    if not args.audit_only:
        run("build_shadow_annotation_dataset.py", "--manifest", teachers["train"], teachers["val"],
            "--output-root", output/"data", "--seed", args.seed)
        run("check_shadow_annotation_data.py", "--data-root", output/"data")
        run("build_joint_shadow_annotation_dataset.py", "--region-data-root", output/"data", "--output-root", output/"joint")
        run("check_joint_shadow_annotation_data.py", "--data-root", output/"joint", "--require-both-classes")
    write_json(output/"rebuild_protocol.json", {**vars(args), "sources": provenance,
        "policy": "receiver explanations separate from caster ownership; no encoder or flow recomputation"})


if __name__ == "__main__":
    main()
