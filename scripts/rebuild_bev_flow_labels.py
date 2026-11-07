"""Rebuild receiver evidence and labels using existing nuScenes/flow/BEV caches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

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
    provenance = {}

    def run(script, *values):
        command = [sys.executable, str(ROOT/"scripts"/script), *map(str, values)]
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)

    def rebuild_split(split):
        origin, destination = source/split, output/split
        destination.mkdir()
        # Shared read-only inputs keep the rebuilt root compatible with the ablation runner.
        for name in ("pairs", "flow", "mature_bev"):
            path = origin/name
            if not path.is_dir():
                raise FileNotFoundError(path)
            (destination/name).symlink_to(path, target_is_directory=True)
        old_protocol_path = origin/"bev_flow/bev_flow_protocol.json"
        old = json.loads(old_protocol_path.read_text(encoding="utf-8"))
        config = old["config"]
        provenance[split] = {"source_bev_protocol_sha256": digest(old_protocol_path),
            "pair_manifest_sha256": digest(origin/"pairs/pairs.jsonl"),
            "motion_protocol_sha256": digest(origin/"flow/analysis_protocol.json")}
        run("analyze_bev_flow_discrepancy.py", "--pairs", destination/"pairs/pairs.jsonl",
            "--analysis-root", destination/"flow", "--feature-root", destination/"mature_bev",
            "--minimum-bev-cosine", old.get("minimum_bev_cosine", 0.3),
            "--extent-m", config["extent_m"], "--resolution-m", config["resolution_m"],
            "--light-threshold", config["light_threshold_L"], "--omit-receiver-features",
            "--output-root", destination/"bev_flow")
        run("build_physics_shadow_labels.py", "--pairs", destination/"pairs/pairs.jsonl",
            "--analysis-root", destination/"flow", "--bev-flow-root", destination/"bev_flow",
            "--decision-policy", "factorized-v3", "--existence-mode", "static-dynamic",
            "--positive-fusion", args.positive_fusion, "--export-weak-labels", "--output-root", destination/"labels")
        run("build_bev_flow_annotation_manifest.py", "--pairs", destination/"pairs/pairs.jsonl",
            "--labels-root", destination/"labels", "--negative-policy", args.negative_policy,
            "--output-root", destination/"teacher")
    with ThreadPoolExecutor(max_workers=args.split_workers, thread_name_prefix="shadow-split") as executor:
        futures = {executor.submit(rebuild_split, split): split for split in ("train", "val")}
        for future, split in futures.items():
            future.result()
            print(f"COMPLETED split={split}", flush=True)
    run("build_shadow_annotation_dataset.py", "--manifest", output/"train/teacher/canonical.jsonl",
        output/"val/teacher/canonical.jsonl", "--output-root", output/"data", "--seed", args.seed)
    run("check_shadow_annotation_data.py", "--data-root", output/"data")
    run("build_joint_shadow_annotation_dataset.py", "--region-data-root", output/"data", "--output-root", output/"joint")
    run("check_joint_shadow_annotation_data.py", "--data-root", output/"joint", "--require-both-classes")
    write_json(output/"rebuild_protocol.json", {**vars(args), "sources": provenance,
        "policy": "static/dynamic existence separate from motion-based ownership; no encoder or flow recomputation"})


if __name__ == "__main__":
    main()
