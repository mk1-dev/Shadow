"""CPU-only projection/flow sampling audit on frozen BEV windows; no label ranking."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.caster_window import (
    extend_pairs, long_observations, receiver_sampling_summary,
)
from shadow_occ.annotation.data import digest, read_rows, resolve_path, write_rows
from shadow_occ.annotation.physics_io import load_motion_analysis
from shadow_occ.utils.io import write_json


def audit_split(source, output, radius, limit):
    pairs_path = source/"pairs/pairs.jsonl"
    pairs, stats, _ = load_motion_analysis(pairs_path, source/"flow")
    by_id = {pair["id"]: pair for pair in pairs}
    forward = {(*[pair[key] for key in ("scene_name", "official_split", "camera")], pair["sample_token"]): pair
               for pair in pairs}
    backward = {(*[pair[key] for key in ("scene_name", "official_split", "camera")], pair["next_sample_token"]): pair
                for pair in pairs}
    windows = read_rows(source/"bev_flow/windows.jsonl")
    if limit:
        windows = windows[:limit]
    records = []
    for window in tqdm(windows, desc=f"{source.name} receiver sampling", unit="window", mininterval=5):
        evidence_path = resolve_path(window["evidence"], source/"bev_flow")
        if digest(evidence_path) != window["evidence_sha256"]:
            raise ValueError("Cached BEV evidence changed")
        with np.load(evidence_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("receiver_world_xyz", "receiver_valid", "visible_by_frame")}
        left, right = (by_id[token] for token in window["pair_ids"])
        chain = extend_pairs(left, right, forward, backward, radius)
        sampled = long_observations(chain, right["frames"][0]["sample_data_token"], arrays,
            None, pairs_path.parent, stats, source/"flow", sampling_only=True)
        records.append({"window_id": window["id"], "scene_name": window["scene_name"], "camera": window["camera"],
            "frames": sampled["sampling_diagnostics"]})
    write_rows(output/"windows.jsonl", records)
    summary = receiver_sampling_summary([frame for row in records for frame in row["frames"]])
    summary["windows"] = len(records)
    write_json(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--temporal-radius", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--max-windows", type=int, default=0, help="Per split; zero audits all cached windows")
    args = parser.parse_args()
    if args.max_windows < 0:
        parser.error("max-windows must be nonnegative")
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if source == output or output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory, not the source cache")
    output.mkdir(parents=True, exist_ok=True)
    summary = {split: audit_split(source/split, output/split, args.temporal_radius, args.max_windows)
               for split in dict.fromkeys(args.splits)}
    write_json(output/"summary.json", summary)
    write_json(output/"protocol.json", {**vars(args), "protocol": "receiver-sampling-audit-v1",
        "source_manifests": {split: {name: digest(source/split/name) for name in
            ("pairs/pairs.jsonl", "flow/analysis_protocol.json", "bev_flow/windows.jsonl")} for split in summary}})
    print(f"Wrote receiver sampling audit to {output}", flush=True)


if __name__ == "__main__":
    main()
