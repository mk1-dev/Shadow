from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import BEVFlowConfig, build_bev_flow
from shadow_occ.annotation.cached_bev import attach_cached_bev, select_bev_flow_receivers
from shadow_occ.annotation.data import checked_write_image, digest, load_array, resolve_path, write_rows
from shadow_occ.annotation.physics_io import check_pair_files, load_motion_analysis, read_evidence
from shadow_occ.annotation.physics_labels import validate_window
from shadow_occ.utils.io import write_json


def review_panel(arrays):
    panels = []
    values = [(arrays["appearance_L_compensated"][i], 0, 100, f"receiver L: t{i}") for i in range(3)]
    values += [(arrays["temporal_illumination_change_L"][i], -15, 15, f"illumination delta {i}->{i+1}") for i in range(2)]
    for value, lower, upper, label in values:
        scaled = np.clip((np.nan_to_num(value, nan=lower)-lower)/(upper-lower)*255, 0, 255).astype(np.uint8)
        panel = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        panel[~np.isfinite(value)] = 0
        panels.append((panel, label))
    state = np.zeros((*arrays["explanation_state"].shape, 3), np.uint8)
    for code, color in ((1, (100, 100, 100)), (2, (0, 100, 255)), (3, (0, 220, 220))):
        state[arrays["explanation_state"] == code] = color
    panels.append((state, "A stable / B box / C light change"))
    result = []
    for panel, label in panels:
        panel = cv2.resize(panel, (384, 384), interpolation=cv2.INTER_NEAREST)
        panel = np.vstack([np.zeros((24, 384, 3), np.uint8), panel])
        cv2.putText(panel, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        result.append(panel)
    return np.vstack([np.hstack(result[:3]), np.hstack(result[3:])])


def main():
    parser = argparse.ArgumentParser("Three-frame physical BEV + observed flow + mature BEVFormer features")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-root", required=True, help="Frozen spatial UniAD/OpenDriveVLA BEV cache")
    parser.add_argument("--minimum-bev-cosine", type=float, default=0.3)
    parser.add_argument("--extent-m", type=float, default=40)
    parser.add_argument("--resolution-m", type=float, default=0.5)
    parser.add_argument("--light-threshold", type=float, default=6)
    parser.add_argument("--review-limit", type=int, default=20)
    args = parser.parse_args()
    if not -1 <= args.minimum_bev_cosine <= 1:
        raise ValueError("BEV cosine threshold must be in [-1,1]")
    config = BEVFlowConfig(extent_m=args.extent_m, resolution_m=args.resolution_m, light_threshold_L=args.light_threshold)
    config.grid()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    base, analysis = Path(args.pairs).resolve().parent, Path(args.analysis_root).resolve()
    pairs, stats, analysis_protocol = load_motion_analysis(args.pairs, analysis)
    lookup = {(p["scene_name"], p["official_split"], p["camera"], p["sample_token"]): p for p in pairs}
    if len(lookup) != len(pairs):
        raise ValueError("Duplicate camera/sample pair")
    windows, counts, checked, audit = [], Counter(), set(), []
    for left in tqdm(pairs, desc="three-frame physical / learned BEV", unit="pair", mininterval=5):
        right = lookup.get((left["scene_name"], left["official_split"], left["camera"], left["next_sample_token"]))
        if right is None:
            continue
        try:
            validate_window(left, right)
        except ValueError as exc:
            audit.append({"pair_ids": [left["id"], right["id"]], "reason": str(exc)})
            continue
        for pair in (left, right):
            if pair["id"] not in checked:
                check_pair_files(pair, base)
                checked.add(pair["id"])
        first, ma = read_evidence(left, stats, analysis)
        second, mb = read_evidence(right, stats, analysis)
        frames = [left["frames"][0], right["frames"][0], right["frames"][1]]
        images = [cv2.imread(str(resolve_path(frame["image"], base))) for frame in frames]
        if any(image is None for image in images):
            raise FileNotFoundError("Missing exported receiver image")
        grounds = [load_array(frame["ground_support"], base) > 0 for frame in frames]
        arrays, metadata = build_bev_flow(left, right, images, grounds, first, second,
                                         [ma["exposure"], mb["exposure"]], config)
        arrays, metadata = attach_cached_bev(arrays, metadata, args.feature_root)
        arrays, metadata = select_bev_flow_receivers(arrays, metadata, args.minimum_bev_cosine)
        stem = hashlib.sha256(f"{left['id']}:{right['id']}".encode()).hexdigest()[:24]
        path = output/"windows"/f"{stem}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        windows.append({**metadata, "id": stem, "evidence": str(path), "evidence_sha256": digest(path)})
        counts.update(metadata["counts"])
        if len(windows) <= args.review_limit:
            checked_write_image(output/"review"/f"{stem}.jpg", review_panel(arrays))
        print(f"windows={len(windows)} scene={left['scene_name']} receiver={metadata['counts']['receiver']} "
              f"events={metadata['counts']['illumination_events']} candidates={metadata['counts']['shadow_candidates']}", flush=True)
    write_rows(output/"windows.jsonl", windows)
    write_rows(output/"window_audit.jsonl", audit)
    write_json(output/"bev_flow_protocol.json", {**vars(args), "config": asdict(config),
        "protocol": "bev-flow-discrepancy-v1", "pair_manifest_sha256": digest(args.pairs),
        "analysis_protocol_sha256": digest(analysis_protocol),
        "pair_metrics_sha256": digest(analysis/"pair_metrics.jsonl"),
        "feature_cache_protocol_sha256": digest(Path(args.feature_root)/"cache_protocol.json") if args.feature_root else None,
        "coordinates": "fixed first-sample ego x/y; common world receiver for all three frames",
        "learned_encoder": "cached UniAD/OpenDriveVLA BEVFormer" if args.feature_root else None,
        "flow_units": "source camera pixels per actual interval; never BEV metres or caster velocity"})
    summary = {"windows": len(windows), "invalid_windows": len(audit), "counts": dict(counts),
               "learned_bev_attached": bool(args.feature_root)}
    write_json(output/"summary.json", summary)
    print(summary, flush=True)
    if not windows:
        raise RuntimeError("No consecutive three-frame windows")


if __name__ == "__main__":
    main()
