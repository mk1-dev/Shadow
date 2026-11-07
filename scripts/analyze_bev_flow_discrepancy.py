from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import BEVFlowConfig, build_bev_flow
from shadow_occ.annotation.cached_bev import attach_cached_bev, select_bev_flow_receivers
from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.annotation.physics_io import check_pair_files, load_motion_analysis, read_evidence
from shadow_occ.annotation.physics_labels import validate_window
from shadow_occ.annotation.receiver_explanations import attach_appearance_prior
from shadow_occ.annotation.mature_objects import attach_mature_objects
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
        state[arrays["receiver_observation_state"] == code] = color
    if "mature_entity_probability" in arrays:
        state[arrays["mature_entity_probability"][1] >= .25] = (255, 100, 60)
    change = arrays["receiver_boundary_formation"] | arrays["receiver_boundary_recovery"]
    state[change] = (0, 220, 220)
    panels.append((state, "ground / GT orange / model blue / light"))
    result = []
    for panel, label in panels:
        panel = cv2.resize(panel, (384, 384), interpolation=cv2.INTER_NEAREST)
        panel = np.vstack([np.zeros((24, 384, 3), np.uint8), panel])
        cv2.putText(panel, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        result.append(panel)
    return np.vstack([np.hstack(result[:3]), np.hstack(result[3:])])


def motion_review_panel(image, rendered, motion, evidence, lighting_comparable):
    overlay = image.copy()
    entity = rendered["confidence"] >= .25
    overlay[entity] = (.6*overlay[entity]+.4*np.array([255, 100, 60])).astype(np.uint8)
    supported = np.isfinite(motion["expected"]).all(axis=-1)
    yy, xx = np.where(supported)
    indices = np.linspace(0, max(len(xx)-1, 0), min(len(xx), 180), dtype=int)
    arrows = []
    for field in (motion["expected"], evidence["flow_t_to_next"]):
        canvas = image.copy()
        for index in indices:
            x, y = int(xx[index]), int(yy[index])
            delta = field[y, x]
            if np.isfinite(delta).all():
                endpoint = tuple(np.rint([x, y]+delta).astype(int))
                cv2.arrowedLine(canvas, (x, y), endpoint, (0, 230, 230), 1, tipLength=.25)
        arrows.append(canvas)
    advantage = motion["ground_residual"]-motion["object_residual"]
    light = evidence.get("delta_L", np.full(image.shape[:2], np.nan))
    light_valid = evidence["visible_ground"] & lighting_comparable & np.isfinite(light)
    maps = []
    for value, valid, extent in ((advantage, np.isfinite(advantage), 5.), (light, light_valid, 15.)):
        scaled = np.clip((np.nan_to_num(value)/extent+1)*127.5, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        heat[~valid] = (image[~valid]*.25).astype(np.uint8)
        maps.append(heat)
    panels = [(image, "current frame"), (overlay, "decoded 3D predictions"),
              (arrows[0], "3D expected surface flow"), (arrows[1], "2D flow at the same pixels"),
              (maps[0], "ground error - object error (px)"),
              (maps[1], "registered light change" if lighting_comparable else "light change unavailable")]
    tiles = []
    for value, label in panels:
        tile = np.zeros((240, 384, 3), np.uint8)
        scale = min(384/value.shape[1], 216/value.shape[0])
        value = cv2.resize(value, (max(1, round(value.shape[1]*scale)), max(1, round(value.shape[0]*scale))))
        tile[24:24+value.shape[0], :value.shape[1]] = value
        cv2.putText(tile, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])


def main():
    parser = argparse.ArgumentParser("Three-frame physical BEV + observed flow + mature BEVFormer features")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-root", help="Frozen spatial UniAD/OpenDriveVLA BEV cache")
    parser.add_argument("--feature-gate", choices=["aligned", "none", "shuffled", "matched-random"], default="aligned")
    parser.add_argument("--flow-gate", choices=["registered", "none"], default="registered",
                        help="Remove final registration gate only; cached exposure estimation still uses DIS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-bev-cosine", type=float, default=0.3)
    parser.add_argument("--extent-m", type=float, default=40)
    parser.add_argument("--resolution-m", type=float, default=0.5)
    parser.add_argument("--light-threshold", type=float, default=6)
    parser.add_argument("--review-limit", type=int, default=20)
    parser.add_argument("--omit-receiver-features", action="store_true", help="Retain cosine/gates but do not duplicate sampled feature tensors")
    parser.add_argument("--shadow-baseline-manifest", help="Optional per-frame external shadow probability maps")
    parser.add_argument("--object-root", help="Decoded mature 3D cache; defaults to --feature-root")
    parser.add_argument("--mature-object-mode", choices=["required", "off"], default="required",
                        help="Active path requires decoded predictions; off is the explicit GT-geometry baseline")
    args = parser.parse_args()
    if args.feature_gate != "none" and not args.feature_root:
        parser.error("--feature-root is required unless --feature-gate none")
    object_root = args.object_root or args.feature_root
    if args.mature_object_mode == "required" and not object_root:
        parser.error("Decoded 3D comparison requires --object-root or --feature-root")
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
    prior_rows = read_rows(args.shadow_baseline_manifest) if args.shadow_baseline_manifest else []
    prior_lookup = {row["sample_data_token"]: row for row in prior_rows}
    if len(prior_lookup) != len(prior_rows):
        raise ValueError("Shadow baseline manifest has duplicate frames")
    motion_protocol = json.loads(analysis_protocol.read_text(encoding="utf-8"))
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
        if args.shadow_baseline_manifest:
            arrays, prior_metadata = attach_appearance_prior(arrays, frames, prior_lookup,
                Path(args.shadow_baseline_manifest).resolve().parent)
            metadata["appearance_baseline"] = prior_metadata
        stem = hashlib.sha256(f"{left['id']}:{right['id']}".encode()).hexdigest()[:24]
        if args.feature_gate != "none":
            arrays, metadata = attach_cached_bev(arrays, metadata, args.feature_root)
        if args.mature_object_mode == "required":
            feature_sources = metadata.get("learned_bev", {}).get("sources")
            arrays, metadata, rendered, object_motion = attach_mature_objects(
                arrays, metadata, frames, (first, second), object_root, feature_sources)
            motion_path = output/"image_motion"/f"{stem}.npz"
            motion_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(motion_path,
                predicted_entity_probability=np.stack([value["confidence"] for value in rendered]),
                predicted_entity_depth=np.stack([value["depth"] for value in rendered]),
                expected_object_flow_px=np.stack([value["expected"] for value in object_motion]),
                observed_optical_flow_px=np.stack([first["flow_t_to_next"], second["flow_t_to_next"]]),
                object_flow_residual_px=np.stack([value["object_residual"] for value in object_motion]),
                ground_flow_residual_px=np.stack([value["ground_residual"] for value in object_motion]),
                entity_motion_fit=np.stack([value["entity_fit"] for value in object_motion]))
            metadata["mature_image_motion"] = {"path": str(motion_path), "sha256": digest(motion_path),
                "coordinates": "working camera pixels; intervals 0->1 and 1->2", "gt_used": False}
            if len(windows) < args.review_limit:
                checked_write_image(output/"motion_review"/f"{stem}.jpg", motion_review_panel(
                    images[1], rendered[1], object_motion[1], second, bool(mb["exposure"]["valid"])))
        window_seed = int(hashlib.sha256(f"{args.seed}:{stem}".encode()).hexdigest()[:16], 16)
        arrays, metadata = select_bev_flow_receivers(arrays, metadata, args.minimum_bev_cosine,
            feature_gate=args.feature_gate, flow_gate=args.flow_gate == "registered", seed=window_seed)
        if args.omit_receiver_features:
            arrays.pop("learned_bev_receiver_features", None)
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
        "receiver_explanation_code_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/receiver_explanations.py"),
        "shadow_baseline_manifest_sha256": digest(args.shadow_baseline_manifest) if args.shadow_baseline_manifest else None,
        "protocol": "bev-flow-discrepancy-v1", "pair_manifest_sha256": digest(args.pairs),
        "analysis_protocol_sha256": digest(analysis_protocol),
        "pair_metrics_sha256": digest(analysis/"pair_metrics.jsonl"),
        "feature_cache_protocol_sha256": digest(Path(args.feature_root)/"cache_protocol.json") if args.feature_root else None,
        "object_root": str(Path(object_root).resolve()) if object_root else None,
        "mature_objects_code_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/mature_objects.py"),
        "coordinates": "fixed first-sample ego x/y; common world receiver for all three frames",
        "learned_encoder": "cached UniAD/OpenDriveVLA BEVFormer" if args.feature_root else None,
        "registration_mode": motion_protocol.get("registration_mode", "geometry-dis"),
        "flow_ablation_scope": "final receiver gate; upstream registration/exposure source recorded separately",
        "flow_units": "source camera pixels per actual interval; never BEV metres or caster velocity"})
    summary = {"windows": len(windows), "invalid_windows": len(audit), "counts": dict(counts),
               "learned_bev_attached": bool(args.feature_root)}
    write_json(output/"summary.json", summary)
    print(summary, flush=True)
    if not windows:
        raise RuntimeError("No consecutive three-frame windows")


if __name__ == "__main__":
    main()
