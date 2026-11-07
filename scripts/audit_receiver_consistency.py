"""Read existing pair masks and BEV caches; isolate receiver provenance and precision."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import ground_grid
from shadow_occ.annotation.caster_window import extend_pairs
from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.annotation.receiver_consistency import (
    compare_masks, empty_receiver_reason, frame_provenance, identity_comparison,
    index_observations, sample_support, summarize_comparisons, summarize_windows,
)
from shadow_occ.annotation.receiver_review import mask_image, sheet, tile
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.utils.io import write_json


def audit_split(source, output, radius=3, soft_threshold=.95, review_limit=20,
                metadata_only=False, sampling_root=None):
    pairs_path = source/"pairs/pairs.jsonl"
    pairs = read_rows(pairs_path)
    protocol_path = source/"bev_flow/bev_flow_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["pair_manifest_sha256"] != digest(pairs_path):
        raise ValueError("BEV cache and pair manifest differ")
    groups = index_observations(pairs)
    base = pairs_path.parent

    @lru_cache(maxsize=32)
    def read_ground(encoded, expected):
        ref = json.loads(encoded)
        path = resolve_path(ref["path"] if isinstance(ref, dict) else ref, base)
        if digest(path) != expected:
            raise ValueError("Ground support changed since pair export")
        return load_array(ref, base) > 0

    def ground(frame):
        return read_ground(json.dumps(frame["ground_support"], sort_keys=True), frame["ground_sha256"])

    tokens, mask_rows = [], []
    for (scene, camera, token), observations in tqdm(groups.items(), desc=f"{source.name} frame variants", mininterval=5):
        hashes = sorted({item["frame"]["ground_sha256"] for item in observations})
        tokens.append({"scene_name": scene, "camera": camera, "camera_token": token,
            "ground_hash_versions": len(hashes), "ground_hashes": hashes,
            "ground_model_versions": len({item["ground_model_id"] for item in observations}),
            "image_hash_versions": len({item["frame"]["image_sha256"] for item in observations}),
            "timestamp_versions": len({item["frame"]["camera_timestamp"] for item in observations}),
            "projection_versions": len({np.asarray(item["frame"]["projection_world_to_image"], np.float64).tobytes() for item in observations}),
            "observations": [frame_provenance(item) for item in observations]})
        if not metadata_only:
            # One comparison per distinct hash pair, not per repeated occurrence.
            unique = {item["frame"]["ground_sha256"]: item for item in observations}
            for hash_a, hash_b in combinations(hashes, 2):
                first, second = unique[hash_a], unique[hash_b]
                a, b = ground(first["frame"]), ground(second["frame"])
                mask_rows.append({"scene_name": scene, "camera": camera, "camera_token": token,
                    "first": frame_provenance(first), "second": frame_provenance(second),
                    "identity": identity_comparison(first, second),
                    "status": "compared" if a.shape == b.shape else "incompatible_pixel_grids",
                    "mask_comparison": compare_masks(a, b) if a.shape == b.shape else None})
    write_rows(output/"frame_observations.jsonl", tokens)
    write_rows(output/"ground_mask_comparisons.jsonl", mask_rows)
    summary = {"camera_tokens": len(tokens),
        "camera_tokens_with_multiple_ground_hashes": sum(item["ground_hash_versions"] > 1 for item in tokens),
        "ground_hash_version_counts": dict(Counter(item["ground_hash_versions"] for item in tokens)),
        "camera_tokens_with_multiple_ground_models": sum(item["ground_model_versions"] > 1 for item in tokens),
        "camera_tokens_with_image_identity_conflicts": sum(item["image_hash_versions"] > 1 or item["timestamp_versions"] > 1 or item["projection_versions"] > 1 for item in tokens),
        "metadata_only": metadata_only,
        "different_hash_mask_metrics": summarize_comparisons([row["mask_comparison"] for row in mask_rows if row["mask_comparison"] is not None]),
        "hash_different_but_masks_identical": sum(row["mask_comparison"] is not None and row["mask_comparison"]["xor"] == 0 for row in mask_rows)}
    if metadata_only:
        write_json(output/"summary.json", summary)
        return summary
    by_id = {pair["id"]: pair for pair in pairs}
    forward = {(pair["scene_name"], pair["official_split"], pair["camera"], pair["sample_token"]): pair for pair in pairs}
    backward = {(pair["scene_name"], pair["official_split"], pair["camera"], pair["next_sample_token"]): pair for pair in pairs}
    rows, alternate_rows = [], []
    for window in tqdm(read_rows(source/"bev_flow/windows.jsonl"), desc=f"{source.name} grid comparisons", mininterval=5):
        path = resolve_path(window["evidence"], source/"bev_flow")
        if digest(path) != window["evidence_sha256"]:
            raise ValueError("Cached BEV evidence changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files if name in {
                "receiver_world_xyz", "receiver_valid", "visible_by_frame", "physical_box_footprint", "physical_ray_volume_support"}}
        left, right = (by_id[token] for token in window["pair_ids"])
        world64 = ground_grid(left["plane_basis_world"], np.asarray(left["anchor_ego_to_world"]), BEVGrid(**window["grid"]))
        world32 = arrays["receiver_world_xyz"]
        original = [left["frames"][0], right["frames"][0], right["frames"][1]]
        chain = extend_pairs(left, right, forward, backward, radius)
        selected = [pair["frames"][0] for pair in chain]+[chain[-1]["frames"][1]]
        middle = next(index for index, frame in enumerate(selected) if frame["sample_data_token"] == original[1]["sample_data_token"])
        frame_rows = []
        for index, old in enumerate(original):
            current = selected[middle+index-1]
            old64 = sample_support(world64, old, ground(old))
            old32 = sample_support(world32, old, ground(old))
            selected32 = sample_support(world32, current, ground(current))
            softened = sample_support(world32, old, ground(old), soft_threshold)
            cached = arrays["visible_by_frame"][index].astype(bool)
            receiver = arrays["receiver_valid"].astype(bool)
            isolated = (old["sample_data_token"] == current["sample_data_token"]
                and old["camera_timestamp"] == current["camera_timestamp"]
                and old["image_sha256"] == current["image_sha256"]
                and old["image_size_wh"] == current["image_size_wh"]
                and np.array_equal(old["projection_world_to_image"], current["projection_world_to_image"]))
            frame_rows.append({"relative_frame": index-1, "camera_token": old["sample_data_token"],
                "ground_variant_isolated": bool(isolated),
                "cached_ground_hash_from_pair": old["ground_sha256"], "selected_ground_hash": current["ground_sha256"],
                "same_ground_hash": old["ground_sha256"] == current["ground_sha256"],
                "original64_vs_cached": compare_masks(old64, cached),
                "precision_only": compare_masks(old64, old32),
                "ground_variant_only": compare_masks(old32, selected32) if isolated else None,
                "selected32_vs_cached": compare_masks(selected32, cached),
                "selected32_vs_cached_on_receiver": compare_masks(selected32 & receiver, cached & receiver),
                "soft_support_counterfactual": compare_masks(old32, softened)})
        for frame_index, frame in enumerate(selected):
            key = (window["scene_name"], window["camera"], frame["sample_data_token"])
            selected_mask = sample_support(world32, frame, ground(frame))
            seen_hashes = {frame["ground_sha256"]}
            for other in groups[key]:
                alternate = other["frame"]
                if alternate["ground_sha256"] in seen_hashes:
                    continue
                seen_hashes.add(alternate["ground_sha256"])
                alternate_mask = sample_support(world32, alternate, ground(alternate))
                alternate_rows.append({"window_id": window["id"], "relative_frame": frame_index-middle,
                    "camera_token": frame["sample_data_token"], "selected_ground_hash": frame["ground_sha256"],
                    "alternate_ground_hash": alternate["ground_sha256"],
                    "mask_comparison": compare_masks(selected_mask, alternate_mask)})
        receiver = empty_receiver_reason(arrays)
        receiver["source_reference_ground_pixels"] = int(ground(original[1]).sum())
        if receiver["receiver_pixels"] == 0 and receiver["source_reference_ground_pixels"] == 0:
            receiver["reason"] = "reference_source_ground_empty"
        rows.append({"window_id": window["id"], "scene_name": window["scene_name"], "camera": window["camera"],
            "receiver": receiver, "central_frames": frame_rows,
            "world_quantization_max_m": float(np.linalg.norm(world64-world32, axis=-1).max()),
            "ground_provenance_scope": "cache frame support inferred from hashed source pair manifest, not embedded separately in old BEV NPZ"})
    write_rows(output/"window_consistency.jsonl", rows)
    write_rows(output/"window_ground_variants.jsonl", alternate_rows)
    summary["grid_consistency"] = summarize_windows(rows)
    summary["alternate_ground_by_relative_frame"] = {str(offset): summarize_comparisons([
        row["mask_comparison"] for row in alternate_rows if row["relative_frame"] == offset])
        for offset in sorted({row["relative_frame"] for row in alternate_rows})}
    if sampling_root is not None:
        sampling = read_rows(sampling_root/source.name/"windows.jsonl")
        source_ids = {row["window_id"] for row in rows}
        sampled_ids = {window["window_id"] for window in sampling}
        frames = [frame for window in sampling if window["window_id"] in source_ids for frame in window["frames"]]
        flows = [frame["flow"] for frame in frames if "flow" in frame]
        summary["existing_flow_sampling"] = {"source": str(sampling_root/source.name),
            "window_coverage": {"matched": len(source_ids & sampled_ids), "source_only": len(source_ids-sampled_ids),
                                "sampling_only": len(sampled_ids-source_ids)},
            "flow_reasons": dict(Counter(flow["zero_support_reason"] for flow in flows)),
            "chain_no_visible_overlap": sum(not frame["is_reference"] and frame["chain_visible_overlap_pixels"] == 0 for frame in frames),
            "chain_removed_all_support": sum(not frame["is_reference"] and frame["chain_visible_overlap_pixels"] > 0 and frame["chain_supported_pixels"] == 0 for frame in frames),
            "raw_dis_finite_rate": None, "scope": "read existing sampling diagnostics; raw DIS validity is not inferred from receiver support"}
    worst = sorted((row for row in mask_rows if row["mask_comparison"] is not None and row["mask_comparison"]["xor"] > 0),
                   key=lambda row: -(row["mask_comparison"]["mismatch_ratio"] or 0))[:review_limit]
    previews = []
    for index, row in enumerate(worst):
        a = ground({"ground_support": row["first"]["ground_support"], "ground_sha256": row["first"]["ground_sha256"]})
        b = ground({"ground_support": row["second"]["ground_support"], "ground_sha256": row["second"]["ground_sha256"]})
        difference = np.full((*a.shape, 3), 24, np.uint8)
        difference[a & b], difference[a & ~b], difference[b & ~a] = (120, 120, 120), (60, 60, 240), (255, 255, 0)
        image_path = resolve_path(row["first"]["image"], base)
        if digest(image_path) != row["first"]["image_sha256"]:
            raise ValueError("Review RGB changed since pair export")
        image = cv2.imread(str(image_path))
        panels = [tile(image, "Original camera RGB"), tile(mask_image(a), "Ground A"), tile(mask_image(b), "Ground B"),
                  tile(difference, f"A-only red; B-only cyan; IoU={row['mask_comparison']['iou']}")]
        checked_write_image(output/"review"/f"{index:03}_{row['camera_token']}.jpg", sheet(panels, 4))
        previews.append(tile(difference, row["camera_token"][:20]))
    if previews:
        checked_write_image(output/"ground_difference_contact_sheet.jpg", sheet(previews, 3))
    summary["soft_support_threshold_diagnostic_only"] = soft_threshold
    write_json(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sampling-root", help="Existing geometry audit's sampling directory; optional")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--metadata-only", action="store_true", help="Hash/model index only; no images or BEV arrays read")
    parser.add_argument("--temporal-radius", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--soft-support-threshold", type=float, default=.95, help="Counterfactual comparison only")
    parser.add_argument("--review-limit", type=int, default=20)
    args = parser.parse_args()
    if not 0 < args.soft_support_threshold <= .999 or args.review_limit < 0:
        parser.error("Invalid diagnostic threshold or review count")
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if source == output or output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    sampling = Path(args.sampling_root).resolve() if args.sampling_root else None
    summary = {split: audit_split(source/split, output/split, args.temporal_radius,
        args.soft_support_threshold, args.review_limit, args.metadata_only, sampling) for split in dict.fromkeys(args.splits)}
    write_json(output/"summary.json", summary)
    write_json(output/"protocol.json", {**vars(args), "protocol": "receiver-consistency-audit-v1",
        "pair_manifest_hashes": {split: digest(source/split/"pairs/pairs.jsonl") for split in summary},
        "scope": "read-only existing inputs; no canonical frame cache, label gates or training changes"})
    print(f"Wrote receiver consistency audit to {output}", flush=True)


if __name__ == "__main__":
    main()
