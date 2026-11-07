"""Read-only comparisons of pair-local receiver observations and cached grids."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json

import cv2
import numpy as np

from shadow_occ.annotation.physics_motion import project_world, sample_at


def compare_masks(first, second):
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    if first.shape != second.shape:
        raise ValueError("Receiver masks use different pixel grids")
    intersection = int((first & second).sum())
    union = int((first | second).sum())
    changed = first ^ second
    edges = np.zeros(first.shape, np.uint8)
    for mask in (first, second):
        edges |= cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    difference = int(changed.sum())
    return {"first_pixels": int(first.sum()), "second_pixels": int(second.sum()),
        "intersection": intersection, "union": union, "xor": difference,
        "first_only": int((first & ~second).sum()), "second_only": int((second & ~first).sum()),
        "iou": intersection/union if union else None,
        "mismatch_ratio": difference/union if union else None,
        "changed_on_mask_boundary_fraction": float((changed & (edges > 0)).sum()/difference) if difference else None,
        "both_empty": union == 0}


def index_observations(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        model = {key: pair[key] for key in ("plane_world", "plane_basis_world", "anchor_ego_to_world")}
        model_id = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for slot, frame in enumerate(pair["frames"]):
            key = (pair["scene_name"], pair["camera"], frame["sample_data_token"])
            groups[key].append({"pair_id": pair["id"], "slot": slot, "frame": frame,
                "lidar_sample_token": pair["sample_token"] if slot == 0 else pair["next_sample_token"],
                "ground_model_id": model_id, **model})
    return groups


def frame_provenance(observation):
    frame = observation["frame"]
    return {"pair_id": observation["pair_id"], "slot": observation["slot"],
        "camera_token": frame["sample_data_token"], "timestamp": frame["camera_timestamp"],
        "image_sha256": frame["image_sha256"], "ground_sha256": frame["ground_sha256"],
        "image": frame["image"],
        "ground_support": frame["ground_support"], "ground_model_id": observation["ground_model_id"],
        "lidar_sample_token": observation["lidar_sample_token"],
        "plane_world": observation["plane_world"], "world_anchor": observation["anchor_ego_to_world"],
        "ground_evidence": frame.get("ground_evidence"),
        "object_state_failures": frame.get("object_state_failures")}


def identity_comparison(first, second):
    a, b = first["frame"], second["frame"]
    pa, pb = np.asarray(first["plane_world"], float), np.asarray(second["plane_world"], float)
    pa, pb = pa/np.linalg.norm(pa[:3]), pb/np.linalg.norm(pb[:3])
    origin = np.asarray(first["anchor_ego_to_world"], float)[:, 3]
    return {"same_camera_token": a["sample_data_token"] == b["sample_data_token"],
        "same_timestamp": a["camera_timestamp"] == b["camera_timestamp"],
        "same_image_hash": a["image_sha256"] == b["image_sha256"],
        "same_image_size": a["image_size_wh"] == b["image_size_wh"],
        "projection_max_abs_difference": float(np.max(np.abs(
            np.asarray(a["projection_world_to_image"])-np.asarray(b["projection_world_to_image"])))),
        "same_ground_hash": a["ground_sha256"] == b["ground_sha256"],
        "same_ground_model_id": first["ground_model_id"] == second["ground_model_id"],
        "same_lidar_sample": first["lidar_sample_token"] == second["lidar_sample_token"],
        "plane_normal_difference_degrees": float(np.degrees(np.arccos(np.clip(pa[:3] @ pb[:3], -1, 1)))),
        "plane_height_difference_at_first_anchor_m": float(abs(origin @ (pa-pb)))}


def sample_support(world, frame, ground, threshold=.999):
    height, width = ground.shape
    if (width, height) != tuple(frame["image_size_wh"]):
        raise ValueError("Ground support and camera pixel frame differ")
    uv, _, front = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
    valid = front & (uv >= 0).all(axis=1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1)
    coordinates = np.where(valid[:, None], uv, -1).reshape(*world.shape[:2], 2).astype(np.float32)
    return valid.reshape(world.shape[:2]) & (sample_at(ground.astype(np.float32), coordinates) >= threshold)


def empty_receiver_reason(arrays):
    visible = np.asarray(arrays["visible_by_frame"], bool)
    receiver = np.asarray(arrays["receiver_valid"], bool)
    common = visible.all(axis=0)
    if receiver.any():
        reason = "nonempty_receiver"
    elif not visible[1].any():
        reason = "reference_grid_no_ground_support"
    elif not common.any():
        reason = "central_view_missing_support" if not visible.any(axis=(1, 2)).all() else "central_views_disjoint"
    else:
        reason = "post_visibility_receiver_exclusion"
    counts = {"reason": reason, "visible_pixels_by_frame": visible.sum(axis=(1, 2)).tolist(),
        "common_visible_pixels": int(common.sum()), "receiver_pixels": int(receiver.sum())}
    for key in ("physical_box_footprint", "physical_ray_volume_support"):
        if key in arrays:
            counts[key+"_on_common_pixels"] = int((common & np.asarray(arrays[key], bool).any(axis=0)).sum())
    return counts


def summarize_comparisons(rows):
    nonempty = [row for row in rows if row["union"] > 0]
    union = sum(row["union"] for row in rows)
    return {"comparisons": len(rows), "both_empty": len(rows)-len(nonempty),
        "mismatched": sum(row["xor"] > 0 for row in rows), "xor_pixels": sum(row["xor"] for row in rows),
        "union_weighted_mismatch_ratio": sum(row["xor"] for row in rows)/union if union else None,
        "mean_mismatch_ratio": float(np.mean([row["mismatch_ratio"] for row in nonempty])) if nonempty else None,
        "maximum_mismatch_ratio": max((row["mismatch_ratio"] for row in nonempty), default=None)}


def summarize_windows(rows):
    fields = ("precision_only", "ground_variant_only", "original64_vs_cached", "selected32_vs_cached",
              "soft_support_counterfactual", "selected32_vs_cached_on_receiver")
    frames = [frame for row in rows for frame in row["central_frames"]]
    return {"windows": len(rows),
        "empty_receiver_reasons": dict(Counter(row["receiver"]["reason"] for row in rows)),
        "comparisons_by_relative_frame": {str(offset): {name: summarize_comparisons([
            frame[name] for frame in frames if frame["relative_frame"] == offset and frame[name] is not None]) for name in fields}
            for offset in (-1, 0, 1)},
        "frame_identity_failures": sum(not frame["ground_variant_isolated"] for frame in frames),
        "scope": "ground variant, grid precision and soft-support effects are independent diagnostic comparisons; not production changes"}
