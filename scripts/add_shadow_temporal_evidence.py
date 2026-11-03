from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import WEAK_SOURCES, checked_write_image, load_array, make_target, read_rows, resolve_path, write_rows
from shadow_occ.annotation.temporal import ground_homographies, temporal_observations, yaw_difference_degrees
from shadow_occ.utils.io import write_json


def absolute_row(row, base):
    row = dict(row)
    row["image"] = str(resolve_path(row["image"], base))
    for key in ("shadow_mask", "negative_mask", "valid_mask", "ground_support"):
        if row.get(key) is not None:
            value = row[key]
            row[key] = {**value, "path": str(resolve_path(value["path"], base))} if isinstance(value, dict) else str(resolve_path(value, base))
    row["regions"] = [{**region, "mask": {**region["mask"], "path": str(resolve_path(region["mask"]["path"], base))} if isinstance(region["mask"], dict) else str(resolve_path(region["mask"], base))} for region in row.get("regions", [])]
    return row


def main():
    parser = argparse.ArgumentParser("Add caster-GT temporal continuity and photometric evidence to shadow data.")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-gap-seconds", type=float, default=0.75)
    parser.add_argument("--static-speed", type=float, default=0.3)
    parser.add_argument("--max-yaw-change-degrees", type=float, default=5.0)
    parser.add_argument("--min-overlap", type=int, default=64)
    parser.add_argument("--max-background-residual", type=float, default=8.0)
    parser.add_argument("--max-exposure-offset", type=float, default=25.0)
    parser.add_argument("--propagate-positive", action="store_true", help="Recover only unknown weak-label pixels supported by a prior positive and current image evidence")
    parser.add_argument("--propagation-confidence", type=float, default=0.7)
    parser.add_argument("--review-limit", type=int, default=50)
    args = parser.parse_args()
    if args.max_gap_seconds <= 0 or args.static_speed < 0 or args.min_overlap < 1:
        raise ValueError("Invalid temporal gap, static speed or overlap threshold")
    if not 0 < args.propagation_confidence <= 1 or args.review_limit < 0:
        raise ValueError("Invalid propagation confidence or review limit")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [absolute_row(row, Path(path).resolve().parent) for path in args.manifest for row in read_rows(path)]
    tracks = defaultdict(list)
    counts = Counter()
    for row in rows:
        if row.get("instance_token") and row.get("camera") and row.get("timestamp") is not None:
            tracks[(row["scene_name"], row["camera"], row["instance_token"])].append(row)
        else:
            row["temporal_status"] = "missing_track_metadata"
    audits = []
    for track in tqdm(tracks.values(), desc="caster track temporal evidence"):
        track.sort(key=lambda row: int(row["timestamp"]))
        for index, current in enumerate(track):
            current["temporal_status"] = "no_previous_observation"
            if index == 0:
                continue
            previous = track[index-1]
            seconds = (int(current["timestamp"])-int(previous["timestamp"]))/1e6
            if seconds <= 0:
                raise ValueError("Duplicate/reversed frame within a caster track")
            if seconds > args.max_gap_seconds:
                current["temporal_status"] = "time_gap"
                continue
            if current["official_split"] != previous["official_split"]:
                raise ValueError("A caster track crosses an official data split")
            required = ("ego_to_global", "ego_to_img", "caster_center_ego", "caster_rotation_global", "ground_support")
            if any(key not in current or key not in previous for key in required):
                current["temporal_status"] = "missing_geometry_or_ground"
                continue
            yaw = yaw_difference_degrees(current["caster_rotation_global"], previous["caster_rotation_global"])
            if yaw > args.max_yaw_change_degrees:
                current["temporal_status"] = "caster_turning"
                continue
            try:
                matrices = ground_homographies(current, previous, current.get("ground_z_ego", 0.0))
            except np.linalg.LinAlgError:
                current["temporal_status"] = "singular_projection"
                continue
            image = cv2.imread(current["image"])
            old_image = cv2.imread(previous["image"])
            if image is None or old_image is None:
                raise FileNotFoundError("Missing temporal frame image")
            target = make_target(current, image.shape[:2], Path("."))
            old_target = make_target(previous, old_image.shape[:2], Path("."))
            ground = load_array(current["ground_support"], Path(".")) > 0
            old_ground = load_array(previous["ground_support"], Path(".")) > 0
            if ground.shape != image.shape[:2] or old_ground.shape != old_image.shape[:2]:
                raise ValueError("Ground support must share its image pixel coordinates")
            observation = temporal_observations(image, old_image, target, old_target, ground, old_ground, matrices, static_speed=args.static_speed, min_overlap=args.min_overlap, max_background_residual=args.max_background_residual, max_exposure_offset=args.max_exposure_offset)
            details = {**observation["details"], "dt_seconds": seconds, "yaw_change_degrees": yaw, "caster_speed_mps": matrices["caster_speed_mps"], "caster_displacement_m": matrices["caster_displacement_m"]}
            stem = hashlib.sha256(current["id"].encode()).hexdigest()[:24]
            target_path = output/"targets"/f"{stem}.png"
            propagated = np.zeros(target.shape, dtype=bool)
            if args.propagate_positive and current.get("label_source") in WEAK_SOURCES and details["lighting_comparable"]:
                light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                background = cv2.GaussianBlur(light, (0, 0), max(15.0, min(light.shape)*0.18))
                dark = light/np.maximum(background, 1.0) < 0.85
                propagated = (target == 255) & observation["shared_caster"] & (observation["previous_target_caster"] == 1) & dark & (observation["photometric_confidence"] >= args.propagation_confidence)
                target[propagated] = 1
                if propagated.any():
                    observation["weight"][propagated] = observation["photometric_confidence"][propagated]*(1.0 if details["mode"] == "static" else 0.5)
            checked_write_image(target_path, target)
            sidecar = output/"evidence"/f"{stem}.npz"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sidecar, weight=observation["weight"].astype(np.float16), delta_l=observation["delta_l"].astype(np.float16), shared_world=observation["shared_world"].astype(np.uint8), shared_caster=observation["shared_caster"].astype(np.uint8), propagated=propagated.astype(np.uint8))
            if len(audits) < args.review_limit:
                for name, array in (("current", image), ("previous_world", observation["previous_world"]), ("previous_caster", observation["previous_caster"])):
                    checked_write_image(output/"review"/f"{stem}_{name}.jpg", array)
                checked_write_image(output/"review"/f"{stem}_consistency_weight.png", np.clip(observation["weight"]*255, 0, 255).astype(np.uint8))
            details["propagated_shadow_pixels"] = int(propagated.sum())
            details["consistency_weight_sum"] = float(observation["weight"].sum())
            current["temporal"] = {**details, "previous_id": previous["id"], "previous_image": previous["image"], "previous_scene_name": previous["scene_name"], "previous_instance_token": previous["instance_token"], "previous_camera": previous["camera"], "evidence": str(sidecar), "world_previous_to_current": matrices["world_previous_to_current"].tolist(), "caster_previous_to_current": matrices["caster_previous_to_current"].tolist()}
            current["temporal_status"] = "comparable" if details["lighting_comparable"] else "insufficient_shared_background_or_light_change"
            if propagated.any():
                current["shadow_mask"] = str(target_path)
                current["mask_encoding"] = "trimap"
                current["label_source"] = "temporal_weak"
                # Keep prior region supervision sources; recovery has no new GT answer.
            counts[details["mode"]+"_pairs"] += 1
            counts["propagated_pixels"] += int(propagated.sum())
            audits.append({"id": current["id"], "scene_name": current["scene_name"], "instance_token": current["instance_token"], "status": current["temporal_status"], **details})
    counts.update(row.get("temporal_status", "missing_track_metadata") for row in rows)
    write_rows(output/"manifest.jsonl", rows)
    write_rows(output/"temporal_audit.jsonl", audits)
    write_json(output/"temporal_protocol.json", {**vars(args), "counts": dict(counts), "constraint": "conditional caster-track continuity; object GT is not shadow GT", "static_rule": "small aligned light differences preserve stable shadow evidence", "moving_rule": "GT translation aligns shadow masks; world-fixed alignment measures light change"})


if __name__ == "__main__":
    main()
