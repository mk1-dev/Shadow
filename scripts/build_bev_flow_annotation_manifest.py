from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import bev_mask_to_image, image_mask_to_bev
from shadow_occ.annotation.bev_teacher import PROTOCOL, endpoint_negative_masks, lit_receiver_negative_mask, merge_trimaps
from shadow_occ.annotation.data import checked_write_image, digest, read_rows, resolve_path, write_rows
from shadow_occ.annotation.physics_labels import light_bank, validate_window, window_tracks
from shadow_occ.annotation.physics_motion import ground_homography
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.utils.io import write_json


CASTER_FIELDS = ("instance_token", "annotation_token", "associated_annotation_token", "caster_category",
                 "caster_bbox_xyxy", "caster_source", "association_source", "association_status", "association_evidence")


@lru_cache(maxsize=1)
def load_receiver_evidence(path, sha256):
    if digest(path) != sha256:
        raise ValueError("BEV evidence changed after label selection")
    keys = ("bev_world_to_grid", "receiver_world_xyz", "ground_basis_world", "receiver_valid",
            "shadow_boundary_onset", "shadow_boundary_recovery", "learned_bev_receiver_support",
            "shadow_flow_supported", "appearance_local_ratio")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (*keys, "appearance_L_raw", "shadow_candidate") if key in archive.files}


def main():
    parser = argparse.ArgumentParser("Build the new BEV-flow training manifest, without existing shadow labels")
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--minimum-pixels", type=int, default=32)
    parser.add_argument("--minimum-lit-ratio", type=float, default=0.97)
    parser.add_argument("--negative-exclusion-root", help="Optional camera-image risk masks named <frame sample_data_token>.png")
    parser.add_argument("--negative-policy", choices=["endpoints", "endpoints+lit-receivers"], default="endpoints+lit-receivers")
    parser.add_argument("--lit-radius-m", type=float, default=12.0)
    parser.add_argument("--lit-negative-ratio", type=float, default=2.0)
    args = parser.parse_args()
    if args.minimum_pixels < 1 or not 0.8 <= args.minimum_lit_ratio <= 1.5:
        raise ValueError("Invalid region size or lit-endpoint ratio")
    if not np.isfinite([args.lit_radius_m, args.lit_negative_ratio]).all() or min(args.lit_radius_m, args.lit_negative_ratio) <= 0:
        raise ValueError("Lit receiver radius and sample ratio must be positive")
    labels_root, base = Path(args.labels_root).resolve(), Path(args.pairs).resolve().parent
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    pairs = {row["id"]: row for row in read_rows(args.pairs)}
    decisions = {row["id"]: row for row in read_rows(labels_root/"decisions.jsonl")}
    frames, counts, lit_windows = defaultdict(list), Counter(), set()
    negative_audit = []
    starts = {(p["scene_name"], p["camera"], p["sample_token"]): p for p in pairs.values()}
    ends = {(p["scene_name"], p["camera"], p["next_sample_token"]): p for p in pairs.values()}

    def add(row, target):
        key = (row["scene_name"], row["camera"], row["sample_token"], digest(row["image"]))
        frames[key].append((row, target))

    def context_for(row, identifier):
        key = (row["scene_name"], row["camera"], row["sample_token"])
        if key not in starts or key not in ends:
            return None
        previous, following = ends[key], starts[key]
        try:
            validate_window(previous, following)
        except ValueError:
            return None
        middle = following["frames"][0]
        current_ground = cv2.imread(row["ground_support"], cv2.IMREAD_GRAYSCALE) > 0
        projection = middle["projection_world_to_image"]
        basis = np.asarray(previous["plane_basis_world"])
        context = {"scene_name": row["scene_name"], "camera": row["camera"], "middle_timestamp": row["timestamp"],
            "middle_image_sha256": digest(row["image"]), "alignment": "same_world_receiver_not_caster_translation", "views": []}
        for slot, frame in (("previous", previous["frames"][0]), ("next", following["frames"][1])):
            image = cv2.imread(str(resolve_path(frame["image"], base)))
            support = cv2.imread(str(resolve_path(frame["ground_support"], base)), cv2.IMREAD_GRAYSCALE)
            h = ground_homography(frame["projection_world_to_image"], projection, basis)
            size = (current_ground.shape[1], current_ground.shape[0])
            aligned = cv2.warpPerspective(image, h, size)
            valid = cv2.warpPerspective(support, h, size, flags=cv2.INTER_NEAREST) > 0
            aligned[~(valid & current_ground)] = 0
            path = output/"context"/f"{identifier}_{slot}.png"
            checked_write_image(path, aligned)
            context["views"].append({"slot": slot, "image": str(path), "sha256": digest(path),
                                     "timestamp": frame["camera_timestamp"]})
        if row.get("physics_context", {}).get("transitions"):
            context["transitions"] = row["physics_context"]["transitions"]
        return context

    for row in tqdm(read_rows(labels_root/"canonical.jsonl"), desc="BEV shadow / lit-endpoint labels", mininterval=5):
        decision = decisions[row["id"]]
        if not decision["shadow_decision"]["shadow_accepted"]:
            continue
        if not row["physics_export_enabled"] or not row.get("bev_flow", {}).get("selection"):
            raise ValueError("Run learned BEV-flow selection and export its labels first")
        target = cv2.imread(row["shadow_mask"], cv2.IMREAD_GRAYSCALE)
        if target is None or not (target == 1).any():
            raise ValueError("Accepted receiver has no exported positive pixels")
        evidence = row["bev_flow"]
        arrays = load_receiver_evidence(evidence["evidence"], evidence["evidence_sha256"])
        left, right = [pairs[key] for key in row["physics_pair_ids"]]
        grid = BEVGrid(**evidence["grid"])
        middle = right["frames"][0]
        positive_bev = image_mask_to_bev(target == 1, middle["projection_world_to_image"], arrays["receiver_world_xyz"])
        positive_bev &= arrays["receiver_valid"]
        tracks, _ = window_tracks(left, right)
        endpoint_frames = {0: left["frames"][0], 1: middle, 2: right["frames"][1]}
        exclusions = {}
        if args.negative_exclusion_root:
            for slot, frame in endpoint_frames.items():
                path = Path(args.negative_exclusion_root)/f"{frame['sample_data_token']}.png"
                if path.exists():
                    exclusion = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if exclusion is None or exclusion.shape[::-1] != tuple(frame["image_size_wh"]):
                        raise ValueError(f"Expected exported camera-image exclusion mask: {path}")
                    exclusions[slot] = image_mask_to_bev(exclusion > 0, frame["projection_world_to_image"],
                        arrays["receiver_world_xyz"])
        masks, details = endpoint_negative_masks(arrays, positive_bev, tracks, left["plane_world"], light_bank(),
            decision["shadow_decision"]["plausible_hypotheses"], minimum_lit_ratio=args.minimum_lit_ratio,
            frame_volumes={slot: frame.get("object_volumes_world", []) for slot, frame in endpoint_frames.items()},
            exclusion_masks=exclusions)
        negative_audit.append({"parent_id": row["id"], "pair_ids": row["physics_pair_ids"], "endpoints": details})
        row = {**row, "physics_protocol": PROTOCOL}
        row["regions"] = [{**region, "id": f"{row['id']}:{region['id']}", "bev_flow": evidence, "teacher_evidence": {"kind": "selected_shadow",
            "parent_id": row["id"]}, **{key: row[key] for key in CASTER_FIELDS if key in row}} for region in row["regions"]]
        add(row, target)
        counts["accepted_shadow_regions"] += 1
        window = tuple(row["physics_pair_ids"])
        if args.negative_policy == "endpoints+lit-receivers" and window not in lit_windows:
            lit_bev, lit_details = lit_receiver_negative_mask(arrays, positive_bev, tracks, left["plane_world"],
                light_bank(), decision["shadow_decision"]["plausible_hypotheses"], resolution=grid.resolution,
                volumes=middle.get("object_volumes_world", []), exclusion=exclusions.get(1),
                minimum_lit_ratio=args.minimum_lit_ratio, radius_m=args.lit_radius_m, negative_ratio=args.lit_negative_ratio)
            negative_audit.append({"parent_id": row["id"], "pair_ids": row["physics_pair_ids"], "lit_receiver": lit_details})
            ground = cv2.imread(str(resolve_path(middle["ground_support"], base)), cv2.IMREAD_GRAYSCALE)
            lit = bev_mask_to_image(lit_bev, middle["projection_world_to_image"], arrays["ground_basis_world"], grid, ground.shape) & (ground > 0)
            counts[f"lit_receiver:{lit_details['reason']}"] += 1
            if lit.sum() >= args.minimum_pixels:
                negative = np.full(lit.shape, 255, np.uint8)
                negative[lit] = 0
                identifier = f"{row['id']}:lit_receiver"
                add({"id": identifier, "scene_name": row["scene_name"], "camera": row["camera"],
                    "official_split": row["official_split"], "sample_token": row["sample_token"],
                    "timestamp": row["timestamp"], "image": row["image"], "ground_support": row["ground_support"],
                    "label_source": "temporal_weak", "physics_protocol": PROTOCOL, "physics_export_enabled": True,
                    "bev_flow": evidence, "regions": [{"id": identifier, "shadow_label": "NO",
                        "label_source": "temporal_weak", "bev_flow": evidence,
                        "teacher_evidence": {"kind": "observed_lit_receiver", "parent_id": row["id"], **lit_details},
                        "association_source": "unresolved", "association_status": "not_applicable"}]}, negative)
                lit_windows.add(window)
                counts["lit_receiver_regions"] += 1
            else:
                counts["lit_receiver_negative_too_small"] += 1
        for slot, frame, token in ((0, left["frames"][0], left["sample_token"]),
                                    (2, right["frames"][1], right["next_sample_token"])):
            ground = cv2.imread(str(resolve_path(frame["ground_support"], base)), cv2.IMREAD_GRAYSCALE)
            if ground is None:
                raise FileNotFoundError(frame["ground_support"])
            mask = bev_mask_to_image(masks[slot], frame["projection_world_to_image"],
                                    arrays["ground_basis_world"], grid, ground.shape) & (ground > 0)
            if mask.sum() < args.minimum_pixels:
                counts["endpoint_negative_too_small"] += 1
                continue
            negative = np.full(mask.shape, 255, np.uint8)
            negative[mask] = 0
            identifier = f"{row['id']}:lit_endpoint_{slot}"
            add({"id": identifier, "scene_name": row["scene_name"], "camera": row["camera"],
                "official_split": row["official_split"], "sample_token": token,
                "timestamp": frame["camera_timestamp"], "image": str(resolve_path(frame["image"], base)),
                "ground_support": str(resolve_path(frame["ground_support"], base)),
                "label_source": "temporal_weak", "physics_protocol": PROTOCOL, "physics_export_enabled": True,
                "bev_flow": evidence, "regions": [{"id": identifier, "shadow_label": "NO",
                    "label_source": "temporal_weak", "bev_flow": evidence,
                    "teacher_evidence": {"kind": "observed_lit_endpoint", "parent_id": row["id"],
                                         "slot": slot, **details[str(slot)]},
                    "association_source": "unresolved", "association_status": "not_applicable"}]}, negative)
            counts["lit_endpoint_regions"] += 1

    canonical = []
    for key, observations in tqdm(frames.items(), desc="merge physical frames / temporal views", mininterval=5):
        # Preserve a real three-frame context when this frame also has a positive observation.
        parent = next((row for row, target in observations if (target == 1).any()), observations[0][0])
        target, conflicts = merge_trimaps([value for _, value in observations])
        counts["conflicting_pixels_ignored"] += conflicts
        identifier = hashlib.sha256(":".join(key).encode()).hexdigest()[:24]
        target_path = output/"trimaps"/f"{identifier}.png"
        checked_write_image(target_path, target)
        regions = []
        for observation, raw_target in observations:
            value = 1 if (raw_target == 1).any() else 0
            if value == 1:
                raw_regions = [(region, cv2.imread(region["mask"], cv2.IMREAD_GRAYSCALE) > 0)
                               for region in observation["regions"]]
            else:
                raw_regions = [(observation["regions"][0], raw_target == 0)]
            for region, mask in raw_regions:
                mask &= target == value
                if mask.sum() < args.minimum_pixels:
                    continue
                stem = hashlib.sha256(region["id"].encode()).hexdigest()[:24]
                path = output/"regions"/f"{stem}.png"
                checked_write_image(path, mask.astype(np.uint8)*255)
                regions.append({**region, "id": stem, "mask": str(path), "mask_sha256": digest(path),
                    "proposal_source": "bev_flow_selected_receiver" if value else f"bev_flow_{region['teacher_evidence']['kind']}"})
        if not regions:
            continue
        record = {**parent, "id": identifier, "shadow_mask": str(target_path), "mask_encoding": "trimap",
            "physics_target_sha256": digest(target_path), "regions": regions,
            "require_three_frame_vlm": True, "teacher_observation_ids": [row["id"] for row, _ in observations]}
        context = context_for(parent, identifier)
        if context:
            record["physics_context"] = context
        else:
            record.pop("physics_context", None)
            counts["segmentation_only_boundary_frames"] += 1
        canonical.append(record)
        counts["exported_frames"] += 1
        counts.update(f"exported_regions:{region['shadow_label']}" for region in regions)
        counts["known_shadow_pixels"] += int((target == 1).sum())
        counts["known_lit_pixels"] += int((target == 0).sum())
    write_rows(output/"canonical.jsonl", canonical)
    write_rows(output/"negative_audit.jsonl", negative_audit)
    write_json(output/"protocol.json", {**vars(args), "protocol": PROTOCOL, "counts": dict(counts),
        "pair_manifest_sha256": digest(args.pairs), "decisions_sha256": digest(labels_root/"decisions.jsonl"),
        "negative_source": "observed lit endpoints plus compact persistently lit neighbouring receivers outside competing shadows",
        "negative_exclusion_scope": "all annotated endpoint volumes plus optional static-risk masks; not exhaustive unannotated casters",
        "unknown_policy": "all other pixels ignored", "image_policy": "original camera images for BOTH classes"})
    print(dict(counts), flush=True)
    if not canonical or not counts["known_shadow_pixels"] or not counts["known_lit_pixels"]:
        raise RuntimeError("BEV-flow teacher needs observed shadow and lit-receiver pixels; inspect protocol.json for separate positive/negative counts")


if __name__ == "__main__":
    main()
