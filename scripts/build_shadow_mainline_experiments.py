"""Bridge frozen discovery outputs to E1 predictions and E2 segmenter data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.annotation.bev_flow import BEVFlowConfig
from shadow_occ.annotation.bev_teacher import lit_receiver_negative_mask
from shadow_occ.annotation.receiver_negatives import independent_lit_receiver_mask
from shadow_occ.annotation.physics_labels import light_bank
from shadow_occ.annotation.physics_motion import plane_basis, project_world, sample_at
from shadow_occ.annotation.video_review import find_clips, load_clip, load_npz
from shadow_occ.annotation.video_review_render import project_mask, read_rgb
from shadow_occ.utils.io import write_json
from review_shadow_video_changes import camera_index


def frame_id(scene, camera, token):
    return hashlib.sha256(json.dumps([scene, camera, token]).encode()).hexdigest()[:24]


def simple_name(value):
    if not value or Path(value).name != value or value in {".", ".."} or "\\" in value:
        raise ValueError("Method must be a simple directory name")
    return value


def projected_score(arrays, frame, plane):
    """Project saved ranking scores, without changing discovery or acceptance."""
    name = next((key for key in ("proposal_score_map", "cross_space_discovery_score", "discrepancy_score_map") if key in arrays), None)
    if name is None:
        return None, None
    value = arrays[name]
    basis = plane_basis(plane)
    homography = np.asarray(frame["projection_world_to_image"]) @ basis @ np.linalg.inv(arrays["bev_world_to_grid"] @ basis)
    front = project_world(arrays["receiver_world_xyz"].reshape(-1, 3), frame["projection_world_to_image"])[2].reshape(value.shape)
    valid = np.isfinite(value) & front
    if "receiver_valid_by_frame" in arrays:
        valid &= arrays["receiver_valid_by_frame"].astype(bool)
    if np.any((value[valid] < 0) | (value[valid] > 1)):
        raise ValueError("Saved ranking scores must be in [0,1], not unbounded residuals")
    size = tuple(frame["image_size_wh"])
    warp = lambda x: cv2.warpPerspective(x, homography, size, flags=cv2.INTER_NEAREST)
    supported = warp(valid.astype(np.uint8)) > 0
    score = warp(np.where(valid, value, 0).astype(np.float32))
    return np.where(supported, score, 0), {"field": name, "support": supported}


def negative_index(paths):
    result = defaultdict(list)
    for path in paths:
        path = Path(path).resolve()
        manifest_sha = digest(path)
        for row in read_rows(path):
            if (row.get("label_source") not in {"temporal_weak", "geometry_weak"}
                    or row.get("mask_encoding") != "trimap" or not row.get("physics_export_enabled")
                    or not row.get("physics_protocol")):
                raise ValueError("Supplemental negatives require exported physical weak trimaps, never human test GT")
            token = row.get("frame_token") or row.get("sample_data_token")
            identity = token or "image:"+str(resolve_path(row["image"], path.parent))
            key = (row["scene_name"], row["camera"], identity)
            reference = row.get("target") or row.get("shadow_mask")
            reference = dict(reference) if isinstance(reference, dict) else {"path": reference}
            reference["path"] = str(resolve_path(reference["path"], path.parent))
            result[key].append((reference, {"manifest": str(path), "manifest_sha256": manifest_sha,
                "physics_protocol": row["physics_protocol"], "reference": reference,
                "expected_target_sha256": row.get("physics_target_sha256")}))
    return result


def automatic_lit_negatives(clip, camera_frames):
    """Combine shadow-neighbour negatives with independent observed lit receivers."""
    required = {"receiver_valid_by_frame", "bev_feature_valid"}
    ground_reference = clip["pair"].get("ground_motion_reference_valid_map")
    if not all(required <= arrays.keys() for arrays in clip["arrays"]):
        return {}, {"reason": "missing_saved_receiver_or_feature_support", "frames": {}}
    resolution = float(clip["protocol"]["grid"]["resolution"])
    raw, ratios = [], []
    for arrays, (frame, base) in zip(clip["arrays"], camera_frames):
        image = read_rgb(frame, base)
        light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)*(100/255)
        points = arrays["receiver_world_xyz"]
        uv, _, _ = project_world(points.reshape(-1, 3), frame["projection_world_to_image"])
        light = sample_at(light, uv.reshape(*points.shape[:2], 2).astype(np.float32))
        support = arrays["receiver_valid_by_frame"].astype(bool) & np.isfinite(light)
        sigma = BEVFlowConfig().background_sigma_m/resolution
        density = cv2.GaussianBlur(support.astype(np.float32), (0, 0), sigma)
        mean = cv2.GaussianBlur(np.where(support, light, 0), (0, 0), sigma)/np.maximum(density, 1e-6)
        raw.append(light)
        ratios.append(np.where(support, light/np.maximum(mean, 1), np.nan))
    masks, details = {}, {}
    for slot in range(1, len(camera_frames)-1):
        arrays = clip["arrays"][slot]
        positive = arrays["shadow_target_bev"] == 1
        frame, base = camera_frames[slot]
        bank_path = clip["folder"]/f"{slot:03d}_light_bank.npz"
        directions = load_npz(bank_path)["directions_world"] if bank_path.is_file() else light_bank()
        explanations = []
        for region in clip["rows"][slot]["regions"]:
            physics = region.get("physics", {})
            selected = physics.get("selected")
            if physics.get("shadow_accepted") and selected and 0 <= selected["light_id"] < len(directions):
                explanations.append(selected)
        nearby = clip["arrays"][slot-1:slot+2]
        receiver = np.logical_and.reduce([row["receiver_valid_by_frame"].astype(bool) for row in nearby])
        features = np.logical_and.reduce([row["bev_feature_valid"].astype(bool) for row in nearby])
        motion = ground_reference[slot-1].astype(bool) & ground_reference[slot].astype(bool) if ground_reference is not None else np.zeros_like(receiver)
        volumes = frame.get("object_volumes_world")
        input_arrays = {"bev_world_to_grid": arrays["bev_world_to_grid"], "receiver_valid": receiver,
            "learned_bev_receiver_support": features, "shadow_flow_supported": motion,
            "appearance_L_raw": np.stack(raw[slot-1:slot+2]),
            "appearance_local_ratio": np.stack(ratios[slot-1:slot+2]),
            "shadow_candidate": arrays["shadow_proposal_bev"].astype(bool)}
        negative, detail = np.zeros_like(receiver), {"reason": "no_positive_core_or_explanation", "cells": 0}
        if positive.any() and explanations and volumes is not None:
            negative, detail = lit_receiver_negative_mask(input_arrays, positive, [],
                clip["protocol"]["ground_plane_world"], directions, explanations,
                resolution=resolution, volumes=volumes, exclusion=arrays.get("entity_occlusion_mask"))
        exclusion = np.logical_or.reduce([row["shadow_proposal_bev"].astype(bool)
            | (row["shadow_target_bev"] == 1) | row.get("entity_occlusion_mask", np.zeros_like(receiver)).astype(bool)
            for row in nearby])
        aligned = all(np.allclose(row["receiver_world_xyz"][receiver], arrays["receiver_world_xyz"][receiver], atol=1e-4, rtol=0)
                      for row in nearby)
        independent, independent_detail = np.zeros_like(receiver), {"reason": "unaligned_world_receivers", "cells": 0}
        if aligned:
            independent, independent_detail = independent_lit_receiver_mask(
                {**input_arrays, "receiver_valid": receiver & features, "exclusion": exclusion},
                clip["protocol"]["ground_plane_world"], directions, volumes)
        negative |= independent
        projected = project_mask(negative, arrays, frame, clip["protocol"]["ground_plane_world"])
        if frame.get("ground_sha256"):
            reference = frame["ground_support"]
            path = reference["path"] if isinstance(reference, dict) else reference
            if digest(resolve_path(path, base)) != frame["ground_sha256"]:
                raise ValueError("Receiver support changed since pair export")
        ground = load_array(frame["ground_support"], base) > 0
        projected &= ground & ~clip["images"][slot]["shadow_proposal_image"].astype(bool)
        projected &= clip["images"][slot]["shadow_target_image"] != 1
        masks[slot] = projected
        details[str(slot)] = {**detail, "image_negative_pixels": int(projected.sum()),
            "source": "observed_lit_receiver_union_v2", "weak_not_shadow_gt": True,
            "independent": independent_detail,
            "support": "aligned observed receivers and BEV validity; legacy neighbour branch additionally uses flow support"}
    return masks, {"reason": "automatic_physical_lit_receivers", "frames": details,
        "scope": "weak negatives outside modelled caster projections, not exhaustive unmodelled static-caster exclusion"}


def build_dataset(runs_path, output_root, negative_manifests=(), size=(384, 704), freeze_lock=None, auto_negatives=False,
                  write_manual_template=False, negative_weight=0.5):
    if not np.isfinite(negative_weight) or not 0 < negative_weight <= 1:
        raise ValueError("Weak-negative loss coefficient must be in (0, 1]")
    runs_path = Path(runs_path).resolve()
    runs = read_rows(runs_path)
    if not runs:
        raise ValueError("Supply frozen run rows")
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    for run in runs:
        root = resolve_path(run["run_root"], runs_path.parent)
        if output == root or root in output.parents:
            raise ValueError("Export outside frozen runs")
    development = set(json.loads(Path(freeze_lock).read_text(encoding="utf-8"))["excluded_development_scenes"]) if freeze_lock else set()
    negatives = negative_index(negative_manifests)
    output.mkdir(parents=True, exist_ok=True)
    staging = output/".merge"
    staging.mkdir()
    records, scene_roles, input_hashes = {}, {}, {str(runs_path): digest(runs_path)}
    for run in runs:
        method, split = simple_name(run["method"]), run["split"]
        if split not in {"train", "dev", "eval"}:
            raise ValueError("Declare scene-disjoint train/dev/eval roles before export")
        root = resolve_path(run["run_root"], runs_path.parent)
        if output == root or root in output.parents:
            raise ValueError("Export outside frozen runs")
        pair_paths = [resolve_path(path, runs_path.parent) for path in run["pairs"]]
        cameras = camera_index(pair_paths)
        official = {}
        for path in pair_paths:
            for pair in read_rows(path):
                identity = pair["scene_name"], pair["camera"]
                if official.setdefault(identity, pair["official_split"]) != pair["official_split"]:
                    raise ValueError("Conflicting official scene splits")
        input_hashes.update({str(path): digest(path) for path in pair_paths})
        for key, entry in find_clips(root).items():
            scene, camera, tokens = key
            if split == "eval" and scene in development:
                raise ValueError("Independent eval cannot reuse teacher-development scenes")
            official_split = official[(scene, camera)]
            if split in {"train", "dev"} and official_split not in {"train", "mini_train"}:
                raise ValueError("Official validation/test scenes cannot become training or model-selection data")
            if scene_roles.setdefault(scene, split) != split:
                raise ValueError(f"Scene leaks across dataset roles: {scene}")
            clip = load_clip(entry)
            input_hashes.update(clip["fingerprints"])
            automatic, automatic_details = ({}, {"frames": {}})
            if auto_negatives:
                automatic, automatic_details = automatic_lit_negatives(clip, [cameras[(scene, camera, token)] for token in tokens])
            for slot, token in enumerate(tokens):
                frame, base = cameras[(scene, camera, token)]
                if frame["camera_timestamp"] != clip["rows"][slot]["timestamp"]:
                    raise ValueError("Camera timestamps differ from frozen outputs")
                saved = clip["images"][slot]["shadow_target_image"]
                if not np.isin(saved, [1, 255]).all():
                    raise ValueError("Video teacher contract is positive/IGNORE; use explicit negative evidence")
                score, score_info = projected_score(clip["arrays"][slot], frame, clip["protocol"]["ground_plane_world"])
                identity = (method, split, scene, camera, token)
                image = resolve_path(frame["image"], base)
                if str(image) not in input_hashes:
                    input_hashes[str(image)] = digest(image)
                if frame.get("image_sha256") and input_hashes[str(image)] != frame["image_sha256"]:
                    raise ValueError("Source RGB changed since frozen pair export")
                shape = (frame["image_size_wh"][1], frame["image_size_wh"][0])
                if saved.shape != shape:
                    raise ValueError("Frozen image target must use calibrated RGB coordinates")
                source = {"run_root": str(root), "clip_id": clip["protocol"]["clip"]["id"],
                    "protocol_sha256": digest(clip["folder"]/"protocol.json"), "score_field": score_info["field"] if score_info else None,
                    "frame_evidence": str(clip["folder"]/f"{slot:03d}.npz"),
                    "image_evidence": str(clip["folder"]/f"{slot:03d}_image.npz"),
                    "frame_manifest": str(clip["folder"]/"frames.jsonl"), "frame_slot": slot,
                    "observed_region_ids": [row["id"] for row in clip["rows"][slot]["regions"]],
                    "automatic_negative_evidence": automatic_details["frames"].get(str(slot), {"reason": automatic_details.get("reason", "disabled")})}
                frame_negative = automatic.get(slot, np.zeros(shape, bool))
                if identity not in records:
                    merged = {"positive": saved == 1, "score": score if score is not None else np.zeros(shape, np.float32),
                              "score_support": score_info["support"] if score_info else np.zeros(shape, bool), "negative": frame_negative}
                    records[identity] = {"image": image,
                        "timestamp": frame["camera_timestamp"], "sources": [source], "all_scores_available": score is not None,
                        "official_split": official_split,
                        "staging": staging/(hashlib.sha256(json.dumps(identity).encode()).hexdigest()+".npz")}
                else:
                    previous = records[identity]
                    if image != previous["image"] or previous["timestamp"] != frame["camera_timestamp"]:
                        raise ValueError("Overlapping clips disagree on frame identity")
                    with np.load(previous["staging"], allow_pickle=False) as archive:
                        merged = {key: archive[key] for key in archive.files}
                    merged["positive"] |= saved == 1
                    merged["negative"] |= frame_negative
                    previous["all_scores_available"] &= score is not None
                    if score is not None:
                        merged["score"] = np.maximum(merged["score"], score)
                        merged["score_support"] |= score_info["support"]
                    previous["sources"].append(source)
                np.savez_compressed(records[identity]["staging"], **merged)
    rows, predictions, automatic_manifests = defaultdict(list), defaultdict(list), defaultdict(list)
    counts = defaultdict(Counter)
    scene_counts = defaultdict(Counter)
    for (method, split, scene, camera, token), record in sorted(records.items()):
        name = frame_id(scene, camera, token)
        with np.load(record["staging"], allow_pickle=False) as archive:
            merged = {key: archive[key] for key in archive.files}
        positive = merged["positive"]
        negative = merged["negative"]
        automatic_negative = negative.copy()
        provenance = []
        evidence = negatives.get((scene, camera, token), [])+negatives.get((scene, camera, "image:"+str(record["image"])), [])
        for reference, source in evidence:
            mask = load_array(reference, output)
            if not np.isin(mask, [0, 1, 255]).all():
                raise ValueError("Negative evidence must retain 0/1/255 encoding")
            if source["expected_target_sha256"] and digest(reference["path"]) != source["expected_target_sha256"]:
                raise ValueError("Physical negative trimap changed since teacher export")
            input_hashes[source["manifest"]] = source["manifest_sha256"]
            if reference["path"] not in input_hashes:
                input_hashes[reference["path"]] = digest(reference["path"])
            mask = mask == 0
            if mask.shape != positive.shape:
                raise ValueError("Negative evidence uses different image coordinates")
            negative |= mask
            provenance.append(source)
        conflict = positive & negative
        target = np.full(positive.shape, 255, np.uint8)
        target[positive & ~negative] = 1
        target[negative & ~positive] = 0
        target_path = output/method/"targets"/f"{name}.png"
        checked_write_image(target_path, target)
        row = {"id": name, "frame_token": token, "scene_name": scene, "camera": camera,
            "timestamp": record["timestamp"], "image": str(record["image"]), "target": str(target_path),
            "official_split": record["official_split"],
            "split": split, "label_source": "temporal_weak", "mask_encoding": "trimap",
            "method": method, "sources": record["sources"], "negative_evidence": provenance,
            "conflicting_pixels_ignored": int(conflict.sum()), "negative_weight": negative_weight}
        rows[(method, split)].append(row)
        if automatic_negative.any():
            automatic_path = output/method/"automatic_negative_targets"/f"{name}.png"
            automatic_target = np.full(target.shape, 255, np.uint8)
            automatic_target[automatic_negative & (target == 0)] = 0
            checked_write_image(automatic_path, automatic_target)
            automatic_manifests[method].append({"id": name, "scene_name": scene, "camera": camera,
                "frame_token": token, "image": str(record["image"]), "split": split,
                "shadow_mask": str(automatic_path), "mask_encoding": "trimap", "label_source": "temporal_weak",
                "physics_export_enabled": True, "physics_protocol": "video-lit-receiver-negatives-v2",
                "physics_target_sha256": digest(automatic_path), "sources": record["sources"]})
        frozen_prediction = output/method/"discovery"/f"{name}.npz"
        frozen_prediction.parent.mkdir(parents=True, exist_ok=True)
        payload = {"prediction": positive, "score_support": merged["score_support"]}
        if record["all_scores_available"]:
            payload["probability"] = merged["score"]  # Ranking evidence, not calibrated probability.
        np.savez_compressed(frozen_prediction, **payload)
        predictions[(method, split)].append({"id": name, "frame_token": token, "scene_name": scene, "camera": camera,
            "prediction": {"path": str(frozen_prediction), "key": "prediction"},
            "probability": str(frozen_prediction) if record["all_scores_available"] else None,
            "score_support": {"path": str(frozen_prediction), "key": "score_support"},
            "prediction_sha256": digest(frozen_prediction), "scores_are_probabilities": False})
        resized = cv2.resize(target, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        row["resized_class_pixels"] = {label: int((resized == value).sum())
            for value, label in ((0, "negative"), (1, "positive"), (255, "ignore"))}
        row["resized_positive_image_fraction"] = float((resized == 1).mean())
        row["resized_ignore_image_fraction"] = float((resized == 255).mean())
        for value, label in ((0, "negative"), (1, "positive"), (255, "ignore")):
            counts[(method, split)][label+"_pixels"] += int((target == value).sum())
            counts[(method, split)]["resized_"+label+"_pixels"] += int((resized == value).sum())
            scene_counts[(method, split, scene)][label+"_pixels"] += int((target == value).sum())
            scene_counts[(method, split, scene)]["resized_"+label+"_pixels"] += int((resized == value).sum())
        scene_counts[(method, split, scene)]["frames"] += 1
        scene_counts[(method, split, scene)]["frames_with_positive"] += int((target == 1).any())
        scene_counts[(method, split, scene)]["frames_with_negative"] += int((target == 0).any())
        counts[(method, split)]["frames"] += 1
        counts[(method, split)]["frames_with_positive"] += int((resized == 1).any())
        counts[(method, split)]["frames_with_negative"] += int((resized == 0).any())
        counts[(method, split)]["positive_only_frames"] += int((resized == 1).any() and not (resized == 0).any())
        counts[(method, split)]["conflict_pixels"] += int(conflict.sum())
        record["staging"].unlink()
    staging.rmdir()
    summary = {"runs_manifest_sha256": digest(runs_path), "size_hw": list(size), "methods": {},
        "data_adapter_sha256": digest(__file__),
        "negative_rule_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/bev_teacher.py"),
        "scene_roles": scene_roles, "training_performed": False, "unlabeled_is_background": False,
        "development_scenes": sorted(development), "freeze_lock_sha256": digest(freeze_lock) if freeze_lock else None,
        "independent_negative_rule_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/receiver_negatives.py"),
        "automatic_negative_rule": "observed_lit_receiver_union_v2" if auto_negatives else None,
        "negative_loss_weight": negative_weight,
        "negative_weight_contract": "weak-negative reliability; class-balanced loss normalizes within each class, legacy loss applies an absolute coefficient",
        "overlapping_clip_merge": "union of positive observations per camera frame; conflicting negative evidence -> IGNORE"}
    for method in sorted({key[0] for key in rows}):
        write_rows(output/method/"automatic_negatives.jsonl", automatic_manifests[method])
        details = {}
        for split in ("train", "dev", "eval"):
            group = rows.get((method, split), [])
            write_rows(output/method/f"seg_{split}.jsonl", group)
            write_rows(output/method/f"discovery_{split}.jsonl", predictions.get((method, split), []))
            detail = dict(counts[(method, split)])
            detail["scenes"] = sorted({row["scene_name"] for row in group})
            detail["negative_scenes"] = sorted({row["scene_name"] for row in group if row["resized_class_pixels"]["negative"] > 0})
            detail["both_classes_survive_resize"] = all(detail.get("resized_"+label+"_pixels", 0) > 0 for label in ("positive", "negative"))
            details[split] = detail
        summary["methods"][method] = details
    evaluation_frames = {}
    for (method, split), group in rows.items():
        if split == "eval":
            for row in group:
                evaluation_frames[row["id"]] = {key: row[key] for key in ("id", "frame_token", "scene_name", "camera", "image")}
    if write_manual_template:
        write_rows(output/"manual_eval_template.jsonl", ({**row, "target": None, "mask_encoding": "trimap",
            "label_source": "manual_shadow_gt", "split": "eval", "independent": False, "annotation_status": "unannotated"}
            for _, row in sorted(evaluation_frames.items())))
    write_json(output/"dataset_summary.json", summary)
    write_json(output/"annotation_summary.json", summary)
    write_json(output/"class_distribution.json", {"size_hw": list(size), "methods": summary["methods"],
        "per_scene": [{"method": method, "split": split, "scene_name": scene, **dict(values)}
            for (method, split, scene), values in sorted(scene_counts.items())]})
    write_json(output/"dataset_manifest.json", {"scene_roles": scene_roles, "deduplication": "method/split/scene/camera/frame_token",
        "input_fingerprints": input_hashes, "freeze_lock_sha256": summary["freeze_lock_sha256"],
        "manifests": {str(output/method/f"seg_{split}.jsonl"): digest(output/method/f"seg_{split}.jsonl")
            for method in summary["methods"] for split in ("train", "dev", "eval")}})
    write_rows(output/"scene_dataset_counts.jsonl", ({"method": method, "split": split, "scene_name": scene, **dict(values)}
        for (method, split, scene), values in sorted(scene_counts.items())))
    write_json(output/"input_fingerprints.json", input_hashes)
    return summary


def prepare_suite(dataset_root, methods, output_root, size=(384, 704), epochs=10, batch_size=2, accumulation=4, seed=0,
                  backbone_weights="imagenet", init_checkpoint=None, reference_method=None, scene_sampling="supervision", skip_untrainable=False):
    if scene_sampling not in {"image", "uniform", "supervision"}:
        raise ValueError("Choose image, uniform or supervision scene sampling")
    root, output = Path(dataset_root).resolve(), Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    methods = [simple_name(method) for method in methods]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Choose unique existing method datasets; absent baselines are not synthesized")
    if min(*size, epochs, batch_size, accumulation) < 1:
        raise ValueError("Training dimensions and budget must be positive")
    availability = {method: {"method": method, "status": "available", "reasons": []} for method in methods}
    requested_methods = list(methods)
    reference_method = reference_method or ("anchor_first_v8plus" if "anchor_first_v8plus" in methods else methods[-1])
    if skip_untrainable:
        available = []
        for method in methods:
            reasons = []
            for split in ("train", "dev"):
                path = root/method/f"seg_{split}.jsonl"
                if not path.is_file():
                    reasons.append(split+": missing actual method manifest")
                    continue
                totals = np.zeros(2, np.int64)
                for row in read_rows(path):
                    mask = cv2.resize(load_array(row["target"], root), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
                    totals += [(mask == 0).sum(), (mask == 1).sum()]
                if not totals.all():
                    reasons.append(split+": both known classes do not survive resize")
            availability[method].update(status="untrainable" if reasons else "available", reasons=reasons)
            if not reasons:
                available.append(method)
        output.mkdir(parents=True)
        write_json(output/"method_availability.json", list(availability.values()))
        methods = available
        if not methods:
            raise ValueError("No requested method has real two-class train/dev supervision")
    if backbone_weights not in {"none", "imagenet"}:
        raise ValueError("Use an existing supported backbone initialization")
    initialization = str(Path(init_checkpoint).resolve()) if init_checkpoint else None
    initialization_sha = digest(initialization) if initialization else None
    if reference_method not in requested_methods:
        raise ValueError("Choose a reference teacher from the actual supplied methods")
    reference_path = root/reference_method/"seg_eval.jsonl"
    if not reference_path.is_file() or not read_rows(reference_path):
        raise ValueError("Held-out teacher evaluation needs an actual nonempty eval manifest")
    indexed, class_pixels = {}, {}
    for split in ("train", "dev"):
        for method in methods:
            group = read_rows(root/method/f"seg_{split}.jsonl")
            indexed[(method, split)] = {row["id"]: row for row in group}
            if len(indexed[(method, split)]) != len(group):
                raise ValueError("Duplicate observation IDs in a method dataset")
            for identity, row in indexed[(method, split)].items():
                if row["label_source"] != "temporal_weak":
                    raise ValueError("This weak-supervision suite never imports manual test labels")
                mask = load_array(row["target"], root)
                resized = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
                class_pixels[(method, split, identity)] = np.array([(resized == 0).sum(), (resized == 1).sum()])
    priority = [reference_method]+[method for method in requested_methods if method != reference_method]
    alignment_drops = []
    while True:
        common, failures = {}, []
        for split in ("train", "dev"):
            usable = [{identity for identity in indexed[(method, split)]
                       if class_pixels[(method, split, identity)].sum()} for method in methods]
            common[split] = set.intersection(*usable)
            for method in methods:
                totals = sum((class_pixels[(method, split, identity)] for identity in common[split]), np.zeros(2, np.int64))
                if not totals.all():
                    failures.append(f"{method}/{split} lacks reliable positive or negative supervision after paired-image alignment")
        if not failures:
            break
        if not skip_untrainable or len(methods) == 1:
            raise ValueError("; ".join(failures))
        # Keep the reference teacher arm first; relax comparisons, never fabricate labels.
        removed = max(methods, key=priority.index)
        methods.remove(removed)
        availability[removed].update(status="alignment_incompatible", reasons=failures,
            scope="paired suite; this method may still be trainable independently")
        alignment_drops.append(removed)
        write_json(output/"method_availability.json", list(availability.values()))
    aligned, scene_sets = {}, {}
    for split in ("train", "dev"):
        for identity in common[split]:
            identities = {(indexed[(method, split)][identity]["scene_name"], indexed[(method, split)][identity]["frame_token"],
                           indexed[(method, split)][identity]["image"]) for method in methods}
            if len(identities) != 1:
                raise ValueError("Shared IDs do not identify the same RGB observations")
        for method in methods:
            aligned[(method, split)] = [indexed[(method, split)][identity] for identity in sorted(common[split])]
        scene_sets[split] = {row["scene_name"] for row in aligned[(methods[0], split)]}
    if scene_sets["train"] & scene_sets["dev"]:
        raise ValueError("Training and dev scenes overlap")
    output.mkdir(parents=True, exist_ok=True)
    arms = []
    for method in methods:
        for split in ("train", "dev"):
            write_rows(output/method/f"seg_{split}.jsonl", aligned[(method, split)])
        arms.append({"method": method, "train": str(output/method/"seg_train.jsonl"), "dev": str(output/method/"seg_dev.jsonl")})
    plan = {"arms": arms, "size_hw": list(size), "epochs": epochs, "batch_size": batch_size,
        "accumulation": accumulation, "seed": seed, "lr": 1e-4, "backbone_weights": backbone_weights,
        "init_checkpoint": initialization, "init_checkpoint_sha256": initialization_sha,
        "temporal_weight": 0.0, "receiver_pair_weight": 0.0, "weak_weight": 1.0,
        "scene_sampling": scene_sampling, "pixel_balance": "loss", "dice_weight": 0.5,
        "loss_mode": "class-balanced", "selection_metric": "balanced_accuracy",
        "requested_methods": requested_methods, "method_availability": list(availability.values()),
        "alignment_dropped_methods": alignment_drops, "alignment_priority": priority,
        "scene_sets": {key: sorted(value) for key, value in scene_sets.items()},
        "aligned_frames": {split: len(aligned[(methods[0], split)]) for split in ("train", "dev")},
        "budget": "same eligible image IDs, sampler policy/draw count, size, epochs, batch/accumulation and seed; supervision-dependent draws may differ between weak trimaps",
        "architecture": "DeepLabV3-MobileNetV3", "evaluation_gt_used_for_training": False,
        "weak_eval_reference": str(root/reference_method/"seg_eval.jsonl"),
        "weak_eval_reference_sha256": digest(root/reference_method/"seg_eval.jsonl"),
        "weak_eval_reference_method": reference_method,
        "dataset_summary": str(root/"dataset_summary.json"),
        "manifest_sha256": {str(output/method/f"seg_{split}.jsonl"): digest(output/method/f"seg_{split}.jsonl")
                            for method in methods for split in ("train", "dev")}}
    write_json(output/"suite.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--runs", required=True)
    export.add_argument("--negative-manifests", nargs="*", default=[])
    export.add_argument("--output-root", required=True)
    export.add_argument("--size", nargs=2, type=int, default=[384, 704])
    export.add_argument("--freeze-lock")
    export.add_argument("--auto-negatives", action="store_true", help="Generate physical weak negatives automatically from saved observations")
    export.add_argument("--negative-weight", type=float, default=0.5, help="Weak-negative loss coefficient, not a probability")
    export.add_argument("--manual-eval-template", action="store_true", help="Optional separate manual-evaluation task list; never required for training")
    suite = sub.add_parser("suite")
    suite.add_argument("--dataset-root", required=True)
    suite.add_argument("--methods", nargs="+", required=True)
    suite.add_argument("--output-root", required=True)
    suite.add_argument("--size", nargs=2, type=int, default=[384, 704])
    suite.add_argument("--epochs", type=int, default=10)
    suite.add_argument("--batch-size", type=int, default=2)
    suite.add_argument("--accumulation", type=int, default=4)
    suite.add_argument("--seed", type=int, default=0)
    suite.add_argument("--backbone-weights", choices=["none", "imagenet"], default="imagenet")
    suite.add_argument("--init-checkpoint")
    suite.add_argument("--reference-method", help="One frozen teacher reference shared by all held-out agreement evaluations")
    suite.add_argument("--scene-sampling", choices=["uniform", "image", "supervision"], default="supervision")
    suite.add_argument("--skip-untrainable", action="store_true", help="Report absent/one-class arms without fabricating labels")
    args = parser.parse_args()
    if min(args.size) < 1:
        raise ValueError("Image size must be positive")
    if args.command == "export":
        result = build_dataset(args.runs, args.output_root, args.negative_manifests, args.size, args.freeze_lock, args.auto_negatives, args.manual_eval_template, args.negative_weight)
    else:
        result = prepare_suite(args.dataset_root, args.methods, args.output_root, args.size, args.epochs, args.batch_size, args.accumulation, args.seed,
                               args.backbone_weights, args.init_checkpoint, args.reference_method, args.scene_sampling, args.skip_untrainable)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
