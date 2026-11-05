from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import GT_SOURCES, assert_disjoint, checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.diagnostics import decision
from shadow_occ.annotation.self_training import pseudo_annotation, teacher_gate, temporal_agreement, validate_policy
from shadow_occ.annotation.teacher_provenance import cached_digest, input_provenance
from shadow_occ.annotation.readiness import require_prediction_for_proposals
from shadow_occ.utils.io import write_json


def indexed(path):
    rows = read_rows(path)
    result = {row["id"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows, result


def protocol_for(path, data_path, hash_key):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get(hash_key) != digest(data_path):
        raise ValueError(f"Stale/missing provenance: {path}/{hash_key}; rerun teacher inference")
    if value.get("in_sample_annotation"):
        raise ValueError("In-sample teacher predictions cannot create cross-fit labels")
    return value


def main():
    parser = argparse.ArgumentParser("Build ONE anchored offline self-training pilot round; dry-run by default.")
    parser.add_argument("--anchor-train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--seg-anchor-train", required=True)
    parser.add_argument("--seg-dev", required=True)
    parser.add_argument("--teacher-scores", required=True)
    parser.add_argument("--score-protocol", required=True)
    parser.add_argument("--teacher-annotations", required=True)
    parser.add_argument("--annotation-protocol", required=True)
    parser.add_argument("--seg-predictions", required=True)
    parser.add_argument("--seg-metrics", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["dry-run", "build"], default="dry-run")
    parser.add_argument("--allow-weak-pilot", action="store_true", help="Explicit consent to a weakly evaluated experiment, not real shadow-GT verification")
    parser.add_argument("--cache-parents", type=int, default=8, help="Bound full-resolution arrays held in CPU memory")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.cache_parents < 2:
        raise ValueError("cache-parents must be at least two for temporal pairing")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    validate_policy(policy)
    if args.mode == "build" and policy.get("eligible_for_pilot") is not True:
        raise ValueError("Teacher quality gate failed; use dry-run for diagnostics, not label expansion")
    if args.mode == "build" and policy.get("supervision_scope") != "gt" and not args.allow_weak_pilot:
        raise ValueError("Weak-label pilots require --allow-weak-pilot; this is not GT validation")
    anchors, anchor_ids = indexed(args.anchor_train)
    dev, _ = indexed(args.dev)
    seg_anchors, seg_anchor_ids = indexed(args.seg_anchor_train)
    seg_dev, _ = indexed(args.seg_dev)
    scores, _ = indexed(args.teacher_scores)
    _, generated = indexed(args.teacher_annotations)
    _, segments = indexed(args.seg_predictions)
    if args.mode == "build":
        for row in segments.values():
            require_prediction_for_proposals(row)
    if not anchors or not seg_anchors or not dev or not seg_dev:
        raise ValueError("Original supervision and disjoint dev must remain present")
    if any(row.get("self_training") for row in anchors+seg_anchors):
        raise ValueError("Do not recursively self-train; re-evaluate new teachers before designing another round")
    assert_disjoint(anchors+seg_anchors+scores, dev+seg_dev)
    training_images = {str(Path(row["image"]).resolve()) for row in anchors+seg_anchors+scores if row.get("image")}
    dev_images = {str(Path(row["image"]).resolve()) for row in dev+seg_dev if row.get("image")}
    if training_images & dev_images:
        raise ValueError("Training/self-label pool and dev share source images")
    if any(row.get("split") != "dev" or row.get("official_split") in {"val", "mini_val", "eval", "test"} for row in dev+seg_dev):
        raise ValueError("Student dev must remain train-derived and unchanged")
    if not set(policy["dev_scenes"]) <= {row["scene_name"] for row in dev}:
        raise ValueError("Policy was selected using another dev split")
    score_protocol = protocol_for(args.score_protocol, args.teacher_scores, "scores_sha256")
    generation_protocol = protocol_for(args.annotation_protocol, args.teacher_annotations, "annotations_sha256")
    segmentation_protocol = protocol_for(args.seg_metrics, args.seg_predictions, "predictions_sha256")
    for value in (score_protocol, generation_protocol):
        if value.get("scorer_id") != policy["scorer_id"]:
            raise ValueError("Policy/scoring/generation teachers differ")
    candidate_scenes = {row["scene_name"] for row in scores}
    for value, train_key, dev_key in ((score_protocol, "teacher_train_scenes", "teacher_dev_scenes"),
                                     (generation_protocol, "teacher_train_scenes", "teacher_dev_scenes"),
                                     (segmentation_protocol, "checkpoint_train_scenes", "checkpoint_dev_scenes")):
        if value.get(train_key) is None or value.get(dev_key) is None:
            raise ValueError("Teacher train/dev provenance is required; no in-sample fallback")
        if value.get("teacher_self_training_round", 0) != 0:
            raise ValueError("Recursive self-training teachers are blocked in this one-round pilot")
        if candidate_scenes & (set(value[train_key]) | set(value[dev_key])):
            raise ValueError("Teacher trained or selected checkpoints on pseudo-label candidate scenes")
    if any(row.get("split") != "train" or row.get("official_split") in {"val", "mini_val", "test", "eval"} for row in scores):
        raise ValueError("Only training-pool candidates may be self-labeled; eval/dev are forbidden")
    if set(generated) != {row["id"] for row in scores}:
        raise ValueError("Generation and score candidates must have identical unique IDs")
    counts, audit, selected = Counter(), [], []
    by_class, by_track, by_scene, frames, masks_seen = Counter(), Counter(), Counter(), set(), set()
    anchor_classes = Counter(decision(row) for row in anchors)
    maximum = int(len(anchors)*policy["maximum_pseudo_ratio"])
    cache, provenance_cache = OrderedDict(), {}

    def arrays(parent):
        if parent["id"] not in cache:
            for key, hash_key in (("image", "source_image_sha256"), ("target", "source_target_sha256"), ("ground_support", "source_ground_sha256")):
                if not parent.get(key) or parent.get(hash_key) != cached_digest(parent[key], provenance_cache):
                    raise ValueError("Segmentation source data changed or lacks provenance; rerun inference")
            if parent.get("probability_sha256") != cached_digest(parent["probability"], provenance_cache):
                raise ValueError("Teacher segmentation probabilities changed after inference")
            with np.load(parent["probability"], allow_pickle=False) as archive:
                probability = np.asarray(archive["probability"], np.float32)
            target = cv2.imread(parent["target"], cv2.IMREAD_GRAYSCALE)
            ground = cv2.imread(parent["ground_support"], cv2.IMREAD_GRAYSCALE) if parent.get("ground_support") else None
            if target is None or ground is None:
                raise ValueError("Self-labeling requires original target and observed-ground masks")
            if probability.shape != target.shape or ground.shape != target.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError("Invalid or misaligned teacher masks")
            if not np.isin(target, [0, 1, 255]).all():
                raise ValueError("Original target must preserve unknown pixels as 255")
            cache[parent["id"]] = probability, target, ground > 0
            if len(cache) > args.cache_parents:
                cache.popitem(last=False)
        cache.move_to_end(parent["id"])
        return cache[parent["id"]]

    for score in sorted(scores, key=lambda row: (-max(row["score"], 1-row["score"]), row["id"])):
        counts["candidates"] += 1
        reason, label = teacher_gate(score, generated[score["id"]], policy)
        parent = segments.get(score.get("parent_id"))
        if reason is None and (score["id"] in anchor_ids or parent is None or parent["id"] not in seg_anchor_ids):
            reason = "existing_annotation_or_missing_anchor_parent"
        if reason is not None:
            counts[reason] += 1
            audit.append({"id": score["id"], "reason": reason})
            continue
        if score["scene_name"] != parent["scene_name"] or score.get("image") != parent.get("image"):
            raise ValueError("Pseudo annotation and segmentation parent disagree")
        if input_provenance(score, provenance_cache) != {key: score.get(key) for key in ("input_content_sha256", "region_mask_sha256")}:
            raise ValueError("Teacher input images or proposal mask changed after inference")
        region = cv2.imread(score["region_mask"], cv2.IMREAD_GRAYSCALE)
        if region is None or digest(score["region_mask"]) != score["region_mask_sha256"]:
            raise ValueError("Candidate mask changed after teacher inference")
        probability, old, ground = arrays(parent)
        region = region > 0
        if region.shape != old.shape or not region.any():
            raise ValueError("Empty/misaligned candidate region")
        observed = region & ground
        supported = probability >= policy["pixel_threshold"] if label == "YES" else probability <= 1-policy["pixel_threshold"]
        usable = observed & supported & (old == 255)
        if parent["label_source"] in GT_SOURCES and np.any(region & (old == (0 if label == "YES" else 1))):
            reason = "conflicts_with_real_shadow_gt"
        elif observed.sum()/region.sum() < policy["minimum_ground_fraction"]:
            reason = "insufficient_observed_ground"
        elif not observed.any() or supported[observed].mean() < policy["minimum_pixel_agreement"]:
            reason = "segmentation_disagrees"
        elif usable.sum() < policy["minimum_pixels"] or usable.sum()/region.sum() < policy["minimum_new_fraction"]:
            reason = "insufficient_previously_unknown_pixels"
        temporal_report = None
        if reason is None and policy["require_temporal_consistency"]:
            temporal = score.get("temporal", {})
            previous = segments.get(temporal.get("previous_id"))
            if not score.get("camera") or not score.get("instance_token"):
                reason = "missing_temporal_track_identity"
            elif previous is None:
                reason = "missing_previous_teacher_prediction"
            elif previous["scene_name"] != score["scene_name"] or previous.get("camera") != score.get("camera") or previous.get("instance_token") != score.get("instance_token"):
                raise ValueError("Temporal teacher pair changes scene/camera/caster")
            else:
                previous_probability, _, previous_ground = arrays(previous)
                temporal_report = temporal_agreement(region, probability, previous_probability, ground, previous_ground, temporal, label, policy)
                if not temporal_report["accepted"]:
                    reason = temporal_report.get("reason", "temporal_disagreement")
        track = (score["scene_name"], score.get("instance_token") or score.get("sample_token") or parent["id"], label)
        frame = (score.get("image"), label)
        mask_key = (score.get("image"), score["region_mask_sha256"])
        if reason is None and (len(selected) >= maximum or by_class[label] >= int(anchor_classes[label]*policy["maximum_pseudo_ratio"])):
            reason = "anchor_relative_budget_reached"
        if reason is None and (by_track[track] >= policy["maximum_per_track"] or by_scene[score["scene_name"]] >= policy["maximum_per_scene"] or frame in frames or mask_key in masks_seen):
            reason = "duplicate_or_diversity_cap"
        if reason is not None:
            counts[reason] += 1
            audit.append({"id": score["id"], "reason": reason, "temporal": temporal_report})
            continue
        target, association_enabled, association_source = pseudo_annotation(score, generated[score["id"]]["prediction"], label,
                                                                         score_protocol.get("training_state_support", {}).get("outside_bev", 0))
        agreement_proxy = min(float(supported[observed].mean()), float(temporal_report["agreement"]) if temporal_report else 1.0,
                              score["score"] if label == "YES" else 1-score["score"])
        metadata = {"round": 1, "teacher_scorer_id": policy["scorer_id"], "policy_sha256": digest(args.policy),
                    "score": score["score"], "score_is_calibrated_probability": False, "temporal": temporal_report,
                    "agreement_weight_proxy": agreement_proxy, "weight_is_probability_of_correctness": False,
                    "annotation_source_id": score["id"], "pixel_source_parent": parent["id"], "ground_pixels": int(usable.sum())}
        pseudo = {**score, "id": "self1:"+score["id"], "annotation_target": target, "answer": json.dumps(target, separators=(",", ":")),
                  "label_source": "model_pseudo", "shadow_label_source": "model_pseudo", "association_source": association_source,
                  "association_supervision_enabled": association_enabled, "pseudo_loss_weight": policy["pseudo_loss_weight"]*agreement_proxy, "self_training": metadata}
        selected.append((pseudo, parent, np.flatnonzero(usable).astype(np.int32), usable.shape, label))
        by_class[label] += 1
        by_track[track] += 1
        by_scene[score["scene_name"]] += 1
        frames.add(frame)
        masks_seen.add(mask_key)
        counts["selected_candidates"] += 1
        audit.append({"id": score["id"], "reason": "selected_for_pilot" if args.mode == "build" else "would_select_under_rules", "temporal": temporal_report})
    write_rows(output/"selection_audit.jsonl", audit)
    report = {"mode": args.mode, "teacher_eligible": policy["eligible_for_pilot"], "teacher_blocking_reasons": policy.get("blocking_reasons", []), "student_data_written": False,
              "counts": dict(counts), "selected_by_class": dict(by_class), "anchors_retained": len(anchors),
              "legacy_teacher_history_assumptions": {"vlm": score_protocol.get("legacy_history_assumed_seed", False), "segmentation": segmentation_protocol.get("legacy_history_assumed_seed", False)},
              "scope": "Weak self-training pilot; not independent shadow precision or production approval"}
    if args.mode == "build":
        if not selected:
            report["student_data_written"] = False
            write_json(output/"self_training_protocol.json", report)
            raise ValueError("No eligible new labels; refusing an empty/no-op self-training round. Inspect the audit.")
        additions, grouped = [], {}
        for pseudo, parent, pixels, shape, label in selected:
            additions.append(pseudo)
            entry = grouped.setdefault(parent["id"], {"parent": parent, "pixels": [], "shape": shape, "rows": [], "loss_weights": []})
            entry["pixels"].append((pixels, label))
            entry["rows"].append(pseudo["id"])
            entry["loss_weights"].append(pseudo["pseudo_loss_weight"])
        seg_additions = []
        for identifier, entry in grouped.items():
            path = output/"pseudo_masks"/(hashlib.sha256(identifier.encode()).hexdigest()[:24]+".png")
            target = np.full(entry["shape"], 255, np.uint8)
            for pixels, label in entry["pixels"]:
                target.ravel()[pixels] = 1 if label == "YES" else 0
            checked_write_image(path, target)
            seg_additions.append({**entry["parent"], "id": "self1:"+identifier, "target": str(path), "label_source": "model_pseudo", "sample_weight": 1.0,
                                  "pseudo_loss_weight": min(entry["loss_weights"]), "self_training": {"round": 1, "source_annotations": entry["rows"], "policy_sha256": digest(args.policy)}})
            # The anchor cache weights refer to OLD labeled pixels, not this new mask.
            seg_additions[-1].pop("temporal", None)
            seg_additions[-1]["temporal_status"] = "teacher_filtered_no_student_temporal_cache"
        write_rows(output/"annotation_train.jsonl", anchors+additions)
        write_rows(output/"annotation_dev.jsonl", dev)
        write_rows(output/"seg_train.jsonl", seg_anchors+seg_additions)
        write_rows(output/"seg_dev.jsonl", seg_dev)
        report["student_data_written"] = True
        report["input_sha256"] = {name: digest(getattr(args, name)) for name in ("anchor_train", "dev", "seg_anchor_train", "seg_dev", "teacher_scores", "teacher_annotations", "seg_predictions", "policy")}
    write_json(output/"self_training_protocol.json", report)
    print(report)


if __name__ == "__main__":
    main()
