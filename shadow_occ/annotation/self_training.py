from __future__ import annotations

import math

import numpy as np

from shadow_occ.annotation.data import GT_SOURCES
from shadow_occ.annotation.diagnostics import SCORING_VERSION, classify, decision
from shadow_occ.annotation.joint import annotation, bbox_iou, validate
from shadow_occ.annotation.supervision import visible_target
from shadow_occ.annotation.temporal import warp


POLICY_VERSION = "shadow-self-training-pilot-v1"


def score_vote(row, policy):
    calibration = policy["decision_policy"]
    label = classify(row["score"], calibration["threshold"], calibration["abstain_margin"],
                     row.get("uncertainty_score", 0.0), calibration["max_uncertainty"])
    if label == "YES" and row["score"] < policy["positive_cutoff"]:
        return "UNCERTAIN"
    if label == "NO" and (not policy["allow_negative_pseudo_labels"] or row["score"] > policy["negative_cutoff"]):
        return "UNCERTAIN"
    return label


def make_policy(calibration, rows, *, minimum_precision=0.9, minimum_recall=0.2,
                minimum_positives=20, minimum_scenes=3, positive_cutoff=0.9,
                negative_cutoff=0.1, allow_negative=False):
    if not rows or any(row.get("split") != "dev" or row.get("official_split") in {"val", "mini_val", "eval", "test"} for row in rows):
        raise ValueError("Self-training policy requires train-derived dev scores only")
    if calibration.get("fitted_split") != "dev" or calibration.get("scoring_version") != SCORING_VERSION:
        raise ValueError("Need a current frozen dev calibration")
    if any(row.get("scorer_id") != calibration["scorer_id"] or row.get("scoring_version") != SCORING_VERSION or row.get("in_sample_annotation") for row in rows):
        raise ValueError("Policy scorer mismatch or in-sample teacher scores")
    if not (0.5 < positive_cutoff <= 1 and 0 <= negative_cutoff < 0.5 and 0 < minimum_precision <= 1 and 0 < minimum_recall <= 1):
        raise ValueError("Invalid self-training decision gates")
    if minimum_positives < 1 or minimum_scenes < 2:
        raise ValueError("Require independent positive evidence, not a single scene")
    policy = {"version": POLICY_VERSION, "scorer_id": calibration["scorer_id"], "decision_policy": calibration,
              "dev_scenes": sorted({row["scene_name"] for row in rows}), "positive_cutoff": positive_cutoff,
              "negative_cutoff": negative_cutoff, "allow_negative_pseudo_labels": allow_negative,
              "maximum_pseudo_ratio": 0.5, "maximum_per_track": 1, "maximum_per_scene": 10,
              "pixel_threshold": 0.8, "minimum_pixels": 64, "minimum_ground_fraction": 0.8,
              "minimum_pixel_agreement": 0.8, "minimum_new_fraction": 0.5, "minimum_motion_advantage": 0.05,
              "require_temporal_consistency": True, "pseudo_loss_weight": 0.25,
              "round": 1, "approved_for_production": False, "scope": "offline cross-fit pilot, not EMA/FixMatch reproduction",
              "teacher_gate_requirements": {"minimum_precision": minimum_precision, "minimum_recall": minimum_recall,
                                            "minimum_positives": minimum_positives, "minimum_scenes": minimum_scenes}}
    positives = [row for row in rows if decision(row) == "YES"]
    accepted = [row for row in rows if decision(row) in {"YES", "NO"} and score_vote(row, policy) == "YES"]
    correct = [row for row in accepted if decision(row) == "YES"]
    precision = len(correct)/len(accepted) if accepted else None
    recall = len(correct)/len(positives) if positives else 0.0
    reasons = []
    if precision is None or precision < minimum_precision:
        reasons.append("insufficient_dev_score_gate_precision")
    if recall < minimum_recall:
        reasons.append("insufficient_dev_positive_recall")
    if len(correct) < minimum_positives:
        reasons.append("too_few_accepted_dev_positives")
    if len({row["scene_name"] for row in correct}) < minimum_scenes:
        reasons.append("too_few_independent_positive_scenes")
    negative = [row for row in rows if decision(row) in {"YES", "NO"} and score_vote(row, policy) == "NO"]
    negative_correct = [row for row in negative if decision(row) == "NO"]
    if allow_negative and (len(negative_correct) < minimum_positives or not negative or len(negative_correct)/len(negative) < minimum_precision):
        reasons.append("negative_pseudo_label_gate_failed")
    policy.update(eligible_for_pilot=not reasons, blocking_reasons=reasons,
                  supervision_scope="gt" if all(row.get("shadow_label_source") in GT_SOURCES for row in rows if decision(row) in {"YES", "NO"}) else "weak",
                  dev_score_gate={"precision": precision, "positive_recall": recall, "correct_accepted_positives": len(correct),
                                  "positive_scenes": len({row["scene_name"] for row in correct}),
                                  "warning": "Score-gate agreement only, not precision of the final multi-evidence selected set"})
    return policy


def validate_policy(policy):
    if policy.get("version") != POLICY_VERSION or policy.get("round") != 1 or policy.get("approved_for_production") is not False:
        raise ValueError("Only a current one-round experimental policy is supported")
    if policy.get("require_temporal_consistency") is not True:
        raise ValueError("This pilot requires temporal teacher agreement")
    for key in ("maximum_pseudo_ratio", "minimum_ground_fraction", "minimum_pixel_agreement", "minimum_new_fraction", "pseudo_loss_weight"):
        if not math.isfinite(policy[key]) or not 0 < policy[key] <= 1:
            raise ValueError(f"Invalid policy bound: {key}")
    if not 0.5 < policy["pixel_threshold"] <= 1 or not 0.5 < policy["positive_cutoff"] <= 1 or not 0 <= policy["negative_cutoff"] < 0.5:
        raise ValueError("Invalid frozen score/pixel cutoffs")
    if any(not isinstance(policy[key], int) or policy[key] < 1 for key in ("minimum_pixels", "maximum_per_track", "maximum_per_scene")):
        raise ValueError("Invalid diversity/pixel limits")
    if policy["decision_policy"].get("scorer_id") != policy["scorer_id"] or policy["decision_policy"].get("fitted_split") != "dev":
        raise ValueError("Self-training policy and frozen dev scorer differ")


def teacher_gate(score, generated, policy):
    if score.get("scoring_version") != SCORING_VERSION:
        return "obsolete_or_missing_score_version", None
    if score.get("split") != "train" or score.get("official_split") in {"val", "mini_val", "eval", "test"}:
        return "not_training_pool", None
    if score.get("view_mode", "full") != "full" or score.get("self_training"):
        return "ablation_or_recursive_pseudo_input", None
    if score.get("annotation_target") is not None or score.get("answer") is not None:
        return "already_labeled_candidate", None
    if score.get("in_sample_annotation") is not False or score.get("scorer_id") != policy["scorer_id"]:
        return "teacher_not_heldout_or_wrong_model", None
    for name in ("scorer_id", "input_content_sha256", "region_mask_sha256"):
        if not score.get(name) or score.get(name) != generated.get(name):
            return "generation_score_input_mismatch", None
    prediction = generated.get("prediction")
    try:
        validate(prediction, score.get("candidates", []))
    except (ValueError, KeyError, TypeError):
        return "parse_error", None
    vote = score_vote(score, policy)
    if vote == "UNCERTAIN" or generated.get("parse_error") or prediction["shadow"] != vote:
        return "low_score_or_generation_disagreement", None
    return None, vote


def temporal_agreement(region, current, previous, ground, previous_ground, temporal, label, policy):
    if not temporal.get("lighting_comparable"):
        return {"accepted": False, "reason": "lighting_or_ground_overlap_unverified"}
    shape = region.shape
    if current.shape != shape or ground.shape != shape or previous.shape != previous_ground.shape:
        raise ValueError("Temporal probability/ground coordinates disagree")
    mode = temporal.get("mode")
    if mode not in {"static", "moving"}:
        return {"accepted": False, "reason": "unknown_motion_mode"}
    matrix = temporal["world_previous_to_current"] if mode == "static" or label == "NO" else temporal["caster_previous_to_current"]
    for value in (matrix, temporal["world_previous_to_current"]):
        value = np.asarray(value)
        if value.shape != (3, 3) or not np.isfinite(value).all() or abs(np.linalg.det(value)) < 1e-12:
            raise ValueError("Invalid temporal teacher homography")
    aligned = warp(previous.astype(np.float32), matrix, shape)
    visible = warp(previous_ground.astype(np.uint8), matrix, shape, nearest=True) > 0
    valid = region & ground & visible
    count = int(valid.sum())
    if count < policy["minimum_pixels"]:
        return {"accepted": False, "reason": "insufficient_temporal_region_overlap", "pixels": count}
    threshold = policy["pixel_threshold"]
    supported = (aligned >= threshold) & (current >= threshold) if label == "YES" else (aligned <= 1-threshold) & (current <= 1-threshold)
    agreement = float(supported[valid].mean())
    result = {"accepted": agreement >= policy["minimum_pixel_agreement"], "agreement": agreement, "pixels": count, "mode": mode}
    if mode == "moving" and label == "YES":
        world = warp(previous.astype(np.float32), temporal["world_previous_to_current"], shape)
        world_visible = warp(previous_ground.astype(np.uint8), temporal["world_previous_to_current"], shape, nearest=True) > 0
        comparison = valid & world_visible
        advantage = float((aligned[comparison]-world[comparison]).mean()) if comparison.any() else None
        result["motion_advantage"] = advantage
        result["accepted"] = bool(result["accepted"] and comparison.sum() >= policy["minimum_pixels"] and advantage is not None and advantage >= policy["minimum_motion_advantage"])
    if not result["accepted"]:
        result["reason"] = "temporal_teacher_disagreement"
    return result


def pseudo_annotation(row, prediction, label, outside_support):
    if label == "NO":
        return annotation("NO", "none"), False, "unresolved"
    expected = visible_target(row)
    if expected["caster"]["state"] == "visible" and prediction["caster"]["state"] == "visible":
        if bbox_iou(expected["caster"]["bbox_1000_context"], prediction["caster"].get("bbox_1000_context")) >= 0.5 and expected["caster"]["category"] == prediction["caster"]["category"]:
            return expected, True, row["association_source"]
    if outside_support > 0 and prediction["caster"]["state"] == "outside_bev" and row.get("association_source") in {"geometry_weak", "temporal_weak"}:
        selected = prediction["caster"]["outside_candidate_id"]
        if selected in row.get("valid_candidate_ids", []):
            return annotation("YES", "outside_bev", candidate_id=selected), True, row["association_source"]
    # Teach shadow existence without forcing the student to learn a false refusal.
    return annotation("YES", "unresolved"), False, "unresolved"
