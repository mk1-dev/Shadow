from __future__ import annotations

from collections import Counter
import math

import numpy as np

from shadow_occ.annotation.data import WEAK_SOURCES
from shadow_occ.annotation.joint import parse
from shadow_occ.annotation.metrics import auc


SCORING_VERSION = "joint-shadow-value-v1"
AUDIT_CATEGORIES = {"unreviewed", "dark_pavement", "vehicle_underbody", "wet_ground", "vegetation", "suspected_missing_shadow_label", "other"}


def decision(row):
    target = row.get("annotation_target", row.get("target"))
    if target is None and row.get("answer"):
        target = parse(row["answer"], row.get("candidates", []))
    return (target or {}).get("shadow")


def example_weight(row, weak_weight):
    sources = {row.get("label_source"), row.get("shadow_label_source"), row.get("association_source")}
    multiplier = float(row.get("temporal_weight_multiplier", 1.0))
    if not math.isfinite(multiplier) or not 0 < multiplier <= 1:
        raise ValueError("Temporal evidence can only downweight, not amplify examples")
    if row.get("self_training"):
        pseudo_weight = float(row.get("pseudo_loss_weight", 0.25))
        if not math.isfinite(pseudo_weight) or not 0 < pseudo_weight <= 1:
            raise ValueError("Pseudo-label loss weight must be in (0,1]")
        multiplier *= pseudo_weight
    return (weak_weight if sources & WEAK_SOURCES else 1.0)*multiplier


def temporal_multiplier(row, policy):
    evidence = row.get("temporal", {})
    if policy.get("validated_for_weighting") is not True or policy.get("fitted_split") != "dev":
        raise ValueError("Temporal weighting requires an explicitly validated, frozen dev policy")
    downweight = float(policy["downweight"])
    if not math.isfinite(downweight) or not 0 < downweight <= 1 or policy["minimum_pixels"] < 1:
        raise ValueError("Invalid temporal downweight/minimum-pixel policy")
    if not math.isfinite(policy["minimum_motion_advantage"]) or not -1 <= policy["minimum_motion_advantage"] <= 1 or not 0 <= policy["minimum_transition_ratio"] <= 1:
        raise ValueError("Invalid temporal evidence thresholds")
    if row.get("shadow_label_source", row.get("label_source")) not in WEAK_SOURCES:
        return 1.0
    # Static small differences and unavailable evidence are never penalized.
    if evidence.get("mode") != "moving" or not evidence.get("lighting_comparable"):
        return 1.0
    if evidence.get("motion_alignment_comparison_pixels", 0) >= policy["minimum_pixels"]:
        advantage = evidence.get("motion_alignment_advantage")
        if advantage is not None and advantage < policy["minimum_motion_advantage"]:
            return downweight
    for count, feature in (("onset_pixels", "onset_darkening_ratio"), ("recovery_pixels", "recovery_brightening_ratio")):
        if evidence.get(count, 0) >= policy["minimum_pixels"] and evidence.get(feature) is not None and evidence[feature] < policy["minimum_transition_ratio"]:
            return downweight
    return 1.0


def balance_audit(rows, weights, weak_weight):
    groups = {}
    for row in rows:
        label = decision(row) or "unlabeled"
        group = groups.setdefault(label, {"rows": 0, "sum_base_weight": 0.0, "sum_decision_weight": 0.0, "scenes": set(), "tracks": set(), "missing_track_rows": 0})
        base = example_weight(row, weak_weight)
        group["rows"] += 1
        group["sum_base_weight"] += base
        group["sum_decision_weight"] += base*weights.get(label, 1.0)
        group["scenes"].add(row["scene_name"])
        if row.get("instance_token"):
            group["tracks"].add((row["scene_name"], row["instance_token"]))
        else:
            group["missing_track_rows"] += 1
    for group in groups.values():
        group["unique_scenes"] = len(group.pop("scenes"))
        group["unique_tracks"] = len(group.pop("tracks"))
        group["mean_decision_weight"] = group["sum_decision_weight"]/group["rows"]
        group["expected_sample_fraction"] = group["rows"]/len(rows)
    states = Counter((row.get("annotation_target") or (parse(row["answer"], row.get("candidates", [])) if row.get("answer") else {})).get("caster", {}).get("state", "unlabeled") for row in rows)
    return {"sampler": "ordinary shuffled/distributed sampler; no class oversampling", "rows": len(rows), "classes": groups, "caster_states": dict(states), "outside_training_supported": states["outside_bev"] > 0}


def average_precision(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    tp = np.cumsum(labels)[ends]
    precision, recall = tp/(ends+1), tp/positives
    return float(np.sum(np.diff(np.r_[0.0, recall])*precision))


def classify(score, threshold, margin=0.0, uncertainty=0.0, max_uncertainty=1.0):
    if not all(math.isfinite(value) for value in (score, threshold, margin, uncertainty, max_uncertainty)):
        raise ValueError("Non-finite decision scores or thresholds")
    if not 0 <= score <= 1 or not 0 <= margin <= 0.5 or not 0 <= uncertainty <= 1 or not 0 <= max_uncertainty <= 1:
        raise ValueError("Decision score/margin outside its range")
    if abs(score-threshold) < margin or uncertainty > max_uncertainty:
        return "UNCERTAIN"
    return "YES" if score >= threshold else "NO"


def score_metrics(rows, threshold, margin=0.0, max_uncertainty=1.0):
    labeled = [row for row in rows if decision(row) in {"YES", "NO"}]
    counts = Counter()
    predictions = []
    for row in labeled:
        label = decision(row)
        prediction = classify(row["score"], threshold, margin, row.get("uncertainty_score", 0.0), max_uncertainty)
        counts[label] += 1
        counts[f"{label}->{prediction}"] += 1
        predictions.append(prediction)
    ratio = lambda a, b: a/b if b else None
    tp, fp, tn = counts["YES->YES"], counts["NO->YES"], counts["NO->NO"]
    positive, negative = counts["YES"], counts["NO"]
    covered = sum(value != "UNCERTAIN" for value in predictions)
    labels = [decision(row) == "YES" for row in labeled]
    scores = [row["score"] for row in labeled]
    return {"labeled": len(labeled), "positive": positive, "negative": negative, "tp": tp, "fp": fp, "tn": tn,
            "positive_abstentions": counts["YES->UNCERTAIN"], "negative_abstentions": counts["NO->UNCERTAIN"],
            "positive_recall": ratio(tp, positive), "negative_specificity": ratio(tn, negative), "false_positive_rate": ratio(fp, negative),
            "balanced_accuracy": (tp/positive+tn/negative)/2 if positive and negative else None,
            "precision": ratio(tp, tp+fp), "coverage": ratio(covered, len(labeled)), "selective_accuracy": ratio(tp+tn, covered),
            "auroc": auc(labels, scores), "pr_auc_average_precision": average_precision(labels, scores) if labeled else None,
            "positive_prevalence": ratio(positive, len(labeled)), "confusion": dict(counts),
            "scope": "UNCERTAIN counts as a miss in class recalls; weak-target metrics are agreement, not real accuracy"}


def fit_threshold(rows, max_fpr=0.1, margin=0.0, max_uncertainty=1.0):
    if not rows or any(row.get("split") != "dev" or row.get("official_split") in {"val", "mini_val", "eval", "test"} for row in rows):
        raise ValueError("Threshold selection requires only train-derived dev rows")
    if not 0 <= max_fpr <= 1:
        raise ValueError("max-fpr must be in [0,1]")
    if {decision(row) for row in rows} & {"YES", "NO"} != {"YES", "NO"}:
        raise ValueError("Dev calibration needs both YES and NO")
    # Include both sides of each score and abstention boundary; ties stay intact.
    values = {0.0, 0.5, 1.0}
    for row in rows:
        for value in (row["score"]-margin, row["score"], row["score"]+margin):
            if 0 <= value <= 1:
                values.update((float(value), float(np.nextafter(value, np.inf))))
    scan = [{"threshold": value, **score_metrics(rows, value, margin, max_uncertainty)} for value in sorted(values)]
    feasible = [item for item in scan if item["false_positive_rate"] <= max_fpr]
    if not feasible:
        raise ValueError("No threshold satisfies this FPR/abstention policy")
    best = max(feasible, key=lambda item: (item["balanced_accuracy"], item["positive_recall"], item["coverage"], item["threshold"]))
    return {"threshold": best["threshold"], "abstain_margin": margin, "max_uncertainty": max_uncertainty, "max_dev_fpr": max_fpr,
            "dev_metrics": best, "has_positive_recall": best["positive_recall"] > 0,
            "approved_for_relabeling": False, "policy": "Dev operating point only; quality and independent-scene audit still required"}, scan


def cluster_intervals(rows, threshold, margin=0.0, max_uncertainty=1.0, bootstrap=1000, seed=0):
    scenes = sorted({row["scene_name"] for row in rows})
    if not scenes or bootstrap < 1:
        return {}
    if len(scenes) < 2:
        return {"scenes": len(scenes), "ci95": None, "warning": "At least two independent scenes are required for cluster intervals"}
    clusters = {scene: [row for row in rows if row["scene_name"] == scene] for scene in scenes}
    names = ("balanced_accuracy", "precision", "positive_recall", "false_positive_rate", "coverage", "auroc", "pr_auc_average_precision")
    draws = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        sampled = [row for index in rng.integers(0, len(scenes), len(scenes)) for row in clusters[scenes[index]]]
        metrics = score_metrics(sampled, threshold, margin, max_uncertainty)
        for name in names:
            if metrics[name] is not None:
                draws[name].append(metrics[name])
    return {"scenes": len(scenes), "bootstrap": bootstrap, "fixed_policy": True,
            "ci95": {name: {"interval": np.percentile(values, [2.5, 97.5]).tolist() if values else None, "valid_draws": len(values)} for name, values in draws.items()}}
