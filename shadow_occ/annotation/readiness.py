"""Non-collapse checks are necessary, never sufficient approval for pseudo labels."""
from __future__ import annotations

import math

from shadow_occ.annotation.data import GT_SOURCES


def segmentation_readiness(by_source):
    preferred = {key: value for key, value in by_source.items() if key in GT_SOURCES}
    by_source = preferred or by_source
    totals = {name: sum(int(value.get(name, 0)) for value in by_source.values())
              for name in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (totals[name] for name in ("tp", "fp", "fn", "tn"))
    reasons = []
    if tp+fn == 0 or tn+fp == 0:
        reasons.append("missing_dev_class")
    if tp+fp == 0:
        reasons.append("all_background")
    if tn+fn == 0:
        reasons.append("all_foreground")
    if tp == 0:
        reasons.append("zero_true_positive")
    if tn == 0:
        reasons.append("zero_true_negative")
    return {"noncollapsed": not reasons, "reasons": reasons, "counts": totals,
            "source_scope": sorted(by_source),
            "approved_for_relabeling": False,
            "scope": "dev source agreement only; independent quality audit still required"}


def require_segmenter_checkpoint(checkpoint, diagnostics_only=False):
    metrics = checkpoint.get("dev_metrics_by_source")
    readiness = segmentation_readiness(metrics) if isinstance(metrics, dict) and metrics else None
    if not diagnostics_only and (not readiness or not readiness["noncollapsed"]):
        raise ValueError("Collapsed or unvalidated segmenter checkpoint; diagnostic-only inference must be explicit")
    return readiness or {"noncollapsed": False, "reasons": ["missing_dev_metrics"],
                         "approved_for_relabeling": False}


def binary_risk_weights(fractions, maximum=50.0):
    """One loss-weighting mechanism; no oversampling or implicit background labels."""
    if len(fractions) != 2 or not math.isfinite(maximum) or maximum < 1:
        raise ValueError("Invalid balancing configuration")
    total = sum(fractions)
    if not all(math.isfinite(x) and x > 0 for x in fractions) or total <= 0:
        raise ValueError("Balancing requires both observed classes")
    return [min(maximum, total/(2*x)) for x in fractions]


def require_prediction_for_proposals(row):
    readiness = row.get("segmenter_readiness") or {}
    if row.get("diagnostics_only") or not readiness.get("noncollapsed"):
        raise ValueError("Diagnostic, collapsed or legacy-unvalidated segmentation cannot generate proposals")
