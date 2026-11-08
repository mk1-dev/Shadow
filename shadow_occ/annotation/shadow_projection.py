"""Projection capacity before any shadow acceptance or association gate."""
from __future__ import annotations

import numpy as np


def projection_coverage(observed, projections, light_ids=None, valid=None):
    """Measure a box/light search space against observed pixels, not its own ROI.

    This is model coverage of a proposal unless the caller supplies an independent
    reference mask. Best-cover and best-IoU directions need not be the same.
    """
    observed = np.asarray(observed, bool)
    projections = np.asarray(projections, bool)
    if projections.ndim != 3 or projections.shape[1:] != observed.shape:
        raise ValueError("Projection bank and observation must share a receiver grid")
    ids = list(range(len(projections))) if light_ids is None else list(light_ids)
    if len(ids) != len(projections) or len(set(ids)) != len(ids):
        raise ValueError("Projection light IDs must be unique and aligned")
    total = len(projections)
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != (total,):
            raise ValueError("Projection validity must have one value per light")
        projections = projections[valid]
        ids = [value for value, keep in zip(ids, valid) if keep]
    counts = {"hypotheses_total": total, "hypotheses": len(projections),
              "invalid_hypotheses": total-len(projections)}
    area = int(observed.sum())
    if not len(projections) or not area:
        return {"observed_cells": area, **counts,
            "status": "empty_reference" if not area else "no_valid_hypotheses",
            "best_cover": None, "best_iou": None, "union_cover": None,
            "best_cover_light_id": None, "best_iou_light_id": None, "by_light": []}
    intersections = (projections & observed).sum(axis=(1, 2))
    unions = (projections | observed).sum(axis=(1, 2))
    covers = intersections/area
    ious = intersections/np.maximum(unions, 1)
    best_cover, best_iou = int(np.argmax(covers)), int(np.argmax(ious))
    supported_direction = bool(intersections.any())
    return {"observed_cells": area, **counts,
        "status": "supported_direction" if supported_direction else "no_supported_direction",
        "best_cover": float(covers[best_cover]), "best_iou": float(ious[best_iou]),
        "union_cover": float((observed & projections.any(axis=0)).sum()/area),
        "best_cover_light_id": ids[best_cover] if supported_direction else None,
        "best_iou_light_id": ids[best_iou] if supported_direction else None,
        "by_light": [{"light_id": light_id, "intersection": int(intersection),
            "union": int(union), "cover": float(cover), "iou": float(iou)}
            for light_id, intersection, union, cover, iou in zip(ids, intersections, unions, covers, ious)]}


def selected_direction_coverage(observed, projections, light_id, valid=None):
    """Evaluate a preselected bank index without optimizing on the reference."""
    observed, projections = np.asarray(observed, bool), np.asarray(projections, bool)
    if projections.ndim != 3 or projections.shape[1:] != observed.shape:
        raise ValueError("Projection bank and reference must share a receiver grid")
    supported = np.ones(len(projections), bool) if valid is None else np.asarray(valid, bool)
    if supported.shape != (len(projections),):
        raise ValueError("Projection validity must have one value per light")
    result = {"light_id": light_id, "observed_cells": int(observed.sum()),
              "cover": None, "iou": None, "intersection": None, "union": None}
    if light_id is None:
        return {**result, "status": "not_selected"}
    if not isinstance(light_id, (int, np.integer)) or not 0 <= light_id < len(projections):
        raise ValueError("Selected light ID is outside the projection bank")
    if not supported[light_id]:
        return {**result, "status": "invalid_direction"}
    if not observed.any():
        return {**result, "status": "empty_reference"}
    mask = projections[light_id]
    intersection, union = int((observed & mask).sum()), int((observed | mask).sum())
    return {**result, "status": "evaluated", "intersection": intersection, "union": union,
            "cover": intersection/int(observed.sum()), "iou": intersection/max(union, 1)}


def reviewed_projection_coverage(reference, projections, selected_light_id, valid=None):
    """Separate reference-selected oracle capacity from the frozen model choice."""
    return {
        "projection_oracle_capacity": {**projection_coverage(reference, projections, valid=valid),
            "evaluation_kind": "reference_optimized_capacity_upper_bound_not_generalization"},
        "projection_selected_direction": {**selected_direction_coverage(reference, projections, selected_light_id, valid),
            "selection_source": "observed_proposal_best_cover_without_review_masks"},
    }


def summarize_signals(inputs, maps, slot, mask):
    """Read-only distributions for independently identified receiver regions."""
    mask = np.asarray(mask, bool)
    if mask.shape != inputs["receiver"][slot].shape:
        raise ValueError("Reviewed region and receiver grid differ")
    intervals = [i for i in (slot-1, slot) if 0 <= i < len(inputs["delta_L"])]
    reference_bev = maps["bev_feature_change"][intervals]
    values = {
        "illumination_change_abs_L": np.where(maps["exposure_valid"][intervals], np.abs(inputs["delta_L"][intervals]), np.nan),
        "flow_residual_px": maps["ground_motion_residual_px"][intervals],
        "bev_feature_change": reference_bev,
        "entity_explanation": np.where(maps["entity_explanation_valid_map"][intervals], maps["entity_explanation_map"][intervals], np.nan),
        "ground_reference_valid": maps["ground_motion_reference_valid_map"][intervals].astype(float),
        "discrepancy": maps["discrepancy_score_map"][slot:slot+1],
        "partial_appearance": maps["appearance_partial_score_map"][slot:slot+1],
    }
    output = {}
    for name, field in values.items():
        selected = np.asarray(field)[:, mask].ravel()
        valid = selected[np.isfinite(selected)]
        output[name] = {"finite_samples": len(valid), "missing_samples": int(len(selected)-len(valid)),
            "mean": float(valid.mean()) if len(valid) else None,
            "median": float(np.median(valid)) if len(valid) else None,
            "p10": float(np.percentile(valid, 10)) if len(valid) else None,
            "p90": float(np.percentile(valid, 90)) if len(valid) else None}
    return {"region_cells": int(mask.sum()), "visible_receiver_cells": int((mask & inputs["receiver"][slot]).sum()),
            "complete_discrepancy_cells": int((mask & maps["discrepancy_evidence_complete"][slot]).sum()),
            "partial_candidate_cells": int((mask & maps["partial_evidence_candidate"][slot]).sum()),
            "signals": output}
