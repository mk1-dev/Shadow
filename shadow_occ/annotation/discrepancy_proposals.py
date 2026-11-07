"""Active proposals from projected 3D explanations and 2D appearance dynamics."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


PROTOCOL = "physical-appearance-proposals-v1"
SOURCE_BITS = {"appearance": 1, "discrepancy": 2, "caster_guided": 4}


@dataclass(frozen=True)
class DiscrepancyConfig:
    minimum_score: float = .35
    minimum_light_change_L: float = 1.5
    flow_scale_px: float = 3.0

    def validate(self):
        if (not np.isfinite(list(asdict(self).values())).all()
                or not 0 < self.minimum_score <= 1
                or min(self.minimum_light_change_L, self.flow_scale_px) <= 0):
            raise ValueError("Invalid discrepancy proposal configuration")


def projected_motion_explanation(arrays, domain, flow_scale):
    """Compare flow with ground and available tracked box-surface predictions.

    No box prediction is not free-space evidence. Object motion explains a
    receiver observation only when it fits better than the ground prediction.
    """
    shape = (2, *domain.shape)
    valid = np.zeros(shape, bool)
    ground_residual = np.full(shape, np.nan, np.float32)
    best_residual = ground_residual.copy()
    entity_fit = np.zeros(shape, np.float32)
    entity_valid = np.zeros(shape, bool)
    source = np.zeros(shape, np.uint8)
    observed, ground = arrays.get("observed_optical_flow_px"), arrays.get("expected_geometric_flow_px")
    if observed is None or ground is None or "observed_flow_valid_by_interval" not in arrays:
        return valid, ground_residual, best_residual, entity_fit, entity_valid, source
    valid = (arrays["observed_flow_valid_by_interval"] & domain[None]
        & np.isfinite(observed).all(axis=-1) & np.isfinite(ground).all(axis=-1))
    ground_residual = np.where(valid, np.linalg.norm(observed-ground, axis=-1), np.nan)
    best_residual = ground_residual.copy()
    source[valid] = 1
    models = []
    if "box_surface_expected_flow_px" in arrays:
        models.append((2, arrays["box_surface_expected_flow_px"], arrays["box_surface_flow_valid"], 1.0))
    if "mature_expected_object_flow_px" in arrays:
        confidence = arrays["mature_entity_probability"]
        model_domain = arrays["mature_inference_domain"]
        supported = model_domain[:-1] & model_domain[1:] & np.isfinite(confidence[:-1])
        models.append((3, arrays["mature_expected_object_flow_px"], supported,
            np.clip(np.nan_to_num(confidence[:-1]), 0, 1)))
    for code, expected, supported, confidence in models:
        supported = valid & supported & np.isfinite(expected).all(axis=-1)
        residual = np.linalg.norm(observed-expected, axis=-1)
        advantage = np.clip((ground_residual-residual)/flow_scale, 0, 1)
        fit = np.where(supported, np.exp(-residual/flow_scale)*advantage*confidence, 0)
        improve = supported & (fit > entity_fit)
        best_residual[improve] = residual[improve]
        source[improve] = code
        entity_fit = np.maximum(entity_fit, fit)
        entity_valid |= supported
    return valid, ground_residual, best_residual, entity_fit, entity_valid, source


def propose_discrepancy(arrays, *, cosine=None, baseline=None, light_threshold=6.0, config=None):
    """Evaluate the whole receiver grid, independent of dark-region seeds."""
    config = config or DiscrepancyConfig()
    config.validate()
    if not np.isfinite(light_threshold) or light_threshold <= 0:
        raise ValueError("Light-change normalization must be positive")
    domain = arrays["receiver_valid"].astype(bool).copy()
    for key in ("physical_box_footprint", "physical_ray_volume_support"):
        if key in arrays:
            domain &= ~arrays[key].any(axis=0)
    delta = arrays["temporal_illumination_change_L"]
    comparable = np.broadcast_to(arrays.get("temporal_exposure_valid_by_interval",
        arrays.get("temporal_exposure_valid", False)), delta.shape)
    quality = domain[None] & comparable & np.isfinite(delta)
    if "visible_by_frame" in arrays:
        quality &= arrays["visible_by_frame"][:-1] & arrays["visible_by_frame"][1:]
    # Positive means the centre frame is darker than its neighbour. No local
    # darkness threshold or pre-existing component is used in this branch.
    signed = np.stack([-delta[0], delta[1]])
    chroma = arrays.get("appearance_chroma_change")
    color = np.ones(delta.shape, np.float32)
    if chroma is not None:
        color = np.where(np.isfinite(chroma), np.exp(-chroma/12), 0)
    appearance = np.where(quality, np.clip(signed/light_threshold, 0, 1)*color, 0)
    bev_valid = np.zeros(delta.shape, bool)
    bev_stability = np.full(delta.shape, np.nan, np.float32)
    if cosine is not None:
        feature_valid = arrays["learned_bev_valid"]
        bev_valid = quality & feature_valid[:-1] & feature_valid[1:] & np.isfinite(cosine)
        bev_stability = np.where(bev_valid, np.clip((cosine+1)/2, 0, 1), np.nan)
    flow_valid, ground_residual, best_residual, entity_fit, entity_valid, motion_source = (
        projected_motion_explanation(arrays, domain, config.flow_scale_px))
    # Stable BEV favours receiver-preserving change. Missing BEV uses the middle
    # of the bounded factor range, not a claim of semantic stability.
    spatial = np.where(bev_valid, .5+.5*np.nan_to_num(bev_stability), .75)
    motion = np.where(flow_valid, 1+np.clip(np.nan_to_num(best_residual)/config.flow_scale_px, 0, 1), 1)
    score_by_interval = np.clip(appearance*spatial*motion*(1-entity_fit), 0, 1).astype(np.float32)
    proposed = quality & (signed >= config.minimum_light_change_L) & (score_by_interval >= config.minimum_score)
    candidate = proposed.any(axis=0)
    baseline = np.zeros_like(domain) if baseline is None else baseline
    novel = candidate & ~baseline
    # Proposal thresholds do not lower the brightness threshold for physical
    # transition verification. Weak proposals remain candidates, not events/GT.
    verification = proposed & (signed >= light_threshold) & (color >= .5)
    arrays.update(discrepancy_candidate=candidate, discrepancy_novel_candidate=novel,
        discrepancy_baseline_candidate=baseline.copy(), discrepancy_proposed_by_interval=proposed,
        discrepancy_score_by_interval=score_by_interval, discrepancy_score=score_by_interval.max(axis=0),
        discrepancy_appearance_evidence=appearance.astype(np.float32),
        discrepancy_receiver_quality=quality, discrepancy_bev_valid=bev_valid,
        discrepancy_bev_stability=bev_stability, discrepancy_flow_valid=flow_valid,
        discrepancy_ground_flow_residual_px=ground_residual, discrepancy_best_3d_flow_residual_px=best_residual,
        discrepancy_entity_explanation=entity_fit, discrepancy_entity_explanation_valid=entity_valid,
        discrepancy_motion_source=motion_source, discrepancy_onset=verification[0],
        discrepancy_recovery=verification[1])
    return {"protocol": PROTOCOL, "config": asdict(config), "light_threshold_L": light_threshold,
        "score_contract": {"appearance": "positive signed L change / light threshold * exp(-chroma/12)",
            "bev_factor": "0.5+0.5*stability; missing=0.75",
            "motion_factor": "1+clipped unexplained reliable flow residual / flow scale; missing=1",
            "entity_factor": "1-best supported object-motion explanation",
            "score": "clipped product; uncalibrated discovery score, not shadow probability"},
        "motion_source_legend": {"0": "unobserved", "1": "static_ground", "2": "GT_box_surface", "3": "predicted_box_surface"},
        "source_bits": SOURCE_BITS, "candidate_cells": int(candidate.sum()), "novel_cells": int(novel.sum()),
        "overlap_baseline_cells": int((candidate & baseline).sum()),
        "evaluated_cells": int(quality.any(axis=0).sum()),
        "bev_supported_cells": int((candidate & bev_valid.any(axis=0)).sum()),
        "flow_supported_cells": int((candidate & flow_valid.any(axis=0)).sum()),
        "entity_motion_explained_cells": int((entity_fit > 0).any(axis=0).sum()),
        "raw_flow_validity_available": "observed_flow_valid_by_interval" in arrays,
        "chroma_available": chroma is not None, "candidate_is_shadow_label": False}
