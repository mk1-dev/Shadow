"""Ground-registered 2D appearance versus 3D explanations, not shadow truth."""
from __future__ import annotations

import numpy as np


PROTOCOL = "cross-space-discovery-v1"


def discover_cross_space(arrays, *, cosine=None, light_threshold=6.0, darkness_ratio=.88):
    """Propose observed dark receivers; BEV/flow rank evidence without vetoing it.

    A missing flow estimate is unknown, not a large residual. Stable darkness is
    retained separately from temporal discrepancy and still needs caster physics.
    """
    receiver = arrays["receiver_valid"].astype(bool).copy()
    for key in ("physical_box_footprint", "physical_ray_volume_support"):
        if key in arrays:
            receiver &= ~arrays[key].any(axis=0)
    ratio = arrays["appearance_local_ratio"]
    delta = arrays["temporal_illumination_change_L"]
    comparable = np.broadcast_to(arrays.get("temporal_exposure_valid_by_interval",
        arrays.get("temporal_exposure_valid", False)), delta.shape)
    valid = comparable & np.isfinite(delta) & receiver[None]
    chroma = arrays.get("appearance_chroma_change")
    if chroma is not None:
        valid &= np.isfinite(chroma) & (chroma <= 12*np.log(2))
    dark = receiver & np.isfinite(ratio[1]) & (ratio[1] < darkness_ratio)
    onset = dark & valid[0] & (delta[0] <= -light_threshold)
    recovery = dark & valid[1] & (delta[1] >= light_threshold)
    change = onset | recovery
    persistent = receiver & np.isfinite(ratio).all(axis=0) & (ratio < darkness_ratio).all(axis=0)
    original = arrays.get("shadow_candidate_photometric", arrays["shadow_candidate"])
    candidate = receiver & (original | change | persistent)

    bev_valid = np.zeros(receiver.shape, bool)
    stability = np.full(receiver.shape, np.nan, np.float32)
    if cosine is not None:
        cosine = np.asarray(cosine, np.float32)
        feature_valid = arrays["learned_bev_valid"]
        pairs = feature_valid[:-1] & feature_valid[1:] & np.isfinite(cosine)
        count = pairs.sum(axis=0)
        bev_valid = receiver & (count > 0)
        mean = np.where(pairs, cosine, 0).sum(axis=0)/np.maximum(count, 1)
        stability[bev_valid] = np.clip((mean[bev_valid]+1)/2, 0, 1)

    flow_valid = np.zeros(delta.shape, bool)
    residual = np.full(delta.shape, np.nan, np.float32)
    if "observed_flow_valid_by_interval" in arrays:
        observed = arrays["observed_optical_flow_px"]
        expected = arrays["expected_geometric_flow_px"]
        flow_valid = (arrays["observed_flow_valid_by_interval"] & receiver[None]
            & np.isfinite(observed).all(axis=-1) & np.isfinite(expected).all(axis=-1))
        residual = np.where(flow_valid, np.linalg.norm(observed-expected, axis=-1), np.nan)
    flow_change = np.max(np.where(flow_valid, np.clip(np.nan_to_num(residual)/6, 0, 1), 0), axis=0)
    magnitude = np.maximum(np.where(onset, -delta[0], 0), np.where(recovery, delta[1], 0))
    appearance = np.clip(magnitude/(2*light_threshold), 0, 1)
    entity_fit = np.zeros(receiver.shape, np.float32)
    if "mature_entity_motion_fit" in arrays:
        fit = arrays["mature_entity_motion_fit"]
        entity_fit = np.max(np.where(np.isfinite(fit), np.clip(fit, 0, 1), 0), axis=0)
    # Uncalibrated evidence ranking. Missing BEV/flow is neutral, not agreement.
    bev_factor = np.where(bev_valid, .5+.5*np.nan_to_num(stability), .75)
    flow_factor = np.where(flow_valid.any(axis=0), .75+.25*flow_change, .875)
    discrepancy = np.where(receiver, appearance*bev_factor*flow_factor*(1-entity_fit), 0)
    arrays.update(cross_space_receiver=receiver, cross_space_temporal_candidate=change,
        cross_space_static_candidate=persistent, cross_space_bev_valid=bev_valid,
        cross_space_bev_stability=stability, cross_space_flow_valid=flow_valid,
        cross_space_flow_residual_px=residual, cross_space_entity_motion_explanation=entity_fit,
        cross_space_appearance_evidence=appearance.astype(np.float32),
        cross_space_discrepancy=discrepancy.astype(np.float32))
    arrays["shadow_candidate"] = candidate
    arrays["shadow_boundary_onset"] = onset & candidate
    arrays["shadow_boundary_recovery"] = recovery & candidate
    arrays["shadow_same_patch_cycle"] = onset & recovery & candidate
    return {"protocol": PROTOCOL, "light_threshold_L": light_threshold, "darkness_ratio": darkness_ratio,
        "rule": "observed appearance on nonentity receivers; BEV and flow are evidence, not vetoes",
        "score_role": "uncalibrated temporal evidence ranking; not shadow probability or label",
        "flow_validity_available": "observed_flow_valid_by_interval" in arrays,
        "receiver_cells": int(receiver.sum()), "temporal_cells": int(change.sum()),
        "static_cells": int(persistent.sum()), "candidate_cells": int(candidate.sum()),
        "candidate_without_bev": int((candidate & ~bev_valid).sum()),
        "candidate_without_registered_flow": int((candidate & ~arrays["shadow_flow_supported"]).sum()),
        "temporally_supported_bev_cells": int((change & bev_valid).sum()),
        "reliable_flow_residual_cells": int(flow_valid.any(axis=0).sum())}
