"""Signed same-world illumination evidence; stable dark appearance stays unknown."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shadow_occ.annotation.physics_motion import ground_homography


@dataclass(frozen=True)
class IlluminationReasoningConfig:
    minimum_log_darkening: float = .08
    log_darkening_scale: float = .2
    minimum_chroma_agreement: float = .6
    minimum_texture_agreement: float = .6
    minimum_correspondence_quality: float = .2
    geometry_only_quality: float = .6
    flow_residual_scale_px: float = 3.


def same_world_anchor_pairs(features, valid):
    features, valid = np.asarray(features), np.asarray(valid, bool)
    norms = np.linalg.norm(features, axis=-1)
    available = valid & np.isfinite(features).all(axis=-1) & (norms > 1e-6)
    good = available[:-1] & available[1:]
    cosine = (features[:-1]*features[1:]).sum(axis=-1)/np.maximum(norms[:-1]*norms[1:], 1e-8)
    return np.where(good, np.clip(cosine, 0, 1), 0).astype(np.float32), good


def illumination_reasoning(images, receivers, frames, basis, exposures, slot, anchor_pairs, *,
                          pair_flows=None, prior_states=None, config=IlluminationReasoningConfig()):
    """Dark at the current receiver versus a registered brighter observation.

    Appearance persistence is not a material label. Anchor similarity never adds
    illumination evidence. Missing flow can use direct geometry; large residuals
    never make a candidate more shadow-like.
    """
    shape = images[slot].shape[:2]
    current = cv2.cvtColor(images[slot], cv2.COLOR_BGR2LAB).astype(np.float32)
    current[..., 0] *= 100/255
    light = np.maximum(current[..., 0], 1)
    texture = np.log(light)-cv2.GaussianBlur(np.log(light), (0, 0), 2)
    result = {key: np.zeros(shape, np.float32) for key in
        ("score", "log_darkening", "log_recovery", "correspondence_quality", "anchor_consistency",
         "shadow_state_increase", "shadow_state_conflict", "joint_score")}
    result.update({key: np.zeros(shape, bool) for key in ("qualified", "observed_pair", "anchor_available")})
    result.update({key: np.zeros(shape, bool) for key in ("joint_onset", "joint_recovery", "shadow_state_available")})
    result["flow_state"] = np.zeros(shape, np.uint8)  # missing/low-confidence/reliable
    for other in (slot-1, slot+1):
        interval = min(slot, other)
        if other < 0 or other >= len(images) or interval >= len(exposures) or not exposures[interval]["valid"]:
            continue
        fit = exposures[interval]
        homography = ground_homography(frames[other]["projection_world_to_image"],
            frames[slot]["projection_world_to_image"], basis)
        warp = lambda value: cv2.warpPerspective(value, homography, shape[::-1], flags=cv2.INTER_LINEAR)
        adjacent = cv2.cvtColor(images[other], cv2.COLOR_BGR2LAB).astype(np.float32)
        adjacent[..., 0] *= 100/255
        adjacent = warp(adjacent)
        reference = (fit["scale"]*adjacent[..., 0]+fit["offset_L"] if other > slot else
                     (adjacent[..., 0]-fit["offset_L"])/fit["scale"])
        reference = np.maximum(reference, 1)
        signed = np.log(reference/light)
        chroma = np.exp(-np.linalg.norm(current[..., 1:]-adjacent[..., 1:], axis=-1)/12.)
        detail = np.log(reference)-cv2.GaussianBlur(np.log(reference), (0, 0), 2)
        texture_agreement = np.exp(-np.abs(detail-texture)/.15)
        confidence = np.minimum(receivers[slot]["confidence"], warp(receivers[other]["confidence"]))
        valid = (confidence > 0) & receivers[slot]["visible"] & (warp(receivers[other]["visible"].astype(np.float32)) >= .999)
        # These maps already use this frame's ground-ray coordinates.
        anchor, anchor_valid = anchor_pairs[interval][slot]
        quality = np.full(shape, config.geometry_only_quality, np.float32)
        flow_state = np.zeros(shape, np.uint8)
        if pair_flows is not None and interval < len(pair_flows):
            flow = pair_flows[interval]
            residual = np.asarray(flow["residual"], np.float32)
            reliable = np.asarray(flow["reliable"], np.float32)
            finite = np.asarray(flow["finite"], np.float32)
            if other < slot:
                residual, reliable, finite = warp(residual), warp(reliable), warp(finite)
            finite = finite >= .999
            good = (reliable >= .999) & finite & np.isfinite(residual)
            flow_quality = np.exp(-np.maximum(np.nan_to_num(residual, nan=1e6), 0)/config.flow_residual_scale_px)
            # Missing correspondence and observed geometric contradiction are different.
            # Keep geometry-only fallback for unavailable flow, not for a reliable
            # image match that demonstrably disagrees with the receiver projection.
            quality = np.where(good, flow_quality, quality)
            flow_state[finite] = 1
            flow_state[good & (residual <= config.flow_residual_scale_px)] = 2
        correspondence = confidence*quality
        qualified = valid & anchor_valid & (signed >= config.minimum_log_darkening)
        qualified &= ((chroma >= config.minimum_chroma_agreement) & (texture_agreement >= config.minimum_texture_agreement)
                      & (correspondence >= config.minimum_correspondence_quality))
        score = np.clip(signed/config.log_darkening_scale, 0, 1)*chroma*texture_agreement
        score *= correspondence*(.5+.5*anchor)
        score = np.where(qualified, score, 0)
        if prior_states is not None and prior_states[slot] is not None and prior_states[other] is not None:
            change = np.asarray(prior_states[slot], np.float32)-warp(np.asarray(prior_states[other], np.float32))
            increase, conflict = np.maximum(change, 0), np.maximum(-change, 0)
            joint = qualified & (increase >= .05)
            result["shadow_state_available"] |= valid
            result["shadow_state_increase"] = np.maximum(result["shadow_state_increase"], np.where(valid, increase, 0))
            result["shadow_state_conflict"] = np.maximum(result["shadow_state_conflict"], np.where(valid, conflict, 0))
            result["joint_score"] = np.maximum(result["joint_score"], np.where(joint, score*increase, 0))
            result["joint_onset" if other < slot else "joint_recovery"] |= joint
        result["score"] = np.maximum(result["score"], score)
        result["log_darkening"] = np.maximum(result["log_darkening"], np.where(valid, np.maximum(signed, 0), 0))
        result["log_recovery"] = np.maximum(result["log_recovery"], np.where(valid, np.maximum(-signed, 0), 0))
        result["correspondence_quality"] = np.maximum(result["correspondence_quality"], np.where(valid, correspondence, 0))
        result["anchor_consistency"] = np.maximum(result["anchor_consistency"], np.where(valid & anchor_valid, anchor, 0))
        result["flow_state"] = np.maximum(result["flow_state"], np.where(valid, flow_state, 0))
        result["qualified"] |= qualified
        result["observed_pair"] |= valid
        result["anchor_available"] |= valid & anchor_valid
    return result
