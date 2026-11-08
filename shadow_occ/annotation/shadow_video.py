"""Receiver-aligned video proposals and BEV-aware region memory.

Scores are evidence, not calibrated shadow probabilities. Physical acceptance
is performed separately by the existing caster/light verifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


ENTITY_STATES = {0: "missing", 1: "explained", 2: "unexplained"}


@dataclass(frozen=True)
class VideoShadowConfig:
    seed_score: float = .35
    grow_score: float = .18
    light_scale_L: float = 6.0
    chroma_scale: float = 12.0
    flow_scale_px: float = 3.0
    minimum_area: int = 2
    memory_frames: int = 3
    maximum_gap_seconds: float = 1.5
    link_distance_m: float = 4.0
    minimum_link_score: float = .35

    def validate(self):
        values = np.asarray(list(asdict(self).values()), float)
        if (not np.isfinite(values).all() or (values <= 0).any()
                or not 0 < self.grow_score < self.seed_score <= 1
                or not 0 < self.minimum_link_score <= 1
                or not isinstance(self.minimum_area, int) or not isinstance(self.memory_frames, int)):
            raise ValueError("Invalid video shadow configuration")


def _cosine(a, b):
    denominator = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    value = (a*b).sum(axis=-1)/np.maximum(denominator, 1e-8)
    return np.clip(value, -1, 1), denominator > 1e-8


def bev_signatures(features, valid):
    """Compare both cell features and local spatial contrast at metric locations."""
    features = np.nan_to_num(np.asarray(features, np.float32))
    valid = np.asarray(valid, bool) & (np.linalg.norm(features, axis=-1) > 1e-6)
    local = np.zeros_like(features)
    for index, (feature, support) in enumerate(zip(features, valid)):
        count = cv2.boxFilter(support.astype(np.float32), -1, (3, 3), normalize=False,
                             borderType=cv2.BORDER_CONSTANT)
        mean = cv2.boxFilter(feature*support[..., None], -1, (3, 3), normalize=False,
                            borderType=cv2.BORDER_CONSTANT)
        if mean.ndim == 2:
            mean = mean[..., None]
        local[index] = np.where(support[..., None], feature-mean/np.maximum(count[..., None], 1), 0)
    cell, good = _cosine(features[:-1], features[1:])
    neighbourhood, local_good = _cosine(local[:-1], local[1:])
    supported = valid[:-1] & valid[1:] & good
    stability = np.where(supported, (cell+1)/2, np.nan)
    local_supported = supported & local_good
    stability[local_supported] = .5*stability[local_supported]+.5*(neighbourhood[local_supported]+1)/2
    return {"bev_pair_stability": stability.astype(np.float32),
            "bev_feature_valid": valid, "bev_feature_change": np.where(supported, (1-cell)/2, np.nan),
            "bev_neighbourhood_change": np.where(local_supported, (1-neighbourhood)/2, np.nan)}


def video_proposals(lab, receiver, delta_L, exposure_valid, *, observed_flow, expected_flow,
                    fb_consistent, texture_support, features, feature_valid,
                    caster_prior=None, entity_explanation=None, fb_error_px=None,
                    occlusion_mask=None, config=None, bev_role="context", entity_explanation_valid=None):
    """Discover on each valid frame pair, never the all-frame intersection.

    delta_L[t] is the exposure-corrected L(t+1)-L(t) at fixed world cells.
    Darkening belongs to frame t+1; recovery belongs to frame t.
    """
    config = config or VideoShadowConfig()
    config.validate()
    if bev_role not in {"context", "stability", "off"}:
        raise ValueError("BEV role must be context, stability or off")
    lab, receiver = np.asarray(lab, np.float32), np.asarray(receiver, bool)
    length, height, width = receiver.shape
    if (length < 2 or lab.shape != (length, height, width, 3)
            or delta_L.shape != (length-1, height, width)
            or observed_flow.shape != (length-1, height, width, 2)
            or expected_flow.shape != observed_flow.shape
            or fb_consistent.shape != delta_L.shape or texture_support.shape != delta_L.shape
            or features.shape[:3] != receiver.shape or feature_valid.shape != receiver.shape):
        raise ValueError("Video evidence must use one shared receiver grid")
    receiver = receiver & np.isfinite(lab).all(axis=-1)
    pair_visible = receiver[:-1] & receiver[1:]
    finite_delta = np.isfinite(delta_L)
    comparable = pair_visible & np.broadcast_to(exposure_valid, delta_L.shape) & finite_delta
    chroma = np.linalg.norm(np.diff(lab[..., 1:], axis=0), axis=-1)
    color = np.where(np.isfinite(chroma), np.exp(-chroma/config.chroma_scale), 0)
    occluded = np.zeros(delta_L.shape, bool) if occlusion_mask is None else np.asarray(occlusion_mask, bool)
    if occluded.shape != delta_L.shape:
        raise ValueError("Occlusion support must share the interval grid")
    flow_present = np.isfinite(observed_flow).all(axis=-1)
    flow_valid = pair_visible & ~occluded & flow_present & fb_consistent & np.isfinite(expected_flow).all(axis=-1)
    confidence = flow_valid.astype(np.float32)
    if fb_error_px is not None:
        error = np.asarray(fb_error_px, np.float32)
        if error.shape != delta_L.shape:
            raise ValueError("FB error must share the interval grid")
        confidence *= np.where(np.isfinite(error), np.exp(-np.maximum(error, 0)/config.flow_scale_px), 0)
    flow_residual = np.where(flow_valid, np.linalg.norm(observed_flow-expected_flow, axis=-1), np.nan)
    signatures = bev_signatures(features, feature_valid & receiver)
    entity = np.zeros_like(delta_L) if entity_explanation is None else np.asarray(entity_explanation, np.float32)
    if entity.shape != delta_L.shape or not np.isfinite(entity).all() or ((entity < 0) | (entity > 1)).any():
        raise ValueError("Entity explanations must be supported scores in [0,1]")
    entity_valid = (entity > 0) if entity_explanation_valid is None else np.asarray(entity_explanation_valid, bool)
    if entity_valid.shape != delta_L.shape:
        raise ValueError("Entity explanation validity must share the interval grid")
    entity = np.where(entity_valid, entity, 0)
    entity_state = np.zeros(delta_L.shape, np.uint8)
    entity_state[entity_valid & (entity > 0)] = 1
    entity_state[entity_valid & (entity == 0)] = 2
    complete_by_interval = comparable & entity_valid
    # The default does not assume BEV illumination invariance. Keep the old
    # hypothesis as an explicit control, while descriptors still link regions.
    stability = signatures["bev_pair_stability"]
    spatial = np.where(np.isfinite(stability), stability, 1) if bev_role == "stability" else np.ones_like(delta_L)
    ground_correspondence = confidence*np.exp(-np.nan_to_num(flow_residual, nan=0)/config.flow_scale_px)
    appearance = np.zeros(receiver.shape, np.float32)
    discrepancy = np.zeros_like(appearance)
    partial = np.zeros_like(appearance)
    complete_by_frame = np.zeros(receiver.shape, bool)
    onset, recovery = np.zeros(receiver.shape, bool), np.zeros(receiver.shape, bool)
    pair_score = np.zeros_like(delta_L, np.float32)
    pair_partial = np.zeros_like(pair_score)
    for index in range(length-1):
        change = np.where(comparable[index], delta_L[index], 0)
        darken = np.clip(-change/config.light_scale_L, 0, 1)*color[index]
        restore = np.clip(change/config.light_scale_L, 0, 1)*color[index]
        for frame, value in ((index+1, darken), (index, restore)):
            appearance[frame] = np.maximum(appearance[frame], value)
            score = np.clip(value*spatial[index]*(1-entity[index]), 0, 1)
            complete = np.where(complete_by_interval[index], score, 0)
            partial_score = np.where(comparable[index] & ~entity_valid[index], value*spatial[index], 0)
            discrepancy[frame] = np.maximum(discrepancy[frame], complete)
            partial[frame] = np.maximum(partial[frame], partial_score)
            complete_by_frame[frame] |= complete_by_interval[index]
            pair_score[index] = np.maximum(pair_score[index], complete)
            pair_partial[index] = np.maximum(pair_partial[index], partial_score)
        onset[index+1] |= comparable[index] & (-change >= config.light_scale_L) & (color[index] >= .5)
        recovery[index] |= comparable[index] & (change >= config.light_scale_L) & (color[index] >= .5)
    # Static appearance is a separate proposal source, never a fabricated event.
    static = np.zeros_like(appearance)
    boundary = np.zeros_like(appearance)
    for index in range(length):
        light = np.nan_to_num(lab[index, ..., 0])
        support = receiver[index].astype(np.float32)
        density = cv2.GaussianBlur(support, (0, 0), 4)
        mean = cv2.GaussianBlur(light*support, (0, 0), 4)/np.maximum(density, 1e-6)
        static[index] = np.where(receiver[index], np.clip((1-light/np.maximum(mean, 1))/.12, 0, 1), 0)
        gx = cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=3)/8
        gy = cv2.Sobel(light, cv2.CV_32F, 0, 1, ksize=3)/8
        boundary[index] = np.where(receiver[index], np.clip(np.hypot(gx, gy)/config.light_scale_L, 0, 1), 0)
    prior = np.zeros_like(appearance) if caster_prior is None else np.asarray(caster_prior, np.float32)
    if prior.shape != receiver.shape or not np.isfinite(prior).all() or ((prior < 0) | (prior > 1)).any():
        raise ValueError("Caster prior must share video grid and lie in [0,1]")
    # A projected footprint alone never creates an observed candidate.
    caster_score = prior*np.maximum(appearance, static)
    dynamic = np.maximum(discrepancy, partial)
    combined = np.maximum(dynamic, np.maximum(.5*static, .5*caster_score))*receiver
    source = ((appearance >= config.seed_score).astype(np.uint8)
              | ((discrepancy >= config.seed_score).astype(np.uint8) << 1)
              | ((caster_score*.5 >= config.seed_score).astype(np.uint8) << 2))
    source[(static*.5 >= config.seed_score) & receiver] |= 1
    return {**signatures, "appearance_score_map": np.maximum(appearance, .5*static),
        "discrepancy_score_map": np.where(complete_by_frame, discrepancy, np.nan),
        "discrepancy_evidence_complete": complete_by_frame,
        "appearance_partial_score_map": partial, "dynamic_discovery_score_map": dynamic,
        "partial_evidence_candidate": partial >= config.seed_score, "proposal_score_map": combined,
        "caster_prior_map": prior, "entity_explanation_map": entity,
        "entity_explanation_valid_map": entity_valid, "entity_explanation_state": entity_state,
        "discrepancy_evidence_complete_by_interval": complete_by_interval,
        "flow_present": flow_present, "fb_consistent": fb_consistent & flow_present,
        "flow_out_of_receiver": ~pair_visible, "flow_low_texture": pair_visible & ~texture_support,
        "flow_fb_inconsistent": flow_present & ~fb_consistent,
        "flow_occluded": occluded,
        "flow_occlusion_known": np.full(delta_L.shape, occlusion_mask is not None, bool),
        "flow_interpolation_filled": np.zeros(delta_L.shape, bool),
        "flow_confidence_map": confidence,
        "ground_motion_residual_px": flow_residual,
        "pair_discrepancy_score_map": np.where(complete_by_interval, pair_score, np.nan),
        "pair_partial_appearance_score_map": pair_partial,
        "ground_flow_correspondence_score": ground_correspondence,
        "ground_consistent_illumination_score": ground_correspondence*np.maximum(pair_score, pair_partial),
        "boundary_confidence": boundary, "proposal_source_bits": source,
        "onset": onset, "recovery": recovery, "receiver_valid_by_frame": receiver,
        "pair_correspondence_valid": pair_visible, "exposure_valid": comparable}


def refine_regions(score, boundary, receiver, features, feature_valid, config=None, memory_mask=None):
    """Seeded OpenCV watershed with boundaries and current observed support."""
    config = config or VideoShadowConfig()
    allowed = receiver & (score >= config.grow_score)
    seed = receiver & (score >= config.seed_score)
    if memory_mask is not None:
        seed |= allowed & memory_mask
    if not seed.any():
        return []
    count, labels = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
    markers = labels.astype(np.int32)+1
    markers[allowed & ~seed] = 0
    markers[~allowed] = 1
    # Strong observed boundaries are watershed barriers, not foreground seeds.
    terrain = np.maximum(boundary, 1-score)
    terrain = cv2.cvtColor((np.clip(terrain, 0, 1)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(terrain, markers)
    regions = []
    for label in range(1, count):
        mask = ((markers == label+1) | (labels == label)) & allowed
        if mask.sum() < config.minimum_area:
            continue
        supported = mask & feature_valid & np.isfinite(features).all(axis=-1)
        descriptor = None
        if supported.any():
            descriptor = features[supported].mean(axis=0).astype(np.float32)
            descriptor /= max(float(np.linalg.norm(descriptor)), 1e-6)
        rim = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        rank = float(.75*score[mask].mean()+.25*boundary[rim & receiver].mean()) if (rim & receiver).any() else float(score[mask].mean())
        regions.append({"mask": mask, "descriptor": descriptor, "rank_score": rank,
                        "boundary_score": float(boundary[rim & receiver].mean()) if (rim & receiver).any() else 0.0})
    return sorted(regions, key=lambda region: -region["rank_score"])


class ShadowRegionMemory:
    """Hungarian region linking with BEV signatures and bounded disappearance."""
    def __init__(self, resolution_m, config=None):
        self.config = config or VideoShadowConfig()
        self.config.validate()
        if not np.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("Resolution must be positive")
        self.resolution = resolution_m
        self.entries = {}
        self.next_id = 1
        self.last_timestamp = None

    def update(self, regions, timestamp, *, predicted_shift_xy=None):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Region memory requires strictly increasing timestamps")
        self.last_timestamp = timestamp
        self.entries = {key: value for key, value in self.entries.items()
                        if (timestamp-value["timestamp"])/1e6 <= self.config.maximum_gap_seconds
                        and value["missed"] < self.config.memory_frames}
        identities = list(self.entries)
        scores = np.zeros((len(identities), len(regions)), np.float32)
        for row, identity in enumerate(identities):
            previous = self.entries[identity]
            old = previous["mask"]
            if predicted_shift_xy is not None:
                shift = np.asarray(predicted_shift_xy)*((timestamp-previous["timestamp"])/1e6)
                matrix = np.array([[1, 0, shift[0]/self.resolution], [0, 1, shift[1]/self.resolution]], np.float32)
                old = cv2.warpAffine(old.astype(np.uint8), matrix, old.shape[::-1], flags=cv2.INTER_NEAREST) > 0
            for col, region in enumerate(regions):
                mask = region["mask"]
                distance = np.linalg.norm(np.argwhere(old).mean(axis=0)-np.argwhere(mask).mean(axis=0))*self.resolution if old.any() else np.inf
                if distance > self.config.link_distance_m:
                    continue
                overlap = (old & mask).sum()/max(int((old | mask).sum()), 1)
                spatial = np.exp(-distance/self.config.link_distance_m)
                descriptors = previous["descriptor"], region["descriptor"]
                if all(value is not None for value in descriptors):
                    similarity = float(np.clip(np.dot(*descriptors), 0, 1))
                    scores[row, col] = .35*overlap+.25*spatial+.4*similarity
                else:
                    scores[row, col] = .6*overlap+.4*spatial
        assigned = {}
        if scores.size:
            rows, cols = linear_sum_assignment(-scores)
            for row, col in zip(rows, cols):
                if scores[row, col] >= self.config.minimum_link_score:
                    assigned[col] = (identities[row], float(scores[row, col]))
        for entry in self.entries.values():
            entry["missed"] += 1
        for index, region in enumerate(regions):
            identity, consistency = assigned.get(index, (self.next_id, None))
            if index not in assigned:
                self.next_id += 1
            region.update(shadow_track_id=identity, temporal_consistency=consistency)
            self.entries[identity] = {"mask": region["mask"].copy(), "descriptor": region["descriptor"],
                                      "timestamp": timestamp, "missed": 0}
        return regions
