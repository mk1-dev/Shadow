"""Metric flow/BEV correspondence on one fixed world receiver grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shadow_occ.annotation.physics_motion import project_world, sample_at


@dataclass(frozen=True)
class CorrespondenceConfig:
    radius_cells: int = 2
    minimum_cosine: float = .2
    minimum_margin: float = .02
    cycle_tolerance_cells: float = 1.0
    displacement_scale_m: float = .75  # Retained for legacy proxy readers, not positive evidence.
    flow_weight: float = .35  # Reliable material correspondence supporting observed appearance.
    feature_weight: float = .25  # Same-world anchor consistency, not feature-change magnitude.

    def validate(self):
        if (not isinstance(self.radius_cells, int) or not 1 <= self.radius_cells <= 8
                or not -1 <= self.minimum_cosine <= 1 or not 0 < self.minimum_margin <= 2
                or not np.isfinite(list(self.__dict__.values())).all()
                or min(self.cycle_tolerance_cells, self.displacement_scale_m) <= 0
                or min(self.flow_weight, self.feature_weight) < 0):
            raise ValueError("Invalid metric BEV correspondence settings")


@dataclass(frozen=True)
class CrossSpaceGrowthConfig:
    radius_m: float = 2.0
    light_tolerance_L: float = 6.0
    chroma_tolerance: float = 12.0
    maximum_boundary: float = .75
    spatial_decay: float = .9
    temporal_decay: float = .9

    def validate(self):
        if (not np.isfinite(list(self.__dict__.values())).all()
                or min(self.radius_m, self.light_tolerance_L, self.chroma_tolerance) <= 0
                or not 0 < self.maximum_boundary <= 1
                or not 0 < self.spatial_decay < 1 or not 0 < self.temporal_decay < 1):
            raise ValueError("Invalid cross-space region growth settings")


@dataclass(frozen=True)
class FlowConfidenceConfig:
    maximum_ground_residual_px: float = 3.0
    residual_scale_px: float = 3.0
    minimum_fb_confidence: float = .5

    def validate(self):
        if (not np.isfinite(list(self.__dict__.values())).all()
                or min(self.maximum_ground_residual_px, self.residual_scale_px) <= 0
                or not 0 < self.minimum_fb_confidence <= 1):
            raise ValueError("Invalid ground-flow confidence settings")


SPATIAL_MATCH_STATES = {0: "missing", 1: "low_confidence", 2: "reliable", 3: "ambiguous"}
FLOW_STATES = {0: "missing", 1: "low_confidence", 2: "reliable"}


def ground_flow_confidence(maps, config=None):
    """Judge material correspondence, never score large residuals as shadows."""
    config = config or FlowConfidenceConfig()
    config.validate()
    residual = maps["ground_motion_residual_observed_px"]
    observed = maps["pair_correspondence_valid"] & maps["flow_present"] & ~maps["flow_occluded"] & np.isfinite(residual)
    fb_quality = maps["flow_confidence_map"]
    texture = ~maps["flow_low_texture"]
    quality = fb_quality*np.exp(-np.nan_to_num(residual, nan=0)/config.residual_scale_px)*texture
    reliable = (observed & maps["fb_consistent"] & texture
                & (fb_quality >= config.minimum_fb_confidence) & (residual <= config.maximum_ground_residual_px))
    return {"m2_flow_state": np.where(reliable, 2, np.where(observed, 1, 0)).astype(np.uint8),
        "m2_flow_quality": np.where(observed, quality, np.nan).astype(np.float32),
        "m2_flow_reliable": reliable,
        "ground_flow_geometry_residual_px": np.where(observed, residual, np.nan)}


def lift_flow_to_bev(world, frames, pixels, flow, flow_valid, receiver, basis, projection, resolution):
    """Intersect observed NEXT-image endpoints with the receiver plane.

    Camera motion cancels through the actual next camera projection. Output
    vectors are horizontal anchor-grid x/y meters, NOT image pixels or speed.
    """
    height, width = world.shape[:2]
    yy, xx = np.indices((height, width))
    start = np.stack([xx, yy], axis=-1)
    vectors = np.full(flow.shape, np.nan, np.float32)
    valid = np.zeros(flow.shape[:-1], bool)
    for slot in range(len(flow)):
        endpoints = pixels[slot]+flow[slot]
        next_frame = frames[slot+1]
        image_width, image_height = next_frame["image_size_wh"]
        good = receiver[slot] & flow_valid[slot] & np.isfinite(endpoints).all(axis=-1)
        good &= ((endpoints >= 0).all(axis=-1) & (endpoints[..., 0] <= image_width-1)
                 & (endpoints[..., 1] <= image_height-1))
        transform = np.asarray(next_frame["projection_world_to_image"]) @ basis
        homogeneous = np.linalg.solve(transform, np.c_[endpoints.reshape(-1, 2), np.ones(height*width)].T).T
        good &= (np.isfinite(homogeneous).all(axis=-1) & (homogeneous[:, 2] > 1e-8)).reshape(height, width)
        xy = homogeneous[:, :2]/np.where(homogeneous[:, 2:] > 1e-8, homogeneous[:, 2:], 1)
        destination_world = (np.c_[xy, np.ones(len(xy))] @ basis.T)[:, :3]
        grid_uv, _, front = project_world(destination_world, projection)
        grid_uv = grid_uv.reshape(height, width, 2).astype(np.float32)
        good &= front.reshape(height, width) & np.isfinite(grid_uv).all(axis=-1)
        good &= ((grid_uv >= 0).all(axis=-1) & (grid_uv[..., 0] <= width-1) & (grid_uv[..., 1] <= height-1))
        safe = np.where(np.isfinite(grid_uv), grid_uv, -1)
        good &= sample_at(receiver[slot+1].astype(np.float32), safe) >= .999
        valid[slot] = good
        vectors[slot] = np.where(good[..., None], (grid_uv-start)*resolution, np.nan)
    return {"flow_bev_displacement_m": vectors, "flow_bev_valid": valid}


def _local_match(first, second, first_valid, second_valid, config):
    height, width = first_valid.shape
    best = np.full((height, width), -np.inf, np.float32)
    runner_up = best.copy()
    displacement = np.zeros((height, width, 2), np.float32)
    count = np.zeros((height, width), np.uint16)
    offsets = sorted(((dy, dx) for dy in range(-config.radius_cells, config.radius_cells+1)
                       for dx in range(-config.radius_cells, config.radius_cells+1)),
                     key=lambda d: (d[0]**2+d[1]**2, d))
    for dy, dx in offsets:
        ya, yb = max(0, -dy), min(height, height-dy)
        xa, xb = max(0, -dx), min(width, width-dx)
        if ya >= yb or xa >= xb:
            continue
        source, target = np.s_[ya:yb, xa:xb], np.s_[ya+dy:yb+dy, xa+dx:xb+dx]
        cosine = np.einsum("hwc,hwc->hw", first[source], second[target])
        cosine = np.where(first_valid[source] & second_valid[target], np.clip(cosine, -1, 1), -np.inf)
        count[source] += np.isfinite(cosine)
        previous = best[source].copy()
        better = cosine > previous
        runner_up[source] = np.where(better, previous, np.maximum(runner_up[source], cosine))
        best[source] = np.maximum(previous, cosine)
        displacement[source] = np.where(better[..., None], [dx, dy], displacement[source])
    # A unique vector cannot be established from constant/repeated descriptors.
    margin = np.zeros_like(best)
    finite = np.isfinite(best) & np.isfinite(runner_up)
    np.subtract(best, runner_up, out=margin, where=finite)
    supported = first_valid & finite & (best >= config.minimum_cosine) & (margin >= config.minimum_margin)
    return displacement, best, margin, supported, count


def match_spatial_bev(features, valid, resolution, config=None):
    """Refine the SAME WORLD receiver anchor on current-only descriptors.

    Inputs are already mapped through each frame's world-to-native-BEV transform.
    Thus the expected ground correspondence is the same array cell, not an
    unconstrained feature guess. Output offsets are feature-registration
    residuals around that anchor, NOT measured physical ground motion.
    """
    config = config or CorrespondenceConfig()
    config.validate()
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution must be positive")
    features = np.asarray(features, np.float32)
    norms = np.linalg.norm(features, axis=-1)
    valid = np.asarray(valid, bool) & np.isfinite(features).all(axis=-1) & (norms > 1e-6)
    normalized = np.nan_to_num(features)/np.maximum(norms[..., None], 1e-6)
    length, height, width = valid.shape
    yy, xx = np.indices((height, width))
    start = np.stack([xx, yy], axis=-1)
    vectors = np.full((length-1, height, width, 2), np.nan, np.float32)
    good = np.zeros(vectors.shape[:-1], bool)
    confidence = np.zeros(good.shape, np.float32)
    state = np.zeros(good.shape, np.uint8)
    candidate_count = np.zeros(good.shape, np.uint16)
    cycle_error = np.full(good.shape, np.nan, np.float32)
    match_cosine = np.full(good.shape, np.nan, np.float32)
    match_margin = np.full(good.shape, np.nan, np.float32)
    change = np.full(good.shape, np.nan, np.float32)
    for slot in range(length-1):
        if not valid[slot].any() or not valid[slot+1].any():
            continue
        forward, best, margin, supported, count = _local_match(normalized[slot], normalized[slot+1], valid[slot], valid[slot+1], config)
        backward, reverse_best, reverse_margin, reverse_valid, reverse_count = _local_match(
            normalized[slot+1], normalized[slot], valid[slot+1], valid[slot], config)
        endpoints = (start+forward).astype(np.float32)
        reverse = sample_at(backward, endpoints, nearest=True)
        reverse_best = sample_at(reverse_best, endpoints, nearest=True)
        reverse_margin = sample_at(reverse_margin, endpoints, nearest=True)
        reverse_count = sample_at(reverse_count, endpoints, nearest=True)
        cycle = np.linalg.norm(forward+reverse, axis=-1)
        anchor_valid = valid[slot] & valid[slot+1]
        available = anchor_valid & (count > 0) & np.isfinite(best) & (best >= config.minimum_cosine)
        reversible = available & (reverse_count > 0) & np.isfinite(reverse_best) & (reverse_best >= config.minimum_cosine)
        supported &= sample_at(reverse_valid.astype(np.uint8), endpoints, nearest=True) > 0
        supported &= anchor_valid & (cycle <= config.cycle_tolerance_cells)
        # A forced argmax on a singleton/tie is not a measured motion vector.
        ambiguous = available & ((count < 2) | (margin <= 1e-6)
                                  | (reversible & ((reverse_count < 2) | (reverse_margin <= 1e-6))))
        uniqueness = np.clip(margin/.1, 0, 1)
        reverse_uniqueness = np.clip(reverse_margin/.1, 0, 1)
        quality = np.minimum(uniqueness, reverse_uniqueness)
        quality *= np.where(supported, 1, np.exp(-cycle/config.cycle_tolerance_cells))
        quality = np.where(reversible & ~ambiguous, quality, 0)
        usable = reversible & (quality > 0)
        vectors[slot] = np.where(usable[..., None], forward*resolution, np.nan)
        good[slot] = supported
        confidence[slot] = quality
        state[slot] = np.where(ambiguous, 3, np.where(supported, 2, np.where(available, 1, 0)))
        candidate_count[slot] = count
        cycle_error[slot] = np.where(reversible, cycle, np.nan)
        match_cosine[slot] = np.where(available, best, np.nan)
        match_margin[slot] = np.where(available & (count > 1), margin, np.nan)
        same_cell = valid[slot] & valid[slot+1]
        cosine = np.einsum("hwc,hwc->hw", normalized[slot], normalized[slot+1])
        change[slot] = np.where(same_cell, np.clip(1-cosine, 0, 2), np.nan)
    return {"spatial_bev_displacement_m": vectors, "spatial_bev_match_valid": good,
        "spatial_bev_match_confidence": confidence, "spatial_bev_feature_change": change,
        "spatial_bev_match_state": state, "spatial_bev_match_usable": confidence > 0,
        "spatial_bev_candidate_count": candidate_count, "spatial_bev_cycle_error_cells": cycle_error,
        "spatial_bev_match_cosine": match_cosine, "spatial_bev_match_margin": match_margin,
        "spatial_bev_geometry_pair_valid": valid[:-1] & valid[1:],
        "spatial_bev_anchor_cosine": np.where(np.isfinite(change), 1-change, np.nan),
        "spatial_bev_local_offset_m": np.where(good[..., None], vectors, np.nan),
        "spatial_bev_local_residual_m": np.where(good, np.linalg.norm(vectors, axis=-1), np.nan)}


def cross_space_proposals(maps, spatial_features, spatial_valid, flow_bev, flow_valid,
                          resolution, config=None, *, flow_config=None):
    """Anchor-first appearance reasoning; local matching is optional refinement."""
    config = config or CorrespondenceConfig()
    signals = match_spatial_bev(spatial_features, spatial_valid & maps["receiver_valid_by_frame"], resolution, config)
    flow_signals = ground_flow_confidence(maps, flow_config)
    length = maps["receiver_valid_by_frame"].shape[0]
    score = np.zeros_like(maps["proposal_score_map"])
    complete_score, partial_score = np.zeros_like(score), np.zeros_like(score)
    complete_valid = np.zeros(score.shape, bool)
    anchor_score, flow_score = np.zeros_like(score), np.zeros_like(score)
    anchor_by_frame = np.full(score.shape, np.nan, np.float32)
    known_entity = maps["entity_explanation_valid_map"]
    entity_factor = np.where(known_entity, 1-maps["entity_explanation_map"], np.nan)
    paired = maps["exposure_valid"]
    anchor_valid = signals["spatial_bev_geometry_pair_valid"]
    anchor = np.where(anchor_valid, np.clip(signals["spatial_bev_anchor_cosine"], 0, 1), np.nan)
    reliable = flow_valid & signals["spatial_bev_match_valid"] & flow_signals["m2_flow_reliable"] & paired
    difference = np.linalg.norm(flow_bev-signals["spatial_bev_local_offset_m"], axis=-1)
    flow_quality = np.where(flow_signals["m2_flow_reliable"] & flow_valid, flow_signals["m2_flow_quality"], 0)
    for slot in range(length-1):
        for at in (slot, slot+1):
            base = maps["appearance_score_map"][at]
            observed = paired[slot] & anchor_valid[slot] & maps["receiver_valid_by_frame"][at]
            anchor_support = base*config.feature_weight*np.nan_to_num(anchor[slot])
            flow_support = base*config.flow_weight*flow_quality[slot]
            raw = np.clip(base+anchor_support+flow_support, 0, 1)
            anchor_score[at] = np.maximum(anchor_score[at], np.where(observed, anchor_support, 0))
            flow_score[at] = np.maximum(flow_score[at], np.where(observed, flow_support, 0))
            anchor_by_frame[at] = np.fmax(anchor_by_frame[at], np.where(observed, anchor[slot], np.nan))
            # The unknown channel does NOT apply an invented entity complement.
            # It proposes unexplained observations, then defers to M3 physics.
            full = np.where(observed & known_entity[slot], raw*entity_factor[slot], 0)
            partial = np.where(observed & ~known_entity[slot], raw, 0)
            complete_valid[at] |= observed & known_entity[slot]
            complete_score[at] = np.maximum(complete_score[at], full)
            partial_score[at] = np.maximum(partial_score[at], partial)
            score[at] = np.maximum(complete_score[at], partial_score[at])
    maps.update(signals)
    maps.update(flow_signals)
    geometric_pair = maps["receiver_valid_by_frame"][:-1] & maps["receiver_valid_by_frame"][1:]
    maps.update(flow_bev_displacement_m=flow_bev, flow_bev_valid=flow_valid,
        bev_flow_observable_pair=geometric_pair,
        bev_flow_matchable_pair=geometric_pair & signals["spatial_bev_geometry_pair_valid"],
        ground_flow_geometry_residual_m=np.where(flow_valid, np.linalg.norm(flow_bev, axis=-1), np.nan),
        flow_bev_discrepancy_m=np.where(reliable, difference, np.nan),
        flow_bev_discrepancy_valid=reliable, flow_bev_discrepancy_supported=reliable,
        bev_anchor_consistency=anchor, bev_anchor_consistency_by_frame=anchor_by_frame,
        anchor_appearance_support_score=anchor_score, flow_appearance_support_score=flow_score,
        cross_space_discovery_score=score,
        cross_space_complete_score=np.where(complete_valid, complete_score, np.nan),
        cross_space_partial_score=partial_score, cross_space_complete_valid=complete_valid,
        cross_space_entity_factor=entity_factor,
        cross_space_entity_factor_known=known_entity)
    return maps


def correspondence_summary(maps):
    """Coverage denominators and consistency diagnostics, not true accuracy."""
    rows = []
    for slot, observable in enumerate(maps["bev_flow_observable_pair"]):
        matchable = maps["bev_flow_matchable_pair"][slot]
        reliable = maps["spatial_bev_match_valid"][slot] & matchable
        row = {"interval": slot, "observable_receiver_cells": int(observable.sum()),
            "geometry_anchored_descriptor_cells": int(matchable.sum()),
            "reliable_local_match_cells": int(reliable.sum()),
            "ambiguous_local_match_cells": int(((maps["spatial_bev_match_state"][slot] == 3) & matchable).sum())}
        for value, label in FLOW_STATES.items():
            row[label+"_flow_cells"] = int((matchable & (maps["m2_flow_state"][slot] == value)).sum())
        row["descriptor_coverage"] = float(matchable.sum()/observable.sum()) if observable.any() else None
        row["reliable_local_match_coverage"] = float(reliable.sum()/matchable.sum()) if matchable.any() else None
        for name in ("spatial_bev_anchor_cosine", "spatial_bev_cycle_error_cells",
                     "spatial_bev_local_residual_m", "ground_flow_geometry_residual_px"):
            values = maps[name][slot][matchable & np.isfinite(maps[name][slot])]
            row[name+"_median"] = float(np.median(values)) if len(values) else None
        rows.append(row)
    totals = {key: sum(row[key] for row in rows) for key in ("observable_receiver_cells",
        "geometry_anchored_descriptor_cells", "reliable_local_match_cells", "ambiguous_local_match_cells",
        "missing_flow_cells", "low_confidence_flow_cells", "reliable_flow_cells")}
    observable, matchable = totals["observable_receiver_cells"], totals["geometry_anchored_descriptor_cells"]
    totals["descriptor_coverage"] = totals["geometry_anchored_descriptor_cells"]/observable if observable else None
    totals["reliable_local_match_coverage"] = totals["reliable_local_match_cells"]/matchable if matchable else None
    return {"pairs": rows, "totals": totals, "correctness_is_measured_accuracy": False,
        "contract": "Coverage uses independently observable receiver pairs. Cosine, cycle and residuals are consistency diagnostics, not audited correspondence correctness."}


def grow_cross_space_regions(maps, lab, resolution, video_config, *, timestamps=None, config=None):
    """Grow observed regions from M2 seeds, not from rendered footprints.

    Temporal seeds use independently valid adjacent pairs at the same world
    receiver cell. They are never recursively propagated or advected as shadow
    motion by material flow. Every destination needs its own appearance support.
    """
    config = config or CrossSpaceGrowthConfig()
    config.validate()
    video_config.validate()
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution must be positive")
    lab = np.asarray(lab, np.float32)
    score = maps["cross_space_discovery_score"]
    receiver = maps["receiver_valid_by_frame"]
    if lab.shape != (*score.shape, 3):
        raise ValueError("Region growth requires frame-aligned Lab observations")
    allowed = receiver & maps["search_valid_by_frame"] & np.isfinite(lab).all(axis=-1)
    appearance = maps["appearance_score_map"]
    observed_appearance = appearance >= video_config.minimum_observed_appearance
    seeds = allowed & observed_appearance & (score >= video_config.seed_score)
    temporal_score = np.zeros_like(score)
    temporal_source = np.full(score.shape, -1, np.int32)
    if timestamps is not None:
        timestamps = np.asarray(timestamps, np.float64)
        if timestamps.shape != (len(score),) or not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
            raise ValueError("Growth timestamps must be strictly increasing camera microseconds")
    for slot in range(len(score)-1):
        if timestamps is not None and (timestamps[slot+1]-timestamps[slot])/1e6 > video_config.maximum_gap_seconds:
            continue
        pair = (maps["pair_correspondence_valid"][slot] & maps["exposure_valid"][slot]
                & np.isfinite(maps["spatial_bev_feature_change"][slot]))
        color = np.linalg.norm(lab[slot+1, ..., 1:]-lab[slot, ..., 1:], axis=-1)
        pair &= color <= config.chroma_tolerance
        for source, target in ((slot, slot+1), (slot+1, slot)):
            supported = (seeds[source] & allowed[target] & observed_appearance[target] & pair
                         & (lab[target, ..., 0] <= lab[source, ..., 0]+config.light_tolerance_L)
                         & (maps["boundary_confidence"][target] <= config.maximum_boundary))
            value = np.where(supported, score[source]*config.temporal_decay, 0)
            if "bev_anchor_consistency" in maps:
                value *= np.nan_to_num(maps["bev_anchor_consistency"][slot])
            better = value > temporal_score[target]
            temporal_score[target] = np.maximum(temporal_score[target], value)
            temporal_source[target][better] = source
    temporal_seed = (temporal_score >= video_config.grow_score) & ~seeds
    region_score = np.where(seeds, score, 0)
    region_score = np.maximum(region_score, np.where(temporal_seed, temporal_score, 0))
    anchors = lab.copy()
    anchor_appearance = appearance.copy()
    region_appearance = np.where(seeds | temporal_seed, appearance, 0)
    # Anchor colors travel with the seed: local gradients cannot accumulate
    # across a different material or into a bright road region.
    height, width = receiver.shape[1:]
    for _ in range(int(np.floor(config.radius_m/resolution))):
        previous, previous_anchors = region_score.copy(), anchors.copy()
        previous_appearance = anchor_appearance.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ya, yb = max(0, -dy), min(height, height-dy)
            xa, xb = max(0, -dx), min(width, width-dx)
            source = np.s_[:, ya:yb, xa:xb]
            target = np.s_[:, ya+dy:yb+dy, xa+dx:xb+dx]
            value = previous[source]*config.spatial_decay
            light_distance = np.abs(lab[target][..., 0]-previous_anchors[source][..., 0])
            color_distance = np.linalg.norm(lab[target][..., 1:]-previous_anchors[source][..., 1:], axis=-1)
            compatible = (light_distance <= config.light_tolerance_L) & (color_distance <= config.chroma_tolerance)
            # Uniform shadow interiors may have zero local darkness score. Use
            # their observed color agreement with the seed, not that old gate.
            observed = previous_appearance[source]*np.exp(
                -light_distance/config.light_tolerance_L-color_distance/config.chroma_tolerance)
            better = (allowed[target] & compatible & (value >= video_config.grow_score)
                      & (observed >= video_config.minimum_observed_appearance)
                      & (maps["boundary_confidence"][target] <= config.maximum_boundary)
                      & (value > region_score[target]))
            region_score[target] = np.where(better, value, region_score[target])
            anchors[target] = np.where(better[..., None], previous_anchors[source], anchors[target])
            anchor_appearance[target] = np.where(better, previous_appearance[source], anchor_appearance[target])
            region_appearance[target] = np.where(better, observed, region_appearance[target])
        if np.array_equal(previous, region_score):
            break
    support = region_score >= video_config.grow_score
    maps.update(cross_space_region_score=region_score, cross_space_region_support=support,
        cross_space_region_appearance_support=region_appearance,
        cross_space_direct_seed=seeds, cross_space_temporal_seed=temporal_seed,
        cross_space_temporal_source_frame=temporal_source,
        cross_space_spatial_growth=support & ~seeds & ~temporal_seed)
    return maps
