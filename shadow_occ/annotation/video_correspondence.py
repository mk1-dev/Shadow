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
    displacement_scale_m: float = .75
    flow_weight: float = .35
    feature_weight: float = .25

    def validate(self):
        if (not isinstance(self.radius_cells, int) or not 1 <= self.radius_cells <= 8
                or not -1 <= self.minimum_cosine <= 1 or not 0 < self.minimum_margin <= 2
                or not np.isfinite(list(self.__dict__.values())).all()
                or min(self.cycle_tolerance_cells, self.displacement_scale_m) <= 0
                or min(self.flow_weight, self.feature_weight) < 0):
            raise ValueError("Invalid metric BEV correspondence settings")


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
    return displacement, best, margin, supported


def match_spatial_bev(features, valid, resolution, config=None):
    """Local cosine matching + reverse-cycle check on CURRENT-ONLY features."""
    config = config or CorrespondenceConfig()
    config.validate()
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
    change = np.full(good.shape, np.nan, np.float32)
    for slot in range(length-1):
        if not valid[slot].any() or not valid[slot+1].any():
            continue
        forward, _, margin, supported = _local_match(normalized[slot], normalized[slot+1], valid[slot], valid[slot+1], config)
        backward, _, _, reverse_valid = _local_match(normalized[slot+1], normalized[slot], valid[slot+1], valid[slot], config)
        endpoints = (start+forward).astype(np.float32)
        reverse = sample_at(backward, endpoints, nearest=True)
        supported &= sample_at(reverse_valid.astype(np.uint8), endpoints, nearest=True) > 0
        supported &= np.linalg.norm(forward+reverse, axis=-1) <= config.cycle_tolerance_cells
        vectors[slot] = np.where(supported[..., None], forward*resolution, np.nan)
        good[slot] = supported
        confidence[slot] = np.where(supported, np.clip(margin/.1, 0, 1), 0)
        same_cell = valid[slot] & valid[slot+1]
        cosine = np.einsum("hwc,hwc->hw", normalized[slot], normalized[slot+1])
        change[slot] = np.where(same_cell, np.clip(1-cosine, 0, 2), np.nan)
    return {"spatial_bev_displacement_m": vectors, "spatial_bev_match_valid": good,
        "spatial_bev_match_confidence": confidence, "spatial_bev_feature_change": change}


def cross_space_proposals(maps, spatial_features, spatial_valid, flow_bev, flow_valid,
                          resolution, config=None):
    """Propose from metric cross-space discrepancy; physics still accepts labels."""
    config = config or CorrespondenceConfig()
    signals = match_spatial_bev(spatial_features, spatial_valid & maps["receiver_valid_by_frame"], resolution, config)
    length = maps["receiver_valid_by_frame"].shape[0]
    score = np.zeros_like(maps["proposal_score_map"])
    complete_score, partial_score = np.zeros_like(score), np.zeros_like(score)
    complete_valid = np.zeros(score.shape, bool)
    known_entity = maps["entity_explanation_valid_map"]
    entity_factor = np.where(known_entity, 1-maps["entity_explanation_map"], np.nan)
    paired = maps["exposure_valid"]
    flow_quality = signals["spatial_bev_match_confidence"]*maps["flow_confidence_map"]
    matched = flow_valid & signals["spatial_bev_match_valid"] & paired
    difference = np.linalg.norm(flow_bev-signals["spatial_bev_displacement_m"], axis=-1)
    flow_score = np.where(matched, np.nan_to_num(difference/config.displacement_scale_m), 0)
    feature_change = signals["spatial_bev_feature_change"]
    feature_score = np.where(paired & np.isfinite(feature_change), feature_change/2, 0)
    extra = config.flow_weight*flow_quality*np.clip(flow_score, 0, 1)+config.feature_weight*feature_score
    for slot in range(length-1):
        for at in (slot, slot+1):
            base = maps["appearance_score_map"][at]
            observed = (paired[slot] & (np.isfinite(feature_change[slot]) | matched[slot])
                        & maps["receiver_valid_by_frame"][at])
            raw = np.clip(base+extra[slot], 0, 1)
            # The unknown channel does NOT apply an invented entity complement.
            # It proposes unexplained observations, then defers to M3 physics.
            full = np.where(observed & known_entity[slot], raw*entity_factor[slot], 0)
            partial = np.where(observed & ~known_entity[slot], raw, 0)
            complete_valid[at] |= observed & known_entity[slot]
            complete_score[at] = np.maximum(complete_score[at], full)
            partial_score[at] = np.maximum(partial_score[at], partial)
            score[at] = np.maximum(complete_score[at], partial_score[at])
    maps.update(signals)
    maps.update(flow_bev_displacement_m=flow_bev, flow_bev_valid=flow_valid,
        flow_bev_discrepancy_m=np.where(matched, difference, np.nan),
        flow_bev_discrepancy_valid=matched, cross_space_discovery_score=score,
        cross_space_complete_score=np.where(complete_valid, complete_score, np.nan),
        cross_space_partial_score=partial_score, cross_space_complete_valid=complete_valid,
        cross_space_entity_factor=entity_factor,
        cross_space_entity_factor_known=known_entity)
    return maps
