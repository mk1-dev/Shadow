"""Auditable physical BEV / image-motion evidence, not neural BEV or occupancy GT."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shadow_occ.annotation.physics_labels import validate_window, window_tracks
from shadow_occ.annotation.physics_motion import decompose_object_motion, project_world, sample_at
from shadow_occ.annotation.receiver_explanations import (
    PROTOCOL as RECEIVER_PROTOCOL, OBSERVATION_STATES, box_ray_evidence, receiver_evidence,
)
from shadow_occ.geometry.bev import BEVGrid


@dataclass(frozen=True)
class BEVFlowConfig:
    extent_m: float = 40.0
    resolution_m: float = 0.5
    light_threshold_L: float = 6.0
    darkness_ratio: float = 0.88
    background_sigma_m: float = 4.0
    chroma_scale: float = 12.0

    def grid(self):
        values = [self.extent_m, self.resolution_m, self.light_threshold_L,
                  self.background_sigma_m, self.chroma_scale, self.darkness_ratio]
        if not np.isfinite(values).all() or min(values) <= 0 or self.darkness_ratio >= 1:
            raise ValueError("Invalid BEV-flow configuration")
        cells = 2*self.extent_m/self.resolution_m
        if not np.isclose(cells, round(cells)) or not 2 <= cells <= 1024:
            raise ValueError("BEV bounds must form an integer grid of 2..1024 cells per side")
        return BEVGrid(-self.extent_m, self.extent_m, -self.extent_m, self.extent_m, self.resolution_m)


def bev_projection(anchor, grid):
    anchor = np.asarray(anchor, np.float64)
    if (anchor.shape != (4, 4) or not np.isfinite(anchor).all()
            or not np.allclose(anchor[3], [0, 0, 0, 1])
            or not np.allclose(anchor[:3, :3].T @ anchor[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(anchor[:3, :3]) < 0.99):
        raise ValueError("Expected a rigid anchor ego-to-world transform")
    pixels = np.array([[1/grid.resolution, 0, 0, -grid.x_min/grid.resolution-0.5],
                       [0, 1/grid.resolution, 0, -grid.y_min/grid.resolution-0.5], [0, 0, 0, 1]])
    return pixels @ np.linalg.inv(anchor)


def ground_grid(basis_world, anchor, grid):
    basis = np.asarray(basis_world, np.float64)
    if basis.shape != (4, 3) or not np.isfinite(basis).all():
        raise ValueError("Invalid metric receiver basis")
    # The exported basis uses horizontal x/y coordinates of this exact anchor.
    local = np.linalg.inv(anchor) @ basis
    if (not np.allclose(local[:2], [[1, 0, 0], [0, 1, 0]], atol=1e-5)
            or not np.allclose(local[3], [0, 0, 1], atol=1e-5)):
        raise ValueError("Ground basis and BEV anchor disagree; re-export pairs")
    yy, xx = np.indices(grid.shape)
    xy = grid.ij_to_xy(np.c_[yy.ravel(), xx.ravel()])
    world = (basis @ np.c_[xy, np.ones(len(xy))].T)[:3].T
    return world.reshape(*grid.shape, 3)


def rasterize_volumes(frame, anchor, grid):
    if "object_volumes_world" not in frame:
        raise ValueError("Missing all-class physical volumes; re-export nuScenes pairs")
    occupied = np.zeros(grid.shape, bool)
    low = np.full(grid.shape, np.nan, np.float32)
    high = low.copy()
    inverse = np.linalg.inv(anchor)
    for volume in frame["object_volumes_world"]:
        corners = np.asarray(volume["corners_world"], np.float64)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError("Invalid physical box volume")
        points = (np.c_[corners, np.ones(8)] @ inverse.T)[:, :3]
        pixels = (points[:, :2]-[grid.x_min, grid.y_min])/grid.resolution-0.5
        polygon = cv2.convexHull(np.rint(np.clip(pixels, -100000, 100000)).astype(np.int32))
        mask = np.zeros(grid.shape, np.uint8)
        cv2.fillConvexPoly(mask, polygon, 1)
        mask = mask > 0
        low[mask] = np.fmin(low[mask], points[:, 2].min())
        high[mask] = np.fmax(high[mask], points[:, 2].max())
        occupied |= mask
    return occupied, low, high


def _image_projection(world, frame, shape):
    uv, _, valid = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
    height, width = shape
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= width-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height-1)
    return np.where(valid[:, None], uv, -1).reshape(*world.shape[:2], 2).astype(np.float32), valid.reshape(world.shape[:2])


def build_bev_flow(left, right, images, grounds, first, second, exposures, config=None):
    """Sample ALL supported receivers before extracting dark components."""
    config = config or BEVFlowConfig()
    grid = config.grid()
    validate_window(left, right)
    if "anchor_ego_to_world" not in left:
        raise ValueError("Missing physical BEV anchor; re-export pairs")
    anchor = np.asarray(left["anchor_ego_to_world"], np.float64)
    projection = bev_projection(anchor, grid)
    world = ground_grid(left["plane_basis_world"], anchor, grid)
    frames = [left["frames"][0], right["frames"][0], right["frames"][1]]
    if len(images) != 3 or len(grounds) != 3:
        raise ValueError("Expected exactly three receiver observations")
    uv, visible, labs, occupied, low, high = [], [], [], [], [], []
    for frame, image, ground in zip(frames, images, grounds):
        if image.shape[:2] != ground.shape or image.shape[1::-1] != tuple(frame["image_size_wh"]):
            raise ValueError("BEV evidence image coordinates differ from pair manifest")
        pixels, geometric = _image_projection(world, frame, ground.shape)
        # Every bilinear contributor must be supported; never interpolate unseen road as observed.
        valid = geometric & (sample_at(ground.astype(np.float32), pixels) >= 0.999)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] *= 100/255
        uv.append(pixels)
        visible.append(valid)
        labs.append(sample_at(lab, pixels))
        o, bottom, top = rasterize_volumes(frame, anchor, grid)
        occupied.append(o)
        low.append(bottom)
        high.append(top)
    occupied = np.stack(occupied)
    visible = np.stack(visible)
    sampled_flows = [sample_at(np.asarray(value["flow_t_to_next"], np.float32), uv[index])
                     for index, value in enumerate((first, second))]
    sampled_flow_valid = [sample_at(np.asarray(value.get("flow_reliable",
        np.zeros(images[index].shape[:2], bool)), np.float32), uv[index]) >= .999
        for index, value in enumerate((first, second))]
    volume_evidence = box_ray_evidence(world, frames, uv, sampled_flows, sampled_flow_valid)
    receiver = (visible.all(axis=0) & ~occupied.any(axis=0)
                & ~volume_evidence["physical_ray_volume_support"].any(axis=0))
    observed, expected, reliability, residual = [], [], [], []
    for index, evidence in enumerate((first, second)):
        flow = np.asarray(evidence["flow_t_to_next"], np.float32)
        if flow.shape != (*images[index].shape[:2], 2):
            raise ValueError("Flow and source image coordinates differ")
        sampled = sample_at(flow, uv[index])
        geometric = uv[index+1]-uv[index]
        valid = (sample_at(np.asarray(evidence["receiver_flow_supported"], np.float32), uv[index]) >= 0.999)
        valid &= receiver & np.isfinite(sampled).all(axis=-1)
        observed.append(sampled)
        expected.append(geometric)
        reliability.append(valid)
        residual.append(np.linalg.norm(sampled-geometric, axis=-1))
    comparable = all(item.get("valid") is True for item in exposures)
    raw_L = np.stack([lab[..., 0] for lab in labs])
    appearance = raw_L.copy()
    scale0, offset0 = exposures[0]["scale"], exposures[0]["offset_L"]
    scale1, offset1 = exposures[1]["scale"], exposures[1]["offset_L"]
    if not exposures[0].get("valid"):
        scale0, offset0 = 1.0, 0.0
    if not exposures[1].get("valid"):
        scale1, offset1 = 1.0, 0.0
    if not np.isfinite([scale0, offset0, scale1, offset1]).all():
        raise ValueError("Nonfinite valid exposure transforms")
    appearance[1] = scale0*raw_L[1]+offset0
    appearance[2] = scale0*(scale1*raw_L[2]+offset1)+offset0
    delta = np.diff(appearance, axis=0)
    sigma = config.background_sigma_m/grid.resolution
    support = receiver.astype(np.float32)
    density = cv2.GaussianBlur(support, (0, 0), sigma)
    means = np.stack([cv2.GaussianBlur(value*support, (0, 0), sigma)/np.maximum(density, 1e-6)
                      for value in raw_L])
    local_ratio = raw_L/np.maximum(means, 1)
    dark = receiver & (density > 1e-6) & (local_ratio[1] < config.darkness_ratio)
    chroma = np.stack([np.linalg.norm(labs[i+1][..., 1:]-labs[i][..., 1:], axis=-1) for i in range(2)])
    color = (np.exp(-chroma/config.chroma_scale) >= 0.5).all(axis=0)
    usable = receiver & color & comparable
    onset = usable & (delta[0] <= -config.light_threshold_L)
    recovery = usable & (delta[1] >= config.light_threshold_L)
    previous_support = onset | (np.abs(delta[0]) < config.light_threshold_L)
    next_support = recovery | (np.abs(delta[1]) < config.light_threshold_L)
    persistent_dark = receiver & (density > 1e-6) & (local_ratio < config.darkness_ratio).all(axis=0)
    candidate = (dark & usable & previous_support & next_support) | persistent_dark
    flow_support = np.stack(reliability).all(axis=0)
    changed = usable & (np.abs(delta) >= config.light_threshold_L).any(axis=0)
    # A/B/C are explanatory states, not non-shadow/shadow truth. Zero is unknown.
    state = np.zeros(grid.shape, np.uint8)
    state[visible.any(axis=0) & occupied.any(axis=0)] = 2
    state[usable & flow_support & ~changed] = 1
    state[changed] = 3
    tracks, _ = window_tracks(left, right)
    landmarks = np.asarray([track["points"] for track in tracks], np.float64).reshape(-1, 3, 9, 3)
    displacement = np.diff(landmarks[:, :, 0, :], axis=1)
    intervals = np.array([left["dt_seconds"], right["dt_seconds"]])
    object_flows = {key: [] for key in ("total_displacement_px", "camera_displacement_px", "object_residual_displacement_px")}
    for track in tracks:
        decomposed = [decompose_object_motion(track["points"][i], track["points"][i+1],
            frames[i]["projection_world_to_image"], frames[i+1]["projection_world_to_image"], intervals[i]) for i in range(2)]
        for key in object_flows:
            object_flows[key].append([item[key] for item in decomposed])
    arrays = {"receiver_world_xyz": world.astype(np.float32), "bev_world_to_grid": projection,
        "anchor_ego_to_world": anchor, "ground_basis_world": np.asarray(left["plane_basis_world"]),
        "visible_by_frame": visible, "physical_box_footprint": occupied,
        "physical_box_bottom_z_anchor": np.stack(low), "physical_box_top_z_anchor": np.stack(high),
        "receiver_valid": receiver, "appearance_L_raw": raw_L,
        "appearance_L_compensated": appearance, "temporal_illumination_change_L": delta,
        "appearance_local_ratio": np.where(receiver[None], local_ratio, np.nan),
        "expected_geometric_flow_px": np.stack(expected), "observed_optical_flow_px": np.stack(observed),
        "flow_residual_px": np.stack(residual), "flow_reliability": np.stack(reliability),
        "explanation_state": state, "illumination_event": changed,
        "shadow_boundary_onset": onset & candidate, "shadow_boundary_recovery": recovery & candidate,
        "shadow_candidate": candidate, "shadow_flow_supported": flow_support,
        "shadow_persistent_dark": persistent_dark, "temporal_exposure_valid": np.full(grid.shape, comparable, bool),
        "temporal_exposure_valid_by_interval": np.broadcast_to(
            np.array([bool(item.get("valid")) for item in exposures])[:, None, None], (2, *grid.shape)).copy(),
        "shadow_same_patch_cycle": onset & recovery & candidate,
        "track_landmarks_world": landmarks.astype(np.float32),
        "physical_object_displacement_world_m": displacement.astype(np.float32),
        "physical_object_velocity_world_mps": (displacement/intervals[None, :, None]).astype(np.float32),
        "physical_object_displacement_anchor_m": (displacement @ anchor[:3, :3]).astype(np.float32)}
    arrays.update(volume_evidence)
    arrays.update(receiver_evidence(arrays, light_threshold=config.light_threshold_L,
                                    darkness_ratio=config.darkness_ratio))
    for key, values in object_flows.items():
        arrays[f"box_landmark_{key}"] = np.asarray(values, np.float32).reshape(-1, 2, 9, 2)
    metadata = {"protocol": "bev-flow-discrepancy-v1", "receiver_explanation_protocol": RECEIVER_PROTOCOL,
        "receiver_observation_legend": OBSERVATION_STATES,
        "pair_ids": [left["id"], right["id"]],
        "scene_name": left["scene_name"], "camera": left["camera"], "official_split": left["official_split"],
        "sample_tokens": [left["sample_token"], right["sample_token"], right["next_sample_token"]],
        "frame_tokens": [frame["sample_data_token"] for frame in frames],
        "timestamps": [frame["camera_timestamp"] for frame in frames],
        "track_tokens": [track["instance_token"] for track in tracks], "exposure_comparable": comparable,
        "grid": {key: getattr(grid, key) for key in ("x_min", "x_max", "y_min", "y_max", "resolution")},
        "counts": {"visible_all": int(visible.all(axis=0).sum()), "receiver": int(receiver.sum()),
                   "physical_box_excluded": int(occupied.any(axis=0).sum()),
                   "physical_ray_volume_excluded": int(volume_evidence["physical_ray_volume_support"].any(axis=0).sum()),
                   "illumination_events": int(changed.sum()), "shadow_candidates": int(candidate.sum()),
                   "persistent_dark_receivers": int(persistent_dark.sum()),
                   "flow_supported_receivers": int((flow_support & receiver).sum())},
        "state_legend": {"0": "unknown", "1": "stable_receiver_observation_not_negative_label",
                         "2": "physical_box_projection_support_not_dense_surface_motion",
                         "3": "stable_receiver_appearance_change_not_shadow_truth"},
        "units": "BEV x/y and object displacement: metres; flow: source image pixels per actual interval",
        "is_shadow_gt": False, "train_approved": False}
    # Unsupported values are explicitly absent, not zero-valued physical evidence.
    for key in ("appearance_L_raw", "appearance_L_compensated"):
        arrays[key] = np.where(visible, arrays[key], np.nan)
    arrays["temporal_illumination_change_L"] = np.where(receiver[None], delta, np.nan)
    for key in ("expected_geometric_flow_px", "observed_optical_flow_px"):
        arrays[key] = np.where(receiver[None, ..., None], arrays[key], np.nan)
    arrays["flow_residual_px"] = np.where(receiver[None], arrays["flow_residual_px"], np.nan)
    return arrays, metadata


def image_mask_to_bev(mask, projection, world):
    uv, _, valid = project_world(world.reshape(-1, 3), projection)
    height, width = mask.shape
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    coordinates = np.where(valid[:, None], uv, -1).reshape(*world.shape[:2], 2).astype(np.float32)
    return (sample_at(mask.astype(np.uint8), coordinates, nearest=True) > 0) & valid.reshape(world.shape[:2])


def bev_mask_to_image(mask, projection, basis_world, grid, image_shape):
    """Nearest-cell pullback, to be intersected with actual observed image support."""
    if mask.shape != grid.shape:
        raise ValueError("BEV mask and metric grid differ")
    yy, xx = np.indices(image_shape)
    matrix = np.asarray(projection) @ np.asarray(basis_world)
    if np.linalg.cond(matrix) > 1e10:
        raise ValueError("Singular receiver projection")
    coordinates = np.linalg.solve(matrix, np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)]))
    valid = np.isfinite(coordinates).all(axis=0) & (coordinates[2] > 1e-8)
    xy = np.zeros((xx.size, 2))
    xy[valid] = (coordinates[:2, valid]/coordinates[2, valid]).T
    i, j, inside = grid.xy_to_ij(xy)
    selected = valid & inside
    output = np.zeros(xx.size, bool)
    output[selected] = mask[i[selected], j[selected]]
    return output.reshape(image_shape)
