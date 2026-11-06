"""Stage 1/2 evidence only: no shadow labels or caster truth are inferred here."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EvidenceConfig:
    min_background: int = 128
    fb_absolute_px: float = 1.5
    fb_relative: float = 0.05
    ground_flow_tolerance_px: float = 3.0
    max_background_residual_L: float = 4.0
    max_exposure_offset_L: float = 20.0
    light_threshold_L: float = 6.0
    chroma_scale: float = 12.0
    darkness_ratio: float = 0.88
    receiver_support_radius: int = 7
    receiver_support_min_pixels: int = 16
    receiver_support_min_fraction: float = 0.6
    receiver_texture_L: float = 1.0

    def validate(self):
        if self.min_background < 16 or not 0 < self.darkness_ratio < 1:
            raise ValueError("Invalid background count or darkness ratio")
        values = [self.fb_absolute_px, self.fb_relative, self.ground_flow_tolerance_px,
                  self.max_background_residual_L, self.max_exposure_offset_L,
                  self.light_threshold_L, self.chroma_scale]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Evidence thresholds must be finite and positive")
        if (not isinstance(self.receiver_support_radius, int) or self.receiver_support_radius < 1
                or not isinstance(self.receiver_support_min_pixels, int) or self.receiver_support_min_pixels < 1
                or not np.isfinite([self.receiver_support_min_fraction, self.receiver_texture_L]).all()
                or not 0 < self.receiver_support_min_fraction <= 1 or self.receiver_texture_L <= 0):
            raise ValueError("Invalid local receiver-flow support settings")


def project_world(points, projection):
    points = np.asarray(points, np.float64)
    matrix = np.asarray(projection, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (3, 4):
        raise ValueError("Expected Nx3 world points and a 3x4 projection")
    pixels = np.c_[points, np.ones(len(points))] @ matrix.T
    depth = pixels[:, 2]
    valid = np.isfinite(pixels).all(axis=1) & (depth > 0.1)
    uv = np.full((len(points), 2), np.nan)
    uv[valid] = pixels[valid, :2] / depth[valid, None]
    return uv, depth, valid


def decompose_object_motion(points_t, points_next, projection_t, projection_next, dt):
    """Decompose box landmarks, NOT surface flow, at the object's own depth."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Object motion requires a positive time interval")
    a, b = np.asarray(points_t, np.float64), np.asarray(points_next, np.float64)
    if a.shape != b.shape:
        raise ValueError("Object landmark correspondences must match")
    p0, _, v0 = project_world(a, projection_t)
    counterfactual, _, vc = project_world(a, projection_next)
    p1, _, v1 = project_world(b, projection_next)
    valid = v0 & vc & v1
    total, camera, physical = p1-p0, counterfactual-p0, p1-counterfactual
    for array in (total, camera, physical):
        array[~valid] = np.nan
    return {"valid": valid, "pixels_t": p0, "pixels_next": p1,
            "total_displacement_px": total, "camera_displacement_px": camera,
            "object_residual_displacement_px": physical,
            "landmark_velocity_world_mps": (b-a)/dt}


def interpolate_box_state(left, right, timestamp):
    """Interpolate one track only inside its annotated time bracket."""
    from pyquaternion import Quaternion

    if left["instance_token"] != right["instance_token"]:
        raise ValueError("Cannot interpolate across object identities")
    t0, t1, t = int(left["timestamp"]), int(right["timestamp"]), int(timestamp)
    if t1 < t0 or t < t0 or t > t1 or (t1 == t0 and t != t0):
        raise ValueError("Object state extrapolation is not allowed")
    amount = (t-t0)/(t1-t0) if t1 > t0 else 0.0
    center = (1-amount)*np.asarray(left["translation"])+amount*np.asarray(right["translation"])
    size = (1-amount)*np.asarray(left["size"])+amount*np.asarray(right["size"])
    rotation = Quaternion.slerp(Quaternion(left["rotation"]), Quaternion(right["rotation"]), amount)
    # Corner ordering matches nuScenes Box.corners: local x length, y width.
    signs = np.array([[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1],
                      [-1, 1, 1], [-1, -1, 1], [-1, -1, -1], [-1, 1, -1]])
    corners = signs*np.array([size[1], size[0], size[2]])/2
    corners = corners @ rotation.rotation_matrix.T+center
    return {"instance_token": left["instance_token"], "timestamp": t,
            "translation": center, "size": size, "rotation": rotation.elements,
            "corners": corners, "interpolation_fraction": amount}


def fit_ground_plane(points_ego, *, seed=0, iterations=192, tolerance=0.15,
                     minimum_points=80, maximum_tilt_degrees=15.0):
    """Conservative local lower-envelope LiDAR plane, not semantic ground GT."""
    points = np.asarray(points_ego, np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    radius = np.linalg.norm(points[:, :2], axis=1)
    # Ego origin can be near the road, not at LiDAR height. Include both cases.
    points = points[(radius >= 2) & (radius <= 35) & (points[:, 2] >= -3.5) & (points[:, 2] <= 1.0)]
    if len(points) < minimum_points:
        raise ValueError("insufficient_ground_plane_points")
    cells = np.floor(points[:, :2]).astype(np.int32)
    _, cell_ids = np.unique(cells, axis=0, return_inverse=True)
    order = np.lexsort((points[:, 2], cell_ids))
    first = np.r_[True, np.diff(cell_ids[order]) != 0]
    points = points[order[first]]
    if len(points) < minimum_points:
        raise ValueError("insufficient_independent_ground_cells")
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if normal[2] < 0:
            normal *= -1
        if normal[2] < np.cos(np.deg2rad(maximum_tilt_degrees)):
            continue
        distance = np.abs(points @ normal-normal @ a)
        count = int((distance < tolerance).sum())
        if count > best_count:
            best, best_count = distance < tolerance, count
    if best is None or best_count < minimum_points or best_count/len(points) < 0.5:
        raise ValueError("ground_plane_not_supported")
    # z = ax + by + c is stable for the bounded, near-horizontal plane.
    coefficients = np.linalg.lstsq(np.c_[points[best, :2], np.ones(best_count)], points[best, 2], rcond=None)[0]
    normal = np.r_[-coefficients[:2], 1.0]
    norm = np.linalg.norm(normal)
    plane = np.r_[normal, -coefficients[2]]/norm
    residuals = np.abs(points @ plane[:3]+plane[3])
    inliers = residuals <= tolerance
    if (inliers.sum() < minimum_points or inliers.mean() < 0.5
            or plane[2] < np.cos(np.deg2rad(maximum_tilt_degrees))):
        raise ValueError("refined_ground_plane_not_supported")
    return plane, {"source": "local_lidar_lower_envelope", "independent_cells": len(points),
                   "inlier_cells": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
                   "residual_p95_m": float(np.percentile(residuals[inliers], 95)),
                   "semantic_ground_verified": False}


def plane_basis(plane):
    plane = np.asarray(plane, np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all() or abs(plane[2]) < 1e-8:
        raise ValueError("Expected a finite plane with a nonzero z normal")
    a, b, c, d = plane
    return np.array([[1, 0, 0], [0, 1, 0], [-a/c, -b/c, -d/c], [0, 0, 1]])


def ground_homography(projection_t, projection_next, basis_world):
    a, b = np.asarray(projection_t) @ basis_world, np.asarray(projection_next) @ basis_world
    if not np.isfinite([a, b]).all() or max(np.linalg.cond(a), np.linalg.cond(b)) > 1e10:
        raise ValueError("singular_ground_projection")
    return b @ np.linalg.inv(a)


def ground_correspondence(projection_t, projection_next, basis_world, shape):
    """Forward pixel endpoints with positive plane-intersection depths in BOTH views."""
    height, width = shape
    y, x = np.indices(shape, dtype=np.float64)
    grid = np.stack([x, y], axis=-1)
    a, b = np.asarray(projection_t) @ basis_world, np.asarray(projection_next) @ basis_world
    ground_homography(projection_t, projection_next, basis_world)
    coordinates = np.linalg.solve(a, np.stack([x.ravel(), y.ravel(), np.ones(height*width)]))
    divisor = coordinates[2]
    valid = np.isfinite(coordinates).all(axis=0) & (divisor > 1e-8)
    coordinates /= np.where(valid, divisor, 1)[None]
    destination = b @ coordinates
    valid &= np.isfinite(destination).all(axis=0) & (destination[2] > 0.1)
    uv = (destination[:2]/np.where(valid, destination[2], 1)).T.reshape(height, width, 2)
    valid = valid.reshape(shape) & (uv[..., 0] >= 0) & (uv[..., 0] <= width-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= height-1)
    uv[~valid] = -1
    return uv.astype(np.float32), (uv-grid).astype(np.float32), valid


def sample_at(array, endpoints, nearest=False):
    return cv2.remap(np.asarray(array), endpoints[..., 0], endpoints[..., 1],
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def forward_backward_evidence(forward, backward, config):
    forward, backward = np.asarray(forward, np.float32), np.asarray(backward, np.float32)
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[-1] != 2:
        raise ValueError("Forward/backward flow must have identical HxWx2 layouts")
    y, x = np.indices(forward.shape[:2], dtype=np.float32)
    endpoints = np.stack([x, y], axis=-1)+forward
    height, width = forward.shape[:2]
    valid = np.isfinite(endpoints).all(axis=-1)
    valid &= (endpoints[..., 0] >= 0) & (endpoints[..., 0] <= width-1) & (endpoints[..., 1] >= 0) & (endpoints[..., 1] <= height-1)
    safe = np.where(np.isfinite(endpoints), endpoints, -1).astype(np.float32)
    reverse = sample_at(backward, safe)
    error = np.linalg.norm(forward+reverse, axis=-1)
    tolerance = config.fb_absolute_px+config.fb_relative*(np.linalg.norm(forward, axis=-1)+np.linalg.norm(reverse, axis=-1))
    reliable = valid & np.isfinite(error) & (error <= tolerance)
    return error, reliable


def exposure_fit(old_L, new_L, background, config):
    """Trimmed affine fit on flow-consistent ground; assumes majority stable lighting."""
    old, new = old_L[background], new_L[background]
    result = {"valid": False, "background_pixels": len(old), "scale": 1.0,
              "offset_L": 0.0, "residual_median_L": None, "reason": "insufficient_background"}
    if len(old) < config.min_background:
        return result
    old_iqr = float(np.percentile(old, 75)-np.percentile(old, 25))
    new_iqr = float(np.percentile(new, 75)-np.percentile(new, 25))
    scale = new_iqr/old_iqr if old_iqr >= 2 else 1.0
    offset = float(np.median(new-scale*old))
    # Fit exposure without manufacturing negative shadow labels.
    for _ in range(3):
        errors = np.abs(new-(scale*old+offset))
        keep = errors <= max(1.0, float(np.percentile(errors, 75)))
        if keep.sum() < config.min_background:
            break
        if np.std(old[keep]) >= 2:
            scale, offset = np.linalg.lstsq(np.c_[old[keep], np.ones(keep.sum())], new[keep], rcond=None)[0]
        else:
            offset = float(np.median(new[keep]-scale*old[keep]))
    residual = float(np.median(np.abs(new-(scale*old+offset))))
    valid = bool(np.isfinite([scale, offset, residual]).all() and 0.7 <= scale <= 1.4
                 and abs(offset) <= config.max_exposure_offset_L and residual <= config.max_background_residual_L)
    result.update(valid=valid, scale=float(scale), offset_L=float(offset), residual_median_L=residual,
                  reason="comparable" if valid else "unstable_exposure_or_background")
    return result


def receiver_flow_evidence(lightness, visible, forward, geometric_flow, flow_reliable, config):
    """Validate receiver registration without requiring shadow boundaries to follow material flow.

    Neighbour votes are restricted to the same connected supported-ground patch.
    Textureless patches without observable neighbours remain unsupported.
    """
    residual = np.linalg.norm(forward-geometric_flow, axis=-1)
    gx = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)/8
    textured = np.hypot(gx, gy) >= config.receiver_texture_L
    anchors = visible & textured & flow_reliable & np.isfinite(residual)
    anchors &= residual <= config.ground_flow_tolerance_px
    window = 2*config.receiver_support_radius+1
    count = np.zeros(lightness.shape, np.float32)
    fraction = np.zeros_like(count)
    number, components, stats, _ = cv2.connectedComponentsWithStats(visible.astype(np.uint8), connectivity=8)
    for index in range(1, number):
        x, y, width, height, _ = stats[index]
        section = np.s_[y:y+height, x:x+width]
        patch = components[section] == index
        if not np.any(anchors[section] & patch):
            continue
        def total(values):
            return cv2.boxFilter(values.astype(np.float32), -1, (window, window),
                                 normalize=False, borderType=cv2.BORDER_CONSTANT)
        votes = total(anchors[section] & patch)
        observable = total(visible[section] & textured[section] & patch)
        count[section][patch] = votes[patch]
        fraction[section][patch] = (votes/np.maximum(observable, 1))[patch]
    neighbourhood = (count >= config.receiver_support_min_pixels) & (fraction >= config.receiver_support_min_fraction)
    supported = visible & (anchors | neighbourhood)
    return {"receiver_flow_supported": supported, "receiver_flow_anchor": anchors,
            "receiver_flow_support_count": count, "receiver_flow_support_fraction": fraction,
            "receiver_texture": textured & visible}


def pair_evidence(image_t, image_next, ground_t, ground_next, projection_t,
                  projection_next, basis_world, forward, backward, config=None, *, use_observed_flow=True):
    """All output maps use t coordinates; positive delta means recovery at that surface."""
    config = config or EvidenceConfig()
    config.validate()
    shape = image_t.shape[:2]
    if image_t.shape != image_next.shape or ground_t.shape != shape or ground_next.shape != shape:
        raise ValueError("Images and ground masks must share the working pixel frame")
    if np.asarray(forward).shape != (*shape, 2):
        raise ValueError("Flow must share the working image pixel frame")
    endpoints, geometric_flow, geometric_valid = ground_correspondence(projection_t, projection_next, basis_world, shape)
    visible = geometric_valid & (ground_t > 0) & (sample_at((ground_next > 0).astype(np.uint8), endpoints, True) > 0)
    if use_observed_flow:
        fb_error, flow_reliable = forward_backward_evidence(forward, backward, config)
    else:
        fb_error, flow_reliable = np.full(shape, np.nan, np.float32), np.zeros(shape, bool)
    flow_residual = np.linalg.norm(forward-geometric_flow, axis=-1)
    background = visible & flow_reliable & np.isfinite(flow_residual) & (flow_residual <= config.ground_flow_tolerance_px)
    if not use_observed_flow:
        background = visible.copy()
    # Convert before interpolation: Lab remapping and BGR remapping are not equivalent.
    lab_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_next = cv2.cvtColor(image_next, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_t[..., 0] *= 100/255
    lab_next[..., 0] *= 100/255
    receiver = receiver_flow_evidence(lab_t[..., 0], visible, forward, geometric_flow, flow_reliable, config)
    aligned = sample_at(lab_next, endpoints)
    exposure = exposure_fit(aligned[..., 0], lab_t[..., 0], background, config)
    delta = exposure["scale"]*aligned[..., 0]+exposure["offset_L"]-lab_t[..., 0]
    chroma = np.linalg.norm(aligned[..., 1:]-lab_t[..., 1:], axis=-1)
    sigma = max(8.0, min(shape)*0.12)
    ratio_t = lab_t[..., 0]/np.maximum(cv2.GaussianBlur(lab_t[..., 0], (0, 0), sigma), 1)
    ratio_next = lab_next[..., 0]/np.maximum(cv2.GaussianBlur(lab_next[..., 0], (0, 0), sigma), 1)
    darkness_t = np.clip((1-ratio_t)/(1-config.darkness_ratio), 0, 1)
    darkness_next = sample_at(np.clip((1-ratio_next)/(1-config.darkness_ratio), 0, 1), endpoints)
    comparable = visible & bool(exposure["valid"])
    color_support = np.exp(-chroma/config.chroma_scale)
    onset = comparable & (delta <= -config.light_threshold_L) & (darkness_next >= 1) & (color_support >= 0.5)
    recovery = comparable & (delta >= config.light_threshold_L) & (darkness_t >= 1) & (color_support >= 0.5)
    stable = comparable & (np.abs(delta) < config.light_threshold_L) & (darkness_t >= 1) & (darkness_next >= 1) & (color_support >= 0.5)

    def median(array, mask):
        values = array[mask & np.isfinite(array)]
        return float(np.median(values)) if values.size else None

    details = {"visible_ground_pixels": int(visible.sum()),
               "flow_consistent_background_pixels": int(background.sum()) if use_observed_flow else 0,
               "exposure_background_pixels": int(background.sum()),
               "registration_mode": "geometry-dis" if use_observed_flow else "geometry-only",
               "exposure_background": "flow_consistent_ground" if use_observed_flow else "geometrically_visible_ground",
               "flow_fb_reliable_ground_ratio": float(flow_reliable[visible].mean()) if use_observed_flow and visible.any() else None,
               "ground_flow_residual_median_px": median(flow_residual, visible & flow_reliable),
               "fit_background_flow_residual_median_px": median(flow_residual, background),
               "flow_fb_error_median_px": median(fb_error, visible), "exposure": exposure,
               "onset_candidate_pixels": int(onset.sum()), "recovery_candidate_pixels": int(recovery.sum()),
               "stable_dark_candidate_pixels": int(stable.sum()), "shadow_accuracy_available": False,
               "flow_evidence_version": "receiver-support-v2",
               "receiver_flow_supported_pixels": int(receiver["receiver_flow_supported"].sum()),
               "receiver_flow_anchor_pixels": int(receiver["receiver_flow_anchor"].sum()),
               "transition_pixels_without_direct_reliable_flow": int(((onset | recovery) & ~flow_reliable).sum()),
               "status": "evidence_available" if exposure["valid"] else exposure["reason"]}
    return {**receiver, "ground_flow": geometric_flow, "flow_residual_px": flow_residual,
            "flow_fb_error_px": fb_error, "flow_reliable": flow_reliable,
            "visible_ground": visible, "delta_L": np.where(visible, delta, 0),
            "chroma_difference": np.where(visible, chroma, 0),
            "onset_candidate": onset, "recovery_candidate": recovery,
            "stable_dark_candidate": stable, "details": details}
