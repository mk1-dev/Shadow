from __future__ import annotations

import cv2
import numpy as np


def plane_matrix(z: float) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, z], [0, 0, 1]], dtype=np.float64)


def ground_homographies(current: dict, previous: dict, z: float = 0.0) -> dict:
    """Separate same-world-ground alignment from caster-translation alignment.

    H matrices map PREVIOUS image pixels into CURRENT image pixels. Caster
    translation approximates its shadow displacement under stationary light;
    it must not be used to compare texture at the same world ground position.
    """
    for key in ("scene_name", "camera", "instance_token"):
        if current[key] != previous[key]:
            raise ValueError(f"Temporal pairing mismatch: {key}")
    ec, ep = np.asarray(current["ego_to_global"]), np.asarray(previous["ego_to_global"])
    pc, pp = np.asarray(current["ego_to_img"]), np.asarray(previous["ego_to_img"])
    cc = (ec @ np.r_[current["caster_center_ego"], 1])[:3]
    cp = (ep @ np.r_[previous["caster_center_ego"], 1])[:3]
    shift = np.eye(4)
    shift[:2, 3] = cp[:2]-cc[:2]
    current_plane = pc @ plane_matrix(z)
    previous_world = pp @ np.linalg.inv(ep) @ ec @ plane_matrix(z)
    previous_caster = pp @ np.linalg.inv(ep) @ shift @ ec @ plane_matrix(z)
    world = current_plane @ np.linalg.inv(previous_world)
    caster = current_plane @ np.linalg.inv(previous_caster)
    seconds = (int(current["timestamp"])-int(previous["timestamp"]))/1e6
    if seconds <= 0:
        raise ValueError("Previous frame must precede the current frame")
    return {"world_previous_to_current": world, "caster_previous_to_current": caster, "dt_seconds": seconds, "caster_displacement_m": float(np.linalg.norm(cc[:2]-cp[:2])), "caster_speed_mps": float(np.linalg.norm(cc[:2]-cp[:2])/seconds)}


def yaw_difference_degrees(a, b) -> float:
    def yaw(q):
        w, x, y, z = q
        return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    delta = yaw(a)-yaw(b)
    return float(abs(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta)))))


def warp(array, homography, shape, nearest=False, border=0):
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Homography must be a finite 3x3 matrix")
    return cv2.warpPerspective(array, matrix, (shape[1], shape[0]), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def sampling_grid(homography, current_shape, previous_shape, resized_shape):
    """grid_sample coordinates for the previous prediction, in current layout."""
    height, width = resized_shape
    cy, cx = np.indices((height, width), dtype=np.float64)
    # Resizing follows align_corners=False pixel-center coordinates.
    x = (cx+0.5)*current_shape[1]/width-0.5
    y = (cy+0.5)*current_shape[0]/height-0.5
    mapped = np.linalg.inv(np.asarray(homography)) @ np.stack([x.ravel(), y.ravel(), np.ones(height*width)])
    denominator = mapped[2]
    valid = np.abs(denominator) > 1e-8
    px = mapped[0]/np.where(valid, denominator, 1)
    py = mapped[1]/np.where(valid, denominator, 1)
    valid &= (px >= 0) & (px <= previous_shape[1]-1) & (py >= 0) & (py <= previous_shape[0]-1)
    gx = 2*(px+0.5)/previous_shape[1]-1
    gy = 2*(py+0.5)/previous_shape[0]-1
    grid = np.stack([gx, gy], axis=-1).reshape(height, width, 2).astype(np.float32)
    grid[~valid.reshape(height, width)] = 2.0
    return grid, valid.reshape(height, width)


def temporal_observations(current_bgr, previous_bgr, current_target, previous_target,
                          current_ground, previous_ground, matrices, *,
                          static_speed=0.3, min_overlap=64, max_background_residual=8.0,
                          max_exposure_offset=25.0, light_scale=8.0, chroma_scale=18.0):
    """Measure stable evidence and motion-related light change without a diff gate."""
    shape = current_bgr.shape[:2]
    hw = matrices["world_previous_to_current"]
    hc = matrices["caster_previous_to_current"]
    previous_world = warp(previous_bgr, hw, shape)
    previous_caster = warp(previous_bgr, hc, shape)
    world_ground = warp(previous_ground.astype(np.uint8), hw, shape, nearest=True) > 0
    caster_ground = warp(previous_ground.astype(np.uint8), hc, shape, nearest=True) > 0
    shared_world = current_ground & world_ground
    shared_caster = current_ground & caster_ground
    world_target = warp(previous_target, hw, shape, nearest=True, border=255)
    caster_target = warp(previous_target, hc, shape, nearest=True, border=255)
    lab = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    old_lab = cv2.cvtColor(previous_world, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100/255
    old_lab[..., 0] *= 100/255
    background = shared_world & (current_target == 0) & (world_target == 0)
    enough = int(background.sum()) >= min_overlap
    scale, offset, residual = 1.0, 0.0, None
    if enough:
        old, new = old_lab[..., 0][background], lab[..., 0][background]
        old_iqr = float(np.percentile(old, 75)-np.percentile(old, 25))
        new_iqr = float(np.percentile(new, 75)-np.percentile(new, 25))
        scale = float(np.clip(new_iqr/max(old_iqr, 1), 0.7, 1.4)) if old_iqr >= 1 else 1.0
        offset = float(np.median(new-scale*old))
        residual = float(np.median(np.abs(new-(scale*old+offset))))
    delta_l = lab[..., 0]-(old_lab[..., 0]*scale+offset)
    delta_chroma = np.linalg.norm(lab[..., 1:]-old_lab[..., 1:], axis=-1)
    light_ok = enough and residual <= max_background_residual and abs(offset) <= max_exposure_offset
    mode = "static" if matrices["caster_speed_mps"] <= static_speed else "moving"
    photometric = np.exp(-np.abs(delta_l)/light_scale)*np.exp(-delta_chroma/chroma_scale)
    if mode == "moving":
        # Change at a world-fixed surface supports moving-shadow onset. Small
        # differences remain allowed where the moved shadow covers a similar patch.
        photometric = (0.5+0.5*np.clip(-delta_l/light_scale, 0, 1))*np.exp(-delta_chroma/chroma_scale)
    positive_agree = shared_caster & (current_target == 1) & (caster_target == 1)
    negative_agree = shared_caster & (current_target == 0) & (caster_target == 0)
    kernel = np.ones((3, 3), np.uint8)
    interior = (cv2.erode(positive_agree.astype(np.uint8), kernel) > 0) | (cv2.erode(negative_agree.astype(np.uint8), kernel) > 0)
    weight = interior.astype(np.float32)*photometric*float(light_ok)
    if mode == "moving":
        weight *= 0.5
    before = (current_target == 1) & shared_caster
    repeated = before & (caster_target == 1)
    world_repeated = (current_target == 1) & (world_target == 1) & shared_world
    onset = shared_world & (current_target == 1) & (world_target == 0)
    recovery = shared_world & (current_target == 0) & (world_target == 1)
    comparison = shared_world & shared_caster & (current_target == 1) & (world_target != 255) & (caster_target != 255)
    motion_advantage = float(((caster_target[comparison] == 1).astype(np.float32)-(world_target[comparison] == 1)).mean()) if comparison.any() else None
    details = {
        "mode": mode, "background_points": int(background.sum()),
        "shared_world_pixels": int(shared_world.sum()), "shared_caster_pixels": int(shared_caster.sum()),
        "exposure_scale": scale, "exposure_offset_L": offset,
        "background_residual_L": residual, "lighting_comparable": bool(light_ok),
        "current_shadow_pixels_in_overlap": int(before.sum()), "repeated_shadow_pixels": int(repeated.sum()),
        "shadow_retention": float(repeated.sum()/before.sum()) if before.any() else None,
        "world_abs_light_change_mean": float(np.abs(delta_l[shared_world]).mean()) if shared_world.any() else None,
        "repeated_shadow_small_change_ratio": float((np.abs(delta_l[world_repeated]) < 6.0).mean()) if world_repeated.any() else None,
        "small_change_threshold_L": 6.0,
        "consistency_weight_sum": float(weight.sum()),
        "onset_pixels": int(onset.sum()), "recovery_pixels": int(recovery.sum()),
        "onset_darkening_ratio": float((delta_l[onset] < -6.0).mean()) if onset.any() and light_ok else None,
        "recovery_brightening_ratio": float((delta_l[recovery] > 6.0).mean()) if recovery.any() and light_ok else None,
        "motion_alignment_comparison_pixels": int(comparison.sum()),
        "motion_alignment_advantage": motion_advantage if light_ok else None,
    }
    return {"previous_world": previous_world, "previous_caster": previous_caster, "previous_target_caster": caster_target, "shared_caster": shared_caster, "shared_world": shared_world, "delta_l": delta_l, "delta_chroma": delta_chroma, "photometric_confidence": photometric, "weight": weight, "details": details}
