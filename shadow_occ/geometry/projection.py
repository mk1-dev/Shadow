from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shadow_occ.geometry.transforms import apply_transform, invert_transform


@dataclass
class CameraCalibration:
    intrinsic: np.ndarray
    sensor_to_ego: np.ndarray
    ego_to_global: np.ndarray
    image_size: tuple[int, int] | None = None

    @property
    def global_to_sensor(self) -> np.ndarray:
        return invert_transform(self.ego_to_global @ self.sensor_to_ego)


def project_points_to_image(
    points: np.ndarray,
    occ_to_global: np.ndarray,
    camera: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points from an occupancy frame to a nuScenes camera image."""
    if points.size == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=bool), np.zeros((0,))

    points_global = apply_transform(points, occ_to_global)
    points_cam = apply_transform(points_global, camera.global_to_sensor)
    depth = points_cam[:, 2]
    valid = depth > 1e-4

    uvw = points_cam @ camera.intrinsic.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)

    if camera.image_size is not None:
        width, height = camera.image_size
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid, depth


def sample_mask_at_uv(mask: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(len(uv), dtype=bool)
    h, w = mask.shape[:2]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    inside = valid & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out[inside] = mask[y[inside], x[inside]].astype(bool)
    return out


def ground_reprojection_error(
    points: np.ndarray,
    occ_to_global: np.ndarray,
    camera: CameraCalibration,
    ground_z: float,
) -> dict[str, float | int]:
    """Measure image -> ground-plane -> image round-trip error in pixels."""
    uv, valid, _ = project_points_to_image(points, occ_to_global, camera)
    if not np.any(valid):
        return {"count": 0, "mean_px": float("nan"), "median_px": float("nan"), "p95_px": float("nan"), "max_px": float("nan")}

    uv_valid = uv[valid]
    k_inv = np.linalg.inv(camera.intrinsic)
    rays_sensor = (k_inv @ np.concatenate([uv_valid, np.ones((len(uv_valid), 1))], axis=1).T).T
    sensor_to_global = camera.ego_to_global @ camera.sensor_to_ego
    origin = sensor_to_global[:3, 3]
    rays_global = rays_sensor @ sensor_to_global[:3, :3].T
    denominator = rays_global[:, 2]
    intersect_valid = np.abs(denominator) > 1e-8
    scale = (ground_z - origin[2]) / np.where(intersect_valid, denominator, 1.0)
    reconstructed = origin[None, :] + rays_global * scale[:, None]
    reconstructed_uv, reconstructed_valid, _ = _project_global_points(reconstructed, camera)
    good = intersect_valid & reconstructed_valid
    if not np.any(good):
        return {"count": 0, "mean_px": float("nan"), "median_px": float("nan"), "p95_px": float("nan"), "max_px": float("nan")}
    errors = np.linalg.norm(reconstructed_uv[good] - uv_valid[good], axis=1)
    return {
        "count": int(errors.size),
        "mean_px": float(np.mean(errors)),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "max_px": float(np.max(errors)),
    }


def _project_global_points(points_global: np.ndarray, camera: CameraCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_cam = apply_transform(points_global, camera.global_to_sensor)
    depth = points_cam[:, 2]
    valid = depth > 1e-4
    uvw = points_cam @ camera.intrinsic.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    if camera.image_size is not None:
        width, height = camera.image_size
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid, depth
