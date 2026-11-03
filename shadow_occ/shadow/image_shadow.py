from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

from shadow_occ.config import ShadowConfig
from shadow_occ.geometry.projection import CameraCalibration


@dataclass
class TemporalImageDifference:
    """Previous-frame evidence sampled in the current camera image plane."""

    mask: np.ndarray
    confidence: np.ndarray
    light_difference: np.ndarray
    valid: np.ndarray
    threshold: float
    overlap: int
    exposure_scale: float
    exposure_offset: float


def read_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def dark_region_mask(image_bgr: np.ndarray, cfg: ShadowConfig) -> np.ndarray:
    """Find low-illumination image regions using HSV value and light morphology."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    adaptive_thr = np.percentile(value, cfg.dark_percentile)
    threshold = min(float(cfg.max_dark_value), float(adaptive_thr))
    mask = (value <= threshold) & (saturation >= cfg.min_saturation)

    kernel_size = max(1, int(cfg.morph_kernel))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_u8 = mask.astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask = mask_u8.astype(bool)
    return mask


def temporal_lab_difference(
    current_bgr: np.ndarray,
    previous_bgr: np.ndarray,
    current_camera: CameraCalibration,
    previous_camera: CameraCalibration,
    ground_z: float,
    cfg: ShadowConfig,
) -> TemporalImageDifference:
    """Compare two frames after ego-motion alignment on an approximate ground plane.

    The homography is derived from the two calibrated camera poses and the
    global ground plane. This keeps the comparison in a common world frame,
    so vehicle motion does not become a false illumination change.
    """
    current_lab = _lab_float(current_bgr)
    previous_lab = _lab_float(previous_bgr)
    h, w = current_bgr.shape[:2]
    map_x, map_y, valid = _previous_to_current_ground_warp(
        current_camera, previous_camera, ground_z, (h, w), previous_lab.shape[:2]
    )
    previous_aligned = cv2.remap(
        previous_lab,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid &= np.isfinite(current_lab).all(axis=2) & np.isfinite(previous_aligned).all(axis=2)
    valid &= previous_aligned[:, :, 0] > 0.0
    current_l = current_lab[:, :, 0]
    previous_l = previous_aligned[:, :, 0]
    exposure_scale = 1.0
    exposure_offset = 0.0
    overlap = int(np.count_nonzero(valid))
    if cfg.exposure_compensation and overlap >= cfg.temporal_min_overlap:
        previous_l, exposure_scale, exposure_offset = _local_exposure_alignment(
            previous_l,
            current_l,
            valid,
        )

    light_difference = np.abs(current_l - previous_l).astype(np.float32)
    chroma_difference = np.linalg.norm(
        current_lab[:, :, 1:3] - previous_aligned[:, :, 1:3], axis=2
    ).astype(np.float32)
    if overlap:
        threshold = max(
            float(cfg.min_temporal_l_diff),
            float(np.percentile(light_difference[valid], cfg.temporal_diff_percentile)),
        )
    else:
        threshold = float(cfg.min_temporal_l_diff)
    candidate = valid & (light_difference >= threshold)
    if cfg.max_temporal_chroma_diff > 0:
        candidate &= chroma_difference <= cfg.max_temporal_chroma_diff
    kernel_size = max(1, int(cfg.temporal_dilate_kernel))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        candidate = cv2.dilate(candidate.astype(np.uint8), kernel, iterations=1).astype(bool)
        candidate &= valid
    confidence = np.clip(light_difference / max(2.0 * threshold, 1e-6), 0.0, 1.0)
    confidence = confidence.astype(np.float32) * valid.astype(np.float32)
    return TemporalImageDifference(
        mask=candidate.astype(np.uint8),
        confidence=confidence,
        light_difference=light_difference,
        valid=valid,
        threshold=threshold,
        overlap=overlap,
        exposure_scale=float(exposure_scale),
        exposure_offset=float(exposure_offset),
    )


def _lab_float(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    return lab


def _robust_exposure_alignment(previous_l: np.ndarray, current_l: np.ndarray) -> tuple[float, float]:
    previous_l = np.asarray(previous_l, dtype=np.float32)
    current_l = np.asarray(current_l, dtype=np.float32)
    previous_iqr = float(np.percentile(previous_l, 75) - np.percentile(previous_l, 25))
    current_iqr = float(np.percentile(current_l, 75) - np.percentile(current_l, 25))
    scale = current_iqr / max(previous_iqr, 1e-3)
    scale = float(np.clip(scale, 0.6, 1.7))
    offset = float(np.median(current_l) - scale * np.median(previous_l))
    return scale, offset


def _local_exposure_alignment(
    previous_l: np.ndarray,
    current_l: np.ndarray,
    valid: np.ndarray,
    window: int = 31,
) -> tuple[np.ndarray, float, float]:
    """Match low-frequency local illumination while retaining local contrast."""
    global_scale, global_offset = _robust_exposure_alignment(
        previous_l[valid], current_l[valid]
    )
    kernel_size = max(3, int(window) | 1)
    kernel = (kernel_size, kernel_size)
    valid_float = valid.astype(np.float32)
    denominator = np.maximum(cv2.GaussianBlur(valid_float, kernel, 0), 1e-3)
    previous_mean = cv2.GaussianBlur(previous_l * valid_float, kernel, 0) / denominator
    current_mean = cv2.GaussianBlur(current_l * valid_float, kernel, 0) / denominator
    previous_residual = (previous_l - previous_mean) * valid_float
    current_residual = (current_l - current_mean) * valid_float
    previous_std = np.sqrt(
        cv2.GaussianBlur(previous_residual * previous_residual, kernel, 0) / denominator
    )
    previous_std = np.maximum(previous_std, 1.0)
    current_std = np.sqrt(
        cv2.GaussianBlur(current_residual * current_residual, kernel, 0) / denominator
    )
    current_std = np.maximum(current_std, 1.0)
    local_scale = np.clip(current_std / previous_std, 0.6, 1.7)
    aligned = (previous_l - previous_mean) * local_scale + current_mean
    aligned = np.where(valid, aligned, previous_l * global_scale + global_offset)
    return aligned.astype(np.float32), global_scale, global_offset


def _previous_to_current_ground_warp(
    current_camera: CameraCalibration,
    previous_camera: CameraCalibration,
    ground_z: float,
    current_shape: tuple[int, int],
    previous_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_to_global = _ground_plane_image_matrix(current_camera, ground_z)
    previous_to_global = _ground_plane_image_matrix(previous_camera, ground_z)
    try:
        current_to_previous = previous_to_global @ np.linalg.inv(current_to_global)
    except np.linalg.LinAlgError:
        h, w = current_shape
        return (
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=bool),
        )
    h, w = current_shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    pixels = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(h * w, dtype=np.float32)], axis=0)
    mapped = current_to_previous @ pixels
    denom = mapped[2].reshape(h, w)
    map_x = (mapped[0].reshape(h, w) / np.maximum(denom, 1e-6)).astype(np.float32)
    map_y = (mapped[1].reshape(h, w) / np.maximum(denom, 1e-6)).astype(np.float32)
    previous_h, previous_w = previous_shape
    valid = (denom > 1e-6) & (map_x >= 0) & (map_x < previous_w) & (map_y >= 0) & (map_y < previous_h)
    return map_x, map_y, valid


def _ground_plane_image_matrix(camera: CameraCalibration, ground_z: float) -> np.ndarray:
    projection = camera.intrinsic @ camera.global_to_sensor[:3, :]
    return np.stack(
        [projection[:, 0], projection[:, 1], projection[:, 2] * ground_z + projection[:, 3]],
        axis=1,
    )
