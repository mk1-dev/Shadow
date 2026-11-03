from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shadow_occ.config import ShadowConfig
from shadow_occ.data.occ3d_loader import Occ3DSample
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.geometry.projection import (
    CameraCalibration,
    ground_reprojection_error,
    project_points_to_image,
    sample_mask_at_uv,
)
from shadow_occ.shadow.image_shadow import TemporalImageDifference, temporal_lab_difference


@dataclass
class CameraShadowProjection:
    bev_mask: np.ndarray
    confidence: np.ndarray
    projected_points: int
    selected_points: int
    temporal_selected_points: int
    temporal_confidence: np.ndarray
    reprojection: dict[str, float | int]
    light_difference_mean: float
    temporal_overlap: int
    temporal_threshold: float
    exposure_scale: float
    exposure_offset: float


def project_dark_ground_to_bev(
    ground_points: np.ndarray,
    dark_mask: np.ndarray,
    occ_to_global: np.ndarray,
    camera: CameraCalibration,
    grid: BEVGrid,
    current_image: np.ndarray | None = None,
    previous_image: np.ndarray | None = None,
    previous_camera: CameraCalibration | None = None,
    ground_z: float | None = None,
    cfg: ShadowConfig | None = None,
) -> CameraShadowProjection:
    uv, valid, _depth = project_points_to_image(ground_points, occ_to_global, camera)
    selected = sample_mask_at_uv(dark_mask, uv, valid)
    temporal_confidence = np.zeros(grid.shape, dtype=np.float32)
    temporal_selected = selected.copy()
    light_difference_mean = 0.0
    temporal_overlap = 0
    temporal_threshold = 0.0
    exposure_scale = 1.0
    exposure_offset = 0.0
    temporal_observation: TemporalImageDifference | None = None
    if (
        cfg is not None
        and cfg.use_temporal_difference
        and current_image is not None
        and previous_image is not None
        and previous_camera is not None
        and ground_z is not None
    ):
        temporal_observation = temporal_lab_difference(
            current_image,
            previous_image,
            camera,
            previous_camera,
            ground_z,
            cfg,
        )
        temporal_selected = sample_mask_at_uv(temporal_observation.mask, uv, valid)
        selected &= temporal_selected
        light_values = temporal_observation.light_difference
        temporal_overlap = temporal_observation.overlap
        temporal_threshold = temporal_observation.threshold
        exposure_scale = temporal_observation.exposure_scale
        exposure_offset = temporal_observation.exposure_offset
        if np.any(temporal_observation.valid):
            light_difference_mean = float(np.mean(light_values[temporal_observation.valid]))
    points = ground_points[selected]

    bev = np.zeros(grid.shape, dtype=np.uint8)
    conf = np.zeros(grid.shape, dtype=np.float32)
    if len(points):
        i, j, inside = grid.xy_to_ij(points[:, :2])
        bev[i[inside], j[inside]] = 1
        np.add.at(conf, (i[inside], j[inside]), 1.0)
        conf = np.clip(conf, 0.0, 1.0)
        if temporal_observation is not None:
            temporal_values = temporal_observation.confidence
            temporal_confidence[i[inside], j[inside]] = np.maximum(
                temporal_confidence[i[inside], j[inside]],
                temporal_values[  # type: ignore[index]
                    np.rint(uv[selected, 1][inside]).astype(np.int64),
                    np.rint(uv[selected, 0][inside]).astype(np.int64),
                ],
            )

    reprojection = (
        ground_reprojection_error(
            ground_points,
            occ_to_global,
            camera,
            float(ground_z),
        )
        if ground_z is not None
        else {"count": 0, "mean_px": float("nan"), "median_px": float("nan"), "p95_px": float("nan"), "max_px": float("nan")}
    )

    return CameraShadowProjection(
        bev_mask=bev,
        confidence=conf,
        projected_points=int(np.count_nonzero(valid)),
        selected_points=int(np.count_nonzero(selected)),
        temporal_selected_points=int(np.count_nonzero(temporal_selected)) if temporal_observation is not None else 0,
        temporal_confidence=temporal_confidence,
        reprojection=reprojection,
        light_difference_mean=light_difference_mean,
        temporal_overlap=temporal_overlap,
        temporal_threshold=temporal_threshold,
        exposure_scale=exposure_scale,
        exposure_offset=exposure_offset,
    )


def merge_camera_projections(
    projections: list[CameraShadowProjection],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not projections:
        raise ValueError("No projections to merge.")
    masks = np.stack([p.bev_mask for p in projections], axis=0).astype(bool)
    conf = np.stack([p.confidence for p in projections], axis=0)
    temporal = np.stack([p.temporal_confidence for p in projections], axis=0)
    return masks.any(axis=0).astype(np.uint8), np.clip(conf.sum(axis=0), 0.0, 1.0), np.max(temporal, axis=0)


def cleanup_bev_mask(mask: np.ndarray, cfg: ShadowConfig) -> np.ndarray:
    kernel_size = max(1, int(cfg.morph_kernel))
    out = mask.astype(np.uint8)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return (out > 0).astype(np.uint8)


def pseudo_shadow_from_occ_and_images(
    sample: Occ3DSample,
    ground_points: np.ndarray,
    camera_dark_masks: list[tuple[np.ndarray, CameraCalibration]],
    occ_to_global: np.ndarray,
    grid: BEVGrid,
    cfg: ShadowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    projections = [
        _project_observation(ground_points, observation, occ_to_global, grid, cfg)
        for observation in camera_dark_masks
    ]
    if projections:
        mask, confidence, temporal_confidence = merge_camera_projections(projections)
    else:
        mask = np.zeros(grid.shape, dtype=np.uint8)
        confidence = np.zeros(grid.shape, dtype=np.float32)
        temporal_confidence = np.zeros(grid.shape, dtype=np.float32)
    mask = cleanup_bev_mask(mask, cfg)
    meta = {
        "geometry_path": str(getattr(sample, "path", "unknown")),
        "num_ground_points": int(len(ground_points)),
        "camera_projection": [
            {
                "projected_points": p.projected_points,
                "selected_points": p.selected_points,
                "temporal_selected_points": p.temporal_selected_points,
                "light_difference_mean": p.light_difference_mean,
                "temporal_overlap": p.temporal_overlap,
                "temporal_threshold": p.temporal_threshold,
                "exposure_scale": p.exposure_scale,
                "exposure_offset": p.exposure_offset,
                "reprojection": p.reprojection,
            }
            for p in projections
        ],
        "temporal_difference_enabled": bool(cfg.use_temporal_difference),
    }
    return mask, confidence, temporal_confidence, meta


def _project_observation(
    ground_points: np.ndarray,
    observation: tuple,
    occ_to_global: np.ndarray,
    grid: BEVGrid,
    cfg: ShadowConfig,
) -> CameraShadowProjection:
    """Accept old (mask, calibration) tuples and new temporal observations."""
    dark_mask, camera = observation[0], observation[1]
    current_image = observation[2] if len(observation) > 2 else None
    previous_image = observation[3] if len(observation) > 3 else None
    previous_camera = observation[4] if len(observation) > 4 else None
    ground_z = observation[5] if len(observation) > 5 else None
    return project_dark_ground_to_bev(
        ground_points,
        dark_mask,
        occ_to_global,
        camera,
        grid,
        current_image=current_image,
        previous_image=previous_image,
        previous_camera=previous_camera,
        ground_z=ground_z,
        cfg=cfg,
    )
