from __future__ import annotations

import cv2
import numpy as np

from shadow_occ.config import ShadowConfig
from shadow_occ.geometry.projection import CameraCalibration, ground_reprojection_error
from shadow_occ.shadow.image_shadow import temporal_lab_difference


def _downward_camera() -> CameraCalibration:
    intrinsic = np.asarray([[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]])
    rotation = np.diag([1.0, -1.0, -1.0])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[2, 3] = 1.0
    return CameraCalibration(
        intrinsic=intrinsic,
        sensor_to_ego=np.eye(4),
        ego_to_global=pose,
        image_size=(32, 24),
    )


def test_temporal_lab_difference_uses_aligned_overlap() -> None:
    camera = _downward_camera()
    previous = np.full((24, 32, 3), 120, dtype=np.uint8)
    current = previous.copy()
    cv2.rectangle(current, (8, 8), (16, 16), (50, 50, 50), -1)
    cfg = ShadowConfig(
        exposure_compensation=False,
        min_temporal_l_diff=2.0,
        temporal_diff_percentile=50.0,
        max_temporal_chroma_diff=20.0,
        temporal_dilate_kernel=3,
    )
    result = temporal_lab_difference(current, previous, camera, camera, 0.0, cfg)
    assert result.overlap > 0
    assert result.mask.shape == (24, 32)
    assert np.count_nonzero(result.mask) > 0


def test_ground_projection_round_trip_is_small() -> None:
    camera = _downward_camera()
    points = np.asarray([[-0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    result = ground_reprojection_error(points, np.eye(4), camera, 0.0)
    assert result["count"] == 3
    assert result["p95_px"] < 1e-4
