from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shadow_occ.config import LidarConfig
from shadow_occ.geometry.bev import BEVGrid


@dataclass
class LidarPseudoGeometry:
    path: Path
    ground_points: np.ndarray
    height_map: np.ndarray
    point_count_map: np.ndarray


def build_lidar_pseudo_geometry(
    points_ego: np.ndarray,
    grid: BEVGrid,
    cfg: LidarConfig,
    sample_token: str,
) -> LidarPseudoGeometry:
    """Build ground samples and a coarse height map from nuScenes LIDAR_TOP only.

    The height map is cell-local height above the lowest observed point, not
    semantic occupancy ground truth. It is useful for weak shadow geometry,
    not for Occ3D benchmark reporting.
    """
    points = np.asarray(points_ego, dtype=np.float32).reshape(-1, 3)
    height_map = np.full(grid.shape, np.nan, dtype=np.float32)
    min_z = np.full(grid.shape, np.inf, dtype=np.float32)
    max_z = np.full(grid.shape, -np.inf, dtype=np.float32)
    point_count = np.zeros(grid.shape, dtype=np.int32)

    if len(points) == 0:
        return LidarPseudoGeometry(Path(f"nuscenes_lidar/{sample_token}"), points, height_map, point_count)

    ij_i, ij_j, valid = grid.xy_to_ij(points[:, :2])
    for row, col, z in zip(ij_i[valid], ij_j[valid], points[:, 2][valid]):
        point_count[row, col] += 1
        min_z[row, col] = min(min_z[row, col], z)
        max_z[row, col] = max(max_z[row, col], z)

    enough = point_count >= cfg.min_points_per_cell
    local_height = max_z - min_z
    local_height = np.clip(local_height, cfg.min_height_m, cfg.max_height_m)
    height_map[enough] = local_height[enough]

    ground_points = _extract_ground_points(points, ij_i, ij_j, valid, min_z, point_count, cfg)
    return LidarPseudoGeometry(
        path=Path(f"nuscenes_lidar/{sample_token}"),
        ground_points=ground_points,
        height_map=height_map,
        point_count_map=point_count,
    )


def _extract_ground_points(
    points: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    valid: np.ndarray,
    min_z: np.ndarray,
    point_count: np.ndarray,
    cfg: LidarConfig,
) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3)
    selected = []
    valid_indices = np.where(valid)[0]
    for idx in valid_indices:
        row = rows[idx]
        col = cols[idx]
        if point_count[row, col] < cfg.min_points_per_cell:
            continue
        if points[idx, 2] <= min_z[row, col] + cfg.ground_margin_m:
            selected.append(idx)
    if not selected:
        return np.zeros((0, 3), dtype=np.float32)
    return points[np.asarray(selected, dtype=np.int64)]
