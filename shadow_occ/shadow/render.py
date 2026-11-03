from __future__ import annotations

import math

import numpy as np

from shadow_occ.config import ShadowConfig
from shadow_occ.geometry.bev import BEVGrid


def render_shadow_from_height(
    height_map: np.ndarray,
    grid: BEVGrid,
    cfg: ShadowConfig,
) -> np.ndarray:
    """Approximate BEV shadow prior from an OCC height map.

    This is intentionally simple: every occupied height cell casts a ray on the
    ground plane according to a fixed sun direction and elevation.
    """
    prior = np.zeros(grid.shape, dtype=np.uint8)
    valid = np.isfinite(height_map) & (height_map >= cfg.render_height_threshold_m)
    if not np.any(valid):
        return prior

    direction = np.asarray(cfg.sun_direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm

    elevation = max(1.0, min(89.0, float(cfg.sun_elevation_deg)))
    tan_elev = math.tan(math.radians(elevation))
    rows, cols = np.column_stack(np.where(valid)).T
    for row, col in zip(rows, cols):
        height = float(height_map[row, col])
        length_m = height / max(tan_elev, 1e-6)
        steps = max(1, int(math.ceil(length_m / grid.resolution)))
        xy0 = grid.ij_to_xy(np.asarray([[row, col]], dtype=np.float32))[0]
        for step in range(1, steps + 1):
            xy = xy0 + direction * (step * grid.resolution)
            i, j, inside = grid.xy_to_ij(xy.reshape(1, 2))
            if bool(inside[0]):
                prior[int(i[0]), int(j[0])] = 1
    return prior
