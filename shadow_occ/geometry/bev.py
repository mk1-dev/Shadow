from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shadow_occ.config import BevConfig


@dataclass(frozen=True)
class BEVGrid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    @classmethod
    def from_config(cls, cfg: BevConfig) -> "BEVGrid":
        return cls(cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, cfg.resolution)

    @property
    def width(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def xy_to_ij(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        j = np.floor((xy[:, 0] - self.x_min) / self.resolution).astype(np.int64)
        i = np.floor((xy[:, 1] - self.y_min) / self.resolution).astype(np.int64)
        valid = (i >= 0) & (i < self.height) & (j >= 0) & (j < self.width)
        return i, j, valid

    def ij_to_xy(self, ij: np.ndarray) -> np.ndarray:
        ij = np.asarray(ij, dtype=np.float64).reshape(-1, 2)
        y = self.y_min + (ij[:, 0] + 0.5) * self.resolution
        x = self.x_min + (ij[:, 1] + 0.5) * self.resolution
        return np.stack([x, y], axis=1)

    def distance_map(self) -> np.ndarray:
        rows, cols = np.indices(self.shape)
        xy = self.ij_to_xy(np.stack([rows.ravel(), cols.ravel()], axis=1))
        return np.linalg.norm(xy, axis=1).reshape(self.shape)


def rasterize_points(xy: np.ndarray, grid: BEVGrid, values: np.ndarray | None = None) -> np.ndarray:
    canvas = np.zeros(grid.shape, dtype=np.float32 if values is not None else np.uint8)
    i, j, valid = grid.xy_to_ij(xy)
    if values is None:
        canvas[i[valid], j[valid]] = 1
    else:
        canvas[i[valid], j[valid]] = np.maximum(canvas[i[valid], j[valid]], values[valid])
    return canvas
