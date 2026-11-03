from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shadow_occ.config import OccConfig
from shadow_occ.geometry.bev import BEVGrid


@dataclass
class Occ3DSample:
    path: Path
    semantics: np.ndarray
    mask_camera: np.ndarray | None = None
    mask_lidar: np.ndarray | None = None

    def occupied_mask(self, cfg: OccConfig) -> np.ndarray:
        sem = self.semantics
        return (sem != cfg.free_label) & (sem != cfg.ignore_label)

    def class_mask(self, class_ids: list[int]) -> np.ndarray:
        if not class_ids:
            return np.zeros_like(self.semantics, dtype=bool)
        return np.isin(self.semantics, np.asarray(class_ids, dtype=self.semantics.dtype))


class Occ3DLoader:
    def __init__(self, root: str | Path, cfg: OccConfig) -> None:
        self.root = Path(root)
        self.cfg = cfg

    def find_label_path(self, sample_token: str, scene_name: str | None = None) -> Path:
        candidates: list[Path] = []
        if scene_name:
            candidates.extend(
                [
                    self.root / "gts" / scene_name / sample_token / "labels.npz",
                    self.root / scene_name / sample_token / "labels.npz",
                    self.root / "labels" / scene_name / f"{sample_token}.npz",
                ]
            )
        candidates.extend(
            [
                self.root / sample_token / "labels.npz",
                self.root / f"{sample_token}.npz",
                self.root / "labels" / f"{sample_token}.npz",
            ]
        )
        for path in candidates:
            if path.exists():
                return path
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find Occ3D label for sample {sample_token}. Tried:\n{tried}")

    def load(self, sample_token: str, scene_name: str | None = None) -> Occ3DSample:
        path = self.find_label_path(sample_token, scene_name)
        data = np.load(path)
        semantics = _first_existing(data, ["semantics", "semantic", "labels", "voxel_label"])
        mask_camera = _maybe_existing(data, ["mask_camera", "camera_mask", "valid_camera_mask"])
        mask_lidar = _maybe_existing(data, ["mask_lidar", "lidar_mask", "valid_lidar_mask"])
        return Occ3DSample(
            path=path,
            semantics=np.asarray(semantics),
            mask_camera=None if mask_camera is None else np.asarray(mask_camera).astype(bool),
            mask_lidar=None if mask_lidar is None else np.asarray(mask_lidar).astype(bool),
        )

    def voxel_centers(self, shape: tuple[int, int, int]) -> np.ndarray:
        z_axis = self.cfg.z_axis
        if len(shape) != 3:
            raise ValueError(f"Expected 3D occupancy labels, got shape={shape}")
        axes = [np.arange(s, dtype=np.float32) for s in shape]
        grids = np.meshgrid(*axes, indexing="ij")

        coords_by_axis = [None, None, None]
        horizontal_axes = [axis for axis in range(3) if axis != z_axis]
        coords_by_axis[horizontal_axes[0]] = "x"
        coords_by_axis[horizontal_axes[1]] = "y"
        coords_by_axis[z_axis] = "z"

        pc = self.cfg.pc_range
        voxel = self.cfg.voxel_size
        out = {}
        for axis, name in enumerate(coords_by_axis):
            start = {"x": pc[0], "y": pc[1], "z": pc[2]}[name]
            step = {"x": voxel[0], "y": voxel[1], "z": voxel[2]}[name]
            out[name] = start + (grids[axis] + 0.5) * step
        return np.stack([out["x"], out["y"], out["z"]], axis=-1).reshape(-1, 3)

    def height_map(self, sample: Occ3DSample, grid: BEVGrid) -> np.ndarray:
        centers = self.voxel_centers(sample.semantics.shape)
        occupied = sample.occupied_mask(self.cfg).reshape(-1)
        if sample.mask_camera is not None and sample.mask_camera.shape == sample.semantics.shape:
            occupied &= sample.mask_camera.reshape(-1)
        pts = centers[occupied]
        heights = pts[:, 2] if len(pts) else np.zeros((0,), dtype=np.float32)
        return bev_max_height(pts[:, :2] if len(pts) else pts.reshape(0, 3)[:, :2], heights, grid)

    def ground_points(self, sample: Occ3DSample) -> tuple[np.ndarray, np.ndarray]:
        centers = self.voxel_centers(sample.semantics.shape)
        ground = sample.class_mask(self.cfg.ground_class_ids).reshape(-1)
        if not np.any(ground):
            occupied = sample.occupied_mask(self.cfg).reshape(-1)
            z = centers[:, 2]
            ground = occupied & (z < (self.cfg.pc_range[2] + self.cfg.voxel_size[2] * 3.0))
        if sample.mask_camera is not None and sample.mask_camera.shape == sample.semantics.shape:
            ground &= sample.mask_camera.reshape(-1)
        pts = centers[ground]
        labels = sample.semantics.reshape(-1)[ground]
        return pts, labels


def _first_existing(data: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the label keys exist in {data.files}. Expected one of {keys}.")


def _maybe_existing(data: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def bev_max_height(xy: np.ndarray, height: np.ndarray, grid: BEVGrid) -> np.ndarray:
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    if len(xy) == 0:
        return out
    i, j, valid = grid.xy_to_ij(xy)
    for row, col, h in zip(i[valid], j[valid], height[valid]):
        current = out[row, col]
        if np.isnan(current) or h > current:
            out[row, col] = h
    return out
