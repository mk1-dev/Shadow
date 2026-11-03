from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use shadow_occ.data.bevformer_dataset.") from exc

from shadow_occ.config import ProjectConfig
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.data.occ3d_loader import Occ3DLoader
from shadow_occ.geometry.bev import BEVGrid


class NuScenesOccSequenceDataset(Dataset):
    """Multi-frame, multi-camera nuScenes + Occ3D dataset for the local BEVFormer."""

    def __init__(self, cfg: ProjectConfig, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split
        self.grid = BEVGrid.from_config(cfg.bev)
        self.nusc = NuScenesAdapter(cfg.data.version, cfg.data.dataroot, verbose=True)
        self.occ_loader = Occ3DLoader(cfg.data.occ_root, cfg.occ)
        self.num_frames = int(cfg.data.num_frames)
        if self.num_frames < 1:
            raise ValueError("data.num_frames must be >= 1")
        scene_names = self._scene_names_for_split(split)
        raw_records = list(
            self.nusc.iter_samples(
                scene_names=scene_names,
                max_samples=None,
            )
        )
        self.records = self._filter_records_with_occ_labels(raw_records)
        if cfg.data.max_samples is not None:
            self.records = self.records[: cfg.data.max_samples]
        if not self.records:
            scene_hint = ", ".join(cfg.data.scene_names) if cfg.data.scene_names else "all configured scenes"
            raise FileNotFoundError(
                "No Occ3D labels were found for the requested nuScenes samples.\n"
                f"occ_root: {cfg.data.occ_root}\n"
                f"scenes: {scene_hint}\n"
                "Check that occ_root points to the real Occ3D-nuScenes directory and that its layout "
                "contains labels like gts/{scene_name}/{sample_token}/labels.npz."
            )
        self.shadow_label_root = _shadow_label_root(cfg)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_record = self.records[index]
        frame_records = self._collect_frame_records(target_record)

        images = []
        ego_to_img = []
        ego_to_global = []
        for record in frame_records:
            frame_images = []
            frame_ego_to_img = []
            for cam_name in self.cfg.data.cams:
                camera = self.nusc.get_camera(record, cam_name)
                image, intrinsic = _load_and_resize_image(camera.image_path, camera.calibration.intrinsic, self.cfg.data.image_size)
                ego_to_cam = camera.calibration.global_to_sensor @ record.ego_to_global
                projection = intrinsic @ ego_to_cam[:3, :]
                frame_images.append(image)
                frame_ego_to_img.append(projection.astype(np.float32))
            images.append(np.stack(frame_images, axis=0))
            ego_to_img.append(np.stack(frame_ego_to_img, axis=0))
            ego_to_global.append(record.ego_to_global.astype(np.float32))

        occ = self.occ_loader.load(target_record.token, target_record.scene_name)
        semantic_target = occ_to_semantic_target(
            occ.semantics,
            self.occ_loader,
            self.grid,
            self.cfg.occ.free_label,
            self.cfg.occ.ignore_label,
        )
        occ_target = semantic_to_binary_target(
            semantic_target,
            self.cfg.occ.free_label,
            self.cfg.occ.ignore_label,
        )
        height_target = np.nan_to_num(self.occ_loader.height_map(occ, self.grid), nan=0.0).astype(np.float32)
        shadow = self._load_shadow_labels(target_record, height_target)

        sample = {
            "images": torch.from_numpy(np.stack(images, axis=0)).float(),
            "ego_to_img": torch.from_numpy(np.stack(ego_to_img, axis=0)).float(),
            "ego_to_global": torch.from_numpy(np.stack(ego_to_global, axis=0)).float(),
            "occ_target": torch.from_numpy(occ_target).float(),
            "semantic_target": torch.from_numpy(semantic_target).long(),
            "sample_token": target_record.token,
            "scene_name": target_record.scene_name,
        }
        sample.update({key: torch.from_numpy(value).float() for key, value in shadow.items()})
        return sample

    def _collect_frame_records(self, target_record) -> list:
        samples = []
        sample = target_record.sample
        for _ in range(self.num_frames):
            samples.append(sample)
            prev_token = sample.get("prev", "")
            if not prev_token:
                break
            sample = self.nusc.nusc.get("sample", prev_token)
        while len(samples) < self.num_frames:
            samples.append(samples[-1])
        samples = list(reversed(samples))
        return [self.nusc.make_sample_record(sample) for sample in samples]

    def _filter_records_with_occ_labels(self, records: list) -> list:
        valid = []
        missing_examples = []
        for record in records:
            try:
                self.occ_loader.find_label_path(record.token, record.scene_name)
            except FileNotFoundError:
                if len(missing_examples) < 5:
                    missing_examples.append(f"{record.scene_name}/{record.token}")
                continue
            valid.append(record)

        missing = len(records) - len(valid)
        if missing:
            print(
                f"[Occ3D] skipped {missing} nuScenes samples without labels; "
                f"using {len(valid)} labeled samples."
            )
            if missing_examples:
                print("[Occ3D] missing examples: " + ", ".join(missing_examples))
        return valid

    def _load_shadow_labels(self, record, height_target: np.ndarray) -> dict[str, np.ndarray]:
        label_path = self.shadow_label_root / record.scene_name / f"{record.token}.npz"
        if label_path.exists():
            data = np.load(label_path, allow_pickle=True)
            h_occ = _first_label_array(
                data,
                ["height_target", "H_occ", "height_prior"],
                self.grid.shape,
                default=height_target,
            )
            return {
                "shadow_mask": _label_array(data, "shadow_mask", self.grid.shape)[None],
                "shadow_root_heatmap": _label_array(data, "shadow_root_heatmap", self.grid.shape)[None],
                "shadow_tip_heatmap": _label_array(data, "shadow_tip_heatmap", self.grid.shape)[None],
                "shadow_caster_heatmap": _label_array(data, "shadow_caster_heatmap", self.grid.shape)[None],
                "shadow_direction": _direction_array(data, self.grid.shape),
                # Always take H_occ from the independently loaded Occ3D target.
                # A pseudo-label file may contain a LiDAR height map, but that
                # must not replace the benchmark height supervision.
                "height_target": height_target[None],
                "H_occ": height_target[None],
                "shadow_reliability": _label_array(data, "shadow_reliability", self.grid.shape)[None],
                "rendered_shadow_prior": _label_array(data, "rendered_shadow_prior", self.grid.shape)[None],
            }
        zeros = np.zeros(self.grid.shape, dtype=np.float32)
        return {
            "shadow_mask": zeros[None],
            "shadow_root_heatmap": zeros[None],
            "shadow_tip_heatmap": zeros[None],
            "shadow_caster_heatmap": zeros[None],
            "shadow_direction": np.zeros((2, *self.grid.shape), dtype=np.float32),
            "height_target": height_target[None],
            "H_occ": height_target[None],
            "shadow_reliability": zeros[None],
            "rendered_shadow_prior": zeros[None],
        }

    def _scene_names_for_split(self, split: str) -> list[str]:
        if self.cfg.data.scene_names:
            return list(self.cfg.data.scene_names)
        if split == "train" and self.cfg.data.train_scene_names:
            return list(self.cfg.data.train_scene_names)
        if split in {"val", "validation"} and self.cfg.data.val_scene_names:
            return list(self.cfg.data.val_scene_names)
        official_split = "val" if split in {"val", "validation"} else split
        return self.nusc.scene_names_for_split(official_split)


def occ_to_binary_target(
    semantics: np.ndarray,
    loader: Occ3DLoader,
    grid: BEVGrid,
    free_label: int,
    ignore_label: int,
) -> np.ndarray:
    semantic = occ_to_semantic_target(semantics, loader, grid, free_label, ignore_label)
    return semantic_to_binary_target(semantic, free_label, ignore_label)


def semantic_to_binary_target(
    semantic_target: np.ndarray,
    free_label: int,
    ignore_label: int,
) -> np.ndarray:
    valid = semantic_target != ignore_label
    occupied = valid & (semantic_target != free_label)
    return occupied.astype(np.float32)


def occ_to_semantic_target(
    semantics: np.ndarray,
    loader: Occ3DLoader,
    grid: BEVGrid,
    free_label: int,
    ignore_label: int,
) -> np.ndarray:
    centers = loader.voxel_centers(semantics.shape)
    labels = semantics.reshape(-1)
    z_min = loader.cfg.pc_range[2]
    z_max = loader.cfg.pc_range[5]
    dz = loader.cfg.voxel_size[2]
    num_z = int(round((z_max - z_min) / dz))
    target = np.full((num_z, grid.height, grid.width), ignore_label, dtype=np.int64)
    valid_label = labels != ignore_label
    if not np.any(valid_label):
        return target
    pts = centers[valid_label]
    cls = labels[valid_label].astype(np.int64)
    z_idx = np.floor((pts[:, 2] - z_min) / dz).astype(np.int64)
    i, j, valid_xy = grid.xy_to_ij(pts[:, :2])
    valid = valid_xy & (z_idx >= 0) & (z_idx < num_z)
    target[z_idx[valid], i[valid], j[valid]] = cls[valid]
    return target


def _load_and_resize_image(
    image_path: Path,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    orig_h, orig_w = image.shape[:2]
    out_h, out_w = image_size
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)

    k = intrinsic.astype(np.float32).copy()
    k[0, :] *= float(out_w) / float(orig_w)
    k[1, :] *= float(out_h) / float(orig_h)
    return image, k


def _shadow_label_root(cfg: ProjectConfig) -> Path:
    if cfg.data.shadow_label_root:
        return Path(cfg.data.shadow_label_root)
    return Path(cfg.data.output_root) / "shadow_labels"


def _label_array(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray:
    if key not in data:
        return np.zeros(shape, dtype=np.float32)
    value = np.asarray(data[key], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"Label {key} has shape {value.shape}, expected {shape}.")
    return value


def _first_label_array(
    data: np.lib.npyio.NpzFile,
    keys: list[str],
    shape: tuple[int, int],
    default: np.ndarray,
) -> np.ndarray:
    for key in keys:
        if key in data:
            return _label_array(data, key, shape)
    return default.astype(np.float32)


def _direction_array(data: np.lib.npyio.NpzFile, shape: tuple[int, int]) -> np.ndarray:
    if "shadow_direction" not in data:
        return np.zeros((2, *shape), dtype=np.float32)
    value = np.asarray(data["shadow_direction"], dtype=np.float32)
    if value.shape != (2, *shape):
        raise ValueError(f"shadow_direction has shape {value.shape}, expected {(2, *shape)}.")
    return value
