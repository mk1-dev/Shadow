from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from shadow_occ.geometry.projection import CameraCalibration
from shadow_occ.geometry.transforms import transform_matrix
from shadow_occ.geometry.transforms import apply_transform


@dataclass
class CameraFrame:
    name: str
    token: str
    image_path: Path
    calibration: CameraCalibration
    timestamp: int


@dataclass
class SampleRecord:
    token: str
    scene_token: str
    scene_name: str
    timestamp: int
    sample: dict
    lidar_token: str
    lidar_ego_to_global: np.ndarray
    lidar_sensor_to_ego: np.ndarray

    @property
    def ego_to_global(self) -> np.ndarray:
        return self.lidar_ego_to_global


class NuScenesAdapter:
    def __init__(self, version: str, dataroot: str | Path, verbose: bool = True) -> None:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise ImportError(
                "nuscenes-devkit is required. Install with `pip install nuscenes-devkit`."
            ) from exc

        self.dataroot = Path(dataroot)
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)
        self._scene_name_by_token = {scene["token"]: scene["name"] for scene in self.nusc.scene}

    def iter_samples(
        self,
        scene_names: list[str] | None = None,
        max_samples: int | None = None,
    ) -> Iterator[SampleRecord]:
        scene_filter = set(scene_names or [])
        count = 0
        for sample in self.nusc.sample:
            scene_name = self._scene_name_by_token[sample["scene_token"]]
            if scene_filter and scene_name not in scene_filter:
                continue
            yield self.make_sample_record(sample)
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    def scene_names_for_split(self, split: str) -> list[str]:
        """Return the official nuScenes scene names for a benchmark split."""
        try:
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as exc:
            raise ImportError(
                "nuscenes-devkit is required to resolve official train/val splits."
            ) from exc

        splits = create_splits_scenes()
        if split not in splits:
            raise ValueError(f"Unknown nuScenes split {split!r}; available: {sorted(splits)}")
        available = set(self._scene_name_by_token.values())
        return [name for name in splits[split] if name in available]

    def make_sample_record(self, sample: dict) -> SampleRecord:
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = self.nusc.get("sample_data", lidar_token)
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        return SampleRecord(
            token=sample["token"],
            scene_token=sample["scene_token"],
            scene_name=self._scene_name_by_token[sample["scene_token"]],
            timestamp=sample["timestamp"],
            sample=sample,
            lidar_token=lidar_token,
            lidar_ego_to_global=transform_matrix(lidar_pose["translation"], lidar_pose["rotation"]),
            lidar_sensor_to_ego=transform_matrix(lidar_cs["translation"], lidar_cs["rotation"]),
        )

    def get_camera(self, record: SampleRecord, camera_name: str) -> CameraFrame:
        token = record.sample["data"][camera_name]
        sample_data = self.nusc.get("sample_data", token)
        calibrated = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

        image_path = self.dataroot / sample_data["filename"]
        intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=np.float64)
        width = int(sample_data["width"])
        height = int(sample_data["height"])
        calibration = CameraCalibration(
            intrinsic=intrinsic,
            sensor_to_ego=transform_matrix(calibrated["translation"], calibrated["rotation"]),
            ego_to_global=transform_matrix(ego_pose["translation"], ego_pose["rotation"]),
            image_size=(width, height),
        )
        return CameraFrame(
            name=camera_name,
            token=token,
            image_path=image_path,
            calibration=calibration,
            timestamp=sample_data["timestamp"],
        )

    def load_lidar_points_ego(self, record: SampleRecord) -> np.ndarray:
        """Load LIDAR_TOP points and transform them into the sample ego frame."""
        try:
            from nuscenes.utils.data_classes import LidarPointCloud
        except ImportError as exc:
            raise ImportError(
                "nuscenes-devkit is required to load nuScenes LiDAR point clouds."
            ) from exc

        sample_data = self.nusc.get("sample_data", record.lidar_token)
        lidar_path = self.dataroot / sample_data["filename"]
        cloud = LidarPointCloud.from_file(str(lidar_path))
        points = cloud.points[:3].T.astype(np.float64)
        return apply_transform(points, record.lidar_sensor_to_ego)

    def annotation_centers_ego(self, record: SampleRecord) -> np.ndarray:
        """Return annotated object centers in the sample ego frame.

        These nuScenes annotations are used only to associate a candidate
        shadow root with a possible caster. They are not used to construct
        occupancy targets or height targets.
        """
        centers = []
        global_to_ego = np.linalg.inv(record.ego_to_global)
        for annotation_token in record.sample.get("anns", []):
            annotation = self.nusc.get("sample_annotation", annotation_token)
            category = str(annotation.get("category_name", ""))
            if not (category.startswith("vehicle.") or category.startswith("human.") or category.startswith("movable_object.")):
                continue
            center = np.asarray(annotation["translation"], dtype=np.float64).reshape(1, 3)
            centers.append(apply_transform(center, global_to_ego)[0])
        if not centers:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(centers, dtype=np.float32)

    def occ_to_global(self, record: SampleRecord, occ_frame: str = "ego") -> np.ndarray:
        if occ_frame == "ego":
            return record.lidar_ego_to_global
        if occ_frame == "lidar":
            return record.lidar_ego_to_global @ record.lidar_sensor_to_ego
        raise ValueError(f"Unsupported occ_frame={occ_frame!r}. Use 'ego' or 'lidar'.")
