from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    version: str = "v1.0-trainval"
    dataroot: str = "E:/datasets/nuscenes"
    occ_root: str = "E:/datasets/Occ3D-nuScenes"
    output_root: str = "outputs"
    # Use LiDAR/image geometry for pseudo shadow labels by default. Occ3D is
    # reserved for occupancy supervision and final evaluation.
    geometry_source: str = "nuscenes_lidar"
    shadow_label_root: str | None = None
    image_size: tuple[int, int] = (256, 704)
    num_frames: int = 3
    cams: list[str] = field(
        default_factory=lambda: [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
    )
    scene_names: list[str] = field(default_factory=list)
    train_scene_names: list[str] = field(default_factory=list)
    val_scene_names: list[str] = field(default_factory=list)
    max_samples: int | None = None
    occ_frame: str = "ego"
    use_annotation_caster: bool = False


@dataclass
class OccConfig:
    num_classes: int = 18
    free_label: int = 17
    ignore_label: int = 255
    semantic_eval_class_ids: list[int] = field(default_factory=lambda: list(range(17)))
    z_axis: int = 2
    voxel_size: tuple[float, float, float] = (0.4, 0.4, 0.4)
    pc_range: tuple[float, float, float, float, float, float] = (
        -40.0,
        -40.0,
        -1.0,
        40.0,
        40.0,
        5.4,
    )
    ground_class_ids: list[int] = field(default_factory=lambda: [11, 12, 13, 14, 15])
    object_class_ids: list[int] = field(default_factory=lambda: list(range(1, 11)))


@dataclass
class BevConfig:
    x_min: float = -40.0
    x_max: float = 40.0
    y_min: float = -40.0
    y_max: float = 40.0
    resolution: float = 0.4


@dataclass
class ShadowConfig:
    dark_percentile: float = 28.0
    max_dark_value: int = 130
    min_saturation: int = 0
    morph_kernel: int = 5
    min_component_area: int = 8
    min_length_m: float = 1.2
    min_elongation: float = 1.8
    min_track_age: int = 2
    root_height_threshold_m: float = 0.6
    heatmap_sigma: float = 1.8
    sun_direction_xy: tuple[float, float] = (1.0, -1.0)
    sun_elevation_deg: float = 35.0
    render_height_threshold_m: float = 0.6
    caster_match_distance_m: float = 6.0
    # Temporal image evidence. The previous frame is warped onto the current
    # ground plane before Lab differences are measured.
    use_temporal_difference: bool = True
    temporal_diff_percentile: float = 65.0
    min_temporal_l_diff: float = 6.0
    max_temporal_chroma_diff: float = 18.0
    temporal_dilate_kernel: int = 9
    exposure_compensation: bool = True
    temporal_min_overlap: int = 32
    caster_max_motion_m: float = 0.75
    tip_max_motion_m: float = 3.0
    tip_smoothness_sigma_m: float = 1.5
    ground_plane_z_offset_m: float = 0.0


@dataclass
class LidarConfig:
    ground_margin_m: float = 0.25
    min_points_per_cell: int = 2
    min_height_m: float = 0.15
    max_height_m: float = 5.0


@dataclass
class EvalConfig:
    distance_bins_m: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    )


@dataclass
class ModelConfig:
    bev_channels: int = 128
    image_feature_channels: int = 128
    num_bev_layers: int = 2
    num_heads: int = 4
    shadow_hidden_channels: int = 128
    z_anchors: tuple[float, ...] = (0.4, 1.2, 2.0, 3.2)
    dropout: float = 0.1
    use_shadow: bool = True
    use_shadow_fusion: bool = True
    use_height_prior: bool = True
    use_uncertainty_gate: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 12
    lr: float = 2e-4
    weight_decay: float = 0.01
    log_interval: int = 10
    save_interval: int = 1
    resume: str | None = None
    device: str = "cuda"
    occ_loss_weight: float = 1.0
    semantic_loss_weight: float = 1.0
    shadow_loss_weight: float = 1.0
    temporal_loss_weight: float = 1.0
    motion_smooth_loss_weight: float = 0.2
    caster_geometry_loss_weight: float = 0.2
    render_loss_weight: float = 0.5
    uncertainty_loss_weight: float = 1.0
    seed: int = 0
    deterministic: bool = False


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    occ: OccConfig = field(default_factory=OccConfig)
    bev: BevConfig = field(default_factory=BevConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ProjectConfig:
    raw: dict[str, Any] = {}
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _merge_dict(raw, overrides)

    data_raw = raw.get("data", {})
    if "image_size" in data_raw:
        data_raw["image_size"] = _as_tuple(data_raw["image_size"])
    data = DataConfig(**data_raw)

    occ_raw = raw.get("occ", {})
    for key in ("voxel_size", "pc_range"):
        if key in occ_raw:
            occ_raw[key] = _as_tuple(occ_raw[key])
    occ = OccConfig(**occ_raw)

    bev = BevConfig(**raw.get("bev", {}))
    shadow_raw = raw.get("shadow", {})
    if "sun_direction_xy" in shadow_raw:
        shadow_raw["sun_direction_xy"] = _as_tuple(shadow_raw["sun_direction_xy"])
    shadow = ShadowConfig(**shadow_raw)
    lidar = LidarConfig(**raw.get("lidar", {}))

    eval_raw = raw.get("eval", {})
    if "distance_bins_m" in eval_raw:
        eval_raw["distance_bins_m"] = [tuple(x) for x in eval_raw["distance_bins_m"]]
    eval_cfg = EvalConfig(**eval_raw)
    model_raw = raw.get("model", {})
    if "z_anchors" in model_raw:
        model_raw["z_anchors"] = _as_tuple(model_raw["z_anchors"])
    model = ModelConfig(**model_raw)
    train = TrainConfig(**raw.get("train", {}))

    return ProjectConfig(
        data=data,
        occ=occ,
        bev=bev,
        shadow=shadow,
        lidar=lidar,
        eval=eval_cfg,
        model=model,
        train=train,
    )
