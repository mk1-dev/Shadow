from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from shadow_occ.config import ShadowConfig
from shadow_occ.geometry.bev import BEVGrid


@dataclass
class ShadowComponent:
    component_id: int
    area: int
    centroid_xy: tuple[float, float]
    root_xy: tuple[float, float]
    tip_xy: tuple[float, float]
    direction_xy: tuple[float, float]
    length_m: float
    elongation: float
    mean_height_near_root: float
    reliability: float = 0.0
    track_id: int | None = None
    track_age: int = 1
    stable: bool = False
    caster_stability: float = 0.0
    tip_motion_m: float = 0.0
    tip_smoothness: float = 0.0
    temporal_light_score: float = 0.0
    caster_xy: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShadowGeometryLabels:
    shadow_mask: np.ndarray
    root_heatmap: np.ndarray
    tip_heatmap: np.ndarray
    caster_heatmap: np.ndarray
    direction: np.ndarray
    length_map: np.ndarray
    height_target: np.ndarray
    reliability: np.ndarray
    components: list[ShadowComponent]
    component_id_map: np.ndarray | None = None

    def as_npz_dict(self) -> dict:
        return {
            "shadow_mask": self.shadow_mask.astype(np.uint8),
            "shadow_root_heatmap": self.root_heatmap.astype(np.float32),
            "shadow_tip_heatmap": self.tip_heatmap.astype(np.float32),
            "shadow_caster_heatmap": self.caster_heatmap.astype(np.float32),
            "shadow_direction": self.direction.astype(np.float32),
            "V_shadow": self.direction.astype(np.float32),
            "shadow_length": self.length_map.astype(np.float32),
            "height_target": self.height_target.astype(np.float32),
            "H_occ": self.height_target.astype(np.float32),
            # Backward-compatible alias for labels generated before the naming cleanup.
            "height_prior": self.height_target.astype(np.float32),
            "shadow_reliability": self.reliability.astype(np.float32),
            "U_shadow": self.reliability.astype(np.float32),
        }


def build_shadow_geometry_labels(
    shadow_mask: np.ndarray,
    height_map: np.ndarray,
    grid: BEVGrid,
    cfg: ShadowConfig,
    projection_confidence: np.ndarray | None = None,
    caster_points: np.ndarray | None = None,
    temporal_confidence: np.ndarray | None = None,
) -> ShadowGeometryLabels:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        shadow_mask.astype(np.uint8), connectivity=8
    )

    root_heatmap = np.zeros(grid.shape, dtype=np.float32)
    tip_heatmap = np.zeros(grid.shape, dtype=np.float32)
    caster_heatmap = np.zeros(grid.shape, dtype=np.float32)
    direction = np.zeros((2, *grid.shape), dtype=np.float32)
    length_map = np.zeros(grid.shape, dtype=np.float32)
    height_target = np.nan_to_num(height_map, nan=0.0).astype(np.float32)
    reliability = np.zeros(grid.shape, dtype=np.float32)
    components: list[ShadowComponent] = []

    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < cfg.min_component_area:
            continue

        pixels = np.column_stack(np.where(labels == component_id))
        component = _component_geometry(component_id, pixels, height_map, grid, cfg)
        if component is None:
            continue

        if component.length_m < cfg.min_length_m or component.elongation < cfg.min_elongation:
            continue

        component_pixels = labels == component_id
        component.reliability = _component_reliability(
            component,
            component_pixels,
            projection_confidence,
            cfg,
            temporal_confidence,
        )
        if temporal_confidence is not None and temporal_confidence.shape == component_pixels.shape:
            component.temporal_light_score = float(np.mean(temporal_confidence[component_pixels]))
        components.append(component)
        _draw_gaussian(root_heatmap, _xy_to_pixel(component.root_xy, grid), cfg.heatmap_sigma)
        _draw_gaussian(tip_heatmap, _xy_to_pixel(component.tip_xy, grid), cfg.heatmap_sigma)
        caster_xy = _match_caster(component.root_xy, caster_points, cfg.caster_match_distance_m)
        component.caster_xy = caster_xy
        _draw_gaussian(caster_heatmap, _xy_to_pixel(caster_xy, grid), cfg.heatmap_sigma)

        direction[0, component_pixels] = component.direction_xy[0]
        direction[1, component_pixels] = component.direction_xy[1]
        length_map[component_pixels] = component.length_m
        reliability[component_pixels] = component.reliability

    kept = np.zeros_like(shadow_mask, dtype=np.uint8)
    for component in components:
        pixel = _xy_to_pixel(component.centroid_xy, grid)
        label_id = labels[pixel[0], pixel[1]]
        if label_id > 0:
            kept[labels == label_id] = 1

    return ShadowGeometryLabels(
        shadow_mask=kept,
        root_heatmap=root_heatmap,
        tip_heatmap=tip_heatmap,
        caster_heatmap=caster_heatmap,
        direction=direction,
        length_map=length_map,
        height_target=height_target,
        reliability=reliability,
        components=components,
        component_id_map=labels,
    )


def _match_caster(
    root_xy: tuple[float, float], caster_points: np.ndarray | None, max_distance_m: float
) -> tuple[float, float]:
    if caster_points is None:
        return root_xy
    points = np.asarray(caster_points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return root_xy
    root = np.asarray(root_xy, dtype=np.float32)
    distances = np.linalg.norm(points[:, :2] - root[None, :], axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) > max_distance_m:
        return root_xy
    return (float(points[index, 0]), float(points[index, 1]))


def apply_temporal_reliability(labels: ShadowGeometryLabels, min_track_age: int) -> ShadowGeometryLabels:
    """Boost reliable long-lived tracks and suppress one-frame shadow candidates."""
    if labels.component_id_map is None:
        return labels

    updated = np.zeros_like(labels.reliability, dtype=np.float32)
    for component in labels.components:
        age_score = min(1.0, float(component.track_age) / max(1.0, float(min_track_age)))
        stable_bonus = 1.0 if component.stable else 0.65
        geometry_bonus = 0.5 + 0.5 * min(
            1.0,
            max(0.0, component.caster_stability)
            * max(0.0, component.tip_smoothness),
        )
        component.reliability = float(
            np.clip(
                component.reliability
                * (0.5 + 0.5 * age_score)
                * stable_bonus
                * geometry_bonus,
                0.0,
                1.0,
            )
        )
        updated[labels.component_id_map == component.component_id] = component.reliability
    labels.reliability = updated
    return labels


def _component_geometry(
    component_id: int,
    pixels: np.ndarray,
    height_map: np.ndarray,
    grid: BEVGrid,
    cfg: ShadowConfig,
) -> ShadowComponent | None:
    if len(pixels) < 2:
        return None
    xy = grid.ij_to_xy(pixels)
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    cov = centered.T @ centered / max(1, len(centered) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor_val = float(max(eigvals[order[1]], 1e-6))
    major_val = float(max(eigvals[order[0]], 1e-6))
    elongation = float(np.sqrt(major_val / minor_val))

    projection = centered @ major
    a = xy[int(np.argmin(projection))]
    b = xy[int(np.argmax(projection))]
    length = float(np.linalg.norm(b - a))
    if length <= 1e-6:
        return None

    root, tip, root_height = _choose_root_tip(a, b, height_map, grid, cfg)
    direction = (tip - root) / max(np.linalg.norm(tip - root), 1e-6)

    return ShadowComponent(
        component_id=component_id,
        area=int(len(pixels)),
        centroid_xy=(float(centroid[0]), float(centroid[1])),
        root_xy=(float(root[0]), float(root[1])),
        tip_xy=(float(tip[0]), float(tip[1])),
        direction_xy=(float(direction[0]), float(direction[1])),
        length_m=length,
        elongation=elongation,
        mean_height_near_root=float(root_height),
    )


def _component_reliability(
    component: ShadowComponent,
    component_pixels: np.ndarray,
    projection_confidence: np.ndarray | None,
    cfg: ShadowConfig,
    temporal_confidence: np.ndarray | None = None,
) -> float:
    length_score = min(1.0, component.length_m / max(cfg.min_length_m * 3.0, 1e-6))
    elongation_score = min(1.0, component.elongation / max(cfg.min_elongation * 1.8, 1e-6))
    area_score = min(1.0, component.area / max(cfg.min_component_area * 6.0, 1.0))
    height_score = min(
        1.0,
        component.mean_height_near_root / max(cfg.root_height_threshold_m * 2.0, 1e-6),
    )
    projection_score = 1.0
    if projection_confidence is not None and projection_confidence.shape == component_pixels.shape:
        values = projection_confidence[component_pixels]
        if len(values):
            projection_score = float(np.clip(np.mean(values), 0.0, 1.0))
    temporal_score = 1.0
    if temporal_confidence is not None and temporal_confidence.shape == component_pixels.shape:
        temporal_score = float(np.clip(np.mean(temporal_confidence[component_pixels]), 0.0, 1.0))
    score = (
        0.25 * length_score
        + 0.20 * elongation_score
        + 0.12 * area_score
        + 0.18 * height_score
        + 0.10 * projection_score
        + 0.15 * temporal_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _choose_root_tip(
    a: np.ndarray,
    b: np.ndarray,
    height_map: np.ndarray,
    grid: BEVGrid,
    cfg: ShadowConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    ha = _nearby_height(a, height_map, grid, cfg)
    hb = _nearby_height(b, height_map, grid, cfg)
    if ha >= hb:
        return a, b, ha
    return b, a, hb


def _nearby_height(xy: np.ndarray, height_map: np.ndarray, grid: BEVGrid, cfg: ShadowConfig) -> float:
    i, j, valid = grid.xy_to_ij(xy.reshape(1, 2))
    if not bool(valid[0]):
        return 0.0
    radius = max(1, int(round(1.2 / grid.resolution)))
    r0 = max(0, int(i[0]) - radius)
    r1 = min(grid.height, int(i[0]) + radius + 1)
    c0 = max(0, int(j[0]) - radius)
    c1 = min(grid.width, int(j[0]) + radius + 1)
    patch = height_map[r0:r1, c0:c1]
    valid_heights = patch[np.isfinite(patch)]
    valid_heights = valid_heights[valid_heights >= cfg.root_height_threshold_m]
    if len(valid_heights) == 0:
        return 0.0
    return float(np.mean(valid_heights))


def _draw_gaussian(canvas: np.ndarray, pixel: tuple[int, int], sigma: float) -> None:
    row, col = pixel
    radius = max(1, int(round(sigma * 3.0)))
    r0 = max(0, row - radius)
    r1 = min(canvas.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(canvas.shape[1], col + radius + 1)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    gaussian = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2.0 * sigma**2))
    canvas[r0:r1, c0:c1] = np.maximum(canvas[r0:r1, c0:c1], gaussian)


def _xy_to_pixel(xy: tuple[float, float], grid: BEVGrid) -> tuple[int, int]:
    i, j, valid = grid.xy_to_ij(np.asarray(xy, dtype=np.float32).reshape(1, 2))
    if not bool(valid[0]):
        return int(np.clip(i[0], 0, grid.height - 1)), int(np.clip(j[0], 0, grid.width - 1))
    return int(i[0]), int(j[0])
