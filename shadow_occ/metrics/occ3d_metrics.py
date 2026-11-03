from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shadow_occ.geometry.bev import BEVGrid


@dataclass
class Occ3DMetrics:
    num_classes: int
    free_label: int
    ignore_label: int
    semantic_eval_class_ids: list[int]
    grid: BEVGrid
    z_min: float
    z_size: float
    distance_bins_m: list[tuple[float, float]]
    ray_angle_bins: int = 720
    ray_distance_resolution: float = 0.5
    semantic_confusion: np.ndarray = field(init=False)
    occupancy_confusion: np.ndarray = field(init=False)
    occupancy_distance_confusion: dict[tuple[float, float], np.ndarray] = field(init=False)
    ray_stats: dict[tuple[float, float], dict[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.semantic_confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.occupancy_confusion = np.zeros((2, 2), dtype=np.int64)
        self.occupancy_distance_confusion = {
            tuple(bin_): np.zeros((2, 2), dtype=np.int64) for bin_ in self.distance_bins_m
        }
        self.ray_stats = {
            tuple(bin_): {"intersection": 0, "union": 0} for bin_ in self.distance_bins_m
        }

    def update(self, pred_semantic: np.ndarray, target_semantic: np.ndarray) -> None:
        pred = np.asarray(pred_semantic, dtype=np.int64)
        target = np.asarray(target_semantic, dtype=np.int64)
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

        valid = target != self.ignore_label
        pred = np.where((pred >= 0) & (pred < self.num_classes), pred, self.free_label)
        self._update_semantic(pred, target, valid)
        self._update_occupancy(pred, target, valid)
        self._update_ray_iou(pred, target, valid)

    def summarize(self) -> dict:
        occ = _iou_from_confusion(self.occupancy_confusion)
        sem_ious = _iou_from_confusion(self.semantic_confusion)
        eval_ids = [idx for idx in self.semantic_eval_class_ids if 0 <= idx < len(sem_ious)]
        sem_values = sem_ious[eval_ids] if eval_ids else np.asarray([], dtype=np.float64)

        ray_rows = []
        for start, end in self.distance_bins_m:
            stats = self.ray_stats[(start, end)]
            union = stats["union"]
            ray_rows.append(
                {
                    "range_m": [float(start), float(end)],
                    "intersection": int(stats["intersection"]),
                    "union": int(union),
                    "RayIoU": float(stats["intersection"] / union) if union else float("nan"),
                }
            )

        return {
            "Occupancy_mIoU": _safe_nanmean(occ),
            "Occupied_IoU": float(occ[1]) if len(occ) > 1 else float("nan"),
            "Free_IoU": float(occ[0]) if len(occ) > 0 else float("nan"),
            "Occupancy_mIoU_by_distance": [
                {
                    "range_m": [float(start), float(end)],
                    "Occupancy_mIoU": _safe_nanmean(
                        _iou_from_confusion(self.occupancy_distance_confusion[(start, end)])
                    ),
                    "Occupied_IoU": float(
                        _iou_from_confusion(self.occupancy_distance_confusion[(start, end)])[1]
                    ),
                }
                for start, end in self.distance_bins_m
            ],
            "Semantic_mIoU": _safe_nanmean(sem_values),
            "Semantic_IoU_by_class": {
                str(class_id): float(sem_ious[class_id]) for class_id in eval_ids
            },
            "RayIoU_by_distance": ray_rows,
            "RayIoU": _safe_nanmean(np.asarray([row["RayIoU"] for row in ray_rows])),
        }

    def _update_semantic(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        pred_flat = pred[valid].reshape(-1)
        target_flat = target[valid].reshape(-1)
        in_range = (target_flat >= 0) & (target_flat < self.num_classes)
        pred_flat = pred_flat[in_range]
        target_flat = target_flat[in_range]
        index = target_flat * self.num_classes + pred_flat
        counts = np.bincount(index, minlength=self.num_classes * self.num_classes)
        self.semantic_confusion += counts.reshape(self.num_classes, self.num_classes)

    def _update_occupancy(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        pred_occ = (pred != self.free_label).astype(np.int64)
        target_occ = (target != self.free_label).astype(np.int64)
        pred_flat = pred_occ[valid].reshape(-1)
        target_flat = target_occ[valid].reshape(-1)
        index = target_flat * 2 + pred_flat
        counts = np.bincount(index, minlength=4)
        self.occupancy_confusion += counts.reshape(2, 2)
        distance = self.grid.distance_map()
        for start, end in self.distance_bins_m:
            region = valid & (distance >= start) & (distance < end)
            if not np.any(region):
                continue
            region_index = target_occ[region].reshape(-1) * 2 + pred_occ[region].reshape(-1)
            self.occupancy_distance_confusion[(start, end)] += np.bincount(
                region_index, minlength=4
            ).reshape(2, 2)

    def _update_ray_iou(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        pred_occ = (pred != self.free_label) & valid
        target_occ = (target != self.free_label) & valid
        for start, end in self.distance_bins_m:
            pred_ray = _ray_occupancy_map(
                pred_occ,
                self.grid,
                start,
                end,
                self.ray_angle_bins,
                self.ray_distance_resolution,
            )
            target_ray = _ray_occupancy_map(
                target_occ,
                self.grid,
                start,
                end,
                self.ray_angle_bins,
                self.ray_distance_resolution,
            )
            inter = int(np.count_nonzero(pred_ray & target_ray))
            union = int(np.count_nonzero(pred_ray | target_ray))
            stats = self.ray_stats[(start, end)]
            stats["intersection"] += inter
            stats["union"] += union


@dataclass
class HeightPriorCorrelation:
    """Track corr(H_shadow, H_occ) over BEV distance bins."""

    grid: BEVGrid
    distance_bins_m: list[tuple[float, float]]
    min_target_height_m: float = 0.05
    stats: dict[tuple[float, float], dict[str, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.stats = {tuple(bin_): _empty_corr_stats() for bin_ in self.distance_bins_m}
        self.stats[("all", "all")] = _empty_corr_stats()

    def update(self, h_shadow: np.ndarray, h_occ: np.ndarray) -> None:
        pred = np.asarray(h_shadow, dtype=np.float64).squeeze()
        target = np.asarray(h_occ, dtype=np.float64).squeeze()
        if pred.shape != self.grid.shape or target.shape != self.grid.shape:
            raise ValueError(
                f"Expected height maps with shape {self.grid.shape}, got {pred.shape} and {target.shape}"
            )
        valid = np.isfinite(pred) & np.isfinite(target) & (target > self.min_target_height_m)
        distance = self.grid.distance_map()
        self._accumulate(("all", "all"), pred, target, valid)
        for start, end in self.distance_bins_m:
            region = valid & (distance >= start) & (distance < end)
            self._accumulate((start, end), pred, target, region)

    def summarize(self) -> dict:
        rows = []
        for start, end in self.distance_bins_m:
            stats = self.stats[(start, end)]
            rows.append(
                {
                    "range_m": [float(start), float(end)],
                    "count": int(stats["n"]),
                    "corr_H_shadow_H_occ": _corr_from_stats(stats),
                    "mae_H_shadow_H_occ": _mae_from_stats(stats),
                }
            )
        all_stats = self.stats[("all", "all")]
        return {
            "HeightPrior_corr": _corr_from_stats(all_stats),
            "HeightPrior_MAE": _mae_from_stats(all_stats),
            "HeightPrior_corr_by_distance": rows,
        }

    def _accumulate(
        self,
        key: tuple[float, float] | tuple[str, str],
        pred: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        x = pred[valid].reshape(-1)
        y = target[valid].reshape(-1)
        if x.size == 0:
            return
        stats = self.stats[key]
        stats["n"] += float(x.size)
        stats["sum_x"] += float(np.sum(x))
        stats["sum_y"] += float(np.sum(y))
        stats["sum_x2"] += float(np.sum(x * x))
        stats["sum_y2"] += float(np.sum(y * y))
        stats["sum_xy"] += float(np.sum(x * y))
        stats["sum_abs_err"] += float(np.sum(np.abs(x - y)))


def pred_semantic_from_logits(
    semantic_logits: np.ndarray,
    occ_logits: np.ndarray | None,
    free_label: int,
    occ_threshold: float = 0.5,
) -> np.ndarray:
    """Convert model logits to Occ3D-style voxel labels."""
    sem = np.argmax(semantic_logits, axis=0).astype(np.int64)
    if occ_logits is None:
        return sem
    occ_prob = 1.0 / (1.0 + np.exp(-occ_logits))
    sem = np.where(occ_prob >= occ_threshold, sem, free_label)
    return sem.astype(np.int64)


def _ray_occupancy_map(
    occ: np.ndarray,
    grid: BEVGrid,
    start_m: float,
    end_m: float,
    angle_bins: int,
    distance_resolution: float,
) -> np.ndarray:
    z_count, height, width = occ.shape
    rows, cols = np.indices((height, width))
    xy = grid.ij_to_xy(np.stack([rows.ravel(), cols.ravel()], axis=1))
    dist = np.linalg.norm(xy, axis=1)
    region = (dist >= start_m) & (dist < end_m)
    if not np.any(region):
        return np.zeros((angle_bins, 1), dtype=bool)

    angle = np.arctan2(xy[:, 1], xy[:, 0])
    angle_idx = np.floor((angle + np.pi) / (2.0 * np.pi) * angle_bins).astype(np.int64)
    angle_idx = np.clip(angle_idx, 0, angle_bins - 1)
    distance_bins = max(1, int(np.ceil((end_m - start_m) / distance_resolution)))
    dist_idx = np.floor((dist - start_m) / distance_resolution).astype(np.int64)
    dist_idx = np.clip(dist_idx, 0, distance_bins - 1)

    bev_occ = np.any(occ, axis=0).reshape(-1)
    selected = region & bev_occ
    out = np.zeros((angle_bins, distance_bins), dtype=bool)
    out[angle_idx[selected], dist_idx[selected]] = True
    return out


def _iou_from_confusion(confusion: np.ndarray) -> np.ndarray:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    denom = tp + fp + fn
    return np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)


def _safe_nanmean(values: np.ndarray) -> float:
    if values.size == 0 or np.all(~np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))


def _empty_corr_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "sum_abs_err": 0.0,
    }


def _corr_from_stats(stats: dict[str, float]) -> float:
    n = stats["n"]
    if n < 2:
        return float("nan")
    cov = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n
    var_x = stats["sum_x2"] - stats["sum_x"] * stats["sum_x"] / n
    var_y = stats["sum_y2"] - stats["sum_y"] * stats["sum_y"] / n
    denom = float(np.sqrt(max(var_x, 0.0) * max(var_y, 0.0)))
    if denom <= 1e-12:
        return float("nan")
    return float(cov / denom)


def _mae_from_stats(stats: dict[str, float]) -> float:
    if stats["n"] <= 0:
        return float("nan")
    return float(stats["sum_abs_err"] / stats["n"])


@dataclass
class PseudoShadowMetrics:
    """Agreement with generated shadow targets, reported separately from OCC."""

    threshold: float = 0.5
    confusion: dict[str, np.ndarray] = field(init=False)
    direction_cosine_sum: float = 0.0
    direction_count: int = 0
    reliability_abs_error: float = 0.0
    reliability_count: int = 0

    def __post_init__(self) -> None:
        self.confusion = {
            key: np.zeros((2, 2), dtype=np.int64)
            for key in ("shadow", "root", "tip", "caster")
        }

    def update(self, pred: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> None:
        pairs = {
            "shadow": (pred["P_shadow"], target["shadow_mask"]),
            "root": (pred["P_root"], target["shadow_root_heatmap"]),
            "tip": (pred["P_tip"], target["shadow_tip_heatmap"]),
            "caster": (pred["P_caster"], target["shadow_caster_heatmap"]),
        }
        for key, (logits, labels) in pairs.items():
            prediction = _binary_from_logits(logits, self.threshold)
            truth = np.asarray(labels) >= self.threshold
            index = truth.reshape(-1).astype(np.int64) * 2 + prediction.reshape(-1).astype(np.int64)
            self.confusion[key] += np.bincount(index, minlength=4).reshape(2, 2)

        shadow_valid = np.asarray(target["shadow_mask"]).squeeze() >= self.threshold
        if np.any(shadow_valid):
            direction = np.asarray(pred["V_shadow"], dtype=np.float64)
            target_direction = np.asarray(target["shadow_direction"], dtype=np.float64)
            direction = direction / np.maximum(np.linalg.norm(direction, axis=0, keepdims=True), 1e-8)
            target_direction = target_direction / np.maximum(
                np.linalg.norm(target_direction, axis=0, keepdims=True), 1e-8
            )
            cosine = np.sum(direction * target_direction, axis=0)
            self.direction_cosine_sum += float(np.sum(cosine[shadow_valid]))
            self.direction_count += int(np.count_nonzero(shadow_valid))

        predicted_reliability = _sigmoid(np.asarray(pred["U_shadow"], dtype=np.float64))
        reliability = np.asarray(target["shadow_reliability"], dtype=np.float64)
        self.reliability_abs_error += float(np.sum(np.abs(predicted_reliability - reliability)))
        self.reliability_count += int(reliability.size)

    def summarize(self) -> dict:
        result: dict[str, float] = {}
        for key, confusion in self.confusion.items():
            iou = _iou_from_confusion(confusion)
            precision = _precision_from_confusion(confusion)
            recall = _recall_from_confusion(confusion)
            result[f"PseudoShadow_{key}_IoU"] = float(iou[1])
            result[f"PseudoShadow_{key}_F1"] = float(
                2.0 * precision * recall / max(precision + recall, 1e-12)
            )
        result["PseudoShadow_direction_cosine"] = (
            self.direction_cosine_sum / self.direction_count
            if self.direction_count
            else float("nan")
        )
        result["PseudoShadow_reliability_MAE"] = (
            self.reliability_abs_error / self.reliability_count
            if self.reliability_count
            else float("nan")
        )
        return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _binary_from_logits(values: np.ndarray, threshold: float) -> np.ndarray:
    return (_sigmoid(np.asarray(values, dtype=np.float64)) >= threshold).astype(np.int64)


def _precision_from_confusion(confusion: np.ndarray) -> float:
    tp = float(confusion[1, 1])
    return tp / max(float(confusion[:, 1].sum()), 1.0)


def _recall_from_confusion(confusion: np.ndarray) -> float:
    tp = float(confusion[1, 1])
    return tp / max(float(confusion[1, :].sum()), 1.0)
