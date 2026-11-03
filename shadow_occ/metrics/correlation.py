from __future__ import annotations

from pathlib import Path

import numpy as np

from shadow_occ.geometry.bev import BEVGrid


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def load_length_height_pairs(label_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths: list[float] = []
    heights: list[float] = []
    distances: list[float] = []
    for path in label_paths:
        data = np.load(path, allow_pickle=True)
        if "components" not in data:
            continue
        for item in data["components"]:
            comp = item.item() if hasattr(item, "item") else item
            length = float(comp.get("length_m", np.nan))
            height = float(comp.get("mean_height_near_root", np.nan))
            root = np.asarray(comp.get("root_xy", [np.nan, np.nan]), dtype=np.float64)
            lengths.append(length)
            heights.append(height)
            distances.append(float(np.linalg.norm(root)))
    return np.asarray(lengths), np.asarray(heights), np.asarray(distances)


def correlation_by_distance(
    lengths: np.ndarray,
    heights: np.ndarray,
    distances: np.ndarray,
    bins: list[tuple[float, float]],
) -> list[dict]:
    rows = []
    for start, end in bins:
        mask = (distances >= start) & (distances < end)
        rows.append(
            {
                "range_m": [float(start), float(end)],
                "count": int(np.count_nonzero(mask)),
                "pearson_length_height": pearson_corr(lengths[mask], heights[mask]),
                "mean_length_m": _safe_mean(lengths[mask]),
                "mean_height_m": _safe_mean(heights[mask]),
            }
        )
    rows.append(
        {
            "range_m": ["all"],
            "count": int(len(lengths)),
            "pearson_length_height": pearson_corr(lengths, heights),
            "mean_length_m": _safe_mean(lengths),
            "mean_height_m": _safe_mean(heights),
        }
    )
    return rows


def occupancy_iou_by_distance(
    pred: np.ndarray,
    target: np.ndarray,
    grid: BEVGrid,
    bins: list[tuple[float, float]],
) -> list[dict]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    distance = grid.distance_map()
    rows = []
    for start, end in bins:
        region = (distance >= start) & (distance < end)
        inter = np.count_nonzero(pred & target & region)
        union = np.count_nonzero((pred | target) & region)
        rows.append(
            {
                "range_m": [float(start), float(end)],
                "intersection": int(inter),
                "union": int(union),
                "iou": float(inter / union) if union else float("nan"),
            }
        )
    return rows


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))
