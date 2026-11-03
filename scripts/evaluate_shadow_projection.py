from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shadow_occ.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        "Evaluate camera-ground projection and temporal light diagnostics."
    )
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    label_root = Path(args.label_root)
    json_paths = sorted(label_root.glob("*/*.json")) + sorted(label_root.glob("*.json"))
    reprojection_errors: list[float] = []
    projected_points: list[int] = []
    temporal_overlaps: list[int] = []
    temporal_selected: list[int] = []
    selected_points: list[int] = []
    light_differences: list[float] = []
    samples = 0
    cameras = 0
    for path in json_paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        samples += 1
        for camera in payload.get("meta", {}).get("camera_projection", []):
            cameras += 1
            reprojection = camera.get("reprojection", {})
            count = int(reprojection.get("count", 0))
            if count > 0 and np.isfinite(reprojection.get("p95_px", np.nan)):
                reprojection_errors.append(float(reprojection["p95_px"]))
            projected_points.append(int(camera.get("projected_points", 0)))
            temporal_overlaps.append(int(camera.get("temporal_overlap", 0)))
            temporal_selected.append(int(camera.get("temporal_selected_points", 0)))
            selected_points.append(int(camera.get("selected_points", 0)))
            light_value = float(camera.get("light_difference_mean", np.nan))
            if np.isfinite(light_value):
                light_differences.append(light_value)

    summary = {
        "samples": samples,
        "cameras": cameras,
        "reprojection_p95_px_mean": _mean(reprojection_errors),
        "reprojection_p95_px_median": _median(reprojection_errors),
        "reprojection_p95_px_max": _max(reprojection_errors),
        "projected_ground_points_mean": _mean(projected_points),
        "temporal_overlap_pixels_mean": _mean(temporal_overlaps),
        "temporal_selected_ground_points_mean": _mean(temporal_selected),
        "temporal_selected_ground_point_ratio": _ratio(temporal_selected, projected_points),
        "final_shadow_ground_points_mean": _mean(selected_points),
        "final_shadow_ground_point_ratio": _ratio(selected_points, projected_points),
        "light_difference_mean": _mean(light_differences),
    }
    output = ensure_dir(Path(args.output_root) / "metrics") / "shadow_projection_metrics.json"
    write_json(output, summary)
    print(summary)
    print(f"wrote {output}")


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def _ratio(numerators: list[int], denominators: list[int]) -> float:
    if not numerators or not denominators:
        return float("nan")
    numerator = float(np.sum(numerators))
    denominator = float(np.sum(denominators))
    return numerator / denominator if denominator else float("nan")


if __name__ == "__main__":
    main()
