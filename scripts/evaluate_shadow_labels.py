from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_occ.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate generated shadow-label statistics.")
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--output-name", default="shadow_label_metrics.json")
    args = parser.parse_args()

    root = Path(args.label_root)
    npz_paths = sorted(root.glob("*/*.npz")) + sorted(root.glob("*.npz"))
    json_paths = {path.stem: path for path in root.glob("*/*.json")}
    component_rows: list[dict] = []
    shadow_occupancy: list[float] = []
    reliability: list[float] = []
    nonempty = 0
    temporal_selected = 0
    projected_points = 0
    temporal_overlap = []
    light_differences = []
    geometry_sources: set[str] = set()

    for path in npz_paths:
        with np.load(path, allow_pickle=True) as data:
            mask = np.asarray(data.get("shadow_mask", 0), dtype=np.float32)
            rel = np.asarray(data.get("shadow_reliability", 0), dtype=np.float32)
            shadow_occupancy.append(float(np.mean(mask > 0.5)))
            reliability.append(float(np.mean(rel)))
            if np.any(mask > 0.5):
                nonempty += 1

        sidecar = json_paths.get(path.stem)
        if sidecar is None:
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        source = payload.get("geometry_source")
        if source:
            geometry_sources.add(str(source))
        for component in payload.get("components", []):
            component_rows.append(component)
        for camera in payload.get("meta", {}).get("camera_projection", []):
            projected_points += int(camera.get("projected_points", 0))
            temporal_selected += int(camera.get("temporal_selected_points", 0))
            temporal_overlap.append(int(camera.get("temporal_overlap", 0)))
            value = float(camera.get("light_difference_mean", np.nan))
            if np.isfinite(value):
                light_differences.append(value)

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [float(row[key]) for row in component_rows if row.get(key) is not None],
            dtype=np.float64,
        )

    stable = sum(bool(row.get("stable", False)) for row in component_rows)
    summary = {
        "samples": len(npz_paths),
        "nonempty_shadow_samples": nonempty,
        "nonempty_shadow_ratio": _ratio(nonempty, len(npz_paths)),
        "geometry_sources": sorted(geometry_sources),
        "mean_shadow_bev_occupancy": _mean(shadow_occupancy),
        "mean_reliability": _mean(reliability),
        "components": len(component_rows),
        "components_per_sample": _ratio(len(component_rows), len(npz_paths)),
        "stable_component_ratio": _ratio(stable, len(component_rows)),
        "length_m_mean": _mean(values("length_m")),
        "elongation_mean": _mean(values("elongation")),
        "caster_stability_mean": _mean(values("caster_stability")),
        "tip_motion_m_mean": _mean(values("tip_motion_m")),
        "tip_smoothness_mean": _mean(values("tip_smoothness")),
        "temporal_light_score_mean": _mean(values("temporal_light_score")),
        "projected_points": projected_points,
        "temporal_selected_points": temporal_selected,
        "temporal_selected_ratio": _ratio(temporal_selected, projected_points),
        "temporal_overlap_pixels_mean": _mean(temporal_overlap),
        "light_difference_mean": _mean(light_differences),
    }
    output = ensure_dir(Path(args.output_root) / "metrics") / args.output_name
    write_json(output, summary)
    print(summary)
    print(f"wrote {output}")


def _mean(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


if __name__ == "__main__":
    main()
