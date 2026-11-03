from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_occ.utils.io import ensure_dir, write_json


SCALARS = (
    "Occupancy_mIoU",
    "Occupied_IoU",
    "Semantic_mIoU",
    "RayIoU",
    "HeightPrior_corr",
    "HeightPrior_MAE",
    "PseudoShadow_shadow_F1",
    "PseudoShadow_direction_cosine",
)


def main() -> None:
    parser = argparse.ArgumentParser("Compare matched Shadow-OCC experiments.")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="NAME=METRICS_JSON, repeat once per experiment; first is the baseline.",
    )
    parser.add_argument("--output-root", default="outputs/metrics_compare")
    args = parser.parse_args()

    experiments: list[tuple[str, dict]] = []
    for item in args.experiment:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        metrics_path = Path(path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiments.append((name, payload))

    baseline = experiments[0][1]
    rows = []
    for name, payload in experiments:
        row = {"experiment": name}
        for key in SCALARS:
            value = payload.get(key)
            row[key] = _number(value)
            row[f"delta_{key}"] = _number(value) - _number(baseline.get(key))
        for distance_row in payload.get("RayIoU_by_distance", []):
            start, end = distance_row["range_m"]
            key = f"RayIoU_{start:g}_{end:g}m"
            value = distance_row.get("RayIoU")
            row[key] = _number(value)
            row[f"delta_{key}"] = _number(value) - _baseline_distance(
                baseline.get("RayIoU_by_distance", []), start, end
            )
        for distance_row in payload.get("Occupancy_mIoU_by_distance", []):
            start, end = distance_row["range_m"]
            key = f"Occupancy_mIoU_{start:g}_{end:g}m"
            value = distance_row.get("Occupancy_mIoU")
            row[key] = _number(value)
            row[f"delta_{key}"] = _number(value) - _baseline_distance(
                baseline.get("Occupancy_mIoU_by_distance", []), start, end, "Occupancy_mIoU"
            )
        rows.append(row)

    output_root = ensure_dir(args.output_root)
    output_json = output_root / "experiment_comparison.json"
    write_json(
        output_json,
        {
            "baseline": experiments[0][0],
            "experiments": [name for name, _ in experiments],
            "rows": rows,
        },
    )
    output_csv = output_root / "experiment_comparison.csv"
    keys = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _baseline_distance(rows: list[dict], start: float, end: float, key: str = "RayIoU") -> float:
    for row in rows:
        current_start, current_end = row.get("range_m", [None, None])
        if float(current_start) == float(start) and float(current_end) == float(end):
            return _number(row.get(key))
    return float("nan")


if __name__ == "__main__":
    main()
