from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import read_rows
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Summarize caster-track visibility, light change and shadow retention.")
    parser.add_argument("--manifest", required=True, help="Output of add_shadow_temporal_evidence.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    result = {"examples": len(rows), "status_counts": dict(Counter(row.get("temporal_status", "missing") for row in rows)), "by_motion": {}}
    keys = ("shadow_retention", "world_abs_light_change_mean", "repeated_shadow_small_change_ratio", "caster_speed_mps", "background_residual_L")
    for mode in ("static", "moving"):
        selected = [row["temporal"] for row in rows if row.get("temporal", {}).get("mode") == mode]
        comparable = [item for item in selected if item["lighting_comparable"]]
        stats = {"pairs": len(selected), "lighting_comparable_pairs": len(comparable), "positive_consistency_weight_pairs": sum(item["consistency_weight_sum"] > 0 for item in comparable)}
        for key in keys:
            values = [item[key] for item in comparable if item.get(key) is not None]
            stats[key+"_mean"] = float(np.mean(values)) if values else None
            stats[key+"_count"] = len(values)
        stats["propagated_shadow_pixels"] = sum(item.get("propagated_shadow_pixels", 0) for item in selected)
        result["by_motion"][mode] = stats
    result["interpretation"] = "Consistency of weak masks in common visible ground; not measured shadow recall or proof that every GT caster casts a visible shadow"
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
