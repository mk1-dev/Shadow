from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import assert_disjoint, digest, read_rows
from shadow_occ.annotation.diagnostics import balance_audit, decision
from shadow_occ.annotation.metrics import auc
from shadow_occ.utils.io import write_json


def profile(rows):
    groups = {}
    for label in ("YES", "NO", "UNCERTAIN", None):
        selected = [row for row in rows if decision(row) == label]
        counters = {key: Counter() for key in ("camera", "distance", "illumination", "ground_type", "temporal_mode")}
        for row in selected:
            center = row.get("caster_center_ego")
            distance = "unknown" if center is None else "0-10m" if np.linalg.norm(center[:2]) < 10 else "10-20m" if np.linalg.norm(center[:2]) < 20 else "20m+"
            for key, value in (("camera", row.get("camera")), ("distance", distance), ("illumination", row.get("illumination")), ("ground_type", row.get("ground_type")), ("temporal_mode", row.get("temporal", {}).get("mode"))):
                counters[key][str(value) if value is not None else "unknown"] += 1
        groups[label or "unlabeled"] = {key: dict(value) for key, value in counters.items()}
    temporal = {}
    for name in ("onset_darkening_ratio", "recovery_brightening_ratio", "motion_alignment_advantage", "shadow_retention"):
        selected = [row for row in rows if decision(row) in {"YES", "NO"} and row.get("temporal", {}).get(name) is not None]
        temporal[name] = {"observed": len(selected), "auroc_against_existing_targets": auc([decision(row) == "YES" for row in selected], [row["temporal"][name] for row in selected]),
                          "scope": "Availability-biased weak audit; not independent shadow verification"}
    return {"coverage": groups, "temporal_feature_audit": temporal, "rows": len(rows),
            "duplicate_source_image_rows": len(rows)-len({row.get("image", row["id"]) for row in rows})}


def main():
    parser = argparse.ArgumentParser("Audit independent-positive coverage, temporal evidence and outside-BEV attrition.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--eval")
    parser.add_argument("--stage-protocols", nargs="*", default=[])
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    datasets = {split: read_rows(path) for split, path in (("train", args.train), ("dev", args.dev), ("eval", args.eval)) if path}
    assert_disjoint(datasets["train"], datasets["dev"]+datasets.get("eval", []))
    if {row["scene_name"] for row in datasets["dev"]} & {row["scene_name"] for row in datasets.get("eval", [])}:
        raise ValueError("Dev and eval overlap")
    reports = {split: {"sampling": balance_audit(rows, {}, 0.25), **profile(rows)} for split, rows in datasets.items()}
    stages = {path: json.loads(Path(path).read_text(encoding="utf-8")) for path in args.stage_protocols}
    write_json(output/"data_audit.json", {"splits": reports, "stage_protocols": stages, "input_sha256": {path: digest(path) for path in (args.train, args.dev, args.eval) if path},
               "policy": "Do not replace unknown illumination/ground metadata with inferred GT. Stage counters may use different units; compare matching units only."})
    write_json(output/"temporal_policy_template.json", {"fitted_split": "dev", "dev_scenes": sorted({row["scene_name"] for row in datasets["dev"]}),
               "validated_for_weighting": False, "downweight": 0.5, "minimum_pixels": 64, "minimum_motion_advantage": 0.1,
               "minimum_transition_ratio": 0.6, "scope": "Must be independently reviewed/frozen on dev before optional training use; never creates positive labels"})


if __name__ == "__main__":
    main()
