from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.diagnostics import SCORING_VERSION, classify, cluster_intervals, decision, fit_threshold, score_metrics
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Select a decision policy on dev, or apply a frozen policy without eval threshold search.")
    parser.add_argument("mode", choices=["fit-dev", "evaluate"])
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--max-fpr", type=float, default=0.1)
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    parser.add_argument("--max-uncertainty", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = read_rows(args.scores)
    if not rows or len({row["id"] for row in rows}) != len(rows) or len({row["scorer_id"] for row in rows}) != 1:
        raise ValueError("Need unique rows from exactly one scorer")
    if any(row.get("scoring_version") != SCORING_VERSION or row.get("in_sample_annotation") for row in rows):
        raise ValueError("Only heldout, current-version scores may be calibrated/evaluated")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.mode == "fit-dev":
        if args.calibration:
            raise ValueError("Do not supply an existing calibration when fitting dev")
        policy, scan = fit_threshold(rows, args.max_fpr, args.abstain_margin, args.max_uncertainty)
        policy.update(scorer_id=rows[0]["scorer_id"], scoring_version=SCORING_VERSION, fitted_split="dev", dev_scenes=sorted({row["scene_name"] for row in rows}),
                      dev_ids=[row["id"] for row in rows], dev_scores_sha256=digest(args.scores))
        write_json(output/"calibration.json", policy)
        write_json(output/"dev_threshold_scan.json", scan)
    else:
        if not args.calibration:
            raise ValueError("Evaluation requires --calibration from fit-dev")
        policy = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if policy.get("fitted_split") != "dev" or policy.get("scorer_id") != rows[0]["scorer_id"] or policy.get("scoring_version") != SCORING_VERSION:
            raise ValueError("Calibration/scorer/model/view mismatch")
        if any(row.get("split") != "eval" for row in rows):
            raise ValueError("Final evaluation requires eval rows, not train/dev")
        if set(policy["dev_scenes"]) & {row["scene_name"] for row in rows} or set(policy["dev_ids"]) & {row["id"] for row in rows}:
            raise ValueError("Calibration and evaluation overlap")
    threshold, margin, uncertainty = policy["threshold"], policy["abstain_margin"], policy["max_uncertainty"]
    by_source = {source: score_metrics([row for row in rows if row["shadow_label_source"] == source], threshold, margin, uncertainty)
                 for source in sorted({row["shadow_label_source"] for row in rows})}
    results = [{**row, "target_decision": decision(row), "score_prediction": classify(row["score"], threshold, margin, row.get("uncertainty_score", 0.0), uncertainty)} for row in rows]
    write_rows(output/"predictions.jsonl", results)
    intervals = cluster_intervals(rows, threshold, margin, uncertainty, args.bootstrap, args.seed)
    write_json(output/"metrics.json", {"mode": args.mode, "policy": policy, "all_sources_diagnostic": score_metrics(rows, threshold, margin, uncertainty), "by_shadow_source": by_source, "scene_cluster_intervals": intervals,
               "scope": "Frozen score decisions, not caster localization; weak labels remain weak. Eval never selects thresholds."})


if __name__ == "__main__":
    main()
