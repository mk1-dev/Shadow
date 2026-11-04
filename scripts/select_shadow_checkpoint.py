from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows
from shadow_occ.annotation.diagnostics import SCORING_VERSION, fit_threshold
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Select checkpoint AND threshold using identical dev observations, never eval.")
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-fpr", type=float, default=0.1)
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    parser.add_argument("--max-uncertainty", type=float, default=1.0)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    reference, results = None, []
    for path in args.scores:
        rows = read_rows(path)
        if not rows or len({row["id"] for row in rows}) != len(rows) or len({row["scorer_id"] for row in rows}) != 1:
            raise ValueError("Empty/duplicate/mixed checkpoint score file")
        signature = {row["id"]: (row.get("annotation_target"), row.get("shadow_label_source"), row.get("input_signature")) for row in rows}
        if reference is not None and signature != reference:
            raise ValueError("Checkpoint comparison requires identical targets and inputs")
        reference = signature
        if any(row.get("in_sample_annotation") or row.get("scoring_version") != SCORING_VERSION for row in rows):
            raise ValueError("In-sample or obsolete scores cannot select checkpoints")
        policy, _ = fit_threshold(rows, args.max_fpr, args.abstain_margin, args.max_uncertainty)
        policy.update(fitted_split="dev", scorer_id=rows[0]["scorer_id"], scoring_version=SCORING_VERSION,
                      dev_scenes=sorted({row["scene_name"] for row in rows}), dev_ids=[row["id"] for row in rows], dev_scores_sha256=digest(path))
        results.append({"scores": str(Path(path).resolve()), "policy": policy})
    best = max(results, key=lambda item: (item["policy"]["dev_metrics"]["balanced_accuracy"], item["policy"]["dev_metrics"]["positive_recall"]))
    write_json(output/"calibration.json", best["policy"])
    write_json(output/"checkpoint_selection.json", {"selected": best, "candidates": results, "scope": "Dev selection; no eval feedback. No adapter files are changed."})


if __name__ == "__main__":
    main()
