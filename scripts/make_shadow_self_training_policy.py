from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows
from shadow_occ.annotation.self_training import make_policy
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Gate an offline self-training pilot using frozen dev scores; never approve production labels.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--dev-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--minimum-recall", type=float, default=0.2)
    parser.add_argument("--minimum-positives", type=int, default=20)
    parser.add_argument("--minimum-scenes", type=int, default=3)
    parser.add_argument("--positive-cutoff", type=float, default=0.9)
    parser.add_argument("--negative-cutoff", type=float, default=0.1)
    parser.add_argument("--allow-negative-pseudo-labels", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("dev_scores_sha256") != digest(args.dev_scores):
        raise ValueError("Use exactly the dev scores that fitted this calibration")
    policy = make_policy(calibration, read_rows(args.dev_scores), minimum_precision=args.minimum_precision,
                         minimum_recall=args.minimum_recall, minimum_positives=args.minimum_positives,
                         minimum_scenes=args.minimum_scenes, positive_cutoff=args.positive_cutoff,
                         negative_cutoff=args.negative_cutoff, allow_negative=args.allow_negative_pseudo_labels)
    policy["calibration_sha256"] = digest(args.calibration)
    policy["dev_scores_sha256"] = digest(args.dev_scores)
    write_json(args.output, policy)
    print({"eligible_for_pilot": policy["eligible_for_pilot"], "blocking_reasons": policy["blocking_reasons"], "dev_score_gate": policy["dev_score_gate"]})


if __name__ == "__main__":
    main()
