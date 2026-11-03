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


METRICS = (
    "Occupancy_mIoU",
    "Occupied_IoU",
    "Semantic_mIoU",
    "RayIoU",
    "HeightPrior_corr",
    "HeightPrior_MAE",
)


def main() -> None:
    parser = argparse.ArgumentParser("Summarize repeated seeds.")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="NAME=PATH1,PATH2,...; repeat for each experiment.",
    )
    parser.add_argument("--output-root", default="outputs/seed_summary")
    args = parser.parse_args()

    summary = {}
    for item in args.experiment:
        name, paths_text = item.split("=", 1)
        payloads = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in paths_text.split(",")
        ]
        values = {}
        for key in METRICS:
            array = np.asarray([float(payload.get(key, np.nan)) for payload in payloads])
            values[key] = {
                "mean": float(np.nanmean(array)),
                "std": float(np.nanstd(array)),
                "values": array.tolist(),
            }
        summary[name] = {"runs": len(payloads), "metrics": values}

    output = ensure_dir(args.output_root) / "seed_summary.json"
    write_json(output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
