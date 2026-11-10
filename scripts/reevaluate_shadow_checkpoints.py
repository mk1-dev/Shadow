"""Re-run saved checkpoints sequentially in one inference process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.segmentation import predict_main
from shadow_occ.annotation.data import digest
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Unique NAME=CHECKPOINT entries")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    names, results = set(), []
    for entry in args.checkpoints:
        name, checkpoint = entry.split("=", 1)
        if not name or name in names or Path(name).name != name or name in {".", ".."} or "\\" in name:
            raise ValueError("Checkpoint review names must be unique simple names")
        names.add(name)
        predict_main(["--manifest", args.manifest, "--checkpoint", checkpoint,
                      "--output-root", str(Path(args.output_root)/name), "--device", args.device,
                      "--threshold", "0.5"])
        metrics = json.loads((Path(args.output_root)/name/"segmentation_metrics.json").read_text(encoding="utf-8"))
        results.append({"name": name, "checkpoint_sha256": digest(checkpoint),
                        "by_label_source": metrics["by_label_source"]})
    write_json(Path(args.output_root)/"reevaluation_summary.json", {"process_id": os.getpid(),
        "reference_sha256": digest(args.manifest), "threshold": .5, "checkpoints": results,
        "scope": "same-process frozen-teacher agreement; not manual shadow accuracy"})


if __name__ == "__main__":
    main()
