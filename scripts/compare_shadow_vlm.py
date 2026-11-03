from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import read_rows
from shadow_occ.annotation.metrics import region_summary
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Compare frozen base and fine-tuned VLM on identical region IDs.")
    parser.add_argument("--base-votes", required=True)
    parser.add_argument("--tuned-votes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-input-ablation", action="store_true", help="Explicitly compare different frame/context inputs, not only different adapters")
    args = parser.parse_args()
    base_rows, tuned_rows = read_rows(args.base_votes), read_rows(args.tuned_votes)
    base, tuned = {row["id"]: row for row in base_rows}, {row["id"]: row for row in tuned_rows}
    if len(base) != len(base_rows) or len(tuned) != len(tuned_rows) or set(base) != set(tuned):
        raise ValueError("Comparison requires identical, unique example IDs")
    for key in base:
        for field in ("answer", "label_source", "scene_name", "group", "pair_id"):
            if base[key].get(field) != tuned[key].get(field):
                raise ValueError(f"Metadata mismatch: {key}/{field}")
        if not args.allow_input_ablation and base[key].get("input_signature") != tuned[key].get("input_signature"):
            raise ValueError("Input images/prompts changed; use --allow-input-ablation for an explicit input ablation")
    paired = {}
    for source in sorted({row["label_source"] for row in base_rows}):
        keys = [key for key in base if base[key]["label_source"] == source and base[key].get("answer") in {"YES", "NO"}]
        if not keys:
            continue
        scenes = sorted({base[key]["scene_name"] for key in keys})
        delta = np.asarray([int((tuned[key]["score"] >= args.threshold) == (base[key]["answer"] == "YES"))-int((base[key]["score"] >= args.threshold) == (base[key]["answer"] == "YES")) for key in keys])
        clusters = {scene: [i for i, key in enumerate(keys) if base[key]["scene_name"] == scene] for scene in scenes}
        rng = np.random.default_rng(args.seed)
        distribution = [float(delta[np.concatenate([clusters[scenes[i]] for i in rng.integers(0, len(scenes), len(scenes))])].mean()) for _ in range(args.bootstrap)]
        paired[source] = {"examples": len(keys), "accuracy_delta": float(delta.mean()), "scene_cluster_ci95": np.percentile(distribution, [2.5, 97.5]).tolist() if distribution else None}
    write_json(args.output, {"base": region_summary(base_rows, args.threshold, args.bootstrap, args.seed), "tuned": region_summary(tuned_rows, args.threshold, args.bootstrap, args.seed), "paired_by_label_source": paired, "input_ablation_allowed": args.allow_input_ablation})


if __name__ == "__main__":
    main()
