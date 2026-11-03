from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import read_rows
from shadow_occ.annotation.joint import summarize
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Compare two runs of the same GT-constrained shadow annotation task.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    a_rows, b_rows = read_rows(args.base), read_rows(args.tuned)
    a, b = {row["id"]: row for row in a_rows}, {row["id"]: row for row in b_rows}
    if set(a) != set(b) or len(a) != len(a_rows) or len(b) != len(b_rows):
        raise ValueError("Annotation runs must contain identical unique IDs")
    for key in a:
        for field in ("target", "shadow_label_source", "association_source", "scene_name", "input_signature"):
            if a[key].get(field) != b[key].get(field):
                raise ValueError(f"Different targets or inputs: {key}/{field}")
    paired = {}
    for source in sorted({row["shadow_label_source"] for row in a_rows}):
        keys = [key for key in a if a[key]["shadow_label_source"] == source and a[key].get("target") is not None]
        if not keys:
            continue
        def correct(row):
            return int(row["prediction"] is not None and row["prediction"]["shadow"] == row["target"]["shadow"])
        delta = np.asarray([correct(b[key])-correct(a[key]) for key in keys])
        scenes = sorted({a[key]["scene_name"] for key in keys})
        clusters = {scene: [i for i, key in enumerate(keys) if a[key]["scene_name"] == scene] for scene in scenes}
        rng = np.random.default_rng(args.seed)
        samples = [float(delta[np.concatenate([clusters[scenes[i]] for i in rng.integers(0, len(scenes), len(scenes))])].mean()) for _ in range(args.bootstrap)]
        paired[source] = {"labeled": len(keys), "shadow_agreement_delta": float(delta.mean()), "scene_cluster_ci95": np.percentile(samples, [2.5, 97.5]).tolist() if samples else None}
    write_json(args.output, {"base": summarize(a_rows), "tuned": summarize(b_rows), "paired_by_shadow_source": paired})


if __name__ == "__main__":
    main()
