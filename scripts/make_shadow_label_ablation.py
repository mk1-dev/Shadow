"""Change training trimaps only; keep the same dev/eval reference manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import checked_write_image, digest, read_rows, resolve_path, write_rows
from shadow_occ.utils.io import write_json


def ablate_target(target, policy, rng, ground=None):
    result = target.copy()
    if policy == "positive-only":
        result[result == 0] = 255
    elif policy == "shuffled-negatives":
        count = int((target == 0).sum())
        result[result == 0] = 255
        support = np.ones(target.shape, bool) if ground is None else ground
        pool = np.flatnonzero(support & (target != 1) & (target != 0))
        if len(pool) < count:
            pool = np.flatnonzero(support & (target != 1))
        chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
        result.flat[chosen] = 0
    elif policy == "all-background":
        result[result != 1] = 0
    elif policy != "endpoints":
        raise ValueError(f"Unknown label policy: {policy}")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policy", choices=["positive-only", "endpoints", "shuffled-negatives", "all-background"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source, output = Path(args.data_root).resolve(), Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    counts, inputs = Counter(), {}
    for split in ("train", "dev", "eval"):
        manifest = source/f"seg_{split}.jsonl"
        rows = read_rows(manifest)
        inputs[split] = digest(manifest)
        for row in rows:
            for key in ("image", "target", "ground_support"):
                if row.get(key):
                    row[key] = str(resolve_path(row[key], source))
            if split != "train":
                continue
            target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
            if target is None or not np.isin(target, [0, 1, 255]).all():
                raise ValueError("Training targets must be 0/1/255 trimaps")
            ground = None
            if row.get("ground_support"):
                ground = cv2.imread(row["ground_support"], cv2.IMREAD_GRAYSCALE)
                if ground is None or ground.shape != target.shape:
                    raise ValueError("Ground support and target coordinates differ")
                ground = ground > 0
            seed = int(hashlib.sha256(f"{args.seed}:{row['id']}".encode()).hexdigest()[:16], 16)
            altered = ablate_target(target, args.policy, np.random.default_rng(seed), ground)
            name = hashlib.sha256(row["id"].encode()).hexdigest()[:24]
            path = output/"targets"/f"{name}.png"
            checked_write_image(path, altered)
            row.update(target=str(path), known_pixel_count=int((altered != 255).sum()),
                training_label_ablation=args.policy, original_target_sha256=digest(row["target"]))
            counts["positive_pixels"] += int((altered == 1).sum())
            counts["negative_pixels"] += int((altered == 0).sum())
            counts["ignore_pixels"] += int((altered == 255).sum())
            counts["original_negative_pixels"] += int((target == 0).sum())
            counts["negative_overlap_with_original"] += int(((target == 0) & (altered == 0)).sum())
        write_rows(output/f"seg_{split}.jsonl", rows)
    write_json(output/"label_ablation_protocol.json", {**vars(args), "input_manifest_sha256": inputs,
        "counts": dict(counts), "scope": "segmenter training only; dev/eval targets unchanged",
        "positive_only_training_flag": "--positive-only-ablation" if args.policy == "positive-only" else None})


if __name__ == "__main__":
    main()
