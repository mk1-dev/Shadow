from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import checked_write_image, read_rows, write_rows
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Combine segmentation probabilities with single-region VLM votes.")
    parser.add_argument("--seg-predictions", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--vlm-votes", help="Legacy binary region votes")
    inputs.add_argument("--annotation-results", help="Unified annotations.jsonl from evaluate_shadow_annotator.py")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pixel-threshold", type=float, default=0.7)
    parser.add_argument("--region-threshold", type=float, default=0.8)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for vote in read_rows(args.vlm_votes or args.annotation_results):
        if args.annotation_results:
            if not vote.get("parent_id"):
                # Boundary annotations already carry their full image mask reference;
                # they need a sample/camera join rather than the caster-crop parent join.
                continue
            vote = {**vote, "annotation_decision": (vote.get("prediction") or {}).get("shadow", "UNCERTAIN")}
        if not vote.get("parent_id") or not vote.get("region_mask"):
            raise ValueError("Votes must come from evaluate_shadow_vlm.py with parent IDs and region masks")
        groups[vote["parent_id"]].append(vote)
    rows, report = [], {"examples": 0, "accepted_regions": 0, "positive_pixels": 0, "no_votes": 0}
    for index, row in enumerate(read_rows(args.seg_predictions)):
        if row["id"] not in groups:
            report["no_votes"] += 1
            continue
        with np.load(row["probability"], allow_pickle=False) as archive:
            probability = np.asarray(archive["probability"], dtype=np.float32)
        old_target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        accepted = np.zeros(probability.shape, dtype=bool)
        for vote in groups[row["id"]]:
            mask = cv2.imread(vote["region_mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != probability.shape:
                raise ValueError("Region and segmentation pixel coordinates differ")
            accept = vote["annotation_decision"] == "YES" if "annotation_decision" in vote else vote["score"] >= args.region_threshold
            if accept:
                accepted |= mask > 0
                report["accepted_regions"] += 1
        target = np.full(probability.shape, 255, dtype=np.uint8)
        target[(old_target == 0) & (probability <= 1-args.pixel_threshold) & ~accepted] = 0
        target[accepted & (probability >= args.pixel_threshold)] = 1
        path = output/"targets"/f"{index:07d}.png"
        checked_write_image(path, target)
        rows.append({**row, "target": str(path), "shadow_mask": str(path), "mask_encoding": "trimap", "label_source": "model_pseudo", "refinement": {"pixel_threshold": args.pixel_threshold, "region_threshold": args.region_threshold, "original_label_source": row["label_source"]}})
        report["examples"] += 1
        report["positive_pixels"] += int((target == 1).sum())
    write_rows(output/"manifest.jsonl", rows)
    write_json(output/"refinement_summary.json", report)


if __name__ == "__main__":
    main()
