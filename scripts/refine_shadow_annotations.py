from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import GT_SOURCES, checked_write_image, read_rows, write_rows
from shadow_occ.annotation.refinement import refine_target
from shadow_occ.annotation.readiness import require_prediction_for_proposals
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
    parser.add_argument("--allow-weak-pilot", action="store_true", help="Explicit experimental relabeling after independent teacher diagnostics; not approval")
    args = parser.parse_args()
    if not args.allow_weak_pilot:
        raise ValueError("Refinement requires --allow-weak-pilot; unverified segmenter/VLM agreement is not ground truth")
    if not 0.5 < args.pixel_threshold <= 1 or not 0.5 < args.region_threshold <= 1:
        raise ValueError("Pixel and region thresholds must be in (0.5, 1]")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    seen_votes = set()
    skipped_boundary = 0
    for vote in read_rows(args.vlm_votes or args.annotation_results):
        if args.annotation_results:
            if not vote.get("parent_id"):
                # Boundary annotations already carry their full image mask reference;
                # they need a sample/camera join rather than the caster-crop parent join.
                skipped_boundary += 1
                continue
            vote = {**vote, "annotation_decision": (vote.get("prediction") or {}).get("shadow", "UNCERTAIN")}
        if not vote.get("parent_id") or not vote.get("region_mask"):
            raise ValueError("Votes must come from evaluate_shadow_vlm.py with parent IDs and region masks")
        if vote["id"] in seen_votes:
            raise ValueError("Duplicate annotation vote ID")
        seen_votes.add(vote["id"])
        groups[vote["parent_id"]].append(vote)
    predictions = read_rows(args.seg_predictions)
    ids = [row["id"] for row in predictions]
    if len(ids) != len(set(ids)) or set(groups)-set(ids):
        raise ValueError("Duplicate segmentation ID or annotation parent absent from predictions")
    rows, report = [], {"examples": 0, "accepted_regions": 0, "positive_pixels": 0, "no_votes": 0, "skipped_boundary_annotations": skipped_boundary}
    for index, row in enumerate(predictions):
        require_prediction_for_proposals(row)
        if row["id"] not in groups:
            report["no_votes"] += 1
        with np.load(row["probability"], allow_pickle=False) as archive:
            probability = np.asarray(archive["probability"], dtype=np.float32)
        old_target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        if old_target is None:
            raise FileNotFoundError(row["target"])
        decisions = []
        for vote in groups[row["id"]]:
            mask = cv2.imread(vote["region_mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != probability.shape:
                raise ValueError("Region and segmentation pixel coordinates differ")
            if "annotation_decision" in vote:
                decision = vote["annotation_decision"]
            else:
                score = float(vote["score"])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid legacy VLM score")
                decision = "YES" if score >= args.region_threshold else "NO" if score <= 1-args.region_threshold else "UNCERTAIN"
            decisions.append((mask > 0, decision))
            if decision == "YES":
                report["accepted_regions"] += 1
        preserve_gt = row["label_source"] in GT_SOURCES
        target, changes = refine_target(probability, old_target, decisions, args.pixel_threshold, preserve_gt)
        path = output/"targets"/f"{index:07d}.png"
        checked_write_image(path, target)
        source = row["label_source"] if preserve_gt and not changes["new_positive_pixels"] else "model_pseudo"
        rows.append({**row, "target": str(path), "shadow_mask": str(path), "mask_encoding": "trimap", "label_source": source, "refinement": {"pixel_threshold": args.pixel_threshold, "region_threshold": args.region_threshold, "original_label_source": row["label_source"], **changes}})
        for name, count in changes.items():
            report[name] = report.get(name, 0)+count
        report["examples"] += 1
        report["positive_pixels"] += int((target == 1).sum())
    write_rows(output/"manifest.jsonl", rows)
    write_json(output/"refinement_summary.json", report)


if __name__ == "__main__":
    main()
