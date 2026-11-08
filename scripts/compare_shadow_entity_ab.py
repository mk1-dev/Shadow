"""Paired entity suppression and memory-owner comparison on shared raw caches."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.discover_shadow_video import run_clip, sample_clip, select_chain
from shadow_occ.annotation.bev_flow import BEVFlowConfig
from shadow_occ.annotation.data import digest, load_array, read_rows
from shadow_occ.annotation.entity_memory import EntityMotionConfig
from shadow_occ.annotation.frame_receiver import frame_index
from shadow_occ.annotation.physics_io import check_pair_files, load_motion_analysis
from shadow_occ.annotation.shadow_video import VideoShadowConfig
from shadow_occ.utils.io import write_json


VARIANTS = {
    "A_no_suppression": (False, "geometry"),
    "B_suppression_geometry": (True, "geometry"),
    "B_suppression_memory": (True, "geometry+appearance+bev"),
}


def compare_clip(clip, chain, inputs, context, output, config, reviews=None):
    """Identical features, motion matches and caster memory in every arm."""
    results = {}
    for name, (enabled, owner_mode) in VARIANTS.items():
        arm = {**inputs, "entity_suppression_mask": inputs["entity_suppression_mask"].copy()}
        if not enabled:
            arm["entity_suppression_mask"][:] = False
        state = copy.deepcopy(context)
        state["entity_suppression_enabled"] = enabled
        state["entity_maps"]["entity_suppression_mask"] = arm["entity_suppression_mask"]
        state["entity_maps"]["entity_uncertain_mask"] = arm["entity_occlusion_mask"] & ~arm["entity_suppression_mask"]
        results[name] = run_clip(clip, chain, arm, state, output/name/clip["id"], config,
                                stage="full", bev_role="context", reviews=reviews, owner_scoring=owner_mode,
                                discovery_mode="appearance")
    return results


def summarize(results, output):
    table = {}
    for name in VARIANTS:
        clips = [result[name] for result in results]
        row = {key: sum(clip[key] for clip in clips) for key in (
            "frames", "regions", "candidate_cells", "entity_candidate_cells", "positive_cells", "owner_cells",
            "unique_caster_regions", "additional_search_removed_cells")}
        row["timing"] = {key: sum(clip["timing"][key] for clip in clips) for key in clips[0]["timing"]}
        eligible = retained = proposed = review_rows = 0
        for clip in clips:
            for review in read_rows(output/name/clip["id"]/"reviewed_signals.jsonl"):
                if review["region_type"] == "shadow":
                    metrics = review["search_preservation"]
                    eligible += metrics["eligible_review_cells"]
                    retained += metrics["retained_review_cells"]
                    proposed += metrics["candidate_review_cells"]
                    review_rows += 1
        row["reviewed_shadow_preservation"] = {
            "reference": "independent review, sampled on common receiver-ray grid",
            "review_rows": review_rows, "eligible_cells": eligible,
            "retained_cells": retained, "candidate_cells": proposed,
            "search_retention": retained/eligible if eligible else None,
            "candidate_coverage": proposed/eligible if eligible else None}
        table[name] = row
    a, b, m = (table[name] for name in VARIANTS)
    return {"variants": table, "comparisons": {
        "suppression_B_minus_A": {
            "candidate_cells": b["candidate_cells"]-a["candidate_cells"],
            "positive_cells": b["positive_cells"]-a["positive_cells"],
            "unique_caster_regions": b["unique_caster_regions"]-a["unique_caster_regions"],
            "downstream_seconds": sum(b["timing"][key]-a["timing"][key] for key in (
                "proposal_seconds", "region_seconds", "physics_seconds", "owner_seconds"))},
        "memory_minus_geometry": {
            "positive_cells": m["positive_cells"]-b["positive_cells"],
            "owner_cells": m["owner_cells"]-b["owner_cells"],
            "unique_caster_regions": m["unique_caster_regions"]-b["unique_caster_regions"]}},
        "memory_only_preserves_proposals_and_targets": all(
            result["B_suppression_geometry"].get("mask_hashes") is not None
            and all(result["B_suppression_geometry"]["mask_hashes"][key]
                    == result["B_suppression_memory"]["mask_hashes"][key] for key in ("proposal", "target"))
            for result in results),
        "runtime_scope": "shared sampling excluded; downstream stages timed separately; total_seconds includes visualization/export",
        "order": list(VARIANTS), "timing_is_single_pass_not_benchmark": True,
        "new_owner_policy": "local linked-shadow memory, not legacy independent shared-light validation",
        "shadow_precision_recall_computed": False, "training_performed": False}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for flag in ("pairs", "analysis-root", "feature-root", "clips", "output-root"):
        parser.add_argument("--"+flag, required=True)
    parser.add_argument("--review-regions", help="Independent masks; same format as discover_shadow_video.py")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        (output/name).mkdir()
    cv2.setNumThreads(1)
    pairs, stats, _ = load_motion_analysis(args.pairs, args.analysis_root)
    index = frame_index(pairs)
    clips = read_rows(args.clips)
    if not clips or len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("Choose nonempty uniquely identified clips")
    reviews = read_rows(args.review_regions) if args.review_regions else []
    if not {review["clip_id"] for review in reviews} <= {clip["id"] for clip in clips}:
        raise ValueError("Review references a clip outside this run")
    review_base = Path(args.review_regions).resolve().parent if args.review_regions else None
    results, sampling_seconds = [], 0.
    for clip in tqdm(clips, desc="paired entity suppression", unit="clip"):
        if Path(clip["id"]).name != clip["id"] or clip["id"] in {".", ".."}:
            raise ValueError("Clip id must be a filename")
        chain = select_chain(pairs, clip)
        for pair in chain:
            check_pair_files(pair, Path(args.pairs).resolve().parent)
        tick = perf_counter()
        inputs, context = sample_clip(chain, index, Path(args.pairs).resolve().parent,
            stats, Path(args.analysis_root).resolve(), args.feature_root, BEVFlowConfig(extent_m=40),
            entity_config=EntityMotionConfig(), discovery_domain="image-rays")
        sampling_seconds += perf_counter()-tick
        clip_reviews = {}
        for review in reviews:
            if review["clip_id"] == clip["id"]:
                if review["region_type"] not in {"shadow", "stable_ground", "material_boundary", "entity"}:
                    raise ValueError("Unknown review type")
                clip_reviews.setdefault(review["frame_token"], []).append({**review,
                    "mask": load_array(review["mask"], review_base) > 0})
        if not set(clip_reviews) <= {frame["sample_data_token"] for frame in context["frames"]}:
            raise ValueError("Reviewed frame is outside selected clip")
        results.append(compare_clip(clip, chain, inputs, context, output, VideoShadowConfig(), clip_reviews))
    report = summarize(results, output)
    write_json(output/"summary.json", {**report, "clips": results, "shared_sampling_seconds": sampling_seconds,
        "pair_manifest_sha256": digest(args.pairs), "clip_manifest_sha256": digest(args.clips),
        "review_manifest_sha256": digest(args.review_regions) if args.review_regions else None,
        "discovery_domain": "image-rays", "bev_role": "context"})
    print(report, flush=True)


if __name__ == "__main__":
    main()
