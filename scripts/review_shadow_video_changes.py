"""Build a CPU-only review pack from two frozen video runs, without rerunning M1-M4."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.video_review import (CHANGE_NAMES, FRAME_SIGNALS, PAIR_SIGNALS, change_components,
    compare_frame, evidence_record, find_clips, load_clip, observation_timeline, owner_region_transitions, summarize_tracks)
from shadow_occ.annotation.video_review_render import render_case, render_continuity, sheet, tile
from shadow_occ.utils.io import write_json


def camera_index(paths):
    result = {}
    for path in paths:
        base = Path(path).resolve().parent
        for pair in read_rows(path):
            for frame in pair["frames"]:
                key = (pair["scene_name"], pair["camera"], frame["sample_data_token"])
                if key in result:
                    previous = result[key][0]
                    if (previous["camera_timestamp"] != frame["camera_timestamp"]
                            or not np.allclose(previous["projection_world_to_image"], frame["projection_world_to_image"], atol=1e-6, rtol=0)):
                        raise ValueError(f"Inconsistent camera geometry for {key}")
                else:
                    result[key] = frame, base
    return result


def write_case_arrays(path, before, after, slot, domain, changes):
    yy, xx = np.where(domain)
    box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
    x0, y0, x1, y1 = box
    crop = np.s_[y0:y1, x0:x1]
    payload = {"grid_box_xyxy": np.asarray(box), "affected_proposal_region": domain[crop]}
    payload.update({name: mask[crop] & domain[crop] for name, mask in changes.items()})
    for label, clip in (("before", before), ("after", after)):
        arrays = clip["arrays"][slot]
        for name in (*FRAME_SIGNALS, "shadow_target_bev", "shadow_owner_id", "shadow_track_id", "receiver_world_xyz"):
            if name in arrays:
                payload[label+"_"+name] = arrays[name][crop]
        for at in (slot-1, slot):
            if 0 <= at < len(clip["rows"])-1:
                for name in PAIR_SIGNALS:
                    if name in clip["pair"]:
                        payload[f"{label}_interval_{at:03d}_{name}"] = clip["pair"][name][at][crop]
    np.savez_compressed(path, **payload)
    return box


def build_review(before_root, after_root, pairs, output_root, context_radius=1):
    before_root, after_root, output = map(lambda value: Path(value).resolve(), (before_root, after_root, output_root))
    if any(output == root or root in output.parents for root in (before_root, after_root)):
        raise ValueError("Place review outputs outside both frozen input runs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new review output directory; existing reviews are not overwritten")
    before_clips, after_clips = find_clips(before_root), find_clips(after_root)
    if before_clips.keys() != after_clips.keys():
        raise ValueError("Frozen runs must contain the same scene/camera/frame windows")
    if not isinstance(context_radius, int) or not 0 <= context_radius <= 3:
        raise ValueError("Context radius must be 0..3 frames")
    cameras = camera_index(pairs)
    output.mkdir(parents=True, exist_ok=True)
    (output/"continuity").mkdir()
    fingerprints = {str(Path(path).resolve()): digest(path) for path in pairs}
    source_summaries = []
    for root in (before_root, after_root):
        if (root/"summary.json").is_file():
            fingerprints[str(root/"summary.json")] = digest(root/"summary.json")
            source_summaries.append(json.loads((root/"summary.json").read_text(encoding="utf-8")))
    if (len(source_summaries) == 2 and all(row.get("pair_manifest_sha256") for row in source_summaries)
            and source_summaries[0]["pair_manifest_sha256"] != source_summaries[1]["pair_manifest_sha256"]):
        raise ValueError("Frozen runs used different pair manifests; this is not a paired algorithm comparison")
    cases, timelines, tracks, transitions, continuity = [], [], [], [], []
    counts = Counter({key: 0 for key in (*CHANGE_NAMES, "retained_positive", "before_positive", "after_positive")})
    for clip_number, key in enumerate(tqdm(sorted(before_clips), desc="frozen video review", unit="clip")):
        before, after = load_clip(before_clips[key]), load_clip(after_clips[key])
        fingerprints.update(before["fingerprints"])
        fingerprints.update(after["fingerprints"])
        scene, camera, tokens = key
        camera_frames = [cameras[(scene, camera, token)] for token in tokens]
        identity = f"clip_{clip_number:03d}"
        for label, clip in (("before", before), ("after", after)):
            timeline = observation_timeline(clip, label)
            scope = {"clip_key": identity, "source_clip_id": clip["protocol"]["clip"]["id"], "scene_name": scene, "camera": camera}
            timelines.extend({**scope, **row} for row in timeline)
            tracks.extend({**scope, **row} for row in summarize_tracks(timeline, len(tokens)))
            path = output/"continuity"/f"{identity}_{label}.jpg"
            render_continuity(path, clip, camera_frames, timeline)
            continuity.append({**scope, "version": label, "contact_sheet": str(path.relative_to(output)),
                "original_review_video": str(clip["folder"]/"review.mp4"),
                "memory_trace_available": (clip["folder"]/"region_memory.jsonl").is_file()})
        for slot, token in enumerate(tokens):
            old, new = before["arrays"][slot], after["arrays"][slot]
            if before["rows"][slot]["timestamp"] != after["rows"][slot]["timestamp"]:
                raise ValueError("Same camera token has different frozen timestamps")
            if before["rows"][slot]["timestamp"] != camera_frames[slot][0]["camera_timestamp"]:
                raise ValueError("Pair manifest timestamp differs from frozen output")
            changes, old_tokens, new_tokens = compare_frame(old, new, before["protocol"], after["protocol"])
            for name, mask in changes.items():
                counts[name] += int(mask.sum())
            counts["before_positive"] += int((old["shadow_target_bev"] == 1).sum())
            counts["after_positive"] += int((new["shadow_target_bev"] == 1).sum())
            counts["retained_positive"] += int(((old["shadow_target_bev"] == 1) & (new["shadow_target_bev"] == 1)).sum())
            frame_transitions = owner_region_transitions(before, after, slot, old_tokens, new_tokens)
            frame_cases = []
            for domain in change_components(old, new, changes):
                changed = domain & np.logical_or.reduce(list(changes.values()))
                case_id = hashlib.sha256(json.dumps(key).encode()+token.encode()+np.packbits(domain).tobytes()).hexdigest()[:16]
                folder = output/"cases"/case_id
                folder.mkdir(parents=True)
                evidence = {label: evidence_record(clip, slot, domain, changed)
                            for label, clip in (("before", before), ("after", after))}
                visuals = render_case(folder, before, after, slot, domain, changes, evidence, camera_frames, context_radius)
                box = write_case_arrays(folder/"evidence.npz", before, after, slot, domain, changes)
                added = domain & changes["added_positive"]
                bits = new.get("proposal_source_bits")
                bit_counts = Counter(map(int, bits[added])) if bits is not None else None
                row = {"case_id": case_id, "clip_key": identity, "scene_name": scene, "camera": camera,
                    "frame_token": token, "frame_slot": slot, "timestamp": after["rows"][slot]["timestamp"],
                    "unit": "affected connected proposal region; counts are BEV cells, not independent shadows",
                    "counts": {name: int((mask & domain).sum()) for name, mask in changes.items()},
                    "affected_region_cells": int(domain.sum()), "grid_box_xyxy": box,
                    "added_cell_source_bit_histogram": dict(bit_counts) if bit_counts is not None else None,
                    "source_bits_are_not_causal_attribution": True,
                    "before_owner_tokens": sorted(set(old_tokens[domain])-{ "" }),
                    "after_owner_tokens": sorted(set(new_tokens[domain])-{ "" }),
                    "before_region_track_ids": sorted(map(int, set(np.unique(old["shadow_track_id"][domain]))-{0})),
                    "after_region_track_ids": sorted(map(int, set(np.unique(new["shadow_track_id"][domain]))-{0})),
                    "review_image": f"cases/{case_id}/review.jpg", "clean_crop": f"cases/{case_id}/clean_crop.jpg",
                    "owner_review_image": f"cases/{case_id}/owner_changes.jpg",
                    "temporal_context": f"cases/{case_id}/temporal_context.jpg",
                    "evidence_json": f"cases/{case_id}/evidence.json", "evidence_arrays": f"cases/{case_id}/evidence.npz",
                    "review_status": "unreviewed", **visuals}
                write_json(folder/"evidence.json", {**row, **evidence,
                    "before_owner_vocabulary": before["protocol"]["owner_index_to_instance_token"],
                    "after_owner_vocabulary": after["protocol"]["owner_index_to_instance_token"],
                    "before_frame_manifest": str(before["folder"]/"frames.jsonl"),
                    "after_frame_manifest": str(after["folder"]/"frames.jsonl")})
                cases.append(row)
                frame_cases.append(row)
            for row in frame_transitions:
                row["case_ids"] = [case["case_id"] for case in frame_cases if
                    {item["track_id_within_run"] for item in row["before"]}.intersection(case["before_region_track_ids"])
                    or {item["track_id_within_run"] for item in row["after"]}.intersection(case["after_region_track_ids"])]
                transitions.append({"clip_key": identity, "scene_name": scene, "camera": camera, **row})
    counts["net_positive_change"] = counts["after_positive"]-counts["before_positive"]
    write_rows(output/"review_manifest.jsonl", cases)
    write_rows(output/"reviews_template.jsonl", ({"case_id": row["case_id"], "shadow_judgment": "unreviewed",
        "added_positive_judgment": "unreviewed" if row["counts"]["added_positive"] else "not_applicable",
        "removed_positive_judgment": "unreviewed" if row["counts"]["removed_positive"] else "not_applicable",
        "caster_judgment": "unreviewed", "correct_caster_token": None, "notes": ""} for row in cases))
    write_rows(output/"owner_region_transitions.jsonl", transitions)
    write_rows(output/"observation_timeline.jsonl", timelines)
    write_rows(output/"track_observations.jsonl", tracks)
    write_rows(output/"continuity_manifest.jsonl", continuity)
    pages = []
    for start in range(0, len(cases), 12):
        tiles = []
        for row in cases[start:start+12]:
            image = cv2.imread(str(output/row["clean_crop"]))
            tiles.append(tile(image, f"{row['case_id'][:8]}: add {row['counts']['added_positive']}, remove {row['counts']['removed_positive']}"))
        path = output/"contact_sheets"/f"page_{start//12:03d}.jpg"
        checked_write_image(path, sheet(tiles))
        pages.append(str(path.relative_to(output)))
    summary = {"review_protocol": "frozen-video-review-v1", "before_root": str(before_root), "after_root": str(after_root),
        "clips": len(before_clips), "frames": sum(len(key[2]) for key in before_clips),
        "counts": dict(counts), "review_cases": len(cases), "contact_sheets": pages,
        "context_radius": context_radius,
        "owner_region_relations": dict(Counter(row["relation"] for row in transitions)),
        "owner_region_changes": dict(Counter(row["owner_change"] for row in transitions)),
        "mask_derived_regions_with_resolved_cells": {label: sum(item["resolved_cells"] > 0 for row in transitions
             for item in row[label]) for label in ("before", "after")},
        "algorithm_recomputed": False, "training_performed": False, "ground_truth_generated": False,
        "change_categories_can_overlap": True, "source_bits_are_not_causal_attribution": True,
        "owner_identity": "caster token, never run-local numeric owner/track equality",
        "continuity": "current frame image/BEV masks separate from memory-only retention",
        "precision_recall_computed": False}
    write_json(output/"input_fingerprints.json", fingerprints)
    write_json(output/"summary.json", summary)
    lines = ["# Frozen Shadow Video Review", "", "Input discovery results are read-only. All cases start unreviewed.",
        "Source-bit-only does not mean no appearance evidence. Grid cells are not independent shadow regions.",
        "", "## Changes", "", f"Added {counts['added_positive']}; removed {counts['removed_positive']}; net {counts['net_positive_change']} cells.",
        "", "## Cases", "", "| Case | Scene | Added | Removed | Review | Owner | Context |", "|---|---|---:|---:|---|---|---|"]
    for row in cases:
        lines.append(f"| {row['case_id']} | {row['scene_name']} | {row['counts']['added_positive']} | {row['counts']['removed_positive']} | [image]({row['review_image']}) | [owner]({row['owner_review_image']}) | [frames]({row['temporal_context']}) |")
    lines.extend(["", "## Continuity", "", "Image masks are observations, not manual shadow GT. Memory-only states emit no observed mask.", ""])
    lines.extend(f"- {row['clip_key']} {row['version']}: [contact sheet]({row['contact_sheet']})" for row in continuity)
    lines.extend(["", "Use reviews_template.jsonl for independent judgments; it is not a training manifest.",
        "Owner-region transitions explicitly retain splits, merges and token changes. Selected box/light overlays are geometry, not shadow truth."])
    (output/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="Original camera pair manifests, including train/val when needed")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--context-radius", type=int, choices=range(4), default=1)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(build_review(args.before_root, args.after_root, args.pairs, args.output_root, args.context_radius)), flush=True)


if __name__ == "__main__":
    main()
