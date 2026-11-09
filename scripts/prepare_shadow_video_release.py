"""Blind review, result reveal and qualitative videos from frozen outputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.video_review import find_clips, load_clip, observation_timeline
from shadow_occ.annotation.video_review_render import heat, overlay, project_mask, read_rgb, selected_projection, sheet, tile
from shadow_occ.utils.io import write_json
from review_shadow_video_changes import camera_index


def new_output(value, inputs):
    output = Path(value).resolve()
    if any(output == Path(root).resolve() or Path(root).resolve() in output.parents for root in inputs):
        raise ValueError("Keep exports outside frozen input directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_video(path, frames, timestamps, fps=None):
    """Preserve camera timing on a constant-rate MP4 timeline, not a fixed 2 Hz guess."""
    times = np.asarray(timestamps, dtype=np.float64)/1e6
    if len(times) == 0 or len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Video needs increasing camera timestamps")
    fps = float(fps or (1/np.median(np.diff(times)) if len(times) > 1 else 2))
    if not np.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("Invalid playback FPS")
    writer, size, count, processed = None, None, 0, 0
    try:
        for slot, image in enumerate(frames):
            processed += 1
            if slot >= len(times):
                raise ValueError("Video frame count differs from timestamps")
            # Encoders may silently trim odd dimensions; pad without rescaling pixels.
            image = cv2.copyMakeBorder(image, 0, image.shape[0] % 2, 0, image.shape[1] % 2, cv2.BORDER_CONSTANT)
            if writer is None:
                size = image.shape[1::-1]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError(f"MP4 encoder unavailable: {path}")
            if image.shape[1::-1] != size:
                raise ValueError("Video image dimensions changed")
            end = round((times[slot+1]-times[0])*fps) if slot+1 < len(times) else round((times[-1]-times[0])*fps)+1
            while count < end:
                writer.write(image)
                count += 1
        if processed != len(times):
            raise ValueError("Video frame count differs from timestamps")
    finally:
        if writer is not None:
            writer.release()
    return {"fps": fps, "encoded_frames": count, "camera_timestamps_us": list(map(int, timestamps))}


def prepare_blind(review_root, run_root, pairs, output_root):
    review, run = Path(review_root).resolve(), Path(run_root).resolve()
    cases = read_rows(review/"review_manifest.jsonl")
    entries = find_clips(run)
    cameras = camera_index(pairs)
    output = new_output(output_root, [review, run])
    public, private = output/"blind", output/"private"
    public.mkdir()
    private.mkdir()
    # Opaque IDs and full clean clips avoid exposing source bits, ROI geometry or owner guesses.
    mapping, tasks, videos = [], [], {}
    for case in cases:
        matching = [key for key in entries if key[:2] == (case["scene_name"], case["camera"])
                    and case["frame_token"] in key[2]]
        # Overlapping clips are identified by their saved full ordered frame window.
        saved = json.loads((review/case["evidence_json"]).read_text(encoding="utf-8"))
        manifest = Path(saved["after_frame_manifest"])
        matching = [key for key in matching if entries[key][0].name == manifest.parent.name]
        if len(matching) != 1:
            raise ValueError(f"Cannot uniquely locate reviewed clip: {case['case_id']}")
        key = matching[0]
        blind_clip = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        if blind_clip not in videos:
            frames = [cameras[(key[0], key[1], token)] for token in key[2]]
            times = [frame[0]["camera_timestamp"] for frame in frames]
            if times != [row["timestamp"] for row in entries[key][2]]:
                raise ValueError("Blind RGB timestamps differ from frozen outputs")
            name = f"{blind_clip}.mp4"
            info = write_video(public/name, (read_rgb(*item) for item in frames), times)
            videos[blind_clip] = {"video": name, **info}
        identity = hashlib.sha256((case["case_id"]+blind_clip).encode()).hexdigest()[:16]
        tasks.append({"review_id": identity, "video": videos[blind_clip]["video"],
            "focus_frame_slot": key[2].index(case["frame_token"]),
            "shadow_visibility": "unreviewed", "shadow_types": [], "caster_visibility": "unreviewed",
            "caster_description": "", "notes": ""})
        mapping.append({"review_id": identity, "case_id": case["case_id"], "clip_key": blind_clip})
    write_rows(public/"judgments.jsonl", tasks)
    write_rows(private/"tasks_frozen.jsonl", tasks)
    write_json(public/"videos.json", list(videos.values()))
    (public/"README.txt").write_text(
        "Stage 1: watch clean full clips before seeing model outputs.\n"
        "shadow_visibility: clear / weak / not_observed / unjudgeable\n"
        "shadow_types: vehicle / non_vehicle / ambiguous (zero or more)\n"
        "caster_visibility: visible / outside_view / multiple_plausible / unjudgeable\n"
        "Describe possible casters visually; no model token or mask is provided.\n"
        "Record observations across the full scene, including non-vehicle shadows.\n", encoding="utf-8")
    write_rows(private/"mapping.jsonl", mapping)
    write_json(private/"protocol.json", {"review_root": str(review), "run_root": str(run),
        "review_manifest_sha256": digest(review/"review_manifest.jsonl"),
        "template_sha256": digest(private/"tasks_frozen.jsonl"),
        "blind_media_sha256": {row["video"]: digest(public/row["video"]) for row in videos.values()},
        "pairs_sha256": {str(Path(path).resolve()): digest(path) for path in pairs},
        "algorithm_recomputed": False})
    return {"cases": len(tasks), "clean_clips": len(videos), "reviewer_directory": str(public),
        "operator_only_directory": str(private)}


def reveal_review(pack_root, judgments_path, output_root, reviewer):
    pack = Path(pack_root).resolve()
    protocol = json.loads((pack/"private/protocol.json").read_text(encoding="utf-8"))
    review = Path(protocol["review_root"])
    if digest(review/"review_manifest.jsonl") != protocol["review_manifest_sha256"]:
        raise ValueError("Review case manifest changed after blind preparation")
    for name, sha in protocol["blind_media_sha256"].items():
        if digest(pack/"blind"/name) != sha:
            raise ValueError("Blind video changed")
    mapping = read_rows(pack/"private/mapping.jsonl")
    judgments = read_rows(judgments_path)
    indexed = {row["review_id"]: row for row in judgments}
    if len(indexed) != len(judgments) or set(indexed) != {row["review_id"] for row in mapping}:
        raise ValueError("Submit exactly one blind judgment for every case")
    if digest(pack/"private/tasks_frozen.jsonl") != protocol["template_sha256"]:
        raise ValueError("Frozen blind task template changed")
    templates = {row["review_id"]: row for row in read_rows(pack/"private/tasks_frozen.jsonl")}
    for identity, row in indexed.items():
        if any(row.get(key) != templates[identity][key] for key in ("video", "focus_frame_slot")):
            raise ValueError("Blind task references changed")
        if row.get("shadow_visibility") not in {"clear", "weak", "not_observed", "unjudgeable"}:
            raise ValueError("Complete shadow visibility judgments before revealing outputs")
        if row.get("caster_visibility") not in {"visible", "outside_view", "multiple_plausible", "unjudgeable"}:
            raise ValueError("Complete caster visibility judgments")
        if not isinstance(row.get("shadow_types"), list) or not set(row["shadow_types"]) <= {"vehicle", "non_vehicle", "ambiguous"}:
            raise ValueError("Invalid blind shadow types")
    output = new_output(output_root, [pack, review, protocol["run_root"]])
    # Snapshot submitted judgments before publishing any model overlay into the reveal pack.
    shutil.copyfile(judgments_path, output/"blind_judgments_submitted.jsonl")
    cases = {row["case_id"]: row for row in read_rows(review/"review_manifest.jsonl")}
    revealed = []
    for item in mapping:
        case = cases[item["case_id"]]
        target = output/"cases"/item["review_id"]
        target.parent.mkdir(exist_ok=True)
        shutil.copytree(review/"cases"/item["case_id"], target)
        revealed.append({"review_id": item["review_id"], "case_id": item["case_id"],
            "blind_judgment": indexed[item["review_id"]], "model_review": f"cases/{item['review_id']}/review.jpg",
            "region_quality": "unreviewed", "added_positive_quality": "unreviewed" if case["counts"]["added_positive"] else "not_applicable",
            "removed_positive_quality": "unreviewed" if case["counts"]["removed_positive"] else "not_applicable",
            "owner_judgment": "unreviewed", "correct_caster_tokens": [], "notes": ""})
    write_rows(output/"overlay_judgments.jsonl", revealed)
    write_json(output/"protocol.json", {**protocol, "reviewer": reviewer,
        "submitted_judgments_sha256": digest(output/"blind_judgments_submitted.jsonl"),
        "blind_review_recorded_before_reveal": True, "human_independence_verified": False})
    (output/"README.txt").write_text(
        "Stage 2: compare saved model overlays with your frozen Stage 1 observations.\n"
        "Region quality: shadow_core / boundary_extension / material_expansion / entity_leakage / non_vehicle_shadow / uncertain / mixed.\n"
        "Owner judgment: unique_correct / incorrect / multiple_plausible / unjudgeable.\n"
        "Model tokens and projections are hypotheses, not reviewer ground truth.\n"
        "Use correct_caster_tokens only when identity is independently established.\n"
        "These judgments are not automatically converted to training labels.\n", encoding="utf-8")
    return {"revealed_cases": len(revealed), "output": str(output)}


def showcase_frames(clip, camera_frames):
    for slot, item in enumerate(camera_frames):
        rgb = read_rgb(*item)
        frame = item[0]
        arrays, saved = clip["arrays"][slot], clip["images"][slot]
        plane = clip["protocol"]["ground_plane_world"]
        missing = np.full(arrays["shadow_target_bev"].shape, np.nan)
        suppressed = arrays.get("entity_suppression_mask", np.zeros_like(missing, bool))
        entities = overlay(rgb, project_mask(suppressed, arrays, frame, plane), (30, 30, 230))
        projection, _, _ = selected_projection(clip, slot, clip["rows"][slot]["regions"])
        casters = overlay(rgb, project_mask(projection, arrays, frame, plane), (0, 180, 240))
        draw_caster_tracks(casters, clip, slot, frame)
        proposal = overlay(rgb, saved["shadow_proposal_image"].astype(bool), (220, 180, 0))
        target = saved["shadow_target_image"] == 1
        boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        observed = overlay(rgb, target, (30, 220, 60))
        observed[boundary] = (0, 255, 255)
        bev = np.zeros((*missing.shape, 3), np.uint8)
        bev[arrays["shadow_proposal_bev"].astype(bool)] = (120, 90, 0)
        bev[arrays["shadow_target_bev"] == 1] = (30, 220, 60)
        flow = np.zeros_like(rgb)
        flow_title = "Raw flow yellow / geometric flow blue"
        if slot < len(clip["rows"])-1 and "observed_flow_px" in clip["pair"]:
            uv = project_world_grid(arrays, frame)
            actual = clip["pair"]["observed_flow_px"][slot]
            expected = clip["pair"].get("expected_flow_px")
            for y, x in np.ndindex(*missing.shape):
                if (y*missing.shape[1]+x) % 12 or not np.isfinite(uv[y, x]).all():
                    continue
                start = uv[y, x]
                if not (0 <= start[0] < rgb.shape[1] and 0 <= start[1] < rgb.shape[0]):
                    continue
                for vector, color in ((actual[y, x], (0, 220, 220)),
                                      (expected[slot, y, x] if expected is not None else [np.nan]*2, (230, 120, 30))):
                    if np.isfinite(vector).all():
                        end = start+vector
                        if np.max(np.abs(end)) < 1e6:
                            cv2.arrowedLine(flow, tuple(start.astype(int)), tuple(end.astype(int)), color, 1, tipLength=.2)
        else:
            flow_title = "No outgoing flow (last frame)" if slot == len(clip["rows"])-1 else "Saved flow evidence unavailable"
        entity_title = "M1: entity core suppression overlay" if "entity_suppression_mask" in arrays else "M1: saved suppression map unavailable"
        yield sheet([tile(rgb, f"Clean RGB; camera time {frame['camera_timestamp']}"),
            tile(entities, entity_title),
            tile(heat(arrays.get("bev_anchor_consistency_by_frame", missing)), "M2: same-world BEV anchor", nearest=True),
            tile(flow, flow_title),
            tile(heat(arrays.get("appearance_score_map", missing)), "Observed appearance evidence", nearest=True),
            tile(proposal, "Observed continuous proposal; not GT"),
            tile(casters, "M3: caster track + saved light projection"),
            tile(observed, "M4: weak image target and boundary"),
            tile(bev, "Separate Shadow-BEV; not occupancy", nearest=True)])


def draw_caster_tracks(image, clip, slot, frame):
    from shadow_occ.annotation.physics_motion import project_world
    tokens = set()
    for region in clip["rows"][slot]["regions"]:
        for role in ("selected", "association"):
            hypothesis = region.get("physics", {}).get(role)
            if hypothesis:
                tokens.add(hypothesis["instance_token"])
    for token in sorted(tokens):
        previous = None
        for observation in sorted(clip["casters"].get(token, {}).get("observations", []), key=lambda row: row["slot"]):
            if observation["slot"] > slot:
                continue
            corners = np.asarray(observation["corners_world"], float)
            uv, _, front = project_world(np.vstack([corners, corners.mean(axis=0)]), frame["projection_world_to_image"])
            inside = front & np.isfinite(uv).all(axis=-1) & (uv >= 0).all(axis=-1)
            inside &= (uv[:, 0] < image.shape[1]) & (uv[:, 1] < image.shape[0])
            if inside[-1]:
                point = tuple(uv[-1].astype(int))
                if previous is not None and observation["slot"] == previous[0]+1:
                    cv2.line(image, previous[1], point, (255, 180, 0), 2)
                previous = observation["slot"], point
                cv2.circle(image, point, 3, (255, 180, 0), -1)
                if observation["slot"] == slot:
                    cv2.putText(image, token[:8], point, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 180, 0), 1)
            else:
                previous = None
            if observation["slot"] == slot:
                for point in uv[:-1][inside[:-1]]:
                    cv2.circle(image, tuple(point.astype(int)), 2, (0, 180, 240), -1)


def project_world_grid(arrays, frame):
    from shadow_occ.annotation.physics_motion import project_world
    uv, _, front = project_world(arrays["receiver_world_xyz"].reshape(-1, 3), frame["projection_world_to_image"])
    uv[~front] = np.nan
    return uv.reshape(*arrays["shadow_target_bev"].shape, 2)


def export_showcase(run_root, pairs, output_root, clip_ids=None):
    entries = find_clips(run_root)
    selected = set(clip_ids) if clip_ids else {entry[1]["clip"]["id"] for entry in entries.values()}
    if not selected <= {entry[1]["clip"]["id"] for entry in entries.values()}:
        raise ValueError("Unknown requested clip ID")
    output = new_output(output_root, [run_root])
    cameras, rows = camera_index(pairs), []
    for key, entry in entries.items():
        if entry[1]["clip"]["id"] not in selected:
            continue
        clip = load_clip(entry)
        name = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        frames = [cameras[(key[0], key[1], token)] for token in key[2]]
        times = [row[0]["camera_timestamp"] for row in frames]
        for slot, time in enumerate(times):
            if time != clip["rows"][slot]["timestamp"]:
                raise ValueError("Showcase RGB timestamps differ from saved outputs")
        clean = write_video(output/f"{name}_clean.mp4", (read_rgb(*row) for row in frames), times)
        panels = write_video(output/f"{name}_mainline.mp4", showcase_frames(clip, frames), times)
        rows.append({"clip_id": clip["protocol"]["clip"]["id"], "scene_name": key[0], "camera": key[1],
            "clean_video": f"{name}_clean.mp4", "mainline_video": f"{name}_mainline.mp4",
            "playback": panels, "clean_playback": clean,
            "observation_timeline": observation_timeline(clip, "frozen"),
            "input_sha256": clip["fingerprints"], "quality_status": "unreviewed"})
    write_rows(output/"showcase_manifest.jsonl", rows)
    write_json(output/"protocol.json", {"algorithm_recomputed": False, "training_performed": False,
        "selection": "operator-selected qualitative examples" if clip_ids else "all input clips; no quality filtering",
        "pairs_sha256": {str(Path(path).resolve()): digest(path) for path in pairs},
        "is_shadow_gt": False, "memory_masks_synthesized": False})
    return {"clips": len(rows), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    blind = sub.add_parser("blind")
    blind.add_argument("--review-root", required=True)
    blind.add_argument("--run-root", required=True)
    blind.add_argument("--pairs", nargs="+", required=True)
    blind.add_argument("--output-root", required=True)
    reveal = sub.add_parser("reveal")
    reveal.add_argument("--pack-root", required=True)
    reveal.add_argument("--judgments", required=True)
    reveal.add_argument("--reviewer", required=True)
    reveal.add_argument("--output-root", required=True)
    showcase = sub.add_parser("showcase")
    showcase.add_argument("--run-root", required=True)
    showcase.add_argument("--pairs", nargs="+", required=True)
    showcase.add_argument("--clip-ids", nargs="+")
    showcase.add_argument("--output-root", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.command == "blind":
        result = prepare_blind(args.review_root, args.run_root, args.pairs, args.output_root)
    elif args.command == "reveal":
        result = reveal_review(args.pack_root, args.judgments, args.output_root, args.reviewer)
    else:
        result = export_showcase(args.run_root, args.pairs, args.output_root, args.clip_ids)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
