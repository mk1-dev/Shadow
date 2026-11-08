"""Track-centric BEV/flow discrepancy discovery using exported raw observations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_mask_to_image, bev_projection, ground_grid, image_mask_to_bev, rasterize_volumes
from shadow_occ.annotation.cached_bev import sample_cached_bev
from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.annotation.frame_receiver import frame_index
from shadow_occ.annotation.physics_io import check_pair_files, load_motion_analysis, read_evidence
from shadow_occ.annotation.physics_labels import (
    LabelConfig, factorized_shadow_decision, footprint_mask, light_bank,
    light_support_index, rank_hypotheses, window_tracks,
)
from shadow_occ.annotation.physics_motion import pair_evidence, project_world, sample_at
from shadow_occ.annotation.receiver_explanations import box_ray_evidence
from shadow_occ.annotation.shadow_video import ENTITY_STATES, ShadowRegionMemory, VideoShadowConfig, refine_regions, video_proposals
from shadow_occ.annotation.shadow_projection import projection_coverage, reviewed_projection_coverage, summarize_signals
from shadow_occ.utils.io import write_json


def select_chain(pairs, clip):
    lookup = {pair["sample_token"]: pair for pair in pairs
              if pair["scene_name"] == clip["scene_name"] and pair["camera"] == clip["camera"]}
    requested = int(clip.get("num_frames", 9))
    if not 3 <= requested <= 31:
        raise ValueError("Choose 3..31 frames per clip")
    token, chain = clip["start_sample_token"], []
    for _ in range(requested-1):
        if token not in lookup:
            raise ValueError(f"Clip is not contiguous at sample {token}")
        pair = lookup[token]
        if not any(obj["instance_token"] == clip["track_id"] for obj in pair["objects"]):
            raise ValueError(f"Selected track has no interpolated state at {token}")
        chain.append(pair)
        token = pair["next_sample_token"]
    if len({pair["official_split"] for pair in chain}) != 1:
        raise ValueError("Clip crosses official splits")
    return chain


def sample_clip(chain, index, base, stats, analysis, feature_root, config):
    """One fixed world grid; per-frame masks and independent valid frame pairs."""
    observations = [index[(chain[0]["scene_name"], chain[0]["camera"], chain[0]["frames"][0]["sample_data_token"])]]
    observations += [index[(pair["scene_name"], pair["camera"], pair["frames"][1]["sample_data_token"])] for pair in chain]
    frames = [item["frame"] for item in observations]
    anchor_item = observations[len(observations)//2]
    frame, pair = anchor_item["frame"], anchor_item["pair"]
    anchor = np.asarray(frame.get("receiver_anchor_ego_to_world", pair["anchor_ego_to_world"]), float)
    basis = np.asarray(frame.get("receiver_basis_world", pair["plane_basis_world"]), float)
    plane = np.asarray(frame.get("receiver_plane_world", pair["plane_world"]), float)
    grid = config.grid()
    world = ground_grid(basis, anchor, grid)
    images, grounds, coordinates, visible, labs, occupied, features, feature_valid = ([] for _ in range(8))
    for item in observations:
        frame = item["frame"]
        image = cv2.imread(str(resolve_path(frame["image"], base)))
        ground = load_array(frame["ground_support"], base) > 0
        if image is None or image.shape[:2] != ground.shape or image.shape[1::-1] != tuple(frame["image_size_wh"]):
            raise ValueError("Receiver image, support and calibrated image size disagree")
        uv, _, front = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
        height, width = image.shape[:2]
        inside = front & (uv >= 0).all(axis=1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1)
        pixels = np.where(inside[:, None], uv, -1).reshape(*grid.shape, 2).astype(np.float32)
        support = inside.reshape(grid.shape) & (sample_at(ground.astype(np.float32), pixels) >= .999)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] *= 100/255
        feature, supported, _ = sample_cached_bev(feature_root, item["sample_token"], world)
        images.append(image)
        grounds.append(ground)
        coordinates.append(pixels)
        visible.append(support)
        labs.append(sample_at(lab, pixels))
        occupied.append(rasterize_volumes(frame, anchor, grid)[0])
        features.append(feature)
        feature_valid.append(supported & support)
    visible, occupied = np.stack(visible), np.stack(occupied)
    observed, expected, fb, texture, deltas, comparable, fits, fb_error = ([] for _ in range(8))
    for slot, pair in enumerate(chain):
        evidence, _ = read_evidence(pair, stats, analysis)
        # Re-register cached flow against canonical frame support, without flow inference.
        fresh = pair_evidence(images[slot], images[slot+1], grounds[slot], grounds[slot+1],
            frames[slot]["projection_world_to_image"], frames[slot+1]["projection_world_to_image"],
            basis, evidence["flow_t_to_next"], evidence["flow_next_to_t"])
        observed.append(sample_at(evidence["flow_t_to_next"], coordinates[slot]))
        expected.append(coordinates[slot+1]-coordinates[slot])
        fb.append(sample_at(fresh["flow_reliable"].astype(np.float32), coordinates[slot]) >= .999)
        fb_error.append(sample_at(fresh["flow_fb_error_px"], coordinates[slot]))
        texture.append(sample_at(fresh["receiver_texture"].astype(np.float32), coordinates[slot]) >= .999)
        fit = fresh["details"]["exposure"]
        fits.append(fit)
        comparable.append(bool(fit["valid"]))
        deltas.append(fit["scale"]*labs[slot+1][..., 0]+fit["offset_L"]-labs[slot][..., 0])
    observed, expected, fb = np.stack(observed), np.stack(expected), np.stack(fb)
    volumes = box_ray_evidence(world, frames, coordinates, observed, fb)
    receiver = visible & ~occupied & ~volumes["physical_ray_volume_support"]
    entity_good = volumes["box_surface_flow_valid"] & fb
    ground_error = np.linalg.norm(observed-expected, axis=-1)
    entity_error = volumes["box_surface_flow_residual_px"]
    entity_fit = np.where(entity_good, np.exp(-entity_error/3)*np.clip((ground_error-entity_error)/3, 0, 1), 0)
    return {"lab": np.stack(labs), "receiver": receiver, "delta_L": np.stack(deltas),
        "exposure_valid": np.asarray(comparable)[:, None, None], "observed_flow": observed,
        "expected_flow": expected, "fb_consistent": fb, "texture_support": np.stack(texture),
        "features": np.stack(features), "feature_valid": np.stack(feature_valid),
        "entity_explanation": entity_fit.astype(np.float32), "entity_explanation_valid": entity_good,
        "fb_error_px": np.stack(fb_error),
        "occlusion_mask": volumes["physical_ray_volume_support"][:-1] | volumes["physical_ray_volume_support"][1:]}, {
        "frames": frames, "images": images, "grounds": grounds, "world": world,
        "plane": plane, "basis": basis, "anchor": anchor, "grid": grid,
        "projection": bev_projection(anchor, grid), "exposures": fits, "coordinates": coordinates,
        "observed_flow": observed, "expected_flow": expected, "occupied": occupied,
        "sample_tokens": [item["sample_token"] for item in observations],
        "receiver_sources": [{"camera_token": frame["sample_data_token"], "ground_sha256": frame["ground_sha256"],
            "ground_model_id": frame.get("ground_model_id"),
            "policy": "frame-centric" if frame.get("receiver_protocol") else "canonical_legacy_pair_reference"} for frame in frames]}


def caster_priors(chain, context, selected_track, directions, *, project=True):
    shape, frames = context["grid"].shape, context["frames"]
    states = {}
    for slot, pair in enumerate(chain):
        for obj in pair["objects"]:
            if obj["instance_token"] == selected_track:
                states[slot] = np.asarray(obj["landmarks_world_t"])
                states[slot+1] = np.asarray(obj["landmarks_world_next"])
    prior = np.zeros((len(frames), *shape), np.float32)
    banks, valid_banks = {}, {}
    for slot, points in states.items():
        if not project:
            continue
        hypotheses = np.zeros((len(directions), *shape), bool)
        valid = np.zeros(len(directions), bool)
        for light_id, direction in enumerate(directions):
            try:
                hypotheses[light_id] = footprint_mask(points[1:], context["plane"], direction,
                                                      context["projection"], shape)
                valid[light_id] = True
            except ValueError:
                continue
        banks[slot] = hypotheses
        valid_banks[slot] = valid
        prior[slot] = hypotheses.any(axis=0)
    context["selected_track_states"] = states
    context["caster_projection_bank"] = banks
    context["caster_projection_valid"] = valid_banks
    return prior


def panel(image, context, slot, maps, regions, proposed, accepted):
    def heat(value):
        canvas = cv2.applyColorMap((np.clip(np.nan_to_num(value), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas[~np.isfinite(value)] = 0
        return canvas
    overlay = image.copy()
    overlay[proposed] = (.55*overlay[proposed]+.45*np.array([0, 200, 240])).astype(np.uint8)
    overlay[accepted] = (.5*overlay[accepted]+.5*np.array([60, 240, 60])).astype(np.uint8)
    boxes = image.copy()
    for volume in context["frames"][slot]["object_volumes_world"]:
        uv, _, valid = project_world(volume["corners_world"], context["frames"][slot]["projection_world_to_image"])
        if valid.all():
            hull = cv2.convexHull(np.rint(np.clip(uv, -10000, 10000)).astype(np.int32))
            cv2.polylines(boxes, [hull], True, (80, 200, 80), 1)
    bev = heat(maps["proposal_score_map"][slot])
    for region in regions:
        cv2.drawContours(bev, cv2.findContours(region["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 255), 1)
    interval = min(slot, len(maps["flow_confidence_map"])-1)
    flow_panel = context["images"][interval].copy()
    support = maps["flow_confidence_map"][interval] > 0
    yy, xx = np.where(support)
    for chosen in np.linspace(0, max(len(xx)-1, 0), min(len(xx), 100), dtype=int):
        y, x = yy[chosen], xx[chosen]
        start = context["coordinates"][interval][y, x]
        for field, color in ((context["observed_flow"][interval], (0, 230, 230)),
                             (context["expected_flow"][interval], (230, 150, 60))):
            end = start+field[y, x]
            if np.isfinite(end).all():
                cv2.arrowedLine(flow_panel, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), color, 1)
    shadow_bev = np.zeros_like(bev)
    shadow_bev[context["occupied"][slot]] = (90, 90, 90)
    for region in regions:
        shadow_bev[region["mask"]] = (0, 170, 220)
    world = np.array([points[0] for _, points in sorted(context["selected_track_states"].items())])
    if len(world):
        trail, _, valid = project_world(world, context["projection"])
        if valid.sum() >= 2:
            cv2.polylines(shadow_bev, [np.rint(trail[valid]).astype(np.int32)], False, (255, 255, 255), 1)
    bev_change = maps["bev_feature_change"][interval]
    residual = maps["ground_motion_residual_px"][interval]
    delta = context["delta_L"][interval]
    projection = context.get("projection_review_mask", np.zeros_like(regions[0]["mask"]) if regions else np.zeros(bev.shape[:2], bool))
    tiles = [(image, "RGB"), (boxes, "3D GT projection"),
             (flow_panel, "flow: observed yellow / geometry blue"),
             (heat(.5+delta/24), "A: signed illumination change"),
             (heat(residual/6), "B: flow residual (not shadow speed)"),
             (heat(bev_change), "C: aligned BEV feature change"),
             (heat(np.where(maps["entity_explanation_valid_map"][interval], maps["entity_explanation_map"][interval], np.nan)), "D: entity motion explanation"),
             (bev, "E: actual discrepancy proposals"),
             (heat(projection.astype(float)), "box/light coverage before scoring"),
             (overlay, "proposal yellow / physics green"), (shadow_bev, "shadow BEV / GT vehicle trail"),
             (heat(maps["flow_confidence_map"][interval]), "flow confidence"),
             ]
    result = []
    for value, label in tiles:
        tile = np.zeros((256, 384, 3), np.uint8)
        scale = min(384/value.shape[1], 230/value.shape[0])
        resized = cv2.resize(value, (max(1, round(value.shape[1]*scale)), max(1, round(value.shape[0]*scale))))
        tile[26:26+resized.shape[0], :resized.shape[1]] = resized
        cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        result.append(tile)
    return np.vstack([np.hstack(result[start:start+3]) for start in range(0, len(result), 3)])


def run_clip(clip, chain, inputs, context, output, config, *, stage="full", bev_role="context", reviews=None):
    if stage not in {"signal", "projection", "full"}:
        raise ValueError("Video stage must be signal, projection or full")
    output.mkdir(parents=True, exist_ok=False)
    directions = light_bank()
    prior = caster_priors(chain, context, clip["track_id"], directions, project=stage != "signal")
    maps = video_proposals(**inputs, caster_prior=prior, config=config, bev_role=bev_role)
    context["delta_L"] = inputs["delta_L"]
    regions_by_frame, pool = [], []
    memory = ShadowRegionMemory(context["grid"].resolution, config)
    physical_config = LabelConfig(minimum_area=config.minimum_area, minimum_change_pixels=1,
                                 allow_static_existence=True, allow_partial_temporal=True)
    for slot, frame in enumerate(context["frames"]):
        # Memory can seed current weak observed evidence, never empty/unseen cells.
        remembered = np.zeros(context["grid"].shape, bool)
        for entry in memory.entries.values():
            if (frame["camera_timestamp"]-entry["timestamp"])/1e6 <= config.maximum_gap_seconds:
                remembered |= entry["mask"]
        regions = refine_regions(maps["proposal_score_map"][slot], maps["boundary_confidence"][slot],
            maps["receiver_valid_by_frame"][slot], inputs["features"][slot],
            maps["bev_feature_valid"][slot] & (bev_role != "off"), config, remembered)
        regions = memory.update(regions, frame["camera_timestamp"])
        polygon_cache = {}
        for number, region in enumerate(regions):
            region["id"] = f"{slot:03d}_{number:03d}"
            if stage == "full" and 0 < slot < len(chain):
                tracks, _ = window_tracks(chain[slot-1], chain[slot])
                rank_maps = {"visible": inputs["receiver"][slot], "dark": region["mask"],
                    "onset": maps["onset"][slot], "recovery": maps["recovery"][slot],
                    "flow_supported": maps["flow_confidence_map"][slot-1:slot+1].any(axis=0) > 0,
                    "geometric_correspondence": maps["pair_correspondence_valid"][slot-1:slot+1].any(axis=0),
                    "same_patch_cycle": maps["onset"][slot] & maps["recovery"][slot]}
                rankings, report = rank_hypotheses(region["mask"], rank_maps, tracks, context["plane"], context["basis"],
                    context["projection"], directions, physical_config, polygon_cache=polygon_cache)
                candidate = {"id": region["id"], "scene_name": clip["scene_name"],
                    "official_split": chain[0]["official_split"],
                    "frame_tokens": [value["sample_data_token"] for value in context["frames"][slot-1:slot+2]],
                    "sample_tokens": context["sample_tokens"][slot-1:slot+2], "hypotheses": rankings}
                region["physics_report"] = report
                region["candidate"] = candidate
                pool.append(candidate)
        regions_by_frame.append(regions)
    support_index = light_support_index(pool)
    rows, video, reviewed_rows = [], None, []
    owner_tokens = sorted({volume["instance_token"] for frame in context["frames"] for volume in frame["object_volumes_world"]})
    try:
        for slot, (frame, image, regions) in enumerate(zip(context["frames"], context["images"], regions_by_frame)):
            proposal = np.zeros(context["grid"].shape, bool)
            target = np.full(context["grid"].shape, 255, np.uint8)
            owner = np.full(context["grid"].shape, -1, np.int32)
            track_ids = np.zeros(context["grid"].shape, np.int32)
            confidence = np.zeros(context["grid"].shape, np.float32)
            moving = np.full(context["grid"].shape, -1, np.int8)
            association_status = np.zeros(context["grid"].shape, np.uint8)
            frame_rows = []
            for region in regions:
                mask = region["mask"]
                proposal |= mask
                track_ids[mask] = region["shadow_track_id"]
                decision = factorized_shadow_decision(region["candidate"], pool, directions, physical_config,
                    support_index=support_index) if "candidate" in region else {"shadow_accepted": False,
                        "association": None, "reason": "endpoint_missing_two_sided_physics" if stage == "full" else "physics_not_run"}
                if decision["shadow_accepted"]:
                    hypothesis = decision["selected"]
                    selected_volume = next(volume for volume in frame["object_volumes_world"]
                                           if volume["instance_token"] == hypothesis["instance_token"])
                    supported_mask = mask & footprint_mask(selected_volume["corners_world"], context["plane"],
                        directions[hypothesis["light_id"]], context["projection"], mask.shape)
                    target[supported_mask] = 1
                    confidence[supported_mask] = region["rank_score"]
                    moving[supported_mask] = int(hypothesis["motion_supported"])
                    association_status[supported_mask] = 1
                    if decision["association"] is not None:
                        owner[supported_mask] = owner_tokens.index(decision["association"]["instance_token"])
                        association_status[supported_mask] = 2
                frame_rows.append({"id": region["id"], "shadow_track_id": region["shadow_track_id"],
                    "ranking_score": region["rank_score"], "temporal_consistency": region["temporal_consistency"],
                    "area_cells": int(mask.sum()), "source_bits": np.unique(maps["proposal_source_bits"][slot][mask]).tolist(),
                    "complete_discrepancy_cells": int((mask & maps["discrepancy_evidence_complete"][slot]).sum()),
                    "partial_evidence_cells": int((mask & maps["partial_evidence_candidate"][slot]).sum()),
                    "physics": decision})
            bank = context["caster_projection_bank"].get(slot, np.zeros((0, *target.shape), bool))
            bank_valid = context["caster_projection_valid"].get(slot, np.ones(len(bank), bool))
            coverage = projection_coverage(proposal, bank, valid=bank_valid)
            coverage["reference_kind"] = "observed_proposal_not_shadow_GT"
            best_id = coverage["best_cover_light_id"]
            best_projection = bank[best_id] if best_id is not None else np.zeros_like(proposal)
            context["projection_review_mask"] = best_projection
            for review in (reviews or {}).get(frame["sample_data_token"], []):
                review_mask = review["mask"]
                if review_mask.shape != image.shape[:2]:
                    raise ValueError("Independent review mask must use the calibrated image dimensions")
                receiver_mask = image_mask_to_bev(review_mask, frame["projection_world_to_image"], context["world"])
                reviewed_rows.append({"frame_token": frame["sample_data_token"], "region_type": review["region_type"],
                    "reference_kind": "independent_review_only_not_training",
                    **summarize_signals(inputs, maps, slot, receiver_mask),
                    **reviewed_projection_coverage(receiver_mask, bank, best_id, valid=bank_valid)})
            positive = target == 1
            to_image = lambda mask: bev_mask_to_image(mask, frame["projection_world_to_image"], context["basis"], context["grid"], image.shape[:2]) & context["grounds"][slot]
            image_proposal, image_positive = to_image(proposal), to_image(positive)
            image_target = np.full(image.shape[:2], 255, np.uint8)
            image_target[image_positive] = 1
            path = output/f"{slot:03d}.npz"
            np.savez_compressed(path, shadow_receiver_mask_bev=positive, shadow_proposal_bev=proposal,
                shadow_target_bev=target, shadow_owner_id=owner, shadow_confidence=confidence,
                static_or_moving_shadow=moving, shadow_track_id=track_ids,
                resolved_or_unresolved=association_status,
                receiver_world_xyz=context["world"], bev_world_to_grid=context["projection"],
                best_cover_projection_bev=best_projection,
                projection_union_bev=bank.any(axis=0) if len(bank) else np.zeros_like(proposal),
                **{key: value[slot] for key, value in maps.items() if value.shape[0] == len(context["frames"])})
            checked_write_image(output/f"{slot:03d}_target.png", image_target)
            checked_write_image(output/f"{slot:03d}_proposal.png", image_proposal.astype(np.uint8)*255)
            mosaic = panel(image, context, slot, maps, regions, image_proposal, image_positive)
            checked_write_image(output/f"{slot:03d}_review.jpg", mosaic)
            if video is None:
                video = cv2.VideoWriter(str(output/"review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2,
                                        (mosaic.shape[1], mosaic.shape[0]))
                if not video.isOpened():
                    raise RuntimeError("OpenCV MP4 encoder could not open review.mp4")
            video.write(mosaic)
            if len(bank):
                np.savez_compressed(output/f"{slot:03d}_light_bank.npz", projected_box_masks=bank,
                                    directions_world=directions, projection_valid=bank_valid)
            rows.append({"frame_token": frame["sample_data_token"], "timestamp": frame["camera_timestamp"],
                "evidence": str(path), "regions": frame_rows, "positive_cells": int(positive.sum()),
                "projection_capacity": coverage,
                "selected_direction": {"light_id": best_id, "selection_source": "observed_proposal_best_cover_without_review_masks"},
                "requested_track_id": clip["track_id"], "requested_track_is_not_owner_gt": True})
    finally:
        if video is not None:
            video.release()
    np.savez_compressed(output/"pair_evidence.npz", delta_L=inputs["delta_L"],
        observed_flow_px=inputs["observed_flow"], expected_flow_px=inputs["expected_flow"],
        **{key: value for key, value in maps.items() if value.shape[0] == len(chain)})
    write_rows(output/"frames.jsonl", rows)
    write_rows(output/"reviewed_signals.jsonl", reviewed_rows)
    write_json(output/"protocol.json", {"protocol": "track-centric-appearance-geometry-video-v3", "clip": clip,
        "stage": stage, "bev_role": bev_role,
        "discrepancy_definition": "registered illumination change * (1-entity explanation) only where entity coverage is valid",
        "entity_state_legend": ENTITY_STATES,
        "partial_evidence_contract": "missing entity coverage retains appearance candidates, not evidence of entity exclusion; incomplete discrepancy is NaN",
        "projection_review_contract": "oracle capacity optimizes on reference; selected direction is frozen on proposals before reading review masks",
        "bev_role_contract": "context=descriptors/linking only; stability=explicit invariance hypothesis; off=no BEV linking",
        "discovery_config": asdict(config), "physics_config": asdict(physical_config),
        "owner_index_to_instance_token": owner_tokens, "receiver_sources": context["receiver_sources"],
        "exposures": context["exposures"], "grid": asdict(context["grid"]),
        "frame_intervals_seconds": (np.diff([frame["camera_timestamp"] for frame in context["frames"]])/1e6).tolist(),
        "ground_plane_world": context["plane"].tolist(),
        "coordinate_frame": "one fixed world receiver grid per clip; pairwise visibility, not temporal intersection",
        "scores_are_probabilities": False, "target_encoding": "1=weak shadow,255=IGNORE; no inferred negatives",
        "motion_encoding": "-1=unknown,0=static,1=moving", "owner_unknown": -1,
        "association_encoding": "0=unknown,1=unresolved,2=resolved",
        "bev_semantics": "feature consistency, not a learned free-ground classifier",
        "flow_occlusion": "FB inconsistency is not an occlusion label; occlusion cause unknown",
        "training_performed": False, "is_shadow_gt": False})
    return {"id": clip["id"], "frames": len(rows), "regions": sum(len(row["regions"]) for row in rows),
            "positive_cells": sum(row["positive_cells"] for row in rows), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--clips", required=True, help="JSONL: id,scene_name,camera,start_sample_token,track_id,num_frames")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--extent-m", type=float, default=40)
    parser.add_argument("--resolution-m", type=float, default=.5)
    parser.add_argument("--seed-score", type=float, default=.35)
    parser.add_argument("--grow-score", type=float, default=.18)
    parser.add_argument("--stage", choices=["signal", "projection", "full"], default="full")
    parser.add_argument("--bev-role", choices=["context", "stability", "off"], default="context")
    parser.add_argument("--review-regions", help="Optional independent image masks: clip_id,frame_token,region_type,mask")
    args = parser.parse_args()
    config = VideoShadowConfig(seed_score=args.seed_score, grow_score=args.grow_score)
    config.validate()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    pairs, stats, _ = load_motion_analysis(args.pairs, args.analysis_root)
    index = frame_index(pairs)
    clips = read_rows(args.clips)
    if not clips or len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("Choose nonempty uniquely identified clips")
    summaries = []
    reviews = read_rows(args.review_regions) if args.review_regions else []
    if not {row["clip_id"] for row in reviews} <= {clip["id"] for clip in clips}:
        raise ValueError("Independent review refers to a clip outside this run")
    review_base = Path(args.review_regions).resolve().parent if args.review_regions else None
    for clip in tqdm(clips, desc="track-centric shadow video", unit="clip"):
        if Path(clip["id"]).name != clip["id"] or clip["id"] in {".", ".."}:
            raise ValueError("Clip id must be a filename")
        chain = select_chain(pairs, clip)
        for pair in chain:
            check_pair_files(pair, Path(args.pairs).resolve().parent)
        inputs, context = sample_clip(chain, index, Path(args.pairs).resolve().parent,
            stats, Path(args.analysis_root).resolve(), args.feature_root,
            BEVFlowConfig(extent_m=args.extent_m, resolution_m=args.resolution_m))
        clip_reviews = {}
        for review in reviews:
            if review["clip_id"] == clip["id"]:
                if review["region_type"] not in {"shadow", "stable_ground", "material_boundary", "entity"}:
                    raise ValueError("Unknown independent review region type")
                clip_reviews.setdefault(review["frame_token"], []).append({**review,
                    "mask": load_array(review["mask"], review_base) > 0})
        known_frames = {frame["sample_data_token"] for frame in context["frames"]}
        if not set(clip_reviews) <= known_frames:
            raise ValueError("Reviewed frame does not belong to the selected clip")
        summary = run_clip(clip, chain, inputs, context, output/clip["id"], config,
                           stage=args.stage, bev_role=args.bev_role, reviews=clip_reviews)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(output/"summary.json", {"clips": summaries, "pair_manifest_sha256": digest(args.pairs),
        "clip_manifest_sha256": digest(args.clips), "stage": args.stage, "bev_role": args.bev_role,
        "review_manifest_sha256": digest(args.review_regions) if args.review_regions else None,
        "training_performed": False})


if __name__ == "__main__":
    main()
