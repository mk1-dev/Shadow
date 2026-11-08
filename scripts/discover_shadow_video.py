"""Track-centric BEV/flow discrepancy discovery using exported raw observations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_image_cell_indices, bev_mask_to_image, bev_projection, ground_grid, image_mask_to_bev, rasterize_volumes
from shadow_occ.annotation.cached_bev import sample_cached_bev
from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.annotation.frame_receiver import frame_index
from shadow_occ.annotation.entity_memory import CasterMemory, EntityMotionConfig, OwnerScoringConfig, match_entity_motion
from shadow_occ.annotation.physics_io import check_pair_files, load_motion_analysis, read_evidence
from shadow_occ.annotation.physics_labels import (
    LabelConfig, factorized_shadow_decision, footprint_mask, light_bank,
    light_support_index, rank_hypotheses,
)
from shadow_occ.annotation.physics_motion import pair_evidence, project_world, sample_at
from shadow_occ.annotation.receiver_explanations import box_ray_evidence
from shadow_occ.annotation.shadow_video import ENTITY_STATES, PHYSICAL_REFERENCE_SOURCES, ShadowRegionMemory, VideoShadowConfig, refine_regions, video_proposals
from shadow_occ.annotation.shadow_projection import projection_coverage, reviewed_projection_coverage, summarize_signals
from shadow_occ.annotation.video_correspondence import (CorrespondenceConfig, CrossSpaceGrowthConfig,
    SPATIAL_MATCH_STATES, cross_space_proposals, grow_cross_space_regions, lift_flow_to_bev)
from shadow_occ.annotation.shadow_representation import describe_regions, memory_state, paste_region_image, project_region_memory, refine_image_region
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


def sample_clip(chain, index, base, stats, analysis, feature_root, config, *,
                entity_config=None, entity_suppression=True, discovery_domain="receiver", spatial_feature_root=None):
    """One fixed world grid; per-frame masks and independent valid frame pairs."""
    if discovery_domain not in {"receiver", "image-rays"}:
        raise ValueError("Discovery domain must be receiver or image-rays")
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
    images, grounds, coordinates, visible, labs, occupied, features, feature_valid, in_image = ([] for _ in range(9))
    spatial_features, spatial_valid, spatial_sources = [], [], []
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
        feature, supported, feature_source = sample_cached_bev(feature_root, item["sample_token"], world)
        spatial, spatial_supported, spatial_source = feature, supported, feature_source
        if spatial_feature_root is not None and Path(spatial_feature_root).resolve() != Path(feature_root).resolve():
            spatial, spatial_supported, spatial_source = sample_cached_bev(spatial_feature_root, item["sample_token"], world)
        if spatial_feature_root is not None and spatial_source.get("history_mode") != "current-only":
            raise ValueError("Spatial discovery requires --history-mode current-only BEV exports")
        if spatial_source.get("history_mode") != "current-only":
            spatial, spatial_supported = np.zeros_like(feature), np.zeros(supported.shape, bool)
        for key in ("checkpoint_sha256", "model_config_sha256"):
            if feature_source.get(key) is not None and spatial_source.get(key) is not None and feature_source[key] != spatial_source[key]:
                raise ValueError("Spatial and temporal BEV caches use different mature models")
        spatial_features.append(spatial)
        spatial_valid.append(spatial_supported & support)
        spatial_sources.append({"sample_token": item["sample_token"], "history_mode": spatial_source.get("history_mode"),
            "available_for_discovery": spatial_source.get("history_mode") == "current-only"})
        images.append(image)
        grounds.append(ground)
        coordinates.append(pixels)
        visible.append(support)
        in_image.append(inside.reshape(grid.shape))
        labs.append(sample_at(lab, pixels))
        occupied.append(rasterize_volumes(frame, anchor, grid)[0])
        features.append(feature)
        feature_valid.append(supported & support)
    visible, occupied = np.stack(visible), np.stack(occupied)
    observed, expected, fb, texture, deltas, comparable, fits, fb_error, ground_reference = ([] for _ in range(9))
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
        ground_reference.append(sample_at(fresh["receiver_flow_supported"].astype(np.float32), coordinates[slot]) >= .999)
        fit = fresh["details"]["exposure"]
        fits.append(fit)
        comparable.append(bool(fit["valid"]))
        deltas.append(fit["scale"]*labs[slot+1][..., 0]+fit["offset_L"]-labs[slot][..., 0])
    observed, expected, fb = np.stack(observed), np.stack(expected), np.stack(fb)
    volumes = box_ray_evidence(world, frames, coordinates, observed, fb)
    entity_config = entity_config or EntityMotionConfig()
    entity_maps = match_entity_motion(volumes, fb, np.stack(fb_error), entity_config, enabled=entity_suppression)
    caster_memory = CasterMemory(chain, frames)
    for slot in range(len(frames)):
        matched = entity_maps["entity_motion_matched_by_interval"][slot] if slot < len(chain) else np.zeros(grid.shape, bool)
        caster_memory.observe(slot, labs[slot], volumes["box_surface_owner_index"][slot], matched,
            feature_root, observations[slot]["sample_token"], sample_cached_bev)
    receiver = visible & ~occupied & ~volumes["physical_ray_volume_support"]
    metric_flow = lift_flow_to_bev(world, frames, coordinates, observed, fb, receiver, basis,
                                  bev_projection(anchor, grid), grid.resolution)
    entity_good = volumes["box_surface_flow_valid"] & fb
    ground_error = np.linalg.norm(observed-expected, axis=-1)
    entity_error = volumes["box_surface_flow_residual_px"]
    entity_fit = np.where(entity_good, np.exp(-entity_error/3)*np.clip((ground_error-entity_error)/3, 0, 1), 0)
    return {"lab": np.stack(labs), "receiver": receiver, "delta_L": np.stack(deltas),
        "exposure_valid": np.asarray(comparable)[:, None, None], "observed_flow": observed,
        "expected_flow": expected, "fb_consistent": fb, "texture_support": np.stack(texture),
        "features": np.stack(features), "feature_valid": np.stack(feature_valid),
        "entity_explanation": entity_fit.astype(np.float32), "entity_explanation_valid": entity_good,
        "ground_motion_reference_valid": np.stack(ground_reference),
        "entity_suppression_mask": entity_maps["entity_suppression_mask"],
        "entity_occlusion_mask": entity_maps["entity_occlusion_mask"],
        "search_domain": (receiver | (np.stack(in_image) & volumes["physical_ray_volume_support"])) if discovery_domain == "image-rays" else None,
        "fb_error_px": np.stack(fb_error),
        "occlusion_mask": volumes["physical_ray_volume_support"][:-1] | volumes["physical_ray_volume_support"][1:]}, {
        "frames": frames, "images": images, "grounds": grounds, "world": world,
        "plane": plane, "basis": basis, "anchor": anchor, "grid": grid,
        "projection": bev_projection(anchor, grid), "exposures": fits, "coordinates": coordinates,
        "entity_maps": entity_maps, "caster_memory": caster_memory,
        "entity_motion_config": asdict(entity_config), "entity_suppression_enabled": entity_suppression,
        "discovery_domain": discovery_domain,
        "spatial_feature_sources": spatial_sources,
        "spatial_features": np.stack(spatial_features), "spatial_feature_valid": np.stack(spatial_valid),
        "metric_flow": metric_flow["flow_bev_displacement_m"], "metric_flow_valid": metric_flow["flow_bev_valid"],
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
    if "caster_memory" in context:
        states = context["caster_memory"].states(selected_track)
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
    for y, x in np.argwhere(maps["entity_suppression_mask"][slot]):
        point = context["coordinates"][slot][y, x]
        if np.isfinite(point).all() and (point >= 0).all():
            cv2.circle(boxes, tuple(np.rint(point).astype(int)), 2, (255, 240, 0), -1)
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
    spatial_panel = "spatial_bev_feature_change" in maps
    bev_change = maps["spatial_bev_feature_change"][interval]/2 if spatial_panel else maps["bev_feature_change"][interval]
    metric_panel = "flow_bev_discrepancy_m" in maps
    residual = maps["flow_bev_discrepancy_m"][interval]/.75 if metric_panel else maps["ground_motion_residual_px"][interval]/6
    delta = np.where(maps["exposure_valid"][interval], context["delta_L"][interval], np.nan)
    projection = context.get("projection_review_mask", np.zeros_like(regions[0]["mask"]) if regions else np.zeros(bev.shape[:2], bool))
    tiles = [(image, "RGB"), (boxes, "3D GT / matched cores cyan"),
             (flow_panel, "flow: observed yellow / geometry blue"),
             (heat(.5+delta/24), "A: signed illumination change"),
             (heat(residual), "B: metric Flow - BEV" if metric_panel else "B: image flow residual"),
             (heat(bev_change), "C: current-only BEV change" if spatial_panel else "C: temporal BEV context change"),
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


def run_clip(clip, chain, inputs, context, output, config, *, stage="full", bev_role="context", reviews=None,
             owner_scoring="physical", owner_config=None, discovery_mode="cross-space"):
    if stage not in {"signal", "projection", "full"}:
        raise ValueError("Video stage must be signal, projection or full")
    if owner_scoring not in {"physical", "geometry", "geometry+appearance+bev"}:
        raise ValueError("Unknown owner scoring mode")
    if discovery_mode not in {"cross-space", "appearance"}:
        raise ValueError("Unknown video discovery mode")
    owner_config = owner_config or OwnerScoringConfig()
    owner_config.validate()
    started = perf_counter()
    timing = {"proposal_seconds": 0., "region_seconds": 0., "physics_seconds": 0., "owner_seconds": 0.}
    output.mkdir(parents=True, exist_ok=False)
    if "caster_memory" not in context:
        context["caster_memory"] = CasterMemory(chain, context["frames"])
    caster_memory = context["caster_memory"]
    directions = light_bank()
    prior = caster_priors(chain, context, clip["track_id"], directions, project=stage != "signal")
    tick = perf_counter()
    maps = video_proposals(**inputs, caster_prior=prior, config=config, bev_role=bev_role)
    correspondence_config = CorrespondenceConfig()
    growth_config = CrossSpaceGrowthConfig()
    if discovery_mode == "cross-space" and "spatial_features" in context:
        original = maps["proposal_score_map"].copy()
        maps = cross_space_proposals(maps, context["spatial_features"], context["spatial_feature_valid"],
            context["metric_flow"], context["metric_flow_valid"], context["grid"].resolution, correspondence_config)
        maps = grow_cross_space_regions(maps, inputs["lab"], context["grid"].resolution, config,
            timestamps=[frame["camera_timestamp"] for frame in context["frames"]], config=growth_config)
        maps["proposal_score_map"] = np.maximum(original,
            np.maximum(maps["cross_space_discovery_score"], maps["cross_space_region_score"]))
        maps["cross_space_new_seed"] = (maps["cross_space_discovery_score"] >= config.seed_score) & (original < config.seed_score)
        maps["proposal_source_bits"][(maps["cross_space_discovery_score"] >= config.seed_score)
                                     | maps["cross_space_region_support"]] |= 8
    timing["proposal_seconds"] = perf_counter()-tick
    maps.update(context.get("entity_maps", {}))
    context["delta_L"] = inputs["delta_L"]
    regions_by_frame, pool, region_memory_rows = [], [], []
    image_cell_maps = []
    memory = ShadowRegionMemory(context["grid"].resolution, config)
    physical_config = LabelConfig(minimum_area=config.minimum_area, minimum_change_pixels=1,
                                 allow_static_existence=True, allow_partial_temporal=True)
    for slot, frame in enumerate(context["frames"]):
        # Memory can seed current weak observed evidence, never empty/unseen cells.
        remembered = np.zeros(context["grid"].shape, bool)
        projection_masks = project_region_memory(memory.entries, caster_memory, slot, context["plane"], directions, context["projection"])
        for identity, entry in memory.entries.items():
            if ((frame["camera_timestamp"]-entry["timestamp"])/1e6 <= config.maximum_gap_seconds
                    and entry["missed"] < config.memory_frames):
                remembered |= entry["mask"]
                if identity in projection_masks:
                    remembered |= projection_masks[identity]
        tick = perf_counter()
        if "cross_space_temporal_seed" in maps:
            remembered |= maps["cross_space_temporal_seed"][slot]
        regions = refine_regions(maps["proposal_score_map"][slot], maps["boundary_confidence"][slot],
            maps["search_valid_by_frame"][slot], inputs["features"][slot],
            maps["bev_feature_valid"][slot] & (bev_role != "off"), config, remembered)
        flow_slot = min(slot, len(chain)-1)
        flow = context.get("metric_flow")
        flow_valid = context.get("metric_flow_valid")
        describe_regions(regions, inputs["lab"][slot], maps["boundary_confidence"][slot],
                         flow[flow_slot] if flow is not None and slot < len(chain) else None,
                         flow_valid[flow_slot] if flow_valid is not None and slot < len(chain) else None)
        image_shape = context["images"][slot].shape[:2]
        cell_indices = bev_image_cell_indices(frame["projection_world_to_image"], context["basis"], context["grid"], image_shape)
        image_cell_maps.append(cell_indices)
        image_domain = bev_mask_to_image(maps["receiver_valid_by_frame"][slot], frame["projection_world_to_image"],
            context["basis"], context["grid"], image_shape, cell_indices=cell_indices) & context["grounds"][slot]
        available_tracks = caster_memory.tracks(slot) if 0 < slot < len(chain) else []
        for number, region in enumerate(regions):
            region["id"] = f"{slot:03d}_{number:03d}"
            region["frame_slot"] = slot
            region["receiver_mask"] = region["mask"] & maps["receiver_valid_by_frame"][slot]
            if discovery_mode == "cross-space":
                appearance_support = maps["appearance_score_map"][slot]
                if "cross_space_region_appearance_support" in maps:
                    appearance_support = np.maximum(appearance_support, maps["cross_space_region_appearance_support"][slot])
                region["receiver_mask"] &= appearance_support >= config.minimum_observed_appearance
            seed = bev_mask_to_image(region["receiver_mask"], frame["projection_world_to_image"],
                context["basis"], context["grid"], image_domain.shape, cell_indices=cell_indices)
            region.update(refine_image_region(context["images"][slot], seed, image_domain))
            center = np.median(context["world"][region["mask"]], axis=0)
            region["caster_candidates"] = [track["instance_token"] for track in available_tracks
                if np.linalg.norm(track["points"][1, 0, :2]-center[:2]) <= physical_config.maximum_caster_distance_m]
        regions = memory.update(regions, frame["camera_timestamp"], receiver=maps["receiver_valid_by_frame"][slot],
                                occlusion=maps["entity_occlusion_mask"][slot], projection_masks=projection_masks)
        timing["region_seconds"] += perf_counter()-tick
        polygon_cache = {}
        for number, region in enumerate(regions):
            if stage == "full" and 0 < slot < len(chain):
                tick = perf_counter()
                tracks = caster_memory.tracks(slot)
                rank_maps = {"visible": maps["receiver_valid_by_frame"][slot], "dark": region["receiver_mask"],
                    "onset": maps["onset"][slot], "recovery": maps["recovery"][slot],
                    "flow_supported": maps["flow_confidence_map"][slot-1:slot+1].any(axis=0) > 0,
                    "geometric_correspondence": maps["pair_correspondence_valid"][slot-1:slot+1].any(axis=0),
                    "same_patch_cycle": maps["onset"][slot] & maps["recovery"][slot]}
                rankings, report = rank_hypotheses(region["receiver_mask"], rank_maps, tracks, context["plane"], context["basis"],
                    context["projection"], directions, physical_config, polygon_cache=polygon_cache)
                region["caster_memory_candidates"] = [caster_memory.association_context(track["instance_token"], slot)
                                                       for track in tracks]
                candidate = {"id": region["id"], "scene_name": clip["scene_name"],
                    "frame_slot": slot, "shadow_track_id": region["shadow_track_id"],
                    "observed_transition_pixels": report.get("onset", 0)+report.get("recovery", 0),
                    "official_split": chain[0]["official_split"],
                    "frame_tokens": [value["sample_data_token"] for value in context["frames"][slot-1:slot+2]],
                    "sample_tokens": context["sample_tokens"][slot-1:slot+2], "hypotheses": rankings}
                region["physics_report"] = report
                region["candidate"] = candidate
                memory.remember_physics(region, rankings)
                caster_memory.remember_shadow_projection(region["shadow_track_id"], slot, rankings)
                pool.append(candidate)
                timing["physics_seconds"] += perf_counter()-tick
        regions_by_frame.append(regions)
        region_memory_rows.append({"frame_token": frame["sample_data_token"], "timestamp": frame["camera_timestamp"],
            "regions": [memory_state(entry, identity) for identity, entry in memory.entries.items()]})
    support_index = light_support_index(pool)
    rows, video, reviewed_rows = [], None, []
    mask_hashes = {name: hashlib.sha256() for name in ("proposal", "target", "owner")}
    owner_tokens = sorted(caster_memory.records)
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
            image_regions = np.zeros(image.shape[:2], np.int32)
            for region in regions:
                mask = region["mask"]
                assigned_owner_cells = 0
                proposal |= mask
                paste_region_image(region, image_regions, region["shadow_track_id"])
                track_ids[mask] = region["shadow_track_id"]
                tick = perf_counter()
                decision = factorized_shadow_decision(region["candidate"], pool, directions, physical_config,
                    support_index=support_index) if "candidate" in region else {"shadow_accepted": False,
                        "association": None, "reason": "endpoint_missing_two_sided_physics" if stage == "full" else "physics_not_run"}
                timing["physics_seconds"] += perf_counter()-tick
                if decision["shadow_accepted"] and owner_scoring != "physical":
                    tick = perf_counter()
                    association, evidence = caster_memory.score_owner(region["candidate"], pool, owner_scoring, owner_config)
                    decision = {**decision, "physical_association": decision["association"], "association": association,
                        "association_reason": evidence["reason"], "memory_owner_scoring": evidence}
                    used = {item["instance_token"]: any(value["pairs"] > 0 for value in item["descriptor_consistency"].values())
                            for item in evidence["ranked_owners"]}
                    for item in region["caster_memory_candidates"]:
                        item["descriptors_used_for_owner_score"] = used.get(item["caster_memory_id"], False)
                    timing["owner_seconds"] += perf_counter()-tick
                if decision["shadow_accepted"]:
                    hypothesis = decision["selected"]
                    caster_state = caster_memory.states(hypothesis["instance_token"])[slot]
                    supported_mask = region["receiver_mask"] & footprint_mask(caster_state[1:], context["plane"],
                        directions[hypothesis["light_id"]], context["projection"], mask.shape)
                    target[supported_mask] = 1
                    confidence[supported_mask] = region["rank_score"]
                    moving[supported_mask] = int(hypothesis["motion_supported"])
                    association_status[supported_mask] = 1
                    if decision["association"] is not None:
                        association = decision["association"]
                        owner_state = caster_memory.states(association["instance_token"])[slot]
                        owner_mask = supported_mask & footprint_mask(owner_state[1:], context["plane"],
                            directions[association["light_id"]], context["projection"], mask.shape)
                        owner[owner_mask] = owner_tokens.index(association["instance_token"])
                        association_status[owner_mask] = 2
                        assigned_owner_cells = int(owner_mask.sum())
                frame_rows.append({"id": region["id"], "shadow_track_id": region["shadow_track_id"],
                    "ranking_score": region["rank_score"], "temporal_consistency": region["temporal_consistency"],
                    "area_cells": int(mask.sum()), "source_bits": np.unique(maps["proposal_source_bits"][slot][mask]).tolist(),
                    "receiver_area_cells": int(region["receiver_mask"].sum()),
                    "assigned_owner_cells": assigned_owner_cells,
                    "image_area_pixels": int(region["image_mask_roi"].sum()),
                    "image_mask_box_xyxy": region["image_mask_box_xyxy"],
                    "visibility_status": region["visibility_status"],
                    "boundary_descriptor": region["boundary_descriptor"].tolist(),
                    "flow_correspondence": region.get("flow_correspondence"),
                    "caster_candidates": region["caster_candidates"],
                    "complete_discrepancy_cells": int((mask & maps["discrepancy_evidence_complete"][slot]).sum()),
                    "partial_evidence_cells": int((mask & maps["partial_evidence_candidate"][slot]).sum()),
                    "caster_memory_candidates": region.get("caster_memory_candidates", []),
                    "physics": decision})
            refined_image_proposal = image_regions > 0
            continuous_bev = image_mask_to_bev(refined_image_proposal, frame["projection_world_to_image"], context["world"])
            continuous_bev &= maps["receiver_valid_by_frame"][slot]
            rejected_by_image = (target == 1) & ~continuous_bev
            target[rejected_by_image] = 255
            owner[rejected_by_image] = -1
            confidence[rejected_by_image] = 0
            moving[rejected_by_image] = -1
            association_status[rejected_by_image] = 0
            for row, region in zip(frame_rows, regions):
                row["assigned_owner_cells"] = int(((owner >= 0) & region["mask"]).sum())
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
                before_search = inputs.get("search_domain")
                before_search = inputs["receiver"][slot] if before_search is None else before_search[slot]
                eligible_review = receiver_mask & before_search
                reviewed_rows.append({"frame_token": frame["sample_data_token"], "region_type": review["region_type"],
                    "reference_kind": "independent_review_only_not_training",
                    "search_preservation": {"review_cells_on_grid": int(receiver_mask.sum()),
                        "eligible_review_cells": int(eligible_review.sum()),
                        "retained_review_cells": int((eligible_review & maps["search_valid_by_frame"][slot]).sum()),
                        "candidate_review_cells": int((eligible_review & proposal).sum()),
                        "positive_review_cells": int((eligible_review & (target == 1)).sum()),
                        "coordinate_domain": "independent image mask sampled on receiver-ray grid"},
                    **summarize_signals(inputs, maps, slot, receiver_mask),
                    **reviewed_projection_coverage(receiver_mask, bank, best_id, valid=bank_valid)})
            positive = target == 1
            for name, value in (("proposal", proposal), ("target", target), ("owner", owner)):
                mask_hashes[name].update(np.ascontiguousarray(value).tobytes())
            to_image = lambda mask: bev_mask_to_image(mask, frame["projection_world_to_image"], context["basis"], context["grid"], image.shape[:2], cell_indices=image_cell_maps[slot])
            image_proposal = refined_image_proposal
            image_positive = to_image(positive) & refined_image_proposal & context["grounds"][slot]
            image_target = np.full(image.shape[:2], 255, np.uint8)
            image_target[image_positive] = 1
            path = output/f"{slot:03d}.npz"
            np.savez_compressed(path, shadow_receiver_mask_bev=positive, shadow_proposal_bev=proposal,
                shadow_continuous_proposal_bev=continuous_bev,
                shadow_target_bev=target, shadow_owner_id=owner, shadow_confidence=confidence,
                static_or_moving_shadow=moving, shadow_track_id=track_ids,
                resolved_or_unresolved=association_status,
                receiver_world_xyz=context["world"], bev_world_to_grid=context["projection"],
                best_cover_projection_bev=best_projection,
                projection_union_bev=bank.any(axis=0) if len(bank) else np.zeros_like(proposal),
                **{key: value[slot] for key, value in maps.items() if value.shape[0] == len(context["frames"])})
            image_path = output/f"{slot:03d}_image.npz"
            np.savez_compressed(image_path, shadow_proposal_image=refined_image_proposal,
                shadow_target_image=image_target, shadow_region_id_image=image_regions,
                coordinate_frame="calibrated camera pixels; RGB-refined observed regions")
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
                "image_representation": str(image_path), "shadow_bev_is_separate_from_caster_geometry": True,
                "candidate_cells": int(proposal.sum()),
                "receiver_candidate_cells": int((proposal & maps["receiver_valid_by_frame"][slot]).sum()),
                "entity_candidate_cells": int((proposal & maps["entity_occlusion_mask"][slot]).sum()),
                "search_cells": int(maps["search_valid_by_frame"][slot].sum()),
                "owner_cells": int((owner >= 0).sum()),
                "unique_caster_regions": sum(region["assigned_owner_cells"] > 0 for region in frame_rows),
                "entity_suppressed_cells": int(maps["entity_suppression_mask"][slot].sum()),
                "entity_occluded_cells": int(maps["entity_occlusion_mask"][slot].sum()),
                "additional_search_removed_cells": int(maps["entity_search_removed_mask"][slot].sum()),
                "entity_surface_index_to_instance_token": [volume["instance_token"] for volume in frame["object_volumes_world"]],
                "projection_capacity": coverage,
                "selected_direction": {"light_id": best_id, "status": coverage["status"],
                    "selection_source": "observed_proposal_best_cover_without_review_masks"},
                "requested_track_id": clip["track_id"], "requested_track_is_not_owner_gt": True})
    finally:
        if video is not None:
            video.release()
    np.savez_compressed(output/"pair_evidence.npz", delta_L=inputs["delta_L"],
        observed_flow_px=inputs["observed_flow"], expected_flow_px=inputs["expected_flow"],
        **{key: value for key, value in maps.items() if value.shape[0] == len(chain)})
    write_rows(output/"frames.jsonl", rows)
    write_rows(output/"caster_memory.jsonl", caster_memory.rows())
    decisions = {region["id"]: region for row in rows for region in row["regions"]}
    for row in region_memory_rows:
        for state in row["regions"]:
            observed_region = decisions.get(state["last_seen_region_id"])
            if observed_region is not None:
                decision = observed_region["physics"]
                state["existence_score"] = decision.get("existence_score")
                state["owner_status"] = ("resolved" if observed_region["assigned_owner_cells"] else
                                         "unresolved" if decision.get("shadow_accepted") else "unknown")
                state["owner_score"] = (decision.get("memory_owner_scoring", {}).get("ranked_owners") or [{}])[0].get("score")
    write_rows(output/"region_memory.jsonl", region_memory_rows)
    write_rows(output/"reviewed_signals.jsonl", reviewed_rows)
    timing["total_seconds"] = perf_counter()-started
    write_json(output/"protocol.json", {"protocol": "entity-shadow-disentangled-video-v8", "clip": clip,
        "stage": stage, "bev_role": bev_role,
        "modules": ["M1_entity_decomposition", "M2_metric_BEV_flow_discovery", "M3_caster_light_reasoning", "M4_temporal_shadow_representation"],
        "discovery_mode": discovery_mode, "correspondence_config": asdict(correspondence_config),
        "cross_space_growth_config": asdict(growth_config), "spatial_match_states": SPATIAL_MATCH_STATES,
        "spatial_match_contract": "reliable retains cosine/uniqueness/cycle gates; low confidence contributes a weighted proposal, not reliable geometry; repeated features have zero motion weight",
        "cross_space_growth_contract": "bounded Lab-anchor and boundary constrained growth on currently observed ground; uniform interiors use measured seed-color agreement as appearance support; one-hop same-world temporal seeds require independent valid pairs and current appearance; no flow-as-shadow-speed or masks in missing frames",
        "spatial_feature_sources": context.get("spatial_feature_sources", []),
        "feature_roles": "current-only spatial features for BEV correspondence; temporal/context features for region identity; absent spatial cache is missing evidence",
        "flow_BEV_units": "horizontal anchor-grid x/y meters per actual camera interval; ground-ray lift in destination camera",
        "proposal_source_bits": {"1": "appearance", "2": "ground-referenced illumination", "4": "caster-guided appearance", "8": "metric flow-BEV cross-space proposal"},
        "temporal_representation": "RGB-refined observed masks + boundary/Lab/BEV/material correspondence + alternative caster/light motion; missing tracks keep state but emit no masks",
        "image_label_contract": "weak positive image pixels require RGB region AND physical BEV support; grid and image masks are separate, not occupancy",
        "discovery_domain": context.get("discovery_domain", "receiver"),
        "domain_contract": "image-rays adds observed entity surfaces to candidate search only; physics, illumination events and positive targets remain on visible ground",
        "owner_scoring": owner_scoring, "owner_scoring_config": asdict(owner_config), "timing": timing,
        "discrepancy_definition": "appearance plus confidence-weighted metric flow/BEV disagreement and current-only feature change; union with base/static/caster proposals",
        "base_discrepancy_definition": "registered illumination change * supported entity complement with validated ground or entity motion reference",
        "entity_state_legend": ENTITY_STATES,
        "entity_motion_config": context.get("entity_motion_config", asdict(EntityMotionConfig())),
        "entity_suppression_enabled": context.get("entity_suppression_enabled", True),
        "entity_suppression_contract": "front box-surface proxy + image-pixel core margin + FB-valid flow match + predicted surface visibility; never whole 2D rectangles",
        "entity_map_coordinates": "fixed world receiver-ray grid; entity surface along ray, occluded ground endpoint; not dense image segmentation",
        "caster_memory_contract": "checked GT triplets drive projection; optional descriptor consistency weights same-shadow-track temporal owner support, never car-color versus shadow-color similarity",
        "entity_occlusion_contract": "unmatched or boundary box rays remain unobservable ground, never background negatives; last frame has no outgoing suppression prediction",
        "compute_contract": "shared cached BEV/flow unchanged; time downstream proposal, region and physics work separately from export; receiver-only domain can make incremental removal zero",
        "physical_reference_source_legend": PHYSICAL_REFERENCE_SOURCES,
        "ground_reference_contract": "canonical per-pair ground geometry validated by direct or same-patch textured DIS anchors; never copied box motion or inferred free space",
        "partial_evidence_contract": "missing physical motion reference retains appearance candidates; incomplete discrepancy is NaN; entity state remains missing on receivers without box-surface predictions",
        "projection_review_contract": "oracle capacity optimizes on reference; selected direction is frozen on proposals before reading review masks",
        "bev_role_contract": "context/stability/off control temporal region descriptors; independent current-only spatial features drive M2 separately",
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
            "entity_suppressed_cells": sum(row["entity_suppressed_cells"] for row in rows),
            "additional_search_removed_cells": sum(row["additional_search_removed_cells"] for row in rows),
            "caster_memory_tracks": len(caster_memory.records),
            "candidate_cells": sum(row["candidate_cells"] for row in rows),
            "entity_candidate_cells": sum(row["entity_candidate_cells"] for row in rows),
            "owner_cells": sum(row["owner_cells"] for row in rows),
            "unique_caster_regions": sum(row["unique_caster_regions"] for row in rows),
            "timing": timing,
            "mask_hashes": {name: value.hexdigest() for name, value in mask_hashes.items()},
            "positive_cells": sum(row["positive_cells"] for row in rows), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--feature-root", required=True, help="Temporal/context BEV cache for region association")
    parser.add_argument("--spatial-feature-root", help="Independent --history-mode current-only cache for metric BEV discovery")
    parser.add_argument("--clips", required=True, help="JSONL: id,scene_name,camera,start_sample_token,track_id,num_frames")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--extent-m", type=float, default=40)
    parser.add_argument("--resolution-m", type=float, default=.5)
    parser.add_argument("--seed-score", type=float, default=.35)
    parser.add_argument("--grow-score", type=float, default=.18)
    parser.add_argument("--stage", choices=["signal", "projection", "full"], default="full")
    parser.add_argument("--bev-role", choices=["context", "stability", "off"], default="context")
    parser.add_argument("--entity-motion-suppression", choices=["on", "off"], default="on")
    parser.add_argument("--discovery-domain", choices=["receiver", "image-rays"], default="receiver")
    parser.add_argument("--owner-scoring", choices=["physical", "geometry", "geometry+appearance+bev"], default="geometry+appearance+bev")
    parser.add_argument("--discovery-mode", choices=["cross-space", "appearance"], default="cross-space")
    parser.add_argument("--entity-core-margin-px", type=float, default=2.0)
    parser.add_argument("--entity-max-residual-px", type=float, default=2.0)
    parser.add_argument("--entity-max-fb-error-px", type=float, default=1.5)
    parser.add_argument("--review-regions", help="Optional independent image masks: clip_id,frame_token,region_type,mask")
    args = parser.parse_args()
    config = VideoShadowConfig(seed_score=args.seed_score, grow_score=args.grow_score)
    config.validate()
    entity_config = EntityMotionConfig(args.entity_core_margin_px, args.entity_max_residual_px, args.entity_max_fb_error_px)
    entity_config.validate()
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
            BEVFlowConfig(extent_m=args.extent_m, resolution_m=args.resolution_m),
            entity_config=entity_config, entity_suppression=args.entity_motion_suppression == "on", discovery_domain=args.discovery_domain,
            spatial_feature_root=args.spatial_feature_root)
        if args.discovery_mode == "cross-space" and not all(source["available_for_discovery"] for source in context["spatial_feature_sources"]):
            raise ValueError("Cross-space discovery needs current-only BEV features. Export with --history-mode current-only and set --spatial-feature-root; use --discovery-mode appearance for legacy caches.")
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
                           stage=args.stage, bev_role=args.bev_role, reviews=clip_reviews, owner_scoring=args.owner_scoring,
                           discovery_mode=args.discovery_mode)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(output/"summary.json", {"clips": summaries, "pair_manifest_sha256": digest(args.pairs),
        "clip_manifest_sha256": digest(args.clips), "stage": args.stage, "bev_role": args.bev_role,
        "discovery_domain": args.discovery_domain, "owner_scoring": args.owner_scoring,
        "discovery_mode": args.discovery_mode, "spatial_feature_root": args.spatial_feature_root,
        "review_manifest_sha256": digest(args.review_regions) if args.review_regions else None,
        "training_performed": False})


if __name__ == "__main__":
    main()
