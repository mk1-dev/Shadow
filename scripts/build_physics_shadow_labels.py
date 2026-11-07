from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, resolve_path, write_rows
from shadow_occ.annotation.bev_flow import bev_mask_to_image
from shadow_occ.annotation.physics_io import load_motion_analysis, check_pair_files, read_evidence
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.annotation.physics_labels import (
    LabelConfig, footprint_mask, light_bank, light_support_index, middle_evidence, rank_hypotheses, receiver_points,
    select_shadow_and_caster, factorized_shadow_decision, window_tracks, fuse_shadow_support,
)
from shadow_occ.annotation.physics_motion import ground_homography, project_world
from shadow_occ.annotation.illumination import PROTOCOL as LIGHT_PROTOCOL, validate_prior
from shadow_occ.annotation.caster_window import (
    caster_proposals, component_transition_diagnostics, extend_pairs, long_observations,
)
from shadow_occ.annotation.receiver_explanations import (
    PROTOCOL as RECEIVER_PROTOCOL, SCORE_PROTOCOL, CompetitionConfig, apply_competition, explain_component,
    receiver_evidence,
)
from shadow_occ.utils.io import write_json


def visible_box(track, projection, shape):
    pixels, _, good = project_world(track["points"][1, 1:], projection)
    if not good.all():
        return None
    height, width = shape
    lower = np.maximum(pixels.min(axis=0), [0, 0])
    upper = np.minimum(pixels.max(axis=0), [width, height])
    if np.any(upper-lower < 2):
        return None
    return np.r_[lower, upper].tolist()


def projectable_shadow_support(mask, explanations, track_map, plane, directions, projection,
                               policy, consensus_fraction):
    """Fuse only explanations whose shadow volume projects into the camera."""
    footprints, rejected = [], []
    for explanation in explanations:
        try:
            footprint = footprint_mask(track_map[explanation["instance_token"]]["points"][1, 1:],
                plane, directions[explanation["light_id"]], projection, mask.shape)
        except ValueError as exc:
            rejected.append({"instance_token": explanation["instance_token"],
                             "light_id": explanation["light_id"], "reason": str(exc)})
            continue
        footprints.append(footprint)
    if not footprints:
        return None, rejected
    return fuse_shadow_support(mask, footprints, policy, consensus_fraction), rejected


def main():
    parser = argparse.ArgumentParser("Joint 3D/2D weak-label gate; audit-only unless explicitly enabled.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--bev-flow-root", help="BEV components after learned receiver and flow selection")
    parser.add_argument("--minimum-bev-area-m2", type=float, default=0.5)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--export-weak-labels", action="store_true", help="Opt in to experimental positive-only training labels, NOT verified GT")
    parser.add_argument("--max-components-per-window", type=int, default=50)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--review-limit", type=int, default=20)
    parser.add_argument("--minimum-temporal-gain", type=float, default=0.1,
                        help="Real trajectory minus hardest frozen/wrong-time score; zero is an ablation")
    parser.add_argument("--decision-policy", choices=["strict-v2", "factorized-v3"], default="strict-v2")
    parser.add_argument("--existence-mode", choices=["motion", "static-dynamic"],
                        help="Default: static-dynamic for factorized-v3, motion for strict-v2")
    parser.add_argument("--light-search", choices=["bank", "fixed"], default="bank")
    parser.add_argument("--fixed-light-id", type=int, default=0, help="Predeclared light ID, never chosen on eval")
    parser.add_argument("--trajectory-contrast", choices=["on", "off"], default="on")
    parser.add_argument("--independent-light", choices=["on", "off"], default="on")
    parser.add_argument("--positive-fusion", choices=["best", "intersection", "consensus"],
                        help="Default: consensus for factorized-v3, intersection for strict-v2")
    parser.add_argument("--consensus-fraction", type=float, default=2/3)
    parser.add_argument("--receiver-competition", choices=["on", "off", "score-only"],
                        help="Default: on for factorized BEV labels; off for legacy/image-only labels")
    parser.add_argument("--control-shift-m", type=float, default=3.0)
    parser.add_argument("--minimum-projection-gain", type=float, default=0.05)
    parser.add_argument("--minimum-explanation-margin", type=float, default=0.05)
    parser.add_argument("--appearance-prior-weight", type=float, default=0.2)
    parser.add_argument("--temporal-evidence", choices=["available", "complete"],
                        help="Use observed signed boundary events; default available for factorized-v3, complete for strict-v2")
    parser.add_argument("--light-prior-manifest", help="Frozen learned or astronomical per-window light distributions")
    parser.add_argument("--light-prior-weight", type=float, default=.15)
    parser.add_argument("--candidate-source", choices=["photometric", "caster"],
                        help="Default caster for factorized BEV labels; GT projection expands search, never creates positive pixels")
    parser.add_argument("--caster-proposal-darkness-ratio", type=float, default=.92,
                        help="Observed-middle-frame darkness for the caster-conditioned proposal pool; not a final score")
    parser.add_argument("--temporal-radius", type=int, default=3, help="1=3 frames, 2=5, 3=7, 4=9 when consecutive cached pairs exist")
    args = parser.parse_args()
    args.receiver_competition = args.receiver_competition or (
        "on" if args.bev_flow_root and args.decision_policy == "factorized-v3" else "off")
    if args.receiver_competition != "off" and not args.bev_flow_root:
        parser.error("Receiver competition requires --bev-flow-root")
    competition_config = CompetitionConfig(control_shift_m=args.control_shift_m,
        minimum_projection_gain=args.minimum_projection_gain,
        minimum_winner_margin=args.minimum_explanation_margin,
        appearance_prior_weight=args.appearance_prior_weight)
    competition_config.validate()
    args.positive_fusion = args.positive_fusion or ("consensus" if args.decision_policy == "factorized-v3" else "intersection")
    args.temporal_evidence = args.temporal_evidence or ("available" if args.decision_policy == "factorized-v3" else "complete")
    args.candidate_source = args.candidate_source or ("caster" if args.bev_flow_root and args.decision_policy == "factorized-v3" else "photometric")
    if not 1 <= args.temporal_radius <= 4:
        parser.error("Temporal radius must be between 1 and 4")
    if args.max_components_per_window < 1 or args.review_limit < 0 or not 0 < args.caster_proposal_darkness_ratio <= 1:
        raise ValueError("Invalid processing limits")
    if not 0 <= args.fixed_light_id < len(light_bank()) or not 0 < args.consensus_fraction <= 1:
        parser.error("Invalid fixed light ID or consensus fraction")
    config = LabelConfig(minimum_area=args.minimum_area, minimum_temporal_gain=args.minimum_temporal_gain,
        allow_partial_temporal=args.temporal_evidence == "available",
        light_prior_weight=args.light_prior_weight,
        require_temporal_contrast=args.trajectory_contrast == "on",
        require_independent_light=args.independent_light == "on",
        allow_static_existence=(args.existence_mode or ("static-dynamic" if args.decision_policy == "factorized-v3" else "motion")) == "static-dynamic",
        light_ids=(args.fixed_light_id,) if args.light_search == "fixed" else None)
    config.validate()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    base = Path(args.pairs).resolve().parent
    analysis = Path(args.analysis_root).resolve()
    pairs, stats, protocol_path = load_motion_analysis(args.pairs, analysis)
    light_rows = read_rows(args.light_prior_manifest) if args.light_prior_manifest else []
    light_priors = {row["window_id"]: row for row in light_rows}
    if len(light_priors) != len(light_rows):
        raise ValueError("Duplicate light-prior windows")
    bev_windows = {}
    if args.bev_flow_root:
        bev_root = Path(args.bev_flow_root).resolve()
        bev_protocol = json.loads((bev_root/"bev_flow_protocol.json").read_text(encoding="utf-8"))
        for key, path in (("pair_manifest_sha256", args.pairs), ("analysis_protocol_sha256", protocol_path),
                          ("pair_metrics_sha256", analysis/"pair_metrics.jsonl")):
            if bev_protocol[key] != digest(path):
                raise ValueError(f"BEV-flow input mismatch: {key}")
        rows = read_rows(bev_root/"windows.jsonl")
        bev_windows = {tuple(row["pair_ids"]): row for row in rows}
        if len(bev_windows) != len(rows) or not rows:
            raise ValueError("BEV windows must be nonempty and unique")
        if not np.isfinite(args.minimum_bev_area_m2) or args.minimum_bev_area_m2 <= 0:
            raise ValueError("BEV component area must be positive square metres")
    successor, predecessor = {}, {}
    for pair in pairs:
        key = (pair["scene_name"], pair["official_split"], pair["camera"], pair["sample_token"])
        if key in successor:
            raise ValueError("Duplicate camera/sample pair")
        successor[key] = pair
        predecessor[(*key[:3], pair["next_sample_token"])] = pair
    counts, candidates, audit, transition_audit = Counter(), [], [], []
    directions = light_bank()
    checked = set()
    for left in tqdm(pairs, desc="3D / ground / flow / light joint gate", unit="pair", mininterval=5):
        right = successor.get((left["scene_name"], left["official_split"], left["camera"], left["next_sample_token"]))
        if right is None:
            counts["pairs_without_successor"] += 1
            continue
        counts["windows_attempted"] += 1
        for pair in (left, right):
            if pair["id"] not in checked:
                check_pair_files(pair, base)
                checked.add(pair["id"])
        first, first_metric = read_evidence(left, stats, analysis)
        second, second_metric = read_evidence(right, stats, analysis)
        try:
            maps = middle_evidence(left, right, first, second)
        except ValueError as exc:
            audit.append({"left_pair": left["id"], "right_pair": right["id"], "status": "invalid_window", "reason": str(exc)})
            counts["invalid_windows"] += 1
            continue
        if not first_metric["exposure"]["valid"] or not second_metric["exposure"]["valid"]:
            counts["unstable_exposure_windows"] += 1
            audit.append({"left_pair": left["id"], "right_pair": right["id"], "status": "unstable_exposure"})
            if not config.allow_static_existence:
                continue
            counts["unstable_exposure_windows_retained_for_static"] += 1
        tracks, rejected = window_tracks(left, right)
        if rejected:
            audit.append({"left_pair": left["id"], "right_pair": right["id"], "status": "track_state_rejections", "tracks": rejected})
        counts["track_state_rejections"] += len(rejected)
        frame = right["frames"][0]
        projection = np.asarray(frame["projection_world_to_image"])
        basis = np.asarray(left["plane_basis_world"])
        plane = np.asarray(left["plane_world"])
        rank_maps, rank_projection, rank_config = maps, projection, config
        window_id = hashlib.sha256(f"{left['id']}:{right['id']}".encode()).hexdigest()[:24]
        light_prior = light_priors.get(window_id)
        if light_prior is not None:
            validate_prior(light_prior, directions)
            if (light_prior["protocol"] != LIGHT_PROTOCOL or light_prior["scene_name"] != left["scene_name"]
                    or light_prior["sample_tokens"] != [left["sample_token"], right["sample_token"], right["next_sample_token"]]
                    or not np.allclose(light_prior["directions_world"], directions)):
                raise ValueError("Light prior does not match the window/direction bank")
            counts["windows_with_light_prior"] += 1
        bev_ref, bev_arrays, grid = None, None, None
        if args.bev_flow_root:
            bev_ref = bev_windows[left["id"], right["id"]]
            evidence_path = resolve_path(bev_ref["evidence"], bev_root)
            if digest(evidence_path) != bev_ref["evidence_sha256"]:
                raise ValueError("BEV-flow evidence changed")
            with np.load(evidence_path, allow_pickle=False) as archive:
                bev_arrays = {key: archive[key].copy() for key in archive.files}
            bev_arrays["temporal_exposure_valid_by_interval"] = np.broadcast_to(
                np.array([bool(value["exposure"]["valid"]) for value in (first_metric, second_metric)])[:, None, None],
                (2, *bev_arrays["receiver_valid"].shape)).copy()
            bev_arrays.update(receiver_evidence(bev_arrays,
                light_threshold=bev_protocol.get("config", {}).get("light_threshold_L", 6.0)))
            if args.receiver_competition != "off" and bev_ref.get("receiver_explanation_protocol") != RECEIVER_PROTOCOL:
                raise ValueError("Rebuild BEV analysis from cached pairs/flow/features for receiver competition")
            grid = BEVGrid(**bev_ref["grid"])
            rank_projection = bev_arrays["bev_world_to_grid"]
            rank_maps = {"visible": bev_arrays["receiver_valid"], "dark": bev_arrays["shadow_candidate"],
                "onset": bev_arrays["shadow_boundary_onset"], "recovery": bev_arrays["shadow_boundary_recovery"],
                "flow_supported": bev_arrays["shadow_flow_supported"],
                "same_patch_cycle": bev_arrays["shadow_same_patch_cycle"]}
            if config.allow_partial_temporal:
                rank_maps["onset"] = bev_arrays["receiver_boundary_formation"] & rank_maps["dark"]
                rank_maps["recovery"] = bev_arrays["receiver_boundary_recovery"] & rank_maps["dark"]
                rank_maps["same_patch_cycle"] = rank_maps["onset"] & rank_maps["recovery"]
            if bev_ref.get("learned_bev_selection", {}).get("flow_gate") is False:
                maps["flow_supported"] = maps["visible"].copy()
            rank_config = replace(config, minimum_area=max(1, int(np.ceil(args.minimum_bev_area_m2/grid.resolution**2))),
                                  minimum_change_pixels=1)
        polygon_cache, extended_context, expanded = {}, None, None
        if grid is not None:
            roi, expanded = caster_proposals(bev_arrays, tracks, directions, plane, rank_projection,
                polygon_cache, light_prior, expand=args.candidate_source == "caster",
                proposal_darkness_ratio=args.caster_proposal_darkness_ratio)
            rank_maps["dark"] = bev_arrays["shadow_candidate"]
            if config.allow_partial_temporal:
                rank_maps["onset"] = bev_arrays["receiver_boundary_formation"] & rank_maps["dark"]
                rank_maps["recovery"] = bev_arrays["receiver_boundary_recovery"] & rank_maps["dark"]
                rank_maps["same_patch_cycle"] = rank_maps["onset"] & rank_maps["recovery"]
            counts["caster_added_observed_pixels"] += expanded["added_observed_pixels"]
            for key in ("caster_roi_pixels", "caster_roi_receiver_pixels", "caster_roi_dark_pixels",
                        "caster_roi_flow_pixels", "caster_roi_bev_pixels", "caster_own_component_pixels",
                        "overlap_original"):
                counts[key] += expanded[key]
            counts["caster_added_pixels"] += expanded["added"]
            counts["caster_windows_with_added"] += int(expanded["added"] > 0)
            chain = extend_pairs(left, right, successor, predecessor, args.temporal_radius)
            if len(chain) > 2:
                extended_context = long_observations(chain, frame["sample_data_token"], bev_arrays, roi, base, stats, analysis)
            counts[f"temporal_frames:{len(chain)+1}"] += 1
            audit.append({"left_pair": left["id"], "right_pair": right["id"], "window_id": bev_ref["id"],
                "status": "caster_proposals", **expanded,
                "temporal_frames": len(chain)+1,
                "track_state_counts": extended_context["track_state_counts"] if extended_context else None})
        transition_audit.append({"window_id": window_id, "scene_name": left["scene_name"],
            "camera": left["camera"], "long_window_available": extended_context is not None,
            "timestamps": extended_context["timestamps"] if extended_context else None,
            "tracks_with_complete_states": len(extended_context["tracks"]) if extended_context else None,
            "track_state_counts": extended_context["track_state_counts"] if extended_context else None,
            "frames": extended_context["frame_diagnostics"] if extended_context else [],
            "central_onset_pixels": int(rank_maps["onset"].sum()),
            "central_recovery_pixels": int(rank_maps["recovery"].sum())})
        # Never bridge gaps across unsupported receivers by dilation or closing.
        number, labels, areas, _ = cv2.connectedComponentsWithStats(rank_maps["dark"].astype(np.uint8), connectivity=8)
        components = [index for index in range(1, number) if areas[index, cv2.CC_STAT_AREA] >= rank_config.minimum_area]
        counts["small_components"] += number-1-len(components)
        if len(components) > args.max_components_per_window:
            counts["component_limit_windows"] += 1
            audit.append({"left_pair": left["id"], "right_pair": right["id"], "status": "too_many_components", "count": len(components)})
            continue
        track_map = {track["instance_token"]: track for track in tracks}
        context = {"scene_name": left["scene_name"], "camera": left["camera"],
                   "middle_timestamp": frame["camera_timestamp"], "middle_image_sha256": frame["image_sha256"],
                   "alignment": "same_world_receiver_not_caster_translation", "views": []}
        window_stem = hashlib.sha256(f"{left['id']}:{right['id']}".encode()).hexdigest()[:24]
        transition_path = output/"transitions"/f"{window_stem}.npz"
        transition_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(transition_path, onset=(maps["onset"] & maps["flow_supported"]).astype(np.uint8),
                            recovery=(maps["recovery"] & maps["flow_supported"]).astype(np.uint8))
        context["transitions"] = {"path": str(transition_path), "sha256": digest(transition_path),
            "semantics": "relative_brightness_evidence_not_absolute_nonshadow_labels"}
        for slot, adjacent in (("previous", left["frames"][0]), ("next", right["frames"][1])):
            adjacent_image = cv2.imread(str(resolve_path(adjacent["image"], base)))
            if adjacent_image is None:
                raise FileNotFoundError(adjacent["image"])
            h = ground_homography(adjacent["projection_world_to_image"], projection, basis)
            aligned = cv2.warpPerspective(adjacent_image, h, (maps["dark"].shape[1], maps["dark"].shape[0]))
            aligned[~maps["visible"]] = 0
            path = output/"context"/f"{window_stem}_{slot}.png"
            checked_write_image(path, aligned)
            context["views"].append({"slot": slot, "image": str(path), "sha256": digest(path),
                                     "timestamp": adjacent["camera_timestamp"]})
        for component in components:
            rank_mask = labels == component
            hypotheses, report = rank_hypotheses(rank_mask, rank_maps, tracks, plane, basis, rank_projection,
                                                  directions, rank_config, polygon_cache=polygon_cache, light_prior=light_prior,
                                                  temporal_context=extended_context)
            competition = None
            if args.receiver_competition != "off":
                competition = explain_component(rank_mask, bev_arrays, hypotheses, polygon_cache,
                                                grid.resolution, competition_config)
            mask = rank_mask
            if grid is not None:
                mask = bev_mask_to_image(rank_mask, projection, basis, grid, maps["visible"].shape) & maps["visible"]
                if mask.sum() < config.minimum_area:
                    counts["bev_components_without_image_support"] += 1
                    continue
            stem = hashlib.sha256(f"{left['id']}:{right['id']}:{component}".encode()).hexdigest()[:24]
            path = output/"regions"/f"{stem}.png"
            checked_write_image(path, mask.astype(np.uint8)*255)
            ground_path = output/"ground_points"/f"{stem}.npz"
            ground_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ground_path, observed_receiver_points_world=receiver_points(mask, projection, basis))
            eligible = {item["instance_token"] for item in hypotheses}
            eligible.update(alt["instance_token"] for item in hypotheses for alt in item.get("alternatives", []))
            caster_metadata = {}
            for token in eligible:
                track = track_map[token]
                caster_metadata[token] = {"annotation_token": track["annotation_token"], "category": track["category"],
                                          "bbox": visible_box(track, projection, mask.shape)}
            bev_component = None
            if grid is not None:
                component_path = output/"bev_regions"/f"{stem}.npz"
                component_path.parent.mkdir(parents=True, exist_ok=True)
                pooled = {}
                if "learned_bev_receiver_features" in bev_arrays:
                    features = bev_arrays["learned_bev_receiver_features"].astype(np.float32)
                    support = bev_arrays["learned_bev_valid"] & rank_mask[None]
                    pooled["receiver_feature_mean"] = np.stack([
                        features[t][support[t]].mean(axis=0) if support[t].any() else np.zeros(features.shape[-1])
                        for t in range(3)]).astype(np.float32)
                np.savez_compressed(component_path, component_mask=rank_mask, **pooled)
                bev_component = {"window_id": bev_ref["id"], "evidence": str(evidence_path),
                    "mature_objects": bev_ref.get("mature_objects"),
                    "mature_image_motion": bev_ref.get("mature_image_motion"),
                    "grid": bev_ref["grid"],
                    "selection": bev_ref.get("learned_bev_selection"),
                    "evidence_sha256": bev_ref["evidence_sha256"], "component": str(component_path),
                    "component_sha256": digest(component_path), "area_m2": float(rank_mask.sum()*grid.resolution**2),
                    "coordinate_frame": "fixed anchor ego x/y; rows=y, columns=x",
                    "caster_footprints": [{"instance_token": h["instance_token"], "light_id": h["light_id"],
                        "polygons": [p.tolist() for p in polygon_cache[h["instance_token"], h["light_id"]]]}
                        for h in sorted(hypotheses, key=lambda h: -h["score"])[:5]]}
            candidates.append({"id": stem, "scene_name": left["scene_name"], "official_split": left["official_split"],
                "camera": left["camera"], "sample_token": right["sample_token"],
                "frame_tokens": [left["frames"][0]["sample_data_token"], frame["sample_data_token"], right["frames"][1]["sample_data_token"]],
                "sample_tokens": [left["sample_token"], right["sample_token"], right["next_sample_token"]],
                "pair_ids": [left["id"], right["id"]], "image": str(resolve_path(frame["image"], base)),
                "ground_support": str(resolve_path(frame["ground_support"], base)),
                "timestamp": frame["camera_timestamp"], "mask": str(path), "mask_sha256": digest(path),
                "ground_points": str(ground_path), "hypotheses": hypotheses,
                "light_prior": light_prior,
                "caster_proposals": expanded,
                "association_frame_tokens": [value["sample_data_token"] for value in extended_context["frames"]] if extended_context else [],
                "association_sample_tokens": extended_context["sample_tokens"] if extended_context else [],
                "temporal_window": {"frames": len(extended_context["frames"]) if extended_context else 3,
                    "timestamps": extended_context["timestamps"] if extended_context else [value["camera_timestamp"] for value in
                        (left["frames"][0], frame, right["frames"][1])],
                    "exposures": extended_context["exposures"] if extended_context else None,
                    "tracks_with_complete_states": len(extended_context["tracks"]) if extended_context else len(tracks),
                    "track_state_counts": extended_context["track_state_counts"] if extended_context else
                        {"seen": len(tracks), "complete": len(tracks), "partial": 0, "inconsistent": 0, "missing_samples": 0},
                    **({"gt_tracks_seen": extended_context["track_state_counts"]["seen"],
                        "gt_tracks_complete": extended_context["track_state_counts"]["complete"],
                        "gt_tracks_partial": extended_context["track_state_counts"]["partial"],
                        "gt_tracks_inconsistent": extended_context["track_state_counts"]["inconsistent"],
                        "gt_missing_samples": extended_context["track_state_counts"]["missing_samples"]}
                       if extended_context else {
                        "gt_tracks_seen": len(tracks), "gt_tracks_complete": len(tracks), "gt_tracks_partial": 0,
                        "gt_tracks_inconsistent": 0, "gt_missing_samples": 0})},
                "transition_diagnostics": {
                    "central_onset_pixels": int((rank_mask & rank_maps["onset"]).sum()),
                    "central_recovery_pixels": int((rank_mask & rank_maps["recovery"]).sum()),
                    "long_window": component_transition_diagnostics(extended_context, rank_mask)},
                "evidence": report, "casters": caster_metadata, "physics_context": context,
                "bev_flow": bev_component, "receiver_competition": competition})
            counts.update(f"evidence_reason:{reason}" for reason in report["reasons"])
            counts.update(f"temporal_evidence_reason:{reason}" for reason in report.get("temporal_reasons", []))
        counts["windows_processed"] += 1
        print(f"windows={counts['windows_processed']} components={len(components)} tracks={len(tracks)} scene={left['scene_name']}", flush=True)
    # Only other tracks outside the query's camera frames AND physical samples vote.
    pools = defaultdict(list)
    for item in candidates:
        pools[item["scene_name"], item["official_split"]].append(item)
    support_index = light_support_index(candidates)
    pair_lookup = {pair["id"]: pair for pair in pairs}
    canonical, decisions = [], []
    for index, item in enumerate(tqdm(candidates, desc="independent light support / trimaps", mininterval=5)):
        selector = factorized_shadow_decision if args.decision_policy == "factorized-v3" else select_shadow_and_caster
        decision = selector(item, pools[item["scene_name"], item["official_split"]],
                            directions, config, support_index=support_index)
        counts["baseline_projection_accepted"] += int(decision["shadow_accepted"])
        if item["receiver_competition"] is not None:
            competition = item["receiver_competition"]
            decision = {**decision, "baseline_projection_decision": {
                key: decision[key] for key in ("shadow_accepted", "reason", "existence_score") if key in decision}}
            counts[f"explanation:{competition['winner']}"] += 1
            if args.receiver_competition == "on":
                decision = apply_competition(decision, competition, item["hypotheses"])
            else:
                decision = {**decision, "receiver_competition": competition}
        selection, association, reason = decision["selected"], decision["association"], decision["reason"]
        mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE) > 0
        positive = np.zeros_like(mask)
        if decision["shadow_accepted"]:
            left, right = (pair_lookup[key] for key in item["pair_ids"])
            tracks, _ = window_tracks(left, right)
            track_map = {track["instance_token"]: track for track in tracks}
            projection = np.asarray(right["frames"][0]["projection_world_to_image"])
            explanations = ([selection] if args.positive_fusion == "best" else decision["plausible_hypotheses"])
            positive, rejected = projectable_shadow_support(mask, explanations, track_map,
                left["plane_world"], directions, projection, args.positive_fusion, args.consensus_fraction)
            counts.update(f"footprint_rejection:{item['reason']}" for item in rejected)
            if positive is None:
                reason = "no_projectable_positive_footprint"
                positive = np.zeros_like(mask)
            elif positive.sum() < config.minimum_area:
                reason = "insufficient_supported_positive_area"
            else:
                reason = None
            if reason is not None:
                selection, association = None, None
                decision = {**decision, "shadow_accepted": False, "selected": None,
                            "association": None, "reason": reason}
                positive[:] = False
        accepted = decision["shadow_accepted"]
        label = "YES" if accepted and args.export_weak_labels else "IGNORE"
        counts["weak_gate_passed" if accepted else "uncertain"] += 1
        if accepted:
            counts[f"accepted:{decision.get('existence_evidence', 'motion_contrast')}"] += 1
        counts[f"reason:{reason}"] += 1
        target = np.full(mask.shape, 255, np.uint8)
        score_map = np.full(mask.shape, -1, np.float32)
        if label == "YES":
            target[positive] = 1
            score_map[positive] = decision.get("existence_score", selection["score"])
        region_path = Path(item["mask"])
        if label == "YES":
            region_path = output/"training_regions"/f"{item['id']}.png"
            checked_write_image(region_path, positive.astype(np.uint8)*255)
        target_path = output/"trimaps"/f"{item['id']}.png"
        checked_write_image(target_path, target)
        score_path = output/"scores"/f"{item['id']}.npz"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(score_path, uncalibrated_rule_score=score_map)
        row = {key: item[key] for key in ("id", "scene_name", "official_split", "camera", "sample_token", "image", "ground_support", "timestamp")}
        row.update(label_source="temporal_weak", shadow_mask=str(target_path), mask_encoding="trimap",
            physics_protocol="factorized-receiver-shadow-v3" if args.decision_policy == "factorized-v3" else "three-frame-track-contrast-v2", physics_export_enabled=args.export_weak_labels,
            physics_target_sha256=digest(target_path),
            physics_pair_ids=item["pair_ids"], physics_frame_tokens=item["frame_tokens"],
            physics_sample_tokens=item["sample_tokens"],
            physics_context=item["physics_context"],
            bev_flow=item["bev_flow"],
            physics_evidence={**item["evidence"], "gate_reason": reason, "selected": association,
                              "shadow_decision": decision, "minimum_temporal_gain": config.minimum_temporal_gain,
                              "trajectory_contrast_enabled": config.require_temporal_contrast,
                              "independent_light_enabled": config.require_independent_light,
                              "positive_fusion": args.positive_fusion,
                              "static_existence_enabled": config.allow_static_existence,
                              "probability_calibrated": False, "is_shadow_gt": False},
            regions=[{"id": "observed_region", "mask": str(region_path), "mask_sha256": digest(region_path), "shadow_label": label,
                      "label_source": "temporal_weak", "proposal_source": "bev_flow_receiver_component" if args.bev_flow_root else "ground_aligned_three_frame_dark_region"}])
        if accepted and association is None:
            counts["shadow_supported_owner_unresolved"] += 1
        if association is not None:
            caster = item["casters"][association["instance_token"]]
            row.update(instance_token=association["instance_token"], annotation_token=caster["annotation_token"],
                associated_annotation_token=caster["annotation_token"], caster_category=caster["category"],
                caster_source="annotation_gt", association_source="temporal_weak",
                association_status="temporal_supported" if label == "YES" else "unapproved_physics_candidate",
                association_evidence={"selection": association, "gt_scope": "object_state_only_not_shadow_association"})
            if caster["bbox"] is not None:
                row["caster_bbox_xyxy"] = caster["bbox"]
        canonical.append(row)
        decisions.append({**item, "selection": selection, "shadow_decision": decision, "gate_reason": reason,
                          "decision": "YES" if accepted else "UNCERTAIN", "exported_label": label,
                          "supported_positive_pixels": int(positive.sum()),
                          "probability_calibrated": False, "is_shadow_gt": False})
        if index < args.review_limit:
            image = cv2.imread(item["image"])
            if image is None:
                raise FileNotFoundError(item["image"])
            marked = image.copy()
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(marked, contours, -1, (0, 200, 0) if accepted else (0, 180, 255), 1)
            if association is not None:
                bbox = item["casters"][association["instance_token"]]["bbox"]
                if bbox is not None:
                    x0, y0, x1, y1 = np.rint(bbox).astype(int)
                    cv2.rectangle(marked, (x0, y0), (x1, y1), (255, 200, 0), 1)
            cv2.putText(marked, "WEAK gate pass" if accepted else "UNCERTAIN / IGNORE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            adjacent = [cv2.imread(view["image"]) for view in item["physics_context"]["views"]]
            if any(view is None for view in adjacent):
                raise FileNotFoundError("Missing registered diagnostic context")
            checked_write_image(output/"review"/f"{item['id']}.jpg",
                                np.vstack([np.hstack([adjacent[0], image]), np.hstack([adjacent[1], marked])]))
    write_rows(output/"canonical.jsonl", canonical)
    write_rows(output/"decisions.jsonl", decisions)
    write_rows(output/"window_audit.jsonl", audit)
    write_rows(output/"transition_diagnostics.jsonl", transition_audit)
    split_counts = Counter(row["official_split"] for row in canonical if row["regions"][0]["shadow_label"] == "YES")
    summary = {"counts": dict(counts), "components": len(candidates), "exported_positive_by_official_split": dict(split_counts),
        "decision_policy": args.decision_policy,
        "existence_mode": "static-dynamic" if config.allow_static_existence else "motion",
        "ablation": {key: getattr(args, key) for key in ("light_search", "fixed_light_id", "trajectory_contrast",
            "independent_light", "positive_fusion", "consensus_fraction")},
        "candidate_space": "physical_bev" if args.bev_flow_root else "middle_camera",
        "stage_counts": {"receiver_windows": counts["windows_processed"], "observed_components": len(candidates),
            "absolute_projection_fits": sum(any(h.get("shadow_fit_passes") for parent in item["hypotheses"]
                for h in [parent, *parent.get("alternatives", [])]) for item in candidates),
            "contrast_fits": sum(any(h.get("shadow_passes") and h["temporal_gain"] + 1e-12 >= config.minimum_temporal_gain
                for h in item["hypotheses"]) for item in candidates),
            "configured_trajectory_gate_fits": sum(any(h.get("shadow_passes") for h in item["hypotheses"]) for item in candidates),
            "accepted_shadow": counts["weak_gate_passed"],
            "baseline_projection_accepted": counts["baseline_projection_accepted"],
            "accepted_static_projection": counts["accepted:static_projection"],
            "accepted_moving_projection": counts["accepted:moving_projection"],
            "accepted_fraction_of_observed_components": counts["weak_gate_passed"]/max(len(candidates), 1),
            "unique_weak_association": sum(item["shadow_decision"]["association"] is not None for item in decisions)},
        "scenes": len({item["scene_name"] for item in candidates}),
        "weak_gate_pass_fraction": counts["weak_gate_passed"]/len(candidates) if candidates else None,
        "temporal_contrast": {
            "minimum_gain": config.minimum_temporal_gain,
            "accepted_gain_mean": float(np.mean([row["selection"]["temporal_gain"] for row in decisions
                                                   if row["selection"]])) if counts["weak_gate_passed"] else None,
            "controls_are_shadow_negatives": False},
        "distinct_proposed_caster_tracks": len({row["selection"]["instance_token"] for row in decisions if row["selection"]}),
        "exported_negative_pixels": 0, "export_weak_labels": args.export_weak_labels,
        "shadow_precision": None, "shadow_recall": None, "shadow_iou": None, "caster_accuracy": None,
        "training_readiness": "positive_only_requires_independent_negative_supervision_and_disjoint_dev",
        "static_dark_policy": "projection_supported_yes_else_ignore" if config.allow_static_existence else "retain_unknown_not_negative",
        "receiver_competition": {"mode": args.receiver_competition, "score_protocol": SCORE_PROTOCOL,
            "config": asdict(competition_config),
            "winner_counts": {key: counts[f"explanation:{key}"] for key in
                              ("unknown", "physical_entity", "ground_material", "cast_shadow")},
            "negative_policy": "material/entity explanations do not automatically create NO labels"},
        "probability_calibrated": False}
    write_json(output/"summary.json", summary)
    write_json(output/"label_protocol.json", {**vars(args), "config": asdict(config),
        "receiver_score_protocol": SCORE_PROTOCOL,
        "light_prior_sha256": digest(args.light_prior_manifest) if args.light_prior_manifest else None,
        "receiver_explanation_code_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/receiver_explanations.py"),
        "caster_window_code_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/caster_window.py"),
        "pair_manifest_sha256": digest(args.pairs), "analysis_protocol_sha256": digest(protocol_path),
        "pair_metrics_sha256": digest(analysis/"pair_metrics.jsonl"), "light_directions_world": directions.tolist(),
        "coordinate_frame": "BEV caster matching; observed middle-camera trimaps" if args.bev_flow_root else "middle camera pixels; metric receiver points in world ground plane",
        "protocol": "factorized-receiver-shadow-v3" if args.decision_policy == "factorized-v3" else "three-frame-track-contrast-v2",
        "consensus": "same scene/split; leave all plausible caster tracks, camera frames AND physical samples out",
        "track_contrast": "same light; real time order versus hardest permutation and frozen-middle control",
        "limitations": ["bounding volumes are not object silhouettes", "single locally planar receiver",
                        "constant scene-light hypothesis may fail", "uncalibrated weak scores, not real shadow probabilities",
                        "static existence uses weak projection support; unique ownership still requires motion evidence",
                        "GT-assisted offline labeling is not evidence of GT-free deployment"]})
    print(summary, flush=True)
    if not counts["windows_processed"]:
        raise RuntimeError("No usable consecutive windows; inspect window_audit.jsonl and re-export consecutive pairs")


if __name__ == "__main__":
    main()
