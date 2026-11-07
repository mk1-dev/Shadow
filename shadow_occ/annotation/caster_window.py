"""GT-caster proposals and longer observations on the same world receiver grid."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from shadow_occ.annotation.data import digest, load_array, resolve_path
from shadow_occ.annotation.physics_io import read_evidence
from shadow_occ.annotation.physics_motion import EvidenceConfig, exposure_fit, project_world, sample_at


def extend_pairs(left, right, forward, backward, radius):
    chain = [left, right]
    prefix = (left["scene_name"], left["official_split"], left["camera"])
    for _ in range(radius-1):
        before = backward.get((*prefix, chain[0]["sample_token"]))
        if before is None:
            break
        if before["frames"][1]["sample_data_token"] != chain[0]["frames"][0]["sample_data_token"]:
            break
        chain.insert(0, before)
    for _ in range(radius-1):
        after = forward.get((*prefix, chain[-1]["next_sample_token"]))
        if after is None:
            break
        if chain[-1]["frames"][1]["sample_data_token"] != after["frames"][0]["sample_data_token"]:
            break
        chain.append(after)
    return chain


def caster_proposals(arrays, tracks, directions, plane, projection, polygon_cache, light_prior=None,
                     expand=True, proposal_darkness_ratio=.92):
    from shadow_occ.annotation.physics_labels import footprint_polygon

    if not 0 < proposal_darkness_ratio <= 1:
        raise ValueError("Caster proposal darkness ratio must be in (0, 1]")
    roi = np.zeros(arrays["receiver_valid"].shape, np.uint8)
    for track in tracks:
        for light_id, direction in enumerate(directions):
            key = (track["instance_token"], light_id)
            try:
                if key not in polygon_cache:
                    rays = np.asarray(light_prior["directions_by_frame_world"])[:, light_id] if light_prior else [direction]*3
                    polygon_cache[key] = [footprint_polygon(points[1:], plane, ray, projection)
                        for points, ray in zip(track["points"], rays)]
                for polygon in polygon_cache[key]:
                    cv2.fillConvexPoly(roi, polygon, 1)
            except ValueError:
                continue
    roi = roi > 0
    receiver = arrays["receiver_valid"]
    ratio = arrays["appearance_local_ratio"][1]
    dark = receiver & np.isfinite(ratio) & (ratio < proposal_darkness_ratio)
    learned = arrays.get("learned_bev_receiver_support", np.ones_like(receiver)) & receiver
    flow = arrays["shadow_flow_supported"] & receiver
    original = arrays["shadow_candidate"].copy()
    # This caster-conditioned pool is built from observed signals directly. It is
    # intentionally not defined as a subset of the photometric candidate mask.
    caster_pool = roi & dark & learned & flow
    added = caster_pool & ~original
    arrays["shadow_candidate"] = original | caster_pool if expand else original
    pixel_counts = {
        "caster_roi_pixels": int(roi.sum()),
        "caster_roi_receiver_pixels": int((roi & receiver).sum()),
        "caster_roi_dark_pixels": int((roi & dark).sum()),
        "caster_roi_flow_pixels": int((roi & flow).sum()),
        "caster_roi_bev_pixels": int((roi & learned).sum()),
        "caster_own_component_pixels": int(caster_pool.sum()),
        "overlap_original": int((caster_pool & original).sum()),
        "added": int(added.sum()),
    }
    return roi, {"original_pixels": int(original.sum()), **pixel_counts,
        "added_observed_pixels": int(added.sum()) if expand else 0,
        "proposal_darkness_ratio": float(proposal_darkness_ratio),
        "scope": "GT/light ROI proposes a separate observed dark/receiver/BEV/flow pool; projection never creates positive pixels"}


def summarize_track_states(states, inconsistent, samples):
    """Separate complete GT tracks from partial observations; never fill gaps."""
    complete, diagnostics = {}, {}
    counts = {"seen": len(states), "complete": 0, "partial": 0, "inconsistent": 0, "missing_samples": 0}
    for token, values in states.items():
        missing = [sample for sample in samples if sample not in values]
        if token in inconsistent:
            status = "inconsistent"
            counts["inconsistent"] += 1
        elif missing:
            status = "partial"
            counts["partial"] += 1
        else:
            status = "complete"
            counts["complete"] += 1
            complete[token] = np.asarray([values[sample] for sample in samples], np.float64)
        counts["missing_samples"] += len(missing)
        diagnostics[token] = {"status": status, "observed_samples": len(samples)-len(missing),
            "missing_samples": len(missing), "missing_sample_tokens": missing}
    return complete, diagnostics, counts


def trajectory_track_state(context, token):
    if context is None:
        return {"status": "no_long_window", "observed_samples": None, "missing_samples": None}
    if len(context["frames"]) <= 3:
        return {"status": "three_frame_only", "observed_samples": len(context["frames"]), "missing_samples": None}
    return context["track_state_diagnostics"].get(token, {
        "status": "unseen", "observed_samples": 0, "missing_samples": len(context["sample_tokens"])})


def receiver_chain_support(visible, flow, middle, index):
    overlap = visible[index] & visible[middle]
    support, steps = overlap.copy(), []
    for step in range(min(index, middle), max(index, middle)):
        before = int(support.sum())
        support &= flow[step]
        steps.append({"pair_index": step, "before_pixels": before, "after_pixels": int(support.sum())})
    return overlap, support, steps


def long_transition_evidence(observations, visible, flow, middle, roi):
    """Keep the event rule unchanged; expose every upstream support loss separately."""
    onset, recovery = np.zeros_like(visible), np.zeros_like(visible)
    raw, normalized_events, supports = (np.zeros_like(visible) for _ in range(3))
    exposures, diagnostics = [], []
    config = EvidenceConfig()
    for index in range(len(observations)):
        overlap, support, step_support = receiver_chain_support(visible, flow, middle, index)
        supports[index] = support
        background = support & ~roi
        fit = exposure_fit(observations[index], observations[middle], background, config)
        exposures.append(fit)
        delta = observations[middle]-observations[index]
        raw[index] = overlap & (delta <= -6)
        if index != middle and fit["valid"]:
            normalized = fit["scale"]*observations[index]+fit["offset_L"]
            normalized_events[index] = overlap & (observations[middle]-normalized <= -6)
            event = normalized_events[index] & support
            if index < middle:
                onset[index] = event
            else:
                recovery[index] = event
        if index == middle:
            stage = "reference_frame"
        elif not overlap.any():
            stage = "no_visible_overlap"
        elif not support.any():
            stage = "flow_chain_removed_support"
        elif not fit["valid"]:
            stage = ("roi_excluded_background" if fit["reason"] == "insufficient_background"
                     and support.sum() >= config.min_background else fit["reason"])
        elif not normalized_events[index].any():
            stage = "no_normalized_darkening"
        elif not (onset[index] | recovery[index]).any():
            stage = "flow_removed_darkening"
        else:
            stage = "observed_event"
        diagnostics.append({"frame_index": index, "is_reference": index == middle,
            "visible_pixels": int(visible[index].sum()), "visible_overlap_pixels": int(overlap.sum()),
            "flow_supported_pixels": int(support.sum()), "flow_chain": step_support,
            "roi_excluded_pixels": int((support & roi).sum()),
            "minimum_background_pixels": config.min_background, "exposure": fit,
            "raw_darkening_pixels": int(raw[index].sum()),
            "normalized_darkening_pixels": int(normalized_events[index].sum()) if fit["valid"] and index != middle else None,
            "event_pixels": int((onset[index] | recovery[index]).sum()), "stage": stage})
    return {"onset": onset, "recovery": recovery, "exposures": exposures,
        "raw_darkening": raw, "normalized_darkening": normalized_events,
        "flow_support": supports, "frame_diagnostics": diagnostics}


def component_transition_diagnostics(context, mask):
    """Count observations without conditioning on a surviving caster projection."""
    if context is None:
        return None
    frames = []
    for index, info in enumerate(context["frame_diagnostics"]):
        frames.append({"frame_index": index, "is_reference": info["is_reference"], "stage": info["stage"],
            "raw_darkening_pixels": int((context["raw_darkening"][index] & mask).sum()),
            "normalized_darkening_pixels": int((context["normalized_darkening"][index] & mask).sum())
                if info["normalized_darkening_pixels"] is not None else None,
            "flow_supported_pixels": int((context["flow_support"][index] & mask).sum()),
            "onset_pixels": int((context["onset"][index] & mask).sum()),
            "recovery_pixels": int((context["recovery"][index] & mask).sum())})
    return {"frames": frames, "event_pixels": int(((context["onset"] | context["recovery"]).any(axis=0) & mask).sum()),
        "scope": "observed component events before caster/light selection; raw darkness is not a shadow label"}


def sample_receiver_frame(world, frame, ground, receiver):
    """Trace geometric projection and sparse-ground sampling without relaxing support."""
    height, width = ground.shape
    if (width, height) != tuple(frame["image_size_wh"]):
        raise ValueError("Receiver ground mask and camera pixel frame differ")
    uv, depth, front = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
    inside = front & (uv >= 0).all(axis=1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1)
    coordinates = np.where(inside[:, None], uv, -1).reshape(*world.shape[:2], 2).astype(np.float32)
    geometric = inside.reshape(receiver.shape)
    fraction = sample_at(ground.astype(np.float32), coordinates)
    nearest = sample_at(ground.astype(np.float32), coordinates, nearest=True) > 0
    supported = geometric & (fraction >= .999)
    valid = supported & receiver

    def pixel_range(values):
        return [float(values.min()), float(values.max())] if len(values) else None

    reason = ("supported" if valid.any() else "empty_anchor_receiver" if not receiver.any()
        else "no_front_projection" if not front.any() else "outside_image" if not inside.any()
        else "empty_source_ground" if not ground.any() else "ground_bilinear_rejection" if (geometric & nearest).any() and not supported.any()
        else "no_ground_grid_intersection" if not supported.any() else "anchor_receiver_disjoint")
    diagnostics = {"sample_data_token": frame["sample_data_token"],
        "grid_pixels": int(receiver.size), "anchor_receiver_pixels": int(receiver.sum()),
        "source_ground_pixels": int(ground.sum()), "front_projection_pixels": int(front.sum()),
        "in_image_pixels": int(inside.sum()),
        "ground_nearest_pixels": int((geometric & nearest).sum()),
        "ground_bilinear_pixels": int(supported.sum()), "receiver_visible_pixels": int(valid.sum()),
        "anchor_ground_nearest_pixels": int((geometric & nearest & receiver).sum()),
        "projection_depth_range_m": pixel_range(depth[front]),
        "front_u_range_px": pixel_range(uv[front, 0]), "front_v_range_px": pixel_range(uv[front, 1]),
        "zero_support_reason": reason,
        "sampling_rule": "bilinear >= 0.999; nearest counts are diagnostic only"}
    return coordinates, valid, diagnostics


def sample_receiver_flow(evidence, coordinates, image_shape, receiver):
    """Sample cached source-frame evidence on the fixed world receiver grid."""
    source = np.asarray(evidence["receiver_flow_supported"], np.float32)
    if source.shape != image_shape:
        raise ValueError("Receiver flow support and source camera pixel frame differ")
    sampled = sample_at(source, coordinates)
    support = sampled >= .999
    nearest = sample_at(source, coordinates, nearest=True) > 0
    diagnostics = {"source_supported_pixels": int((source > 0).sum()),
        "sampled_nearest_pixels": int(nearest.sum()), "sampled_bilinear_pixels": int(support.sum()),
        "anchor_sampled_nearest_pixels": int((nearest & receiver).sum()),
        "anchor_sampled_bilinear_pixels": int((support & receiver).sum())}
    for key in ("visible_ground", "flow_reliable", "receiver_flow_anchor", "receiver_texture"):
        if key in evidence:
            values = np.asarray(evidence[key], np.float32)
            if values.shape != image_shape:
                raise ValueError(f"{key} and source camera pixel frame differ")
            diagnostics[key] = {"source_pixels": int((values > 0).sum()),
                "anchor_sampled_pixels": int(((sample_at(values, coordinates) >= .999) & receiver).sum())}
    diagnostics["zero_support_reason"] = ("supported" if (support & receiver).any()
        else "empty_anchor_receiver" if not receiver.any() else "empty_source_flow_support" if not source.any()
        else "flow_bilinear_rejection" if (nearest & receiver).any() else "no_flow_grid_intersection")
    return support, diagnostics


def receiver_sampling_summary(frames):
    flows = [frame["flow"] for frame in frames if "flow" in frame]
    central = [frame for frame in frames if "cached_central_visibility_mismatch_pixels" in frame]
    comparisons = [frame for frame in frames if not frame["is_reference"] and "chain_supported_pixels" in frame]
    return {"frame_comparisons": len(frames),
        "reference_frames": sum(frame["is_reference"] for frame in frames),
        "empty_reference_receiver_frames": sum(frame["is_reference"] and frame["anchor_receiver_pixels"] == 0 for frame in frames),
        "visibility_reasons": dict(Counter(frame["zero_support_reason"] for frame in frames)),
        "visibility_reasons_by_relative_frame": {str(offset): dict(Counter(frame["zero_support_reason"]
            for frame in frames if frame["relative_frame"] == offset))
            for offset in sorted({frame["relative_frame"] for frame in frames})},
        "flow_comparisons": len(flows),
        "flow_reasons": dict(Counter(flow["zero_support_reason"] for flow in flows)),
        "chain_comparisons": len(comparisons),
        "chain_no_visible_overlap": sum(frame["chain_visible_overlap_pixels"] == 0 for frame in comparisons),
        "chain_removed_all_support": sum(frame["chain_visible_overlap_pixels"] > 0 and frame["chain_supported_pixels"] == 0
                                         for frame in comparisons),
        "empty_source_ground_frames": sum(frame["source_ground_pixels"] == 0 for frame in frames),
        "empty_source_flow_frames": sum(flow["source_supported_pixels"] == 0 for flow in flows),
        "cached_central_comparisons": len(central),
        "cached_central_mismatch_frames": sum(frame["cached_central_visibility_mismatch_pixels"] > 0 for frame in central),
        "cached_central_mismatch_pixels": sum(frame["cached_central_visibility_mismatch_pixels"] for frame in central),
        "scope": "fixed cached receiver grid; no labels, thresholds, scores or support masks changed"}


def transition_stage_accounting(frames):
    stages = Counter(frame.get("stage", "other") for frame in frames if not frame["is_reference"])
    known = ("no_visible_overlap", "flow_chain_removed_support", "insufficient_background",
             "roi_excluded_background", "unstable_exposure_or_background", "no_normalized_darkening",
             "flow_removed_darkening", "observed_event")
    counts = {name: stages[name] for name in known}
    counts["other"] = sum(count for name, count in stages.items() if name not in known)
    return {"total": sum(stages.values()), "counts": counts,
        "other_details": {name: count for name, count in stages.items() if name not in known},
        "closed": sum(counts.values()) == sum(stages.values())}


def long_observations(chain, middle_token, arrays, roi, base, stats, analysis, *, sampling_only=False,
                      include_observations=False):
    frames = [pair["frames"][0] for pair in chain]+[chain[-1]["frames"][1]]
    samples = [pair["sample_token"] for pair in chain]+[chain[-1]["next_sample_token"]]
    middle = next(index for index, frame in enumerate(frames) if frame["sample_data_token"] == middle_token)
    world = arrays["receiver_world_xyz"]
    observations, visible, flow, sampling = [], [], [], []
    states, inconsistent = {}, set()
    for pair in chain:
        for item in pair["objects"]:
            track = states.setdefault(item["instance_token"], {})
            for sample, points in ((pair["sample_token"], item["landmarks_world_t"]),
                                   (pair["next_sample_token"], item["landmarks_world_next"])):
                if sample in track and not np.allclose(track[sample], points, atol=1e-5, rtol=0):
                    inconsistent.add(item["instance_token"])
                track[sample] = points
    for frame_index, frame in enumerate(frames):
        image_path = resolve_path(frame["image"], base)
        if digest(image_path) != frame["image_sha256"]:
            raise ValueError("Long-window image changed")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        if image.shape[1::-1] != tuple(frame["image_size_wh"]):
            raise ValueError("Long-window image and camera pixel frame differ")
        if digest(resolve_path(frame["ground_support"], base)) != frame["ground_sha256"]:
            raise ValueError("Long-window ground support changed")
        ground = load_array(frame["ground_support"], base) > 0
        coordinates, valid, diagnostic = sample_receiver_frame(world, frame, ground, arrays["receiver_valid"])
        diagnostic.update(frame_index=frame_index, is_reference=frame_index == middle,
                          relative_frame=frame_index-middle)
        central_index = frame_index-middle+1
        if 0 <= central_index <= 2 and "visible_by_frame" in arrays:
            resampled = valid & arrays["receiver_valid"]
            cached = arrays["visible_by_frame"][central_index] & arrays["receiver_valid"]
            diagnostic["cached_central_visibility_mismatch_pixels"] = int((resampled != cached).sum())
        if not sampling_only:
            lab = cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            observations.append(sample_at(lab[..., 0], coordinates))
        visible.append(valid)
        if frame_index < len(chain):
            pair = chain[frame_index]
            if pair["frames"][0]["sample_data_token"] != frame["sample_data_token"]:
                raise ValueError("Flow evidence must be sampled in its source frame")
            evidence, _ = read_evidence(pair, stats, analysis)
            supported, trace = sample_receiver_flow(evidence, coordinates, image.shape[:2], arrays["receiver_valid"])
            trace.update(pair_id=pair["id"], source_frame_token=frame["sample_data_token"])
            diagnostic["flow"] = trace
            flow.append(supported)
        sampling.append(diagnostic)
    visible = np.stack(visible)
    for index, trace in enumerate(sampling):
        overlap, support, steps = receiver_chain_support(visible, flow, middle, index)
        trace.update(chain_visible_overlap_pixels=int(overlap.sum()), chain_supported_pixels=int(support.sum()),
                     chain_steps=steps)
    if sampling_only:
        return {"frames": frames, "middle": middle, "visible": visible, "flow": flow,
                "sampling_diagnostics": sampling}
    evidence = long_transition_evidence(np.stack(observations), visible, flow, middle, roi)
    for diagnostic, trace in zip(evidence["frame_diagnostics"], sampling):
        diagnostic["sampling"] = trace
    tracks, diagnostics, state_counts = summarize_track_states(states, inconsistent, samples)
    return {"frames": frames, "sample_tokens": samples, "middle": middle, **evidence,
        **({"lightness": np.stack(observations)} if include_observations else {}),
        "visible": visible, "tracks": tracks,
        "polygon_cache": {}, "timestamps": [frame["camera_timestamp"] for frame in frames],
        "track_state_diagnostics": diagnostics, "track_state_counts": state_counts}


def extended_trajectory_fit(context, token, light_id, direction, plane, projection, mask):
    from shadow_occ.annotation.physics_labels import footprint_polygon

    if token not in context["tracks"] or len(context["frames"]) <= 3:
        return None
    key = (token, light_id)
    if key not in context["polygon_cache"]:
        try:
            context["polygon_cache"][key] = [footprint_polygon(points[1:], plane, direction, projection)
                for points in context["tracks"][token]]
        except ValueError:
            return None
    yy, xx = np.where(mask)
    section = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
    local = mask[section]
    offset = np.array([[[xx.min(), yy.min()]]], np.int32)
    footprints = []
    for polygon in context["polygon_cache"][key]:
        raster = np.zeros(local.shape, np.uint8)
        cv2.fillConvexPoly(raster, polygon-offset, 1)
        footprints.append(raster > 0)
    events = [(index, kind, context[kind][index][section] & local)
        for kind in ("onset", "recovery") for index in range(len(footprints))]
    events = [(index, kind, event) for index, kind, event in events if event.any()]
    middle = context["middle"]

    def fit(order):
        current = footprints[order[middle]]
        coverage = float((current & local).sum()/max(local.sum(), 1))
        values = [float((event & current & ~footprints[order[index]]).sum()/event.sum())
                  for index, _, event in events]
        edge = float(np.mean(values)) if values else 0.
        return .4*coverage+.6*edge, edge

    order = np.arange(len(footprints))
    score, edge = fit(order)
    alignment = []
    for index, kind, event in events:
        current_event = event & footprints[middle]
        change = footprints[middle] & ~footprints[index] & local
        matched = event & change
        reason = ("matched" if matched.any() else "outside_current_footprint" if not current_event.any()
                  else "no_predicted_change" if not change.any() else "outside_predicted_change")
        alignment.append({"frame_index": index, "kind": kind, "event_pixels": int(event.sum()),
            "in_current_footprint_pixels": int(current_event.sum()),
            "predicted_change_pixels": int(change.sum()), "matched_pixels": int(matched.sum()),
            "reason": reason})
    controls = [fit(order[::-1])[0], fit(np.roll(order, 1))[0], fit(np.roll(order, -1))[0],
                fit(np.full(len(order), middle))[0]]
    before = np.logical_or.reduce(context["onset"][:, section[0], section[1]], axis=0) & local
    after = np.logical_or.reduce(context["recovery"][:, section[0], section[1]], axis=0) & local
    centers = context["tracks"][token][:, 0]
    movement = float(np.max(np.linalg.norm(centers-centers[middle], axis=-1)))
    return {"score": score, "temporal_gain": score-max(controls), "edge_fit": edge,
        "onset_pixels": int(before.sum()), "recovery_pixels": int(after.sum()),
        "transition_fraction": float((before | after).sum()/max(local.sum(), 1)),
        "observed_transition_types": int(before.any())+int(after.any()), "movement_m": movement,
        "event_alignment": alignment,
        "frames": len(order), "control": "hardest reversed/cyclic/frozen trajectory on identical observations"}
