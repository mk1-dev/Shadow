"""GT-caster proposals and longer observations on the same world receiver grid."""
from __future__ import annotations

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


def caster_proposals(arrays, tracks, directions, plane, projection, polygon_cache, light_prior=None, expand=True):
    from shadow_occ.annotation.physics_labels import footprint_polygon

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
    dark = arrays["receiver_valid"] & (arrays["appearance_local_ratio"][1] < .88)
    learned = arrays.get("learned_bev_receiver_support", np.ones_like(dark))
    flow = arrays["shadow_flow_supported"]
    original = arrays["shadow_candidate"].copy()
    added = dark & (roi > 0) & learned & flow
    arrays["shadow_candidate"] = original | added if expand else original
    return roi > 0, {"original_pixels": int(original.sum()), "caster_roi_pixels": int((roi > 0).sum()),
        "added_observed_pixels": int((added & ~original).sum()) if expand else 0,
        "scope": "GT/light swept ROI proposes search only; observed darkness, BEV and flow are still required"}


def long_transition_evidence(observations, visible, flow, middle, roi):
    """Keep the event rule unchanged; expose every upstream support loss separately."""
    onset, recovery = np.zeros_like(visible), np.zeros_like(visible)
    raw, normalized_events, supports = (np.zeros_like(visible) for _ in range(3))
    exposures, diagnostics = [], []
    config = EvidenceConfig()
    for index in range(len(observations)):
        overlap = visible[index] & visible[middle]
        support = overlap.copy()
        step_support = []
        for step in range(min(index, middle), max(index, middle)):
            before = int(support.sum())
            support &= flow[step]
            step_support.append({"pair_index": step, "before_pixels": before,
                                 "after_pixels": int(support.sum())})
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


def long_observations(chain, middle_token, arrays, roi, base, stats, analysis):
    frames = [pair["frames"][0] for pair in chain]+[chain[-1]["frames"][1]]
    samples = [pair["sample_token"] for pair in chain]+[chain[-1]["next_sample_token"]]
    middle = next(index for index, frame in enumerate(frames) if frame["sample_data_token"] == middle_token)
    world = arrays["receiver_world_xyz"]
    observations, visible, flow = [], [], []
    states, inconsistent = {}, set()
    for pair in chain:
        for item in pair["objects"]:
            track = states.setdefault(item["instance_token"], {})
            for sample, points in ((pair["sample_token"], item["landmarks_world_t"]),
                                   (pair["next_sample_token"], item["landmarks_world_next"])):
                if sample in track and not np.allclose(track[sample], points, atol=1e-5, rtol=0):
                    inconsistent.add(item["instance_token"])
                track[sample] = points
    for frame in frames:
        image_path = resolve_path(frame["image"], base)
        if digest(image_path) != frame["image_sha256"]:
            raise ValueError("Long-window image changed")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        lab = cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        uv, _, good = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
        width, height = frame["image_size_wh"]
        good &= (uv >= 0).all(axis=1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1)
        coordinates = np.where(good[:, None], uv, -1).reshape(*world.shape[:2], 2).astype(np.float32)
        if digest(resolve_path(frame["ground_support"], base)) != frame["ground_sha256"]:
            raise ValueError("Long-window ground support changed")
        ground = load_array(frame["ground_support"], base) > 0
        valid = good.reshape(world.shape[:2]) & (sample_at(ground.astype(np.float32), coordinates) >= .999)
        observations.append(sample_at(lab[..., 0], coordinates))
        visible.append(valid & arrays["receiver_valid"])
        if len(flow) < len(chain):
            evidence, _ = read_evidence(chain[len(flow)], stats, analysis)
            flow.append(sample_at(evidence["receiver_flow_supported"].astype(np.float32), coordinates) >= .999)
    observations, visible = np.stack(observations), np.stack(visible)
    evidence = long_transition_evidence(observations, visible, flow, middle, roi)
    tracks = {token: np.asarray([values[sample] for sample in samples], np.float64)
              for token, values in states.items() if token not in inconsistent and all(sample in values for sample in samples)}
    return {"frames": frames, "sample_tokens": samples, "middle": middle, **evidence,
        "visible": visible, "tracks": tracks,
        "polygon_cache": {}, "timestamps": [frame["camera_timestamp"] for frame in frames]}


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
