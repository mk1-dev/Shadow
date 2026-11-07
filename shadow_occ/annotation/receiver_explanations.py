"""Competing explanations for appearance on an observed ground receiver."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from shadow_occ.annotation.data import digest, resolve_path
from shadow_occ.annotation.physics_motion import project_world, sample_at


PROTOCOL = "receiver-explanations-v1"
OBSERVATION_STATES = {0: "unknown", 1: "observed_ground_receiver", 2: "annotated_volume_support"}
SCORE_CONTRACT = {
    "projection": {"coverage": .65, "boundary": .2, "dark_fraction": .15, "lit_conflict_penalty": .35},
    "shadow": {"projection_strength": .65, "boundary_transition": .35},
    "projection_gain_scale": .12, "transition_fraction_scale": .1,
    "material_ratio_range_scale": .1, "near_projection_margin": .12,
}


@dataclass(frozen=True)
class CompetitionConfig:
    control_shift_m: float = 3.0
    ring_m: float = 2.0
    minimum_projection_gain: float = 0.05
    minimum_winner_margin: float = 0.05
    darkness_ratio: float = 0.88
    minimum_lit_ratio: float = 0.97
    appearance_prior_weight: float = 0.2

    def validate(self):
        if (not np.isfinite(list(asdict(self).values())).all()
                or min(self.control_shift_m, self.ring_m) <= 0
                or not 0 < self.darkness_ratio < self.minimum_lit_ratio <= 1
                or not 0 <= self.minimum_projection_gain < 1
                or not 0 <= self.minimum_winner_margin < 1
                or not 0 <= self.appearance_prior_weight <= 1):
            raise ValueError("Invalid receiver competition configuration")


def attach_appearance_prior(arrays, frames, lookup, base):
    """Consume exported SILT/other baseline probabilities, never its hard labels."""
    world = arrays["receiver_world_xyz"]
    probabilities = np.full((3, *world.shape[:2]), np.nan, np.float32)
    sources = []
    for slot, frame in enumerate(frames):
        row = lookup.get(frame["sample_data_token"])
        if row is None:
            continue
        if row["image_sha256"] != frame["image_sha256"]:
            raise ValueError("Baseline probability/image fingerprint mismatch")
        path = resolve_path(row["probability"], base)
        with np.load(path, allow_pickle=False) as archive:
            probability = np.asarray(archive["probability"], np.float32)
        width, height = frame["image_size_wh"]
        if (probability.shape != (height, width) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise ValueError("Baseline probabilities must be finite [0,1] in exported frame coordinates")
        pixels, _, valid = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
        valid &= ((pixels >= 0).all(axis=1) & (pixels[:, 0] <= width-1) & (pixels[:, 1] <= height-1))
        coordinates = np.where(valid[:, None], pixels, -1).reshape(*world.shape[:2], 2).astype(np.float32)
        sampled = sample_at(probability, coordinates)
        observed = valid.reshape(world.shape[:2]) & arrays["receiver_valid"]
        probabilities[slot] = np.where(observed, sampled, np.nan)
        sources.append({"sample_data_token": frame["sample_data_token"], "model_id": row["model_id"],
                        "probability_sha256": digest(path), "image_sha256": frame["image_sha256"]})
    arrays["external_shadow_probability"] = probabilities
    return arrays, {"sources": sources, "role": "appearance score only; cannot establish geometry or caster identity"}


def box_ray_evidence(world, frames, pixels, observed_flows, observed_flow_valid=None):
    """Intersect camera-to-ground segments with all-class convex GT volumes.

    Front surface motion is an affine BOX-SURFACE proxy, not dense object flow.
    Annotation-time fallback boxes exclude geometry but never supply motion.
    """
    shape = world.shape[:2]
    receivers = world.reshape(-1, 3)
    support = np.zeros((3, *shape), bool)
    flow_valid = np.zeros((2, *shape), bool)
    expected = np.full((2, *shape, 2), np.nan, np.float32)
    residual = np.full((2, *shape), np.nan, np.float32)
    for slot, frame in enumerate(frames):
        projection = np.asarray(frame["projection_world_to_image"], np.float64)
        camera = np.linalg.solve(projection[:, :3], -projection[:, 3])
        directions = receivers-camera
        uv = pixels[slot].reshape(-1, 2)
        _, _, in_front = project_world(receivers, projection)
        width, height = frame["image_size_wh"]
        in_front &= ((uv >= 0).all(axis=1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1))
        nearest = np.full(len(receivers), np.inf)
        next_volumes = ({value["instance_token"]: value for value in frames[slot+1]["object_volumes_world"]
                         if value.get("instance_token")} if slot < 2 else {})
        for volume in frame["object_volumes_world"]:
            corners = np.asarray(volume["corners_world"], np.float64)
            corner_uv, corner_depth, valid = project_world(corners, projection)
            if (corner_depth <= 0).all():
                continue
            eligible = in_front.copy()
            if valid.all():
                eligible &= ((uv >= corner_uv.min(axis=0)-1).all(axis=1)
                             & (uv <= corner_uv.max(axis=0)+1).all(axis=1))
            indices = np.flatnonzero(eligible)
            if not len(indices):
                continue
            planes = ConvexHull(corners).equations
            origin = planes[:, :3] @ camera+planes[:, 3]
            rate = directions[indices] @ planes[:, :3].T
            parallel = np.abs(rate) < 1e-10
            crossing = np.divide(-origin[None], rate, out=np.zeros_like(rate), where=~parallel)
            enter = np.maximum(0, np.max(np.where(rate < -1e-10, crossing, -np.inf), axis=1))
            leave = np.minimum(1, np.min(np.where(rate > 1e-10, crossing, np.inf), axis=1))
            hit = ((enter <= leave) & (enter < 1-1e-5)
                   & ~(parallel & (origin[None] > 1e-8)).any(axis=1)
                   & (enter < nearest[indices]))
            indices, fraction = indices[hit], enter[hit]
            if not len(indices):
                continue
            nearest[indices] = fraction
            support[slot].flat[indices] = True
            if slot == 2:
                continue
            # A nearer fallback/non-track box must clear any farther motion proxy.
            flow_valid[slot].flat[indices] = False
            expected[slot].reshape(-1, 2)[indices] = np.nan
            residual[slot].flat[indices] = np.nan
            following = next_volumes.get(volume.get("instance_token"))
            if (volume.get("source") != "camera_time_interpolated_box" or following is None
                    or following.get("source") != "camera_time_interpolated_box"):
                continue
            transform = np.linalg.lstsq(np.c_[corners, np.ones(8)],
                np.asarray(following["corners_world"], np.float64), rcond=None)[0]
            surface = camera+fraction[:, None]*directions[indices]
            predicted_world = np.c_[surface, np.ones(len(surface))] @ transform
            following_uv, _, good = project_world(predicted_world, frames[slot+1]["projection_world_to_image"])
            next_width, next_height = frames[slot+1]["image_size_wh"]
            good &= ((following_uv >= 0).all(axis=1) & (following_uv[:, 0] <= next_width-1)
                     & (following_uv[:, 1] <= next_height-1))
            observed = observed_flows[slot].reshape(-1, 2)[indices]
            good &= np.isfinite(observed).all(axis=1)
            if observed_flow_valid is not None:
                good &= observed_flow_valid[slot].reshape(-1)[indices]
            chosen = indices[good]
            displacement = following_uv[good]-uv[chosen]
            expected[slot].reshape(-1, 2)[chosen] = displacement
            residual[slot].flat[chosen] = np.linalg.norm(observed[good]-displacement, axis=1)
            flow_valid[slot].flat[chosen] = True
    return {"physical_ray_volume_support": support, "box_surface_flow_valid": flow_valid,
            "box_surface_expected_flow_px": expected, "box_surface_flow_residual_px": residual}


def receiver_evidence(arrays, *, light_threshold=6.0, darkness_ratio=0.88):
    """Ground-registered appearance boundaries are separate from material flow."""
    receiver = arrays["receiver_valid"]
    ratio = arrays["appearance_local_ratio"]
    raw = arrays["appearance_L_raw"]
    comparable = np.broadcast_to(arrays.get("temporal_exposure_valid", False), receiver.shape)
    delta = arrays["temporal_illumination_change_L"]
    dark = (ratio < darkness_ratio) & receiver[None]
    boundaries = np.stack([value & ~cv2.erode(value.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                           for value in dark])
    formed = dark[1] & ~dark[0] & comparable & (delta[0] <= -light_threshold)
    recovered = dark[1] & ~dark[2] & comparable & (delta[1] >= light_threshold)
    finite_ratio = np.isfinite(ratio).all(axis=0) & receiver
    persistence = np.exp(-np.ptp(np.nan_to_num(ratio, nan=0), axis=0)/SCORE_CONTRACT["material_ratio_range_scale"])
    texture, texture_valid = [], []
    gradients = []
    for value in raw:
        value = np.log(np.maximum(np.nan_to_num(value, nan=0), 1))
        gradients.append(np.stack([cv2.Sobel(value, cv2.CV_32F, 1, 0)/8,
                                   cv2.Sobel(value, cv2.CV_32F, 0, 1)/8], axis=-1))
    # Do not count support boundaries or changing shadow edges as material texture.
    interior = cv2.erode(receiver.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for first, second in zip(gradients, gradients[1:]):
        norms = np.linalg.norm(first, axis=-1)*np.linalg.norm(second, axis=-1)
        valid = interior & (norms > 1e-4) & ~boundaries.any(axis=0)
        cosine = (first*second).sum(axis=-1)/np.maximum(norms, 1e-6)
        texture.append(np.where(valid, np.clip(cosine, -1, 1), np.nan))
        texture_valid.append(valid)
    entity = arrays["physical_box_footprint"].any(axis=0) | arrays["physical_ray_volume_support"].any(axis=0)
    observation = np.zeros(receiver.shape, np.uint8)
    observation[receiver] = 1
    observation[entity] = 2
    return {"receiver_observation_state": observation,
            "receiver_dark_by_frame": dark, "receiver_appearance_boundary": boundaries,
            "receiver_boundary_formation": formed, "receiver_boundary_recovery": recovered,
            "receiver_material_persistence": np.where(finite_ratio, persistence, np.nan).astype(np.float32),
            "receiver_texture_cosine": np.stack(texture).astype(np.float32),
            "receiver_texture_valid": np.stack(texture_valid)}


def projection_score(mask, footprint, domain, dark, lit):
    covered = mask & footprint
    cover = float(covered.sum()/max(mask.sum(), 1))
    supported = footprint & domain
    precision = float((supported & dark).sum()/max(supported.sum(), 1))
    conflict = float((supported & lit).sum()/max(supported.sum(), 1))
    kernel = np.ones((3, 3), np.uint8)
    edge = mask & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    border = footprint & ~cv2.erode(footprint.astype(np.uint8), kernel).astype(bool)
    near = cv2.dilate(border.astype(np.uint8), kernel).astype(bool)
    edge_fit = float((edge & near).sum()/max(edge.sum(), 1))
    weights = SCORE_CONTRACT["projection"]
    score = float(np.clip(weights["coverage"]*cover+weights["boundary"]*edge_fit
        +weights["dark_fraction"]*precision-weights["lit_conflict_penalty"]*conflict, 0, 1))
    return {"score": score, "cover": cover, "boundary_fit": edge_fit,
            "observed_dark_fraction": precision, "observed_lit_conflict": conflict}


def explain_component(mask, arrays, hypotheses, polygons, resolution, config=None):
    """Compare real and shifted casters with the SAME track/light search budget."""
    config = config or CompetitionConfig()
    config.validate()
    area = max(int(mask.sum()), 1)
    ground = arrays["receiver_valid"]
    observation = arrays["receiver_observation_state"]
    ground_fraction = float((mask & ground).sum()/area)
    entity_fraction = float((mask & (observation == 2)).sum()/area)
    unknown_fraction = float((mask & (observation == 0)).sum()/area)
    radius = max(1, int(np.ceil(config.ring_m/resolution)))
    domain = cv2.dilate(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool) & ground
    ratio = arrays["appearance_local_ratio"]
    lightness = arrays["appearance_L_raw"]
    dark = (ratio[1] < config.darkness_ratio) & ground
    reference = float(np.median(lightness[1][mask & ground])) if (mask & ground).any() else 0.0
    lit = ((ratio >= config.minimum_lit_ratio).all(axis=0) & ground
           & (lightness[1] >= reference+6))
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("Receiver competition requires a nonempty component")
    y0, y1 = max(0, int(yy.min())-radius-1), min(mask.shape[0], int(yy.max())+radius+2)
    x0, x1 = max(0, int(xx.min())-radius-1), min(mask.shape[1], int(xx.max())+radius+2)
    section = np.s_[y0:y1, x0:x1]
    local_mask, local_domain, local_dark, local_lit = (value[section] for value in (mask, domain, dark, lit))
    evaluations, controls = [], []
    shift = max(1, int(round(config.control_shift_m/resolution)))
    offsets = [(shift, 0), (-shift, 0), (0, shift), (0, -shift)]
    explanations = [value for parent in hypotheses for value in [parent, *parent.get("alternatives", [])]]
    for item in explanations:
        key = (item["instance_token"], item["light_id"])
        footprint = np.zeros(local_mask.shape, np.uint8)
        polygon = polygons[key][1]-np.array([[[x0, y0]]], np.int32)
        cv2.fillConvexPoly(footprint, polygon, 1)
        values = projection_score(local_mask, footprint > 0, local_domain, local_dark, local_lit)
        evaluations.append({"instance_token": key[0], "light_id": key[1],
                            "projection_fit_passes": bool(item.get("shadow_fit_passes")), **values})
        for dx, dy in offsets:
            control = np.zeros(local_mask.shape, np.uint8)
            cv2.fillConvexPoly(control, polygon+np.array([[[dx, dy]]], np.int32), 1)
            controls.append({"instance_token": key[0], "light_id": key[1], "offset_cells": [dx, dy],
                             **projection_score(local_mask, control > 0, local_domain, local_dark, local_lit)})
    fits = [value for value in evaluations if value["projection_fit_passes"]]
    best = max(fits, key=lambda value: value["score"]) if fits else None
    # Each displacement arm searches every real track/light; compare its best to the real best.
    best_control = max(controls, key=lambda value: value["score"]) if controls else None
    gain = best["score"]-best_control["score"] if best is not None and best_control is not None else 0.0
    motion_gain = max((float(value.get("temporal_gain", 0)) for value in explanations
                       if best and value["instance_token"] == best["instance_token"]
                       and value["light_id"] == best["light_id"] and value.get("shadow_passes")), default=0.0)
    projection_strength = (best["score"]*float(np.clip(
        max(gain, motion_gain)/SCORE_CONTRACT["projection_gain_scale"], 0, 1))) if best else 0.0
    events = arrays["receiver_boundary_formation"] | arrays["receiver_boundary_recovery"]
    transition = float(np.clip((events & mask).sum()/area/SCORE_CONTRACT["transition_fraction_scale"], 0, 1))
    persistence_values = arrays["receiver_material_persistence"][mask & ground]
    persistence_values = persistence_values[np.isfinite(persistence_values)]
    persistence = float(persistence_values.mean()) if len(persistence_values) else 0.0
    scores = {"physical_entity": entity_fraction,
              "ground_material": ground_fraction*persistence*(1-projection_strength),
              "cast_shadow": ground_fraction*(SCORE_CONTRACT["shadow"]["projection_strength"]*projection_strength
                  +SCORE_CONTRACT["shadow"]["boundary_transition"]*transition) if best else 0.0,
              "unknown": unknown_fraction}
    mature_evidence = None
    mature_unexplained_score = None
    if "mature_entity_probability" in arrays:
        mature_domain = arrays["mature_inference_domain"][1] & mask
        probability = arrays["mature_entity_probability"][1]
        motion_fit = arrays["mature_entity_motion_fit"][1]
        if mature_domain.any():
            available = mature_domain & np.isfinite(motion_fit)
            confidence = probability[mature_domain]
            effective_fit = np.where(np.isfinite(motion_fit[mature_domain]), motion_fit[mature_domain], .5)
            entity_score = float(np.mean(confidence*effective_fit))
            scores["physical_entity"] = max(scores["physical_entity"], entity_score)
            mature_unexplained_score = float(1-confidence.mean())
            mature_evidence = {"predicted_entity_mean": float(confidence.mean()),
                "entity_motion_score": entity_score, "motion_observed_fraction": float(available.sum()/area),
                "inference_domain_fraction": float(mature_domain.sum()/area),
                "unexplained_receiver_fraction": float((mature_domain & (probability < .25)).sum()/area),
                "absence_is_free_space": False, "gt_used": False}
        else:
            mature_evidence = {"inference_domain_fraction": 0.0, "absence_is_free_space": False, "gt_used": False}
    prior = arrays.get("external_shadow_probability")
    prior_mean, context_mean, prior_contrast, prior_coverage = None, None, 0.0, 0.0
    if prior is not None:
        observed_prior = np.isfinite(prior[1]) & ground
        core = mask & observed_prior
        context = domain & ~mask & observed_prior
        prior_coverage = float(core.sum()/area)
        if core.any():
            prior_mean = float(prior[1][core].mean())
        if context.any():
            context_mean = float(prior[1][context].mean())
        if prior_mean is not None and context_mean is not None and prior_coverage >= .5:
            prior_contrast = prior_mean-context_mean
            scores["cast_shadow"] = float(np.clip(scores["cast_shadow"]
                +config.appearance_prior_weight*prior_contrast*ground_fraction*projection_strength, 0, 1))
    ordered = sorted(scores, key=scores.get, reverse=True)
    winner = ordered[0]
    margin = scores[winner]-scores[ordered[1]]
    discriminative = max(gain, motion_gain) >= config.minimum_projection_gain
    accepted = bool(best and winner == "cast_shadow" and discriminative
                    and margin >= config.minimum_winner_margin)
    texture = arrays["receiver_texture_cosine"][:, mask]
    texture = texture[np.isfinite(texture)]
    ratio_values = ratio[1][mask & ground]
    darkness = float(np.clip(1-np.mean(ratio_values), 0, 1)) if len(ratio_values) else 0.0
    motion_diagnostics = {}
    for name in ("flow_residual_px", "box_surface_flow_residual_px"):
        values = arrays[name][:, mask] if name in arrays else np.empty(0)
        values = values[np.isfinite(values)]
        motion_diagnostics[name+"_median"] = float(np.median(values)) if len(values) else None
    for name in ("mature_object_flow_residual_px", "mature_ground_flow_residual_px"):
        values = arrays[name][1][mask] if name in arrays else np.empty(0)
        values = values[np.isfinite(values)]
        motion_diagnostics[name+"_median"] = float(np.median(values)) if len(values) else None
    return {"protocol": PROTOCOL, "winner": winner, "winner_margin": float(margin), "scores": scores,
            "shadow_supported": accepted, "ground_fraction": ground_fraction,
            "entity_fraction": entity_fraction, "unknown_fraction": unknown_fraction,
            "transition_fraction": float((events & mask).sum()/area), "material_persistence": persistence,
            "texture_cosine_mean": float(texture.mean()) if len(texture) else None,
            "texture_observed_fraction": float(arrays["receiver_texture_valid"][:, mask].mean()),
            "best_projection": best, "best_shifted_control": best_control,
            "projection_control_gain": float(gain), "trajectory_control_gain": float(motion_gain),
            "hypothesis_scores": evaluations,
            "control_search": {"offsets_cells": offsets, "real_hypotheses": len(evaluations),
                               "hypotheses_per_offset": len(evaluations)},
            "baseline_scores": {"appearance": darkness,
                "geometry_flow": darkness*ground_fraction*float(arrays["shadow_flow_supported"][mask].mean()),
                "projection_coverage": max((value["cover"] for value in evaluations), default=0.0),
                "external_shadow_model": prior_mean,
                "mature_3d_unexplained": darkness*mature_unexplained_score if mature_unexplained_score is not None else None,
                "receiver_competition": scores["cast_shadow"]-max(scores[key] for key in scores if key != "cast_shadow")},
            "appearance_prior": {"mean": prior_mean, "context_mean": context_mean,
                "contrast": prior_contrast, "coverage": prior_coverage,
                "requires_discriminative_projection": True},
            "motion_diagnostics": motion_diagnostics,
            "mature_3d": mature_evidence,
            "config": asdict(config), "score_contract": SCORE_CONTRACT, "scores_are_probabilities": False}


def apply_competition(decision, evidence, hypotheses):
    baseline = {key: decision[key] for key in ("shadow_accepted", "reason", "existence_score") if key in decision}
    result = {**decision, "baseline_projection_decision": baseline, "receiver_competition": evidence}
    if not decision["shadow_accepted"] or not evidence["shadow_supported"]:
        return {**result, "shadow_accepted": False, "selected": None, "association": None,
                "reason": "receiver_competition_"+evidence["winner"]}
    best = evidence["best_projection"]
    explanations = [value for parent in hypotheses for value in [parent, *parent.get("alternatives", [])]]
    selected = next(value for value in explanations if value["instance_token"] == best["instance_token"]
                    and value["light_id"] == best["light_id"])
    near = {(value["instance_token"], value["light_id"]) for value in evidence["hypothesis_scores"]
            if value["projection_fit_passes"]
            and best["score"]-value["score"] < SCORE_CONTRACT["near_projection_margin"]}
    plausible = [value for value in explanations if (value["instance_token"], value["light_id"]) in near]
    association = result["association"]
    if association is not None and (association["instance_token"] != selected["instance_token"]
                                   or len({token for token, _ in near}) != 1):
        association = None
    return {**result, "selected": selected, "association": association, "plausible_hypotheses": plausible,
            "reason": "receiver_competition_shadow_support", "existence_score": best["score"],
            "existence_evidence": "moving_projection" if selected.get("motion_supported") else "static_projection"}
