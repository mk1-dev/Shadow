"""Observed shadow-region descriptors and RGB boundary refinement."""
from __future__ import annotations

import cv2
import numpy as np

from shadow_occ.annotation.physics_labels import cast_box
from shadow_occ.annotation.physics_motion import project_world


def _appearance(values):
    histogram = np.concatenate([np.histogram(values[:, ch], bins=8, range=limits)[0]
        for ch, limits in enumerate(((0, 100), (0, 256), (0, 256)))])
    return histogram/max(np.linalg.norm(histogram), 1e-8)


def describe_regions(regions, lab, boundary, flow_m=None, flow_valid=None):
    for region in regions:
        mask = region["mask"]
        values = lab[mask & np.isfinite(lab).all(axis=-1)]
        if len(values):
            region["appearance_descriptor"] = _appearance(values)
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        yy, xx = np.where(mask)
        aspect = (np.ptp(xx)+1)/(np.ptp(yy)+1)
        shape = np.array([min(1., 4*np.pi*mask.sum()/max(perimeter**2, 1)),
                          min(aspect, 1/aspect), float(boundary[mask].mean())], np.float32)
        region["boundary_descriptor"] = shape/max(np.linalg.norm(shape), 1e-8)
        if flow_m is not None:
            observed = mask & flow_valid & np.isfinite(flow_m).all(axis=-1)
            region["flow_correspondence"] = {"valid_cells": int(observed.sum()),
                "displacement_m": np.median(flow_m[observed], axis=0).tolist() if observed.any() else None,
                "meaning": "material correspondence, not shadow boundary velocity"}
    return regions


def refine_image_region(image, seed, ground):
    """Watershed within a small observed RGB ROI, not a rendered silhouette."""
    yy, xx = np.where(seed & ground)
    if not len(xx):
        return {"image_mask_box_xyxy": [0, 0, 0, 0], "image_mask_roi": np.zeros((0, 0), bool)}
    x0, x1 = max(0, int(xx.min())-12), min(seed.shape[1], int(xx.max())+13)
    y0, y1 = max(0, int(yy.min())-12), min(seed.shape[0], int(yy.max())+13)
    section = np.s_[y0:y1, x0:x1]
    seed = seed[section] & ground[section]
    supported = ground[section]
    rgb = image[section]
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[..., 0]
    density = cv2.GaussianBlur(supported.astype(np.float32), (0, 0), 8)
    local = cv2.GaussianBlur(light*supported, (0, 0), 8)/np.maximum(density, 1e-6)
    ring = supported & ~cv2.dilate(seed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if ring.any():
        local = np.maximum(local, np.percentile(light[ring], 70))
    dark = light < .98*np.maximum(local, 1)
    allowed = supported & dark & cv2.dilate(seed.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    core = cv2.erode(seed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & allowed
    if not core.any():
        core = seed & allowed
    result = np.zeros(seed.shape, bool)
    if core.any():
        markers = np.where(allowed, 0, 1).astype(np.int32)
        markers[core] = 2
        cv2.watershed(np.ascontiguousarray(rgb), markers)
        result = ((markers == 2) | core) & allowed
    output = {"image_mask_box_xyxy": [x0, y0, x1, y1], "image_mask_roi": result}
    if result.any():
        values = lab[result].copy()
        values[:, 0] *= 100/255
        output["appearance_descriptor"] = _appearance(values)
    return output


def paste_region_image(region, canvas, value):
    x0, y0, x1, y1 = region["image_mask_box_xyxy"]
    if x1 > x0 and y1 > y0:
        view = canvas[y0:y1, x0:x1]
        view[region["image_mask_roi"] & (view == 0)] = value


def project_region_memory(entries, casters, slot, plane, directions, projection):
    """Alternative motion hypotheses from caster/light geometry, not image flow."""
    result = {}
    for identity, entry in entries.items():
        hint = entry.get("projection_hint")
        if hint is None:
            continue
        states = casters.states(hint["instance_token"])
        previous = hint["frame_slot"]
        if previous not in states or slot not in states:
            continue
        try:
            centers = np.array([cast_box(states[at][1:], plane, directions[hint["light_id"]]).mean(axis=0)
                                for at in (previous, slot)])
        except ValueError:
            continue
        points, _, valid = project_world(centers, projection)
        if not valid.all():
            continue
        shift = points[1]-points[0]
        matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
        result[identity] = cv2.warpAffine(entry["mask"].astype(np.uint8), matrix,
            entry["mask"].shape[::-1], flags=cv2.INTER_NEAREST) > 0
    return result


def memory_state(entry, identity):
    """Small serializable representation; masks stay in frame NPZ/ROI payloads."""
    return {"shadow_track_id": identity, "last_seen_timestamp": entry["timestamp"],
        "last_seen_region_id": entry.get("region_id"), "missed_frames": entry["missed"],
        "visibility_status": entry.get("visibility_status", "unknown"),
        "receiver_visible_fraction": entry.get("receiver_visible_fraction"),
        "receiver_area_cells": int(entry["mask"].sum()),
        "image_mask_box_xyxy": entry.get("image_mask_box_xyxy"),
        "mask_frame_slot": entry.get("frame_slot"),
        "mask_lookup": "shadow_track_id in per-frame BEV NPZ; shadow_region_id_image in per-frame image NPZ",
        "bev_descriptor": None if entry.get("descriptor") is None else entry["descriptor"].tolist(),
        "boundary_descriptor": None if entry.get("boundary_descriptor") is None else entry["boundary_descriptor"].tolist(),
        "appearance_history": entry.get("appearance_history", []),
        "flow_correspondence": entry.get("flow_correspondence"),
        "caster_candidates": entry.get("caster_candidates", []),
        "projection_hint": entry.get("projection_hint"),
        "existence_score": entry.get("existence_score"), "owner_score": entry.get("owner_score"),
        "scores_are_probabilities": False, "missing_masks_are_not_emitted": True}
