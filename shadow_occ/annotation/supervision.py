from __future__ import annotations

import hashlib
import random

import cv2
import numpy as np

from shadow_occ.annotation.data import SOURCES
from shadow_occ.annotation.joint import annotation
from shadow_occ.annotation.offbev import ASSOCIATION_GT


def competing_association(shadow_xy, casters, max_attachment=1.5, min_margin=1.0):
    """A unique footprint attachment is weak evidence, never association GT."""
    shadow = cv2.convexHull(np.asarray(shadow_xy, np.float32).reshape(-1, 2))
    candidates = []
    for caster in casters:
        corners = caster.get("caster_corners_ego")
        if corners is None:
            continue
        footprint = cv2.convexHull(np.asarray(corners, np.float32)[:, :2])
        if len(shadow) < 3 or len(footprint) < 3:
            continue
        intersection, _ = cv2.intersectConvexConvex(shadow, footprint)
        distance = 0.0 if intersection > 0 else min(
            [abs(cv2.pointPolygonTest(footprint, tuple(map(float, point)), True)) for point in shadow[:, 0]]
            + [abs(cv2.pointPolygonTest(shadow, tuple(map(float, point)), True)) for point in footprint[:, 0]]
        )
        candidates.append({"annotation_token": caster["annotation_token"], "attachment_distance_m": distance})
    candidates.sort(key=lambda item: (item["attachment_distance_m"], item["annotation_token"]))
    status, owner = "unsupported", None
    if candidates and candidates[0]["attachment_distance_m"] <= max_attachment:
        if len(candidates) == 1 or candidates[1]["attachment_distance_m"]-candidates[0]["attachment_distance_m"] >= min_margin:
            status, owner = "weak_geometry_supported", candidates[0]["annotation_token"]
        else:
            status = "ambiguous"
    return {"association_status": status, "associated_annotation_token": owner,
            "association_source": "geometry_weak" if owner else "unresolved",
            "association_candidates": candidates,
            "association_evidence": {"method": "convex_footprint_attachment_competition", "max_attachment_m": max_attachment,
                                     "min_margin_m": min_margin, "scope": "exported vehicle candidates only; not exhaustive", "calibrated_probability": False}}


def visible_target(row):
    source = row.get("association_source", "unresolved")
    supported = source in ASSOCIATION_GT or (
        source in {"geometry_weak", "temporal_weak"}
        and row.get("association_status") in {"weak_geometry_supported", "temporal_supported"}
        and row.get("associated_annotation_token") == row.get("annotation_token")
        and row.get("annotation_token") is not None
    )
    box, context = row.get("caster_bbox_xyxy"), row.get("context_crop_xyxy")
    if not supported or not box or not context or row.get("caster_source") != "annotation_gt":
        return annotation("YES", "unresolved")
    x0, y0, x1, y1 = context
    if x1 <= x0 or y1 <= y0 or not (x0 <= box[0] < box[2] <= x1 and y0 <= box[1] < box[3] <= y1):
        return annotation("YES", "unresolved")
    normalized = [round(1000*(box[0]-x0)/(x1-x0)), round(1000*(box[1]-y0)/(y1-y0)),
                  round(1000*(box[2]-x0)/(x1-x0)), round(1000*(box[3]-y0)/(y1-y0))]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return annotation("YES", "unresolved")
    return annotation("YES", "visible", category=row.get("caster_category", "unknown"), bbox=normalized)


def exterior_target(row):
    # Object existence/association metadata alone must not manufacture shadow YES.
    decision = row.get("shadow_label") if row.get("shadow_label_source") in SOURCES else None
    if decision == "NO":
        return annotation("NO", "none")
    if decision != "YES":
        return annotation("UNCERTAIN", "unresolved") if row.get("answer") is not None else None
    source = row.get("association_source", row.get("label_source", "unlabeled"))
    if row.get("selected_candidate_id") and source in ASSOCIATION_GT | {"geometry_weak", "temporal_weak"}:
        return annotation("YES", "outside_bev", candidate_id=row["selected_candidate_id"])
    return annotation("YES", "unresolved")


def permute_candidates(candidates, identifier, seed):
    """Randomize order AND IDs independently of the associated GT object."""
    bank = [dict(item) for item in candidates]
    if len({item["id"] for item in bank}) != len(bank):
        raise ValueError("Duplicate exterior candidate identifiers")
    rng = random.Random(hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest())
    rng.shuffle(bank)
    names = [f"H{i+1}" for i in range(len(bank))]
    rng.shuffle(names)
    mapping = {}
    for item, name in zip(bank, names):
        mapping[item["id"]] = name
        item["id"] = name
    return bank, mapping
