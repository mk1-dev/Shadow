from __future__ import annotations

import json

import cv2
import numpy as np


ASSOCIATION_GT = {"manual_caster_association_gt", "synthetic_caster_association_gt"}


def footprint_status(corners, bounds) -> str:
    xmin, ymin, xmax, ymax = bounds
    polygon = cv2.convexHull(np.asarray(corners, dtype=np.float32)[:, :2])
    rectangle = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
    intersection, _ = cv2.intersectConvexConvex(polygon, rectangle)
    area = float(cv2.contourArea(polygon))
    if intersection <= 1e-6:
        return "outside"
    return "inside" if intersection >= area-1e-4 else "crossing_boundary"


def contains(candidate, point) -> bool:
    x0, y0, x1, y1 = candidate["xyxy_ego"]
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def boxes_overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def boundary_hypotheses(xy, bounds, *, near_boundary=8.0, distances=(0.0, 4.0, 8.0, 16.0), angular_tolerance=15.0):
    """Extrapolate BOTH shadow-axis directions; never inspect caster annotations."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        return []
    xmin, ymin, xmax, ymax = bounds
    if np.min(np.c_[xy[:, 0]-xmin, xmax-xy[:, 0], xy[:, 1]-ymin, ymax-xy[:, 1]]) > near_boundary:
        return []
    center = xy.mean(axis=0)
    covariance = np.cov((xy-center).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, np.argmax(values)]
    hypotheses = []
    for direction in (axis, -axis):
        exits = []
        for dimension, low, high, low_name, high_name in ((0, xmin, xmax, "x_min", "x_max"), (1, ymin, ymax, "y_min", "y_max")):
            if abs(direction[dimension]) < 1e-8:
                continue
            for boundary, side in ((low, low_name), (high, high_name)):
                distance = (boundary-center[dimension])/direction[dimension]
                point = center+direction*distance
                if distance > 0 and xmin-1e-5 <= point[0] <= xmax+1e-5 and ymin-1e-5 <= point[1] <= ymax+1e-5:
                    exits.append((distance, point, side))
        if not exits:
            continue
        distance_to_edge, edge, side = min(exits, key=lambda item: item[0])
        # A component far from this side supplies no supported extrapolation.
        endpoint = xy[np.argmax((xy-center)@direction)]
        if np.linalg.norm(endpoint-edge) > near_boundary:
            continue
        for start, end in zip(distances[:-1], distances[1:]):
            lateral = max(1.5, end*np.tan(np.deg2rad(angular_tolerance)))
            perpendicular = np.array([-direction[1], direction[0]])
            vertices = np.asarray([edge+direction*d+perpendicular*s*lateral for d in (start, end) for s in (-1, 1)])
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            if side == "x_min":
                upper[0] = min(upper[0], xmin-1e-3)
            elif side == "x_max":
                lower[0] = max(lower[0], xmax+1e-3)
            elif side == "y_min":
                upper[1] = min(upper[1], ymin-1e-3)
            else:
                lower[1] = max(lower[1], ymax+1e-3)
            if np.any(upper <= lower):
                continue
            hypotheses.append({"id": f"H{len(hypotheses)+1}", "side": side, "xyxy_ego": np.r_[lower, upper].tolist(), "center_xy_ego": ((lower+upper)/2).tolist(), "outside_distance_m": [start, end], "geometry_score": float(1/(1+distance_to_edge+0.5*(start+end))), "uncertainty": "shadow-axis extrapolation; light direction and caster height are unknown"})
    return hypotheses


def answer_text(candidate_id=None) -> str:
    return json.dumps({"decision": "POTENTIAL" if candidate_id is not None else "ABSTAIN", "candidate_id": candidate_id}, separators=(",", ":"))


def choices(row):
    return [(None, answer_text())]+[(candidate["id"], answer_text(candidate["id"])) for candidate in row["candidates"]]


def localization_metrics(rows: list[dict]) -> dict:
    by_source = {}
    for source in sorted({row["label_source"] for row in rows}):
        group = [row for row in rows if row["label_source"] == source]
        labeled = [row for row in group if row.get("target_decision") in {"POTENTIAL", "ABSTAIN"}]
        positive = [row for row in labeled if row["target_decision"] == "POTENTIAL"]
        abstain = [row for row in labeled if row["target_decision"] == "ABSTAIN"]
        matched = [row for row in positive if row.get("selected_candidate_id") in row.get("valid_candidate_ids", [])]
        baseline = sum(row.get("geometry_candidate_id") in row.get("valid_candidate_ids", []) for row in positive)
        errors = [float(np.linalg.norm(np.asarray(row["selected_center_xy_ego"])-np.asarray(row["target_center_xy_ego"]))) for row in matched if row.get("target_center_xy_ego") is not None]
        ratio = lambda a, b: float(a/b) if b else None
        by_source[source] = {"examples": len(group), "labeled": len(labeled), "positive_associations": len(positive), "explicit_abstention_targets": len(abstain), "candidate_bank_recall": ratio(sum(bool(row.get("valid_candidate_ids")) for row in positive), len(positive)), "region_localization_recall": ratio(len(matched), len(positive)), "geometry_top1_recall": ratio(baseline, len(positive)), "positive_hypothesis_rate": ratio(sum(row.get("selected_candidate_id") is not None for row in positive), len(positive)), "abstention_accuracy": ratio(sum(row.get("selected_candidate_id") is None for row in abstain), len(abstain)), "matched_cell_center_error_m": float(np.mean(errors)) if errors else None}
    return {"examples": len(rows), "by_association_source": by_source, "metric_scope": "caster association and outside-BEV region hypotheses; not occupancy mIoU or a precise 3D reconstruction"}
