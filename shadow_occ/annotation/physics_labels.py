"""Conservative three-frame, caster-conditioned WEAK labels, not shadow GT."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shadow_occ.annotation.physics_motion import ground_homography, project_world


@dataclass(frozen=True)
class LabelConfig:
    minimum_area: int = 32
    minimum_change_pixels: int = 4
    minimum_change_fraction: float = 0.03
    minimum_flow_fraction: float = 0.35
    minimum_cover: float = 0.65
    minimum_edge_fit: float = 0.55
    minimum_score: float = 0.65
    caster_margin: float = 0.12
    minimum_motion_m: float = 0.2
    maximum_caster_distance_m: float = 25.0
    minimum_other_tracks: int = 2
    light_tolerance_degrees: float = 16.0

    def validate(self):
        for name in ("minimum_change_fraction", "minimum_flow_fraction", "minimum_cover",
                     "minimum_edge_fit", "minimum_score", "caster_margin"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"Invalid fraction: {name}")
        if (self.minimum_area < 1 or self.minimum_change_pixels < 1 or self.minimum_other_tracks < 2
                or not np.isfinite([self.minimum_motion_m, self.maximum_caster_distance_m,
                                   self.light_tolerance_degrees]).all()
                or self.minimum_motion_m <= 0 or self.maximum_caster_distance_m <= 0
                or not 0 < self.light_tolerance_degrees < 45):
            raise ValueError("Invalid physics-label gates")


def validate_window(left, right):
    """Reject temporal joins that merely share a sample name but not a camera frame."""
    for pair in (left, right):
        if pair.get("protocol") != "physics-motion-evidence-v1" or pair.get("train_approved") is not False:
            raise ValueError("Expected unapproved physics-motion pairs")
        if len(pair.get("frames", [])) != 2:
            raise ValueError("Expected two frames per pair")
        times = [frame["camera_timestamp"] for frame in pair["frames"]]
        if not 0 < (times[1]-times[0])/1e6 <= 0.75:
            raise ValueError("Invalid camera interval")
        if not np.isclose((times[1]-times[0])/1e6, pair["dt_seconds"]):
            raise ValueError("Camera interval differs from pair metadata")
    for key in ("scene_name", "camera", "official_split"):
        if left[key] != right[key]:
            raise ValueError(f"Window crosses {key}")
    if left["next_sample_token"] != right["sample_token"]:
        raise ValueError("Pairs are not consecutive")
    a, b = left["frames"][1], right["frames"][0]
    for key in ("sample_data_token", "camera_timestamp", "image_sha256", "image_size_wh"):
        if a[key] != b[key]:
            raise ValueError(f"Middle camera frame mismatch: {key}")
    if not np.allclose(a["projection_world_to_image"], b["projection_world_to_image"], atol=1e-8, rtol=0):
        raise ValueError("Middle camera projection mismatch")
    pa, pb = np.asarray(left["plane_world"]), np.asarray(right["plane_world"])
    na, nb = np.linalg.norm(pa[:3]), np.linalg.norm(pb[:3])
    if min(na, nb) < 1e-8 or not np.isfinite([pa, pb]).all():
        raise ValueError("Invalid ground planes")
    pa, pb = pa/na, pb/nb
    basis = np.asarray(left["plane_basis_world"])
    origin = basis[:, 2]
    if np.dot(pa[:3], pb[:3]) < np.cos(np.deg2rad(5)) or abs(origin @ (pa-pb)) > 0.3:
        raise ValueError("Window ground planes disagree")


def middle_evidence(left, right, evidence_left, evidence_right):
    """Align two pairwise observations to the SAME middle receiver surface."""
    validate_window(left, right)
    width, height = left["frames"][1]["image_size_wh"]
    h = ground_homography(left["frames"][0]["projection_world_to_image"],
                          left["frames"][1]["projection_world_to_image"],
                          np.asarray(left["plane_basis_world"]))

    def mask(archive, name):
        value = np.asarray(archive[name])
        if value.shape != (height, width) or not np.isfinite(value).all():
            raise ValueError(f"Invalid evidence map: {name}")
        return value > 0

    def aligned(name):
        return cv2.warpPerspective(mask(evidence_left, name).astype(np.uint8), h,
                                   (width, height), flags=cv2.INTER_NEAREST) > 0

    valid = aligned("visible_ground") & mask(evidence_right, "visible_ground")
    onset = aligned("onset_candidate") & valid
    recovery = mask(evidence_right, "recovery_candidate") & valid
    dark_before = onset | aligned("stable_dark_candidate")
    dark_after = recovery | mask(evidence_right, "stable_dark_candidate")
    # Both terms observe darkness at t1, never union t0 darkness with t2 darkness.
    dark = dark_before & dark_after & valid
    return {"visible": valid, "dark": dark, "onset": onset & dark,
            "recovery": recovery & dark,
            "flow_supported": aligned("flow_reliable") & mask(evidence_right, "flow_reliable") & valid,
            "same_patch_cycle": onset & recovery & dark}


def window_tracks(left, right):
    a = {obj["instance_token"]: obj for obj in left["objects"]}
    b = {obj["instance_token"]: obj for obj in right["objects"]}
    tracks, rejected = [], []
    for token in sorted(a.keys() & b.keys()):
        before, after = a[token], b[token]
        try:
            points = np.asarray([before["landmarks_world_t"], before["landmarks_world_next"],
                                 after["landmarks_world_next"]], np.float64)
            shared = np.asarray(after["landmarks_world_t"], np.float64)
            if (points.shape != (3, 9, 3) or shared.shape != (9, 3)
                    or not np.isfinite(points).all() or not np.isfinite(shared).all()
                    or not np.allclose(points[1], shared, atol=1e-5, rtol=0)
                    or before["category"] != after["category"]
                    or before["annotation_token_next"] != after["annotation_token_t"]):
                raise ValueError("inconsistent_track_state")
            tracks.append({"instance_token": token, "category": before["category"],
                           "annotation_token": after["annotation_token_t"], "points": points})
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append({"instance_token": token, "reason": str(exc)})
    return tracks, rejected


def light_bank():
    """World-frame downward ray hypotheses; not estimated/calibrated sunlight."""
    return np.asarray([[np.cos(np.deg2rad(e))*np.cos(np.deg2rad(a)),
                        np.cos(np.deg2rad(e))*np.sin(np.deg2rad(a)), -np.sin(np.deg2rad(e))]
                       for e in (20, 35, 50, 65) for a in range(0, 360, 15)], np.float64)


def cast_box(corners, plane, direction):
    """Project a bounding volume onto a receiver plane, not a true object silhouette."""
    corners, plane, direction = (np.asarray(value, np.float64) for value in (corners, plane, direction))
    if corners.shape != (8, 3) or plane.shape != (4,) or direction.shape != (3,):
        raise ValueError("Invalid box/light/plane shape")
    if not all(np.isfinite(value).all() for value in (corners, plane, direction)):
        raise ValueError("Nonfinite box/light/plane")
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-8 or np.linalg.norm(direction) < 1e-8:
        raise ValueError("Degenerate plane or light")
    plane = plane/norm
    denominator = plane[:3] @ direction
    distances = corners @ plane[:3]+plane[3]
    if denominator >= -1e-4 or distances.min() < -0.35:
        raise ValueError("Light does not reach receiver or box is below receiver")
    # Small ground-fit/box-bottom errors do not create backwards rays.
    origins = corners-np.minimum(distances, 0)[:, None]*plane[:3]
    distances = np.maximum(distances, 0)
    return origins-distances[:, None]/denominator*direction


def footprint_polygon(corners, plane, direction, projection):
    shadow = cast_box(corners, plane, direction)
    pixels, _, valid = project_world(shadow, projection)
    if not valid.all():
        raise ValueError("Shadow volume crosses camera near plane")
    pixels = np.clip(pixels, -10000, 10000)
    return cv2.convexHull(np.rint(pixels).astype(np.int32))


def footprint_mask(corners, plane, direction, projection, shape):
    hull = footprint_polygon(corners, plane, direction, projection)
    result = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(result, hull, 1)
    return result > 0


def receiver_points(mask, projection, basis, limit=256):
    """Sample observed component pixels into a metric ground coordinate frame."""
    yy, xx = np.where(mask)
    if not len(xx):
        return np.empty((0, 3))
    indices = np.linspace(0, len(xx)-1, min(limit, len(xx)), dtype=int)
    matrix = np.asarray(projection) @ np.asarray(basis)
    if np.linalg.cond(matrix) > 1e10:
        raise ValueError("Singular receiver projection")
    coordinates = np.linalg.solve(matrix, np.stack([xx[indices], yy[indices], np.ones(len(indices))]))
    good = np.isfinite(coordinates).all(axis=0) & (coordinates[2] > 1e-8)
    coordinates = coordinates[:, good]/coordinates[2, good]
    return (np.asarray(basis) @ coordinates)[:3].T


def candidate_evidence(mask, maps, config):
    area = int(mask.sum())
    counts = {key: int(np.count_nonzero(mask & maps[key]))
              for key in ("onset", "recovery", "flow_supported", "same_patch_cycle")}
    reasons = []
    if area < config.minimum_area:
        reasons.append("small_component")
    for key in ("onset", "recovery"):
        if counts[key] < max(config.minimum_change_pixels, config.minimum_change_fraction*area):
            reasons.append(f"insufficient_{key}_static_dark_is_not_negative")
    if counts["flow_supported"]/max(area, 1) < config.minimum_flow_fraction:
        reasons.append("insufficient_flow_correspondence")
    return {"area": area, **counts, "reasons": reasons}


def rank_hypotheses(mask, maps, tracks, plane, basis, projection, directions, config, *, polygon_cache=None):
    """Score the SAME light at all three GT states; never equate flow and box velocity."""
    report = candidate_evidence(mask, maps, config)
    if report["reasons"]:
        return [], report
    world = receiver_points(mask, projection, basis)
    if not len(world):
        report["reasons"].append("no_receiver_intersections")
        return [], report
    center = np.median(world, axis=0)
    eligible = [track for track in tracks if
                np.linalg.norm(track["points"][1, 0, :2]-center[:2]) <= config.maximum_caster_distance_m]
    report["nearby_competing_tracks"] = len(eligible)
    yy, xx = np.where(mask)
    x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
    local_mask = mask[y0:y1, x0:x1]
    onset = local_mask & maps["onset"][y0:y1, x0:x1]
    recovery = local_mask & maps["recovery"][y0:y1, x0:x1]
    cache = {} if polygon_cache is None else polygon_cache
    rankings = []
    for light_id, direction in enumerate(directions):
        ranked = []
        for track in eligible:
            points = track["points"]
            # Large turns can change shadows too, but are deliberately deferred here.
            if min(np.linalg.norm(points[1, 0]-points[0, 0]),
                   np.linalg.norm(points[2, 0]-points[1, 0])) < config.minimum_motion_m:
                continue
            try:
                key = (track["instance_token"], light_id)
                if key not in cache:
                    cache[key] = [footprint_polygon(p[1:], plane, direction, projection) for p in points]
                footprints = []
                for polygon in cache[key]:
                    footprint = np.zeros(local_mask.shape, np.uint8)
                    cv2.fillConvexPoly(footprint, polygon-np.array([[[x0, y0]]], np.int32), 1)
                    footprints.append(footprint > 0)
            except ValueError:
                continue
            cover = float(np.count_nonzero(local_mask & footprints[1])/report["area"])
            onset_fit = float(np.count_nonzero(onset & footprints[1] & ~footprints[0])/onset.sum())
            recovery_fit = float(np.count_nonzero(recovery & footprints[1] & ~footprints[2])/recovery.sum())
            score = 0.4*cover+0.3*onset_fit+0.3*recovery_fit
            ranked.append({"instance_token": track["instance_token"], "light_id": light_id,
                           "score": score, "cover": cover, "onset_fit": onset_fit,
                           "recovery_fit": recovery_fit})
        ranked.sort(key=lambda item: (-item["score"], item["instance_token"]))
        if ranked:
            # Include runner-up even when it fails its own absolute acceptance gate.
            best = dict(ranked[0])
            best["caster_margin"] = best["score"]-(ranked[1]["score"] if len(ranked) > 1 else 0)
            best["competing_caster"] = ranked[1]["instance_token"] if len(ranked) > 1 else None
            best["passes"] = bool(best["cover"] >= config.minimum_cover
                and min(best["onset_fit"], best["recovery_fit"]) >= config.minimum_edge_fit
                and best["score"] >= config.minimum_score and best["caster_margin"] >= config.caster_margin)
            rankings.append(best)
    if not any(item["passes"] for item in rankings):
        report["reasons"].append("no_unique_three_frame_projection_fit")
    return rankings, report


def light_support_index(pool):
    """Deduplicate votes once, instead of rescanning every hypothesis for every region."""
    index = {}
    for other in pool:
        frames = tuple(sorted(set(other["frame_tokens"])))
        for vote in other["hypotheses"]:
            if vote["passes"]:
                key = (other["scene_name"], other["official_split"], vote["light_id"])
                index.setdefault(key, {}).setdefault(vote["instance_token"], set()).add(frames)
    return index


def select_with_independent_light(candidate, pool, directions, config, *, support_index=None):
    """Leave the entire proposed caster track OUT of its scene-light support."""
    possible = [item for item in candidate["hypotheses"] if item["passes"]]
    index = light_support_index(pool) if support_index is None else support_index
    query_frames = set(candidate["frame_tokens"])
    supported = []
    for item in possible:
        others = set()
        compatible = np.flatnonzero(directions @ directions[item["light_id"]] >= np.cos(np.deg2rad(config.light_tolerance_degrees)))
        for light_id in compatible:
            key = (candidate["scene_name"], candidate["official_split"], int(light_id))
            for token, windows in index.get(key, {}).items():
                if token != item["instance_token"] and token not in others:
                    if any(query_frames.isdisjoint(frames) for frames in windows):
                        others.add(token)
        if len(others) >= config.minimum_other_tracks:
            supported.append({**item, "independent_light_tracks": sorted(others)})
    if not supported:
        return None, "insufficient_leave_track_and_frame_out_light_support"
    supported.sort(key=lambda item: (-item["score"], item["light_id"]))
    best = supported[0]
    competing = [item["score"] for item in supported if item["instance_token"] != best["instance_token"]]
    if competing and best["score"]-max(competing) < config.caster_margin:
        return None, "caster_ambiguous_across_light_hypotheses"
    return best, "weak_physics_supported_not_verified"
