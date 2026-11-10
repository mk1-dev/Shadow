"""Bounded planar interpolation of receivers, separate from sparse BEV evidence."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import Delaunay
try:
    from scipy.spatial import QhullError
except ImportError:  # SciPy versions used by the existing local environment.
    from scipy.spatial.qhull import QhullError

from shadow_occ.annotation.physics_motion import project_world
from shadow_occ.annotation.receiver_explanations import _ray_box_entry


@dataclass(frozen=True)
class DenseReceiverConfig:
    maximum_triangle_edge_m: float = 3.0
    full_confidence_edge_m: float = 1.5
    interpolation_confidence: float = .5
    observed_confidence: float = .8


def ground_ray_occlusion(frame, basis, shape, domain):
    """Exclude convex GT volumes *before* the receiver, not whole rectangles."""
    projection = np.asarray(frame["projection_world_to_image"], np.float64)
    camera = np.linalg.solve(projection[:, :3], -projection[:, 3])
    yy, xx = np.where(domain)
    hidden = np.zeros(shape, bool)
    if not len(xx):
        return hidden
    coordinates = np.linalg.solve(projection @ basis, np.stack([xx, yy, np.ones(len(xx))]))
    good = np.isfinite(coordinates).all(axis=0) & (coordinates[2] > 1e-8)
    coordinates[:, good] /= coordinates[2, good]
    world = (basis @ coordinates)[:3].T
    for volume in frame.get("object_volumes_world", []):
        corners = np.asarray(volume["corners_world"], np.float64)
        uv, depth, front = project_world(corners, projection)
        if (depth <= 0).all():
            continue
        eligible = good.copy()
        if front.all():
            eligible &= ((xx >= uv[:, 0].min()-1) & (xx <= uv[:, 0].max()+1)
                         & (yy >= uv[:, 1].min()-1) & (yy <= uv[:, 1].max()+1))
        indices = np.flatnonzero(eligible)
        if len(indices):
            entry = _ray_box_entry(camera, world[indices], corners)
            selected = indices[entry < 1-1e-5]
            hidden[yy[selected], xx[selected]] = True
    return hidden


def dense_receiver_observation(frame, world, sparse_valid, sparse_image, ground_image, basis,
                               config=DenseReceiverConfig()):
    """Interpolate only inside short, observed-ground triangles in metric space.

    Interpolated pixels are estimated support, never promoted to observed LiDAR
    ground. No convex-hull extrapolation, hole dilation or image-darkness test is
    used to define the receiver. Image semantics/depth, when absent, stay absent.
    """
    shape = ground_image.shape
    observed = np.asarray(sparse_image, bool) & np.asarray(ground_image, bool)
    confidence = np.zeros(shape, np.float32)
    confidence[observed] = config.observed_confidence
    points = np.asarray(world)[sparse_valid]
    uv, _, front = project_world(points, frame["projection_world_to_image"])
    height, width = shape
    good = front & np.isfinite(points).all(axis=1) & (uv >= 0).all(axis=1)
    good &= (uv[:, 0] < width) & (uv[:, 1] < height)
    points, uv = points[good], uv[good]
    accepted = 0
    if len(points) >= 3:
        try:
            triangles = Delaunay(points[:, :2]).simplices
        except QhullError:
            triangles = np.empty((0, 3), int)
        for triangle in triangles:
            vertices = points[triangle]
            edges = vertices-np.roll(vertices, 1, axis=0)
            longest = np.linalg.norm(edges, axis=1).max()
            if longest > config.maximum_triangle_edge_m:
                continue
            polygon = np.rint(uv[triangle]).astype(np.int32)
            x0, y0 = polygon.min(axis=0)
            x1, y1 = polygon.max(axis=0)+1
            local = np.zeros((y1-y0, x1-x0), np.uint8)
            cv2.fillConvexPoly(local, polygon-[x0, y0], 1)
            value = config.interpolation_confidence*min(1., config.full_confidence_edge_m/longest)**2
            section = confidence[y0:y1, x0:x1]
            np.maximum(section, np.where(local, value, 0), out=section)
            accepted += 1
    confidence[observed] = config.observed_confidence
    visible = confidence > 0
    occluded = ground_ray_occlusion(frame, np.asarray(basis), shape, visible)
    confidence[occluded] = 0
    state = np.zeros(shape, np.uint8)  # 0=unknown, 1=observed local support, 2=interpolated, 3=occluded
    state[confidence > 0] = 2
    state[observed & ~occluded] = 1
    state[occluded] = 3
    return {"visible": confidence > 0, "confidence": confidence, "state": state,
        "occluded": occluded, "sparse_support": observed & ~occluded,
        "details": {"ground_anchors": len(points), "accepted_metric_triangles": accepted,
            "observed_pixels": int((state == 1).sum()), "interpolated_pixels": int((state == 2).sum()),
            "occluded_pixels": int(occluded.sum()), "dense_ground_is_estimated_not_GT": True,
            "unmodelled_occlusion": "no dense semantic/depth model is inferred from empty LiDAR support"}}
