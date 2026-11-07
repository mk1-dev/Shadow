"""Training labels from selected BEV receivers and caster/light trajectories."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from shadow_occ.annotation.physics_labels import footprint_mask


PROTOCOL = "bev-flow-annotation-v2"
TEACHER_PROTOCOLS = {PROTOCOL, "bev-flow-annotation-v1"}


def endpoint_negative_masks(arrays, positive_core, tracks, plane, directions, explanations, *,
                            minimum_lit_ratio=0.97, frame_volumes=None, exclusion_masks=None):
    """Weak lit endpoints: observed recovery/onset AND outside all plausible tracked shadows."""
    shape = positive_core.shape
    light_ids = sorted({item["light_id"] for item in explanations})
    result, details = {}, {}
    for slot, event in ((0, "shadow_boundary_onset"), (2, "shadow_boundary_recovery")):
        possible_shadow = np.zeros(shape, bool)
        invalid = False
        for light_id in light_ids:
            for track in tracks:
                try:
                    possible_shadow |= footprint_mask(track["points"][slot, 1:], plane,
                        directions[light_id], arrays["bev_world_to_grid"], shape)
                except ValueError:
                    invalid = True
            for volume in (frame_volumes or {}).get(slot, []):
                try:
                    possible_shadow |= footprint_mask(np.asarray(volume["corners_world"]), plane,
                        directions[light_id], arrays["bev_world_to_grid"], shape)
                except ValueError:
                    invalid = True
        mask = positive_core & arrays[event] & arrays["receiver_valid"]
        mask &= arrays["learned_bev_receiver_support"] & arrays["shadow_flow_supported"]
        mask &= arrays["appearance_local_ratio"][slot] >= minimum_lit_ratio
        mask &= ~possible_shadow
        exclusion = (exclusion_masks or {}).get(slot)
        if exclusion is not None:
            mask &= ~exclusion
        if invalid or not light_ids:
            mask[:] = False
        result[slot] = mask
        details[str(slot)] = {"cells": int(mask.sum()), "invalid_competing_volume": invalid,
                             "excluded_projected_shadow_cells": int(possible_shadow.sum()),
                             "extra_exclusion_cells": int(exclusion.sum()) if exclusion is not None else 0,
                             "all_class_volumes": len((frame_volumes or {}).get(slot, [])),
                             "light_ids": light_ids, "minimum_lit_ratio": minimum_lit_ratio}
    return result, details


def merge_trimaps(targets):
    """Combine frame observations; contradictory teacher pixels become IGNORE."""
    values = np.stack(targets)
    positive, negative = (values == 1).any(axis=0), (values == 0).any(axis=0)
    merged = np.full(values.shape[1:], 255, np.uint8)
    merged[positive & ~negative] = 1
    merged[negative & ~positive] = 0
    return merged, int((positive & negative).sum())


def lit_receiver_negative_mask(arrays, positive_core, tracks, plane, directions, explanations, *,
                               resolution, volumes=(), exclusion=None, minimum_lit_ratio=0.97,
                               minimum_contrast_L=6.0, radius_m=12.0, negative_ratio=2.0):
    """A compact, persistently lit neighbour; no motion or brightness event required."""
    shape = positive_core.shape
    empty = np.zeros(shape, bool)
    if "appearance_L_raw" not in arrays or not positive_core.any():
        return empty, {"reason": "missing_receiver_appearance", "cells": 0}
    light_ids = sorted({item["light_id"] for item in explanations})
    if not light_ids:
        return empty, {"reason": "missing_projection_explanation", "cells": 0}
    possible = np.zeros(shape, bool)
    normalized_plane = np.asarray(plane, np.float64)/np.linalg.norm(np.asarray(plane)[:3])
    corners_list = [track["points"][1, 1:] for track in tracks]
    corners_list += [np.asarray(volume["corners_world"]) for volume in volumes]
    for corners in corners_list:
        # Exclusion volumes can extend below the locally fitted road plane.
        distances = corners @ normalized_plane[:3]+normalized_plane[3]
        clipped = corners-np.minimum(distances, 0)[:, None]*normalized_plane[:3]
        for light_id in light_ids:
            possible |= footprint_mask(clipped, plane, directions[light_id], arrays["bev_world_to_grid"], shape)
    distance = cv2.distanceTransform((~positive_core).astype(np.uint8), cv2.DIST_L2, 5)*resolution
    positive_L = float(np.median(arrays["appearance_L_raw"][1][positive_core]))
    candidate = arrays["receiver_valid"] & arrays["learned_bev_receiver_support"] & arrays["shadow_flow_supported"]
    stages = {"supported_receiver_cells": int(candidate.sum())}
    candidate &= (arrays["appearance_local_ratio"] >= minimum_lit_ratio).all(axis=0)
    stages["persistent_lit_cells"] = int(candidate.sum())
    candidate &= arrays["appearance_L_raw"][1] >= positive_L+minimum_contrast_L
    stages["contrast_supported_cells"] = int(candidate.sum())
    candidate &= (distance <= radius_m) & ~positive_core & ~possible
    if "shadow_candidate" in arrays:
        candidate &= ~arrays["shadow_candidate"]
    if exclusion is not None:
        candidate &= ~exclusion
    details = {"stages": stages, "candidate_cells": int(candidate.sum()), "excluded_shadow_cells": int(possible.sum()),
               "positive_median_L": positive_L, "minimum_contrast_L": minimum_contrast_L,
               "minimum_lit_ratio": minimum_lit_ratio, "radius_m": radius_m, "negative_ratio": negative_ratio}
    if not candidate.any():
        return empty, {**details, "reason": "no_persistently_lit_receiver", "cells": 0}
    # Grow one local connected patch, not scattered easy negatives over the image.
    start = tuple(np.unravel_index(np.argmin(np.where(candidate, distance, np.inf)), shape))
    pending, visited = deque([start]), {start}
    result = empty.copy()
    budget = max(1, int(np.ceil(positive_core.sum()*negative_ratio)))
    accepted = 0
    while pending and accepted < budget:
        y, x = pending.popleft()
        result[y, x] = True
        accepted += 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            point = (y+dy, x+dx)
            if (0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]
                    and point not in visited and candidate[point]):
                visited.add(point)
                pending.append(point)
    return result, {**details, "reason": "observed_lit_receiver", "cells": int(result.sum())}
