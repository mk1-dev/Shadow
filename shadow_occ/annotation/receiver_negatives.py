"""Observed lit-ground weak negatives independent of accepted shadow cores."""
from __future__ import annotations

import cv2
import numpy as np

from shadow_occ.annotation.physics_labels import footprint_mask


def independent_lit_receiver_mask(arrays, plane, directions, volumes, *, maximum_cells=64,
                                  minimum_L=30.0, minimum_contrast_L=6.0,
                                  minimum_local_ratio=.98, maximum_log_change=.12,
                                  maximum_fraction=.1):
    if maximum_cells < 1 or not 0 < maximum_fraction <= 1:
        raise ValueError("Negative budget must be positive and fraction in (0,1]")
    receiver = np.asarray(arrays["receiver_valid"], bool)
    empty = np.zeros_like(receiver)
    details = {"protocol": "independent-lit-receiver-v1", "requires_positive_core": False,
        "minimum_L": minimum_L, "minimum_contrast_L": minimum_contrast_L,
        "minimum_local_ratio": minimum_local_ratio, "maximum_log_change": maximum_log_change,
        "maximum_cells": maximum_cells, "maximum_fraction": maximum_fraction,
        "selection": "spatial_tile_round_robin_brightness_rank"}
    if volumes is None:
        return empty, {**details, "reason": "missing_all_class_exclusion_volumes", "cells": 0}
    light = np.asarray(arrays["appearance_L_raw"], np.float64)
    ratio = np.asarray(arrays["appearance_local_ratio"], np.float64)
    supported = receiver & np.isfinite(light).all(axis=0) & np.isfinite(ratio).all(axis=0)
    details["supported_receiver_cells"] = int(supported.sum())
    if not supported.any():
        return empty, {**details, "reason": "missing_geometric_receiver_correspondence", "cells": 0}
    low, high = np.quantile(light[1][supported], [.1, .9])
    lit_floor = max(minimum_L, float(np.quantile(light[1][supported], .7)))
    details.update(context_contrast_L=float(high-low), lit_floor_L=lit_floor)
    if high-low < minimum_contrast_L:
        return empty, {**details, "reason": "insufficient_lit_context_contrast", "cells": 0}
    exposure = np.maximum(np.median(light[:, supported], axis=1), 1.0)
    normalized = np.maximum(light, 1.0)/exposure[:, None, None]
    change = np.log(np.maximum(normalized.max(axis=0), 1e-6)/np.maximum(normalized.min(axis=0), 1e-6))
    candidate = supported & (light[1] >= lit_floor) & (ratio >= minimum_local_ratio).all(axis=0)
    candidate &= change <= maximum_log_change
    excluded = np.asarray(arrays["exclusion"], bool).copy()
    # No accepted light is needed: exclude all modelled volumes under the fixed bank.
    if candidate.any():
        normal = np.asarray(plane, np.float64)
        normal = normal/np.linalg.norm(normal[:3])
        for volume in volumes:
            corners = np.asarray(volume["corners_world"], np.float64)
            distances = corners @ normal[:3]+normal[3]
            corners = corners-np.minimum(distances, 0)[:, None]*normal[:3]
            for direction in directions:
                excluded |= footprint_mask(corners, plane, direction, arrays["bev_world_to_grid"], receiver.shape)
    excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    candidate &= ~excluded
    details.update(excluded_cells=int(excluded.sum()), candidate_cells=int(candidate.sum()),
        light_hypotheses=len(directions), all_class_volumes=len(volumes))
    result = empty.copy()
    if candidate.any():
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=4)
        usable = np.zeros_like(candidate)
        for identity in range(1, count):
            if statistics[identity, cv2.CC_STAT_AREA] >= 2:
                usable |= labels == identity
        indices = np.flatnonzero(usable)
        # Cap the pool independently of positive area, retaining observed evidence only.
        budget = min(maximum_cells, max(1, int(np.ceil(supported.sum()*maximum_fraction))))
        # Spread the existing evidence over the receiver instead of concentrating
        # every retained negative in the single brightest road patch.
        yy, xx = np.unravel_index(indices, receiver.shape)
        tiles = (yy*4//receiver.shape[0])*4 + xx*4//receiver.shape[1]
        remaining, ordered = list(np.unique(tiles)), []
        while remaining:
            tile = remaining[0] if not ordered else max(remaining, key=lambda value:
                min((value//4-other//4)**2+(value%4-other%4)**2 for other in ordered))
            ordered.append(tile)
            remaining.remove(tile)
        queues = [indices[tiles == tile][np.argsort(-light[1].ravel()[indices[tiles == tile]], kind="stable")]
                  for tile in ordered]
        selected = [int(queue[level]) for level in range(max(map(len, queues), default=0))
                    for queue in queues if level < len(queue)][:budget]
        result.ravel()[selected] = True
    return result, {**details, "reason": "independent_observed_lit_receiver" if result.any() else "no_independent_lit_receiver",
                    "cells": int(result.sum()), "unlabelled_is_background": False}
