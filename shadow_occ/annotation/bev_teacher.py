"""Training labels from selected BEV receivers and caster/light trajectories."""
from __future__ import annotations

import numpy as np

from shadow_occ.annotation.physics_labels import footprint_mask


PROTOCOL = "bev-flow-annotation-v1"


def endpoint_negative_masks(arrays, positive_core, tracks, plane, directions, explanations, *, minimum_lit_ratio=0.97):
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
        mask = positive_core & arrays[event] & arrays["receiver_valid"]
        mask &= arrays["learned_bev_receiver_support"] & arrays["shadow_flow_supported"]
        mask &= arrays["appearance_local_ratio"][slot] >= minimum_lit_ratio
        mask &= ~possible_shadow
        if invalid or not light_ids:
            mask[:] = False
        result[slot] = mask
        details[str(slot)] = {"cells": int(mask.sum()), "invalid_competing_volume": invalid,
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
