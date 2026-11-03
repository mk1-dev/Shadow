from __future__ import annotations

import numpy as np


def refine_target(probability, original, decisions, threshold=0.7, preserve_gt=False):
    probability = np.asarray(probability, np.float32)
    original = np.asarray(original)
    if probability.shape != original.shape or probability.ndim != 2:
        raise ValueError("Probability and original trimap must share 2D coordinates")
    if not 0.5 < threshold <= 1 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid pixel threshold/probabilities")
    if not np.isin(original, [0, 1, 255]).all():
        raise ValueError("Original target must be a 0/1/255 trimap")
    yes, no, uncertain = [np.zeros(original.shape, bool) for _ in range(3)]
    for mask, decision in decisions:
        mask = np.asarray(mask, bool)
        if mask.shape != original.shape:
            raise ValueError("Region and segmentation pixel coordinates differ")
        if decision == "YES":
            yes |= mask
        elif decision == "NO":
            no |= mask
        elif decision == "UNCERTAIN":
            uncertain |= mask
        else:
            raise ValueError("Invalid region decision")
    conflict = (yes & no) | uncertain
    reviewed = yes | no | uncertain
    high, low = probability >= threshold, probability <= 1-threshold
    target = np.full(original.shape, 255, np.uint8)
    target[(original == 0) & low & ~yes & ~uncertain] = 0
    # Unreviewed existing positives may persist; only reviewed YES can expand them.
    target[(original == 1) & high & ~reviewed] = 1
    target[yes & high & ~conflict & (original != 0)] = 1
    if preserve_gt:
        target[original != 255] = original[original != 255]
    report = {
        "retained_positive_pixels": int(((original == 1) & (target == 1)).sum()),
        "new_positive_pixels": int(((original == 255) & (target == 1)).sum()),
        "removed_positive_pixels": int(((original == 1) & (target != 1)).sum()),
        "conflicting_vote_pixels": int((yes & no).sum()),
        "uncertain_vote_pixels": int(uncertain.sum()),
        "positive_vs_known_negative_pixels": int((yes & high & (original == 0)).sum()),
    }
    return target, report
