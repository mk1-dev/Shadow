"""Deterministic scene and supervision-support sampling, independent of torch."""
from collections import Counter
import hashlib
import math

import numpy as np


def scene_rank(value, seed, scope):
    return hashlib.sha256(f"{seed}:{scope}:{value}".encode()).hexdigest()


def partition_scenes(training, seed=0, dev_fraction=.15):
    ordered = sorted(set(training), key=lambda scene: scene_rank(scene, seed, "dev"))
    if len(ordered) < 2 or not 0 < dev_fraction < 1:
        raise ValueError("Scene partition needs at least two training scenes and a dev fraction in (0, 1)")
    dev_count = max(1, min(len(ordered)-1, round(len(ordered)*dev_fraction)))
    dev = set(ordered[:dev_count])
    train_order = sorted(set(ordered)-dev, key=lambda scene: scene_rank(scene, seed, "train"))
    return dev, {fraction: set(train_order[:math.ceil(len(train_order)*fraction/100)]) for fraction in (25, 50, 100)}


def scene_probabilities(rows, mode="image"):
    if not rows or mode not in {"image", "uniform", "supervision"}:
        raise ValueError("Choose nonempty training rows and image/uniform/supervision sampling")
    if mode == "supervision":
        groups = _supervision_groups(rows)
        result = np.zeros(len(rows))
        for indices in groups:
            result[indices] += .5*scene_probabilities([rows[index] for index in indices], "uniform")
        return result
    counts = Counter(row["scene_name"] for row in rows)
    weights = np.array([1.0/counts[row["scene_name"]] if mode == "uniform" else 1.0 for row in rows])
    return weights/weights.sum()


def _supervision_groups(rows):
    groups = [[index for index, row in enumerate(rows) if row["sampling_class_pixels"][label] > 0]
              for label in (0, 1)]
    if not all(groups):
        raise ValueError("Supervision sampling needs both classes after the actual training resize")
    return groups


def supervision_epoch_indices(rows, seed, epoch, rank=0, world=1):
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid distributed sampler shard")
    groups = _supervision_groups(rows)
    count = math.ceil(len(rows)/(2*world))*2*world
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    positions = np.arange(count)
    # Each local pair draws a negative-support and a positive-support image.
    kinds = (positions//world+positions % world) % 2
    indices = np.zeros(count, dtype=np.int64)
    for kind, group in enumerate(groups):
        indices[kinds == kind] = generator.choice(group, size=int((kinds == kind).sum()), replace=True,
            p=scene_probabilities([rows[index] for index in group], "uniform"))
    return indices[rank::world].tolist()


def scene_epoch_indices(rows, seed, epoch, rank=0, world=1):
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid distributed sampler shard")
    count = math.ceil(len(rows)/world)*world
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    indices = generator.choice(len(rows), size=count, replace=True, p=scene_probabilities(rows, "uniform"))
    return indices[rank::world].tolist()
