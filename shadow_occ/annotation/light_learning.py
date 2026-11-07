"""Shared data and diagnostics for the learned light-direction head."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from shadow_occ.annotation.data import digest, read_rows


class LightDataset(Dataset):
    def __init__(self, rows, mode="image+geo"):
        self.rows, self.mode = rows, mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["features"])
        if digest(path) != row["features_sha256"]:
            raise ValueError("Light training feature cache changed")
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name].astype(np.float32) for name in
                      ("images", "receiver", "metadata", "directions", "target", "solar_target")}
        values["images"] = values["images"]/255-.5
        if self.mode == "geo-only":
            values["images"][:] = 0
            values["receiver"][:] = 0
        elif self.mode == "no-geo":
            values["metadata"][:5] = 0
            values["directions"][:, 9:] = 0
        values["weight"] = np.float32(row["target_weight"])
        values["confidence"] = np.float32(row["confidence_target"])
        return {key: torch.as_tensor(value) for key, value in values.items()}


def scene_weights(rows):
    counts = Counter(row["scene_name"] for row in rows)
    return [1/counts[row["scene_name"]] for row in rows]


def load_light_rows(root):
    root = Path(root)
    rows = {split: read_rows(root/f"{split}.jsonl") for split in ("train", "dev", "eval")}
    scenes = {split: {row["scene_name"] for row in values} for split, values in rows.items()}
    if any(scenes[a] & scenes[b] for a, b in (("train", "dev"), ("train", "eval"), ("dev", "eval"))):
        raise ValueError("Light dataset scene leakage")
    if not rows["train"] or not rows["dev"]:
        raise ValueError("Light training requires nonempty scene-disjoint train and dev")
    return rows


def light_loss(prediction, batch):
    log_probability = prediction["logits"].log_softmax(-1)
    distribution = -(batch["target"]*log_probability).sum(-1)
    weight = batch["weight"]
    direction = (distribution*weight).sum()/weight.sum().clamp_min(1e-6)
    confidence = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction["confidence_logits"], batch["confidence"], reduction="none")
    observed = weight > 0
    confidence = (confidence*observed).sum()/observed.sum().clamp_min(1)
    return direction+.2*confidence


def forward_light(model, batch, device):
    batch = {key: value.to(device) for key, value in batch.items()}
    return model(batch["images"], batch["receiver"], batch["metadata"], batch["directions"]), batch
