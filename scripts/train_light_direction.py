"""Train the small light head, with scene-held-out priors for training frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest
from shadow_occ.annotation.light_learning import LightDataset, forward_light, light_loss, load_light_rows, scene_weights
from shadow_occ.models.illumination import GeoTemporalLightNet
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crossfit-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.lr) <= 0 or args.crossfit_folds < 0 or args.crossfit_folds == 1:
        parser.error("Positive training settings and either zero or at least two crossfit folds are required")
    root, output = Path(args.data_root).resolve(), Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_light_rows(root)
    if any(not any(row["target_weight"] > 0 for row in rows[split]) for split in ("train", "dev")):
        raise ValueError("Both train and dev require astronomical or physical light evidence")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scenes = sorted({row["scene_name"] for row in rows["train"]})
    if args.crossfit_folds > len(scenes):
        raise ValueError("Crossfit folds exceed independent training scenes")
    fold_by_scene = {scene: index % args.crossfit_folds for index, scene in enumerate(scenes)} if args.crossfit_folds else {}
    protocol = {**vars(args), "data_protocol_sha256": digest(root/"protocol.json"),
        "manifest_sha256": {split: digest(root/f"{split}.jsonl") for split in rows},
        "scene_folds": fold_by_scene, "selection": "dev weak loss, never official eval",
        "targets": "weak astronomy/geometry distributions, not illumination GT"}
    write_json(output/"training_protocol.json", protocol)

    def fit(name, training, heldout):
        if not training or not any(row["target_weight"] > 0 for row in training):
            raise ValueError(f"No supervised windows in {name}")
        model = GeoTemporalLightNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sampler = WeightedRandomSampler(scene_weights(training), len(training), replacement=True)
        loader = DataLoader(LightDataset(training), batch_size=args.batch_size, sampler=sampler)
        dev = DataLoader(LightDataset(rows["dev"]), batch_size=args.batch_size)
        best, history = float("inf"), []
        for epoch in range(args.epochs):
            model.train()
            total = 0.
            for batch in loader:
                prediction, batch = forward_light(model, batch, device)
                loss = light_loss(prediction, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total += loss.item()
            model.eval()
            validation = []
            with torch.no_grad():
                for batch in dev:
                    prediction, batch = forward_light(model, batch, device)
                    validation.append(light_loss(prediction, batch).item())
            value = float(np.mean(validation))
            history.append({"epoch": epoch, "train_loss": total/max(len(loader), 1), "dev_weak_loss": value})
            print({"model": name, **history[-1]}, flush=True)
            if value < best:
                best = value
                torch.save({"model": model.state_dict(), "protocol": protocol,
                    "trained_scenes": sorted({row["scene_name"] for row in training}),
                    "heldout_train_scenes": heldout, "epoch": epoch}, output/f"{name}.pth")
        write_json(output/f"{name}_history.json", history)

    fit("best", rows["train"], [])
    for fold in range(args.crossfit_folds):
        heldout = [scene for scene, value in fold_by_scene.items() if value == fold]
        fit(f"fold_{fold}", [row for row in rows["train"] if row["scene_name"] not in heldout], heldout)


if __name__ == "__main__":
    main()
