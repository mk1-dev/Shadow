from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from shadow_occ.annotation.data import (
    GT_SOURCES, SOURCES, WEAK_SOURCES, assert_disjoint, checked_write_image, digest, read_rows, write_rows,
)
from shadow_occ.annotation.metrics import BinaryMetrics
from shadow_occ.annotation.temporal import sampling_grid
from shadow_occ.utils.io import write_json
from shadow_occ.utils.seed import seed_everything, seed_worker


class ShadowDataset(Dataset):
    def __init__(self, rows, size, augment=False):
        self.rows, self.size, self.augment = rows, tuple(size), augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row["image"])
        target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            raise FileNotFoundError(f"Missing image/target for {row['id']}")
        if image.shape[:2] != target.shape:
            raise ValueError(f"Image-target mismatch for {row['id']}")
        original_shape = image.shape[:2]
        height, width = self.size
        previous = image
        grid = np.zeros((height, width, 2), dtype=np.float32)
        weight = np.zeros((height, width), dtype=np.float32)
        temporal = row.get("temporal")
        if temporal and temporal.get("lighting_comparable"):
            previous = cv2.imread(temporal["previous_image"])
            if previous is None:
                raise FileNotFoundError(temporal["previous_image"])
            grid, valid = sampling_grid(temporal["caster_previous_to_current"], original_shape, previous.shape[:2], self.size)
            with np.load(temporal["evidence"], allow_pickle=False) as archive:
                original_weight = np.asarray(archive["weight"], dtype=np.float32)
            if original_weight.shape != original_shape:
                raise ValueError("Temporal evidence and current image coordinates differ")
            weight = cv2.resize(original_weight, (width, height), interpolation=cv2.INTER_LINEAR)*valid
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        previous = cv2.resize(previous, (width, height), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST)
        if self.augment and np.random.random() < 0.5:
            image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
            previous, weight = previous[:, ::-1].copy(), weight[:, ::-1].copy()
            grid = grid[:, ::-1].copy()
            grid[..., 0] *= -1

        def tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            rgb = (rgb-np.array([0.485, 0.456, 0.406], np.float32))/np.array([0.229, 0.224, 0.225], np.float32)
            return torch.from_numpy(rgb.transpose(2, 0, 1).copy())

        sample_weight = float(row.get("sample_weight", 1.0))
        if not np.isfinite(sample_weight) or not 0 < sample_weight <= 1:
            raise ValueError("sample_weight must be in (0, 1]")
        return {"image": tensor(image), "previous_image": tensor(previous), "target": torch.from_numpy(target.astype(np.int64)), "temporal_grid": torch.from_numpy(grid), "temporal_weight": torch.from_numpy(weight), "source": row["label_source"], "sample_weight": sample_weight}


def build_segmenter(backbone_weights=None):
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.DEFAULT if backbone_weights == "imagenet" else None
    return deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=1, aux_loss=False)


def masked_loss(logits, target, dice_weight, weights=None):
    logits = logits[:, 0].float()
    valid = target != 255
    truth = (target == 1).to(logits.dtype)
    count = valid.flatten(1).sum(1)
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none")*valid).flatten(1).sum(1)/count.clamp_min(1)
    probability = logits.sigmoid()*valid
    dice = 1-(2*(probability*truth).flatten(1).sum(1)+1)/(probability.flatten(1).sum(1)+(truth*valid).flatten(1).sum(1)+1)
    loss = (bce+dice_weight*dice)*(count > 0)
    if weights is not None:
        loss = loss*weights
    return loss.sum()/(count > 0).sum().clamp_min(1)


def temporal_loss(current, previous, grid, weight):
    previous_probability = F.grid_sample(previous.sigmoid().detach().float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    error = (current[:, 0].sigmoid().float()-previous_probability).abs()
    return (error*weight).sum()/weight.sum().clamp_min(1)


def evaluate(model, rows, size, device, rank=0, world=1):
    # Uneven, non-padding shards are evaluated without a DDP wrapper.
    loader = DataLoader(ShadowDataset(rows[rank::world], size), batch_size=1, num_workers=0)
    states = {source: BinaryMetrics() for source in sorted({row["label_source"] for row in rows})}
    model.eval()
    temporal_error, temporal_count = 0.0, 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device))["out"]
            probability = logits[:, 0].sigmoid().cpu().numpy()
            states[batch["source"][0]].update(probability[0] >= 0.5, batch["target"][0].numpy())
            count = float(batch["temporal_weight"].sum())
            if count > 0:
                previous = model(batch["previous_image"].to(device))["out"]
                error = temporal_loss(logits, previous, batch["temporal_grid"].to(device), batch["temporal_weight"].to(device))
                temporal_error += float(error)*count
                temporal_count += count
    if world > 1:
        for state in states.values():
            values = torch.as_tensor(state.confusion, device=device)
            dist.all_reduce(values)
            state.confusion = values.cpu().numpy()
        totals = torch.tensor([temporal_error, temporal_count], device=device, dtype=torch.float64)
        dist.all_reduce(totals)
        temporal_error, temporal_count = totals.cpu().tolist()
    return {source: state.summary() for source, state in states.items()}, {"weighted_probability_MAE": temporal_error/temporal_count if temporal_count else None, "weight_sum": temporal_count}


def train_main():
    parser = argparse.ArgumentParser("Train RGB shadow segmentation, retaining uncertain pixels as ignore.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--size", nargs=2, type=int, default=[384, 704], metavar=("H", "W"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.1)
    parser.add_argument("--temporal-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--allow-weak-labels", action="store_true")
    parser.add_argument("--weak-weight", type=float, default=0.25)
    parser.add_argument("--backbone-weights", choices=["none", "imagenet"], default="none")
    parser.add_argument("--init-checkpoint", help="Optional segmenter checkpoint for pretraining -> adaptation")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation) < 1:
        raise ValueError("epochs, batch-size and accumulation must be positive")
    if not 0 < args.weak_weight <= 1 or args.workers < 0 or args.temporal_weight < 0:
        raise ValueError("Invalid weak-label weight or worker count")
    train_all, dev_all = read_rows(args.train), read_rows(args.dev)
    assert_disjoint(train_all, dev_all)
    if any(row.get("split") != "dev" for row in dev_all):
        raise ValueError("Use the train-derived dev split, not official eval, for model selection")
    allowed = SOURCES if args.allow_weak_labels else GT_SOURCES
    train = [row for row in train_all if row["label_source"] in allowed]
    dev = [row for row in dev_all if row["label_source"] in allowed]
    if not train or not dev:
        raise ValueError("No usable train/dev shadow masks; object boxes alone cannot supervise segmentation")
    class_counts = np.zeros(2, dtype=np.int64)
    usable = []
    for row in train:
        mask = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(row["target"])
        counts = np.array([(mask == 0).sum(), (mask == 1).sum()])
        class_counts += counts
        if counts.sum():
            usable.append(row)
    train = usable
    if (class_counts == 0).any():
        raise ValueError("Both known shadow AND known non-shadow pixels are required")
    dev_counts = np.zeros(2, dtype=np.int64)
    for row in dev:
        mask = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(row["target"])
        dev_counts += np.array([(mask == 0).sum(), (mask == 1).sum()])
    if (dev_counts == 0).any():
        raise ValueError("Dev masks must contain known shadow and non-shadow pixels")
    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    if args.device == "cuda":
        torch.cuda.set_device(local)
    device = torch.device("cuda", local) if args.device == "cuda" else torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_everything(args.seed)
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output root")
    if world > 1:
        dist.barrier()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/"protocol.json", {**vars(args), "train_sha256": digest(args.train), "dev_sha256": digest(args.dev), "train_scenes": sorted({row['scene_name'] for row in train}), "dev_scenes": sorted({row['scene_name'] for row in dev}), "pixel_counts": class_counts.tolist()})
    if world > 1:
        dist.barrier()
    sampler = DistributedSampler(ShadowDataset(train, args.size), seed=args.seed) if world > 1 else None
    loader = DataLoader(ShadowDataset(train, args.size, augment=True), batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=args.workers, worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    model = build_segmenter(args.backbone_weights).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best = [], -1.0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        # Freeze running statistics, including ASPP's 1x1 pooled BatchNorm at batch size 1.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"segmenter epoch {epoch}", disable=rank != 0)):
            start = (step//args.accumulation)*args.accumulation
            window = min(args.accumulation, len(loader)-start)
            boundary = (step+1) % args.accumulation == 0 or step+1 == len(loader)
            sync = contextlib.nullcontext() if world == 1 or boundary else model.no_sync()
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    images = batch["image"].to(device)
                    use_temporal = args.temporal_weight > 0 and bool(batch["temporal_weight"].any())
                    inputs = torch.cat([images, batch["previous_image"].to(device)]) if use_temporal else images
                    predictions = model(inputs)["out"]
                    logits = predictions[:images.shape[0]]
                    weights = torch.tensor([args.weak_weight if source in WEAK_SOURCES else 1.0 for source in batch["source"]], device=device)
                    weights = weights*batch["sample_weight"].to(device)
                    loss = masked_loss(logits, batch["target"].to(device), args.dice_weight, weights)
                    if use_temporal:
                        previous = predictions[images.shape[0]:]
                        loss = loss + args.temporal_weight*temporal_loss(logits, previous, batch["temporal_grid"].to(device), batch["temporal_weight"].to(device))
                    loss = loss/window
                scaler.scale(loss).backward()
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        raw = model.module if world > 1 else model
        metrics, temporal_metrics = evaluate(raw, dev, args.size, device, rank, world)
        preferred = [v["iou"] for k, v in metrics.items() if k in GT_SOURCES and v["iou"] is not None]
        values = preferred or [v["iou"] for v in metrics.values() if v["iou"] is not None]
        score = float(np.mean(values)) if values else -1.0
        if rank == 0:
            state = {"model": raw.state_dict(), "epoch": epoch, "size": args.size, "args": vars(args), "dev_metrics_by_source": metrics, "dev_temporal_metrics": temporal_metrics, "train_scenes": sorted({row['scene_name'] for row in train}), "dev_scenes": sorted({row['scene_name'] for row in dev})}
            torch.save(state, output/"last.pth")
            if score > best:
                best = score
                torch.save(state, output/"best.pth")
            history.append({"epoch": epoch, "dev_metrics_by_source": metrics, "temporal": temporal_metrics})
            write_json(output/"history.json", history)
            print({"epoch": epoch, "dev_metrics_by_source": metrics, "temporal": temporal_metrics}, flush=True)
    if world > 1:
        dist.destroy_process_group()


def predict_main():
    parser = argparse.ArgumentParser("Predict pixel masks and evaluate by label provenance.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-train-scenes", action="store_true", help="Explicit in-sample annotation, not held-out evaluation")
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    overlap = sorted(set(checkpoint.get("train_scenes", [])) & {row["scene_name"] for row in rows})
    if overlap and not args.allow_train_scenes:
        raise ValueError("Manifest overlaps training scenes; use --allow-train-scenes only for in-sample annotation")
    model = build_segmenter()
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    states, predictions = defaultdict(BinaryMetrics), []
    temporal_error, temporal_count = 0.0, 0.0
    with torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc="shadow segmentation")):
            sample = ShadowDataset([row], checkpoint["size"])[0]
            image = sample["image"]
            original = cv2.imread(row["image"])
            logits = model(image[None].to(device))["out"]
            count = float(sample["temporal_weight"].sum())
            if count > 0:
                previous = model(sample["previous_image"][None].to(device))["out"]
                error = temporal_loss(logits, previous, sample["temporal_grid"][None].to(device), sample["temporal_weight"][None].to(device))
                temporal_error += float(error)*count
                temporal_count += count
            probability = F.interpolate(logits, size=original.shape[:2], mode="bilinear", align_corners=False)[0, 0].sigmoid().cpu().numpy()
            target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
            states[row["label_source"]].update(probability >= args.threshold, target)
            path = output/"probabilities"/f"{index:07d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probability=probability.astype(np.float16))
            checked_write_image(output/"masks"/f"{index:07d}.png", (probability >= args.threshold).astype(np.uint8)*255)
            predictions.append({**row, "probability": str(path), "threshold": args.threshold})
    write_rows(output/"predictions.jsonl", predictions)
    write_json(output/"segmentation_metrics.json", {"checkpoint": str(Path(args.checkpoint).resolve()), "manifest_sha256": digest(args.manifest), "threshold": args.threshold, "in_sample_annotation": bool(overlap), "overlapping_train_scenes": overlap, "by_label_source": {source: state.summary() for source, state in states.items()}, "temporal_consistency": {"weighted_probability_MAE": temporal_error/temporal_count if temporal_count else None, "weight_sum": temporal_count}})
