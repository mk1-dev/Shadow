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
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from tqdm import tqdm

from shadow_occ.annotation.data import (
    GT_SOURCES, SOURCES, WEAK_SOURCES, assert_disjoint, checked_write_image, digest, read_rows, write_rows,
)
from shadow_occ.annotation.metrics import BinaryMetrics, pixel_score_diagnostics
from shadow_occ.annotation.temporal import sampling_grid
from shadow_occ.annotation.readiness import binary_risk_weights, require_segmenter_checkpoint, segmentation_readiness
from shadow_occ.annotation.teacher_provenance import extend_scene_history
from shadow_occ.annotation.scene_sampling import scene_epoch_indices, scene_probabilities
from shadow_occ.utils.io import write_json
from shadow_occ.utils.seed import seed_everything, seed_worker


class SceneBalancedSampler(Sampler):
    def __init__(self, rows, seed, rank=0, world=1):
        self.rows, self.seed, self.rank, self.world, self.epoch = rows, seed, rank, world, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (len(self.rows)+self.world-1)//self.world

    def __iter__(self):
        return iter(scene_epoch_indices(self.rows, self.seed, self.epoch, self.rank, self.world))


class ShadowDataset(Dataset):
    def __init__(self, rows, size, augment=False, receiver_pairs=False):
        self.rows, self.size, self.augment = rows, tuple(size), augment
        self.receiver_pairs = receiver_pairs

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
        if not np.isin(target, [0, 1, 255]).all():
            raise ValueError("Segmentation targets must be 0/1/255 trimaps")
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
        receiver_previous, receiver_next = image, image
        onset_weight = np.zeros((height, width), np.float32)
        recovery_weight = onset_weight.copy()
        transition = row.get("physics_transition")
        if self.receiver_pairs and transition:
            if (transition["semantics"] != "relative_brightness_evidence_not_absolute_nonshadow_labels"
                    or digest(transition["path"]) != transition["sha256"]
                    or [view["slot"] for view in transition["views"]] != ["previous", "next"]):
                raise ValueError("Invalid or changed same-receiver transition evidence")
            views = []
            for view in transition["views"]:
                if digest(view["image"]) != view["sha256"]:
                    raise ValueError("Changed registered receiver view")
                aligned = cv2.imread(view["image"])
                if aligned is None or aligned.shape[:2] != original_shape:
                    raise ValueError("Receiver views must use middle image coordinates")
                views.append(cv2.resize(aligned, (width, height), interpolation=cv2.INTER_LINEAR))
            receiver_previous, receiver_next = views
            with np.load(transition["path"], allow_pickle=False) as archive:
                values = []
                for name in ("onset", "recovery"):
                    array = archive[name]
                    if array.shape != original_shape or not np.isfinite(array).all():
                        raise ValueError("Receiver transition map/image mismatch")
                    values.append(cv2.resize((array > 0).astype(np.float32), (width, height),
                        interpolation=cv2.INTER_NEAREST)*(target == 1))
                onset_weight, recovery_weight = values
        if self.augment and np.random.random() < 0.5:
            image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
            previous, weight = previous[:, ::-1].copy(), weight[:, ::-1].copy()
            grid = grid[:, ::-1].copy()
            grid[..., 0] *= -1
            receiver_previous, receiver_next = receiver_previous[:, ::-1].copy(), receiver_next[:, ::-1].copy()
            onset_weight, recovery_weight = onset_weight[:, ::-1].copy(), recovery_weight[:, ::-1].copy()

        def tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            rgb = (rgb-np.array([0.485, 0.456, 0.406], np.float32))/np.array([0.229, 0.224, 0.225], np.float32)
            return torch.from_numpy(rgb.transpose(2, 0, 1).copy())

        sample_weight = float(row.get("sample_weight", 1.0))
        if not np.isfinite(sample_weight) or not 0 < sample_weight <= 1:
            raise ValueError("sample_weight must be in (0, 1]")
        negative_weight = float(row.get("negative_weight", 1.0))
        if not np.isfinite(negative_weight) or not 0 < negative_weight <= 1:
            raise ValueError("negative_weight must be in (0, 1]")
        return {"image": tensor(image), "previous_image": tensor(previous), "target": torch.from_numpy(target.astype(np.int64)), "temporal_grid": torch.from_numpy(grid), "temporal_weight": torch.from_numpy(weight), "source": row["label_source"], "sample_weight": sample_weight,
                "negative_weight": negative_weight,
                "receiver_previous": tensor(receiver_previous), "receiver_next": tensor(receiver_next),
                "onset_weight": torch.from_numpy(onset_weight), "recovery_weight": torch.from_numpy(recovery_weight)}


def build_segmenter(backbone_weights=None):
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.DEFAULT if backbone_weights == "imagenet" else None
    return deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=1, aux_loss=False)


def masked_loss(logits, target, dice_weight, weights=None, class_weights=None, negative_weights=None, *, return_components=False):
    logits = logits[:, 0].float()
    valid = target != 255
    truth = (target == 1).to(logits.dtype)
    count = valid.flatten(1).sum(1)
    pixel_weights = torch.ones_like(logits)
    if class_weights is not None:
        pixel_weights = torch.where(target == 1, float(class_weights[1]), float(class_weights[0]))
    reliability = torch.ones_like(logits)
    if negative_weights is not None:
        reliability = torch.where(target == 0, negative_weights.to(logits).reshape(-1, 1, 1), reliability)
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none")*valid*pixel_weights*reliability).flatten(1).sum(1)/count.clamp_min(1)
    probability = logits.sigmoid()*valid*reliability
    dice = 1-(2*(probability*truth).flatten(1).sum(1)+1)/(probability.flatten(1).sum(1)+(truth*valid).flatten(1).sum(1)+1)
    # Empty foreground images already have a BCE term; do not let their Dice dominate.
    scale = (count > 0).to(logits.dtype)
    if weights is not None:
        scale = scale*weights
    denominator = (count > 0).sum().clamp_min(1)
    bce = (bce*scale).sum()/denominator
    dice = (dice*(truth.flatten(1).sum(1) > 0)*scale).sum()/denominator
    loss = bce+dice_weight*dice
    return (loss, {"bce": bce, "dice": dice}) if return_components else loss


def temporal_loss(current, previous, grid, weight, sample_weight=None):
    previous_probability = F.grid_sample(previous.sigmoid().detach().float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    error = (current[:, 0].sigmoid().float()-previous_probability).abs()
    values = (error*weight).flatten(1).sum(1)/weight.flatten(1).sum(1).clamp_min(1)
    if sample_weight is not None:
        values = values*sample_weight.to(device=values.device, dtype=values.dtype).reshape(-1)
    return values.mean()


def receiver_order_loss(current, previous, following, onset, recovery, sample_weight=None, margin=0.5):
    """Weak relative ordering at observed transitions, NOT 0/1 labels for brighter endpoints."""
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Receiver ordering margin must be finite and positive")
    current, previous, following = (item[:, 0].float() for item in (current, previous, following))
    errors = F.relu(margin-current+previous)*onset + F.relu(margin-current+following)*recovery
    count = (onset+recovery).flatten(1).sum(1)
    values = errors.flatten(1).sum(1)/count.clamp_min(1)
    if sample_weight is not None:
        values *= sample_weight.to(values)
    return values.sum()/(count > 0).sum().clamp_min(1)


def evaluate(model, rows, size, device, rank=0, world=1):
    # Uneven, non-padding shards are evaluated without a DDP wrapper.
    loader = DataLoader(ShadowDataset(rows[rank::world], size), batch_size=1, num_workers=0)
    states = {source: BinaryMetrics() for source in sorted({row["label_source"] for row in rows})}
    histograms = {source: np.zeros((2, 256), np.int64) for source in states}
    model.eval()
    temporal_error, temporal_count = 0.0, 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device))["out"]
            probability = logits[:, 0].sigmoid().cpu().numpy()
            if not np.isfinite(probability).all():
                raise RuntimeError("Nonfinite dev segmentation probabilities")
            source, target = batch["source"][0], batch["target"][0].numpy()
            states[source].update(probability[0] >= 0.5, target)
            known = (target == 0) | (target == 1)
            bucket = np.minimum((probability[0][known]*256).astype(np.int64), 255)
            histograms[source] += np.bincount(target[known].astype(np.int64)*256+bucket,
                                              minlength=512).reshape(2, 256)
            count = float(batch["temporal_weight"].sum())
            if count > 0:
                previous = model(batch["previous_image"].to(device))["out"]
                error = temporal_loss(logits, previous, batch["temporal_grid"].to(device), batch["temporal_weight"].to(device))
                temporal_error += float(error)*count
                temporal_count += count
    if world > 1:
        for source, state in states.items():
            values = torch.as_tensor(state.confusion, device=device)
            dist.all_reduce(values)
            state.confusion = values.cpu().numpy()
            histogram = torch.as_tensor(histograms[source], device=device)
            dist.all_reduce(histogram)
            histograms[source] = histogram.cpu().numpy()
        totals = torch.tensor([temporal_error, temporal_count], device=device, dtype=torch.float64)
        dist.all_reduce(totals)
        temporal_error, temporal_count = totals.cpu().tolist()
    return {source: {**state.summary(), "score_diagnostics": pixel_score_diagnostics(histograms[source])}
            for source, state in states.items()}, {"weighted_probability_MAE": temporal_error/temporal_count if temporal_count else None, "weight_sum": temporal_count}


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
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--pixel-balance", choices=["loss", "none"], default="loss")
    parser.add_argument("--scene-sampling", choices=["image", "uniform"], default="image")
    parser.add_argument("--positive-only-ablation", action="store_true",
                        help="Run the positive-only training control against unchanged two-class dev labels")
    parser.add_argument("--max-pixel-weight", type=float, default=50.0)
    parser.add_argument("--temporal-weight", type=float, default=0.2)
    parser.add_argument("--receiver-pair-weight", type=float, default=0.0,
                        help="Experimental relative transition loss; does not create endpoint NO labels")
    parser.add_argument("--receiver-pair-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--allow-weak-labels", action="store_true")
    parser.add_argument("--weak-weight", type=float, default=0.25)
    parser.add_argument("--backbone-weights", choices=["none", "imagenet"], default="none")
    parser.add_argument("--init-checkpoint", help="Optional segmenter checkpoint for pretraining -> adaptation")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, *args.size) < 1:
        raise ValueError("epochs, batch-size and accumulation must be positive")
    if (not 0 < args.weak_weight <= 1 or args.workers < 0 or args.temporal_weight < 0
            or not np.isfinite([args.dice_weight, args.max_pixel_weight, args.receiver_pair_weight, args.receiver_pair_margin]).all()
            or args.dice_weight < 0 or args.max_pixel_weight < 1 or args.receiver_pair_weight < 0 or args.receiver_pair_margin <= 0):
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
    effective_mass = np.zeros(2, dtype=np.float64)
    resized_counts = np.zeros(2, dtype=np.int64)
    receiver_pair_rows = 0
    usable, row_mass = [], []
    for row in train:
        mask = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(row["target"])
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("Training targets must be 0/1/255 trimaps")
        counts = np.array([(mask == 0).sum(), (mask == 1).sum()])
        class_counts += counts
        mask = cv2.resize(mask, (args.size[1], args.size[0]), interpolation=cv2.INTER_NEAREST)
        resized = np.array([(mask == 0).sum(), (mask == 1).sum()])
        resized_counts += resized
        if resized.sum():
            base = (args.weak_weight if row["label_source"] in WEAK_SOURCES else 1.0)*float(row.get("sample_weight", 1.0))
            if not np.isfinite(base) or base <= 0:
                raise ValueError("Invalid effective segmentation sample weight")
            row_mass.append(base*resized/resized.sum())
            receiver_pair_rows += int(bool(row.get("physics_transition")) and resized[1] > 0)
            usable.append(row)
    train = usable
    if not train:
        raise ValueError("No known training pixels survive resize")
    probabilities = scene_probabilities(train, args.scene_sampling)
    effective_mass = (np.asarray(row_mass)*probabilities[:, None]).sum(axis=0)*len(train)
    positive_only = args.positive_only_ablation and resized_counts[0] == 0 and resized_counts[1] > 0
    if (resized_counts == 0).any() and not positive_only:
        raise ValueError("Both known classes must survive the actual training resize; increase resolution or fix labels")
    pixel_weights = binary_risk_weights(effective_mass.tolist(), args.max_pixel_weight) if args.pixel_balance == "loss" and not positive_only else [1.0, 1.0]
    if args.receiver_pair_weight > 0 and not receiver_pair_rows:
        raise ValueError("Receiver-pair loss requested but no positive transition rows exist; rebuild physics data first")
    dev_counts = np.zeros(2, dtype=np.int64)
    for row in dev:
        mask = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(row["target"])
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("Dev targets must be 0/1/255 trimaps")
        mask = cv2.resize(mask, (args.size[1], args.size[0]), interpolation=cv2.INTER_NEAREST)
        dev_counts += np.array([(mask == 0).sum(), (mask == 1).sum()])
    if (dev_counts == 0).any():
        raise ValueError("Dev masks must contain known shadow and non-shadow pixels")
    train_scenes = {row["scene_name"] for row in train}
    dev_scenes = {row["scene_name"] for row in dev}
    initial_checkpoint = None
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        train_scenes, dev_scenes = extend_scene_history(train_scenes, dev_scenes, initial_checkpoint)
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
        write_json(output/"protocol.json", {**vars(args), "train_sha256": digest(args.train), "dev_sha256": digest(args.dev), "train_scenes": sorted(train_scenes), "dev_scenes": sorted(dev_scenes), "pixel_counts": class_counts.tolist(), "resized_pixel_counts": resized_counts.tolist(), "dev_resized_pixel_counts": dev_counts.tolist(), "effective_class_mass": effective_mass.tolist(), "pixel_loss_weights": pixel_weights, "sampler": "uniform scene then uniform image with replacement; no class sampling" if args.scene_sampling == "uniform" else "ordinary shuffled/distributed; no class oversampling", "approved_for_relabeling": False,
            "expected_scene_sample_fraction": {scene: float(sum(probabilities[index] for index, row in enumerate(train) if row["scene_name"] == scene)) for scene in sorted({row["scene_name"] for row in train})},
            "negative_weight_contract": "per-row weak-negative loss coefficient, not calibrated confidence; class balancing computed before reliability weighting",
            "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("segmentation.py", "readiness.py", "scene_sampling.py")},
            "initialization_sha256": digest(args.init_checkpoint) if args.init_checkpoint else None})
        print({"train_resized_pixels": resized_counts.tolist(), "effective_class_mass": effective_mass.tolist(),
               "pixel_loss_weights": pixel_weights, "receiver_pair_rows": receiver_pair_rows}, flush=True)
    if world > 1:
        dist.barrier()
    sampler = (SceneBalancedSampler(train, args.seed, rank, world) if args.scene_sampling == "uniform"
               else DistributedSampler(ShadowDataset(train, args.size), seed=args.seed) if world > 1 else None)
    loader = DataLoader(ShadowDataset(train, args.size, augment=True, receiver_pairs=args.receiver_pair_weight > 0), batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=args.workers, worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    model = build_segmenter(args.backbone_weights).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model"])
        del initial_checkpoint
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
        pair_weight_sum, pair_loss_sum, pair_batches = 0.0, 0.0, 0
        train_totals = torch.zeros(6, device=device, dtype=torch.float64)
        for step, batch in enumerate(tqdm(loader, desc=f"segmenter epoch {epoch}", disable=rank != 0)):
            start = (step//args.accumulation)*args.accumulation
            window = min(args.accumulation, len(loader)-start)
            boundary = (step+1) % args.accumulation == 0 or step+1 == len(loader)
            sync = contextlib.nullcontext() if world == 1 or boundary else model.no_sync()
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    images = batch["image"].to(device)
                    use_temporal = args.temporal_weight > 0 and bool(batch["temporal_weight"].any())
                    use_receiver = args.receiver_pair_weight > 0 and bool((batch["onset_weight"]+batch["recovery_weight"]).any())
                    input_parts = [images]
                    if use_temporal:
                        input_parts.append(batch["previous_image"].to(device))
                    if use_receiver:
                        input_parts.extend([batch["receiver_previous"].to(device), batch["receiver_next"].to(device)])
                    inputs = torch.cat(input_parts)
                    predictions = model(inputs)["out"]
                    logits = predictions[:images.shape[0]]
                    weights = torch.tensor([args.weak_weight if source in WEAK_SOURCES else 1.0 for source in batch["source"]], device=device)
                    weights = weights*batch["sample_weight"].to(device)
                    loss, terms = masked_loss(logits, batch["target"].to(device), args.dice_weight, weights, pixel_weights,
                                       batch["negative_weight"].to(device), return_components=True)
                    temporal_value, pair_value = loss.new_zeros(()), loss.new_zeros(())
                    if use_temporal:
                        previous = predictions[images.shape[0]:2*images.shape[0]]
                        temporal_value = temporal_loss(logits, previous, batch["temporal_grid"].to(device), batch["temporal_weight"].to(device), weights)
                        loss = loss + args.temporal_weight*temporal_value
                    if use_receiver:
                        offset = images.shape[0]*(2 if use_temporal else 1)
                        before, after = predictions[offset:].chunk(2)
                        pair_loss = receiver_order_loss(logits, before, after, batch["onset_weight"].to(device),
                            batch["recovery_weight"].to(device), weights, args.receiver_pair_margin)
                        loss = loss + args.receiver_pair_weight*pair_loss
                        pair_value = pair_loss
                        pair_weight_sum += float((batch["onset_weight"]+batch["recovery_weight"]).sum())
                        pair_loss_sum += float(pair_loss.detach())
                        pair_batches += 1
                    train_totals[:5] += torch.stack([loss, terms["bce"], terms["dice"], temporal_value, pair_value]).detach().double()*len(images)
                    train_totals[5] += len(images)
                    loss = loss/window
                scaler.scale(loss).backward()
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        raw = model.module if world > 1 else model
        if world > 1:
            dist.all_reduce(train_totals)
        totals = train_totals.cpu().tolist()
        train_metrics = {name: totals[index]/totals[5] if totals[5] else None
            for index, name in enumerate(("total", "bce", "dice", "temporal", "receiver_pair"))}
        train_metrics.update(images=int(totals[5]), dice_weight=args.dice_weight,
            temporal_weight=args.temporal_weight, receiver_pair_weight=args.receiver_pair_weight,
            reduction="sample-mean before gradient accumulation scaling, across all DDP ranks")
        metrics, temporal_metrics = evaluate(raw, dev, args.size, device, rank, world)
        readiness = segmentation_readiness(metrics)
        preferred = [v["iou"] for k, v in metrics.items() if k in GT_SOURCES and v["iou"] is not None]
        values = preferred or [v["iou"] for v in metrics.values() if v["iou"] is not None]
        score = float(np.mean(values)) if values else -1.0
        if rank == 0:
            state = {"model": raw.state_dict(), "epoch": epoch, "size": args.size, "args": vars(args), "dev_metrics_by_source": metrics, "dev_temporal_metrics": temporal_metrics, "readiness": readiness, "train_scenes": sorted(train_scenes), "dev_scenes": sorted(dev_scenes)}
            torch.save(state, output/"last.pth")
            if readiness["noncollapsed"] and score > best:
                best = score
                torch.save(state, output/"best.pth")
            history.append({"epoch": epoch, "train_loss": train_metrics, "dev_metrics_by_source": metrics, "temporal": temporal_metrics, "readiness": readiness,
                "receiver_pairs_rank0": {"active_transition_pixels": pair_weight_sum, "batches": pair_batches,
                    "mean_relative_loss": pair_loss_sum/pair_batches if pair_batches else None}})
            write_json(output/"history.json", history)
            write_json(output/"readiness.json", {"has_noncollapsed_best": best >= 0,
                "last": readiness, "approved_for_relabeling": False})
            compact = {source: {key: value for key, value in result.items() if key != "score_diagnostics"}
                       for source, result in metrics.items()}
            print({"epoch": epoch, "train_loss": train_metrics, "dev_metrics_by_source": compact, "temporal": temporal_metrics,
                   "noncollapsed": readiness["noncollapsed"], "reasons": readiness["reasons"]}, flush=True)
    usable_best = torch.tensor(int(best >= 0), device=device)
    if world > 1:
        dist.broadcast(usable_best, src=0)
        dist.destroy_process_group()
    if not bool(usable_best):
        raise RuntimeError("All segmenter checkpoints failed non-collapse checks; last.pth is diagnostic only, no best.pth exported")


def predict_main():
    parser = argparse.ArgumentParser("Predict pixel masks and evaluate by label provenance.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-train-scenes", action="store_true", help="Explicit in-sample annotation, not held-out evaluation")
    parser.add_argument("--diagnostics-only", action="store_true", help="Inspect collapsed/legacy models; outputs are quarantined from relabeling")
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    readiness = require_segmenter_checkpoint(checkpoint, args.diagnostics_only)
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
    histograms = defaultdict(lambda: np.zeros((2, 256), np.int64))
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
            known = (target == 0) | (target == 1)
            bucket = np.minimum((probability[known]*256).astype(np.int64), 255)
            histograms[row["label_source"]] += np.bincount(target[known].astype(np.int64)*256+bucket,
                minlength=512).reshape(2, 256)
            path = output/"probabilities"/f"{index:07d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probability=probability.astype(np.float16))
            checked_write_image(output/"masks"/f"{index:07d}.png", (probability >= args.threshold).astype(np.uint8)*255)
            predictions.append({**row, "probability": str(path), "threshold": args.threshold,
                "diagnostics_only": args.diagnostics_only, "segmenter_readiness": readiness,
                "probability_sha256": digest(path), "source_image_sha256": digest(row["image"]),
                "source_target_sha256": digest(row["target"]),
                "source_ground_sha256": digest(row["ground_support"]) if row.get("ground_support") else None})
    write_rows(output/"predictions.jsonl", predictions)
    write_json(output/"segmentation_metrics.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(args.manifest),
        "predictions_sha256": digest(output/"predictions.jsonl"), "threshold": args.threshold,
        "checkpoint_train_scenes": checkpoint.get("train_scenes"), "checkpoint_dev_scenes": checkpoint.get("dev_scenes"),
        "diagnostics_only": args.diagnostics_only, "segmenter_readiness": readiness,
        "in_sample_annotation": bool(overlap), "overlapping_train_scenes": overlap,
        "by_label_source": {source: {**state.summary(), "score_diagnostics": pixel_score_diagnostics(histograms[source])}
                            for source, state in states.items()},
        "temporal_consistency": {"weighted_probability_MAE": temporal_error/temporal_count if temporal_count else None, "weight_sum": temporal_count}})
