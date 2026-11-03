from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to evaluate the BEVFormer OCC model.") from exc

try:
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
except ImportError:  # pragma: no cover
    dist = None
    DistributedSampler = None

from shadow_occ.config import load_config
from shadow_occ.data.bevformer_dataset import NuScenesOccSequenceDataset
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.metrics.occ3d_metrics import (
    HeightPriorCorrelation,
    Occ3DMetrics,
    PseudoShadowMetrics,
    pred_semantic_from_logits,
)
from shadow_occ.models.bevformer_occ import build_bevformer_occ_shadow
from shadow_occ.utils.io import ensure_dir, write_json
from shadow_occ.utils.seed import seed_everything, seed_worker


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate BEVFormer OCC + Shadow model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataroot")
    parser.add_argument("--occ-root")
    parser.add_argument("--output-root")
    parser.add_argument("--shadow-label-root")
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--metrics-name", default="occ3d_metrics.json")
    args = parser.parse_args()

    distributed = _init_distributed()

    overrides: dict[str, Any] = {"data": {}}
    if args.dataroot:
        overrides["data"]["dataroot"] = args.dataroot
    if args.occ_root:
        overrides["data"]["occ_root"] = args.occ_root
    if args.output_root:
        overrides["data"]["output_root"] = args.output_root
    if args.shadow_label_root:
        overrides["data"]["shadow_label_root"] = args.shadow_label_root
    if args.scene:
        overrides["data"]["scene_names"] = args.scene
    if args.max_samples is not None:
        overrides["data"]["max_samples"] = args.max_samples
    cfg = load_config(args.config, overrides=overrides)
    seed_everything(cfg.train.seed if args.seed is None else args.seed, cfg.train.deterministic)
    device = _resolve_device(args.device or cfg.train.device, distributed)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    dataset = NuScenesOccSequenceDataset(cfg, split="val")
    sampler = DistributedSampler(dataset, shuffle=False) if distributed["distributed"] else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if cfg.train.num_workers > 0 else None,
    )
    model = build_bevformer_occ_shadow(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    grid = BEVGrid.from_config(cfg.bev)
    metrics = Occ3DMetrics(
        num_classes=cfg.occ.num_classes,
        free_label=cfg.occ.free_label,
        ignore_label=cfg.occ.ignore_label,
        semantic_eval_class_ids=cfg.occ.semantic_eval_class_ids,
        grid=grid,
        z_min=cfg.occ.pc_range[2],
        z_size=cfg.occ.voxel_size[2],
        distance_bins_m=cfg.eval.distance_bins_m,
    )
    height_metrics = HeightPriorCorrelation(
        grid=grid,
        distance_bins_m=cfg.eval.distance_bins_m,
    )
    shadow_metrics = PseudoShadowMetrics()
    start_time = time.perf_counter()

    if _is_main_process():
        print(
            f"Starting evaluation: {len(dataset)} samples, "
            f"{len(loader)} samples on rank 0/{distributed['world_size']}"
        )

    with torch.no_grad():
        batches = tqdm(
            loader,
            total=len(loader),
            desc="OCC evaluation",
            unit="sample",
            disable=not _is_main_process(),
        )
        for batch in batches:
            batch = _move_batch(batch, device)
            out = model(batch["images"], batch["ego_to_img"], batch["ego_to_global"])
            semantic_logits = out["semantic_logits"].detach().cpu().numpy()
            occ_logits = out["occ_logits"].detach().cpu().numpy()
            h_shadow = out["H_prior"].detach().cpu().numpy()
            h_occ = batch["H_occ"].detach().cpu().numpy()
            target = batch["semantic_target"].detach().cpu().numpy()
            for idx in range(semantic_logits.shape[0]):
                pred_sem = pred_semantic_from_logits(
                    semantic_logits[idx],
                    occ_logits[idx],
                    free_label=cfg.occ.free_label,
                )
                metrics.update(pred_sem, target[idx])
                height_metrics.update(h_shadow[idx, 0], h_occ[idx, 0])
                shadow_metrics.update(
                    pred={
                        "P_shadow": out["P_shadow"].detach().cpu().numpy()[idx],
                        "P_root": out["P_root"].detach().cpu().numpy()[idx],
                        "P_tip": out["P_tip"].detach().cpu().numpy()[idx],
                        "P_caster": out["P_caster"].detach().cpu().numpy()[idx],
                        "V_shadow": out["V_shadow"].detach().cpu().numpy()[idx],
                        "U_shadow": out["U_shadow"].detach().cpu().numpy()[idx],
                    },
                    target={
                        "shadow_mask": batch["shadow_mask"].detach().cpu().numpy()[idx],
                        "shadow_root_heatmap": batch["shadow_root_heatmap"].detach().cpu().numpy()[idx],
                        "shadow_tip_heatmap": batch["shadow_tip_heatmap"].detach().cpu().numpy()[idx],
                        "shadow_caster_heatmap": batch["shadow_caster_heatmap"].detach().cpu().numpy()[idx],
                        "shadow_direction": batch["shadow_direction"].detach().cpu().numpy()[idx],
                        "shadow_reliability": batch["shadow_reliability"].detach().cpu().numpy()[idx],
                    },
                )

    _reduce_metric_state(metrics, height_metrics, shadow_metrics, device, distributed)
    summary = metrics.summarize()
    summary.update(height_metrics.summarize())
    summary.update(shadow_metrics.summarize())
    summary["checkpoint"] = str(args.checkpoint)
    summary["seed"] = int(cfg.train.seed if args.seed is None else args.seed)
    summary["num_samples"] = int(len(dataset))
    summary["elapsed_seconds"] = float(time.perf_counter() - start_time)
    summary["samples_per_second"] = float(
        len(dataset) / max(summary["elapsed_seconds"], 1e-6)
    )
    if _is_main_process():
        out_path = ensure_dir(Path(cfg.data.output_root) / "metrics") / args.metrics_name
        write_json(out_path, summary)
        print(
            {
                "Occupancy_mIoU": summary["Occupancy_mIoU"],
                "RayIoU": summary["RayIoU"],
                "Semantic_mIoU": summary["Semantic_mIoU"],
                "HeightPrior_corr": summary["HeightPrior_corr"],
            }
        )
        for row in summary["Occupancy_mIoU_by_distance"]:
            print(row)
        for row in summary["RayIoU_by_distance"]:
            print(row)
        for row in summary["HeightPrior_corr_by_distance"]:
            print(row)
        print(f"wrote {out_path}")
    _cleanup_distributed()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _init_distributed() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if dist is None:
            raise RuntimeError("torch.distributed is required for multi-GPU evaluation.")
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
    return {"distributed": distributed, "world_size": world_size, "rank": rank, "local_rank": local_rank}


def _resolve_device(name: str, distributed: dict[str, int | bool]) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        if distributed["distributed"]:
            local_rank = int(distributed["local_rank"])
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cuda")
    return torch.device(name)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _cleanup_distributed() -> None:
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _reduce_metric_state(
    metrics: Occ3DMetrics,
    heights: HeightPriorCorrelation,
    shadows: PseudoShadowMetrics,
    device: torch.device,
    distributed: dict[str, int | bool],
) -> None:
    if not distributed["distributed"]:
        return

    def reduce_array(array: Any) -> np.ndarray:
        tensor = torch.as_tensor(array, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().numpy()

    metrics.semantic_confusion = reduce_array(metrics.semantic_confusion).astype(np.int64)
    metrics.occupancy_confusion = reduce_array(metrics.occupancy_confusion).astype(np.int64)
    for key, stats in metrics.occupancy_distance_confusion.items():
        metrics.occupancy_distance_confusion[key] = reduce_array(stats).astype(np.int64)
    for key, stats in metrics.ray_stats.items():
        values = reduce_array(np.asarray([stats["intersection"], stats["union"]], dtype=np.int64))
        stats["intersection"], stats["union"] = int(values[0]), int(values[1])
    for stats in heights.stats.values():
        keys = ["n", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy", "sum_abs_err"]
        values = reduce_array(np.asarray([stats[key] for key in keys], dtype=np.float64))
        for key, value in zip(keys, values):
            stats[key] = float(value)
    for confusion in shadows.confusion.values():
        confusion[:] = reduce_array(confusion).astype(np.int64)
    values = reduce_array(
        np.asarray(
            [
                shadows.direction_cosine_sum,
                shadows.direction_count,
                shadows.reliability_abs_error,
                shadows.reliability_count,
            ],
            dtype=np.float64,
        )
    )
    shadows.direction_cosine_sum = float(values[0])
    shadows.direction_count = int(values[1])
    shadows.reliability_abs_error = float(values[2])
    shadows.reliability_count = int(values[3])


if __name__ == "__main__":
    main()
