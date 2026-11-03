from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to train the BEVFormer OCC model.") from exc

from shadow_occ.config import load_config
from shadow_occ.data.bevformer_dataset import NuScenesOccSequenceDataset
from shadow_occ.models.bevformer_occ import build_bevformer_occ_shadow
from shadow_occ.models.losses import weak_shadow_total_loss
from shadow_occ.utils.io import ensure_dir
from shadow_occ.utils.seed import seed_everything, seed_worker


def main() -> None:
    parser = argparse.ArgumentParser("Train local BEVFormer OCC + Shadow model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataroot")
    parser.add_argument("--occ-root")
    parser.add_argument("--output-root")
    parser.add_argument("--shadow-label-root")
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true", help="Run a random forward/backward smoke test.")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return
    train(args)


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, overrides=_overrides_from_args(args))
    seed = cfg.train.seed if args.seed is None else args.seed
    seed_everything(seed, cfg.train.deterministic)
    ddp = _init_distributed()
    device = _resolve_train_device(cfg.train.device if args.device is None else args.device, ddp)
    out_dir = ensure_dir(Path(cfg.data.output_root) / "checkpoints") if _is_main_process() else None

    dataset = NuScenesOccSequenceDataset(cfg, split="train")
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if ddp["distributed"] else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if cfg.train.num_workers > 0 else None,
    )
    model = build_bevformer_occ_shadow(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    start_epoch = 0

    if cfg.train.resume:
        checkpoint = torch.load(cfg.train.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if ddp["distributed"]:
        # Ablation configs intentionally leave parts of the shadow branch
        # unused. Let DDP handle those parameters instead of failing at the
        # next iteration when fusion or shadow objectives are disabled.
        model = DDP(
            model,
            device_ids=[ddp["local_rank"]] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    model.train()
    global_step = 0
    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            pred = model(batch["images"], batch["ego_to_img"], batch["ego_to_global"])
            occ_loss = F.binary_cross_entropy_with_logits(pred["occ_logits"], batch["occ_target"])
            semantic_loss = F.cross_entropy(
                pred["semantic_logits"],
                batch["semantic_target"],
                ignore_index=cfg.occ.ignore_label,
            )
            target = _shadow_target(batch)
            losses = weak_shadow_total_loss(
                pred=pred,
                target=target,
                pred_prev_warped={"P_caster": pred["temporal_prev_P_caster"]},
                occ_loss=cfg.train.occ_loss_weight * occ_loss
                + cfg.train.semantic_loss_weight * semantic_loss,
                weights={
                    "shadow": cfg.train.shadow_loss_weight,
                    "root": cfg.train.shadow_loss_weight,
                    "tip": cfg.train.shadow_loss_weight,
                    "caster": cfg.train.shadow_loss_weight,
                    "direction": cfg.train.shadow_loss_weight,
                    "height": cfg.train.shadow_loss_weight,
                    "reliability": cfg.train.shadow_loss_weight,
                    "render": cfg.train.render_loss_weight,
                    "uncertainty": cfg.train.uncertainty_loss_weight,
                    "temporal": cfg.train.temporal_loss_weight,
                    "motion_smooth": cfg.train.motion_smooth_loss_weight,
                    "caster_geometry": cfg.train.caster_geometry_loss_weight,
                },
            )
            loss = losses["loss_total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if _is_main_process() and global_step % cfg.train.log_interval == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} occ={occ_loss.item():.4f} "
                    f"sem={semantic_loss.item():.4f} "
                    f"shadow={losses['loss_shadow_total'].item():.4f} "
                    f"temporal={losses['L_temporal_consistency'].item():.4f}"
                )
            global_step += 1

        if _is_main_process() and (epoch + 1) % cfg.train.save_interval == 0:
            assert out_dir is not None
            path = out_dir / f"bevformer_occ_shadow_epoch_{epoch:03d}.pth"
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_to_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                },
                path,
            )
            print(f"saved {path}")
    _cleanup_distributed()


def run_smoke() -> None:
    cfg = load_config(
        None,
        overrides={
            "data": {
                "image_size": [64, 96],
                "num_frames": 2,
                "cams": ["CAM_FRONT", "CAM_BACK"],
            },
            "bev": {"x_min": -4.0, "x_max": 4.0, "y_min": -4.0, "y_max": 4.0, "resolution": 2.0},
            "occ": {"pc_range": [-4.0, -4.0, -1.0, 4.0, 4.0, 3.0], "voxel_size": [2.0, 2.0, 1.0]},
            "model": {"bev_channels": 16, "image_feature_channels": 16, "num_bev_layers": 1, "num_heads": 4, "shadow_hidden_channels": 16, "z_anchors": [0.0, 1.0]},
        },
    )
    device = _device("cuda")
    model = build_bevformer_occ_shadow(cfg).to(device)
    b, t, n = 1, cfg.data.num_frames, len(cfg.data.cams)
    h, w = cfg.data.image_size
    images = torch.randn(b, t, n, 3, h, w, device=device)
    ego_to_img = torch.zeros(b, t, n, 3, 4, device=device)
    ego_to_img[..., 0, 0] = 60.0
    ego_to_img[..., 1, 1] = 60.0
    ego_to_img[..., 0, 2] = w / 2.0
    ego_to_img[..., 1, 2] = h / 2.0
    ego_to_img[..., 2, 2] = 1.0
    ego_to_global = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(b, t, 1, 1)
    out = model(images, ego_to_img, ego_to_global)
    target = torch.zeros_like(out["occ_logits"])
    semantic_target = torch.zeros(
        b,
        out["semantic_logits"].shape[2],
        out["semantic_logits"].shape[3],
        out["semantic_logits"].shape[4],
        dtype=torch.long,
        device=device,
    )
    loss = F.binary_cross_entropy_with_logits(out["occ_logits"], target)
    loss = loss + F.cross_entropy(out["semantic_logits"], semantic_target)
    loss.backward()
    print(
        "BEVFormer smoke passed: "
        f"occ_logits={tuple(out['occ_logits'].shape)} "
        f"semantic_logits={tuple(out['semantic_logits'].shape)} "
        f"P_shadow={tuple(out['P_shadow'].shape)}"
    )


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if args.dataroot:
        data["dataroot"] = args.dataroot
    if args.occ_root:
        data["occ_root"] = args.occ_root
    if args.output_root:
        data["output_root"] = args.output_root
    if args.shadow_label_root:
        data["shadow_label_root"] = args.shadow_label_root
    if args.scene:
        data["scene_names"] = args.scene
    if args.max_samples is not None:
        data["max_samples"] = args.max_samples
    overrides: dict[str, Any] = {"data": data} if data else {}
    if args.seed is not None:
        overrides["train"] = {"seed": args.seed}
    return overrides


def _shadow_target(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    keys = [
        "shadow_mask",
        "shadow_root_heatmap",
        "shadow_tip_heatmap",
        "shadow_caster_heatmap",
        "shadow_direction",
        "height_target",
        "H_occ",
        "shadow_reliability",
        "rendered_shadow_prior",
    ]
    return {key: batch[key] for key in keys}


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _init_distributed() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }


def _resolve_train_device(name: str, ddp: dict[str, int | bool]) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        if ddp["distributed"]:
            local_rank = int(ddp["local_rank"])
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cuda")
    return _device(name)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
