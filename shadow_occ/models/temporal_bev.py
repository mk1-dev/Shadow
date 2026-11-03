from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use shadow_occ.models.") from exc

from shadow_occ.config import BevConfig
from shadow_occ.models.shadow_head import ConvBlock, ShadowGeometryHead


class TemporalBEVAligner(nn.Module):
    """Warp multi-frame BEV features into the current ego frame."""

    def __init__(self, bev_cfg: BevConfig) -> None:
        super().__init__()
        self.x_min = float(bev_cfg.x_min)
        self.y_min = float(bev_cfg.y_min)
        self.resolution = float(bev_cfg.resolution)

    def forward(
        self,
        bev_features: torch.Tensor,
        ego_to_global: torch.Tensor,
    ) -> torch.Tensor:
        """Align features to the last frame.

        Args:
            bev_features: [B, T, C, H, W]
            ego_to_global: [B, T, 4, 4]

        Returns:
            [B, T, C, H, W] feature volume in frame t coordinates.
        """
        if bev_features.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(bev_features.shape)}")
        b, t, c, h, w = bev_features.shape
        target_pose = ego_to_global[:, -1]
        aligned = []
        for frame_idx in range(t):
            if frame_idx == t - 1:
                aligned.append(bev_features[:, frame_idx])
            else:
                aligned.append(
                    warp_bev_feature(
                        bev_features[:, frame_idx],
                        src_ego_to_global=ego_to_global[:, frame_idx],
                        dst_ego_to_global=target_pose,
                        x_min=self.x_min,
                        y_min=self.y_min,
                        resolution=self.resolution,
                    )
                )
        return torch.stack(aligned, dim=1)


class TemporalBEVShadowNet(nn.Module):
    """Temporal BEV proposal network for the B design.

    The image backbone and lift-splat / transformer BEV encoder stay outside
    this module. This module receives already-built BEV features from at least
    two frames, ego-aligns them, aggregates the temporal volume, then predicts
    shadow geometry and reliability.
    """

    def __init__(
        self,
        bev_cfg: BevConfig,
        in_channels: int,
        num_frames: int,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError("TemporalBEVShadowNet expects at least two frames.")
        self.aligner = TemporalBEVAligner(bev_cfg)
        self.temporal_fusion = ConvBlock(in_channels * num_frames, in_channels)
        self.head = ShadowGeometryHead(in_channels, hidden_channels)

    def forward(
        self,
        bev_features: torch.Tensor,
        ego_to_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        aligned = self.aligner(bev_features, ego_to_global)
        b, t, c, h, w = aligned.shape
        fused = self.temporal_fusion(aligned.reshape(b, t * c, h, w))
        out = self.head(fused)
        out["temporal_bev_volume"] = aligned
        out["temporal_bev_feature"] = fused
        return out


def warp_bev_feature(
    feature: torch.Tensor,
    src_ego_to_global: torch.Tensor,
    dst_ego_to_global: torch.Tensor,
    x_min: float,
    y_min: float,
    resolution: float,
) -> torch.Tensor:
    """Sample a source-frame BEV feature on destination-frame BEV coordinates."""
    b, _c, h, w = feature.shape
    device = feature.device
    dtype = feature.dtype

    ys = y_min + (torch.arange(h, device=device, dtype=dtype) + 0.5) * resolution
    xs = x_min + (torch.arange(w, device=device, dtype=dtype) + 0.5) * resolution
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    zeros = torch.zeros_like(xx)
    ones = torch.ones_like(xx)
    dst_points = torch.stack([xx, yy, zeros, ones], dim=-1).view(1, h * w, 4).repeat(b, 1, 1)

    src_from_dst = torch.linalg.inv(src_ego_to_global) @ dst_ego_to_global
    src_points = torch.bmm(dst_points, src_from_dst.transpose(1, 2))
    x_src = src_points[:, :, 0]
    y_src = src_points[:, :, 1]

    j = (x_src - x_min) / resolution - 0.5
    i = (y_src - y_min) / resolution - 0.5
    grid_x = 2.0 * j / max(w - 1, 1) - 1.0
    grid_y = 2.0 * i / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(b, h, w, 2)
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
