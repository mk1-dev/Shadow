from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use shadow_occ.models.") from exc

from shadow_occ.config import ProjectConfig
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.models.shadow_head import HeightPriorFusion, ShadowGeometryHead
from shadow_occ.models.temporal_bev import TemporalBEVAligner


class ImageBackbone(nn.Module):
    """Small self-contained CNN backbone for multi-camera BEVFormer experiments."""

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNReLU(3, 32, stride=2),
            _ConvBNReLU(32, 64, stride=2),
            _ConvBNReLU(64, 96, stride=2),
            _ConvBNReLU(96, out_channels, stride=1),
            _ConvBNReLU(out_channels, out_channels, stride=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class BEVFormerSpatialCrossAttention(nn.Module):
    """Project BEV reference points into camera feature maps and attend over cameras/z anchors."""

    def __init__(
        self,
        embed_dims: int,
        num_cams: int,
        num_z_anchors: int,
        image_size: tuple[int, int],
        dropout: float = 0.1,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_z_anchors = num_z_anchors
        self.image_h, self.image_w = image_size
        self.chunk_size = chunk_size
        self.attn = nn.Linear(embed_dims, num_cams * num_z_anchors)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        camera_features: torch.Tensor,
        ego_to_img: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        b, num_query, _ = query.shape
        outputs = []
        for start in range(0, num_query, self.chunk_size):
            end = min(start + self.chunk_size, num_query)
            q = query[:, start:end]
            ref = reference_points[start:end]
            sampled, valid = self._sample_camera_features(camera_features, ego_to_img, ref)
            logits = self.attn(q).view(b, end - start, self.num_cams, self.num_z_anchors)
            valid = valid.permute(0, 2, 1, 3)
            logits = logits.masked_fill(~valid, -1.0e4)
            weights = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)
            weights = weights * valid.float()
            weights = weights / weights.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
            mixed = (weights.unsqueeze(-1) * sampled).sum(dim=(2, 3))
            outputs.append(self.output_proj(mixed))
        return query + self.dropout(torch.cat(outputs, dim=1))

    def _sample_camera_features(
        self,
        camera_features: torch.Tensor,
        ego_to_img: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, num_cams, c, _h, _w = camera_features.shape
        q, z = reference_points.shape[:2]
        points = reference_points.reshape(q * z, 4)
        projected = torch.einsum("bnij,mj->bnmi", ego_to_img, points)
        depth = projected[..., 2]
        u = projected[..., 0] / depth.clamp_min(1.0e-5)
        v = projected[..., 1] / depth.clamp_min(1.0e-5)
        grid_x = 2.0 * u / max(self.image_w - 1, 1) - 1.0
        grid_y = 2.0 * v / max(self.image_h - 1, 1) - 1.0
        valid = (depth > 1.0e-5) & (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)

        feature_flat = camera_features.reshape(b * num_cams, c, camera_features.shape[-2], camera_features.shape[-1])
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(b * num_cams, q * z, 1, 2)
        sampled = F.grid_sample(
            feature_flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, num_cams, q, z, c)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()
        return sampled, valid.reshape(b, num_cams, q, z)


class BEVFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_cams: int,
        num_z_anchors: int,
        image_size: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial_attn = BEVFormerSpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            num_z_anchors=num_z_anchors,
            image_size=image_size,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        camera_features: torch.Tensor,
        ego_to_img: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        query = self.norm1(self.spatial_attn(query, camera_features, ego_to_img, reference_points))
        query = self.norm2(query + self.dropout(self.ffn(query)))
        return query


class BEVFormerEncoder(nn.Module):
    def __init__(
        self,
        grid: BEVGrid,
        embed_dims: int,
        num_cams: int,
        z_anchors: tuple[float, ...],
        image_size: tuple[int, int],
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.embed_dims = embed_dims
        num_query = grid.height * grid.width
        self.bev_queries = nn.Parameter(torch.empty(num_query, embed_dims))
        self.bev_pos = nn.Parameter(torch.empty(num_query, embed_dims))
        nn.init.normal_(self.bev_queries, std=0.02)
        nn.init.normal_(self.bev_pos, std=0.02)
        self.layers = nn.ModuleList(
            [
                BEVFormerEncoderLayer(
                    embed_dims=embed_dims,
                    num_cams=num_cams,
                    num_z_anchors=len(z_anchors),
                    image_size=image_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer("reference_points", _make_reference_points(grid, z_anchors), persistent=False)

    def forward(self, camera_features: torch.Tensor, ego_to_img: torch.Tensor) -> torch.Tensor:
        b = camera_features.shape[0]
        query = (self.bev_queries + self.bev_pos).unsqueeze(0).repeat(b, 1, 1)
        ref = self.reference_points.to(device=query.device, dtype=query.dtype)
        for layer in self.layers:
            query = layer(query, camera_features, ego_to_img, ref)
        return query.transpose(1, 2).reshape(b, self.embed_dims, self.grid.height, self.grid.width)


class TemporalSelfAttentionFuser(nn.Module):
    """BEVFormer-style temporal attention after ego-motion alignment."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, aligned_bev: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = aligned_bev.shape
        seq = aligned_bev.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        query = seq[:, -1:, :]
        attended, _ = self.attn(query, seq, seq, need_weights=False)
        out = self.norm1(query + self.dropout(attended))
        out = self.norm2(out + self.dropout(self.ffn(out)))
        return out[:, 0].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class OccupancyDecoder(nn.Module):
    def __init__(self, channels: int, num_z: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            _ConvBNReLU(channels, channels),
            _ConvBNReLU(channels, channels),
            nn.Conv2d(channels, num_z, kernel_size=1),
        )

    def forward(self, bev_feature: torch.Tensor) -> torch.Tensor:
        return self.decoder(bev_feature)


class SemanticOccupancyDecoder(nn.Module):
    def __init__(self, channels: int, num_z: int, num_classes: int) -> None:
        super().__init__()
        self.num_z = num_z
        self.num_classes = num_classes
        self.decoder = nn.Sequential(
            _ConvBNReLU(channels, channels),
            _ConvBNReLU(channels, channels),
            nn.Conv2d(channels, num_z * num_classes, kernel_size=1),
        )

    def forward(self, bev_feature: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(bev_feature)
        b, _c, h, w = logits.shape
        return logits.view(b, self.num_classes, self.num_z, h, w)


class BEVFormerOccShadow(nn.Module):
    """Complete local BEVFormer occupancy model with the shadow-prior branch."""

    def __init__(self, cfg: ProjectConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = BEVGrid.from_config(cfg.bev)
        self.num_frames = int(cfg.data.num_frames)
        self.num_cams = len(cfg.data.cams)
        self.num_z = int(round((cfg.occ.pc_range[5] - cfg.occ.pc_range[2]) / cfg.occ.voxel_size[2]))

        self.image_backbone = ImageBackbone(out_channels=cfg.model.image_feature_channels)
        self.channel_proj = nn.Conv2d(cfg.model.image_feature_channels, cfg.model.bev_channels, kernel_size=1)
        self.spatial_encoder = BEVFormerEncoder(
            grid=self.grid,
            embed_dims=cfg.model.bev_channels,
            num_cams=self.num_cams,
            z_anchors=cfg.model.z_anchors,
            image_size=cfg.data.image_size,
            num_layers=cfg.model.num_bev_layers,
            dropout=cfg.model.dropout,
        )
        self.temporal_aligner = TemporalBEVAligner(cfg.bev)
        self.temporal_fuser = TemporalSelfAttentionFuser(
            channels=cfg.model.bev_channels,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
        )
        self.shadow_head = ShadowGeometryHead(
            in_channels=cfg.model.bev_channels,
            hidden_channels=cfg.model.shadow_hidden_channels,
        )
        self.shadow_fusion = HeightPriorFusion(
            bev_channels=cfg.model.bev_channels,
            shadow_channels=cfg.model.shadow_hidden_channels,
        )
        self.occ_decoder = OccupancyDecoder(cfg.model.bev_channels, self.num_z)
        self.semantic_decoder = SemanticOccupancyDecoder(
            cfg.model.bev_channels,
            self.num_z,
            cfg.occ.num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
        ego_to_img: torch.Tensor,
        ego_to_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, n, c, h, w = images.shape
        if t != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {t}.")
        if t < 2:
            raise ValueError("BEVFormerOccShadow requires at least two frames for temporal supervision.")
        if n != self.num_cams:
            raise ValueError(f"Expected {self.num_cams} cameras, got {n}.")

        flat_images = images.reshape(b * t * n, c, h, w)
        flat_features = self.channel_proj(self.image_backbone(flat_images))
        _, feat_c, feat_h, feat_w = flat_features.shape
        camera_features = flat_features.reshape(b, t, n, feat_c, feat_h, feat_w)

        bev_frames = []
        for frame_idx in range(t):
            bev = self.spatial_encoder(camera_features[:, frame_idx], ego_to_img[:, frame_idx])
            bev_frames.append(bev)
        bev_volume = torch.stack(bev_frames, dim=1)
        aligned_volume = self.temporal_aligner(bev_volume, ego_to_global)
        temporal_bev = self.temporal_fuser(aligned_volume)

        if self.cfg.model.use_shadow:
            shadow = self.shadow_head(temporal_bev)
        else:
            shadow = _empty_shadow_outputs(temporal_bev)

        if self.cfg.model.use_shadow and self.cfg.model.use_shadow_fusion:
            fused_bev = self.shadow_fusion(
                temporal_bev,
                shadow["F_shadow"],
                shadow["U_shadow"] if self.cfg.model.use_uncertainty_gate else None,
                shadow["H_prior"] if self.cfg.model.use_height_prior else None,
            )
        else:
            fused_bev = temporal_bev

        # The aligned per-frame predictions provide a real previous-frame
        # target for the temporal caster consistency loss. They are kept out of
        # the OCC decoder and only used by the training objective.
        if self.cfg.model.use_shadow:
            frame_shadow = self.shadow_head(
                aligned_volume.reshape(b * t, feat_c, aligned_volume.shape[-2], aligned_volume.shape[-1])
            )
            prev_caster = frame_shadow["P_caster"].reshape(
                b, t, 1, aligned_volume.shape[-2], aligned_volume.shape[-1]
            )[:, -2]
        else:
            prev_caster = temporal_bev.new_zeros(
                b, 1, aligned_volume.shape[-2], aligned_volume.shape[-1]
            )
        occ_logits = self.occ_decoder(fused_bev)
        semantic_logits = self.semantic_decoder(fused_bev)

        out = {
            "occ_logits": occ_logits,
            "semantic_logits": semantic_logits,
            "bev_feature": temporal_bev,
            "bev_feature_fused": fused_bev,
            "temporal_bev_volume": aligned_volume,
            "temporal_prev_P_caster": prev_caster,
        }
        out.update(shadow)
        return out


def build_bevformer_occ_shadow(cfg: ProjectConfig) -> BEVFormerOccShadow:
    return BEVFormerOccShadow(cfg)


def _empty_shadow_outputs(bev_feature: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return shape-compatible outputs for the strict camera-only baseline."""
    b, _, h, w = bev_feature.shape
    zeros = bev_feature.new_zeros
    return {
        "P_shadow": zeros(b, 1, h, w),
        "P_root": zeros(b, 1, h, w),
        "P_tip": zeros(b, 1, h, w),
        "P_caster": zeros(b, 1, h, w),
        "V_shadow": zeros(b, 2, h, w),
        "V_direction": zeros(b, 2, h, w),
        "H_prior": zeros(b, 1, h, w),
        "U_shadow": zeros(b, 1, h, w),
        "F_shadow": zeros(b, 1, h, w),
    }


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_reference_points(grid: BEVGrid, z_anchors: tuple[float, ...]) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.arange(grid.height, dtype=torch.float32),
        torch.arange(grid.width, dtype=torch.float32),
        indexing="ij",
    )
    x = grid.x_min + (cols.reshape(-1) + 0.5) * grid.resolution
    y = grid.y_min + (rows.reshape(-1) + 0.5) * grid.resolution
    z = torch.tensor(z_anchors, dtype=torch.float32)
    num_query = x.numel()
    points = torch.zeros(num_query, len(z_anchors), 4, dtype=torch.float32)
    points[..., 0] = x[:, None]
    points[..., 1] = y[:, None]
    points[..., 2] = z[None, :]
    points[..., 3] = 1.0
    return points
