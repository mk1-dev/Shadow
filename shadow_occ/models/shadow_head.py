from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use shadow_occ.models.") from exc


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ShadowGeometryHead(nn.Module):
    """BEV head that predicts shadow geometry and a height prior."""

    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.trunk = ConvBlock(in_channels, hidden_channels)
        self.confidence = nn.Conv2d(hidden_channels, 1, 1)
        self.root = nn.Conv2d(hidden_channels, 1, 1)
        self.tip = nn.Conv2d(hidden_channels, 1, 1)
        self.caster = nn.Conv2d(hidden_channels, 1, 1)
        self.direction = nn.Conv2d(hidden_channels, 2, 1)
        self.height_prior = nn.Conv2d(hidden_channels, 1, 1)
        self.reliability = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, bev_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.trunk(bev_feature)
        direction = F.normalize(self.direction(feat), dim=1, eps=1e-6)
        return {
            "P_shadow": self.confidence(feat),
            "P_root": self.root(feat),
            "P_tip": self.tip(feat),
            "P_caster": self.caster(feat),
            "V_shadow": direction,
            "V_direction": direction,
            "H_prior": self.height_prior(feat),
            "U_shadow": self.reliability(feat),
            "F_shadow": feat,
        }


class HeightPriorFusion(nn.Module):
    """Fuse a shadow feature into OCC BEV features through confidence-gated residuals."""

    def __init__(self, bev_channels: int, shadow_channels: int) -> None:
        super().__init__()
        self.shadow_to_bev = nn.Conv2d(shadow_channels, bev_channels, 1)
        self.alpha = nn.Sequential(
            nn.Conv2d(1, bev_channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        bev_feature: torch.Tensor,
        shadow_feature: torch.Tensor,
        shadow_reliability_logits: torch.Tensor | None,
        height_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shadow_reliability_logits is None:
            alpha = torch.ones(
                shadow_feature.shape[0],
                self.shadow_to_bev.out_channels,
                shadow_feature.shape[2],
                shadow_feature.shape[3],
                device=shadow_feature.device,
                dtype=shadow_feature.dtype,
            )
        else:
            reliability = torch.sigmoid(shadow_reliability_logits)
            alpha = self.alpha(reliability)
        if height_prior is not None:
            # Use the predicted height prior as a spatial signal before the
            # shared shadow projection. This keeps the fusion checkpoint
            # compatible while making H_prior causally affect OCC features.
            height_signal = torch.tanh(height_prior).expand(-1, shadow_feature.shape[1], -1, -1)
            shadow_feature = shadow_feature + height_signal
        return bev_feature + alpha * self.shadow_to_bev(shadow_feature)


def shadow_geometry_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    from shadow_occ.models.losses import shadow_geometry_loss as _loss

    return _loss(pred, target, mask=mask, weights=weights)
