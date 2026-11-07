"""Small geo-temporal light estimator; no downloads or pretrained weights needed."""
from __future__ import annotations

import torch
from torch import nn


def encoder(channels, width):
    return nn.Sequential(nn.Conv2d(channels, width, 5, stride=2, padding=2), nn.GroupNorm(4, width), nn.SiLU(),
        nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GroupNorm(4, width), nn.SiLU(),
        nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())


class GeoTemporalLightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = encoder(9, 64)
        self.receiver = encoder(7, 32)
        self.context = nn.Sequential(nn.Linear(64+32+9, 128), nn.SiLU(), nn.Dropout(.1))
        self.direction = nn.Sequential(nn.Linear(128+12, 64), nn.SiLU(), nn.Linear(64, 1))
        self.confidence = nn.Linear(128, 1)

    def forward(self, images, receiver, metadata, directions):
        context = self.context(torch.cat([self.image(images), self.receiver(receiver), metadata], dim=-1))
        shared = context[:, None].expand(-1, directions.shape[1], -1)
        logits = self.direction(torch.cat([shared, directions], dim=-1)).squeeze(-1)
        return {"logits": logits, "confidence_logits": self.confidence(context).squeeze(-1)}
