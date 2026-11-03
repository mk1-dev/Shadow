from __future__ import annotations

import numpy as np

from shadow_occ.config import load_config
from shadow_occ.metrics.occ3d_metrics import PseudoShadowMetrics


def test_baseline_disables_all_shadow_objectives() -> None:
    cfg = load_config("configs/baseline.yaml")
    assert cfg.model.use_shadow is False
    assert cfg.model.use_shadow_fusion is False
    assert cfg.train.shadow_loss_weight == 0.0
    assert cfg.train.temporal_loss_weight == 0.0
    assert cfg.train.render_loss_weight == 0.0
    assert cfg.train.uncertainty_loss_weight == 0.0


def test_pseudo_shadow_metrics_handles_channel_dimension() -> None:
    shape = (1, 4, 4)
    labels = np.zeros(shape, dtype=np.float32)
    labels[:, 1:3, 1:3] = 1.0
    direction = np.zeros((2, 4, 4), dtype=np.float32)
    direction[0] = 1.0
    metrics = PseudoShadowMetrics()
    metrics.update(
        pred={
            "P_shadow": np.where(labels > 0, 5.0, -5.0),
            "P_root": np.where(labels > 0, 5.0, -5.0),
            "P_tip": np.where(labels > 0, 5.0, -5.0),
            "P_caster": np.where(labels > 0, 5.0, -5.0),
            "V_shadow": direction,
            "U_shadow": np.where(labels > 0, 5.0, -5.0),
        },
        target={
            "shadow_mask": labels,
            "shadow_root_heatmap": labels,
            "shadow_tip_heatmap": labels,
            "shadow_caster_heatmap": labels,
            "shadow_direction": direction,
            "shadow_reliability": labels,
        },
    )
    summary = metrics.summarize()
    assert summary["PseudoShadow_shadow_F1"] > 0.99
    assert summary["PseudoShadow_direction_cosine"] > 0.99
