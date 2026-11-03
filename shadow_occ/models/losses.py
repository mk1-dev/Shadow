from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use shadow_occ.models.") from exc


def shadow_geometry_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise all proposal outputs when pseudo labels are available."""
    weights = weights or {}
    valid = 1.0 if mask is None else mask.float()

    loss_shadow = F.binary_cross_entropy_with_logits(
        pred["P_shadow"], target["shadow_mask"].float()
    )
    loss_root = F.binary_cross_entropy_with_logits(
        pred["P_root"], target["shadow_root_heatmap"].float()
    )
    loss_tip = F.binary_cross_entropy_with_logits(
        pred["P_tip"], target["shadow_tip_heatmap"].float()
    )
    loss_caster = F.binary_cross_entropy_with_logits(
        pred["P_caster"], target["shadow_caster_heatmap"].float()
    )
    loss_dir = (
        F.smooth_l1_loss(pred["V_shadow"], target["shadow_direction"].float(), reduction="none")
        * valid
    ).mean()
    height_target = target.get("height_target", target.get("H_occ", target.get("height_prior")))
    if height_target is None:
        raise KeyError("Expected height_target, H_occ, or height_prior for H_prior supervision.")
    loss_height = (F.smooth_l1_loss(pred["H_prior"], height_target.float(), reduction="none") * valid).mean()
    loss_reliability = F.binary_cross_entropy_with_logits(
        pred["U_shadow"], target["shadow_reliability"].float()
    )

    total = (
        weights.get("shadow", 1.0) * loss_shadow
        + weights.get("root", 1.0) * loss_root
        + weights.get("tip", 1.0) * loss_tip
        + weights.get("caster", 1.0) * loss_caster
        + weights.get("direction", 0.5) * loss_dir
        + weights.get("height", 1.0) * loss_height
        + weights.get("reliability", 1.0) * loss_reliability
    )
    return {
        "loss_shadow_total": total,
        "loss_shadow_confidence": loss_shadow,
        "loss_shadow_root": loss_root,
        "loss_shadow_tip": loss_tip,
        "loss_shadow_caster": loss_caster,
        "loss_shadow_direction": loss_dir,
        "loss_shadow_height": loss_height,
        "loss_shadow_reliability": loss_reliability,
    }


def temporal_caster_consistency_loss(
    pred_t: dict[str, torch.Tensor],
    pred_prev_warped: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Caster base should stay stable after ego-motion BEV warping."""
    current = torch.sigmoid(pred_t["P_caster"])
    previous = torch.sigmoid(pred_prev_warped["P_caster"])
    loss = F.smooth_l1_loss(current, previous, reduction="none")
    if mask is not None:
        loss = loss * mask.float()
    return loss.mean()


def shadow_motion_smooth_loss(
    pred: dict[str, torch.Tensor],
    shadow_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage neighboring BEV cells to agree on shadow motion/direction."""
    vec = pred["V_shadow"]
    dx = vec[:, :, :, 1:] - vec[:, :, :, :-1]
    dy = vec[:, :, 1:, :] - vec[:, :, :-1, :]
    if shadow_weight is None:
        weight_x = torch.sigmoid(pred["P_shadow"][:, :, :, 1:]).detach()
        weight_y = torch.sigmoid(pred["P_shadow"][:, :, 1:, :]).detach()
    else:
        weight_x = shadow_weight[:, :, :, 1:].float()
        weight_y = shadow_weight[:, :, 1:, :].float()
    return (dx.abs() * weight_x).mean() + (dy.abs() * weight_y).mean()


def caster_shadow_geometry_loss(pred: dict[str, torch.Tensor]) -> torch.Tensor:
    """Weak geometric prior: caster/root/tip ordering should agree with V_shadow."""
    caster_xy = _spatial_expectation(pred["P_caster"])
    tip_xy = _spatial_expectation(pred["P_tip"])
    expected_dir = F.normalize(tip_xy - caster_xy, dim=1, eps=1e-6)
    shadow_prob = torch.sigmoid(pred["P_shadow"]).detach()
    pred_dir = (pred["V_shadow"] * shadow_prob).flatten(2).sum(dim=2)
    pred_dir = F.normalize(pred_dir, dim=1, eps=1e-6)
    cosine = (pred_dir * expected_dir).sum(dim=1)
    return (1.0 - cosine).mean()


def render_shadow_loss(
    pred: dict[str, torch.Tensor],
    rendered_shadow_prior: torch.Tensor,
) -> torch.Tensor:
    """OCC -> shadow supervision, using rendered BEV shadow priors."""
    return F.binary_cross_entropy_with_logits(pred["P_shadow"], rendered_shadow_prior.float())


def uncertainty_gating_loss(
    pred: dict[str, torch.Tensor],
    target_reliability: torch.Tensor,
    shadow_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train U_shadow to open the shadow branch only for useful candidates."""
    reliability = target_reliability.float()
    loss = F.binary_cross_entropy_with_logits(pred["U_shadow"], reliability)
    if shadow_target is None:
        return loss
    gate = torch.sigmoid(pred["U_shadow"])
    shadow_prob = torch.sigmoid(pred["P_shadow"])
    leakage = ((1.0 - gate) * shadow_prob * (1.0 - shadow_target.float())).mean()
    return loss + 0.1 * leakage


def weak_shadow_total_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    pred_prev_warped: dict[str, torch.Tensor] | None = None,
    occ_loss: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Loss bundle matching the B-design equation."""
    weights = weights or {}
    losses = shadow_geometry_loss(pred, target, weights=weights)
    total = losses["loss_shadow_total"]

    if occ_loss is not None:
        losses["L_occ"] = occ_loss
        total = total + weights.get("occ", 1.0) * occ_loss

    if pred_prev_warped is not None:
        l_temporal = temporal_caster_consistency_loss(pred, pred_prev_warped)
        losses["L_temporal_consistency"] = l_temporal
        total = total + weights.get("temporal", 1.0) * l_temporal

    l_motion = shadow_motion_smooth_loss(pred)
    losses["L_shadow_motion_smooth"] = l_motion
    total = total + weights.get("motion_smooth", 0.2) * l_motion

    l_geometry = caster_shadow_geometry_loss(pred)
    losses["L_caster_shadow_geometry"] = l_geometry
    total = total + weights.get("caster_geometry", 0.2) * l_geometry

    if "rendered_shadow_prior" in target:
        l_render = render_shadow_loss(pred, target["rendered_shadow_prior"])
        losses["L_render_shadow"] = l_render
        total = total + weights.get("render", 0.5) * l_render

    l_gate = uncertainty_gating_loss(
        pred,
        target["shadow_reliability"],
        target.get("shadow_mask"),
    )
    losses["L_uncertainty_gating"] = l_gate
    total = total + weights.get("uncertainty", 1.0) * l_gate

    losses["loss_total"] = total
    return losses


def _spatial_expectation(logits: torch.Tensor) -> torch.Tensor:
    b, _, h, w = logits.shape
    prob = torch.softmax(logits.flatten(2), dim=-1).view(b, 1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (prob[:, 0] * xx).flatten(1).sum(dim=1)
    y = (prob[:, 0] * yy).flatten(1).sum(dim=1)
    return torch.stack([x, y], dim=1)
