import numpy as np
import pytest

from shadow_occ.annotation.metrics import BinaryMetrics, pixel_score_diagnostics
from shadow_occ.annotation.readiness import segmentation_selection_score
from shadow_occ.annotation.scene_sampling import scene_probabilities, supervision_epoch_indices
from shadow_occ.annotation.receiver_negatives import independent_lit_receiver_mask


def test_best_checkpoint_uses_both_classes_not_foreground_majority():
    almost_all_positive, discriminative = BinaryMetrics(), BinaryMetrics()
    almost_all_positive.confusion = np.array([1, 99, 1, 999])
    discriminative.confusion = np.array([80, 20, 100, 900])
    a, b = almost_all_positive.summary(), discriminative.summary()
    assert a["iou"] > b["iou"]
    assert b["balanced_accuracy"] > a["balanced_accuracy"]
    assert segmentation_selection_score({"temporal_weak": b}) > segmentation_selection_score({"temporal_weak": a})
    assert b["mean_iou"] == pytest.approx((b["iou"]+b["negative_iou"])/2)
    assert a["all_foreground_iou"] == pytest.approx(1000/1100)


def test_constant_pixel_scores_have_auc_half_even_with_high_ap():
    histogram = np.zeros((2, 256), np.int64)
    histogram[:, 250] = [5, 95]
    summary = pixel_score_diagnostics(histogram)
    assert summary["approximate_average_precision"] == pytest.approx(.95)
    assert summary["approximate_auroc"] == .5


def test_supervision_sampler_gives_each_local_pair_both_support_pools():
    rows = [{"scene_name": "positive", "sampling_class_pixels": [0, 100]} for _ in range(99)]
    rows += [{"scene_name": "negative", "sampling_class_pixels": [10, 0]}]
    probabilities = scene_probabilities(rows, "supervision")
    assert probabilities[-1] == pytest.approx(.5)
    for rank in range(3):
        indices = supervision_epoch_indices(rows, 4, 2, rank, 3)
        assert len(indices) == 34
        for start in range(0, len(indices), 2):
            classes = [np.array(rows[index]["sampling_class_pixels"]) > 0 for index in indices[start:start+2]]
            assert np.logical_or.reduce(classes).all()
        assert indices == supervision_epoch_indices(rows, 4, 2, rank, 3)


def lit_arrays():
    light = np.full((3, 12, 12), 20.0)
    light[:, :, 7:] = 70.0
    return {"receiver_valid": np.ones((12, 12), bool), "appearance_L_raw": light,
        "appearance_local_ratio": np.ones_like(light), "exclusion": np.zeros((12, 12), bool),
        "bev_world_to_grid": np.eye(3, 4)}


def test_independent_negatives_do_not_require_shadow_core_and_have_a_budget():
    arrays = lit_arrays()
    mask, details = independent_lit_receiver_mask(arrays, [0, 0, 1, 0], [[0, 0, -1]], [])
    assert mask.any() and mask.sum() <= 15
    assert not details["requires_positive_core"]
    assert not mask[:, :7].any()
    assert not details["unlabelled_is_background"]


def test_independent_negatives_exclude_proposals_occlusion_and_unstable_appearance():
    arrays = lit_arrays()
    arrays["exclusion"][:, 7:] = True
    mask, _ = independent_lit_receiver_mask(arrays, [0, 0, 1, 0], [[0, 0, -1]], [])
    assert not mask.any()
    arrays = lit_arrays()
    arrays["appearance_L_raw"][0, :, 7:] = 35
    mask, _ = independent_lit_receiver_mask(arrays, [0, 0, 1, 0], [[0, 0, -1]], [])
    assert not mask.any()


def test_independent_negatives_need_observed_lit_contrast_and_geometry():
    arrays = lit_arrays()
    arrays["appearance_L_raw"][:] = 70
    mask, details = independent_lit_receiver_mask(arrays, [0, 0, 1, 0], [[0, 0, -1]], [])
    assert not mask.any() and details["reason"] == "insufficient_lit_context_contrast"
    arrays = lit_arrays()
    arrays["receiver_valid"][:] = False
    mask, _ = independent_lit_receiver_mask(arrays, [0, 0, 1, 0], [[0, 0, -1]], [])
    assert not mask.any()


def test_class_balanced_loss_cannot_reward_a_constant_foreground_prediction():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import masked_loss
    target = torch.ones(1, 10, 10, dtype=torch.long)
    target[0, 0, 0] = 0
    logits = torch.zeros(1, 1, 10, 10, requires_grad=True)
    total, parts = masked_loss(logits, target, .5, negative_weights=torch.tensor([.5]),
        objective="class-balanced", return_components=True)
    total.backward()
    assert logits.grad.sum().abs() < 1e-6
    assert float(total.detach()) == pytest.approx(float((parts["bce"]+.5*parts["dice"]).detach()))
    foreground = masked_loss(torch.full_like(logits, 5), target, .5, objective="class-balanced")
    assert foreground > total.detach()


def test_class_balanced_loss_respects_ignore_and_single_class_has_no_dice():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import masked_loss
    target = torch.tensor([[[1, 255, 255]]])
    logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
    total, terms = masked_loss(logits, target, .5, objective="class-balanced", return_components=True)
    total.backward()
    assert terms["dice"] == 0 and not logits.grad[..., 1:].any()
    assert float(logits.grad[..., 0]) < 0


def test_balanced_class_reliability_is_relative_not_a_hidden_class_prior():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import masked_loss
    target = torch.tensor([[[1, 0, 255]]])
    logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
    a = masked_loss(logits, target, .5, negative_weights=torch.tensor([.1]), objective="class-balanced")
    b = masked_loss(logits, target, .5, negative_weights=torch.tensor([1.]), objective="class-balanced")
    assert torch.allclose(a, b)


def _distributed_loss_worker(rank, rendezvous, output):
    import torch
    from datetime import timedelta
    from shadow_occ.annotation.segmentation import masked_loss
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank,
        world_size=2, timeout=timedelta(seconds=60))
    logits = torch.tensor([[[[.2, -.3]]]], requires_grad=True)
    target = torch.tensor([[[1, 1] if rank == 0 else [0, 255]]])
    loss = masked_loss(logits, target, .5, negative_weights=torch.tensor([.5]), objective="class-balanced")
    loss.backward()
    torch.save({"loss": loss.detach(), "gradient": logits.grad}, output+str(rank)+".pth")
    torch.distributed.destroy_process_group()


def test_distributed_class_loss_matches_one_global_batch_even_with_missing_local_class(tmp_path):
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import masked_loss
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    output = str(tmp_path/"rank")
    torch.multiprocessing.spawn(_distributed_loss_worker,
        args=((tmp_path/"rendezvous").as_uri(), output), nprocs=2, join=True)
    logits = torch.tensor([[[[.2, -.3]]], [[[.2, -.3]]]], requires_grad=True)
    target = torch.tensor([[[1, 1]], [[0, 255]]])
    loss = masked_loss(logits, target, .5, negative_weights=torch.tensor([.5, .5]), objective="class-balanced")
    loss.backward()
    for rank in range(2):
        actual = torch.load(output+str(rank)+".pth", weights_only=True)
        assert torch.allclose(actual["loss"], loss.detach(), atol=1e-6)
        assert torch.allclose(actual["gradient"]/2, logits.grad[rank:rank+1], atol=1e-6)
