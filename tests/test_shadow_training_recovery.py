import json

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import checked_write_image, digest
from shadow_occ.annotation.diagnostics import decision_loss_summary
from shadow_occ.annotation.metrics import pixel_score_diagnostics
from shadow_occ.annotation.readiness import (
    binary_risk_weights, require_prediction_for_proposals, require_segmenter_checkpoint,
    segmentation_readiness,
)


def test_all_background_cannot_be_best_even_with_high_accuracy():
    metrics = {"temporal_weak": {"tp": 0, "fp": 0, "fn": 15744, "tn": 378000}}
    ready = segmentation_readiness(metrics)
    assert not ready["noncollapsed"] and not ready["approved_for_relabeling"]
    assert "all_background" in ready["reasons"]
    with pytest.raises(ValueError, match="Collapsed"):
        require_segmenter_checkpoint({"dev_metrics_by_source": metrics})
    assert not require_segmenter_checkpoint({"dev_metrics_by_source": metrics}, True)["noncollapsed"]


def test_weak_hits_cannot_hide_collapse_on_available_gt_sources():
    ready = segmentation_readiness({"manual_shadow_gt": {"tp": 0, "tn": 100, "fn": 10, "fp": 0},
                                    "geometry_weak": {"tp": 100, "tn": 100, "fn": 0, "fp": 0}})
    assert not ready["noncollapsed"] and ready["source_scope"] == ["manual_shadow_gt"]


@pytest.mark.parametrize("row", [{}, {"diagnostics_only": True, "segmenter_readiness": {"noncollapsed": True}},
                                 {"segmenter_readiness": {"noncollapsed": False}}])
def test_unvalidated_predictions_cannot_enter_proposals(row):
    with pytest.raises(ValueError, match="cannot generate"):
        require_prediction_for_proposals(row)


def test_noncollapse_is_not_automatic_relabeling_approval():
    ready = segmentation_readiness({"geometry_weak": {"tp": 4, "tn": 7, "fp": 2, "fn": 3}})
    assert ready["noncollapsed"] and not ready["approved_for_relabeling"]


def test_score_histogram_distinguishes_subthreshold_separation_from_all_low_scores():
    histogram = np.zeros((2, 256), np.int64)
    histogram[0, 10], histogram[1, 70] = 100, 10
    result = pixel_score_diagnostics(histogram)
    low = next(item for item in result["threshold_sweep"] if item["threshold"] == 0.1015625)
    half = next(item for item in result["threshold_sweep"] if item["threshold"] == 0.5)
    assert half["tp"] == 0 and low["tp"] == 10 and low["fp"] == 0
    assert not result["approved_for_relabeling"]


def test_macro_decision_loss_cannot_be_dominated_by_many_easy_no_rows():
    records = [(str(i), "NO", 0.001) for i in range(100)] + [("positive", "YES", 1.0)]
    summary = decision_loss_summary(records + [records[0]])
    assert summary["macro"] == pytest.approx(0.5005)
    assert summary["counts"] == {"NO": 100, "YES": 1}
    with pytest.raises(ValueError, match="Conflicting"):
        decision_loss_summary([("same", "NO", 0.1), ("same", "YES", 0.1)])


def test_pixel_balance_uses_effective_known_mass_and_caps_weights():
    assert binary_risk_weights([99, 1]) == pytest.approx([100/198, 50])
    assert binary_risk_weights([9999, 1], maximum=25)[1] == 25
    with pytest.raises(ValueError):
        binary_risk_weights([100, 0])


def test_balanced_masked_bce_equalizes_rare_foreground_gradient():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import masked_loss
    logits = torch.zeros((1, 1, 10, 10), requires_grad=True)
    target = torch.zeros((1, 10, 10), dtype=torch.long)
    target[0, 0, 0] = 1
    loss = masked_loss(logits, target, 0, class_weights=binary_risk_weights([99, 1]))
    loss.backward()
    gradient = logits.grad[0, 0]
    assert -float(gradient[0, 0]) == pytest.approx(float(gradient.sum()-gradient[0, 0]))


def test_ignore_pixels_never_receive_segmentation_gradient():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import masked_loss
    logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
    target = torch.tensor([[[1, 0], [255, 255]]])
    masked_loss(logits, target, 0.5, class_weights=[1, 5]).backward()
    assert not logits.grad[0, 0, 1].any()


def test_receiver_order_loss_has_signal_at_collapsed_equal_predictions():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import receiver_order_loss
    current, before, after = [torch.zeros((1, 1, 2, 2), requires_grad=True) for _ in range(3)]
    mask = torch.ones((1, 2, 2))
    loss = receiver_order_loss(current, before, after, mask, mask)
    loss.backward()
    assert (current.grad < 0).all() and (before.grad > 0).all() and (after.grad > 0).all()
    assert float(receiver_order_loss(current, before, after, mask*0, mask*0)) == 0


def test_receiver_pairs_are_restricted_to_positive_core_and_hash_checked(tmp_path):
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import ShadowDataset
    image, target, events = tmp_path/"image.png", tmp_path/"target.png", tmp_path/"events.npz"
    checked_write_image(image, np.full((8, 8, 3), 100, np.uint8))
    trimap = np.full((8, 8), 255, np.uint8)
    trimap[2:4, 2:4] = 1
    trimap[0, 0] = 0
    checked_write_image(target, trimap)
    np.savez_compressed(events, onset=np.ones((8, 8)), recovery=np.ones((8, 8)))
    row = {"id": "paired", "image": str(image), "target": str(target), "label_source": "temporal_weak",
           "physics_transition": {"path": str(events), "sha256": digest(events),
               "semantics": "relative_brightness_evidence_not_absolute_nonshadow_labels",
               "views": [{"slot": slot, "image": str(image), "sha256": digest(image)} for slot in ("previous", "next")]}}
    sample = ShadowDataset([row], (8, 8), receiver_pairs=True)[0]
    assert float(sample["onset_weight"].sum()) == 4
    assert not sample["onset_weight"][0, 0]
    row["physics_transition"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="transition evidence"):
        ShadowDataset([row], (8, 8), receiver_pairs=True)[0]


def test_component_source_weight_covers_all_components_class_weight_only_decision():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.objectives import component_loss
    logits = torch.zeros((1, 4, 3), requires_grad=True)
    labels = torch.tensor([[-100, 0, 1, 2]])
    groups = torch.tensor([[0, 1, 2, 3]])
    weights = {"decision": 1.0, "association": 0.5, "format": 0.1}
    baseline, _ = component_loss(logits, labels, groups, 1, weights)
    weak, _ = component_loss(logits, labels, groups, 1, weights, example_weight=0.25)
    balanced, _ = component_loss(logits, labels, groups, 5, weights, example_weight=0.25)
    assert float(weak) == pytest.approx(float(baseline)*0.25)
    assert float(balanced) == pytest.approx(float(baseline)*0.25*(5+0.5+0.1)/1.6)


def test_vlm_dataset_does_not_apply_class_weight_twice(monkeypatch):
    pytest.importorskip("torch")
    from shadow_occ.annotation import vlm
    from shadow_occ.annotation.joint import annotation
    monkeypatch.setattr(vlm, "encode_example", lambda *args, **kwargs: {})
    row = {"id": "yes", "answer": json.dumps(annotation("YES")), "label_source": "temporal_weak"}
    encoded = vlm.SFTDataset([row], None, 1024, 0.25, {"YES": 25})[0]
    assert encoded["example_weight"] == 0.25 and encoded["decision_class_weight"] == 25


def test_vlm_class_balance_accounts_for_source_weight_before_frequency():
    pytest.importorskip("torch")
    from shadow_occ.annotation.vlm import shadow_decision_weights
    from shadow_occ.annotation.joint import annotation
    from shadow_occ.annotation.diagnostics import balance_audit
    rows = [{"id": "positive", "scene_name": "a", "answer": json.dumps(annotation("YES")),
             "label_source": "manual_shadow_gt"}]
    rows += [{"id": str(i), "scene_name": "b", "answer": json.dumps(annotation("NO", "none")),
              "label_source": "geometry_weak"} for i in range(4)]
    weights = shadow_decision_weights(rows, maximum=50, weak_weight=0.25)
    assert weights == pytest.approx({"YES": 1, "NO": 1})
    report = balance_audit(rows, weights, 0.25)
    assert report["classes"]["YES"]["effective_decision_mass_fraction"] == 0.5
