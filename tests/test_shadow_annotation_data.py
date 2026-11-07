import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import assert_disjoint, group_split, group_splits, make_target
from shadow_occ.annotation.metrics import BinaryMetrics, auc, region_summary


def test_weak_positive_mask_does_not_create_negative_gt(tmp_path):
    path = tmp_path/"mask.png"
    cv2.imwrite(str(path), np.array([[255, 0], [0, 0]], dtype=np.uint8))
    target = make_target({"shadow_mask": str(path), "label_source": "geometry_weak"}, (2, 2), tmp_path)
    assert target.tolist() == [[1, 255], [255, 255]]


def test_binary_gt_and_trimap_have_distinct_255_meanings(tmp_path):
    path = tmp_path/"mask.png"
    cv2.imwrite(str(path), np.array([[255, 0]], dtype=np.uint8))
    gt = make_target({"shadow_mask": str(path), "label_source": "public_shadow_gt"}, (1, 2), tmp_path)
    assert gt.tolist() == [[1, 0]]
    trimap = make_target({"shadow_mask": str(path), "label_source": "geometry_weak", "mask_encoding": "trimap"}, (1, 2), tmp_path)
    assert trimap.tolist() == [[255, 0]]


def test_official_val_cannot_be_randomly_assigned_to_training():
    assert group_split("scene-0001", "val", 0.15, 0) == "eval"
    with pytest.raises(ValueError, match="Scene leakage"):
        assert_disjoint([{"scene_name": "a", "split": "train"}], [{"scene_name": "a"}])


def test_small_scene_partition_keeps_class_aware_nonempty_dev():
    labels = {"scene-a": {"YES"}, "scene-b": {"YES", "NO"}, "scene-c": {"NO"},
              "scene-d": {"YES", "NO"}, "scene-e": {"YES"}}
    splits = group_splits(labels, "train", 0.15, 0, labels)
    dev = {scene for scene, split in splits.items() if split == "dev"}
    train = set(splits)-dev
    assert dev and train
    assert {"YES", "NO"} <= set().union(*(labels[scene] for scene in dev))
    assert {"YES", "NO"} <= set().union(*(labels[scene] for scene in train))
    assert set(group_splits(["scene-0001"], "val", 0.15, 0).values()) == {"eval"}


def test_pixel_metrics_exclude_unknown_and_auc_handles_ties():
    state = BinaryMetrics()
    state.update(np.array([1, 1, 1]), np.array([1, 0, 255]))
    assert state.summary()["valid_count"] == 2
    assert state.summary()["iou"] == 0.5
    assert auc([1, 0], [0.5, 0.5]) == 0.5


def test_weak_control_groups_never_become_shadow_ground_truth():
    rows = [{"id": "a", "scene_name": "scene-a", "pair_id": "p", "group": "candidate", "score": 0.9, "answer": None, "label_source": "unlabeled"}, {"id": "b", "scene_name": "scene-a", "pair_id": "p", "group": "weak_control", "score": 0.2, "answer": None, "label_source": "unlabeled"}]
    result = region_summary(rows, 0.5, bootstrap=10)
    assert result["by_label_source"] == {}
    assert result["weak_pair_separation"]["yes_difference"] == 1.0
