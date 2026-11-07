from copy import deepcopy

import numpy as np
import pytest

from shadow_occ.annotation.receiver_consistency import (
    compare_masks, empty_receiver_reason, identity_comparison, index_observations,
    sample_support, summarize_comparisons,
)
from tests.test_bev_flow import example


def test_receiver_iou_and_symmetric_difference_are_complementary():
    a, b = np.zeros((5, 5), bool), np.zeros((5, 5), bool)
    a[1:3, 1:3], b[1:3, 2:4] = True, True
    result = compare_masks(a, b)
    assert result["intersection"] == 2 and result["union"] == 6 and result["xor"] == 4
    assert result["iou"]+result["mismatch_ratio"] == pytest.approx(1)
    assert result["first_only"] == result["second_only"] == 2


def test_two_empty_masks_do_not_claim_perfect_observation_iou():
    result = compare_masks(np.zeros((3, 3)), np.zeros((3, 3)))
    assert result["both_empty"] and result["iou"] is None and result["mismatch_ratio"] is None
    assert summarize_comparisons([result])["union_weighted_mismatch_ratio"] is None


def test_receiver_mask_comparison_rejects_different_pixel_grids():
    with pytest.raises(ValueError, match="pixel grids"):
        compare_masks(np.zeros((3, 3)), np.zeros((4, 3)))


def test_same_camera_token_can_have_pair_local_ground_models():
    pairs = deepcopy(example()[0])
    for pair in pairs:
        for frame in pair["frames"]:
            frame.update(image="image.png", ground_support="ground.png", ground_sha256="same")
    pairs[1]["frames"][0] = deepcopy(pairs[1]["frames"][0])
    pairs[1]["frames"][0]["ground_sha256"] = "different"
    pairs[1]["plane_world"] = [0, 0, 1, -.1]
    items = index_observations(pairs)["scene1", "CAM_FRONT", "cam1"]
    identity = identity_comparison(*items)
    assert identity["same_camera_token"] and identity["same_image_hash"] and identity["same_lidar_sample"]
    assert not identity["same_ground_hash"] and not identity["same_ground_model_id"]
    assert identity["projection_max_abs_difference"] == 0
    assert identity["plane_height_difference_at_first_anchor_m"] == pytest.approx(.1)


def test_ground_variant_effect_is_not_mislabeled_as_grid_precision():
    world = np.array([[[2., 2., 1.]]])
    frame = {"projection_world_to_image": np.eye(3, 4), "image_size_wh": [8, 6]}
    ground = np.ones((6, 8), bool)
    original = sample_support(world, frame, ground)
    cached = sample_support(world.astype(np.float32), frame, ground)
    alternate = sample_support(world.astype(np.float32), frame, ~ground)
    assert compare_masks(original, cached)["xor"] == 0
    assert compare_masks(cached, alternate)["xor"] == 1


def test_soft_support_is_only_a_separate_counterfactual():
    world = np.array([[[2.03, 2., 1.]]])
    frame = {"projection_world_to_image": np.eye(3, 4), "image_size_wh": [8, 6]}
    ground = np.zeros((6, 8), bool)
    ground[:, 2] = True
    before = ground.copy()
    assert not sample_support(world, frame, ground).any()
    assert sample_support(world, frame, ground, .95).any()
    np.testing.assert_array_equal(ground, before)


@pytest.mark.parametrize("reason", ["reference_grid_no_ground_support", "central_view_missing_support",
    "central_views_disjoint", "post_visibility_receiver_exclusion"])
def test_empty_receiver_has_distinct_visibility_failure_types(reason):
    visible = np.ones((3, 4, 4), bool)
    if reason == "reference_grid_no_ground_support":
        visible[1] = False
    elif reason == "central_view_missing_support":
        visible[0] = False
    elif reason == "central_views_disjoint":
        visible[:] = False
        for index in range(3):
            visible[index, index] = True
    result = empty_receiver_reason({"visible_by_frame": visible, "receiver_valid": np.zeros((4, 4), bool)})
    assert result["reason"] == reason
