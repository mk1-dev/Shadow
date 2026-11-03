import numpy as np
import pytest

from shadow_occ.annotation.temporal import ground_homographies, sampling_grid, temporal_observations, warp


def record(center, timestamp, instance="vehicle-a"):
    return {"scene_name": "scene-a", "camera": "CAM_FRONT", "instance_token": instance, "timestamp": timestamp, "ego_to_global": np.eye(4).tolist(), "ego_to_img": [[10, 0, 0, 16], [0, 10, 0, 12], [0, 0, 0, 1]], "caster_center_ego": center}


def test_caster_and_world_alignment_are_separate():
    matrices = ground_homographies(record([1, 0, 0], 500000), record([0, 0, 0], 0))
    np.testing.assert_allclose(matrices["world_previous_to_current"], np.eye(3), atol=1e-12)
    point = matrices["caster_previous_to_current"] @ np.array([16, 12, 1])
    np.testing.assert_allclose(point[:2]/point[2], [26, 12])
    assert matrices["caster_speed_mps"] == 2.0


def test_temporal_pair_cannot_change_instance_identity():
    with pytest.raises(ValueError, match="instance_token"):
        ground_homographies(record([0, 0, 0], 500000), record([0, 0, 0], 0, "other"))


def test_static_shadow_keeps_weight_when_light_difference_is_zero():
    image = np.full((32, 32, 3), 160, dtype=np.uint8)
    image[8:24, 8:24] = 60
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8:24, 8:24] = 1
    matrices = {"world_previous_to_current": np.eye(3), "caster_previous_to_current": np.eye(3), "caster_speed_mps": 0.0}
    evidence = temporal_observations(image, image, target, target, np.ones(target.shape, bool), np.ones(target.shape, bool), matrices)
    assert evidence["details"]["lighting_comparable"]
    assert evidence["details"]["shadow_retention"] == 1.0
    assert evidence["details"]["repeated_shadow_small_change_ratio"] == 1.0
    assert evidence["weight"][12:20, 12:20].min() > 0.99


def test_nonoverlapping_ground_disables_temporal_supervision():
    image = np.full((16, 16, 3), 100, dtype=np.uint8)
    target = np.zeros((16, 16), dtype=np.uint8)
    matrices = {"world_previous_to_current": np.eye(3), "caster_previous_to_current": np.eye(3), "caster_speed_mps": 0.0}
    observation = temporal_observations(image, image, target, target, np.ones(target.shape, bool), np.zeros(target.shape, bool), matrices)
    assert observation["weight"].sum() == 0
    assert not observation["details"]["lighting_comparable"]


def test_moving_shadow_reports_ground_darkening_recovery_and_motion_advantage():
    previous = np.full((32, 32, 3), 160, np.uint8)
    current = previous.copy()
    previous[8:16, 8:16] = 60
    current[8:16, 16:24] = 60
    old_target = np.zeros((32, 32), np.uint8)
    target = old_target.copy()
    old_target[8:16, 8:16] = 1
    target[8:16, 16:24] = 1
    matrices = {"world_previous_to_current": np.eye(3), "caster_previous_to_current": np.array([[1, 0, 8], [0, 1, 0], [0, 0, 1]]), "caster_speed_mps": 2.0}
    ground = np.ones(target.shape, bool)
    details = temporal_observations(current, previous, target, old_target, ground, ground, matrices)["details"]
    assert details["onset_pixels"] == details["recovery_pixels"] == 64
    assert details["onset_darkening_ratio"] == details["recovery_brightening_ratio"] == 1.0
    assert details["motion_alignment_advantage"] == 1.0


def test_identity_sampling_grid_uses_pixel_centers():
    grid, valid = sampling_grid(np.eye(3), (16, 24), (16, 24), (16, 24))
    np.testing.assert_allclose(grid[0, 0], [-1+1/24, -1+1/16], atol=1e-6)
    assert valid.all()


def test_warp_accepts_json_serialized_homography():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    warped = warp(image, np.eye(3).tolist(), image.shape)
    np.testing.assert_array_equal(warped, image)


def test_segmentation_dice_stays_finite_for_large_half_precision_masks():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from shadow_occ.annotation.segmentation import masked_loss

    logits = torch.zeros((1, 1, 384, 704), dtype=torch.float16)
    target = torch.ones((1, 384, 704), dtype=torch.long)
    assert bool(torch.isfinite(masked_loss(logits, target, 0.1)))
