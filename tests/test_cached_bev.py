from types import SimpleNamespace

import numpy as np
import pytest

from shadow_occ.annotation.bev_flow import BEVFlowConfig, ground_grid
from shadow_occ.annotation.cached_bev import attach_cached_bev, sample_cached_bev, select_bev_flow_receivers, spatial_bev, write_cached_bev
from shadow_occ.annotation.physics_motion import plane_basis


def test_uniad_sequence_layout_preserves_y_rows_x_columns():
    value = np.arange(3*4*2).reshape(12, 1, 2)
    result = spatial_bev(value, 3, 4)
    assert result.shape == (2, 3, 4)
    np.testing.assert_array_equal(result[:, 2, 1], value[9, 0])
    np.testing.assert_array_equal(result, spatial_bev(value.transpose(1, 0, 2), 3, 4, "BHWC"))
    with pytest.raises(ValueError, match="spatial"):
        spatial_bev(np.zeros((900, 256)), 200, 200)


def test_sampling_uses_lidar_calibration_not_ego_grid_resize(tmp_path):
    grid = BEVFlowConfig(extent_m=2, resolution_m=1).grid()
    world = ground_grid(plane_basis([0, 0, 1, 0]), np.eye(4), grid)
    feature = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4)
    sensor = np.eye(4)
    sensor[0, 3] = 1
    record = SimpleNamespace(scene_name="scene1", lidar_ego_to_global=np.eye(4), lidar_sensor_to_ego=sensor)
    write_cached_bev(tmp_path, "s0", feature, record, 100, [-2, -2, -1, 2, 2, 1], {})
    actual, valid, _ = sample_cached_bev(tmp_path, "s0", world)
    np.testing.assert_array_equal(actual[:, 1:, 0], feature[0, :, :-1])
    assert not valid[:, 0].any() and valid[:, 1:].all()


def test_native_geometry_preserves_non_square_grid_rotation_and_metric_resolution(tmp_path):
    ego = np.eye(4)
    ego[:2, :2] = [[0, -1], [1, 0]]
    ego[:3, 3] = [5, 9, 1]
    sensor = np.eye(4)
    sensor[:3, 3] = [1, -.5, 0]
    record = SimpleNamespace(scene_name="scene1", lidar_ego_to_global=ego, lidar_sensor_to_ego=sensor)
    feature = np.arange(1, 16, dtype=np.float32).reshape(1, 3, 5)
    write_cached_bev(tmp_path, "s0", feature, record, 12345, [-5, -1.5, -2, 5, 1.5, 2], {})
    yy, xx = np.indices((3, 5))
    local = np.stack([-5+(xx+.5)*2, -1.5+yy+.5, np.zeros_like(xx)], axis=-1)
    world = (np.c_[local.reshape(-1, 3), np.ones(15)] @ (ego @ sensor).T)[:, :3].reshape(3, 5, 3)
    actual, valid, source, geometry = sample_cached_bev(tmp_path, "s0", world, return_geometry=True)
    np.testing.assert_allclose(actual[..., 0], feature[0])
    np.testing.assert_allclose(geometry["native_uv"], np.stack([xx, yy], axis=-1))
    np.testing.assert_allclose(geometry["resolution_xy_m"], [2, 1])
    assert valid.all() and geometry["native_valid"].all()
    assert source["timestamp"] == 12345


def test_same_world_ground_descriptor_survives_ego_translation_and_rotation(tmp_path):
    poses = [np.eye(4), np.eye(4)]
    poses[1][:2, :2] = [[0, -1], [1, 0]]
    poses[1][0, 3] = 1
    yy, xx = np.indices((8, 8))
    local = np.stack([xx-3.5, yy-3.5, np.zeros_like(xx)], axis=-1)
    grid = BEVFlowConfig(extent_m=1.5, resolution_m=1).grid()
    world = ground_grid(plane_basis([0, 0, 1, 0]), np.eye(4), grid)
    sampled = []
    anchors = []
    for slot, pose in enumerate(poses):
        record = SimpleNamespace(scene_name="scene", lidar_ego_to_global=pose, lidar_sensor_to_ego=np.eye(4))
        centres = (np.c_[local.reshape(-1, 3), np.ones(64)] @ pose.T)[:, :3].reshape(8, 8, 3)
        feature = np.stack([centres[..., 0], centres[..., 1], np.ones((8, 8))])
        write_cached_bev(tmp_path, f"s{slot}", feature, record, slot*500_000, [-4, -4, -1, 4, 4, 1], {})
        values, valid, _, geometry = sample_cached_bev(tmp_path, f"s{slot}", world, return_geometry=True)
        assert valid.all()
        sampled.append(values)
        anchors.append(geometry["native_uv"])
    np.testing.assert_allclose(sampled[0], sampled[1], atol=1e-6)
    assert not np.allclose(anchors[0], anchors[1])


def test_bilinear_sampling_does_not_invent_support_across_invalid_native_cells(tmp_path):
    record = SimpleNamespace(scene_name="scene", lidar_ego_to_global=np.eye(4), lidar_sensor_to_ego=np.eye(4))
    valid = np.ones((4, 4), bool)
    valid[1, 1] = False
    write_cached_bev(tmp_path, "s0", np.ones((2, 4, 4)), record, 0, [-2, -2, -1, 2, 2, 1], {}, valid_mask=valid)
    world = np.array([[[-.5, -.5, 0], [0, -.5, 0], [.5, -.5, 0]]])
    values, supported, _, geometry = sample_cached_bev(tmp_path, "s0", world, return_geometry=True)
    np.testing.assert_array_equal(supported, [[False, False, True]])
    np.testing.assert_allclose(geometry["interpolation_support"], [[0, .5, 1]])
    assert geometry["in_range"].all()
    assert not values[0, :2].any() and np.isfinite(values).all()


def test_three_frame_features_are_aligned_at_world_receivers(tmp_path):
    grid = BEVFlowConfig(extent_m=2, resolution_m=1).grid()
    world = ground_grid(plane_basis([0, 0, 1, 0]), np.eye(4), grid)
    record = SimpleNamespace(scene_name="scene1", lidar_ego_to_global=np.eye(4), lidar_sensor_to_ego=np.eye(4))
    for i in range(3):
        write_cached_bev(tmp_path, f"s{i}", np.ones((2, 4, 4)), record, i, [-2, -2, -1, 2, 2, 1], {})
    arrays, meta = attach_cached_bev({"receiver_world_xyz": world, "visible_by_frame": np.ones((3, 4, 4), bool)},
                                  {"sample_tokens": ["s0", "s1", "s2"]}, tmp_path)
    assert arrays["learned_bev_receiver_features"].shape == (3, 4, 4, 2)
    np.testing.assert_allclose(arrays["learned_bev_temporal_cosine"], 1, atol=1e-6)
    assert meta["learned_bev"]["layout"] == "T,H,W,C"


def test_learned_bev_changes_actual_candidate_selection():
    full = np.ones((2, 2), bool)
    arrays = {"learned_bev_temporal_cosine": np.ones((2, 2, 2), np.float32),
        "learned_bev_valid": np.ones((3, 2, 2), bool), "shadow_candidate": full.copy(),
        "shadow_flow_supported": full.copy(), "shadow_boundary_onset": full.copy(),
        "shadow_boundary_recovery": full.copy(), "shadow_same_patch_cycle": full.copy()}
    arrays["learned_bev_temporal_cosine"][0, 0, 0] = -0.2
    arrays["shadow_flow_supported"][1, 1] = False
    arrays, metadata = select_bev_flow_receivers(arrays, {"counts": {}}, 0.3, discovery_mode="hard-intersection")
    assert arrays["shadow_candidate"].sum() == 2
    assert not arrays["shadow_candidate"][0, 0]
    assert not arrays["shadow_boundary_onset"][0, 0]
    assert metadata["learned_bev_selection"]["feature_rejected_cells"] == 1
    assert metadata["learned_bev_selection"]["flow_rejected_cells"] == 1
