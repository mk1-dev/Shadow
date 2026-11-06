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
    feature = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    sensor = np.eye(4)
    sensor[0, 3] = 1
    record = SimpleNamespace(scene_name="scene1", lidar_ego_to_global=np.eye(4), lidar_sensor_to_ego=sensor)
    write_cached_bev(tmp_path, "s0", feature, record, 100, [-2, -2, -1, 2, 2, 1], {})
    actual, valid, _ = sample_cached_bev(tmp_path, "s0", world)
    np.testing.assert_array_equal(actual[:, 1:, 0], feature[0, :, :-1])
    assert not valid[:, 0].any() and valid[:, 1:].all()


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
    arrays, metadata = select_bev_flow_receivers(arrays, {"counts": {}}, 0.3)
    assert arrays["shadow_candidate"].sum() == 2
    assert not arrays["shadow_candidate"][0, 0]
    assert not arrays["shadow_boundary_onset"][0, 0]
    assert metadata["learned_bev_selection"]["feature_rejected_cells"] == 1
    assert metadata["learned_bev_selection"]["flow_rejected_cells"] == 1
