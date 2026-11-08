import numpy as np

from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_projection, ground_grid
from shadow_occ.annotation.physics_motion import project_world
from shadow_occ.annotation.video_correspondence import cross_space_proposals, lift_flow_to_bev, match_spatial_bev
from shadow_occ.annotation.shadow_video import video_proposals
from tests.test_bev_flow import example
from tests.test_shadow_video import video_case


def test_flow_is_lifted_through_next_camera_not_scaled_as_image_pixels():
    pairs = example()[0]
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0]]
    frames[1]["projection_world_to_image"][0][3] += 40  # Camera-only displacement = 4 image pixels.
    grid = BEVFlowConfig(extent_m=4).grid()
    basis = np.asarray(pairs[0]["plane_basis_world"])
    world = ground_grid(basis, np.eye(4), grid)
    uv = project_world(world.reshape(-1, 3), frames[0]["projection_world_to_image"])[0].reshape(*grid.shape, 2)
    receiver = np.ones((2, *grid.shape), bool)
    flow = np.zeros((1, *grid.shape, 2), np.float32)
    flow[..., 0] = 4
    args = (world, frames, [uv], flow, np.ones(flow.shape[:-1], bool), receiver, basis, bev_projection(np.eye(4), grid), .5)
    maps = lift_flow_to_bev(*args)
    assert maps["flow_bev_valid"].all()
    np.testing.assert_allclose(maps["flow_bev_displacement_m"], 0, atol=1e-6)
    flow[..., 0] = 6  # Two residual pixels -> half a meter in this calibration.
    maps = lift_flow_to_bev(*args)
    valid = maps["flow_bev_valid"]
    assert valid.any() and not valid.all()
    np.testing.assert_allclose(maps["flow_bev_displacement_m"][valid], np.tile([.5, 0], (valid.sum(), 1)), atol=1e-6)


def test_spatial_feature_correspondence_returns_displacement_in_meters():
    rng = np.random.default_rng(4)
    first = rng.normal(size=(9, 9, 32)).astype(np.float32)
    features = np.stack([first, np.roll(first, 1, axis=1)])
    valid = np.ones((2, 9, 9), bool)
    valid[1, :, 0] = False
    result = match_spatial_bev(features, valid, .5)
    assert result["spatial_bev_match_valid"][0, 2:7, 2:7].all()
    np.testing.assert_allclose(result["spatial_bev_displacement_m"][0, 2:7, 2:7],
                               np.broadcast_to([.5, 0], (5, 5, 2)))


def test_constant_or_missing_bev_is_not_a_confident_zero_motion_match():
    features = np.ones((2, 9, 9, 4), np.float32)
    for valid in (np.ones((2, 9, 9), bool), np.zeros((2, 9, 9), bool)):
        result = match_spatial_bev(features, valid, .5)
        assert not result["spatial_bev_match_valid"].any()
        assert np.isnan(result["spatial_bev_displacement_m"]).all()


def test_metric_difference_can_propose_below_appearance_seed_threshold():
    inputs = video_case(3)
    inputs["lab"][1:, 5:10, 5:10, 0] = 68
    inputs["delta_L"] = np.diff(inputs["lab"][..., 0], axis=0)
    maps = video_proposals(**inputs)
    assert maps["proposal_score_map"][1, 6, 6] < .35
    rng = np.random.default_rng(2)
    features = np.broadcast_to(rng.normal(size=(16, 16, 32)).astype(np.float32), (3, 16, 16, 32)).copy()
    valid = np.ones((3, 16, 16), bool)
    flow = np.zeros((2, 16, 16, 2), np.float32)
    flow[..., 0] = .75
    result = cross_space_proposals(maps, features, valid, flow, np.ones(valid[:-1].shape, bool), .5)
    assert result["cross_space_discovery_score"][1, 6, 6] >= .35
    assert result["flow_bev_discrepancy_valid"][0, 6, 6]
    assert np.isnan(result["cross_space_entity_factor"][0, 6, 6])  # Missing entity prediction remains missing.
    assert np.isnan(result["cross_space_complete_score"][1, 6, 6])
    assert result["cross_space_partial_score"][1, 6, 6] >= .35
    result = cross_space_proposals(maps, features, valid, flow, np.zeros(valid[:-1].shape, bool), .5)
    assert result["cross_space_discovery_score"][1, 6, 6] < .35


def test_temporal_only_missing_spatial_features_cannot_claim_cross_space_discovery():
    inputs = video_case(3)
    maps = video_proposals(**inputs)
    result = cross_space_proposals(maps, inputs["features"], np.zeros((3, 16, 16), bool),
        np.zeros((2, 16, 16, 2)), np.zeros((2, 16, 16), bool), .5)
    assert not result["cross_space_discovery_score"].any()
    assert not result["spatial_bev_match_valid"].any()
