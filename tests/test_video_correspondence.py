import numpy as np

from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_projection, ground_grid
from shadow_occ.annotation.physics_motion import project_world
from shadow_occ.annotation.video_correspondence import (CrossSpaceGrowthConfig, cross_space_proposals,
    correspondence_summary, grow_cross_space_regions, lift_flow_to_bev, match_spatial_bev)
from shadow_occ.annotation.shadow_video import VideoShadowConfig, refine_regions, video_proposals
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


def test_local_search_cannot_replace_a_missing_geometric_anchor_with_a_neighbour():
    rng = np.random.default_rng(4)
    first = rng.normal(size=(9, 9, 16)).astype(np.float32)
    features = np.stack([first, np.roll(first, 1, axis=1)])
    valid = np.ones((2, 9, 9), bool)
    valid[1, 4, 4] = False
    result = match_spatial_bev(features, valid, .5)
    assert result["spatial_bev_candidate_count"][0, 4, 4] > 0
    assert not result["spatial_bev_geometry_pair_valid"][0, 4, 4]
    assert result["spatial_bev_match_state"][0, 4, 4] == 0
    assert np.isnan(result["spatial_bev_local_offset_m"][0, 4, 4]).all()


def test_constant_or_missing_bev_is_not_a_confident_zero_motion_match():
    features = np.ones((2, 9, 9, 4), np.float32)
    for valid in (np.ones((2, 9, 9), bool), np.zeros((2, 9, 9), bool)):
        result = match_spatial_bev(features, valid, .5)
        assert not result["spatial_bev_match_valid"].any()
        assert np.isnan(result["spatial_bev_displacement_m"]).all()
        assert not result["spatial_bev_match_confidence"].any()
        assert not result["spatial_bev_match_usable"].any()
        assert (result["spatial_bev_match_state"] == (3 if valid.any() else 0)).all()
        assert result["spatial_bev_geometry_pair_valid"].all() == bool(valid.any())


def test_sparse_single_candidate_is_ambiguous_without_forcing_a_motion_vector():
    features = np.zeros((2, 5, 5, 2), np.float32)
    valid = np.zeros((2, 5, 5), bool)
    features[:, 2, 2] = [1, 0]
    valid[:, 2, 2] = True
    result = match_spatial_bev(features, valid, .5)
    at = (0, 2, 2)
    assert result["spatial_bev_match_state"][at] == 3
    assert not result["spatial_bev_match_usable"][at]
    assert not result["spatial_bev_match_valid"][at]
    assert result["spatial_bev_match_confidence"][at] == 0
    assert np.isnan(result["spatial_bev_local_offset_m"][at]).all()
    assert result["spatial_bev_geometry_pair_valid"][at]
    np.testing.assert_allclose(result["spatial_bev_anchor_cosine"][at], 1)
    assert np.isnan(result["spatial_bev_match_margin"][at])
    assert result["spatial_bev_candidate_count"][at] == 1


def test_ambiguous_match_has_small_weight_instead_of_becoming_reliable_geometry():
    features = np.zeros((2, 5, 5, 2), np.float32)
    valid = np.zeros((2, 5, 5), bool)
    features[0, 2, 2] = features[1, 2, 2] = [1, 0]
    features[0, 2, 3] = [0, 1]
    features[1, 2, 3] = [1, .01]
    valid[:, 2, 2:4] = True
    result = match_spatial_bev(features, valid, .5)
    at = (0, 2, 2)
    assert result["spatial_bev_match_state"][at] == 1
    assert 0 < result["spatial_bev_match_confidence"][at] < .01
    assert result["spatial_bev_match_usable"][at]
    assert not result["spatial_bev_match_valid"][at]


def test_low_confidence_flow_bev_vote_is_weighted_and_not_marked_reliable():
    inputs = video_case(3)
    inputs["lab"][..., 0] = 70
    inputs["delta_L"][:] = 0
    maps = video_proposals(**inputs)
    features = np.zeros((3, 16, 16, 2), np.float32)
    valid = np.zeros((3, 16, 16), bool)
    features[:, 6, 6] = [1, 0]
    features[:, 6, 7] = [1, .01]
    valid[:, 6, 6:8] = True
    flow = np.zeros((2, 16, 16, 2), np.float32)
    flow[..., 0] = .75
    result = cross_space_proposals(maps, features, valid, flow, np.ones((2, 16, 16), bool), .5)
    assert result["flow_bev_discrepancy_supported"][:, 6, 6].all()
    assert not result["flow_bev_discrepancy_valid"].any()
    assert (result["cross_space_discovery_score"][:, 6, 6] > 0).all()
    assert (result["cross_space_discovery_score"][:, 6, 6] < .01).all()


def test_stable_geometry_and_ambiguous_semantics_preserve_illumination_evidence():
    inputs = video_case(3)
    maps = video_proposals(**inputs)
    features = np.ones((3, 16, 16, 4), np.float32)
    result = cross_space_proposals(maps, features, np.ones((3, 16, 16), bool),
        np.zeros((2, 16, 16, 2), np.float32), np.ones((2, 16, 16), bool), .5)
    assert (result["spatial_bev_match_state"] == 3).all()
    assert result["bev_flow_matchable_pair"].all()
    np.testing.assert_allclose(result["ground_flow_geometry_residual_m"], 0)
    assert result["cross_space_discovery_score"][1, 6, 6] >= .35
    assert np.isnan(result["spatial_bev_local_residual_m"]).all()


def test_coverage_separates_observable_geometry_and_local_descriptor_matching():
    inputs = video_case(3)
    inputs["receiver"][0, :, :8] = False
    maps = video_proposals(**inputs)
    valid = np.ones((3, 16, 16), bool)
    valid[:, :, 12:] = False
    result = cross_space_proposals(maps, np.ones((3, 16, 16, 2)), valid,
        np.zeros((2, 16, 16, 2)), np.ones((2, 16, 16), bool), .5)
    report = correspondence_summary(result)
    assert report["totals"]["observable_receiver_cells"] == 384
    assert report["totals"]["geometry_anchored_descriptor_cells"] == 256
    assert report["totals"]["descriptor_coverage"] == 256/384
    assert report["totals"]["reliable_local_match_cells"] == 0
    assert not report["correctness_is_measured_accuracy"]


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


def growth_case(length=3):
    shape = (length, 13, 13)
    lab = np.zeros((*shape, 3), np.float32)
    lab[..., 0] = 40
    lab[..., 1:] = 128
    maps = {"cross_space_discovery_score": np.zeros(shape, np.float32),
        "receiver_valid_by_frame": np.ones(shape, bool),
        "search_valid_by_frame": np.ones(shape, bool),
        "appearance_score_map": np.full(shape, .08, np.float32),
        "boundary_confidence": np.zeros(shape, np.float32),
        "pair_correspondence_valid": np.ones((length-1, 13, 13), bool),
        "exposure_valid": np.ones((length-1, 13, 13), bool),
        "spatial_bev_feature_change": np.zeros((length-1, 13, 13), np.float32)}
    return maps, lab


def test_cross_space_seed_grows_weak_current_appearance_with_bounded_radius():
    maps, lab = growth_case()
    maps["cross_space_discovery_score"][1, 6, 6] = .36
    result = grow_cross_space_regions(maps, lab, .5, VideoShadowConfig())
    assert result["cross_space_spatial_growth"][1, 6, 8]
    assert not result["cross_space_region_support"][1, 6, 11]
    assert (maps["appearance_score_map"][result["cross_space_spatial_growth"]] < .18).all()
    regions = refine_regions(result["cross_space_region_score"][1], maps["boundary_confidence"][1],
        maps["receiver_valid_by_frame"][1], np.ones((13, 13, 2)), np.ones((13, 13), bool), VideoShadowConfig())
    assert regions and sum(int(region["mask"].sum()) for region in regions) > 1


def test_uniform_shadow_interior_does_not_need_old_darkness_detector_to_fire():
    maps, lab = growth_case()
    maps["appearance_score_map"][:] = 0
    maps["appearance_score_map"][1, 6, 6] = .08
    maps["cross_space_discovery_score"][1, 6, 6] = .36
    result = grow_cross_space_regions(maps, lab, .5, VideoShadowConfig())
    assert result["cross_space_spatial_growth"][1, 6, 8]
    assert result["cross_space_region_appearance_support"][1, 6, 8] >= .05
    assert not result["cross_space_region_support"][0].any()


def test_growth_does_not_cross_material_light_boundary_missing_or_suppressed_cells():
    maps, lab = growth_case()
    maps["cross_space_discovery_score"][1, 6, 6] = .36
    lab[:, :, 8:, 0] = 60
    lab[:, 8:, :, 1] = 160
    maps["boundary_confidence"][:, :, 4] = 1
    maps["receiver_valid_by_frame"][:, 4] = False
    maps["search_valid_by_frame"][:, 5, 6] = False
    lab[:, 6, 7] = np.nan
    result = grow_cross_space_regions(maps, lab, .5, VideoShadowConfig())
    support = result["cross_space_region_support"]
    assert not support[:, :, 8:].any()
    assert not support[:, 8:].any()
    assert not support[:, :, :5].any()
    assert not support[:, :5].any()
    assert not support[:, 5, 6].any()
    assert not support[:, 6, 7].any()


def test_temporal_seed_is_one_hop_only_and_can_start_a_current_observed_region():
    maps, lab = growth_case(5)
    maps["cross_space_discovery_score"][1, 6, 6] = .36
    maps["pair_correspondence_valid"][0] = False
    result = grow_cross_space_regions(maps, lab, .5, VideoShadowConfig())
    assert result["cross_space_temporal_seed"][2, 6, 6]
    assert result["cross_space_temporal_source_frame"][2, 6, 6] == 1
    assert not result["cross_space_region_support"][[0, 3, 4]].any()
    assert not result["cross_space_direct_seed"][2].any()
    regions = refine_regions(result["cross_space_region_score"][2], maps["boundary_confidence"][2],
        maps["receiver_valid_by_frame"][2], np.ones((13, 13, 2)), np.ones((13, 13), bool),
        VideoShadowConfig(), result["cross_space_temporal_seed"][2])
    assert regions  # Below the original seed threshold, but observed now.


def test_temporal_support_requires_each_destination_observation_and_pair_separately():
    for failure in ("invisible", "exposure", "missing_bev", "no_appearance", "bright", "time_gap"):
        maps, lab = growth_case()
        maps["cross_space_discovery_score"][0, 6, 6] = .36
        if failure == "invisible":
            maps["receiver_valid_by_frame"][1] = False
        elif failure == "exposure":
            maps["exposure_valid"][0] = False
        elif failure == "missing_bev":
            maps["spatial_bev_feature_change"][0] = np.nan
        elif failure == "no_appearance":
            maps["appearance_score_map"][1] = 0
        elif failure == "bright":
            lab[1, ..., 0] = 70
        timestamps = [0, 2_000_000, 2_500_000] if failure == "time_gap" else [0, 500_000, 1_000_000]
        result = grow_cross_space_regions(maps, lab, .5, VideoShadowConfig(), timestamps=timestamps)
        assert not result["cross_space_region_support"][1:].any(), failure


def test_growth_never_creates_regions_without_cross_space_seeds():
    maps, lab = growth_case()
    maps["appearance_score_map"][:] = .5
    result = grow_cross_space_regions(maps, lab, .5, VideoShadowConfig(), config=CrossSpaceGrowthConfig())
    assert not result["cross_space_region_support"].any()
