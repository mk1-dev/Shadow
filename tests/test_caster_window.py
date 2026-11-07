import numpy as np
import pytest

from shadow_occ.annotation.caster_window import (
    caster_proposals, component_transition_diagnostics, extend_pairs, extended_trajectory_fit,
    long_transition_evidence, receiver_sampling_summary, sample_receiver_flow, sample_receiver_frame,
    summarize_track_states,
)
from shadow_occ.annotation.physics_labels import footprint_mask
from tests.test_physics_shadow_labels import boxes, projection
from tests.test_receiver_explanations import receiver_case


def chain():
    frames = [{"sample_data_token": f"c{index}"} for index in range(7)]
    return [{"sample_token": f"s{index}", "next_sample_token": f"s{index+1}", "scene_name": "scene",
        "official_split": "train", "camera": "CAM_FRONT", "frames": frames[index:index+2]}
        for index in range(6)]


def test_gt_roi_expands_only_observed_dark_supported_ground():
    arrays, mask, _, polygons = receiver_case()
    arrays["shadow_candidate"] = np.zeros_like(mask)
    arrays["learned_bev_receiver_support"] = np.ones_like(mask)
    cache = {("car", 0): polygons["car", 0]}
    roi, metadata = caster_proposals(arrays, [{"instance_token": "car"}], [[1, 0, -1]],
        [0, 0, 1, 0], projection(), cache)
    assert roi.any() and metadata["added_observed_pixels"] == mask.sum()
    np.testing.assert_array_equal(arrays["shadow_candidate"], mask)
    arrays["shadow_candidate"][:] = False
    arrays["learned_bev_receiver_support"][:] = False
    caster_proposals(arrays, [{"instance_token": "car"}], [[1, 0, -1]], [0, 0, 1, 0], projection(), cache)
    assert not arrays["shadow_candidate"].any()


def test_caster_pool_is_observed_signal_not_an_original_candidate_subset():
    arrays, mask, _, polygons = receiver_case()
    arrays["shadow_candidate"] = np.zeros_like(mask)
    arrays["learned_bev_receiver_support"] = np.ones_like(mask)
    arrays["appearance_local_ratio"][1][mask] = .90
    roi, metadata = caster_proposals(arrays, [{"instance_token": "car"}], [[1, 0, -1]],
        [0, 0, 1, 0], projection(), {("car", 0): polygons["car", 0]}, proposal_darkness_ratio=.92)
    assert roi.any() and metadata["caster_own_component_pixels"] == mask.sum()
    assert metadata["overlap_original"] == 0 and metadata["added"] == mask.sum()
    assert arrays["shadow_candidate"][mask].all()


def test_partial_gt_tracks_are_reported_and_never_extrapolated():
    points = np.zeros((9, 3))
    states = {"complete": {"s0": points, "s1": points, "s2": points},
              "partial": {"s0": points, "s1": points},
              "inconsistent": {"s0": points, "s1": points, "s2": points}}
    complete, diagnostics, counts = summarize_track_states(states, {"inconsistent"}, ["s0", "s1", "s2"])
    assert set(complete) == {"complete"}
    assert counts == {"seen": 3, "complete": 1, "partial": 1, "inconsistent": 1, "missing_samples": 1}
    assert diagnostics["partial"]["missing_sample_tokens"] == ["s2"]


def test_long_window_uses_consecutive_camera_frames_without_crossing_scene():
    pairs = chain()
    prefix = ("scene", "train", "CAM_FRONT")
    forward = {(*prefix, item["sample_token"]): item for item in pairs}
    backward = {(*prefix, item["next_sample_token"]): item for item in pairs}
    result = extend_pairs(pairs[2], pairs[3], forward, backward, 3)
    assert len(result)+1 == 7
    assert extend_pairs(pairs[2], pairs[3], forward, backward, 1) == pairs[2:4]
    broken = {**backward, (*prefix, "s2"): {**pairs[1], "frames": [{}, {"sample_data_token": "other_camera"}]}}
    assert extend_pairs(pairs[2], pairs[3], forward, broken, 3)[0] is pairs[2]


def test_remote_observations_recover_events_missing_from_central_triplet():
    shape = (64, 96)
    points = np.stack([boxes()[1]+[offset, 0, 0] for offset in (-4, 0, 0, 0, 4)])
    direction = np.array([1, 0, -1])/np.sqrt(2)
    footprints = [footprint_mask(value[1:], [0, 0, 1, 0], direction, projection(), shape) for value in points]
    mask = footprints[2]
    onset, recovery = np.zeros((5, *shape), bool), np.zeros((5, *shape), bool)
    onset[0], recovery[4] = mask & ~footprints[0], mask & ~footprints[4]
    assert not onset[1:4].any() and not recovery[1:4].any()
    context = {"frames": [{}]*5, "middle": 2, "tracks": {"car": points},
               "polygon_cache": {}, "onset": onset, "recovery": recovery}
    fit = extended_trajectory_fit(context, "car", 0, direction, [0, 0, 1, 0], projection(), mask)
    assert fit["frames"] == 5 and fit["observed_transition_types"] == 2
    assert fit["edge_fit"] == 1 and fit["temporal_gain"] > 0
    assert all(item["reason"] == "matched" for item in fit["event_alignment"])
    assert extended_trajectory_fit(context, "missing_track", 0, direction, [0, 0, 1, 0], projection(), mask) is None


def transition_case():
    observations = np.full((5, 20, 20), 60., np.float32)
    mask = np.zeros((20, 20), bool)
    mask[8:12, 8:12] = True
    observations[1:4, mask] = 30
    visible = np.ones_like(observations, bool)
    return observations, visible, np.ones((4, 20, 20), bool), mask


def test_long_diagnostics_preserve_event_rule_and_separate_reference():
    observations, visible, flow, mask = transition_case()
    result = long_transition_evidence(observations, visible, flow, 2, mask)
    for index, fit in enumerate(result["exposures"]):
        expected = np.zeros_like(mask)
        if index != 2 and fit["valid"]:
            expected = visible[index] & visible[2]
            for step in range(min(index, 2), max(index, 2)):
                expected &= flow[step]
            expected &= observations[2]-(fit["scale"]*observations[index]+fit["offset_L"]) <= -6
        np.testing.assert_array_equal(result["onset"][index] | result["recovery"][index], expected)
    assert result["frame_diagnostics"][2]["stage"] == "reference_frame"
    assert result["frame_diagnostics"][2]["normalized_darkening_pixels"] is None
    component = component_transition_diagnostics(result, mask)
    assert component["event_pixels"] == mask.sum()
    assert component["frames"][0]["onset_pixels"] == mask.sum()
    assert component["frames"][4]["recovery_pixels"] == mask.sum()


def test_diagnostics_distinguish_roi_exposure_failure_from_no_event():
    observations, visible, flow, mask = transition_case()
    result = long_transition_evidence(observations, visible, flow, 2, np.ones_like(mask))
    frame = result["frame_diagnostics"][0]
    assert frame["stage"] == "roi_excluded_background"
    assert frame["raw_darkening_pixels"] == mask.sum()
    assert frame["normalized_darkening_pixels"] is None and frame["event_pixels"] == 0
    assert frame["exposure"]["reason"] == "insufficient_background"
    assert not result["onset"].any() and not result["recovery"].any()


def test_diagnostics_locate_flow_chain_support_loss():
    observations, visible, flow, mask = transition_case()
    flow[1] = False
    result = long_transition_evidence(observations, visible, flow, 2, mask)
    frame = result["frame_diagnostics"][0]
    assert frame["stage"] == "flow_chain_removed_support"
    assert frame["flow_chain"] == [{"pair_index": 0, "before_pixels": 400, "after_pixels": 400},
                                    {"pair_index": 1, "before_pixels": 400, "after_pixels": 0}]


def test_static_footprint_events_report_no_predicted_change():
    points = np.stack([boxes()[1]]*5)
    direction = np.array([1, 0, -1])/np.sqrt(2)
    mask = footprint_mask(points[2, 1:], [0, 0, 1, 0], direction, projection(), (64, 96))
    onset = np.zeros((5, *mask.shape), bool)
    onset[0] = mask
    context = {"frames": [{}]*5, "middle": 2, "tracks": {"car": points},
        "polygon_cache": {}, "onset": onset, "recovery": np.zeros_like(onset)}
    fit = extended_trajectory_fit(context, "car", 0, direction, [0, 0, 1, 0], projection(), mask)
    assert fit["edge_fit"] == 0 and fit["event_alignment"][0]["reason"] == "no_predicted_change"


def sampling_case():
    world = np.array([[[2.25, 2.25, 1.]]])
    frame = {"sample_data_token": "camera", "image_size_wh": [8, 6],
        "projection_world_to_image": np.eye(3, 4).tolist()}
    return world, frame, np.ones((6, 8), bool), np.ones((1, 1), bool)


def test_receiver_projection_uses_xy_pixels_and_preserves_strict_support():
    world, frame, ground, receiver = sampling_case()
    coordinates, visible, trace = sample_receiver_frame(world, frame, ground, receiver)
    np.testing.assert_array_equal(coordinates, [[[2.25, 2.25]]])
    assert visible.all() and trace["in_image_pixels"] == 1
    assert trace["zero_support_reason"] == "supported"


@pytest.mark.parametrize("failure", ["empty_source_ground", "empty_anchor_receiver", "no_front_projection", "outside_image"])
def test_receiver_sampling_separates_geometry_and_source_mask_failures(failure):
    world, frame, ground, receiver = sampling_case()
    if failure == "empty_source_ground":
        ground[:] = False
    elif failure == "empty_anchor_receiver":
        receiver[:] = False
    elif failure == "no_front_projection":
        world[..., 2] = -1
    else:
        world[..., 0] = 20
    _, visible, trace = sample_receiver_frame(world, frame, ground, receiver)
    assert not visible.any() and trace["zero_support_reason"] == failure


def test_binary_sampling_erosion_is_diagnostic_not_relaxed_support():
    world, frame, ground, receiver = sampling_case()
    ground[:] = False
    ground[2, 2] = True
    coordinates, visible, trace = sample_receiver_frame(world, frame, ground, receiver)
    assert not visible.any() and trace["zero_support_reason"] == "ground_bilinear_rejection"
    assert trace["ground_nearest_pixels"] == 1 and trace["ground_bilinear_pixels"] == 0
    supported, flow_trace = sample_receiver_flow({"receiver_flow_supported": ground}, coordinates, ground.shape, receiver)
    assert not supported.any() and flow_trace["zero_support_reason"] == "flow_bilinear_rejection"
    assert flow_trace["source_supported_pixels"] == 1


def test_receiver_sampling_rejects_different_image_pixel_frames():
    world, frame, ground, receiver = sampling_case()
    with pytest.raises(ValueError, match="pixel frame"):
        sample_receiver_frame(world, frame, ground[:4], receiver)
    coordinates, _, _ = sample_receiver_frame(world, frame, ground, receiver)
    with pytest.raises(ValueError, match="pixel frame"):
        sample_receiver_flow({"receiver_flow_supported": ground[:4]}, coordinates, ground.shape, receiver)


def test_receiver_sampling_summary_keeps_empty_reference_separate():
    world, frame, ground, receiver = sampling_case()
    receiver[:] = False
    _, _, trace = sample_receiver_frame(world, frame, ground, receiver)
    trace.update(is_reference=True, relative_frame=0, cached_central_visibility_mismatch_pixels=0)
    summary = receiver_sampling_summary([trace])
    assert summary["empty_reference_receiver_frames"] == 1
    assert summary["cached_central_mismatch_frames"] == 0
    assert summary["flow_comparisons"] == 0
