import numpy as np

from shadow_occ.annotation.caster_window import (
    caster_proposals, extend_pairs, extended_trajectory_fit, summarize_track_states,
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
    assert extended_trajectory_fit(context, "missing_track", 0, direction, [0, 0, 1, 0], projection(), mask) is None
