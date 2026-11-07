import copy

import numpy as np

from shadow_occ.annotation.cached_bev import select_bev_flow_receivers
from shadow_occ.annotation.cross_space import discover_cross_space
from shadow_occ.annotation.physics_labels import LabelConfig, candidate_evidence


def evidence():
    shape = (8, 8)
    mask = np.zeros(shape, bool)
    mask[2:6, 2:6] = True
    ratio = np.ones((3, *shape), np.float32)
    ratio[1, mask] = .7
    delta = np.zeros((2, *shape), np.float32)
    delta[0, mask], delta[1, mask] = -12, 12
    arrays = {"receiver_valid": np.ones(shape, bool), "appearance_local_ratio": ratio,
        "temporal_illumination_change_L": delta, "temporal_exposure_valid": np.ones(shape, bool),
        "shadow_candidate": np.zeros(shape, bool), "shadow_flow_supported": np.zeros(shape, bool),
        "learned_bev_valid": np.ones((3, *shape), bool),
        "learned_bev_temporal_cosine": np.ones((2, *shape), np.float32),
        "observed_flow_valid_by_interval": np.zeros((2, *shape), bool),
        "observed_optical_flow_px": np.zeros((2, *shape, 2), np.float32),
        "expected_geometric_flow_px": np.zeros((2, *shape, 2), np.float32)}
    for key in ("shadow_boundary_onset", "shadow_boundary_recovery", "shadow_same_patch_cycle"):
        arrays[key] = np.zeros(shape, bool)
    return arrays, mask


def test_discrepancy_discovers_change_without_bev_or_registered_flow():
    arrays, mask = evidence()
    arrays["learned_bev_valid"][:] = False
    arrays, meta = select_bev_flow_receivers(arrays, {"counts": {}}, discovery_mode="cross-space")
    np.testing.assert_array_equal(arrays["shadow_candidate"], mask)
    assert arrays["cross_space_discrepancy"][mask].min() > 0
    assert not arrays["cross_space_flow_valid"].any()
    assert meta["cross_space_discovery"]["candidate_without_bev"] == mask.sum()
    assert meta["learned_bev_selection"]["hard_intersection_cells"] == 0
    assert not arrays["shadow_flow_supported"].any()


def test_bev_stability_changes_evidence_not_candidate_existence():
    arrays, mask = evidence()
    first, _ = select_bev_flow_receivers(copy.deepcopy(arrays), {"counts": {}}, discovery_mode="cross-space")
    arrays["learned_bev_temporal_cosine"][:] = -1
    second, _ = select_bev_flow_receivers(arrays, {"counts": {}}, discovery_mode="cross-space")
    np.testing.assert_array_equal(first["shadow_candidate"], second["shadow_candidate"])
    assert (first["cross_space_discrepancy"][mask] > second["cross_space_discrepancy"][mask]).all()


def test_missing_remote_feature_keeps_valid_pair_measurement():
    arrays, mask = evidence()
    arrays["learned_bev_valid"][0] = False
    arrays["learned_bev_temporal_cosine"][0] = np.nan
    arrays, _ = select_bev_flow_receivers(arrays, {"counts": {}}, discovery_mode="cross-space")
    assert arrays["cross_space_bev_valid"][mask].all()
    np.testing.assert_allclose(arrays["cross_space_bev_stability"][mask], 1)


def test_unreliable_flow_is_not_residual_evidence():
    arrays, _ = evidence()
    arrays["observed_optical_flow_px"][:] = 100
    discover_cross_space(arrays)
    assert np.isnan(arrays["cross_space_flow_residual_px"]).all()
    arrays["observed_flow_valid_by_interval"][:] = True
    discover_cross_space(arrays)
    assert np.nanmin(arrays["cross_space_flow_residual_px"]) > 100


def test_flow_residual_and_stable_bev_alone_cannot_create_shadow():
    arrays, _ = evidence()
    arrays["appearance_local_ratio"][:] = 1
    arrays["temporal_illumination_change_L"][:] = 0
    arrays["observed_flow_valid_by_interval"][:] = True
    arrays["observed_optical_flow_px"][:] = 100
    select_bev_flow_receivers(arrays, {"counts": {}}, discovery_mode="cross-space")
    assert not arrays["shadow_candidate"].any()
    assert not arrays["cross_space_discrepancy"].any()


def test_entity_and_invalid_geometry_remain_hard_exclusions():
    arrays, mask = evidence()
    arrays["physical_box_footprint"] = np.broadcast_to(mask, (3, *mask.shape)).copy()
    discover_cross_space(arrays)
    assert not arrays["shadow_candidate"].any()
    del arrays["physical_box_footprint"]
    arrays["receiver_valid"][:] = False
    discover_cross_space(arrays)
    assert not arrays["shadow_candidate"].any()


def test_static_appearance_is_retained_without_temporal_discrepancy():
    arrays, mask = evidence()
    arrays["appearance_local_ratio"][:, mask] = .7
    arrays["temporal_illumination_change_L"][:] = 0
    discover_cross_space(arrays)
    np.testing.assert_array_equal(arrays["shadow_candidate"], mask)
    assert not arrays["cross_space_temporal_candidate"].any()
    assert not arrays["cross_space_discrepancy"].any()


def test_one_exposure_interval_is_sufficient_but_untrusted_intervals_are_not_events():
    arrays, mask = evidence()
    arrays["temporal_exposure_valid_by_interval"] = np.ones((2, *mask.shape), bool)
    arrays["temporal_exposure_valid_by_interval"][0] = False
    discover_cross_space(arrays)
    assert not arrays["shadow_boundary_onset"].any()
    assert arrays["shadow_boundary_recovery"][mask].all()
    arrays["temporal_exposure_valid_by_interval"][:] = False
    discover_cross_space(arrays)
    assert not arrays["cross_space_temporal_candidate"].any()


def test_large_chroma_change_does_not_become_illumination_event():
    arrays, _ = evidence()
    arrays["appearance_chroma_change"] = np.full((2, 8, 8), 50, np.float32)
    discover_cross_space(arrays)
    assert not arrays["cross_space_temporal_candidate"].any()


def test_switching_modes_restores_original_not_derived_candidates():
    arrays, mask = evidence()
    arrays, meta = select_bev_flow_receivers(arrays, {"counts": {}}, discovery_mode="cross-space")
    assert arrays["shadow_candidate"][mask].all()
    arrays, meta = select_bev_flow_receivers(arrays, meta, discovery_mode="hard-intersection")
    assert not arrays["shadow_candidate"].any()
    assert "cross_space_discovery" not in meta
    assert not any(key.startswith("cross_space_") for key in arrays)
    arrays, _ = select_bev_flow_receivers(arrays, meta, discovery_mode="cross-space")
    assert arrays["shadow_candidate"][mask].all()


def test_old_cache_missing_raw_flow_validity_is_unknown_not_perfect():
    arrays, _ = evidence()
    del arrays["observed_flow_valid_by_interval"]
    meta = discover_cross_space(arrays)
    assert not meta["flow_validity_available"]
    assert np.isnan(arrays["cross_space_flow_residual_px"]).all()


def test_geometric_correspondence_does_not_fake_optical_flow_support():
    arrays, mask = evidence()
    arrays, _ = select_bev_flow_receivers(arrays, {"counts": {}}, discovery_mode="cross-space")
    maps = {"onset": arrays["shadow_boundary_onset"], "recovery": arrays["shadow_boundary_recovery"],
        "flow_supported": arrays["shadow_flow_supported"], "same_patch_cycle": arrays["shadow_same_patch_cycle"],
        "geometric_correspondence": arrays["cross_space_receiver"]}
    report = candidate_evidence(mask, maps, LabelConfig(minimum_area=1, minimum_change_pixels=1))
    assert "insufficient_flow_correspondence" not in report["reasons"]
    assert report["flow_supported"] == 0
    assert report["geometric_correspondence"] == mask.sum()


def test_caster_projection_does_not_require_bev_or_flow_and_cannot_paint_lit_pixels():
    from shadow_occ.annotation.caster_window import caster_proposals
    from tests.test_physics_shadow_labels import projection
    from tests.test_receiver_explanations import receiver_case

    arrays, mask, _, polygons = receiver_case()
    arrays["cross_space_receiver"] = arrays["receiver_valid"].copy()
    arrays["shadow_candidate"] = np.zeros_like(mask)
    arrays["learned_bev_receiver_support"] = np.zeros_like(mask)
    arrays["shadow_flow_supported"][:] = False
    caster_proposals(arrays, [{"instance_token": "car"}], [[1, 0, -1]], [0, 0, 1, 0],
        projection(), {("car", 0): polygons["car", 0]})
    np.testing.assert_array_equal(arrays["shadow_candidate"], mask)
    arrays["appearance_local_ratio"][:] = 1
    arrays["shadow_candidate"][:] = False
    caster_proposals(arrays, [{"instance_token": "car"}], [[1, 0, -1]], [0, 0, 1, 0],
        projection(), {("car", 0): polygons["car", 0]})
    assert not arrays["shadow_candidate"].any()


def test_long_events_can_use_geometry_without_relabeling_failed_flow_as_valid():
    from shadow_occ.annotation.caster_window import long_transition_evidence
    from tests.test_caster_window import transition_case

    observations, visible, flow, roi = transition_case()
    flow[:] = False
    legacy = long_transition_evidence(observations, visible, flow, 2, roi)
    current = long_transition_evidence(observations, visible, flow, 2, roi, correspondence="geometry-pair")
    assert not legacy["onset"].any() and not legacy["recovery"].any()
    assert current["onset"].any() and current["recovery"].any()
    assert not current["flow_support"][0].any()
    assert current["correspondence_support"][0].all()
    assert current["frame_diagnostics"][0]["flow_supported_pixels"] == 0


def test_raw_flow_validity_is_separate_from_registered_ground_flow():
    from tests.test_bev_flow import build, example

    data = example()
    for flow in data[3]:
        flow["flow_reliable"] = np.ones((64, 64), bool)
        flow["receiver_flow_supported"][:] = False
        flow["flow_t_to_next"][:] = 10
    arrays, meta = build(data)
    arrays, meta = select_bev_flow_receivers(arrays, meta, feature_gate="none", discovery_mode="cross-space")
    assert arrays["observed_flow_valid_by_interval"].any()
    assert not arrays["shadow_flow_supported"].any()
    assert arrays["cross_space_flow_valid"].any()
    assert arrays["shadow_candidate"].any()


def test_prediction_motion_is_an_explanation_not_a_new_shadow_seed():
    arrays, mask = evidence()
    arrays["mature_entity_motion_fit"] = np.ones((2, *mask.shape), np.float32)
    discover_cross_space(arrays)
    assert arrays["shadow_candidate"][mask].all()
    assert not arrays["cross_space_discrepancy"].any()


def test_shuffled_bev_changes_scores_without_changing_observed_candidates():
    arrays, mask = evidence()
    arrays["learned_bev_receiver_features"] = np.broadcast_to(
        np.random.default_rng(8).normal(size=(1, 8, 8, 4)), (3, 8, 8, 4)).copy()
    aligned, _ = select_bev_flow_receivers(copy.deepcopy(arrays), {"counts": {}}, discovery_mode="cross-space")
    shuffled, _ = select_bev_flow_receivers(arrays, {"counts": {}}, feature_gate="shuffled", seed=4, discovery_mode="cross-space")
    np.testing.assert_array_equal(aligned["shadow_candidate"], shuffled["shadow_candidate"])
    assert not np.allclose(aligned["cross_space_discrepancy"][mask], shuffled["cross_space_discrepancy"][mask])
