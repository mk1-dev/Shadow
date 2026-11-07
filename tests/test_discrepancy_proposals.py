import copy

import numpy as np
import pytest

from shadow_occ.annotation.cached_bev import select_bev_flow_receivers
from shadow_occ.annotation.discrepancy_proposals import DiscrepancyConfig, propose_discrepancy
from tests.test_cross_space import evidence


def weak_case(change=3.0):
    arrays, mask = evidence()
    arrays["appearance_local_ratio"][:] = .99
    arrays["temporal_illumination_change_L"][0, mask] = -change
    arrays["temporal_illumination_change_L"][1, mask] = change
    return arrays, mask


def test_active_branch_adds_candidates_outside_all_original_dark_seeds():
    arrays, mask = weak_case()
    baseline, _ = select_bev_flow_receivers(copy.deepcopy(arrays), {"counts": {}}, discovery_mode="cross-space")
    current, meta = select_bev_flow_receivers(arrays, {"counts": {}})
    assert not baseline["shadow_candidate"].any()
    np.testing.assert_array_equal(current["shadow_candidate"], mask)
    np.testing.assert_array_equal(current["discrepancy_novel_candidate"], mask)
    assert (current["proposal_source_bits"][mask] == 2).all()
    assert meta["discrepancy_discovery"]["novel_cells"] == mask.sum()
    assert meta["counts"]["discrepancy_novel_proposals"] == mask.sum()


def test_bev_reference_now_changes_proposal_selection_not_just_scores():
    arrays, mask = weak_case()
    stable, _ = select_bev_flow_receivers(copy.deepcopy(arrays), {"counts": {}})
    arrays["learned_bev_temporal_cosine"][:] = -1
    changed, _ = select_bev_flow_receivers(arrays, {"counts": {}})
    assert stable["discrepancy_candidate"][mask].all()
    assert not changed["discrepancy_candidate"].any()


def test_trusted_flow_residual_can_propose_weak_change_below_appearance_threshold():
    arrays, mask = weak_case(2)
    arrays["observed_optical_flow_px"][..., 0] = 3
    unavailable, _ = select_bev_flow_receivers(copy.deepcopy(arrays), {"counts": {}})
    arrays["observed_flow_valid_by_interval"][:] = True
    available, _ = select_bev_flow_receivers(arrays, {"counts": {}})
    assert not unavailable["discrepancy_candidate"].any()
    assert available["discrepancy_candidate"][mask].all()


def test_missing_or_unreliable_flow_is_not_an_anomaly():
    arrays, _ = weak_case(2)
    arrays["observed_optical_flow_px"][:] = 100
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert not arrays["discrepancy_candidate"].any()
    assert np.isnan(arrays["discrepancy_best_3d_flow_residual_px"]).all()


def test_ground_texture_flow_can_match_geometry_while_shadow_appearance_changes():
    arrays, mask = weak_case(6)
    arrays["observed_flow_valid_by_interval"][:] = True
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert arrays["discrepancy_candidate"][mask].all()
    np.testing.assert_allclose(arrays["discrepancy_ground_flow_residual_px"], 0)
    assert arrays["discrepancy_onset"][mask].all()


def test_flow_error_alone_cannot_become_shadow_candidate():
    arrays, _ = weak_case(0)
    arrays["observed_flow_valid_by_interval"][:] = True
    arrays["observed_optical_flow_px"][:] = 100
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert not arrays["shadow_candidate"].any()


def predicted_motion_case():
    arrays, mask = weak_case(2)
    arrays["observed_flow_valid_by_interval"][:] = True
    arrays["observed_optical_flow_px"][..., 0] = 3
    arrays["mature_expected_object_flow_px"] = arrays["observed_optical_flow_px"].copy()
    arrays["mature_entity_probability"] = np.ones((3, *mask.shape), np.float32)
    arrays["mature_inference_domain"] = np.ones((3, *mask.shape), bool)
    return arrays, mask


def test_projected_entity_motion_explains_change_instead_of_creating_proposal():
    arrays, _ = predicted_motion_case()
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert not arrays["discrepancy_candidate"].any()
    assert arrays["discrepancy_entity_explanation_valid"].all()
    np.testing.assert_allclose(arrays["discrepancy_entity_explanation"], 1)
    assert (arrays["discrepancy_motion_source"] == 3).all()


def test_wrong_predicted_motion_cannot_explain_observed_flow():
    arrays, mask = predicted_motion_case()
    arrays["mature_expected_object_flow_px"][..., 0] = -3
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert arrays["discrepancy_candidate"][mask].all()
    assert not arrays["discrepancy_entity_explanation"].any()
    assert (arrays["discrepancy_motion_source"] == 1).all()


def test_prediction_outside_its_inference_domain_is_unknown_not_entity_support():
    arrays, mask = predicted_motion_case()
    arrays["mature_inference_domain"][:] = False
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert arrays["discrepancy_candidate"][mask].all()
    assert not arrays["discrepancy_entity_explanation_valid"].any()


def test_gt_box_surface_motion_is_a_distinct_explanation_source():
    arrays, _ = predicted_motion_case()
    for key in list(arrays):
        if key.startswith("mature_"):
            del arrays[key]
    arrays["box_surface_expected_flow_px"] = arrays["observed_optical_flow_px"].copy()
    arrays["box_surface_flow_valid"] = arrays["observed_flow_valid_by_interval"].copy()
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert not arrays["discrepancy_candidate"].any()
    assert (arrays["discrepancy_motion_source"] == 2).all()


@pytest.mark.parametrize("reason", ["geometry", "exposure", "chroma", "entity"])
def test_invalid_observations_and_known_entity_do_not_create_new_proposals(reason):
    arrays, mask = weak_case(6)
    if reason == "geometry":
        arrays["receiver_valid"][:] = False
    elif reason == "exposure":
        arrays["temporal_exposure_valid"][:] = False
    elif reason == "chroma":
        arrays["appearance_chroma_change"] = np.full((2, *mask.shape), 60, np.float32)
    else:
        arrays["physical_box_footprint"] = np.broadcast_to(mask, (3, *mask.shape)).copy()
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert not arrays["discrepancy_candidate"].any()


def test_weak_proposal_does_not_lower_physical_transition_threshold():
    arrays, mask = weak_case(3)
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert arrays["discrepancy_candidate"][mask].all()
    assert not arrays["discrepancy_onset"].any()
    assert not arrays["shadow_boundary_onset"].any()


def test_static_dark_branch_survives_without_fabricating_discrepancy():
    arrays, mask = weak_case(0)
    arrays["appearance_local_ratio"][:, mask] = .7
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert arrays["shadow_candidate"][mask].all()
    assert not arrays["discrepancy_candidate"].any()
    assert (arrays["proposal_source_bits"][mask] == 1).all()


def test_central_brightening_has_the_wrong_sign_for_shadow_proposal():
    arrays, _ = weak_case(-6)
    select_bev_flow_receivers(arrays, {"counts": {}})
    assert not arrays["discrepancy_candidate"].any()


def test_old_cache_without_raw_flow_validity_still_uses_valid_brightness_evidence():
    arrays, mask = weak_case(6)
    del arrays["observed_flow_valid_by_interval"]
    _, meta = select_bev_flow_receivers(arrays, {"counts": {}})
    assert arrays["discrepancy_candidate"][mask].all()
    assert not meta["discrepancy_discovery"]["raw_flow_validity_available"]


def test_switching_to_baseline_cannot_reuse_active_candidates_or_source_bits():
    arrays, mask = weak_case()
    arrays, meta = select_bev_flow_receivers(arrays, {"counts": {}})
    assert arrays["shadow_candidate"][mask].all()
    arrays, meta = select_bev_flow_receivers(arrays, meta, discovery_mode="cross-space")
    assert not arrays["shadow_candidate"].any()
    assert "discrepancy_discovery" not in meta
    assert not any(key.startswith("discrepancy_") for key in arrays)
    assert "proposal_source_bits" not in arrays


def test_score_threshold_is_used_by_the_proposal_mask():
    arrays, mask = weak_case()
    select_bev_flow_receivers(arrays, {"counts": {}}, discrepancy_config=DiscrepancyConfig(minimum_score=.75))
    assert not arrays["discrepancy_candidate"].any()
    assert not arrays["shadow_candidate"][mask].any()


def test_standalone_branch_does_not_need_a_dark_seed_or_component_mask():
    arrays, mask = weak_case()
    del arrays["shadow_candidate"]
    meta = propose_discrepancy(arrays, cosine=arrays["learned_bev_temporal_cosine"])
    assert arrays["discrepancy_candidate"][mask].all()
    assert meta["novel_cells"] == mask.sum()


def test_caster_union_tracks_incremental_pixels_without_double_counting():
    from shadow_occ.annotation.caster_window import caster_proposals
    from tests.test_physics_shadow_labels import projection
    from tests.test_receiver_explanations import receiver_case

    arrays, mask, _, polygons = receiver_case()
    arrays["shadow_candidate"] = mask.copy()
    arrays["cross_space_receiver"] = arrays["receiver_valid"].copy()
    arrays["discrepancy_candidate"] = mask.copy()
    arrays["proposal_source_bits"] = mask.astype(np.uint8)*2
    caster_proposals(arrays, [{"instance_token": "car"}], [[1, 0, -1]], [0, 0, 1, 0],
        projection(), {("car", 0): polygons["car", 0]})
    assert (arrays["proposal_source_bits"][mask] == 6).all()
    assert not arrays["discrepancy_exclusive_candidate"].any()


def test_new_events_reach_receiver_physics_without_a_dark_ratio_gate():
    from shadow_occ.annotation.receiver_explanations import receiver_evidence

    arrays, mask = weak_case(6)
    arrays["appearance_L_raw"] = np.full((3, *mask.shape), 70, np.float32)
    select_bev_flow_receivers(arrays, {"counts": {}})
    physical = receiver_evidence(arrays)
    assert physical["receiver_boundary_formation"][mask].all()
    assert physical["receiver_boundary_recovery"][mask].all()
    assert not physical["receiver_dark_by_frame"].any()


def test_cached_cli_exports_new_proposals_and_keeps_source_unchanged(tmp_path, monkeypatch):
    import json
    import sys

    from shadow_occ.annotation.data import digest, read_rows, write_rows
    from shadow_occ.utils.io import write_json
    from tests.test_bev_flow_ablations import script

    arrays, mask = weak_case()
    arrays["shadow_candidate_photometric"] = arrays["shadow_candidate"].copy()
    source, output = tmp_path/"source", tmp_path/"active"
    source.mkdir()
    path = source/"window.npz"
    np.savez_compressed(path, **arrays)
    before = digest(path)
    write_rows(source/"windows.jsonl", [{"id": "window", "counts": {}, "evidence": str(path), "evidence_sha256": before}])
    write_json(source/"bev_flow_protocol.json", {"config": {"light_threshold_L": 6}})
    monkeypatch.setattr(sys, "argv", ["reselect", "--source-root", str(source), "--output-root", str(output),
        "--mature-object-mode", "off", "--discovery-mode", "discrepancy"])
    script("reselect_bev_flow").main()
    row = read_rows(output/"windows.jsonl")[0]
    with np.load(row["evidence"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["discrepancy_novel_candidate"], mask)
        np.testing.assert_array_equal(archive["shadow_candidate"], mask)
    assert digest(path) == before
    protocol = json.loads((output/"bev_flow_protocol.json").read_text())
    assert protocol["discovery_mode"] == "discrepancy"
    assert protocol["discrepancy_minimum_score"] == .35


def test_ablation_pair_changes_only_discrepancy_discovery():
    import json
    from pathlib import Path

    variants = json.loads((Path(__file__).parents[1]/"configs/bev_flow_ablations.json").read_text())["variants"]
    before, after = variants["appearance_only"], variants["discrepancy_proposal"]
    assert before["candidate_source"] == after["candidate_source"] == "photometric"
    assert before["discovery_mode"] == "cross-space" and after["discovery_mode"] == "discrepancy"
    assert variants["bidirectional_proposal"]["candidate_source"] == "caster"
