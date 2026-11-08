import copy
import json

import numpy as np
import pytest

from shadow_occ.annotation.entity_memory import CasterMemory
from shadow_occ.annotation.shadow_video import video_proposals
from tests.test_entity_memory import memory_case
from tests.test_shadow_video import video_case
from tests.test_bev_flow_ablations import script


def test_search_can_observe_entities_without_claiming_ground_or_complete_discrepancy():
    inputs = video_case()
    blocked = np.zeros(inputs["receiver"].shape, bool)
    blocked[:, 5:7, 5:7] = True
    inputs["receiver"][blocked] = False
    inputs["entity_occlusion_mask"] = blocked
    inputs["search_domain"] = np.ones_like(blocked)
    inputs["ground_motion_reference_valid"] = np.ones_like(inputs["delta_L"], bool)
    a = video_proposals(**inputs)
    b = video_proposals(**inputs, entity_suppression_mask=blocked)
    assert a["proposal_score_map"][1, 6, 6] > .9
    assert not b["proposal_score_map"][blocked].any()
    assert not a["receiver_valid_by_frame"][blocked].any()
    assert not a["discrepancy_evidence_complete"][blocked].any()
    assert np.isnan(a["discrepancy_score_map"][1, 6, 6])
    assert a["proposal_score_map"][1, 8, 8] > .9 and b["proposal_score_map"][1, 8, 8] > .9
    np.testing.assert_array_equal(a["receiver_valid_by_frame"], b["receiver_valid_by_frame"])


def owner_case():
    pairs, frames = memory_case()
    memory = CasterMemory(pairs, frames)
    memory.records["rival"] = copy.deepcopy(memory.records["car"])
    memory.records["rival"]["instance_token"] = "rival"
    for token in ("car", "rival"):
        for slot, state in memory.records[token]["states"].items():
            state["appearance_descriptor"] = [1., 0.] if token == "car" or slot == 1 else [0., 1.]
            state["bev_descriptor"] = state["appearance_descriptor"]
    def candidate(slot, shadow_id=1):
        return {"scene_name": "scene", "official_split": "train", "shadow_track_id": shadow_id,
            "frame_slot": slot, "hypotheses": [{"instance_token": token, "light_id": 0,
                "score": score, "existence_score": score, "shadow_fit_passes": True}
                for token, score in (("car", .8), ("rival", .88))]}
    current, other = candidate(1), candidate(2)
    return memory, current, [current, other]


def test_owner_descriptor_weighting_actually_changes_ranking_of_physical_candidates():
    memory, current, pool = owner_case()
    geometry, report = memory.score_owner(current, pool, "geometry")
    richer, rich = memory.score_owner(current, pool, "geometry+appearance+bev")
    assert geometry["instance_token"] == "rival"
    assert richer["instance_token"] == "car"
    assert rich["ranked_owners"][0]["descriptor_consistency"]["appearance_descriptor"]["pairs"] == 1
    assert report["shadow_existence_changed"] is False


def test_missing_descriptors_fall_back_to_identical_geometry_scores():
    memory, current, pool = owner_case()
    for record in memory.records.values():
        for state in record["states"].values():
            state.pop("appearance_descriptor")
            state.pop("bev_descriptor")
    a, ar = memory.score_owner(current, pool, "geometry")
    b, br = memory.score_owner(current, pool, "geometry+appearance+bev")
    assert a == b
    assert [r["score"] for r in ar["ranked_owners"]] == [r["score"] for r in br["ranked_owners"]]


def test_different_shadow_track_or_same_frame_cannot_supply_temporal_owner_support():
    memory, current, pool = owner_case()
    pool[1]["shadow_track_id"] = 2
    selected, report = memory.score_owner(current, pool, "geometry")
    assert selected is None and report["reason"] == "insufficient_linked_shadow_frames"
    pool[1]["shadow_track_id"] = 1
    pool[1]["frame_slot"] = 1
    assert memory.score_owner(current, pool, "geometry")[0] is None


def test_descriptor_agreement_cannot_rescue_a_failed_physical_fit():
    memory, current, pool = owner_case()
    for row in pool:
        row["hypotheses"][0]["shadow_fit_passes"] = False
    association, report = memory.score_owner(current, pool, "geometry+appearance+bev")
    assert association is None
    assert all(row["instance_token"] != "car" for row in report["ranked_owners"])


def test_light_direction_must_agree_across_owner_history():
    memory, current, pool = owner_case()
    for item in pool[1]["hypotheses"]:
        item["light_id"] = 1
    assert memory.score_owner(current, pool, "geometry")[0] is None


def test_observed_boundary_motion_and_counterfactual_gain_enter_owner_reasoning():
    memory, current, pool = owner_case()
    for row in pool:
        row["observed_transition_pixels"] = 5
        row["hypotheses"][0].update(available_edge_fit=1., temporal_gain=.1)
        row["hypotheses"][1].update(available_edge_fit=0., temporal_gain=0.)
    association, report = memory.score_owner(current, pool, "geometry")
    assert association["instance_token"] == "car"
    assert report["ranked_owners"][0]["geometry_includes_observed_transition_fit"]
    assert not report["shadow_existence_changed"]


def test_paired_runner_changes_only_suppression_and_owner_policy(tmp_path, monkeypatch):
    module = script("compare_shadow_entity_ab")
    inputs = video_case(3)
    inputs["entity_suppression_mask"] = np.ones((3, 16, 16), bool)
    inputs["entity_occlusion_mask"] = np.ones((3, 16, 16), bool)
    inputs["search_domain"] = np.ones((3, 16, 16), bool)
    context = {"entity_maps": {"entity_suppression_mask": inputs["entity_suppression_mask"]}}
    calls = []
    def run(clip, chain, arm, state, path, config, **kwargs):
        calls.append((arm, state, kwargs))
        return {"id": clip["id"]}
    monkeypatch.setattr(module, "run_clip", run)
    module.compare_clip({"id": "clip"}, [], inputs, context, tmp_path, None)
    assert len(calls) == 3
    assert not calls[0][0]["entity_suppression_mask"].any()
    assert calls[1][0]["entity_suppression_mask"].all()
    assert calls[0][2]["owner_scoring"] == calls[1][2]["owner_scoring"] == "geometry"
    assert calls[2][2]["owner_scoring"] == "geometry+appearance+bev"
    for name in ("lab", "receiver", "observed_flow", "features", "search_domain"):
        assert calls[0][0][name] is calls[1][0][name] is calls[2][0][name]
    assert inputs["entity_suppression_mask"].all()  # No mutation of shared evidence.


def test_missing_independent_review_does_not_report_perfect_shadow_retention(tmp_path):
    module = script("compare_shadow_entity_ab")
    results = {}
    for name in module.VARIANTS:
        folder = tmp_path/name/"clip"
        folder.mkdir(parents=True)
        (folder/"reviewed_signals.jsonl").write_text("")
        results[name] = {"id": "clip", "timing": {"proposal_seconds": .1, "region_seconds": .1,
            "physics_seconds": .1, "owner_seconds": .1, "total_seconds": 1.},
            **{key: 0 for key in ("frames", "regions", "candidate_cells", "entity_candidate_cells", "positive_cells",
            "owner_cells", "unique_caster_regions", "additional_search_removed_cells")}}
    report = module.summarize([results], tmp_path)
    for row in report["variants"].values():
        assert row["reviewed_shadow_preservation"]["search_retention"] is None
