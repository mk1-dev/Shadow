import numpy as np
import pytest

from shadow_occ.annotation.entity_memory import CasterMemory, EntityMotionConfig, match_entity_motion
from shadow_occ.annotation.receiver_explanations import box_ray_evidence
from shadow_occ.annotation.shadow_video import video_proposals
from tests.test_receiver_explanations import ray_case
from tests.test_shadow_video import video_case


def matched_case():
    values = box_ray_evidence(*ray_case())
    valid = np.ones((2, 1, 3), bool)
    return values, valid, np.zeros(valid.shape, np.float32)


def test_surface_motion_matching_requires_geometry_and_preserves_boundary_and_ground():
    values, valid, error = matched_case()
    maps = match_entity_motion(values, valid, error)
    assert maps["entity_suppression_mask"][0, 0, 2]  # Interior front surface.
    assert not maps["entity_suppression_mask"][0, 0, 0]  # Near silhouette.
    assert maps["entity_uncertain_mask"][0, 0, 0]
    assert not maps["entity_suppression_mask"][0, 0, 1]  # Ground, same observed flow.
    assert not maps["entity_suppression_mask"][-1].any()  # No outgoing evidence.
    assert (maps["entity_suppression_mask"] <= maps["entity_occlusion_mask"]).all()


@pytest.mark.parametrize("failure", ["flow", "fb", "nonfinite", "target_visibility", "fallback"])
def test_unreliable_entity_motion_remains_occluded_not_shadow_or_background(failure):
    values, valid, error = matched_case()
    if failure == "flow":
        values["box_surface_flow_residual_px"][:] = 20
    elif failure == "fb":
        valid[:] = False
    elif failure == "nonfinite":
        error[:] = np.nan
    elif failure == "target_visibility":
        values["box_surface_target_visible"][:] = False
    else:
        values["box_surface_flow_valid"][:] = False
    maps = match_entity_motion(values, valid, error)
    assert not maps["entity_suppression_mask"].any()
    np.testing.assert_array_equal(maps["entity_uncertain_mask"], values["physical_ray_volume_support"])


def test_predicted_surface_occluded_by_another_box_is_not_motion_matched():
    data = ray_case()
    data[1][1]["object_volumes_world"].append({"instance_token": "closer",
        "source": "annotation_time_exclusion_only", "corners_world": np.array([
            [x, y, z] for x in (-.5, 1) for y in (-1, 1) for z in (6, 8)])})
    values = box_ray_evidence(*data)
    assert values["box_surface_flow_valid"][0, 0, 2]
    assert not values["box_surface_target_visible"][0, 0, 2]
    assert not match_entity_motion(values, np.ones((2, 1, 3), bool),
        np.zeros((2, 1, 3), np.float32))["entity_suppression_mask"][0, 0, 2]


def test_disabling_suppression_does_not_turn_occlusion_into_free_ground():
    values, valid, error = matched_case()
    on = match_entity_motion(values, valid, error)
    off = match_entity_motion(values, valid, error, enabled=False)
    assert on["entity_suppression_mask"].any() and not off["entity_suppression_mask"].any()
    np.testing.assert_array_equal(on["entity_occlusion_mask"], off["entity_occlusion_mask"])
    np.testing.assert_array_equal(on["entity_motion_matched_by_interval"], off["entity_motion_matched_by_interval"])


def test_entity_exclusion_precedes_static_and_dynamic_candidate_generation():
    inputs = video_case()
    suppression = np.zeros(inputs["receiver"].shape, bool)
    suppression[:, 5:7, 5:7] = True
    occlusion = suppression.copy()
    occlusion[:, 7, 5:7] = True
    maps = video_proposals(**inputs, entity_suppression_mask=suppression, entity_occlusion_mask=occlusion)
    assert not maps["proposal_score_map"][occlusion].any()
    assert not maps["receiver_valid_by_frame"][occlusion].any()
    assert maps["proposal_score_map"][1, 8, 8] > .9  # Adjacent shadow retained.
    np.testing.assert_array_equal(maps["entity_search_removed_mask"], occlusion)


def memory_case():
    from tests.test_physics_shadow_labels import pair_rows
    pairs = pair_rows()
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]
    for frame in frames:
        frame["object_volumes_world"] = []
    return pairs, frames


def test_memory_preserves_caster_without_surface_observations_and_drives_geometry():
    pairs, frames = memory_case()
    memory = CasterMemory(pairs, frames)
    from shadow_occ.annotation.physics_labels import window_tracks
    expected = window_tracks(*pairs)[0][0]
    np.testing.assert_array_equal(memory.tracks(1)[0]["points"], expected["points"])
    assert memory.states("car").keys() == {0, 1, 2}
    called = []
    def features(root, token, centers):
        called.append(centers.copy())
        return np.ones((*centers.shape[:2], 4)), np.ones(centers.shape[:2], bool), {}
    memory.observe(1, np.ones((4, 4, 3)), np.full((4, 4), -1), np.zeros((4, 4), bool),
                   "cache", "s1", features)
    np.testing.assert_array_equal(called[0][0, 0], expected["points"][1, 0])
    context = memory.association_context("car", 1)
    assert context["bev_descriptor_available"] and not context["appearance_available"]
    assert not context["matched_entity_cells"]
    row = memory.rows()[0]
    assert row["instance_token"] == "car" and len(row["observations"]) == 3
    np.testing.assert_allclose(row["observations"][1]["velocity_world_mps"], [4, 0, 0])


def test_memory_keeps_continuity_validation_and_does_not_invent_missing_tracks():
    pairs, frames = memory_case()
    pairs[1]["objects"][0]["annotation_token_t"] = "wrong_identity"
    memory = CasterMemory(pairs, frames)
    assert not memory.tracks(1) and memory.rejected[1]
    assert len(memory.states("car")) == 3  # Identity remains for context, not validated triplet.
    assert memory.states("absent") == {}


def test_memory_appearance_uses_only_matched_surface_of_the_same_identity():
    pairs, frames = memory_case()
    points = np.asarray(pairs[0]["objects"][0]["landmarks_world_t"])
    frames[0]["object_volumes_world"] = [{"instance_token": "car", "corners_world": points[1:]}]
    memory = CasterMemory(pairs, frames)
    lab = np.full((3, 3, 3), [50, 128, 128], np.float32)
    owners = np.full((3, 3), -1)
    owners[1, 1] = 0
    def features(root, token, centers):
        return np.ones((*centers.shape[:2], 2)), np.zeros(centers.shape[:2], bool), {}
    memory.observe(0, lab, owners, np.ones((3, 3), bool), None, "s0", features)
    state = memory.rows()[0]["observations"][0]
    assert state["matched_entity_cells"] == 1
    assert len(state["appearance_descriptor"]) == 24
    assert state["bev_descriptor"] is None


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_entity_thresholds_are_explicit_and_checked(value):
    with pytest.raises(ValueError):
        EntityMotionConfig(core_margin_px=value).validate()
