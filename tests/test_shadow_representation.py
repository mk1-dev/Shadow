import cv2
import numpy as np

from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_image_cell_indices, bev_mask_to_image, bev_projection
from shadow_occ.annotation.entity_memory import CasterMemory
from shadow_occ.annotation.physics_labels import light_bank
from shadow_occ.annotation.shadow_representation import describe_regions, memory_state, paste_region_image, project_region_memory, refine_image_region
from shadow_occ.annotation.shadow_video import ShadowRegionMemory
from tests.test_entity_memory import memory_case
from tests.test_shadow_video import region
from tests.test_bev_flow import example


def test_rgb_region_refinement_obeys_observed_dark_boundary_and_ground():
    image = np.full((100, 100, 3), 200, np.uint8)
    image[35:65, 35:65] = 50
    seed = np.zeros((100, 100), bool)
    seed[30:70, 30:70] = True
    ground = np.ones(seed.shape, bool)
    ground[55:] = False
    refined = refine_image_region(image, seed, ground)
    actual = np.zeros(seed.shape, bool)
    paste_region_image(refined, actual, True)
    assert actual.any() and not actual[55:].any()
    assert not actual[:30].any() and not actual[:, :30].any()
    assert not refine_image_region(np.full_like(image, 200), seed, ground)["image_mask_roi"].any()


def test_image_grid_lookup_can_be_reused_without_changing_coordinates():
    pairs = example()[0]
    grid = BEVFlowConfig(extent_m=4).grid()
    projection = pairs[0]["frames"][0]["projection_world_to_image"]
    basis = pairs[0]["plane_basis_world"]
    mask = np.zeros(grid.shape, bool)
    mask[5:10, 5:10] = True
    cells = bev_image_cell_indices(projection, basis, grid, (64, 64))
    np.testing.assert_array_equal(bev_mask_to_image(mask, projection, basis, grid, (64, 64)),
        bev_mask_to_image(mask, projection, basis, grid, (64, 64), cell_indices=cells))


def test_region_memory_preserves_occluded_identity_without_emitting_a_mask():
    memory = ShadowRegionMemory(.5)
    first = region()
    first.update(id="000_000", boundary_descriptor=np.array([1., 0., 0.]),
        appearance_descriptor=np.array([1., 0.]), caster_candidates=["car"])
    identity = memory.update([first], 0)[0]["shadow_track_id"]
    assert memory.update([], 500000, receiver=np.zeros((16, 16), bool), occlusion=np.ones((16, 16), bool)) == []
    state = memory_state(memory.entries[identity], identity)
    assert state["visibility_status"] == "occluded" and state["last_seen_region_id"] == "000_000"
    assert memory.update([first.copy()], 1000000)[0]["shadow_track_id"] == identity
    assert len(memory.entries[identity]["appearance_history"]) == 2


def test_projected_shadow_geometry_is_an_alternative_for_fast_region_linking():
    memory = ShadowRegionMemory(.5)
    old = {"mask": np.zeros((32, 32), bool), "descriptor": None, "rank_score": .8}
    old["mask"][4:8, 4:8] = True
    identity = memory.update([old], 0)[0]["shadow_track_id"]
    moved = {**old, "mask": np.roll(old["mask"], 20, axis=1)}
    new = memory.update([moved], 500000, projection_masks={identity: moved["mask"]})
    assert new[0]["shadow_track_id"] == identity


def test_region_prediction_uses_caster_light_projection_not_material_flow():
    pairs, frames = memory_case()
    casters = CasterMemory(pairs, frames)
    mask = region()["mask"]
    entries = {1: {"mask": mask, "projection_hint": {"instance_token": "car", "light_id": 0, "frame_slot": 1}}}
    grid = BEVFlowConfig(extent_m=4).grid()
    predicted = project_region_memory(entries, casters, 2, [0, 0, 1, 0], light_bank(), bev_projection(np.eye(4), grid))
    np.testing.assert_array_equal(predicted[1], np.roll(mask, 4, axis=1))


def test_region_descriptors_retain_appearance_shape_and_material_correspondence():
    regions = [region()]
    lab = np.full((16, 16, 3), [60, 128, 128], np.float32)
    flow = np.zeros((16, 16, 2), np.float32)
    flow[..., 0] = .5
    describe_regions(regions, lab, np.ones((16, 16)), flow, np.ones((16, 16), bool))
    assert regions[0]["appearance_descriptor"].shape == (24,)
    assert regions[0]["boundary_descriptor"].shape == (3,)
    assert regions[0]["flow_correspondence"]["displacement_m"] == [.5, 0.]
