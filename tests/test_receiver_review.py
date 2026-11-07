import cv2
import numpy as np

from shadow_occ.annotation.physics_labels import cast_box
from shadow_occ.annotation.receiver_review import (
    exposure_review, light_projection_review, sheet, tile, transition_projection_review,
)
from tests.test_caster_window import transition_case
from tests.test_physics_shadow_labels import boxes, projection
from shadow_occ.annotation.caster_window import long_transition_evidence
from tests.test_receiver_explanations import script


def test_cast_shadow_is_not_the_vertical_box_footprint():
    corners = boxes()[1, 1:]
    ray = np.array([1., 0, -1.])
    cast = cast_box(corners, [0, 0, 1, 0], ray)
    np.testing.assert_allclose(cast[:, 0]-corners[:, 0], corners[:, 2])
    np.testing.assert_allclose(cast[:, 2], 0)
    assert cast[:, 0].max() > corners[:, 0].max()


def test_light_review_records_all_96_directions_without_a_best_gate():
    image = np.full((64, 96, 3), 100, np.uint8)
    observed = np.zeros((64, 96), bool)
    observed[28:35, 47:58] = True
    tracks = [{"instance_token": "car", "points": boxes()}]
    overlay, atlas, records = light_projection_review(image, observed, tracks, [0, 0, 1, 0], projection(), "car")
    assert overlay.shape == image.shape and atlas.shape == (2400, 2560, 3)
    assert len(records) == 96 and {item["light_id"] for item in records} == set(range(96))
    assert all(item["direction_world"][2] < 0 for item in records)
    assert all(item["status"] == "projected" for item in records)
    assert np.ptp(atlas) > 0


def test_exposure_review_records_candidate_pixels_used_as_background():
    observations, visible, flow, mask = transition_case()
    context = long_transition_evidence(observations, visible, flow, 2, np.zeros_like(mask))
    context.update(lightness=observations, middle=2, visible=visible)
    panels, details = exposure_review(context, 0, mask, np.zeros_like(mask))
    assert len(panels) == 3
    assert details["component_background_overlap_pixels"] == mask.sum()
    assert details["component_raw_delta_L_quantiles"][2] == -30
    assert details["component_normalized_delta_L_quantiles"] is not None
    assert len(details["background_pixel_coordinates_yx"]) == mask.size


def test_missing_complete_track_is_explicit_in_visual_review():
    panel, details = transition_projection_review({"tracks": {}}, "missing", [0, 0, 1, 0], projection(), np.zeros((8, 8), bool))
    assert panel is None and details["status"] == "no_complete_track"


def test_visual_sampling_counts_components_not_hypothesis_event_copies():
    module = script("audit_shadow_geometry")
    item = {"id": "component", "scene_name": "scene", "bev_flow": {"window_id": "window"},
        "transition_diagnostics": None, "exported_label": "UNCERTAIN",
        "hypotheses": [{"long_window": {"event_alignment": [{"reason": "outside_current_footprint"}]*96}}]}
    pools = module.case_pool([], [item])
    assert len(pools["outside_footprint"]) == 1
    assert len(module.sample_cases(pools, 20, 17)) == 1


def test_stage_accounting_includes_previously_omitted_exposure_state():
    module = script("analyze_receiver_competition")
    frames = [{"is_reference": False, "stage": stage, "exposure": {"reason": stage}}
              for stage in ("no_visible_overlap", "unstable_exposure_or_background", "unrecognized_reason")]
    result = module.transition_stage_summary([], [{"frames": frames, "long_window_available": True}])
    accounting = result["stage_accounting"]
    assert accounting["total"] == 3 and accounting["closed"]
    assert accounting["counts"]["unstable_exposure_or_background"] == 1
    assert accounting["counts"]["other"] == 1
    assert accounting["other_details"] == {"unrecognized_reason": 1}


def test_panel_layout_has_stable_dimensions(tmp_path):
    image = np.full((64, 96, 3), 120, np.uint8)
    panel = sheet([tile(image, "RGB")]*9, 3)
    assert panel.shape == (900, 1440, 3)
    assert cv2.imwrite(str(tmp_path/"panel.jpg"), panel)
