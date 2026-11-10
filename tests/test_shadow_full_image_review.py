import numpy as np
import pytest

from shadow_occ.annotation.full_image_review import prediction_coverage, review_panel, summarize_coverage


def test_ignore_predictions_are_counted_without_becoming_accuracy():
    target = np.array([[1, 0], [255, 255]], np.uint8)
    probability = np.array([[.9, .8], [.9, .1]])
    result = prediction_coverage(probability, target)
    assert result["ignore_predicted_shadow_fraction"] == .5
    assert result["full_image_predicted_shadow_fraction"] == .75
    assert not result["ignore_prediction_is_accuracy"]
    panel = review_panel(np.zeros((2, 2, 3), np.uint8), probability, target, width=80)
    assert panel.shape == (108, 240, 3)


def test_empty_known_region_and_scene_macro_have_explicit_denominators():
    a = prediction_coverage(np.ones((2, 2)), np.full((2, 2), 255))
    b = prediction_coverage(np.zeros((10, 10)), np.zeros((10, 10)))
    assert a["known_predicted_shadow_fraction"] is None
    result = summarize_coverage([{"scene_name": "a", "full_image_coverage": a},
                                 {"scene_name": "b", "full_image_coverage": b}])
    assert result["scene_macro_predicted_shadow_fraction"] == .5
    assert result["by_scene"]["b"]["ignore_predicted_shadow_fraction"] is None


def test_bad_probabilities_do_not_enter_full_image_reports():
    with pytest.raises(ValueError, match="finite"):
        prediction_coverage(np.array([[np.nan]]), np.array([[255]]))
