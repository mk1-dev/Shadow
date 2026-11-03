import numpy as np

from shadow_occ.annotation.offbev import boundary_hypotheses, contains, footprint_status, localization_metrics


def test_outside_requires_entire_footprint_not_only_box_center():
    bounds = [-40, -40, 40, 40]
    assert footprint_status([[41,-1,0],[43,-1,0],[43,1,0],[41,1,0]], bounds) == "outside"
    assert footprint_status([[39,-1,0],[43,-1,0],[43,1,0],[39,1,0]], bounds) == "crossing_boundary"


def test_shadow_axis_generates_external_uncertain_regions_without_gt():
    xy = np.c_[np.linspace(35, 39.8, 30), np.zeros(30)]
    bank = boundary_hypotheses(xy, [-40,-40,40,40])
    assert bank
    assert any(contains(candidate, [42, 0]) for candidate in bank)
    for candidate in bank:
        x0, y0, x1, y1 = candidate["xyxy_ego"]
        assert footprint_status([[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]], [-40,-40,40,40]) == "outside"


def test_no_association_is_not_evaluated_as_an_empty_area():
    result = localization_metrics([{"label_source": "unlabeled", "target_decision": None, "selected_candidate_id": "H1"}])
    assert result["by_association_source"]["unlabeled"]["labeled"] == 0
    assert result["by_association_source"]["unlabeled"]["abstention_accuracy"] is None
