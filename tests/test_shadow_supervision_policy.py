import numpy as np

from shadow_occ.annotation.joint import annotation, summarize
from shadow_occ.annotation.refinement import refine_target
from shadow_occ.annotation.supervision import competing_association, exterior_target, permute_candidates, visible_target


def visible_row():
    return {"annotation_token": "car-a", "caster_source": "annotation_gt", "caster_category": "vehicle.car",
            "caster_bbox_xyxy": [10, 20, 30, 40], "context_crop_xyxy": [0, 0, 100, 100]}


def test_object_gt_alone_never_establishes_a_shadow_association():
    assert visible_target(visible_row())["caster"]["state"] == "unresolved"


def test_supported_weak_visible_association_keeps_exact_identity():
    row = {**visible_row(), "association_source": "geometry_weak", "association_status": "weak_geometry_supported", "associated_annotation_token": "car-a"}
    assert visible_target(row)["caster"]["bbox_1000_context"] == [100, 200, 300, 400]
    row["associated_annotation_token"] = "car-b"
    assert visible_target(row)["caster"]["state"] == "unresolved"


def test_two_nearby_footprints_remain_ambiguous():
    xy = [[0, 0], [1, 0], [1, 3], [0, 3]]
    casters = [{"annotation_token": name, "caster_corners_ego": [[x, 0, 0], [x+1, 0, 0], [x+1, 1, 0], [x, 1, 0]]} for name, x in (("a", -1), ("b", 1))]
    result = competing_association(xy, casters)
    assert result["association_status"] == "ambiguous"
    assert result["associated_annotation_token"] is None


def test_unique_object_or_even_association_gt_is_not_shadow_gt():
    row = {"answer": "present", "selected_candidate_id": "H1", "association_source": "manual_caster_association_gt"}
    assert exterior_target(row) == annotation("UNCERTAIN", "unresolved")
    row.update(shadow_label="YES", shadow_label_source="geometry_weak")
    assert exterior_target(row) == annotation("YES", "outside_bev", candidate_id="H1")
    row.update(shadow_label="NO")
    assert exterior_target(row) == annotation("NO", "none")


def test_candidate_permutation_is_reproducible_and_bijective():
    bank = [{"id": f"H{i}", "geometry_score": i} for i in range(1, 7)]
    shuffled, mapping = permute_candidates(bank, "sample", 3)
    assert (shuffled, mapping) == permute_candidates(bank, "sample", 3)
    assert len(set(mapping.values())) == len(bank)
    assert {item["geometry_score"] for item in shuffled} == set(range(1, 7))
    assert bank[0]["id"] == "H1"


def test_unknown_can_expand_only_after_a_yes_review():
    probability = np.full((3, 3), 0.9)
    original = np.full((3, 3), 255, np.uint8)
    mask = np.ones(original.shape, bool)
    without, _ = refine_target(probability, original, [])
    assert np.all(without == 255)
    with_yes, report = refine_target(probability, original, [(mask, "YES")])
    assert np.all(with_yes == 1)
    assert report["new_positive_pixels"] == 9


def test_no_and_parse_uncertainty_never_create_negative_pixels():
    original = np.ones((3, 3), np.uint8)
    mask = np.ones(original.shape, bool)
    for decision in ("NO", "UNCERTAIN"):
        target, _ = refine_target(np.full(original.shape, 0.9), original, [(mask, decision)])
        assert np.all(target == 255)


def test_conflicts_and_known_negatives_are_not_silently_rewritten():
    probability = np.full((3, 3), 0.9)
    mask = np.ones(probability.shape, bool)
    conflict, _ = refine_target(probability, np.ones(probability.shape, np.uint8), [(mask, "YES"), (mask, "NO")])
    assert np.all(conflict == 255)
    negative, _ = refine_target(probability, np.zeros(probability.shape, np.uint8), [(mask, "YES")])
    assert np.all(negative == 255)
    ground_truth, _ = refine_target(probability, np.zeros(probability.shape, np.uint8), [(mask, "YES")], preserve_gt=True)
    assert np.all(ground_truth == 0)


def test_unreviewed_existing_positive_can_survive_and_false_positives_are_reported():
    target, _ = refine_target(np.full((2, 2), 0.9), np.ones((2, 2), np.uint8), [])
    assert np.all(target == 1)
    summary = summarize([{"target": annotation("NO", "none"), "prediction": annotation("YES", "unresolved"),
                          "shadow_label_source": "geometry_weak", "association_source": "unresolved"}])
    assert summary["by_shadow_source"]["geometry_weak"]["negative_false_positive_rate"] == 1
