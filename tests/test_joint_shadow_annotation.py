import json

import pytest

from shadow_occ.annotation.joint import annotation, parse, shadow_confusion, summarize
from shadow_occ.annotation.vlm import shadow_decision_weights


def test_one_annotation_includes_shadow_and_exterior_caster():
    value = annotation("YES", "outside_bev", candidate_id="H1")
    assert parse(json.dumps(value), [{"id": "H1"}])["caster"]["state"] == "outside_bev"
    with pytest.raises(ValueError, match="candidate"):
        parse(json.dumps(value), [])


def test_no_shadow_cannot_carry_an_associated_caster():
    with pytest.raises(ValueError, match="NO region"):
        annotation("NO", "outside_bev", candidate_id="H1")


def test_unlabeled_annotation_is_not_scored_as_negative_gt():
    summary = summarize([{"target": None, "prediction": annotation("YES", "unresolved"), "shadow_label_source": "unlabeled", "association_source": "unresolved"}])
    assert summary["by_shadow_source"] == {}


def test_shadow_decision_weights_balance_positive_and_negative_examples():
    rows = [{"answer": json.dumps(annotation("NO", "none"))} for _ in range(9)] + [{"answer": json.dumps(annotation("YES"))}]
    weights = shadow_decision_weights(rows)
    assert weights == {"NO": 0.5555555555555556, "YES": 5.0}


def test_shadow_confusion_exposes_majority_class_collapse():
    rows = [{"target": annotation("YES"), "prediction": annotation("NO", "none")} for _ in range(2)]
    rows += [{"target": annotation("NO", "none"), "prediction": annotation("NO", "none")} for _ in range(8)]
    result = shadow_confusion(rows)
    assert result["positive_recall"] == 0.0
    assert result["negative_specificity"] == 1.0
    assert result["balanced_accuracy"] == 0.5
