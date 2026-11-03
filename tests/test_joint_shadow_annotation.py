import json

import pytest

from shadow_occ.annotation.joint import annotation, parse, summarize


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
