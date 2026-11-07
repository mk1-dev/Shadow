from copy import deepcopy
import json
from pathlib import Path

import pytest

from shadow_occ.annotation.observability import (
    PROTOCOL, for_window, input_signature, parse_assessment, rgb_windows, stratum, vote_index,
)
from tests.test_bev_flow import example


def assessment(**updates):
    return {"illumination": "direct", "shadow_observability": "clear", "receiver_visibility": "sufficient",
        "temporal_usability": "stable", "object_shadow_relation": "plausible",
        "evidence": "Distinct projected boundaries on visible ground.", "uncertain": False, **updates}


def make_vote(row, value=None):
    value = assessment() if value is None else value
    result = {key: row[key] for key in ("id", "protocol", "scene_name", "official_split", "camera",
        "frame_tokens", "timestamps", "image_sha256", "input_signature")}
    return {**result, "assessment": value, "stratum": stratum(value), "parse_error": None,
            "inference_seconds": .1, "is_shadow_gt": False}


def test_stable_shadow_is_not_removed_and_weather_is_not_a_hard_gate():
    assert stratum(assessment(temporal_usability="stable")) == "PRIORITY"
    assert stratum(assessment(illumination="diffuse")) == "PRIORITY"
    assert stratum(assessment(object_shadow_relation="unclear")) == "PRIORITY"


def test_low_signal_and_unknown_are_strata_not_shadow_no():
    assert stratum(assessment(shadow_observability="not_observed")) == "LOW_SIGNAL"
    assert stratum(assessment(shadow_observability="weak")) == "LOW_SIGNAL"
    assert stratum(assessment(uncertain=True)) == "UNCERTAIN"
    assert stratum(assessment(temporal_usability="insufficient")) == "UNCERTAIN"
    assert stratum(None) == "UNCERTAIN"


@pytest.mark.parametrize("updates", [{"uncertain": "false"}, {"receiver_visibility": "no road GT"},
    {"evidence": " "}, {"shadow_observability": "NO"}, {"confidence": .95}])
def test_parser_rejects_invalid_or_extra_supervision_fields(updates):
    with pytest.raises(ValueError):
        parse_assessment(json.dumps(assessment(**updates)))


def test_clean_rgb_manifest_has_no_gt_or_geometry_inputs(tmp_path):
    pairs = deepcopy(example()[0])
    for pair in pairs:
        pair["objects"] = [{"instance_token": "GT_SECRET"}]
        pair["plane_world"] = None  # Observable RGB does not depend on a successful ground model.
        for frame in pair["frames"]:
            frame["image"] = f"clean_{frame['sample_data_token']}.png"
    rows = rgb_windows(pairs, tmp_path, "FIXED RGB PROMPT")
    assert len(rows) == 1 and rows[0]["prompt"] == "FIXED RGB PROMPT"
    assert len(rows[0]["images"]) == 3 and all("clean_" in path for path in rows[0]["images"])
    assert "GT_SECRET" not in json.dumps(rows[0])
    assert rows[0]["input_signature"] == input_signature(rows[0])
    assert rows[0]["protocol"] == PROTOCOL
    vote = make_vote(rows[0])
    assert for_window(vote_index([vote]), *pairs)["stratum"] == "PRIORITY"
    assert for_window({}, *pairs)["stratum"] == "UNRATED"
    vote["image_sha256"] = ["different"]*3
    with pytest.raises(ValueError, match="RGB window"):
        for_window(vote_index([vote]), *pairs)


def test_duplicate_or_inconsistent_votes_are_rejected(tmp_path):
    pairs = example()[0]
    for pair in pairs:
        for frame in pair["frames"]:
            frame["image"] = "clean.png"
    row = rgb_windows(pairs, tmp_path, "prompt")[0]
    vote = make_vote(row)
    with pytest.raises(ValueError, match="Duplicate"):
        vote_index([vote, vote])
    vote["stratum"] = "NO"
    with pytest.raises(ValueError, match="stratum"):
        vote_index([vote])


def test_audit_cases_are_deduplicated_without_feeding_categories_to_vlm(tmp_path, monkeypatch):
    import sys
    from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
    import numpy as np
    from tests.test_receiver_explanations import script

    source, audit, output = tmp_path/"source", tmp_path/"audit", tmp_path/"prepared"
    for split in ("train", "val"):
        pairs = deepcopy(example()[0])
        for pair in pairs:
            pair.update(id=split+pair["id"], scene_name="scene_"+split, official_split=split)
            for frame in pair["frames"]:
                image = tmp_path/f"{split}_{frame['sample_data_token']}.png"
                checked_write_image(image, np.full((64, 64, 3), 180, np.uint8))
                frame.update(image=str(image), image_sha256=digest(image))
        write_rows(source/split/"pairs/pairs.jsonl", pairs)
        key = rgb_windows(pairs, tmp_path, "prompt")[0]["id"]
        write_rows(audit/split/"cases.jsonl", [{"window_id": key, "id": "a", "category": "accepted"},
                                              {"window_id": key, "id": "b", "category": "flow_chain_zero"}])
    prompt = Path(__file__).parents[1]/"prompts/shadow_observability_v1.txt"
    monkeypatch.setattr(sys, "argv", ["prepare", "--source-root", str(source), "--audit-root", str(audit),
        "--prompt", str(prompt), "--output-root", str(output)])
    script("prepare_shadow_observability").main()
    rows = read_rows(output/"manifest.jsonl")
    assert len(rows) == 2 and len(read_rows(output/"audit_links.jsonl")) == 4
    assert all("category" not in row and "answer" not in row for row in rows)
    assert all(row["prompt"] == prompt.read_text(encoding="utf-8") for row in rows)
