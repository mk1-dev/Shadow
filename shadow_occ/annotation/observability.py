"""RGB-only observation priority, separate from shadow truth and physical gates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json


PROTOCOL = "shadow-observability-v1"
FIELDS = {"illumination": {"direct", "diffuse", "mixed", "unknown"},
    "shadow_observability": {"clear", "weak", "not_observed", "unknown"},
    "receiver_visibility": {"sufficient", "limited", "unknown"},
    "temporal_usability": {"changing", "stable", "insufficient"},
    "object_shadow_relation": {"plausible", "unclear", "not_observed"}}


def window_id(left, right):
    return hashlib.sha256(f"{left['id']}:{right['id']}".encode()).hexdigest()[:24]


def parse_assessment(text):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != set(FIELDS) | {"evidence", "uncertain"}:
        raise ValueError("Observability needs exactly the seven protocol fields")
    for field, options in FIELDS.items():
        if not isinstance(value[field], str) or value[field] not in options:
            raise ValueError(f"Invalid observability field: {field}")
    if not isinstance(value["evidence"], str) or not value["evidence"].strip():
        raise ValueError("Observability evidence must be a nonempty string")
    if type(value["uncertain"]) is not bool:
        raise ValueError("uncertain must be a JSON boolean")
    return value


def stratum(assessment):
    if assessment is None:
        return "UNCERTAIN"
    if (assessment["uncertain"] or assessment["shadow_observability"] == "unknown"
            or assessment["receiver_visibility"] == "unknown"
            or assessment["temporal_usability"] == "insufficient"):
        return "UNCERTAIN"
    if assessment["shadow_observability"] == "clear" and assessment["receiver_visibility"] == "sufficient":
        return "PRIORITY"
    return "LOW_SIGNAL"


def input_signature(row):
    values = {key: row[key] for key in ("scene_name", "camera", "frame_tokens", "timestamps",
                                       "image_sha256", "prompt")}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rgb_windows(pairs, base, prompt):
    from shadow_occ.annotation.data import resolve_path

    successor = {}
    for pair in pairs:
        key = (pair["scene_name"], pair["official_split"], pair["camera"], pair["sample_token"])
        if key in successor:
            raise ValueError("Duplicate scene/camera/sample pair")
        successor[key] = pair
    rows = []
    for left in pairs:
        right = successor.get((left["scene_name"], left["official_split"], left["camera"], left["next_sample_token"]))
        if right is None:
            continue
        shared, middle = left["frames"][1], right["frames"][0]
        for key in ("sample_data_token", "image_sha256", "camera_timestamp", "image_size_wh"):
            if shared[key] != middle[key]:
                raise ValueError(f"RGB time-context mismatch: {key}")
        frames = [left["frames"][0], middle, right["frames"][1]]
        timestamps = [frame["camera_timestamp"] for frame in frames]
        if not timestamps[0] < timestamps[1] < timestamps[2]:
            raise ValueError("RGB windows need strictly increasing timestamps")
        row = {"id": window_id(left, right), "protocol": PROTOCOL, "task": "scene_observability",
            "scene_name": left["scene_name"], "official_split": left["official_split"], "camera": left["camera"],
            "pair_ids": [left["id"], right["id"]], "frame_tokens": [frame["sample_data_token"] for frame in frames],
            "timestamps": timestamps, "time_segment_us": [timestamps[0], timestamps[-1]],
            "images": [str(resolve_path(frame["image"], base)) for frame in frames],
            "image_sha256": [frame["image_sha256"] for frame in frames], "prompt": prompt,
            "input_regime": "clean_rgb_temporal_no_gt_or_overlay", "is_shadow_gt": False}
        row["input_signature"] = input_signature(row)
        rows.append(row)
    return rows


def vote_index(votes):
    result = {}
    for vote in votes:
        if vote["id"] in result:
            raise ValueError("Duplicate observability vote ID")
        if vote.get("protocol") != PROTOCOL:
            raise ValueError("Unsupported observability protocol")
        assessment = vote.get("assessment")
        if assessment is not None:
            assessment = parse_assessment(json.dumps(assessment))
        if vote["stratum"] != stratum(assessment):
            raise ValueError("Observability stratum differs from the frozen policy")
        result[vote["id"]] = vote
    return result


def for_window(index, left, right):
    key = window_id(left, right)
    context = {"window_id": key, "scene_name": left["scene_name"], "camera": left["camera"],
        "time_segment_us": [left["frames"][0]["camera_timestamp"], right["frames"][1]["camera_timestamp"]]}
    vote = index.get(key)
    if vote is None:
        return {**context, "stratum": "UNRATED", "assessment": None, "status": "not_tagged"}
    frames = [left["frames"][0], right["frames"][0], right["frames"][1]]
    if (vote["scene_name"] != left["scene_name"] or vote["camera"] != left["camera"]
            or vote["official_split"] != left["official_split"]
            or vote["image_sha256"] != [frame["image_sha256"] for frame in frames]
            or vote["frame_tokens"] != [frame["sample_data_token"] for frame in frames]
            or vote["timestamps"] != [frame["camera_timestamp"] for frame in frames]):
        raise ValueError("Observability vote does not match this RGB window")
    return {**context, "stratum": vote["stratum"], "assessment": vote.get("assessment"),
        "status": "parse_error" if vote.get("parse_error") else "tagged",
        "input_signature": vote["input_signature"], "is_shadow_gt": False}


def summarize_votes(votes):
    return {"windows": len(votes), "strata": dict(Counter(vote["stratum"] for vote in votes)),
        "parse_errors": sum(vote.get("parse_error") is not None for vote in votes),
        "dimensions": {field: dict(Counter(vote["assessment"][field] for vote in votes if vote.get("assessment")))
                       for field in FIELDS},
        "scope": "observation priority only; stable is retained, not_observed is not a negative label"}
