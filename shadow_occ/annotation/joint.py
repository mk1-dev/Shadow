from __future__ import annotations

import json

import numpy as np


def annotation(shadow, state="unresolved", *, candidate_id=None, category="unknown", bbox=None):
    value = {"shadow": shadow, "caster": {"state": state, "category": category, "bbox_1000_context": bbox, "outside_candidate_id": candidate_id}}
    validate(value)
    return value


def validate(value, candidates=None):
    if not isinstance(value, dict) or value.get("shadow") not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError("shadow must be YES, NO or UNCERTAIN")
    caster = value.get("caster")
    if not isinstance(caster, dict) or caster.get("state") not in {"visible", "outside_bev", "unresolved", "none"}:
        raise ValueError("Invalid caster association")
    if not isinstance(caster.get("category"), str):
        raise ValueError("caster.category must be a string")
    if value["shadow"] == "NO" and caster["state"] != "none":
        raise ValueError("A NO region cannot be associated with a caster")
    if caster["state"] == "visible":
        box = caster.get("bbox_1000_context")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in box):
            raise ValueError("Visible caster needs a numeric context box")
        if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            raise ValueError("Invalid normalized caster box")
    elif caster.get("bbox_1000_context") is not None:
        raise ValueError("Only a visible caster has an image box")
    if caster["state"] == "outside_bev":
        if not isinstance(caster.get("outside_candidate_id"), str):
            raise ValueError("Outside association requires a candidate ID")
        if candidates is not None and caster["outside_candidate_id"] not in {item["id"] for item in candidates}:
            raise ValueError("Outside candidate is not present in the observation")
    elif caster.get("outside_candidate_id") is not None:
        raise ValueError("Unexpected outside candidate")
    return value


def parse(text, candidates=None):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return validate(json.loads(text), candidates)


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return float(intersection/max(area_a+area_b-intersection, 1e-12))


def shadow_confusion(rows):
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0, "unclassified": 0}
    for row in rows:
        target = row.get("target")
        prediction = row.get("prediction")
        target_decision = (target or {}).get("shadow")
        predicted_decision = (prediction or {}).get("shadow")
        if target_decision not in {"YES", "NO"}:
            continue
        if predicted_decision not in {"YES", "NO"}:
            counts["unclassified"] += 1
        elif target_decision == predicted_decision == "YES":
            counts["true_positive"] += 1
        elif target_decision == "NO" and predicted_decision == "YES":
            counts["false_positive"] += 1
        elif target_decision == "YES" and predicted_decision == "NO":
            counts["false_negative"] += 1
        else:
            counts["true_negative"] += 1
    positive = counts["true_positive"] + counts["false_negative"]
    negative = counts["true_negative"] + counts["false_positive"]
    counts["positive_recall"] = counts["true_positive"]/positive if positive else None
    counts["negative_specificity"] = counts["true_negative"]/negative if negative else None
    if positive and negative:
        counts["balanced_accuracy"] = (counts["positive_recall"]+counts["negative_specificity"])/2
    else:
        counts["balanced_accuracy"] = None
    return counts


def summarize(rows):
    result = {"examples": len(rows), "parse_errors": sum(row["prediction"] is None for row in rows), "shadow_confusion": shadow_confusion(rows), "by_shadow_source": {}, "by_association_source": {}}
    for source in sorted({row["shadow_label_source"] for row in rows}):
        selected = [row for row in rows if row["shadow_label_source"] == source and row.get("target") is not None]
        if not selected:
            continue
        correct = sum(row["prediction"] is not None and row["prediction"]["shadow"] == row["target"]["shadow"] for row in selected)
        result["by_shadow_source"][source] = {"labeled": len(selected), "shadow_decision_agreement": correct/len(selected), "shadow_confusion": shadow_confusion(selected)}
    for source in sorted({row["association_source"] for row in rows}):
        visible = [row for row in rows if row["association_source"] == source and (row.get("target") or {}).get("caster", {}).get("state") == "visible"]
        outside = [row for row in rows if row["association_source"] == source and (row.get("known_outside_caster") or (row.get("target") or {}).get("caster", {}).get("state") == "outside_bev")]
        visible_hits, outside_hits = 0, 0
        for row in visible:
            prediction = row["prediction"]
            visible_hits += int(prediction is not None and prediction["shadow"] == "YES" and prediction["caster"]["state"] == "visible" and bbox_iou(prediction["caster"].get("bbox_1000_context"), row["target"]["caster"]["bbox_1000_context"]) >= 0.5)
        for row in outside:
            prediction = row["prediction"]
            outside_hits += int(prediction is not None and prediction["shadow"] == "YES" and prediction["caster"]["state"] == "outside_bev" and prediction["caster"].get("outside_candidate_id") in row.get("valid_candidate_ids", []))
        result["by_association_source"][source] = {"visible_associations": len(visible), "visible_box_recall_iou50": visible_hits/len(visible) if visible else None, "outside_associations": len(outside), "outside_candidate_bank_recall": sum(bool(row.get("valid_candidate_ids")) for row in outside)/len(outside) if outside else None, "outside_region_recall": outside_hits/len(outside) if outside else None}
    result["interpretation"] = "One GT-constrained shadow annotation model; object boxes, weak shadow masks and association GT retain separate provenance"
    return result
