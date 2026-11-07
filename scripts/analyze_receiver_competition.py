"""Compare receiver explanations and image/geometry/projection baselines."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.diagnostics import average_precision
from shadow_occ.annotation.metrics import auc
from shadow_occ.utils.io import write_json


def review_ids(items, limit, seed):
    groups = defaultdict(list)
    for item in items:
        groups[item["scene_name"]].append(item)
    rng = np.random.default_rng(seed)
    scenes = sorted(groups)
    rng.shuffle(scenes)
    for values in groups.values():
        rng.shuffle(values)
        values.sort(key=lambda item: not (
            item["shadow_decision"].get("baseline_projection_decision", item["shadow_decision"])["shadow_accepted"]
            and not (item.get("receiver_competition") or {}).get("shadow_supported", False)), reverse=True)
    selected = set()
    while len(selected) < limit:
        added = False
        for scene in scenes:
            if groups[scene] and len(selected) < limit:
                selected.add(groups[scene].pop()["id"])
                added = True
        if not added:
            break
    return selected


def rank_metrics(rows, reference, bootstrap=1000, seed=0):
    """Use supplied region labels only; never turn teacher acceptance into truth."""
    matched = [row for row in rows if row["id"] in reference]
    if not matched:
        raise ValueError("No reference IDs match this fixed component population")
    for row in matched:
        if reference[row["id"]]["scene_name"] != row["scene_name"]:
            raise ValueError("Reference scene differs from component scene")
    labels = np.array([reference[row["id"]]["shadow"] == "YES" for row in matched])
    by_scene = defaultdict(list)
    for index, row in enumerate(matched):
        by_scene[row["scene_name"]].append(index)
    clusters = [np.asarray(indices) for indices in by_scene.values()]
    rng = np.random.default_rng(seed)
    samples = [np.concatenate([clusters[index] for index in rng.integers(len(clusters), size=len(clusters))])
               for _ in range(bootstrap)] if len(clusters) > 1 else []
    result = {}
    for name in matched[0]["baseline_scores"]:
        available = sum(row["baseline_scores"].get(name) is not None for row in matched)
        if available != len(matched):
            result[name] = {"status": "missing_scores", "available": available,
                            "policy": "do not compare a smaller easier population"}
            continue
        scores = np.array([row["baseline_scores"][name] for row in matched])
        estimate = {"auroc": auc(labels, scores), "average_precision": average_precision(labels, scores)}
        intervals = defaultdict(list)
        for indices in samples:
            if len(np.unique(labels[indices])) < 2:
                continue
            intervals["auroc"].append(auc(labels[indices], scores[indices]))
            intervals["average_precision"].append(average_precision(labels[indices], scores[indices]))
        estimate["scene_bootstrap_ci95"] = {key: np.percentile(values, [2.5, 97.5]).tolist()
                                             for key, values in intervals.items()}
        result[name] = estimate
    return {"matched": len(matched), "unmatched_components": len(rows)-len(matched),
            "unmatched_reference": len(reference)-len(matched), "scenes": len(clusters),
            "positive_prevalence": float(labels.mean()), "scores": result}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-manifest", help="Optional fixed independent {id,scene_name,shadow:YES|NO} JSONL")
    parser.add_argument("--review-limit", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.review_limit, args.bootstrap) < 0:
        parser.error("Review and bootstrap counts must be nonnegative")
    root, output = Path(args.labels_root), Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    rows, winners, scenes, panels = [], Counter(), defaultdict(Counter), []
    items = read_rows(root/"decisions.jsonl")
    selected_reviews = review_ids(items, args.review_limit, args.seed)
    for item in items:
        evidence = item.get("receiver_competition")
        if evidence is None:
            raise ValueError("Run labels with --receiver-competition on or score-only")
        decision = item["shadow_decision"]
        baseline = decision.get("baseline_projection_decision", decision)
        row = {"id": item["id"], "scene_name": item["scene_name"], "camera": item["camera"],
               "image": item["image"], "mask": item["mask"], "winner": evidence["winner"],
               "scores": evidence["scores"], "winner_margin": evidence["winner_margin"],
               "baseline_scores": evidence["baseline_scores"],
               "mature_3d": evidence.get("mature_3d"),
               "motion_diagnostics": evidence.get("motion_diagnostics"),
               "mature_image_motion": (item.get("bev_flow") or {}).get("mature_image_motion"),
               "projection_control_gain": evidence["projection_control_gain"],
               "trajectory_control_gain": evidence["trajectory_control_gain"],
               "baseline_accepted": baseline["shadow_accepted"],
               "competition_supported": evidence["shadow_supported"],
               "exported_positive": item["exported_label"] == "YES"}
        rows.append(row)
        winners[row["winner"]] += 1
        scenes[row["scene_name"]].update({"components": 1, "baseline_accepted": int(row["baseline_accepted"]),
            "competition_supported": int(row["competition_supported"]), "exported_positive": int(row["exported_positive"])})
        if item["id"] in selected_reviews:
            image = cv2.imread(row["image"])
            mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise FileNotFoundError(row["id"])
            yy, xx = np.where(mask > 0)
            if not len(xx):
                continue
            margin = max(24, int(max(np.ptp(xx), np.ptp(yy))*.75))
            x0, x1 = max(0, xx.min()-margin), min(image.shape[1], xx.max()+margin+1)
            y0, y1 = max(0, yy.min()-margin), min(image.shape[0], yy.max()+margin+1)
            crop = image[y0:y1, x0:x1].copy()
            contours, _ = cv2.findContours(mask[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean = crop.copy()
            cv2.drawContours(crop, contours, -1, (0, 220, 220), 1)
            tile = np.zeros((288, 640, 3), np.uint8)
            for index, value in enumerate((clean, crop)):
                scale = min(320/value.shape[1], 240/value.shape[0])
                value = cv2.resize(value, (max(1, round(value.shape[1]*scale)), max(1, round(value.shape[0]*scale))))
                tile[48:48+value.shape[0], index*320:index*320+value.shape[1]] = value
            cv2.putText(tile, f"{row['scene_name']} {row['winner']} base={int(row['baseline_accepted'])}",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            cv2.putText(tile, f"control_gain={row['projection_control_gain']:.3f} margin={row['winner_margin']:.3f}",
                        (5, 37), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            checked_write_image(output/"review"/f"{row['id']}.jpg", tile)
            panels.append(tile)
    if not rows:
        raise ValueError("No components to compare")
    reference_metrics = None
    if args.reference_manifest:
        references = read_rows(args.reference_manifest)
        if (len({row["id"] for row in references}) != len(references)
                or any(row.get("shadow") not in {"YES", "NO"} for row in references)):
            raise ValueError("Reference needs unique IDs and explicit YES/NO labels")
        reference_metrics = rank_metrics(rows, {row["id"]: row for row in references}, args.bootstrap, args.seed)
    write_rows(output/"components.jsonl", rows)
    if panels:
        per_row = 2
        panels += [np.zeros_like(panels[0]) for _ in range((-len(panels)) % per_row)]
        checked_write_image(output/"contact_sheet.jpg", np.vstack([
            np.hstack(panels[index:index+per_row]) for index in range(0, len(panels), per_row)]))
    write_json(output/"summary.json", {"components": len(rows), "winner_counts": dict(winners),
        "mature_3d_components": sum(row["mature_3d"] is not None for row in rows),
        "mature_motion_components": sum((row["mature_3d"] or {}).get("motion_observed_fraction", 0) > 0 for row in rows),
        "baseline_accepted": sum(row["baseline_accepted"] for row in rows),
        "competition_supported": sum(row["competition_supported"] for row in rows),
        "exported_positive": sum(row["exported_positive"] for row in rows),
        "scenes": {key: dict(value) for key, value in scenes.items()}, "reference_metrics": reference_metrics,
        "reference_sha256": digest(args.reference_manifest) if args.reference_manifest else None,
        "baseline_scope": "projection acceptance before positive fusion; exported_positive includes fusion and camera support",
        "scope": "Receiver explanations and baseline ranking; no precision claim without independent reference"})


if __name__ == "__main__":
    main()
