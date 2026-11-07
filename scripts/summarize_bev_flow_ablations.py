"""Report fixed-window coverage and optional independent-reference pixel quality."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, resolve_path
from shadow_occ.utils.io import write_json


def ratio(a, b):
    return float(a/b) if b else None


def reference_metrics(counts):
    tp, fp, tn, fn, assigned, known, positives = map(int, counts)
    return {"positive_precision": ratio(tp, tp+fp), "negative_precision": ratio(tn, tn+fn),
        "known_pixel_coverage": ratio(assigned, known), "positive_reference_capture": ratio(tp, positives),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "reference_known_pixels": known}


def identity(row, base):
    if all(row.get(key) is not None for key in ("scene_name", "camera", "sample_token")):
        return tuple(row[key] for key in ("scene_name", "camera", "sample_token"))
    return (digest(resolve_path(row["image"], base)),)


def evaluate_reference(canonical, reference, reference_base, bootstrap, seed):
    predictions = {identity(row, Path(".")): row for row in canonical}
    if len(predictions) != len(canonical):
        raise ValueError("Duplicate predicted physical frames")
    by_scene = defaultdict(lambda: np.zeros(7, np.int64))
    seen = set()
    matched = 0
    for row in reference:
        key = identity(row, reference_base)
        if key in seen:
            raise ValueError("Reference pool must contain unique physical frames")
        seen.add(key)
        target = cv2.imread(str(resolve_path(row["target"], reference_base)), cv2.IMREAD_GRAYSCALE)
        if target is None or not np.isin(target, [0, 1, 255]).all():
            raise ValueError("Reference targets must be 0/1/255 trimaps")
        predicted = np.full(target.shape, 255, np.uint8)
        if key in predictions:
            predicted = cv2.imread(predictions[key]["shadow_mask"], cv2.IMREAD_GRAYSCALE)
            if predicted is None or predicted.shape != target.shape:
                raise ValueError("Reference and teacher image coordinates differ")
            matched += 1
        valid = target != 255
        positive, negative = (predicted == 1) & valid, (predicted == 0) & valid
        counts = np.array([(positive & (target == 1)).sum(), (positive & (target == 0)).sum(),
            (negative & (target == 0)).sum(), (negative & (target == 1)).sum(),
            ((positive | negative) & valid).sum(), valid.sum(), (target == 1).sum()], np.int64)
        by_scene[row["scene_name"]] += counts
    totals = sum(by_scene.values(), np.zeros(7, np.int64))
    result = {**reference_metrics(totals), "reference_frames": len(reference), "matched_frames": matched,
        "reference_scenes": len(by_scene), "missing_prediction_policy": "IGNORE; zero coverage, not dropped"}
    if bootstrap and by_scene:
        values = np.stack(list(by_scene.values()))
        rng, samples = np.random.default_rng(seed), defaultdict(list)
        for _ in range(bootstrap):
            metrics = reference_metrics(values[rng.integers(len(values), size=len(values))].sum(axis=0))
            for key in ("positive_precision", "negative_precision", "known_pixel_coverage", "positive_reference_capture"):
                if metrics[key] is not None:
                    samples[key].append(metrics[key])
        result["scene_bootstrap_ci95"] = {key: np.percentile(value, [2.5, 97.5]).tolist() for key, value in samples.items()}
    return result


def paired_discovery(root, left, right, split):
    """Compare masks on identical cached windows, not independently surviving regions."""
    folders = [Path(root)/name/split/"bev_flow" for name in (left, right)]
    if not all((folder/"windows.jsonl").exists() for folder in folders):
        return None
    rows = [{row["id"]: row for row in read_rows(folder/"windows.jsonl")} for folder in folders]
    if rows[0].keys() != rows[1].keys():
        raise ValueError("Paired discovery requires identical windows")
    counts = {key: {"left_only": 0, "right_only": 0, "intersection": 0} for key in
              ("discrepancy_candidate", "shadow_candidate")}
    coverage = {name: {"available_windows": 0, "observed_pixels": 0, "explained_pixels": 0} for name in ("left", "right")}
    for window_id in rows[0]:
        a, b = [values[window_id] for values in rows]
        if a["grid"] != b["grid"]:
            raise ValueError("Paired discovery grids differ")
        with np.load(resolve_path(a["evidence"], folders[0]), allow_pickle=False) as first, np.load(
                resolve_path(b["evidence"], folders[1]), allow_pickle=False) as second:
            if not np.array_equal(first["receiver_valid"], second["receiver_valid"]):
                raise ValueError("Paired discovery receiver geometry differs")
            for name, archive in (("left", first), ("right", second)):
                if "discrepancy_entity_explanation_valid" in archive:
                    coverage[name]["available_windows"] += 1
                    coverage[name]["observed_pixels"] += int(archive["discrepancy_entity_explanation_valid"].any(axis=0).sum())
                    coverage[name]["explained_pixels"] += int((archive["discrepancy_entity_explanation"] > 0).any(axis=0).sum())
            for key in counts:
                x, y = first[key].astype(bool), second[key].astype(bool)
                counts[key]["left_only"] += int((x & ~y).sum())
                counts[key]["right_only"] += int((y & ~x).sum())
                counts[key]["intersection"] += int((x & y).sum())
    return {"left": left, "right": right, "windows": len(rows[0]), "pixels": counts, "entity_motion_coverage": coverage,
        "scope": "fixed-window branch and appearance/discrepancy union before caster expansion; not shadow correctness"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-manifest", help="Fixed independent image/target trimaps, same coordinates as exported images")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    reference = read_rows(args.reference_manifest) if args.reference_manifest else None
    results, expected_windows, expected_inputs = [], {}, {}
    for protocol_path in sorted(Path(args.root).glob("*/ablation_protocol.json")):
        root = protocol_path.parent
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        for split in ("train", "val"):
            folder = root/split
            bev_protocol_path = folder/"bev_flow/bev_flow_protocol.json"
            if not bev_protocol_path.exists():
                results.append({"variant": root.name, "split": split, "status": "incomplete"})
                continue
            bev_protocol = json.loads(bev_protocol_path.read_text(encoding="utf-8"))
            inputs = bev_protocol["pair_manifest_sha256"]
            windows = read_rows(folder/"bev_flow/windows.jsonl")
            ids = {row["id"] for row in windows}
            if len(ids) != len(windows) or ids != expected_windows.setdefault(split, ids) or inputs != expected_inputs.setdefault(split, inputs):
                raise ValueError("Ablations must share the exact source inputs and window population")
            counts = sum((int(row["counts"]["photometric_candidates"]) for row in windows))
            selected = sum(int(row["counts"]["shadow_candidates"]) for row in windows)
            summary_path = folder/"labels/summary.json"
            labels = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            teacher_path = folder/"teacher/protocol.json"
            teacher = json.loads(teacher_path.read_text(encoding="utf-8")) if teacher_path.exists() else {}
            competition_path = folder/"competition/summary.json"
            competition = json.loads(competition_path.read_text(encoding="utf-8")) if competition_path.exists() else {}
            decision_path = folder/"labels/decisions.jsonl"
            decisions = read_rows(decision_path) if decision_path.exists() else []
            positive = [item for item in decisions if item["exported_label"] == "YES"]
            result = {"variant": root.name, "split": split, "recipe": protocol["recipe"],
                "windows": len(windows), "photometric_cells": counts, "selected_cells": selected,
                "analysis_protocol_sha256": bev_protocol["analysis_protocol_sha256"],
                "candidate_retention": ratio(selected, counts), "label_stages": labels.get("stage_counts"),
                "receiver_competition": labels.get("receiver_competition"),
                "teacher_counts": teacher.get("counts"), "reference_quality": None}
            result.update(receiver_cells=sum(row["counts"]["receiver"] for row in windows),
                discrepancy_proposals=competition.get("discrepancy_proposals"),
                label_counts=labels.get("counts"),
                exported_unique_caster=sum(item["shadow_decision"]["association"] is not None for item in positive),
                exported_yes_unresolved=sum(item["shadow_decision"]["association"] is None for item in positive),
                positive_window_coverage=ratio(len({(item["bev_flow"] or {})["window_id"] for item in positive}), len(windows)),
                candidate_budget_policy="whole over-budget windows skipped; no silent top-k selection")
            if reference is not None:
                official = [row for row in reference if row["official_split"] == split]
                canonical_path = folder/"teacher/canonical.jsonl"
                canonical = read_rows(canonical_path) if canonical_path.exists() else []
                result["reference_quality"] = evaluate_reference(canonical, official,
                    Path(args.reference_manifest).resolve().parent, args.bootstrap, args.seed)
            results.append(result)
    if not results:
        raise ValueError("No ablation protocols found")
    controls = {split: {
        "entity_motion": paired_discovery(args.root, "C_discrepancy", "D_no_entity_motion", split),
        "gt_time": paired_discovery(args.root, "C_discrepancy", "C_gt_reverse", split),
        "bev_shuffle": paired_discovery(args.root, "C_discrepancy", "C_bev_shuffled", split)} for split in ("train", "val")}
    write_json(Path(args.output_root)/"summary.json", {"variants": results, "paired_controls": controls,
        "reference_sha256": digest(args.reference_manifest) if args.reference_manifest else None,
        "quality_scope": "agreement with supplied independent reference" if reference is not None else "coverage only; precision not measured"})
    lines = ["# BEV-Flow Ablations", "", "| Variant | Split | Windows | Selected cells | Retention | Accepted regions |", "|---|---|---:|---:|---:|---:|"]
    for row in results:
        if "windows" in row:
            rate = row["candidate_retention"]
            lines.append(f"| {row['variant']} | {row['split']} | {row['windows']} | {row['selected_cells']} | {rate if rate is not None else 'NA'} | {(row['label_stages'] or {}).get('accepted_shadow', 'NA')} |")
    lines += ["", "All arms share source windows. Retention is not precision. Reference quality and scene-bootstrap intervals are in summary.json."]
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Wrote {output/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
