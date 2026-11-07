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
from shadow_occ.annotation.caster_window import receiver_sampling_summary, transition_stage_accounting
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


def projection_transition_diagnostics(hypotheses):
    explanations = [item for parent in hypotheses for item in [parent, *parent.get("alternatives", [])]]
    long_fits = [item["long_window"] for item in explanations if item.get("long_window") is not None]
    alignment = [event for fit in long_fits for event in fit.get("event_alignment", [])]
    return {"tested_hypotheses": len(explanations), "long_fits": len(long_fits),
        "long_fits_with_events": sum(fit["observed_transition_types"] > 0 for fit in long_fits),
        "long_fits_with_matches": sum(fit["edge_fit"] > 0 for fit in long_fits),
        "maximum_edge_fit": max((fit["edge_fit"] for fit in long_fits), default=None),
        "event_alignment_available": any("event_alignment" in fit for fit in long_fits),
        "event_alignment_reasons": dict(Counter(event["reason"] for event in alignment)),
        "scope": "all tested caster/light hypotheses, including those failing projection acceptance; not independent events"}


def transition_stage_summary(rows, windows):
    components = [row["transition_diagnostics"] for row in rows if row.get("transition_diagnostics") is not None]
    long_components = [item["long_window"] for item in components if item.get("long_window") is not None]
    # Window records are unique even when many components share the same exposure fit.
    frames = [frame for window in windows for frame in window["frames"] if not frame["is_reference"]]
    long_frames = [frame for item in long_components for frame in item["frames"] if not frame.get("is_reference", False)]
    alignments = Counter()
    for row in rows:
        alignments.update(row["projection_transition_diagnostics"]["event_alignment_reasons"])
    return {"windows_with_diagnostics": len(windows),
        "stage_accounting": transition_stage_accounting(frames),
        "receiver_sampling": receiver_sampling_summary([
            frame["sampling"] for window in windows for frame in window["frames"] if "sampling" in frame]),
        "windows_with_long_observations": sum(window["long_window_available"] for window in windows),
        "nonreference_frame_comparisons": len(frames),
        "frame_stage_counts": dict(Counter(frame["stage"] for frame in frames)),
        "exposure_reason_counts": dict(Counter(frame["exposure"]["reason"] for frame in frames)),
        "components_with_diagnostics": len(components), "components_with_long_observations": len(long_components),
        "central_event_components": sum(item["central_onset_pixels"]+item["central_recovery_pixels"] > 0 for item in components),
        "raw_long_darkening_components": sum(any(frame["raw_darkening_pixels"] > 0 for frame in item["frames"])
                                              for item in long_components),
        "normalized_long_darkening_components": sum(any((frame["normalized_darkening_pixels"] or 0) > 0
                                                        for frame in item["frames"]) for item in long_components),
        "observed_long_event_components": sum(item["event_pixels"] > 0 for item in long_components),
        "component_frame_exposure_unavailable": sum(frame["normalized_darkening_pixels"] is None for frame in long_frames),
        "any_hypothesis_long_match_components": sum(row["projection_transition_diagnostics"]["long_fits_with_matches"] > 0
                                                     for row in rows),
        "components_without_tested_hypotheses": sum(row["projection_transition_diagnostics"]["tested_hypotheses"] == 0
                                                     for row in rows),
        "components_with_alignment_diagnostics": sum(row["projection_transition_diagnostics"]["event_alignment_available"]
                                                       for row in rows),
        "hypothesis_event_alignment_reasons": dict(alignments),
        "scope": "observations before best_projection selection; raw differences are uncorrected, not accepted events; missing diagnostics are not zero events"}


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
    window_audit = read_rows(root/"window_audit.jsonl")
    caster_windows = {row["window_id"]: row for row in window_audit
                      if row.get("status") == "caster_proposals" and row.get("window_id")}
    for item in items:
        evidence = item.get("receiver_competition")
        if evidence is None:
            raise ValueError("Run labels with --receiver-competition on or score-only")
        decision = item["shadow_decision"]
        baseline = decision.get("baseline_projection_decision", decision)
        best = evidence.get("best_projection") or {}
        selected_explanation = next((value for parent in item.get("hypotheses", [])
            for value in [parent, *parent.get("alternatives", [])]
            if best and value["instance_token"] == best["instance_token"]
            and value["light_id"] == best["light_id"]), {})
        selected_track_state = best.get("long_window_track_state", selected_explanation.get("long_window_track_state"))
        selected_missing = best.get("long_window_missing_samples", selected_explanation.get("long_window_missing_samples"))
        row = {"id": item["id"], "scene_name": item["scene_name"], "camera": item["camera"],
               "window_id": (item.get("bev_flow") or {}).get("window_id"),
               "image": item["image"], "mask": item["mask"], "winner": evidence["winner"],
               "scores": evidence["scores"], "winner_margin": evidence["winner_margin"],
               "baseline_scores": evidence["baseline_scores"],
               "mature_3d": evidence.get("mature_3d"),
               "motion_diagnostics": evidence.get("motion_diagnostics"),
               "score_protocol": evidence.get("score_protocol", "receiver-competition-v1"),
               "without_mature_3d": evidence.get("without_mature_3d"),
               "matched_transition_strength": evidence.get("matched_transition_strength", 0),
               "material_nonresponse_fraction": evidence.get("material_nonresponse_fraction", 0),
               "raw_transition_events": evidence.get("raw_transition_events"),
               "best_projection_transitions": best.get("transitions"),
               "best_projection_instance_token": best.get("instance_token"),
               "best_projection_light_id": best.get("light_id"),
               "long_window_track_state": selected_track_state,
               "long_window_missing_samples": selected_missing,
               "light_prior": {"confidence": item["light_prior"]["confidence"],
                   "model_source": item["light_prior"].get("model_source")} if item.get("light_prior") else None,
               "temporal_window": item.get("temporal_window"), "caster_proposals": item.get("caster_proposals"),
               "transition_diagnostics": item.get("transition_diagnostics"),
               "projection_transition_diagnostics": projection_transition_diagnostics(item.get("hypotheses", [])),
               "mature_image_motion": (item.get("bev_flow") or {}).get("mature_image_motion"),
               "projection_control_gain": evidence["projection_control_gain"],
               "trajectory_control_gain": evidence["trajectory_control_gain"],
               "baseline_accepted": baseline["shadow_accepted"],
               "competition_supported": evidence["shadow_supported"],
               "exported_positive": item["exported_label"] == "YES"}
        rows.append(row)
        if row["window_id"] is not None and row["caster_proposals"] is not None:
            caster_windows.setdefault(row["window_id"], row["caster_proposals"])
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
    transition_path = root/"transition_diagnostics.jsonl"
    transition_windows = read_rows(transition_path) if transition_path.exists() else []
    transition_summary = transition_stage_summary(rows, transition_windows)
    write_json(output/"transition_stages.json", transition_summary)
    if panels:
        per_row = 2
        panels += [np.zeros_like(panels[0]) for _ in range((-len(panels)) % per_row)]
        checked_write_image(output/"contact_sheet.jpg", np.vstack([
            np.hstack(panels[index:index+per_row]) for index in range(0, len(panels), per_row)]))
    caster_keys = ("caster_roi_pixels", "caster_roi_receiver_pixels", "caster_roi_dark_pixels",
        "caster_roi_flow_pixels", "caster_roi_bev_pixels", "caster_own_component_pixels",
        "overlap_original", "added_observed_pixels")
    caster_summary = {key: sum(int(value.get(key, 0)) for value in caster_windows.values()) for key in caster_keys}
    caster_summary.update({"windows": len(caster_windows), "windows_with_added":
        sum(int(value.get("added_observed_pixels", 0) > 0) for value in caster_windows.values()),
        "proposal_darkness_ratios": sorted({value.get("proposal_darkness_ratio") for value in caster_windows.values()
                                            if value.get("proposal_darkness_ratio") is not None}),
        "scope": "unique BEV windows; component rows repeat their source-window proposal counts"})
    raw_events = [row["raw_transition_events"] or {} for row in rows]
    long_states = Counter(row["long_window_track_state"] or "no_best_projection" for row in rows)
    track_windows = {row["window_id"]: row.get("track_state_counts") for row in window_audit
                     if row.get("status") == "caster_proposals" and row.get("track_state_counts")}
    track_totals = {key: sum(int(value.get(key, 0)) for value in track_windows.values())
                    for key in ("seen", "complete", "partial", "inconsistent", "missing_samples")}
    write_json(output/"summary.json", {"components": len(rows), "winner_counts": dict(winners),
        "transition_stages": transition_summary,
        "transition_counter_scope": "observed_transition_components and long_event_components below are conditional on best_projection",
        "score_protocols": dict(Counter(row["score_protocol"] for row in rows)),
        "caster_proposals": caster_summary,
        "raw_event_components": sum(value.get("observed_transition_types", 0) > 0 for value in raw_events),
        "raw_formation_pixels": sum(value.get("formation_pixels", 0) for value in raw_events),
        "raw_recovery_pixels": sum(value.get("recovery_pixels", 0) for value in raw_events),
        "matched_transition_components": sum(row["matched_transition_strength"] > 0 for row in rows),
        "observed_transition_components": sum((row["best_projection_transitions"] or {}).get(
            "observed_transition_types", 0) > 0 for row in rows),
        "material_nonresponse_components": sum(row["material_nonresponse_fraction"] > 0 for row in rows),
        "light_prior_components": sum(row["light_prior"] is not None for row in rows),
        "long_event_components": sum(((row["best_projection_transitions"] or {}).get("long_window") or {}).get(
            "observed_transition_types", 0) > 0 for row in rows),
        "long_matched_components": sum(((row["best_projection_transitions"] or {}).get("long_window") or {}).get(
            "edge_fit", 0) > 0 for row in rows),
        "temporal_frame_counts": dict(Counter((row["temporal_window"] or {}).get("frames", 3) for row in rows)),
        "temporal_window_frame_counts": dict(Counter(row.get("temporal_frames", 3) for row in window_audit
                                                     if row.get("status") == "caster_proposals")),
        "gt_track_state_totals": {"windows": len(track_windows), **track_totals,
            "scope": "sum over unique windows; GT vehicle track states, not shadow truth"},
        "long_window_track_state_counts": dict(long_states),
        "transition_metric_scope": {
            "raw_event_components": "component mask in the central triplet, unconditional on caster projection",
            "observed_transition_components": "selected best_projection transitions only",
            "long_event_components": "selected best_projection long-window transitions only",
            "long_matched_components": "selected best_projection long-window edge_fit > 0"},
        "mature_3d_counterfactual": {
            "available": sum(row["without_mature_3d"] is not None for row in rows),
            "without_mature_supported": sum(bool((row["without_mature_3d"] or {}).get("shadow_supported")) for row in rows),
            "winner_changed": sum(row["without_mature_3d"] is not None
                and row["without_mature_3d"]["winner"] != row["winner"] for row in rows),
            "shadow_rejected_by_mature": sum(bool((row["without_mature_3d"] or {}).get("shadow_supported"))
                and not row["competition_supported"] for row in rows),
            "scope": "same candidates and scores; remove only the mature entity-motion contribution"},
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
