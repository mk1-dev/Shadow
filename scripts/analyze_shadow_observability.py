"""Compare frozen physical annotation yields across zero-shot observation strata."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import digest, read_rows, resolve_path, write_rows
from shadow_occ.annotation.observability import for_window, vote_index
from shadow_occ.utils.io import write_json


def summarize(rows):
    components = sum(row["components"] for row in rows)
    count = len(rows)
    event_rows = [row for row in rows if row["uncorrected_event_pixels"] is not None]
    timed = [row for row in rows if row["physics_processing_seconds"] is not None]
    inferred = [row for row in rows if row["vlm_inference_seconds"] is not None]
    return {"windows": count, "scenes": len({row["scene_name"] for row in rows}),
        "windows_with_physics_diagnostics": sum(row["has_physics_diagnostics"] for row in rows),
        "receiver_supported_windows": sum(row["receiver_pixels"] > 0 for row in rows),
        "receiver_supported_window_rate": sum(row["receiver_pixels"] > 0 for row in rows)/count if count else None,
        "components": components,
        "uncorrected_event_evaluated_windows": len(event_rows),
        "uncorrected_event_pixels": sum(row["uncorrected_event_pixels"] for row in event_rows),
        "supported_central_event_pixels": sum(row["supported_central_event_pixels"] for row in rows),
        "event_matched_components": sum(row["event_matched_components"] for row in rows),
        "event_matched_component_rate": sum(row["event_matched_components"] for row in rows)/components if components else None,
        "physical_pass_components": sum(row["physical_pass_components"] for row in rows),
        "exported_positive": sum(row["exported_positive"] for row in rows),
        "positives_per_100_windows": 100*sum(row["exported_positive"] for row in rows)/count if count else None,
        "positive_unresolved": sum(row["positive_unresolved"] for row in rows),
        "positive_unique_caster": sum(row["positive_unique_caster"] for row in rows),
        "timed_physics_windows": len(timed),
        "inferred_windows": len(inferred), "vlm_inference_seconds": sum(row["vlm_inference_seconds"] for row in inferred),
        "physics_processing_seconds": sum(row["physics_processing_seconds"] for row in timed) if timed else None,
        "timing_scope": "window processing only where recorded; old outputs have no retroactively inferred runtime"}


def component_counts(items):
    matched, physical = 0, 0
    for item in items:
        hypotheses = [value for parent in item["hypotheses"] for value in [parent, *parent.get("alternatives", [])]]
        matched += any(value.get("onset_fit", 0) > 0 or value.get("recovery_fit", 0) > 0
            or (value.get("long_window") or {}).get("edge_fit", 0) > 0 for value in hypotheses)
        physical += any(value.get("shadow_passes", False) for value in hypotheses)
    positive = [item for item in items if item["exported_label"] == "YES"]
    return {"components": len(items), "event_matched_components": matched, "physical_pass_components": physical,
        "exported_positive": len(positive),
        "positive_unresolved": sum(item["shadow_decision"]["association"] is None for item in positive),
        "positive_unique_caster": sum(item["shadow_decision"]["association"] is not None for item in positive)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--votes", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    args = parser.parse_args()
    if not 0 <= args.audit_fraction <= 1:
        parser.error("audit-fraction must lie in [0, 1]")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest_rows = read_rows(args.manifest)
    manifest = {row["id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate observability input IDs")
    votes = vote_index(read_rows(args.votes))
    if set(votes) != set(manifest) or any(votes[key]["input_signature"] != manifest[key]["input_signature"] for key in votes):
        raise ValueError("Complete matching observability votes are required for this comparison")
    rows = []
    for split in dict.fromkeys(args.splits):
        root = Path(args.source_root)/split
        pairs = {pair["id"]: pair for pair in read_rows(root/"pairs/pairs.jsonl")}
        diagnostics = {row["window_id"]: row for row in read_rows(root/"labels/transition_diagnostics.jsonl")}
        components = defaultdict(list)
        for item in read_rows(root/"labels/decisions.jsonl"):
            components[item["bev_flow"]["window_id"]].append(item)
        for window in read_rows(root/"bev_flow/windows.jsonl"):
            left, right = (pairs[token] for token in window["pair_ids"])
            tag = for_window(votes, left, right)
            diagnostic = diagnostics.get(window["id"], {})
            raw_events = None
            if window["id"] in votes:
                path = resolve_path(window["evidence"], root/"bev_flow")
                if digest(path) != window["evidence_sha256"]:
                    raise ValueError("Physical cache changed")
                with np.load(path, allow_pickle=False) as archive:
                    lightness, receiver = archive["appearance_L_raw"], archive["receiver_valid"]
                    raw_events = int(((lightness[1]-lightness[0] <= -6) & receiver).sum()
                        +((lightness[2]-lightness[1] >= 6) & receiver).sum())
            rows.append({"window_id": window["id"], "pair_ids": window["pair_ids"], "scene_name": window["scene_name"],
                "camera": window["camera"], "official_split": split, "stratum": tag["stratum"],
                "assessment": tag["assessment"], "receiver_pixels": window["counts"]["receiver"],
                "has_physics_diagnostics": bool(diagnostic),
                "uncorrected_event_pixels": raw_events,
                "supported_central_event_pixels": diagnostic.get("central_onset_pixels", 0)+diagnostic.get("central_recovery_pixels", 0),
                "physics_processing_seconds": diagnostic.get("processing_seconds"),
                "vlm_inference_seconds": votes.get(window["id"], {}).get("inference_seconds"),
                **component_counts(components[window["id"]])})
    rng = random.Random(args.seed)
    background = [row for row in rows if row["stratum"] != "PRIORITY"]
    rng.shuffle(background)
    audit_count = min(len(background), max(1, round(args.audit_fraction*len(background)))) if args.audit_fraction else 0
    audit_ids = {row["window_id"] for row in background[:audit_count]}
    queue = sorted(rows, key=lambda row: (0 if row["stratum"] == "PRIORITY" else 1 if row["window_id"] in audit_ids else 2,
                                          row["official_split"], row["window_id"]))
    write_rows(output/"windows.jsonl", rows)
    write_rows(output/"priority_windows.jsonl", [dict(row, queue_rank=index, audit_sample=row["window_id"] in audit_ids)
                                               for index, row in enumerate(queue)])
    summary = {}
    for split in ("train", "val"):
        population = [row for row in rows if row["official_split"] == split]
        summary[split] = {"ALL_SOURCE": summarize(population),
            "A_ALL_TAGGED": summarize([row for row in population if row["window_id"] in votes]),
            **{name: summarize([row for row in population if row["stratum"] == name])
               for name in ("PRIORITY", "LOW_SIGNAL", "UNCERTAIN", "UNRATED")}}
    matched = {row["window_id"] for row in rows} & set(votes)
    write_json(output/"summary.json", {"splits": summary, "tagged_windows_without_cached_physics": len(set(votes)-matched),
        "votes_strata": dict(Counter(vote["stratum"] for vote in votes.values())),
        "inference_seconds": sum(vote["inference_seconds"] for vote in votes.values()),
        "manifest_sha256": digest(args.manifest), "votes_sha256": digest(args.votes),
        "audit_fraction": args.audit_fraction, "seed": args.seed,
        "scope": "same physical method across observational strata, not algorithm ablations or shadow precision; audit cases are failure-enriched",
        "policy": "all source windows retained; priority is ordering only; unknown and low-signal never become NO labels"})
    print(f"Wrote observability comparison and full-population queue to {output}", flush=True)


if __name__ == "__main__":
    main()
