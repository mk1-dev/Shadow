"""Repeat receiver gates on identical cached windows without recomputing flow."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.cached_bev import attach_cached_bev, select_bev_flow_receivers
from shadow_occ.annotation.discrepancy_proposals import DiscrepancyConfig
from shadow_occ.annotation.data import digest, read_rows, resolve_path, write_rows
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-root", help="Optional replacement cache, e.g. current-only features")
    parser.add_argument("--discovery-mode", choices=["discrepancy", "cross-space", "hard-intersection"], default="discrepancy")
    parser.add_argument("--discrepancy-minimum-score", type=float, default=.35)
    parser.add_argument("--discrepancy-minimum-light-change", type=float, default=1.5)
    parser.add_argument("--discrepancy-flow-scale", type=float, default=3.0)
    parser.add_argument("--proposal-entity-motion", choices=["on", "off"], default="on")
    parser.add_argument("--feature-gate", choices=["aligned", "none", "shuffled", "matched-random"], default="aligned")
    parser.add_argument("--flow-gate", choices=["registered", "none"], default="registered")
    parser.add_argument("--minimum-bev-cosine", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mature-object-mode", choices=["required", "off"], default="required")
    args = parser.parse_args()
    proposal_config = DiscrepancyConfig(args.discrepancy_minimum_score,
        args.discrepancy_minimum_light_change, args.discrepancy_flow_scale)
    proposal_config.validate()
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    protocol_path = source/"bev_flow_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, counts = [], Counter()
    for row in tqdm(read_rows(source/"windows.jsonl"), desc="cached receiver ablation", mininterval=5):
        path = resolve_path(row["evidence"], source)
        if digest(path) != row["evidence_sha256"]:
            raise ValueError(f"Changed source window: {row['id']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files
                if key != "learned_bev_receiver_features" or args.feature_gate == "shuffled"}
        if args.mature_object_mode == "required":
            if not row.get("mature_objects") or "mature_entity_probability" not in arrays:
                raise ValueError("Source lacks decoded 3D results; upgrade the mature object cache and rebuild first")
        else:
            arrays = {key: value for key, value in arrays.items() if not key.startswith("mature_")}
            row = {key: value for key, value in row.items() if key not in {"mature_objects", "mature_image_motion"}}
            row["counts"] = {key: value for key, value in row["counts"].items() if not key.startswith("mature_")}
        original = arrays["shadow_candidate_photometric"]
        threshold = protocol["config"]["light_threshold_L"]
        delta = arrays["temporal_illumination_change_L"]
        # Older caches stored transitions only after the original feature gate.
        events = {"shadow_boundary_onset": delta[0] <= -threshold,
                  "shadow_boundary_recovery": delta[1] >= threshold,
                  "shadow_same_patch_cycle": (delta[0] <= -threshold) & (delta[1] >= threshold)}
        comparable = arrays.get("temporal_exposure_valid", row.get("exposure_comparable", True))
        for key, event in events.items():
            arrays[f"{key}_photometric"] = event & original & comparable
        feature_root = args.feature_root
        if args.feature_gate == "shuffled" and "learned_bev_receiver_features" not in arrays:
            feature_root = feature_root or protocol.get("feature_root")
            if not feature_root:
                raise ValueError("Shuffled features need the original --feature-root cache")
        if feature_root:
            previous_sources = row.get("learned_bev", {}).get("sources", [])
            arrays, row = attach_cached_bev(arrays, row, feature_root)
            for old, new in zip(previous_sources, row["learned_bev"]["sources"]):
                for key in ("checkpoint_sha256", "model_config_sha256"):
                    if old.get(key) != new.get(key):
                        raise ValueError(f"Feature-history comparison changed encoder {key}")
        seed = int(hashlib.sha256(f"{args.seed}:{row['id']}".encode()).hexdigest()[:16], 16)
        row["discovery_config"] = {key: protocol["config"][key] for key in
            ("light_threshold_L", "darkness_ratio") if key in protocol["config"]}
        arrays, row = select_bev_flow_receivers(arrays, row, args.minimum_bev_cosine,
            feature_gate=args.feature_gate, flow_gate=args.flow_gate == "registered", seed=seed,
            discovery_mode=args.discovery_mode, discrepancy_config=proposal_config,
            proposal_entity_motion=args.proposal_entity_motion)
        arrays.pop("learned_bev_receiver_features", None)
        destination = output/"windows"/f"{row['id']}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        rows.append({**row, "evidence": str(destination), "evidence_sha256": digest(destination)})
        counts.update(row["counts"])
    if not rows:
        raise ValueError("Source must contain frozen three-frame windows")
    write_rows(output/"windows.jsonl", rows)
    write_json(output/"summary.json", {"windows": len(rows), "counts": dict(counts),
        "candidate_retention": counts["shadow_candidates"]/max(counts["photometric_candidates"], 1)})
    write_json(output/"bev_flow_protocol.json", {**protocol, **vars(args),
        "cross_space_code_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/cross_space.py"),
        "discrepancy_code_sha256": digest(Path(__file__).resolve().parents[1]/"shadow_occ/annotation/discrepancy_proposals.py"),
        "source_protocol_sha256": digest(protocol_path),
        "source_windows_sha256": digest(source/"windows.jsonl"),
        "receiver_feature_storage": "shared source cache; derived evidence keeps gate results only",
        "feature_root": str(feature_root) if feature_root else protocol.get("feature_root"),
        "feature_cache_protocol_sha256": digest(Path(feature_root)/"cache_protocol.json") if feature_root
            else protocol.get("feature_cache_protocol_sha256"),
        "ablation_scope": "identical geometry, images, illumination and exposure; configurable final gates"})


if __name__ == "__main__":
    main()
