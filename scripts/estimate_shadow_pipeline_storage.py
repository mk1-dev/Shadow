"""Plan the official scene population and estimate cache/training disk from real files."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.scene_sampling import partition_scenes as scene_partition
from shadow_occ.annotation.data import digest, read_rows, resolve_path, write_rows
from shadow_occ.utils.io import write_json


def size_quantiles(paths):
    sizes = [path.stat().st_size for path in set(paths) if path.is_file()]
    return {"files": len(sizes), "bytes": sum(sizes),
        "median_bytes": float(np.median(sizes)) if sizes else None,
        "p90_bytes": float(np.percentile(sizes, 90)) if sizes else None}


def estimate(population, catalogue_path, lock_path, output_root, destination, fraction=25,
             seed=0, dev_fraction=.15, pairs_per_scene=24, cameras=3, methods=2,
             reference_run=None, parameter_count=11_020_337):
    if fraction not in {25, 50, 100} or min(pairs_per_scene, cameras, methods, parameter_count) < 1:
        raise ValueError("Use positive cache dimensions and stage 25/50/100")
    training, evaluation = set(population["official_train_scenes"]), set(population["official_eval_scenes"])
    if training & evaluation:
        raise ValueError("Official train/eval scene identities overlap")
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    excluded = set(lock["excluded_development_scenes"])
    dev, selected = scene_partition(training, seed, dev_fraction)
    roles = {scene: "train" for scene in selected[fraction]}
    roles.update({scene: "dev" for scene in dev})
    roles.update({scene: "eval" for scene in evaluation-excluded})
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    catalogue = Path(catalogue_path).resolve()
    inventory, paths = defaultdict(list), defaultdict(set)
    tokens, observed_frames, pair_ids = set(), set(), set()
    for source in read_rows(catalogue):
        source_paths = {key: resolve_path(source[key], catalogue.parent) for key in
            ("pairs", "analysis_root", "feature_root", "spatial_feature_root")}
        metrics_path = source_paths["analysis_root"]/"pair_metrics.jsonl"
        metrics = {row["id"]: row for row in read_rows(metrics_path)} if metrics_path.is_file() else {}
        flow_protocol_path = source_paths["analysis_root"]/"analysis_protocol.json"
        flow_protocol = json.loads(flow_protocol_path.read_text()) if flow_protocol_path.is_file() else {}
        for pair in read_rows(source_paths["pairs"]):
            pair_key = pair["scene_name"], pair["camera"], pair["sample_token"]
            if pair_key in pair_ids:
                raise ValueError("Catalogue repeats scene/camera/sample pairs")
            pair_ids.add(pair_key)
            canonical = all(frame.get("receiver_protocol") == "frame-receiver-v1" for frame in pair["frames"])
            row_ready = canonical and flow_protocol.get("flow_evidence_version") == "receiver-support-v2"
            if pair["id"] in metrics:
                path = resolve_path(metrics[pair["id"]]["evidence"], source_paths["analysis_root"])
                paths["flow"].add(path)
                row_ready &= path.is_file()
            else:
                row_ready = False
            for frame in pair["frames"]:
                observed_frames.add(frame["sample_data_token"])
                for name in ("image", "ground_support"):
                    path = resolve_path(frame[name], source_paths["pairs"].parent)
                    paths["receiver"].add(path)
                    row_ready &= path.is_file()
            for token in (pair["sample_token"], pair["next_sample_token"]):
                tokens.add(token)
                for key, role in (("feature_root", "temporal_bev"), ("spatial_feature_root", "current_bev")):
                    path = source_paths[key]/(token+".npz")
                    paths[role].add(path)
                    row_ready &= path.is_file() and path.with_suffix(".json").is_file()
            inventory[pair["scene_name"]].append({"ready": bool(row_ready), "canonical": canonical})
    measurements = {key: size_quantiles(value) for key, value in paths.items()}
    requirements = []
    for scene, role in sorted(roles.items()):
        cached = inventory.get(scene, [])
        reusable = len(cached) >= pairs_per_scene and all(row["ready"] for row in cached)
        requirements.append({"scene_name": scene, "split": role, "cached_pairs": len(cached),
            "canonical_pairs": sum(row["canonical"] for row in cached),
            "metadata_reuse_candidate": reusable,
            "action": "verify_frozen_hashes_before_reuse" if reusable else "prepare_missing_or_legacy_cache"})
    missing = sum(not row["metadata_reuse_candidate"] for row in requirements)
    new_pairs = missing*pairs_per_scene
    sample_ratio = len(tokens)/len(pair_ids) if pair_ids else None
    frame_ratio = len(observed_frames)/len(pair_ids) if pair_ids else None
    components = {}
    for key in ("temporal_bev", "current_bev"):
        measurement = measurements.get(key, {})
        components[key] = {"estimated_units": math.ceil(new_pairs*sample_ratio) if sample_ratio is not None else None,
            "typical_bytes": math.ceil(new_pairs*sample_ratio*measurement["median_bytes"]) if sample_ratio is not None and measurement.get("median_bytes") is not None else None,
            "p90_bytes": math.ceil(new_pairs*sample_ratio*measurement["p90_bytes"]) if sample_ratio is not None and measurement.get("p90_bytes") is not None else None}
    flow = measurements.get("flow", {})
    components["flow"] = {"estimated_units": new_pairs,
        "typical_bytes": math.ceil(new_pairs*flow["median_bytes"]) if flow.get("median_bytes") is not None else None,
        "p90_bytes": math.ceil(new_pairs*flow["p90_bytes"]) if flow.get("p90_bytes") is not None else None}
    # Account for both image and support files; use their measured combined bytes per unique frame.
    receiver_per_frame = measurements.get("receiver", {}).get("bytes", 0)/len(observed_frames) if observed_frames and measurements.get("receiver", {}).get("files") else None
    components["receiver"] = {"typical_bytes": math.ceil(new_pairs*frame_ratio*receiver_per_frame) if frame_ratio is not None and receiver_per_frame is not None else None}
    reference_bytes, reference_frames = 0, 0
    if reference_run:
        for path in Path(reference_run).glob("*/frames.jsonl"):
            rows = read_rows(path)
            reference_frames += len(rows)
            reference_bytes += sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    total_frames = math.ceil(len(roles)*pairs_per_scene*(frame_ratio or 0))
    components["teacher_outputs"] = {"typical_bytes": math.ceil(reference_bytes/reference_frames*total_frames*methods*7/3) if reference_frames else None,
        "includes": "measured clip outputs; seven-frame/stride-three window overlap factor 7/3 before deduplication"}
    # Float32 best+last checkpoints; the stable checkpoint alias is a hardlink on HPC.
    components["student_checkpoints"] = {"typical_bytes": parameter_count*4*2*methods,
        "parameter_count": parameter_count, "optimizer_saved": False}
    eval_frames = math.ceil(sum(role == "eval" for role in roles.values())*pairs_per_scene*(frame_ratio or 0))
    components["student_probabilities_and_targets"] = {"typical_bytes": int((total_frames*methods+eval_frames*methods*5)*800*450),
        "assumption": "800x450; one byte target per frame/method and float32 score plus byte mask per eval frame/method, before NPZ/PNG compression"}
    existing = Path(destination).resolve()
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    known = [row["typical_bytes"] for row in components.values() if row.get("typical_bytes") is not None]
    total = sum(known)
    complete = all(row.get("typical_bytes") is not None for row in components.values())
    report = {"protocol": "shadow-storage-estimate-v1", "stage": fraction, "seed": seed,
        "official_train_scenes": len(training), "train_pool_after_dev": len(selected[100]),
        "selected_train_scenes": len(selected[fraction]), "dev_scenes": len(dev),
        "eval_scenes": len(evaluation-excluded), "prior_development_eval_excluded": sorted(evaluation & excluded),
        "scene_requiring_cache_preparation": missing, "metadata_reuse_candidates": len(roles)-missing,
        "measurements": measurements, "components": components,
        "additional_typical_bytes": total, "additional_with_30_percent_headroom_bytes": math.ceil(total*1.3),
        "estimated_total_is_complete": complete, "free_bytes": disk.free,
        "fits_typical_with_headroom": total*1.3 <= disk.free if complete else None,
        "scope": "planning estimate, not disk reservation or cache integrity approval; label yields are unknown",
        "assumptions": {"pairs_per_scene": pairs_per_scene, "cameras": cameras, "methods": methods,
            "observed_samples_per_pair": sample_ratio, "observed_camera_frames_per_pair": frame_ratio,
            "cache_npz_includes_metadata": False, "new_scene_density_may_differ": True},
        "catalogue_sha256": digest(catalogue), "freeze_lock_sha256": digest(lock_path),
        "jobs_submitted": False}
    output.mkdir(parents=True)
    write_json(output/"population.json", population)
    write_json(output/"storage_estimate.json", report)
    write_rows(output/"cache_requirements.jsonl", requirements)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--population", help="Optional JSON with official_train_scenes and official_eval_scenes")
    parser.add_argument("--fraction", type=int, choices=[25, 50, 100], default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=.15)
    parser.add_argument("--pairs-per-scene", type=int, default=24)
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--methods", type=int, default=2)
    parser.add_argument("--parameter-count", type=int, default=11_020_337)
    parser.add_argument("--reference-run")
    args = parser.parse_args()
    if args.population:
        population = json.loads(Path(args.population).read_text(encoding="utf-8"))
    else:
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        population = {"official_train_scenes": splits["train"], "official_eval_scenes": splits["val"]}
    result = estimate(population, args.catalogue, args.lock, args.output_root, args.destination,
        args.fraction, args.seed, args.dev_fraction, args.pairs_per_scene, args.cameras, args.methods,
        args.reference_run, args.parameter_count)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
