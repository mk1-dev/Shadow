"""Automatically plan incremental 25/50/100% scene annotation from prepared caches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, resolve_path, write_rows
from shadow_occ.annotation.physics_io import load_motion_analysis
from shadow_occ.annotation.scene_sampling import partition_scenes as scene_partition
from shadow_occ.utils.io import write_json


def rank(value, seed, scope):
    return hashlib.sha256(f"{seed}:{scope}:{value}".encode()).hexdigest()


def automatic_clips(pairs, num_frames=7, stride=3, seed=0):
    """One deterministic GT track per window; no image/shadow-quality selection."""
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["camera"]].append(pair)
    result, counts = [], Counter()
    for camera, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["frames"][0]["camera_timestamp"])
        lookup = {row["sample_token"]: row for row in rows}
        if len(lookup) != len(rows):
            raise ValueError("Duplicate sample/camera pairs in automatic catalogue")
        for first in rows[::stride]:
            token, chain, common = first["sample_token"], [], None
            for _ in range(num_frames-1):
                if token not in lookup:
                    break
                pair = lookup[token]
                tracks = {obj["instance_token"] for obj in pair["objects"]
                          if obj.get("category", "vehicle.").startswith("vehicle.")}
                compatible = tracks if common is None else common & tracks
                if not compatible:
                    break
                chain.append(pair)
                common = compatible
                token = pair["next_sample_token"]
            counts["attempted_windows"] += 1
            if len(chain) < 2:
                counts["without_three_frame_gt_track"] += 1
                continue
            track = min(common, key=lambda value: rank(value, seed, first["id"]))
            identity = hashlib.sha256(json.dumps([first["scene_name"], camera, first["sample_token"], track]).encode()).hexdigest()[:24]
            result.append({"id": "auto_"+identity, "scene_name": first["scene_name"], "camera": camera,
                "start_sample_token": first["sample_token"], "track_id": track, "num_frames": len(chain)+1,
                "selection": "deterministic_contiguous_GT_track_without_shadow_or_VLM_filter"})
            counts["exported_clips"] += 1
    return result, dict(counts)


def plan_scenes(catalogue_path, lock_path, output_root, annotation_root, seed=0, dev_fraction=.15,
                num_frames=7, stride=3, expected_train_scenes=None, population_path=None,
                methods=("anchor_first_v8plus",)):
    catalogue_path = Path(catalogue_path).resolve()
    output, annotations = Path(output_root).resolve(), Path(annotation_root).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    excluded = set(lock["excluded_development_scenes"])
    if not 0 < dev_fraction < 1 or not 3 <= num_frames <= 31 or stride < 1:
        raise ValueError("Invalid automatic scene/window plan")
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {"appearance_only", "anchor_first_v8plus"}:
        raise ValueError("Only implemented discovery modes can be planned; historical baselines need their own frozen runs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    sources = read_rows(catalogue_path)
    if not sources:
        raise ValueError("Provide prepared pairs/flow/BEV catalogue rows")
    scene_splits, prepared, seen = {}, [], set()
    input_hashes = {str(catalogue_path): digest(catalogue_path), str(Path(lock_path).resolve()): digest(lock_path)}
    for at, source in enumerate(sources):
        paths = {key: resolve_path(source[key], catalogue_path.parent) for key in
                 ("pairs", "analysis_root", "feature_root", "spatial_feature_root")}
        pairs, stats, protocol_path = load_motion_analysis(paths["pairs"], paths["analysis_root"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        parent_hashes = {"parent_pairs_sha256": digest(paths["pairs"]), "parent_protocol_sha256": digest(protocol_path),
                         "parent_metrics_sha256": digest(paths["analysis_root"]/"pair_metrics.jsonl")}
        input_hashes.update({str(paths["pairs"]): parent_hashes["parent_pairs_sha256"],
            str(protocol_path): parent_hashes["parent_protocol_sha256"],
            str(paths["analysis_root"]/"pair_metrics.jsonl"): parent_hashes["parent_metrics_sha256"]})
        groups = defaultdict(list)
        for pair in pairs:
            scene, official = pair["scene_name"], pair["official_split"]
            if official not in {"train", "mini_train", "val", "mini_val"}:
                raise ValueError("Automatic weak plan expects official nuScenes train/val scenes")
            role = "train" if official in {"train", "mini_train"} else "eval"
            if scene_splits.setdefault(scene, role) != role:
                raise ValueError("A scene occurs in conflicting official splits")
            key = scene, pair["camera"], pair["sample_token"]
            if key in seen:
                raise ValueError("Input catalogues overlap the same scene/camera/sample")
            seen.add(key)
            groups[scene].append(pair)
        prepared.append((at, paths, groups, stats, protocol, parent_hashes))
    training = sorted(scene for scene, role in scene_splits.items() if role == "train")
    if len(training) < 2:
        raise ValueError("Automatic train/dev isolation needs at least two prepared train scenes")
    population = json.loads(Path(population_path).read_text(encoding="utf-8")) if population_path else None
    if population_path:
        input_hashes[str(Path(population_path).resolve())] = digest(population_path)
    official_training = population["official_train_scenes"] if population else training
    if population and (not set(training) <= set(official_training)
                       or not {scene for scene, role in scene_splits.items() if role == "eval"} <= set(population["official_eval_scenes"])):
        raise ValueError("Prepared scenes are outside the declared official population")
    if expected_train_scenes is not None and len(official_training) != expected_train_scenes:
        raise ValueError(f"Prepared catalogue has {len(training)} train scenes, expected {expected_train_scenes}; do not call a small cache full-data")
    dev, selected = scene_partition(official_training, seed, dev_fraction)
    assigned = {scene: next(fraction for fraction in (25, 50, 100) if scene in selected[fraction]) for scene in selected[100]}
    batches = {key: [] for key in ("shared", "stage_025", "stage_050", "stage_100")}
    export_rows = defaultdict(list)
    diagnostics, usable_scenes = [], set()
    output.mkdir(parents=True)
    for source_id, paths, groups, stats, protocol, parent_hashes in prepared:
        for scene, rows in sorted(groups.items()):
            split = "dev" if scene in dev else scene_splits[scene]
            if split == "eval" and scene in excluded:
                diagnostics.append({"scene_name": scene, "reason": "prior_teacher_development_scene_excluded_from_eval"})
                continue
            clips, counts = automatic_clips(rows, num_frames, stride, seed)
            if not clips:
                diagnostics.append({"scene_name": scene, "split": split, "status": "no_valid_clip",
                    "reason": "no_contiguous_GT_windows", **counts})
                continue
            tag = hashlib.sha256(f"{paths['pairs']}:{scene}".encode()).hexdigest()[:16]
            folder = output/"scene_inputs"/tag
            shard_rows, shard_stats = [], []
            for pair in rows:
                shard_rows.append({**pair, "frames": [{**frame,
                    "image": str(resolve_path(frame["image"], paths["pairs"].parent)),
                    "ground_support": str(resolve_path(frame["ground_support"], paths["pairs"].parent))}
                    for frame in pair["frames"]]})
                metric = stats[pair["id"]]
                shard_stats.append({**metric, "evidence": str(resolve_path(metric["evidence"], paths["analysis_root"]))})
            write_rows(folder/"pairs.jsonl", shard_rows)
            write_rows(folder/"flow/pair_metrics.jsonl", shard_stats)
            write_json(folder/"flow/analysis_protocol.json", {**protocol, "manifest_sha256": digest(folder/"pairs.jsonl"),
                "derived_metadata_subset": True, "raw_flow_recomputed": False,
                **parent_hashes})
            write_rows(folder/"clips.jsonl", clips)
            stage = "shared" if split in {"dev", "eval"} else f"stage_{assigned[scene]:03d}"
            for method in methods:
                batch_id = split+"_"+tag+("_appearance" if method == "appearance_only" else "")
                batches[stage].append({"id": batch_id, "split": split, "method": method,
                    "discovery_mode": "appearance" if method == "appearance_only" else "cross-space",
                    "pairs": str(folder/"pairs.jsonl"), "analysis_root": str(folder/"flow"), "clips": str(folder/"clips.jsonl"),
                    "feature_root": str(paths["feature_root"]), "spatial_feature_root": str(paths["spatial_feature_root"])})
                export_rows[scene].append({"method": method, "split": split,
                    "run_root": str(annotations/stage/batch_id), "pairs": [str(folder/"pairs.jsonl")]})
            usable_scenes.add(scene)
            diagnostics.append({"scene_name": scene, "split": split, "stage": stage, "status": "ready",
                "source_catalogue_index": source_id, **counts})
    for stage, rows in batches.items():
        write_rows(output/stage/"batches.jsonl", rows)
    for fraction in (25, 50, 100):
        rows = [row for scene in sorted(export_rows) for row in export_rows[scene]
                if row["split"] != "train" or scene in selected[fraction]]
        write_rows(output/f"stage_{fraction:03d}"/"experiment_runs.jsonl", rows)
    write_rows(output/"scene_coverage.jsonl", diagnostics)
    evaluation = (set(population["official_eval_scenes"]) if population else {scene for scene, role in scene_splits.items() if role == "eval"})-excluded
    targets = {"shared": dev | evaluation,
        **{f"stage_{fraction:03d}": selected[fraction] | dev | evaluation for fraction in selected}}
    coverage = {stage: {"requested_scenes": sorted(target),
        "missing_cache_scenes": sorted(target-set(scene_splits)),
        "no_valid_clip_scenes": sorted((target & set(scene_splits))-usable_scenes),
        "usable_scenes": sorted(target & usable_scenes),
        "complete_clip_coverage": target <= usable_scenes} for stage, target in targets.items()}
    payload = {"protocol": "automatic-shadow-scene-plan-v1", "seed": seed, "dev_fraction": dev_fraction,
        "catalogue_sha256": digest(catalogue_path), "freeze_lock_sha256": digest(lock_path),
        "catalogue_path": str(catalogue_path), "input_manifest_sha256": input_hashes,
        "population_scope": "prepared input catalogues, not an assertion of complete nuScenes coverage",
        "population_manifest_sha256": digest(population_path) if population_path else None,
        "official_train_population": len(official_training), "methods": list(methods),
        "missing_stage_scenes": {str(fraction): sorted((selected[fraction] | dev | (set(population["official_eval_scenes"])-excluded if population else set()))-set(scene_splits)) for fraction in selected},
        "missing_shared_scenes": sorted((dev | (set(population["official_eval_scenes"])-excluded if population else set()))-set(scene_splits)),
        "prepared_official_train_scenes": len(training), "dev_scenes": sorted(dev),
        "train_scene_stages": {str(fraction): sorted(selected[fraction]) for fraction in selected},
        "incremental_batch_counts": {stage: len(rows) for stage, rows in batches.items()},
        "stage_coverage": coverage,
        "coverage_contract": "cache presence, usable GT clips and exported label coverage are distinct; no-valid-clip scenes are explicit skips, not completed annotations",
        "annotation_output_root": str(annotations), "manual_shadow_selection": False,
        "raw_observations_recomputed": False, "dev_eval_annotations_reused_between_stages": True}
    write_json(output/"plan.json", payload)
    return payload


def check_plan(plan_root, stage, lock_path=None):
    root = Path(plan_root).resolve()
    plan = json.loads((root/"plan.json").read_text(encoding="utf-8"))
    if stage not in {"shared", "stage_025", "stage_050", "stage_100"}:
        raise ValueError("Unknown automatic stage")
    if lock_path and digest(lock_path) != plan["freeze_lock_sha256"]:
        raise ValueError("Plan uses a different freeze lock; generate a new plan")
    for path, expected in plan.get("input_manifest_sha256", {}).items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f"Prepared inputs changed: {path}; refresh matching flow evidence and generate a new plan")
    if "stage_coverage" not in plan:
        raise ValueError("Legacy plan lacks usable-clip coverage; generate a new plan")
    coverage = plan["stage_coverage"][stage]
    if coverage["missing_cache_scenes"]:
        raise ValueError(f"{stage}: {len(coverage['missing_cache_scenes'])} target scenes lack prepared caches; prepare them and regenerate the plan")
    shared = read_rows(root/"shared/batches.jsonl")
    if not all(any(row["split"] == split for row in shared) for split in ("dev", "eval")):
        raise ValueError("Shared stage requires usable dev AND eval clips; an empty shared stage is not complete")
    if stage != "shared":
        runs = read_rows(root/stage/"experiment_runs.jsonl")
        if not any(row["split"] == "train" for row in runs):
            raise ValueError("No usable training clips in the cumulative stage")
    return {"stage": stage, **coverage, "no_valid_clip_policy": "explicitly skipped; never counted as annotated"}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        parser = argparse.ArgumentParser("Check fresh inputs and effective annotation coverage")
        parser.add_argument("--plan", required=True)
        parser.add_argument("--stage", required=True)
        parser.add_argument("--lock")
        args = parser.parse_args(sys.argv[2:])
        print(json.dumps(check_plan(args.plan, args.stage, args.lock)), flush=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=.15)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--expected-train-scenes", type=int)
    parser.add_argument("--population", help="Official scene population, fixed before partial cache preparation")
    parser.add_argument("--methods", nargs="+", choices=["appearance_only", "anchor_first_v8plus"], default=["anchor_first_v8plus"])
    args = parser.parse_args()
    print(json.dumps(plan_scenes(args.catalogue, args.lock, args.output_root, args.annotation_root,
        args.seed, args.dev_fraction, args.num_frames, args.stride, args.expected_train_scenes, args.population, args.methods)), flush=True)


if __name__ == "__main__":
    main()
