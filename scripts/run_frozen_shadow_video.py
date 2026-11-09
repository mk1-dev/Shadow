"""Lock v8+ settings, then run unchanged discovery on scene-disjoint clips."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import BEVFlowConfig
from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.entity_memory import EntityMotionConfig, OwnerScoringConfig
from shadow_occ.annotation.physics_labels import LabelConfig
from shadow_occ.annotation.shadow_video import VideoShadowConfig
from shadow_occ.annotation.video_correspondence import CorrespondenceConfig, CrossSpaceGrowthConfig, FlowConfidenceConfig
from shadow_occ.annotation.video_review import find_clips
from shadow_occ.utils.io import write_json


SETTINGS = ("stage", "bev_role", "discovery_mode", "discovery_domain", "owner_scoring",
    "entity_suppression_enabled", "entity_motion_config", "discovery_config", "physics_config",
    "owner_scoring_config", "correspondence_config", "m2_flow_config", "cross_space_growth_config", "grid", "m2_correspondence")
MODEL_FIELDS = ("checkpoint_sha256", "model_config_sha256", "external_exporter_sha256", "history_mode",
                "pc_range", "shape_chw", "coordinate_frame", "layout")


def source_hashes(repo):
    paths = sorted((repo/"shadow_occ").rglob("*.py"))+[repo/"scripts/discover_shadow_video.py"]
    return {path.relative_to(repo).as_posix(): digest(path) for path in paths}


def git_revision(repo):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def settings(protocol):
    return {key: protocol[key] for key in SETTINGS}


def model_signatures(protocol):
    signatures = []
    for row in protocol.get("spatial_feature_sources", []):
        signature = {key: row.get(key) for key in MODEL_FIELDS}
        if not signature["checkpoint_sha256"] or not signature["model_config_sha256"]:
            raise ValueError("Frozen BEV model needs checkpoint and config hashes")
        if signature not in signatures:
            signatures.append(signature)
    if not signatures or any(row["history_mode"] != "current-only" for row in signatures):
        raise ValueError("Freeze a run with recorded current-only BEV model provenance")
    return signatures


def cli_options(config):
    discovery = config["discovery_config"]
    grid = config["grid"]
    extent = grid["x_max"]
    if not (grid["x_min"] == -extent and grid["y_min"] == -extent and grid["y_max"] == extent):
        raise ValueError("Discovery CLI requires the frozen square symmetric receiver grid")
    expected = {
        "discovery_config": asdict(VideoShadowConfig(seed_score=discovery["seed_score"], grow_score=discovery["grow_score"])),
        "physics_config": asdict(LabelConfig(minimum_area=discovery["minimum_area"], minimum_change_pixels=1,
                                             allow_static_existence=True, allow_partial_temporal=True)),
        "owner_scoring_config": asdict(OwnerScoringConfig()),
        "correspondence_config": asdict(CorrespondenceConfig()),
        "m2_flow_config": asdict(FlowConfidenceConfig()),
        "cross_space_growth_config": asdict(CrossSpaceGrowthConfig()),
        "grid": asdict(BEVFlowConfig(extent_m=extent, resolution_m=grid["resolution"]).grid()),
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"Current runtime does not reproduce frozen {key}; do not silently substitute defaults")
    entity = EntityMotionConfig(**config["entity_motion_config"])
    entity.validate()
    options = {"extent-m": extent, "resolution-m": grid["resolution"],
        "seed-score": discovery["seed_score"], "grow-score": discovery["grow_score"],
        "stage": config["stage"], "bev-role": config["bev_role"], "discovery-mode": config["discovery_mode"],
        "discovery-domain": config["discovery_domain"], "owner-scoring": config["owner_scoring"],
        "entity-motion-suppression": "on" if config["entity_suppression_enabled"] else "off",
        "entity-core-margin-px": entity.core_margin_px, "entity-max-residual-px": entity.maximum_residual_px,
        "entity-max-fb-error-px": entity.maximum_fb_error_px}
    return [value for key, item in options.items() for value in ("--"+key, str(item))]


def check_batch_cache(paths, clips, signatures):
    """Check only selected sample metadata; discovery verifies referenced NPZ hashes."""
    pairs = read_rows(paths["pairs"])
    lookup = {(row["scene_name"], row["camera"], row["sample_token"]): row for row in pairs}
    fingerprints = {}
    for clip in clips:
        token = clip["start_sample_token"]
        for _ in range(clip["num_frames"]-1):
            key = (clip["scene_name"], clip["camera"], token)
            if key not in lookup:
                raise ValueError(f"Noncontiguous new-scene clip: {key}")
            pair = lookup[key]
            for sample in (token, pair["next_sample_token"]):
                for role in ("feature_root", "spatial_feature_root"):
                    path = paths[role]/f"{sample}.json"
                    meta = json.loads(path.read_text(encoding="utf-8"))
                    signature = {field: meta.get(field) for field in MODEL_FIELDS}
                    if role == "spatial_feature_root" and signature not in signatures:
                        raise ValueError("New cache differs from the frozen current-only BEV model")
                    if (signature["checkpoint_sha256"], signature["model_config_sha256"]) not in {
                            (item["checkpoint_sha256"], item["model_config_sha256"]) for item in signatures}:
                        raise ValueError("Temporal/context BEV cache uses a different mature model")
                    fingerprints[str(path)] = digest(path)
            token = pair["next_sample_token"]
    for name in ("analysis_protocol.json", "pair_metrics.jsonl"):
        path = paths["analysis_root"]/name
        fingerprints[str(path)] = digest(path)
    return fingerprints


def freeze(reference_root, development_roots, output_path, repo=None, development_pairs=()):
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(Path(root).resolve() in output.parents for root in [reference_root, *development_roots]):
        raise ValueError("Store freeze lock outside frozen reference runs")
    entries = list(find_clips(reference_root).values())
    configs = [settings(entry[1]) for entry in entries]
    if any(row != configs[0] for row in configs[1:]):
        raise ValueError("Reference clips used different settings")
    if configs[0]["m2_correspondence"] != "anchor-first-reasoning-v1" or configs[0]["stage"] != "full":
        raise ValueError("Use the completed Anchor-first full run as reference")
    signatures = model_signatures(entries[0][1])
    if any(model_signatures(entry[1]) != signatures for entry in entries[1:]):
        raise ValueError("Reference clips used different BEV extraction models")
    cli_options(configs[0])
    excluded, protocols = set(), {}
    for root in [reference_root, *development_roots]:
        for folder, protocol, _ in find_clips(root).values():
            excluded.add(protocol["clip"]["scene_name"])
            protocols[str(folder/"protocol.json")] = digest(folder/"protocol.json")
    development_manifests = {}
    for path in development_pairs:
        path = Path(path).resolve()
        rows = read_rows(path)
        if not rows or any(not row.get("scene_name") for row in rows):
            raise ValueError("Development pair manifests must identify actual previously used scenes")
        excluded.update(row["scene_name"] for row in rows)
        development_manifests[str(path)] = digest(path)
    payload = {"protocol": "frozen-anchor-first-generalization-v1", "source_commit": git_revision(repo),
        "source_sha256": source_hashes(repo), "reference_protocol_sha256": protocols,
        "development_pair_manifest_sha256": development_manifests,
        "excluded_development_scenes": sorted(excluded), "settings": configs[0],
        "spatial_model_signatures": signatures, "discovery_cli_options": cli_options(configs[0]),
        "historical_reference_source_identity_verified": False,
        "note": "Source hashes freeze current algorithm files; old run provenance must be checked by the operator."}
    write_json(output, payload)
    return payload


def run_frozen(lock_path, batches_path, output_root, repo=None):
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if source_hashes(repo) != lock["source_sha256"]:
        raise ValueError("Algorithm source differs from freeze lock")
    if cli_options(lock["settings"]) != lock["discovery_cli_options"]:
        raise ValueError("Frozen CLI settings changed")
    batches_path = Path(batches_path).resolve()
    batches = read_rows(batches_path)
    if not batches or len({row["id"] for row in batches}) != len(batches):
        raise ValueError("Choose nonempty uniquely identified batches")
    prepared, manifest_hashes = [], {}
    excluded = set(lock["excluded_development_scenes"])
    seen_clips = set()
    for row in batches:
        if not row["id"] or Path(row["id"]).name != row["id"] or row["id"] in {".", ".."} or "\\" in row["id"]:
            raise ValueError("Batch ID must be a simple directory name")
        paths = {key: (batches_path.parent/row[key]).resolve() for key in
                 ("pairs", "analysis_root", "feature_root", "spatial_feature_root", "clips")}
        clips = read_rows(paths["clips"])
        if not clips or len({clip["id"] for clip in clips}) != len(clips):
            raise ValueError("Batch needs nonempty uniquely identified clips")
        overlap = {clip["scene_name"] for clip in clips} & excluded
        split = row.get("split", "eval")
        if split not in {"train", "dev", "eval"}:
            raise ValueError("Unknown annotation batch split")
        if overlap and split == "eval":
            raise ValueError(f"New-scene batch overlaps development scenes: {sorted(overlap)}")
        if split in {"train", "dev"} and any(pair.get("official_split") not in {"train", "mini_train"} for pair in read_rows(paths["pairs"])):
            raise ValueError("Only official training scenes may be annotated as train/dev")
        mode = row.get("discovery_mode", lock["settings"]["discovery_mode"])
        if mode not in {lock["settings"]["discovery_mode"], "appearance"}:
            raise ValueError("Only the explicit appearance proposal ablation may differ from the frozen teacher")
        for clip in clips:
            key = (mode, clip["scene_name"], clip["camera"], clip["start_sample_token"], clip["track_id"], clip["num_frames"])
            if key in seen_clips:
                raise ValueError("Duplicate clip across batches")
            seen_clips.add(key)
        for key in ("pairs", "clips"):
            manifest_hashes[str(paths[key])] = digest(paths[key])
        for key in ("analysis_root", "feature_root", "spatial_feature_root"):
            if not paths[key].is_dir():
                raise FileNotFoundError(paths[key])
        manifest_hashes.update(check_batch_cache(paths, clips, lock["spatial_model_signatures"]))
        prepared.append((row["id"], paths, clips, {**lock["settings"], "discovery_mode": mode}))
    output = Path(output_root).resolve()
    inputs = [path for _, paths, _, _ in prepared for key, path in paths.items() if key.endswith("root")]
    if any(output == path or path in output.parents for path in inputs):
        raise ValueError("Use a separate output directory outside frozen inputs/manifests")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"freeze_lock.json", lock)
    write_rows(output/"batches_submitted.jsonl", [{"id": identity, "discovery_mode": config["discovery_mode"], **{key: str(path) for key, path in paths.items()}}
                                                 for identity, paths, _, config in prepared])
    protocol = {"freeze_lock_sha256": digest(lock_path), "batch_manifest_sha256": digest(batches_path),
        "input_manifest_sha256": manifest_hashes, "excluded_development_scenes": sorted(excluded),
        "algorithm_settings_overridden": any(config != lock["settings"] for _, _, _, config in prepared),
        "ablation_contract": "appearance-only proposal ablation retains frozen receiver, physics, caster and memory; not a geometry-free pipeline",
        "training_performed": False, "batches": []}
    protocol["batch_splits"] = {row["id"]: row.get("split", "eval") for row in batches}
    protocol["development_scene_policy"] = "prior teacher-development scenes may be training data, never held-out eval"
    write_json(output/"frozen_run.json", protocol)
    for identity, paths, clips, config in prepared:
        command = [sys.executable, "-u", str(repo/"scripts/discover_shadow_video.py")]
        for key, path in paths.items():
            command.extend(["--"+key.replace("_", "-"), str(path)])
        command.extend(["--output-root", str(output/identity), *cli_options(config)])
        record = {"id": identity, "command": command, "status": "running"}
        protocol["batches"].append(record)
        write_json(output/"frozen_run.json", protocol)
        print(f"Frozen batch {identity}: starting", flush=True)
        result = subprocess.run(command, cwd=str(repo))
        record["exit_code"] = result.returncode
        record["status"] = "failed" if result.returncode else "verifying"
        write_json(output/"frozen_run.json", protocol)
        if result.returncode:
            raise RuntimeError(f"Batch {identity} failed: {result.returncode}")
        try:
            actual_entries = list(find_clips(output/identity).values())
            if {item[1]["clip"]["id"] for item in actual_entries} != {clip["id"] for clip in clips}:
                raise ValueError("Discovery did not export the exact submitted clip list")
            expected_clips = {clip["id"]: clip for clip in clips}
            for _, actual, rows in actual_entries:
                if actual["clip"] != expected_clips[actual["clip"]["id"]] or len(rows) != actual["clip"]["num_frames"]:
                    raise ValueError("Discovery clip identity or frame count differs from submitted manifest")
                if settings(actual) != config:
                    raise ValueError("Actual run settings differ from frozen reference")
                if model_signatures(actual) != lock["spatial_model_signatures"]:
                    raise ValueError("New data used a different frozen BEV model/extraction protocol")
            if source_hashes(repo) != lock["source_sha256"] or any(digest(path) != sha for path, sha in manifest_hashes.items()):
                raise ValueError("Frozen source or input manifests changed during execution")
        except Exception:
            record["status"] = "verification_failed"
            write_json(output/"frozen_run.json", protocol)
            raise
        record["status"] = "completed"
        write_json(output/"frozen_run.json", protocol)
        print(f"Frozen batch {identity}: completed and verified", flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("freeze")
    lock.add_argument("--reference-root", required=True)
    lock.add_argument("--development-roots", nargs="*", default=[])
    lock.add_argument("--development-pairs", nargs="*", default=[], help="Pair manifests actually used in previous algorithm development, not future cache coverage")
    lock.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--lock", required=True)
    run.add_argument("--batches", required=True)
    run.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = freeze(args.reference_root, args.development_roots, args.output, development_pairs=args.development_pairs) if args.command == "freeze" else run_frozen(args.lock, args.batches, args.output_root)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
