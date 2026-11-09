"""Plan or submit the frozen teacher -> weak labels -> segmenter Slurm pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, resolve_path, write_rows
from shadow_occ.annotation.scene_sampling import partition_scenes
from shadow_occ.utils.io import write_json


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_pipeline(config_path, output_root, fraction=25):
    config_path, output = Path(config_path).resolve(), Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if fraction not in {25, 50, 100} or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose stage 25/50/100 and a new pipeline output root")
    repo = Path(__file__).resolve().parents[1]
    for key in ("freeze_lock", "dataroot", "external_root", "checkpoint", "train_ann_file", "val_ann_file"):
        config[key] = str(resolve_path(config[key], config_path.parent))
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])
    config["model_config"] = str(resolve_path(config.get("model_config", "projects/configs/stage1_track_map/base_track_map.py"), Path(config["external_root"])))
    if not Path(config["model_config"]).is_file():
        raise FileNotFoundError(config["model_config"])
    lock = json.loads(Path(config["freeze_lock"]).read_text(encoding="utf-8"))
    if not lock.get("context_model_signatures"):
        raise ValueError("Create a new freeze lock with context BEV provenance before planning jobs")
    for path, expected in lock["source_sha256"].items():
        if digest(repo/path) != expected:
            raise ValueError("Algorithm source changed; create a new freeze lock before submitting")
    if config.get("population"):
        population = json.loads(resolve_path(config["population"], config_path.parent).read_text(encoding="utf-8"))
    else:
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        population = {"official_train_scenes": splits["train"], "official_eval_scenes": splits["val"]}
    defaults = {"seed": 0, "dev_fraction": .15, "num_frames": 7, "stride": 3,
        "pairs_per_scene": 24, "working_width": 800, "version": "v1.0-trainval",
        "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        "methods": ["appearance_only", "anchor_first_v8plus"],
        "cpu_partition": "AMD", "gpu_partition": "NV_RTX4090", "cpu_parallel": 4, "bev_parallel": 1,
        "annotation_env": "shadow_annotator", "bev_env": "drivevla", "student_gpus": 1,
        "epochs": 10, "batch_size": 2, "accumulation": 4, "backbone_weights": "imagenet", "negative_weight": .5,
        "conda_sh": "/public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh"}
    config = {**defaults, **config}
    config["config_source"] = str(config_path)
    if config["version"] != "v1.0-trainval":
        raise ValueError("This official train/val pipeline requires v1.0-trainval")
    if set(population["official_train_scenes"]) & set(population["official_eval_scenes"]):
        raise ValueError("Official train and eval scenes must be disjoint")
    if config.get("init_checkpoint"):
        config["init_checkpoint"] = str(resolve_path(config["init_checkpoint"], config_path.parent))
        if not Path(config["init_checkpoint"]).is_file():
            raise FileNotFoundError(config["init_checkpoint"])
    if (min(config[key] for key in ("pairs_per_scene", "working_width", "cpu_parallel", "bev_parallel", "student_gpus", "epochs", "batch_size", "accumulation")) < 1
            or not config["cameras"] or not 3 <= config["num_frames"] <= 31 or config["stride"] < 1
            or config["working_width"] < 64 or config["pairs_per_scene"] < 2
            or set(config["methods"])-{"appearance_only", "anchor_first_v8plus"}
            or len(set(config["methods"])) != len(config["methods"])
            or "anchor_first_v8plus" not in config["methods"]):
        raise ValueError("Invalid automatic pipeline dimensions or actual teacher methods")
    dev, stages = partition_scenes(population["official_train_scenes"], config["seed"], config["dev_fraction"])
    evaluation = set(population["official_eval_scenes"])-set(lock["excluded_development_scenes"])
    if not evaluation:
        raise ValueError("No independent official eval scenes remain")
    roles = {scene: "train" for scene in stages[fraction]}
    roles.update({scene: "dev" for scene in dev})
    roles.update({scene: "eval" for scene in evaluation})
    parent = None
    if config.get("previous_run"):
        parent_path = resolve_path(config["previous_run"], config_path.parent)
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if (parent["fraction"], fraction) not in {(25, 50), (50, 100)}:
            raise ValueError("Expand a completed 25% run to 50%, then 50% to 100%")
    elif fraction != 25:
        raise ValueError("Stage 50/100 requires previous_run pointing to the preceding pipeline.json")
    for key, name in (("cache_root", "cache"), ("annotation_root", "annotations")):
        default = parent["config"][key] if parent else str(output/name)
        config[key] = str(resolve_path(config.get(key, default), config_path.parent))
    annotation_keys = ("dataroot", "external_root", "model_config", "checkpoint", "train_ann_file", "val_ann_file",
        "seed", "dev_fraction", "num_frames", "stride", "pairs_per_scene", "working_width", "cameras", "methods",
        "version", "cache_root", "annotation_root")
    annotation_signature = signature({"config": {key: config[key] for key in annotation_keys}, "population": population})
    if parent and (parent["annotation_signature"] != annotation_signature or parent["freeze_lock_sha256"] != digest(config["freeze_lock"])):
        raise ValueError("Expansion must retain the preceding scene/teacher/cache protocol")
    if parent:
        for stage in ("shared", f"stage_{parent['fraction']:03d}"):
            path = Path(config["annotation_root"])/stage/"frozen_run.json"
            if not path.is_file() or not json.loads(path.read_text()).get("batches") or any(row["status"] != "completed" for row in json.loads(path.read_text())["batches"]):
                raise ValueError("Previous shared/train annotations are not complete")
        status_path = Path(parent["output_root"])/"student/suite_status.json"
        if not status_path.is_file() or not json.loads(status_path.read_text()).get("all_arms_completed") or json.loads(status_path.read_text()).get("metrics_exit_code", 1):
            raise ValueError("Previous student training/evaluation must finish before scaling")
    tasks = []
    for scene, role in sorted(roles.items()):
        root = Path(config["cache_root"])/scene
        tasks.append({"scene_name": scene, "split": role, "official_split": "val" if role == "eval" else "train",
            "pairs": str(root/"pairs/pairs.jsonl"), "analysis_root": str(root/"flow"),
            "feature_root": str(root/"temporal_bev"), "spatial_feature_root": str(root/"current_bev")})
    output.mkdir(parents=True)
    write_json(output/"population.json", population)
    write_rows(output/"cache_tasks.jsonl", tasks)
    plan = {"protocol": "automatic-shadow-slurm-pipeline-v1", "project_root": str(repo), "output_root": str(output),
        "fraction": fraction, "config": config, "freeze_lock_sha256": digest(config["freeze_lock"]),
        "annotation_signature": annotation_signature, "reuse_shared_annotations": parent is not None,
        "target_scenes": {role: sum(value == role for value in roles.values()) for role in ("train", "dev", "eval")},
        "tasks": str(output/"cache_tasks.jsonl"), "raw_nuscenes_copied": False,
        "jobs_submitted": False, "vlm_training": False, "occ_training": False}
    write_json(output/"pipeline.json", plan)
    return plan


def job_specs(plan):
    config, output = plan["config"], Path(plan["output_root"])
    repo = Path(plan["project_root"])
    common = {"SHADOW_PROJECT_ROOT": str(repo), "SHADOW_CONDA_ENV": config["annotation_env"],
        "SHADOW_CONDA_SH": config["conda_sh"], "SHADOW_AUTO_JOB": str(output/"pipeline.json")}
    count = len(read_rows(plan["tasks"]))
    resources = {"cpu": ["--partition", config["cpu_partition"], "--cpus-per-task", "4", "--mem", "32G"],
        "gpu": ["--partition", config["gpu_partition"], "--cpus-per-task", "8", "--mem", "48G", "--gres", "gpu:1"]}
    specs = []
    for name, phase, mode, parallel, after in (("cache_cpu", "cpu", "cpu", config["cpu_parallel"], []),
        ("cache_bev", "bev", "gpu", config["bev_parallel"], ["cache_cpu"]),
        ("plan", "plan", "cpu", None, ["cache_bev"])):
        env = {**common, "SHADOW_CACHE_PHASE": phase,
            "SHADOW_CONDA_ENV": config["bev_env"] if phase == "bev" else config["annotation_env"]}
        args = resources[mode]+(["--array", f"0-{count-1}%{parallel}"] if parallel else [])
        specs.append({"name": name, "after": after, "script": str(repo/"slurm/prepare_automatic_shadow_cache.slurm"), "args": args, "env": env})
    shared_env = {**common, "SHADOW_AUTO_PLAN": str(output/"plan"), "SHADOW_VIDEO_FREEZE_LOCK": config["freeze_lock"],
        "SHADOW_AUTO_DATA_ROOT": str(output/"data"), "SHADOW_AUTO_SUITE_ROOT": str(output/"suite"),
        "SHADOW_SEG_EPOCHS": str(config["epochs"]), "SHADOW_SEG_BATCH_SIZE": str(config["batch_size"]),
        "SHADOW_SEG_ACCUMULATION": str(config["accumulation"]), "SHADOW_SEED": str(config["seed"]),
        "SHADOW_SEG_BACKBONE_WEIGHTS": config["backbone_weights"], "SHADOW_NEGATIVE_WEIGHT": str(config["negative_weight"]),
        "SHADOW_SEG_METHODS": " ".join(config["methods"])}
    if config.get("init_checkpoint"):
        shared_env["SHADOW_SEG_INIT_CHECKPOINT"] = str(resolve_path(config["init_checkpoint"], Path(config["config_source"]).parent))
    if not plan["reuse_shared_annotations"]:
        specs.append({"name": "shared", "after": ["plan"], "script": str(repo/"slurm/prepare_automatic_shadow_stage.slurm"),
            "args": resources["cpu"], "env": {**shared_env, "SHADOW_AUTO_STAGE": "shared"}})
    specs.append({"name": "data", "after": ["plan"] if plan["reuse_shared_annotations"] else ["shared"],
        "script": str(repo/"slurm/prepare_automatic_shadow_stage.slurm"), "args": resources["cpu"],
        "env": {**shared_env, "SHADOW_AUTO_STAGE": f"stage_{plan['fraction']:03d}"}})
    student_args = resources["gpu"][:-2]+["--gres", f"gpu:{config['student_gpus']}"]
    specs.append({"name": "student", "after": ["data"], "script": str(repo/"slurm/run_shadow_segmenter_suite.slurm"),
        "args": student_args, "env": {**common, "SHADOW_SEG_SUITE": str(output/"suite/suite.json"),
            "SHADOW_SEG_SUITE_OUTPUT": str(output/"student"), "SHADOW_SEG_GPUS": str(config["student_gpus"])}})
    return specs


def submit_jobs(plan, submit=False):
    output = Path(plan["output_root"])
    previous = output/"jobs.json"
    if previous.is_file() and json.loads(previous.read_text()).get("jobs_submitted"):
        raise ValueError("This pipeline already has submitted jobs; inspect jobs.json instead of submitting duplicates")
    if submit:
        lock_path = plan["config"]["freeze_lock"]
        if digest(lock_path) != plan["freeze_lock_sha256"]:
            raise ValueError("Freeze lock changed after pipeline planning")
        for path, expected in json.loads(Path(lock_path).read_text())["source_sha256"].items():
            if digest(Path(plan["project_root"])/path) != expected:
                raise ValueError("Source changed after pipeline planning; create a new freeze lock and plan")
    output.joinpath("logs").mkdir(exist_ok=True)
    status = {"jobs_submitted": False, "jobs": []}
    ids = {}
    for spec in job_specs(plan):
        command = ["sbatch", "--parsable", "--export=ALL", "--job-name", "shadow_"+spec["name"], "--chdir", plan["project_root"],
            "--time", "2-00:00:00", "--output", str(output/"logs/%x_%A_%a.out"), "--error", str(output/"logs/%x_%A_%a.err"), *spec["args"]]
        if spec["after"]:
            command += ["--dependency", "afterok:"+":".join(ids.get(key, "<"+key+">") for key in spec["after"])]
        command += [spec["script"]]
        record = {**spec, "command": command, "job_id": None}
        if submit:
            env = dict(os.environ)
            for key in ("SHADOW_EVAL_GT", "SHADOW_SEG_INIT_CHECKPOINT", "SHADOW_SEG_METHODS", "SHADOW_AUTO_STAGE"):
                env.pop(key, None)
            env.update(spec["env"])
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {result.stdout}")
            ids[spec["name"]] = job_id
            record["job_id"] = job_id
        status["jobs"].append(record)
        status["jobs_submitted"] = bool(ids)
        write_json(output/"jobs.json", status)
    return status


def run_command(command, repo):
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=str(repo), check=True)


def cached_phase(stamp_path, request, commands, artifacts, repo):
    if stamp_path.is_file():
        stamp = json.loads(stamp_path.read_text())
        if stamp["request"] != request:
            raise ValueError("Cache protocol changed; use a new cache_root instead of overwriting existing evidence")
        if not all(Path(path).is_file() and digest(path) == sha for path, sha in stamp["manifest_sha256"].items()):
            raise ValueError("Completed cache metadata changed")
        if not all(Path(path).is_file() for path in stamp["required_payloads"]):
            raise ValueError("Completed cache payloads are missing")
        print(f"Reusing completed cache: {stamp_path}", flush=True)
        return
    started = stamp_path.with_suffix(".request.json")
    if started.is_file():
        if json.loads(started.read_text())["request"] != request:
            raise ValueError("Partial cache belongs to a different request; use a new cache_root")
    else:
        if any(path.exists() for _, path in commands):
            raise ValueError("Existing unstamped cache must be imported/verified separately; do not adopt it silently")
        write_json(started, {"request": request})
    for command, protocol_path in commands:
        if not protocol_path.is_file():
            run_command(command, repo)
    manifests, payloads = artifacts()
    if not all(Path(path).is_file() for path in payloads):
        raise ValueError("Cache phase did not produce all required payload files")
    write_json(stamp_path, {"request": request, "manifest_sha256": {str(path): digest(path) for path in manifests},
        "required_payloads": [str(path) for path in payloads],
        "payload_validation": "existence here; discovery verifies original payload hashes before use"})


def run_worker(job_path, phase, index=0):
    job_path = Path(job_path).resolve()
    plan = json.loads(job_path.read_text())
    config, output, repo = plan["config"], Path(plan["output_root"]), Path(plan["project_root"])
    tasks = read_rows(plan["tasks"])
    python = sys.executable
    if phase == "plan":
        catalogue = output/"catalogue.jsonl"
        write_rows(catalogue, [{key: task[key] for key in ("pairs", "analysis_root", "feature_root", "spatial_feature_root")} for task in tasks])
        run_command([python, str(repo/"scripts/plan_automatic_shadow_scenes.py"), "--catalogue", str(catalogue),
            "--lock", config["freeze_lock"], "--output-root", str(output/"plan"), "--annotation-root", config["annotation_root"],
            "--population", str(output/"population.json"), "--seed", str(config["seed"]), "--dev-fraction", str(config["dev_fraction"]),
            "--num-frames", str(config["num_frames"]), "--stride", str(config["stride"]), "--methods", *config["methods"]], repo)
        return
    task = tasks[index]
    pairs, flow = Path(task["pairs"]), Path(task["analysis_root"])
    root = pairs.parent.parent
    if phase == "cpu":
        pair_command = [python, str(repo/"scripts/export_nuscenes_motion_pairs.py"), "--dataroot", config["dataroot"],
            "--version", config["version"], "--split", task["official_split"], "--scene", task["scene_name"],
            "--cams", *config["cameras"], "--max-pairs", str(config["pairs_per_scene"]),
            "--max-pairs-per-scene", str(config["pairs_per_scene"]), "--max-attempts", "12000",
            "--working-width", str(config["working_width"]), "--seed", str(config["seed"]), "--output-root", str(pairs.parent)]
        flow_command = [python, str(repo/"scripts/analyze_shadow_motion_pairs.py"), "--pairs", str(pairs),
            "--flow-backend", "dis", "--device", "cpu", "--review-limit", "0", "--output-root", str(flow)]
        def artifacts():
            images = {resolve_path(frame[key], pairs.parent) for row in read_rows(pairs) for frame in row["frames"] for key in ("image", "ground_support")}
            return [pairs, pairs.parent/"export_protocol.json", flow/"analysis_protocol.json", flow/"pair_metrics.jsonl"], \
                sorted(images)+[resolve_path(row["evidence"], flow) for row in read_rows(flow/"pair_metrics.jsonl")]
        request = signature({"pair": pair_command[1:], "flow": flow_command[1:],
            "source": {name: digest(repo/name) for name in ("scripts/export_nuscenes_motion_pairs.py", "scripts/analyze_shadow_motion_pairs.py",
                "shadow_occ/annotation/frame_receiver.py", "shadow_occ/annotation/physics_motion.py", "shadow_occ/annotation/optical_flow.py", "shadow_occ/data/nuscenes_adapter.py")}})
        cached_phase(root/"cpu_complete.json", request, [(pair_command, pairs.parent/"export_protocol.json"),
            (flow_command, flow/"analysis_protocol.json")], artifacts, repo)
    elif phase == "bev":
        commands = []
        for mode, key in (("temporal", "feature_root"), ("current-only", "spatial_feature_root")):
            command = [python, str(repo/"scripts/export_mature_bev.py"), "--external-root", config["external_root"],
                "--model-config", config["model_config"], "--checkpoint", config["checkpoint"],
                "--dataroot", config["dataroot"], "--version", config["version"], "--pairs", str(pairs),
                "--ann-file", config["val_ann_file"] if task["official_split"] == "val" else config["train_ann_file"],
                "--history-mode", mode, "--output-root", task[key]]
            commands.append((command, Path(task[key])/"cache_protocol.json"))
        def artifacts():
            tokens = {row[key] for row in read_rows(pairs) for key in ("sample_token", "next_sample_token")}
            roots = [Path(task[key]) for key in ("feature_root", "spatial_feature_root")]
            return [path/"cache_protocol.json" for path in roots]+[path/(token+".json") for path in roots for token in sorted(tokens)], \
                [path/(token+".npz") for path in roots for token in sorted(tokens)]
        request = signature({"commands": [command[1:] for command, _ in commands], "pairs": digest(pairs),
            "model_config": digest(config["model_config"]), "freeze": plan["freeze_lock_sha256"],
            "exporter": digest(repo/"scripts/export_mature_bev.py")})
        cached_phase(root/"bev_complete.json", request, commands, artifacts, repo)
    else:
        raise ValueError("Unknown worker phase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("prepare")
    start.add_argument("--config", required=True)
    start.add_argument("--output-root", required=True)
    start.add_argument("--fraction", type=int, choices=[25, 50, 100], default=25)
    start.add_argument("--submit", action="store_true")
    launch = sub.add_parser("submit")
    launch.add_argument("--job", required=True, help="Previously inspected dry-run pipeline.json")
    worker = sub.add_parser("worker")
    worker.add_argument("--job", required=True)
    worker.add_argument("--phase", choices=["cpu", "bev", "plan"], required=True)
    worker.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "submit":
        print(json.dumps(submit_jobs(json.loads(Path(args.job).read_text()), submit=True)), flush=True)
    elif args.command == "worker":
        run_worker(args.job, args.phase, args.index)
    else:
        plan = prepare_pipeline(args.config, args.output_root, args.fraction)
        print(json.dumps(submit_jobs(plan, args.submit)), flush=True)


if __name__ == "__main__":
    main()
