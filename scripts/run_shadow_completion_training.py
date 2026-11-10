"""Expand cached scenes, annotate with motion/illumination evidence, train students."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.utils.io import write_json


def diverse_clips(clips, limit, seed):
    """Round-robin cameras, then deterministic windows; no teacher score selection."""
    groups = {}
    for clip in clips:
        groups.setdefault(clip["camera"], []).append(clip)
    order = lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()
    groups = {camera: sorted(rows, key=order) for camera, rows in groups.items()}
    result = []
    while groups and len(result) < limit:
        for camera in sorted(list(groups)):
            result.append(groups[camera].pop(0))
            if not groups[camera]:
                del groups[camera]
            if len(result) == limit:
                break
    return result


def prepare(source_plan, output_root, counts, clips_per_scene=3, seed=0, epochs=10):
    if set(counts) != {"train", "dev", "eval"} or min(*counts.values(), clips_per_scene, epochs) < 1:
        raise ValueError("Choose positive scene/clip counts and training epochs")
    source, output = Path(source_plan).resolve(), Path(output_root).resolve()
    runs = read_rows(source/"stage_025/experiment_runs.jsonl")
    references = {(row["split"], str(Path(row["pairs"][0]).resolve())): row["run_root"]
                  for row in runs if row["method"] == "anchor_first_v8plus"}
    batches = read_rows(source/"stage_025/batches.jsonl")+read_rows(source/"shared/batches.jsonl")
    available = {role: {} for role in counts}
    for batch in batches:
        if batch["method"] != "anchor_first_v8plus":
            continue
        reference = references.get((batch["split"], str(Path(batch["pairs"]).resolve())))
        if not reference or not (Path(reference)/"summary.json").is_file():
            continue
        clips = read_rows(batch["clips"])
        if not clips:
            continue
        scenes = {clip["scene_name"] for clip in clips}
        if len(scenes) != 1:
            raise ValueError("Each source batch must identify one scene")
        scene = next(iter(scenes))
        if scene in available[batch["split"]]:
            raise ValueError("Duplicate ready scene in source plan")
        available[batch["split"]][scene] = (batch, reference, clips)
    selected = {}
    for split, count in counts.items():
        order = sorted(available[split], key=lambda scene: hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest())
        if len(order) < count:
            raise ValueError(f"{split}: {len(order)} cached scenes available, requested {count}")
        selected[split] = order[:count]
    if any(set(selected[a]) & set(selected[b]) for a, b in (("train", "dev"), ("train", "eval"), ("dev", "eval"))):
        raise ValueError("Scene roles overlap")
    output.mkdir(parents=True, exist_ok=False)
    planned, pairs_all, clips_all, fingerprints = [], {}, [], {}
    for split, scenes in selected.items():
        for scene in scenes:
            batch, reference, candidates = available[split][scene]
            clips = diverse_clips(candidates, clips_per_scene, seed)
            path = output/"clips"/(batch["id"]+".jsonl")
            write_rows(path, clips)
            planned.append({**batch, "scene_name": scene, "clips": str(path), "core_reference": reference,
                            "output": str(output/"discovery"/batch["id"])})
            for name in (batch["pairs"], batch["clips"], str(Path(reference)/"summary.json")):
                fingerprints[name] = digest(name)
            for pair in read_rows(batch["pairs"]):
                if pair["id"] in pairs_all and pairs_all[pair["id"]] != pair:
                    raise ValueError("Pair identity collision")
                pairs_all[pair["id"]] = pair
            clips_all.extend(clips)
    write_rows(output/"pairs.jsonl", pairs_all.values())
    write_rows(output/"clips.jsonl", clips_all)
    write_rows(output/"batches.jsonl", planned)
    plan = {"protocol": "cached-motion-illumination-training-v1", "source_plan": str(source),
        "scene_sets": selected, "available_cached_scenes": {role: len(rows) for role, rows in available.items()},
        "clips": len(clips_all), "clips_per_scene": clips_per_scene, "seed": seed, "epochs": epochs,
        "sampling": "deterministic scene sampling, camera round-robin; no positive-count or eval-score selection",
        "nuScenes_or_BEV_download_required": False, "new_prior_inference_required": True,
        "source_fingerprints": fingerprints,
        "manifest_fingerprints": {str(output/name): digest(output/name) for name in ("pairs.jsonl", "clips.jsonl", "batches.jsonl")}}
    write_json(output/"plan.json", plan)
    return plan


def run(plan_path, assets, device="cuda", stages=("prior", "annotation", "train"), resume=False, workers=1, prior_manifest=None):
    path = Path(plan_path).resolve()
    root, plan = path.parent, json.loads(path.read_text(encoding="utf-8"))
    for name, expected in {**plan["source_fingerprints"], **plan["manifest_fingerprints"]}.items():
        if digest(name) != expected:
            raise ValueError("Prepared training inputs changed: "+name)
    repo, assets = Path(__file__).resolve().parents[1], Path(assets).resolve()
    prior_manifest = Path(prior_manifest).resolve() if prior_manifest else root/"prior/manifest.jsonl"
    if workers < 1:
        raise ValueError("Choose positive annotation workers")
    status_path = root/("pipeline_"+"_".join(stages)+"_status.json")
    status = {"plan_sha256": digest(path), "stages": list(stages), "steps": [], "training_performed": False}
    lock = threading.Lock()
    def execute(script, arguments):
        command = [sys.executable, "-u", str(repo/"scripts"/script), *map(str, arguments)]
        step = {"script": script, "command": command, "status": "running"}
        with lock:
            status["steps"].append(step)
            write_json(status_path, status)
        result = subprocess.run(command, cwd=repo)
        with lock:
            step.update(exit_code=result.returncode, status="completed" if result.returncode == 0 else "failed")
            write_json(status_path, status)
        if result.returncode:
            raise RuntimeError(f"{script} exited {result.returncode}")
    if "prior" in stages and prior_manifest == root/"prior/manifest.jsonl":
        if not (resume and (root/"prior/manifest.jsonl").is_file()):
            execute("export_fsd_shadow_priors.py", ["--pairs", root/"pairs.jsonl", "--clips", root/"clips.jsonl",
                "--fsd-root", assets/"source", "--checkpoint", assets/"FSD_best.ckpt", "--output-root", root/"prior"])
    runs = []
    batches = read_rows(root/"batches.jsonl")
    def annotate(batch):
        output = Path(batch["output"]).resolve()
        if output.parent != (root/"discovery").resolve():
            raise ValueError("Annotation output must stay inside this prepared run")
        if resume and (output/"summary.json").is_file():
            from shadow_occ.annotation.video_review import find_clips
            entries = find_clips(output)
            if {item[1]["clip"]["id"] for item in entries.values()} != {item["id"] for item in read_rows(batch["clips"])}:
                raise ValueError("Completed annotation has different selected clips")
            from shadow_occ.annotation.shadow_completion import COMPLETION_PROTOCOL
            if any(item[1]["protocol"] != COMPLETION_PROTOCOL for item in entries.values()):
                raise ValueError("Cannot resume a different teacher protocol")
            return
        if resume and output.exists():
            output.rename(output.with_name(output.name+".partial_"+datetime.now().strftime("%Y%m%d_%H%M%S")))
        arguments = []
        for field in ("pairs", "clips", "analysis_root", "feature_root", "spatial_feature_root"):
            arguments.extend(["--"+field.replace("_", "-"), batch[field]])
        arguments.extend(["--shadow-prior-manifest", prior_manifest, "--core-reference-root", batch["core_reference"],
                          "--output-root", batch["output"], "--stage", "full", "--discovery-mode", "cross-space"])
        execute("discover_shadow_video.py", arguments)
    if "annotation" in stages:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(annotate, batches))
    for batch in batches:
        for method, output in (("v8_core", batch["core_reference"]), ("motion_illumination", batch["output"])):
            runs.append({"method": method, "split": batch["split"], "run_root": output, "pairs": [batch["pairs"]],
                         "clip_ids": [clip["id"] for clip in read_rows(batch["clips"])]})
    write_rows(root/"experiment_runs.jsonl", runs)
    if "train" not in stages and "dataset" not in stages:
        status["completed"] = True
        write_json(status_path, status)
        return status
    if "dataset" in stages or not (root/"suite/suite.json").is_file():
        execute("build_shadow_mainline_experiments.py", ["export", "--runs", root/"experiment_runs.jsonl",
            "--output-root", root/"dataset", "--auto-negatives"])
        execute("build_shadow_mainline_experiments.py", ["suite", "--dataset-root", root/"dataset", "--methods",
            "v8_core", "motion_illumination", "--reference-method", "motion_illumination", "--output-root", root/"suite",
            "--epochs", plan["epochs"], "--seed", plan["seed"], "--batch-size", "2", "--accumulation", "4"])
    if "train" not in stages:
        status["completed"] = True
        write_json(status_path, status)
        return status
    status["training_performed"] = True
    execute("run_shadow_segmenter_suite.py", ["--plan", root/"suite/suite.json", "--output-root", root/"training", "--device", device])
    status["completed"] = True
    write_json(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-plan", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    for split, default in (("train", 8), ("dev", 3), ("eval", 3)):
        prepare_parser.add_argument("--"+split+"-scenes", type=int, default=default)
    prepare_parser.add_argument("--clips-per-scene", type=int, default=3)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--epochs", type=int, default=10)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--assets", required=True)
    run_parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run_parser.add_argument("--stages", nargs="+", choices=["prior", "annotation", "dataset", "train"], default=["prior", "annotation", "dataset", "train"])
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--annotation-workers", type=int, default=1)
    run_parser.add_argument("--prior-manifest", help="Reuse raw FSD observations, never old training targets")
    args = parser.parse_args()
    result = (prepare(args.source_plan, args.output_root, {split: getattr(args, split+"_scenes") for split in ("train", "dev", "eval")},
                      args.clips_per_scene, args.seed, args.epochs) if args.command == "prepare" else
              run(args.plan, args.assets, args.device, args.stages, args.resume, args.annotation_workers, args.prior_manifest))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
