"""Run the existing segmenter with one shared E2 training budget per weak teacher."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.utils.io import write_json


def run_suite(plan_path, output_root, ground_truth=None, gpus=1, device="cuda"):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[1]
    if gpus < 1 or device not in {"cuda", "cpu"}:
        raise ValueError("Choose a positive GPU process count or explicit CPU mode")
    for path, expected in plan["manifest_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Aligned training manifests changed after suite preparation")
    if plan.get("init_checkpoint") and digest(plan["init_checkpoint"]) != plan["init_checkpoint_sha256"]:
        raise ValueError("Shared initialization checkpoint changed")
    for path, expected in plan.get("source_suite_sha256", {}).items():
        if digest(path) != expected:
            raise ValueError("Source comparison suite changed")
    if plan.get("binary_observations"):
        checked = {}
        def content_hash(path):
            if path not in checked:
                checked[path] = digest(path)
            return checked[path]
        for arm in plan["arms"]:
            for split in ("train", "dev"):
                expected = {row["id"]: row for row in plan["binary_observations"][split]}
                rows = read_rows(arm[split])
                if {row["id"] for row in rows} != expected.keys():
                    raise ValueError("Paired binary observation IDs changed")
                for row in rows:
                    if (content_hash(row["target"]) != expected[row["id"]]["target_sha256"]
                            or content_hash(row["image"]) != expected[row["id"]]["image_sha256"]):
                        raise ValueError("Paired RGB/trimap contents changed")
    evaluation = []
    reference_kind = "manual" if ground_truth else "frozen-teacher"
    if ground_truth:
        base = Path(ground_truth).resolve().parent
        exposed = set(plan["scene_sets"]["train"]) | set(plan["scene_sets"]["dev"])
        for row in read_rows(ground_truth):
            if row.get("label_source") != "manual_shadow_gt" or row.get("split") != "eval" or row.get("independent") is not True:
                raise ValueError("Only independent manual eval masks may be used for final inference")
            if row["scene_name"] in exposed:
                raise ValueError("Manual eval overlaps training or checkpoint-selection scenes")
            evaluation.append((row, base))
        if not evaluation:
            raise ValueError("Manual evaluation manifest is empty")
    elif plan.get("weak_eval_reference"):
        reference = Path(plan["weak_eval_reference"])
        if digest(reference) != plan["weak_eval_reference_sha256"]:
            raise ValueError("Frozen held-out teacher reference changed")
        exposed = set(plan["scene_sets"]["train"]) | set(plan["scene_sets"]["dev"])
        for row in read_rows(reference):
            if row["scene_name"] in exposed or row.get("split") != "eval" or row.get("label_source") not in {"temporal_weak", "geometry_weak"}:
                raise ValueError("Weak agreement evaluation must use held-out physical teacher labels")
            evaluation.append((row, reference.parent))
        if not evaluation:
            raise ValueError("Frozen teacher evaluation manifest is empty")
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    eval_path = None
    if evaluation:
        normalized = []
        for index, (row, base) in enumerate(evaluation):
            target = load_array(row["target"], base)
            if reference_kind == "manual" and row.get("mask_encoding", "trimap") == "binary":
                target = (target > 0).astype(np.uint8)
            if not np.isin(target, [0, 1, 255]).all():
                raise ValueError("Evaluation requires 0/1/255 trimaps")
            image = str(resolve_path(row["image"], base))
            path = output/("manual_eval_targets" if reference_kind == "manual" else "weak_eval_targets")/f"{index:07d}.png"
            checked_write_image(path, target)
            normalized.append({**row, "target": str(path), "image": image, "mask_encoding": "trimap"})
            del target
        eval_path = output/("manual_eval.jsonl" if reference_kind == "manual" else "weak_eval.jsonl")
        write_rows(eval_path, normalized)
    status = {"suite_sha256": digest(plan_path), "gpus": gpus, "device": device, "arms": [],
        "evaluation_gt_used_for_training": False, "architecture": plan["architecture"],
        "equal_image_and_epoch_budget": True, "ground_truth_sha256": digest(ground_truth) if ground_truth else None}
    status["evaluation_reference_kind"] = reference_kind if evaluation else None
    status["is_true_shadow_accuracy"] = bool(evaluation) and reference_kind == "manual"
    evaluated = []
    shared_initializations, paired_orders = {}, {}
    for arm in plan["arms"]:
        settings = {**plan, **arm.get("training", {})}
        method, folder = arm["method"], output/arm["method"]
        if Path(method).name != method or method in {".", ".."} or "\\" in method:
            raise ValueError("Invalid method directory")
        command = [sys.executable]
        if device == "cuda" and gpus > 1:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(gpus)]
        command += [str(repo/"scripts/train_shadow_segmenter.py"), "--train", arm["train"], "--dev", arm["dev"],
                    "--output-root", str(folder), "--allow-weak-labels", "--device", device]
        for field in ("epochs", "batch_size", "accumulation", "seed", "lr", "backbone_weights", "temporal_weight", "receiver_pair_weight", "weak_weight"):
            command += ["--"+field.replace("_", "-"), str(settings[field])]
        for field in ("scene_sampling", "pixel_balance", "dice_weight", "loss_mode", "selection_metric",
                      "supervision_mode", "amp", "workers", "inherited_weight", "expanded_weight"):
            if field in settings:
                command += ["--"+field.replace("_", "-"), str(settings[field])]
        if settings.get("deterministic"):
            command += ["--deterministic"]
        for field in ("expansion_warmup_epochs", "expansion_ramp_epochs"):
            if field in settings:
                command += ["--"+field.replace("_", "-"), str(settings[field])]
        command += ["--size", *map(str, plan["size_hw"])]
        initialization = plan.get("init_checkpoint")
        if plan.get("paired_initialization"):
            import torch
            from shadow_occ.annotation.segmentation import build_segmenter, model_state_sha256
            from shadow_occ.utils.seed import seed_everything
            seed = settings["seed"]
            if seed not in shared_initializations:
                seed_everything(seed)
                initial = build_segmenter(settings["backbone_weights"])
                initialization = output/f"initialization_seed_{seed}.pth"
                torch.save({"model": initial.state_dict(), "train_scenes": [], "dev_scenes": []}, initialization)
                shared_initializations[seed] = (str(initialization), model_state_sha256(initial))
                del initial
            initialization = shared_initializations[seed][0]
        if initialization:
            command += ["--init-checkpoint", str(initialization)]
        record = {"method": method, "train_command": command, "status": "training"}
        status["arms"].append(record)
        write_json(output/"suite_status.json", status)
        result = subprocess.run(command, cwd=str(repo))
        record["train_exit_code"] = result.returncode
        if (folder/"history.json").is_file():
            shutil.copy2(folder/"history.json", folder/"segmenter_history.json")
        if result.returncode or not (folder/"best.pth").is_file():
            record["status"] = "training_failed_or_collapsed"
            write_json(output/"suite_status.json", status)
            continue
        record["status"] = "trained"
        record["checkpoint_sha256"] = digest(folder/"best.pth")
        if plan.get("paired_initialization"):
            repro = json.loads((folder/"reproducibility.json").read_text(encoding="utf-8"))
            if repro["initial_state_sha256"] != shared_initializations[settings["seed"]][1]:
                raise ValueError("Paired arm did not load the shared model initialization")
            history = json.loads((folder/"history.json").read_text(encoding="utf-8"))
            order = [row["train_loss"]["sample_order_sha256"] for row in history]
            if settings["seed"] in paired_orders and order != paired_orders[settings["seed"]]:
                raise ValueError("Paired arms used different sample orders")
            paired_orders[settings["seed"]] = order
            record.update(seed=settings["seed"], initial_state_sha256=repro["initial_state_sha256"],
                          sample_order_sha256=order)
        try:
            os.link(folder/"best.pth", folder/"checkpoint_best.pth")
        except OSError:
            shutil.copy2(folder/"best.pth", folder/"checkpoint_best.pth")
        if eval_path:
            prediction_root = output/(method+"_eval")
            predict = [sys.executable, str(repo/"scripts/evaluate_shadow_segmenter.py"),
                       "--manifest", str(eval_path), "--checkpoint", str(folder/"best.pth"),
                       "--output-root", str(prediction_root), "--device", device, "--threshold", "0.5"]
            record["eval_command"] = predict
            result = subprocess.run(predict, cwd=str(repo))
            record["eval_exit_code"] = result.returncode
            record["status"] = "inferred" if result.returncode == 0 else "inference_failed"
            if result.returncode == 0:
                evaluated.append({"method": method, "experiment": "E2" if reference_kind == "manual" else "TeacherAgreement", "predictions": str(prediction_root/"predictions.jsonl"),
                                  "training_protocol": str(folder/"protocol.json")})
        write_json(output/"suite_status.json", status)
    write_rows(output/"evaluation_methods.jsonl", evaluated)
    if evaluated:
        metrics = [sys.executable, str(repo/"scripts/evaluate_shadow_mainline_experiments.py"),
                   "--ground-truth", str(eval_path), "--methods", str(output/"evaluation_methods.jsonl"),
                   "--dataset-summary", plan["dataset_summary"], "--output-root", str(output/"metrics"),
                   "--reference-kind", reference_kind]
        result = subprocess.run(metrics, cwd=str(repo))
        status["metrics_exit_code"] = result.returncode
        if result.returncode == 0 and plan.get("paired_initialization"):
            results = json.loads((output/"metrics/metrics.json").read_text(encoding="utf-8"))["results"]
            groups = {}
            metadata = {arm["method"]: arm for arm in plan["arms"]}
            for row in results:
                arm = metadata[row["method"]]
                whole = json.loads((output/(row["method"]+"_eval")/"full_image_summary.json").read_text(encoding="utf-8"))
                groups.setdefault(arm["comparison_group"], []).append({"seed": arm["training"]["seed"],
                    "balanced_accuracy": row["balanced_accuracy"], "negative_recall": row["negative_recall"],
                    "iou": row["iou"], "auroc": row["score_diagnostics"]["approximate_auroc"],
                    "full_image_fraction": whole["scene_macro_predicted_shadow_fraction"]})
            summaries = {}
            for group, rows in groups.items():
                summaries[group] = {"seeds": rows, "metrics": {}}
                for key in ("balanced_accuracy", "negative_recall", "iou", "auroc", "full_image_fraction"):
                    values = [row[key] for row in rows if row[key] is not None]
                    summaries[group]["metrics"][key] = {"n": len(values),
                        "mean": statistics.mean(values) if values else None,
                        "std": statistics.stdev(values) if len(values) > 1 else None,
                        "min": min(values) if values else None, "max": max(values) if values else None}
            write_json(output/"paired_seed_summary.json", {"groups": summaries,
                "scope": "paired-seed teacher agreement, not true shadow accuracy"})
    status["all_arms_completed"] = all(row["status"] == ("inferred" if eval_path else "trained") for row in status["arms"])
    write_json(output/"suite_status.json", status)
    with (output/"training_status.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["method", "status", "train_exit_code", "eval_exit_code"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in status["arms"])
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ground-truth")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    status = run_suite(args.plan, args.output_root, args.ground_truth, args.gpus, args.device)
    print(json.dumps(status), flush=True)
    if not status["all_arms_completed"] or status.get("metrics_exit_code", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
