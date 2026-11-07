"""Rebuild ablation labels/datasets from a completed shared BEV-flow preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadow_occ.annotation.data import digest
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(ROOT/"configs/bev_flow_ablations.json"))
    parser.add_argument("--variants", nargs="+", default=["full", "geometry_only", "shuffled", "matched_random", "cosine_01", "cosine_05"])
    parser.add_argument("--current-cache-root", help="Contains train/ and val/ caches exported with --history-mode current-only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels-only", action="store_true", help="Skip segmenter/VLM dataset crop generation")
    args = parser.parse_args()
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if len(set(args.variants)) != len(args.variants) or any(name not in config["variants"] for name in args.variants):
        parser.error("Select unique named variants from the config")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if any(config["variants"][name].get("feature_cache") == "current-only" for name in args.variants) and not args.current_cache_root:
        parser.error("current_only requires a separately exported --current-cache-root")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__).resolve(),
        ROOT/"scripts/reselect_bev_flow.py", ROOT/"scripts/build_physics_shadow_labels.py",
        ROOT/"scripts/build_bev_flow_annotation_manifest.py", ROOT/"shadow_occ/annotation/cached_bev.py",
        ROOT/"shadow_occ/annotation/physics_labels.py", ROOT/"shadow_occ/annotation/bev_teacher.py",
        ROOT/"shadow_occ/annotation/physics_motion.py"]}

    def run(script, *values, check=True):
        command = [sys.executable, str(ROOT/"scripts"/script), *map(str, values)]
        print("RUN " + " ".join(command), flush=True)
        return subprocess.run(command, check=check, cwd=ROOT)

    for name in args.variants:
        recipe = {"feature_gate": "aligned", "flow_gate": "registered", "minimum_bev_cosine": 0.3,
            "registration": "geometry-dis",
            "decision_policy": "factorized-v3", "light_search": "bank", "fixed_light_id": 0,
            "trajectory_contrast": "on", "independent_light": "on", "positive_fusion": "consensus",
            "negative_policy": "endpoints+lit-receivers",
            "consensus_fraction": 2/3, **config["variants"][name]}
        root = output/name
        root.mkdir()
        write_json(root/"ablation_protocol.json", {"variant": name, "recipe": recipe, "seed": args.seed,
            "source_root": str(source), "config_sha256": digest(args.config),
            "code_sha256": code_hashes,
            "evaluation_policy": "same official scene split and independent reference; no per-variant eval tuning"})
        teacher_ready = True
        for split in ("train", "val"):
            origin, target = source/split, root/split
            analysis_root = origin/"flow"
            replacement = []
            if recipe.get("feature_cache") == "current-only":
                cache = Path(args.current_cache_root).resolve()/split
                cache_protocol = json.loads((cache/"cache_protocol.json").read_text(encoding="utf-8"))
                if cache_protocol.get("history_mode") != "current-only":
                    raise ValueError(f"Not a current-only feature cache: {cache}")
                replacement = ["--feature-root", cache]
            if recipe["registration"] == "geometry-only":
                analysis_root = output/"shared_geometry_only"/split/"flow"
                if not (analysis_root/"analysis_protocol.json").exists():
                    run("analyze_shadow_motion_pairs.py", "--pairs", origin/"pairs/pairs.jsonl",
                        "--flow-backend", "none", "--review-limit", 0, "--output-root", analysis_root)
                features = replacement or (["--feature-root", origin/"mature_bev"] if recipe["feature_gate"] != "none" else [])
                original_protocol = json.loads((origin/"bev_flow/bev_flow_protocol.json").read_text(encoding="utf-8"))
                geometry = original_protocol["config"]
                run("analyze_bev_flow_discrepancy.py", "--pairs", origin/"pairs/pairs.jsonl",
                    "--analysis-root", analysis_root, "--output-root", target/"bev_flow", "--review-limit", 0, "--omit-receiver-features",
                    "--extent-m", geometry["extent_m"], "--resolution-m", geometry["resolution_m"],
                    "--light-threshold", geometry["light_threshold_L"],
                    "--feature-gate", recipe["feature_gate"], "--flow-gate", recipe["flow_gate"],
                    "--minimum-bev-cosine", recipe["minimum_bev_cosine"], "--seed", args.seed, *features)
            else:
                run("reselect_bev_flow.py", "--source-root", origin/"bev_flow", "--output-root", target/"bev_flow",
                    "--feature-gate", recipe["feature_gate"], "--flow-gate", recipe["flow_gate"],
                    "--minimum-bev-cosine", recipe["minimum_bev_cosine"], "--seed", args.seed, *replacement)
            options = [value for key in ("decision_policy", "light_search", "fixed_light_id", "trajectory_contrast",
                "independent_light", "positive_fusion", "consensus_fraction")
                for value in ("--"+key.replace("_", "-"), recipe[key])]
            if "existence_mode" in recipe:
                options += ["--existence-mode", recipe["existence_mode"]]
            run("build_physics_shadow_labels.py", "--pairs", origin/"pairs/pairs.jsonl", "--analysis-root", analysis_root,
                "--bev-flow-root", target/"bev_flow", "--output-root", target/"labels", "--export-weak-labels", *options)
            summary = json.loads((target/"labels/summary.json").read_text(encoding="utf-8"))
            if summary["stage_counts"]["accepted_shadow"] == 0:
                teacher_ready = False
                results.append({"variant": name, "split": split, "status": "zero_accepted_shadow"})
                continue
            teacher_run = run("build_bev_flow_annotation_manifest.py", "--pairs", origin/"pairs/pairs.jsonl",
                "--labels-root", target/"labels", "--output-root", target/"teacher",
                "--negative-policy", recipe["negative_policy"], check=False)
            if teacher_run.returncode:
                teacher_protocol = target/"teacher/protocol.json"
                counts = json.loads(teacher_protocol.read_text(encoding="utf-8"))["counts"] if teacher_protocol.exists() else {}
                if counts and (not counts.get("known_shadow_pixels") or not counts.get("known_lit_pixels")):
                    teacher_ready = False
                    results.append({"variant": name, "split": split, "status": "missing_teacher_class", "counts": counts})
                    continue
                teacher_run.check_returncode()
            results.append({"variant": name, "split": split, "status": "teacher_ready"})
        if teacher_ready and not args.labels_only:
            run("build_shadow_annotation_dataset.py", "--manifest", root/"train/teacher/canonical.jsonl",
                root/"val/teacher/canonical.jsonl", "--output-root", root/"data", "--seed", args.seed)
            run("build_joint_shadow_annotation_dataset.py", "--region-data-root", root/"data", "--output-root", root/"joint")
        write_json(output/"run_status.json", results)
    run("summarize_bev_flow_ablations.py", "--root", output, "--output-root", output/"comparison")


if __name__ == "__main__":
    main()
