from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.offbev import choices, localization_metrics
from shadow_occ.annotation.vlm import encode_example, load_model, model_options
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Infer potential outside-BEV casters from visible shadow evidence.")
    model_options(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--minimum-choice-score", type=float, default=0.3)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use plain python for single-GPU off-BEV VLM evaluation")
    if not 0 <= args.minimum_choice_score <= 1:
        raise ValueError("minimum-choice-score must be in [0,1]")
    rows = read_rows(args.manifest)
    if not rows or any(row["task"] != "offbev_caster" for row in rows):
        raise ValueError("Use an offbev train/dev/eval manifest")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.adapter:
        protocol = json.loads((Path(args.adapter).parent/"training_protocol.json").read_text(encoding="utf-8"))
        if set(protocol["train_scenes"]) & {row["scene_name"] for row in rows}:
            raise ValueError("Held-out off-BEV evaluation overlaps adapter training scenes")
        if protocol.get("primary_task") != "offbev_caster":
            raise ValueError("Adapter was not trained for the outside-BEV caster task")
    model, processor = load_model(args)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.inference_mode():
        for row in tqdm(rows, desc="shadow-induced outside-BEV hypotheses"):
            alternatives = choices(row)
            likelihoods = []
            for _, text in alternatives:
                inputs = {key: value.to(device) for key, value in encode_example(processor, row, text, args.max_length).items()}
                tokens = int((inputs["labels"][:, 1:] != -100).sum())
                log_probability = -float(model(**inputs).loss)*tokens
                if not math.isfinite(log_probability):
                    raise RuntimeError(f"Non-finite candidate score for {row['id']}")
                likelihoods.append(log_probability)
            scores = np.exp(np.asarray(likelihoods)-max(likelihoods))
            scores /= scores.sum()
            best = int(np.argmax(scores))
            chosen_id = alternatives[best][0] if float(scores[best]) >= args.minimum_choice_score else None
            selected = next((cell for cell in row["candidates"] if cell["id"] == chosen_id), None)
            geometry = max(row["candidates"], key=lambda cell: cell["geometry_score"], default=None)
            signature = hashlib.sha256(json.dumps({"images": row["images"], "prompt": row["prompt"]}, sort_keys=True).encode()).hexdigest()
            predictions.append({"id": row["id"], "scene_name": row["scene_name"], "sample_token": row["sample_token"], "camera": row["camera"], "input_signature": signature, "label_source": row["label_source"], "target_decision": row["target_decision"], "valid_candidate_ids": row["valid_candidate_ids"], "target_annotation_token": row.get("target_annotation_token"), "target_center_xy_ego": row.get("target_center_xy_ego"), "target_visibility": row["target_visibility"], "selected_candidate_id": chosen_id, "selected_center_xy_ego": selected["center_xy_ego"] if selected else None, "potential_caster_region_xyxy_ego": selected["xyxy_ego"] if selected else None, "boundary_side": selected["side"] if selected else None, "geometry_candidate_id": geometry["id"] if geometry else None, "decision": "POTENTIAL" if selected else "ABSTAIN", "choice_scores": [{"candidate_id": key, "score": float(score)} for (key, _), score in zip(alternatives, scores)], "nominal_bev_bounds": row["bev_bounds_xyxy"], "uncertainty": "outside region hypothesis, not verified occupancy or exact object extent"})
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/"hypotheses.jsonl", predictions)
    write_json(output/"metrics.json", {**localization_metrics(predictions), "manifest_sha256": digest(args.manifest), "model": args.model, "adapter": args.adapter, "minimum_choice_score": args.minimum_choice_score, "score_definition": "normalized likelihood over ABSTAIN and supplied candidate regions; uncalibrated"})


if __name__ == "__main__":
    main()
