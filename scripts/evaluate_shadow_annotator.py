from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.joint import parse, summarize
from shadow_occ.annotation.vlm import encode_example, load_model, model_options
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Evaluate the GT-constrained shadow annotator and its caster associations.")
    model_options(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--allow-train-scenes", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Use plain python for single-GPU annotator evaluation")
    rows = read_rows(args.manifest)
    if not rows or any(row.get("task") != "shadow_annotation" for row in rows):
        raise ValueError("Use annotation_train/dev/eval.jsonl from the unified dataset builder")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    overlap = set()
    if args.adapter:
        protocol = json.loads((Path(args.adapter).parent/"training_protocol.json").read_text(encoding="utf-8"))
        if protocol.get("primary_task") != "shadow_annotation":
            raise ValueError("Use an adapter trained for the unified shadow annotation task")
        overlap = set(protocol["train_scenes"]) & {row["scene_name"] for row in rows}
        if overlap and not args.allow_train_scenes:
            raise ValueError("Evaluation overlaps training scenes")
    model, processor = load_model(args)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    results = []
    with torch.inference_mode():
        for row in tqdm(rows, desc="GT-constrained shadow annotation"):
            inputs = {key: value.to(device) for key, value in encode_example(processor, row, max_length=args.max_length).items()}
            tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = processor.tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                prediction = parse(text, row.get("candidates", []))
                error = None
            except (ValueError, TypeError, KeyError) as exc:
                prediction, error = None, str(exc)
            signature = hashlib.sha256(json.dumps({"images": row["images"], "prompt": row["prompt"]}, sort_keys=True).encode()).hexdigest()
            target = row.get("annotation_target")
            results.append({"id": row["id"], "parent_id": row.get("parent_id"), "scene_name": row["scene_name"], "input_signature": signature, "target": target, "shadow_label_source": row["shadow_label_source"], "association_source": row["association_source"], "known_outside_caster": row.get("known_outside_caster", False), "prediction": prediction, "raw_answer": text, "parse_error": error, "region_mask": row.get("region_mask"), "shadow_mask_reference": row.get("region_mask") if prediction and prediction["shadow"] == "YES" else None, "valid_candidate_ids": row.get("valid_candidate_ids", []), "exterior_candidates": row.get("candidates", [])})
    for result, row in zip(results, rows):
        result["input_regime"] = row.get("input_regime", "unspecified")
        result["shadow_evidence"] = row.get("shadow_evidence")
        result["association_status"] = row.get("association_status", "unspecified")
    write_rows(output/"annotations.jsonl", results)
    write_json(output/"metrics.json", {**summarize(results), "manifest_sha256": digest(args.manifest), "model": args.model, "adapter": args.adapter, "in_sample_annotation": bool(overlap)})


if __name__ == "__main__":
    main()
