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
from shadow_occ.annotation.diagnostics import SCORING_VERSION
from shadow_occ.annotation.joint import annotation
from shadow_occ.annotation.teacher_provenance import fingerprint, input_provenance
from shadow_occ.annotation.vlm import encode_example, load_model, model_options
from shadow_occ.utils.io import write_json


def score_values(model, processor, row, device, max_length):
    values, reference = {}, None
    for label in ("YES", "NO", "UNCERTAIN"):
        # Complete hypothetical outputs preserve the trained JSON boundary tokens.
        # Only the decision span is scored; future caster tokens are ignored.
        answer = json.dumps(annotation(label, "none" if label == "NO" else "unresolved"), separators=(",", ":"))
        inputs = encode_example(processor, row, answer, max_length, return_groups=True)
        groups, labels = inputs.pop("token_groups"), inputs.pop("labels")
        mask = groups[0] == 1
        start = int(torch.nonzero(mask)[0].item())
        prefix = inputs["input_ids"][0, :start]
        if reference is not None and not torch.equal(reference, prefix):
            raise ValueError("YES/NO/UNCERTAIN do not share an identical token prefix")
        reference = prefix.clone()
        outputs = model(**{key: value.to(device) for key, value in inputs.items()})
        selected = mask[1:].to(device)
        logits = outputs.logits[0, :-1][selected].float()
        targets = labels[0, 1:][mask[1:]].to(device)
        value = logits.log_softmax(-1).gather(-1, targets[:, None]).sum()
        if not bool(torch.isfinite(value)):
            raise RuntimeError("Non-finite shadow decision likelihood")
        values[label] = float(value)
    vector = torch.tensor([values[label] for label in ("YES", "NO", "UNCERTAIN")], dtype=torch.float64)
    probabilities = vector.softmax(0).tolist()
    binary = vector[:2].softmax(0).tolist()
    return {"score": binary[0], "uncertainty_score": probabilities[2], "decision_scores": dict(zip(("YES", "NO", "UNCERTAIN"), probabilities)), "log_likelihoods": values}


def main():
    parser = argparse.ArgumentParser("Score only the shadow value in the unified JSON task, never caster/box/end tokens.")
    model_options(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-train-scenes", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use plain python and one GPU")
    rows = read_rows(args.manifest)
    if not rows or any(row.get("task") != "shadow_annotation" for row in rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique unified annotation rows")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    protocol, overlap = {}, set()
    if args.adapter:
        protocol = json.loads((Path(args.adapter).parent/"training_protocol.json").read_text(encoding="utf-8"))
        if protocol.get("primary_task") != "shadow_annotation":
            raise ValueError("Adapter was not trained for the unified annotation task")
        overlap = set(protocol["train_scenes"]) & {row["scene_name"] for row in rows}
        if overlap and not args.allow_train_scenes:
            raise ValueError("Scoring overlaps adapter training scenes")
    fingerprints, scorer_id = fingerprint(args, rows)
    model, processor = load_model(args)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    results, content_cache = [], {}
    with torch.inference_mode():
        for row in tqdm(rows, desc="shadow-value conditional scores"):
            result = score_values(model, processor, row, device, args.max_length)
            results.append({**row, **result, **input_provenance(row, content_cache), "scorer_id": scorer_id, "scoring_version": SCORING_VERSION,
                            "in_sample_annotation": bool(overlap), "input_signature": hashlib.sha256(json.dumps({"images": row["images"], "prompt": row["prompt"]}, sort_keys=True).encode()).hexdigest()})
    write_rows(output/"scores.jsonl", results)
    write_json(output/"scoring_protocol.json", {**vars(args), "manifest_sha256": digest(args.manifest), "fingerprints": fingerprints,
               "scores_sha256": digest(output/"scores.jsonl"), "teacher_train_scenes": protocol.get("train_scenes"), "teacher_dev_scenes": protocol.get("dev_scenes"),
               "teacher_self_training_round": protocol.get("self_training_round", 0), "legacy_history_assumed_seed": "self_training_round" not in protocol,
               "scorer_id": scorer_id, "training_state_support": protocol.get("balance_audit", {}).get("caster_states", {}),
               "score_definition": "relative likelihood of only shadow value tokens including closing quote; not a calibrated probability",
               "outside_policy": "This scores shadow existence only; it does not train or evaluate outside-BEV localization"})


if __name__ == "__main__":
    main()
