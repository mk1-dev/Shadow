from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset

from shadow_occ.annotation.data import GT_SOURCES, WEAK_SOURCES, assert_disjoint, digest, read_rows, write_rows
from shadow_occ.annotation.metrics import region_summary
from shadow_occ.annotation.offbev import ASSOCIATION_GT, choices
from shadow_occ.annotation.joint import parse as parse_annotation
from shadow_occ.utils.io import write_json


def load_model(args, training=False):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL training/scoring requires a CUDA device in this entry point")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa", "local_files_only": not args.allow_download}
    if args.load_in_4bit:
        kwargs.update({"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True), "device_map": {"": local}})
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    if not args.load_in_4bit:
        model.to(torch.device("cuda", local))
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=56*56, max_pixels=args.max_pixels, local_files_only=not args.allow_download)
    model.config.use_cache = False
    if training and args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, processor


def encode_example(processor, row, answer=None, max_length=4096):
    images = []
    for path in row["images"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    messages = [{"role": "user", "content": [{"type": "image", "image": path} for path in row["images"]] + [{"type": "text", "text": row["prompt"]}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = processor(text=[prompt_text], images=images, return_tensors="pt", padding=False)
    if answer is None:
        result = dict(prompt)
    else:
        full_text = processor.apply_chat_template(messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}], tokenize=False, add_generation_prompt=False)
        full = processor(text=[full_text], images=images, return_tensors="pt", padding=False)
        prefix_length = prompt["input_ids"].shape[1]
        if not torch.equal(full["input_ids"][:, :prefix_length], prompt["input_ids"]):
            raise ValueError(f"Chat-template prefix mismatch for {row['id']}; update processor/model together")
        result = dict(full)
        result["labels"] = full["input_ids"].clone()
        result["labels"][:, :prefix_length] = -100
        if not (result["labels"] != -100).any():
            raise ValueError("No supervised assistant tokens")
    if result["input_ids"].shape[1] > max_length:
        raise ValueError(f"{row['id']} exceeds {max_length} tokens; reduce --max-pixels, do not truncate visual tokens")
    return result


class SFTDataset(Dataset):
    def __init__(self, rows, processor, max_length, weak_weight, decision_weights=None):
        self.rows, self.processor, self.max_length, self.weak_weight = rows, processor, max_length, weak_weight
        self.decision_weights = decision_weights or {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = encode_example(self.processor, row, row["answer"], self.max_length)
        weight = self.weak_weight if row["label_source"] in WEAK_SOURCES else 1.0
        if self.decision_weights:
            decision = parse_annotation(row["answer"], row.get("candidates", []))["shadow"]
            weight *= self.decision_weights.get(decision, 1.0)
        encoded["example_weight"] = weight
        return encoded


def shadow_decision_weights(rows, maximum=10.0):
    decisions = Counter(parse_annotation(row["answer"], row.get("candidates", []))["shadow"] for row in rows)
    total = sum(decisions.values())
    if not total or not decisions:
        raise ValueError("Cannot balance an empty annotation dataset")
    return {decision: min(maximum, total/(len(decisions)*count)) for decision, count in sorted(decisions.items())}


def one_example_collator(items):
    if len(items) != 1:
        raise ValueError("Use micro-batch size 1; increase gradient accumulation for the effective batch")
    item = dict(items[0])
    item["example_weight"] = torch.tensor(item["example_weight"], dtype=torch.float32)
    return item


def model_options(parser):
    parser.add_argument("--model", required=True, help="Local Qwen3-VL-8B-Instruct model directory")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--max-length", type=int, default=4096)


def train_main():
    parser = argparse.ArgumentParser("Train the GT-constrained shadow annotator; legacy diagnostics are explicit opt-ins.")
    model_options(parser)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--primary-task", choices=["shadow_annotation", "shadow", "offbev_caster"], default="shadow_annotation")
    parser.add_argument("--auxiliary-train", help="Optional existing caster-GT task manifest")
    parser.add_argument("--auxiliary-dev")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use a small positive value for an HPC smoke run")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--caster-ratio", type=float, default=0.0, help="Legacy diagnostic tasks only; unified annotation incorporates caster GT into each annotation target")
    parser.add_argument("--weak-weight", type=float, default=0.25)
    parser.add_argument("--allow-weak-labels", action="store_true")
    parser.add_argument("--balance-shadow", action="store_true", help="Balance YES/NO/UNCERTAIN weights using the training split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    if not 0 <= args.caster_ratio <= 1 or not 0 < args.weak_weight <= 1:
        raise ValueError("Invalid auxiliary ratio or weak-label weight")
    train_all, dev_all = read_rows(args.train), read_rows(args.dev)
    if bool(args.auxiliary_train) != bool(args.auxiliary_dev):
        raise ValueError("Supply both auxiliary train and dev manifests")
    if args.auxiliary_train:
        train_all += [row for row in read_rows(args.auxiliary_train) if row.get("task") == "caster"]
        dev_all += [row for row in read_rows(args.auxiliary_dev) if row.get("task") == "caster"]
    if args.primary_task == "shadow_annotation" and (args.auxiliary_train or args.caster_ratio != 0):
        raise ValueError("The unified annotation task uses GT in annotation targets; do not add a separate caster task")
    assert_disjoint(train_all, dev_all)
    if any(row.get("split") != "dev" for row in dev_all):
        raise ValueError("Model selection requires a train-derived dev manifest")
    primary_gt = ASSOCIATION_GT if args.primary_task == "offbev_caster" else GT_SOURCES | ASSOCIATION_GT
    allowed = primary_gt | WEAK_SOURCES if args.allow_weak_labels else primary_gt

    def select(rows):
        shadow = [row for row in rows if row.get("task") == args.primary_task and row.get("answer") is not None and row.get("label_source") in allowed]
        if args.primary_task == "shadow_annotation":
            decisions = set()
            for row in shadow:
                target = parse_annotation(row["answer"], row.get("candidates", []))
                decisions.add(target["shadow"])
            if "YES" not in decisions or not decisions & {"NO", "UNCERTAIN"}:
                raise ValueError("Annotation training needs positive and negative/uncertain shadow examples")
        elif args.primary_task == "shadow":
            if {row["answer"] for row in shadow} != {"YES", "NO"}:
                raise ValueError("Both YES and NO shadow targets are required; object GT alone cannot train shadow decisions")
        else:
            if {row["target_decision"] for row in shadow} != {"POTENTIAL", "ABSTAIN"}:
                raise ValueError("Outside-BEV training needs potential-target associations and abstention examples")
            for row in shadow:
                if row["answer"] not in {text for _, text in choices(row)}:
                    raise ValueError("Off-BEV answer references a candidate not available in its inputs")
        casters = [row for row in rows if row.get("task") == "caster" and row.get("label_source") == "object_annotation_gt"]
        random.Random(args.seed).shuffle(casters)
        return shadow + casters[:int(len(shadow)*args.caster_ratio)]

    train, dev = select(train_all), select(dev_all)
    decision_weights = shadow_decision_weights(train) if args.balance_shadow else {}
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new LoRA output directory")
    set_seed(args.seed)
    model, processor = load_model(args, training=True)
    # Explicit full module names prevent matching Qwen's visual attention blocks.
    targets = [name for name, _ in model.named_modules() if ".language_model." in name and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    if not targets:
        raise ValueError("No language attention modules found; inspect this model's module names")
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=2*args.lora_rank, lora_dropout=0.05, target_modules=targets, bias="none"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weight = inputs.pop("example_weight")
            outputs = model(**inputs)
            loss = outputs.loss*weight
            return (loss, outputs) if return_outputs else loss

    training = TrainingArguments(output_dir=str(output), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.accumulation, num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, logging_steps=10, warmup_ratio=0.05, weight_decay=0.01, remove_unused_columns=False, label_names=["labels"], report_to="none", dataloader_num_workers=0, ddp_find_unused_parameters=False, seed=args.seed, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, prediction_loss_only=True)
    trainer = WeightedTrainer(model=model, args=training, train_dataset=SFTDataset(train, processor, args.max_length, args.weak_weight, decision_weights), eval_dataset=SFTDataset(dev, processor, args.max_length, args.weak_weight, decision_weights), data_collator=one_example_collator)
    trainer.model_accepts_loss_kwargs = False
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/"training_protocol.json", {**vars(args), "train_sha256": digest(args.train), "dev_sha256": digest(args.dev), "train_scenes": sorted({row['scene_name'] for row in train}), "dev_scenes": sorted({row['scene_name'] for row in dev}), "target_modules": targets, "training_rows": len(train), "dev_rows": len(dev), "shadow_decision_weights": decision_weights})
    trainer.train()
    trainer.save_model(str(output/"adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(output/"adapter")


def score_main():
    parser = argparse.ArgumentParser("Score single-proposal shadow decisions before or after LoRA.")
    model_options(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-train-scenes", action="store_true", help="Explicit in-sample annotation, not held-out evaluation")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use plain python for single-GPU VLM scoring")
    rows = [row for row in read_rows(args.manifest) if row.get("task") == "shadow"]
    if not rows:
        raise ValueError("No single-proposal shadow examples")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    overlap = set()
    if args.adapter:
        protocol_path = Path(args.adapter).parent/"training_protocol.json"
        if not protocol_path.exists():
            raise FileNotFoundError("Adapter training_protocol.json is required to check scene leakage")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("primary_task", "shadow") != "shadow":
            raise ValueError("Use evaluate_shadow_annotator.py for the unified annotation adapter")
        overlap = set(protocol["train_scenes"]) & {row["scene_name"] for row in rows}
        if overlap and not args.allow_train_scenes:
            raise ValueError(f"Evaluation overlaps adapter training scenes: {sorted(overlap)[:10]}")
    model, processor = load_model(args)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    output.mkdir(parents=True, exist_ok=True)
    results = []
    from tqdm import tqdm

    with torch.inference_mode():
        for row in tqdm(rows, desc="VLM single-region scoring"):
            likelihoods = []
            for answer in ("YES", "NO"):
                inputs = {key: value.to(device) for key, value in encode_example(processor, row, answer, args.max_length).items()}
                # Score the full answer, including its terminator, with the same prompt.
                tokens = int((inputs["labels"][:, 1:] != -100).sum())
                response = model(**inputs)
                value = -float(response.loss)*tokens
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite sequence likelihood for {row['id']}")
                likelihoods.append(value)
            difference = max(-60.0, min(60.0, likelihoods[1]-likelihoods[0]))
            probability = 1/(1+math.exp(difference))
            input_signature = hashlib.sha256(json.dumps({"images": row["images"], "prompt": row["prompt"]}, sort_keys=True).encode()).hexdigest()
            results.append({"id": row["id"], "parent_id": row.get("parent_id"), "scene_name": row["scene_name"], "split": row["split"], "region_mask": row.get("region_mask"), "pair_id": row.get("pair_id"), "group": row.get("group"), "answer": row.get("answer"), "label_source": row["label_source"], "temporal_mode": row.get("temporal", {}).get("mode", "none"), "temporal_context": row.get("temporal_context", False), "input_signature": input_signature, "score": probability, "prediction": "YES" if probability >= args.threshold else "NO", "log_likelihood_yes": likelihoods[0], "log_likelihood_no": likelihoods[1]})
    write_rows(output/"votes.jsonl", results)
    summary = region_summary(results, args.threshold, args.bootstrap, args.seed)
    summary.update({"manifest_sha256": digest(args.manifest), "model": args.model, "adapter": args.adapter, "score_definition": "normalized YES/NO sequence likelihood; not a calibrated probability"})
    summary["caster_context_sources"] = dict(Counter(row.get("caster_source", "none") for row in rows))
    summary["in_sample_annotation"] = bool(args.adapter and overlap)
    summary["temporal_mode_counts"] = dict(Counter(row["temporal_mode"] for row in results))
    summary["temporal_context_examples"] = sum(row["temporal_context"] for row in results)
    write_json(output/"summary.json", summary)
