"""Zero-shot Qwen scene observability; no annotation adapter, masks or caster outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import digest, read_rows
from shadow_occ.annotation.observability import (
    PROTOCOL, input_signature, parse_assessment, stratum, summarize_votes, vote_index,
)
from shadow_occ.utils.io import write_json


def main():
    import torch
    from tqdm import tqdm
    from shadow_occ.annotation.vlm import encode_example, load_model, model_options

    parser = argparse.ArgumentParser(__doc__)
    model_options(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use one GPU and plain python")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if (Path(args.model)/"adapter_config.json").exists():
        parser.error("Use an untuned base VLM directory, not an annotation adapter")
    rows = read_rows(args.manifest)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Observability manifest must contain unique windows")
    for row in rows:
        if row.get("protocol") != PROTOCOL or row.get("task") != "scene_observability" or input_signature(row) != row["input_signature"]:
            raise ValueError("Invalid RGB observability manifest")
    output = Path(args.output_root).resolve()
    protocol = {"protocol": PROTOCOL, "manifest_sha256": digest(args.manifest), "model": args.model,
        "load_in_4bit": args.load_in_4bit, "max_pixels": args.max_pixels, "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens, "adapter": None, "generation": "greedy"}
    results = []
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(output)
        if json.loads((output/"protocol.json").read_text(encoding="utf-8")) != protocol:
            raise ValueError("Resume model/manifest/settings differ")
        results = read_rows(output/"votes.jsonl") if (output/"votes.jsonl").exists() else []
    output.mkdir(parents=True, exist_ok=True)
    previous = vote_index(results)
    signatures = {row["id"]: row["input_signature"] for row in rows}
    if any(signatures.get(key) != vote["input_signature"] for key, vote in previous.items()):
        raise ValueError("Resume votes do not belong to this manifest")
    write_json(output/"protocol.json", protocol)
    pending = [row for row in rows if row["id"] not in previous]
    if pending:
        model, processor = load_model(args)
        model.eval()
        device = next(model.parameters()).device
        checked = {}
        with (output/"votes.jsonl").open("a", encoding="utf-8") as handle, torch.inference_mode():
            for row in tqdm(pending, desc="Zero-shot observation conditions", unit="window"):
                for path, expected in zip(row["images"], row["image_sha256"]):
                    if path not in checked:
                        checked[path] = digest(path)
                    if checked[path] != expected:
                        raise ValueError("Clean RGB changed before VLM inference")
                started = time.perf_counter()
                inputs = {key: value.to(device) for key, value in encode_example(processor, row, max_length=args.max_length).items()}
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True)
                raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
                error, assessment = None, None
                try:
                    assessment = parse_assessment(raw)
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                result = {key: row[key] for key in ("id", "protocol", "scene_name", "official_split", "camera",
                    "frame_tokens", "timestamps", "image_sha256", "input_signature")}
                result.update(assessment=assessment, stratum=stratum(assessment), parse_error=error, raw=raw,
                              inference_seconds=time.perf_counter()-started, is_shadow_gt=False)
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
                handle.flush()
                results.append(result)
    write_json(output/"summary.json", summarize_votes(results))
    print(f"Wrote observability tags to {output}", flush=True)


if __name__ == "__main__":
    main()
