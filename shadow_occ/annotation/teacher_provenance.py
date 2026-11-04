from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shadow_occ.annotation.data import digest
from shadow_occ.annotation.diagnostics import SCORING_VERSION


def fingerprint(args, rows):
    values = {"model": str(Path(args.model).resolve()), "version": SCORING_VERSION,
              "view_mode": sorted({row.get("view_mode", "full") for row in rows}),
              "max_pixels": args.max_pixels, "max_length": args.max_length, "load_in_4bit": args.load_in_4bit}
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        path = Path(args.model)/name
        if path.is_file():
            values[name] = digest(path)
    if args.adapter:
        weights = sorted(Path(args.adapter).glob("adapter_model*.safetensors"))+sorted(Path(args.adapter).glob("adapter_model*.bin"))
        if not weights:
            raise FileNotFoundError("No local adapter weights available for fingerprinting")
        for path in [Path(args.adapter)/"adapter_config.json", *weights]:
            if path.is_file():
                values[path.name] = digest(path)
    return values, hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def cached_digest(path, cache):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        cache[key] = digest(path)
    return cache[key]


def extend_scene_history(train, dev, initial):
    if initial.get("train_scenes") is None or initial.get("dev_scenes") is None:
        raise ValueError("Initial segmenter needs scene provenance; it cannot hide fold leakage")
    train, dev = set(train) | set(initial["train_scenes"]), set(dev) | set(initial["dev_scenes"])
    if train & dev:
        raise ValueError("Initial segmentation checkpoint crosses the current train/dev boundary")
    return train, dev


def input_provenance(row, cache):
    identity = {"prompt": row["prompt"], "images": [cached_digest(path, cache) for path in row["images"]]}
    return {"input_content_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "region_mask_sha256": cached_digest(row["region_mask"], cache) if row.get("region_mask") else None}
