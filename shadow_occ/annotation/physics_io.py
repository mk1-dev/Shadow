"""Shared loading of exported camera/flow evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shadow_occ.annotation.data import digest, read_rows, resolve_path


def load_motion_analysis(pairs_path, analysis):
    analysis = Path(analysis)
    protocol_path = analysis/"analysis_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("flow_evidence_version") != "receiver-support-v2":
        raise ValueError("Re-run motion analysis with receiver-support-v2")
    if protocol["manifest_sha256"] != digest(pairs_path):
        raise ValueError("Flow analysis was not produced from this exact pair manifest")
    pairs = read_rows(pairs_path)
    if not pairs or len({row["id"] for row in pairs}) != len(pairs):
        raise ValueError("Pairs must be nonempty and uniquely identified")
    metrics = read_rows(analysis/"pair_metrics.jsonl")
    stats = {row["id"]: row for row in metrics}
    if len(stats) != len(metrics) or set(stats) != {row["id"] for row in pairs}:
        raise ValueError("Evidence must cover the exact pair manifest, without duplicates")
    return pairs, stats, protocol_path


def read_evidence(row, stats, base):
    metric = stats[row["id"]]
    for key in ("scene_name", "camera", "official_split"):
        if row[key] != metric[key]:
            raise ValueError(f"Pair evidence metadata mismatch: {key}")
    path = resolve_path(metric["evidence"], Path(base))
    if digest(path) != metric["evidence_sha256"]:
        raise ValueError("Pair evidence changed since analysis")
    with np.load(path, allow_pickle=False) as archive:
        if "receiver_flow_supported" not in archive.files:
            raise ValueError("Missing v2 receiver-flow support; rebuild motion analysis")
        return {key: archive[key].copy() for key in archive.files}, metric


def check_pair_files(pair, base):
    for frame in pair["frames"]:
        for key, sha in (("image", "image_sha256"), ("ground_support", "ground_sha256")):
            if digest(resolve_path(frame[key], Path(base))) != frame[sha]:
                raise ValueError(f"Exported {key} changed: {pair['id']}")
