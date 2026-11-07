"""Prepare scene-disjoint light-estimator data from completed BEV/flow caches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, group_splits, read_rows, resolve_path, write_rows
from shadow_occ.annotation.illumination import PROTOCOL, direction_inputs, sunlight_context, weak_light_target
from shadow_occ.annotation.physics_labels import light_bank
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dataroot", help="Enrich old caches lacking map location from original nuScenes tables")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-fraction", type=float, default=.2)
    args = parser.parse_args()
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    adapter = NuScenesAdapter("v1.0-trainval", args.dataroot) if args.dataroot else None
    directions = light_bank()
    rows, fingerprints = [], {}
    for official in ("train", "val"):
        root = source/official
        pairs_path, windows_path = root/"pairs/pairs.jsonl", root/"bev_flow/windows.jsonl"
        pairs = {row["id"]: row for row in read_rows(pairs_path)}
        metrics_path = root/"flow/pair_metrics.jsonl"
        metrics = {row["id"]: row for row in read_rows(metrics_path)}
        decisions_path = root/"labels/decisions.jsonl"
        components = defaultdict(list)
        if decisions_path.exists():
            for item in read_rows(decisions_path):
                if item.get("bev_flow"):
                    components[item["bev_flow"]["window_id"]].append(item)
        fingerprints[official] = {"pairs_sha256": digest(pairs_path), "windows_sha256": digest(windows_path),
            "pair_metrics_sha256": digest(metrics_path),
            "weak_decisions_sha256": digest(decisions_path) if decisions_path.exists() else None}
        for window in tqdm(read_rows(windows_path), desc=f"light data {official}", mininterval=5):
            left, right = [pairs[key] for key in window["pair_ids"]]
            frames = [left["frames"][0], right["frames"][0], right["frames"][1]]
            location = left.get("location")
            if not location and adapter:
                sample = adapter.nusc.get("sample", left["sample_token"])
                scene = adapter.nusc.get("scene", sample["scene_token"])
                location = adapter.nusc.get("log", scene["log_token"])["location"]
            anchor = np.asarray(left["anchor_ego_to_world"])
            times = [frame["camera_timestamp"] for frame in frames]
            sun = sunlight_context(location, anchor[:2, 3], times)
            validity = [bool(metrics[pair["id"]]["exposure"]["valid"]) for pair in (left, right)]
            intervals = np.diff(times)/1e6
            features, metadata = direction_inputs(directions, frames, sun, validity, intervals)
            image_inputs = []
            for frame in frames:
                image_path = resolve_path(frame["image"], pairs_path.parent)
                if digest(image_path) != frame["image_sha256"]:
                    raise ValueError("Light input image fingerprint mismatch")
                image = cv2.imread(str(image_path))
                if image is None:
                    raise FileNotFoundError(image_path)
                image_inputs.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (160, 96)))
            evidence_path = resolve_path(window["evidence"], windows_path.parent)
            if digest(evidence_path) != window["evidence_sha256"]:
                raise ValueError("Light input BEV fingerprint mismatch")
            with np.load(evidence_path, allow_pickle=False) as evidence:
                valid = evidence["receiver_valid"]
                lightness = np.nan_to_num(evidence["appearance_L_compensated"])/100
                delta = np.clip(np.nan_to_num(evidence["temporal_illumination_change_L"])/15, -3, 3)
                receiver = np.concatenate([lightness*valid, delta*valid,
                    np.stack([valid*flag for flag in validity])])
            receiver = np.stack([cv2.resize(value.astype(np.float32), (80, 80)) for value in receiver])
            target = weak_light_target(directions, components[window["id"]], sun)
            path = output/"features"/f"{window['id']}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, images=np.concatenate(image_inputs, axis=-1).transpose(2, 0, 1),
                receiver=receiver.astype(np.float32), metadata=metadata, directions=features,
                target=target["probability"], solar_target=target["solar_probability"])
            rows.append({"id": window["id"], "scene_name": window["scene_name"], "official_split": official,
                "sample_tokens": window["sample_tokens"], "camera": window["camera"],
                "features": str(path), "features_sha256": digest(path), "sun": sun,
                "target_weight": target["weight"], "confidence_target": target["confidence"],
                "physical_tracks": target["physical_tracks"], "geometry_quality": target["geometry_quality"],
                "target_source": target["source"], "timestamps": times,
                "source_window_sha256": window["evidence_sha256"]})
    train_scenes = {row["scene_name"] for row in rows if row["official_split"] == "train"}
    eval_scenes = {row["scene_name"] for row in rows if row["official_split"] == "val"}
    if train_scenes & eval_scenes:
        raise ValueError("Official train/eval scenes overlap")
    mapping = group_splits(train_scenes, "train", args.dev_fraction, args.seed)
    for row in rows:
        row["split"] = "eval" if row["official_split"] == "val" else mapping[row["scene_name"]]
    for split in ("train", "dev", "eval"):
        selected = [row for row in rows if row["split"] == split]
        write_rows(output/f"{split}.jsonl", selected)
    write_json(output/"protocol.json", {"protocol": PROTOCOL, **vars(args), "directions_world": directions.tolist(),
        "sources": fingerprints, "counts": dict(Counter(row["split"] for row in rows)),
        "scene_split": {**mapping, **{scene: "eval" for scene in eval_scenes}},
        "supervision": "weak astronomy/geometry; no manual light or shadow GT; not a ground-truth benchmark"})


if __name__ == "__main__":
    main()
