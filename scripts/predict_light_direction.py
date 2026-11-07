"""Export frozen light priors and input-ablation diagnostics, never shadow masks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, write_rows
from shadow_occ.annotation.illumination import prior_record
from shadow_occ.annotation.light_learning import LightDataset, forward_light, load_light_rows
from shadow_occ.models.illumination import GeoTemporalLightNet
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", choices=["learned", "solar"], default="learned")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root, output = Path(args.data_root).resolve(), Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_light_rows(root)
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    directions = np.asarray(protocol["directions_world"])
    predictions, diagnostics = [], defaultdict(list)
    if args.method == "solar":
        for split, values in rows.items():
            for row in values:
                item = LightDataset([row])[0]
                prediction = prior_record(row, item["solar_target"].numpy(),
                    .25 if row["sun"].get("direct_sun_possible") else 0, row["sun"], directions,
                    {"method": "astronomy_only", "input_split": split, "training": "none"})
                predictions.append(prediction)
    else:
        if not args.checkpoint_root:
            parser.error("Learned prediction requires --checkpoint-root")
        checkpoints = Path(args.checkpoint_root).resolve()
        training = json.loads((checkpoints/"training_protocol.json").read_text(encoding="utf-8"))
        if training["data_protocol_sha256"] != digest(root/"protocol.json"):
            raise ValueError("Checkpoint was trained on a different light-data protocol")
        for split, path_hash in training["manifest_sha256"].items():
            if digest(root/f"{split}.jsonl") != path_hash:
                raise ValueError("Light-data manifest changed after training")
        groups = defaultdict(list)
        for split, values in rows.items():
            for row in values:
                fold = training["scene_folds"].get(row["scene_name"]) if split == "train" else None
                groups[f"fold_{fold}" if fold is not None else "best"].append(row)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable")
        for name, values in groups.items():
            path = checkpoints/f"{name}.pth"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if name != "best" and any(row["scene_name"] in checkpoint["trained_scenes"] for row in values):
                raise ValueError("Training-frame prior is not scene-held-out")
            model = GeoTemporalLightNet().to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            for mode in ("image+geo", "geo-only", "no-geo"):
                loader = DataLoader(LightDataset(values, mode), batch_size=args.batch_size)
                offset = 0
                with torch.no_grad():
                    for batch in loader:
                        prediction, moved = forward_light(model, batch, device)
                        probabilities = prediction["logits"].softmax(-1).cpu().numpy()
                        confidence = prediction["confidence_logits"].sigmoid().cpu().numpy()
                        targets = batch["target"].numpy()
                        for index, (probability, certainty, target) in enumerate(zip(probabilities, confidence, targets)):
                            row = values[offset+index]
                            diagnostics[f"{row['split']}:{mode}"].append({
                                "weak_cross_entropy": float(-np.sum(target*np.log(np.maximum(probability, 1e-8)))),
                                "weak_top1_agreement": bool(probability.argmax() == target.argmax()),
                                "weak_target_weight": row["target_weight"]})
                            if mode == "image+geo":
                                predictions.append(prior_record(row, probability, float(certainty), row["sun"], directions,
                                    {"method": "learned_geo_temporal", "checkpoint_sha256": digest(path),
                                     "input_split": row["split"], "checkpoint_name": name,
                                     "scene_held_out": row["scene_name"] not in checkpoint["trained_scenes"]}))
                        offset += len(probabilities)
    write_rows(output/"light_priors.jsonl", predictions)
    write_json(output/"metrics.json", {"method": args.method, "windows": len(predictions),
        "diagnostics": {key: {"windows": len(values),
            "weak_cross_entropy": float(np.mean([row["weak_cross_entropy"] for row in values if row["weak_target_weight"] > 0]))
                if any(row["weak_target_weight"] > 0 for row in values) else None,
            "weak_top1_agreement": float(np.mean([row["weak_top1_agreement"] for row in values if row["weak_target_weight"] > 0]))
                if any(row["weak_target_weight"] > 0 for row in values) else None}
            for key, values in diagnostics.items()},
        "metric_scope": "weak-target agreement, not measured light accuracy or shadow precision",
        "ablation_scope": "frozen-model input ablations; not independently trained baselines",
        "data_protocol_sha256": digest(root/"protocol.json")})


if __name__ == "__main__":
    main()
