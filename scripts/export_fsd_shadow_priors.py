"""Cache official FSD responses in calibrated RGB coordinates, once per camera token."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, resolve_path, write_rows
from shadow_occ.utils.io import write_json
from fetch_shadow_prior_assets import FSD_COMMIT, FSD_SHA256, checksum


def unique_frames(pairs, clips=None):
    selected = None
    if clips is not None:
        from discover_shadow_video import select_chain
        selected = {frame["sample_data_token"] for clip in clips for pair in select_chain(pairs, clip)
                    for frame in pair["frames"]}
    rows = {}
    for pair in pairs:
        for frame in pair["frames"]:
            token = frame["sample_data_token"]
            if selected is not None and token not in selected:
                continue
            row = {"frame_token": token, "scene_name": pair["scene_name"], "camera": pair["camera"],
                "timestamp": frame["camera_timestamp"], "image": frame["image"], "image_size_wh": frame["image_size_wh"]}
            if token in rows and rows[token] != row:
                raise ValueError("Same camera token has inconsistent RGB metadata")
            rows[token] = row
    return [rows[token] for token in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--clips", help="Limit inference to selected clips; otherwise cache all pair frames")
    parser.add_argument("--fsd-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-width", type=int, default=800, help="Inference width; response is restored to calibrated RGB size")
    args = parser.parse_args()
    import torch
    from torch.nn import functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("Official FSD constructs a CUDA DIM submodule; use a GPU job")
    if args.max_width < 32:
        raise ValueError("FSD inference width must be at least 32")
    source, checkpoint = Path(args.fsd_root).resolve(), Path(args.checkpoint).resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != FSD_COMMIT or checksum(checkpoint) != FSD_SHA256:
        raise ValueError("Use the pinned official FSD source and checksum-verified checkpoint")
    sys.path.insert(0, str(source))
    from networks.fdrnet import FDRNet
    model = FDRNet(backbone="efficientnet-b3", proj_planes=16, pred_planes=32, use_pretrained=False,
        fix_backbone=False, has_se=False, dropout_2d=0, normalize=True, mu_init=.5, reweight_mode="manual")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
    model.cuda().eval()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = Path(args.pairs).resolve().parent
    frames = unique_frames(read_rows(args.pairs), read_rows(args.clips) if args.clips else None)
    write_json(output/"protocol.json", {"model": "FSD", "external_repository": "https://github.com/IMU-Group/FSD",
        "external_commit": commit, "checkpoint_sha256": FSD_SHA256, "exporter_sha256": digest(__file__),
        "external_source_sha256": {name: digest(source/name) for name in ("networks/fdrnet.py", "networks/DIM.py")},
        "pairs_sha256": digest(args.pairs), "clips_sha256": digest(args.clips) if args.clips else None,
        "max_width": args.max_width, "rgb_input": "RGB float [0,1]; model applies ImageNet normalization internally",
        "response_contract": "binary_mask is nonnegative ReLU response, NOT logits/probability; official threshold 0.5; no extra sigmoid",
        "semantic_probability_contract": "logit field is already sigmoid; stored separately without second sigmoid",
        "source_supervision": "external annotated shadow data; no nuScenes manual shadow labels",
        "training_performed": False})
    rows = []
    for row in frames:
        path = resolve_path(row["image"], base)
        image = cv2.imread(str(path))
        if image is None or list(image.shape[1::-1]) != row["image_size_wh"]:
            raise ValueError(f"Missing or mismatched RGB: {path}")
        height, width = image.shape[:2]
        scale = min(1., args.max_width/width)
        resized = cv2.resize(image, (max(32, round(width*scale)), max(32, round(height*scale))))
        tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float()[None].cuda()/255
        with torch.inference_mode():
            prediction = model(tensor)
            payload = {key: F.interpolate(prediction[field], size=(height, width), mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
                for key, field in (("binary_response", "binary_mask"), ("opacity_response", "soft_feat"), ("semantic_probability", "logit"))}
        if any(not np.isfinite(value).all() for value in payload.values()):
            raise ValueError("FSD emitted nonfinite values")
        target = output/(row["frame_token"]+".npz")
        np.savez_compressed(target, **payload)
        rows.append({**row, "image": str(path), "image_sha256": digest(path), "response": target.name,
            "response_sha256": digest(target)})
        print(json.dumps({"completed": len(rows), "total": len(frames), "frame_token": row["frame_token"]}), flush=True)
    write_rows(output/"manifest.jsonl", rows)
    write_json(output/"summary.json", {"frames": len(rows), "bytes": sum(path.stat().st_size for path in output.glob("*.npz")),
        "scenes": sorted({row["scene_name"] for row in rows}), "no_shadow_labels_created": True})


if __name__ == "__main__":
    main()
