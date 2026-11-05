"""Run in the mature OpenDriveVLA environment; export only spatial BEV tensors."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.cached_bev import spatial_bev, write_cached_bev
from shadow_occ.annotation.data import digest, read_rows
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Import OpenDriveVLA export_uniad_pth.py --full-results caches")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pairs", required=True, help="Export the exact three-frame sample tokens needed by Shadow")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.external_root).resolve()))
    # Reuse the model's own config parser; Shadow's annotation env needs no MMCV.
    from mmengine import Config

    cfg = Config.fromfile(args.model_config)
    pc_range = list(cfg.point_cloud_range)
    height, width = int(cfg.bev_h_), int(cfg.bev_w_)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    nusc = NuScenesAdapter(args.version, args.dataroot)
    tokens = sorted({row[key] for row in read_rows(args.pairs) for key in ("sample_token", "next_sample_token")})
    source = {"encoder": "UniAD/OpenDriveVLA BEVFormer", "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": digest(args.checkpoint), "model_config_sha256": digest(args.model_config)}
    for token in tqdm(tokens, desc="import mature spatial BEV", unit="sample"):
        path = Path(args.cache_root)/f"{token}.pth"
        # Full results contain UniAD instance classes: use the original trusted export environment.
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("sample_token") != token:
            raise ValueError(f"Sample token mismatch: {path}")
        value = cache["result_track"].get("bev_embed")
        if value is None:
            raise ValueError(f"No spatial bev_embed in {path}; export with --full-results")
        h, w = height, width
        if h == w == 100 and value.shape[0] == 200*200:
            h = w = 200
        feature = spatial_bev(value.detach().float().cpu().numpy(), h, w)
        record = nusc.make_sample_record(nusc.nusc.get("sample", token))
        write_cached_bev(output, token, feature, record, nusc.nusc.get("sample_data", record.lidar_token)["timestamp"],
                         pc_range, {**source, "source_cache_sha256": digest(path)})
    write_json(output/"cache_protocol.json", {**source, "pair_manifest_sha256": digest(args.pairs),
        "samples": len(tokens), "height": height, "width": width, "pc_range": pc_range,
        "coordinate_frame": "LIDAR_TOP", "model_executed_here": False})


if __name__ == "__main__":
    main()
