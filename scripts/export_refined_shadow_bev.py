from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import read_rows
from shadow_occ.config import load_config
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Back-project refined image masks to BEV override masks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ground-z", type=float, default=0.0)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    grid = BEVGrid.from_config(cfg.bev)
    yy, xx = np.indices(grid.shape)
    xy = grid.ij_to_xy(np.stack([yy.ravel(), xx.ravel()], axis=1))
    points = np.c_[xy, np.full(len(xy), args.ground_z), np.ones(len(xy))]
    groups = defaultdict(list)
    for row in read_rows(args.manifest):
        if not row.get("sample_token") or row.get("ego_to_img") is None:
            raise ValueError("BEV export requires original sample_token and ego_to_img metadata")
        groups[(row["scene_name"], row["sample_token"])].append(row)
    for (scene, token), rows in groups.items():
        positive = np.zeros(len(points), dtype=bool)
        observed = np.zeros(len(points), dtype=bool)
        for row in rows:
            target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
            if target is None:
                raise FileNotFoundError(row["target"])
            projected = points @ np.asarray(row["ego_to_img"]).T
            depth = projected[:, 2]
            uv = np.rint(projected[:, :2]/np.maximum(depth[:, None], 1e-8)).astype(np.int64)
            valid = (depth > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < target.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < target.shape[0])
            indices = np.where(valid)[0]
            values = target[uv[indices, 1], uv[indices, 0]]
            positive[indices[values == 1]] = True
            observed[indices[values != 255]] = True
        folder = output/scene
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/f"{token}.npz", shadow_mask=positive.reshape(grid.shape).astype(np.uint8), observed_mask=observed.reshape(grid.shape).astype(np.uint8))
    write_json(output/"export_protocol.json", {**vars(args), "samples": len(groups), "format": "BEV masks with observed_mask; not complete OCC-training shadow labels", "next_step": "Preserve observed_mask as an ignore mask and rebuild geometry labels before any future OCC training integration"})


if __name__ == "__main__":
    main()
