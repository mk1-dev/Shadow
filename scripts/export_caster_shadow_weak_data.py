from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shadow_occ.annotation.data import checked_write_image, read_rows, write_rows
from shadow_occ.config import load_config
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.geometry.projection import project_points_to_image
from shadow_occ.utils.io import write_json


def components(mask, minimum):
    count, ids, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [ids == index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= minimum]


def main():
    parser = argparse.ArgumentParser("Bridge nuScenes caster GT and existing BEV pseudo masks to image weak supervision.")
    parser.add_argument("--casters", nargs="+", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ground-z", type=float, default=0.0, help="Approximate ego-frame ground height, in metres")
    parser.add_argument("--ground-tolerance", type=float, default=0.25)
    parser.add_argument("--support-radius", type=int, default=5)
    parser.add_argument("--depth-tolerance", type=float, default=0.3)
    parser.add_argument("--bright-percentile", type=float, default=75)
    parser.add_argument("--min-ground-fraction", type=float, default=0.3)
    parser.add_argument("--min-region-area", type=int, default=50)
    parser.add_argument("--max-regions-per-class", type=int, default=3)
    parser.add_argument("--context-metres", type=float, default=6.0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    grid = BEVGrid.from_config(cfg.bev)
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    adapter = NuScenesAdapter(cfg.data.version, args.dataroot)
    groups = defaultdict(list)
    for caster_manifest in args.casters:
        for row in read_rows(caster_manifest):
            if row.get("caster_bbox_xyxy"):
                groups[(row["sample_token"], row["camera"])].append(row)
    rows, counters = [], Counter()
    yy, xx = np.indices(grid.shape)
    points = np.stack([grid.x_min+(xx+0.5)*grid.resolution, grid.y_min+(yy+0.5)*grid.resolution, np.full(grid.shape, args.ground_z), np.ones(grid.shape)], axis=-1)
    grid_to_ego = np.array([[grid.resolution, 0, grid.x_min+grid.resolution/2], [0, grid.resolution, grid.y_min+grid.resolution/2], [0, 0, args.ground_z], [0, 0, 1]])
    for (token, camera_name), casters in groups.items():
        record = adapter.make_sample_record(adapter.nusc.get("sample", token))
        path = Path(args.label_root)/record.scene_name/f"{token}.npz"
        if not path.exists():
            counters["missing_pseudo_mask"] += 1
            continue
        with np.load(path, allow_pickle=False) as archive:
            pseudo = np.asarray(archive["shadow_mask"]) > 0
        if pseudo.shape != grid.shape:
            raise ValueError(f"BEV-grid mismatch at {path}")
        camera = adapter.get_camera(record, camera_name)
        image = cv2.imread(str(camera.image_path))
        if image is None:
            raise FileNotFoundError(camera.image_path)
        height, width = image.shape[:2]
        projection = np.asarray(casters[0]["ego_to_img"])
        forward = np.einsum("ij,hwj->hwi", projection, points)[..., 2] > 0.1
        lidar = adapter.load_lidar_points_ego(record)
        uv, valid, depths = project_points_to_image(lidar, record.ego_to_global, camera.calibration)
        pixels = np.rint(uv).astype(np.int64)
        valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        zbuffer = np.full((height, width), np.inf, dtype=np.float32)
        np.minimum.at(zbuffer, (pixels[valid, 1], pixels[valid, 0]), depths[valid])
        zbuffer = cv2.erode(zbuffer, np.ones((5, 5), np.uint8))
        ground = valid & (np.abs(lidar[:, 2]-args.ground_z) <= args.ground_tolerance)
        indices = np.where(ground)[0]
        ground[indices] &= depths[indices] <= zbuffer[pixels[indices, 1], pixels[indices, 0]]+args.depth_tolerance
        support = np.zeros((height, width), dtype=np.uint8)
        support[pixels[ground, 1], pixels[ground, 0]] = 1
        support = cv2.dilate(support, np.ones((2*args.support_radius+1, 2*args.support_radius+1), np.uint8)) > 0
        # Keep observed ground in front of a ground-plane ray intersection.
        object_mask = np.zeros((height, width), dtype=np.uint8)
        for caster in casters:
            x0, y0, x1, y1 = np.rint(caster["caster_bbox_xyxy"]).astype(int)
            object_mask[max(0,y0):min(height,y1), max(0,x0):min(width,x1)] = 1
        support &= object_mask == 0
        if not support.any():
            counters["no_ground_observations"] += 1
            continue
        support_path = output/"ground_support"/f"{token}_{camera_name}.png"
        checked_write_image(support_path, support.astype(np.uint8)*255)
        light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
        bright = light >= np.percentile(light[support], args.bright_percentile)
        full_shadow = cv2.warpPerspective((pseudo & forward).astype(np.uint8), projection@grid_to_ego, (width, height), flags=cv2.INTER_NEAREST) > 0
        shadow_buffer = cv2.dilate(full_shadow.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        for caster in casters:
            center = np.asarray(caster["caster_center_ego"])
            footprint_radius = np.linalg.norm(caster["caster_size_wlh"][:2])/2
            nearby = np.linalg.norm(points[..., :2]-center[:2], axis=-1) <= footprint_radius+args.context_metres
            local = cv2.warpPerspective((nearby & forward).astype(np.uint8), projection@grid_to_ego, (width, height), flags=cv2.INTER_NEAREST) > 0
            target = np.full((height, width), 255, dtype=np.uint8)
            target[local & support & bright & ~shadow_buffer] = 0
            target[local & support & full_shadow] = 1
            proposals = []
            for index, mask in enumerate(sorted(components(full_shadow & local & (object_mask == 0), args.min_region_area), key=lambda m: int(m.sum()), reverse=True)):
                if np.count_nonzero(mask & support)/mask.sum() >= args.min_ground_fraction:
                    proposals.append((f"positive_{index}", mask, "YES"))
                    if len(proposals) >= args.max_regions_per_class:
                        break
            # Bright-ground examples are weak negatives, never shadow GT.
            for index, mask in enumerate(sorted(components(target == 0, args.min_region_area), key=lambda m: int(m.sum()), reverse=True)[:args.max_regions_per_class]):
                proposals.append((f"negative_{index}", mask, "NO"))
            if not (target == 1).any() or not (target == 0).any():
                counters["missing_positive_or_negative"] += 1
                continue
            stem = hashlib.sha256(caster["id"].encode()).hexdigest()[:24]
            mask_path = output/"targets"/f"{stem}.png"
            checked_write_image(mask_path, target)
            regions = []
            for region_id, mask, label in proposals:
                mask_file = output/"regions"/f"{stem}_{region_id}.png"
                checked_write_image(mask_file, mask.astype(np.uint8)*255)
                regions.append({"id": region_id, "mask": str(mask_file), "shadow_label": label, "label_source": "geometry_weak"})
            rows.append({**caster, "shadow_mask": str(mask_path), "mask_encoding": "trimap", "label_source": "geometry_weak", "regions": regions, "ground_support": str(support_path), "z0_assumption": True, "ground_z_ego": args.ground_z})
            counters["examples"] += 1
            counters["regions"] += len(regions)
    write_rows(output/"manifest.jsonl", rows)
    write_json(output/"export_protocol.json", {**vars(args), "counts": dict(counters), "supervision": "object GT + image/BEV weak labels; no shadow GT is created", "ground_evidence": "sparse LiDAR near the assumed ego ground plane; not semantic ground truth"})
    print(f"Wrote {len(rows)} weak examples to {output}")


if __name__ == "__main__":
    main()
