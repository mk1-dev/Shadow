from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shadow_occ.annotation.data import write_rows
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.geometry.projection import project_points_to_image
from shadow_occ.geometry.transforms import apply_transform


def main() -> None:
    parser = argparse.ArgumentParser("Export full nuScenes caster GT for annotation training.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", choices=["train", "val", "mini_train", "mini_val"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--exclude-scenes", help="JSON list of scene names")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-distance", type=float, default=30.0)
    parser.add_argument("--min-lidar-points", type=int, default=5)
    parser.add_argument("--min-visible-fraction", type=float, default=0.5)
    parser.add_argument("--min-box-area", type=float, default=400.0)
    parser.add_argument("--include-offscreen", action="store_true", help="Retain GT casters not visible in this camera for outside-BEV hypothesis supervision")
    parser.add_argument("--cams", nargs="+", default=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    adapter = NuScenesAdapter(args.version, args.dataroot)
    official = set(adapter.scene_names_for_split(args.split))
    selected = set(args.scene) if args.scene else official
    if not selected <= official:
        raise ValueError("Requested scenes are outside the specified official split")
    excluded = set(json.loads(Path(args.exclude_scenes).read_text(encoding="utf-8"))) if args.exclude_scenes else set()
    selected -= excluded
    if not selected:
        raise ValueError("No scenes selected")

    def records():
        for record in adapter.iter_samples(sorted(selected), args.max_samples):
            for token in record.sample["anns"]:
                annotation = adapter.nusc.get("sample_annotation", token)
                category = annotation.get("category_name", "")
                if not category.startswith("vehicle."):
                    continue
                if int(annotation.get("num_lidar_pts", 0)) < args.min_lidar_points:
                    continue
                box = adapter.nusc.get_box(token)
                global_to_ego = np.linalg.inv(record.ego_to_global)
                corners_ego = apply_transform(box.corners().T, global_to_ego)
                center_ego = apply_transform(np.asarray([annotation["translation"]]), global_to_ego)[0]
                if np.linalg.norm(center_ego[:2]) > args.max_distance:
                    continue
                for name in args.cams:
                    camera = adapter.get_camera(record, name)
                    uv, _, depths = project_points_to_image(corners_ego, record.ego_to_global, camera.calibration)
                    width, height = camera.calibration.image_size
                    near_plane_unknown = bool((np.any(depths > 0.1) and np.any(depths <= 0.1)) or (np.any(depths > 0.1) and not np.isfinite(uv).all()))
                    visible_box, target_box = None, None
                    if np.all(depths > 0.1) and np.isfinite(uv).all():
                        lower, upper = uv.min(axis=0), uv.max(axis=0)
                        clipped_lower = np.maximum(lower, [0, 0])
                        clipped_upper = np.minimum(upper, [width, height])
                        area = float(np.prod(np.maximum(clipped_upper - clipped_lower, 0)))
                        raw_area = float(np.prod(np.maximum(upper - lower, 0)))
                        if area > 0:
                            visible_box = np.r_[clipped_lower, clipped_upper].tolist()
                        if area >= args.min_box_area and area/max(raw_area, 1.0) >= args.min_visible_fraction:
                            target_box = visible_box
                    if target_box is None and not args.include_offscreen:
                        continue
                    yield {
                        "id": f"{record.token}_{name}_{token}",
                        "scene_name": record.scene_name,
                        "sample_token": record.token,
                        "camera": name,
                        "official_split": args.split,
                        "image": str(camera.image_path.resolve()),
                        "annotation_token": token,
                        "instance_token": annotation["instance_token"],
                        "caster_source": "annotation_gt",
                        "caster_category": category,
                        "caster_bbox_xyxy": target_box,
                        "caster_visible_bbox_xyxy": visible_box,
                        "caster_projection_near_plane_unknown": near_plane_unknown,
                        "caster_corners_ego": corners_ego.tolist(),
                        "caster_center_ego": center_ego.tolist(),
                        "caster_size_wlh": annotation["size"],
                        "caster_rotation_global": annotation["rotation"],
                        "caster_center_global": annotation["translation"],
                        "previous_annotation_token": annotation.get("prev", ""),
                        "next_annotation_token": annotation.get("next", ""),
                        "visibility_token": annotation.get("visibility_token", ""),
                        "timestamp": record.timestamp,
                        "ego_to_global": record.ego_to_global.tolist(),
                        "ego_to_img": (camera.calibration.intrinsic @ (camera.calibration.global_to_sensor @ record.ego_to_global)[:3]).tolist(),
                        "image_size_wh": [width, height],
                    }

    write_rows(output, records())
    print(f"Wrote caster GT to {output}. These are object annotations, not shadow labels.")


if __name__ == "__main__":
    main()
