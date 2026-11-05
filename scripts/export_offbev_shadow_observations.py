from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import checked_write_image, read_rows, write_rows
from shadow_occ.annotation.readiness import require_prediction_for_proposals
from shadow_occ.annotation.offbev import boundary_hypotheses
from shadow_occ.annotation.temporal import plane_matrix, warp
from shadow_occ.config import load_config
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.geometry.projection import project_points_to_image
from shadow_occ.utils.io import write_json


def crop_box(mask, scale):
    yy, xx = np.where(mask)
    x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
    cx, cy = (x0+x1)/2, (y0+y1)/2
    hw, hh = max(24, (x1-x0)*scale/2), max(24, (y1-y0)*scale/2)
    return [max(0, int(cx-hw)), max(0, int(cy-hh)), min(mask.shape[1], int(np.ceil(cx+hw))), min(mask.shape[0], int(np.ceil(cy+hh)))]


def main():
    parser = argparse.ArgumentParser("Export visible boundary-shadow observations without using caster GT.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--label-root", required=True, help="Existing or refined BEV shadow mask directory")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=["train", "val", "mini_train", "mini_val"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--near-boundary", type=float, default=8.0)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--min-visible-ratio", type=float, default=0.5)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seg-predictions", help="Optional segmenter predictions.jsonl, in source image coordinates")
    parser.add_argument("--segmenter-threshold", type=float, default=0.7)
    parser.add_argument("--exclude-scenes", help="JSON list")
    args = parser.parse_args()
    import json

    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    grid = BEVGrid.from_config(cfg.bev)
    bounds = [grid.x_min, grid.y_min, grid.x_max, grid.y_max]
    adapter = NuScenesAdapter(cfg.data.version, args.dataroot)
    excluded = set(json.loads(Path(args.exclude_scenes).read_text(encoding="utf-8"))) if args.exclude_scenes else set()
    scenes = [scene for scene in adapter.scene_names_for_split(args.split) if scene not in excluded]
    if not scenes:
        raise ValueError("No official scenes selected")
    prediction_index = {}
    if args.seg_predictions:
        for row in read_rows(args.seg_predictions):
            require_prediction_for_proposals(row)
            prediction_index.setdefault((row["sample_token"], row["camera"]), []).append(row)
    grid_to_ego = np.array([[grid.resolution, 0, grid.x_min+grid.resolution/2], [0, grid.resolution, grid.y_min+grid.resolution/2], [0, 0, args.ground_z], [0, 0, 1]])
    observations = []
    for record in adapter.iter_samples(scenes, args.max_samples):
        label_path = Path(args.label_root)/record.scene_name/f"{record.token}.npz"
        if not label_path.exists():
            continue
        with np.load(label_path, allow_pickle=False) as archive:
            mask = np.asarray(archive["shadow_mask"], dtype=np.uint8)
        if mask.shape != grid.shape:
            raise ValueError("BEV mask and configured bounds disagree")
        count, component_ids, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = []
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < args.min_area:
                continue
            pixels = np.column_stack(np.where(component_ids == component))
            xy = grid.ij_to_xy(pixels)
            bank = boundary_hypotheses(xy, bounds, near_boundary=args.near_boundary)
            if bank:
                components.append((component, xy, bank))
        for camera_name in cfg.data.cams:
            camera = adapter.get_camera(record, camera_name)
            image = cv2.imread(str(camera.image_path))
            if image is None:
                raise FileNotFoundError(camera.image_path)
            projection = camera.calibration.intrinsic @ (camera.calibration.global_to_sensor @ record.ego_to_global)[:3]
            previous_image = None
            previous_sample_token, previous_homography = None, None
            previous = record.sample.get("prev")
            if previous:
                previous_record = adapter.make_sample_record(adapter.nusc.get("sample", previous))
                if 0 < (record.timestamp-previous_record.timestamp)/1e6 <= 0.75:
                    old_camera = adapter.get_camera(previous_record, camera_name)
                    old_projection = old_camera.calibration.intrinsic @ (old_camera.calibration.global_to_sensor @ previous_record.ego_to_global)[:3]
                    previous_image = cv2.imread(str(old_camera.image_path))
                    if previous_image is not None:
                        try:
                            previous_to_current = projection@plane_matrix(args.ground_z)@np.linalg.inv(old_projection@np.linalg.inv(previous_record.ego_to_global)@record.ego_to_global@plane_matrix(args.ground_z))
                            previous_image = warp(previous_image, previous_to_current, image.shape[:2])
                            previous_sample_token = previous_record.token
                            previous_homography = previous_to_current.tolist()
                        except np.linalg.LinAlgError:
                            previous_image = None
            segmenter = None
            if args.seg_predictions:
                predictions = prediction_index.get((record.token, camera_name), [])
                if not predictions:
                    continue
                for prediction in predictions:
                    with np.load(prediction["probability"], allow_pickle=False) as archive:
                        probability = np.asarray(archive["probability"], dtype=np.float32)
                    if probability.shape != image.shape[:2]:
                        raise ValueError("Segmenter output must use the full source image coordinates")
                    segmenter = probability if segmenter is None else np.maximum(segmenter, probability)
            for component, xy, bank in components:
                points = np.c_[xy, np.full(len(xy), args.ground_z)]
                _, visible, _ = project_points_to_image(points, record.ego_to_global, camera.calibration)
                if float(visible.mean()) < args.min_visible_ratio:
                    continue
                component_mask = (component_ids == component).astype(np.uint8)
                depths = (np.c_[points, np.ones(len(points))]@projection.T)[:, 2]
                cells = np.column_stack(np.where(component_ids == component))
                component_mask[cells[depths <= 0.1, 0], cells[depths <= 0.1, 1]] = 0
                image_mask = cv2.warpPerspective(component_mask, projection@grid_to_ego, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST) > 0
                if segmenter is not None:
                    image_mask &= segmenter >= args.segmenter_threshold
                if image_mask.sum() < 40:
                    continue
                tight, context = crop_box(image_mask, 1.5), crop_box(image_mask, 2.2)
                x0, y0, x1, y1 = tight
                cx0, cy0, cx1, cy1 = context
                clean = image[y0:y1, x0:x1]
                reference = clean.copy()
                contours, _ = cv2.findContours(image_mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(reference, contours, -1, (245, 245, 245), 1, cv2.LINE_AA)
                identifier = f"{record.token}:{camera_name}:{component}"
                stem = hashlib.sha256(identifier.encode()).hexdigest()[:24]
                region_path = output/"observed_masks"/f"{stem}.png"
                checked_write_image(region_path, image_mask.astype(np.uint8)*255)
                views = [clean, reference, image[cy0:cy1, cx0:cx1]]
                if previous_image is not None:
                    views.append(previous_image[cy0:cy1, cx0:cx1])
                paths = []
                for i, view in enumerate(views):
                    path = output/"views"/f"{stem}_{i}.png"
                    checked_write_image(path, view)
                    paths.append(str(path))
                observations.append({"id": identifier, "region_mask": str(region_path), "scene_name": record.scene_name, "sample_token": record.token, "camera": camera_name, "official_split": args.split, "images": paths, "image_crops_xyxy": [tight, tight, context], "previous_context_crop_xyxy": context if previous_image is not None else None, "previous_sample_token": previous_sample_token, "previous_to_current_homography": previous_homography, "image": str(camera.image_path.resolve()), "timestamp": record.timestamp, "candidates": bank, "bev_bounds_xyxy": bounds, "visible_shadow_point_ratio": float(visible.mean()), "shadow_centroid_xy_ego": xy.mean(axis=0).tolist(), "label_source": "unlabeled", "observation_source": "segmenter_filtered_pseudo_mask" if segmenter is not None else "pseudo_shadow_mask", "caster_gt_used_for_observation": False})
    for observation in observations:
        observation.update(shadow_label="YES", shadow_label_source="geometry_weak",
                           shadow_evidence={"method": observation["observation_source"], "is_shadow_gt": False})
    write_rows(output/"observations.jsonl", observations)
    write_json(output/"observation_protocol.json", {**vars(args), "observations": len(observations), "bounds": bounds, "candidates": "both shadow-axis directions, outside-region uncertainty cells; no caster GT used", "previous_frame_alignment": "ego-motion world ground only; no hidden-caster GT motion is supplied as input"})


if __name__ == "__main__":
    main()
