from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import checked_write_image, digest, group_split, write_rows
from shadow_occ.annotation.physics_motion import (
    decompose_object_motion, fit_ground_plane, ground_homography, interpolate_box_state,
    plane_basis, project_world,
)
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.geometry.transforms import apply_transform
from shadow_occ.utils.io import write_json


def annotation_state(adapter, annotation, camera_timestamp, max_gap):
    def timed(item):
        return {**item, "timestamp": adapter.nusc.get("sample", item["sample_token"])["timestamp"]}

    anchor = timed(annotation)
    time = int(anchor["timestamp"])
    if time == camera_timestamp:
        return interpolate_box_state(anchor, anchor, camera_timestamp)
    neighbor_token = annotation.get("prev" if camera_timestamp < time else "next")
    if not neighbor_token:
        raise ValueError("camera_timestamp_outside_track_bracket")
    neighbor = timed(adapter.nusc.get("sample_annotation", neighbor_token))
    left, right = sorted([anchor, neighbor], key=lambda item: item["timestamp"])
    if (right["timestamp"]-left["timestamp"])/1e6 > max_gap:
        raise ValueError("annotation_interpolation_gap")
    return interpolate_box_state(left, right, camera_timestamp)


def without_objects(points, annotations):
    from pyquaternion import Quaternion

    keep = np.ones(len(points), bool)
    for annotation in annotations:
        rotation = Quaternion(annotation["rotation"]).rotation_matrix
        local = (points-np.asarray(annotation["translation"])) @ rotation
        w, length, height = annotation["size"]
        keep &= ~np.all(np.abs(local) <= np.array([length, w, height])/2+0.25, axis=1)
    return keep


def scaled_projection(camera, working_size):
    width, height = working_size
    original_width, original_height = camera.calibration.image_size
    sx, sy = width/original_width, height/original_height
    # Match cv2.resize's pixel-center convention, not just a scale of focal lengths.
    resize = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    return resize @ camera.calibration.intrinsic @ camera.calibration.global_to_sensor[:3]


def object_image_polygon(corners, projection, shape, *, near_depth=0.1, margin=3):
    corners = np.asarray(corners, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    if (corners.shape != (8, 3) or projection.shape != (3, 4)
            or not np.isfinite(corners).all() or not np.isfinite(projection).all()):
        raise ValueError("invalid_object_projection")
    projected = np.c_[corners, np.ones(8)] @ projection.T
    front = projected[:, 2] >= near_depth
    crossing = bool(front.any() and not front.all())
    empty = np.empty((0, 1, 2), dtype=np.int32)
    if not front.any():
        return empty, crossing
    vertices = [*projected[front]]
    if crossing:
        # All vertex chords include the box edges; extra intersections lie inside
        # the clipped convex box. This does not depend on corner ordering.
        for a in projected[front]:
            for b in projected[~front]:
                fraction = (near_depth-a[2])/(b[2]-a[2])
                vertices.append(a+fraction*(b-a))
    vertices = np.asarray(vertices)
    uv = vertices[:, :2]/vertices[:, 2:3]
    hull = cv2.convexHull(uv.astype(np.float32))
    if len(hull) < 3 or cv2.contourArea(hull) <= 0:
        return empty, crossing
    height, width = shape
    viewport = np.array([[-margin, -margin], [width-1+margin, -margin],
                         [width-1+margin, height-1+margin], [-margin, height-1+margin]], np.float32)
    area, clipped = cv2.intersectConvexConvex(hull, viewport)
    if area <= 0 or clipped is None:
        return empty, crossing
    return np.rint(clipped).astype(np.int32), crossing


def observed_ground(points_world, object_free, projection, plane_world, shape,
                    object_corners, *, basis_world=None, support_radius=3, plane_tolerance=0.15):
    height, width = shape
    uv, depth, valid = project_world(points_world, projection)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= width-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height-1)
    pixels = np.rint(np.nan_to_num(uv, nan=-1)).astype(np.int64)
    zbuffer = np.full(shape, 1e9, np.float32)
    np.minimum.at(zbuffer, (pixels[valid, 1], pixels[valid, 0]), depth[valid])
    zbuffer = cv2.erode(zbuffer, np.ones((3, 3), np.uint8))
    distance = np.abs(points_world @ plane_world[:3]+plane_world[3])
    ground = valid & object_free & (distance <= plane_tolerance)
    indices = np.flatnonzero(ground)
    ground[indices] &= depth[indices] <= zbuffer[pixels[indices, 1], pixels[indices, 0]]+0.3
    support = np.zeros(shape, np.uint8)
    support[pixels[ground, 1], pixels[ground, 0]] = 1
    support = cv2.dilate(support, np.ones((2*support_radius+1, 2*support_radius+1), np.uint8)) > 0
    # All annotated objects, not just candidate vehicles, exclude visible support.
    objects_crossing_near_plane = 0
    near_plane_objects_clipped = 0
    object_exclusion = np.zeros(shape, bool)
    for corners in object_corners:
        polygon, crossing = object_image_polygon(corners, projection, shape)
        objects_crossing_near_plane += int(crossing)
        if not len(polygon):
            continue
        near_plane_objects_clipped += int(crossing)
        object_mask = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(object_mask, polygon, 1)
        object_exclusion |= cv2.dilate(object_mask, np.ones((7, 7), np.uint8)) > 0
    support &= ~object_exclusion
    # A dilated sample is only local planar support, NOT dense visibility truth.
    basis = plane_basis(plane_world) if basis_world is None else np.asarray(basis_world)
    matrix = projection @ basis
    if np.linalg.cond(matrix) > 1e10:
        raise ValueError("singular_ground_projection")
    yy, xx = np.indices(shape, dtype=np.float64)
    parameters = np.linalg.solve(matrix, np.stack([xx.ravel(), yy.ravel(), np.ones(height*width)]))
    inverse_depth = parameters[2].reshape(shape)
    plane_depth = 1/np.maximum(inverse_depth, 1e-9)
    support &= (inverse_depth > 0) & (plane_depth <= zbuffer+0.3)
    return support, {"projected_lidar_points": int(valid.sum()), "projected_ground_points": int(ground.sum()),
                     "supported_pixels": int(support.sum()), "support_radius_px": support_radius,
                     "objects_crossing_near_plane": objects_crossing_near_plane,
                     "near_plane_objects_clipped": near_plane_objects_clipped,
                     "object_exclusion_pixels": int(object_exclusion.sum()),
                     "visibility_scope": "sparse_lidar_local_planar_support_not_dense_ground_gt"}


def finite_list(values):
    array = np.asarray(values)
    return np.where(np.isfinite(array), array, None).tolist()


def main():
    parser = argparse.ArgumentParser("Export Stage 1/2 pairs without reading any shadow labels or Occ3D.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", choices=["train", "val", "mini_train", "mini_val"], required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--cams", nargs="+", default=["CAM_FRONT"])
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--max-gap-seconds", type=float, default=0.75)
    parser.add_argument("--max-camera-lidar-offset", type=float, default=0.1)
    parser.add_argument("--working-width", type=int, default=800)
    parser.add_argument("--support-radius", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if (args.max_pairs < 1 or args.max_attempts < args.max_pairs or args.working_width < 64 or args.support_radius < 0
            or not 0 < args.max_gap_seconds <= 0.75 or args.max_camera_lidar_offset <= 0):
        raise ValueError("Invalid pair, image, support or time limits")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    adapter = NuScenesAdapter(args.version, args.dataroot)
    official = set(adapter.scene_names_for_split(args.split))
    scenes = set(args.scene) if args.scene else official
    if not scenes or not scenes <= official:
        raise ValueError("Select nonempty scenes from the requested official split")
    counts, audit = Counter(), []

    @lru_cache(maxsize=3)
    def geometry(sample_token):
        record = adapter.make_sample_record(adapter.nusc.get("sample", sample_token))
        annotations = [adapter.nusc.get("sample_annotation", token) for token in record.sample["anns"]]
        points_world = apply_transform(adapter.load_lidar_points_ego(record), record.ego_to_global)
        clean = without_objects(points_world, annotations)
        points_ego = apply_transform(points_world, np.linalg.inv(record.ego_to_global))
        plane, plane_details = fit_ground_plane(points_ego[clean], seed=args.seed)
        local_support = clean & (np.linalg.norm(points_ego[:, :2], axis=1) <= 35)
        return record, annotations, points_world, local_support, plane, plane_details

    def records():
        written = 0
        progress = tqdm(total=args.max_pairs, desc="export physics pairs", unit="pair")
        try:
            for record in adapter.iter_samples(sorted(scenes)):
                if not record.sample.get("next"):
                    continue
                next_sample = adapter.nusc.get("sample", record.sample["next"])
                if next_sample["scene_token"] != record.scene_token:
                    raise ValueError("Sample link crosses scenes")
                for name in args.cams:
                    if counts["pairs_attempted"] >= args.max_attempts:
                        return
                    pair_id = f"{record.token}_{next_sample['token']}_{name}"
                    counts["pairs_attempted"] += 1
                    print(f"attempt={counts['pairs_attempted']}/{args.max_attempts} "
                          f"accepted={written}/{args.max_pairs} scene={record.scene_name} camera={name}", flush=True)
                    try:
                        previous, anns_t, points_t, clean_t, plane_t, plane_info = geometry(record.token)
                        following, anns_next, points_next, clean_next, plane_next, _ = geometry(next_sample["token"])
                        camera_t = adapter.get_camera(previous, name)
                        camera_next = adapter.get_camera(following, name)
                        seconds = (camera_next.timestamp-camera_t.timestamp)/1e6
                        if not 0 < seconds <= args.max_gap_seconds:
                            raise ValueError("camera_time_gap")
                        for sample_record, camera in ((previous, camera_t), (following, camera_next)):
                            lidar_time = adapter.nusc.get("sample_data", sample_record.lidar_token)["timestamp"]
                            if abs(camera.timestamp-lidar_time)/1e6 > args.max_camera_lidar_offset:
                                raise ValueError("camera_lidar_time_offset")
                        world_plane = np.linalg.inv(previous.ego_to_global).T @ plane_t
                        next_world_plane = np.linalg.inv(following.ego_to_global).T @ plane_next
                        normal_cos = np.clip(world_plane[:3] @ next_world_plane[:3], -1, 1)
                        origin = previous.ego_to_global[:3, 3]
                        height_disagreement = abs(np.r_[origin, 1] @ (world_plane-next_world_plane))
                        if np.rad2deg(np.arccos(normal_cos)) > 5 or height_disagreement > 0.3:
                            raise ValueError("local_ground_planes_disagree")
                        basis_world = previous.ego_to_global @ plane_basis(plane_t)
                        frames, states = [], []
                        for slot, camera, anns, points, clean in (
                            ("t", camera_t, anns_t, points_t, clean_t),
                            ("next", camera_next, anns_next, points_next, clean_next),
                        ):
                            image = cv2.imread(str(camera.image_path))
                            if image is None:
                                raise FileNotFoundError(camera.image_path)
                            original_height, original_width = image.shape[:2]
                            if (original_width, original_height) != camera.calibration.image_size:
                                raise ValueError("Camera table and image dimensions differ")
                            width = min(original_width, args.working_width)
                            height = max(64, int(round(original_height*width/original_width/8))*8)
                            width = max(64, int(round(width/8))*8)
                            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                            projection = scaled_projection(camera, (width, height))
                            timed_states, object_corners, state_failures = {}, [], []
                            for ann in anns:
                                try:
                                    state = annotation_state(adapter, ann, camera.timestamp, args.max_gap_seconds)
                                    timed_states[ann["instance_token"]] = (ann, state)
                                    object_corners.append(state["corners"])
                                except ValueError as exc:
                                    counts[str(exc)] += 1
                                    state_failures.append({"instance_token": ann["instance_token"], "reason": str(exc)})
                                    # Conservative fallback is for EXCLUSION only, never motion evidence.
                                    object_corners.append(adapter.nusc.get_box(ann["token"]).corners().T)
                            support, support_info = observed_ground(points, clean, projection, world_plane,
                                                                      image.shape[:2], object_corners,
                                                                      basis_world=basis_world,
                                                                      support_radius=args.support_radius)
                            image_path = output/"images"/f"{pair_id}_{slot}.png"
                            ground_path = output/"ground"/f"{pair_id}_{slot}.png"
                            checked_write_image(image_path, image)
                            checked_write_image(ground_path, support.astype(np.uint8)*255)
                            frames.append({"image": str(image_path), "ground_support": str(ground_path),
                                           "image_sha256": digest(image_path), "ground_sha256": digest(ground_path),
                                           "source_image": str(camera.image_path.resolve()), "camera_timestamp": camera.timestamp,
                                           "sample_data_token": camera.token, "projection_world_to_image": projection.tolist(),
                                           "object_state_failures": state_failures,
                                           "image_size_wh": [width, height], "ground_evidence": support_info})
                            states.append(timed_states)
                        if frames[0]["image_size_wh"] != frames[1]["image_size_wh"]:
                            raise ValueError("Temporal image dimensions differ")
                        motions = []
                        for instance in sorted(states[0].keys() & states[1].keys()):
                            ann, a = states[0][instance]
                            ann_next, b = states[1][instance]
                            if not ann["category_name"].startswith("vehicle."):
                                continue
                            landmarks_t = np.vstack([a["translation"], a["corners"]])
                            landmarks_next = np.vstack([b["translation"], b["corners"]])
                            motion = decompose_object_motion(landmarks_t, landmarks_next,
                                frames[0]["projection_world_to_image"], frames[1]["projection_world_to_image"], seconds)
                            motions.append({"instance_token": instance, "category": ann["category_name"],
                                            "annotation_token_t": ann["token"],
                                            "annotation_token_next": ann_next["token"],
                                            "landmarks_world_t": landmarks_t.tolist(),
                                            "landmarks_world_next": landmarks_next.tolist(),
                                            "source": "interpolated_object_gt_not_shadow_gt",
                                            "interpolation_fraction_t": a["interpolation_fraction"],
                                            "interpolation_fraction_next": b["interpolation_fraction"],
                                            "landmark_order": "center_then_eight_box_corners",
                                            **{key: value.tolist() if value.dtype == bool else finite_list(value) for key, value in motion.items()}})
                        homography = ground_homography(frames[0]["projection_world_to_image"], frames[1]["projection_world_to_image"], basis_world)
                        row = {"id": pair_id, "scene_name": previous.scene_name, "camera": name,
                               "official_split": args.split, "split": group_split(previous.scene_name, args.split, 0.1, args.seed),
                               "sample_token": previous.token, "next_sample_token": following.token,
                               "dt_seconds": seconds, "frames": frames, "objects": motions,
                               "plane_basis_world": basis_world.tolist(), "plane_world": world_plane.tolist(),
                               "ground_plane_fit": plane_info, "world_t_to_next": homography.tolist(),
                               "protocol": "physics-motion-evidence-v1", "shadow_label_source": "none",
                               "is_shadow_gt": False, "train_approved": False}
                        yield row
                        counts["pairs_written"] += 1
                        counts["object_tracks_with_motion"] += len(motions)
                        written += 1
                        progress.update(1)
                        progress.set_postfix(scene=previous.scene_name, tracks=len(motions), refresh=False)
                        if written >= args.max_pairs:
                            return
                    except ValueError as exc:
                        counts["pairs_skipped"] += 1
                        audit.append({"id": pair_id, "status": "skipped", "reason": str(exc)})
        finally:
            progress.close()

    write_rows(output/"pairs.jsonl", records())
    write_rows(output/"export_audit.jsonl", audit)
    write_json(output/"export_protocol.json", {**vars(args), "counts": dict(counts),
               "protocol": "physics-motion-evidence-v1", "pair_manifest_sha256": digest(output/"pairs.jsonl"),
               "boundary": "Stage 1/2 only; no shadow masks, Occ3D or automatic training labels used",
               "ground_limitations": "single local LiDAR plane; sparse dilated visibility; no semantic ground truth",
               "motion_limitations": "bracketed GT box interpolation, not dense object surface flow"})
    print(dict(counts), flush=True)
    if not counts["pairs_written"]:
        raise RuntimeError("No valid pairs; inspect export_audit.jsonl instead of treating skipped pairs as negatives")


if __name__ == "__main__":
    main()
