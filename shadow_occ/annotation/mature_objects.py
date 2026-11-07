"""Decoded UniAD 3D predictions and their camera-space motion explanations."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from shadow_occ.annotation.data import digest
from shadow_occ.annotation.physics_motion import project_world, sample_at
from shadow_occ.utils.io import write_json


PROTOCOL = "mature-3d-objects-v1"
# LiDARInstance3DBoxes.corners, not nuScenes Box.corners ordering.
FACES = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
         (3, 2, 6, 7), (0, 3, 7, 4), (1, 5, 6, 2))


def numpy_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def objects_from_uniad(result, record, timestamp, class_names, minimum_score=0.25):
    """Use decoded predicted boxes' OWN corners/centers, never GT/normalized queries."""
    if not 0 <= minimum_score <= 1:
        raise ValueError("Minimum object score must be in [0,1]")
    transform = np.asarray(record.lidar_ego_to_global) @ np.asarray(record.lidar_sensor_to_ego)
    objects, branches = [], []
    for suffix, kind in (("", "tracking"), ("_det", "detection")):
        key = "boxes_3d"+suffix
        if key not in result:
            continue
        branches.append(kind)
        boxes = result[key]
        corners = numpy_value(boxes.corners).astype(np.float64)
        centers = numpy_value(boxes.gravity_center).astype(np.float64)
        tensor = numpy_value(boxes.tensor)
        scores = numpy_value(result["scores_3d"+suffix]).reshape(-1)
        labels = numpy_value(result["labels_3d"+suffix]).reshape(-1)
        ids = numpy_value(result["track_ids"]).reshape(-1) if kind == "tracking" else np.full(len(scores), -1)
        if (corners.shape != (len(scores), 8, 3) or centers.shape != (len(scores), 3)
                or len(ids) != len(scores) or len(labels) != len(scores)
                or tensor.ndim != 2 or tensor.shape[0] != len(scores)):
            raise ValueError("Decoded UniAD box/score/label/track arrays are not aligned")
        if not np.isfinite(corners).all() or not np.isfinite(centers).all():
            raise ValueError("Nonfinite decoded 3D predictions")
        world_corners = corners @ transform[:3, :3].T+transform[:3, 3]
        world_centers = centers @ transform[:3, :3].T+transform[:3, 3]
        for index, score in enumerate(scores):
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid decoded object confidence")
            if score < minimum_score:
                continue
            label, track_id = int(labels[index]), int(ids[index])
            if label < 0 or label >= len(class_names):
                raise ValueError("Prediction label is outside the model class vocabulary")
            token = f"{record.scene_name}:pred-track:{track_id}" if track_id >= 0 else None
            velocity = None
            if tensor.shape[1] >= 9 and np.isfinite(tensor[index, 7:9]).all():
                velocity = (transform[:3, :3] @ np.r_[tensor[index, 7:9], 0.]).tolist()
            objects.append({"id": token or f"{record.token}:pred-det:{kind}:{index}",
                "track_token": token, "track_id": track_id if track_id >= 0 else None,
                "prediction_kind": kind, "category": str(class_names[label]), "label": label,
                "score": float(score), "center_world": world_centers[index].tolist(),
                "box_lidar": tensor[index, :7].astype(float).tolist(),
                "corners_world": world_corners[index].tolist(), "velocity_world_mps": velocity})
    if not branches:
        raise ValueError("No decoded boxes_3d/boxes_3d_det; feature-only results cannot explain 3D entities")
    return {"protocol": PROTOCOL, "sample_token": record.token, "scene_name": record.scene_name,
        "timestamp": int(timestamp), "coordinate_frame": "world", "corner_order": "mmdet3d_lidar",
        "box_lidar_format": "x,y,z_bottom,dx,dy,dz,yaw; use model corners for projection",
        "minimum_score": minimum_score, "class_names": list(class_names), "branches": branches,
        "objects": objects, "gt_used": False, "empty_prediction_is_free_space": False}


def write_cached_objects(root, token, bundle, source):
    path = Path(root)/f"{token}.objects.json"
    write_json(path, {**bundle, "model_source": source})
    return {"objects_file": path.name, "objects_sha256": digest(path), "objects_protocol": PROTOCOL}


def object_frame_source(record, timestamp, pc_range, source):
    return {**source, "lidar_to_world": (np.asarray(record.lidar_ego_to_global)
            @ np.asarray(record.lidar_sensor_to_ego)).tolist(), "pc_range": list(pc_range),
            "timestamp": int(timestamp)}


def load_cached_objects(root, token, feature_source=None):
    path = Path(root)/f"{token}.objects.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mature 3D predictions: {path}; re-export or import full UniAD results")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if (bundle["protocol"] != PROTOCOL or bundle["sample_token"] != token
            or bundle["coordinate_frame"] != "world" or bundle["gt_used"] is not False):
        raise ValueError("Mature object cache sample/frame/provenance mismatch")
    if feature_source is not None:
        if feature_source.get("objects_sha256") and digest(path) != feature_source["objects_sha256"]:
            raise ValueError("Mature object cache changed")
        for key in ("checkpoint_sha256", "model_config_sha256"):
            if bundle["model_source"].get(key) != feature_source.get(key):
                raise ValueError(f"Feature/object model mismatch: {key}")
    return bundle, {"sample_token": token, "path": str(path.resolve()), "sha256": digest(path),
                    "objects": len(bundle["objects"]), "model_source": bundle["model_source"]}


def camera_time_objects(bundle, frame):
    offset = (frame["camera_timestamp"]-bundle["timestamp"])/1e6
    if abs(offset) > .2:
        raise ValueError("Prediction/camera timestamps differ by more than 0.2 seconds")
    result = []
    for item in bundle["objects"]:
        corners = np.asarray(item["corners_world"], np.float64)
        center = np.asarray(item["center_world"], np.float64)
        velocity = item.get("velocity_world_mps")
        if velocity is not None:
            shift = np.asarray(velocity)*offset
            corners, center = corners+shift, center+shift
        result.append({**item, "corners": corners, "center": center,
                       "motion_trackable": bundle.get("model_source", {}).get("history_mode") != "current-only",
                       "time_aligned": velocity is not None or abs(offset) <= .02,
                       "camera_offset_seconds": float(offset)})
    return result


def render_objects(objects, frame):
    """Project box faces with inverse-depth z-buffering; no GT and no ray tracing."""
    width, height = frame["image_size_wh"]
    depth = np.full((height, width), np.inf, np.float32)
    confidence = np.zeros((height, width), np.float32)
    owner = np.full((height, width), -1, np.int32)
    face_map = owner.copy()
    projection = np.asarray(frame["projection_world_to_image"], np.float64)
    camera = np.linalg.solve(projection[:, :3], -projection[:, 3])
    faces = {}
    for index, item in enumerate(objects):
        corners = item["corners"]
        uv, z, valid = project_world(corners, projection)
        for face_id, face in enumerate(FACES):
            ids = np.asarray(face)
            if not valid[ids].all():
                continue
            xyz = corners[ids]
            normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
            midpoint = xyz.mean(axis=0)
            if normal @ (midpoint-item["center"]) < 0:
                normal = -normal
            if normal @ (camera-midpoint) <= 0:
                continue
            pixels = uv[ids].astype(np.float32)
            if cv2.contourArea(pixels) < 1:
                continue
            lower = np.maximum(np.floor(pixels.min(axis=0)).astype(int), 0)
            upper = np.minimum(np.ceil(pixels.max(axis=0)).astype(int)+1, [width, height])
            x0, y0 = lower
            x1, y1 = upper
            if x1 <= x0 or y1 <= y0:
                continue
            inverse_depth = np.linalg.lstsq(np.c_[pixels, np.ones(4)], 1/z[ids], rcond=None)[0]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            inv_z = inverse_depth[0]*xx+inverse_depth[1]*yy+inverse_depth[2]
            patch = np.zeros(xx.shape, np.uint8)
            cv2.fillConvexPoly(patch, np.rint(pixels-[x0, y0]).astype(np.int32), 1)
            face_depth = np.divide(1., inv_z, out=np.full_like(inv_z, np.inf), where=inv_z > 0)
            section = np.s_[y0:y1, x0:x1]
            nearest = (patch > 0) & (inv_z > 0) & (face_depth < depth[section]-1e-3)
            depth[section][nearest] = face_depth[nearest]
            owner[section][nearest] = index
            face_map[section][nearest] = face_id
            confidence[section][nearest] = item["score"]
            faces[index, face_id] = (pixels, inverse_depth)
    return {"depth": depth, "confidence": confidence, "owner": owner, "face": face_map, "faces": faces}


def projected_object_motion(frames, bundles, evidence):
    objects = [camera_time_objects(bundle, frame) for bundle, frame in zip(bundles, frames)]
    rendered = [render_objects(values, frame) for values, frame in zip(objects, frames)]
    motion = []
    for slot, flow_evidence in enumerate(evidence):
        height, width = rendered[slot]["depth"].shape
        expected = np.full((height, width, 2), np.nan, np.float32)
        object_residual = np.full((height, width), np.nan, np.float32)
        ground_residual = object_residual.copy()
        fit = object_residual.copy()
        next_lookup = {item["track_token"]: index for index, item in enumerate(objects[slot+1])
                       if item["track_token"] is not None and item["prediction_kind"] == "tracking"}
        reliable = flow_evidence.get("flow_reliable", np.zeros((height, width), bool))
        for (index, face_id), (pixels, _) in rendered[slot]["faces"].items():
            item = objects[slot][index]
            target = next_lookup.get(item["track_token"])
            if (target is None or not item["time_aligned"] or not item["motion_trackable"]
                    or not objects[slot+1][target]["time_aligned"] or not objects[slot+1][target]["motion_trackable"]):
                continue
            target_face = rendered[slot+1]["faces"].get((target, face_id))
            if target_face is None:
                continue
            yy, xx = np.where((rendered[slot]["owner"] == index) & (rendered[slot]["face"] == face_id) & reliable)
            if not len(xx):
                continue
            homography = cv2.getPerspectiveTransform(pixels, target_face[0])
            if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
                continue
            source = np.c_[xx, yy].astype(np.float32)
            projected = cv2.perspectiveTransform(source[None], homography)[0]
            next_width, next_height = frames[slot+1]["image_size_wh"]
            valid = (np.isfinite(projected).all(axis=1) & (projected >= 0).all(axis=1)
                     & (projected[:, 0] <= next_width-1) & (projected[:, 1] <= next_height-1))
            owners = sample_at(rendered[slot+1]["owner"].astype(np.float32), projected[None], nearest=True)[0]
            faces = sample_at(rendered[slot+1]["face"].astype(np.float32), projected[None], nearest=True)[0]
            valid &= (owners == target) & (faces == face_id)
            yy, xx, projected, source = yy[valid], xx[valid], projected[valid], source[valid]
            delta = projected-source
            observed = flow_evidence["flow_t_to_next"][yy, xx]
            err_object = np.linalg.norm(observed-delta, axis=1)
            ground_flow = flow_evidence["ground_flow"][yy, xx]
            ground_endpoints = source+ground_flow
            ground_valid = ((ground_endpoints >= 0).all(axis=1)
                & (ground_endpoints[:, 0] <= width-1) & (ground_endpoints[:, 1] <= height-1))
            err_ground = np.where(ground_valid, np.linalg.norm(observed-ground_flow, axis=1), np.nan)
            finite = np.isfinite(observed).all(axis=1) & np.isfinite(err_object)
            yy, xx = yy[finite], xx[finite]
            expected[yy, xx] = delta[finite]
            object_residual[yy, xx] = err_object[finite]
            ground_residual[yy, xx] = err_ground[finite]
            advantage = np.clip(np.nan_to_num((err_ground[finite]-err_object[finite])/3, nan=0), -1, 1)
            fit[yy, xx] = np.exp(-err_object[finite]/3)*(.5+.5*advantage)
        motion.append({"expected": expected, "object_residual": object_residual,
                       "ground_residual": ground_residual, "entity_fit": fit})
    return rendered, motion


def attach_mature_objects(arrays, metadata, frames, evidence, root, feature_sources=None):
    sources, bundles = [], []
    for slot, token in enumerate(metadata["sample_tokens"]):
        bundle, source = load_cached_objects(root, token, feature_sources[slot] if feature_sources else None)
        if bundle["scene_name"] != metadata["scene_name"]:
            raise ValueError("Mature predictions cross scene boundaries")
        bundles.append(bundle)
        sources.append(source)
    rendered, motion = projected_object_motion(frames, bundles, evidence)
    world = arrays["receiver_world_xyz"]
    probabilities, expected, residuals, ground_residuals, fits, domains = [], [], [], [], [], []
    for slot, frame in enumerate(frames):
        uv, _, valid = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
        width, height = frame["image_size_wh"]
        valid &= ((uv >= 0).all(axis=1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1))
        coordinates = np.where(valid[:, None], uv, -1).reshape(*world.shape[:2], 2).astype(np.float32)
        source = bundles[slot]["model_source"]
        transform = np.asarray(source["lidar_to_world"])
        local = np.c_[world.reshape(-1, 3), np.ones(world.shape[0]*world.shape[1])] @ np.linalg.inv(transform).T
        bounds = np.asarray(source["pc_range"])
        domain = valid & (local[:, :2] >= bounds[:2]).all(axis=1) & (local[:, :2] <= bounds[3:5]).all(axis=1)
        domains.append(domain.reshape(world.shape[:2]))
        probabilities.append(np.where(domains[-1], sample_at(rendered[slot]["confidence"], coordinates), np.nan))
        if slot < 2:
            for target, name in ((expected, "expected"), (residuals, "object_residual"),
                                 (ground_residuals, "ground_residual"), (fits, "entity_fit")):
                sampled = sample_at(motion[slot][name], coordinates, nearest=True)
                valid_sample = domains[-1][..., None] if sampled.ndim == 3 else domains[-1]
                target.append(np.where(valid_sample, sampled, np.nan))
    arrays.update(mature_entity_probability=np.stack(probabilities).astype(np.float32),
        mature_inference_domain=np.stack(domains), mature_expected_object_flow_px=np.stack(expected),
        mature_object_flow_residual_px=np.stack(residuals), mature_ground_flow_residual_px=np.stack(ground_residuals),
        mature_entity_motion_fit=np.stack(fits))
    arrays["mature_receiver_without_predicted_entity"] = (arrays["receiver_valid"]
        & arrays["mature_inference_domain"][1] & (arrays["mature_entity_probability"][1] < .25))
    metadata["mature_objects"] = {"protocol": PROTOCOL, "sources": sources, "gt_used": False,
        "motion": "matched predicted track, projected box-face homography; not box-center flow",
        "absence": "no predicted entity is not observed free space"}
    metadata["counts"]["mature_predicted_objects"] = sum(len(bundle["objects"]) for bundle in bundles)
    metadata["counts"]["mature_object_motion_cells"] = int(np.isfinite(arrays["mature_entity_motion_fit"]).sum())
    metadata["counts"]["mature_unexplained_dark_cells"] = int((arrays["shadow_candidate"]
        & arrays["mature_receiver_without_predicted_entity"]).sum())
    return arrays, metadata, rendered, motion
