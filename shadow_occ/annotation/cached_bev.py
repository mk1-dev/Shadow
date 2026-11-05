"""Spatial BEV cache shared with the existing UniAD/OpenDriveVLA environment."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from shadow_occ.annotation.data import digest
from shadow_occ.utils.io import write_json


def write_cached_bev(output, token, feature, record, timestamp, pc_range, source):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/f"{token}.npz"
    np.savez_compressed(path, bev_chw=feature.astype(np.float16))
    write_json(output/f"{token}.json", {**source, "sample_token": token, "scene_name": record.scene_name,
        "coordinate_frame": "LIDAR_TOP", "lidar_to_world": (record.lidar_ego_to_global @ record.lidar_sensor_to_ego).tolist(),
        "pc_range": list(pc_range), "shape_chw": list(feature.shape), "feature_sha256": digest(path),
        "layout": "C,H,W; rows=y; columns=x; cell centres", "timestamp": timestamp})


def spatial_bev(value, height, width, layout="HWBC"):
    """UniAD get_bevs returns [H*W, B, C], with columns along metric x."""
    value = np.asarray(value)
    if layout == "HWBC":
        if value.ndim != 3 or value.shape[:2] != (height*width, 1):
            raise ValueError("Expected UniAD spatial bev_embed [H*W, 1, C]")
        value = value[:, 0].reshape(height, width, -1).transpose(2, 0, 1)
    elif layout == "BHWC":
        if value.ndim != 3 or value.shape[:2] != (1, height*width):
            raise ValueError("Expected spatial bev_embed [1, H*W, C]")
        value = value[0].reshape(height, width, -1).transpose(2, 0, 1)
    elif layout == "BCHW":
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[2:] != (height, width):
            raise ValueError("Expected spatial BEV [1, C, H, W]")
        value = value[0]
    else:
        raise ValueError(f"Unknown BEV layout: {layout}")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite BEV features")
    return np.ascontiguousarray(value, dtype=np.float32)


def sample_cached_bev(root, token, world_points):
    """Map receiver world points to the mature model's current LIDAR_TOP grid."""
    root = Path(root)
    meta = json.loads((root/f"{token}.json").read_text(encoding="utf-8"))
    if meta["sample_token"] != token or meta["coordinate_frame"] != "LIDAR_TOP":
        raise ValueError("BEV cache sample/frame mismatch")
    path = root/f"{token}.npz"
    if digest(path) != meta["feature_sha256"]:
        raise ValueError("BEV feature cache changed")
    with np.load(path, allow_pickle=False) as archive:
        feature = archive["bev_chw"].astype(np.float32)
    lower, upper = np.asarray(meta["pc_range"][:3]), np.asarray(meta["pc_range"][3:])
    height, width = feature.shape[1:]
    xyz = np.c_[world_points.reshape(-1, 3), np.ones(world_points.shape[0]*world_points.shape[1])]
    local = (xyz @ np.linalg.inv(np.asarray(meta["lidar_to_world"])).T)[:, :3]
    column = (local[:, 0]-lower[0])/(upper[0]-lower[0])*width-0.5
    row = (local[:, 1]-lower[1])/(upper[1]-lower[1])*height-0.5
    valid = (column >= 0) & (column <= width-1) & (row >= 0) & (row <= height-1)
    shape = world_points.shape[:2]
    sampled = cv2.remap(feature.transpose(1, 2, 0), column.reshape(shape).astype(np.float32),
                        row.reshape(shape).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    if sampled.ndim == 2:
        sampled = sampled[..., None]
    sampled[~valid.reshape(shape)] = 0
    return sampled, valid.reshape(shape), meta


def attach_cached_bev(arrays, metadata, root):
    features, valid, sources = [], [], []
    for token in metadata["sample_tokens"]:
        feature, support, source = sample_cached_bev(root, token, arrays["receiver_world_xyz"])
        features.append(feature)
        valid.append(support)
        sources.append(source)
    features = np.stack(features)
    valid = np.stack(valid) & arrays["visible_by_frame"]
    norm = np.linalg.norm(features, axis=-1)
    similarity = []
    for index in range(2):
        supported = valid[index] & valid[index+1] & (norm[index] > 1e-6) & (norm[index+1] > 1e-6)
        dot = (features[index]*features[index+1]).sum(axis=-1)
        cosine = dot/np.maximum(norm[index]*norm[index+1], 1e-6)
        similarity.append(np.where(supported, np.clip(cosine, -1, 1), np.nan))
    arrays["learned_bev_receiver_features"] = np.where(valid[..., None], features, 0).astype(np.float16)
    arrays["learned_bev_valid"] = valid
    arrays["learned_bev_temporal_cosine"] = np.stack(similarity).astype(np.float32)
    metadata["learned_bev"] = {"sources": sources, "layout": "T,H,W,C",
        "alignment": "world receiver -> each sample LIDAR_TOP; no resize-only alignment"}
    return arrays, metadata
