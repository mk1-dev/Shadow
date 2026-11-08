"""One receiver observation per camera token, shared by pairs and video clips."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from shadow_occ.annotation.data import checked_write_image, digest
from shadow_occ.utils.io import write_json


def geometry_id(plane, basis, anchor):
    value = {"plane_world": np.asarray(plane, float).tolist(),
             "plane_basis_world": np.asarray(basis, float).tolist(),
             "anchor_ego_to_world": np.asarray(anchor, float).tolist()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_frame_receiver(root, frame, image, support, plane, basis, anchor):
    root = Path(root)
    token = frame["sample_data_token"]
    folder = root/"frames"/token
    folder.mkdir(parents=True, exist_ok=True)
    image_path, ground_path = folder/"image.png", folder/"ground.png"
    checked_write_image(image_path, image)
    checked_write_image(ground_path, np.asarray(support, np.uint8)*255)
    model_id = geometry_id(plane, basis, anchor)
    observation = {**frame, "image": str(image_path), "ground_support": str(ground_path),
        "image_sha256": digest(image_path), "ground_sha256": digest(ground_path),
        "receiver_protocol": "frame-receiver-v1", "ground_model_id": model_id,
        "receiver_plane_world": np.asarray(plane).tolist(),
        "receiver_basis_world": np.asarray(basis).tolist(),
        "receiver_anchor_ego_to_world": np.asarray(anchor).tolist()}
    write_json(folder/"observation.json", observation)
    return observation


def frame_index(pairs):
    """Resolve legacy duplicates deterministically; never union incompatible masks.

    New exports have identical references. Legacy input prefers the frame's own
    pair-start geometry and explicitly reports that it is not a rebuilt cache.
    """
    choices = {}
    for pair in sorted(pairs, key=lambda value: value["id"]):
        for slot, frame in enumerate(pair["frames"]):
            key = (pair["scene_name"], pair["camera"], frame["sample_data_token"])
            item = {"frame": frame, "pair": pair, "slot": slot,
                    "sample_token": pair["sample_token"] if slot == 0 else pair["next_sample_token"]}
            old = choices.get(key)
            if old is not None:
                a = old["frame"]
                if (a["camera_timestamp"] != frame["camera_timestamp"]
                        or a["image_sha256"] != frame["image_sha256"]
                        or a["image_size_wh"] != frame["image_size_wh"]
                        or not np.allclose(a["projection_world_to_image"], frame["projection_world_to_image"], atol=1e-8)):
                    raise ValueError(f"Conflicting camera identity: {key}")
                if a.get("receiver_protocol") and frame.get("receiver_protocol"):
                    if (a["ground_model_id"] != frame["ground_model_id"]
                            or a["ground_sha256"] != frame["ground_sha256"]):
                        raise ValueError(f"Frame receiver cache is inconsistent: {key}")
            if old is None or slot < old["slot"]:
                choices[key] = item
    return choices
