import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_mask_to_image, bev_projection, ground_grid
from shadow_occ.annotation.bev_teacher import PROTOCOL, endpoint_negative_masks, merge_trimaps
from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.physics_labels import footprint_mask, light_bank
from shadow_occ.annotation.physics_motion import plane_basis
from shadow_occ.utils.io import write_json


def fixture():
    grid = BEVFlowConfig(extent_m=8).grid()
    basis = plane_basis([0, 0, 1, 0])
    world = ground_grid(basis, np.eye(4), grid)
    box = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0, 2)])
    points = np.array([np.vstack([[0, y, 1], box+[0, y, 0]]) for y in (-3, 0, 3)])
    projection = bev_projection(np.eye(4), grid)
    mask = footprint_mask(points[1, 1:], [0, 0, 1, 0], light_bank()[0], projection, grid.shape)
    mask &= world[..., 0] > 1.5
    arrays = {"bev_world_to_grid": projection, "receiver_world_xyz": world, "ground_basis_world": basis,
        "receiver_valid": np.ones(grid.shape, bool), "shadow_boundary_onset": mask.copy(),
        "shadow_boundary_recovery": mask.copy(), "learned_bev_receiver_support": np.ones(grid.shape, bool),
        "shadow_flow_supported": np.ones(grid.shape, bool), "appearance_local_ratio": np.ones((3, *grid.shape))}
    return grid, arrays, mask, [{"instance_token": "car", "points": points}]


def test_endpoint_negatives_need_observed_light_and_no_competing_shadow():
    _, arrays, mask, tracks = fixture()
    masks, _ = endpoint_negative_masks(arrays, mask, tracks, [0, 0, 1, 0], light_bank(), [{"light_id": 0}])
    assert masks[0].any() and masks[2].any()
    arrays["appearance_local_ratio"][0] = .6
    masks, _ = endpoint_negative_masks(arrays, mask, tracks, [0, 0, 1, 0], light_bank(), [{"light_id": 0}])
    assert not masks[0].any() and masks[2].any()
    arrays["appearance_local_ratio"][0] = 1
    stationary = {"instance_token": "other", "points": np.repeat(tracks[0]["points"][1:2], 3, axis=0)}
    masks, _ = endpoint_negative_masks(arrays, mask, tracks+[stationary], [0, 0, 1, 0], light_bank(), [{"light_id": 0}])
    assert not masks[0].any() and not masks[2].any()


def test_static_bright_receiver_alone_is_not_a_negative():
    _, arrays, mask, tracks = fixture()
    arrays["shadow_boundary_onset"][:] = False
    arrays["shadow_boundary_recovery"][:] = False
    masks, _ = endpoint_negative_masks(arrays, mask, tracks, [0, 0, 1, 0], light_bank(), [{"light_id": 0}])
    assert not masks[0].any() and not masks[2].any()


def test_merge_contradictory_frame_observations_as_ignore():
    a, b = np.full((3, 3), 255, np.uint8), np.full((3, 3), 255, np.uint8)
    a[0, 0], b[0, 0], a[1, 1], b[2, 2] = 1, 0, 1, 0
    target, count = merge_trimaps([a, b])
    assert target[0, 0] == 255 and count == 1
    assert target[1, 1] == 1 and target[2, 2] == 0


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_teacher_builds_native_image_labels_and_shared_training_data(tmp_path, monkeypatch):
    grid, arrays, core, tracks = fixture()
    k = np.array([[30, 0, 48], [0, 30, 32], [0, 0, 1.]])
    r = np.diag([1., -1., -1.])
    projection = k @ np.c_[r, -r @ [0, 0, 10]]
    frames = []
    for i in range(4):
        image = np.full((64, 96, 3), 180, np.uint8)
        native_shadow = footprint_mask(tracks[0]["points"][min(i, 2), 1:], [0, 0, 1, 0], light_bank()[0], projection, image.shape[:2])
        image[native_shadow] = 45
        ip, gp = tmp_path/f"i{i}.png", tmp_path/f"g{i}.png"
        checked_write_image(ip, image)
        checked_write_image(gp, np.full(image.shape[:2], 255, np.uint8))
        frames.append({"image": str(ip), "image_sha256": digest(ip), "ground_support": str(gp),
            "sample_data_token": f"cam{i}", "image_size_wh": [96, 64],
            "camera_timestamp": 500000*(i+1), "projection_world_to_image": projection.tolist()})
    pairs = []
    for i in range(3):
        points = tracks[0]["points"]
        pairs.append({"id": f"p{i}", "sample_token": f"s{i}", "next_sample_token": f"s{i+1}",
            "scene_name": "scene-0001", "camera": "CAM_FRONT",
            "official_split": "train", "dt_seconds": .5, "protocol": "physics-motion-evidence-v1", "train_approved": False,
            "frames": frames[i:i+2], "plane_world": [0, 0, 1, 0], "plane_basis_world": arrays["ground_basis_world"].tolist(),
            "objects": [{"instance_token": "car", "category": "vehicle.car", "annotation_token_t": f"a{i}",
                "annotation_token_next": f"a{i+1}", "landmarks_world_t": points[min(i, 2)].tolist(),
                "landmarks_world_next": points[min(i+1, 2)].tolist()}]})
    pairs_path = tmp_path/"pairs.jsonl"
    write_rows(pairs_path, pairs)
    evidence_path = tmp_path/"bev.npz"
    np.savez_compressed(evidence_path, **arrays)
    labels = tmp_path/"labels"
    labels.mkdir()
    mask = bev_mask_to_image(core, projection, arrays["ground_basis_world"], grid, (64, 96))
    target = np.full(mask.shape, 255, np.uint8)
    target[mask] = 1
    tp, mp = labels/"target.png", labels/"region.png"
    checked_write_image(tp, target)
    checked_write_image(mp, mask.astype(np.uint8)*255)
    evidence = {"evidence": str(evidence_path), "evidence_sha256": digest(evidence_path),
        "grid": {key: getattr(grid, key) for key in ("x_min", "x_max", "y_min", "y_max", "resolution")},
        "selection": {"minimum_cosine": .3}}
    row = {"id": "positive", "image": frames[1]["image"], "ground_support": frames[1]["ground_support"],
        "timestamp": frames[1]["camera_timestamp"], "scene_name": "scene-0001", "camera": "CAM_FRONT",
        "official_split": "train", "sample_token": "s1", "label_source": "temporal_weak",
        "physics_export_enabled": True, "physics_pair_ids": ["p0", "p1"], "bev_flow": evidence,
        "shadow_mask": str(tp), "regions": [{"id": "region", "mask": str(mp), "shadow_label": "YES",
                                               "label_source": "temporal_weak"}]}
    write_rows(labels/"canonical.jsonl", [row])
    write_rows(labels/"decisions.jsonl", [{"id": "positive", "shadow_decision": {"shadow_accepted": True,
        "plausible_hypotheses": [{"light_id": 0}]}}])
    output = tmp_path/"teacher"
    monkeypatch.setattr(sys, "argv", ["teacher", "--labels-root", str(labels), "--pairs", str(pairs_path),
        "--output-root", str(output), "--minimum-pixels", "1"])
    script("build_bev_flow_annotation_manifest").main()
    rows = read_rows(output/"canonical.jsonl")
    assert len(rows) == 3
    assert {row["image"] for row in rows} == {frame["image"] for frame in frames[:3]}
    assert all(row["physics_protocol"] == PROTOCOL for row in rows)
    assert any(region["shadow_label"] == "NO" for row in rows for region in row["regions"])
    data = tmp_path/"data"
    monkeypatch.setattr(sys, "argv", ["dataset", "--manifest", str(output/"canonical.jsonl"), "--output-root", str(data)])
    script("build_shadow_annotation_dataset").main()
    vlm = [row for split in ("train", "dev", "eval") for row in read_rows(data/f"vlm_{split}.jsonl")]
    assert {row["answer"] for row in vlm} == {"YES", "NO"}
    assert all(len(row["images"]) == 5 for row in vlm)
    joint = tmp_path/"joint"
    monkeypatch.setattr(sys, "argv", ["joint", "--region-data-root", str(data), "--output-root", str(joint)])
    script("build_joint_shadow_annotation_dataset").main()
    annotations = [row for split in ("train", "dev", "eval") for row in read_rows(joint/f"annotation_{split}.jsonl")]
    assert {json.loads(row["answer"])["shadow"] for row in annotations} == {"YES", "NO"}


def test_default_preparation_has_no_legacy_shadow_label_dependency():
    root = Path(__file__).parents[1]
    preparation = (root/"slurm/prepare_shadow_annotation.slurm").read_text()
    for required in ("analyze_shadow_motion_pairs.py", "export_mature_bev.py", "analyze_bev_flow_discrepancy.py",
                     "build_bev_flow_annotation_manifest.py", "build_joint_shadow_annotation_dataset.py"):
        assert required in preparation
    assert "export_caster_shadow_weak_data.py" not in preparation
    assert "SHADOW_LABEL_ROOT" not in preparation and "outputs_temporal" not in preparation


def test_mature_bev_export_unwraps_last_frame_transforms():
    module = script("export_mature_bev")
    marker = object()
    assert module._pick_last_tensor_like([[[marker]]]) is marker
