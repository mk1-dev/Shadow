import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_mask_to_image, bev_projection, ground_grid
from shadow_occ.annotation.bev_teacher import PROTOCOL, lit_receiver_negative_mask
from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.physics_labels import LabelConfig, candidate_evidence, factorized_shadow_decision, footprint_mask, light_bank, rank_hypotheses
from shadow_occ.annotation.physics_motion import plane_basis
from shadow_occ.utils.io import write_json


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_case():
    grid = BEVFlowConfig(extent_m=8).grid()
    basis = plane_basis([0, 0, 1, 0])
    world = ground_grid(basis, np.eye(4), grid)
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0, 2)])
    state = np.vstack([[0, 0, 1], corners])
    tracks = [{"instance_token": "car", "points": np.repeat(state[None], 3, axis=0)}]
    projection = bev_projection(np.eye(4), grid)
    mask = footprint_mask(corners, [0, 0, 1, 0], light_bank()[0], projection, grid.shape)
    mask &= world[..., 0] > 1.5
    full, empty = np.ones(grid.shape, bool), np.zeros(grid.shape, bool)
    raw = np.full((3, *grid.shape), 70., np.float32)
    raw[:, mask] = 30
    ratios = np.ones_like(raw)
    ratios[:, mask] = .6
    arrays = {"bev_world_to_grid": projection, "receiver_world_xyz": world, "ground_basis_world": basis,
        "receiver_valid": full, "shadow_candidate": mask.copy(), "shadow_boundary_onset": empty.copy(),
        "shadow_boundary_recovery": empty.copy(), "shadow_same_patch_cycle": empty.copy(),
        "learned_bev_receiver_support": full.copy(), "shadow_flow_supported": full.copy(),
        "appearance_local_ratio": ratios, "appearance_L_raw": raw}
    return grid, arrays, mask, tracks


def test_static_shadow_reaches_projection_without_motion_or_events():
    _, arrays, mask, tracks = static_case()
    maps = {"onset": arrays["shadow_boundary_onset"], "recovery": arrays["shadow_boundary_recovery"],
        "flow_supported": arrays["shadow_flow_supported"], "same_patch_cycle": arrays["shadow_same_patch_cycle"]}
    config = LabelConfig(minimum_area=1, allow_static_existence=True)
    report = candidate_evidence(mask, maps, config)
    assert not report["reasons"] and len(report["temporal_reasons"]) == 2
    hypotheses, report = rank_hypotheses(mask, maps, tracks, [0, 0, 1, 0], arrays["ground_basis_world"],
        arrays["bev_world_to_grid"], light_bank(), config)
    assert report["static_projection_fit_hypotheses"] > 0
    candidate = {"hypotheses": hypotheses, "scene_name": "scene-0001", "official_split": "train", "frame_tokens": ["f0", "f1", "f2"]}
    result = factorized_shadow_decision(candidate, [], light_bank(), config)
    assert result["shadow_accepted"] and result["association"] is None
    assert result["existence_evidence"] == "static_projection"
    assert result["existence_score"] >= .65
    assert not any(h["shadow_passes"] for h in hypotheses)


def test_static_lit_negatives_do_not_need_onset_recovery():
    grid, arrays, mask, tracks = static_case()
    result, info = lit_receiver_negative_mask(arrays, mask, tracks, [0, 0, 1, 0], light_bank(),
        [{"light_id": 0}], resolution=grid.resolution)
    assert result.any() and not (result & mask).any()
    assert result.sum() <= 2*mask.sum()
    assert info["reason"] == "observed_lit_receiver"
    assert (arrays["appearance_local_ratio"][:, result] >= .97).all()
    count, _ = cv2.connectedComponents(result.astype(np.uint8), connectivity=4)
    assert count == 2


def test_bright_negative_needs_contrast_not_just_an_empty_projection():
    grid, arrays, mask, tracks = static_case()
    arrays["appearance_L_raw"][:] = 30
    result, _ = lit_receiver_negative_mask(arrays, mask, tracks, [0, 0, 1, 0], light_bank(),
        [{"light_id": 0}], resolution=grid.resolution)
    assert not result.any()


def test_static_window_survives_invalid_exposure_through_both_training_targets(tmp_path, monkeypatch):
    grid, arrays, core, tracks = static_case()
    k = np.array([[60, 0, 96], [0, 60, 64], [0, 0, 1.]])
    rotation = np.diag([1., -1., -1.])
    projection = k @ np.c_[rotation, -rotation @ [0, 0, 10]]
    native = bev_mask_to_image(core, projection, arrays["ground_basis_world"], grid, (128, 192))
    frames = []
    for i in range(3):
        image = np.full((128, 192, 3), 180, np.uint8)
        image[native] = 45
        ip, gp = tmp_path/f"i{i}.png", tmp_path/f"g{i}.png"
        checked_write_image(ip, image)
        checked_write_image(gp, np.full(native.shape, 255, np.uint8))
        frames.append({"image": str(ip), "image_sha256": digest(ip), "ground_support": str(gp),
            "ground_sha256": digest(gp), "sample_data_token": f"cam{i}", "image_size_wh": [192, 128],
            "camera_timestamp": (i+1)*500000, "projection_world_to_image": projection.tolist(),
            "object_volumes_world": []})
    pairs = []
    state = tracks[0]["points"][0].tolist()
    for i in range(2):
        pairs.append({"id": f"p{i}", "sample_token": f"s{i}", "next_sample_token": f"s{i+1}",
            "scene_name": "scene-0001", "camera": "CAM_FRONT", "official_split": "train",
            "dt_seconds": .5, "protocol": "physics-motion-evidence-v1", "train_approved": False,
            "frames": frames[i:i+2], "plane_world": [0, 0, 1, 0], "plane_basis_world": arrays["ground_basis_world"].tolist(),
            "objects": [{"instance_token": "car", "category": "vehicle.car", "annotation_token_t": f"a{i}",
                "annotation_token_next": f"a{i+1}", "landmarks_world_t": state, "landmarks_world_next": state}]})
    pairs_path = tmp_path/"pairs.jsonl"
    write_rows(pairs_path, pairs)
    flow = tmp_path/"flow"
    flow.mkdir()
    stats = []
    for pair in pairs:
        path = flow/f"{pair['id']}.npz"
        np.savez_compressed(path, visible_ground=np.ones(native.shape, bool), receiver_flow_supported=np.ones(native.shape, bool),
            onset_candidate=np.zeros(native.shape, bool), recovery_candidate=np.zeros(native.shape, bool), stable_dark_candidate=native)
        stats.append({"id": pair["id"], "scene_name": pair["scene_name"], "camera": pair["camera"], "official_split": "train",
            "evidence": str(path), "evidence_sha256": digest(path), "exposure": {"valid": False}})
    write_rows(flow/"pair_metrics.jsonl", stats)
    write_json(flow/"analysis_protocol.json", {"flow_evidence_version": "receiver-support-v2", "manifest_sha256": digest(pairs_path)})
    bev = tmp_path/"bev"
    bev.mkdir()
    path = bev/"evidence.npz"
    np.savez_compressed(path, **arrays)
    grid_meta = {key: getattr(grid, key) for key in ("x_min", "x_max", "y_min", "y_max", "resolution")}
    write_rows(bev/"windows.jsonl", [{"id": "window", "pair_ids": ["p0", "p1"], "evidence": str(path),
        "evidence_sha256": digest(path), "grid": grid_meta, "learned_bev_selection": {"feature_gate": "aligned"}}])
    write_json(bev/"bev_flow_protocol.json", {"pair_manifest_sha256": digest(pairs_path),
        "analysis_protocol_sha256": digest(flow/"analysis_protocol.json"), "pair_metrics_sha256": digest(flow/"pair_metrics.jsonl")})
    labels = tmp_path/"labels"
    monkeypatch.setattr(sys, "argv", ["physics", "--pairs", str(pairs_path), "--analysis-root", str(flow),
        "--bev-flow-root", str(bev), "--decision-policy", "factorized-v3", "--export-weak-labels",
        "--review-limit", "0", "--output-root", str(labels)])
    script("build_physics_shadow_labels").main()
    summary = json.loads((labels/"summary.json").read_text())
    assert summary["stage_counts"]["accepted_static_projection"] > 0
    assert summary["counts"]["unstable_exposure_windows_retained_for_static"] == 1
    teacher = tmp_path/"teacher"
    monkeypatch.setattr(sys, "argv", ["teacher", "--pairs", str(pairs_path), "--labels-root", str(labels), "--output-root", str(teacher)])
    script("build_bev_flow_annotation_manifest").main()
    canonical = read_rows(teacher/"canonical.jsonl")
    assert {region["shadow_label"] for row in canonical for region in row["regions"]} == {"YES", "NO"}
    assert all(row["physics_protocol"] == PROTOCOL for row in canonical)
    data = tmp_path/"data"
    monkeypatch.setattr(sys, "argv", ["dataset", "--manifest", str(teacher/"canonical.jsonl"), "--output-root", str(data)])
    script("build_shadow_annotation_dataset").main()
    vlm = [row for split in ("train", "dev", "eval") for row in read_rows(data/f"vlm_{split}.jsonl") if row["task"] == "shadow"]
    assert {row["answer"] for row in vlm} == {"YES", "NO"}
    assert all(len(row["images"]) == 5 for row in vlm)
