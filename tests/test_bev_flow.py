import cv2
import numpy as np
import pytest

from shadow_occ.annotation.bev_flow import (
    BEVFlowConfig, bev_mask_to_image, bev_projection, build_bev_flow, ground_grid,
)
from shadow_occ.annotation.physics_labels import LabelConfig, candidate_evidence, rank_hypotheses
from shadow_occ.annotation.physics_motion import plane_basis, project_world


def example():
    k = np.array([[40, 0, 32], [0, 40, 32], [0, 0, 1]])
    rotation = np.diag([1., -1., -1.])
    projection = k @ np.c_[rotation, -rotation @ [0, 0, 10]]
    frames = [{"sample_data_token": f"cam{i}", "camera_timestamp": (i+1)*500000,
        "image_sha256": f"sha{i}", "image_size_wh": [64, 64], "object_volumes_world": [],
        "projection_world_to_image": projection.tolist()} for i in range(3)]
    pairs = [{"id": f"p{i}", "scene_name": "scene1", "official_split": "train", "camera": "CAM_FRONT",
        "sample_token": f"s{i}", "next_sample_token": f"s{i+1}", "frames": frames[i:i+2],
        "dt_seconds": .5, "plane_world": [0, 0, 1, 0],
        "plane_basis_world": plane_basis([0, 0, 1, 0]).tolist(),
        "anchor_ego_to_world": np.eye(4).tolist(), "objects": [],
        "protocol": "physics-motion-evidence-v1", "train_approved": False} for i in range(2)]
    images = [np.full((64, 64, 3), 180, np.uint8) for _ in range(3)]
    images[1][26:38, 26:38] = 50
    ground = [np.ones((64, 64), bool) for _ in range(3)]
    flow = [{"flow_t_to_next": np.zeros((64, 64, 2), np.float32),
             "receiver_flow_supported": np.ones((64, 64), bool)} for _ in range(2)]
    exposures = [{"valid": True, "scale": 1., "offset_L": 0.} for _ in range(2)]
    return pairs, images, ground, flow, exposures


def build(data):
    pairs, images, ground, flow, exposures = data
    return build_bev_flow(*pairs, images, ground, *flow, exposures, BEVFlowConfig(extent_m=4))


def test_fixed_receiver_light_changes_do_not_require_nonzero_flow():
    arrays, metadata = build(example())
    candidate = arrays["shadow_candidate"]
    assert candidate.any()
    assert (arrays["explanation_state"][candidate] == 3).all()
    assert arrays["shadow_same_patch_cycle"][candidate].all()
    np.testing.assert_allclose(arrays["flow_residual_px"], 0)
    assert not metadata["is_shadow_gt"] and not metadata["train_approved"]


def test_static_darkness_is_retained_but_not_accepted_as_motion_shadow():
    data = example()
    data[1][0][:] = data[1][1]
    data[1][2][:] = data[1][1]
    arrays, _ = build(data)
    mask = arrays["shadow_candidate"]
    assert mask.any() and not arrays["illumination_event"].any()
    maps = {"onset": arrays["shadow_boundary_onset"], "recovery": arrays["shadow_boundary_recovery"],
            "flow_supported": arrays["shadow_flow_supported"], "same_patch_cycle": arrays["shadow_same_patch_cycle"]}
    report = candidate_evidence(mask, maps, LabelConfig(minimum_area=1))
    assert any("static_dark" in reason for reason in report["reasons"])


def test_exposure_failure_keeps_persistent_appearance_without_false_transitions():
    data = example()
    data[1][0][:] = data[1][1]
    data[1][2][:] = data[1][1]
    data[4][0].update(valid=False, scale=float("nan"), offset_L=float("nan"))
    arrays, metadata = build(data)
    assert arrays["shadow_candidate"].any() and arrays["shadow_persistent_dark"].any()
    assert not metadata["exposure_comparable"]
    assert not arrays["shadow_boundary_onset"].any() and not arrays["shadow_boundary_recovery"].any()
    assert not arrays["temporal_exposure_valid"].any()


def test_expected_flow_uses_camera_projection_not_bev_pixel_displacement():
    data = example()
    pairs, images, _, flow, _ = data
    for i, frame in enumerate([pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]):
        p = np.asarray(frame["projection_world_to_image"])
        p[0, 3] += i*40  # Constant ground depth 10 -> four camera pixels per interval.
        frame["projection_world_to_image"] = p.tolist()
        images[i][:] = 180
    for value in flow:
        value["flow_t_to_next"][..., 0] = 4
    arrays, _ = build(data)
    np.testing.assert_allclose(arrays["expected_geometric_flow_px"][..., 0], 4)
    np.testing.assert_allclose(arrays["flow_residual_px"], 0)
    assert not arrays["illumination_event"].any()


def test_exposure_transforms_compose_into_first_frame_reference():
    data = example()
    levels = []
    for image, gray in zip(data[1], (120, 140, 160)):
        image[:] = gray
        levels.append(float(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[0, 0, 0])*100/255)
    for i in range(2):
        data[4][i]["offset_L"] = levels[i]-levels[i+1]
    arrays, _ = build(data)
    np.testing.assert_allclose(arrays["temporal_illumination_change_L"], 0, atol=1e-4)
    assert not arrays["shadow_candidate"].any()


def test_unobserved_or_box_supported_receivers_cannot_become_candidates():
    data = example()
    data[2][1][:] = False
    arrays, _ = build(data)
    assert not arrays["receiver_valid"].any()
    assert not arrays["shadow_candidate"].any()
    assert np.isnan(arrays["temporal_illumination_change_L"]).all()
    data = example()
    corners = np.array([[x, y, z] for x in (-2, 2) for y in (-2, 2) for z in (0, 2)])
    for pair in data[0]:
        for frame in pair["frames"]:
            frame["object_volumes_world"] = [{"corners_world": corners.tolist()}]
    arrays, _ = build(data)
    excluded = arrays["physical_box_footprint"].any(axis=0)
    assert excluded.any() and not arrays["shadow_candidate"][excluded].any()
    assert (arrays["explanation_state"][excluded] == 2).all()


def test_flow_error_alone_does_not_generate_shadow_candidates():
    data = example()
    for image in data[1]:
        image[:] = 180
    data[3][0]["flow_t_to_next"][:] = 20
    arrays, _ = build(data)
    assert np.nanmax(arrays["flow_residual_px"]) > 20
    assert not arrays["shadow_candidate"].any()


def test_rotated_translated_anchor_and_image_pullback_share_cell_centres():
    grid = BEVFlowConfig(extent_m=4).grid()
    anchor = np.array([[0, -1, 0, 15], [1, 0, 0, -8], [0, 0, 1, 0], [0, 0, 0, 1.]])
    basis = anchor @ plane_basis([0, 0, 1, 0])
    world = ground_grid(basis, anchor, grid)
    p = bev_projection(anchor, grid)
    pixels, _, valid = project_world(world.reshape(-1, 3), p)
    yy, xx = np.indices(grid.shape)
    np.testing.assert_allclose(pixels, np.c_[xx.ravel(), yy.ravel()], atol=1e-6)
    assert valid.all()
    mask = (xx > 4) & (yy < 10)
    np.testing.assert_array_equal(bev_mask_to_image(mask, p, basis, grid, grid.shape), mask)


def test_bev_matching_uses_metric_receivers_and_same_light_three_frame_footprints():
    from shadow_occ.annotation.physics_labels import footprint_mask
    grid = BEVFlowConfig(extent_m=8).grid()
    p = bev_projection(np.eye(4), grid)
    basis = plane_basis([0, 0, 1, 0])
    box = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0, 2)])
    points = np.array([np.vstack([[x, 0, 1], box+[x, 0, 0]]) for x in (-3, 0, 3)])
    ray = np.array([[1, 0, -1.]])/np.sqrt(2)
    footprints = [footprint_mask(state[1:], [0, 0, 1, 0], ray[0], p, grid.shape) for state in points]
    mask = footprints[1]
    maps = {"onset": mask & ~footprints[0], "recovery": mask & ~footprints[2],
            "flow_supported": np.ones_like(mask), "same_patch_cycle": mask & ~footprints[0] & ~footprints[2]}
    ranks, report = rank_hypotheses(mask, maps, [{"instance_token": "car", "points": points}],
        [0, 0, 1, 0], basis, p, ray, LabelConfig(minimum_area=1, minimum_change_pixels=1))
    assert not report["reasons"]
    assert ranks[0]["passes"] and ranks[0]["score"] == 1


def test_old_pairs_require_reexport_for_physical_bev():
    data = example()
    del data[0][0]["anchor_ego_to_world"]
    with pytest.raises(ValueError, match="anchor"):
        build(data)
    data = example()
    del data[0][0]["frames"][0]["object_volumes_world"]
    with pytest.raises(ValueError, match="volumes"):
        build(data)


def test_bev_pipeline_matches_in_bev_and_exports_camera_regions(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import sys
    from types import SimpleNamespace

    from shadow_occ.annotation.cached_bev import write_cached_bev
    from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
    from shadow_occ.utils.io import write_json

    def script(name):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/"scripts"/f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    pairs, images, grounds, flows, exposures = example()
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]
    for i, frame in enumerate(frames):
        ip, gp = tmp_path/f"i{i}.png", tmp_path/f"g{i}.png"
        checked_write_image(ip, images[i])
        checked_write_image(gp, grounds[i].astype(np.uint8)*255)
        frame.update(image=str(ip), ground_support=str(gp), image_sha256=digest(ip), ground_sha256=digest(gp))
    manifest = tmp_path/"pairs.jsonl"
    write_rows(manifest, pairs)
    analysis = tmp_path/"flow"
    analysis.mkdir()
    write_json(analysis/"analysis_protocol.json", {"flow_evidence_version": "receiver-support-v2",
                                                "manifest_sha256": digest(manifest)})
    patch = np.zeros((64, 64), bool)
    patch[26:38, 26:38] = True
    zero = np.zeros_like(patch)
    metrics = []
    for i, pair in enumerate(pairs):
        path = analysis/f"e{i}.npz"
        np.savez_compressed(path, **flows[i], visible_ground=np.ones_like(patch),
            onset_candidate=patch if i == 0 else zero, recovery_candidate=patch if i == 1 else zero,
            stable_dark_candidate=zero)
        metrics.append({**{k: pair[k] for k in ("id", "scene_name", "camera", "official_split")},
            "evidence": str(path), "evidence_sha256": digest(path), "exposure": exposures[i]})
    write_rows(analysis/"pair_metrics.jsonl", metrics)
    feature_root = tmp_path/"features"
    record = SimpleNamespace(scene_name="scene1", lidar_ego_to_global=np.eye(4), lidar_sensor_to_ego=np.eye(4))
    for i in range(3):
        write_cached_bev(feature_root, f"s{i}", np.ones((2, 16, 16)), record, i, [-4, -4, -1, 4, 4, 1], {})
    write_json(feature_root/"cache_protocol.json", {"encoder": "synthetic test"})
    bev = tmp_path/"bev"
    monkeypatch.setattr(sys, "argv", ["analyze", "--pairs", str(manifest), "--analysis-root", str(analysis),
        "--output-root", str(bev), "--feature-root", str(feature_root), "--extent-m", "4", "--mature-object-mode", "off"])
    script("analyze_bev_flow_discrepancy").main()
    builder = script("build_physics_shadow_labels")
    calls, original = [], builder.rank_hypotheses

    def captured(mask, maps, tracks, plane, basis, projection, *args, **kwargs):
        calls.append((mask.shape, np.asarray(projection)))
        return original(mask, maps, tracks, plane, basis, projection, *args, **kwargs)

    monkeypatch.setattr(builder, "rank_hypotheses", captured)
    output = tmp_path/"labels"
    monkeypatch.setattr(sys, "argv", ["labels", "--pairs", str(manifest), "--analysis-root", str(analysis),
        "--output-root", str(output), "--bev-flow-root", str(bev)])
    builder.main()
    assert calls and all(shape == (16, 16) for shape, _ in calls)
    np.testing.assert_allclose(calls[0][1], bev_projection(np.eye(4), BEVFlowConfig(extent_m=4).grid()))
    rows = read_rows(output/"canonical.jsonl")
    assert rows and rows[0]["bev_flow"] is not None
    assert rows[0]["regions"][0]["proposal_source"] == "bev_flow_receiver_component"
    target = cv2.imread(rows[0]["shadow_mask"], cv2.IMREAD_GRAYSCALE)
    assert target.shape == (64, 64)
    with np.load(rows[0]["bev_flow"]["component"], allow_pickle=False) as archive:
        assert archive["receiver_feature_mean"].shape == (3, 2)
    assert json.loads((output/"summary.json").read_text())["candidate_space"] == "physical_bev"
