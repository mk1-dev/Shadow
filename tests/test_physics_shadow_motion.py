import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.physics_motion import (
    EvidenceConfig, decompose_object_motion, exposure_fit, fit_ground_plane,
    forward_backward_evidence, ground_correspondence, ground_homography,
    interpolate_box_state, pair_evidence, plane_basis, project_world,
)


def projection(camera_x=0.0, camera_height=10.0):
    intrinsic = np.array([[30, 0, 48], [0, 30, 32], [0, 0, 1]])
    rotation = np.diag([1.0, -1.0, -1.0])
    return intrinsic @ np.c_[rotation, -rotation @ np.array([camera_x, 0, camera_height])]


def image_pair():
    first = np.full((64, 96, 3), 160, np.uint8)
    second = first.copy()
    first[20:40, 20:36] = 55
    second[20:40, 28:44] = 55
    return first, second


def evidence(first, second, ground=None, forward=None, backward=None):
    ground = np.ones(first.shape[:2], bool) if ground is None else ground
    flow = np.zeros((*first.shape[:2], 2), np.float32)
    return pair_evidence(first, second, ground, ground, projection(), projection(),
                         plane_basis([0, 0, 1, 0]),
                         flow if forward is None else forward,
                         flow if backward is None else backward)


def test_object_motion_uses_same_depth_counterfactual():
    points = np.array([[0.0, 0, 2], [0.5, 0, 3]])
    result = decompose_object_motion(points, points+[1, 0, 0], projection(), projection(0.5), 0.5)
    assert result["valid"].all()
    np.testing.assert_allclose(result["total_displacement_px"],
                               result["camera_displacement_px"]+result["object_residual_displacement_px"])
    np.testing.assert_allclose(result["object_residual_displacement_px"][:, 0], [30/8, 30/7])
    assert not np.isclose(result["object_residual_displacement_px"][0, 0], 30/10)


def test_stationary_object_has_zero_physical_motion_with_moving_camera():
    points = np.array([[0.0, 0, 2]])
    result = decompose_object_motion(points, points, projection(), projection(1), 0.5)
    np.testing.assert_allclose(result["object_residual_displacement_px"], 0)
    assert np.linalg.norm(result["total_displacement_px"]) > 0


def test_invalid_object_depth_not_used_as_landmark_motion():
    result = decompose_object_motion([[0, 0, 11]], [[1, 0, 11]], projection(), projection(), 0.5)
    assert not result["valid"].any()
    assert np.isnan(result["total_displacement_px"]).all()
    with pytest.raises(ValueError, match="positive"):
        decompose_object_motion([[0, 0, 0]], [[0, 0, 0]], projection(), projection(), 0)


def test_box_interpolation_rejects_extrapolation_and_identity_change():
    pytest.importorskip("pyquaternion")
    a = {"instance_token": "car", "timestamp": 0, "translation": [0, 0, 1],
         "size": [2, 4, 2], "rotation": [1, 0, 0, 0]}
    b = {**a, "timestamp": 500000, "translation": [2, 0, 1]}
    state = interpolate_box_state(a, b, 250000)
    np.testing.assert_allclose(state["translation"], [1, 0, 1])
    assert state["corners"].shape == (8, 3)
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_box_state(a, b, 600000)
    with pytest.raises(ValueError, match="identities"):
        interpolate_box_state(a, {**b, "instance_token": "other"}, 250000)


def test_ground_correspondence_matches_direct_plane_projection():
    basis = plane_basis([0.05, 0, 1, 1.5])
    uv, flow, valid = ground_correspondence(projection(), projection(0.4), basis, (64, 96))
    homography = ground_homography(projection(), projection(0.4), basis)
    y, x = 30, 40
    expected = homography @ [x, y, 1]
    assert valid[y, x]
    np.testing.assert_allclose(uv[y, x], expected[:2]/expected[2], atol=1e-5)
    np.testing.assert_allclose(flow[y, x], uv[y, x]-[x, y], atol=1e-5)


def test_ground_behind_camera_is_unknown_even_for_identity_homography():
    _, _, valid = ground_correspondence(projection(), projection(), plane_basis([0, 0, 1, -11]), (64, 96))
    assert not valid.any()


def test_singular_ground_projection_rejected():
    bad = np.zeros((3, 4))
    with pytest.raises(ValueError, match="singular"):
        ground_homography(bad, projection(), plane_basis([0, 0, 1, 0]))


@pytest.mark.parametrize("height", [-1.7, 0.0])
def test_ground_plane_does_not_assume_sensor_or_ego_height(height):
    rng = np.random.default_rng(9)
    yy, xx = np.meshgrid(np.arange(-12, 13), np.arange(-12, 13))
    z = height+0.03*xx+rng.normal(0, 0.005, xx.shape)
    points = np.c_[xx.ravel(), yy.ravel(), z.ravel()]
    plane, report = fit_ground_plane(points)
    assert report["inlier_ratio"] > 0.95
    assert not report["semantic_ground_verified"]
    assert abs(plane[3]/plane[2]+height) < 0.02
    assert abs(plane[0]/plane[2]+0.03) < 0.01


def test_ground_fit_refuses_too_few_points():
    with pytest.raises(ValueError, match="insufficient"):
        fit_ground_plane(np.zeros((5, 3)))


def test_flow_fb_check_rejects_out_of_bounds_and_inconsistent_flow():
    forward = np.zeros((16, 24, 2), np.float32)
    backward = forward.copy()
    forward[..., 0] = 3
    backward[..., 0] = -3
    errors, valid = forward_backward_evidence(forward, backward, EvidenceConfig())
    assert valid[:, :-3].all()
    assert not valid[:, -3:].any()
    np.testing.assert_allclose(errors[:, :-3], 0)
    _, wrong = forward_backward_evidence(forward, np.zeros_like(backward), EvidenceConfig())
    assert not wrong.any()


def test_flow_disagreement_alone_does_not_create_shadow_evidence():
    image = np.full((64, 96, 3), 160, np.uint8)
    flow = np.zeros((64, 96, 2), np.float32)
    flow[20:40, 20:40, 0] = 4
    report = evidence(image, image, forward=flow, backward=-flow)
    assert not report["onset_candidate"].any()
    assert not report["recovery_candidate"].any()
    assert not report["stable_dark_candidate"].any()


def test_pair_residual_sign_reports_onset_and_recovery_without_labels():
    first, second = image_pair()
    result = evidence(first, second)
    assert result["details"]["exposure"]["valid"]
    assert result["onset_candidate"][25:35, 38:42].all()
    assert result["recovery_candidate"][25:35, 22:26].all()
    assert result["delta_L"][30, 40] < -6
    assert result["delta_L"][30, 23] > 6
    assert result["details"]["shadow_accuracy_available"] is False


def test_static_dark_region_survives_zero_motion_and_zero_light_difference():
    first, _ = image_pair()
    result = evidence(first, first)
    assert result["stable_dark_candidate"][25:35, 24:32].all()
    assert not result["onset_candidate"].any()
    assert not result["recovery_candidate"].any()


def test_insufficient_ground_disables_all_change_candidates():
    first, second = image_pair()
    result = evidence(first, second, np.zeros(first.shape[:2], bool))
    assert result["details"]["status"] == "insufficient_background"
    for kind in ("onset", "recovery", "stable_dark"):
        assert not result[f"{kind}_candidate"].any()


def test_global_exposure_change_is_compensated_not_called_shadow():
    # Work in Lab to make the intended global exposure perturbation affine in L.
    yy, xx = np.indices((64, 96))
    old = (20+xx/4+yy/8).astype(np.float32)
    new = old*1.1+2
    result = exposure_fit(old, new, np.ones(old.shape, bool), EvidenceConfig())
    assert result["valid"]
    assert abs(result["scale"]-1.1) < 1e-5
    assert abs(result["offset_L"]-2) < 1e-4
    assert result["residual_median_L"] < 1e-4


def test_extreme_exposure_change_refuses_evidence():
    old = np.full((32, 32), 20, np.float32)
    new = old+40
    assert not exposure_fit(old, new, np.ones(old.shape, bool), EvidenceConfig())["valid"]


def test_flow_with_nan_cannot_validate_background():
    first, second = image_pair()
    flow = np.full((64, 96, 2), np.nan, np.float32)
    result = evidence(first, second, forward=flow, backward=flow)
    assert not result["flow_reliable"].any()
    assert not result["details"]["exposure"]["valid"]


def test_dis_backend_runs_on_cpu_without_torch_or_downloads():
    from shadow_occ.annotation.optical_flow import DenseFlow

    engine = DenseFlow()
    image = np.random.default_rng(0).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    flow = engine(image, image)
    assert flow.shape == (64, 96, 2)
    assert np.isfinite(flow).all()
    assert np.max(np.abs(flow)) < 0.1
    with pytest.raises(ValueError, match="explicit local"):
        DenseFlow("raft-small")
    with pytest.raises(ValueError, match="CPU baseline"):
        DenseFlow("dis", device="cuda")


def load_script(name):
    path = Path(__file__).resolve().parents[1]/"scripts"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_resize_projection_uses_pixel_centers():
    from types import SimpleNamespace

    exporter = load_script("export_nuscenes_motion_pairs")
    camera = SimpleNamespace(calibration=SimpleNamespace(
        image_size=(1600, 900), intrinsic=np.eye(3), global_to_sensor=np.eye(4)))
    matrix = exporter.scaled_projection(camera, (800, 448))
    uv, _, valid = project_world([[10, 20, 1]], matrix)
    assert valid.all()
    np.testing.assert_allclose(uv[0], [(10+0.5)*0.5-0.5, (20+0.5)*448/900-0.5])


def test_stage2_pair_validation_rejects_wrong_camera_time_and_training_data():
    analyzer = load_script("analyze_shadow_motion_pairs")
    row = {"protocol": "physics-motion-evidence-v1", "train_approved": False,
           "official_split": "train", "dt_seconds": 0.5,
           "frames": [{"camera_timestamp": 0}, {"camera_timestamp": 500000}]}
    analyzer.validate_pair(row)
    with pytest.raises(ValueError, match="timestamps"):
        analyzer.validate_pair({**row, "dt_seconds": 0.4})
    with pytest.raises(ValueError, match="unapproved"):
        analyzer.validate_pair({**row, "train_approved": True})


def test_diagnostic_panel_does_not_mutate_clean_image():
    analyzer = load_script("analyze_shadow_motion_pairs")
    first, second = image_pair()
    original = first.copy()
    panel = analyzer.diagnostic_panel(first, second, {"objects": []}, evidence(first, second))
    assert panel.shape[1] == 960
    assert panel.dtype == np.uint8
    np.testing.assert_array_equal(first, original)


def test_stage2_integration_keeps_candidates_unknown_and_checks_input_hashes(tmp_path, monkeypatch):
    from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
    import json
    import sys

    analyzer = load_script("analyze_shadow_motion_pairs")
    first, second = image_pair()
    frames = []
    for index, image in enumerate((first, second)):
        image_path, ground_path = tmp_path/f"image_{index}.png", tmp_path/f"ground_{index}.png"
        checked_write_image(image_path, image)
        checked_write_image(ground_path, np.full(image.shape[:2], 255, np.uint8))
        frames.append({"image": str(image_path), "ground_support": str(ground_path),
                       "image_sha256": digest(image_path), "ground_sha256": digest(ground_path),
                       "camera_timestamp": index*500000, "projection_world_to_image": projection().tolist()})
    pair = {"id": "pair0", "scene_name": "train_scene", "official_split": "train", "camera": "CAM_FRONT",
            "dt_seconds": 0.5, "frames": frames, "objects": [], "plane_basis_world": plane_basis([0, 0, 1, 0]).tolist(),
            "protocol": "physics-motion-evidence-v1", "train_approved": False}
    manifest = tmp_path/"pairs.jsonl"
    write_rows(manifest, [pair])
    monkeypatch.setattr(analyzer, "DenseFlow", lambda *a, **k: lambda first, second: np.zeros((*first.shape[:2], 2), np.float32))
    monkeypatch.setattr(sys, "argv", ["analysis", "--pairs", str(manifest), "--output-root", str(tmp_path/"run")])
    analyzer.main()
    candidates = read_rows(tmp_path/"run"/"candidates.jsonl")
    assert candidates
    assert all(row["decision"] == "UNCERTAIN" and row["shadow_label"] == "IGNORE"
               and row["train_approved"] is False and row["caster_id"] is None for row in candidates)
    summary = json.loads((tmp_path/"run"/"summary.json").read_text())
    assert summary["shadow_precision"] is None
    assert summary["caster_accuracy"] is None
    assert not list((tmp_path/"run").glob("*train*.jsonl"))
    changed = first.copy()
    changed[0, 0] = 0
    checked_write_image(Path(frames[0]["image"]), changed)
    monkeypatch.setattr(sys, "argv", ["analysis", "--pairs", str(manifest), "--output-root", str(tmp_path/"changed")])
    with pytest.raises(ValueError, match="changed"):
        analyzer.main()
