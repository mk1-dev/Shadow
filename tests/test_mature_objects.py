import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from shadow_occ.annotation.cached_bev import write_cached_bev
from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.mature_objects import (
    attach_mature_objects, camera_time_objects, load_cached_objects, object_frame_source,
    objects_from_uniad, projected_object_motion, render_objects, write_cached_objects,
)
from shadow_occ.annotation.physics_motion import plane_basis
from shadow_occ.annotation.receiver_explanations import explain_component
from shadow_occ.utils.io import write_json
from tests.test_bev_flow import example
from tests.test_receiver_explanations import receiver_case


class Boxes:
    def __init__(self, centers, velocity=(2., 0.)):
        centers = np.asarray(centers, np.float64).reshape(-1, 3)
        # Match LiDARInstance3DBoxes.corners, including its bottom-z tensor origin.
        signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
                          [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]])
        self.corners = centers[:, None]+signs
        self.gravity_center = centers
        self.tensor = np.zeros((len(centers), 9))
        self.tensor[:, :3] = centers-[0, 0, 1]
        self.tensor[:, 3:6] = 2
        self.tensor[:, 7:9] = velocity


def record(token="s0"):
    return SimpleNamespace(token=token, scene_name="scene1", lidar_ego_to_global=np.eye(4),
                           lidar_sensor_to_ego=np.eye(4))


def result(centers=((0, 0, 1),), ids=None):
    count = len(centers)
    return {"boxes_3d": Boxes(centers), "scores_3d": np.full(count, .9),
            "labels_3d": np.zeros(count, np.int64),
            "track_ids": np.arange(count) if ids is None else np.asarray(ids),
            "gt_bboxes_3d": "not used", "matched_idx": "not used"}


def scene():
    k = np.array([[40, 0, 48], [0, 40, 48], [0, 0, 1.]])
    rotation = np.diag([1., -1., -1.])
    projection = k @ np.c_[rotation, -rotation @ [0, 0, 10]]
    frames, bundles = [], []
    for index in range(3):
        frames.append({"camera_timestamp": index*500000, "image_size_wh": [96, 96],
                       "projection_world_to_image": projection, "sample_data_token": f"c{index}"})
        bundle = objects_from_uniad(result(((index, 0, 1),)), record(f"s{index}"), index*500000, ["car"])
        bundle["model_source"] = object_frame_source(record(f"s{index}"), index*500000,
            [-20, -20, -5, 20, 20, 5], {"checkpoint_sha256": "checkpoint", "model_config_sha256": "config"})
        bundles.append(bundle)
    flow = [{"flow_t_to_next": np.full((96, 96, 2), [5., 0], np.float32),
             "ground_flow": np.zeros((96, 96, 2), np.float32),
             "flow_reliable": np.ones((96, 96), bool)} for _ in range(2)]
    return frames, bundles, flow


def script(name):
    path = Path(__file__).parents[1]/"scripts"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(name+"_mature_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decoded_predictions_preserve_model_center_corner_order_and_velocity():
    item = record()
    item.lidar_sensor_to_ego = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 2], [0, 0, 0, 1.]])
    bundle = objects_from_uniad(result(), item, 0, ["car"])
    prediction = bundle["objects"][0]
    np.testing.assert_allclose(prediction["center_world"], [4, 5, 3])
    np.testing.assert_allclose(prediction["velocity_world_mps"], [0, 2, 0])
    assert prediction["track_id"] == 0 and prediction["track_token"] == "scene1:pred-track:0"
    assert bundle["corner_order"] == "mmdet3d_lidar" and not bundle["gt_used"]


def test_detection_outputs_do_not_invent_temporal_track_ids():
    values = result()
    values.update(boxes_3d_det=Boxes(((5, 0, 1),)), scores_3d_det=np.array([.8]), labels_3d_det=np.array([0]))
    bundle = objects_from_uniad(values, record(), 0, ["car"])
    assert len(bundle["objects"]) == 2
    assert bundle["objects"][1]["track_token"] is None
    assert bundle["objects"][1]["prediction_kind"] == "detection"


def test_empty_predictions_are_available_but_not_free_space():
    bundle = objects_from_uniad(result(()), record(), 0, ["car"])
    assert bundle["objects"] == [] and bundle["branches"] == ["tracking"]
    assert bundle["empty_prediction_is_free_space"] is False
    with pytest.raises(ValueError, match="decoded"):
        objects_from_uniad({"bev_embed": np.ones(4)}, record(), 0, ["car"])


def test_cached_feature_and_object_payloads_are_bound_together(tmp_path):
    frames, bundles, _ = scene()
    source = {"checkpoint_sha256": "checkpoint", "model_config_sha256": "config"}
    write_cached_bev(tmp_path, "s0", np.ones((2, 4, 4)), record(), 0,
                     [-20, -20, -5, 20, 20, 5], source, objects=bundles[0])
    metadata = json.loads((tmp_path/"s0.json").read_text())
    bundle, evidence = load_cached_objects(tmp_path, "s0", metadata)
    assert evidence["objects"] == 1 and bundle["sample_token"] == "s0"
    with pytest.raises(ValueError, match="model mismatch"):
        load_cached_objects(tmp_path, "s0", {**metadata, "checkpoint_sha256": "another-model"})
    with pytest.raises(FileNotFoundError, match="re-export"):
        load_cached_objects(tmp_path, "missing")


def test_camera_time_compensation_uses_predicted_velocity_not_gt():
    frames, bundles, _ = scene()
    frame = {**frames[0], "camera_timestamp": 100000}
    values = camera_time_objects(bundles[0], frame)
    np.testing.assert_allclose(values[0]["center"], [.2, 0, 1])
    bundles[0]["objects"][0]["velocity_world_mps"] = None
    assert not camera_time_objects(bundles[0], frame)[0]["time_aligned"]
    with pytest.raises(ValueError, match="timestamps"):
        camera_time_objects(bundles[0], {**frame, "camera_timestamp": 500000})


def test_face_motion_uses_surface_depth_not_box_center_or_ground():
    frames, bundles, evidence = scene()
    rendered, motion = projected_object_motion(frames, bundles, evidence)
    valid = np.isfinite(motion[0]["object_residual"])
    assert valid.any() and rendered[0]["confidence"].max() == pytest.approx(.9)
    np.testing.assert_allclose(motion[0]["expected"][valid], np.tile([5., 0.], (valid.sum(), 1)), atol=1e-4)
    assert np.median(motion[0]["object_residual"][valid]) < 1e-4
    assert np.median(motion[0]["ground_residual"][valid]) == pytest.approx(5)
    assert np.median(motion[0]["entity_fit"][valid]) > .99


def test_unmatched_tracks_and_unreliable_flow_remain_unknown():
    frames, bundles, evidence = scene()
    bundles[1]["objects"][0]["track_token"] = "another-track"
    _, motion = projected_object_motion(frames, bundles, evidence)
    assert np.isnan(motion[0]["expected"]).all()
    frames, bundles, evidence = scene()
    evidence[0]["flow_reliable"][:] = False
    _, motion = projected_object_motion(frames, bundles, evidence)
    assert np.isnan(motion[0]["expected"]).all()


def test_current_only_ids_cannot_be_mistaken_for_persistent_tracks():
    frames, bundles, evidence = scene()
    for bundle in bundles:
        bundle["model_source"]["history_mode"] = "current-only"
    rendered, motion = projected_object_motion(frames, bundles, evidence)
    assert rendered[0]["confidence"].any()
    assert np.isnan(motion[0]["expected"]).all()


def test_depth_ordering_does_not_choose_the_highest_confidence_box():
    frames, bundles, _ = scene()
    second = objects_from_uniad(result(((0, 0, 3),), ids=[2]), record(), 0, ["car"])["objects"][0]
    second["score"] = .4
    bundles[0]["objects"].append(second)
    values = render_objects(camera_time_objects(bundles[0], frames[0]), frames[0])
    assert values["confidence"][48, 48] == pytest.approx(.4)
    assert values["depth"][48, 48] == pytest.approx(6)


def test_real_3d_entity_motion_changes_shadow_competition():
    arrays, mask, hypotheses, polygons = receiver_case()
    arrays["mature_entity_probability"] = np.full((3, *mask.shape), .95, np.float32)
    arrays["mature_entity_motion_fit"] = np.ones((2, *mask.shape), np.float32)
    arrays["mature_inference_domain"] = np.ones((3, *mask.shape), bool)
    result_value = explain_component(mask, arrays, hypotheses, polygons, .5)
    assert result_value["winner"] == "physical_entity" and not result_value["shadow_supported"]
    assert result_value["mature_3d"]["entity_motion_score"] > .9
    assert result_value["without_mature_3d"]["winner"] == "cast_shadow"
    assert result_value["without_mature_3d"]["shadow_supported"]
    arrays["mature_entity_probability"][:] = 0
    arrays["mature_entity_motion_fit"][:] = np.nan
    assert explain_component(mask, arrays, [], {}, .5)["shadow_supported"] is False


def test_prediction_cache_attaches_to_receiver_bev_and_retains_provenance(tmp_path):
    frames, bundles, evidence = scene()
    for index, bundle in enumerate(bundles):
        write_cached_objects(tmp_path, f"s{index}", bundle, bundle["model_source"])
    yy, xx = np.mgrid[-2:2:8j, -2:2:8j]
    arrays = {"receiver_world_xyz": np.stack([xx, yy, np.zeros_like(xx)], axis=-1),
              "receiver_valid": np.ones(xx.shape, bool), "shadow_candidate": np.ones(xx.shape, bool),
              "learned_bev_temporal_cosine": np.full((2, *xx.shape), .8, np.float32),
              "shadow_flow_supported": np.ones(xx.shape, bool)}
    feature_evidence = arrays["learned_bev_temporal_cosine"].copy()
    flow_evidence = arrays["shadow_flow_supported"].copy()
    metadata = {"sample_tokens": ["s0", "s1", "s2"], "scene_name": "scene1", "counts": {}}
    arrays, metadata, _, _ = attach_mature_objects(arrays, metadata, frames, evidence, tmp_path)
    assert arrays["mature_entity_probability"].shape == (3, 8, 8)
    assert arrays["mature_expected_object_flow_px"].shape == (2, 8, 8, 2)
    assert not metadata["mature_objects"]["gt_used"]
    assert len(metadata["mature_objects"]["sources"]) == 3
    np.testing.assert_array_equal(arrays["learned_bev_temporal_cosine"], feature_evidence)
    np.testing.assert_array_equal(arrays["shadow_flow_supported"], flow_evidence)


def test_active_analysis_requires_actual_decoded_cache_and_writes_image_motion(tmp_path, monkeypatch):
    pairs, images, grounds, flows, exposures = example()
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]
    cache = tmp_path/"cache"
    source = {"checkpoint_sha256": "checkpoint", "model_config_sha256": "config"}
    for index, frame in enumerate(frames):
        image, ground = tmp_path/f"i{index}.png", tmp_path/f"g{index}.png"
        checked_write_image(image, images[index])
        checked_write_image(ground, grounds[index].astype(np.uint8)*255)
        frame.update(image=str(image), ground_support=str(ground), image_sha256=digest(image), ground_sha256=digest(ground))
        item = record(f"s{index}")
        bundle = objects_from_uniad(result(()), item, frame["camera_timestamp"], ["car"])
        write_cached_bev(cache, f"s{index}", np.ones((2, 16, 16)), item, frame["camera_timestamp"],
                         [-4, -4, -1, 4, 4, 3], source, objects=bundle)
    write_json(cache/"cache_protocol.json", {**source, "encoder": "synthetic mature model"})
    manifest = tmp_path/"pairs.jsonl"
    write_rows(manifest, pairs)
    flow = tmp_path/"flow"
    flow.mkdir()
    write_json(flow/"analysis_protocol.json", {"flow_evidence_version": "receiver-support-v2", "manifest_sha256": digest(manifest)})
    patch = np.zeros((64, 64), bool)
    patch[26:38, 26:38] = True
    metrics = []
    for index, pair in enumerate(pairs):
        path = flow/f"e{index}.npz"
        np.savez_compressed(path, **flows[index], visible_ground=np.ones_like(patch),
            ground_flow=np.zeros((64, 64, 2), np.float32), flow_reliable=np.ones_like(patch),
            onset_candidate=patch if index == 0 else np.zeros_like(patch),
            recovery_candidate=patch if index == 1 else np.zeros_like(patch), stable_dark_candidate=np.zeros_like(patch))
        metrics.append({**{key: pair[key] for key in ("id", "scene_name", "camera", "official_split")},
                        "evidence": str(path), "evidence_sha256": digest(path), "exposure": exposures[index]})
    write_rows(flow/"pair_metrics.jsonl", metrics)
    output = tmp_path/"bev"
    monkeypatch.setattr(sys, "argv", ["analyze", "--pairs", str(manifest), "--analysis-root", str(flow),
        "--feature-root", str(cache), "--extent-m", "4", "--output-root", str(output)])
    script("analyze_bev_flow_discrepancy").main()
    windows = read_rows(output/"windows.jsonl")
    assert windows and windows[0]["mature_objects"]["protocol"] == "mature-3d-objects-v1"
    assert Path(windows[0]["mature_image_motion"]["path"]).is_file()
    assert (output/"motion_review"/f"{windows[0]['id']}.jpg").is_file()
    with np.load(windows[0]["evidence"], allow_pickle=False) as archive:
        assert {"learned_bev_temporal_cosine", "shadow_flow_supported",
                "mature_entity_probability"} <= set(archive.files)
