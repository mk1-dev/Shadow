import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.physics_motion import project_world
from shadow_occ.annotation.receiver_explanations import (
    CompetitionConfig, apply_competition, attach_appearance_prior, box_ray_evidence, explain_component,
    receiver_evidence,
)
from tests.test_bev_flow import build, example


def script(name):
    path = Path(__file__).parents[1]/"scripts"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(name+"_receiver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receiver_case(dynamic=False):
    shape = (32, 32)
    mask = np.zeros(shape, bool)
    mask[12:16, 12:16] = True
    raw = np.full((3, *shape), 70., np.float32)
    ratio = np.ones_like(raw)
    slots = [1] if dynamic else [0, 1, 2]
    for slot in slots:
        raw[slot, mask] = 30
        ratio[slot, mask] = .6
    arrays = {"receiver_valid": np.ones(shape, bool), "appearance_L_raw": raw,
        "appearance_local_ratio": ratio, "temporal_exposure_valid": np.full(shape, dynamic),
        "temporal_illumination_change_L": np.diff(raw, axis=0),
        "physical_box_footprint": np.zeros((3, *shape), bool),
        "physical_ray_volume_support": np.zeros((3, *shape), bool),
        "shadow_flow_supported": np.ones(shape, bool)}
    arrays.update(receiver_evidence(arrays))
    hypothesis = {"instance_token": "car", "light_id": 0, "shadow_fit_passes": True,
                  "score": 1., "shadow_passes": False, "alternatives": [], "motion_supported": False}
    polygon = np.array([[[12, 12]], [[15, 12]], [[15, 15]], [[12, 15]]], np.int32)
    return arrays, mask, [hypothesis], {("car", 0): [polygon]*3}


def test_static_shadow_can_win_without_events_or_exposure_fit():
    arrays, mask, hypotheses, polygons = receiver_case()
    evidence = explain_component(mask, arrays, hypotheses, polygons, .5)
    assert evidence["winner"] == "cast_shadow" and evidence["shadow_supported"]
    assert evidence["transition_fraction"] == 0 and evidence["projection_control_gain"] > .5
    assert evidence["control_search"]["hypotheses_per_offset"] == 1


def test_oversized_projection_no_longer_wins_by_cover_alone():
    arrays, mask, hypotheses, polygons = receiver_case()
    polygon = np.array([[[0, 0]], [[31, 0]], [[31, 31]], [[0, 31]]], np.int32)
    polygons["car", 0] = [polygon]*3
    evidence = explain_component(mask, arrays, hypotheses, polygons, .5)
    assert evidence["baseline_scores"]["projection_coverage"] == 1
    assert evidence["best_projection"]["observed_lit_conflict"] > .5
    assert not evidence["shadow_supported"] and evidence["winner"] == "ground_material"


def test_shifted_controls_do_not_wrap_and_have_equal_light_search():
    arrays, mask, hypotheses, polygons = receiver_case()
    second = {**hypotheses[0], "light_id": 1}
    hypotheses[0]["alternatives"] = [second]
    polygons["car", 1] = polygons["car", 0]
    evidence = explain_component(mask, arrays, hypotheses, polygons, .5)
    assert evidence["control_search"]["real_hypotheses"] == 2
    assert evidence["control_search"]["hypotheses_per_offset"] == 2
    assert evidence["best_shifted_control"]["cover"] == 0


def test_dynamic_boundaries_are_not_ground_texture_flow():
    arrays, mask, hypotheses, polygons = receiver_case(dynamic=True)
    evidence = explain_component(mask, arrays, hypotheses, polygons, .5)
    assert arrays["receiver_boundary_formation"][mask].all()
    assert arrays["receiver_boundary_recovery"][mask].all()
    assert evidence["transition_fraction"] == 1
    assert evidence["shadow_supported"]


def test_unknown_geometry_and_entities_cannot_supply_shadow_absence_evidence():
    arrays, mask, hypotheses, polygons = receiver_case()
    arrays["receiver_valid"][:] = False
    arrays["appearance_local_ratio"][:] = np.nan
    arrays.update(receiver_evidence(arrays))
    evidence = explain_component(mask, arrays, hypotheses, polygons, .5)
    assert evidence["winner"] == "unknown" and not evidence["shadow_supported"]
    arrays["physical_ray_volume_support"][:, mask] = True
    arrays.update(receiver_evidence(arrays))
    evidence = explain_component(mask, arrays, hypotheses, polygons, .5)
    assert evidence["winner"] == "physical_entity" and not evidence["shadow_supported"]


def test_textureless_ground_does_not_get_fabricated_texture_similarity():
    arrays, _, _, _ = receiver_case()
    assert np.isnan(arrays["receiver_texture_cosine"][:, 2:8, 2:8]).all()
    assert not arrays["receiver_texture_valid"][:, 2:8, 2:8].any()


def ray_case():
    world = np.array([[[1.4, 0, 0], [3., 0, 0], [0., 0, 0]]])
    k = np.array([[20, 0, 32], [0, 20, 32], [0, 0, 1.]])
    rotation = np.diag([1., -1., -1.])
    projection = k @ np.c_[rotation, -rotation @ [0, 0, 10]]
    frames = []
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (2, 4)])
    for slot in range(3):
        frames.append({"projection_world_to_image": projection, "image_size_wh": [64, 64],
            "object_volumes_world": [{"instance_token": "car", "source": "camera_time_interpolated_box",
                                       "corners_world": corners+[slot, 0, 0]}]})
    pixels, _, _ = project_world(world.reshape(-1, 3), projection)
    pixels = [pixels.reshape(1, 3, 2).astype(np.float32)]*3
    flows = [np.full((1, 3, 2), [20/6, 0], np.float32)]*2
    return world, frames, pixels, flows


def test_ray_volume_detects_occlusion_outside_vertical_bev_footprint():
    values = box_ray_evidence(*ray_case())
    assert values["physical_ray_volume_support"][0, 0, 0]  # x=1.4, outside box x<=1.
    assert not values["physical_ray_volume_support"][0, 0, 1]
    assert values["box_surface_flow_valid"][0, 0, 0]
    assert values["box_surface_flow_residual_px"][0, 0, 0] < 1e-5


def test_fallback_boxes_never_supply_object_motion():
    data = ray_case()
    for frame in data[1]:
        frame["object_volumes_world"][0]["source"] = "annotation_time_exclusion_only"
    values = box_ray_evidence(*data)
    assert values["physical_ray_volume_support"].any()
    assert not values["box_surface_flow_valid"].any()
    assert np.isnan(values["box_surface_expected_flow_px"]).all()


def test_boxes_behind_camera_do_not_occlude_ground():
    data = ray_case()
    for frame in data[1]:
        frame["object_volumes_world"][0]["corners_world"][:, 2] += 10
    values = box_ray_evidence(*data)
    assert not values["physical_ray_volume_support"].any()


def test_unreliable_flow_does_not_supply_box_surface_motion():
    data = ray_case()
    result = box_ray_evidence(*data, observed_flow_valid=[np.zeros((1, 3), bool)]*2)
    assert result["physical_ray_volume_support"].any()
    assert not result["box_surface_flow_valid"].any()


def test_bev_builder_exports_separate_geometry_boundary_and_motion_arrays():
    arrays, metadata = build(example())
    assert metadata["receiver_explanation_protocol"] == "receiver-explanations-v1"
    assert arrays["receiver_boundary_formation"].any()
    assert arrays["box_surface_expected_flow_px"].shape == (2, 16, 16, 2)
    assert np.isnan(arrays["box_surface_expected_flow_px"]).all()
    assert not arrays["physical_ray_volume_support"].any()


def test_competition_keeps_uncertain_ownership_separate():
    arrays, mask, hypotheses, polygons = receiver_case()
    evidence = explain_component(mask, arrays, hypotheses, polygons, .5)
    decision = {"shadow_accepted": True, "selected": hypotheses[0], "association": None,
                "reason": "old_projection", "existence_score": 1.}
    updated = apply_competition(decision, evidence, hypotheses)
    assert updated["shadow_accepted"] and updated["association"] is None
    assert updated["baseline_projection_decision"]["reason"] == "old_projection"
    assert updated["plausible_hypotheses"] == hypotheses


def test_configuration_and_scores_are_not_probabilities():
    with pytest.raises(ValueError):
        CompetitionConfig(control_shift_m=0).validate()
    arrays, mask, hypotheses, polygons = receiver_case()
    assert not explain_component(mask, arrays, hypotheses, polygons, .5)["scores_are_probabilities"]


def test_darkness_baseline_does_not_saturate_all_dark_candidates():
    arrays, mask, hypotheses, polygons = receiver_case()
    first = explain_component(mask, arrays, hypotheses, polygons, .5)["baseline_scores"]["appearance"]
    arrays["appearance_local_ratio"][:, mask] = .8
    second = explain_component(mask, arrays, hypotheses, polygons, .5)["baseline_scores"]["appearance"]
    assert 0 < second < first < 1


def test_external_shadow_model_is_auxiliary_not_a_geometry_replacement():
    arrays, mask, hypotheses, polygons = receiver_case()
    arrays["external_shadow_probability"] = np.full((3, *mask.shape), .1, np.float32)
    arrays["external_shadow_probability"][:, mask] = .9
    result = explain_component(mask, arrays, hypotheses, polygons, .5)
    without = explain_component(mask, arrays, hypotheses, polygons, .5, CompetitionConfig(appearance_prior_weight=0))
    assert result["scores"]["cast_shadow"] > without["scores"]["cast_shadow"]
    assert result["baseline_scores"]["external_shadow_model"] > .8
    assert result["appearance_prior"]["requires_discriminative_projection"]
    polygon = np.array([[[0, 0]], [[31, 0]], [[31, 31]], [[0, 31]]], np.int32)
    polygons["car", 0] = [polygon]*3
    assert not explain_component(mask, arrays, hypotheses, polygons, .5)["shadow_supported"]


def test_external_baseline_maps_are_bound_to_the_exported_image(tmp_path):
    world, frames, _, _ = ray_case()
    for index, frame in enumerate(frames):
        frame.update(sample_data_token=f"frame{index}", image_sha256=f"image{index}")
    path = tmp_path/"baseline.npz"
    np.savez_compressed(path, probability=np.full((64, 64), .75, np.float32))
    rows = {"frame0": {"image_sha256": "image0", "probability": "baseline.npz", "model_id": "external-test"}}
    arrays = {"receiver_world_xyz": world, "receiver_valid": np.ones(world.shape[:2], bool)}
    result, metadata = attach_appearance_prior(arrays, frames, rows, tmp_path)
    np.testing.assert_allclose(result["external_shadow_probability"][0], .75)
    assert np.isnan(result["external_shadow_probability"][1:]).all()
    assert metadata["sources"][0]["model_id"] == "external-test"
    rows["frame0"]["image_sha256"] = "wrong-image"
    with pytest.raises(ValueError, match="fingerprint"):
        attach_appearance_prior(arrays, frames, rows, tmp_path)


def test_baseline_rank_metrics_require_fixed_external_reference():
    module = script("analyze_receiver_competition")
    rows = [{"id": str(i), "scene_name": f"s{i}", "baseline_scores": {"appearance": value}}
            for i, value in enumerate((.9, .1, .8, .2))]
    reference = {str(i): {"scene_name": f"s{i}", "shadow": "YES" if i % 2 == 0 else "NO"} for i in range(4)}
    metrics = module.rank_metrics(rows, reference, bootstrap=20)
    assert metrics["scores"]["appearance"]["auroc"] == 1
    assert metrics["scores"]["appearance"]["average_precision"] == 1
    assert metrics["matched"] == 4 and metrics["scenes"] == 4
    with pytest.raises(ValueError, match="No reference"):
        module.rank_metrics(rows, {})
    rows[0]["baseline_scores"]["appearance"] = None
    assert module.rank_metrics(rows, reference)["scores"]["appearance"]["status"] == "missing_scores"


def test_contact_sheet_covers_scenes_and_prioritizes_baseline_disagreements():
    items = [{"id": f"{scene}-{index}", "scene_name": scene,
              "shadow_decision": {"shadow_accepted": True},
              "receiver_competition": {"shadow_supported": index != 1}}
             for scene in ("a", "b") for index in range(4)]
    selected = script("analyze_receiver_competition").review_ids(items, limit=2, seed=0)
    assert selected == {"a-1", "b-1"}


def test_active_bev_label_and_baseline_audit_pipeline(tmp_path, monkeypatch):
    from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
    from shadow_occ.utils.io import write_json

    pairs, images, grounds, flows, exposures = example()
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]
    for index, frame in enumerate(frames):
        image, ground = tmp_path/f"i{index}.png", tmp_path/f"g{index}.png"
        checked_write_image(image, images[index])
        checked_write_image(ground, grounds[index].astype(np.uint8)*255)
        frame.update(image=str(image), ground_support=str(ground), image_sha256=digest(image), ground_sha256=digest(ground))
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
            onset_candidate=patch if index == 0 else np.zeros_like(patch),
            recovery_candidate=patch if index == 1 else np.zeros_like(patch), stable_dark_candidate=np.zeros_like(patch))
        metrics.append({**{key: pair[key] for key in ("id", "scene_name", "camera", "official_split")},
                        "evidence": str(path), "evidence_sha256": digest(path), "exposure": exposures[index]})
    write_rows(flow/"pair_metrics.jsonl", metrics)
    bev, labels, audit = tmp_path/"bev", tmp_path/"labels", tmp_path/"audit"
    monkeypatch.setattr(sys, "argv", ["analyze", "--pairs", str(manifest), "--analysis-root", str(flow),
        "--feature-gate", "none", "--extent-m", "4", "--output-root", str(bev)])
    script("analyze_bev_flow_discrepancy").main()
    monkeypatch.setattr(sys, "argv", ["labels", "--pairs", str(manifest), "--analysis-root", str(flow),
        "--bev-flow-root", str(bev), "--decision-policy", "factorized-v3", "--export-weak-labels", "--output-root", str(labels)])
    script("build_physics_shadow_labels").main()
    protocol = json.loads((labels/"label_protocol.json").read_text())
    assert protocol["receiver_competition"] == "on"
    decisions = read_rows(labels/"decisions.jsonl")
    assert decisions and all(row["receiver_competition"] for row in decisions)
    assert not any(row["shadow_decision"]["shadow_accepted"] for row in decisions)  # No caster in this fixture.
    monkeypatch.setattr(sys, "argv", ["audit", "--labels-root", str(labels), "--output-root", str(audit)])
    script("analyze_receiver_competition").main()
    report = json.loads((audit/"summary.json").read_text())
    assert report["components"] == len(decisions) and report["reference_metrics"] is None
    assert report["exported_positive"] == 0
    assert (audit/"contact_sheet.jpg").exists()
