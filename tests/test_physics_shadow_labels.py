import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import checked_write_image, digest, make_target, read_rows, write_rows
from shadow_occ.annotation.physics_labels import (
    LabelConfig, candidate_evidence, cast_box, footprint_mask, light_bank,
    middle_evidence, rank_hypotheses, receiver_points, select_with_independent_light,
    select_shadow_and_caster, trajectory_contrast, factorized_shadow_decision,
    validate_window, window_tracks,
)
from shadow_occ.annotation.physics_motion import plane_basis


def projection():
    k = np.array([[30, 0, 48], [0, 30, 32], [0, 0, 1]])
    rotation = np.diag([1.0, -1.0, -1.0])
    return k @ np.c_[rotation, -rotation @ [0, 0, 10]]


def boxes():
    signs = np.array([[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1],
                      [-1, 1, 1], [-1, -1, 1], [-1, -1, -1], [-1, 1, -1]])
    return np.asarray([np.vstack([[x, 0, 1], signs+[x, 0, 1]]) for x in (-2, 0, 2)], np.float64)


def pair_rows():
    frames = [{"sample_data_token": f"cam{index}", "camera_timestamp": index*500000,
               "image_sha256": f"sha{index}", "image_size_wh": [96, 64],
               "projection_world_to_image": projection().tolist()} for index in range(3)]
    points = boxes()
    pairs = []
    for index in range(2):
        pairs.append({"id": f"pair{index}", "scene_name": "scene-0001", "official_split": "train",
            "camera": "CAM_FRONT", "sample_token": f"s{index}", "next_sample_token": f"s{index+1}",
            "frames": frames[index:index+2], "dt_seconds": 0.5,
            "plane_world": [0, 0, 1, 0], "plane_basis_world": plane_basis([0, 0, 1, 0]).tolist(),
            "protocol": "physics-motion-evidence-v1", "train_approved": False,
            "objects": [{"instance_token": "car", "category": "vehicle.car",
                         "annotation_token_t": f"ann{index}", "annotation_token_next": f"ann{index+1}",
                         "landmarks_world_t": points[index].tolist(), "landmarks_world_next": points[index+1].tolist()}]})
    return pairs


def test_light_cast_intersects_plane_and_rejects_backwards_or_below_ground():
    corners = boxes()[1, 1:]
    direction = np.array([1, 0, -1])/np.sqrt(2)
    shadow = cast_box(corners, [0, 0, 1, 0], direction)
    np.testing.assert_allclose(shadow[:, 2], 0, atol=1e-8)
    np.testing.assert_allclose(shadow[:, 0], corners[:, 0]+corners[:, 2])
    with pytest.raises(ValueError, match="does not reach"):
        cast_box(corners, [0, 0, 1, 0], [1, 0, 1])
    with pytest.raises(ValueError, match="below"):
        cast_box(corners-[0, 0, 1], [0, 0, 1, 0], direction)


def test_light_bank_is_unit_downward_and_frozen():
    rays = light_bank()
    assert rays.shape == (96, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1)
    assert (rays[:, 2] < 0).all()
    np.testing.assert_array_equal(rays, light_bank())


def test_middle_darkness_intersects_observations_not_wrong_time_union():
    left, right = pair_rows()
    zero = np.zeros((64, 96), np.uint8)
    first = {key: zero.copy() for key in ("onset_candidate", "recovery_candidate", "stable_dark_candidate")}
    second = {key: value.copy() for key, value in first.items()}
    for archive in (first, second):
        archive["visible_ground"] = np.ones_like(zero)
        archive["flow_reliable"] = np.ones_like(zero)
    first["onset_candidate"][20:40, 20:40] = 1
    second["recovery_candidate"][20:40, 30:50] = 1
    result = middle_evidence(left, right, first, second)
    assert result["dark"].sum() == 200
    assert result["same_patch_cycle"].sum() == 200
    assert not result["dark"][:, :30].any()


@pytest.mark.parametrize("key,value", [("scene_name", "other"), ("camera", "CAM_BACK"), ("official_split", "val")])
def test_window_rejects_cross_scene_camera_or_split(key, value):
    left, right = pair_rows()
    with pytest.raises(ValueError, match="crosses"):
        validate_window(left, {**right, key: value})


def test_window_rejects_changed_shared_frame_even_if_sample_matches():
    left, right = pair_rows()
    right["frames"] = [{**right["frames"][0], "image_sha256": "tampered"}, right["frames"][1]]
    with pytest.raises(ValueError, match="Middle camera"):
        validate_window(left, right)


def test_old_pair_schema_without_world_landmarks_cannot_silently_associate():
    left, right = pair_rows()
    del left["objects"][0]["landmarks_world_t"]
    tracks, rejected = window_tracks(left, right)
    assert not tracks and len(rejected) == 1


def test_shared_track_state_must_refer_to_same_middle_annotation():
    left, right = pair_rows()
    tracks, rejected = window_tracks(left, right)
    assert len(tracks) == 1 and not rejected
    right["objects"][0]["annotation_token_t"] = "another"
    assert not window_tracks(left, right)[0]


def test_static_darkness_is_retained_unknown_never_negative():
    mask = np.ones((10, 10), bool)
    maps = {"onset": np.zeros_like(mask), "recovery": np.zeros_like(mask),
            "flow_supported": mask, "same_patch_cycle": np.zeros_like(mask)}
    report = candidate_evidence(mask, maps, LabelConfig())
    assert any("static_dark_is_not_negative" in reason for reason in report["reasons"])
    ranks, _ = rank_hypotheses(mask, maps, [], [0, 0, 1, 0], plane_basis([0, 0, 1, 0]),
                               projection(), light_bank(), LabelConfig())
    assert ranks == []


def moving_case():
    points = boxes()
    direction = np.array([[1, 0, -1]])/np.sqrt(2)
    footprints = [footprint_mask(p[1:], [0, 0, 1, 0], direction[0], projection(), (64, 96)) for p in points]
    mask = footprints[1]
    maps = {"onset": mask & ~footprints[0], "recovery": mask & ~footprints[2],
            "flow_supported": np.ones_like(mask), "same_patch_cycle": mask & ~footprints[0] & ~footprints[2]}
    tracks = [{"instance_token": "car", "points": points}]
    return mask, maps, tracks, direction


def test_joint_fit_uses_three_states_and_keeps_disjoint_onset_recovery_edges():
    mask, maps, tracks, directions = moving_case()
    rankings, report = rank_hypotheses(mask, maps, tracks, [0, 0, 1, 0], plane_basis([0, 0, 1, 0]),
                                       projection(), directions, LabelConfig())
    assert not report["reasons"]
    assert rankings[0]["passes"] and rankings[0]["score"] == 1
    assert rankings[0]["instance_token"] == "car"


def test_identical_competing_caster_is_rejected_not_nearest_selected():
    mask, maps, tracks, directions = moving_case()
    tracks.append({**tracks[0], "instance_token": "ambiguous_car"})
    rankings, _ = rank_hypotheses(mask, maps, tracks, [0, 0, 1, 0], plane_basis([0, 0, 1, 0]),
                                  projection(), directions, LabelConfig())
    assert rankings[0]["caster_margin"] == 0
    assert not rankings[0]["passes"]


def test_real_order_must_beat_frozen_and_wrong_time_explanations():
    mask, maps, tracks, directions = moving_case()
    footprints = [footprint_mask(p[1:], [0, 0, 1, 0], directions[0], projection(), mask.shape)
                  for p in tracks[0]["points"]]
    result = trajectory_contrast(footprints, mask, maps["onset"], maps["recovery"])
    assert result["score"] == 1 and result["temporal_gain"] >= 0.1
    reversed_result = trajectory_contrast(footprints[::-1], mask, maps["onset"], maps["recovery"])
    assert reversed_result["temporal_gain"] < 0
    frozen = trajectory_contrast([mask]*3, mask, maps["onset"], maps["recovery"])
    assert frozen["temporal_gain"] == 0


def test_ambiguous_caster_does_not_erase_independently_supported_shadow():
    mask, maps, tracks, directions = moving_case()
    tracks.append({**tracks[0], "instance_token": "second_car"})
    hypotheses, _ = rank_hypotheses(mask, maps, tracks, [0, 0, 1, 0], plane_basis([0, 0, 1, 0]),
                                    projection(), directions, LabelConfig())
    candidate = {"scene_name": "scene", "official_split": "train", "frame_tokens": ["middle"],
                 "hypotheses": hypotheses}
    support = [vote("a", "other1"), vote("b", "other2")]
    decision = select_shadow_and_caster(candidate, support, directions, LabelConfig())
    assert decision["shadow_accepted"] and decision["association"] is None
    assert {item["instance_token"] for item in decision["plausible_hypotheses"]} == {"car", "second_car"}


def test_multicamera_same_physical_sample_is_not_independent_light_evidence():
    target = {**vote("car", "front_middle"), "sample_tokens": ["s0", "s1", "s2"]}
    pool = [{**vote(token, f"back_{token}"), "sample_tokens": ["s1", "s3", "s4"]}
            for token in ("a", "b")]
    assert not select_shadow_and_caster(target, pool, light_bank(), LabelConfig())["shadow_accepted"]


def test_ambiguous_possible_owner_cannot_vote_for_its_own_shadow():
    target = vote("car", "middle")
    target["hypotheses"][0].update(passes=False, shadow_passes=True,
        alternatives=[{"instance_token": "other_car", "score": 0.89, "light_id": 0}])
    pool = [vote("other_car", "other_time"), vote("b", "another_time")]
    assert not select_shadow_and_caster(target, pool, light_bank(), LabelConfig())["shadow_accepted"]


def test_factorized_existence_does_not_require_other_track_light_votes():
    mask, maps, tracks, directions = moving_case()
    hypotheses, _ = rank_hypotheses(mask, maps, tracks, [0, 0, 1, 0], plane_basis([0, 0, 1, 0]),
                                    projection(), directions, LabelConfig())
    candidate = {"scene_name": "scene", "official_split": "train", "frame_tokens": ["f"],
                 "hypotheses": hypotheses}
    assert not select_shadow_and_caster(candidate, [], directions, LabelConfig())["shadow_accepted"]
    decision = factorized_shadow_decision(candidate, [], directions, LabelConfig())
    assert decision["shadow_accepted"] and decision["association"] is None
    assert sum(row["weight"] for row in decision["light_weights"]) == pytest.approx(1)


def test_factorized_policy_needs_a_projection_explanation_not_just_darkness():
    candidate = {"scene_name": "scene", "official_split": "train", "frame_tokens": ["f"], "hypotheses": []}
    assert not factorized_shadow_decision(candidate, [], light_bank(), LabelConfig())["shadow_accepted"]


def test_polygon_cache_matches_uncached_roi_results():
    mask, maps, tracks, directions = moving_case()
    args = (mask, maps, tracks, [0, 0, 1, 0], plane_basis([0, 0, 1, 0]), projection(), directions, LabelConfig())
    cache = {}
    a = rank_hypotheses(*args, polygon_cache=cache)
    assert cache
    assert rank_hypotheses(*args, polygon_cache=cache) == a == rank_hypotheses(*args)


def test_metric_receiver_points_are_on_real_plane_not_object_height():
    mask, _, _, _ = moving_case()
    points = receiver_points(mask, projection(), plane_basis([0, 0, 1, 0]))
    assert 0 < len(points) <= 256
    np.testing.assert_allclose(points[:, 2], 0)


def vote(token, frame, light=0, split="train"):
    return {"scene_name": "scene", "official_split": split, "frame_tokens": [frame],
            "hypotheses": [{"instance_token": token, "light_id": light, "passes": True, "score": 0.9}]}


def test_scene_light_excludes_own_track_shared_frames_duplicates_and_eval():
    rays = light_bank()
    target = vote("car", "middle")
    config = LabelConfig()
    invalid_support = [vote("car", "other"), vote("a", "middle"), vote("b", "different", split="val")]
    assert select_with_independent_light(target, invalid_support, rays, config)[0] is None
    duplicate = [vote("a", "frame1"), vote("a", "frame2")]
    assert select_with_independent_light(target, duplicate, rays, config)[0] is None
    selected, reason = select_with_independent_light(target, duplicate+[vote("b", "frame3")], rays, config)
    assert selected["independent_light_tracks"] == ["a", "b"]
    assert "not_verified" in reason


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_only_manifest_is_blocked_by_training_builder(tmp_path, monkeypatch):
    builder = script("build_shadow_annotation_dataset")
    manifest = tmp_path/"audit.jsonl"
    write_rows(manifest, [{"physics_protocol": "three-frame-physics-weak-v1", "physics_export_enabled": False}])
    monkeypatch.setattr(sys, "argv", ["builder", "--manifest", str(manifest), "--output-root", str(tmp_path/"data")])
    with pytest.raises(ValueError, match="audit-only"):
        builder.main()


def test_default_context_does_not_reveal_gt_caster_location():
    builder = script("build_shadow_annotation_dataset")
    box, shape = [100, 100, 120, 120], (400, 600)
    a = builder.context_bounds(box, shape, [200, 200, 240, 240])
    b = builder.context_bounds(box, shape, [400, 300, 450, 350])
    assert a == b == builder.context_bounds(box, shape)
    assert a != builder.context_bounds(box, shape, [400, 300, 450, 350], "gt-assisted")


@pytest.mark.parametrize("gain", [-0.1, 1.1, float("nan")])
def test_track_contrast_rejects_invalid_threshold(gain):
    with pytest.raises(ValueError, match="contrast"):
        LabelConfig(minimum_temporal_gain=gain).validate()


def test_stage3_integration_exports_no_unapproved_targets(tmp_path, monkeypatch):
    builder = script("build_physics_shadow_labels")
    pairs = pair_rows()
    mask, maps, _, _ = moving_case()
    frames = [pairs[0]["frames"][0], pairs[0]["frames"][1], pairs[1]["frames"][1]]
    for index, frame in enumerate(frames):
        image, ground = tmp_path/f"i{index}.png", tmp_path/f"g{index}.png"
        checked_write_image(image, np.full((64, 96, 3), 150, np.uint8))
        checked_write_image(ground, np.full((64, 96), 255, np.uint8))
        frame.update(image=str(image), ground_support=str(ground), image_sha256=digest(image), ground_sha256=digest(ground))
    # Restore the shared dictionary references explicitly in the second pair.
    pairs[1]["frames"][0] = frames[1]
    manifest = tmp_path/"pairs.jsonl"
    write_rows(manifest, pairs)
    analysis = tmp_path/"flow"
    analysis.mkdir()
    (analysis/"analysis_protocol.json").write_text(json.dumps({"manifest_sha256": digest(manifest),
        "flow_evidence_version": "receiver-support-v2"}))
    metrics = []
    for index, pair in enumerate(pairs):
        path = analysis/f"e{index}.npz"
        # t0->t1 observes middle darkness, t1->t2 observes its recovery.
        np.savez_compressed(path, visible_ground=np.ones_like(mask), flow_reliable=np.ones_like(mask),
                            receiver_flow_supported=np.ones_like(mask),
                            onset_candidate=maps["onset"] if index == 0 else np.zeros_like(mask),
                            recovery_candidate=maps["recovery"] if index == 1 else np.zeros_like(mask),
                            stable_dark_candidate=mask & ~(maps["onset"] if index == 0 else maps["recovery"]))
        metrics.append({**{key: pair[key] for key in ("id", "scene_name", "camera", "official_split")},
                        "evidence": str(path), "evidence_sha256": digest(path), "exposure": {"valid": True}})
    write_rows(analysis/"pair_metrics.jsonl", metrics)
    monkeypatch.setattr(builder, "light_bank", lambda: np.array([[1, 0, -1]])/np.sqrt(2))
    original_selector = builder.select_shadow_and_caster
    for enabled, independent_support in ((False, False), (True, False), (True, True)):
        output = tmp_path/f"labels_{enabled}_{independent_support}"
        if independent_support:
            # Exercise the export/consumer contract separately from the real consensus test above.
            def accepted(candidate, *args, **kwargs):
                best = next(item for item in candidate["hypotheses"] if item["passes"])
                best = {**best, "independent_light_tracks": ["other_a", "other_b"]}
                return {"shadow_accepted": True, "selected": best, "association": best,
                        "reason": "synthetic_test_gate", "plausible_hypotheses": [best]}
            monkeypatch.setattr(builder, "select_shadow_and_caster", accepted)
        else:
            monkeypatch.setattr(builder, "select_shadow_and_caster", original_selector)
        argv = ["builder", "--pairs", str(manifest), "--analysis-root", str(analysis), "--output-root", str(output)]
        if enabled:
            argv.append("--export-weak-labels")
        monkeypatch.setattr(sys, "argv", argv)
        builder.main()
        rows = read_rows(output/"canonical.jsonl")
        assert rows
        # Single track cannot approve itself, even when weak export is enabled.
        expected = "YES" if independent_support else "IGNORE"
        assert all(row["regions"][0]["shadow_label"] == expected for row in rows)
        for row in rows:
            target = make_target(row, (64, 96), output)
            assert not (target == 0).any()
            assert (target == 1).any() if independent_support else (target == 255).all()
        summary = json.loads((output/"summary.json").read_text())
        assert summary["exported_negative_pixels"] == 0 and summary["shadow_precision"] is None
        if independent_support:
            consumer = script("build_shadow_annotation_dataset")
            data = tmp_path/"region_data"
            monkeypatch.setattr(sys, "argv", ["consumer", "--manifest", str(output/"canonical.jsonl"),
                                            "--output-root", str(data)])
            consumer.main()
            vlm = [row for split in ("train", "dev", "eval") for row in read_rows(data/f"vlm_{split}.jsonl")
                   if row["task"] == "shadow"]
            assert vlm and all(row["answer"] == "YES" and len(row["images"]) == 5 for row in vlm)
            assert all(row["physics_temporal_context"] and not row["temporal_context"] for row in vlm)
            joint = script("build_joint_shadow_annotation_dataset")
            monkeypatch.setattr(sys, "argv", ["joint", "--region-data-root", str(data),
                                            "--output-root", str(tmp_path/"joint")])
            joint.main()
            unified = [row for split in ("train", "dev", "eval")
                       for row in read_rows(tmp_path/"joint"/f"annotation_{split}.jsonl")]
            assert unified and all(row["task"] == "shadow_annotation" for row in unified)
            assert "same receiver ground" in unified[0]["prompt"]
            unknown = dict(rows[0])
            unknown["id"] += "_ignore_only"
            target_path = tmp_path/"ignore_target.png"
            checked_write_image(target_path, np.full((64, 96), 255, np.uint8))
            unknown["shadow_mask"] = str(target_path)
            unknown["physics_target_sha256"] = digest(target_path)
            unknown["regions"] = [{**unknown["regions"][0], "shadow_label": "IGNORE"}]
            augmented = tmp_path/"augmented.jsonl"
            write_rows(augmented, [rows[0], unknown])
            weighted = tmp_path/"weighted"
            monkeypatch.setattr(sys, "argv", ["consumer", "--manifest", str(augmented), "--output-root", str(weighted)])
            consumer.main()
            segments = [row for split in ("train", "dev", "eval") for row in read_rows(weighted/f"seg_{split}.jsonl")]
            assert len(segments) == 2 and all(row["sample_weight"] == 1 for row in segments)
