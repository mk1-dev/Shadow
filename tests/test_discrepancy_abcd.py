import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from shadow_occ.annotation.cached_bev import select_bev_flow_receivers
from shadow_occ.annotation.data import digest, write_rows
from shadow_occ.utils.io import write_json
from tests.test_discrepancy_proposals import predicted_motion_case, weak_case


ROOT = Path(__file__).parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name+"_abcd", ROOT/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_abcd_changes_only_declared_discovery_variable():
    config = json.loads((ROOT/"configs/discrepancy_abcd.json").read_text())
    recipes = {name: {**config["shared"], **value} for name, value in config["variants"].items()}
    base = recipes["C_discrepancy"]
    assert recipes["D_no_entity_motion"] == {**base, "proposal_entity_motion": "off"}
    assert recipes["A_hard"] == {**base, "discovery_mode": "hard-intersection"}
    assert recipes["B_cross"] == {**base, "discovery_mode": "cross-space"}
    assert recipes["C_gt_reverse"] == {**base, "caster_motion_control": "reverse-time"}
    assert recipes["C_bev_shuffled"] == {**base, "feature_gate": "shuffled"}
    assert config["shared"]["verification_evidence"] == "geometry"


def test_entity_motion_ablation_keeps_original_3d_evidence_for_verification():
    arrays, mask = predicted_motion_case()
    original = copy.deepcopy(arrays)
    on, _ = select_bev_flow_receivers(copy.deepcopy(arrays), {"counts": {}})
    off, meta = select_bev_flow_receivers(arrays, {"counts": {}}, proposal_entity_motion="off")
    assert not on["discrepancy_candidate"].any()
    assert off["discrepancy_candidate"][mask].all()
    assert not off["discrepancy_entity_explanation_valid"].any()
    for key in original:
        if key.startswith("mature_"):
            np.testing.assert_array_equal(off[key], original[key])
    assert meta["discrepancy_discovery"]["proposal_entity_motion"] == "off"


def test_bev_shuffle_leaves_nonreceiver_features_and_validity_unchanged():
    arrays, receiver = weak_case()
    arrays["receiver_valid"] = receiver.copy()
    features = np.random.default_rng(3).normal(size=(1, 8, 8, 12)).astype(np.float32)
    arrays["learned_bev_receiver_features"] = np.repeat(features, 3, axis=0)
    original_features = arrays["learned_bev_receiver_features"].copy()
    original_valid = arrays["learned_bev_valid"].copy()
    select_bev_flow_receivers(arrays, {"counts": {}}, feature_gate="shuffled", seed=17)
    np.testing.assert_allclose(arrays["learned_bev_receiver_consistency"][~receiver], 1, atol=1e-6)
    np.testing.assert_array_equal(arrays["learned_bev_receiver_features"], original_features)
    np.testing.assert_array_equal(arrays["learned_bev_valid"], original_valid)
    np.testing.assert_array_equal(arrays["receiver_valid"], receiver)


def test_same_geometry_verification_events_for_all_candidate_modes():
    builder = script("build_physics_shadow_labels")
    arrays, mask = weak_case(6)
    expected = None
    for mode in ("hard-intersection", "cross-space", "discrepancy"):
        selected, _ = select_bev_flow_receivers(copy.deepcopy(arrays), {"counts": {}}, discovery_mode=mode)
        events = builder.verification_observations(selected, 6)
        assert events[0][mask].all() and events[1][mask].all()
        if expected is not None:
            for first, second in zip(expected, events):
                np.testing.assert_array_equal(first, second)
        expected = events


def test_wrong_time_control_keeps_centre_state_and_observations_unchanged():
    builder = script("build_physics_shadow_labels")
    points = np.arange(3*9*3).reshape(3, 9, 3)
    extended = np.arange(4*9*3).reshape(4, 9, 3)
    context = {"frames": [{"timestamp": index} for index in range(4)], "middle": 1,
        "tracks": {"car": extended.copy()}, "polygon_cache": {"old": "cached"},
        "onset": np.ones((4, 2, 2), bool)}
    tracks = [{"instance_token": "car", "points": points.copy()}]
    wrong, wrong_context = builder.verification_tracks(tracks, context, "reverse-time")
    np.testing.assert_array_equal(wrong[0]["points"][1], points[1])
    np.testing.assert_array_equal(wrong[0]["points"][0], points[2])
    np.testing.assert_array_equal(wrong_context["tracks"]["car"][1], extended[1])
    assert wrong_context["polygon_cache"] == {}
    assert wrong_context["onset"] is context["onset"]
    np.testing.assert_array_equal(context["tracks"]["car"], extended)


def test_paired_report_counts_masks_before_and_after_candidate_union(tmp_path):
    summary = script("summarize_bev_flow_ablations")
    full = np.ones((2, 2), bool)
    for name, mask in (("C_discrepancy", np.zeros_like(full)), ("D_no_entity_motion", np.eye(2, dtype=bool))):
        root = tmp_path/name/"train/bev_flow"
        root.mkdir(parents=True)
        path = root/"window.npz"
        np.savez_compressed(path, receiver_valid=full, discrepancy_candidate=mask,
            shadow_candidate=full)
        write_rows(root/"windows.jsonl", [{"id": "same", "grid": {"resolution": .5}, "evidence": str(path)}])
    result = summary.paired_discovery(tmp_path, "C_discrepancy", "D_no_entity_motion", "train")
    assert result["pixels"]["discrepancy_candidate"]["right_only"] == 2
    assert result["pixels"]["shadow_candidate"]["right_only"] == 0
    assert result["windows"] == 1


def test_audit_runner_does_not_build_training_datasets(tmp_path, monkeypatch):
    runner = script("run_bev_flow_ablations")
    source, output = tmp_path/"source", tmp_path/"output"
    source.mkdir()
    for split in ("train", "val"):
        cache = source/split/"bev_flow"
        cache.mkdir(parents=True)
        path = cache/"window.npz"
        np.savez_compressed(path, observed_flow_valid_by_interval=np.ones((2, 2, 2), bool),
            expected_geometric_flow_px=np.zeros((2, 2, 2, 2)), observed_optical_flow_px=np.zeros((2, 2, 2, 2)))
        write_rows(cache/"windows.jsonl", [{"id": "window", "evidence": str(path), "evidence_sha256": digest(path)}])
        write_json(cache/"bev_flow_protocol.json", {})
    commands = []

    def run(command, check, cwd):
        commands.append(command)
        if Path(command[1]).name == "build_physics_shadow_labels.py":
            destination = Path(command[command.index("--output-root")+1])
            destination.mkdir(parents=True)
            write_json(destination/"summary.json", {"stage_counts": {"accepted_shadow": 0}})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    variants = ["A_hard", "B_cross", "C_discrepancy", "D_no_entity_motion", "C_bev_shuffled", "C_gt_reverse"]
    monkeypatch.setattr(sys, "argv", ["run", "--source-root", str(source), "--output-root", str(output),
        "--config", str(ROOT/"configs/discrepancy_abcd.json"), "--audit-only", "--variants", *variants])
    runner.main()
    names = {Path(command[1]).name for command in commands}
    assert names == {"reselect_bev_flow.py", "build_physics_shadow_labels.py", "analyze_receiver_competition.py",
        "summarize_bev_flow_ablations.py"}
    for command in commands:
        if Path(command[1]).name == "build_physics_shadow_labels.py":
            assert command[command.index("--verification-evidence")+1] == "geometry"
            assert command[command.index("--candidate-source")+1] == "caster"
        elif Path(command[1]).name == "reselect_bev_flow.py":
            off = "D_no_entity_motion" in command[command.index("--output-root")+1]
            assert command[command.index("--proposal-entity-motion")+1] == ("off" if off else "on")
