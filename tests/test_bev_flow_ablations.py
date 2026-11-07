import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.cached_bev import select_bev_flow_receivers
from shadow_occ.annotation.physics_labels import LabelConfig, fuse_shadow_support, light_bank, select_shadow_and_caster
from shadow_occ.annotation.metrics import pixel_score_diagnostics
from shadow_occ.utils.io import write_json


def script(name):
    path = Path(__file__).parents[1]/"scripts"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receivers():
    full = np.ones((4, 4), bool)
    cosine = np.ones((2, 4, 4), np.float32)
    cosine[:, :2] = -1
    return {"shadow_candidate": full.copy(), "shadow_flow_supported": full.copy(),
        "shadow_boundary_onset": full.copy(), "shadow_boundary_recovery": full.copy(),
        "shadow_same_patch_cycle": full.copy(), "learned_bev_valid": np.ones((3, 4, 4), bool),
        "learned_bev_temporal_cosine": cosine,
        "learned_bev_receiver_features": np.random.default_rng(7).normal(size=(3, 4, 4, 8)).astype(np.float32)}


def test_geometry_only_gate_needs_no_learned_arrays():
    arrays = {key: value for key, value in receivers().items() if not key.startswith("learned")}
    result, meta = select_bev_flow_receivers(arrays, {"counts": {}}, feature_gate="none", discovery_mode="hard-intersection")
    assert result["shadow_candidate"].sum() == 16
    assert meta["learned_bev_selection"]["feature_gate"] == "none"


def test_repeated_gate_uses_unfiltered_candidates_and_transitions():
    arrays, meta = select_bev_flow_receivers(receivers(), {"counts": {}}, .3, discovery_mode="hard-intersection")
    assert arrays["shadow_candidate"].sum() == 8
    arrays, _ = select_bev_flow_receivers(arrays, meta, -.99, feature_gate="none", discovery_mode="hard-intersection")
    assert arrays["shadow_candidate"].sum() == 16
    assert arrays["shadow_boundary_onset"].sum() == 16


def test_random_gate_matches_aligned_coverage_and_repeats_by_seed():
    result, _ = select_bev_flow_receivers(receivers(), {"counts": {}}, feature_gate="matched-random", seed=17, discovery_mode="hard-intersection")
    repeated, _ = select_bev_flow_receivers(receivers(), {"counts": {}}, feature_gate="matched-random", seed=17, discovery_mode="hard-intersection")
    assert result["shadow_candidate"].sum() == 8
    np.testing.assert_array_equal(result["shadow_candidate"], repeated["shadow_candidate"])


def test_flow_gate_can_be_restored_after_removal():
    arrays = receivers()
    arrays["shadow_flow_supported"][:2] = False
    arrays, meta = select_bev_flow_receivers(arrays, {"counts": {}}, feature_gate="none", flow_gate=False, discovery_mode="hard-intersection")
    assert arrays["shadow_candidate"].sum() == 16
    arrays, _ = select_bev_flow_receivers(arrays, meta, feature_gate="none", flow_gate=True, discovery_mode="hard-intersection")
    assert arrays["shadow_candidate"].sum() == 8


def test_shuffled_gate_is_deterministic_and_not_in_place_feature_corruption():
    arrays = receivers()
    original = arrays["learned_bev_receiver_features"].copy()
    result, _ = select_bev_flow_receivers(arrays, {"counts": {}}, feature_gate="shuffled", seed=3, discovery_mode="hard-intersection")
    other, _ = select_bev_flow_receivers(receivers(), {"counts": {}}, feature_gate="shuffled", seed=3, discovery_mode="hard-intersection")
    np.testing.assert_array_equal(result["shadow_candidate"], other["shadow_candidate"])
    np.testing.assert_array_equal(arrays["learned_bev_receiver_features"], original)


def test_physical_support_fusion_preserves_observed_domain():
    observed = np.array([[1, 1, 1, 0]], bool)
    masks = [np.array([[1, 1, 1, 1]], bool), np.array([[1, 1, 0, 1]], bool), np.array([[1, 0, 0, 1]], bool)]
    assert fuse_shadow_support(observed, masks, "best").sum() == 3
    assert fuse_shadow_support(observed, masks, "intersection").sum() == 1
    assert fuse_shadow_support(observed, masks, "consensus").sum() == 2


def test_independent_light_ablation_does_not_require_other_tracks():
    hypothesis = {"instance_token": "car", "light_id": 0, "score": .9, "shadow_passes": True,
                  "passes": True, "alternatives": []}
    candidate = {"hypotheses": [hypothesis], "scene_name": "scene1", "official_split": "train",
                 "frame_tokens": ["f0", "f1", "f2"], "sample_tokens": ["s0", "s1", "s2"]}
    assert not select_shadow_and_caster(candidate, [candidate], light_bank(), LabelConfig())["shadow_accepted"]
    accepted = select_shadow_and_caster(candidate, [candidate], light_bank(), LabelConfig(require_independent_light=False))
    assert accepted["shadow_accepted"] and accepted["association"] is not None


def test_current_only_resets_both_uniad_history_stores():
    model = SimpleNamespace(test_track_instances=object(), prev_bev=object(),
        prev_frame_info={"prev_bev": object(), "scene_token": "old", "prev_pos": [1, 2, 3]})
    script("export_mature_bev").reset_mature_history(model)
    assert model.prev_bev is None and model.test_track_instances is None
    assert model.prev_frame_info["prev_bev"] is None and model.prev_frame_info["scene_token"] is None


def test_weak_label_controls_keep_positive_pixels_and_ignore_when_requested():
    transform = script("make_shadow_label_ablation").ablate_target
    target = np.array([[1, 0, 255, 255], [255, 0, 255, 255]], np.uint8)
    positive = transform(target, "positive-only", np.random.default_rng(0))
    assert (positive == 1).sum() == 1 and not (positive == 0).any()
    shuffled = transform(target, "shuffled-negatives", np.random.default_rng(0))
    assert (shuffled == 0).sum() == 2 and shuffled[0, 0] == 1
    assert not ((shuffled == 0) & (target == 0)).any()
    dense = transform(target, "all-background", np.random.default_rng(0))
    assert not (dense == 255).any() and dense[0, 0] == 1


def test_ablation_config_has_unique_complete_and_strict_ladder():
    root = Path(__file__).parents[1]
    config = json.loads((root/"configs/bev_flow_ablations.json").read_text())
    variants = config["variants"]
    assert "full" in variants and "current_only" in variants
    assert all(variants[name]["decision_policy"] == "strict-v2" for name in
               ("physics_fixed", "physics_bank", "physics_contrast", "physics_shared"))


def test_reference_quality_is_not_coverage():
    metrics = script("summarize_bev_flow_ablations").reference_metrics([6, 2, 8, 4, 20, 40, 16])
    assert metrics["positive_precision"] == .75
    assert metrics["negative_precision"] == 8/12
    assert metrics["known_pixel_coverage"] == .5
    assert metrics["positive_reference_capture"] == 6/16


def test_pixel_pr_curve_and_ap_use_class_ties_not_accuracy():
    histogram = np.zeros((2, 256), np.int64)
    histogram[0, 10], histogram[1, 240] = 900, 100
    result = pixel_score_diagnostics(histogram)
    assert result["approximate_average_precision"] == 1
    assert len(result["pr_curve"]) == 256
    histogram[:] = 0
    histogram[0, 10], histogram[1, 10] = 900, 100
    assert np.isclose(pixel_score_diagnostics(histogram)["approximate_average_precision"], .1)


def test_cached_reselection_restores_old_filtered_transitions_without_feature_copies(tmp_path, monkeypatch):
    import sys

    arrays, row = select_bev_flow_receivers(receivers(), {"id": "window", "counts": {}}, .3, discovery_mode="hard-intersection")
    arrays["temporal_illumination_change_L"] = np.stack([np.full((4, 4), -10), np.full((4, 4), 10)])
    for key in list(arrays):
        if key.endswith("_photometric") and key != "shadow_candidate_photometric":
            del arrays[key]
    source = tmp_path/"source"
    source.mkdir()
    path = source/"evidence.npz"
    np.savez_compressed(path, **arrays)
    write_rows(source/"windows.jsonl", [{**row, "evidence": str(path), "evidence_sha256": digest(path)}])
    write_json(source/"bev_flow_protocol.json", {"config": {"light_threshold_L": 6}})
    output = tmp_path/"none"
    monkeypatch.setattr(sys, "argv", ["reselect", "--source-root", str(source), "--output-root", str(output),
                                     "--feature-gate", "none", "--mature-object-mode", "off", "--discovery-mode", "hard-intersection"])
    script("reselect_bev_flow").main()
    result = read_rows(output/"windows.jsonl")[0]
    with np.load(result["evidence"], allow_pickle=False) as archive:
        assert archive["shadow_candidate"].sum() == 16
        assert archive["shadow_boundary_onset"].sum() == 16
        assert "learned_bev_receiver_features" not in archive.files


def test_reference_report_counts_missing_frames_as_zero_coverage(tmp_path):
    target = np.array([[1, 0]], np.uint8)
    path, predicted_path = tmp_path/"truth.png", tmp_path/"prediction.png"
    checked_write_image(path, target)
    checked_write_image(predicted_path, target)
    reference = [{"scene_name": f"scene{i}", "camera": "CAM_FRONT", "sample_token": f"s{i}", "target": str(path)} for i in range(2)]
    canonical = [{**reference[0], "shadow_mask": str(predicted_path)}]
    result = script("summarize_bev_flow_ablations").evaluate_reference(canonical, reference, tmp_path, 100, 0)
    assert result["positive_precision"] == 1 and result["negative_precision"] == 1
    assert result["known_pixel_coverage"] == .5 and result["positive_reference_capture"] == .5
    assert result["reference_frames"] == 2 and result["matched_frames"] == 1


def test_label_ablation_keeps_dev_eval_targets_unchanged(tmp_path, monkeypatch):
    import sys

    source, output = tmp_path/"data", tmp_path/"ablation"
    source.mkdir()
    image, target_path = source/"image.png", source/"target.png"
    checked_write_image(image, np.full((4, 4, 3), 180, np.uint8))
    target = np.full((4, 4), 255, np.uint8)
    target[0, 0], target[1, 1] = 1, 0
    checked_write_image(target_path, target)
    for split in ("train", "dev", "eval"):
        write_rows(source/f"seg_{split}.jsonl", [{"id": split, "image": str(image), "target": str(target_path)}])
    monkeypatch.setattr(sys, "argv", ["labels", "--data-root", str(source), "--output-root", str(output), "--policy", "positive-only"])
    script("make_shadow_label_ablation").main()
    for split in ("dev", "eval"):
        assert read_rows(output/f"seg_{split}.jsonl")[0]["target"] == str(target_path)
    train = read_rows(output/"seg_train.jsonl")[0]
    assert train["original_target_sha256"] == digest(target_path)
