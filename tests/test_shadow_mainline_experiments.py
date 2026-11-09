import csv
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.utils.io import write_json
from tests.test_bev_flow_ablations import script
from tests.test_shadow_video_review import frozen_runs


def data_fixture(tmp_path, with_negatives=True):
    source = tmp_path/"source"
    source.mkdir()
    before, after, pairs = frozen_runs(source)
    runs, negatives = [], []
    for split in ("train", "dev", "eval"):
        scene = split+"-scene"
        root = tmp_path/split
        root.mkdir()
        pairs_rows = read_rows(pairs)
        for pair in pairs_rows:
            pair.update(scene_name=scene, official_split="train" if split != "eval" else "val")
            for frame in pair["frames"]:
                frame["sample_data_token"] = split+"_"+frame["sample_data_token"]
        pair_path = root/"pairs.jsonl"
        write_rows(pair_path, pairs_rows)
        for method, original in (("appearance", before), ("anchor", after)):
            destination = root/method
            shutil.copytree(original, destination)
            protocol_path = destination/"clip/protocol.json"
            protocol = json.loads(protocol_path.read_text())
            protocol["clip"]["scene_name"] = scene
            write_json(protocol_path, protocol)
            rows = read_rows(destination/"clip/frames.jsonl")
            for slot, row in enumerate(rows):
                row["frame_token"] = split+"_"+row["frame_token"]
                path = destination/"clip"/f"{slot:03d}.npz"
                with np.load(path) as archive:
                    payload = {key: archive[key] for key in archive.files}
                payload["proposal_score_map"] = np.full((8, 8), .4, np.float32)
                np.savez_compressed(path, **payload)
            write_rows(destination/"clip/frames.jsonl", rows)
            runs.append({"method": method, "split": split, "run_root": str(destination), "pairs": [str(pair_path)]})
        if with_negatives:
            target = np.full((80, 80), 255, np.uint8)
            target[8:16, 8:16] = 0
            target_path = root/"lit.png"
            checked_write_image(target_path, target)
            # Canonical legacy teacher rows identify camera observations by image path.
            negatives.append({"scene_name": scene, "camera": "CAM_FRONT", "image": pairs_rows[0]["frames"][1]["image"],
                "shadow_mask": str(target_path), "mask_encoding": "trimap", "label_source": "temporal_weak",
                "physics_export_enabled": True, "physics_protocol": "bev-flow-annotation-v2"})
    runs_path, neg_path = tmp_path/"runs.jsonl", tmp_path/"negatives.jsonl"
    write_rows(runs_path, runs)
    write_rows(neg_path, negatives)
    return runs_path, neg_path


def test_export_bridges_frozen_frames_and_keeps_ignore_and_explicit_negatives(tmp_path):
    runs, negatives = data_fixture(tmp_path)
    module = script("build_shadow_mainline_experiments")
    output = tmp_path/"data"
    result = module.build_dataset(runs, output, [negatives], (80, 80))
    assert result["methods"]["anchor"]["train"]["both_classes_survive_resize"]
    rows = read_rows(output/"anchor/seg_train.jsonl")
    assert len(rows) == 3 and all(row["label_source"] == "temporal_weak" for row in rows)
    target = cv2.imread(rows[1]["target"], 0)
    assert set(np.unique(target)) == {0, 1, 255}
    assert (target[8:16, 8:16] == 0).all()
    assert target[0, 0] == 255
    predictions = read_rows(output/"anchor/discovery_train.jsonl")
    assert predictions[1]["probability"] is not None
    assert not result["training_performed"] and not result["unlabeled_is_background"]


def test_positive_only_export_is_valid_but_cannot_be_passed_as_two_class_training(tmp_path):
    runs, _ = data_fixture(tmp_path, False)
    module = script("build_shadow_mainline_experiments")
    result = module.build_dataset(runs, tmp_path/"data", size=(80, 80))
    assert not result["methods"]["anchor"]["train"]["both_classes_survive_resize"]
    with pytest.raises(ValueError, match="lacks reliable positive or negative"):
        module.prepare_suite(tmp_path/"data", ["appearance", "anchor"], tmp_path/"suite", (80, 80))


def test_export_does_not_train_on_manual_negatives_or_official_val(tmp_path):
    runs, negatives = data_fixture(tmp_path)
    module = script("build_shadow_mainline_experiments")
    rows = read_rows(negatives)
    rows[0]["label_source"] = "manual_shadow_gt"
    write_rows(negatives, rows)
    with pytest.raises(ValueError, match="never human test GT"):
        module.build_dataset(runs, tmp_path/"data", [negatives])
    rows = read_rows(runs)
    rows[-1]["split"] = "train"
    write_rows(runs, rows)
    with pytest.raises(ValueError, match="Official validation"):
        module.build_dataset(runs, tmp_path/"data2")


def test_suite_aligns_same_images_budget_and_scene_sets(tmp_path):
    runs, negatives = data_fixture(tmp_path)
    module = script("build_shadow_mainline_experiments")
    module.build_dataset(runs, tmp_path/"data", [negatives], (80, 80))
    plan = module.prepare_suite(tmp_path/"data", ["appearance", "anchor"], tmp_path/"suite", (80, 80), epochs=2)
    assert plan["aligned_frames"] == {"train": 1, "dev": 1}  # Other synthetic frames are all IGNORE.
    assert plan["epochs"] == 2 and plan["weak_weight"] == 1
    assert plan["temporal_weight"] == plan["receiver_pair_weight"] == 0
    assert plan["scene_sets"] == {"train": ["train-scene"], "dev": ["dev-scene"]}
    ids = [{row["id"] for row in read_rows(arm["train"])} for arm in plan["arms"]]
    assert ids[0] == ids[1]
    runner = script("run_shadow_segmenter_suite")
    # Real model training is exercised on HPC, not faked as a local result.
    assert runner is not None


def evaluation_fixture(tmp_path):
    truth = np.zeros((24, 24), np.uint8)
    truth[8:16, 8:16] = 1
    truth[:2] = 255
    target = tmp_path/"manual.png"
    checked_write_image(target, truth)
    gt = tmp_path/"gt.jsonl"
    row = {"id": "g", "scene_name": "heldout", "camera": "CAM_FRONT", "frame_token": "cam",
        "target": str(target), "image": str(target), "mask_encoding": "trimap", "label_source": "manual_shadow_gt",
        "independent": True, "split": "eval"}
    write_rows(gt, [row])
    probability = np.where(truth == 1, .9, .1).astype(np.float32)
    path = tmp_path/"prediction.npz"
    np.savez_compressed(path, probability=probability)
    predictions = tmp_path/"predictions.jsonl"
    write_rows(predictions, [{**{key: row[key] for key in ("scene_name", "camera", "frame_token")}, "probability": str(path), "threshold": .5}])
    methods = tmp_path/"methods.jsonl"
    write_rows(methods, [{"method": "anchor", "experiment": "E1", "predictions": str(predictions)}])
    summary = tmp_path/"dataset_summary.json"
    write_json(summary, {"scene_roles": {"training": "train", "development": "dev", "heldout": "eval"}})
    return gt, methods, summary, predictions


def test_independent_gt_metrics_and_paper_table_are_real_pixel_results(tmp_path):
    gt, methods, summary, _ = evaluation_fixture(tmp_path)
    module = script("evaluate_shadow_mainline_experiments")
    result = module.evaluate_experiments(gt, methods, summary, tmp_path/"metrics")
    assert result["tables"] == ["E1"]
    metric = json.loads((tmp_path/"metrics/metrics.json").read_text())["results"][0]
    assert metric["tp"] == 64 and metric["fn"] == metric["fp"] == 0
    assert metric["precision"] == metric["recall"] == metric["f1"] == metric["iou"] == 1
    assert metric["boundary"]["f1"] == 1 and metric["average_precision_binned"] == 1
    with (tmp_path/"metrics/table_E1.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "anchor" and float(rows[0]["iou"]) == 1


def test_metrics_cannot_use_weak_gt_missing_predictions_or_training_scenes(tmp_path):
    gt, methods, summary, predictions = evaluation_fixture(tmp_path)
    module = script("evaluate_shadow_mainline_experiments")
    original = read_rows(gt)
    write_rows(gt, [{**original[0], "label_source": "temporal_weak"}])
    with pytest.raises(ValueError, match="independently reviewed"):
        module.evaluate_experiments(gt, methods, summary, tmp_path/"metrics")
    write_rows(gt, original)
    write_json(summary, {"scene_roles": {"heldout": "dev"}})
    with pytest.raises(ValueError, match="overlap"):
        module.evaluate_experiments(gt, methods, summary, tmp_path/"metrics")
    write_json(summary, {"scene_roles": {"heldout": "eval"}})
    write_rows(predictions, [])
    with pytest.raises(ValueError, match="Every method"):
        module.evaluate_experiments(gt, methods, summary, tmp_path/"metrics")


def test_saved_discovery_mask_not_thresholded_ranking_score_defines_operating_point(tmp_path):
    gt, methods, summary, predictions = evaluation_fixture(tmp_path)
    row = read_rows(predictions)[0]
    probability = np.full((24, 24), .9, np.float32)
    path = tmp_path/"accepted.npz"
    mask = np.zeros((24, 24), bool)
    mask[8:12, 8:16] = True
    np.savez_compressed(path, probability=probability, accepted=mask)
    row.update(probability=str(path), prediction={"path": str(path), "key": "accepted"})
    write_rows(predictions, [row])
    script("evaluate_shadow_mainline_experiments").evaluate_experiments(gt, methods, summary, tmp_path/"metrics")
    metric = json.loads((tmp_path/"metrics/metrics.json").read_text())["results"][0]
    assert metric["tp"] == 32 and metric["fp"] == 0 and metric["fn"] == 32
    assert metric["recall"] == .5


def test_no_ranking_score_does_not_fabricate_pr_auc(tmp_path):
    gt, methods, summary, predictions = evaluation_fixture(tmp_path)
    mask = np.zeros((24, 24), np.uint8)
    path = tmp_path/"mask.png"
    checked_write_image(path, mask)
    row = read_rows(predictions)[0]
    row.update(probability=None, prediction=str(path))
    write_rows(predictions, [row])
    script("evaluate_shadow_mainline_experiments").evaluate_experiments(gt, methods, summary, tmp_path/"metrics")
    metric = json.loads((tmp_path/"metrics/metrics.json").read_text())["results"][0]
    assert metric["average_precision_binned"] is None and metric["recall"] == 0


def test_boundary_metric_excludes_unknown_surroundings():
    module = script("evaluate_shadow_mainline_experiments")
    truth = np.full((24, 24), 255, np.uint8)
    truth[6:18, 6:18] = 1
    prediction = truth == 1
    counts = module.boundary_counts(prediction, truth, 2)
    assert counts.sum() == 0  # No known non-shadow side of that boundary.


def test_suite_driver_reuses_shared_settings_and_does_not_evaluate_collapse(tmp_path, monkeypatch):
    runs, negatives = data_fixture(tmp_path)
    builder = script("build_shadow_mainline_experiments")
    builder.build_dataset(runs, tmp_path/"data", [negatives], (80, 80))
    builder.prepare_suite(tmp_path/"data", ["appearance", "anchor"], tmp_path/"suite", (80, 80), epochs=1)
    runner = script("run_shadow_segmenter_suite")
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    status = runner.run_suite(tmp_path/"suite/suite.json", tmp_path/"training", device="cpu")
    assert len(calls) == 2 and not status["all_arms_completed"]
    assert all(row["status"] == "training_failed_or_collapsed" for row in status["arms"])
    assert all("--temporal-weight" in call and "--seed" in call for call in calls)
    assert read_rows(tmp_path/"training/evaluation_methods.jsonl") == []


def test_overlapping_clip_observations_are_deduplicated_per_camera_frame(tmp_path):
    runs, _ = data_fixture(tmp_path, False)
    rows = read_rows(runs)
    rows.append(rows[1].copy())
    write_rows(runs, rows)
    script("build_shadow_mainline_experiments").build_dataset(runs, tmp_path/"data", size=(80, 80))
    frames = read_rows(tmp_path/"data/anchor/seg_train.jsonl")
    assert len(frames) == 3 and len(frames[1]["sources"]) == 2
    assert not (tmp_path/"data/.merge").exists()


def test_conflicting_physical_negative_becomes_ignore_not_a_forced_answer(tmp_path):
    runs, negatives = data_fixture(tmp_path)
    source_path = tmp_path/"train/anchor/clip/001_image.npz"
    with np.load(source_path) as source:
        positive = source["shadow_target_image"] == 1
    row = read_rows(negatives)[0]
    path = Path(row["shadow_mask"])
    negative = cv2.imread(str(path), 0)
    negative[positive] = 0
    checked_write_image(path, negative)
    result = script("build_shadow_mainline_experiments").build_dataset(runs, tmp_path/"data", [negatives], (80, 80))
    row = read_rows(tmp_path/"data/anchor/seg_train.jsonl")[1]
    target = cv2.imread(row["target"], 0)
    assert (target[positive] == 255).all()
    assert result["methods"]["anchor"]["train"]["conflict_pixels"] == int(positive.sum())


def test_unknown_coordinates_are_not_resized_to_make_evaluation_pass(tmp_path):
    gt, methods, summary, predictions = evaluation_fixture(tmp_path)
    path = tmp_path/"wrong.npz"
    np.savez_compressed(path, probability=np.ones((12, 12), np.float32))
    row = read_rows(predictions)[0]
    row["probability"] = str(path)
    write_rows(predictions, [row])
    with pytest.raises(ValueError, match="coordinate mismatch"):
        script("evaluate_shadow_mainline_experiments").evaluate_experiments(gt, methods, summary, tmp_path/"metrics")


def test_development_scene_exclusions_propagate_to_independent_evaluation(tmp_path):
    gt, methods, summary, _ = evaluation_fixture(tmp_path)
    write_json(summary, {"scene_roles": {"heldout": "eval"}, "development_scenes": ["heldout"]})
    with pytest.raises(ValueError, match="development scenes"):
        script("evaluate_shadow_mainline_experiments").evaluate_experiments(gt, methods, summary, tmp_path/"metrics")


def test_e2_success_path_launches_frozen_gt_inference_then_table_evaluation(tmp_path, monkeypatch):
    runs, negatives = data_fixture(tmp_path)
    builder = script("build_shadow_mainline_experiments")
    builder.build_dataset(runs, tmp_path/"data", [negatives], (80, 80))
    builder.prepare_suite(tmp_path/"data", ["appearance", "anchor"], tmp_path/"suite", (80, 80), epochs=1)
    gt, _, _, _ = evaluation_fixture(tmp_path)
    runner = script("run_shadow_segmenter_suite")
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1].endswith("train_shadow_segmenter.py"):
            assert "--manifest" not in command and str(gt) not in command
            folder = Path(command[command.index("--output-root")+1])
            folder.mkdir(parents=True)
            (folder/"best.pth").write_bytes(b"synthetic checkpoint placeholder")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    status = runner.run_suite(tmp_path/"suite/suite.json", tmp_path/"training", gt, device="cpu")
    assert status["all_arms_completed"] and status["metrics_exit_code"] == 0
    assert len(calls) == 5  # Two training, two inference, one aggregate table job.
    assert calls[-1][1].endswith("evaluate_shadow_mainline_experiments.py")
    assert len(read_rows(tmp_path/"training/evaluation_methods.jsonl")) == 2
