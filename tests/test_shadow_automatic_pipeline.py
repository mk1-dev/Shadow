import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.physics_io import load_motion_analysis, read_evidence
from shadow_occ.annotation.video_review import find_clips, load_clip
from shadow_occ.utils.io import write_json
from tests.test_bev_flow_ablations import script
from tests.test_shadow_video_review import frozen_runs
from tests.test_shadow_mainline_experiments import data_fixture, evaluation_fixture
from shadow_occ.annotation.scene_sampling import scene_epoch_indices, scene_probabilities


def catalogue_fixture(tmp_path):
    source = tmp_path/"prepared"
    source.mkdir()
    pairs, metrics = [], []
    for index in range(10):
        scene = f"scene-{index:04d}"
        official = "train" if index < 8 else "val"
        for step in range(4):
            identifier = scene+f"_{step}"
            row = {"id": identifier, "scene_name": scene, "camera": "CAM_FRONT", "official_split": official,
                "sample_token": scene+f"_s{step}", "next_sample_token": scene+f"_s{step+1}",
                "objects": [{"instance_token": scene+"_vehicle", "category": "vehicle.car"}],
                "frames": [{"sample_data_token": scene+f"_cam{at}", "camera_timestamp": at*500000,
                            "image": f"{scene}_{at}.png", "ground_support": f"{scene}_ground_{at}.png"} for at in (step, step+1)]}
            pairs.append(row)
            evidence = source/(identifier+".npz")
            np.savez_compressed(evidence, receiver_flow_supported=np.ones((2, 2), bool))
            metrics.append({"id": identifier, "scene_name": scene, "camera": "CAM_FRONT", "official_split": official,
                            "evidence": evidence.name, "evidence_sha256": digest(evidence)})
    pair_path = source/"pairs.jsonl"
    write_rows(pair_path, pairs)
    write_rows(source/"pair_metrics.jsonl", metrics)
    write_json(source/"analysis_protocol.json", {"flow_evidence_version": "receiver-support-v2", "manifest_sha256": digest(pair_path)})
    catalogue = tmp_path/"catalogue.jsonl"
    write_rows(catalogue, [{"pairs": str(pair_path), "analysis_root": str(source), "feature_root": str(source), "spatial_feature_root": str(source)}])
    lock = tmp_path/"lock.json"
    write_json(lock, {"excluded_development_scenes": ["scene-0008"]})
    return catalogue, lock, source


def test_scene_plan_is_nested_incremental_automatic_and_split_isolated(tmp_path):
    catalogue, lock, source = catalogue_fixture(tmp_path)
    original = {path: digest(path) for path in source.iterdir()}
    output = tmp_path/"plan"
    result = script("plan_automatic_shadow_scenes").plan_scenes(catalogue, lock, output, tmp_path/"runs", expected_train_scenes=8)
    sets = {key: set(value) for key, value in result["train_scene_stages"].items()}
    assert sets["25"] <= sets["50"] <= sets["100"]
    assert sets["100"].isdisjoint(result["dev_scenes"])
    batches = [read_rows(output/stage/"batches.jsonl") for stage in ("stage_025", "stage_050", "stage_100")]
    assert len(set(row["id"] for group in batches for row in group)) == sum(map(len, batches))
    shared = read_rows(output/"shared/batches.jsonl")
    assert {row["split"] for row in shared} == {"dev", "eval"}
    manifests = read_rows(output/"stage_100/experiment_runs.jsonl")
    assert all("scene-0008" != clip["scene_name"] for row in shared if row["split"] == "eval" for clip in read_rows(row["clips"]))
    assert all("shared" in row["run_root"] for row in manifests if row["split"] != "train")
    assert all(digest(path) == sha for path, sha in original.items())


def test_scene_sharding_preserves_flow_payload_and_exact_subset_protocol(tmp_path):
    catalogue, lock, _ = catalogue_fixture(tmp_path)
    output = tmp_path/"plan"
    script("plan_automatic_shadow_scenes").plan_scenes(catalogue, lock, output, tmp_path/"runs")
    batch = read_rows(output/"stage_025/batches.jsonl")[0]
    pairs, stats, _ = load_motion_analysis(batch["pairs"], batch["analysis_root"])
    assert len({row["scene_name"] for row in pairs}) == 1
    evidence, _ = read_evidence(pairs[0], stats, batch["analysis_root"])
    assert evidence["receiver_flow_supported"].all()
    assert Path(pairs[0]["frames"][0]["image"]).is_absolute()
    assert read_rows(batch["clips"])[0]["selection"].endswith("without_shadow_or_VLM_filter")


def test_small_cache_cannot_be_called_complete_nuscenes_training_population(tmp_path):
    catalogue, lock, _ = catalogue_fixture(tmp_path)
    with pytest.raises(ValueError, match="expected 700"):
        script("plan_automatic_shadow_scenes").plan_scenes(catalogue, lock, tmp_path/"plan", tmp_path/"runs", expected_train_scenes=700)


def test_no_gt_track_cannot_be_fabricated_for_automatic_clip_selection():
    rows = [{"id": str(at), "scene_name": "scene", "camera": "CAM_FRONT", "sample_token": str(at),
             "next_sample_token": str(at+1), "frames": [{"camera_timestamp": at}], "objects": []} for at in range(4)]
    clips, counts = script("plan_automatic_shadow_scenes").automatic_clips(rows)
    assert clips == [] and counts["without_three_frame_gt_track"] > 0


def test_present_pairs_without_clips_are_explicitly_unusable_not_complete(tmp_path):
    catalogue, lock, source = catalogue_fixture(tmp_path)
    rows = read_rows(source/"pairs.jsonl")
    for row in rows:
        row["objects"] = []
    write_rows(source/"pairs.jsonl", rows)
    write_json(source/"analysis_protocol.json", {"flow_evidence_version": "receiver-support-v2", "manifest_sha256": digest(source/"pairs.jsonl")})
    module = script("plan_automatic_shadow_scenes")
    result = module.plan_scenes(catalogue, lock, tmp_path/"plan", tmp_path/"runs")
    assert not result["stage_coverage"]["shared"]["missing_cache_scenes"]
    assert result["stage_coverage"]["shared"]["no_valid_clip_scenes"]
    assert not result["stage_coverage"]["shared"]["complete_clip_coverage"]
    with pytest.raises(ValueError, match="usable dev AND eval"):
        module.check_plan(tmp_path/"plan", "shared", lock)


def test_plan_detects_changed_catalogue_instead_of_reusing_stale_batches(tmp_path):
    catalogue, lock, _ = catalogue_fixture(tmp_path)
    module = script("plan_automatic_shadow_scenes")
    module.plan_scenes(catalogue, lock, tmp_path/"plan", tmp_path/"runs")
    assert module.check_plan(tmp_path/"plan", "stage_025", lock)["complete_clip_coverage"]
    catalogue.write_text(catalogue.read_text()+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generate a new plan"):
        module.check_plan(tmp_path/"plan", "stage_025", lock)


def test_alignment_conflict_preserves_reference_arm_without_fabricating_labels(tmp_path):
    runs, negatives = data_fixture(tmp_path)
    module = script("build_shadow_mainline_experiments")
    module.build_dataset(runs, tmp_path/"data", [negatives], (80, 80))
    for split in ("train", "dev"):
        for method, labels in (("appearance", [1, 0, None]), ("anchor", [None, 1, 0])):
            for row, label in zip(read_rows(tmp_path/"data"/method/f"seg_{split}.jsonl"), labels):
                target = np.full((80, 80), 255, np.uint8)
                if label is not None:
                    target[8:16, 8:16] = label
                checked_write_image(Path(row["target"]), target)
    result = module.prepare_suite(tmp_path/"data", ["appearance", "anchor"], tmp_path/"suite", (80, 80),
        reference_method="anchor", skip_untrainable=True)
    assert [arm["method"] for arm in result["arms"]] == ["anchor"]
    assert result["alignment_dropped_methods"] == ["appearance"]
    assert result["aligned_frames"]["train"] == 2
    assert result["method_availability"][0]["status"] == "alignment_incompatible"


def lit_fixture(tmp_path):
    _, after, pairs = frozen_runs(tmp_path)
    for slot in range(3):
        path = after/"clip"/f"{slot:03d}.npz"
        with np.load(path) as data:
            values = {key: data[key] for key in data.files}
        values.update(receiver_valid_by_frame=np.ones((8, 8), bool), bev_feature_valid=np.ones((8, 8), bool))
        np.savez_compressed(path, **values)
    np.savez_compressed(after/"clip/pair_evidence.npz", ground_motion_reference_valid_map=np.ones((2, 8, 8), bool))
    ground = tmp_path/"ground.png"
    checked_write_image(ground, np.ones((80, 80), np.uint8))
    pair_rows = read_rows(pairs)
    corners = [[x, y, z] for x in (2, 3) for y in (2, 3) for z in (0, 1)]
    for row in pair_rows:
        row["official_split"] = "train"
        for frame in row["frames"]:
            frame["ground_support"] = str(ground)
            frame["object_volumes_world"] = [{"instance_token": "A", "corners_world": corners}]
    write_rows(pairs, pair_rows)
    protocol_path = after/"clip/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["grid"] = {"resolution": 1.0}
    write_json(protocol_path, protocol)
    module = script("build_shadow_mainline_experiments")
    cameras = module.camera_index([pairs])
    clip = load_clip(next(iter(find_clips(after).values())))
    frames = [cameras[("scene", "CAM_FRONT", f"cam{slot}")] for slot in range(3)]
    return module, clip, frames, after, pairs


def test_automatic_lit_negative_rule_uses_real_observation_support(tmp_path):
    module, clip, frames, _, _ = lit_fixture(tmp_path)
    masks, details = module.automatic_lit_negatives(clip, frames)
    assert masks[1].sum() > 0
    assert not (masks[1] & clip["images"][1]["shadow_proposal_image"]).any()
    assert details["frames"]["1"]["source"] == "existing_lit_receiver_negative_mask"
    clip["pair"]["ground_motion_reference_valid_map"][:] = False
    missing, _ = module.automatic_lit_negatives(clip, frames)
    assert not missing[1].any()  # Missing correspondence never turns into a known negative.


def test_full_export_creates_two_class_weak_data_and_reusable_negatives_without_manual_input(tmp_path):
    module, clip, _, after, pairs = lit_fixture(tmp_path)
    runs = tmp_path/"runs.jsonl"
    write_rows(runs, [{"method": "anchor_first_v8plus", "split": "train", "run_root": str(after), "pairs": [str(pairs)]}])
    original = {path: digest(path) for path in after.rglob("*") if path.is_file()}
    result = module.build_dataset(runs, tmp_path/"data", size=(80, 80), auto_negatives=True)
    assert result["methods"]["anchor_first_v8plus"]["train"]["both_classes_survive_resize"]
    rows = read_rows(tmp_path/"data/anchor_first_v8plus/seg_train.jsonl")
    target = cv2.imread(rows[1]["target"], 0)
    assert set(np.unique(target)) == {0, 1, 255}
    assert (target == 1).sum() == (clip["images"][1]["shadow_target_image"] == 1).sum()
    pool = tmp_path/"data/anchor_first_v8plus/automatic_negatives.jsonl"
    assert len(read_rows(pool)) == 1
    assert not (tmp_path/"data/manual_eval_template.jsonl").exists()
    assert all(digest(path) == sha for path, sha in original.items())
    module.build_dataset(runs, tmp_path/"common_pool_data", [pool], (80, 80))
    copied = read_rows(tmp_path/"common_pool_data/anchor_first_v8plus/seg_train.jsonl")
    assert np.array_equal(cv2.imread(copied[1]["target"], 0), target)


def test_weak_teacher_metrics_are_separately_named_not_shadow_ground_truth(tmp_path):
    gt, methods, summary, _ = evaluation_fixture(tmp_path)
    rows = read_rows(gt)
    rows[0].update(label_source="temporal_weak", independent=False)
    write_rows(gt, rows)
    training = tmp_path/"training_protocol.json"
    write_json(training, {"train_scenes": ["training"], "dev_scenes": ["development"]})
    rows = read_rows(methods)
    rows[0].update(experiment="TeacherAgreement", training_protocol=str(training))
    write_rows(methods, rows)
    module = script("evaluate_shadow_mainline_experiments")
    module.evaluate_experiments(gt, methods, summary, tmp_path/"metrics", reference_kind="frozen-teacher")
    result = json.loads((tmp_path/"metrics/metrics.json").read_text())
    assert not result["is_true_shadow_accuracy"]
    assert not (tmp_path/"metrics/table_E2.csv").exists()
    with (tmp_path/"metrics/table_teacher_agreement.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert "teacher_agreement_iou" in row and "iou" not in row
    assert float(row["teacher_agreement_iou"]) == 1.0
    rows = read_rows(gt)
    rows[0]["label_source"] = "model_pseudo"
    write_rows(gt, rows)
    with pytest.raises(ValueError, match="not model self-predictions"):
        module.evaluate_experiments(gt, methods, summary, tmp_path/"bad_metrics", reference_kind="frozen-teacher")


def test_without_manual_gt_suite_still_runs_heldout_teacher_inference(tmp_path, monkeypatch):
    runs, negatives = data_fixture(tmp_path)
    builder = script("build_shadow_mainline_experiments")
    builder.build_dataset(runs, tmp_path/"data", [negatives], (80, 80))
    builder.prepare_suite(tmp_path/"data", ["appearance", "anchor"], tmp_path/"suite", (80, 80), epochs=1)
    runner = script("run_shadow_segmenter_suite")
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1].endswith("train_shadow_segmenter.py"):
            folder = Path(command[command.index("--output-root")+1])
            folder.mkdir(parents=True)
            (folder/"best.pth").write_bytes(b"unit test only")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    status = runner.run_suite(tmp_path/"suite/suite.json", tmp_path/"training", device="cpu")
    assert len(calls) == 5 and status["all_arms_completed"]
    assert status["evaluation_reference_kind"] == "frozen-teacher" and not status["is_true_shadow_accuracy"]
    assert calls[-1][-2:] == ["--reference-kind", "frozen-teacher"]
    methods = read_rows(tmp_path/"training/evaluation_methods.jsonl")
    assert all(row["experiment"] == "TeacherAgreement" for row in methods)


def test_scene_sampling_is_class_independent_balanced_and_distributed():
    rows = [{"scene_name": "large", "target": "ignored"} for _ in range(90)]+[{"scene_name": "small"} for _ in range(10)]
    probabilities = scene_probabilities(rows, "uniform")
    assert probabilities[:90].sum() == pytest.approx(.5)
    assert probabilities[90:].sum() == pytest.approx(.5)
    shards = [scene_epoch_indices(rows, 3, 7, rank, 3) for rank in range(3)]
    assert {len(shard) for shard in shards} == {34}
    assert shards[0] == scene_epoch_indices(rows, 3, 7, 0, 3)
    assert shards[0] != scene_epoch_indices(rows, 3, 8, 0, 3)
    assert np.allclose(scene_probabilities(rows), .01)


def test_export_writes_resized_per_image_scene_and_delivery_artifacts(tmp_path):
    runs, negatives = data_fixture(tmp_path)
    builder = script("build_shadow_mainline_experiments")
    root = tmp_path/"data"
    builder.build_dataset(runs, root, [negatives], (80, 80))
    row = read_rows(root/"anchor/seg_train.jsonl")[1]
    assert sum(row["resized_class_pixels"].values()) == 80*80
    assert row["negative_weight"] == .5
    assert row["resized_positive_image_fraction"] == row["resized_class_pixels"]["positive"]/6400
    for name in ["annotation_summary.json", "dataset_manifest.json", "class_distribution.json"]:
        assert (root/name).is_file()
    builder.prepare_suite(root, ["absent_baseline", "anchor"], tmp_path/"suite", (80, 80), skip_untrainable=True)
    availability = json.loads((tmp_path/"suite/method_availability.json").read_text())
    assert availability[0]["status"] == "untrainable"
    plan = json.loads((tmp_path/"suite/suite.json").read_text())
    assert plan["scene_sampling"] == "uniform" and len(plan["arms"]) == 1


def test_full_population_split_does_not_change_when_only_a_cache_subset_is_present(tmp_path):
    catalogue, lock, _ = catalogue_fixture(tmp_path)
    population = tmp_path/"population.json"
    write_json(population, {"official_train_scenes": [f"scene-{index:04d}" for index in range(8)]+[f"extra-{index}" for index in range(20)],
        "official_eval_scenes": ["scene-0008", "scene-0009"]})
    module = script("plan_automatic_shadow_scenes")
    result = module.plan_scenes(catalogue, lock, tmp_path/"plan", tmp_path/"runs",
        expected_train_scenes=28, population_path=population, methods=("appearance_only", "anchor_first_v8plus"))
    dev, stages = module.scene_partition(json.loads(population.read_text())["official_train_scenes"])
    assert result["dev_scenes"] == sorted(dev)
    assert result["train_scene_stages"]["25"] == sorted(stages[25])
    assert result["missing_stage_scenes"]["25"]
    batches = [row for stage in ("shared", "stage_025", "stage_050", "stage_100") for row in read_rows(tmp_path/"plan"/stage/"batches.jsonl")]
    assert {row["discovery_mode"] for row in batches} == {"appearance", "cross-space"}


def test_storage_estimate_uses_real_file_sizes_and_marks_missing_measurements(tmp_path):
    catalogue, lock, _ = catalogue_fixture(tmp_path)
    population = {"official_train_scenes": [f"scene-{index:04d}" for index in range(8)],
        "official_eval_scenes": ["scene-0008", "scene-0009"]}
    # The synthetic catalogue does not contain native BEV NPZs: no invented size.
    result = script("estimate_shadow_pipeline_storage").estimate(population, catalogue, lock,
        tmp_path/"estimate", tmp_path)
    assert result["selected_train_scenes"] == 2 and not result["jobs_submitted"]
    assert not result["estimated_total_is_complete"]
    assert result["components"]["current_bev"]["typical_bytes"] is None
