import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.diagnostics import SCORING_VERSION, example_weight
from shadow_occ.annotation.joint import annotation
from shadow_occ.annotation.self_training import make_policy, pseudo_annotation, teacher_gate, temporal_agreement, validate_policy
from shadow_occ.annotation.teacher_provenance import extend_scene_history, input_provenance


ROOT = Path(__file__).resolve().parents[1]


def dev_rows():
    return [{"id": f"dev-{i}", "scene_name": f"dev-{'a' if i % 2 == 0 else 'b'}", "split": "dev", "official_split": "train",
             "annotation_target": annotation("YES", "unresolved") if i < 2 else annotation("NO", "none"),
             "shadow_label_source": "geometry_weak", "label_source": "geometry_weak", "scorer_id": "teacher",
             "score": 0.99 if i < 2 else 0.01, "scoring_version": SCORING_VERSION, "in_sample_annotation": False} for i in range(4)]


def calibration():
    return {"fitted_split": "dev", "scoring_version": SCORING_VERSION, "scorer_id": "teacher", "threshold": 0.5,
            "abstain_margin": 0.0, "max_uncertainty": 1.0}


def pilot_policy():
    return make_policy(calibration(), dev_rows(), minimum_positives=2, minimum_scenes=2)


def test_small_dev_pool_cannot_approve_default_self_training():
    policy = make_policy(calibration(), dev_rows())
    assert not policy["eligible_for_pilot"]
    assert "too_few_accepted_dev_positives" in policy["blocking_reasons"]
    assert not policy["approved_for_production"]


def test_pilot_quality_is_explicitly_weak_not_real_shadow_precision():
    policy = pilot_policy()
    assert policy["eligible_for_pilot"]
    assert policy["supervision_scope"] == "weak"
    assert policy["teacher_gate_requirements"]["minimum_positives"] == 2
    validate_policy(policy)
    with pytest.raises(ValueError, match="temporal"):
        validate_policy({**policy, "require_temporal_consistency": False})


def test_policy_fitting_rejects_eval_and_wrong_teacher():
    rows = dev_rows()
    rows[0]["official_split"] = "val"
    with pytest.raises(ValueError, match="dev"):
        make_policy(calibration(), rows)
    rows = dev_rows()
    rows[0]["scorer_id"] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        make_policy(calibration(), rows)


def test_independent_generation_must_agree_with_scores_and_content():
    score = {"split": "train", "official_split": "train", "scorer_id": "teacher", "score": 0.99, "in_sample_annotation": False,
             "scoring_version": SCORING_VERSION, "input_content_sha256": "pixels", "region_mask_sha256": "mask", "annotation_target": None, "answer": None}
    generated = {**score, "prediction": annotation("YES")}
    assert teacher_gate(score, generated, pilot_policy()) == (None, "YES")
    assert teacher_gate(score, {**generated, "input_content_sha256": "other"}, pilot_policy())[0] == "generation_score_input_mismatch"
    assert teacher_gate(score, {**generated, "prediction": annotation("NO", "none")}, pilot_policy())[0] == "low_score_or_generation_disagreement"
    assert teacher_gate({**score, "split": "eval"}, generated, pilot_policy())[0] == "not_training_pool"


def test_static_unchanging_shadow_is_kept_but_moving_texture_is_not():
    policy = pilot_policy()
    region = np.ones((16, 16), bool)
    probability = np.full(region.shape, 0.95, np.float32)
    temporal = {"lighting_comparable": True, "mode": "static", "world_previous_to_current": np.eye(3), "caster_previous_to_current": np.eye(3)}
    assert temporal_agreement(region, probability, probability, region, region, temporal, "YES", policy)["accepted"]
    moving = temporal_agreement(region, probability, probability, region, region, {**temporal, "mode": "moving"}, "YES", policy)
    assert not moving["accepted"]
    assert moving["motion_advantage"] == 0


def test_unseen_outside_state_is_not_invented_by_self_training():
    prediction = annotation("YES", "outside_bev", candidate_id="H1")
    row = {"association_source": "geometry_weak", "valid_candidate_ids": ["H1"]}
    target, enabled, _ = pseudo_annotation(row, prediction, "YES", outside_support=0)
    assert target["caster"]["state"] == "unresolved"
    assert not enabled


def test_pseudo_loss_is_weaker_and_cannot_override_original_weight():
    original = {"label_source": "geometry_weak"}
    assert example_weight(original, 0.25) == 0.25
    assert example_weight({**original, "self_training": {"round": 1}, "pseudo_loss_weight": 0.25}, 0.25) == 0.0625
    with pytest.raises(ValueError, match="Pseudo-label"):
        example_weight({**original, "self_training": {"round": 1}, "pseudo_loss_weight": 2.0}, 0.25)


def test_initial_checkpoint_history_cannot_launder_heldout_scenes():
    train, dev = extend_scene_history({"fold-train"}, {"dev"}, {"train_scenes": ["heldout-pool"], "dev_scenes": ["dev"]})
    assert "heldout-pool" in train
    assert dev == {"dev"}
    with pytest.raises(ValueError, match="boundary"):
        extend_scene_history({"fold-train"}, {"dev"}, {"train_scenes": ["dev"], "dev_scenes": []})


@pytest.fixture
def round_inputs(tmp_path):
    image_path, dev_image = tmp_path/"image.png", tmp_path/"dev.png"
    target_path, ground_path, mask_path = tmp_path/"target.png", tmp_path/"ground.png", tmp_path/"region.png"
    cv2.imwrite(str(image_path), np.full((16, 16, 3), 100, np.uint8))
    cv2.imwrite(str(dev_image), np.full((16, 16, 3), 160, np.uint8))
    target = np.zeros((16, 16), np.uint8)
    target[4:12, 4:12] = 255
    target[0:2, 0:2] = 1
    region = np.zeros(target.shape, np.uint8)
    region[4:12, 4:12] = 255
    cv2.imwrite(str(target_path), target)
    cv2.imwrite(str(ground_path), np.full(target.shape, 255, np.uint8))
    cv2.imwrite(str(mask_path), region)
    segments = []
    for identifier in ("previous", "current"):
        probability = np.full(target.shape, 0.05, np.float32)
        probability[4:12, 4:12] = 0.95
        path = tmp_path/f"{identifier}.npz"
        np.savez_compressed(path, probability=probability)
        segments.append({"id": identifier, "scene_name": "pool", "split": "train", "official_split": "train", "camera": "CAM_FRONT", "instance_token": "car-a",
                         "image": str(image_path), "target": str(target_path), "ground_support": str(ground_path), "probability": str(path), "label_source": "geometry_weak",
                         "probability_sha256": digest(path), "source_image_sha256": digest(image_path), "source_target_sha256": digest(target_path), "source_ground_sha256": digest(ground_path),
                         "segmenter_readiness": {"noncollapsed": True, "approved_for_relabeling": False}})
    temporal = {"previous_id": "previous", "lighting_comparable": True, "mode": "static", "world_previous_to_current": np.eye(3).tolist(), "caster_previous_to_current": np.eye(3).tolist()}
    score = {**segments[-1], "id": "candidate", "parent_id": "current", "task": "shadow_annotation", "images": [str(image_path)]*3, "region_mask": str(mask_path),
             "prompt": "Judge the region", "annotation_target": None, "answer": None, "scorer_id": "teacher", "scoring_version": SCORING_VERSION, "score": 0.99, "in_sample_annotation": False, "temporal": temporal}
    score.update(input_provenance(score, {}))
    generation = {"id": score["id"], "scorer_id": "teacher", "input_content_sha256": score["input_content_sha256"], "region_mask_sha256": score["region_mask_sha256"], "prediction": annotation("YES")}
    anchors = [{**score, "id": f"anchor-{i}", "annotation_target": annotation("YES") if i < 2 else annotation("NO", "none"),
                "answer": json.dumps(annotation("YES") if i < 2 else annotation("NO", "none")), "label_source": "geometry_weak"} for i in range(4)]
    dev = [{**row, "task": "shadow_annotation", "image": str(dev_image), "images": [str(dev_image)]*3, "answer": json.dumps(row["annotation_target"])} for row in dev_rows()]
    seg_dev = [{**segments[-1], "id": f"seg-dev-{i}", "scene_name": row["scene_name"], "image": str(dev_image), "split": "dev"} for i, row in enumerate(dev)]
    paths = {key: tmp_path/f"{key}.jsonl" for key in ("anchor_train", "dev", "seg_anchor_train", "seg_dev", "teacher_scores", "teacher_annotations", "seg_predictions")}
    for key, rows in (("anchor_train", anchors), ("dev", dev), ("seg_anchor_train", segments), ("seg_dev", seg_dev), ("teacher_scores", [score]), ("teacher_annotations", [generation]), ("seg_predictions", segments)):
        write_rows(paths[key], rows)
    protocols = {
        "score_protocol": {"scores_sha256": digest(paths["teacher_scores"]), "scorer_id": "teacher", "teacher_train_scenes": ["teacher-train"], "teacher_dev_scenes": ["dev-a", "dev-b"]},
        "annotation_protocol": {"annotations_sha256": digest(paths["teacher_annotations"]), "scorer_id": "teacher", "teacher_train_scenes": ["teacher-train"], "teacher_dev_scenes": ["dev-a", "dev-b"]},
        "seg_metrics": {"predictions_sha256": digest(paths["seg_predictions"]), "checkpoint_train_scenes": ["teacher-train"], "checkpoint_dev_scenes": ["dev-a", "dev-b"]},
        "policy": pilot_policy(),
    }
    for key, value in protocols.items():
        paths[key] = tmp_path/f"{key}.json"
        paths[key].write_text(json.dumps(value), encoding="utf-8")
    return paths


def build_round(paths, output, *extra):
    arguments = [sys.executable, str(ROOT/"scripts"/"build_shadow_self_training_round.py")]
    for key, path in paths.items():
        arguments += ["--"+key.replace("_", "-"), str(path)]
    return subprocess.run(arguments+["--output-root", str(output), *extra], cwd=ROOT, capture_output=True, text=True)


def test_build_retains_anchors_dev_and_marks_only_new_pixels(round_inputs, tmp_path):
    output = tmp_path/"student"
    result = build_round(round_inputs, output, "--mode", "build", "--allow-weak-pilot")
    assert result.returncode == 0, result.stderr
    annotations = read_rows(output/"annotation_train.jsonl")
    assert len(annotations) == 5
    assert annotations[-1]["label_source"] == "model_pseudo"
    assert not annotations[-1]["association_supervision_enabled"]
    assert read_rows(output/"annotation_dev.jsonl") == read_rows(round_inputs["dev"])
    segmentation = read_rows(output/"seg_train.jsonl")
    assert len(segmentation) == 3
    mask = cv2.imread(segmentation[-1]["target"], cv2.IMREAD_GRAYSCALE)
    assert np.count_nonzero(mask == 1) == 64
    assert np.count_nonzero(mask == 255) == 192
    assert "temporal" not in segmentation[-1]
    assert not (output/"annotation_eval.jsonl").exists()


def test_bad_teacher_dry_run_never_writes_training_manifests(round_inputs, tmp_path):
    policy = json.loads(round_inputs["policy"].read_text())
    policy["eligible_for_pilot"] = False
    round_inputs["policy"].write_text(json.dumps(policy), encoding="utf-8")
    output = tmp_path/"dry"
    result = build_round(round_inputs, output)
    assert result.returncode == 0, result.stderr
    assert not (output/"annotation_train.jsonl").exists()
    result = build_round(round_inputs, tmp_path/"blocked", "--mode", "build", "--allow-weak-pilot")
    assert result.returncode != 0


def test_weak_pilot_needs_explicit_consent(round_inputs, tmp_path):
    result = build_round(round_inputs, tmp_path/"blocked", "--mode", "build")
    assert result.returncode != 0
    assert "allow-weak-pilot" in result.stderr


def test_either_teacher_training_scene_overlap_blocks_self_labels(round_inputs, tmp_path):
    protocol = json.loads(round_inputs["seg_metrics"].read_text())
    protocol["checkpoint_train_scenes"] = ["pool"]
    round_inputs["seg_metrics"].write_text(json.dumps(protocol), encoding="utf-8")
    result = build_round(round_inputs, tmp_path/"leak", "--mode", "build", "--allow-weak-pilot")
    assert result.returncode != 0
    assert "pseudo-label candidate scenes" in result.stderr


def test_changed_teacher_probabilities_are_not_accepted(round_inputs, tmp_path):
    row = read_rows(round_inputs["seg_predictions"])[-1]
    np.savez_compressed(row["probability"], probability=np.zeros((16, 16), np.float32))
    result = build_round(round_inputs, tmp_path/"changed", "--mode", "build", "--allow-weak-pilot")
    assert result.returncode != 0
    assert "probabilities changed" in result.stderr


def test_known_recursive_teacher_cannot_start_another_round(round_inputs, tmp_path):
    protocol = json.loads(round_inputs["score_protocol"].read_text())
    protocol["teacher_self_training_round"] = 1
    round_inputs["score_protocol"].write_text(json.dumps(protocol), encoding="utf-8")
    result = build_round(round_inputs, tmp_path/"recursive", "--mode", "build", "--allow-weak-pilot")
    assert result.returncode != 0
    assert "Recursive self-training" in result.stderr


def test_temporal_loss_downweight_does_not_cancel_at_batch_size_one():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from shadow_occ.annotation.segmentation import temporal_loss

    current, previous = torch.ones((1, 1, 4, 4)), torch.zeros((1, 1, 4, 4))
    grid = torch.zeros((1, 4, 4, 2))
    weight = torch.ones((1, 4, 4))
    full = temporal_loss(current, previous, grid, weight, torch.tensor([1.0]))
    reduced = temporal_loss(current, previous, grid, weight, torch.tensor([0.25]))
    assert float(reduced) == pytest.approx(float(full)*0.25)


def test_unknown_pseudo_association_teaches_only_shadow_tokens(monkeypatch):
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation import vlm

    def encode(*args, **kwargs):
        return {"input_ids": torch.tensor([[10, 11, 12, 13]]), "labels": torch.tensor([[-100, 11, 12, 13]]),
                "token_groups": torch.tensor([[0, 3, 1, 2]])}

    monkeypatch.setattr(vlm, "encode_example", encode)
    target = annotation("YES")
    row = {"id": "pseudo-positive", "task": "shadow_annotation", "annotation_target": target, "answer": json.dumps(target), "label_source": "model_pseudo",
           "self_training": {"round": 1}, "association_supervision_enabled": False}
    sample = vlm.SFTDataset([row], None, 100, 0.25, loss_mode="component")[0]
    assert sample["labels"].tolist() == [[-100, -100, 12, -100]]
    assert not sample["association_active"]
    assert sample["example_id"] == "pseudo-positive"
