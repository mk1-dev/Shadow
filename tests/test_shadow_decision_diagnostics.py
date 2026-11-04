import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from shadow_occ.annotation.diagnostics import (
    SCORING_VERSION, average_precision, balance_audit, classify, cluster_intervals,
    example_weight, fit_threshold, score_metrics, temporal_multiplier,
)
from shadow_occ.annotation.data import read_rows, write_rows
from shadow_occ.annotation.joint import annotation, shadow_confusion


def row(label, score, index=0, split="dev"):
    return {"id": str(index), "scene_name": f"scene-{split}-{index}", "split": split, "official_split": "train" if split == "dev" else "val",
            "annotation_target": annotation(label, "none" if label == "NO" else "unresolved"), "score": score,
            "scorer_id": "model-a", "scoring_version": SCORING_VERSION, "shadow_label_source": "geometry_weak", "label_source": "geometry_weak"}


def test_average_precision_groups_ties_and_reports_imbalance_baseline():
    assert average_precision([1, 0], [0.5, 0.5]) == 0.5
    assert average_precision([0, 1], [0.1, 0.9]) == 1.0
    assert average_precision([0, 0], [0.1, 0.9]) is None


def test_abstaining_everywhere_cannot_look_perfect():
    rows = [row("YES", 0.51), row("NO", 0.49, 1)]
    metrics = score_metrics(rows, 0.5, margin=0.1)
    assert metrics["coverage"] == 0
    assert metrics["balanced_accuracy"] == 0
    assert metrics["precision"] is None
    generated = [{"target": item["annotation_target"], "prediction": annotation("UNCERTAIN")} for item in rows]
    assert shadow_confusion(generated)["balanced_accuracy"] == 0


def test_eval_labels_cannot_select_thresholds():
    with pytest.raises(ValueError, match="dev"):
        fit_threshold([row("YES", 0.9, split="eval"), row("NO", 0.1, 1, split="eval")])


def test_calibration_reports_no_useful_recall_when_fpr_constraint_blocks_it():
    rows = [row("YES", 0.1), row("NO", 0.9, 1)]
    policy, scan = fit_threshold(rows, max_fpr=0)
    assert not policy["has_positive_recall"]
    assert policy["approved_for_relabeling"] is False
    assert scan


def test_weights_expose_combined_source_and_independent_tracks():
    rows = [row("YES", 0.9), row("NO", 0.1, 1)]
    rows[0].update(label_source="manual_shadow_gt", shadow_label_source="manual_shadow_gt", association_source="geometry_weak", instance_token="same-track")
    assert example_weight(rows[0], 0.25) == 0.25
    audit = balance_audit(rows, {"YES": 2.0, "NO": 0.5}, 0.25)
    assert audit["classes"]["YES"]["sum_decision_weight"] == 0.5
    assert audit["classes"]["YES"]["unique_tracks"] == 1
    assert not audit["outside_training_supported"]


def test_static_small_difference_and_missing_evidence_are_never_penalized():
    policy = {"validated_for_weighting": True, "fitted_split": "dev", "downweight": 0.5, "minimum_pixels": 64,
              "minimum_motion_advantage": 0.1, "minimum_transition_ratio": 0.6}
    assert temporal_multiplier({"temporal": {"mode": "static", "lighting_comparable": True}}, policy) == 1
    assert temporal_multiplier({}, policy) == 1
    moving = {"label_source": "geometry_weak", "temporal": {"mode": "moving", "lighting_comparable": True, "motion_alignment_comparison_pixels": 100, "motion_alignment_advantage": -0.2}}
    assert temporal_multiplier(moving, policy) == 0.5
    assert temporal_multiplier({**moving, "label_source": "manual_shadow_gt"}, policy) == 1.0
    with pytest.raises(ValueError, match="validated"):
        temporal_multiplier(moving, {**policy, "validated_for_weighting": False})


def test_scene_bootstrap_keeps_fixed_policy_and_reports_valid_draws():
    rows = [row("YES", 0.9), row("NO", 0.1, 1)]
    intervals = cluster_intervals(rows, 0.5, bootstrap=10)
    assert intervals["fixed_policy"]
    assert intervals["ci95"]["coverage"]["valid_draws"] == 10


def test_component_loss_separates_decision_and_association_and_ignores_prompt():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.objectives import component_loss

    logits = torch.zeros((1, 5, 4), requires_grad=True)
    labels = torch.tensor([[-100, 0, 1, 2, 3]])
    groups = torch.tensor([[0, 3, 1, 2, 3]])
    loss, components = component_loss(logits, labels, groups, 2.0, {"decision": 1.0, "association": 0.5, "format": 0.1})
    assert float(loss) == pytest.approx(math.log(4)*2.6/1.6)
    assert set(components) == {"decision", "association", "format"}
    no_association, _ = component_loss(logits, labels, groups, 1.0, {"decision": 1.0, "association": 0.0, "format": 0.1})
    no_association.backward()
    assert torch.all(logits.grad[0, 2] == 0)


def test_token_offsets_align_to_response_and_not_image_prompt():
    pytest.importorskip("torch")
    from shadow_occ.annotation.objectives import assistant_token_groups

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) for char in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}

    answer = json.dumps(annotation("YES"), separators=(",", ":"))
    text = "visual-prompt\n"+answer+"<end>"
    groups = assistant_token_groups(Tokenizer(), text, answer, [ord(char) for char in answer+"<end>"])
    assert len(groups) == len(answer+"<end>")
    assert 1 in groups.tolist() and 2 in groups.tolist()
    with pytest.raises(ValueError, match="alignment"):
        assistant_token_groups(Tokenizer(), text, answer, [])


def test_cli_freezes_dev_policy_and_rejects_a_different_eval_scorer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root/"scripts"/"analyze_shadow_decision_scores.py"
    dev, evaluation = tmp_path/"dev.jsonl", tmp_path/"eval.jsonl"
    write_rows(dev, [row("YES", 0.9), row("NO", 0.1, 1)])
    write_rows(evaluation, [row("YES", 0.8, 2, split="eval"), row("NO", 0.2, 3, split="eval")])
    subprocess.run([sys.executable, str(script), "fit-dev", "--scores", str(dev), "--output-root", str(tmp_path/"fit"), "--bootstrap", "0"], check=True, cwd=root)
    calibration = tmp_path/"fit"/"calibration.json"
    subprocess.run([sys.executable, str(script), "evaluate", "--scores", str(evaluation), "--calibration", str(calibration), "--output-root", str(tmp_path/"evaluation"), "--bootstrap", "0"], check=True, cwd=root)
    assert not (tmp_path/"evaluation"/"dev_threshold_scan.json").exists()
    data = read_rows(evaluation)
    data[0]["scorer_id"] = data[1]["scorer_id"] = "different-model"
    write_rows(evaluation, data)
    result = subprocess.run([sys.executable, str(script), "evaluate", "--scores", str(evaluation), "--calibration", str(calibration), "--output-root", str(tmp_path/"bad")], cwd=root, capture_output=True)
    assert result.returncode != 0


def test_occlusion_removes_region_from_all_four_views(tmp_path):
    import cv2

    root = Path(__file__).resolve().parents[1]
    image = np.full((16, 16, 3), 200, np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    mask[5:11, 5:11] = 255
    image_path, mask_path = tmp_path/"image.png", tmp_path/"mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    manifest = tmp_path/"manifest.jsonl"
    write_rows(manifest, [{"id": "sample", "task": "shadow_annotation", "images": [str(image_path)]*4,
                           "tight_crop_xyxy": [0, 0, 16, 16], "context_crop_xyxy": [0, 0, 16, 16], "region_mask": str(mask_path)}])
    output = tmp_path/"occlusion"
    subprocess.run([sys.executable, str(root/"scripts"/"make_shadow_input_ablation.py"), "--manifest", str(manifest), "--mode", "occlude-region", "--output-root", str(output)], cwd=root, check=True)
    result = read_rows(output/"manifest.jsonl")[0]
    for path in result["images"]:
        changed = cv2.imread(path)
        assert np.all(changed[8, 8] == 127)
        assert np.all(changed[0, 0] == 200)
