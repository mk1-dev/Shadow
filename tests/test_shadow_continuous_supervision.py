import json

import numpy as np
import pytest

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from tests.test_bev_flow_ablations import script


def comparison_fixture(tmp_path):
    image, target = tmp_path/"rgb.png", tmp_path/"mask.png"
    checked_write_image(image, np.zeros((4, 4, 3), np.uint8))
    checked_write_image(target, np.tile(np.array([0, 1, 255, 1], np.uint8), (4, 1)))
    reference = tmp_path/"eval.jsonl"
    write_rows(reference, [{"id": "eval", "scene_name": "eval"}])
    paths = []
    for name in ("v93", "v94"):
        root = tmp_path/name
        arm = {"method": "motion_illumination"}
        for split in ("train", "dev"):
            path = root/(split+".jsonl")
            write_rows(path, [{"id": split, "frame_token": split, "scene_name": split,
                               "image": str(image), "target": str(target), "label_source": "temporal_weak"}])
            arm[split] = str(path)
        plan = {"arms": [arm], "scene_sets": {"train": ["train"], "dev": ["dev"]},
                "weak_eval_reference": str(reference), "weak_eval_reference_sha256": digest(reference)}
        path = root/"suite.json"
        path.write_text(json.dumps(plan))
        paths.append(path)
    return paths


def test_comparison_freezes_labels_and_varies_only_student_objective(tmp_path):
    paths = comparison_fixture(tmp_path)
    plan = script("prepare_shadow_loss_comparison").prepare(*paths, tmp_path/"paired")
    assert len(plan["arms"]) == 9
    assert plan["paired_initialization"] and plan["deterministic"]
    assert plan["amp"] == "off"
    assert {arm["training"]["seed"] for arm in plan["arms"]} == {0, 1, 2}
    assert {arm["training"]["supervision_mode"] for arm in plan["arms"]} == {"v93", "v94", "continuous"}
    assert all(read_rows(arm["train"])[0]["id"] == "train" for arm in plan["arms"])


def test_comparison_refuses_different_binary_targets(tmp_path):
    paths = comparison_fixture(tmp_path)
    other = tmp_path/"different.png"
    checked_write_image(other, np.zeros((4, 4), np.uint8))
    plan = json.loads(paths[1].read_text())
    manifest = plan["arms"][0]["train"]
    rows = read_rows(manifest)
    rows[0]["target"] = str(other)
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Binary labels"):
        script("prepare_shadow_loss_comparison").prepare(*paths, tmp_path/"paired")


def test_inherited_and_expanded_pixels_keep_dice_and_gradients():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import continuous_weighted_loss
    target = torch.tensor([[[0, 1, 1, 255]]])
    tier = torch.tensor([[[3, 4, 2, 0]]])
    confidence = torch.tensor([[[1., .25, .15, 0.]]])
    logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
    loss, terms = continuous_weighted_loss(logits, target, tier, confidence, .5)
    loss.backward()
    assert terms["dice"].item() > 0
    assert logits.grad[0, 0, 0, 0] > 0
    assert logits.grad[0, 0, 0, 1] < 0 and logits.grad[0, 0, 0, 2] < 0
    assert logits.grad[0, 0, 0, 3] == 0
    assert abs(float(logits.grad[0, 0, 0, 1])) > abs(float(logits.grad[0, 0, 0, 2]))


def test_class_normalization_preserves_loss_scale_with_uniform_inherited_weight():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import continuous_weighted_loss
    target = torch.tensor([[[0, 1, 1, 255]]])
    tier = torch.tensor([[[3, 4, 4, 0]]])
    confidence = torch.tensor([[[1., .25, .25, 0.]]])
    logits = torch.tensor([[[[-1., .3, -.2, 10.]]]])
    a, _ = continuous_weighted_loss(logits, target, tier, confidence, .5, inherited_weight=.7)
    b, _ = continuous_weighted_loss(logits, target, tier, confidence, .5, inherited_weight=1.)
    assert a.item() == pytest.approx(b.item())


def test_model_state_hash_tracks_weights_not_checkpoint_serialization(tmp_path):
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import model_state_sha256
    a = torch.nn.Linear(2, 1)
    b = torch.nn.Linear(2, 1)
    b.load_state_dict(a.state_dict())
    assert model_state_sha256(a) == model_state_sha256(b)
    with torch.no_grad():
        b.bias.add_(1)
    assert model_state_sha256(a) != model_state_sha256(b)
