import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from shadow_occ.annotation.data import read_rows, write_rows


ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args):
    subprocess.run([sys.executable, str(ROOT/"scripts"/name), *map(str, args)], cwd=ROOT, check=True, capture_output=True, text=True)


def test_joint_builder_does_not_promote_legacy_object_matches_to_shadow_gt(tmp_path):
    regions, exterior, output = tmp_path/"regions", tmp_path/"exterior", tmp_path/"joint"
    for split in ("train", "dev", "eval"):
        write_rows(regions/f"vlm_{split}.jsonl", [])
        write_rows(exterior/f"offbev_{split}.jsonl", [{
            "id": split, "scene_name": split, "split": split,
            "official_split": "val" if split == "eval" else "train", "images": [],
            "answer": json.dumps({"decision": "POTENTIAL", "candidate_id": "old-id"}),
            "candidates": [{"id": "old-id", "xyxy_ego": [40, 0, 44, 4]}], "valid_candidate_ids": ["old-id"],
            "target_decision": "POTENTIAL", "label_source": "geometry_weak", "bev_bounds_xyxy": [-40, -40, 40, 40],
        }])
    run_script("build_joint_shadow_annotation_dataset.py", "--region-data-root", regions, "--offbev-data-root", exterior, "--output-root", output)
    row = read_rows(output/"annotation_train.jsonl")[0]
    assert row["annotation_target"]["shadow"] == "UNCERTAIN"
    assert row["shadow_label_source"] == "unlabeled"
    assert not row["known_outside_caster"]
    assert row["valid_candidate_ids"] == [row["candidate_id_mapping"]["old-id"]]


def test_prediction_proposal_and_refinement_can_recover_previously_unknown_pixels(tmp_path):
    shape = (16, 16)
    target = np.full(shape, 255, np.uint8)
    ground = np.full(shape, 255, np.uint8)
    target_path, ground_path = tmp_path/"target.png", tmp_path/"ground.png"
    cv2.imwrite(str(target_path), target)
    cv2.imwrite(str(ground_path), ground)
    probability = np.zeros(shape, np.float32)
    probability[4:12, 4:12] = 0.95
    probability_path = tmp_path/"probability.npz"
    np.savez_compressed(probability_path, probability=probability)
    predictions = tmp_path/"predictions.jsonl"
    write_rows(predictions, [{"id": "a", "image": str(tmp_path/"image.png"), "target": str(target_path),
                             "probability": str(probability_path), "ground_support": str(ground_path), "label_source": "geometry_weak",
                             "segmenter_readiness": {"noncollapsed": True, "approved_for_relabeling": False}}])
    proposals = tmp_path/"proposals"
    run_script("propose_shadow_annotation_regions.py", "--seg-predictions", predictions, "--output-root", proposals, "--min-area", 4)
    regions = read_rows(proposals/"manifest.jsonl")[0]["regions"]
    assert len(regions) == 1
    assert regions[0]["shadow_label"] == "IGNORE"
    votes = tmp_path/"votes.jsonl"
    write_rows(votes, [{"id": "a:proposal", "parent_id": "a", "region_mask": regions[0]["mask"], "prediction": {"shadow": "YES"}}])
    refined = tmp_path/"refined"
    run_script("refine_shadow_annotations.py", "--seg-predictions", predictions, "--annotation-results", votes, "--output-root", refined, "--allow-weak-pilot")
    row = read_rows(refined/"manifest.jsonl")[0]
    assert row["refinement"]["new_positive_pixels"] == 64
    assert np.count_nonzero(cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE) == 1) == 64


def test_crossfit_manifest_excludes_heldout_and_dev_scenes(tmp_path):
    seg, joint = tmp_path/"seg", tmp_path/"joint"
    for task, root in (("seg", seg), ("annotation", joint)):
        write_rows(root/f"{task}_train.jsonl", [{"id": f"{task}-{i}", "scene_name": f"scene-{i}", "split": "train", "official_split": "train"} for i in range(6)])
        write_rows(root/f"{task}_dev.jsonl", [{"id": "dev", "scene_name": "dev-scene", "split": "dev", "official_split": "train"}])
    output = tmp_path/"crossfit"
    run_script("build_shadow_crossfit_folds.py", "--seg-data-root", seg, "--annotation-data-root", joint, "--output-root", output)
    heldout_ids = []
    for fold in range(3):
        train = read_rows(output/f"fold_{fold}"/"annotation_train.jsonl")
        heldout = read_rows(output/f"fold_{fold}"/"annotation_heldout.jsonl")
        assert not ({row["scene_name"] for row in train} & {row["scene_name"] for row in heldout})
        heldout_ids += [row["id"] for row in heldout]
    assert len(heldout_ids) == len(set(heldout_ids)) == 6
