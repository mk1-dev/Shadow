from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadow_occ.annotation.data import assert_disjoint, digest, group_split, read_rows, resolve_path, write_rows
from shadow_occ.annotation.offbev import ASSOCIATION_GT, answer_text, boxes_overlap, contains, footprint_status
from shadow_occ.utils.io import write_json


def visible_caster(caster, observation, index):
    if caster.get("caster_projection_near_plane_unknown"):
        return "unknown"
    box = caster.get("caster_visible_bbox_xyxy", caster.get("caster_bbox_xyxy"))
    if box is not None and any(boxes_overlap(box, crop) for crop in observation["image_crops_xyxy"]):
        return "visible"
    if observation.get("previous_context_crop_xyxy"):
        previous = [row for row in index.get((observation["previous_sample_token"], observation["camera"]), []) if row["instance_token"] == caster["instance_token"]]
        if len(previous) != 1 or previous[0].get("caster_projection_near_plane_unknown"):
            return "unknown"
        box = previous[0].get("caster_visible_bbox_xyxy", previous[0].get("caster_bbox_xyxy"))
        if box is not None:
            x0, y0, x1, y1 = box
            corners = np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]]) @ np.asarray(observation["previous_to_current_homography"]).T
            if np.any(corners[:, 2] <= 1e-6):
                return "unknown"
            xy = corners[:, :2]/corners[:, 2:]
            if boxes_overlap(np.r_[xy.min(axis=0), xy.max(axis=0)], observation["previous_context_crop_xyxy"]):
                return "visible"
    return "not_visible_in_inputs"


def main():
    parser = argparse.ArgumentParser("Build outside-BEV caster inference targets from GT-associated shadow observations.")
    parser.add_argument("--observations", nargs="+", required=True)
    parser.add_argument("--casters", nargs="+", required=True)
    parser.add_argument("--associations", help="Optional JSONL with observation_id, annotation_token and association_source")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt", default=str(ROOT/"prompts/offbev_caster.txt"))
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-visible-caster", action="store_true", help="Diagnostic stratum only; default excludes GT target appearances in all input views")
    parser.add_argument("--teach-ambiguous-abstention", action="store_true", help="Weak ABSTAIN for multiple geometrically compatible casters; does not label unannotated space empty")
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        raise ValueError("dev-fraction must be between zero and one")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    index = defaultdict(list)
    gt_ids = set()
    for path in args.casters:
        for row in read_rows(path):
            key = (row["sample_token"], row["camera"], row["annotation_token"])
            if key in gt_ids:
                raise ValueError("Duplicate caster GT record")
            gt_ids.add(key)
            index[(row["sample_token"], row["camera"])].append(row)
    associations = {}
    if args.associations:
        for row in read_rows(args.associations):
            if row["observation_id"] in associations:
                raise ValueError("Duplicate observation association")
            if row["association_source"] not in ASSOCIATION_GT | {"geometry_weak", "temporal_weak"}:
                raise ValueError("Unknown association source")
            associations[row["observation_id"]] = row
    from nuscenes.utils.splits import create_splits_scenes

    official = create_splits_scenes()
    datasets = {split: [] for split in ("train", "dev", "eval")}
    counts, seen, scene_splits = Counter(), set(), {}
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    for path in args.observations:
        for observation in read_rows(path):
            if observation["id"] in seen:
                raise ValueError("Duplicate observation ID")
            seen.add(observation["id"])
            scene, official_split = observation["scene_name"], observation["official_split"]
            if official_split in {"train", "val", "mini_train", "mini_val"} and scene not in official[official_split]:
                raise ValueError("Observation scene does not belong to its official split")
            split = group_split(scene, official_split, args.dev_fraction, args.seed)
            if scene_splits.setdefault(scene, split) != split:
                raise ValueError("Conflicting splits for one scene")
            bank = observation["candidates"]
            if len({item["id"] for item in bank}) != len(bank):
                raise ValueError("Duplicate candidate identifiers")
            bounds = observation["bev_bounds_xyxy"]
            casters = index.get((observation["sample_token"], observation["camera"]), [])
            casters = [caster for caster in casters if footprint_status(caster["caster_corners_ego"], bounds) == "outside"]
            association = associations.get(observation["id"])
            target, source, decision = None, "unlabeled", None
            if association:
                source = association["association_source"]
                decision = association.get("decision", "POTENTIAL")
                if decision == "POTENTIAL":
                    matched = [caster for caster in casters if caster["annotation_token"] == association["annotation_token"]]
                    if len(matched) != 1:
                        raise ValueError("Association GT target is missing or intersects the nominal BEV")
                    target = matched[0]
                elif decision != "ABSTAIN":
                    raise ValueError("Decision must be POTENTIAL or ABSTAIN")
            else:
                compatible = [caster for caster in casters if any(contains(cell, caster["caster_center_ego"]) for cell in bank)]
                if len(compatible) == 1:
                    target, source, decision = compatible[0], "geometry_weak", "POTENTIAL"
                elif len(compatible) > 1 and args.teach_ambiguous_abstention:
                    if not args.allow_visible_caster and any(visible_caster(caster, observation, index) != "not_visible_in_inputs" for caster in compatible):
                        counts["excluded_ambiguous_visible_or_unknown_context"] += 1
                        continue
                    source, decision = "geometry_weak", "ABSTAIN"
                # No compatible box means unknown annotation coverage/association, never a negative target.
            visibility = "no_associated_caster"
            valid_ids, answer = [], None
            if target is not None:
                visibility = visible_caster(target, observation, index)
                if not args.allow_visible_caster and visibility != "not_visible_in_inputs":
                    counts["excluded_"+visibility] += 1
                    continue
                valid_ids = [cell["id"] for cell in bank if contains(cell, target["caster_center_ego"])]
                if valid_ids:
                    selected = min([cell for cell in bank if cell["id"] in valid_ids], key=lambda cell: np.linalg.norm(np.asarray(cell["center_xy_ego"])-np.asarray(target["caster_center_ego"][:2])))
                    answer = answer_text(selected["id"])
                else:
                    counts["associated_caster_outside_candidate_bank"] += 1
            elif decision == "ABSTAIN":
                answer = answer_text()
            images = [str(resolve_path(image, Path(path).resolve().parent)) for image in observation["images"]]
            if not all(Path(image).is_file() for image in images):
                raise FileNotFoundError("Missing off-BEV observation image")
            question = prompt+"\nNominal BEV [xmin,ymin,xmax,ymax] metres: "+json.dumps(bounds)+"\nOutside candidate regions (ego coordinates, metres): "+json.dumps(bank, separators=(",", ":"))+"\nReturn {\"decision\":\"POTENTIAL\",\"candidate_id\":\"H1\"} using a listed ID, or {\"decision\":\"ABSTAIN\",\"candidate_id\":null}."
            item = {**observation, "images": images, "task": "offbev_caster", "split": split, "prompt": question, "answer": answer, "label_source": source, "target_decision": decision, "valid_candidate_ids": valid_ids, "target_annotation_token": target["annotation_token"] if target else None, "target_instance_token": target["instance_token"] if target else None, "target_center_xy_ego": target["caster_center_ego"][:2] if target else None, "target_category": target["caster_category"] if target else None, "target_visibility": visibility, "supervision_note": "GT object existence/location does not itself establish a shadow-caster association"}
            datasets[split].append(item)
            counts[f"{split}_{source}_{decision}"] += 1
    assert_disjoint(datasets["train"], datasets["dev"]+datasets["eval"])
    for split, rows in datasets.items():
        write_rows(output/f"offbev_{split}.jsonl", rows)
    write_json(output/"protocol.json", {**vars(args), "input_sha256": {path: digest(path) for path in args.observations+args.casters}, "counts": dict(counts), "scenes": scene_splits, "negative_policy": "no GT match remains unknown; ABSTAIN means insufficient association, not empty space", "target": "entire caster footprint outside the nominal BEV; input shadow remains inside", "visibility_policy": "check every input view for direct target appearance"})


if __name__ == "__main__":
    main()
