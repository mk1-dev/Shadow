from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shadow_occ.annotation.data import (
    SOURCES, assert_disjoint, checked_write_image, digest, group_split,
    load_array, make_target, read_rows, resolve_path, write_rows,
)
from shadow_occ.utils.io import write_json
from shadow_occ.annotation.temporal import warp


def crop_bounds(box, shape, scale):
    height, width = shape
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = max(16, (x1 - x0) * scale / 2), max(16, (y1 - y0) * scale / 2)
    return [max(0, int(cx-half_w)), max(0, int(cy-half_h)), min(width, int(np.ceil(cx+half_w))), min(height, int(np.ceil(cy+half_h)))]


def main() -> None:
    parser = argparse.ArgumentParser("Build shared segmentation and single-proposal VLM data.")
    parser.add_argument("--manifest", nargs="+", required=True, help="Canonical JSONLs; see docs/shadow_annotation_hpc.md")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-scenes", help="JSON scene list, e.g. prior tuning scenes")
    parser.add_argument("--positive-fraction", type=float, default=0.7)
    parser.add_argument("--negative-fraction", type=float, default=0.05)
    parser.add_argument("--min-known-fraction", type=float, default=0.8)
    parser.add_argument("--prompt", default=str(ROOT / "prompts/shadow_single_region_sft.txt"))
    parser.add_argument("--casters", help="Optional caster GT JSONL exported by export_nuscenes_caster_gt.py")
    parser.add_argument("--no-temporal-context", action="store_true", help="Ablate only the VLM's additional previous-frame view")
    parser.add_argument("--min-temporal-region-overlap", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        raise ValueError("dev-fraction must be between zero and one")
    if not 0 <= args.negative_fraction < args.positive_fraction <= 1 or not 0 < args.min_known_fraction <= 1:
        raise ValueError("Invalid target thresholds")
    if not 0 <= args.min_temporal_region_overlap <= 1:
        raise ValueError("Invalid temporal-region-overlap threshold")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    excluded = set(json.loads(Path(args.exclude_scenes).read_text(encoding="utf-8"))) if args.exclude_scenes else set()
    casters = {row["id"]: row for row in read_rows(args.casters)} if args.casters else {}
    segments = {key: [] for key in ("train", "dev", "eval")}
    vlm = {key: [] for key in segments}
    counts = Counter()
    seen, scene_splits, image_splits = set(), {}, {}
    from nuscenes.utils.splits import create_splits_scenes

    official_scenes = create_splits_scenes()
    inputs = [(row, Path(path).resolve().parent) for path in args.manifest for row in read_rows(path)]
    for original, base in inputs:
        row = dict(original)
        if row.get("caster_id"):
            if row["caster_id"] not in casters:
                raise ValueError(f"Unknown caster_id: {row['caster_id']}")
            caster = casters[row["caster_id"]]
            for key in ("scene_name", "sample_token", "camera", "official_split", "image"):
                if key in row and row[key] != caster[key]:
                    raise ValueError(f"Caster mismatch for {row['caster_id']}: {key}")
            row = {**caster, **row}
        identifier = str(row["id"])
        if identifier in seen:
            raise ValueError(f"Duplicate id: {identifier}")
        seen.add(identifier)
        scene = row["scene_name"]
        official_split = row["official_split"]
        if official_split in {"train", "val", "mini_train", "mini_val"} and scene not in official_scenes[official_split]:
            raise ValueError(f"Scene {scene} is outside the real official {official_split} split")
        if scene in excluded:
            counts["excluded_rows"] += 1
            continue
        split = group_split(scene, row["official_split"], args.dev_fraction, args.seed)
        if scene_splits.setdefault(scene, split) != split:
            raise ValueError(f"Conflicting official splits for {scene}")
        image_path = resolve_path(row["image"], Path(args.casters).resolve().parent if row.get("caster_id") else base)
        if image_splits.setdefault(str(image_path), split) != split:
            raise ValueError("The same image appears across data splits")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        shape = image.shape[:2]
        target = make_target(row, shape, base)
        stem = hashlib.sha256(identifier.encode()).hexdigest()[:24]
        target_path = output / "targets" / f"{stem}.png"
        checked_write_image(target_path, target)
        common = {
            "id": identifier, "scene_name": scene, "official_split": row["official_split"],
            "split": split, "image": str(image_path), "target": str(target_path),
            "label_source": row.get("label_source", "unlabeled"),
            "caster_source": row.get("caster_source", "none"),
        }
        for key in ("sample_token", "camera", "group", "pair_id", "annotation_token", "instance_token", "caster_bbox_xyxy", "caster_category", "association_source", "ego_to_img", "ego_to_global", "timestamp", "temporal_status", "ground_z_ego"):
            if key in row:
                common[key] = row[key]
        temporal_image, temporal_overlap = None, None
        if row.get("temporal"):
            temporal = dict(row["temporal"])
            for key, expected in (("previous_scene_name", scene), ("previous_instance_token", row.get("instance_token")), ("previous_camera", row.get("camera"))):
                if temporal.get(key) != expected:
                    raise ValueError(f"Temporal context mismatch: {identifier}/{key}")
            for key in ("previous_image", "evidence"):
                temporal[key] = str(resolve_path(temporal[key], base))
            common["temporal"] = temporal
            if temporal.get("lighting_comparable") and not args.no_temporal_context:
                previous = cv2.imread(temporal["previous_image"])
                if previous is None:
                    raise FileNotFoundError(temporal["previous_image"])
                temporal_image = warp(previous, temporal["caster_previous_to_current"], shape)
                with np.load(temporal["evidence"], allow_pickle=False) as archive:
                    temporal_overlap = np.asarray(archive["shared_caster"]) > 0
                if temporal_overlap.shape != shape:
                    raise ValueError("Temporal evidence and image shapes differ")
        segments[split].append(common)
        counts[f"{split}_known_pixels"] += int(np.count_nonzero(target != 255))
        for region in row.get("regions", []):
            mask = load_array(region["mask"], base) > 0
            if mask.shape != shape:
                raise ValueError("Region masks must match the source image exactly")
            if not mask.any():
                counts["empty_regions"] += 1
                continue
            yy, xx = np.where(mask)
            box = [xx.min(), yy.min(), xx.max()+1, yy.max()+1]
            tight = crop_bounds(box, shape, 1.6)
            context_box = box
            if row.get("caster_bbox_xyxy"):
                caster_box = row["caster_bbox_xyxy"]
                context_box = [min(box[0], caster_box[0]), min(box[1], caster_box[1]), max(box[2], caster_box[2]), max(box[3], caster_box[3])]
            context = crop_bounds(context_box, shape, 2.5)
            x0, y0, x1, y1 = tight
            clean = image[y0:y1, x0:x1]
            outlined = clean.copy()
            contours, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outlined, contours, -1, (245, 245, 245), 1, cv2.LINE_AA)
            region_id = f"{identifier}:{region['id']}"
            region_stem = hashlib.sha256(region_id.encode()).hexdigest()[:24]
            region_path = output / "regions" / f"{region_stem}.png"
            checked_write_image(region_path, mask.astype(np.uint8)*255)
            paths = [output / "views" / f"{region_stem}_{index}.png" for index in range(3)]
            cx0, cy0, cx1, cy1 = context
            for path, view in zip(paths, [clean, outlined, image[cy0:cy1, cx0:cx1]]):
                checked_write_image(path, view)
            region_prompt = prompt
            has_context = temporal_image is not None and float(np.count_nonzero(mask & temporal_overlap)/mask.sum()) >= args.min_temporal_region_overlap
            if has_context:
                previous_path = output/"views"/f"{region_stem}_previous.png"
                checked_write_image(previous_path, temporal_image[cy0:cy1, cx0:cx1])
                paths.append(previous_path)
                region_prompt += " Image 4 is the previous-frame context aligned using ego motion and the translation of the same tracked caster. Its coordinates match image 3. Compare only jointly visible ground; unmapped black pixels are not evidence. A stationary cast shadow can persist with almost no brightness difference. A moving caster can move its shadow. Continuity supports a decision but does not guarantee that a dark region is a shadow."
            known = mask & (target != 255)
            known_fraction = float(known.sum() / mask.sum())
            positive_fraction = float(np.count_nonzero(target[known] == 1) / max(known.sum(), 1))
            label = region.get("shadow_label")
            source = region.get("label_source", row.get("label_source", "unlabeled"))
            if label is not None:
                if source not in SOURCES or str(label).upper() not in {"YES", "NO", "IGNORE"}:
                    raise ValueError("Explicit shadow_label requires a supported label_source and YES/NO/IGNORE")
                label = str(label).upper()
            elif known_fraction >= args.min_known_fraction:
                label = "YES" if positive_fraction >= args.positive_fraction else "NO" if positive_fraction <= args.negative_fraction else "IGNORE"
            else:
                label = "IGNORE"
            item = {**common, "id": region_id, "parent_id": identifier, "task": "shadow", "region_id": region["id"], "region_mask": str(region_path), "context_crop_xyxy": context, "images": [str(path) for path in paths], "prompt": region_prompt, "temporal_context": has_context, "answer": label if label != "IGNORE" else None, "label_source": source, "known_fraction": known_fraction, "positive_fraction": positive_fraction}
            vlm[split].append(item)
            counts[f"{split}_shadow_{label}"] += 1
        if row.get("caster_source") == "annotation_gt" and row.get("caster_bbox_xyxy"):
            width, height = shape[1], shape[0]
            box = row["caster_bbox_xyxy"]
            answer = {"category": row["caster_category"], "bbox_1000": [round(1000*box[0]/width), round(1000*box[1]/height), round(1000*box[2]/width), round(1000*box[3]/height)]}
            # The box is a language target only; its coordinates are never given in the question.
            anchor = [round(500*(box[0]+box[2])/width), round(500*(box[1]+box[3])/height)]
            vlm[split].append({**common, "id": identifier+":caster", "task": "caster", "images": [str(image_path)], "prompt": f"Identify the object whose center is nearest point {anchor} in normalized 0..1000 image coordinates. Return JSON with category and bbox_1000 in [x0,y0,x1,y1] normalized to 0..1000.", "answer": json.dumps(answer, separators=(",", ":")), "label_source": "object_annotation_gt"})
            counts[f"{split}_caster"] += 1
    assert_disjoint(segments["train"], segments["dev"] + segments["eval"])
    for split in segments:
        write_rows(output / f"seg_{split}.jsonl", segments[split])
        write_rows(output / f"vlm_{split}.jsonl", vlm[split])
    write_json(output / "dataset_protocol.json", {"input_sha256": {path: digest(path) for path in args.manifest}, "casters_sha256": digest(args.casters) if args.casters else None, "prompt_sha256": digest(args.prompt), "seed": args.seed, "dev_fraction": args.dev_fraction, "positive_fraction": args.positive_fraction, "negative_fraction": args.negative_fraction, "min_known_fraction": args.min_known_fraction, "no_temporal_context": args.no_temporal_context, "min_temporal_region_overlap": args.min_temporal_region_overlap, "excluded_scenes": sorted(excluded), "scenes": scene_splits, "counts": dict(counts)})
    print(f"Wrote shared segmentation/VLM datasets to {output}")


if __name__ == "__main__":
    main()
