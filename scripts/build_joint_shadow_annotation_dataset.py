from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import assert_disjoint, read_rows, write_rows
from shadow_occ.annotation.joint import annotation
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Build one GT-constrained shadow annotation task, including exterior caster associations.")
    parser.add_argument("--region-data-root", required=True)
    parser.add_argument("--offbev-data-root", help="Optional additional annotations of observed shadows with exterior caster targets")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt", default=str(Path(__file__).resolve().parents[1]/"prompts/shadow_annotation_gt.txt"))
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    schema = '\nJSON schema: {"shadow":"YES|NO|UNCERTAIN","caster":{"state":"visible|outside_bev|unresolved|none","category":"string","bbox_1000_context":null,"outside_candidate_id":null}}. Fill a box only for visible; fill an exterior candidate ID only for outside_bev. Return JSON only.'
    datasets = {split: [] for split in ("train", "dev", "eval")}
    counts = Counter()
    for split in datasets:
        for row in read_rows(Path(args.region_data_root)/f"vlm_{split}.jsonl"):
            if row.get("task") != "shadow":
                continue
            target = None
            if row.get("answer") == "NO":
                target = annotation("NO", "none")
            elif row.get("answer") == "YES":
                box, context = row.get("caster_bbox_xyxy"), row.get("context_crop_xyxy")
                if box and context and row.get("caster_source") == "annotation_gt":
                    x0, y0, x1, y1 = context
                    normalized = [round(1000*(box[0]-x0)/(x1-x0)), round(1000*(box[1]-y0)/(y1-y0)), round(1000*(box[2]-x0)/(x1-x0)), round(1000*(box[3]-y0)/(y1-y0))]
                    target = annotation("YES", "visible", category=row.get("caster_category", "unknown"), bbox=normalized)
                else:
                    target = annotation("YES", "unresolved")
            question = prompt
            if row.get("temporal_context"):
                question += " Image 4 follows the same caster with GT-assisted translation in this training/diagnostic view; changing shape and occlusion may interrupt continuity."
            item = {**row, "task": "shadow_annotation", "prompt": question+schema, "answer": json.dumps(target, separators=(",", ":")) if target else None, "annotation_target": target, "shadow_label_source": row["label_source"], "association_source": row.get("association_source", "geometry_weak") if target and target["caster"]["state"] == "visible" else "unresolved", "candidates": [], "annotation_origin": "observed_region"}
            datasets[split].append(item)
        if args.offbev_data_root:
            for row in read_rows(Path(args.offbev_data_root)/f"offbev_{split}.jsonl"):
                target = None
                if row.get("answer") is not None:
                    previous_answer = json.loads(row["answer"])
                    target = annotation("YES", "outside_bev", candidate_id=previous_answer["candidate_id"]) if previous_answer["decision"] == "POTENTIAL" else annotation("UNCERTAIN", "unresolved")
                question = prompt+"\nNominal BEV bounds (ego metres): "+json.dumps(row["bev_bounds_xyxy"])+"\nExterior association candidates (no GT used): "+json.dumps(row["candidates"], separators=(",", ":"))+schema
                item = {**row, "id": "exterior:"+row["id"], "task": "shadow_annotation", "prompt": question, "answer": json.dumps(target, separators=(",", ":")) if target else None, "annotation_target": target, "shadow_label_source": row["label_source"], "association_source": row["label_source"], "known_outside_caster": row.get("target_decision") == "POTENTIAL", "annotation_origin": "boundary_shadow"}
                datasets[split].append(item)
        ids = [row["id"] for row in datasets[split]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate unified annotation ID")
        counts.update(f"{split}:{row['label_source']}:{row.get('annotation_target', {}).get('shadow') if row.get('annotation_target') else 'unlabeled'}" for row in datasets[split])
    assert_disjoint(datasets["train"], datasets["dev"]+datasets["eval"])
    for split, rows in datasets.items():
        write_rows(output/f"annotation_{split}.jsonl", rows)
    write_json(output/"protocol.json", {**vars(args), "task": "one GT-constrained shadow annotation model", "counts": dict(counts), "exterior_policy": "outside-BEV potential casters are an annotation association field, not an additional model task"})


if __name__ == "__main__":
    main()
