from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import assert_disjoint, read_rows, write_rows
from shadow_occ.annotation.joint import annotation
from shadow_occ.annotation.supervision import exterior_target, permute_candidates, visible_target
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Build one GT-constrained shadow annotation task, including exterior caster associations.")
    parser.add_argument("--region-data-root", required=True)
    parser.add_argument("--offbev-data-root", help="Optional additional annotations of observed shadows with exterior caster targets")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt", default=str(Path(__file__).resolve().parents[1]/"prompts/shadow_annotation_gt.txt"))
    parser.add_argument("--seed", type=int, default=0, help="Candidate order and ID permutation seed")
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
                target = visible_target(row)
            question = prompt
            if row.get("physics_temporal_context"):
                question += " Images 4 and 5 show earlier and later contexts registered to the same receiver ground as image 3, not to caster translation. Compare formation and recovery; black unsupported pixels are unknown. Future context is an offline annotation input, not a causal deployment claim."
            if row.get("temporal_context"):
                question += " Image 4 follows the same caster with GT-assisted translation in this training/diagnostic view; changing shape and occlusion may interrupt continuity."
            item = {**row, "task": "shadow_annotation", "prompt": question+schema, "answer": json.dumps(target, separators=(",", ":")) if target else None, "annotation_target": target, "shadow_label_source": row["label_source"], "association_source": row.get("association_source", "unresolved") if target and target["caster"]["state"] == "visible" else "unresolved", "candidates": [], "annotation_origin": "observed_region", "input_regime": "gt_assisted_context"}
            datasets[split].append(item)
        if args.offbev_data_root:
            for row in read_rows(Path(args.offbev_data_root)/f"offbev_{split}.jsonl"):
                row = dict(row)
                if row.get("answer") is not None:
                    row["selected_candidate_id"] = json.loads(row["answer"]).get("candidate_id")
                bank, mapping = permute_candidates(row["candidates"], row["id"], args.seed)
                row["candidates"] = bank
                row["valid_candidate_ids"] = [mapping[value] for value in row.get("valid_candidate_ids", [])]
                if row.get("selected_candidate_id"):
                    row["selected_candidate_id"] = mapping[row["selected_candidate_id"]]
                target = exterior_target(row)
                question = prompt+"\nNominal BEV bounds (ego metres): "+json.dumps(row["bev_bounds_xyxy"])+"\nExterior association candidates (no GT used): "+json.dumps(row["candidates"], separators=(",", ":"))+schema
                shadow_source = row.get("shadow_label_source", "unlabeled")
                association_source = row.get("association_source", row["label_source"])
                item = {**row, "id": "exterior:"+row["id"], "task": "shadow_annotation", "prompt": question, "answer": json.dumps(target, separators=(",", ":")) if target else None, "annotation_target": target, "shadow_label_source": shadow_source, "association_source": association_source, "label_source": shadow_source if shadow_source != "unlabeled" else association_source, "known_outside_caster": row.get("target_decision") == "POTENTIAL" and row.get("shadow_label") == "YES" and shadow_source != "unlabeled", "annotation_origin": "boundary_shadow", "candidate_id_mapping": mapping, "input_regime": "observation_only"}
                datasets[split].append(item)
        ids = [row["id"] for row in datasets[split]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate unified annotation ID")
        counts.update(f"{split}:{row['label_source']}:{row.get('annotation_target', {}).get('shadow') if row.get('annotation_target') else 'unlabeled'}" for row in datasets[split])
    assert_disjoint(datasets["train"], datasets["dev"]+datasets["eval"])
    if {row["scene_name"] for row in datasets["dev"]} & {row["scene_name"] for row in datasets["eval"]}:
        raise ValueError("Dev and eval scenes overlap")
    for split, rows in datasets.items():
        write_rows(output/f"annotation_{split}.jsonl", rows)
    write_json(output/"protocol.json", {**vars(args), "task": "one GT-constrained shadow annotation model", "counts": dict(counts), "exterior_policy": "outside-BEV potential casters are an annotation association field, not an additional model task"})


if __name__ == "__main__":
    main()
