from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import assert_disjoint, read_rows
from shadow_occ.annotation.offbev import footprint_status
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("CPU schema/scene checks for outside-BEV caster data.")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    datasets, summary = {}, {}
    for split in ("train", "dev", "eval"):
        rows = read_rows(root/f"offbev_{split}.jsonl")
        datasets[split] = rows
        counts, ids = Counter(), set()
        for row in rows:
            if row["id"] in ids or row["task"] != "offbev_caster" or row["split"] != split:
                raise ValueError("Duplicate ID, wrong task or wrong split")
            ids.add(row["id"])
            if row.get("caster_gt_used_for_observation"):
                raise ValueError("Off-BEV observations must not depend on a known caster's geometry")
            for image in row["images"]:
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
            for candidate in row["candidates"]:
                x0, y0, x1, y1 = candidate["xyxy_ego"]
                corners = [[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]]
                if footprint_status(corners, row["bev_bounds_xyxy"]) != "outside":
                    raise ValueError("A candidate region intersects the nominal BEV")
            if row["answer"] is not None:
                answer = json.loads(row["answer"])
                if answer["decision"] == "POTENTIAL" and answer["candidate_id"] not in row["valid_candidate_ids"]:
                    raise ValueError("Target candidate is not GT-compatible")
                counts[answer["decision"]] += 1
            else:
                counts["unlabeled_or_bank_miss"] += 1
            counts["source:"+row["label_source"]] += 1
            counts["visibility:"+row["target_visibility"]] += 1
        summary[split] = {"examples": len(rows), "scenes": len({row['scene_name'] for row in rows}), "counts": dict(counts)}
    assert_disjoint(datasets["train"], datasets["dev"]+datasets["eval"])
    if {row["scene_name"] for row in datasets["dev"]} & {row["scene_name"] for row in datasets["eval"]}:
        raise ValueError("Dev and eval overlap")
    write_json(root/"data_check.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
