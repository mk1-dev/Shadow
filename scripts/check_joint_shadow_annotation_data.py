from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import SOURCES, assert_disjoint, read_rows
from shadow_occ.annotation.joint import parse
from shadow_occ.annotation.offbev import ASSOCIATION_GT
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Check the single GT-constrained VLM annotation task on HPC.")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    datasets, report = {}, {}
    for split in ("train", "dev", "eval"):
        rows = read_rows(root/f"annotation_{split}.jsonl")
        datasets[split] = rows
        counts, seen = Counter(), set()
        for row in rows:
            if row["id"] in seen or row["task"] != "shadow_annotation" or row["split"] != split:
                raise ValueError("Duplicate ID, wrong annotation task or wrong split")
            seen.add(row["id"])
            for image in row["images"]:
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
            if row["answer"] is not None:
                target = parse(row["answer"], row.get("candidates", []))
                if target != row["annotation_target"]:
                    raise ValueError("Serialized annotation target disagrees with metadata")
                if target["shadow"] in {"YES", "NO"} and row["shadow_label_source"] not in SOURCES:
                    raise ValueError("Shadow YES/NO requires independent shadow provenance")
                if target["caster"]["state"] == "visible" and row["association_source"] not in ASSOCIATION_GT:
                    if row.get("association_status") not in {"weak_geometry_supported", "temporal_supported"} or row.get("associated_annotation_token") != row.get("annotation_token"):
                        raise ValueError("Visible weak association lacks support or changes caster identity")
                counts["shadow:"+target["shadow"]] += 1
                counts["caster:"+target["caster"]["state"]] += 1
            else:
                counts["unlabeled"] += 1
            counts["shadow_source:"+row["shadow_label_source"]] += 1
            counts["association_source:"+row["association_source"]] += 1
            counts["association_status:"+row.get("association_status", "unspecified")] += 1
            counts["input_regime:"+row.get("input_regime", "unspecified")] += 1
        report[split] = {"rows": len(rows), "scenes": len({row['scene_name'] for row in rows}), "counts": dict(counts)}
    assert_disjoint(datasets["train"], datasets["dev"]+datasets["eval"])
    if {row["scene_name"] for row in datasets["dev"]} & {row["scene_name"] for row in datasets["eval"]}:
        raise ValueError("Dev and eval scenes overlap")
    write_json(root/"data_check.json", report)
    print(report)


if __name__ == "__main__":
    main()
