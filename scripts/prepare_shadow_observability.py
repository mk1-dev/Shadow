"""Prepare clean RGB triplets, optionally restricted to existing visual-audit cases."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.observability import PROTOCOL, rgb_windows
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True, help="Parent of train/val pair caches")
    parser.add_argument("--audit-root", help="Existing geometry audit's visual directory; cases are deduplicated by window")
    parser.add_argument("--prompt", default="prompts/shadow_observability_v1.txt")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--random-windows", type=int, default=0, help="Additional unscreened windows per split when using --audit-root")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.random_windows < 0 or args.random_windows and not args.audit_root:
        parser.error("random-windows requires audit-root and a nonnegative count")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    rows, links, summaries, source_hashes, checked = [], [], {}, {}, {}
    rng = random.Random(args.seed)
    for split in ("train", "val"):
        manifest = Path(args.source_root)/split/"pairs/pairs.jsonl"
        windows = rgb_windows(read_rows(manifest), manifest.parent, prompt)
        source_hashes[split] = digest(manifest)
        cases = read_rows(Path(args.audit_root)/split/"cases.jsonl") if args.audit_root else []
        requested = {case["window_id"] for case in cases}
        if requested-set(row["id"] for row in windows):
            raise ValueError("Visual audit contains windows absent from these pair manifests")
        chosen = windows
        if args.audit_root:
            random_pool = [row for row in windows if row["id"] not in requested]
            rng.shuffle(random_pool)
            random_ids = {row["id"] for row in random_pool[:args.random_windows]}
            chosen = [{**row, "selection_source": "audit" if row["id"] in requested else "unscreened_random"}
                      for row in windows if row["id"] in requested | random_ids]
        for row in chosen:
            for image, expected in zip(row["images"], row["image_sha256"]):
                if image not in checked:
                    checked[image] = digest(image)
                if checked[image] != expected:
                    raise ValueError("Clean RGB changed since pair export")
        links.extend({"window_id": case["window_id"], "case_id": case["id"], "category": case["category"],
                      "official_split": split} for case in cases)
        summaries[split] = {"all_rgb_windows": len(windows), "audit_cases": len(cases),
            "unique_audit_windows": len(requested), "selected_windows": len(chosen),
            "scenes": len({row["scene_name"] for row in chosen}),
            "selection_sources": dict(Counter(row.get("selection_source", "all_windows") for row in chosen))}
        rows.extend(chosen)
    if not rows:
        raise ValueError("No consecutive clean RGB windows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Window IDs collide across source splits")
    write_rows(output/"manifest.jsonl", rows)
    write_rows(output/"audit_links.jsonl", links)
    write_json(output/"summary.json", summaries)
    write_json(output/"protocol.json", {**vars(args), "protocol": PROTOCOL,
        "pair_manifest_hashes": source_hashes, "prompt_sha256": digest(args.prompt),
        "input_policy": "only clean RGB triplets and fixed prompt reach the model; no audit categories, GT or overlays"})
    print(f"Prepared {len(rows)} unique observability windows in {output}", flush=True)


if __name__ == "__main__":
    main()
