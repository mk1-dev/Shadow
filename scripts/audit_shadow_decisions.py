from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import read_rows, resolve_path, write_rows
from shadow_occ.annotation.diagnostics import AUDIT_CATEGORIES, decision
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Create an error-review queue; reviews never become training labels automatically.")
    parser.add_argument("--predictions", required=True, help="Score-analysis predictions.jsonl or generation annotations.jsonl")
    parser.add_argument("--manifest", help="Required when generation results omit image paths")
    parser.add_argument("--reviews", help="Optional independent JSONL: id, category, note; NOT shadow GT")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()
    if args.max_images < 0:
        raise ValueError("max-images must be nonnegative")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest = {row["id"]: row for row in read_rows(args.manifest)} if args.manifest else {}
    reviews = read_rows(args.reviews) if args.reviews else []
    if len({row["id"] for row in reviews}) != len(reviews) or any(row["category"] not in AUDIT_CATEGORIES for row in reviews):
        raise ValueError("Duplicate reviews or unsupported audit category")
    reviews = {row["id"]: row for row in reviews}
    queue, counts, seen = [], Counter(), set()
    base = Path(args.manifest or args.predictions).resolve().parent
    for result in read_rows(args.predictions):
        if result["id"] in seen:
            raise ValueError("Duplicate prediction ID")
        seen.add(result["id"])
        row = {**manifest.get(result["id"], {}), **result}
        target = decision(row)
        prediction = row.get("score_prediction") or (row.get("prediction") or {}).get("shadow", "PARSE_ERROR")
        error = "unlabeled" if target not in {"YES", "NO"} else "correct" if prediction == target else "false_positive" if prediction == "YES" else "false_negative" if prediction == "NO" else "abstention_or_parse_error"
        counts[error] += 1
        if error in {"correct", "unlabeled"}:
            continue
        review = reviews.get(row["id"], {"category": "unreviewed", "note": ""})
        item = {"id": row["id"], "scene_name": row["scene_name"], "error_type": error, "target_decision": target, "prediction": prediction,
                "label_source": row.get("shadow_label_source", row.get("label_source")), "category": review["category"], "note": review.get("note", ""),
                "review_is_shadow_gt": False, "images": row.get("images", [])}
        if len(queue) < args.max_images and item["images"]:
            canvas = Image.new("RGB", (800, 340), "white")
            ImageDraw.Draw(canvas).text((8, 8), f"{error} | weak target={target} prediction={prediction}", fill="black")
            indices = (0, min(2, len(item["images"])-1))
            for column, index in enumerate(indices):
                with Image.open(resolve_path(item["images"][index], base)) as image:
                    tile = image.convert("RGB")
                    tile.thumbnail((390, 290))
                    canvas.paste(tile, (column*400+5, 40))
            path = output/"review"/(hashlib.sha256(row["id"].encode()).hexdigest()[:24]+".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path)
            item["review_image"] = str(path)
        queue.append(item)
    if set(reviews)-seen:
        raise ValueError("Review IDs are absent from this prediction run")
    write_rows(output/"review_queue.jsonl", queue)
    write_json(output/"audit_summary.json", {"prediction_counts": dict(counts), "error_categories": dict(Counter(item["category"] for item in queue)),
               "scope": "Weak error audit, not true precision/recall; no pseudo labels are modified", "images_requested": args.max_images})


if __name__ == "__main__":
    main()
