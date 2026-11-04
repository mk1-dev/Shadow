from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import checked_write_image, digest, read_rows, resolve_path, write_rows
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Make matched-input shadow diagnostics without changing labels or inventing GT.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=["full", "region", "context", "occlude-region"], required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    base, results = Path(args.manifest).resolve().parent, []
    for row in read_rows(args.manifest):
        if row.get("task") != "shadow_annotation" or len(row["images"]) not in {3, 4}:
            raise ValueError("Expected clean/reference/context[/previous] unified inputs")
        paths = []
        mask = None
        if args.mode == "occlude-region":
            mask_path = resolve_path(row["region_mask"], base)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            mask = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            crops = row.get("image_crops_xyxy") or [row.get("tight_crop_xyxy"), row.get("tight_crop_xyxy"), row.get("context_crop_xyxy")]
            if any(crop is None for crop in crops):
                raise ValueError("Legacy data lacks exact crop coordinates; rebuild rather than approximate an occlusion box")
        for index, source in enumerate(row["images"]):
            source = resolve_path(source, base)
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(source)
            if (args.mode == "region" and index >= 2) or (args.mode == "context" and index == 3):
                image[:] = 127
            elif mask is not None:
                crop = crops[index] if index < 3 else row.get("previous_context_crop_xyxy", crops[2])
                x0, y0, x1, y1 = map(int, crop)
                local = mask[y0:y1, x0:x1]
                if local.shape != image.shape[:2]:
                    raise ValueError("Exact crop and input image dimensions disagree")
                image[local] = 127
            name = hashlib.sha256(f"{row['id']}:{index}".encode()).hexdigest()[:24]
            path = output/"views"/f"{name}.png"
            checked_write_image(path, image)
            paths.append(str(path))
        results.append({**row, "images": paths, "view_mode": args.mode, "ablation_source_manifest_sha256": digest(args.manifest),
                        "association_evaluation_allowed": args.mode == "full", "ablation_scope": "shadow decision only; masking is an out-of-distribution diagnostic, not a causal proof"})
    write_rows(output/"manifest.jsonl", results)
    write_json(output/"ablation_protocol.json", {**vars(args), "rows": len(results), "policy": "Fixed image slots, gray masked views, unchanged prompts/targets. Context removes only history; region removes context AND history."})


if __name__ == "__main__":
    main()
