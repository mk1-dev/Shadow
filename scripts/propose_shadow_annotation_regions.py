from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import checked_write_image, read_rows, resolve_path, write_rows
from shadow_occ.utils.io import write_json
from shadow_occ.annotation.readiness import require_prediction_for_proposals


def main():
    parser = argparse.ArgumentParser("Propose observed-ground regions from segmentation, including previously unlabeled pixels.")
    parser.add_argument("--seg-predictions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--max-regions", type=int, default=10)
    args = parser.parse_args()
    if not 0.5 < args.threshold <= 1 or min(args.min_area, args.max_regions) < 1:
        raise ValueError("Invalid proposal thresholds")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    base = Path(args.seg_predictions).resolve().parent
    rows, seen = [], set()
    counts = {"examples": 0, "regions": 0, "zero_proposal_examples": 0, "missing_ground_support": 0, "regions_touching_unknown": 0}
    for row in read_rows(args.seg_predictions):
        require_prediction_for_proposals(row)
        if row["id"] in seen:
            raise ValueError("Duplicate segmentation parent ID")
        seen.add(row["id"])
        row = dict(row)
        for key in ("image", "target", "probability", "ground_support"):
            if row.get(key):
                row[key] = str(resolve_path(row[key], base))
        target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(row["target"])
        with np.load(row["probability"], allow_pickle=False) as archive:
            probability = np.asarray(archive["probability"], np.float32)
        if probability.shape != target.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Invalid segmentation probabilities or coordinates")
        ground = cv2.imread(row["ground_support"], cv2.IMREAD_GRAYSCALE) if row.get("ground_support") else None
        regions = []
        if ground is None:
            counts["missing_ground_support"] += 1
            if row.get("ground_support"):
                raise FileNotFoundError(row["ground_support"])
        else:
            if ground.shape != target.shape:
                raise ValueError("Ground support and prediction coordinates differ")
            count, labels, stats, _ = cv2.connectedComponentsWithStats(((probability >= args.threshold) & (ground > 0)).astype(np.uint8), connectivity=8)
            indices = sorted(range(1, count), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
            for index in indices:
                if stats[index, cv2.CC_STAT_AREA] < args.min_area or len(regions) >= args.max_regions:
                    continue
                mask = labels == index
                stem = hashlib.sha256(f"{row['id']}:{index}".encode()).hexdigest()[:24]
                path = output/"regions"/f"{stem}.png"
                checked_write_image(path, mask.astype(np.uint8)*255)
                regions.append({"id": f"prediction_{index}", "mask": str(path), "shadow_label": "IGNORE", "label_source": "model_pseudo",
                                "association_source": "unresolved", "association_status": "unsupported", "proposal_source": "segmenter_observed_ground"})
                counts["regions_touching_unknown"] += int(np.any(mask & (target == 255)))
        rows.append({**row, "shadow_mask": row["target"], "mask_encoding": "trimap", "regions": regions, "proposal_inference_only": True})
        counts["examples"] += 1
        counts["regions"] += len(regions)
        counts["zero_proposal_examples"] += int(not regions)
    write_rows(output/"manifest.jsonl", rows)
    write_json(output/"proposal_protocol.json", {**vars(args), "counts": counts, "policy": "Candidates are unlabeled, not automatic shadow positives. Missing observed-ground evidence never becomes foreground."})


if __name__ == "__main__":
    main()
