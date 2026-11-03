from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_occ.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate a small manually annotated shadow set.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    annotation_path = Path(args.annotations)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = payload.get("annotations", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("annotations JSON must be a list or {'annotations': [...]}.")

    tp = fp = fn = tn = 0
    evaluated = 0
    for row in rows:
        scene = row["scene_name"]
        token = row["sample_token"]
        prediction_path = Path(args.label_root) / scene / f"{token}.npz"
        if not prediction_path.exists():
            continue
        with np.load(prediction_path, allow_pickle=True) as data:
            prediction = np.asarray(data["shadow_mask"], dtype=np.uint8)
        truth = _load_mask(row, annotation_path.parent)
        if truth.shape != prediction.shape:
            truth = cv2.resize(
                truth.astype(np.uint8),
                (prediction.shape[1], prediction.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        valid = np.ones_like(truth, dtype=bool)
        if row.get("valid_mask"):
            valid = _load_mask({"mask_path": row["valid_mask"]}, annotation_path.parent).astype(bool)
            if valid.shape != prediction.shape:
                valid = cv2.resize(
                    valid.astype(np.uint8),
                    (prediction.shape[1], prediction.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        predicted = prediction > 0
        truth = truth > 0
        tp += int(np.count_nonzero(predicted & truth & valid))
        fp += int(np.count_nonzero(predicted & ~truth & valid))
        fn += int(np.count_nonzero(~predicted & truth & valid))
        tn += int(np.count_nonzero(~predicted & ~truth & valid))
        evaluated += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    summary = {
        "annotations": len(rows),
        "evaluated": evaluated,
        "shadow_precision": precision,
        "shadow_recall": recall,
        "shadow_f1": f1,
        "shadow_iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    output = ensure_dir(Path(args.output_root) / "metrics") / "shadow_annotation_metrics.json"
    write_json(output, summary)
    print(summary)
    print(f"wrote {output}")


def _load_mask(row: dict, base: Path) -> np.ndarray:
    if "mask" in row:
        return np.asarray(row["mask"], dtype=np.uint8)
    if "mask_path" not in row:
        raise ValueError("Each annotation needs 'mask' or 'mask_path'.")
    path = Path(row["mask_path"])
    if not path.is_absolute():
        path = base / path
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.uint8)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return (image > 0).astype(np.uint8)


if __name__ == "__main__":
    main()
