from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import assert_disjoint, digest, read_rows
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("CPU data checks to run on HPC before model training.")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    summary, datasets = {}, {}
    for task in ("seg", "vlm"):
        datasets[task] = {}
        for split in ("train", "dev", "eval"):
            path = root/f"{task}_{split}.jsonl"
            rows = read_rows(path)
            datasets[task][split] = rows
            ids, counts = set(), Counter()
            for row in rows:
                if row["id"] in ids or row["split"] != split:
                    raise ValueError(f"Duplicate ID or wrong split at {path}: {row['id']}")
                ids.add(row["id"])
                counts[row["label_source"]] += 1
                if task == "seg":
                    image = cv2.imread(row["image"])
                    target = cv2.imread(row["target"], cv2.IMREAD_GRAYSCALE)
                    if image is None or target is None or image.shape[:2] != target.shape:
                        raise ValueError(f"Missing/misaligned image and target: {row['id']}")
                    if not np.isin(target, [0, 1, 255]).all():
                        raise ValueError("Target must contain 0, 1, 255")
                    for value, name in ((0, "known_background_pixels"), (1, "known_shadow_pixels"), (255, "ignored_pixels")):
                        counts[name] += int((target == value).sum())
                else:
                    for image_path in row["images"]:
                        if not Path(image_path).is_file():
                            raise FileNotFoundError(image_path)
                    if row["task"] == "shadow":
                        counts[str(row.get("answer"))] += 1
                        expected = 5 if row.get("physics_temporal_context") else 4 if row.get("temporal_context") else 3
                        if len(row["images"]) != expected:
                            raise ValueError("Incorrect clean/reference/context/temporal image count")
                        counts["with_temporal_view" if row.get("temporal_context") else "without_temporal_view"] += 1
                if row.get("temporal"):
                    temporal = row["temporal"]
                    if temporal["previous_scene_name"] != row["scene_name"]:
                        raise ValueError("Temporal pair crosses scene/split boundary")
                    if temporal["previous_instance_token"] != row.get("instance_token") or temporal["previous_camera"] != row.get("camera"):
                        raise ValueError("Temporal pair changes caster identity or camera")
                    if not Path(temporal["previous_image"]).is_file() or not Path(temporal["evidence"]).is_file():
                        raise FileNotFoundError("Missing temporal image or evidence")
                    matrix = np.asarray(temporal["caster_previous_to_current"])
                    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                        raise ValueError("Invalid temporal homography")
            summary[f"{task}_{split}"] = {"rows": len(rows), "scenes": len({row['scene_name'] for row in rows}), "sha256": digest(path), "counts": dict(counts)}
        assert_disjoint(datasets[task]["train"], datasets[task]["dev"]+datasets[task]["eval"])
        if {row["scene_name"] for row in datasets[task]["dev"]} & {row["scene_name"] for row in datasets[task]["eval"]}:
            raise ValueError("Dev and eval scenes overlap")
    write_json(root/"data_check.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
