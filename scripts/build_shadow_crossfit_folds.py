from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import assert_disjoint, digest, read_rows, write_rows
from shadow_occ.utils.io import write_json


def main():
    parser = argparse.ArgumentParser("Create scene-disjoint cross-fit training/annotation manifests, never using official val.")
    parser.add_argument("--seg-data-root", required=True)
    parser.add_argument("--annotation-data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("At least two folds are required")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    tasks, inputs = {}, {}
    for task, root in (("seg", args.seg_data_root), ("annotation", args.annotation_data_root)):
        paths = {split: Path(root)/f"{task}_{split}.jsonl" for split in ("train", "dev")}
        rows = {split: read_rows(path) for split, path in paths.items()}
        assert_disjoint(rows["train"], rows["dev"])
        if not rows["dev"] or any(row.get("split") != "dev" or row.get("official_split") in {"val", "mini_val", "test", "eval"} for row in rows["dev"]):
            raise ValueError("Use a nonempty train-derived dev split, never official val")
        tasks[task] = rows
        inputs.update({str(path): digest(path) for path in paths.values()})
    train_scenes = {row["scene_name"] for rows in tasks.values() for row in rows["train"]}
    dev_scenes = {row["scene_name"] for rows in tasks.values() for row in rows["dev"]}
    if train_scenes & dev_scenes or len(train_scenes) < args.folds:
        raise ValueError("Too few training scenes or segmentation/VLM train-dev scene mismatch")
    ordered = sorted(train_scenes, key=lambda scene: hashlib.sha256(f"{args.seed}:{scene}".encode()).hexdigest())
    assignment = {scene: index % args.folds for index, scene in enumerate(ordered)}
    manifest = {}
    for fold in range(args.folds):
        counts = {}
        for task, rows in tasks.items():
            train = [row for row in rows["train"] if assignment[row["scene_name"]] != fold]
            heldout = [{**row, "annotation_role": "crossfit_holdout", "crossfit_fold": fold} for row in rows["train"] if assignment[row["scene_name"]] == fold]
            if not train or not heldout:
                raise ValueError(f"Empty {task} train/heldout fold {fold}")
            assert_disjoint(train, heldout+rows["dev"])
            for name, selected in (("train", train), ("dev", rows["dev"]), ("heldout", heldout)):
                write_rows(output/f"fold_{fold}"/f"{task}_{name}.jsonl", selected)
                counts[f"{task}_{name}"] = len(selected)
        manifest[str(fold)] = counts
    write_json(output/"crossfit_protocol.json", {**vars(args), "input_sha256": inputs, "scene_assignment": assignment, "counts": manifest,
                                              "policy": "Train each fold separately, infer only its heldout scenes, concatenate unique heldout IDs. Heldout outputs are training pseudo labels, not final evaluation."})


if __name__ == "__main__":
    main()
