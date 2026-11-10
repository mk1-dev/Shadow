"""Freeze same-label v9.3/v9.4/v9.5 paired student comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.utils.io import write_json


def prepare(baseline_suite, layered_suite, output_root, seeds=(0, 1, 2)):
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Choose unique nonnegative seeds")
    baseline, layered = [json.loads(Path(path).read_text(encoding="utf-8")) for path in (baseline_suite, layered_suite)]
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    source_arms = []
    for suite in (baseline, layered):
        source_arms.append(next(arm for arm in suite["arms"] if arm["method"] == "motion_illumination"))
    if baseline["scene_sets"] != layered["scene_sets"]:
        raise ValueError("Paired loss comparison requires identical scene sets")
    if baseline["weak_eval_reference_sha256"] != layered["weak_eval_reference_sha256"]:
        raise ValueError("Paired losses require one unchanged eval reference")
    for suite in (baseline, layered):
        if digest(suite["weak_eval_reference"]) != suite["weak_eval_reference_sha256"]:
            raise ValueError("Frozen evaluation reference changed")
    aligned, evidence = {}, {}
    for split in ("train", "dev"):
        indexed = []
        for arm in source_arms:
            rows = read_rows(arm[split])
            index = {row["id"]: row for row in rows}
            if len(index) != len(rows):
                raise ValueError("Duplicate training observations")
            indexed.append(index)
        if indexed[0].keys() != indexed[1].keys():
            raise ValueError("Different paired observation IDs")
        evidence[split] = []
        for key in sorted(indexed[0]):
            a, b = [index[key] for index in indexed]
            if (a["scene_name"], a["frame_token"]) != (b["scene_name"], b["frame_token"]):
                raise ValueError("Paired IDs refer to different frames")
            if digest(a["target"]) != digest(b["target"]) or digest(a["image"]) != digest(b["image"]):
                raise ValueError("Binary labels or source RGB differ")
            for field in ("sample_weight", "negative_weight", "label_source"):
                if a.get(field) != b.get(field):
                    raise ValueError("Non-pixel training reliability differs: "+field)
            evidence[split].append({"id": key, "target_sha256": digest(a["target"]), "image_sha256": digest(a["image"])})
        for at, index in enumerate(indexed):
            aligned[at, split] = [index[key] for key in sorted(index)]
    arms, hashes = [], {}
    output.mkdir(parents=True)
    for group, index, mode in (("A_v93", 0, "v93"), ("B_v94", 1, "v94"), ("C_v95", 1, "continuous")):
        paths = {}
        for split in ("train", "dev"):
            path = output/group/("seg_"+split+".jsonl")
            write_rows(path, aligned[index, split])
            paths[split] = str(path)
            hashes[str(path)] = digest(path)
        for seed in seeds:
            arms.append({"method": group+"_seed"+str(seed), **paths, "comparison_group": group,
                         "training": {"seed": seed, "supervision_mode": mode}})
    plan = {**layered, "arms": arms, "manifest_sha256": hashes, "paired_initialization": True,
        "init_checkpoint": None, "init_checkpoint_sha256": None, "deterministic": True,
        "amp": "off", "workers": 0, "inherited_weight": .7, "expanded_weight": .25,
        "protocol": "same-label-continuous-supervision-v9.5", "seeds": list(seeds),
        "source_suite_sha256": {str(Path(path).resolve()): digest(path) for path in (baseline_suite, layered_suite)},
        "binary_observations": evidence,
        "comparison_contract": "identical RGB/trimaps/scene roles/sample reliability; A uses v9.3 sidecars, B/C share v9.4 sidecars; no teacher rebuild"}
    write_json(output/"suite.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-suite", required=True)
    parser.add_argument("--layered-suite", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    plan = prepare(args.baseline_suite, args.layered_suite, args.output_root, args.seeds)
    print(json.dumps({"arms": len(plan["arms"]), "seeds": plan["seeds"]}))
