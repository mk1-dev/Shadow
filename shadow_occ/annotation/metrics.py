from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BinaryMetrics:
    confusion: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))

    def update(self, prediction, target) -> None:
        target = np.asarray(target)
        prediction = np.asarray(prediction, dtype=bool)
        valid = (target == 0) | (target == 1)
        self.confusion += np.bincount(target[valid].astype(np.int64)*2 + prediction[valid].astype(np.int64), minlength=4)

    def summary(self) -> dict:
        tn, fp, fn, tp = [int(value) for value in self.confusion]
        ratio = lambda a, b: float(a / b) if b else None
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "iou": ratio(tp, tp+fp+fn), "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn), "f1": ratio(2*tp, 2*tp+fp+fn), "accuracy": ratio(tp+tn, tp+tn+fp+fn), "valid_count": tn+fp+fn+tp}


def auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(labels) or len(np.unique(labels)) != 2:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    return float((ranks[labels == 1].sum() - positive*(positive+1)/2)/(positive*negative))


def region_summary(rows: list[dict], threshold: float, bootstrap: int = 1000, seed: int = 0) -> dict:
    by_source = {}
    for source in sorted({row.get("label_source", "unlabeled") for row in rows}):
        labeled = [row for row in rows if row.get("label_source") == source and row.get("answer") in {"YES", "NO"}]
        if not labeled:
            continue
        labels = np.asarray([row["answer"] == "YES" for row in labeled], dtype=int)
        scores = np.asarray([row["score"] for row in labeled])
        metric = BinaryMetrics()
        metric.update(scores >= threshold, labels)
        by_source[source] = {**metric.summary(), "auroc": auc(labels, scores), "examples": len(labeled)}
    pair_map = {}
    for row in rows:
        if row.get("pair_id") is not None and row.get("group") in {"candidate", "control", "weak_control"}:
            group = "candidate" if row["group"] == "candidate" else "control"
            entry = pair_map.setdefault(str(row["pair_id"]), {})
            # Single-region inputs can be pooled to one side-level YES decision.
            if group not in entry or row["score"] > entry[group]["score"]:
                entry[group] = row
    pairs = [entry for entry in pair_map.values() if {"candidate", "control"} <= entry.keys()]
    separation = {"pairs": len(pairs)}
    if pairs:
        if any(p["candidate"]["scene_name"] != p["control"]["scene_name"] for p in pairs):
            raise ValueError("Weak controls and candidates must come from the same scene")
        differences = np.asarray([int(p["candidate"]["score"] >= threshold)-int(p["control"]["score"] >= threshold) for p in pairs])
        scenes = sorted({p["candidate"]["scene_name"] for p in pairs})
        clusters = {scene: np.asarray([i for i, p in enumerate(pairs) if p["candidate"]["scene_name"] == scene]) for scene in scenes}
        rng = np.random.default_rng(seed)
        sampled = []
        for _ in range(bootstrap):
            indices = np.concatenate([clusters[scenes[i]] for i in rng.integers(0, len(scenes), len(scenes))])
            sampled.append(float(differences[indices].mean()))
        separation.update({"candidate_yes_rate": float(np.mean([p["candidate"]["score"] >= threshold for p in pairs])), "control_yes_rate": float(np.mean([p["control"]["score"] >= threshold for p in pairs])), "yes_difference": float(differences.mean()), "scene_cluster_ci95": np.percentile(sampled, [2.5, 97.5]).tolist() if sampled else None, "scenes": len(scenes), "proxy_group_auroc": auc([1, 0]*len(pairs), [p[group]["score"] for p in pairs for group in ("candidate", "control")])})
    return {"examples": len(rows), "threshold": threshold, "by_label_source": by_source, "weak_pair_separation": separation}
