"""Separate optional manual-GT accuracy from held-out frozen-teacher agreement."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import digest, load_array, read_rows, resolve_path
from shadow_occ.annotation.metrics import BinaryMetrics, pixel_score_diagnostics
from shadow_occ.utils.io import write_json


def observation_key(row):
    return row["scene_name"], row["camera"], row["frame_token"]


def boundary_counts(prediction, target, tolerance):
    valid = target != 255
    safe = cv2.erode(valid.astype(np.uint8), np.ones((2*tolerance+3, 2*tolerance+3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    edge = lambda mask: (mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)) & safe
    predicted, truth = edge(prediction & valid), edge(target == 1)
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    match_pred = predicted & (cv2.dilate(truth.astype(np.uint8), kernel) > 0)
    match_truth = truth & (cv2.dilate(predicted.astype(np.uint8), kernel) > 0)
    return np.array([match_pred.sum(), predicted.sum(), match_truth.sum(), truth.sum()], dtype=np.int64)


def boundary_summary(counts):
    matched_pred, predicted, matched_truth, truth = map(int, counts)
    precision = matched_pred/predicted if predicted else (0.0 if truth else None)
    recall = matched_truth/truth if truth else (0.0 if predicted else None)
    f1 = (2*precision*recall/(precision+recall) if precision+recall else 0.0) if precision is not None and recall is not None else None
    return {"precision": precision, "recall": recall, "f1": f1,
        "matched_prediction": matched_pred, "prediction_boundary_pixels": predicted,
        "matched_truth": matched_truth, "truth_boundary_pixels": truth}


def read_probability(value, base):
    if value is None:
        return None
    return load_array(value if isinstance(value, dict) else {"path": value, "key": "probability"}, base).astype(np.float32)


def evaluate_experiments(gt_path, methods_path, dataset_summary, output_root, freeze_lock=None, threshold=.5, boundary_tolerance=2, bins=4096,
                         reference_kind="manual"):
    if reference_kind not in {"manual", "frozen-teacher"}:
        raise ValueError("Unknown evaluation reference kind")
    weak = reference_kind == "frozen-teacher"
    gt_path, methods_path = Path(gt_path).resolve(), Path(methods_path).resolve()
    data = json.loads(Path(dataset_summary).read_text(encoding="utf-8"))
    exposed = {scene for scene, split in data["scene_roles"].items() if split in {"train", "dev"}}
    exposed |= set(data.get("development_scenes", []))
    if freeze_lock:
        exposed |= set(json.loads(Path(freeze_lock).read_text(encoding="utf-8"))["excluded_development_scenes"])
    truth_rows = read_rows(gt_path)
    if not truth_rows or len({observation_key(row) for row in truth_rows}) != len(truth_rows):
        raise ValueError("Manual evaluation frames must be nonempty and unique")
    for row in truth_rows:
        if weak:
            if row.get("label_source") not in {"temporal_weak", "geometry_weak"} or row.get("split") != "eval":
                raise ValueError("Teacher agreement requires held-out frozen physical weak labels, not model self-predictions")
        elif row.get("label_source") != "manual_shadow_gt" or row.get("split") != "eval" or row.get("independent") is not True:
            raise ValueError("Use independently reviewed manual eval masks, never teacher labels or training GT")
    if exposed & {row["scene_name"] for row in truth_rows}:
        raise ValueError("Manual eval scenes overlap training, model-selection or development scenes")
    methods = read_rows(methods_path)
    if not methods or len({(row["experiment"], row["method"]) for row in methods}) != len(methods):
        raise ValueError("Methods must be nonempty and unique within each experiment")
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not 0 <= threshold <= 1 or boundary_tolerance < 0 or bins < 2:
        raise ValueError("Invalid fixed evaluation settings")
    results = []
    for method in methods:
        if method["experiment"] not in ({"TeacherAgreement"} if weak else {"E1", "E2"}):
            raise ValueError("This evaluator measures discovery and segmentation, not owner or temporal accuracy")
        if method["experiment"] in {"E2", "TeacherAgreement"} and not method.get("training_protocol"):
            raise ValueError("E2 needs the checkpoint training protocol for scene isolation")
        path = resolve_path(method["predictions"], methods_path.parent)
        prediction_rows = read_rows(path)
        predictions = {observation_key(row): row for row in prediction_rows}
        if len(predictions) != len(prediction_rows):
            raise ValueError("Duplicate prediction frames; merge overlapping clips before evaluation")
        if not {observation_key(row) for row in truth_rows} <= predictions.keys():
            raise ValueError("Every method must predict every GT frame; missing frames cannot be silently dropped")
        trained = None
        if method.get("training_protocol"):
            trained = json.loads(resolve_path(method["training_protocol"], methods_path.parent).read_text(encoding="utf-8"))
            overlap = (set(trained["train_scenes"]) | set(trained["dev_scenes"])) & {row["scene_name"] for row in truth_rows}
            if overlap:
                raise ValueError("Checkpoint train/dev scenes overlap manual test scenes")
        state, per_scene = BinaryMetrics(), defaultdict(BinaryMetrics)
        histogram = np.zeros((2, bins), np.int64)
        boundaries = np.zeros(4, np.int64)
        all_scores, support_count, known_count = True, 0, 0
        for row in truth_rows:
            truth = load_array(row["target"], gt_path.parent)
            if row.get("mask_encoding", "trimap") == "binary":
                truth = (truth > 0).astype(np.uint8)
            elif not np.isin(truth, [0, 1, 255]).all():
                raise ValueError("Manual trimaps must retain 0/1/255, or declare binary masks explicitly")
            predicted_row = predictions[observation_key(row)]
            if predicted_row.get("in_sample_annotation") or predicted_row.get("diagnostics_only"):
                raise ValueError("In-sample or quarantined predictions are not main-table results")
            score = read_probability(predicted_row.get("probability"), path.parent)
            if score is not None and score.shape != truth.shape:
                raise ValueError("Score/GT coordinate mismatch; evaluation never resizes GT")
            if predicted_row.get("prediction") is not None:
                prediction = load_array(predicted_row["prediction"], path.parent) > 0
            elif score is not None:
                prediction = score >= float(predicted_row.get("threshold", threshold))
            else:
                raise ValueError("Prediction needs a saved binary mask or a score map")
            if prediction.shape != truth.shape:
                raise ValueError("Prediction/GT coordinate mismatch")
            state.update(prediction, truth)
            per_scene[row["scene_name"]].update(prediction, truth)
            boundaries += boundary_counts(prediction, truth, boundary_tolerance)
            valid = (truth == 0) | (truth == 1)
            known_count += int(valid.sum())
            if score is None:
                all_scores = False
                continue
            supported = np.isfinite(score)
            if predicted_row.get("score_support"):
                support = load_array(predicted_row["score_support"], path.parent)
                if support.shape != truth.shape:
                    raise ValueError("Score support uses different coordinates")
                supported &= support > 0
            if ((score[supported] < 0) | (score[supported] > 1)).any():
                raise ValueError("Ranking scores must be normalized, not raw geometric residuals")
            support_count += int((supported & valid).sum())
            values = np.where(supported, score, 0)
            buckets = np.minimum((values[valid]*bins).astype(np.int64), bins-1)
            histogram += np.bincount(truth[valid].astype(np.int64)*bins+buckets, minlength=2*bins).reshape(2, bins)
        metrics = state.summary()
        if not known_count:
            raise ValueError("Manual eval contains no known pixels")
        diagnostics = pixel_score_diagnostics(histogram) if all_scores else None
        operating = []
        if diagnostics is not None:
            descending = np.cumsum(histogram[:, ::-1], axis=1)
            negatives, positives = histogram.sum(axis=1)
            for fpr in (.001, .005, .01, .05):
                eligible = np.where(descending[0] <= fpr*negatives)[0] if negatives else np.array([], int)
                at = int(eligible[-1]) if len(eligible) else None
                operating.append({"fpr_limit": fpr, "recall": float(descending[1, at]/positives) if at is not None and positives else (0.0 if positives and negatives else None),
                    "threshold": (bins-1-at)/bins if at is not None else None,
                    "scope": "descriptive test-curve comparison, not a selected deployment threshold"})
        scene_results = {}
        for scene, value in per_scene.items():
            item = value.summary()
            item["negative_recall"] = item["tn"]/(item["tn"]+item["fp"]) if item["tn"]+item["fp"] else None
            scene_results[scene] = item
        results.append({"experiment": method["experiment"], "method": method["method"], "frames": len(truth_rows),
            "reference_kind": reference_kind, "is_true_shadow_accuracy": not weak,
            "metric_scope": "student agreement with frozen weak teacher on known pixels" if weak else "independent manual shadow masks",
            "scenes": len(per_scene), **metrics,
            "training_resized_weak_positive_pixels": trained.get("resized_pixel_counts", [None, None])[1] if trained else None,
            "training_scene_count": len(trained["train_scenes"]) if trained else None,
            "negative_recall": metrics["tn"]/(metrics["tn"]+metrics["fp"]) if metrics["tn"]+metrics["fp"] else None,
            "fpr": metrics["fp"]/(metrics["fp"]+metrics["tn"]) if metrics["fp"]+metrics["tn"] else None,
            "positive_coverage": (metrics["tp"]+metrics["fp"])/known_count if known_count else None,
            "score_support_coverage": support_count/known_count if all_scores and known_count else None,
            "boundary": boundary_summary(boundaries), "boundary_tolerance_px": boundary_tolerance,
            "average_precision_binned": diagnostics["approximate_average_precision"] if diagnostics else None,
            "pr_auc_is_quantized": True, "score_bins": bins,
            "score_diagnostics": diagnostics, "recall_at_fpr": operating,
            "per_scene": scene_results,
            "prediction_manifest_sha256": digest(path), "operating_mask": "saved prediction when supplied, otherwise frozen probability threshold"})
    output.mkdir(parents=True)
    write_json(output/"metrics.json", {"ground_truth_manifest_sha256": digest(gt_path),
        "methods_manifest_sha256": digest(methods_path), "dataset_summary_sha256": digest(dataset_summary),
        "freeze_lock_sha256": digest(freeze_lock) if freeze_lock else None,
        "manual_gt_used_for_training": False, "threshold_selected_on_eval": False,
        "reference_kind": reference_kind, "is_true_shadow_accuracy": not weak,
        "missing_scores": "lowest rank (zero); missing prediction frames are errors", "results": results})
    write_json(output/"cross_scene_summary.json", {"reference_kind": reference_kind,
        "is_true_shadow_accuracy": not weak,
        "methods": {row["method"]: {"scene_count": row["scenes"], "per_scene": row["per_scene"]} for row in results}})
    columns = ["method", "frames", "scenes", "training_resized_weak_positive_pixels", "training_scene_count", "accuracy", "balanced_accuracy", "precision", "recall", "negative_recall", "f1", "iou", "negative_iou", "mean_iou", "all_foreground_iou", "fpr", "positive_coverage", "average_precision_binned", "boundary_f1"]
    for experiment in sorted({row["experiment"] for row in results}):
        filename = "table_teacher_agreement.csv" if weak else f"table_{experiment}.csv"
        with (output/filename).open("w", newline="", encoding="utf-8") as handle:
            names = {key: "teacher_agreement_"+key if weak and key not in {"method", "frames", "scenes", "training_resized_weak_positive_pixels", "training_scene_count"} else key for key in columns}
            writer = csv.DictWriter(handle, fieldnames=[names[key] for key in columns])
            writer.writeheader()
            for row in results:
                if row["experiment"] == experiment:
                    writer.writerow({names[key]: row["boundary"]["f1"] if key == "boundary_f1" else row[key] for key in columns})
        if weak:
            import shutil
            shutil.copy2(output/filename, output/"teacher_agreement.csv")
    return {"methods": len(results), "frames_per_method": len(truth_rows), "tables": sorted({row["experiment"] for row in results})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", "--reference", dest="ground_truth", required=True)
    parser.add_argument("--reference-kind", choices=["manual", "frozen-teacher"], default="manual")
    parser.add_argument("--methods", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--freeze-lock")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--score-bins", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(evaluate_experiments(args.ground_truth, args.methods, args.dataset_summary, args.output_root,
        args.freeze_lock, args.threshold, args.boundary_tolerance, args.score_bins, args.reference_kind)), flush=True)


if __name__ == "__main__":
    main()
