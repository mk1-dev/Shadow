"""Full-image prediction diagnostics, including unlabelled image regions."""
from __future__ import annotations

import cv2
import numpy as np


def prediction_coverage(probability, target, threshold=.5):
    probability, target = np.asarray(probability), np.asarray(target)
    if probability.shape != target.shape or probability.ndim != 2:
        raise ValueError("Probability and trimap must share image coordinates")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.isin(target, [0, 1, 255]).all() or not 0 <= threshold <= 1:
        raise ValueError("Expected a 1/0/255 trimap and valid threshold")
    prediction = probability >= threshold
    result = {"pixels": int(target.size), "predicted_shadow_pixels": int(prediction.sum())}
    for name, mask in (("known", target != 255), ("ignore", target == 255),
                       ("teacher_positive", target == 1), ("teacher_negative", target == 0)):
        result[name+"_pixels"] = int(mask.sum())
        result[name+"_predicted_shadow_pixels"] = int((prediction & mask).sum())
        result[name+"_predicted_shadow_fraction"] = float(prediction[mask].mean()) if mask.any() else None
    result["full_image_predicted_shadow_fraction"] = float(prediction.mean())
    result["ignore_prediction_is_accuracy"] = False
    return result


def review_panel(image, probability, target, threshold=.5, width=640):
    prediction_coverage(probability, target, threshold)
    predicted = image.copy()
    selected = probability >= threshold
    predicted[selected] = (.55*predicted[selected]+.45*np.array([0, 200, 255])).astype(np.uint8)
    teacher = image.copy()
    teacher[target == 1] = [0, 200, 255]
    teacher[target == 0] = [70, 220, 40]
    height = max(1, round(image.shape[0]*width/image.shape[1]))
    panels = []
    for title, value in (("RGB", image), ("Prediction", predicted), ("Weak teacher: yellow=1 green=0", teacher)):
        panel = cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def summarize_coverage(rows):
    scenes = {}
    for row in rows:
        scenes.setdefault(row["scene_name"], []).append(row["full_image_coverage"])
    result = {}
    for scene, values in sorted(scenes.items()):
        total = sum(value["pixels"] for value in values)
        ignored = sum(value["ignore_pixels"] for value in values)
        result[scene] = {"frames": len(values), "pixels": total,
            "known_fraction": sum(value["known_pixels"] for value in values)/total,
            "predicted_shadow_fraction": sum(value["predicted_shadow_pixels"] for value in values)/total,
            "ignore_predicted_shadow_fraction": sum(value["ignore_predicted_shadow_pixels"] for value in values)/ignored if ignored else None}
    return {"frames": len(rows), "by_scene": result,
        "scene_macro_predicted_shadow_fraction": float(np.mean([value["predicted_shadow_fraction"] for value in result.values()])) if result else None,
        "scope": "Whole-image coverage, not accuracy on IGNORE pixels"}
