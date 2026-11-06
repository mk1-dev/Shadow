from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.annotation.optical_flow import DenseFlow
from shadow_occ.annotation.physics_motion import EvidenceConfig, pair_evidence, sample_at
from shadow_occ.utils.io import write_json


def diagnostic_panel(image_t, image_next, row, evidence):
    shape = image_t.shape[:2]
    yy, xx = np.indices(shape, dtype=np.float32)
    endpoints = np.stack([xx, yy], axis=-1)+evidence["ground_flow"]
    aligned = sample_at(image_next, endpoints)
    aligned[~evidence["visible_ground"]] = 0
    delta = np.clip((evidence["delta_L"]+20)*255/40, 0, 255).astype(np.uint8)
    delta = cv2.applyColorMap(delta, cv2.COLORMAP_JET)
    delta[~evidence["visible_ground"]] = 0
    residual = cv2.applyColorMap(np.clip(np.nan_to_num(evidence["flow_residual_px"])*255/10, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    residual[~evidence["visible_ground"]] = 0
    candidates = np.zeros_like(image_t)
    candidates[evidence["onset_candidate"]] = (255, 100, 0)
    candidates[evidence["recovery_candidate"]] = (0, 100, 255)
    candidates[evidence["stable_dark_candidate"]] = (0, 220, 0)
    vectors = image_t.copy()
    for obj in row["objects"]:
        for p, displacement, valid in zip(obj["pixels_t"], obj["total_displacement_px"], obj["valid"]):
            if not valid or any(item is None for item in p+displacement):
                continue
            start = np.rint(p).astype(int)
            end = np.rint(np.asarray(p)+displacement).astype(int)
            if (0 <= start[0] < shape[1] and 0 <= start[1] < shape[0]
                    and 0 <= end[0] < shape[1] and 0 <= end[1] < shape[0]):
                cv2.arrowedLine(vectors, tuple(start), tuple(end), (0, 255, 255), 1, tipLength=0.2)
    panels = []
    for image, label in ((image_t, "t: clean"), (image_next, "next: clean"),
                         (aligned, "next on t: supported ground"), (delta, "delta L: blue onset / red recovery"),
                         (residual, "flow minus ground geometry: NOT shadow GT"),
                         (candidates, "candidates: blue onset / red recovery / green stable"),
                         (vectors, "GT box landmarks: total apparent displacement")):
        panel = cv2.resize(image, (480, int(shape[0]*480/shape[1])))
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(panel, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        panels.append(panel)
    panels.append(np.zeros_like(panels[0]))
    return np.vstack([np.hstack(panels[index:index+2]) for index in range(0, 8, 2)])


def validate_pair(row):
    if row.get("protocol") != "physics-motion-evidence-v1" or row.get("train_approved") is not False:
        raise ValueError("Expected an unapproved Stage 1/2 pair, not training data")
    if len(row.get("frames", [])) != 2 or not 0 < float(row["dt_seconds"]) <= 0.75:
        raise ValueError("Expected two ordered, nearby camera frames")
    if row.get("official_split") not in {"train", "val", "mini_train", "mini_val"}:
        raise ValueError("Official scene split is required")
    dt = (row["frames"][1]["camera_timestamp"]-row["frames"][0]["camera_timestamp"])/1e6
    if not np.isclose(dt, row["dt_seconds"], atol=1e-9):
        raise ValueError("Pair interval must use actual camera timestamps")


def main():
    parser = argparse.ArgumentParser("Measure Stage 1/2 evidence; never export accepted shadow labels.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--flow-backend", choices=["dis", "raft-small", "none"], default="dis")
    parser.add_argument("--flow-weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-background", type=int, default=128)
    parser.add_argument("--light-threshold", type=float, default=6.0, help="Lab L units in [0, 100]")
    parser.add_argument("--review-limit", type=int, default=20)
    parser.add_argument("--receiver-support-radius", type=int, default=7)
    parser.add_argument("--receiver-support-min-pixels", type=int, default=16)
    parser.add_argument("--receiver-support-min-fraction", type=float, default=0.6)
    args = parser.parse_args()
    if args.review_limit < 0:
        raise ValueError("Review limit cannot be negative")
    config = EvidenceConfig(min_background=args.min_background, light_threshold_L=args.light_threshold,
        receiver_support_radius=args.receiver_support_radius,
        receiver_support_min_pixels=args.receiver_support_min_pixels,
        receiver_support_min_fraction=args.receiver_support_min_fraction)
    config.validate()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    engine = DenseFlow(args.flow_backend, weights=args.flow_weights, device=args.device) if args.flow_backend != "none" else None
    base = Path(args.pairs).resolve().parent
    rows = read_rows(args.pairs)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Pair manifest must be nonempty with unique IDs")
    counts, stats, candidates = Counter(), [], []
    for index, row in enumerate(tqdm(rows, desc="physics flow + light evidence", unit="pair", mininterval=5)):
        validate_pair(row)
        images, grounds = [], []
        for frame in row["frames"]:
            image_path = resolve_path(frame["image"], base)
            ground_path = resolve_path(frame["ground_support"], base)
            if digest(image_path) != frame["image_sha256"] or digest(ground_path) != frame["ground_sha256"]:
                raise ValueError("Pair image or ground support changed since export")
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            images.append(image)
            grounds.append(load_array(str(ground_path), base) > 0)
        if engine is None:
            forward = np.full((*images[0].shape[:2], 2), np.nan, np.float32)
            backward = forward.copy()
        else:
            forward, backward = engine(images[0], images[1]), engine(images[1], images[0])
        evidence = pair_evidence(*images, *grounds,
            row["frames"][0]["projection_world_to_image"], row["frames"][1]["projection_world_to_image"],
            np.asarray(row["plane_basis_world"], np.float64), forward, backward, config, use_observed_flow=engine is not None)
        stem = row["id"]
        arrays = {key: value.astype(np.uint8) if value.dtype == bool else value.astype(np.float32)
                  for key, value in evidence.items() if isinstance(value, np.ndarray)}
        path = output/"evidence"/f"{stem}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, flow_t_to_next=forward, flow_next_to_t=backward, **arrays)
        details = {"id": row["id"], "scene_name": row["scene_name"], "camera": row["camera"],
                   "official_split": row["official_split"], "dt_seconds": row["dt_seconds"],
                   "motion_track_count": len(row["objects"]), **evidence["details"],
                   "evidence": str(path), "evidence_sha256": digest(path)}
        stats.append(details)
        counts[details["status"]] += 1
        for kind in ("onset", "recovery", "stable_dark"):
            count, ids, areas, _ = cv2.connectedComponentsWithStats(evidence[f"{kind}_candidate"].astype(np.uint8), connectivity=8)
            for component in range(1, count):
                if areas[component, cv2.CC_STAT_AREA] < 16:
                    continue
                component_mask = ids == component
                mask_path = output/"candidates"/f"{stem}_{kind}_{component}.png"
                checked_write_image(mask_path, component_mask.astype(np.uint8)*255)
                candidates.append({"id": f"{stem}_{kind}_{component}", "pair_id": stem,
                    "scene_name": row["scene_name"], "official_split": row["official_split"],
                    "kind": kind, "mask": str(mask_path), "coordinate_frame": "receiver_pixels_at_t",
                    "dark_observation_frame": "next" if kind == "onset" else "t",
                    "shadow_label": "IGNORE", "decision": "UNCERTAIN", "caster_id": None,
                    "train_approved": False, "confidence": None, "probability_calibrated": False})
                counts[f"{kind}_components"] += 1
        if index < args.review_limit:
            checked_write_image(output/"review"/f"{stem}.jpg", diagnostic_panel(*images, row, evidence))
        if (index+1) % 5 == 0 or index+1 == len(rows):
            print(f"completed={index+1}/{len(rows)} scene={row['scene_name']} status={details['status']} "
                  f"ground={details['visible_ground_pixels']} onset={details['onset_candidate_pixels']} "
                  f"recovery={details['recovery_candidate_pixels']}", flush=True)
    write_rows(output/"pair_metrics.jsonl", stats)
    write_rows(output/"candidates.jsonl", candidates)
    summary = {"pairs": len(rows), "scenes": len({row["scene_name"] for row in rows}),
               "counts": dict(counts), "shadow_precision": None, "shadow_recall": None,
               "shadow_iou": None, "caster_accuracy": None, "train_approved": False,
               "interpretation": "candidate/evidence diagnostics only; no independent shadow ground truth"}
    write_json(output/"summary.json", summary)
    write_json(output/"analysis_protocol.json", {**vars(args), "config": asdict(config),
               "manifest_sha256": digest(args.pairs), "opencv_version": cv2.__version__,
               "flow_weights_sha256": digest(args.flow_weights) if args.flow_weights else None,
               "protocol": "physics-motion-evidence-v1", "coordinate_frame": "all evidence arrays use image t pixels",
               "flow_evidence_version": "receiver-support-v2",
               "registration_mode": "geometry-only" if engine is None else "geometry-dis",
               "flow_role": "absent; geometry-only alignment and exposure" if engine is None else
                   "local textured receiver registration; boundary flow is not shadow/material ground truth",
               "delta_definition": "exposure_compensated_L_next_at_same_ground_minus_L_t",
               "exposure_assumption": "majority stable lighting among flow-consistent sparse supported ground",
               "not_implemented": ["causal_three_frame_recovery", "light_direction_estimation", "caster_shadow_association",
                                   "calibrated_shadow_probability", "automatic_positive_negative_training_labels"],
               "sources": ["https://docs.opencv.org/4.1.1/javadoc/org/opencv/video/DISOpticalFlow.html",
                           "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_small.html"]})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
