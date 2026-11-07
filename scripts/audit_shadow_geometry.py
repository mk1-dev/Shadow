"""Sample failure types and render RGB/flow/ground/light-projection audits on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.bev_flow import bev_mask_to_image
from shadow_occ.annotation.caster_window import caster_proposals, extend_pairs, long_observations, transition_stage_accounting
from shadow_occ.annotation.data import checked_write_image, digest, load_array, read_rows, resolve_path, write_rows
from shadow_occ.annotation.physics_io import load_motion_analysis, read_evidence
from shadow_occ.annotation.physics_labels import light_bank, window_tracks
from shadow_occ.annotation.physics_motion import EvidenceConfig, project_world
from shadow_occ.annotation.receiver_review import (
    exposure_review, flow_image, light_projection_review, mask_image,
    receiver_visibility_review, sheet, tile, transition_projection_review,
)
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.utils.io import write_json


def case_pool(windows, decisions):
    pools = defaultdict(list)
    for window in windows:
        for category, stage in (("no_visible_overlap", "no_visible_overlap"), ("flow_chain_zero", "flow_chain_removed_support")):
            comparisons = [frame for frame in window["frames"] if not frame["is_reference"] and frame["stage"] == stage]
            if comparisons:
                pools[category].append({"id": window["window_id"], "window_id": window["window_id"],
                    "scene_name": window["scene_name"], "category": category,
                    "frame_index": comparisons[0]["frame_index"], "component": None})
    for item in decisions:
        diagnostic = item.get("transition_diagnostics") or {}
        long_window = diagnostic.get("long_window") or {}
        removed = [frame for frame in long_window.get("frames", [])
            if frame["raw_darkening_pixels"] > 0 and frame["normalized_darkening_pixels"] == 0]
        hypotheses = [value for parent in item["hypotheses"] for value in [parent, *parent.get("alternatives", [])]]
        outside = any(event["reason"] == "outside_current_footprint" for hypothesis in hypotheses
                      for event in (hypothesis.get("long_window") or {}).get("event_alignment", []))
        categories = ([] if not removed else ["exposure_removed"])+([] if not outside else ["outside_footprint"])
        if item["exported_label"] == "YES":
            categories.append("accepted")
        for category in categories:
            pools[category].append({"id": item["id"], "window_id": item["bev_flow"]["window_id"],
                "scene_name": item["scene_name"], "category": category,
                "frame_index": removed[0]["frame_index"] if category == "exposure_removed" else None,
                "component": item})
    return pools


def sample_cases(pools, limit, seed):
    rng, selected = np.random.default_rng(seed), []
    for category in sorted(pools):
        scenes = defaultdict(list)
        for item in pools[category]:
            scenes[item["scene_name"]].append(item)
        names = sorted(scenes)
        rng.shuffle(names)
        for values in scenes.values():
            rng.shuffle(values)
        chosen = []
        while len(chosen) < limit and any(scenes.values()):
            for name in names:
                if scenes[name] and len(chosen) < limit:
                    chosen.append(scenes[name].pop())
        selected.extend(chosen)
    return selected


def render_case(case, window, by_id, forward, backward, source, output, radius, stats):
    base = source/"pairs"
    evidence_path = resolve_path(window["evidence"], source/"bev_flow")
    if digest(evidence_path) != window["evidence_sha256"]:
        raise ValueError("BEV window changed")
    with np.load(evidence_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    left, right = (by_id[token] for token in window["pair_ids"])
    tracks, rejected_tracks = window_tracks(left, right)
    frame = right["frames"][0]
    projection = np.asarray(frame["projection_world_to_image"])
    grid = BEVGrid(**window["grid"])
    roi, _ = caster_proposals(arrays, tracks, light_bank(), left["plane_world"], arrays["bev_world_to_grid"], {}, expand=False)
    chain = extend_pairs(left, right, forward, backward, radius)
    context = long_observations(chain, frame["sample_data_token"], arrays, roi, base, stats, source/"flow",
                                include_observations=True)
    component = case["component"]
    observed = arrays["receiver_valid"] & (arrays["appearance_local_ratio"][1] < .88)
    if component is not None:
        reference = component["bev_flow"]
        path = resolve_path(reference["component"], source/"labels")
        if digest(path) != reference["component_sha256"]:
            raise ValueError("Component mask changed")
        with np.load(path, allow_pickle=False) as archive:
            observed = archive["component_mask"].astype(bool)
    central_frames = [left["frames"][0], frame, right["frames"][1]]
    images = [cv2.imread(str(resolve_path(value["image"], base))) for value in central_frames]
    if any(image is None for image in images):
        raise FileNotFoundError("Missing central RGB observation")
    image = images[1]
    window_images = [cv2.imread(str(resolve_path(value["image"], base))) for value in context["frames"]]
    window_grounds = [load_array(value["ground_support"], base) > 0 for value in context["frames"]]
    checked_write_image(output/"receiver_visibility.jpg", receiver_visibility_review(
        window_images, window_grounds, context, arrays["receiver_world_xyz"], arrays["receiver_valid"]))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"observations.npz", receiver_world_xyz=arrays["receiver_world_xyz"],
        visibility_by_frame=context["visible"], cached_central_visibility=arrays["visible_by_frame"],
        cached_receiver_valid=arrays["receiver_valid"], chain_support=context["flow_support"],
        onset=context["onset"], recovery=context["recovery"], lightness=context["lightness"], observed_component=observed)
    image_mask = bev_mask_to_image(observed, projection, left["plane_basis_world"], grid, image.shape[:2])
    if component is not None:
        image_path = resolve_path(component["mask"], source/"labels")
        if digest(image_path) != component["mask_sha256"]:
            raise ValueError("Source image component mask changed")
        image_mask = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image_mask is None or image_mask.shape != image.shape[:2]:
            raise ValueError("Source component mask and image pixel frame differ")
        image_mask = image_mask > 0
    focus = None
    if tracks:
        # Pick spatial context, not a successful hypothesis. All tracks remain in the overview.
        center = np.median(arrays["receiver_world_xyz"][observed], axis=0) if observed.any() else arrays["receiver_world_xyz"].reshape(-1, 3).mean(axis=0)
        focus = min(tracks, key=lambda track: (np.linalg.norm(track["points"][1, 0]-center), track["instance_token"]))["instance_token"]
    overview, atlas, projections = light_projection_review(image, image_mask, tracks, left["plane_world"], projection, focus)
    checked_write_image(output/"light_bank_96.jpg", atlas)
    checked_write_image(output/"dark_gt_light_overlay.jpg", overview)
    alignment, alignment_details = transition_projection_review(context, focus, left["plane_world"], arrays["bev_world_to_grid"], observed)
    if alignment is not None:
        checked_write_image(output/"event_projection_alignment.jpg", alignment)
    compare = case["frame_index"]
    if compare is None:
        choices = [index for index in range(len(context["frames"])) if index != context["middle"]]
        compare = max(choices, key=lambda index: int(((context["onset"][index] | context["recovery"][index]) & observed).sum()))
    exposure_panels, exposure_details = exposure_review(context, compare, observed, roi)
    checked_write_image(output/"exposure.jpg", sheet(exposure_panels, 3))
    pair_index = min(compare, len(chain)-1)
    flow, metric = read_evidence(chain[pair_index], stats, source/"flow")
    flow_config = json.loads((source/"flow/analysis_protocol.json").read_text(encoding="utf-8")).get("config", {})
    tolerance = flow_config.get("ground_flow_tolerance_px", EvidenceConfig().ground_flow_tolerance_px)
    raw_flow = flow["flow_t_to_next"]
    masks = [("Raw DIS finite (source pixels)", np.isfinite(raw_flow).all(axis=-1)),
        ("Forward/backward valid (source pixels)", flow["flow_reliable"] > 0),
        ("Geometry-consistent (source pixels)", (flow["flow_residual_px"] <= tolerance) & (flow["visible_ground"] > 0)),
        ("Texture support (source pixels)", flow["receiver_texture"] > 0),
        ("Receiver visibility (source pixels)", flow["visible_ground"] > 0),
        ("Pairwise receiver support (source pixels)", flow["receiver_flow_supported"] > 0),
        ("Chain receiver support (BEV grid)", context["flow_support"][compare])]
    stages = [tile(flow_image(raw_flow), "DIS: source image pixels per interval")]
    stages.extend(tile(mask_image(mask), title) for title, mask in masks)
    checked_write_image(output/"flow_stages.jpg", sheet(stages, 4))
    np.savez_compressed(output/"flow_stages.npz", flow_raw=raw_flow,
        flow_raw_finite=masks[0][1], flow_forward_backward_valid=masks[1][1],
        flow_geometric_consistency=masks[2][1], flow_texture_support=masks[3][1],
        flow_receiver_visibility=masks[4][1], flow_pairwise_valid=masks[5][1], flow_chain_valid_bev=masks[6][1])
    ground = load_array(frame["ground_support"], base) > 0
    ground_view = image.copy()
    ground_view[ground] = (.65*ground_view[ground]+.35*np.array([80, 230, 80])).astype(np.uint8)
    uv, _, good = project_world(arrays["receiver_world_xyz"].reshape(-1, 3), projection)
    good &= (uv >= 0).all(axis=-1) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] < image.shape[0])
    indices = np.flatnonzero(good)
    for index in indices[::max(1, len(indices)//2000)]:
        cv2.circle(ground_view, tuple(np.rint(uv[index]).astype(int)), 1, (0, 200, 255), -1)
    panels = [tile(value, title) for value, title in zip(images, ("RGB previous", "RGB reference", "RGB next"))]
    panels += [tile(flow_image(raw_flow), f"DIS pair {pair_index}: source frame"),
        tile(ground_view, "Green=sparse ground; yellow=projected grid"),
        tile(overview, "Cyan=dark; green=GT; heat=96-ray coverage")]
    panels += exposure_panels
    checked_write_image(output/"panel.jpg", sheet(panels, 3))
    write_json(output/"audit.json", {"case_id": case["id"], "category": case["category"],
        "scene_name": case["scene_name"], "window_id": case["window_id"], "focus_track": focus,
        "light_projection_records": projections, "transition_projection": alignment_details,
        "exposure": exposure_details, "sampling_and_exposure_frames": context["frame_diagnostics"],
        "gt_tracks": [{**track, "points": track["points"].tolist()} for track in tracks],
        "gt_state_inputs": [{"pair_id": pair["id"],
            "camera_timestamps": [value["camera_timestamp"] for value in pair["frames"]],
            "objects": [{key: item[key] for key in ("instance_token", "category", "source",
                "annotation_token_t", "annotation_token_next", "interpolation_fraction_t", "interpolation_fraction_next")
                if key in item} for item in pair["objects"]]} for pair in (left, right)],
        "rejected_tracks": rejected_tracks, "plane_world": left["plane_world"],
        "projection_world_to_image": projection.tolist(), "bev_world_to_grid": arrays["bev_world_to_grid"].tolist(),
        "timestamps": context["timestamps"], "flow_metric": metric,
        "flow_geometric_tolerance_px": tolerance,
        "flow_stage_coordinate_frames": "all flow_stages.npz maps use source image pixels except flow_chain_valid_bev",
        "light_convention": "world-space downward propagation ray; azimuth 0=world +x, positive toward +y; degrees",
        "footprint_definition": "bounding-volume cast shadow on receiver plane, not vehicle ground occupancy or true silhouette",
        "visibility_policy": "cached receiver requires central three views; each remote frame uses its own overlap and intervening flow chain, not all remote views",
        "image_source_sha256": frame["image_sha256"], "bev_source_sha256": window["evidence_sha256"],
        "scope": "diagnostic only; no labels, scores or training changed; atlas focuses one track, overview includes all tracks"})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cases-per-type", type=int, default=20)
    parser.add_argument("--temporal-radius", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.cases_per_type < 1:
        parser.error("cases-per-type must be positive")
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if source == output or output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new audit output directory")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    summaries = {}
    for split in ("train", "val"):
        root = source/split
        pairs, stats, _ = load_motion_analysis(root/"pairs/pairs.jsonl", root/"flow")
        by_id = {pair["id"]: pair for pair in pairs}
        forward = {(*(pair[key] for key in ("scene_name", "official_split", "camera")), pair["sample_token"]): pair for pair in pairs}
        backward = {(*(pair[key] for key in ("scene_name", "official_split", "camera")), pair["next_sample_token"]): pair for pair in pairs}
        windows = {window["id"]: window for window in read_rows(root/"bev_flow/windows.jsonl")}
        diagnostics = read_rows(root/"labels/transition_diagnostics.jsonl")
        pools = case_pool(diagnostics, read_rows(root/"labels/decisions.jsonl"))
        selected = sample_cases(pools, args.cases_per_type, args.seed)
        manifest, contacts = [], defaultdict(list)
        for case in tqdm(selected, desc=f"{split} visual audit", unit="case", mininterval=5):
            destination = output/split/case["category"]/case["id"]
            render_case(case, windows[case["window_id"]], by_id, forward, backward, root, destination,
                        args.temporal_radius, stats)
            manifest.append({key: value for key, value in case.items() if key != "component"} | {"panel": str(destination/"panel.jpg")})
            preview = cv2.imread(str(destination/"panel.jpg"))
            contacts[case["category"]].append(tile(preview, f"{case['scene_name']} {case['id'][:12]}", (480, 328)))
        for category, previews in contacts.items():
            checked_write_image(output/split/category/"contact_sheet.jpg", sheet(previews, 2))
        categories = ("no_visible_overlap", "flow_chain_zero", "exposure_removed", "outside_footprint", "accepted")
        chosen = Counter(row["category"] for row in manifest)
        summaries[split] = {category: {"available": len(pools[category]), "selected": chosen[category],
                                      "requested": args.cases_per_type} for category in categories}
        summaries[split]["stage_accounting"] = transition_stage_accounting([
            frame for window in diagnostics for frame in window["frames"]])
        write_rows(output/split/"cases.jsonl", manifest)
    write_json(output/"summary.json", summaries)
    write_json(output/"protocol.json", {**vars(args), "protocol": "shadow-geometry-visual-audit-v1"})
    print(f"Wrote visual geometry audit to {output}", flush=True)


if __name__ == "__main__":
    main()
