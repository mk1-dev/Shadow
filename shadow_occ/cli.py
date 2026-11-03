from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from shadow_occ.config import load_config
from shadow_occ.data.lidar_pseudo import build_lidar_pseudo_geometry
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.data.occ3d_loader import Occ3DLoader
from shadow_occ.geometry.bev import BEVGrid
from shadow_occ.metrics.correlation import correlation_by_distance, load_length_height_pairs
from shadow_occ.shadow.image_shadow import dark_region_mask, read_image_bgr
from shadow_occ.shadow.labels import apply_temporal_reliability, build_shadow_geometry_labels
from shadow_occ.shadow.pseudo_label import pseudo_shadow_from_occ_and_images
from shadow_occ.shadow.render import render_shadow_from_height
from shadow_occ.shadow.tracking import ShadowTracker
from shadow_occ.utils.io import ensure_dir, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser("shadow-occ")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-labels", help="Generate pseudo shadow geometry labels.")
    make.add_argument("--config", default="configs/default.yaml")
    make.add_argument("--dataroot")
    make.add_argument("--occ-root")
    make.add_argument("--geometry-source", choices=["occ3d", "nuscenes_lidar"])
    make.add_argument("--output-root")
    make.add_argument("--scene", action="append", default=None)
    make.add_argument("--max-samples", type=int)
    make.add_argument("--num-workers", type=int, default=1)

    corr = sub.add_parser("corr", help="Precheck shadow length / Occ3D H_occ correlation.")
    corr.add_argument("--config", default="configs/default.yaml")
    corr.add_argument("--label-root")
    corr.add_argument("--output-root")

    smoke = sub.add_parser("smoke-test", help="Run a synthetic pipeline smoke test.")
    smoke.add_argument("--output-root", default="outputs/smoke")

    args = parser.parse_args()
    if args.command == "make-labels":
        run_make_labels(args)
    elif args.command == "corr":
        run_corr(args)
    elif args.command == "smoke-test":
        run_smoke_test(args.output_root)


def run_make_labels(args: argparse.Namespace) -> None:
    overrides = {"data": {}}
    if args.dataroot:
        overrides["data"]["dataroot"] = args.dataroot
    if args.occ_root:
        overrides["data"]["occ_root"] = args.occ_root
    if args.geometry_source:
        overrides["data"]["geometry_source"] = args.geometry_source
    if args.output_root:
        overrides["data"]["output_root"] = args.output_root
    if args.scene:
        overrides["data"]["scene_names"] = args.scene
    if args.max_samples is not None:
        overrides["data"]["max_samples"] = args.max_samples

    cfg = load_config(args.config, overrides=overrides)
    grid = BEVGrid.from_config(cfg.bev)
    out_root = ensure_dir(cfg.data.output_root)
    label_root = ensure_dir(out_root / "shadow_labels")

    num_workers = max(1, int(getattr(args, "num_workers", 1) or 1))
    nusc = NuScenesAdapter(cfg.data.version, cfg.data.dataroot, verbose=True)
    samples = list(nusc.iter_samples(cfg.data.scene_names, cfg.data.max_samples))
    scene_tokens = _group_sample_tokens_by_scene(samples)
    if num_workers == 1:
        occ_loader = Occ3DLoader(cfg.data.occ_root, cfg.occ) if cfg.data.geometry_source == "occ3d" else None
        total = sum(len(tokens) for tokens in scene_tokens.values())
        progress = tqdm(total=total, desc="shadow labels")
        for scene_name, tokens in scene_tokens.items():
            records = [
                nusc.make_sample_record(nusc.nusc.get("sample", token))
                for token in tokens
            ]
            records.sort(key=lambda item: item.timestamp)
            tracker = _make_tracker(cfg)
            for record in records:
                _write_shadow_label_for_record(
                    cfg=cfg,
                    grid=grid,
                    label_root=label_root,
                    nusc=nusc,
                    occ_loader=occ_loader,
                    record=record,
                    tracker=tracker,
                )
                progress.update(1)
        progress.close()
    else:
        futures = {}
        progress_root = ensure_dir(out_root / "label_progress")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for scene_name, tokens in scene_tokens.items():
                progress_path = progress_root / f"{scene_name}.json"
                write_json(
                    progress_path,
                    {"scene_name": scene_name, "done": 0, "total": len(tokens), "status": "queued"},
                )
                future = executor.submit(_process_scene_label_worker, cfg, scene_name, tokens)
                futures[future] = scene_name
            _monitor_label_workers(futures, progress_root, total_samples=sum(map(len, scene_tokens.values())))

    write_json(out_root / "run_config.json", cfg)
    print(f"Wrote labels to {label_root}")


def _group_sample_tokens_by_scene(samples) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in samples:
        grouped[record.scene_name].append(record.token)
    return dict(grouped)


def _process_scene_label_worker(cfg, scene_name: str, sample_tokens: list[str]) -> dict:
    grid = BEVGrid.from_config(cfg.bev)
    label_root = ensure_dir(Path(cfg.data.output_root) / "shadow_labels")
    progress_path = ensure_dir(Path(cfg.data.output_root) / "label_progress") / f"{scene_name}.json"
    write_json(
        progress_path,
        {"scene_name": scene_name, "done": 0, "total": len(sample_tokens), "status": "loading_nuscenes"},
    )
    nusc = NuScenesAdapter(cfg.data.version, cfg.data.dataroot, verbose=False)
    occ_loader = Occ3DLoader(cfg.data.occ_root, cfg.occ) if cfg.data.geometry_source == "occ3d" else None
    records = [
        nusc.make_sample_record(nusc.nusc.get("sample", token))
        for token in sample_tokens
    ]
    records.sort(key=lambda item: item.timestamp)
    tracker = _make_tracker(cfg)
    for index, record in enumerate(records, start=1):
        write_json(
            progress_path,
            {
                "scene_name": scene_name,
                "done": index - 1,
                "total": len(records),
                "current_sample": record.token,
                "status": "processing_sample",
            },
        )
        _write_shadow_label_for_record(
            cfg=cfg,
            grid=grid,
            label_root=label_root,
            nusc=nusc,
            occ_loader=occ_loader,
            record=record,
            tracker=tracker,
        )
        write_json(
            progress_path,
            {
                "scene_name": scene_name,
                "done": index,
                "total": len(records),
                "last_sample": record.token,
                "status": "running" if index < len(records) else "complete",
            },
        )
    return {"scene_name": scene_name, "num_samples": len(records)}


def _monitor_label_workers(futures: dict, progress_root: Path, total_samples: int) -> None:
    """Show sample-level progress while scene-level workers are running."""
    pending = set(futures)
    completed_samples = 0
    with tqdm(total=total_samples, desc="shadow label samples", unit="sample") as progress:
        while pending:
            finished = {future for future in pending if future.done()}
            for future in finished:
                pending.remove(future)
                result = future.result()
                print(
                    f"completed scene {result['scene_name']}: "
                    f"{result['num_samples']} samples"
                )

            current_samples = 0
            active = []
            for scene_name in futures.values():
                state = _read_progress_state(progress_root / f"{scene_name}.json")
                current_samples += min(int(state.get("done", 0)), int(state.get("total", 0)))
                if state.get("status") not in {"complete", "unknown"}:
                    active.append(
                        f"{scene_name}:{state.get('done', 0)}/{state.get('total', 0)}"
                        f"[{state.get('status')}]"
                    )
            delta = max(0, current_samples - completed_samples)
            if delta:
                progress.update(delta)
                completed_samples = current_samples
            progress.set_postfix(
                scenes=f"{len(futures) - len(pending)}/{len(futures)}",
                active=", ".join(active[:3]) if active else "none",
            )
            progress.refresh()
            if pending:
                time.sleep(1.0)


def _read_progress_state(path: Path) -> dict:
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"done": 0, "total": 0, "status": "unknown"}


def _write_shadow_label_for_record(
    cfg,
    grid: BEVGrid,
    label_root: Path,
    nusc: NuScenesAdapter,
    occ_loader: Occ3DLoader | None,
    record,
    tracker: ShadowTracker,
) -> None:
    scene_out = ensure_dir(label_root / record.scene_name)
    if cfg.data.geometry_source == "occ3d":
        if occ_loader is None:
            raise RuntimeError("Occ3DLoader was not initialized.")
        geometry_sample = occ_loader.load(record.token, record.scene_name)
        height_map = occ_loader.height_map(geometry_sample, grid)
        ground_points, _ground_labels = occ_loader.ground_points(geometry_sample)
        geometry_to_global = nusc.occ_to_global(record, cfg.data.occ_frame)
    elif cfg.data.geometry_source == "nuscenes_lidar":
        points_ego = nusc.load_lidar_points_ego(record)
        geometry_sample = build_lidar_pseudo_geometry(
            points_ego,
            grid,
            cfg.lidar,
            sample_token=record.token,
        )
        height_map = geometry_sample.height_map
        ground_points = geometry_sample.ground_points
        geometry_to_global = record.ego_to_global
    else:
        raise ValueError(
            f"Unsupported data.geometry_source={cfg.data.geometry_source!r}. "
            "Use 'occ3d' or 'nuscenes_lidar'."
        )

    previous_record = _previous_record(nusc, record)
    camera_dark_masks = []
    ground_z = float(record.ego_to_global[2, 3] + cfg.shadow.ground_plane_z_offset_m)
    for cam_name in cfg.data.cams:
        camera = nusc.get_camera(record, cam_name)
        image = read_image_bgr(camera.image_path)
        dark = dark_region_mask(image, cfg.shadow)
        previous_image = None
        previous_camera = None
        if previous_record is not None:
            previous_camera = nusc.get_camera(previous_record, cam_name)
            previous_image = read_image_bgr(previous_camera.image_path)
        camera_dark_masks.append(
            (dark, camera.calibration, image, previous_image, 
             previous_camera.calibration if previous_camera is not None else None, ground_z)
        )

    shadow_mask, confidence, temporal_confidence, meta = pseudo_shadow_from_occ_and_images(
        geometry_sample,
        ground_points,
        camera_dark_masks,
        geometry_to_global,
        grid,
        cfg.shadow,
    )
    rendered_shadow_prior = render_shadow_from_height(height_map, grid, cfg.shadow)
    caster_points = (
        nusc.annotation_centers_ego(record)
        if cfg.data.use_annotation_caster
        else None
    )
    labels = build_shadow_geometry_labels(
        shadow_mask,
        height_map,
        grid,
        cfg.shadow,
        projection_confidence=confidence,
        caster_points=caster_points,
        temporal_confidence=temporal_confidence,
    )
    labels.components = tracker.update(labels.components, record.ego_to_global)
    labels = apply_temporal_reliability(labels, cfg.shadow.min_track_age)

    npz_path = scene_out / f"{record.token}.npz"
    np.savez_compressed(
        npz_path,
        **labels.as_npz_dict(),
        confidence=confidence.astype(np.float32),
        temporal_confidence=temporal_confidence.astype(np.float32),
        rendered_shadow_prior=rendered_shadow_prior.astype(np.uint8),
        components=np.asarray([c.to_dict() for c in labels.components], dtype=object),
    )
    write_json(
        scene_out / f"{record.token}.json",
        {
            "sample_token": record.token,
            "scene_name": record.scene_name,
            "timestamp": record.timestamp,
            "num_components": len(labels.components),
            "components": [c.to_dict() for c in labels.components],
            "meta": meta,
            "geometry_source": cfg.data.geometry_source,
        },
    )


def _make_tracker(cfg) -> ShadowTracker:
    return ShadowTracker(
        min_age=cfg.shadow.min_track_age,
        caster_max_motion_m=cfg.shadow.caster_max_motion_m,
        tip_max_motion_m=cfg.shadow.tip_max_motion_m,
        tip_smoothness_sigma_m=cfg.shadow.tip_smoothness_sigma_m,
    )


def _previous_record(nusc: NuScenesAdapter, record):
    previous_token = record.sample.get("prev", "")
    if not previous_token:
        return None
    return nusc.make_sample_record(nusc.nusc.get("sample", previous_token))


def run_corr(args: argparse.Namespace) -> None:
    overrides = {"data": {}}
    if args.output_root:
        overrides["data"]["output_root"] = args.output_root
    cfg = load_config(args.config, overrides=overrides)
    label_root = Path(args.label_root or Path(cfg.data.output_root) / "shadow_labels")
    paths = sorted(label_root.glob("*/*.npz")) + sorted(label_root.glob("*.npz"))
    lengths, heights, distances = load_length_height_pairs(paths)
    rows = correlation_by_distance(lengths, heights, distances, cfg.eval.distance_bins_m)
    out_path = ensure_dir(Path(cfg.data.output_root) / "metrics") / "shadow_height_corr.json"
    write_json(out_path, rows)
    for row in rows:
        print(row)
    print(f"Wrote metrics to {out_path}")


def run_smoke_test(output_root: str | Path) -> None:
    from tests.synthetic_pipeline import run

    run(output_root)


if __name__ == "__main__":
    main()
