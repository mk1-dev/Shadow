"""Reuse the existing OpenDriveVLA UniAD inference stack, without copying its encoder."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_occ.annotation.cached_bev import spatial_bev, write_cached_bev
from shadow_occ.annotation.data import digest, read_rows
from shadow_occ.annotation.mature_objects import object_frame_source, objects_from_uniad, write_cached_objects
from shadow_occ.data.nuscenes_adapter import NuScenesAdapter
from shadow_occ.utils.io import write_json


def _pick_last_tensor_like(value):
    """Pick the current frame from nested list wrappers used by the mature exporter."""
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return value
        if isinstance(value[0], (list, tuple)):
            return _pick_last_tensor_like(value[-1])
        return _pick_last_tensor_like(value[-1])
    return value


def reset_mature_history(model):
    """UniAD tracking stores BEV history separately from prev_frame_info."""
    model.test_track_instances = None
    model.prev_bev = None
    if hasattr(model, "prev_frame_info"):
        model.prev_frame_info.update(prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0)


def main():
    parser = argparse.ArgumentParser("Frozen spatial BEV export using the mature OpenDriveVLA UniAD environment")
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--model-config", default="projects/configs/stage1_track_map/base_track_map.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--ann-file", help="Use train or val nuScenes infos with the existing TEST pipeline")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--history-mode", choices=["temporal", "current-only"], default="temporal")
    parser.add_argument("--minimum-object-score", type=float, default=0.25)
    parser.add_argument("--objects-only", action="store_true", help="Export missing 3D predictions alongside an existing feature cache")
    args = parser.parse_args()
    external = Path(args.external_root).resolve()
    config_path = Path(args.model_config)
    if not config_path.is_absolute():
        config_path = external/config_path
    pairs_path, checkpoint = Path(args.pairs).resolve(), Path(args.checkpoint).resolve()
    output, dataroot = Path(args.output_root).resolve(), Path(args.dataroot).resolve()
    ann_file = str(Path(args.ann_file).resolve()) if args.ann_file else None
    spec = importlib.util.spec_from_file_location("shadow_mature_uniad_export", external/"scripts/export_uniad_pth.py")
    exporter = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(external))
    os.chdir(external)
    spec.loader.exec_module(exporter)
    cfg = exporter.Config.fromfile(str(config_path))
    cfg.data.test.test_mode = True
    cfg.data.test.data_root = str(dataroot)
    if ann_file:
        cfg.data.test.ann_file = ann_file
    adapter = NuScenesAdapter(args.version, dataroot)
    pairs = read_rows(pairs_path)
    needed = {p[key] for p in pairs for key in ("sample_token", "next_sample_token")}
    scenes = {adapter.nusc.get("sample", token)["scene_token"] for token in needed}
    latest_needed = {scene: max(adapter.nusc.get("sample", token)["timestamp"] for token in needed
        if adapter.nusc.get("sample", token)["scene_token"] == scene) for scene in scenes}
    exporter.import_plugin_modules(cfg)
    dataset = exporter.custom_build_dataset(cfg.data.test)
    class_names = list(dataset.CLASSES)
    indices = sorted((i for i, info in enumerate(dataset.data_infos) if info["scene_token"] in scenes
                      and info["timestamp"] <= latest_needed[info["scene_token"]]
                      and (args.history_mode == "temporal" or info["token"] in needed)),
                     key=lambda i: (dataset.data_infos[i]["scene_token"], dataset.data_infos[i]["timestamp"]))
    # Keep the original dataset indices for its map/trajectory lookups and replay whole scenes.
    for index in indices:
        info = dataset.data_infos[index]
        sample = adapter.nusc.get("sample", info["token"])
        for name, camera in info["cams"].items():
            sd = adapter.nusc.get("sample_data", sample["data"][name])
            camera["data_path"] = str(dataroot/sd["filename"])
        sd = adapter.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        info["lidar_path"] = str(dataroot/sd["filename"])
    loader = exporter.build_dataloader(Subset(dataset, indices), samples_per_gpu=1, workers_per_gpu=args.workers,
                                       num_gpus=1, dist=False, shuffle=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Use the mature UniAD CUDA environment for frozen BEV export")
    model = exporter.build_model_from_cfg(cfg, str(checkpoint), torch.device("cuda:0"), False)
    model.requires_grad_(False)
    source = {"encoder": "UniAD/OpenDriveVLA BEVFormer", "mode": "frozen ordered inference",
        "payload": "objects_only" if args.objects_only else "spatial_features_and_3d_predictions",
        "minimum_object_score": args.minimum_object_score,
        "history_mode": args.history_mode,
        "checkpoint_sha256": digest(checkpoint), "model_config_sha256": digest(config_path),
        "external_exporter_sha256": digest(external/"scripts/export_uniad_pth.py")}
    saved, seen, previous_time = set(), set(), {}
    for data in tqdm(loader, desc="mature BEVFormer scene replay", unit="frame", mininterval=5):
        data = exporter.remove_datacontainer(data)
        metas = data["img_metas"][0]
        token = exporter.extract_sample_token(metas)
        record = adapter.make_sample_record(adapter.nusc.get("sample", token))
        if record.scene_token not in scenes:
            continue
        if token in seen or record.timestamp <= previous_time.get(record.scene_token, -1):
            raise ValueError("Mature BEV export requires ordered, nonduplicated scene frames")
        seen.add(token)
        previous_time[record.scene_token] = record.timestamp
        data["img_metas"] = metas
        data["img"] = data["img"][0]
        exporter._normalize_sequence_inputs_for_export(data)
        exporter._normalize_seg_gt_shapes_for_export(data)
        for key in ("l2g_t", "l2g_r_mat"):
            data[key] = _pick_last_tensor_like(data.get(key))
        data = exporter.move_data_to_device(data, "cuda:0")
        if args.history_mode == "current-only":
            reset_mature_history(model)
        with torch.no_grad():
            _, result = model(return_loss=False, rescale=True, **data)
        if token in needed:
            timestamp = adapter.nusc.get("sample_data", record.lidar_token)["timestamp"]
            objects = objects_from_uniad(result["result_track"], record, timestamp,
                                          class_names, args.minimum_object_score)
            if args.objects_only:
                write_cached_objects(output, token, objects,
                    object_frame_source(record, timestamp, cfg.point_cloud_range, source))
            else:
                value = result["result_track"]["bev_embed"].detach().float().cpu().numpy()
                height, width = int(cfg.bev_h_), int(cfg.bev_w_)
                if height == width == 100 and value.shape[0] == 200*200:
                    height = width = 200  # Existing UniAD upsample_bev_if_tiny output.
                feature = spatial_bev(value, height, width)
                write_cached_bev(output, token, feature, record, timestamp, cfg.point_cloud_range, source, objects=objects)
            saved.add(token)
            print(f"saved={len(saved)}/{len(needed)} sample={token} predicted_objects={len(objects['objects'])}", flush=True)
        del result
        if saved == needed:
            break
    if saved != needed:
        raise RuntimeError(f"Missing {len(needed-saved)} requested sample tokens; choose the matching --ann-file")
    write_json(output/"cache_protocol.json", {**source, "samples": len(saved),
        "pair_manifest_sha256": digest(pairs_path), "coordinate_frame": "LIDAR_TOP",
        "pc_range": list(cfg.point_cloud_range), "ann_file": str(cfg.data.test.ann_file)})


if __name__ == "__main__":
    main()
