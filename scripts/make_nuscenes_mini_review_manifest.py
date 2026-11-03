#!/usr/bin/env python3
"""Build a local review manifest from nuScenes mini camera keyframes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing nuScenes metadata: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list: {path}")
    return rows


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_manifest_rows(
    dataroot: Path,
    *,
    version: str,
    cameras: list[str],
    keyframes_only: bool = True,
    scenes: set[str] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    dataroot = dataroot.resolve()
    metadata_root = dataroot / version
    scenes_by_token = {row["token"]: row for row in _load_json(metadata_root / "scene.json")}
    samples_by_token = {row["token"]: row for row in _load_json(metadata_root / "sample.json")}
    sensors_by_token = {row["token"]: row for row in _load_json(metadata_root / "sensor.json")}
    calibrated_by_token = {
        row["token"]: row for row in _load_json(metadata_root / "calibrated_sensor.json")
    }
    sample_data = _load_json(metadata_root / "sample_data.json")

    camera_set = set(cameras)
    order = {camera: index for index, camera in enumerate(CAMERA_ORDER)}
    rows: list[dict[str, Any]] = []
    missing: list[Path] = []

    for record in sample_data:
        if keyframes_only and not bool(record.get("is_key_frame")):
            continue
        calibrated = calibrated_by_token.get(str(record.get("calibrated_sensor_token", "")))
        if calibrated is None:
            continue
        sensor = sensors_by_token.get(str(calibrated.get("sensor_token", "")))
        if sensor is None or sensor.get("channel") not in camera_set:
            continue
        sample = samples_by_token.get(str(record.get("sample_token", "")))
        if sample is None:
            continue
        scene = scenes_by_token.get(str(sample.get("scene_token", "")))
        if scene is None:
            continue
        scene_name = str(scene.get("name", ""))
        if scenes and scene_name not in scenes:
            continue
        filename = str(record.get("filename", ""))
        image_path = dataroot / filename
        if not filename or not image_path.is_file():
            missing.append(image_path)
            continue
        camera = str(sensor["channel"])
        rows.append(
            {
                "id": f"{scene_name}__{record['sample_token']}__{camera}",
                "scene": scene_name,
                "sample_token": str(record["sample_token"]),
                "camera": camera,
                "source": camera,
                "timestamp": int(record.get("timestamp", sample.get("timestamp", 0))),
                "image": filename,
                "filename": filename,
            }
        )

    rows.sort(
        key=lambda row: (
            row["scene"],
            int(row["timestamp"]),
            order.get(row["camera"], len(order)),
            row["sample_token"],
        )
    )
    if limit:
        rows = rows[:limit]
    if missing and not rows:
        raise ValueError(f"no camera images found; first missing path: {missing[0]}")
    return rows


def write_manifest(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    cameras = Counter(row["camera"] for row in rows)
    scenes = Counter(row["scene"] for row in rows)
    return {
        "output": str(output),
        "rows": len(rows),
        "samples": len({row["sample_token"] for row in rows}),
        "scenes": len(scenes),
        "camera_counts": dict(sorted(cameras.items())),
        "scene_counts": dict(sorted(scenes.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build a nuScenes mini image review manifest.")
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("nuscenes-mini"),
    )
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/nuscenes_mini_screening/manifest.jsonl"),
    )
    parser.add_argument("--cameras", default=",".join(CAMERA_ORDER))
    parser.add_argument("--scenes", default="", help="Optional comma-separated scene names.")
    parser.add_argument("--all-frames", action="store_true", help="Include sweep frames too.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cameras = _split(args.cameras)
    unknown = sorted(set(cameras) - set(CAMERA_ORDER))
    if unknown:
        raise SystemExit(f"unknown cameras: {', '.join(unknown)}")
    scenes = set(_split(args.scenes)) or None
    rows = build_manifest_rows(
        args.dataroot,
        version=args.version,
        cameras=cameras,
        keyframes_only=not args.all_frames,
        scenes=scenes,
        limit=args.limit,
    )
    summary = write_manifest(rows, args.out)
    summary_path = args.out.with_name("manifest_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
