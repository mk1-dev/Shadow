#!/usr/bin/env python3
"""Build a nuScenes mini review manifest and launch the local annotation UI."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from make_nuscenes_mini_review_manifest import (  # noqa: E402
    CAMERA_ORDER,
    _split,
    build_manifest_rows,
    write_manifest,
)
from shadow_image_screener import (  # noqa: E402
    AnnotationStore,
    create_server,
    load_manifest_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Launch local nuScenes mini annotation.")
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("nuscenes-mini"),
        help="Path containing v1.0-mini, samples, and sweeps.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("annotation_work"),
        help="Directory for the manifest and annotation JSONL.",
    )
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--cameras", default=",".join(CAMERA_ORDER))
    parser.add_argument("--scenes", default="", help="Optional comma-separated scene names.")
    parser.add_argument("--all-frames", action="store_true", help="Include sweep frames too.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataroot = args.dataroot.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.jsonl"
    annotation_path = work_dir / "review_annotations.jsonl"

    rows = build_manifest_rows(
        dataroot,
        version=args.version,
        cameras=_split(args.cameras),
        keyframes_only=not args.all_frames,
        scenes=set(_split(args.scenes)) or None,
        limit=args.limit,
    )
    summary = write_manifest(rows, manifest_path)
    (work_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    items = load_manifest_items(
        manifest_path,
        root=dataroot,
        view_fields=["image"],
        limit=args.limit,
    )
    store = AnnotationStore(annotation_path, {str(item["id"]) for item in items})
    server = create_server(args.host, args.port, items, store)
    port = int(server.server_address[1])
    url = f"http://{args.host}:{port}/"
    print(f"images: {len(items)}")
    print(f"manifest: {manifest_path}")
    print(f"annotations: {annotation_path}")
    print(f"open: {url}")
    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(0.25), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
