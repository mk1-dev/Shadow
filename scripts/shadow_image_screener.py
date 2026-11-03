#!/usr/bin/env python3
"""Local browser-based image screening with box and freehand annotations.

The tool is intentionally dependency-free beyond the Python standard library.
It binds to localhost, serves a single-page review UI, and appends one JSON
record per save to a JSONL file. Records use normalized image coordinates so
they can be converted to masks without depending on display resolution.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_VIEW_FIELDS = ("clean", "numbered", "proposal_map")
DECISIONS = {"shadow", "not_shadow", "uncertain"}
SHAPE_MODES = {"box", "freehand"}
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _view_label(field: str) -> str:
    labels = {
        "clean": "Clean",
        "numbered": "Numbered",
        "proposal_map": "Proposal map",
        "ground_mask": "Ground mask",
        "context": "Context",
        "image": "Image",
        "path": "Image",
    }
    return labels.get(field, field.replace("_", " ").title())


def _split_fields(value: str) -> list[str]:
    return [field.strip() for field in value.split(",") if field.strip()]


def _resolve_image(raw: Any, root: Path, manifest_path: Path) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    root_path = (root / path).resolve()
    manifest_relative = (manifest_path.parent / path).resolve()
    if root_path.exists():
        return root_path
    if manifest_relative.exists():
        return manifest_relative
    return root_path


def _unique_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{candidate}#{index}" in used:
        index += 1
    value = f"{candidate}#{index}"
    used.add(value)
    return value


def load_manifest_items(
    manifest: Path,
    *,
    root: Path,
    view_fields: Iterable[str],
    limit: int = 0,
) -> list[dict[str, Any]]:
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise ValueError(f"manifest not found: {manifest}")
    fields = list(view_fields)
    if not fields:
        fields = list(DEFAULT_VIEW_FIELDS)

    items: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    missing: list[str] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")

            views: list[dict[str, Any]] = []
            for field in fields:
                path = _resolve_image(row.get(field), root, manifest)
                if path is None:
                    continue
                if not path.is_file():
                    missing.append(f"line {line_number}, field {field}: {path}")
                    continue
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                views.append({"key": field, "label": _view_label(field), "path": path, "mime": mime})

            if not views:
                missing.append(f"line {line_number}: no usable image field in {fields}")
                continue

            raw_id = str(
                row.get("id")
                or row.get("pair_id")
                or row.get("sample_token")
                or f"line-{line_number}"
            )
            item_id = _unique_id(raw_id, used_ids)
            items.append(
                {
                    "id": item_id,
                    "title": item_id,
                    "scene": str(row.get("scene", "")),
                    "side": str(row.get("pair_source") or row.get("source") or ""),
                    "proposal_count": row.get("proposal_count"),
                    "group": str(row.get("pair_id") or row.get("scene") or ""),
                    "views": views,
                }
            )
            if limit and len(items) >= limit:
                break

    if not items:
        detail = "\n".join(missing[:8])
        raise ValueError(f"no reviewable images found\n{detail}")
    if missing:
        print(f"warning: skipped {len(missing)} unusable image fields; showing first few:")
        for message in missing[:5]:
            print(f"  {message}")
    for index, item in enumerate(items):
        item["index"] = index
    return items


def load_directory_items(directory: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"image directory not found: {directory}")
    paths = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no images found under {directory}")

    used_ids: set[str] = set()
    items: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        relative = path.relative_to(directory).as_posix()
        item_id = _unique_id(relative, used_ids)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        items.append(
            {
                "id": item_id,
                "index": index,
                "title": relative,
                "scene": path.parent.name if path.parent != directory else "",
                "side": "",
                "proposal_count": None,
                "group": path.parent.as_posix(),
                "views": [{"key": "image", "label": "Image", "path": path, "mime": mime}],
            }
        )
    return items


def _finite_number(value: Any, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite number")
    number = float(value)
    if not (low <= number <= high):
        raise ValueError(f"number outside [{low}, {high}]")
    return number


def validate_annotation(payload: dict[str, Any], item_ids: set[str]) -> dict[str, Any]:
    item_id = str(payload.get("item_id", ""))
    if item_id not in item_ids:
        raise ValueError("unknown item_id")

    decision = payload.get("decision")
    if decision is not None and decision not in DECISIONS:
        raise ValueError("invalid decision")
    shape_mode = str(payload.get("shape_mode", "box"))
    if shape_mode not in SHAPE_MODES:
        raise ValueError("invalid shape_mode")

    box = payload.get("box")
    if box is not None:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("box must contain four normalized coordinates")
        x0 = _finite_number(box[0], low=0.0, high=1.0)
        y0 = _finite_number(box[1], low=0.0, high=1.0)
        x1 = _finite_number(box[2], low=0.0, high=1.0)
        y1 = _finite_number(box[3], low=0.0, high=1.0)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("box has zero or negative area")
        box = [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)]

    raw_strokes = payload.get("strokes") or []
    if not isinstance(raw_strokes, list):
        raise ValueError("strokes must be a list")
    strokes = []
    for raw_stroke in raw_strokes[:256]:
        if not isinstance(raw_stroke, dict):
            raise ValueError("each stroke must be an object")
        width = _finite_number(raw_stroke.get("width", 0.001), low=0.00001, high=0.25)
        raw_points = raw_stroke.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("each stroke needs points")
        points = []
        for raw_point in raw_points[:10000]:
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise ValueError("stroke points must be [x, y]")
            x = _finite_number(raw_point[0], low=0.0, high=1.0)
            y = _finite_number(raw_point[1], low=0.0, high=1.0)
            points.append([round(x, 6), round(y, 6)])
        strokes.append({"width": round(width, 6), "points": points})

    note = str(payload.get("note", ""))[:500]
    selected_proposal = payload.get("selected_proposal")
    if selected_proposal is not None:
        if isinstance(selected_proposal, bool) or not isinstance(selected_proposal, int):
            raise ValueError("selected_proposal must be an integer or null")
        if not (0 <= selected_proposal <= 999):
            raise ValueError("selected_proposal out of range")

    image_width = int(payload.get("image_width") or 0)
    image_height = int(payload.get("image_height") or 0)
    if not (0 <= image_width <= 100000 and 0 <= image_height <= 100000):
        raise ValueError("invalid image dimensions")

    return {
        "schema_version": 1,
        "item_id": item_id,
        "decision": decision,
        "shape_mode": shape_mode,
        "box": box,
        "strokes": strokes,
        "note": note,
        "selected_proposal": selected_proposal,
        "image_width": image_width,
        "image_height": image_height,
        "updated_at": _utc_now(),
    }


class AnnotationStore:
    def __init__(self, path: Path, item_ids: set[str]) -> None:
        self.path = path.resolve()
        self.item_ids = item_ids
        self.lock = threading.Lock()
        self.latest: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    item_id = str(row["item_id"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    print(f"warning: ignoring malformed annotation line {line_number}")
                    continue
                if item_id in self.item_ids:
                    self.latest[item_id] = row

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = validate_annotation(payload, self.item_ids)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.latest[record["item_id"]] = record
        return record


def _public_item(item: dict[str, Any], annotation: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "index": item["index"],
        "id": item["id"],
        "title": item["title"],
        "scene": item["scene"],
        "side": item["side"],
        "proposal_count": item["proposal_count"],
        "group": item["group"],
        "views": [
            {
                "key": view["key"],
                "label": view["label"],
                "url": f"/api/image/{item['index']}/{view['key']}",
            }
            for view in item["views"]
        ],
        "annotation": annotation,
    }


def make_handler(items: list[dict[str, Any]], store: AnnotationStore) -> type[BaseHTTPRequestHandler]:
    item_by_index = {int(item["index"]): item for item in items}
    item_ids = set(store.item_ids)

    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "ShadowScreener/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[screener] {self.address_string()} {fmt % args}")

        def _write(self, body: bytes, *, status: int = 200, content_type: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._write(body, status=status, content_type="application/json; charset=utf-8")

        def _error(self, message: str, *, status: int = 400) -> None:
            self._json({"error": message}, status=status)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._write(HTML_PAGE.encode("utf-8"), content_type="text/html; charset=utf-8")
                return
            if path == "/api/state":
                annotations = store.latest
                public_items = [_public_item(item, annotations.get(item["id"])) for item in items]
                counts = {"shadow": 0, "not_shadow": 0, "uncertain": 0, "unreviewed": 0}
                for item in public_items:
                    annotation = item["annotation"] or {}
                    decision = annotation.get("decision")
                    if decision in counts:
                        counts[decision] += 1
                    else:
                        counts["unreviewed"] += 1
                self._json(
                    {
                        "items": public_items,
                        "counts": counts,
                        "output": str(store.path),
                    }
                )
                return
            if path == "/api/export":
                latest = [store.latest[item["id"]] for item in items if item["id"] in store.latest]
                body = "".join(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for record in latest
                ).encode("utf-8")
                self._write(
                    body,
                    content_type="application/x-ndjson",
                    extra={"Content-Disposition": 'attachment; filename="shadow_review_annotations.jsonl"'},
                )
                return
            if path.startswith("/api/image/"):
                parts = [unquote(part) for part in path.split("/") if part]
                if len(parts) != 4:
                    self._error("invalid image route", status=404)
                    return
                try:
                    item = item_by_index[int(parts[2])]
                except (ValueError, KeyError):
                    self._error("image not found", status=404)
                    return
                view = next((view for view in item["views"] if view["key"] == parts[3]), None)
                if view is None:
                    self._error("view not found", status=404)
                    return
                try:
                    body = view["path"].read_bytes()
                except OSError as exc:
                    self._error(f"cannot read image: {exc}", status=500)
                    return
                self._write(body, content_type=str(view["mime"]))
                return
            self._error("not found", status=404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/annotation":
                self._error("not found", status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error("invalid content length")
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._error("invalid request size", status=413)
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                record = store.save(payload)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                self._error(str(exc))
                return
            self._json({"ok": True, "annotation": record})

    return ReviewHandler


def create_server(host: str, port: int, items: list[dict[str, Any]], store: AnnotationStore) -> ThreadingHTTPServer:
    handler = make_handler(items, store)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locally screen shadow candidate images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSONL rows containing image paths.")
    source.add_argument("--images-dir", type=Path, help="Directory containing images.")
    parser.add_argument(
        "--view-fields",
        default="clean,numbered,proposal_map",
        help="Comma-separated manifest image fields to expose as views.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Base directory for relative paths.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/local_image_screening/review_annotations.jsonl"),
        help="Annotation JSONL output path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Review at most this many images.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the review UI in the default browser.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.manifest:
        items = load_manifest_items(
            args.manifest,
            root=args.root,
            view_fields=_split_fields(args.view_fields),
            limit=args.limit,
        )
    else:
        items = load_directory_items(args.images_dir, limit=args.limit)

    store = AnnotationStore(args.out, {str(item["id"]) for item in items})
    server = create_server(args.host, args.port, items, store)
    actual_port = int(server.server_address[1])
    url = f"http://{args.host}:{actual_port}/"
    print(f"review images: {len(items)}")
    print(f"annotations: {store.path}")
    print(f"open: {url}")
    if args.open:
        threading.Thread(target=lambda: (time.sleep(0.25), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        server.server_close()
    return 0


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shadow Image Screener</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #edf1ef;
      --panel: #f8faf9;
      --line: #d4dcd8;
      --ink: #17211e;
      --muted: #63716c;
      --accent: #0f766e;
      --accent-soft: #d8eeea;
      --shadow: #16734a;
      --reject: #b43c3c;
      --uncertain: #9a6500;
      --canvas: #101614;
      --radius: 6px;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
      font: 13px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, textarea { font: inherit; }
    button { color: inherit; }
    .app { height: 100%; display: flex; flex-direction: column; }
    .topbar {
      height: 54px;
      flex: 0 0 54px;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
    }
    .brand { font-weight: 700; letter-spacing: 0; white-space: nowrap; }
    .progress { color: var(--muted); white-space: nowrap; }
    .spacer { flex: 1; }
    .save-state { color: var(--muted); min-width: 74px; text-align: right; white-space: nowrap; }
    .layout { min-height: 0; flex: 1; display: grid; grid-template-columns: 304px minmax(0, 1fr); }
    .sidebar {
      min-height: 0;
      min-width: 0;
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    .sidebar-controls { padding: 12px; border-bottom: 1px solid var(--line); }
    .search {
      width: 100%;
      height: 34px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      outline: none;
    }
    .search:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
    .camera-select { margin-top: 8px; }
    .filters { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }
    .filter-button {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      cursor: pointer;
      white-space: nowrap;
    }
    .filter-button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); color: #084e49; }
    .item-list { min-height: 0; overflow: auto; }
    .item-row {
      width: 100%;
      display: grid;
      grid-template-columns: 8px minmax(0, 1fr);
      gap: 9px;
      padding: 10px 12px;
      text-align: left;
      border: 0;
      border-bottom: 1px solid #e4e9e7;
      background: transparent;
      cursor: pointer;
    }
    .item-row:hover { background: #eef4f1; }
    .item-row.active { background: var(--accent-soft); }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 5px; background: #aab5b1; }
    .status-dot.shadow { background: var(--shadow); }
    .status-dot.not_shadow { background: var(--reject); }
    .status-dot.uncertain { background: var(--uncertain); }
    .item-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
    .item-meta { margin-top: 2px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .workspace { min-width: 0; min-height: 0; display: flex; flex-direction: column; }
    .toolbar {
      min-height: 50px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
      overflow-x: auto;
    }
    .tool-group { display: flex; align-items: center; gap: 4px; }
    .tool-button, .action-button, .nav-button {
      height: 32px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      padding: 0 11px;
      cursor: pointer;
      white-space: nowrap;
    }
    a.nav-button { display: inline-flex; align-items: center; text-decoration: none; color: inherit; }
    .tool-button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); color: #084e49; }
    .action-button.accept { border-color: #b9d9c8; color: var(--shadow); }
    .action-button.reject { border-color: #e4c5c5; color: var(--reject); }
    .action-button.uncertain { border-color: #e4d1ae; color: var(--uncertain); }
    .action-button[aria-pressed="true"].accept { background: #d9eee2; border-color: var(--shadow); }
    .action-button[aria-pressed="true"].reject { background: #f3dddd; border-color: var(--reject); }
    .action-button[aria-pressed="true"].uncertain { background: #f3e8cf; border-color: var(--uncertain); }
    .brush-wrap { display: flex; align-items: center; gap: 7px; color: var(--muted); white-space: nowrap; }
    .brush-wrap input { width: 88px; accent-color: var(--accent); }
    .view-tabs { display: flex; gap: 4px; }
    .view-tab { height: 30px; border: 1px solid var(--line); background: #fff; border-radius: var(--radius); padding: 0 9px; cursor: pointer; }
    .view-tab[aria-pressed="true"] { background: #26322f; color: #fff; border-color: #26322f; }
    .canvas-wrap { position: relative; min-height: 0; flex: 1; background: var(--canvas); overflow: hidden; }
    #reviewCanvas { width: 100%; height: 100%; display: block; touch-action: none; cursor: crosshair; }
    #reviewCanvas.panning { cursor: grabbing; }
    .empty-state {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: #aab7b2;
      pointer-events: none;
    }
    .empty-state[hidden] { display: none; }
    .bottom {
      min-height: 76px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: #fbfcfb;
    }
    .image-meta { min-width: 0; color: var(--muted); }
    .image-title { color: var(--ink); font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .meta-line { margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .note-row { display: flex; gap: 8px; align-items: center; }
    .note-input {
      min-width: 0;
      flex: 1;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 0 9px;
      background: #fff;
    }
    .auto-next { display: flex; align-items: center; gap: 6px; color: var(--muted); white-space: nowrap; }
    .auto-next input { accent-color: var(--accent); }
    .error-banner {
      position: absolute;
      left: 12px;
      top: 12px;
      z-index: 10;
      max-width: min(680px, calc(100% - 24px));
      padding: 9px 12px;
      border: 1px solid #e4c5c5;
      border-radius: var(--radius);
      background: #fff1f1;
      color: #8b2727;
      box-shadow: 0 3px 12px rgba(0,0,0,.12);
    }
    .error-banner[hidden] { display: none; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 238px minmax(0, 1fr); }
      .bottom { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .topbar { gap: 8px; padding: 0 10px; }
      .progress { display: none; }
      .save-state { min-width: 58px; }
      .layout {
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: 238px minmax(0, 1fr);
      }
      .sidebar { width: 100%; overflow: hidden; border-right: 0; border-bottom: 1px solid var(--line); }
      .sidebar-controls { padding: 8px 10px; }
      .filters { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .item-list { width: 100%; min-width: 0; display: flex; overflow-x: auto; overflow-y: hidden; }
      .item-row { flex: 0 0 190px; border-right: 1px solid #e4e9e7; }
      .toolbar { padding: 7px 8px; }
      .bottom { min-height: 72px; padding: 8px 10px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">Shadow Image Screener</div>
      <div class="progress" id="progress">0 / 0</div>
      <div class="spacer"></div>
      <div class="save-state" id="saveState">Ready</div>
      <a class="nav-button" href="/api/export">Export JSONL</a>
    </header>
    <main class="layout">
      <aside class="sidebar">
        <div class="sidebar-controls">
          <input class="search" id="search" type="search" placeholder="Filter scene or ID" autocomplete="off">
          <select class="search camera-select" id="cameraFilter">
            <option value="">All cameras</option>
          </select>
          <div class="filters">
            <button class="filter-button" data-filter="all" aria-pressed="true">All</button>
            <button class="filter-button" data-filter="unreviewed" aria-pressed="false">Open</button>
            <button class="filter-button" data-filter="done" aria-pressed="false">Done</button>
          </div>
        </div>
        <div class="item-list" id="itemList"></div>
      </aside>
      <section class="workspace">
        <div class="toolbar">
          <div class="tool-group">
            <button class="tool-button" id="boxMode" aria-pressed="true">Box</button>
            <button class="tool-button" id="freehandMode" aria-pressed="false">Brush</button>
          </div>
          <div class="brush-wrap">
            <span>Size</span>
            <input id="brushSize" type="range" min="4" max="48" step="2" value="14">
          </div>
          <button class="tool-button" id="undoButton">Undo</button>
          <button class="tool-button" id="clearButton">Clear</button>
          <div class="spacer"></div>
          <div class="view-tabs" id="viewTabs"></div>
          <button class="nav-button" id="prevButton">Previous</button>
          <button class="nav-button" id="nextButton">Next</button>
        </div>
        <div class="canvas-wrap" id="canvasWrap">
          <canvas id="reviewCanvas"></canvas>
          <div class="empty-state" id="emptyState">No image</div>
          <div class="error-banner" id="errorBanner" hidden></div>
        </div>
        <div class="bottom">
          <div class="image-meta">
            <div class="image-title" id="imageTitle">-</div>
            <div class="meta-line" id="imageMeta">-</div>
            <div class="tool-group" style="margin-top:7px">
              <button class="action-button accept" data-decision="shadow" aria-pressed="false">Shadow</button>
              <button class="action-button reject" data-decision="not_shadow" aria-pressed="false">Not shadow</button>
              <button class="action-button uncertain" data-decision="uncertain" aria-pressed="false">Uncertain</button>
            </div>
          </div>
          <div class="note-row">
            <input class="note-input" id="noteInput" type="text" maxlength="500" placeholder="Note">
            <label class="auto-next"><input id="autoNext" type="checkbox" checked>Auto next</label>
          </div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const state = {
      items: [],
      current: -1,
      filter: 'all',
      query: '',
      camera: '',
      viewKey: null,
      image: null,
      imageWidth: 0,
      imageHeight: 0,
      zoom: 1,
      offsetX: 0,
      offsetY: 0,
      mode: 'box',
      annotation: null,
      drawing: null,
      panning: null,
      spaceDown: false,
      viewTouched: false,
      dirty: false,
      revision: 0,
      saveTimer: 0,
      saveChain: Promise.resolve(),
      viewRequest: 0,
    };

    const els = {
      canvas: document.getElementById('reviewCanvas'),
      wrap: document.getElementById('canvasWrap'),
      empty: document.getElementById('emptyState'),
      error: document.getElementById('errorBanner'),
      itemList: document.getElementById('itemList'),
      search: document.getElementById('search'),
      cameraFilter: document.getElementById('cameraFilter'),
      progress: document.getElementById('progress'),
      saveState: document.getElementById('saveState'),
      viewTabs: document.getElementById('viewTabs'),
      boxMode: document.getElementById('boxMode'),
      freehandMode: document.getElementById('freehandMode'),
      brushSize: document.getElementById('brushSize'),
      undo: document.getElementById('undoButton'),
      clear: document.getElementById('clearButton'),
      prev: document.getElementById('prevButton'),
      next: document.getElementById('nextButton'),
      imageTitle: document.getElementById('imageTitle'),
      imageMeta: document.getElementById('imageMeta'),
      note: document.getElementById('noteInput'),
      autoNext: document.getElementById('autoNext'),
    };
    const ctx = els.canvas.getContext('2d');

    function emptyAnnotation(item) {
      return {
        item_id: item.id,
        decision: null,
        shape_mode: state.mode,
        box: null,
        strokes: [],
        note: '',
        selected_proposal: null,
        image_width: 0,
        image_height: 0,
      };
    }

    function clone(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function showError(message) {
      els.error.textContent = message;
      els.error.hidden = false;
    }

    function clearError() {
      els.error.hidden = true;
      els.error.textContent = '';
    }

    function setSaveState(text) {
      els.saveState.textContent = text;
    }

    function decisionOf(item) {
      return item.annotation && item.annotation.decision ? item.annotation.decision : 'unreviewed';
    }

    function updateProgress() {
      const done = state.items.filter((item) => decisionOf(item) !== 'unreviewed').length;
      els.progress.textContent = `${done} / ${state.items.length}`;
    }

    function matchesFilter(item) {
      const decision = decisionOf(item);
      if (state.filter === 'unreviewed' && decision !== 'unreviewed') return false;
      if (state.filter === 'done' && decision === 'unreviewed') return false;
      if (state.query) {
        const haystack = `${item.id} ${item.scene} ${item.side}`.toLowerCase();
        if (!haystack.includes(state.query)) return false;
      }
      if (state.camera && item.side !== state.camera) return false;
      return true;
    }

    function renderList() {
      els.itemList.replaceChildren();
      for (const item of state.items) {
        if (!matchesFilter(item)) continue;
        const button = document.createElement('button');
        button.className = `item-row${item.index === state.current ? ' active' : ''}`;
        button.type = 'button';

        const dot = document.createElement('span');
        dot.className = `status-dot ${decisionOf(item)}`;

        const text = document.createElement('span');
        const title = document.createElement('div');
        title.className = 'item-title';
        title.textContent = item.title;
        const meta = document.createElement('div');
        meta.className = 'item-meta';
        const pieces = [item.scene, item.side, item.proposal_count == null ? '' : `${item.proposal_count} regions`];
        meta.textContent = pieces.filter(Boolean).join(' | ');
        text.append(title, meta);
        button.append(dot, text);
        button.addEventListener('click', () => selectItem(item.index));
        els.itemList.append(button);
      }
      updateProgress();
    }

    function updateDecisionButtons() {
      const decision = state.annotation ? state.annotation.decision : null;
      document.querySelectorAll('[data-decision]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.decision === decision));
      });
    }

    function renderViews() {
      els.viewTabs.replaceChildren();
      const item = state.items[state.current];
      if (!item) return;
      for (const view of item.views) {
        const button = document.createElement('button');
        button.className = 'view-tab';
        button.type = 'button';
        button.textContent = view.label;
        button.setAttribute('aria-pressed', String(view.key === state.viewKey));
        button.addEventListener('click', () => switchView(view.key));
        els.viewTabs.append(button);
      }
    }

    function updateMetadata() {
      const item = state.items[state.current];
      if (!item) {
        els.imageTitle.textContent = '-';
        els.imageMeta.textContent = '-';
        return;
      }
      els.imageTitle.textContent = item.title;
      const pieces = [item.scene, item.side, item.proposal_count == null ? '' : `${item.proposal_count} regions`];
      els.imageMeta.textContent = pieces.filter(Boolean).join(' | ');
    }

    function markDirty(delay = 220) {
      state.dirty = true;
      state.revision += 1;
      setSaveState('Saving');
      window.clearTimeout(state.saveTimer);
      state.saveTimer = window.setTimeout(() => { saveCurrent(); }, delay);
    }

    function currentPayload() {
      const item = state.items[state.current];
      if (!item || !state.annotation) return null;
      const annotation = state.annotation;
      return {
        item_id: item.id,
        decision: annotation.decision,
        shape_mode: annotation.shape_mode,
        box: annotation.box,
        strokes: annotation.strokes,
        note: annotation.note,
        selected_proposal: annotation.selected_proposal,
        image_width: state.imageWidth || annotation.image_width || 0,
        image_height: state.imageHeight || annotation.image_height || 0,
      };
    }

    async function saveCurrent() {
      window.clearTimeout(state.saveTimer);
      if (!state.dirty) return;
      const payload = currentPayload();
      if (!payload) return;
      const revision = state.revision;
      state.saveChain = state.saveChain.then(async () => {
        try {
          const response = await fetch('/api/annotation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          const body = await response.json();
          if (!response.ok) throw new Error(body.error || `save failed (${response.status})`);
          const item = state.items.find((entry) => entry.id === payload.item_id);
          if (item) item.annotation = body.annotation;
          const currentItem = state.items[state.current];
          if (revision === state.revision && currentItem && currentItem.id === payload.item_id) {
            state.dirty = false;
            setSaveState('Saved');
          }
          renderList();
          clearError();
        } catch (error) {
          setSaveState('Save failed');
          showError(error.message || String(error));
        }
      });
      return state.saveChain;
    }

    async function selectItem(index, options = {}) {
      if (!Number.isInteger(index) || index < 0 || index >= state.items.length) return;
      if (index === state.current && !options.force) return;
      await saveCurrent();
      state.current = index;
      const item = state.items[index];
      state.annotation = clone(item.annotation || emptyAnnotation(item));
      state.annotation.decision = state.annotation.decision || null;
      state.annotation.strokes = state.annotation.strokes || [];
      state.annotation.box = state.annotation.box || null;
      state.mode = state.annotation.shape_mode || state.mode;
      els.note.value = state.annotation.note || '';
      state.dirty = false;
      state.revision += 1;
      state.viewTouched = false;
      state.image = null;
      state.imageWidth = 0;
      state.imageHeight = 0;
      updateDecisionButtons();
      renderList();
      updateMetadata();
      const defaultView = item.views[0];
      state.viewKey = defaultView ? defaultView.key : null;
      renderViews();
      await loadView();
      localStorage.setItem('shadow-screener-current', item.id);
    }

    async function switchView(viewKey) {
      if (viewKey === state.viewKey) return;
      state.viewKey = viewKey;
      renderViews();
      await loadView();
    }

    async function loadView() {
      const item = state.items[state.current];
      if (!item || !state.viewKey) return;
      const view = item.views.find((entry) => entry.key === state.viewKey);
      if (!view) return;
      const request = ++state.viewRequest;
      const image = new Image();
      image.decoding = 'async';
      image.onload = () => {
        if (request !== state.viewRequest) return;
        const first = !state.image;
        state.image = image;
        state.imageWidth = image.naturalWidth;
        state.imageHeight = image.naturalHeight;
        if (!state.annotation.image_width || !state.annotation.image_height) {
          state.annotation.image_width = image.naturalWidth;
          state.annotation.image_height = image.naturalHeight;
        }
        els.empty.hidden = true;
        if (first || !state.viewTouched) fitView();
        draw();
      };
      image.onerror = () => {
        if (request !== state.viewRequest) return;
        state.image = null;
        els.empty.hidden = false;
        showError(`Cannot load ${view.label} image`);
        draw();
      };
      const version = item.annotation && item.annotation.updated_at ? item.annotation.updated_at : '';
      image.src = `${view.url}?v=${encodeURIComponent(version)}`;
    }

    function resizeCanvas() {
      const rect = els.wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      if (els.canvas.width !== Math.round(width * dpr) || els.canvas.height !== Math.round(height * dpr)) {
        els.canvas.width = Math.round(width * dpr);
        els.canvas.height = Math.round(height * dpr);
      }
      if (state.image && !state.viewTouched) fitView();
      draw();
    }

    function viewport() {
      const rect = els.canvas.getBoundingClientRect();
      return { width: rect.width, height: rect.height, dpr: window.devicePixelRatio || 1 };
    }

    function fitView() {
      if (!state.image) return;
      const view = viewport();
      const scale = Math.min(view.width / state.imageWidth, view.height / state.imageHeight) * 0.96;
      state.zoom = Math.max(0.001, scale);
      state.offsetX = (view.width - state.imageWidth * state.zoom) / 2;
      state.offsetY = (view.height - state.imageHeight * state.zoom) / 2;
    }

    function draw() {
      const view = viewport();
      ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
      ctx.clearRect(0, 0, view.width, view.height);
      ctx.fillStyle = '#101614';
      ctx.fillRect(0, 0, view.width, view.height);
      if (!state.image) return;

      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(
        state.image,
        state.offsetX,
        state.offsetY,
        state.imageWidth * state.zoom,
        state.imageHeight * state.zoom,
      );

      if (state.annotation) {
        const box = state.annotation.box;
        if (box) {
          const x = state.offsetX + box[0] * state.imageWidth * state.zoom;
          const y = state.offsetY + box[1] * state.imageHeight * state.zoom;
          const w = (box[2] - box[0]) * state.imageWidth * state.zoom;
          const h = (box[3] - box[1]) * state.imageHeight * state.zoom;
          ctx.fillStyle = 'rgba(17, 194, 166, 0.13)';
          ctx.fillRect(x, y, w, h);
          ctx.strokeStyle = '#1fd1b0';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, w, h);
        }
        for (const stroke of state.annotation.strokes || []) {
          const points = stroke.points || [];
          if (!points.length) continue;
          ctx.beginPath();
          points.forEach((point, index) => {
            const x = state.offsetX + point[0] * state.imageWidth * state.zoom;
            const y = state.offsetY + point[1] * state.imageHeight * state.zoom;
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.strokeStyle = 'rgba(31, 209, 176, 0.78)';
          ctx.lineWidth = Math.max(1, stroke.width * state.imageWidth * state.zoom);
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.stroke();
        }
      }
    }

    function canvasPoint(event) {
      const rect = els.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function imagePoint(event) {
      if (!state.image) return null;
      const point = canvasPoint(event);
      const x = (point.x - state.offsetX) / Math.max(0.000001, state.imageWidth * state.zoom);
      const y = (point.y - state.offsetY) / Math.max(0.000001, state.imageHeight * state.zoom);
      return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
    }

    function startDrawing(event) {
      const point = imagePoint(event);
      if (!point) return;
      if (state.mode === 'box') {
        state.drawing = { type: 'box', start: point, current: point };
      } else {
        const rect = els.canvas.getBoundingClientRect();
        const width = Number(els.brushSize.value) / Math.max(1, state.imageWidth * state.zoom);
        const stroke = { width, points: [[point.x, point.y]] };
        state.annotation.strokes.push(stroke);
        state.drawing = { type: 'stroke', stroke };
      }
      draw();
    }

    function moveDrawing(event) {
      const point = imagePoint(event);
      if (!point || !state.drawing) return;
      if (state.drawing.type === 'box') {
        state.drawing.current = point;
        const start = state.drawing.start;
        state.annotation.box = [
          Math.min(start.x, point.x),
          Math.min(start.y, point.y),
          Math.max(start.x, point.x),
          Math.max(start.y, point.y),
        ];
      } else {
        const points = state.drawing.stroke.points;
        const last = points[points.length - 1];
        const dx = (point.x - last[0]) * state.imageWidth * state.zoom;
        const dy = (point.y - last[1]) * state.imageHeight * state.zoom;
        if (Math.hypot(dx, dy) >= 1.5) points.push([point.x, point.y]);
      }
      draw();
    }

    function finishDrawing() {
      if (!state.drawing) return;
      if (state.drawing.type === 'box') {
        const box = state.annotation.box;
        if (!box || (box[2] - box[0]) * state.imageWidth < 3 || (box[3] - box[1]) * state.imageHeight < 3) {
          state.annotation.box = null;
        }
      }
      state.drawing = null;
      markDirty(160);
      draw();
    }

    function undoShape() {
      if (!state.annotation) return;
      if (state.annotation.box) {
        state.annotation.box = null;
      } else if (state.annotation.strokes.length) {
        state.annotation.strokes.pop();
      } else {
        return;
      }
      markDirty(80);
      draw();
    }

    function clearShapes() {
      if (!state.annotation) return;
      state.annotation.box = null;
      state.annotation.strokes = [];
      markDirty(80);
      draw();
    }

    function setMode(mode) {
      state.mode = mode;
      if (state.annotation) state.annotation.shape_mode = mode;
      els.boxMode.setAttribute('aria-pressed', String(mode === 'box'));
      els.freehandMode.setAttribute('aria-pressed', String(mode === 'freehand'));
      if (state.annotation) markDirty(120);
    }

    function setDecision(decision) {
      if (!state.annotation) return;
      state.annotation.decision = decision;
      updateDecisionButtons();
      markDirty(20);
      renderList();
      if (els.autoNext.checked) {
        window.setTimeout(async () => {
          await saveCurrent();
          const next = nextUnreviewed(state.current + 1);
          if (next >= 0) await selectItem(next);
        }, 140);
      }
    }

    function nextUnreviewed(start) {
      for (let step = 0; step < state.items.length; step += 1) {
        const index = (start + step) % state.items.length;
        if (decisionOf(state.items[index]) === 'unreviewed') return index;
      }
      return -1;
    }

    function currentVisibleIndexes() {
      return state.items.filter(matchesFilter).map((item) => item.index);
    }

    function stepVisible(direction) {
      const visible = currentVisibleIndexes();
      if (!visible.length) return;
      let position = visible.indexOf(state.current);
      if (position < 0) position = direction > 0 ? -1 : visible.length;
      const nextPosition = Math.max(0, Math.min(visible.length - 1, position + direction));
      selectItem(visible[nextPosition]);
    }

    els.canvas.addEventListener('pointerdown', (event) => {
      if (!state.image) return;
      els.canvas.setPointerCapture(event.pointerId);
      if (event.button === 1 || state.spaceDown) {
        const point = canvasPoint(event);
        state.panning = { x: point.x, y: point.y, offsetX: state.offsetX, offsetY: state.offsetY };
        els.canvas.classList.add('panning');
        event.preventDefault();
        return;
      }
      if (event.button !== 0) return;
      startDrawing(event);
      event.preventDefault();
    });

    els.canvas.addEventListener('pointermove', (event) => {
      if (state.panning) {
        const point = canvasPoint(event);
        state.offsetX = state.panning.offsetX + point.x - state.panning.x;
        state.offsetY = state.panning.offsetY + point.y - state.panning.y;
        state.viewTouched = true;
        draw();
        return;
      }
      if (state.drawing) {
        moveDrawing(event);
        event.preventDefault();
      }
    });

    els.canvas.addEventListener('pointerup', (event) => {
      if (state.panning) {
        state.panning = null;
        els.canvas.classList.remove('panning');
      }
      if (state.drawing) finishDrawing();
      if (els.canvas.hasPointerCapture(event.pointerId)) els.canvas.releasePointerCapture(event.pointerId);
    });

    els.canvas.addEventListener('pointercancel', () => {
      state.panning = null;
      state.drawing = null;
      els.canvas.classList.remove('panning');
    });

    els.canvas.addEventListener('wheel', (event) => {
      if (!state.image) return;
      event.preventDefault();
      const point = canvasPoint(event);
      const oldZoom = state.zoom;
      const factor = Math.exp(-event.deltaY * 0.0012);
      const nextZoom = Math.max(0.02, Math.min(30, oldZoom * factor));
      const imageX = (point.x - state.offsetX) / oldZoom;
      const imageY = (point.y - state.offsetY) / oldZoom;
      state.zoom = nextZoom;
      state.offsetX = point.x - imageX * nextZoom;
      state.offsetY = point.y - imageY * nextZoom;
      state.viewTouched = true;
      draw();
    }, { passive: false });

    document.querySelectorAll('[data-decision]').forEach((button) => {
      button.addEventListener('click', () => setDecision(button.dataset.decision));
    });
    document.querySelectorAll('.filter-button').forEach((button) => {
      button.addEventListener('click', () => {
        state.filter = button.dataset.filter;
        document.querySelectorAll('.filter-button').forEach((entry) => {
          entry.setAttribute('aria-pressed', String(entry === button));
        });
        renderList();
        const visible = currentVisibleIndexes();
        if (visible.length && !matchesFilter(state.items[state.current])) selectItem(visible[0]);
      });
    });
    els.search.addEventListener('input', () => {
      state.query = els.search.value.trim().toLowerCase();
      renderList();
    });
    els.cameraFilter.addEventListener('change', () => {
      state.camera = els.cameraFilter.value;
      renderList();
      const visible = currentVisibleIndexes();
      if (visible.length && !matchesFilter(state.items[state.current])) selectItem(visible[0]);
    });
    els.note.addEventListener('input', () => {
      if (!state.annotation) return;
      state.annotation.note = els.note.value;
      markDirty(350);
    });
    els.boxMode.addEventListener('click', () => setMode('box'));
    els.freehandMode.addEventListener('click', () => setMode('freehand'));
    els.undo.addEventListener('click', undoShape);
    els.clear.addEventListener('click', clearShapes);
    els.prev.addEventListener('click', () => stepVisible(-1));
    els.next.addEventListener('click', () => stepVisible(1));

    document.addEventListener('keydown', (event) => {
      const target = event.target;
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
      if (event.code === 'Space') {
        state.spaceDown = true;
        event.preventDefault();
      } else if (event.key === 'ArrowLeft') {
        stepVisible(-1);
        event.preventDefault();
      } else if (event.key === 'ArrowRight') {
        stepVisible(1);
        event.preventDefault();
      } else if (event.key === '1') {
        setDecision('shadow');
      } else if (event.key === '2') {
        setDecision('not_shadow');
      } else if (event.key === '3') {
        setDecision('uncertain');
      } else if (event.key.toLowerCase() === 'b') {
        setMode('box');
      } else if (event.key.toLowerCase() === 'f') {
        setMode('freehand');
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        undoShape();
        event.preventDefault();
      }
    });
    document.addEventListener('keyup', (event) => {
      if (event.code === 'Space') state.spaceDown = false;
    });
    window.addEventListener('beforeunload', () => {
      if (!state.dirty) return;
      const payload = currentPayload();
      if (!payload) return;
      navigator.sendBeacon('/api/annotation', new Blob([JSON.stringify(payload)], { type: 'application/json' }));
    });

    async function boot() {
      try {
        const response = await fetch('/api/state');
        if (!response.ok) throw new Error(`state failed (${response.status})`);
        const body = await response.json();
        state.items = body.items;
        const cameras = [...new Set(state.items.map((item) => item.side).filter(Boolean))].sort();
        for (const camera of cameras) {
          const option = document.createElement('option');
          option.value = camera;
          option.textContent = camera;
          els.cameraFilter.append(option);
        }
        renderList();
        if (!state.items.length) {
          els.empty.hidden = false;
          return;
        }
        const previous = localStorage.getItem('shadow-screener-current');
        const previousIndex = state.items.findIndex((item) => item.id === previous);
        const firstOpen = nextUnreviewed(0);
        await selectItem(previousIndex >= 0 ? previousIndex : (firstOpen >= 0 ? firstOpen : 0), { force: true });
      } catch (error) {
        setSaveState('Load failed');
        showError(error.message || String(error));
      }
    }

    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(els.wrap);
    setMode('box');
    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
