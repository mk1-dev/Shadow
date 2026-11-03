from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shadow_image_screener.py"
SPEC = importlib.util.spec_from_file_location("shadow_image_screener", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

MINI_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "make_nuscenes_mini_review_manifest.py"
)
MINI_SPEC = importlib.util.spec_from_file_location("make_nuscenes_mini_review_manifest", MINI_SCRIPT)
MINI_MODULE = importlib.util.module_from_spec(MINI_SPEC)
assert MINI_SPEC and MINI_SPEC.loader
MINI_SPEC.loader.exec_module(MINI_MODULE)


class ShadowImageScreenerTest(unittest.TestCase):
    def test_manifest_and_annotation_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "crop.png"
            image.write_bytes(b"not-a-real-image")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"id": "pair-1", "scene": "scene-0001", "clean": "crop.png"}) + "\n",
                encoding="utf-8",
            )

            items = MODULE.load_manifest_items(
                manifest,
                root=root,
                view_fields=["clean", "numbered"],
            )
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["id"], "pair-1")
            self.assertEqual(items[0]["views"][0]["key"], "clean")

            store = MODULE.AnnotationStore(root / "annotations.jsonl", {"pair-1"})
            saved = store.save(
                {
                    "item_id": "pair-1",
                    "decision": "shadow",
                    "shape_mode": "freehand",
                    "box": None,
                    "strokes": [{"width": 0.01, "points": [[0.1, 0.2], [0.2, 0.3]]}],
                    "note": "test",
                    "image_width": 640,
                    "image_height": 360,
                }
            )
            self.assertEqual(saved["decision"], "shadow")
            self.assertEqual(len(store.latest), 1)

            reloaded = MODULE.AnnotationStore(root / "annotations.jsonl", {"pair-1"})
            self.assertEqual(reloaded.latest["pair-1"]["strokes"][0]["points"][1], [0.2, 0.3])

    def test_rejects_unknown_item(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = MODULE.AnnotationStore(Path(temp) / "annotations.jsonl", {"known"})
            with self.assertRaises(ValueError):
                store.save({"item_id": "unknown"})

    def test_server_state_and_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "crop.png"
            image.write_bytes(b"image-bytes")
            items = [
                {
                    "id": "item-1",
                    "index": 0,
                    "title": "item-1",
                    "scene": "scene-1",
                    "side": "candidate",
                    "proposal_count": 1,
                    "group": "pair-1",
                    "views": [
                        {
                            "key": "clean",
                            "label": "Clean",
                            "path": image,
                            "mime": "image/png",
                        }
                    ],
                }
            ]
            store = MODULE.AnnotationStore(root / "annotations.jsonl", {"item-1"})
            server = MODULE.create_server("127.0.0.1", 0, items, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base}/api/state") as response:
                    body = json.loads(response.read())
                self.assertEqual(body["items"][0]["id"], "item-1")

                request = urllib.request.Request(
                    f"{base}/api/annotation",
                    data=json.dumps(
                        {
                            "item_id": "item-1",
                            "decision": "not_shadow",
                            "shape_mode": "box",
                            "box": [0.1, 0.2, 0.5, 0.8],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    saved = json.loads(response.read())
                self.assertTrue(saved["ok"])
                with urllib.request.urlopen(f"{base}/api/state") as response:
                    body = json.loads(response.read())
                self.assertEqual(body["items"][0]["annotation"]["decision"], "not_shadow")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_nuscenes_mini_manifest_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataroot = Path(temp)
            version = "v1.0-mini"
            metadata = dataroot / version
            metadata.mkdir()
            image = dataroot / "samples" / "CAM_FRONT" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")

            def write(name: str, rows: list[dict]) -> None:
                (metadata / name).write_text(json.dumps(rows), encoding="utf-8")

            write("scene.json", [{"token": "scene-token", "name": "scene-0001"}])
            write("sample.json", [{"token": "sample-token", "scene_token": "scene-token"}])
            write("sensor.json", [{"token": "sensor-token", "channel": "CAM_FRONT"}])
            write("calibrated_sensor.json", [{"token": "calib-token", "sensor_token": "sensor-token"}])
            write(
                "sample_data.json",
                [
                    {
                        "sample_token": "sample-token",
                        "calibrated_sensor_token": "calib-token",
                        "is_key_frame": True,
                        "timestamp": 123,
                        "filename": "samples/CAM_FRONT/frame.jpg",
                    }
                ],
            )

            rows = MINI_MODULE.build_manifest_rows(
                dataroot,
                version=version,
                cameras=["CAM_FRONT"],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scene"], "scene-0001")
            self.assertEqual(rows[0]["source"], "CAM_FRONT")
            self.assertEqual(rows[0]["image"], "samples/CAM_FRONT/frame.jpg")


if __name__ == "__main__":
    unittest.main()
