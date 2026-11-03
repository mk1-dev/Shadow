# Local nuScenes-mini annotation

This package runs a local browser annotation UI using only Python 3.9+ and a
modern browser. It does not need Flask, OpenCV, Node.js, or HPC access.

## Contents

```text
run_local_annotation.py
scripts/shadow_image_screener.py
scripts/make_nuscenes_mini_review_manifest.py
tests/test_shadow_image_screener.py
LOCAL_ANNOTATION_README.md
```

## Quick start

The nuScenes-mini directory must contain `v1.0-mini/`, `samples/`, and
optionally `sweeps/`, matching the standard nuScenes layout.

Windows:

```bat
python run_local_annotation.py --dataroot D:\datasets\nuscenes-mini --work-dir D:\annotation\shadow_mini
```

Linux or macOS:

```bash
python3 run_local_annotation.py \
  --dataroot /path/to/nuscenes-mini \
  --work-dir /path/to/shadow_mini_annotation
```

The command builds the six-camera keyframe manifest, starts the UI at
`http://127.0.0.1:8765/`, and opens the browser. Stop it with `Ctrl+C`.

If port 8765 is occupied, use `--port 8766` or choose another port.

## Annotation workflow

- Use `Shadow`, `Not shadow`, or `Uncertain` for each image.
- Filter by camera from the left sidebar.
- Search by scene name, sample token, or image ID.
- Use `Box` for fast image-level screening.
- Use `Brush` for an irregular shadow boundary when a mask is required.
- Mouse wheel zooms; middle-button drag or Space + drag pans.
- `Auto next` advances to the next unreviewed image after a decision.
- Annotations save immediately and resume after restarting the same command.
- `Export JSONL` downloads the latest record for every annotated image.

## Output

The work directory contains:

```text
manifest.jsonl
manifest_summary.json
review_annotations.jsonl
```

`review_annotations.jsonl` is append-only. Each record contains normalized box
and brush coordinates, so the same format works for both annotation modes:

```json
{
  "schema_version": 1,
  "item_id": "scene__sample__CAM_FRONT",
  "decision": "shadow",
  "shape_mode": "freehand",
  "box": [0.12, 0.4, 0.55, 0.8],
  "strokes": [{"width": 0.004, "points": [[0.2, 0.5], [0.3, 0.6]]}],
  "note": "",
  "selected_proposal": null,
  "image_width": 1600,
  "image_height": 900,
  "updated_at": "2026-10-29T00:00:00.000+00:00"
}
```

## Optional controls

Only selected cameras:

```bash
python run_local_annotation.py \
  --dataroot /path/to/nuscenes-mini \
  --cameras CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT
```

Only selected scenes:

```bash
python run_local_annotation.py \
  --dataroot /path/to/nuscenes-mini \
  --scenes scene-0061,scene-0103
```

Include sweeps in addition to keyframes:

```bash
python run_local_annotation.py --dataroot /path/to/nuscenes-mini --all-frames
```

## Test

From the package root:

```bash
python -m unittest tests.test_shadow_image_screener -v
```

## Recommendation

Use box annotation for the first screening pass. Boxes are faster and sufficient
for deciding whether an image is worth keeping. Use brush annotation only for
accepted shadows and ambiguous boundaries, because a rectangular box includes
too much adjacent ground to serve as a final shadow mask.
