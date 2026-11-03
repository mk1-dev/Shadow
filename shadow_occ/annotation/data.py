from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


GT_SOURCES = {"manual_shadow_gt", "public_shadow_gt", "synthetic_shadow_gt"}
WEAK_SOURCES = {"geometry_weak", "temporal_weak", "model_pseudo"}
SOURCES = GT_SOURCES | WEAK_SOURCES


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_array(value, base: Path) -> np.ndarray:
    """A mask reference is a path, or {path: ..., key: ...} for NPZ arrays."""
    ref = value if isinstance(value, dict) else {"path": value}
    path = resolve_path(ref["path"], base)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            array = np.asarray(archive[ref.get("key", "shadow_mask")]).copy()
    elif path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise FileNotFoundError(path)
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask at {path}; got {array.shape}")
    return array


def make_target(row: dict, shape: tuple[int, int], base: Path) -> np.ndarray:
    """Return 0=known background, 1=known shadow, 255=unobserved/uncertain."""
    target = np.full(shape, 255, dtype=np.uint8)
    if row.get("shadow_mask") is not None:
        source = row.get("label_source")
        if source not in SOURCES:
            raise ValueError("shadow_mask requires an explicit shadow label_source")
        raw = load_array(row["shadow_mask"], base)
        if raw.shape != shape:
            raise ValueError("Shadow masks must already be in the image coordinate frame")
        encoding = row.get("mask_encoding", "binary" if source in GT_SOURCES else "positive_only")
        if encoding == "trimap":
            if not np.isin(raw, [0, 1, 255]).all():
                raise ValueError("trimap values must be 0, 1, 255")
            target = raw.astype(np.uint8)
        elif encoding == "binary":
            target = (raw > 0).astype(np.uint8)
        elif encoding == "positive_only":
            target[raw > 0] = 1
        else:
            raise ValueError(f"Unknown mask_encoding: {encoding}")
    if row.get("negative_mask") is not None:
        if row.get("label_source") not in SOURCES:
            raise ValueError("negative_mask requires an explicit shadow label_source")
        negative = load_array(row["negative_mask"], base)
        if negative.shape != shape:
            raise ValueError("negative_mask shape must match the image")
        target[(negative > 0) & (target != 1)] = 0
    if row.get("valid_mask") is not None:
        valid = load_array(row["valid_mask"], base)
        if valid.shape != shape:
            raise ValueError("valid_mask shape must match the image")
        target[valid <= 0] = 255
    return target


def group_split(scene: str, official_split: str, dev_fraction: float, seed: int) -> str:
    if official_split in {"val", "mini_val", "eval", "test"}:
        return "eval"
    if official_split not in {"train", "mini_train", "external"}:
        raise ValueError(f"Unknown official_split: {official_split}")
    number = int(hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest()[:12], 16)
    return "dev" if number / float(16**12) < dev_fraction else "train"


def assert_disjoint(train: list[dict], other: list[dict]) -> None:
    overlap = {row["scene_name"] for row in train} & {row["scene_name"] for row in other}
    if overlap:
        raise ValueError(f"Scene leakage: {sorted(overlap)[:10]}")
    for rows, expected in ((train, "train"), (other, None)):
        if expected and any(row.get("split") != expected for row in rows):
            raise ValueError("Training manifest contains rows outside the train split")
        if expected and any(row.get("official_split") in {"val", "mini_val", "test", "eval"} for row in rows):
            raise ValueError("Official eval/val scenes cannot be used for training")


def checked_write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Could not write {path}")
