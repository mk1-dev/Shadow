from __future__ import annotations

import numpy as np


def transform_matrix(translation: np.ndarray | list[float], rotation: np.ndarray | list[float]) -> np.ndarray:
    """Build a 4x4 transform from nuScenes translation and wxyz quaternion."""
    from pyquaternion import Quaternion

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 3)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (homo @ transform.T)[:, :3]


def yaw_from_transform(transform: np.ndarray) -> float:
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def make_2d_transform_from_ego(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = transform[:2, :2].astype(np.float64)
    trans = transform[:2, 3].astype(np.float64)
    return rot, trans
