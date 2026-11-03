from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shadow_occ.geometry.transforms import apply_transform
from shadow_occ.shadow.labels import ShadowComponent


@dataclass
class TrackState:
    track_id: int
    centroid_global: np.ndarray
    direction_xy: np.ndarray
    root_global: np.ndarray
    tip_global: np.ndarray
    tip_velocity: np.ndarray
    age: int
    missed: int = 0


class ShadowTracker:
    """Simple BEV component tracker in global coordinates."""

    def __init__(
        self,
        max_distance_m: float = 3.0,
        min_direction_cos: float = 0.2,
        max_missed: int = 2,
        min_age: int = 2,
        caster_max_motion_m: float = 0.75,
        tip_max_motion_m: float = 3.0,
        tip_smoothness_sigma_m: float = 1.5,
    ) -> None:
        self.max_distance_m = max_distance_m
        self.min_direction_cos = min_direction_cos
        self.max_missed = max_missed
        self.min_age = min_age
        self.caster_max_motion_m = caster_max_motion_m
        self.tip_max_motion_m = tip_max_motion_m
        self.tip_smoothness_sigma_m = tip_smoothness_sigma_m
        self._next_id = 1
        self.tracks: dict[int, TrackState] = {}

    def update(self, components: list[ShadowComponent], ego_to_global: np.ndarray) -> list[ShadowComponent]:
        centroids = np.asarray([c.centroid_xy for c in components], dtype=np.float64)
        if len(centroids):
            zeros = np.zeros((len(centroids), 1), dtype=np.float64)
            centroids_global = apply_transform(np.concatenate([centroids, zeros], axis=1), ego_to_global)[:, :2]
            roots = np.asarray(
                [c.caster_xy if c.caster_xy is not None else c.root_xy for c in components],
                dtype=np.float64,
            )
            tips = np.asarray([c.tip_xy for c in components], dtype=np.float64)
            roots_global = apply_transform(np.concatenate([roots, zeros], axis=1), ego_to_global)[:, :2]
            tips_global = apply_transform(np.concatenate([tips, zeros], axis=1), ego_to_global)[:, :2]
        else:
            centroids_global = np.zeros((0, 2), dtype=np.float64)
            roots_global = np.zeros((0, 2), dtype=np.float64)
            tips_global = np.zeros((0, 2), dtype=np.float64)

        unmatched_tracks = set(self.tracks.keys())
        for idx, component in enumerate(components):
            direction = np.asarray(component.direction_xy, dtype=np.float64)
            track_id = self._match(centroids_global[idx], direction, unmatched_tracks)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    centroid_global=centroids_global[idx],
                    direction_xy=direction,
                    root_global=roots_global[idx],
                    tip_global=tips_global[idx],
                    tip_velocity=np.zeros(2, dtype=np.float64),
                    age=0,
                )
            else:
                unmatched_tracks.discard(track_id)

            state = self.tracks[track_id]
            root_motion = float(np.linalg.norm(roots_global[idx] - state.root_global))
            tip_delta = tips_global[idx] - state.tip_global
            tip_motion = float(np.linalg.norm(tip_delta))
            tip_velocity = 0.6 * state.tip_velocity + 0.4 * tip_delta
            velocity_change = float(np.linalg.norm(tip_velocity - state.tip_velocity))
            state.centroid_global = 0.6 * state.centroid_global + 0.4 * centroids_global[idx]
            state.direction_xy = _normalize(0.6 * state.direction_xy + 0.4 * direction)
            state.root_global = 0.6 * state.root_global + 0.4 * roots_global[idx]
            state.tip_global = 0.6 * state.tip_global + 0.4 * tips_global[idx]
            state.tip_velocity = tip_velocity
            state.age += 1
            state.missed = 0
            component.track_id = track_id
            component.track_age = state.age
            component.stable = state.age >= self.min_age
            component.caster_stability = float(np.exp(-root_motion / max(self.caster_max_motion_m, 1e-6)))
            motion_score = float(np.exp(-tip_motion / max(self.tip_max_motion_m, 1e-6)))
            smooth_score = float(np.exp(-velocity_change / max(self.tip_smoothness_sigma_m, 1e-6)))
            component.tip_motion_m = tip_motion
            component.tip_smoothness = motion_score * smooth_score

        for track_id in unmatched_tracks:
            self.tracks[track_id].missed += 1
        self.tracks = {
            tid: state for tid, state in self.tracks.items() if state.missed <= self.max_missed
        }
        return components

    def _match(
        self,
        centroid_global: np.ndarray,
        direction: np.ndarray,
        candidate_track_ids: set[int],
    ) -> int | None:
        best_id = None
        best_score = float("inf")
        direction = _normalize(direction)
        for track_id in candidate_track_ids:
            state = self.tracks[track_id]
            dist = float(np.linalg.norm(centroid_global - state.centroid_global))
            if dist > self.max_distance_m:
                continue
            cos = float(np.dot(direction, _normalize(state.direction_xy)))
            if abs(cos) < self.min_direction_cos:
                continue
            if dist < best_score:
                best_score = dist
                best_id = track_id
        return best_id


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return vec
    return vec / norm
