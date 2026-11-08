"""Separate motion-matched entity suppression from persistent caster identity."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from shadow_occ.annotation.physics_labels import window_tracks
from shadow_occ.annotation.physics_motion import project_world


@dataclass(frozen=True)
class EntityMotionConfig:
    core_margin_px: float = 2.0
    maximum_residual_px: float = 2.0
    maximum_fb_error_px: float = 1.5

    def validate(self):
        values = np.asarray(list(asdict(self).values()))
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Entity motion thresholds must be finite and positive")


def match_entity_motion(volumes, fb_consistent, fb_error_px, config=None, *, enabled=True):
    """Match only actual front-surface rays, never a box's rectangular crop.

    Outputs use the world receiver-ray grid. The surface along such a ray is
    an entity; the ground endpoint is occluded. They are not dense image masks.
    No match is propagated to the next ground cell or extrapolated at endpoints.
    """
    config = config or EntityMotionConfig()
    config.validate()
    support = np.asarray(volumes["physical_ray_volume_support"], bool)
    shape = support.shape
    if fb_consistent.shape != (shape[0]-1, *shape[1:]) or fb_error_px.shape != fb_consistent.shape:
        raise ValueError("Entity matching must use the shared interval grid")
    error = volumes["box_surface_flow_residual_px"]
    matched = (volumes["box_surface_flow_valid"] & fb_consistent
        & volumes["box_surface_target_visible"] & support[:-1]
        & (volumes["box_surface_core_distance_px"][:-1] >= config.core_margin_px)
        & np.isfinite(error) & (error <= config.maximum_residual_px)
        & np.isfinite(fb_error_px) & (fb_error_px >= 0) & (fb_error_px <= config.maximum_fb_error_px))
    suppression = np.zeros(shape, bool)
    if enabled:
        suppression[:-1] = matched
    return {"entity_motion_matched_by_interval": matched,
        "entity_suppression_mask": suppression,
        "entity_occlusion_mask": support.copy(),
        "entity_uncertain_mask": support & ~suppression,
        "entity_surface_owner_index": volumes["box_surface_owner_index"],
        "entity_surface_core_distance_px": volumes["box_surface_core_distance_px"],
        "entity_surface_flow_residual_px": error,
        "entity_surface_target_visible": volumes["box_surface_target_visible"]}


def _unit(value):
    value = np.asarray(value, np.float32)
    norm = np.linalg.norm(value)
    return (value/norm).tolist() if np.isfinite(value).all() and norm > 1e-8 else None


class CasterMemory:
    """Clip-scoped GT identity/state store, independent of shadow-search masks.

    Checked triplets retain the existing track-continuity validation. Appearance
    and BEV descriptors are retained separately, not used as shadow truth or as
    an uncalibrated replacement for the physical owner competition.
    """
    def __init__(self, chain, frames):
        self.frames = frames
        self.records = {}
        self.windows = {}
        self.rejected = {}
        timestamps = [frame["camera_timestamp"] for frame in frames]
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Caster memory needs increasing camera timestamps")
        for slot, frame in enumerate(frames):
            for volume in frame["object_volumes_world"]:
                corners = np.asarray(volume["corners_world"], float)
                self._put(volume["instance_token"], volume.get("category"), slot,
                    np.vstack([corners.mean(axis=0), corners]), volume.get("source"))
        for slot, pair in enumerate(chain):
            for obj in pair["objects"]:
                for at, name in ((slot, "landmarks_world_t"), (slot+1, "landmarks_world_next")):
                    self._put(obj["instance_token"], obj.get("category"), at,
                              obj[name], "camera_time_interpolated_track")
        for slot in range(1, len(chain)):
            self.windows[slot], self.rejected[slot] = window_tracks(chain[slot-1], chain[slot])

    def _put(self, token, category, slot, points, source):
        points = np.asarray(points, float)
        if points.shape != (9, 3) or not np.isfinite(points).all():
            raise ValueError("Caster memory requires finite center + eight corners")
        record = self.records.setdefault(token, {"instance_token": token, "category": category, "states": {}})
        if category is not None:
            record["category"] = category
        record["states"][slot] = {"points": points, "source": source}

    def states(self, token):
        return {slot: state["points"] for slot, state in self.records.get(token, {}).get("states", {}).items()}

    def tracks(self, slot):
        return [{**track, "points": np.stack([self.records[track["instance_token"]]["states"][at]["points"]
                    for at in (slot-1, slot, slot+1)])} for track in self.windows.get(slot, [])]

    def observe(self, slot, lab, surface_owner, matched, feature_root, sample_token, sample_bev):
        """Read BEV at entity centers, NOT from receiver-masked ground features."""
        tokens = [token for token, record in self.records.items() if slot in record["states"]]
        if not tokens:
            return
        centers = np.asarray([[self.records[token]["states"][slot]["points"][0] for token in tokens]])
        features, valid, _ = sample_bev(feature_root, sample_token, centers)
        volumes = self.frames[slot]["object_volumes_world"]
        for index, token in enumerate(tokens):
            state = self.records[token]["states"][slot]
            indices = [i for i, volume in enumerate(volumes) if volume["instance_token"] == token]
            selected = np.isin(surface_owner, indices) & matched & np.isfinite(lab).all(axis=-1)
            appearance = None
            if selected.any():
                values = lab[selected]
                appearance = _unit(np.concatenate([np.histogram(values[:, ch], bins=8, range=limits)[0]
                    for ch, limits in enumerate(((0, 100), (0, 256), (0, 256)))]))
            uv, _, in_front = project_world(state["points"][1:], self.frames[slot]["projection_world_to_image"])
            state.update(bev_descriptor=_unit(features[0, index]) if valid[0, index] else None,
                appearance_descriptor=appearance, matched_entity_cells=int(selected.sum()),
                projected_corners_xy=np.where(in_front[:, None], uv, np.nan),
                projection_valid=in_front)

    def association_context(self, token, slot):
        state = self.records[token]["states"][slot]
        return {"caster_memory_id": token, "state_source": state["source"],
            "matched_entity_cells": state.get("matched_entity_cells", 0),
            "appearance_available": state.get("appearance_descriptor") is not None,
            "bev_descriptor_available": state.get("bev_descriptor") is not None,
            "geometry_used_for_projection": True, "descriptors_used_for_owner_score": False}

    def rows(self):
        result = []
        for token, record in sorted(self.records.items()):
            history = []
            previous = None
            for slot, state in sorted(record["states"].items()):
                timestamp = self.frames[slot]["camera_timestamp"]
                center = state["points"][0]
                velocity = None if previous is None else ((center-previous[1])/((timestamp-previous[0])/1e6)).tolist()
                previous = timestamp, center
                uv = state.get("projected_corners_xy")
                history.append({"slot": slot, "camera_token": self.frames[slot]["sample_data_token"],
                    "timestamp": timestamp, "source": state["source"],
                    "center_world": center.tolist(), "corners_world": state["points"][1:].tolist(),
                    "velocity_world_mps": velocity,
                    "projected_corners_xy": None if uv is None else [point.tolist() if np.isfinite(point).all() else None for point in uv],
                    "bev_descriptor": state.get("bev_descriptor"),
                    "appearance_descriptor": state.get("appearance_descriptor"),
                    "matched_entity_cells": state.get("matched_entity_cells", 0)})
            result.append({"instance_token": token, "category": record["category"],
                "identity_source": "nuscenes_GT", "observations": history})
        return result
