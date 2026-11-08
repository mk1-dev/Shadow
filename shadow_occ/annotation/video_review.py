"""Read-only review of frozen video outputs; no discovery or label generation."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from shadow_occ.annotation.data import digest, read_rows


CHANGE_NAMES = ("added_positive", "removed_positive", "owner_added", "owner_lost", "owner_switched")
FRAME_SIGNALS = ("appearance_score_map", "bev_anchor_consistency_by_frame", "anchor_appearance_support_score",
    "flow_appearance_support_score", "cross_space_discovery_score", "cross_space_direct_seed",
    "cross_space_temporal_seed", "cross_space_spatial_growth", "cross_space_region_appearance_support",
    "receiver_valid_by_frame", "boundary_confidence", "proposal_source_bits")
PAIR_SIGNALS = ("delta_L", "spatial_bev_anchor_cosine", "ground_flow_geometry_residual_px",
    "m2_flow_state", "m2_flow_quality", "entity_explanation_map", "entity_explanation_valid_map")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def artifact_path(folder, value, default):
    """Prefer relocated local artifacts over absolute paths from the HPC run."""
    basename = str(value or default).replace("\\", "/").rsplit("/", 1)[-1]
    for path in (folder/basename, folder/default, Path(value) if value else folder/default):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(folder/basename)


def find_clips(root):
    root = Path(root).resolve()
    protocols = [root/"protocol.json"] if (root/"protocol.json").is_file() else sorted(root.glob("*/protocol.json"))
    result = {}
    for path in protocols:
        protocol = json.loads(path.read_text(encoding="utf-8"))
        rows = read_rows(path.parent/"frames.jsonl")
        tokens = tuple(row["frame_token"] for row in rows)
        if not tokens or len(tokens) != len(set(tokens)) or any(
                a["timestamp"] >= b["timestamp"] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"Expected unique, time-ordered camera frames: {path.parent}")
        key = (protocol["clip"]["scene_name"], protocol["clip"]["camera"], tokens)
        if key in result:
            raise ValueError(f"Duplicate clip window: {path.parent}")
        result[key] = (path.parent, protocol, rows)
    if not result:
        raise ValueError(f"No video clip protocols under {root}")
    return result


def load_clip(entry):
    folder, protocol, rows = entry
    arrays, images, fingerprints = [], [], {}
    for name in ("protocol.json", "frames.jsonl", "region_memory.jsonl", "caster_memory.jsonl", "pair_evidence.npz"):
        if (folder/name).is_file():
            fingerprints[str(folder/name)] = digest(folder/name)
    for path in sorted(folder.glob("*_light_bank.npz")):
        fingerprints[str(path)] = digest(path)
    for slot, row in enumerate(rows):
        path = artifact_path(folder, row.get("evidence"), f"{slot:03d}.npz")
        arrays.append(load_npz(path))
        fingerprints[str(path)] = digest(path)
        image_path = artifact_path(folder, row.get("image_representation"), f"{slot:03d}_image.npz")
        images.append(load_npz(image_path))
        fingerprints[str(image_path)] = digest(image_path)
    memory = read_rows(folder/"region_memory.jsonl") if (folder/"region_memory.jsonl").is_file() else []
    casters = read_rows(folder/"caster_memory.jsonl") if (folder/"caster_memory.jsonl").is_file() else []
    return {"folder": folder, "protocol": protocol, "rows": rows, "arrays": arrays, "images": images,
        "memory": {row["frame_token"]: row["regions"] for row in memory},
        "casters": {row["instance_token"]: row for row in casters}, "fingerprints": fingerprints,
        "pair": load_npz(folder/"pair_evidence.npz") if (folder/"pair_evidence.npz").is_file() else {}}


def owner_tokens(arrays, protocol):
    vocabulary = protocol["owner_index_to_instance_token"]
    indices = arrays["shadow_owner_id"]
    positive = arrays["shadow_target_bev"] == 1
    known = positive & (indices >= 0)
    if known.any() and indices[known].max() >= len(vocabulary):
        raise ValueError("Owner index is outside this run's caster vocabulary")
    result = np.full(indices.shape, "", dtype=f"<U{max([len(value) for value in vocabulary]+[1])}")
    if known.any():
        result[known] = np.asarray(vocabulary)[indices[known]]
    return result


def compare_frame(before, after, before_protocol, after_protocol):
    for key in ("receiver_world_xyz", "bev_world_to_grid"):
        if before[key].shape != after[key].shape or not np.allclose(before[key], after[key], atol=1e-5, rtol=0, equal_nan=True):
            raise ValueError(f"Cannot compare different receiver grids: {key}")
    old, new = before["shadow_target_bev"] == 1, after["shadow_target_bev"] == 1
    if old.shape != new.shape:
        raise ValueError("Cannot resize labels for a frozen comparison")
    old_owner, new_owner = owner_tokens(before, before_protocol), owner_tokens(after, after_protocol)
    changes = {"added_positive": new & ~old, "removed_positive": old & ~new,
        "owner_added": (old_owner == "") & (new_owner != ""),
        "owner_lost": (old_owner != "") & (new_owner == ""),
        "owner_switched": (old_owner != "") & (new_owner != "") & (old_owner != new_owner)}
    return changes, old_owner, new_owner


def change_components(before, after, changes):
    """Review changed cells inside their full affected connected proposal region."""
    changed = np.logical_or.reduce(list(changes.values()))
    domain = before["shadow_proposal_bev"] | after["shadow_proposal_bev"] | changed
    count, labels = cv2.connectedComponents(domain.astype(np.uint8), connectivity=8)
    return [labels == label for label in range(1, count) if (changed & (labels == label)).any()]


def _region_record(clip, slot, identity, tokens):
    arrays = clip["arrays"][slot]
    mask = (arrays["shadow_track_id"] == identity) & (arrays["shadow_target_bev"] == 1)
    row = next((row for row in clip["rows"][slot]["regions"] if row["shadow_track_id"] == identity), {})
    return {"track_id_within_run": int(identity), "region_id": row.get("id"),
        "positive_cells": int(mask.sum()), "resolved_cells": int((mask & (tokens != "")).sum()),
        "caster_tokens": sorted(set(tokens[mask])-{ "" }), "physics": row.get("physics")}


def owner_region_transitions(before, after, slot, old_tokens, new_tokens):
    """Bipartite spatial overlap, never equality of run-local track/owner IDs."""
    old, new = before["arrays"][slot], after["arrays"][slot]
    old_positive, new_positive = old["shadow_target_bev"] == 1, new["shadow_target_bev"] == 1
    ids = [old["shadow_track_id"], new["shadow_track_id"]]
    graph = {(side, int(identity)): set() for side, mask in enumerate((old_positive, new_positive))
             for identity in np.unique(ids[side][mask]) if identity > 0}
    overlap = old_positive & new_positive & (ids[0] > 0) & (ids[1] > 0)
    for a, b in np.unique(np.stack([ids[0][overlap], ids[1][overlap]], axis=-1), axis=0):
        graph[(0, int(a))].add((1, int(b)))
        graph[(1, int(b))].add((0, int(a)))
    result = []
    while graph:
        pending, group = [next(iter(graph))], set()
        while pending:
            node = pending.pop()
            if node not in group:
                group.add(node)
                pending.extend(graph[node]-group)
        for node in group:
            del graph[node]
        old_rows = [_region_record(before, slot, identity, old_tokens) for side, identity in sorted(group) if side == 0]
        new_rows = [_region_record(after, slot, identity, new_tokens) for side, identity in sorted(group) if side == 1]
        a = {token for row in old_rows for token in row["caster_tokens"]}
        b = {token for row in new_rows for token in row["caster_tokens"]}
        relation = ("new_only" if not old_rows else "removed_only" if not new_rows else
                    "one_to_one" if len(old_rows) == len(new_rows) == 1 else
                    "split" if len(old_rows) == 1 else "merge" if len(new_rows) == 1 else "many_to_many")
        status = ("resolved_new_region" if not old_rows and b else "unresolved_new_region" if not old_rows else
                  "removed_resolved_region" if not new_rows and a else "removed_unresolved_region" if not new_rows else
                  "unresolved" if not a and not b else "became_resolved" if not a else
                  "lost_resolution" if not b else "same_owner_set" if a == b else "owner_set_changed")
        result.append({"frame_token": before["rows"][slot]["frame_token"], "relation": relation,
            "owner_change": status, "before": old_rows, "after": new_rows})
    return result


def observation_timeline(clip, version):
    result = []
    for slot, frame in enumerate(clip["rows"]):
        arrays, image = clip["arrays"][slot], clip["images"][slot]
        observed = {row["shadow_track_id"]: row for row in frame["regions"]}
        memory = {row["shadow_track_id"]: row for row in clip["memory"].get(frame["frame_token"], [])}
        image_ids = image["shadow_region_id_image"]
        identities = set(observed) | set(memory) | set(np.unique(image_ids)) | set(np.unique(arrays["shadow_track_id"]))
        for identity in sorted(identities-{0}):
            image_mask = (image_ids == identity) & image["shadow_proposal_image"]
            bev_mask = (arrays["shadow_track_id"] == identity) & arrays["shadow_proposal_bev"]
            current = identity in observed
            state = ("observed_image_mask" if current and image_mask.any() else
                     "observed_bev_only" if current and bev_mask.any() else
                     "observed_region_without_mask" if current else
                     "orphan_mask" if image_mask.any() or bev_mask.any() else "retained_memory_only")
            result.append({"version": version, "frame_slot": slot, "frame_token": frame["frame_token"],
                "timestamp": frame["timestamp"], "track_id_within_run": int(identity), "observation_state": state,
                "current_region_present": current, "observed_image_mask": current and bool(image_mask.any()),
                "image_proposal_pixels": int(image_mask.sum()), "bev_proposal_cells": int(bev_mask.sum()),
                "image_positive_pixels": int((image_mask & (image["shadow_target_image"] == 1)).sum()),
                "bev_positive_cells": int((bev_mask & (arrays["shadow_target_bev"] == 1)).sum()),
                "memory_visibility": memory.get(identity, {}).get("visibility_status"),
                "missed_frames": memory.get(identity, {}).get("missed_frames"),
                "last_seen_region_id": memory.get(identity, {}).get("last_seen_region_id"),
                "memory_trace_available": (clip["folder"]/"region_memory.jsonl").is_file(),
                "memory_score_is_not_current_observation": True})
    return result


def summarize_tracks(timeline, frame_count):
    result = []
    identities = sorted({row["track_id_within_run"] for row in timeline})
    for identity in identities:
        rows = [row for row in timeline if row["track_id_within_run"] == identity]
        observed = {row["frame_slot"] for row in rows if row["observed_image_mask"]}
        longest = streak = 0
        for slot in range(frame_count):
            streak = streak+1 if slot in observed else 0
            longest = max(longest, streak)
        result.append({"version": rows[0]["version"], "track_id_within_run": identity,
            "observed_image_frames": len(observed), "longest_observed_image_streak": longest,
            "observed_in_every_clip_frame": len(observed) == frame_count,
            "memory_only_frames": sum(row["observation_state"] == "retained_memory_only" for row in rows),
            "memory_trace_available": all(row["memory_trace_available"] for row in rows),
            "orphan_mask_frames": sum(row["observation_state"] == "orphan_mask" for row in rows),
            "active_or_remembered_frames": len(rows)})
    return result


def finite_stats(value, mask):
    values = value[mask]
    values = values[np.isfinite(values)]
    return {"finite_cells": int(len(values)), "mean": float(values.mean()) if len(values) else None,
        "min": float(values.min()) if len(values) else None, "max": float(values.max()) if len(values) else None}


def evidence_record(clip, slot, domain, changed):
    arrays = clip["arrays"][slot]
    frame = {key: {"affected_region": finite_stats(arrays[key], domain),
                   "changed_cells": finite_stats(arrays[key], changed)}
             for key in FRAME_SIGNALS if key in arrays}
    intervals = []
    for at in (slot-1, slot):
        if 0 <= at < len(clip["rows"])-1:
            intervals.append({"frame_slots": [at, at+1], "timestamps": [clip["rows"][at]["timestamp"], clip["rows"][at+1]["timestamp"]],
                "signals": {key: finite_stats(clip["pair"][key][at], domain)
                            for key in PAIR_SIGNALS if key in clip["pair"]}})
    identities = set(np.unique(arrays["shadow_track_id"][domain]))-{0}
    return {"frame_signals": frame, "missing_frame_signals": [key for key in FRAME_SIGNALS if key not in arrays],
        "intervals": intervals, "missing_pair_signals": [key for key in PAIR_SIGNALS if key not in clip["pair"]],
        "regions": [row for row in clip["rows"][slot]["regions"] if row["shadow_track_id"] in identities],
        "anchor_only_does_not_mean_no_appearance": True, "source_bits_are_provenance_not_causal_attribution": True}
