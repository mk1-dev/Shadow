"""Spatial BEV cache shared with the existing UniAD/OpenDriveVLA environment."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from shadow_occ.annotation.data import digest
from shadow_occ.utils.io import write_json


def write_cached_bev(output, token, feature, record, timestamp, pc_range, source, objects=None, *, valid_mask=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/f"{token}.npz"
    feature = np.asarray(feature, np.float32)
    native_valid = np.isfinite(feature).all(axis=0) & (np.linalg.norm(feature, axis=0) > 1e-6)
    if valid_mask is not None:
        if np.shape(valid_mask) != feature.shape[1:]:
            raise ValueError("BEV validity must use the native feature grid")
        native_valid &= np.asarray(valid_mask, bool)
    np.savez_compressed(path, bev_chw=feature.astype(np.float16), bev_valid_hw=native_valid)
    object_reference = {}
    if objects is not None:
        from shadow_occ.annotation.mature_objects import object_frame_source, write_cached_objects

        object_reference = write_cached_objects(output, token, objects,
            object_frame_source(record, timestamp, pc_range, source))
    write_json(output/f"{token}.json", {**source, "sample_token": token, "scene_name": record.scene_name,
        **object_reference,
        "coordinate_frame": "LIDAR_TOP", "lidar_to_world": (record.lidar_ego_to_global @ record.lidar_sensor_to_ego).tolist(),
        "pc_range": list(pc_range), "shape_chw": list(feature.shape), "feature_sha256": digest(path),
        "layout": "C,H,W; rows=y; columns=x; cell centres", "timestamp": timestamp,
        "validity": "native finite nonzero descriptor support, not camera visibility or free ground",
        "timestamp_reference": "LIDAR_TOP sample_data microseconds"})


def spatial_bev(value, height, width, layout="HWBC"):
    """UniAD get_bevs returns [H*W, B, C], with columns along metric x."""
    value = np.asarray(value)
    if layout == "HWBC":
        if value.ndim != 3 or value.shape[:2] != (height*width, 1):
            raise ValueError("Expected UniAD spatial bev_embed [H*W, 1, C]")
        value = value[:, 0].reshape(height, width, -1).transpose(2, 0, 1)
    elif layout == "BHWC":
        if value.ndim != 3 or value.shape[:2] != (1, height*width):
            raise ValueError("Expected spatial bev_embed [1, H*W, C]")
        value = value[0].reshape(height, width, -1).transpose(2, 0, 1)
    elif layout == "BCHW":
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[2:] != (height, width):
            raise ValueError("Expected spatial BEV [1, C, H, W]")
        value = value[0]
    else:
        raise ValueError(f"Unknown BEV layout: {layout}")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite BEV features")
    return np.ascontiguousarray(value, dtype=np.float32)


def sample_cached_bev(root, token, world_points, *, return_geometry=False):
    """Map receiver world points to the mature model's current LIDAR_TOP grid."""
    root = Path(root)
    meta = json.loads((root/f"{token}.json").read_text(encoding="utf-8"))
    if meta["sample_token"] != token or meta["coordinate_frame"] != "LIDAR_TOP":
        raise ValueError("BEV cache sample/frame mismatch")
    path = root/f"{token}.npz"
    if digest(path) != meta["feature_sha256"]:
        raise ValueError("BEV feature cache changed")
    with np.load(path, allow_pickle=False) as archive:
        feature = archive["bev_chw"].astype(np.float32)
        native_valid = np.isfinite(feature).all(axis=0) & (np.linalg.norm(feature, axis=0) > 1e-6)
        validity_source = "derived_finite_nonzero_legacy"
        if "bev_valid_hw" in archive:
            if archive["bev_valid_hw"].shape != feature.shape[1:]:
                raise ValueError("Cached BEV validity and native feature grid disagree")
            native_valid &= archive["bev_valid_hw"].astype(bool)
            validity_source = "explicit_native_support"
    lower, upper = np.asarray(meta["pc_range"][:3]), np.asarray(meta["pc_range"][3:])
    height, width = feature.shape[1:]
    xyz = np.c_[world_points.reshape(-1, 3), np.ones(world_points.shape[0]*world_points.shape[1])]
    local = (xyz @ np.linalg.inv(np.asarray(meta["lidar_to_world"])).T)[:, :3]
    column = (local[:, 0]-lower[0])/(upper[0]-lower[0])*width-0.5
    row = (local[:, 1]-lower[1])/(upper[1]-lower[1])*height-0.5
    valid = (np.isfinite(local).all(axis=-1) & (column >= 0) & (column <= width-1)
             & (row >= 0) & (row <= height-1) & (local[:, 2] >= lower[2]) & (local[:, 2] <= upper[2]))
    shape = world_points.shape[:2]
    uv = np.stack([column, row], axis=-1).reshape(*shape, 2).astype(np.float32)
    safe_uv = np.where(valid.reshape(*shape, 1), uv, -1)
    weights = cv2.remap(native_valid.astype(np.float32), safe_uv[..., 0], safe_uv[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sampled = cv2.remap(np.where(native_valid[None], feature, 0).transpose(1, 2, 0),
                        safe_uv[..., 0], safe_uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if sampled.ndim == 2:
        sampled = sampled[..., None]
    in_range = valid.reshape(shape)
    valid = in_range & (weights >= .999) & np.isfinite(sampled).all(axis=-1) & (np.linalg.norm(sampled, axis=-1) > 1e-6)
    sampled = np.where(valid[..., None], sampled/np.maximum(weights[..., None], 1e-6), 0)
    if not return_geometry:
        return sampled, valid, meta
    channels = feature[np.arange(3) % len(feature)].transpose(1, 2, 0)
    preview = np.zeros((*native_valid.shape, 3), np.uint8)
    if native_valid.any():
        lo, hi = np.percentile(channels[native_valid], [5, 95], axis=0)
        preview = (24+231*np.clip(np.nan_to_num((channels-lo)/np.maximum(hi-lo, 1e-6)), 0, 1)).astype(np.uint8)
        preview[~native_valid] = 0
    geometry = {"native_uv": uv, "native_valid": native_valid, "native_preview": preview,
        "in_range": in_range, "interpolation_support": weights, "sample_valid": valid,
        "resolution_xy_m": (upper[:2]-lower[:2])/np.array([width, height]),
        "validity_source": validity_source}
    return sampled, valid, meta, geometry


def attach_cached_bev(arrays, metadata, root):
    features, valid, sources = [], [], []
    for token in metadata["sample_tokens"]:
        feature, support, source = sample_cached_bev(root, token, arrays["receiver_world_xyz"])
        features.append(feature)
        valid.append(support)
        sources.append(source)
    features = np.stack(features)
    valid = np.stack(valid) & arrays["visible_by_frame"]
    norm = np.linalg.norm(features, axis=-1)
    similarity = []
    for index in range(2):
        supported = valid[index] & valid[index+1] & (norm[index] > 1e-6) & (norm[index+1] > 1e-6)
        dot = (features[index]*features[index+1]).sum(axis=-1)
        cosine = dot/np.maximum(norm[index]*norm[index+1], 1e-6)
        similarity.append(np.where(supported, np.clip(cosine, -1, 1), np.nan))
    arrays["learned_bev_receiver_features"] = np.where(valid[..., None], features, 0).astype(np.float16)
    arrays["learned_bev_valid"] = valid
    arrays["learned_bev_temporal_cosine"] = np.stack(similarity).astype(np.float32)
    metadata["learned_bev"] = {"sources": sources, "layout": "T,H,W,C",
        "alignment": "world receiver -> each sample LIDAR_TOP; no resize-only alignment"}
    return arrays, metadata


def select_bev_flow_receivers(arrays, metadata, minimum_cosine=0.3, *,
                              feature_gate="aligned", flow_gate=True, seed=0,
                              discovery_mode="discrepancy", discrepancy_config=None,
                              proposal_entity_motion="on"):
    """Keep physical discovery separate from the explicit hard-intersection arm."""
    if discovery_mode not in {"discrepancy", "cross-space", "hard-intersection"}:
        raise ValueError(f"Unknown discovery mode: {discovery_mode}")
    if proposal_entity_motion not in {"on", "off"}:
        raise ValueError(f"Unknown proposal entity-motion mode: {proposal_entity_motion}")
    if discovery_mode != "hard-intersection" and feature_gate == "matched-random":
        raise ValueError("matched-random is a hard-intersection budget control")
    if feature_gate not in {"aligned", "none", "shuffled", "matched-random"}:
        raise ValueError(f"Unknown feature gate: {feature_gate}")
    if not np.isfinite(minimum_cosine) or not -1 <= minimum_cosine <= 1:
        raise ValueError("BEV cosine threshold must be in [-1,1]")
    original = arrays.get("shadow_candidate_photometric", arrays["shadow_candidate"]).copy()
    observed_flow = arrays.get("shadow_flow_supported_observed", arrays["shadow_flow_supported"]).copy()
    flow = observed_flow if flow_gate else np.ones_like(original)
    rng = np.random.default_rng(seed)
    cosine = None
    metadata.pop("discrepancy_discovery", None)
    for key in list(arrays):
        if key.startswith("discrepancy_") or key == "proposal_source_bits":
            del arrays[key]
    consistency = np.full(original.shape, np.nan, np.float32)
    feature_valid = np.ones_like(original)
    if feature_gate == "none":
        learned = np.ones_like(original)
    else:
        cosine = np.asarray(arrays["learned_bev_temporal_cosine"])
        feature_valid = arrays["learned_bev_valid"].all(axis=0) & np.isfinite(cosine).all(axis=0)
        if feature_gate == "shuffled":
            features = arrays["learned_bev_receiver_features"].astype(np.float32).copy()
            for slot in range(3):
                support = arrays["learned_bev_valid"][slot]
                if "receiver_valid" in arrays:
                    support = support & arrays["receiver_valid"]
                values = features[slot][support]
                features[slot][support] = values[rng.permutation(len(values))]
            norms = np.linalg.norm(features, axis=-1)
            cosine = np.stack([(features[i]*features[i+1]).sum(axis=-1)
                / np.maximum(norms[i]*norms[i+1], 1e-6) for i in range(2)])
            feature_valid &= (norms > 1e-6).all(axis=0)
        consistency = np.min(np.nan_to_num(cosine, nan=-1.0), axis=0)
        learned = feature_valid & (consistency >= minimum_cosine)
        if feature_gate == "matched-random":
            # Match the aligned arm's final candidate budget within the same valid domain.
            count = int((original & flow & learned).sum())
            eligible = np.flatnonzero(original & flow & feature_valid)
            learned = np.zeros_like(original)
            learned.flat[rng.choice(eligible, size=count, replace=False)] = True
    selected = original & learned & flow
    arrays["shadow_candidate_photometric"] = original
    arrays["shadow_flow_supported_observed"] = observed_flow
    arrays["shadow_flow_supported"] = flow
    arrays["learned_bev_receiver_support"] = learned
    arrays["learned_bev_receiver_consistency"] = consistency
    arrays["shadow_candidate"] = selected
    for key in ("shadow_boundary_onset", "shadow_boundary_recovery", "shadow_same_patch_cycle"):
        source = f"{key}_photometric"
        if source not in arrays:
            arrays[source] = arrays[key].copy()
        arrays[key] = arrays[source] & selected
    metadata["learned_bev_selection"] = {"minimum_cosine": minimum_cosine,
        "discovery_mode": discovery_mode,
        "feature_gate": feature_gate, "flow_gate": flow_gate, "seed": int(seed),
        "rule": "photometric AND configured_registration AND configured_feature_gate",
        "photometric_candidate_cells": int(original.sum()), "selected_cells": int(selected.sum()),
        "feature_valid_candidate_cells": int((original & feature_valid).sum()),
        "selection_fraction": float(selected.sum()/max(original.sum(), 1)),
        "feature_rejected_cells": int((original & ~learned).sum()),
        "flow_rejected_cells": int((original & ~flow).sum())}
    if discovery_mode in {"cross-space", "discrepancy"}:
        from shadow_occ.annotation.cross_space import discover_cross_space

        config = metadata.get("discovery_config", {})
        metadata["cross_space_discovery"] = discover_cross_space(arrays, cosine=cosine,
            light_threshold=config.get("light_threshold_L", 6.0),
            darkness_ratio=config.get("darkness_ratio", .88))
        if discovery_mode == "discrepancy":
            from shadow_occ.annotation.discrepancy_proposals import propose_discrepancy

            baseline = arrays["shadow_candidate"].copy()
            # Only the discovery arm loses entity-motion inputs in this control.
            # Keep the original cache intact for downstream physical verification.
            proposal_arrays = arrays if proposal_entity_motion == "on" else {
                key: value for key, value in arrays.items()
                if key not in {"box_surface_expected_flow_px", "mature_expected_object_flow_px"}}
            metadata["discrepancy_discovery"] = propose_discrepancy(proposal_arrays, cosine=cosine,
                baseline=baseline, light_threshold=config.get("light_threshold_L", 6.0), config=discrepancy_config)
            if proposal_arrays is not arrays:
                arrays.update({key: value for key, value in proposal_arrays.items() if key.startswith("discrepancy_")})
            metadata["discrepancy_discovery"]["proposal_entity_motion"] = proposal_entity_motion
            arrays["proposal_source_bits"] = baseline.astype(np.uint8) | (arrays["discrepancy_candidate"].astype(np.uint8) << 1)
            arrays["shadow_candidate"] = baseline | arrays["discrepancy_candidate"]
            arrays["shadow_boundary_onset"] |= arrays["discrepancy_onset"]
            arrays["shadow_boundary_recovery"] |= arrays["discrepancy_recovery"]
            arrays["shadow_same_patch_cycle"] = arrays["shadow_boundary_onset"] & arrays["shadow_boundary_recovery"]
        selected = arrays["shadow_candidate"]
        metadata["learned_bev_selection"].update(
            rule="cross-space appearance discovery; hard intersection recorded as counterfactual",
            hard_intersection_cells=int((original & learned & flow).sum()),
            selected_cells=int(selected.sum()), selection_fraction=float(selected.sum()/max(original.sum(), 1)),
            feature_rejected_cells=0, flow_rejected_cells=0)
        if discovery_mode == "discrepancy":
            metadata["learned_bev_selection"]["rule"] = "appearance baseline UNION active 3D/2D discrepancy proposals"
    else:
        metadata.pop("cross_space_discovery", None)
        for key in list(arrays):
            if key.startswith("cross_space_"):
                del arrays[key]
    metadata["counts"]["photometric_candidates"] = int(original.sum())
    metadata["counts"]["shadow_candidates"] = int(selected.sum())
    metadata["counts"]["discrepancy_proposals"] = int(arrays["discrepancy_candidate"].sum()) if "discrepancy_candidate" in arrays else 0
    metadata["counts"]["discrepancy_novel_proposals"] = int(arrays["discrepancy_novel_candidate"].sum()) if "discrepancy_novel_candidate" in arrays else 0
    if "shadow_persistent_dark" in arrays:
        metadata["counts"]["selected_persistent_dark"] = int((selected & arrays["shadow_persistent_dark"]).sum())
    return arrays, metadata
