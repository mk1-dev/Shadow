"""Camera/native-BEV/world-grid views of one geometry-anchored frame pair."""
from __future__ import annotations

import cv2
import numpy as np


def _heat(values):
    result = cv2.applyColorMap((255*np.clip(np.nan_to_num(values), 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO)
    result[~np.isfinite(values)] = 0
    return result


def _points(image, coordinates, valid, color):
    result = image.copy()
    points = coordinates[valid & np.isfinite(coordinates).all(axis=-1)]
    for index in np.linspace(0, max(len(points)-1, 0), min(len(points), 180), dtype=int):
        x, y = np.rint(points[index]).astype(int)
        if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
            cv2.circle(result, (x, y), 1, color, -1)
    return result


def correspondence_panel(context, maps, slot):
    """Offsets are feature-registration residuals; never shadow/ground speed."""
    receiver = maps["receiver_valid_by_frame"]
    coordinates = context["coordinates"]
    rgb = _points(context["images"][slot], coordinates[slot], receiver[slot], (0, 240, 0))
    features = context["spatial_features"]
    valid = context["spatial_feature_valid"] & np.isfinite(features).all(axis=-1)
    channels = features[..., np.arange(3) % features.shape[-1]]
    registered = np.zeros((*receiver.shape[1:], 3), np.uint8)
    if valid.any():
        lo, hi = np.percentile(channels[valid], [5, 95], axis=0)
        registered = (24+231*np.clip(np.nan_to_num((channels[slot]-lo)/np.maximum(hi-lo, 1e-6)), 0, 1)).astype(np.uint8)
        registered[~valid[slot]] = 0
    native = []
    geometry = context.get("spatial_geometry")
    for at in (slot, slot+1):
        if geometry:
            record = geometry[at]
            native.append(_points(record["native_preview"], record["native_uv"],
                record["sample_valid"] & receiver[at], (0, 240, 0)))
        else:
            native.append(np.zeros_like(registered))
    local = _heat(np.where(maps["bev_flow_matchable_pair"][slot],
                           maps["spatial_bev_match_confidence"][slot], np.nan))
    yy, xx = np.where(maps["spatial_bev_match_valid"][slot])
    for index in np.linspace(0, max(len(xx)-1, 0), min(len(xx), 80), dtype=int):
        y, x = yy[index], xx[index]
        vector = maps["spatial_bev_local_offset_m"][slot, y, x]/context["grid"].resolution
        end = tuple(np.rint(np.array([x, y])+vector).astype(int))
        color = (255, 255, 255) if maps["spatial_bev_match_valid"][slot, y, x] else (0, 220, 255)
        cv2.arrowedLine(local, (x, y), end, color, 1, tipLength=.3)
        if np.linalg.norm(vector) < .1:
            cv2.circle(local, (x, y), 1, color, -1)
    flow = context["images"][slot].copy()
    yy, xx = np.where(maps["flow_confidence_map"][slot] > 0)
    for index in np.linspace(0, max(len(xx)-1, 0), min(len(xx), 80), dtype=int):
        y, x = yy[index], xx[index]
        start = coordinates[slot][y, x]
        for key, color in (("observed_flow", (0, 230, 230)), ("expected_flow", (230, 150, 60))):
            end = start+context[key][slot, y, x]
            if np.isfinite(start).all() and np.isfinite(end).all():
                cv2.arrowedLine(flow, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), color, 1)
    delta = np.where(maps["exposure_valid"][slot], context["delta_L"][slot], np.nan)
    flow_state = np.zeros((*receiver.shape[1:], 3), np.uint8)
    flow_state[maps["m2_flow_state"][slot] == 1] = (0, 160, 240)
    flow_state[maps["m2_flow_state"][slot] == 2] = (70, 200, 70)
    tiles = [(rgb, "World receiver anchors: camera"),
        (native[0], "Native BEV t: channels 0..2 + anchors"),
        (native[1], "Native BEV t+1: geometry anchors"),
        (registered, "World-aligned current-only BEV"),
        (_heat(maps["bev_anchor_consistency"][slot]), "Primary: same-world anchor consistency"),
        (local, "Local offsets + confidence (not motion)"),
        (flow, "Image flow: observed / geometric"),
        (_heat(maps["ground_flow_geometry_residual_px"][slot]/6), "Ground geometry flow residual / 6 px"),
        (flow_state, "Flow: low orange / reliable green"),
        (_heat(.5+delta/24), "Exposure-corrected signed Lab-L"),
        (_heat(maps["proposal_score_map"][slot+1]), "Next-frame discrepancy proposals"),
        (_heat(maps["m2_flow_quality"][slot]), "Ground-flow quality (not probability)")]
    panels = []
    for value, title in tiles:
        tile = np.zeros((256, 384, 3), np.uint8)
        scale = min(384/value.shape[1], 230/value.shape[0])
        resized = cv2.resize(value, (max(1, round(value.shape[1]*scale)), max(1, round(value.shape[0]*scale))))
        tile[26:26+resized.shape[0], :resized.shape[1]] = resized
        cv2.putText(tile, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        panels.append(tile)
    return np.vstack([np.hstack(panels[start:start+3]) for start in range(0, len(panels), 3)])
