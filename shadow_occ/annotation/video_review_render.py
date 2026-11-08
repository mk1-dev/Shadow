"""Visualization of saved masks and selected geometry, never a label teacher."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from shadow_occ.annotation.data import checked_write_image, resolve_path
from shadow_occ.annotation.physics_labels import footprint_mask
from shadow_occ.annotation.physics_motion import plane_basis, project_world
from shadow_occ.annotation.video_review import load_npz, owner_tokens


def tile(image, title, size=(420, 280), nearest=False):
    width, height = size
    output = np.zeros((height, width, 3), np.uint8)
    scale = min(width/image.shape[1], (height-30)/image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))),
                         interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    output[30:30+resized.shape[0], :resized.shape[1]] = resized
    cv2.putText(output, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    return output


def sheet(tiles, columns=3):
    tiles = list(tiles)
    tiles.extend(np.zeros_like(tiles[0]) for _ in range((-len(tiles)) % columns))
    return np.vstack([np.hstack(tiles[at:at+columns]) for at in range(0, len(tiles), columns)])


def read_rgb(frame, base):
    path = resolve_path(frame["image"], base)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    if tuple(image.shape[1::-1]) != tuple(frame["image_size_wh"]):
        raise ValueError(f"Raw RGB size disagrees with calibrated camera: {path}")
    return image


def project_mask(mask, arrays, frame, plane):
    basis = plane_basis(plane)
    matrix = np.asarray(frame["projection_world_to_image"]) @ basis @ np.linalg.inv(arrays["bev_world_to_grid"] @ basis)
    front = project_world(arrays["receiver_world_xyz"].reshape(-1, 3), frame["projection_world_to_image"])[2].reshape(mask.shape)
    # Only an overlay of frozen grid cells, not a new image segmentation.
    return cv2.warpPerspective((mask & front).astype(np.uint8), matrix,
        tuple(frame["image_size_wh"]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0


def overlay(image, mask, color):
    if mask.shape != image.shape[:2]:
        raise ValueError("Frozen image mask and clean RGB have different coordinates")
    result = image.copy()
    result[mask] = (.55*result[mask]+.45*np.asarray(color)).astype(np.uint8)
    return result


def heat(value):
    result = cv2.applyColorMap((255*np.clip(np.nan_to_num(value), 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO)
    result[~np.isfinite(value)] = 0
    return result


def selected_projection(clip, slot, regions):
    arrays = clip["arrays"][slot]
    result = np.zeros(arrays["shadow_target_bev"].shape, bool)
    records, corners = [], []
    path = clip["folder"]/f"{slot:03d}_light_bank.npz"
    directions = load_npz(path).get("directions_world") if path.is_file() else None
    seen = set()
    for region in regions:
        physics = region.get("physics", {})
        for role in ("selected", "association"):
            hypothesis = physics.get(role)
            if not hypothesis:
                continue
            token, light = hypothesis["instance_token"], hypothesis["light_id"]
            if (token, light) in seen:
                continue
            seen.add((token, light))
            state = next((row for row in clip["casters"].get(token, {}).get("observations", []) if row["slot"] == slot), None)
            record = {"caster_token": token, "light_id": light, "role": role,
                "geometry_reprojected_for_review_only": True, "light_directions_file": str(path)}
            if state is None or directions is None:
                record["status"] = "missing_saved_caster_or_light_geometry"
            elif not 0 <= light < len(directions):
                record["status"] = "invalid_saved_light_id"
            else:
                try:
                    result |= footprint_mask(state["corners_world"], clip["protocol"]["ground_plane_world"],
                        directions[light], arrays["bev_world_to_grid"], result.shape)
                    corners.append(np.asarray(state["corners_world"]))
                    record["status"] = "rendered_from_saved_selected_hypothesis"
                except ValueError:
                    record["status"] = "invalid_saved_projection_geometry"
            records.append(record)
    return result, corners, records


def render_case(folder, before, after, slot, domain, changes, evidence, camera_frames, radius=1):
    frame, base = camera_frames[slot]
    rgb = read_rgb(frame, base)
    arrays = after["arrays"][slot]
    plane = after["protocol"]["ground_plane_world"]
    projected = project_mask(domain, arrays, frame, plane)
    projection_image = rgb.copy()
    projection_records, visible_corners = {}, []
    for label, clip, color in (("before", before, (230, 150, 30)), ("after", after, (0, 160, 240))):
        footprint, corners, records = selected_projection(clip, slot, evidence[label]["regions"])
        projection_image = overlay(projection_image, project_mask(footprint, clip["arrays"][slot], frame,
            clip["protocol"]["ground_plane_world"]), color)
        for points in corners:
            uv, _, front = project_world(points, frame["projection_world_to_image"])
            inside = front & np.isfinite(uv).all(axis=-1) & (uv >= 0).all(axis=-1)
            inside &= (uv[:, 0] < rgb.shape[1]) & (uv[:, 1] < rgb.shape[0])
            visible_corners.extend(uv[inside].tolist())
            for point in uv[inside]:
                cv2.circle(projection_image, tuple(np.rint(point).astype(int)), 2, color, -1)
        projection_records[label] = records
    yy, xx = np.where(projected)
    points = np.c_[xx, yy].tolist()+visible_corners
    if points:
        points = np.asarray(points)
        lo, hi = points.min(axis=0), points.max(axis=0)+1
        margin = np.maximum((hi-lo)*.5, 32)
        x0, y0 = np.maximum(np.floor(lo-margin), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(hi+margin), rgb.shape[1::-1]).astype(int)
    else:
        x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]
    crop = np.s_[y0:y1, x0:x1]
    checked_write_image(folder/"clean_crop.jpg", rgb[crop])
    checked_write_image(folder/"clean_context.jpg", rgb)
    before_image = overlay(rgb, before["images"][slot]["shadow_target_image"] == 1, (40, 230, 70))
    after_image = overlay(rgb, after["images"][slot]["shadow_target_image"] == 1, (40, 230, 70))
    delta = np.zeros((*domain.shape, 3), np.uint8)
    delta[domain] = (55, 55, 55)
    owner_delta = delta.copy()
    for name, color in (("owner_added", (70, 220, 70)), ("owner_lost", (200, 80, 80)), ("owner_switched", (0, 180, 240))):
        owner_delta[changes[name] & domain] = color
    delta = owner_delta.copy()
    for name, color in (("added_positive", (220, 220, 0)), ("removed_positive", (30, 30, 230))):
        delta[changes[name] & domain] = color
    gy, gx = np.where(domain)
    grid_crop = np.s_[max(0, gy.min()-2):gy.max()+3, max(0, gx.min()-2):gx.max()+3]
    missing = np.full(domain.shape, np.nan, np.float32)
    provenance = np.zeros_like(delta)
    for name, color in (("cross_space_direct_seed", (240, 240, 240)),
                        ("cross_space_temporal_seed", (0, 220, 220)), ("cross_space_spatial_growth", (220, 90, 180))):
        if name in arrays:
            provenance[arrays[name].astype(bool) & domain] = color
    tiles = [tile(rgb[crop], "Clean RGB; no annotation"), tile(before_image[crop], "Before: saved weak image target"),
        tile(after_image[crop], "After: saved weak image target"),
        tile(delta[grid_crop], "Grid: add cyan / remove red / owner color", nearest=True),
        tile(heat(arrays.get("appearance_score_map", missing))[grid_crop], "After: observed appearance score", nearest=True),
        tile(heat(arrays.get("bev_anchor_consistency_by_frame", missing))[grid_crop], "After: same-world anchor support", nearest=True),
        tile(provenance[grid_crop], "Seeds: direct white / temporal yellow", nearest=True),
        tile(heat(arrays.get("cross_space_region_appearance_support", missing))[grid_crop], "After: seed-color appearance support", nearest=True),
        tile(projection_image[crop], "Saved M3 geometry: before blue / after orange")]
    checked_write_image(folder/"review.jpg", sheet(tiles))
    owner_tiles = []
    for label, clip in (("before", before), ("after", after)):
        tokens = owner_tokens(clip["arrays"][slot], clip["protocol"])
        canvas = np.zeros_like(delta)
        for token in sorted(set(tokens[domain])-{ "" }):
            color = 70+np.frombuffer(hashlib.sha256(token.encode()).digest()[:3], dtype=np.uint8).astype(int)%186
            canvas[(tokens == token) & domain] = color
        owner_tiles.append(tile(canvas[grid_crop], label+": caster-token colors (not owner indices)", nearest=True))
    owner_tiles.append(tile(owner_delta[grid_crop], "Owner: add green / loss blue / switch orange", nearest=True))
    checked_write_image(folder/"owner_changes.jpg", sheet(owner_tiles))
    context_tiles = []
    selected_slots = list(range(max(0, slot-radius), min(len(camera_frames), slot+radius+1)))
    context_images = {slot: rgb}
    for at in selected_slots:
        if at not in context_images:
            context_images[at] = read_rgb(*camera_frames[at])
    for label in ("clean", "before", "after"):
        for at in selected_slots:
            neighbour = camera_frames[at][0]
            image = context_images[at]
            if label != "clean":
                clip = before if label == "before" else after
                image = overlay(image, clip["images"][at]["shadow_target_image"] == 1, (40, 230, 70))
            context_tiles.append(tile(image[crop], f"{label}: frame {at}, t={neighbour['camera_timestamp']}"))
    checked_write_image(folder/"temporal_context.jpg", sheet(context_tiles, len(selected_slots)))
    return {"rgb_crop_box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "projected_region_pixels": int(projected.sum()), "context_frame_slots": selected_slots,
        "selected_projection": projection_records, "projection_is_review_overlay_not_shadow_GT": True}


def render_continuity(path, clip, camera_frames, timeline):
    tiles = []
    for slot, (frame, base) in enumerate(camera_frames):
        image = read_rgb(frame, base)
        saved = clip["images"][slot]
        image = overlay(image, saved["shadow_proposal_image"].astype(bool), (0, 190, 220))
        image = overlay(image, saved["shadow_target_image"] == 1, (50, 220, 50))
        rows = [row for row in timeline if row["frame_slot"] == slot]
        observed = sum(row["observed_image_mask"] for row in rows)
        memory_only = sum(row["observation_state"] == "retained_memory_only" for row in rows)
        tiles.append(tile(image, f"Frame {slot}: observed {observed}; memory-only {memory_only}"))
    checked_write_image(path, sheet(tiles))
