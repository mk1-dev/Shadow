"""Visual geometry audits on real cached observations, never pseudo-label decisions."""
from __future__ import annotations

import cv2
import numpy as np

from shadow_occ.annotation.physics_labels import footprint_mask, footprint_polygon, light_bank
from shadow_occ.annotation.physics_motion import project_world, sample_at


BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))


def tile(image, title, size=(480, 300)):
    width, height = size
    panel = np.full((height, width, 3), 24, np.uint8)
    scale = min(width/image.shape[1], (height-28)/image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))))
    x = (width-resized.shape[1])//2
    panel[28:28+resized.shape[0], x:x+resized.shape[1]] = resized
    cv2.putText(panel, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .44, (245, 245, 245), 1)
    return panel


def sheet(panels, columns):
    panels = list(panels)
    panels += [np.zeros_like(panels[0]) for _ in range((-len(panels)) % columns)]
    return np.vstack([np.hstack(panels[index:index+columns]) for index in range(0, len(panels), columns)])


def mask_image(mask):
    return np.repeat((np.asarray(mask, bool)*255).astype(np.uint8)[..., None], 3, axis=-1)


def outline(image, mask, color, width=1):
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, width)
    return image


def signed_image(delta, valid, limit=12):
    values = np.clip(np.nan_to_num(delta)/limit, -1, 1)
    image = np.full((*values.shape, 3), 32, np.uint8)
    image[..., 0] = np.clip(np.maximum(-values, 0)*255, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip(np.maximum(values, 0)*255, 0, 255).astype(np.uint8)
    image[~valid] = 32
    return image


def flow_image(flow):
    valid = np.isfinite(flow).all(axis=-1)
    clean = np.where(valid[..., None], flow, 0).astype(np.float32)
    magnitude, angle = cv2.cartToPolar(clean[..., 0], clean[..., 1])
    scale = float(np.percentile(magnitude[valid], 95)) if valid.any() else 1.
    hsv = np.zeros((*valid.shape, 3), np.uint8)
    hsv[..., 0] = (angle*90/np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude/max(scale, 1)*255, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image[~valid] = 32
    return image


def boxes_and_tracks(image, tracks, projection):
    result = image.copy()
    for track in tracks:
        points = np.asarray(track["points"])
        uv, _, valid = project_world(points[1, 1:], projection)
        pixels = np.rint(np.clip(np.nan_to_num(uv), -10000, 10000)).astype(int)
        for start, end in BOX_EDGES:
            if valid[start] and valid[end]:
                cv2.line(result, tuple(pixels[start]), tuple(pixels[end]), (80, 230, 80), 1)
        centers, _, seen = project_world(points[:, 0], projection)
        for start in range(len(centers)-1):
            if seen[start:start+2].all():
                a, b = np.rint(np.clip(centers[start:start+2], -10000, 10000)).astype(int)
                cv2.arrowedLine(result, tuple(a), tuple(b), (0, 180, 255), 1, tipLength=.2)
    return result


def receiver_visibility_review(images, grounds, context, world, receiver):
    panels = []
    for index, (frame, image, ground) in enumerate(zip(context["frames"], images, grounds)):
        uv, _, good = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
        good &= (uv >= 0).all(axis=1) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] < image.shape[0])
        coordinates = np.where(good[:, None], uv, -1).reshape(*receiver.shape, 2).astype(np.float32)
        supported = sample_at(ground.astype(np.float32), coordinates).ravel() >= .999
        view = image.copy()
        view[ground] = (.8*view[ground]+.2*np.array([60, 220, 60])).astype(np.uint8)
        indices = np.flatnonzero(good)
        for pixel in indices[::max(1, len(indices)//1600)]:
            color = (80, 240, 80) if supported[pixel] and receiver.ravel()[pixel] else (
                (255, 255, 0) if supported[pixel] else (60, 60, 240))
            cv2.circle(view, tuple(np.rint(uv[pixel]).astype(int)), 1, color, -1)
        panels.append(tile(view, f"frame {index}: in-image={int(good.sum())} visible={int(context['visible'][index].sum())}"))
    return sheet(panels, 3)


def light_projection_review(image, observed, tracks, plane, projection, focus_token=None):
    """Show every direction, including failed projections, rather than an accepted best."""
    directions = light_bank()
    heat = np.zeros(observed.shape, np.float32)
    records, focus_masks = [], {}
    for track in tracks:
        for light_id, direction in enumerate(directions):
            record = {"instance_token": track["instance_token"], "light_id": light_id,
                "direction_world": direction.tolist()}
            try:
                polygon = footprint_polygon(track["points"][1, 1:], plane, direction, projection)
                mask = np.zeros(observed.shape, np.uint8)
                cv2.fillConvexPoly(mask, polygon, 1)
                overlap = int((observed & (mask > 0)).sum())
                record.update(status="projected", polygon_image=polygon.tolist(), overlap_pixels=overlap,
                    cover=overlap/max(int(observed.sum()), 1))
                heat += mask
                if track["instance_token"] == focus_token:
                    focus_masks[light_id] = polygon
            except ValueError as exc:
                record.update(status="invalid_projection", reason=str(exc))
            records.append(record)
    overlay = boxes_and_tracks(image, tracks, projection)
    if heat.any():
        colors = cv2.applyColorMap(np.clip(heat/heat.max()*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        overlay[heat > 0] = (.75*overlay[heat > 0]+.25*colors[heat > 0]).astype(np.uint8)
    outline(overlay, observed, (255, 255, 0), 2)
    focus_records = {record["light_id"]: record for record in records if record["instance_token"] == focus_token}
    yy, xx = np.where(observed)
    points = np.column_stack([xx, yy]).astype(np.float64)
    for track in tracks:
        if track["instance_token"] == focus_token:
            projected, _, valid = project_world(np.asarray(track["points"]).reshape(-1, 3), projection)
            valid &= (projected >= 0).all(axis=1) & (projected[:, 0] < image.shape[1]) & (projected[:, 1] < image.shape[0])
            points = np.vstack([points, projected[valid]])
    if len(points):
        center = (points.min(axis=0)+points.max(axis=0))/2
        extent = np.maximum(2*np.ptp(points, axis=0), [256, 160])
        lower = np.maximum(0, np.floor(center-extent/2)).astype(int)
        upper = np.minimum(image.shape[1::-1], np.ceil(center+extent/2)).astype(int)
        x0, y0 = lower
        x1, y1 = upper
    else:
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
    for record in focus_records.values():
        record["atlas_crop_xyxy"] = [int(x0), int(y0), int(x1), int(y1)]
    panels = []
    for light_id, direction in enumerate(directions):
        view = image.copy()
        if light_id in focus_masks:
            cv2.polylines(view, [focus_masks[light_id]], True, (0, 180, 255), 1)
        outline(view, observed, (255, 255, 0), 2)
        record = focus_records.get(light_id, {})
        status = f"cov={record['cover']:.2f}" if "cover" in record else "unprojectable/no GT"
        azimuth = np.degrees(np.arctan2(direction[1], direction[0])) % 360
        elevation = np.degrees(np.arcsin(-direction[2]))
        panels.append(tile(view[y0:y1, x0:x1], f"{light_id:02} a={azimuth:.0f} e={elevation:.0f} {status}", (320, 200)))
    return overlay, sheet(panels, 8), records


def transition_projection_review(context, token, plane, projection, observed):
    """Audit-only maximum overlap over all directions; never fed back into scores."""
    if token not in context["tracks"]:
        return None, {"status": "no_complete_track", "instance_token": token}
    rows = []
    bank = light_bank()
    points = context["tracks"][token]
    for light_id, direction in enumerate(bank):
        try:
            masks = [footprint_mask(state[1:], plane, direction, projection, observed.shape) for state in points]
        except ValueError:
            continue
        events = context["onset"] | context["recovery"]
        current = masks[context["middle"]]
        matched = sum(int((event & observed & current & ~remote).sum()) for event, remote in zip(events, masks))
        rows.append((matched, light_id, masks))
    if not rows:
        return None, {"status": "no_projectable_track", "instance_token": token}
    matched, light_id, masks = max(rows, key=lambda row: (row[0], -row[1]))
    panels, details = [], []
    current = masks[context["middle"]]
    for index, remote in enumerate(masks):
        event = (context["onset"][index] | context["recovery"][index]) & observed
        predicted = current & ~remote
        view = np.full((*observed.shape, 3), 24, np.uint8)
        view[predicted] = (0, 130, 200)
        view[event] = (255, 255, 0)
        view[predicted & event] = (80, 240, 80)
        outline(view, observed, (255, 255, 255))
        details.append({"frame_index": index, "observed_pixels": int(event.sum()),
            "predicted_change_pixels": int((predicted & observed).sum()), "matched_pixels": int((predicted & event).sum())})
        panels.append(tile(view, f"frame {index} light {light_id}: obs={int(event.sum())} match={details[-1]['matched_pixels']}", (360, 260)))
    return sheet(panels, 3), {"status": "rendered", "instance_token": token, "light_id": light_id,
        "matched_pixels": matched, "frames": details,
        "selection": "maximum event overlap across all directions for audit only, not an estimated light"}


def exposure_review(context, index, component, roi):
    values = context["lightness"]
    middle = context["middle"]
    visible = context["visible"][index] & context["visible"][middle]
    raw = values[middle]-values[index]
    fit = context["exposures"][index]
    normalized = values[middle]-(fit["scale"]*values[index]+fit["offset_L"])
    background = context["flow_support"][index] & ~roi

    def quantiles(array, mask):
        selected = array[mask & np.isfinite(array)]
        return np.percentile(selected, [5, 25, 50, 75, 95]).tolist() if selected.size else None

    panels = [tile(signed_image(raw, visible), "Raw L(ref)-L(other): blue=darkening"),
        tile(signed_image(normalized, visible) if fit["valid"] else mask_image(np.zeros_like(component)),
             "Exposure-normalized residual" if fit["valid"] else "INVALID exposure: no normalized event"),
        tile(outline(mask_image(background), component, (255, 255, 0)), "Exposure background; cyan=component")]
    details = {"frame_index": index, "exposure": fit,
        "background_raw_L_quantiles": quantiles(values[index], background),
        "background_reference_L_quantiles": quantiles(values[middle], background),
        "background_fit_residual_L_quantiles": quantiles(normalized, background) if fit["valid"] else None,
        "component_raw_delta_L_quantiles": quantiles(raw, component & visible),
        "component_normalized_delta_L_quantiles": quantiles(normalized, component & visible) if fit["valid"] else None,
        "component_background_overlap_pixels": int((component & background).sum()),
        "background_pixel_coordinates_yx": np.column_stack(np.where(background)).tolist(),
        "quantile_levels": [5, 25, 50, 75, 95]}
    return panels, details
