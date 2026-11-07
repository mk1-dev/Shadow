"""Geographic sunlight priors and weak direction evidence; rays travel DOWNWARD."""
from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np


PROTOCOL = "geo-temporal-light-v1"
# Official nuscenes-devkit map_api.py: SW origins, WGS84 Web Mercator maps.
MAP_ORIGINS = {
    "boston-seaport": (42.336849169438615, -71.05785369873047),
    "singapore-onenorth": (1.2882100868743724, 103.78475189208984),
    "singapore-hollandvillage": (1.2993652317780957, 103.78217697143555),
    "singapore-queenstown": (1.2782562240223188, 103.76741409301758),
}


def solar_ray(azimuth_deg, elevation_deg):
    """pvlib azimuth east of north -> light propagation in world east/north/up."""
    azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
    return -np.array([np.cos(elevation)*np.sin(azimuth),
                      np.cos(elevation)*np.cos(azimuth), np.sin(elevation)])


def geographic_position(location, world_xy):
    from pyproj import Transformer

    latitude, longitude = MAP_ORIGINS[location]
    forward = Transformer.from_crs(4326, 3857, always_xy=True)
    backward = Transformer.from_crs(3857, 4326, always_xy=True)
    x, y = forward.transform(longitude, latitude)
    lon, lat = backward.transform(x+float(world_xy[0]), y+float(world_xy[1]))
    return float(lat), float(lon)


def sunlight_context(location, world_xy, timestamps):
    if location not in MAP_ORIGINS:
        return {"available": False, "reason": "unknown_map_location", "location": location}
    import pandas as pd
    from pvlib.solarposition import get_solarposition

    latitude, longitude = geographic_position(location, world_xy)
    times = pd.to_datetime(np.asarray(timestamps, np.int64), unit="us", utc=True)
    position = get_solarposition(times, latitude, longitude, method="nrel_numpy")
    elevation = position["elevation"].to_numpy()
    azimuth = position["azimuth"].to_numpy()
    return {"available": True, "location": location, "latitude": latitude, "longitude": longitude,
        "timestamp_utc": [datetime.fromtimestamp(int(value)/1e6, timezone.utc).isoformat() for value in timestamps],
        "elevation_deg": elevation.tolist(), "azimuth_deg": azimuth.tolist(),
        "rays_world": [solar_ray(a, e).tolist() for a, e in zip(azimuth, elevation)],
        "direct_sun_possible": bool(elevation[1] > 3),
        "scope": "astronomical prior, not measured illumination; global x=east,y=north,z=up"}


def directional_distribution(directions, ray, width_degrees=25):
    directions, ray = np.asarray(directions), np.asarray(ray)
    logits = (directions @ ray-1)/max(1-np.cos(np.deg2rad(width_degrees)), 1e-6)
    probability = np.exp(logits-logits.max())
    return probability/probability.sum()


def directional_confidence(probability):
    p = np.asarray(probability, np.float64)
    return float(np.clip(1+np.sum(p*np.log(np.maximum(p, 1e-12)))/np.log(len(p)), 0, 1))


def weak_light_target(directions, components, sun):
    """Pool all physical hypotheses, not the previous teacher's accepted YES labels."""
    tracks = {}
    for component in components:
        projection = {(item["instance_token"], item["light_id"]): item["score"] for item in
                      (component.get("receiver_competition") or {}).get("hypothesis_scores", [])}
        for parent in component["hypotheses"]:
            for item in [parent, *parent.get("alternatives", [])]:
                key = item["instance_token"]
                values = tracks.setdefault(key, np.full(len(directions), np.nan))
                light = item["light_id"]
                score = (.5*projection.get((key, light), item.get("cover", 0))
                    +.3*item.get("available_edge_fit", 0)+.2*max(0, item.get("available_temporal_gain", 0)))
                values[light] = max(score, values[light]) if np.isfinite(values[light]) else score
    geometry = np.zeros(len(directions))
    if tracks:
        scores = np.stack(list(tracks.values()))
        for light in range(len(directions)):
            values = np.sort(scores[:, light][np.isfinite(scores[:, light])])
            geometry[light] = float(values[-2:].mean()) if len(values) else 0
    p_geometry = np.exp((geometry-geometry.max())/.15)
    p_geometry /= p_geometry.sum()
    quality = directional_confidence(p_geometry)*min(len(tracks)/2, 1)
    solar_valid = bool(sun.get("available") and sun.get("direct_sun_possible"))
    p_solar = directional_distribution(directions, sun["rays_world"][1]) if solar_valid else np.full(len(directions), 1/len(directions))
    solar_weight, geometry_weight = .25*solar_valid, 2*quality
    total = solar_weight+geometry_weight
    target = (solar_weight*p_solar+geometry_weight*p_geometry)/total if total else p_solar
    return {"probability": target.astype(np.float32), "weight": float(total),
        "confidence": float(np.clip(.2*solar_valid+.8*quality, 0, 1)),
        "physical_tracks": len(tracks), "geometry_quality": quality,
        "solar_probability": p_solar.astype(np.float32), "source": "astronomy_and_physical_weak_not_light_gt"}


def direction_inputs(directions, frames, sun, exposure_valid, intervals):
    rotations = [cv2.decomposeProjectionMatrix(np.asarray(frame["projection_world_to_image"], np.float64))[1]
                 for frame in frames]
    available = bool(sun.get("available"))
    daylight = available and bool(sun["direct_sun_possible"])
    ray = np.asarray(sun["rays_world"][1]) if daylight else np.zeros(3)
    rays = np.concatenate([np.asarray(directions) @ rotation.T for rotation in rotations], axis=1)
    features = np.c_[rays, np.asarray(directions) @ ray,
                     np.full(len(directions), available), np.full(len(directions), daylight)].astype(np.float32)
    metadata = np.r_[rotations[1] @ ray, available,
        sun["elevation_deg"][1]/90 if available else 0, intervals, exposure_valid].astype(np.float32)
    return features, metadata


def rotate_between(source, target):
    source, target = np.asarray(source), np.asarray(target)
    axis = np.cross(source, target)
    sine, cosine = np.linalg.norm(axis), float(np.clip(source @ target, -1, 1))
    if sine < 1e-8:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(source, [1, 0, 0] if abs(source[0]) < .9 else [0, 1, 0])
        return cv2.Rodrigues(axis/np.linalg.norm(axis)*np.pi)[0]
    return cv2.Rodrigues(axis/sine*np.arctan2(sine, cosine))[0]


def prior_record(window, probability, confidence, sun, directions, model_source):
    probability = np.asarray(probability, np.float64)
    if (probability.shape != (len(directions),) or not np.isfinite(probability).all()
            or (probability < 0).any() or probability.sum() <= 0 or not 0 <= confidence <= 1):
        raise ValueError("Invalid learned light distribution or confidence")
    probability /= probability.sum()
    per_frame = np.repeat(np.asarray(directions)[None], 3, axis=0)
    if sun.get("available") and sun.get("direct_sun_possible"):
        reference = np.asarray(sun["rays_world"][1])
        per_frame = np.stack([np.asarray(directions) @ rotate_between(reference, target).T
                              for target in sun["rays_world"]])
    return {"protocol": PROTOCOL, "window_id": window["id"], "scene_name": window["scene_name"],
        "sample_tokens": window["sample_tokens"], "probability": probability.tolist(),
        "confidence": float(confidence*directional_confidence(probability)),
        "directions_world": np.asarray(directions).tolist(), "directions_by_frame_world": per_frame.tolist(),
        "sun": sun, "model_source": model_source,
        "scope": "soft light ranking and ephemeris drift; not shadow truth or exact boundary velocity"}


def light_prior_bonus(prior, light_id, weight):
    if prior is None:
        return 0.0
    p = np.asarray(prior["probability"], np.float64)
    return float(weight*prior["confidence"]*np.tanh(np.log(max(p[light_id]*len(p), 1e-8))))


def validate_prior(prior, directions):
    p = np.asarray(prior["probability"], np.float64)
    rays = np.asarray(prior["directions_by_frame_world"], np.float64)
    if (p.shape != (len(directions),) or rays.shape != (3, len(directions), 3)
            or not np.isfinite(p).all() or not np.isfinite(rays).all() or (p < 0).any()
            or not np.isclose(p.sum(), 1) or not 0 <= prior["confidence"] <= 1
            or (rays[..., 2] >= 0).any()
            or not np.allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-5)
            or not np.allclose(rays[1], directions, atol=1e-5)):
        raise ValueError("Invalid light prior probabilities, coordinate frame or ray bank")
