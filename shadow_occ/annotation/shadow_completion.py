"""Independent semantic proposals and image-space, physics-grounded completion."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from shadow_occ.annotation.data import digest, read_rows, resolve_path
from shadow_occ.annotation.physics_labels import footprint_mask


@dataclass(frozen=True)
class CompletionConfig:
    prior_seed: float = .5
    prior_grow: float = .25
    cross_seed: float = .35
    appearance_minimum: float = .05
    pairwise_weight: float = .8
    color_sigma: float = 15.
    expansion_weight: float = .25


class ShadowPriorCache:
    """Camera-token identity and content hashes, never slot or filename matching."""
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.rows = {}
        for row in read_rows(self.path):
            token = row["frame_token"]
            if token in self.rows:
                raise ValueError("Duplicate shadow-prior camera token")
            self.rows[token] = row
        protocol = self.path.parent/"protocol.json"
        self.provenance = {"manifest": str(self.path), "manifest_sha256": digest(self.path),
            "protocol": json.loads(protocol.read_text(encoding="utf-8")),
            "protocol_sha256": digest(protocol)}

    def load(self, frame, scene, camera, image_path):
        row = self.rows[frame["sample_data_token"]]
        if (row["scene_name"], row["camera"], row["timestamp"]) != (scene, camera, frame["camera_timestamp"]):
            raise ValueError("Shadow prior differs from camera identity/time")
        if digest(image_path) != row["image_sha256"]:
            raise ValueError("Shadow prior RGB changed")
        path = resolve_path(row["response"], self.path.parent)
        if digest(path) != row["response_sha256"]:
            raise ValueError("Shadow prior response changed")
        with np.load(path, allow_pickle=False) as archive:
            response = np.asarray(archive["binary_response"], np.float32)
        shape = tuple(frame["image_size_wh"][::-1])
        if response.shape != shape or not np.isfinite(response).all() or (response < 0).any():
            raise ValueError("Shadow prior needs finite nonnegative calibrated-image responses")
        return np.clip(response, 0, 1)


def fuse_prior_proposals(maps, responses, coordinates, config=CompletionConfig()):
    """Union only: neither the prior nor a missing flow can veto cross-space seeds."""
    from shadow_occ.annotation.physics_motion import sample_at
    sampled = np.stack([sample_at(response, uv) for response, uv in zip(responses, coordinates)])
    sampled = np.where(maps["receiver_valid_by_frame"], np.nan_to_num(sampled), 0).astype(np.float32)
    prior_seed = sampled >= config.prior_seed
    before = maps["proposal_score_map"].copy()
    maps["shadow_prior_response"] = sampled
    maps["shadow_prior_only_seed"] = prior_seed & (before < config.cross_seed)
    maps["proposal_score_map"] = np.maximum(before, np.where(prior_seed, sampled, 0))
    maps["proposal_source_bits"][prior_seed] |= 16
    return maps


def plausible_projection_envelope(decision, states, slot, plane, directions, projection, shape):
    """Union of near-feasible *rendered hypotheses*, not dilation of a mask.

    The existing M3 decision supplies the feasible light/caster hypotheses. This
    expands shape support without changing its acceptance or owner thresholds.
    """
    envelope = np.zeros(shape, bool)
    if not decision.get("shadow_accepted"):
        return envelope
    hypotheses = [decision["selected"], *decision.get("plausible_hypotheses", [])]
    for item in hypotheses:
        state = states(item["instance_token"]).get(slot)
        if state is None:
            continue
        try:
            envelope |= footprint_mask(state[1:], plane, directions[item["light_id"]], projection, shape)
        except ValueError:
            continue
    return envelope


def native_illumination_evidence(images, receivers, frames, basis, exposures, slot):
    """Observed darkening/recovery on native pixels, not upsampled BEV labels."""
    from shadow_occ.annotation.physics_motion import ground_homography
    image = images[slot]
    shape = image.shape[:2]
    current = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    current[..., 0] *= 100/255
    score = np.zeros(shape, np.float32)
    support = np.zeros(shape, bool)
    for other in (slot-1, slot+1):
        interval = min(slot, other)
        if other < 0 or other >= len(images) or interval >= len(exposures) or not exposures[interval]["valid"]:
            continue
        fit = exposures[interval]
        homography = ground_homography(frames[other]["projection_world_to_image"],
            frames[slot]["projection_world_to_image"], basis)
        light = cv2.cvtColor(images[other], cv2.COLOR_BGR2LAB).astype(np.float32)
        light[..., 0] *= 100/255
        aligned = cv2.warpPerspective(light, homography, shape[::-1], flags=cv2.INTER_LINEAR)
        visible = cv2.warpPerspective(receivers[other].astype(np.float32), homography,
            shape[::-1], flags=cv2.INTER_LINEAR) >= .999
        visible &= receivers[slot]
        aligned_L = (fit["scale"]*aligned[..., 0]+fit["offset_L"] if other > slot else
                     (aligned[..., 0]-fit["offset_L"])/fit["scale"])
        chroma = np.exp(-np.linalg.norm(current[..., 1:]-aligned[..., 1:], axis=-1)/12.)
        # High-pass log luminance retains material texture under multiplicative shading.
        log_current, log_aligned = np.log(np.maximum(current[..., 0], 1)), np.log(np.maximum(aligned_L, 1))
        detail_current = log_current-cv2.GaussianBlur(log_current, (0, 0), 2)
        detail_aligned = log_aligned-cv2.GaussianBlur(log_aligned, (0, 0), 2)
        texture = np.exp(-np.abs(detail_current-detail_aligned)/.15)
        measured = np.clip((aligned_L-current[..., 0])/6., 0, 1)*chroma*texture
        score = np.maximum(score, np.where(visible, measured, 0))
        support |= visible
    return score, support


def complete_visible_shadow(image, core, receiver, prior, cross, appearance, envelope, *,
                            negative=None, config=CompletionConfig()):
    """Graph-cut candidate shape; only independently supported pixels get labels.

    PyMaxflow minimizes a binary contrast-sensitive Potts energy. Semantic-only
    regions remain candidates, not supervision. Occluded ground is never a node
    that can acquire a positive training label.
    """
    shape = image.shape[:2]
    arrays = [np.asarray(value) for value in (core, receiver, prior, cross, appearance, envelope)]
    if any(value.shape != shape for value in arrays):
        raise ValueError("Completion maps must share calibrated RGB coordinates")
    core, receiver, prior, cross, appearance, envelope = arrays
    core, receiver, envelope = core.astype(bool), receiver.astype(bool), envelope.astype(bool)
    negative = np.zeros(shape, bool) if negative is None else np.asarray(negative, bool)
    if negative.shape != shape or (core & negative).any():
        raise ValueError("Completion core and negative seeds conflict")
    prior, cross, appearance = [np.clip(np.nan_to_num(value), 0, 1).astype(np.float32)
                                for value in (prior, cross, appearance)]
    core = core & receiver
    seeds = (core | (prior >= config.prior_seed) | (cross >= config.cross_seed)) & receiver & ~negative
    target = np.full(shape, 255, np.uint8)
    weight = np.zeros(shape, np.float32)
    tier = np.zeros(shape, np.uint8)  # 0=unknown, 1=core, 2=expansion, 3=negative
    completed = np.zeros(shape, bool)
    if seeds.any():
        import maxflow
        yy, xx = np.where(seeds | (receiver & envelope))
        x0, x1 = max(0, int(xx.min())-2), min(shape[1], int(xx.max())+3)
        y0, y1 = max(0, int(yy.min())-2), min(shape[0], int(yy.max())+3)
        section = np.s_[y0:y1, x0:x1]
        domain = receiver[section] & ~negative[section]
        # Fixed seeds carry spatial identity; the prior supplies shape rather than
        # multiplying away an independent cross-space illumination observation.
        unary = np.maximum(np.maximum(prior[section], cross[section]), .55*appearance[section])
        unary = np.clip(unary, .01, .99)
        background_cost, foreground_cost = -np.log1p(-unary), -np.log(unary)
        lab = cv2.cvtColor(image[section], cv2.COLOR_BGR2LAB).astype(np.float32)
        height, width = domain.shape
        graph = maxflow.Graph[float]()
        nodes = graph.add_grid_nodes((height, width))
        # Source side is foreground. Hard costs exceed the incident edge sum.
        hard = 100. + 8*config.pairwise_weight
        graph.add_grid_tedges(nodes, np.where(domain, background_cost, 0),
            np.where(core[section], 0, np.where(domain, foreground_cost, hard)))
        graph.add_grid_tedges(nodes, core[section].astype(np.float64)*hard, np.zeros(domain.shape))
        for axis in (0, 1):
            a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            contrast = np.square(lab[a]-lab[b]).sum(axis=-1)
            edges = np.zeros(domain.shape, np.float64)
            edges[a] = config.pairwise_weight*np.exp(-contrast/(2*config.color_sigma**2))*(domain[a] & domain[b])
            structure = np.zeros((3, 3), np.int8)
            structure[2, 1] = 1 if axis == 0 else 0
            structure[1, 2] = 1 if axis == 1 else 0
            graph.add_grid_edges(nodes, weights=edges, structure=structure, symmetric=True)
        graph.maxflow()
        cut = ~graph.get_grid_segments(nodes) & domain
        count, labels = cv2.connectedComponents(cut.astype(np.uint8), connectivity=8)
        anchored = np.unique(labels[seeds[section]])
        anchored = anchored[anchored != 0]
        completed[section] = np.isin(labels, anchored) if count > 1 else False
    completed |= core
    _, components = cv2.connectedComponents(completed.astype(np.uint8), connectivity=8)
    core_ids = np.unique(components[core])
    core_connected = np.isin(components, core_ids[core_ids != 0])
    # No exact per-pixel box intersection: near-feasible projections bound the
    # extension; independent illumination plus semantic/cross evidence licenses it.
    extended = completed & envelope & (appearance >= config.appearance_minimum) & ~core
    extended &= ((prior >= config.prior_seed) | (cross >= config.cross_seed)
                 | (core_connected & (appearance >= config.cross_seed)))
    target[extended | core] = 1
    extension_support = np.maximum(np.maximum(prior, cross), appearance*core_connected)
    weight[extended] = config.expansion_weight*extension_support[extended]
    weight[core] = 1.
    tier[extended], tier[core] = 2, 1
    known_negative = negative & receiver & ~completed
    target[known_negative], weight[known_negative], tier[known_negative] = 0, 1., 3
    return {"visible_mask": completed, "core_mask": core, "expanded_mask": extended,
        "target": target, "confidence": weight, "tier": tier,
        "candidate_source_bits": (prior >= config.prior_seed).astype(np.uint8)*16
                                 | (cross >= config.cross_seed).astype(np.uint8)*8
                                 | (core_connected & (appearance >= config.cross_seed)).astype(np.uint8)*32}
