import copy
import json

import numpy as np
import pytest

from shadow_occ.annotation.cross_space_illumination import illumination_reasoning, same_world_anchor_pairs
from shadow_occ.annotation.dense_receiver import dense_receiver_observation
from shadow_occ.annotation.physics_motion import plane_basis, project_world
from shadow_occ.annotation.shadow_completion import PhysicalCoreReference
from shadow_occ.annotation.data import write_rows


def scene():
    from tests.test_bev_flow import example
    pairs, images, grounds, _, exposures = example()
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]
    receivers = [{"visible": mask, "confidence": mask.astype(np.float32)*.8} for mask in grounds]
    pair = {slot: (np.ones((64, 64), np.float32), np.ones((64, 64), bool)) for slot in range(3)}
    return images, receivers, frames, plane_basis([0, 0, 1, 0]), exposures, 1, [pair, pair]


def test_signed_light_change_detects_dark_current_frame_not_recovered_bright_frame():
    args = scene()
    dark = illumination_reasoning(*args)
    assert dark["qualified"][30:34, 30:34].all()
    assert dark["score"][32, 32] > 0
    assert not dark["qualified"][2, 2]
    recovered_args = list(args)
    recovered_args[5] = 2
    recovered = illumination_reasoning(*recovered_args)
    assert not recovered["qualified"].any()
    assert recovered["log_recovery"][32, 32] > .08


@pytest.mark.parametrize("missing", ["exposure", "anchor", "receiver"])
def test_missing_independent_evidence_never_becomes_a_positive(missing):
    args = list(scene())
    if missing == "exposure":
        for fit in args[4]:
            fit["valid"] = False
    elif missing == "anchor":
        for pair in args[6]:
            for _, available in pair.values():
                available[:] = False
    else:
        args[1][1]["visible"][:] = False
    result = illumination_reasoning(*args)
    assert not result["qualified"].any()
    assert not result["score"].any()


def test_persistent_dark_patch_is_unknown_even_with_identical_high_bev_anchors():
    args = scene()
    args[0][0][:] = args[0][1]
    args[0][2][:] = args[0][1]
    result = illumination_reasoning(*args)
    assert result["observed_pair"].all()
    assert result["anchor_consistency"].min() > .99
    assert not result["qualified"].any()


def test_fsd_state_transition_requires_rgb_illumination_and_preserves_direction():
    args = scene()
    states = [np.full((64, 64), .1, np.float32) for _ in range(3)]
    states[1][26:38, 26:38] = .9
    result = illumination_reasoning(*args, prior_states=states)
    assert result["joint_onset"][32, 32] and result["joint_recovery"][32, 32]
    assert result["joint_score"][32, 32] > 0
    assert result["shadow_state_increase"][32, 32] > .7
    for image in args[0]:
        image[:] = 180
    result = illumination_reasoning(*args, prior_states=states)
    assert not result["joint_onset"].any()
    assert not result["joint_recovery"].any()


def test_fsd_stability_or_missing_state_does_not_veto_independent_rgb_events():
    args = scene()
    states = [np.full((64, 64), .9, np.float32)]*3
    for prior in (states, [None]*3):
        result = illumination_reasoning(*args, prior_states=prior)
        assert result["qualified"][32, 32]
        assert not result["joint_onset"].any()


def test_exposure_fit_is_applied_in_both_time_directions():
    import cv2
    args = scene()
    args[0][0][:] = 180
    args[0][1][:] = 120
    args[0][2][:] = 180
    values = [cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[0, 0, 0]*100/255 for image in args[0]]
    args[4][0]["scale"] = values[0]/values[1]
    args[4][1]["scale"] = values[1]/values[2]
    result = illumination_reasoning(*args)
    assert not result["qualified"].any()
    assert result["log_darkening"].max() < 1e-5


def test_missing_flow_keeps_geometry_but_observed_contradiction_lowers_confidence():
    args = scene()
    shape = args[0][0].shape[:2]
    good = {"finite": np.ones(shape, bool), "reliable": np.ones(shape, bool), "residual": np.zeros(shape)}
    bad = {**good, "residual": np.full(shape, 100.)}
    missing = {**bad, "finite": np.zeros(shape, bool), "reliable": np.zeros(shape, bool)}
    reliable = illumination_reasoning(*args, pair_flows=[good, good])
    low = illumination_reasoning(*args, pair_flows=[bad, bad])
    absent = illumination_reasoning(*args, pair_flows=[missing, missing])
    assert reliable["flow_state"][32, 32] == 2
    assert low["flow_state"][32, 32] == 1
    assert absent["flow_state"][32, 32] == 0
    assert not low["qualified"][32, 32] and absent["qualified"][32, 32]
    assert low["log_darkening"][32, 32] > .08  # Observation remains, but not a licensed label.
    assert reliable["score"][32, 32] > low["score"][32, 32]
    assert absent["score"][32, 32] > low["score"][32, 32]


def test_anchor_pairs_have_independent_native_validity_and_reject_zero_features():
    features = np.ones((3, 2, 2, 4), np.float32)
    valid = np.ones((3, 2, 2), bool)
    features[1, 0, 0] = 0
    valid[2, 1, 1] = False
    cosine, available = same_world_anchor_pairs(features, valid)
    assert not available[:, 0, 0].any()
    assert not available[1, 1, 1]
    assert cosine[0, 1, 1] == 1


def receiver_case(spacing=1.):
    args = scene()
    frame, basis = args[2][0], args[3]
    x, y = np.meshgrid(np.arange(-1, 2)*spacing, np.arange(-1, 2)*spacing)
    world = np.stack([x, y, np.zeros_like(x)], axis=-1)
    visible = np.ones(x.shape, bool)
    sparse = np.zeros((64, 64), bool)
    uv, _, _ = project_world(world.reshape(-1, 3), frame["projection_world_to_image"])
    uv = np.rint(uv).astype(int)
    sparse[uv[:, 1], uv[:, 0]] = True
    return frame, world, visible, sparse, sparse.copy(), basis


def test_short_ground_gaps_are_interpolated_with_lower_confidence_not_scanlines():
    args = receiver_case()
    result = dense_receiver_observation(*args)
    assert result["details"]["accepted_metric_triangles"] > 0
    assert result["details"]["interpolated_pixels"] > result["details"]["observed_pixels"]
    assert result["state"][31, 31] == 2
    assert result["confidence"][31, 31] < result["confidence"][32, 32]
    assert not result["visible"][0, 0]


def test_long_or_missing_ground_gaps_are_not_filled():
    args = receiver_case(spacing=3.)
    result = dense_receiver_observation(*args)
    assert result["details"]["accepted_metric_triangles"] == 0
    np.testing.assert_array_equal(result["visible"], args[3])
    args = list(args)
    args[2][:] = False
    args[3][:] = False
    result = dense_receiver_observation(*args)
    assert not result["visible"].any()


def test_wider_metric_gaps_have_lower_interpolation_confidence():
    near = dense_receiver_observation(*receiver_case(spacing=1.))
    wider = dense_receiver_observation(*receiver_case(spacing=1.5))
    near_values = near["confidence"][near["state"] == 2]
    wide_values = wider["confidence"][wider["state"] == 2]
    assert near_values.size and wide_values.size
    assert wide_values.max() < near_values.max()


def test_dense_ground_excludes_box_ray_occlusion_not_its_entire_rectangle():
    args = list(receiver_case())
    from itertools import product
    corners = np.array(list(product([-.5, .5], [-.5, .5], [0., 2.])))
    args[0]["object_volumes_world"] = [{"corners_world": corners.tolist()}]
    result = dense_receiver_observation(*args)
    assert result["occluded"][32, 32]
    assert result["confidence"][32, 32] == 0
    assert not result["occluded"][28, 28]


def test_core_reference_preserves_only_pre_prior_camera_aligned_labels(tmp_path):
    root = tmp_path/"core"
    root.mkdir()
    images, _, frames, *_ = scene()
    clip = {"scene_name": "scene", "camera": "front"}
    protocol = {"clip": clip}
    (root/"protocol.json").write_text(json.dumps(protocol))
    rows = []
    for slot, frame in enumerate(frames):
        target = np.full((64, 64), 255, np.uint8)
        target[31:33, 31:33] = 1
        np.savez_compressed(root/f"{slot:03d}_image.npz", shadow_target_image=target)
        rows.append({"frame_token": frame["sample_data_token"], "timestamp": frame["camera_timestamp"]})
    write_rows(root/"frames.jsonl", rows)
    masks, provenance = PhysicalCoreReference(root).load(clip, frames, images)
    assert sum(mask.sum() for mask in masks) == 12
    assert len(provenance["sources"]) == 3
    wrong = copy.deepcopy(frames)
    wrong[0]["camera_timestamp"] += 1
    with pytest.raises(ValueError, match="timestamps"):
        PhysicalCoreReference(root).load(clip, wrong, images)
    protocol["shadow_prior"] = {"checkpoint": "FSD"}
    (root/"protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="independent"):
        PhysicalCoreReference(root)


def test_native_illumination_actively_seeds_candidates_without_static_darkness_or_fsd():
    from tests.test_bev_flow_ablations import script
    from shadow_occ.annotation.bev_flow import BEVFlowConfig, ground_grid, bev_image_cell_indices
    from shadow_occ.annotation.shadow_video import VideoShadowConfig
    module = script("discover_shadow_video")
    images, _, frames, basis, exposures, *_ = scene()
    grid = BEVFlowConfig(extent_m=4).grid()
    world = ground_grid(basis, np.eye(4), grid)
    masks = np.zeros((3, *grid.shape), bool)
    masks[:, ::2] = True  # Sparse evidence remains sparse after building the image domain.
    context = {"frames": frames, "images": images, "grounds": [np.ones((64, 64), bool)]*3,
        "grid": grid, "basis": basis, "world": world, "exposures": exposures,
        "spatial_features": np.ones((3, *grid.shape, 4), np.float32),
        "spatial_feature_valid": np.zeros_like(masks),
        "spatial_geometry": [{"sample_valid": np.ones(grid.shape, bool)}]*3}
    maps = {"receiver_valid_by_frame": masks.copy(), "search_valid_by_frame": masks.copy(),
        "proposal_score_map": np.zeros(masks.shape, np.float32),
        "proposal_source_bits": np.zeros(masks.shape, np.uint8)}
    cells = [bev_image_cell_indices(frame["projection_world_to_image"], basis, grid, (64, 64)) for frame in frames]
    receivers, evidence = module.completion_observations(maps, context, cells, VideoShadowConfig())
    assert evidence[1]["qualified"].any()
    assert not evidence[0]["qualified"].any()
    assert (maps["proposal_source_bits"][1] & 64).any()
    assert maps["proposal_score_map"][1].max() >= VideoShadowConfig().seed_score
    np.testing.assert_array_equal(maps["receiver_valid_by_frame"], masks)
    assert receivers[1]["details"]["interpolated_pixels"] > 0
