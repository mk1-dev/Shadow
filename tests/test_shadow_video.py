import copy
import json

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.frame_receiver import frame_index, write_frame_receiver
from shadow_occ.annotation.shadow_video import (
    ShadowRegionMemory, VideoShadowConfig, bev_signatures, refine_regions, video_proposals,
)


def video_case(length=5):
    shape = (length, 16, 16)
    lab = np.zeros((*shape, 3), np.float32)
    lab[..., 0] = 70
    lab[..., 1:] = 128
    lab[1:3, 5:10, 5:10, 0] = 64
    features = np.ones((*shape, 2), np.float32)
    features[..., 1] = np.indices(shape[1:])[1]/16
    return {"lab": lab, "receiver": np.ones(shape, bool), "delta_L": np.diff(lab[..., 0], axis=0),
        "exposure_valid": np.ones((length-1, 1, 1), bool),
        "observed_flow": np.zeros((length-1, 16, 16, 2), np.float32),
        "expected_flow": np.zeros((length-1, 16, 16, 2), np.float32),
        "fb_consistent": np.ones((length-1, 16, 16), bool),
        "texture_support": np.ones((length-1, 16, 16), bool),
        "features": features, "feature_valid": np.ones(shape, bool)}


def test_missing_remote_frame_does_not_clear_other_observed_pairs():
    inputs = video_case()
    inputs["receiver"][-1] = False
    maps = video_proposals(**inputs)
    assert maps["dynamic_discovery_score_map"][1, 6, 6] > .9
    assert not maps["pair_correspondence_valid"][-1].any()
    assert maps["pair_correspondence_valid"][0].all()
    assert not maps["proposal_score_map"][-1].any()


def test_signed_changes_are_assigned_to_dark_observation_not_bright_endpoint():
    maps = video_proposals(**video_case())
    assert maps["onset"][1, 6, 6] and maps["recovery"][2, 6, 6]
    assert not maps["dynamic_discovery_score_map"][0, 6, 6]
    assert not maps["dynamic_discovery_score_map"][3, 6, 6]


def test_missing_flow_is_not_motion_but_brightness_can_still_propose():
    inputs = video_case()
    inputs["observed_flow"][:] = np.nan
    maps = video_proposals(**inputs)
    assert not maps["flow_present"].any()
    assert not maps["flow_confidence_map"].any()
    assert maps["dynamic_discovery_score_map"][1, 6, 6] > .9


def test_flow_only_or_caster_prior_cannot_manufacture_shadow():
    inputs = video_case()
    inputs["lab"][..., 0] = 70
    inputs["delta_L"][:] = 0
    inputs["observed_flow"][:] = 100
    maps = video_proposals(**inputs, caster_prior=np.ones((5, 16, 16), np.float32))
    assert not maps["dynamic_discovery_score_map"].any()
    assert not (maps["proposal_score_map"] >= VideoShadowConfig().grow_score).any()


def test_bev_change_is_observed_without_assuming_illumination_invariance():
    inputs = video_case()
    original = inputs["features"].copy()
    stable = video_proposals(**inputs)
    inputs["features"][1] *= -1
    changed = video_proposals(**inputs)
    np.testing.assert_array_equal(stable["dynamic_discovery_score_map"], changed["dynamic_discovery_score_map"])
    assert changed["dynamic_discovery_score_map"][1, 6, 6] > .9
    assert changed["bev_feature_change"][0, 6, 6] > stable["bev_feature_change"][0, 6, 6]
    np.testing.assert_array_equal(inputs["features"][0], original[0])


def test_bev_stability_is_an_explicit_hypothesis_control_not_default():
    inputs = video_case()
    inputs["features"][1] *= -1
    default = video_proposals(**inputs)
    control = video_proposals(**inputs, bev_role="stability")
    assert default["dynamic_discovery_score_map"][1, 6, 6] > control["dynamic_discovery_score_map"][1, 6, 6]


def test_geometry_following_flow_keeps_shadow_signal_without_error_bonus():
    inputs = video_case()
    matched = video_proposals(**inputs)
    inputs["observed_flow"][:] = 100
    mismatched = video_proposals(**inputs)
    np.testing.assert_array_equal(matched["dynamic_discovery_score_map"], mismatched["dynamic_discovery_score_map"])
    assert matched["ground_consistent_illumination_score"][0, 6, 6] > .9
    assert mismatched["ground_consistent_illumination_score"][0, 6, 6] < .01


def test_entity_motion_explanation_suppresses_dynamic_discrepancy():
    inputs = video_case()
    maps = video_proposals(**inputs, entity_explanation=np.ones_like(inputs["delta_L"]))
    assert not maps["discrepancy_score_map"].any()


def test_unsupported_entity_scores_cannot_suppress_observed_illumination():
    inputs = video_case()
    maps = video_proposals(**inputs, entity_explanation=np.ones_like(inputs["delta_L"]),
                          entity_explanation_valid=np.zeros_like(inputs["delta_L"], bool))
    assert np.isnan(maps["discrepancy_score_map"][1, 6, 6])
    assert maps["appearance_partial_score_map"][1, 6, 6] > .9
    assert maps["proposal_score_map"][1, 6, 6] > .9


def test_fb_failure_is_not_claimed_to_be_occlusion():
    inputs = video_case()
    inputs["fb_consistent"][:] = False
    maps = video_proposals(**inputs)
    assert maps["flow_fb_inconsistent"].all()
    assert not maps["flow_occlusion_known"].any()
    assert not maps["flow_occluded"].any()


def test_explicit_occlusion_and_fb_uncertainty_have_separate_maps():
    inputs = video_case()
    occlusion = np.zeros_like(inputs["delta_L"], bool)
    occlusion[:, 4, 4] = True
    maps = video_proposals(**inputs, occlusion_mask=occlusion,
                          fb_error_px=np.ones_like(inputs["delta_L"]))
    assert maps["flow_occlusion_known"].all()
    assert not maps["flow_confidence_map"][:, 4, 4].any()
    assert 0 < maps["flow_confidence_map"][0, 0, 0] < 1


def test_unknown_bev_features_do_not_claim_free_ground():
    inputs = video_case()
    inputs["feature_valid"][:] = False
    maps = video_proposals(**inputs)
    assert np.isnan(maps["bev_pair_stability"]).all()
    assert maps["dynamic_discovery_score_map"][1, 6, 6] > .9


def test_nonfinite_receiver_appearance_is_unknown_not_static_darkness():
    inputs = video_case()
    inputs["lab"][:, 4, 4] = np.nan
    maps = video_proposals(**inputs)
    assert not maps["receiver_valid_by_frame"][:, 4, 4].any()
    assert not maps["proposal_score_map"][:, 4, 4].any()


def test_neighbourhood_signature_is_finite_only_where_supported():
    inputs = video_case()
    inputs["feature_valid"][:, 0] = False
    maps = bev_signatures(inputs["features"], inputs["feature_valid"])
    assert np.isnan(maps["bev_feature_change"][:, 0]).all()
    np.testing.assert_allclose(maps["bev_feature_change"][:, 1:], 0, atol=1e-6)


def test_watershed_stays_in_observed_support_and_requires_seed():
    inputs = video_case()
    score = np.zeros((16, 16), np.float32)
    score[4:12, 4:12] = .25
    score[6:9, 6:9] = .8
    receiver = np.ones((16, 16), bool)
    receiver[:, 10:] = False
    regions = refine_regions(score, np.zeros_like(score), receiver, inputs["features"][0], inputs["feature_valid"][0])
    assert regions
    assert all(not region["mask"][:, 10:].any() for region in regions)
    assert not refine_regions(score*.1, np.zeros_like(score), receiver, inputs["features"][0], inputs["feature_valid"][0])


def region(x=4, descriptor=(1., 0.)):
    mask = np.zeros((16, 16), bool)
    mask[4:8, x:x+4] = True
    return {"mask": mask, "descriptor": np.asarray(descriptor), "rank_score": .8}


def test_bev_signature_controls_temporal_linking():
    memory = ShadowRegionMemory(.5)
    identity = memory.update([region()], 0)[0]["shadow_track_id"]
    # Equal geometry: a different BEV signature must not win due to list order.
    linked = memory.update([region(descriptor=(0., 1.)), region()], 500000)
    assert linked[1]["shadow_track_id"] == identity
    assert linked[0]["shadow_track_id"] != identity


def test_short_disappearance_retrieval_does_not_output_hallucinated_masks():
    memory = ShadowRegionMemory(.5)
    first = memory.update([region()], 0)[0]["shadow_track_id"]
    assert memory.update([], 500000) == []
    assert memory.update([region(5)], 1000000)[0]["shadow_track_id"] == first
    memory.update([], 1500000)
    assert memory.update([region()], 3000000)[0]["shadow_track_id"] != first


def test_memory_rejects_reversed_timestamps():
    memory = ShadowRegionMemory(.5)
    memory.update([], 10)
    with pytest.raises(ValueError, match="increasing"):
        memory.update([], 10)


def test_frame_export_and_index_use_one_observation_per_token(tmp_path):
    from tests.test_bev_flow import example
    pairs = example()[0]
    frame = pairs[0]["frames"][1]
    frame = write_frame_receiver(tmp_path, frame, np.zeros((64, 64, 3), np.uint8), np.ones((64, 64), bool),
                                 pairs[0]["plane_world"], pairs[0]["plane_basis_world"], np.eye(4))
    pairs[0]["frames"][1] = frame
    pairs[1]["frames"][0] = copy.deepcopy(frame)
    index = frame_index(pairs)
    assert len(index) == 3
    assert index[("scene1", "CAM_FRONT", "cam1")]["frame"]["ground_sha256"] == frame["ground_sha256"]
    assert json.loads((tmp_path/"frames/cam1/observation.json").read_text())["ground_model_id"] == frame["ground_model_id"]
    pairs[1]["frames"][0]["ground_sha256"] = "different"
    with pytest.raises(ValueError, match="inconsistent"):
        frame_index(pairs)


def test_legacy_frame_selection_is_deterministic_and_prefers_own_start_pair():
    from tests.test_bev_flow import example
    pairs = copy.deepcopy(example()[0])
    pairs[1]["frames"][0] = copy.deepcopy(pairs[1]["frames"][0])
    pairs[0]["frames"][1]["ground_sha256"] = "legacy_previous"
    pairs[1]["frames"][0]["ground_sha256"] = "own_plane"
    a, b = frame_index(pairs), frame_index(pairs[::-1])
    key = ("scene1", "CAM_FRONT", "cam1")
    assert a[key]["frame"]["ground_sha256"] == b[key]["frame"]["ground_sha256"] == "own_plane"


@pytest.mark.parametrize("with_boxes", [False, True])
@pytest.mark.parametrize("spatial_mode", [None, "current-only", "temporal"])
def test_raw_pair_sampling_keeps_partial_visibility_on_fixed_clip_grid(tmp_path, monkeypatch, with_boxes, spatial_mode):
    from shadow_occ.annotation.bev_flow import BEVFlowConfig
    from shadow_occ.annotation.data import checked_write_image, digest
    from tests.test_bev_flow import example
    from tests.test_bev_flow_ablations import script
    module = script("discover_shadow_video")
    pairs, images, grounds, _, _ = example()
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]
    if with_boxes:
        corners = np.array([[x, y, z] for x in (-2.5, -1.5) for y in (-2.5, -1.5) for z in (0, 2)])
        for frame in frames:
            frame["object_volumes_world"] = [{"instance_token": "entity", "corners_world": corners.tolist(),
                                             "source": "camera_time_interpolated_box"}]
    grounds[0][:] = False
    stats = {}
    for slot, frame in enumerate(frames):
        image_path, ground_path = tmp_path/f"{slot}.png", tmp_path/f"{slot}_ground.png"
        checked_write_image(image_path, images[slot])
        checked_write_image(ground_path, grounds[slot].astype(np.uint8)*255)
        frame.update(image=str(image_path), image_sha256=digest(image_path),
                     ground_support=str(ground_path), ground_sha256=digest(ground_path))
    for pair in pairs:
        path = tmp_path/f"{pair['id']}.npz"
        np.savez_compressed(path, flow_t_to_next=np.zeros((64, 64, 2), np.float32),
            flow_next_to_t=np.zeros((64, 64, 2), np.float32), receiver_flow_supported=np.ones((64, 64), bool))
        stats[pair["id"]] = {key: pair[key] for key in ("scene_name", "camera", "official_split")}
        stats[pair["id"]].update(evidence=str(path), evidence_sha256=digest(path))
    monkeypatch.setattr(module, "sample_cached_bev", lambda root, token, world:
        (np.ones((*world.shape[:2], 2), np.float32), np.ones(world.shape[:2], bool), {"history_mode": spatial_mode}))
    if spatial_mode == "temporal":
        with pytest.raises(ValueError, match="current-only"):
            module.sample_clip(pairs, frame_index(pairs), tmp_path, stats, tmp_path,
                              tmp_path, BEVFlowConfig(extent_m=4), spatial_feature_root=tmp_path)
        return
    inputs, context = module.sample_clip(pairs, frame_index(pairs), tmp_path, stats, tmp_path,
                                       tmp_path, BEVFlowConfig(extent_m=4),
                                       spatial_feature_root=tmp_path if spatial_mode else None)
    assert context["spatial_feature_valid"].any() == bool(spatial_mode)
    assert not inputs["receiver"][0].any()
    assert inputs["receiver"][1:].any()
    if not with_boxes:
        assert inputs["receiver"][1:].all()
    maps = video_proposals(**inputs)
    assert not maps["pair_correspondence_valid"][0].any()
    assert maps["pair_correspondence_valid"][1].any()
    assert not (inputs["entity_explanation_valid"] & maps["pair_correspondence_valid"]).any()
    if with_boxes:
        assert inputs["entity_explanation_valid"].any()
    # The two domains are intentionally disjoint. A validated ground reference,
    # not spreading box flow, must still make complete receiver evidence possible.
    assert (maps["physical_reference_source"][1] == 2).any()
    assert maps["discrepancy_evidence_complete_by_interval"][1].any()
    assert np.isfinite(maps["discrepancy_score_map"]).any()
    assert context["receiver_sources"][1]["policy"] == "canonical_legacy_pair_reference"
    assert "metric_flow" in context and "metric_flow" not in inputs


def test_all_frame_count_box_motion_support():
    from shadow_occ.annotation.receiver_explanations import box_ray_evidence
    from tests.test_bev_flow import example
    pairs, _, _, _, _ = example()
    frames = [pairs[0]["frames"][0]]*5
    result = box_ray_evidence(np.zeros((4, 4, 3)), frames,
        [np.zeros((4, 4, 2))]*5, [np.zeros((4, 4, 2))]*4)
    assert result["physical_ray_volume_support"].shape == (5, 4, 4)
    assert result["box_surface_flow_valid"].shape == (4, 4, 4)


@pytest.mark.parametrize("stage", ["signal", "projection", "full"])
@pytest.mark.parametrize("entity_search", [False, True])
def test_clip_runner_exports_video_masks_and_separate_shadow_bev(tmp_path, monkeypatch, stage, entity_search):
    from tests.test_bev_flow_ablations import script
    from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_projection, ground_grid
    from shadow_occ.annotation.physics_motion import plane_basis
    from tests.test_bev_flow import example
    module = script("discover_shadow_video")
    inputs = video_case(3)
    if entity_search:
        inputs["receiver"][:] = False
        inputs["entity_occlusion_mask"] = np.ones_like(inputs["receiver"])
        inputs["search_domain"] = np.ones_like(inputs["receiver"])
    pairs, images, grounds, _, _ = example()
    frames = [pairs[0]["frames"][0], pairs[1]["frames"][0], pairs[1]["frames"][1]]
    grid = BEVFlowConfig(extent_m=4).grid()
    basis = plane_basis([0, 0, 1, 0])
    context = {"frames": frames, "images": images, "grounds": grounds,
        "plane": np.array([0, 0, 1, 0]), "basis": basis, "anchor": np.eye(4), "grid": grid,
        "world": ground_grid(basis, np.eye(4), grid), "projection": bev_projection(np.eye(4), grid),
        "sample_tokens": ["s0", "s1", "s2"], "receiver_sources": [], "exposures": [],
        "occupied": np.zeros((3, 16, 16), bool), "coordinates": [np.zeros((16, 16, 2))]*3,
        "observed_flow": inputs["observed_flow"], "expected_flow": inputs["expected_flow"]}
    rng = np.random.default_rng(7)
    spatial = rng.normal(size=(16, 16, 8)).astype(np.float32)
    context.update(spatial_features=np.stack([spatial]*3), spatial_feature_valid=inputs["receiver"].copy(),
        metric_flow=np.zeros((2, 16, 16, 2), np.float32), metric_flow_valid=np.ones((2, 16, 16), bool))
    class Writer:
        def __init__(self, *args):
            self.frames = 0
        def isOpened(self):
            return True
        def write(self, frame):
            assert frame.shape == (1024, 1152, 3)
            self.frames += 1
        def release(self):
            pass
    monkeypatch.setattr(module.cv2, "VideoWriter", Writer)
    if stage != "full":
        def forbidden(*args, **kwargs):
            pytest.fail("Signal/projection stages must not run physical acceptance")
        monkeypatch.setattr(module, "rank_hypotheses", forbidden)
        monkeypatch.setattr(module, "factorized_shadow_decision", forbidden)
    result = module.run_clip({"id": "test", "track_id": "car", "scene_name": "scene1"}, pairs,
        inputs, context, tmp_path/"clip", VideoShadowConfig(), stage=stage,
        reviews={"cam1": [{"region_type": "shadow", "mask": np.ones((64, 64), bool)}]})
    assert result["frames"] == 3 and result["regions"] > 0
    with np.load(tmp_path/"clip/001.npz", allow_pickle=False) as archive:
        assert archive["shadow_target_bev"].shape == (16, 16)
        assert set(np.unique(archive["shadow_target_bev"])) <= {1, 255}
        assert (archive["shadow_owner_id"] == -1).all()
        if entity_search:
            assert (archive["shadow_target_bev"] == 255).all()
            assert not archive["receiver_valid_by_frame"].any()
            assert archive["search_valid_by_frame"].any()
        assert "bev_feature_change" not in archive.files  # interval maps live in pair_evidence
        assert "cross_space_region_score" in archive.files
        assert "cross_space_region_appearance_support" in archive.files
        assert "cross_space_temporal_source_frame" in archive.files
    with np.load(tmp_path/"clip/pair_evidence.npz", allow_pickle=False) as archive:
        assert "bev_feature_change" in archive.files
        assert "spatial_bev_displacement_m" in archive.files
        assert "spatial_bev_match_state" in archive.files
        assert "flow_bev_discrepancy_supported" in archive.files
    with np.load(tmp_path/"clip/001_image.npz", allow_pickle=False) as archive:
        assert archive["shadow_proposal_image"].shape == (64, 64)
        assert set(np.unique(archive["shadow_target_image"])) <= {1, 255}
    assert len((tmp_path/"clip/region_memory.jsonl").read_text().splitlines()) == 3
    assert cv2.imread(str(tmp_path/"clip/001_target.png"), cv2.IMREAD_GRAYSCALE).shape == (64, 64)
    protocol = json.loads((tmp_path/"clip/protocol.json").read_text())
    assert protocol["protocol"] == "entity-shadow-disentangled-video-v8"
    assert protocol["spatial_match_states"] == {"0": "missing", "1": "low_confidence", "2": "reliable"}
    assert protocol["stage"] == stage and protocol["bev_role"] == "context"
    assert len(protocol["modules"]) == 4
    review = json.loads((tmp_path/"clip/reviewed_signals.jsonl").read_text().strip())
    assert "projection_capacity" not in review
    assert "projection_oracle_capacity" in review and "projection_selected_direction" in review
    assert protocol["entity_state_legend"]["0"] == "missing"
    with np.load(tmp_path/"clip/001.npz", allow_pickle=False) as archive:
        assert "appearance_partial_score_map" in archive.files
        assert not archive["discrepancy_evidence_complete"].any()


def test_projection_capacity_does_not_confuse_max_cover_with_iou():
    from shadow_occ.annotation.shadow_projection import projection_coverage
    observed = np.zeros((10, 10), bool)
    observed[4:6, 4:6] = True
    bank = np.ones((2, 10, 10), bool)
    bank[1] = False
    bank[1, 4, 4:6] = True
    report = projection_coverage(observed, bank)
    assert report["best_cover"] == 1 and report["best_cover_light_id"] == 0
    assert report["best_iou"] == .5 and report["best_iou_light_id"] == 1
    assert report["union_cover"] == 1
    invalid = projection_coverage(observed, bank, valid=np.array([False, True]))
    assert invalid["best_cover_light_id"] == 1 and invalid["invalid_hypotheses"] == 1


def test_projection_capacity_empty_reference_is_not_perfect_coverage():
    from shadow_occ.annotation.shadow_projection import projection_coverage
    report = projection_coverage(np.zeros((4, 4), bool), np.ones((96, 4, 4), bool))
    assert report["best_cover"] is None and report["best_iou"] is None


def test_reviewed_signals_are_distributions_not_new_training_targets():
    from shadow_occ.annotation.shadow_projection import summarize_signals
    inputs = video_case()
    maps = video_proposals(**inputs)
    shadow = np.zeros((16, 16), bool)
    shadow[5:10, 5:10] = True
    before = inputs["lab"].copy()
    report = summarize_signals(inputs, maps, 1, shadow)
    assert report["region_cells"] == 25
    assert report["signals"]["flow_residual_px"]["mean"] == 0
    assert report["signals"]["discrepancy"]["mean"] is None
    assert report["signals"]["partial_appearance"]["mean"] > .9
    np.testing.assert_array_equal(inputs["lab"], before)


def test_entity_missing_and_supported_failure_have_distinct_evidence_without_losing_candidates():
    inputs = video_case()
    missing = video_proposals(**inputs)
    supported = video_proposals(**inputs, entity_explanation=np.zeros_like(inputs["delta_L"]),
                                entity_explanation_valid=np.ones_like(inputs["delta_L"], bool))
    explained = video_proposals(**inputs, entity_explanation=np.ones_like(inputs["delta_L"]),
                                entity_explanation_valid=np.ones_like(inputs["delta_L"], bool))
    assert not missing["entity_explanation_state"].any()
    assert (supported["entity_explanation_state"] == 2).all()
    assert (explained["entity_explanation_state"] == 1).all()
    assert np.isnan(missing["discrepancy_score_map"]).all()
    assert supported["discrepancy_score_map"][1, 6, 6] > .9
    assert not explained["discrepancy_score_map"].any()
    assert missing["partial_evidence_candidate"][1, 6, 6]
    assert missing["proposal_source_bits"][1, 6, 6] == 1
    assert supported["proposal_source_bits"][1, 6, 6] == 3
    np.testing.assert_array_equal(missing["proposal_score_map"], supported["proposal_score_map"])


def test_nonvisible_receivers_are_not_partial_appearance_evidence():
    inputs = video_case()
    inputs["receiver"][:] = False
    maps = video_proposals(**inputs)
    assert np.isnan(maps["discrepancy_score_map"]).all()
    assert not maps["appearance_partial_score_map"].any()
    assert not maps["partial_evidence_candidate"].any()


def test_ground_receiver_has_complete_reference_while_box_motion_remains_missing():
    inputs = video_case()
    ground = np.ones_like(inputs["delta_L"], bool)
    maps = video_proposals(**inputs, ground_motion_reference_valid=ground)
    assert not maps["entity_explanation_valid_map"].any()
    assert not maps["entity_explanation_state"].any()
    assert (maps["physical_reference_source"] == 2).all()
    assert maps["discrepancy_score_map"][1, 6, 6] > .9
    assert maps["proposal_source_bits"][1, 6, 6] == 3
    assert not maps["partial_evidence_candidate"].any()


def test_same_patch_ground_reference_does_not_require_boundary_flow_itself():
    inputs = video_case()
    inputs["fb_consistent"][:, 6, 6] = False
    maps = video_proposals(**inputs, ground_motion_reference_valid=np.ones_like(inputs["delta_L"], bool))
    assert not maps["flow_confidence_map"][:, 6, 6].any()
    assert maps["discrepancy_score_map"][1, 6, 6] > .9
    assert (maps["physical_reference_source"][:, 6, 6] == 2).all()


def test_ground_reference_cannot_fill_invisible_occluded_or_nonfinite_geometry():
    inputs = video_case()
    inputs["receiver"][:, 4, 4] = False
    inputs["expected_flow"][:, 5, 5] = np.nan
    occluded = np.zeros_like(inputs["delta_L"], bool)
    occluded[:, 6, 6] = True
    maps = video_proposals(**inputs, ground_motion_reference_valid=np.ones_like(inputs["delta_L"], bool),
                          occlusion_mask=occluded)
    for y, x in ((4, 4), (5, 5), (6, 6)):
        assert not maps["physical_reference_valid_map"][:, y, x].any()
        assert np.isnan(maps["pair_discrepancy_score_map"][:, y, x]).all()
    assert (maps["physical_reference_source"][:, 7, 7] == 2).all()


def test_no_ground_or_entity_reference_preserves_partial_candidates():
    inputs = video_case()
    maps = video_proposals(**inputs, ground_motion_reference_valid=np.zeros_like(inputs["delta_L"], bool))
    assert not maps["physical_reference_source"].any()
    assert np.isnan(maps["discrepancy_score_map"]).all()
    assert maps["appearance_partial_score_map"][1, 6, 6] > .9
    assert maps["proposal_score_map"][1, 6, 6] > .9


def test_zero_projection_overlap_does_not_select_light_zero_or_oracle_fallback():
    from shadow_occ.annotation.shadow_projection import projection_coverage, reviewed_projection_coverage
    bank = np.zeros((96, 8, 8), bool)
    bank[:, :2, :2] = True
    proposal = np.zeros((8, 8), bool)
    proposal[5:7, 5:7] = True
    report = projection_coverage(proposal, bank)
    assert report["status"] == "no_supported_direction"
    assert report["best_cover"] == report["best_iou"] == report["union_cover"] == 0
    assert report["best_cover_light_id"] is report["best_iou_light_id"] is None
    assert len(report["by_light"]) == 96
    review = reviewed_projection_coverage(bank[0], bank, report["best_cover_light_id"])
    assert review["projection_oracle_capacity"]["best_cover"] == 1
    assert review["projection_selected_direction"]["light_id"] is None
    assert review["projection_selected_direction"]["cover"] is None


def test_review_projection_keeps_proposal_selected_direction_when_oracle_disagrees():
    from shadow_occ.annotation.shadow_projection import projection_coverage, reviewed_projection_coverage
    bank = np.zeros((2, 8, 8), bool)
    bank[0, 2:4, 1:3] = True
    bank[1, 2:4, 5:7] = True
    proposal, reference = bank[0], bank[1]
    selected = projection_coverage(proposal, bank)["best_cover_light_id"]
    report = reviewed_projection_coverage(reference, bank, selected)
    assert report["projection_oracle_capacity"]["best_cover_light_id"] == 1
    assert report["projection_oracle_capacity"]["best_cover"] == 1
    assert report["projection_selected_direction"]["light_id"] == 0
    assert report["projection_selected_direction"]["cover"] == 0
    changed = reviewed_projection_coverage(bank[0], bank, selected)
    assert changed["projection_selected_direction"]["light_id"] == 0
    assert changed["projection_selected_direction"]["cover"] == 1


def test_review_projection_does_not_replace_missing_or_invalid_selection_with_oracle():
    from shadow_occ.annotation.shadow_projection import reviewed_projection_coverage
    bank = np.ones((2, 4, 4), bool)
    report = reviewed_projection_coverage(bank[0], bank, None)
    assert report["projection_oracle_capacity"]["best_cover"] == 1
    assert report["projection_selected_direction"]["status"] == "not_selected"
    assert report["projection_selected_direction"]["cover"] is None
    report = reviewed_projection_coverage(bank[0], bank, 0, valid=np.array([False, True]))
    assert report["projection_selected_direction"]["status"] == "invalid_direction"
    assert report["projection_selected_direction"]["cover"] is None


def test_reviewed_illumination_with_invalid_exposure_is_missing_not_evidence():
    from shadow_occ.annotation.shadow_projection import summarize_signals
    inputs = video_case()
    inputs["exposure_valid"][:] = False
    maps = video_proposals(**inputs)
    report = summarize_signals(inputs, maps, 1, np.ones((16, 16), bool))
    assert report["signals"]["illumination_change_abs_L"]["finite_samples"] == 0
    assert report["signals"]["illumination_change_abs_L"]["mean"] is None


def test_caster_capacity_keeps_all_directions_and_does_not_project_in_signal_stage(monkeypatch):
    from shadow_occ.annotation.bev_flow import BEVFlowConfig, bev_projection
    from shadow_occ.annotation.physics_labels import light_bank
    from tests.test_bev_flow_ablations import script
    module = script("discover_shadow_video")
    grid = BEVFlowConfig(extent_m=4).grid()
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0, 1)])
    points = np.vstack([[0, 0, .5], corners])
    chain = [{"objects": [{"instance_token": "car", "landmarks_world_t": points,
                           "landmarks_world_next": points}]}]*2
    context = {"grid": grid, "frames": [{}, {}, {}], "plane": np.array([0, 0, 1, 0]),
               "projection": bev_projection(np.eye(4), grid)}
    prior = module.caster_priors(chain, context, "car", light_bank())
    assert prior.any()
    assert context["caster_projection_bank"][0].shape == (96, 16, 16)
    assert context["caster_projection_valid"][0].all()
    def forbidden(*args, **kwargs):
        pytest.fail("Signal inspection must not run a physical projection search")
    monkeypatch.setattr(module, "footprint_mask", forbidden)
    prior = module.caster_priors(chain, context, "car", light_bank(), project=False)
    assert not prior.any() and not context["caster_projection_bank"]
