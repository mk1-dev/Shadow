import json

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.shadow_completion import (CompletionConfig, ShadowPriorCache,
    PhysicalCoreReference, complete_visible_shadow, fuse_prior_proposals, native_illumination_evidence, plausible_projection_envelope)


def example():
    image = np.full((30, 40, 3), 160, np.uint8)
    image[8:24, 8:30] = 65
    core = np.zeros(image.shape[:2], bool)
    core[12:18, 12:18] = True
    receiver = np.ones(core.shape, bool)
    prior = np.zeros(core.shape, np.float32)
    prior[8:24, 8:30] = .9
    appearance = prior*.5
    envelope = np.zeros_like(core)
    envelope[6:26, 6:32] = True
    return image, core, receiver, prior, np.zeros_like(prior), appearance, envelope


def illumination(args):
    return {"qualified": args[5] > .05, "score": args[5].copy()}


def test_completion_preserves_core_and_expands_only_independently_supported_visible_pixels():
    pytest.importorskip("maxflow")
    args = list(example())
    args[2][:, 25:] = False
    result = complete_visible_shadow(*args, illumination=illumination(args))
    assert result["expanded_mask"].sum() > args[1].sum()
    assert (result["target"][args[1]] == 1).all()
    assert (result["confidence"][args[1]] == 1).all()
    assert (result["confidence"][result["expanded_mask"]] <= .25).all()
    assert (result["target"][:, 25:] == 255).all()
    assert not result["visible_mask"][:, 25:].any()
    assert not (result["expanded_mask"] & ~args[-1]).any()


@pytest.mark.parametrize("missing", ["appearance", "physics"])
def test_prior_only_is_a_candidate_not_a_positive_label(missing):
    pytest.importorskip("maxflow")
    args = list(example())
    args[1][:] = False
    args[5 if missing == "appearance" else 6][:] = 0
    result = complete_visible_shadow(*args, illumination=illumination(args))
    assert result["visible_mask"].any()
    assert not (result["target"] == 1).any()


def test_cross_space_seed_does_not_need_a_pretrained_positive():
    pytest.importorskip("maxflow")
    args = list(example())
    args[4] = args[3].copy()
    args[3][:] = 0
    result = complete_visible_shadow(*args, illumination=illumination(args))
    assert result["expanded_mask"].any()
    assert (result["candidate_source_bits"][result["expanded_mask"]] & 8).all()
    assert not (result["candidate_source_bits"] & 16).any()


def test_native_illumination_can_fill_a_prior_hole_connected_to_physical_core():
    pytest.importorskip("maxflow")
    args = list(example())
    args[3][18:20, 18:20] = 0
    args[5][18:20, 18:20] = 1
    result = complete_visible_shadow(*args, illumination=illumination(args))
    assert result["expanded_mask"][18:20, 18:20].all()
    assert (result["confidence"][18:20, 18:20] > 0).all()


def test_static_dark_material_cannot_bootstrap_new_cores_from_fsd_and_projection():
    pytest.importorskip("maxflow")
    args = list(example())
    args[1][:] = False
    args[4][:] = 1  # Even strong BEV-assisted appearance is not an illumination event.
    args[5][:] = 1
    result = complete_visible_shadow(*args)
    assert result["visible_mask"].any()
    assert result["unsupported_static_prior"].any()
    assert (result["target"] == 255).all()
    assert not result["confidence"].any()


def test_inherited_static_cores_survive_without_new_illumination():
    pytest.importorskip("maxflow")
    args = example()
    result = complete_visible_shadow(*args)
    np.testing.assert_array_equal(result["target"] == 1, args[1])
    assert not result["expanded_mask"].any()


def test_expansion_confidence_tracks_illumination_not_saturated_prior():
    pytest.importorskip("maxflow")
    args = example()
    first = complete_visible_shadow(*args, illumination=illumination(args))
    dim = illumination(args)
    dim["score"] *= .25
    second = complete_visible_shadow(*args, illumination=dim)
    mask = first["expanded_mask"] & second["expanded_mask"]
    assert mask.any()
    np.testing.assert_allclose(second["confidence"][mask], first["confidence"][mask]*.25)


def region_event(args):
    shape = args[1].shape
    seed = np.zeros(shape, bool)
    seed[12:14, 12:14] = True
    return {"qualified": seed, "score": seed.astype(np.float32)*.6,
        "joint_onset": seed, "joint_recovery": np.zeros(shape, bool),
        "anchor_available": np.ones(shape, bool), "correspondence_quality": np.ones(shape, np.float32)*.8,
        "shadow_state_conflict": np.zeros(shape, np.float32)}


def test_joint_events_can_complete_interiors_without_claiming_pixelwise_observations():
    pytest.importorskip("maxflow")
    args = example()
    observed = region_event(args)
    config = CompletionConfig(region_growth_radius_px_at_800=160)
    result = complete_visible_shadow(*args, illumination=observed, config=config)
    inferred = result["region_inferred_mask"]
    assert inferred.any()
    assert not (inferred & observed["qualified"]).any()
    assert not (inferred & ~args[-1]).any()
    assert (result["target"][inferred] == 1).all()
    assert (result["confidence"][inferred] > 0).all()
    assert result["confidence"][inferred].max() < .1
    assert (result["candidate_source_bits"][inferred] & 64).all()


@pytest.mark.parametrize("missing", ["joint_event", "visibility", "anchor", "physics"])
def test_interior_inference_requires_connected_observed_support(missing):
    pytest.importorskip("maxflow")
    args = list(example())
    observed = region_event(args)
    if missing == "joint_event":
        observed["joint_onset"][:] = False
    elif missing == "visibility":
        args[2][:] = False
    elif missing == "anchor":
        observed["anchor_available"][:] = False
    else:
        args[-1][:] = False
    result = complete_visible_shadow(*args, illumination=observed, config=CompletionConfig(region_growth_radius_px_at_800=160))
    assert not result["region_inferred_mask"].any()


def test_region_inference_stops_at_color_and_occlusion_boundaries():
    pytest.importorskip("maxflow")
    args = list(example())
    observed = region_event(args)
    args[0][:, 20:] = 180
    args[2][18:20] = False
    result = complete_visible_shadow(*args, illumination=observed, config=CompletionConfig(region_growth_radius_px_at_800=400))
    assert not result["region_inferred_mask"][:, 20:].any()
    assert not result["region_inferred_mask"][18:].any()


def test_prior_union_does_not_veto_existing_cross_space_proposals():
    maps = {"receiver_valid_by_frame": np.ones((2, 3, 3), bool),
        "proposal_score_map": np.zeros((2, 3, 3), np.float32),
        "proposal_source_bits": np.zeros((2, 3, 3), np.uint8)}
    maps["proposal_score_map"][:, 0, 0] = .8
    maps["proposal_source_bits"][:, 0, 0] = 8
    maps["receiver_valid_by_frame"][:, 2, 2] = False
    uv = np.indices((3, 3))[::-1].transpose(1, 2, 0).astype(float)
    prior = np.zeros((3, 3), np.float32)
    prior[1:, 1:] = .9
    result = fuse_prior_proposals(maps, [prior]*2, [uv]*2)
    assert (result["proposal_score_map"][:, 0, 0] == .8).all()
    assert result["shadow_prior_only_seed"][:, 1, 1].all()
    assert not result["shadow_prior_only_seed"][:, 2, 2].any()


def test_rejected_physics_cannot_supply_a_completion_envelope():
    result = plausible_projection_envelope({"shadow_accepted": False}, lambda _: {}, 0,
        np.array([0, 0, 1, 0]), [], np.eye(3, 4), (3, 3))
    assert not result.any()


def test_prior_cache_checks_token_camera_time_rgb_and_response_hash(tmp_path):
    image = tmp_path/"image.png"
    checked_write_image(image, np.zeros((3, 4, 3), np.uint8))
    response = tmp_path/"frame.npz"
    np.savez_compressed(response, binary_response=np.full((3, 4), 2., np.float32),
                        semantic_probability=np.full((3, 4), .8, np.float32))
    row = {"frame_token": "frame", "scene_name": "scene", "camera": "front", "timestamp": 42,
        "image_sha256": digest(image), "response": response.name, "response_sha256": digest(response)}
    write_rows(tmp_path/"manifest.jsonl", [row])
    (tmp_path/"protocol.json").write_text('{}', encoding="utf-8")
    frame = {"sample_data_token": "frame", "camera_timestamp": 42, "image_size_wh": [4, 3]}
    cache = ShadowPriorCache(tmp_path/"manifest.jsonl")
    assert (cache.load(frame, "scene", "front", image) == 1).all()  # Clip ReLU response, no sigmoid.
    binary, semantic = cache.load(frame, "scene", "front", image, return_semantic=True)
    assert (binary == 1).all()
    np.testing.assert_allclose(semantic, .8)  # Already sigmoid; do not apply it twice.
    with pytest.raises(ValueError, match="identity/time"):
        cache.load(frame, "scene", "other", image)
    np.savez_compressed(response, binary_response=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="response changed"):
        cache.load(frame, "scene", "front", image)


def test_export_retains_tiers_and_confidence_through_dataset_merge(tmp_path):
    from tests.test_shadow_mainline_experiments import data_fixture
    from tests.test_bev_flow_ablations import script
    runs, negatives = data_fixture(tmp_path)
    for run in read_rows(runs):
        if run["method"] != "anchor":
            continue
        from pathlib import Path
        path = Path(run["run_root"])/"clip/001_image.npz"
        with np.load(path) as archive:
            payload = {key: archive[key] for key in archive.files}
        core = payload["shadow_target_image"] == 1
        payload["shadow_target_image"][40:44, 40:44] = 1
        payload["shadow_core_image"] = core
        payload["shadow_confidence_image"] = np.where(core, 1., np.where(payload["shadow_target_image"] == 1, .2, 0)).astype(np.float32)
        np.savez_compressed(path, **payload)
    output = tmp_path/"dataset"
    script("build_shadow_mainline_experiments").build_dataset(runs, output, [negatives], size=(80, 80))
    row = next(row for row in read_rows(output/"anchor/seg_train.jsonl") if row.get("pixel_supervision"))
    with np.load(row["pixel_supervision"]["path"]) as archive:
        assert (archive["tier"][40:44, 40:44] == 2).all()
        assert np.allclose(archive["confidence"][40:44, 40:44], .2)
        assert (archive["confidence"][archive["tier"] == 1] == 1).all()
    assert digest(row["pixel_supervision"]["path"]) == row["pixel_supervision"]["sha256"]


def test_layered_loss_does_not_cancel_expansion_weight_or_train_during_warmup():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import layered_masked_loss
    target = torch.tensor([[[0, 1, 1, 255]]])
    tier = torch.tensor([[[3, 1, 2, 0]]])
    confidence = torch.tensor([[[1., 1., .25, 0.]]])
    logits = torch.zeros((1, 1, 1, 4), requires_grad=True)
    loss, terms = layered_masked_loss(logits, target, tier, confidence, .5, expansion_scale=0)
    loss.backward()
    assert logits.grad[0, 0, 0, 2] == 0
    assert logits.grad[0, 0, 0, 3] == 0
    logits.grad.zero_()
    loss, terms = layered_masked_loss(logits, target, tier, confidence, .5, expansion_scale=1)
    loss.backward()
    assert torch.allclose(terms["expansion"], torch.tensor(.25*np.log(2), dtype=torch.float32))
    assert torch.allclose(logits.grad[0, 0, 0, 2], torch.tensor(-.125))
    assert logits.grad[0, 0, 0, 3] == 0


def test_native_illumination_uses_observed_pixels_and_does_not_treat_missing_exposure_as_event():
    from shadow_occ.annotation.physics_motion import plane_basis
    first = np.full((30, 40, 3), 160, np.uint8)
    second = first.copy()
    second[10:22, 10:25] = 65
    projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    frames = [{"projection_world_to_image": projection}]*2
    receivers = [np.ones((30, 40), bool)]*2
    exposure = [{"valid": True, "scale": 1., "offset_L": 0.}]
    score, valid = native_illumination_evidence([first, second], receivers, frames,
        plane_basis([0, 0, 1, 0]), exposure, 1)
    assert valid.all() and score[15, 15] > .5
    assert score[2, 2] == 0
    exposure[0]["valid"] = False
    score, valid = native_illumination_evidence([first, second], receivers, frames,
        plane_basis([0, 0, 1, 0]), exposure, 1)
    assert not valid.any() and not score.any()


def test_layered_dataset_flips_and_resizes_tiers_with_target(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import ShadowDataset
    image = tmp_path/"rgb.png"
    target = tmp_path/"target.png"
    checked_write_image(image, np.zeros((3, 4, 3), np.uint8))
    checked_write_image(target, np.tile(np.array([0, 1, 1, 255], np.uint8), (3, 1)))
    tiers = np.tile(np.array([3, 1, 2, 0], np.uint8), (3, 1))
    path = tmp_path/"tiers.npz"
    np.savez_compressed(path, tier=tiers, confidence=np.where(tiers == 2, .2, (tiers > 0).astype(float)).astype(np.float32))
    row = {"id": "one", "image": str(image), "target": str(target), "label_source": "temporal_weak",
        "pixel_supervision": {"path": str(path), "sha256": digest(path)}}
    monkeypatch.setattr(np.random, "random", lambda: 0.)
    batch = ShadowDataset([row], (3, 8), augment=True)[0]
    assert batch["supervision_tier"][0].tolist() == [0, 0, 2, 2, 1, 1, 3, 3]
    assert torch.allclose(batch["pixel_confidence"][0, 2:4], torch.tensor([.2, .2]))


def _layered_ddp_worker(rank, rendezvous, output):
    import torch
    from datetime import timedelta
    from shadow_occ.annotation.segmentation import layered_masked_loss
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank,
        world_size=2, timeout=timedelta(seconds=60))
    logits = torch.tensor([[[[.2, -.3, .5]]]], requires_grad=True)
    target = torch.tensor([[[1, 1, 1] if rank == 0 else [0, 255, 255]]])
    tier = torch.tensor([[[1, 2, 2] if rank == 0 else [3, 0, 0]]])
    confidence = torch.tensor([[[1., .2, .1] if rank == 0 else [1., 0., 0.]]])
    loss, _ = layered_masked_loss(logits, target, tier, confidence, .5)
    loss.backward()
    torch.save({"loss": loss.detach(), "gradient": logits.grad}, output+str(rank)+".pth")
    torch.distributed.destroy_process_group()


def test_expansion_reduction_does_not_overflow_half_precision_pixel_counts():
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import layered_masked_loss
    target = torch.ones((1, 256, 512), dtype=torch.long)
    tier = torch.full_like(target, 2)
    confidence = torch.full(target.shape, .25)
    logits = torch.zeros((1, 1, 256, 512), dtype=torch.float16, requires_grad=True)
    loss, terms = layered_masked_loss(logits, target, tier, confidence, .5)
    assert torch.allclose(terms["expansion"], torch.tensor(.25*np.log(2), dtype=torch.float32))
    assert torch.isfinite(loss)


def test_layered_ddp_loss_matches_global_batch_with_no_expansion_on_one_rank(tmp_path):
    torch = pytest.importorskip("torch")
    from shadow_occ.annotation.segmentation import layered_masked_loss
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    output = str(tmp_path/"rank")
    torch.multiprocessing.spawn(_layered_ddp_worker,
        args=((tmp_path/"rendezvous").as_uri(), output), nprocs=2, join=True)
    logits = torch.tensor([[[[.2, -.3, .5]]], [[[.2, -.3, .5]]]], requires_grad=True)
    target = torch.tensor([[[1, 1, 1]], [[0, 255, 255]]])
    tier = torch.tensor([[[1, 2, 2]], [[3, 0, 0]]])
    confidence = torch.tensor([[[1., .2, .1]], [[1., 0., 0.]]])
    loss, _ = layered_masked_loss(logits, target, tier, confidence, .5)
    loss.backward()
    for rank in range(2):
        actual = torch.load(output+str(rank)+".pth", weights_only=True)
        assert torch.allclose(actual["loss"], loss.detach(), atol=1e-6)
        assert torch.allclose(actual["gradient"]/2, logits.grad[rank:rank+1], atol=1e-6)
