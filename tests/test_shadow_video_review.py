import json

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.data import checked_write_image, digest, read_rows, write_rows
from shadow_occ.annotation.video_review import (compare_frame, find_clips, load_clip, observation_timeline,
    owner_region_transitions, summarize_tracks)
from shadow_occ.annotation.video_review_render import selected_projection
from shadow_occ.utils.io import write_json
from tests.test_bev_flow_ablations import script


def frozen_runs(tmp_path):
    shape = (8, 8)
    yy, xx = np.indices(shape)
    world = np.stack([xx, yy, np.zeros(shape)], axis=-1).astype(float)
    grid_projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    camera_projection = np.array([[8, 0, 0, 8], [0, 8, 0, 8], [0, 0, 0, 1]], float)
    frames = []
    for slot in range(3):
        image = np.full((80, 80, 3), 180, np.uint8)
        image[24:56, 24:56] = 60
        path = tmp_path/f"rgb_{slot}.jpg"
        checked_write_image(path, image)
        frames.append({"sample_data_token": f"cam{slot}", "camera_timestamp": slot*500000,
            "image": str(path), "image_size_wh": [80, 80], "projection_world_to_image": camera_projection.tolist()})
    pairs = tmp_path/"pairs.jsonl"
    write_rows(pairs, [{"scene_name": "scene", "camera": "CAM_FRONT", "frames": frames[at:at+2]} for at in range(2)])
    corners = [[x, y, z] for x in (2, 3) for y in (2, 3) for z in (0, 1)]
    for version, identity, vocabulary in (("before", 1, ["A", "B"]), ("after", 7, ["B", "A"])):
        folder = tmp_path/version/"clip"
        folder.mkdir(parents=True)
        protocol = {"clip": {"id": "clip", "scene_name": "scene", "camera": "CAM_FRONT"},
            "owner_index_to_instance_token": vocabulary, "ground_plane_world": [0, 0, 1, 0]}
        write_json(folder/"protocol.json", protocol)
        rows, memories = [], []
        for slot in range(3):
            proposal = np.zeros(shape, bool)
            if slot < 2:
                proposal[2:6, 2:6] = True
            positive = np.zeros(shape, bool)
            if slot == 1:
                positive[3, 3] = True
                positive[3, 4] = version == "before"
                positive[4, 3] = version == "after"
            owner = np.where(positive, vocabulary.index("A"), -1).astype(np.int32)
            tracks = np.where(proposal, identity, 0).astype(np.int32)
            source = np.zeros(shape, np.uint8)
            source[4, 3] = 8
            seed = np.zeros(shape, bool)
            seed[4, 3] = version == "after" and slot == 1
            arrays = {"receiver_world_xyz": world, "bev_world_to_grid": grid_projection,
                "shadow_target_bev": np.where(positive, 1, 255).astype(np.uint8),
                "shadow_proposal_bev": proposal, "shadow_owner_id": owner, "shadow_track_id": tracks,
                "appearance_score_map": np.full(shape, .08, np.float32),
                "proposal_source_bits": source, "cross_space_direct_seed": seed}
            if version == "after":
                arrays["bev_anchor_consistency_by_frame"] = np.full(shape, .95, np.float32)
            np.savez_compressed(folder/f"{slot:03d}.npz", **arrays)
            h = camera_projection @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
            image_proposal = cv2.warpPerspective(proposal.astype(np.uint8), h, (80, 80), flags=cv2.INTER_NEAREST) > 0
            image_positive = cv2.warpPerspective(positive.astype(np.uint8), h, (80, 80), flags=cv2.INTER_NEAREST) > 0
            np.savez_compressed(folder/f"{slot:03d}_image.npz", shadow_proposal_image=image_proposal,
                shadow_target_image=np.where(image_positive, 1, 255).astype(np.uint8),
                shadow_region_id_image=np.where(image_proposal, identity, 0).astype(np.int32))
            regions = [{"id": f"{slot:03d}_000", "shadow_track_id": identity, "image_area_pixels": int(image_proposal.sum()),
                "physics": {"shadow_accepted": slot == 1, "selected": {"instance_token": "A", "light_id": 0},
                            "association": {"instance_token": "A", "light_id": 0}}}] if slot < 2 else []
            rows.append({"frame_token": f"cam{slot}", "timestamp": slot*500000, "regions": regions,
                "evidence": f"/old/hpc/location/{slot:03d}.npz", "image_representation": f"/old/hpc/location/{slot:03d}_image.npz"})
            memories.append({"frame_token": f"cam{slot}", "regions": [{"shadow_track_id": identity,
                "missed_frames": int(slot == 2), "visibility_status": "occluded" if slot == 2 else "visible",
                "last_seen_region_id": "001_000", "existence_score": .99}]})
        write_rows(folder/"frames.jsonl", rows)
        write_rows(folder/"region_memory.jsonl", memories)
        write_rows(folder/"caster_memory.jsonl", [{"instance_token": "A", "observations": [
            {"slot": slot, "corners_world": corners} for slot in range(3)]}])
        # Deliberately not the selected caster mask: only the saved direction is used.
        np.savez_compressed(folder/"001_light_bank.npz", projected_box_masks=np.zeros((1, *shape), bool),
            directions_world=np.array([[1., 0., -1.]]))
        np.savez_compressed(folder/"pair_evidence.npz", delta_L=np.full((2, *shape), -2, np.float32))
    return tmp_path/"before", tmp_path/"after", pairs


def loaded(root):
    return load_clip(next(iter(find_clips(root).values())))


def test_frozen_review_groups_cells_preserves_inputs_and_exports_all_four_tools(tmp_path):
    before, after, pairs = frozen_runs(tmp_path)
    inputs = {path: digest(path) for root in (before, after) for path in root.rglob("*") if path.is_file()}
    output = tmp_path/"review"
    result = script("review_shadow_video_changes").build_review(before, after, [pairs], output)
    assert result["counts"]["added_positive"] == 1
    assert result["counts"]["removed_positive"] == 1
    assert result["counts"]["retained_positive"] == 1
    assert result["counts"]["net_positive_change"] == 0
    assert result["counts"]["owner_switched"] == 0  # Vocabulary order changed, caster did not.
    assert result["review_cases"] == 1  # Whole affected region, not one case per grid cell.
    assert not result["algorithm_recomputed"] and not result["ground_truth_generated"]
    cases = read_rows(output/"review_manifest.jsonl")
    case = cases[0]
    assert case["affected_region_cells"] == 16
    assert case["added_cell_source_bit_histogram"] == {"8": 1} or case["added_cell_source_bit_histogram"] == {8: 1}
    evidence = json.loads((output/case["evidence_json"]).read_text())
    assert evidence["after"]["anchor_only_does_not_mean_no_appearance"]
    assert evidence["after"]["frame_signals"]["appearance_score_map"]["changed_cells"]["mean"] > 0
    assert "bev_anchor_consistency_by_frame" in evidence["before"]["missing_frame_signals"]
    assert evidence["selected_projection"]["after"][0]["caster_token"] == "A"
    assert (output/case["clean_crop"]).is_file() and (output/case["temporal_context"]).is_file()
    assert (output/case["owner_review_image"]).is_file()
    assert cv2.imread(str(output/case["review_image"])).shape == (840, 1260, 3)
    with np.load(output/case["evidence_arrays"], allow_pickle=False) as archive:
        assert archive["added_positive"].sum() == 1
        assert archive["removed_positive"].sum() == 1
        assert "after_interval_000_delta_L" in archive.files
    timeline = read_rows(output/"observation_timeline.jsonl")
    remembered = [row for row in timeline if row["frame_slot"] == 2]
    assert all(row["observation_state"] == "retained_memory_only" for row in remembered)
    assert all(not row["observed_image_mask"] and row["image_positive_pixels"] == 0 for row in remembered)
    tracks = read_rows(output/"track_observations.jsonl")
    assert all(row["observed_image_frames"] == 2 and row["memory_only_frames"] == 1 for row in tracks)
    assert all(row["longest_observed_image_streak"] == 2 for row in tracks)
    assert all(not row["observed_in_every_clip_frame"] for row in tracks)
    assert all(row["shadow_judgment"] == "unreviewed" for row in read_rows(output/"reviews_template.jsonl"))
    assert all(digest(path) == sha for path, sha in inputs.items())
    transitions = read_rows(output/"owner_region_transitions.jsonl")
    assert next(row for row in transitions if row["frame_token"] == "cam1")["case_ids"] == [case["case_id"]]


def test_owner_comparison_uses_tokens_not_numeric_indices_or_run_track_ids(tmp_path):
    before_root, after_root, _ = frozen_runs(tmp_path)
    before, after = loaded(before_root), loaded(after_root)
    changes, old, new = compare_frame(before["arrays"][1], after["arrays"][1], before["protocol"], after["protocol"])
    assert not changes["owner_switched"].any()
    transitions = owner_region_transitions(before, after, 1, old, new)
    assert len(transitions) == 1 and transitions[0]["relation"] == "one_to_one"
    assert transitions[0]["owner_change"] == "same_owner_set"
    assert transitions[0]["before"][0]["track_id_within_run"] == 1
    assert transitions[0]["after"][0]["track_id_within_run"] == 7
    after["arrays"][1]["shadow_owner_id"][3, 3] = 0  # Now actually B, despite same numeric value as old A.
    changes, _, _ = compare_frame(before["arrays"][1], after["arrays"][1], before["protocol"], after["protocol"])
    assert changes["owner_switched"][3, 3]


def test_region_splits_are_not_counted_as_two_independent_new_caster_associations(tmp_path):
    before_root, after_root, _ = frozen_runs(tmp_path)
    before, after = loaded(before_root), loaded(after_root)
    after["arrays"][1]["shadow_target_bev"] = before["arrays"][1]["shadow_target_bev"].copy()
    after["arrays"][1]["shadow_track_id"][3, 4] = 8
    after["arrays"][1]["shadow_owner_id"][3, 4] = 1
    after["rows"][1]["regions"].append({"id": "001_001", "shadow_track_id": 8})
    _, old, new = compare_frame(before["arrays"][1], after["arrays"][1], before["protocol"], after["protocol"])
    transitions = owner_region_transitions(before, after, 1, old, new)
    assert len(transitions) == 1 and transitions[0]["relation"] == "split"
    assert transitions[0]["owner_change"] == "same_owner_set"


def test_new_resolved_region_is_not_reported_as_disambiguating_an_old_region(tmp_path):
    before_root, after_root, _ = frozen_runs(tmp_path)
    before, after = loaded(before_root), loaded(after_root)
    after["arrays"][1]["shadow_target_bev"][3, 3] = 255
    _, old, new = compare_frame(before["arrays"][1], after["arrays"][1], before["protocol"], after["protocol"])
    transitions = owner_region_transitions(before, after, 1, old, new)
    assert {row["owner_change"] for row in transitions} == {"resolved_new_region", "removed_resolved_region"}


def test_memory_masks_are_not_invented_and_orphan_masks_are_flagged(tmp_path):
    before_root, _, _ = frozen_runs(tmp_path)
    clip = loaded(before_root)
    clip["images"][2]["shadow_region_id_image"][20, 20] = 1
    clip["images"][2]["shadow_proposal_image"][20, 20] = True
    timeline = observation_timeline(clip, "before")
    last = next(row for row in timeline if row["frame_slot"] == 2)
    assert last["observation_state"] == "orphan_mask" and not last["observed_image_mask"]
    assert summarize_tracks(timeline, 3)[0]["observed_image_frames"] == 2


def test_selected_geometry_uses_saved_caster_and_light_not_requested_track_bank_mask(tmp_path):
    _, after_root, _ = frozen_runs(tmp_path)
    after = loaded(after_root)
    mask, _, records = selected_projection(after, 1, after["rows"][1]["regions"])
    assert mask.any()
    assert records[0]["status"] == "rendered_from_saved_selected_hypothesis"
    assert records[0]["geometry_reprojected_for_review_only"]


def test_self_comparison_still_exports_continuity_without_fabricating_review_cases(tmp_path):
    before, _, pairs = frozen_runs(tmp_path)
    output = tmp_path/"self_review"
    result = script("review_shadow_video_changes").build_review(before, before, [pairs], output)
    assert result["review_cases"] == 0 and result["counts"]["added_positive"] == 0
    assert read_rows(output/"review_manifest.jsonl") == []
    assert len(read_rows(output/"continuity_manifest.jsonl")) == 2


def test_different_grids_are_rejected_without_resizing_labels(tmp_path):
    before_root, after_root, _ = frozen_runs(tmp_path)
    before, after = loaded(before_root), loaded(after_root)
    after["arrays"][1]["receiver_world_xyz"][..., 0] += 1
    with pytest.raises(ValueError, match="different receiver grids"):
        compare_frame(before["arrays"][1], after["arrays"][1], before["protocol"], after["protocol"])


def test_review_cannot_write_inside_frozen_run(tmp_path):
    before, after, pairs = frozen_runs(tmp_path)
    with pytest.raises(ValueError, match="outside both frozen"):
        script("review_shadow_video_changes").build_review(before, after, [pairs], after/"audit")


def test_existing_manual_reviews_are_not_overwritten(tmp_path):
    before, after, pairs = frozen_runs(tmp_path)
    output = tmp_path/"review"
    output.mkdir()
    original = output/"reviews_template.jsonl"
    original.write_text('{"judgment":"shadow"}\n')
    with pytest.raises(FileExistsError, match="not overwritten"):
        script("review_shadow_video_changes").build_review(before, after, [pairs], output)
    assert original.read_text() == '{"judgment":"shadow"}\n'


def test_mismatched_clip_windows_are_not_silently_dropped(tmp_path):
    before, after, pairs = frozen_runs(tmp_path)
    path = after/"clip/frames.jsonl"
    rows = read_rows(path)
    rows[-1]["frame_token"] = "different_camera_token"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="same scene/camera/frame windows"):
        script("review_shadow_video_changes").build_review(before, after, [pairs], tmp_path/"review")
