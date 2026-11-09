from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from shadow_occ.annotation.bev_flow import BEVFlowConfig
from shadow_occ.annotation.data import digest, read_rows, write_rows
from shadow_occ.annotation.entity_memory import EntityMotionConfig, OwnerScoringConfig
from shadow_occ.annotation.physics_labels import LabelConfig
from shadow_occ.annotation.shadow_video import VideoShadowConfig
from shadow_occ.annotation.video_correspondence import CorrespondenceConfig, CrossSpaceGrowthConfig, FlowConfidenceConfig
from shadow_occ.utils.io import write_json
from tests.test_bev_flow_ablations import script
from tests.test_shadow_video_review import frozen_runs


def release_setup(tmp_path):
    before, after, pairs = frozen_runs(tmp_path)
    review = tmp_path/"review"
    script("review_shadow_video_changes").build_review(before, after, [pairs], review)
    module = script("prepare_shadow_video_release")
    pack = tmp_path/"pack"
    module.prepare_blind(review, after, [pairs], pack)
    return module, pack, review, after, pairs


def complete_blind(pack):
    rows = read_rows(pack/"blind/judgments.jsonl")
    for row in rows:
        row.update(shadow_visibility="weak", shadow_types=["ambiguous"], caster_visibility="unjudgeable")
    write_rows(pack/"blind/judgments.jsonl", rows)
    return pack/"blind/judgments.jsonl"


def video_count(path):
    video = cv2.VideoCapture(str(path))
    assert video.isOpened()
    count = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        count += 1
    video.release()
    return count


def test_blind_pack_contains_no_model_labels_or_case_mapping(tmp_path):
    _, pack, review, _, _ = release_setup(tmp_path)
    public = pack/"blind"
    text = "\n".join(path.read_text() for path in public.iterdir() if path.suffix in {".json", ".jsonl", ".txt"})
    case = read_rows(review/"review_manifest.jsonl")[0]
    assert case["case_id"] not in text
    assert "scene_name" not in text and "owner_token" not in text and "added_positive" not in text
    assert not list(public.glob("*.npz")) and not list(public.glob("*.jpg"))
    assert video_count(next(public.glob("*.mp4"))) == 3
    assert read_rows(public/"judgments.jsonl")[0]["shadow_visibility"] == "unreviewed"


def test_unfinished_blind_judgments_cannot_reveal_model(tmp_path):
    module, pack, _, _, _ = release_setup(tmp_path)
    with pytest.raises(ValueError, match="Complete shadow"):
        module.reveal_review(pack, pack/"blind/judgments.jsonl", tmp_path/"reveal", "reviewer1")
    assert not (tmp_path/"reveal").exists()


def test_reveal_snapshots_blind_observations_and_keeps_owner_uncertainty(tmp_path):
    module, pack, _, _, _ = release_setup(tmp_path)
    judgments = complete_blind(pack)
    submitted_hash = digest(judgments)
    result = module.reveal_review(pack, judgments, tmp_path/"reveal", "reviewer1")
    assert result["revealed_cases"] == 1
    assert digest(tmp_path/"reveal/blind_judgments_submitted.jsonl") == submitted_hash
    row = read_rows(tmp_path/"reveal/overlay_judgments.jsonl")[0]
    assert row["owner_judgment"] == "unreviewed" and row["correct_caster_tokens"] == []
    assert (tmp_path/"reveal"/row["model_review"]).is_file()
    with pytest.raises(FileExistsError):
        module.reveal_review(pack, judgments, tmp_path/"reveal", "reviewer1")


def test_modified_task_reference_or_media_is_rejected(tmp_path):
    module, pack, _, _, _ = release_setup(tmp_path)
    judgments = complete_blind(pack)
    rows = read_rows(judgments)
    rows[0]["focus_frame_slot"] = 0
    write_rows(judgments, rows)
    with pytest.raises(ValueError, match="task references"):
        module.reveal_review(pack, judgments, tmp_path/"reveal", "r")
    complete_blind(pack)
    video = next((pack/"blind").glob("*.mp4"))
    video.write_bytes(video.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="video changed"):
        module.reveal_review(pack, judgments, tmp_path/"reveal", "r")


def test_showcase_exports_actual_saved_frames_without_changing_sources(tmp_path):
    module, _, _, after, pairs = release_setup(tmp_path)
    original = {path: digest(path) for path in after.rglob("*") if path.is_file()}
    result = module.export_showcase(after, [pairs], tmp_path/"showcase")
    assert result["clips"] == 1
    row = read_rows(tmp_path/"showcase/showcase_manifest.jsonl")[0]
    assert video_count(tmp_path/"showcase"/row["mainline_video"]) == 3
    assert video_count(tmp_path/"showcase"/row["clean_video"]) == 3
    assert row["quality_status"] == "unreviewed"
    assert any(item["observation_state"] == "retained_memory_only" for item in row["observation_timeline"])
    assert all(digest(path) == sha for path, sha in original.items())


def test_showcase_unknown_clip_is_not_silently_dropped(tmp_path):
    module, _, _, after, pairs = release_setup(tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        module.export_showcase(after, [pairs], tmp_path/"showcase", ["missing"])


def test_video_writer_preserves_irregular_camera_timing(tmp_path):
    module = script("prepare_shadow_video_release")
    image = np.full((31, 35, 3), 120, np.uint8)
    info = module.write_video(tmp_path/"timing.mp4", [image]*3, [0, 500000, 1500000], fps=4)
    assert info["encoded_frames"] == video_count(tmp_path/"timing.mp4") == 7
    with pytest.raises(ValueError, match="frame count"):
        module.write_video(tmp_path/"empty.mp4", [], [0])


def frozen_protocol():
    config = VideoShadowConfig()
    return {"stage": "full", "bev_role": "context", "discovery_mode": "cross-space", "discovery_domain": "receiver",
        "owner_scoring": "geometry+appearance+bev", "entity_suppression_enabled": True,
        "entity_motion_config": asdict(EntityMotionConfig()), "discovery_config": asdict(config),
        "physics_config": asdict(LabelConfig(minimum_area=config.minimum_area, minimum_change_pixels=1,
                                             allow_static_existence=True, allow_partial_temporal=True)),
        "owner_scoring_config": asdict(OwnerScoringConfig()), "correspondence_config": asdict(CorrespondenceConfig()),
        "m2_flow_config": asdict(FlowConfidenceConfig()), "cross_space_growth_config": asdict(CrossSpaceGrowthConfig()),
        "grid": asdict(BEVFlowConfig().grid()), "m2_correspondence": "anchor-first-reasoning-v1",
        "spatial_feature_sources": [{"checkpoint_sha256": "checkpoint", "model_config_sha256": "config",
            "history_mode": "current-only", "pc_range": [-40, -40, -5, 40, 40, 3],
            "shape_chw": [8, 80, 80], "coordinate_frame": "LIDAR_TOP", "layout": "C,H,W"}]}


def frozen_setup(tmp_path):
    _, after, _ = frozen_runs(tmp_path)
    protocol_path = after/"clip/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(frozen_protocol())
    write_json(protocol_path, protocol)
    module = script("run_frozen_shadow_video")
    lock_path = tmp_path/"config/freeze.json"
    module.freeze(after, [], lock_path)
    data = tmp_path/"new_data"
    data.mkdir()
    pairs = data/"pairs.jsonl"
    write_rows(pairs, [{"scene_name": "new-scene", "camera": "CAM_FRONT", "sample_token": f"s{slot}",
                        "next_sample_token": f"s{slot+1}"} for slot in range(2)])
    clips = data/"clips.jsonl"
    write_rows(clips, [{"id": "new-clip", "scene_name": "new-scene", "camera": "CAM_FRONT",
                        "start_sample_token": "s0", "track_id": "track", "num_frames": 3}])
    feature, spatial, flow = data/"feature", data/"spatial", data/"flow"
    for path in (feature, spatial, flow):
        path.mkdir()
    for slot in range(3):
        for root in (feature, spatial):
            write_json(root/f"s{slot}.json", protocol["spatial_feature_sources"][0])
    write_json(flow/"analysis_protocol.json", {"manifest_sha256": digest(pairs)})
    write_rows(flow/"pair_metrics.jsonl", [])
    batches = data/"batches.jsonl"
    write_rows(batches, [{"id": "batch", "pairs": "pairs.jsonl", "clips": "clips.jsonl",
                         "analysis_root": "flow", "feature_root": "feature", "spatial_feature_root": "spatial"}])
    return module, lock_path, batches, after


def test_freeze_captures_algorithm_configs_model_and_development_scenes(tmp_path):
    _, lock_path, _, _ = frozen_setup(tmp_path)
    lock = json.loads(lock_path.read_text())
    assert lock["excluded_development_scenes"] == ["scene"]
    assert "shadow_occ/annotation/video_correspondence.py" in lock["source_sha256"]
    assert lock["spatial_model_signatures"][0]["checkpoint_sha256"] == "checkpoint"
    assert lock["settings"]["m2_correspondence"] == "anchor-first-reasoning-v1"


def test_frozen_batch_runs_original_entry_and_verifies_output(tmp_path, monkeypatch):
    module, lock, batches, after = frozen_setup(tmp_path)
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output-root")+1])/"new-clip"
        output.mkdir(parents=True)
        protocol = json.loads((after/"clip/protocol.json").read_text())
        protocol["clip"] = read_rows(batches.parent/"clips.jsonl")[0]
        write_json(output/"protocol.json", protocol)
        write_rows(output/"frames.jsonl", read_rows(after/"clip/frames.jsonl"))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module.run_frozen(lock, batches, tmp_path/"run")
    assert result["batches"][0]["status"] == "completed"
    command = calls[0]
    assert command[2].endswith("discover_shadow_video.py")
    assert command[command.index("--seed-score")+1] == "0.35"
    assert not result["algorithm_settings_overridden"] and not result["training_performed"]


def test_frozen_batch_excludes_development_scene_before_running(tmp_path):
    module, lock, batches, _ = frozen_setup(tmp_path)
    clips_path = batches.parent/"clips.jsonl"
    rows = read_rows(clips_path)
    rows[0]["scene_name"] = "scene"
    write_rows(clips_path, rows)
    with pytest.raises(ValueError, match="overlaps development"):
        module.run_frozen(lock, batches, tmp_path/"run")
    assert not (tmp_path/"run").exists()


def test_frozen_batch_rejects_changed_algorithm_or_model(tmp_path):
    module, lock, batches, _ = frozen_setup(tmp_path)
    saved = json.loads(lock.read_text())
    corrupted = {**saved, "source_sha256": {}}
    write_json(lock, corrupted)
    with pytest.raises(ValueError, match="Algorithm source differs"):
        module.run_frozen(lock, batches, tmp_path/"run")
    write_json(lock, saved)
    meta_path = batches.parent/"spatial/s0.json"
    meta = json.loads(meta_path.read_text())
    meta["checkpoint_sha256"] = "different-model"
    write_json(meta_path, meta)
    with pytest.raises(ValueError, match="cache differs"):
        module.run_frozen(lock, batches, tmp_path/"run")


def test_freeze_does_not_replace_unsupported_nondefault_settings(tmp_path):
    module = script("run_frozen_shadow_video")
    protocol = frozen_protocol()
    protocol["correspondence_config"]["feature_weight"] = 99
    with pytest.raises(ValueError, match="frozen correspondence_config"):
        module.cli_options(module.settings(protocol))


def test_frozen_batch_records_failure_and_stops(tmp_path, monkeypatch):
    module, lock, batches, _ = frozen_setup(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=3))
    with pytest.raises(RuntimeError, match="failed: 3"):
        module.run_frozen(lock, batches, tmp_path/"run")
    protocol = json.loads((tmp_path/"run/frozen_run.json").read_text())
    assert protocol["batches"][0]["status"] == "failed"


def test_successful_process_with_different_settings_is_not_marked_verified(tmp_path, monkeypatch):
    module, lock, batches, after = frozen_setup(tmp_path)
    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output-root")+1])/"new-clip"
        output.mkdir(parents=True)
        protocol = json.loads((after/"clip/protocol.json").read_text())
        protocol["clip"] = read_rows(batches.parent/"clips.jsonl")[0]
        protocol["discovery_config"]["seed_score"] = .1
        write_json(output/"protocol.json", protocol)
        write_rows(output/"frames.jsonl", read_rows(after/"clip/frames.jsonl"))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="settings differ"):
        module.run_frozen(lock, batches, tmp_path/"run")
    record = json.loads((tmp_path/"run/frozen_run.json").read_text())["batches"][0]
    assert record["exit_code"] == 0 and record["status"] == "verification_failed"


def test_freeze_does_not_write_into_reference_outputs(tmp_path):
    module, _, _, after = frozen_setup(tmp_path)
    with pytest.raises(ValueError, match="outside frozen"):
        module.freeze(after, [], after/"freeze.json")
