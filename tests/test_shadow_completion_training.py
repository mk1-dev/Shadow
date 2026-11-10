import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadow_occ.annotation.data import write_rows, read_rows
from tests.test_bev_flow_ablations import script


def source_fixture(tmp_path):
    source = tmp_path/"source"
    batches, runs = [], []
    for role in ("train", "dev", "eval"):
        for index in range(2):
            identity = role+str(index)
            reference = tmp_path/"reference"/identity
            reference.mkdir(parents=True)
            (reference/"summary.json").write_text('{}')
            pairs = source/identity/"pairs.jsonl"
            clips = source/identity/"clips.jsonl"
            write_rows(pairs, [{"id": identity}])
            write_rows(clips, [{"id": identity+camera, "scene_name": identity, "camera": camera}
                               for camera in ("front", "left", "right")])
            batches.append({"id": identity, "split": role, "method": "anchor_first_v8plus",
                            "pairs": str(pairs), "clips": str(clips),
                            "analysis_root": str(source/identity/"flow"),
                            "feature_root": str(source/"context"), "spatial_feature_root": str(source/"spatial")})
            runs.append({"split": role, "method": "anchor_first_v8plus", "pairs": [str(pairs)], "run_root": str(reference)})
    write_rows(source/"stage_025/batches.jsonl", [row for row in batches if row["split"] == "train"])
    write_rows(source/"shared/batches.jsonl", [row for row in batches if row["split"] != "train"])
    write_rows(source/"stage_025/experiment_runs.jsonl", runs)
    return source


def test_plan_expands_isolated_scenes_and_camera_coverage_without_label_selection(tmp_path):
    module = script("run_shadow_completion_training")
    source = source_fixture(tmp_path)
    output = tmp_path/"new"
    plan = module.prepare(source, output, {"train": 1, "dev": 1, "eval": 1})
    assert plan["clips"] == 9
    assert plan["available_cached_scenes"] == {"train": 2, "dev": 2, "eval": 2}
    assert len(set(sum(plan["scene_sets"].values(), []))) == 3
    for row in read_rows(output/"batches.jsonl"):
        assert len({clip["camera"] for clip in read_rows(row["clips"])}) == 3
    second = module.prepare(source, tmp_path/"second", {"train": 1, "dev": 1, "eval": 1})
    assert plan["scene_sets"] == second["scene_sets"]


def test_shortage_is_reported_not_filled_with_repeated_scenes(tmp_path):
    module = script("run_shadow_completion_training")
    with pytest.raises(ValueError, match="requested 3"):
        module.prepare(source_fixture(tmp_path), tmp_path/"new", {"train": 3, "dev": 1, "eval": 1})


def test_expansion_keeps_heldout_scenes_and_exact_clips_fixed(tmp_path):
    module = script("run_shadow_completion_training")
    source = source_fixture(tmp_path)
    old = tmp_path/"old"
    previous = module.prepare(source, old, {"train": 1, "dev": 1, "eval": 1}, clips_per_scene=1)
    new = tmp_path/"expanded"
    expanded = module.prepare(source, new, {"train": 2, "dev": 1, "eval": 1}, clips_per_scene=3,
        heldout_plan=old/"plan.json", negative_maximum_cells=256, negative_maximum_fraction=.25)
    assert len(expanded["scene_sets"]["train"]) == 2
    for role in ("dev", "eval"):
        assert expanded["scene_sets"][role] == previous["scene_sets"][role]
        old_batch = next(row for row in read_rows(old/"batches.jsonl") if row["split"] == role)
        new_batch = next(row for row in read_rows(new/"batches.jsonl") if row["split"] == role)
        assert read_rows(old_batch["clips"]) == read_rows(new_batch["clips"])


def test_clip_selection_prefers_distinct_caster_tracks():
    module = script("run_shadow_completion_training")
    clips = [{"camera": "front", "track_id": track, "id": str(index)}
             for index, track in enumerate(["a", "a", "a", "b", "c"])]
    assert len({row["track_id"] for row in module.diverse_clips(clips, 3, 0)}) == 3


def test_changed_plan_inputs_cannot_launch_training(tmp_path, monkeypatch):
    module = script("run_shadow_completion_training")
    root = tmp_path/"new"
    module.prepare(source_fixture(tmp_path), root, {"train": 1, "dev": 1, "eval": 1})
    with (root/"pairs.jsonl").open("a") as handle:
        handle.write('{}\n')
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Changed inputs launched a subprocess"))
    with pytest.raises(ValueError, match="changed"):
        module.run(root/"plan.json", tmp_path/"assets")


def test_parallel_annotation_resumes_complete_scenes_and_preserves_partial_outputs(tmp_path, monkeypatch):
    module = script("run_shadow_completion_training")
    root = tmp_path/"new"
    module.prepare(source_fixture(tmp_path), root, {"train": 1, "dev": 1, "eval": 1})
    first = Path(read_rows(root/"batches.jsonl")[0]["output"])
    first.mkdir(parents=True)
    (first/"old_partial.txt").write_text('retained')
    calls = []
    def process(command, **kwargs):
        assert Path(command[2]).name == "discover_shadow_video.py"
        clips = read_rows(command[command.index("--clips")+1])
        output = Path(command[command.index("--output-root")+1])
        output.mkdir(parents=True)
        for clip in clips:
            folder = output/clip["id"]
            folder.mkdir()
            (folder/"protocol.json").write_text(json.dumps({"clip": clip, "protocol": "event-anchored-region-shadow-completion-v9.3"}))
            write_rows(folder/"frames.jsonl", [{"frame_token": clip["id"], "timestamp": 1}])
        (output/"summary.json").write_text('{}')
        calls.append(output)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", process)
    module.run(root/"plan.json", tmp_path/"assets", stages=("annotation",), resume=True, workers=2)
    assert len(calls) == 3
    assert len(list(first.parent.glob(first.name+".partial_*/old_partial.txt"))) == 1
    module.run(root/"plan.json", tmp_path/"assets", stages=("annotation",), resume=True, workers=2)
    assert len(calls) == 3


def test_dataset_export_restricts_baseline_to_the_same_selected_windows(tmp_path):
    from tests.test_shadow_mainline_experiments import data_fixture
    runs, negatives = data_fixture(tmp_path)
    rows = read_rows(runs)
    for row in rows:
        row["clip_ids"] = ["clip"]
    write_rows(runs, rows)
    module = script("build_shadow_mainline_experiments")
    summary = module.build_dataset(runs, tmp_path/"selected", [negatives], size=(80, 80))
    assert summary["methods"]
    rows[0]["clip_ids"] = ["does_not_exist"]
    write_rows(runs, rows)
    with pytest.raises(ValueError, match="comparison clips"):
        module.build_dataset(runs, tmp_path/"missing", [negatives], size=(80, 80))
