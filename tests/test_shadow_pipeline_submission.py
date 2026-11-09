import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadow_occ.annotation.data import read_rows
from shadow_occ.utils.io import write_json
from tests.test_bev_flow_ablations import script
from tests.test_shadow_video_release import frozen_setup


def pipeline_fixture(tmp_path):
    _, lock, _, _ = frozen_setup(tmp_path)
    external = tmp_path/"external"
    external.mkdir()
    files = {}
    for name in ("model_config", "checkpoint", "train_ann_file", "val_ann_file"):
        path = external/(name+".test")
        path.write_bytes(b"test fixture, never model weights")
        files[name] = str(path)
    population = tmp_path/"population.json"
    write_json(population, {"official_train_scenes": [f"train-{i}" for i in range(8)],
        "official_eval_scenes": ["eval-1", "eval-2"]})
    config = tmp_path/"input.json"
    write_json(config, {**files, "freeze_lock": str(lock), "dataroot": str(tmp_path),
        "external_root": str(external), "population": str(population), "cache_root": str(tmp_path/"caches")})
    return script("submit_automatic_shadow_pipeline"), config


def test_dry_run_builds_the_new_mainline_dag_without_submitting_or_copying_raw_data(tmp_path, monkeypatch):
    module, config = pipeline_fixture(tmp_path)
    plan = module.prepare_pipeline(config, tmp_path/"run")
    assert plan["target_scenes"] == {"train": 2, "dev": 1, "eval": 2}
    assert not plan["raw_nuscenes_copied"]
    assert len(read_rows(plan["tasks"])) == 5
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry run submitted a job"))
    result = module.submit_jobs(plan)
    assert not result["jobs_submitted"]
    assert [row["name"] for row in result["jobs"]] == ["cache_cpu", "cache_bev", "plan", "shared", "data", "student"]
    assert result["jobs"][0]["after"] == []
    assert "--gres" not in result["jobs"][0]["args"]
    assert "--gres" in result["jobs"][1]["args"]
    assert result["jobs"][3]["after"] == ["plan"]
    assert result["jobs"][5]["after"] == ["data"]
    assert all("train_shadow_vlm" not in str(row) for row in result["jobs"])


def test_real_submission_wires_afterok_job_ids_and_removes_legacy_training_overrides(tmp_path, monkeypatch):
    module, config = pipeline_fixture(tmp_path)
    plan = module.prepare_pipeline(config, tmp_path/"run")
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        assert "SHADOW_EVAL_GT" not in kwargs["env"]
        assert "SHADOW_SEG_INIT_CHECKPOINT" not in kwargs["env"]
        return SimpleNamespace(stdout=f"{1000+len(calls)};cluster\n")
    monkeypatch.setenv("SHADOW_EVAL_GT", "old-manual-test.jsonl")
    monkeypatch.setenv("SHADOW_SEG_INIT_CHECKPOINT", "old-shadow-model.pth")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module.submit_jobs(plan, submit=True)
    assert result["jobs_submitted"] and len(calls) == 6
    assert calls[1][calls[1].index("--dependency")+1] == "afterok:1001"
    assert calls[-1][calls[-1].index("--dependency")+1] == "afterok:1005"
    saved = (Path(plan["output_root"])/"jobs.json").read_bytes()
    for submit in (False, True):
        with pytest.raises(ValueError, match="already has submitted"):
            module.submit_jobs(plan, submit=submit)
    assert (Path(plan["output_root"])/"jobs.json").read_bytes() == saved


def test_pipeline_rejects_overlapping_populations_and_duplicate_methods(tmp_path):
    module, config = pipeline_fixture(tmp_path)
    values = json.loads(config.read_text())
    values["methods"] = ["anchor_first_v8plus", "anchor_first_v8plus"]
    write_json(config, values)
    with pytest.raises(ValueError, match="Invalid automatic"):
        module.prepare_pipeline(config, tmp_path/"run")
    values.pop("methods")
    write_json(config, values)
    write_json(Path(values["population"]), {"official_train_scenes": ["same", "other"], "official_eval_scenes": ["same"]})
    with pytest.raises(ValueError, match="disjoint"):
        module.prepare_pipeline(config, tmp_path/"run")


def test_scaling_requires_a_completed_prior_stage(tmp_path):
    module, config = pipeline_fixture(tmp_path)
    with pytest.raises(ValueError, match="previous_run"):
        module.prepare_pipeline(config, tmp_path/"bad50", fraction=50)
    first = module.prepare_pipeline(config, tmp_path/"run25")
    values = json.loads(config.read_text())
    values["previous_run"] = str(tmp_path/"run25/pipeline.json")
    write_json(config, values)
    with pytest.raises(ValueError, match="annotations are not complete"):
        module.prepare_pipeline(config, tmp_path/"bad50", fraction=50)
    for stage in ("shared", "stage_025"):
        write_json(Path(first["config"]["annotation_root"])/stage/"frozen_run.json", {"batches": [{"status": "completed"}]})
    write_json(tmp_path/"run25/student/suite_status.json", {"all_arms_completed": True, "metrics_exit_code": 0})
    second = module.prepare_pipeline(config, tmp_path/"run50", fraction=50)
    assert second["reuse_shared_annotations"]
    assert second["config"]["annotation_root"] == first["config"]["annotation_root"]
    assert "shared" not in [row["name"] for row in module.job_specs(second)]


def test_cache_phase_reuses_completed_outputs_but_not_unstamped_or_changed_inputs(tmp_path, monkeypatch):
    module = script("submit_automatic_shadow_pipeline")
    protocol, payload = tmp_path/"protocol.json", tmp_path/"payload.npz"
    stamp = tmp_path/"complete.json"
    calls = []
    def fake_run(command, repo):
        calls.append(command)
        write_json(protocol, {"test": "synthetic only"})
        payload.write_bytes(b"synthetic payload")
    monkeypatch.setattr(module, "run_command", fake_run)
    artifacts = lambda: ([protocol], [payload])
    module.cached_phase(stamp, "same-request", [(["not-executed"], protocol)], artifacts, tmp_path)
    module.cached_phase(stamp, "same-request", [(["not-executed"], protocol)], artifacts, tmp_path)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="protocol changed"):
        module.cached_phase(stamp, "different", [], artifacts, tmp_path)
    payload.unlink()
    with pytest.raises(ValueError, match="payloads are missing"):
        module.cached_phase(stamp, "same-request", [], artifacts, tmp_path)


def test_cache_workers_use_canonical_pair_export_dis_and_distinct_bev_history_modes(tmp_path, monkeypatch):
    module, config = pipeline_fixture(tmp_path)
    plan = module.prepare_pipeline(config, tmp_path/"run")
    task = read_rows(plan["tasks"])[0]
    pairs = Path(task["pairs"])
    pairs.parent.mkdir(parents=True)
    pairs.write_text("", encoding="utf-8")
    phases = []
    monkeypatch.setattr(module, "cached_phase", lambda stamp, request, commands, artifacts, repo: phases.append(commands))
    job = Path(plan["output_root"])/"pipeline.json"
    module.run_worker(job, "cpu")
    module.run_worker(job, "bev")
    pair_command, flow_command = [row[0] for row in phases[0]]
    assert pair_command[pair_command.index("--scene")+1] == task["scene_name"]
    assert pair_command[pair_command.index("--max-pairs-per-scene")+1] == "24"
    assert flow_command[flow_command.index("--flow-backend")+1] == "dis"
    assert [row[0][row[0].index("--history-mode")+1] for row in phases[1]] == ["temporal", "current-only"]
    assert all(row[0][row[0].index("--ann-file")+1] == plan["config"]["val_ann_file"] for row in phases[1])
