import importlib.util
from pathlib import Path
import sys

from shadow_occ.annotation.data import digest, write_rows
from shadow_occ.utils.io import write_json


def test_replay_shares_exact_windows_without_export_or_analysis(tmp_path, monkeypatch):
    script = Path(__file__).parents[1]/"scripts/rebuild_bev_flow_labels.py"
    spec = importlib.util.spec_from_file_location("rebuild_shadow_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path/"source", tmp_path/"output"
    protocol = {"mature_object_mode": "required",
        "config": {"extent_m": 40, "resolution_m": .5, "light_threshold_L": 6}}
    for split in ("train", "val"):
        origin = source/split
        for name in ("pairs", "flow", "mature_bev", "bev_flow"):
            (origin/name).mkdir(parents=True)
        write_rows(origin/"pairs/pairs.jsonl", [])
        write_json(origin/"flow/analysis_protocol.json", {})
        write_json(origin/"bev_flow/bev_flow_protocol.json", protocol)
        write_rows(origin/"bev_flow/windows.jsonl", [{"id": split}])
        write_json(origin/"bev_flow/summary.json", {"windows": 1})
    before = {split: digest(source/split/"bev_flow/windows.jsonl") for split in ("train", "val")}
    commands = []

    class Process:
        def __init__(self, command, cwd):
            commands.append(command)
            options = {command[index]: command[index+1] for index in range(2, len(command)-1)
                       if command[index].startswith("--") and not command[index+1].startswith("--")}
            root = Path(options["--output-root"])
            root.mkdir(parents=True)
            if Path(command[1]).name == "build_physics_shadow_labels.py":
                write_rows(root/"canonical.jsonl", [])
                write_json(root/"label_protocol.json", {"receiver_competition": "on",
                    **{name: float(options["--"+name.replace("_", "-")]) for name in
                       ("control_shift_m", "minimum_projection_gain", "minimum_explanation_margin", "appearance_prior_weight")}})
            write_json(root/"summary.json", {})

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(sys, "argv", ["rebuild", "--source-root", str(source), "--output-root", str(output),
        "--reuse-source-analysis", "--audit-only"])
    module.main()
    assert len(commands) == 4
    assert {Path(command[1]).name for command in commands} == {
        "build_physics_shadow_labels.py", "analyze_receiver_competition.py"}
    for split in ("train", "val"):
        assert (output/split/"bev_flow").is_symlink()
        assert digest(output/split/"bev_flow/windows.jsonl") == before[split]
        assert digest(source/split/"bev_flow/windows.jsonl") == before[split]
        assert (output/split/"labels/label_protocol.json").is_file()
