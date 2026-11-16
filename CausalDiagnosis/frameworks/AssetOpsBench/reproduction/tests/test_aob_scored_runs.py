"""The reader of scored run folders that acceptance_check.py uses (tables.load_runs) and its statistics, on hand-made scores.csv
files: runs of one model pool, runs of two models, runs without scores and runs with and without one rows file are refused,
and the mean carries an interval from the bootstrap over scenarios."""
import json

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import score_runs as score
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import reproduction_tables as tables
from CausalDiagnosis.utils.csv_rows import write_rows

PORT = "agent_as_tool_paper"


def rows_of_one_run(run="run_a"):
    """Six episodes of the Agent-As-Tool port on three scenarios, two each."""
    rows = []
    for sid, family, kind in ((1, "IoT", "single"), (101, "FMSR", "single"), (601, "Multi-agent", "multi")):
        for repeat in (1, 2):
            rows.append({"run": run, "episode": f"{PORT}_s{sid:03d}_r{repeat}", "agent": PORT, "scenario_id": sid, "family": family,
                         "task_kind": kind, "repeat": repeat, "error": "", "steps": 4 if kind == "single" else 7,
                         "paper_task_completion_majority": "True" if sid == 1 else "False", "single_task_completion": 1.0 if sid == 1 else 0.0})
    return rows


def make_run(folder, model, rows, **changes):
    folder.mkdir()
    (folder / "run.json").write_text(json.dumps({"run": folder.name, "model": model, "repeats": 2, **changes}), encoding="utf-8")
    write_rows(folder / "scores.csv", rows, score.ROW_FIELDS)
    return folder


def test_runs_of_one_model_pool_and_the_statistics_read_them(tmp_path):
    first = make_run(tmp_path / "run_a", "granite-test", rows_of_one_run())
    second = make_run(tmp_path / "run_b", "granite-test", rows_of_one_run("run_b"))
    metas, rows = tables.load_runs([first, second])
    assert [m["run"] for m in metas] == ["run_a", "run_b"] and len(rows) == 12
    assert tables.scenario_count(rows) == 3 and len(tables.select(rows, PORT, kind="multi")) == 4
    steps = tables.mean_interval(tables.select(rows, kind="single"), "steps")
    assert (steps["estimate"], steps["low"], steps["high"]) == (4.0, 4.0, 4.0)
    done = tables.mean_interval(rows, tables.paper_era_field("agent_as_tool", "task_completion"))
    assert round(done["estimate"], 6) == round(1 / 3, 6) and done["low"] <= done["estimate"] <= done["high"]
    assert tables.paper_era_field("plan_execute", "task_completion") == "single_task_completion"
    assert tables.mean_interval(rows, "absent") is None


def test_runs_of_two_models_without_scores_or_with_mixed_telemetry_are_refused(tmp_path):
    plain = make_run(tmp_path / "run_a", "granite-test", rows_of_one_run())
    other = make_run(tmp_path / "run_b", "other-model", rows_of_one_run("run_b"))
    with pytest.raises(SystemExit, match="mix models"):
        tables.load_runs([plain, other])
    with pytest.raises(SystemExit, match="score_runs.py"):
        tables.load_runs([tmp_path])
    with_rows = make_run(tmp_path / "run_c", "granite-test", rows_of_one_run("run_c"), layer_b={"recovered_rows": {"sha256": "a" * 64}})
    assert tables.rows_files([json.loads((with_rows / "run.json").read_text())]) == {"a" * 12}
    with pytest.raises(SystemExit, match="mix telemetry"):
        tables.load_runs([plain, with_rows])
