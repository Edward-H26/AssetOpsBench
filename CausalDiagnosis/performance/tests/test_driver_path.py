"""The path that ships: a run folder written by the tree's own driver with the scripted agent, no model, read by the registered
command of the command table without an error, every check reported for its episode and no file of the run folder changed. The
test asserts relations between the quantities of the profile, never a recorded value."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.utils.paths import ROOT, SRC

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.performance import profile_table, stage_profile  # noqa: E402


def test_a_scripted_run_of_the_driver_is_profiled_through_the_registered_command(tmp_path):
    run_episodes.main(["--config", "static", "--agent", "scripted", "--seeds", "1", "--out", str(tmp_path / "runs")])
    (run,) = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(SRC))), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-m", "CausalDiagnosis", "performance", "profile-table", str(run), "--out", str(tmp_path / "out")],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    with (tmp_path / "out" / profile_table.EPISODES_FILE).open(newline="", encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    assert row["left_out"] == "" and float(row["total_s"]) > 0
    assert all(row[name].isdigit() for name in stage_profile.CHECKS)  # every check is reported, whatever it finds
    # the parts that the rest leaves add up to the total, and a counted call's service lies inside the client time of the calls
    parts = ("creation_s", "end_s", "model_s", "env_tools_client_s", "diagnostics_client_s", "server_starts_s", "rest_s")
    assert sum(float(row[name]) for name in parts) == pytest.approx(float(row["total_s"]))
    assert int(row["counted_calls"]) > 0 and float(row["tool_service_s"]) <= float(row["env_tools_client_s"])
    # the scripted wiring check starts a server process for every call, so its starts lie outside the opening of the episode
    assert int(row["late_server_starts"]) > 0
    summary = json.loads((tmp_path / "out" / profile_table.CELLS_FILE).read_text(encoding="utf-8"))
    assert set(summary["cells"][0]["checks"]) == set(stage_profile.CHECKS) and summary["jobs"] == []
    asset = json.loads((run / "run.json").read_text(encoding="utf-8"))["asset_id"]  # the asset that the driver recorded for the run
    assert row["asset"] == asset and f"cell 1: asset {asset}, agent scripted" in done.stdout
    assert done.stdout == profile_table.render(summary["cells"], summary["jobs"], summary["engine"]) + "\n"  # every printed line
    assert {path: path.read_bytes() for path in run.rglob("*") if path.is_file()} == before
