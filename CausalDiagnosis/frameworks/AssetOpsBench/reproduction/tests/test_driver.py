"""The driver's refusals and parts without a model and without CouchDB: the pinned scenario file, a server that cannot
start, the time limit of a run, the launchers and the scenario sets, and the call log counts of AssetOpsBench's released
runners that the rig's tables_from_runs/ reads, on synthetic records. The paper-era ports play end to end in test_port.py
and test_pe_port.py."""
import asyncio
import json
from pathlib import Path

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_folder_names, scenario_sets
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_scenarios

STUB = Path(__file__).resolve().parent / "stub_server.py"
COUCH = "http://couch.test:5984"
# Fixture rows in the shape of the scenario file, worded for these tests.
ROWS = [{"id": 1, "type": "IoT", "text": "List the IoT sites.", "characteristic_form": "The answer names the site MAIN."},
        {"id": 3, "type": "IoT", "text": "Which assets are at site MAIN?", "characteristic_form": "The answer lists the assets of MAIN."}]


def write_scenarios(folder):
    path = folder / "scenarios.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    return path


@pytest.fixture
def scenario_file(tmp_path):
    return write_scenarios(tmp_path)


def test_a_scenario_file_other_than_the_pinned_one_is_refused(tmp_path, scenario_file, monkeypatch):
    monkeypatch.setenv("API_KEY", "stand-in-key")
    with pytest.raises(SystemExit, match="SHA-256"):
        run_scenarios.main(["--scenarios", str(scenario_file), "--ids", "1", "--model", "m", "--base-url", "http://127.0.0.1:9/v1",
                            "--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()
    # a server that cannot start stops the run before its folder exists
    broken = tmp_path / "broken_server.py"
    broken.write_text("raise SystemExit('this server cannot start')\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="did not start"):
        run_scenarios.main(["--scenarios", str(scenario_file), "--allow-unpinned", "--ids", "1", "--model", "m",
                            "--base-url", "http://127.0.0.1:9/v1", "--out", str(tmp_path / "runs"), "--no-reload", "--server", f"iot={broken}"])
    assert not (tmp_path / "runs").exists()


def test_a_run_past_its_time_limit_is_an_error_and_not_a_hang():
    class Stuck:
        async def run(self, question):
            await asyncio.sleep(30)

    result, error, seconds = run_scenarios.timed_run(Stuck(), "q", 0.2)
    assert result is None and error == "TimeoutError: the run exceeded 0.2 s" and seconds < 5


def test_launchers_set_the_server_settings_and_only_fmsr_gets_the_model_route(tmp_path):
    paths = run_scenarios.write_launchers(tmp_path, {"iot": None, "fmsr": None, "stub": str(STUB)}, {"COUCHDB_URL": COUCH},
                                          {"FMSR_MODEL_ID": "litellm_proxy/granite-test"})
    iot, fmsr, stub = (paths[name].read_text(encoding="utf-8") for name in ("iot", "fmsr", "stub"))
    assert "from servers.iot.main import main" in iot and COUCH in iot and "FMSR_MODEL_ID" not in iot
    assert "from servers.fmsr.main import main" in fmsr and "litellm_proxy/granite-test" in fmsr and COUCH in fmsr
    assert "runpy.run_path" in stub and str(STUB) in stub
    for text in (iot, fmsr, stub):
        compile(text, "launcher", "exec")
    assert run_scenarios.parse_servers([])["vibration"] is None and len(run_scenarios.parse_servers([])) == 6
    with pytest.raises(SystemExit):
        run_scenarios.parse_servers(["iot"])


def test_the_runnable_set_and_the_pilot():
    ids = scenario_sets.SETS["runnable"]
    assert len(ids) == len(set(ids)) == 120
    counts = {}
    for sid in ids:
        counts[scenario_sets.family_of(sid)] = counts.get(scenario_sets.family_of(sid), 0) + 1
    assert counts == {"IoT": 9, "FMSR": 20, "TSFM": 23, "Workorder": 36, "Multi-agent": 32}
    assert len(scenario_sets.PILOT) == 10 and set(scenario_sets.PILOT) <= set(ids)
    assert {scenario_sets.task_kind(i) for i in (1, 120, 215)} == {"single"} and scenario_sets.task_kind(601) == "multi"
    assert sum(scenario_sets.PINNED_COUNTS.values()) == 141 and run_folder_names.episode_name("plan_execute_paper", 9, 2) == "plan_execute_paper_s009_r2"
    with pytest.raises(ValueError):
        scenario_sets.family_of(350)


def test_only_the_two_ports_are_agents_of_the_reproduction():
    assert run_scenarios.parse_args(["--scenarios", "s", "--out", "o"]).agents == list(run_folder_names.PORTS)
    for released in ("plan_execute", "agent_as_tool"):
        with pytest.raises(SystemExit):
            run_scenarios.parse_args(["--scenarios", "s", "--out", "o", "--agents", released])
    assert not any(entry.startswith(("src/agent/agent_as_tool", "src/agent/openai_agent")) for entry in run_scenarios.CODE)
