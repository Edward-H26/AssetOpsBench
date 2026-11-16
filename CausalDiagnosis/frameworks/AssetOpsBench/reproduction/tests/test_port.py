"""The Agent-As-Tool port in the reproduction: run_scenarios plays it beside the Plan-Execute port, with a scripted
stand-in for SGLang's completion routes and the stand-in IoT server, and records the supervisor's ReAct counter as its
steps, the step rule of the paper's Table 3. Then the acceptance rule of acceptance_check.py on synthetic rows."""
from CausalDiagnosis.utils.paths import PACKAGE
import json
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import acceptance_check as acceptance
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_folder_names, scenario_sets
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_scenarios

STUB = Path(__file__).resolve().parent / "stub_server.py"
PORT_TESTS = PACKAGE / "frameworks" / "AssetOpsBench" / "agents" / "Agent_As_Tool_Paper" / "tests"
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402

ROWS = [{"id": 1, "type": "IoT", "text": "List the IoT sites.", "characteristic_form": "The answer names the site MAIN."}]


def test_the_port_is_an_agent_of_the_reproduction_that_every_run_plays_unless_another_is_named():
    assert run_scenarios.parse_args(["--scenarios", "s", "--out", "o"]).agents == ["agent_as_tool_paper", "plan_execute_paper"]
    assert run_scenarios.parse_args(["--scenarios", "s", "--out", "o", "--agents", run_folder_names.PORT]).agents == ["agent_as_tool_paper"]


def test_the_port_plays_a_scenario_and_records_the_supervisors_counter_as_its_steps(tmp_path, monkeypatch):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    script = [*step("I need the list of sites.", "IoTAgent", "request=list all the sites"),
              *step("I need to call the sites tool.", "sites", "{}"), *finish("The sites are MAIN"), REVIEW_OK,
              *finish("answer=The sites are MAIN"), REVIEW_OK]
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    with ScriptedEndpoint(script, model="granite-test") as ep:
        run_dir = run_scenarios.main(["--scenarios", str(scenarios), "--allow-unpinned", "--ids", "1", "--repeats", "1", "--agents",
                                      run_folder_names.PORT, "--model", "granite-test", "--base-url", ep.base_url, "--out", str(tmp_path / "runs"),
                                      "--no-reload", "--server", f"iot={STUB}"])
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["agents"] == ["agent_as_tool_paper"] and meta["agent_as_tool_paper"]["sub_agents"] == ["IoTAgent", "RuleLogicAgent", "jsonreader"]
    episode = json.loads((run_dir / "agent_as_tool_paper_s001_r1" / "episode.json").read_text())
    assert episode["error"] is None and episode["answer"] == "answer=The sites are MAIN"
    assert episode["steps"] == episode["react_steps"] == 3 and episode["tool_calls"] == 1 and episode["plan_steps"] is None
    assert episode["model_calls"] == len(script) == len(ep.completions())
    record = json.loads((run_dir / "agent_as_tool_paper_s001_r1" / episode["trajectory_record"]).read_text())
    assert record["runner"] == "agent-as-tool-paper" and record["answer"] == episode["answer"]
    calls = json.loads((run_dir / "agent_as_tool_paper_s001_r1" / "llm_calls.json").read_text())
    assert [c["kind"] for c in calls][:2] == ["thought", "action"] and calls[0]["agent"] == "supervisor"


def registered(single, multi, accomplished, **changes):
    """(scored rows of the port, the run.json records of their runs): one row per scenario of the runnable set, from one
    finished run with one run per scenario. The steps of each task kind repeat the given values in order, and the first
    ``accomplished`` scenarios carry a task completion by the majority of five judge calls."""
    rows, seen = [], {"single": 0, "multi": 0}
    for n, sid in enumerate(scenario_sets.SETS["runnable"]):
        kind = scenario_sets.task_kind(sid)
        values = single if kind == "single" else multi
        rows.append({"run": "run", "episode": run_folder_names.episode_name(run_folder_names.PORT, sid, 1), "agent": run_folder_names.PORT,
                     "scenario_id": str(sid), "task_kind": kind, "family": scenario_sets.family_of(sid), "error": "",
                     "steps": str(values[seen[kind] % len(values)]), "paper_task_completion_majority": str(n < accomplished),
                     "paper_data_retrieval_accuracy_majority": "False", "paper_generalized_result_verification_majority": "False"})
        seen[kind] += 1
    meta = {"run": "run", "set": "runnable", "repeats": 1, "episodes_planned": len(rows), "episodes_done": len(rows),
            "scenario_ids": [int(r["scenario_id"]) for r in rows], "ended": "2020-07-01T00:00:00", **changes}
    return rows, [meta]


def test_the_acceptance_rule_holds_when_the_counter_meets_the_paper_and_the_bracket_holds_35():
    verdict = acceptance.check(*registered(single=(5, 6), multi=(6, 7), accomplished=40))
    assert verdict["port"] == run_folder_names.PORT and verdict["decidable"] and verdict["pass"] is True
    assert verdict["steps"]["single"]["pass"] and verdict["steps"]["multi"]["pass"]
    # with 40 accomplished among the 120 scenarios run, the bracket over the paper's 141 runs from 40 to 61 accomplished
    low, high = verdict["task_completion"]["low"], verdict["task_completion"]["high"]
    assert (round(low, 2), round(high, 2)) == (28.37, 43.26) and verdict["task_completion"]["pass"] is True
    # the counter on the paper with a task completion whose bracket lies above 35 misses on the judge alone
    verdict = acceptance.check(*registered(single=(5, 6), multi=(6, 7), accomplished=60))
    assert verdict["steps"]["single"]["pass"] and verdict["steps"]["multi"]["pass"]
    assert verdict["task_completion"]["low"] > 35 and verdict["task_completion"]["pass"] is False and verdict["pass"] is False


def test_the_acceptance_rule_fails_on_one_step_runs():
    verdict = acceptance.check(*registered(single=(2,), multi=(2,), accomplished=40))
    assert verdict["decidable"] and not verdict["steps"]["single"]["pass"] and not verdict["steps"]["multi"]["pass"]
    assert verdict["task_completion"]["pass"] is True and verdict["pass"] is False


def test_the_acceptance_rule_decides_nothing_on_a_pilot_however_its_values_read():
    rows, metas = registered(single=(5, 6), multi=(6, 7), accomplished=40, set="pilot")
    pilot = [r for r in rows if int(r["scenario_id"]) in scenario_sets.PILOT]
    verdict = acceptance.check(pilot, metas)
    assert len(pilot) == 10 and verdict["pass"] is None and not verdict["decidable"] and len(verdict["reasons"]) == 2
    # the bracket of ten scenarios over the paper's 141 holds 35 whatever the port did, which is why a pilot cannot decide
    assert verdict["task_completion"]["low"] <= 35 <= verdict["task_completion"]["high"]


def test_the_acceptance_rule_decides_nothing_on_an_agent_as_tool_run_that_was_not_judged():
    """A run scored without the judge has no verdict of the paper protocol, which is no observation of the port, so the runs
    cannot decide and the reason names the unjudged episodes, where a miss would blame the port."""
    rows, metas = registered(single=(5, 6), multi=(6, 7), accomplished=40)
    for row in rows:
        row["paper_task_completion_majority"] = ""
    verdict = acceptance.check(rows, metas)
    assert verdict["pass"] is None and not verdict["decidable"]
    assert verdict["reasons"] == [f"{len(rows)} episode(s) of the port hold no judge verdict under the paper protocol: "
                                  + acceptance.listed(sorted(r["episode"] for r in rows))]
