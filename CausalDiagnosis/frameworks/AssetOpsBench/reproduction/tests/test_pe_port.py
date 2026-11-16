"""The Plan-Execute port in the reproduction, and the acceptance rule of both ports. run_scenarios plays the port beside
the Agent-As-Tool port, with a scripted stand-in for SGLang's completion routes and the stand-in IoT server, and records
its executed plan steps as its steps. The acceptance rule decides only on finished runs of a registered set that hold
every scenario once with every episode counted, its steps check is an equivalence, and a run of either port without a
result keeps its step count."""
from CausalDiagnosis.utils.paths import PACKAGE
import asyncio
import json
import sys
from pathlib import Path

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import acceptance_check as acceptance
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_folder_names, scenario_file, scenario_sets
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_scenarios
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import score_runs as score
from CausalDiagnosis.utils.csv_rows import write_rows

STUB = Path(__file__).resolve().parent / "stub_server.py"
PORT_TESTS = PACKAGE / "frameworks" / "AssetOpsBench" / "agents" / "Agent_As_Tool_Paper" / "tests"
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402

PE_PORT, PORT = run_folder_names.PE_PORT, run_folder_names.PORT
ROWS = [{"id": 1, "type": "IoT", "text": "List the IoT sites.", "characteristic_form": "The answer names the site MAIN."}]
PLAN = "## Step 1\n#Task1: List the IoT sites.\n#Agent1: IoT Data Download\n#Dependency1: None\n#ExpectedOutput1: the sites\n"
VALID = '{"status": "Valid", "reasoning": "The plan covers the question.", "suggestions": "None."}\n(END OF RESPONSE)'
RUNNABLE = scenario_sets.SETS["runnable"]


def test_the_plan_execute_port_is_an_agent_of_the_reproduction_that_every_run_plays_unless_another_is_named():
    assert PE_PORT == "plan_execute_paper" and run_folder_names.PORTS == ("agent_as_tool_paper", "plan_execute_paper")
    assert PE_PORT in run_scenarios.parse_args(["--scenarios", "s", "--out", "o"]).agents
    assert run_scenarios.parse_args(["--scenarios", "s", "--out", "o", "--agents", PE_PORT]).agents == ["plan_execute_paper"]
    assert "CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper" in run_scenarios.CODE  # the identity of a run hashes the package


def test_the_plan_execute_port_plays_a_scenario_and_records_its_executed_plan_steps_as_its_steps(tmp_path, monkeypatch):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    script = [PLAN, VALID, *step("I need the sites.", "sites", "{}"), *finish("The sites are MAIN"), REVIEW_OK, "MAIN"]
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    with ScriptedEndpoint(script, model="granite-test") as ep:
        run_dir = run_scenarios.main(["--scenarios", str(scenarios), "--allow-unpinned", "--ids", "1", "--repeats", "1", "--agents",
                                      PE_PORT, "--model", "granite-test", "--base-url", ep.base_url, "--out", str(tmp_path / "runs"),
                                      "--no-reload", "--server", f"iot={STUB}"])
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["agents"] == ["plan_execute_paper"] and meta["plan_execute_paper"]["agents"] == ["IoT Data Download"]
    assert meta["agent_as_tool_paper"] is None
    folder = run_dir / "plan_execute_paper_s001_r1"
    episode = json.loads((folder / "episode.json").read_text())
    assert episode["error"] is None and episode["answer"] == "MAIN"
    # one plan step ran, and the summarization step is no step
    assert episode["steps"] == 1 and episode["plan_steps"] == 1 and episode["tool_calls"] == 1 and episode["react_steps"] is None
    assert episode["model_calls"] == len(script) == len(ep.completions())
    record = json.loads((folder / episode["trajectory_record"]).read_text())
    assert record["runner"] == "plan-execute-paper" and record["answer"] == "MAIN"
    flow = json.loads((folder / "plan_execute_trajectory.json").read_text())
    assert flow["plan"] == PLAN and [h["agent_name"] for h in flow["history"]] == ["IoT Data Download", "SummarizationAgent"]
    assert flow["executed_steps"] == 1 and flow["plan_steps"] == 1 and flow["end"] == "finish" and len(flow["plans"]) == 1
    calls = json.loads((folder / "llm_calls.json").read_text())
    assert [c["agent"] for c in calls][:2] == ["planner", "reviewer"] and calls[-1]["agent"] == "summarizer"


# ----- the acceptance rule -----

def registered_rows(port, single=(5, 6), multi=(5, 5), accomplished=0, run="run"):
    """Scored rows of a port, one per scenario of the runnable set. The steps of each task kind repeat the given values in
    order, and the first ``accomplished`` scenarios carry a task completion under both judge rules."""
    rows, seen = [], {"single": 0, "multi": 0}
    for n, sid in enumerate(RUNNABLE):
        kind = scenario_sets.task_kind(sid)
        values = single if kind == "single" else multi
        done = n < accomplished
        rows.append({"run": run, "episode": run_folder_names.episode_name(port, sid, 1), "agent": port, "scenario_id": str(sid),
                     "family": scenario_sets.family_of(sid), "task_kind": kind, "repeat": "1", "error": "",
                     "steps": str(values[seen[kind] % len(values)]), "plan_steps": "4",
                     "paper_task_completion_majority": str(done), "paper_data_retrieval_accuracy_majority": "False",
                     "paper_generalized_result_verification_majority": "False", "single_task_completion": str(float(done)),
                     "single_data_retrieval_accuracy": "0.0", "single_generalized_result_verification": "0.0"})
        seen[kind] += 1
    return rows


def run_record(rows, run="run", **changes):
    """run.json of a finished run of the runnable set with one run per scenario that planned the scenarios of the rows."""
    return {"run": run, "model": "granite-test", "set": "runnable", "repeats": 1, "episodes_planned": len(rows), "episodes_done": len(rows),
            "scenario_ids": sorted({int(r["scenario_id"]) for r in rows}), "ended": "2020-07-01T00:00:00", **changes}


def verdict_of(rows, port=PE_PORT, **changes):
    return acceptance.check(rows, [run_record(rows, **changes)], port)


def test_the_runnable_set_holds_both_task_kinds():
    kinds = [scenario_sets.task_kind(sid) for sid in RUNNABLE]
    assert (len(RUNNABLE), kinds.count("single"), kinds.count("multi")) == (120, 88, 32)


def test_the_plan_execute_acceptance_is_the_steps_check_alone_against_its_own_paper_values():
    assert acceptance.paper_steps(PE_PORT) == {"single": 5.2, "multi": 5.1}
    assert acceptance.paper_steps(PORT) == {"single": 5.3, "multi": 6.6}
    # steps on the paper and no task accomplished: the verdict passes, since no judge value decides
    verdict = verdict_of(registered_rows(PE_PORT))
    assert verdict["decidable"] and verdict["reasons"] == [] and verdict["pass"] is True
    assert verdict["steps"]["single"]["pass"] and verdict["steps"]["multi"]["pass"]
    assert all(verdict[field]["pass"] is None and "low" not in verdict[field] for field in acceptance.JUDGED)
    assert verdict["task_completion"]["paper"] == "22/84 (26.19)" and verdict["task_completion"]["rate"] == 0.0
    assert verdict["plan_steps"]["mean"] == 4.0 and (verdict["episodes"], verdict["scenarios"], verdict["errors"]) == (120, 120, 0)
    # the printed rate is the one judge call at temperature 0, the paper-era rule of this design, and not the majority of five
    mixed = registered_rows(PE_PORT)
    for row in mixed[:30]:
        row["single_task_completion"] = "1.0"
    assert verdict_of(mixed)["task_completion"]["rate"] == 25.0
    # every task accomplished in two steps: the verdict misses
    short = verdict_of(registered_rows(PE_PORT, single=(2,), multi=(2,), accomplished=120))
    assert short["decidable"] and short["pass"] is False and not short["steps"]["single"]["pass"] and not short["steps"]["multi"]["pass"]
    # the rows of one port are not the rows of the other
    other = verdict_of(registered_rows(PE_PORT), port=PORT)
    assert other["episodes"] == 0 and other["pass"] is None and other["reasons"] == ["the runs hold no episode of agent_as_tool_paper"]


def test_the_steps_check_is_an_equivalence_so_a_wide_interval_misses():
    # the multi-agent steps lie four below and four above the paper's 5.1 in turn: their mean is on the paper, and the
    # interval reaches more than one step to either side, so the port is not shown to be within one step
    spread = verdict_of(registered_rows(PE_PORT, multi=(1, 9)))
    multi = spread["steps"]["multi"]
    assert multi["mean"] == 5.0 and multi["low"] < 5.1 - acceptance.STEP_TOLERANCE and multi["high"] > 5.1 + acceptance.STEP_TOLERANCE
    assert multi["pass"] is False and spread["steps"]["single"]["pass"] is True and spread["pass"] is False
    # a narrow interval that leaves the margin on one side misses as well
    high = verdict_of(registered_rows(PE_PORT, multi=(6, 7)))
    assert high["steps"]["multi"]["low"] > 5.1 and high["steps"]["multi"]["high"] > 5.1 + acceptance.STEP_TOLERANCE
    assert high["steps"]["multi"]["pass"] is False and high["pass"] is False


def test_one_function_decides_the_steps_check_and_the_check_reports_the_half_width_and_the_difference(monkeypatch):
    paper, margin = 5.1, acceptance.STEP_TOLERANCE
    # an interval on the edges of the margin passes, one that leaves it on either side misses, and no interval misses
    assert acceptance.steps_meet_paper({"low": paper - margin, "high": paper + margin}, paper)
    assert not acceptance.steps_meet_paper({"low": paper - margin - 0.01, "high": paper}, paper)
    assert not acceptance.steps_meet_paper({"low": paper, "high": paper + margin + 0.01}, paper)
    assert not acceptance.steps_meet_paper(None, paper)
    # beside the verdict of each task kind stand the half-width of its interval and the mean minus the paper's value
    rows = registered_rows(PE_PORT, multi=(1, 9))
    multi = verdict_of(rows)["steps"]["multi"]
    assert multi["half_width"] == (multi["high"] - multi["low"]) / 2 > margin and multi["difference"] == multi["mean"] - paper
    assert multi["pass"] is False
    # the steps check reads that function alone: under a rule that passes every interval the wide spread passes as well
    monkeypatch.setattr(acceptance, "steps_meet_paper", lambda found, paper: True)
    verdict = verdict_of(rows)
    assert verdict["steps"]["multi"]["pass"] is True and verdict["pass"] is True
    # a task kind without any counted step has no interval, and the function decides on None
    seen = []
    monkeypatch.setattr(acceptance, "steps_meet_paper", lambda found, paper: seen.append((found, paper)) or False)
    alone = acceptance.steps_check([r for r in rows if r["task_kind"] == "single"], PE_PORT)
    assert alone["multi"] == {"paper": 5.1, "mean": None, "low": None, "high": None, "half_width": None, "difference": None, "pass": False}
    assert seen[-1] == (None, 5.1)


def test_the_pilot_and_a_list_of_ids_decide_nothing():
    pilot = [r for r in registered_rows(PE_PORT) if int(r["scenario_id"]) in scenario_sets.PILOT]
    assert len(pilot) == 10
    for name in ("pilot", "ids"):
        verdict = verdict_of(pilot, set=name)
        assert verdict["pass"] is None and not verdict["decidable"]
        assert any(reason.startswith(f"run run is no run of a registered set: set {name}, 1 run(s) per scenario") for reason in verdict["reasons"])
        assert any(reason.startswith("110 scenario(s) of the registered set have no episode of the port") for reason in verdict["reasons"])
    # three runs per scenario are not the registered count
    rows = registered_rows(PE_PORT)
    assert verdict_of(rows, repeats=3)["reasons"] == [
        "run run is no run of a registered set: set runnable, 3 run(s) per scenario, where the rule asks for runnable or recovered with 1"]
    # the recovered set is registered: the runnable scenarios and those the rows file unlocks
    assert verdict_of(rows, set="recovered")["pass"] is True


def test_an_unfinished_run_decides_nothing():
    rows = registered_rows(PE_PORT)
    n = len(rows)
    assert verdict_of(rows, ended=None)["reasons"] == [f"run run is not finished: {n} of {n} episodes done, and run.json has no end"]
    played = n - 20
    cut = verdict_of(rows[:played], episodes_planned=n, scenario_ids=sorted(RUNNABLE), ended=None)
    assert cut["pass"] is None and cut["reasons"][0] == f"run run is not finished: {played} of {n} episodes done, and run.json has no end"
    # a second run folder that is not finished stops the check as well, although it holds no episode of the port
    done, other = run_record(rows), run_record([], run="other", episodes_planned=5, episodes_done=2)
    assert acceptance.check(rows, [done, other], PE_PORT)["reasons"] == [
        f"run other is not finished: {other['episodes_done']} of {other['episodes_planned']} episodes done"]


def test_a_missing_scenario_and_a_scenario_held_twice_decide_nothing():
    rows = registered_rows(PE_PORT)
    missing = acceptance.check(rows[2:], [run_record(rows, episodes_planned=120, episodes_done=120)], PE_PORT)
    assert missing["pass"] is None and missing["reasons"] == ["2 scenario(s) of the registered set have no episode of the port: 1, 2"]
    twice = acceptance.check(rows + [dict(rows[0], episode="plan_execute_paper_s001_r2")], [run_record(rows)], PE_PORT)
    assert twice["pass"] is None and twice["reasons"] == ["1 scenario(s) have more than one episode of the port: 1"]
    # two finished runs that both played the set hold every scenario twice
    again = registered_rows(PE_PORT, run="again")
    both = acceptance.check(rows + again, [run_record(rows), run_record(again, run="again")], PE_PORT)
    assert both["reasons"] == ["120 scenario(s) have more than one episode of the port: " + ", ".join(map(str, sorted(RUNNABLE)[:12])) + " and 108 more"]


def test_an_episode_without_a_step_count_decides_nothing():
    rows = registered_rows(PE_PORT)
    rows[0]["steps"], rows[0]["error"] = "", "RuntimeError: server iot did not start"
    verdict = verdict_of(rows)
    assert verdict["pass"] is None and verdict["errors"] == 1
    assert verdict["reasons"] == ["1 episode(s) have no step count, since their agent was never built: plan_execute_paper_s001_r1"]


class OutlastsTheLimit:
    """A stand-in for the runner of either port whose run does not return within the time limit of the driver."""

    def __init__(self, **keywords):
        self.calls, self.trajectory = [], [{"tool": "sites"}, {"tool": "assets"}]

    async def run(self, question):
        await asyncio.sleep(60)

    def steps_reached(self):
        return 4

    def record(self, result=None):
        return {"end": "no_result", "plan": PLAN, "plan_steps": 6, "plans": [{"plan": PLAN, "review": {"status": "Valid"}}], "history": [],
                "executed_steps": 4, "step_agents": []}


def test_a_run_of_either_port_without_a_result_keeps_its_step_count(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setattr(run_scenarios.play_episode, "pe_port_runner", lambda: OutlastsTheLimit)
    monkeypatch.setattr(run_scenarios.agent_runs, "agent_as_tool_runner_class", lambda: OutlastsTheLimit)
    settings = run_scenarios.Settings(launchers={}, model="granite-test", base_url="http://127.0.0.1:9/v1", api_key="stand-in-key",
                                      reload=False, timeout_s=0.05)
    for port in (PE_PORT, PORT):
        record = run_scenarios.play(port, dict(ROWS[0], text="List the IoT sites."), 1, tmp_path, settings)
        assert record["error"] == "TimeoutError: the run exceeded 0.05 s" and record["answer"] == ""
        assert record["steps"] == 4 and record["tool_calls"] == 2  # the count the runner kept, where the episode had none before
        folder = tmp_path / run_folder_names.episode_name(port, 1, 1)
        assert json.loads((folder / "episode.json").read_text())["steps"] == 4
        if port == PE_PORT:
            assert record["plan_steps"] == 6 and record["react_steps"] is None
            flow = json.loads((folder / "plan_execute_trajectory.json").read_text())
            assert flow["end"] == "no_result" and flow["executed_steps"] == 4 and flow["plan"] == PLAN
        else:
            assert record["react_steps"] == 4 and record["plan_steps"] is None and not (folder / "react_trajectory.json").exists()


def write_run(folder, rows, **changes):
    """A scored run folder: run.json of run_record and scores.csv."""
    folder.mkdir(parents=True)
    meta = {**run_record(rows, run=folder.name, **changes), "scenarios": {"sha256": scenario_file.LAYER_SCENARIO_SHA256},
            "endpoint_models": [{"id": "granite-test", "max_model_len": 131072}]}
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    write_rows(folder / "scores.csv", rows, score.ROW_FIELDS)
    return folder


def test_the_report_of_the_plan_execute_port_prints_the_plan_figures_and_says_that_the_steps_alone_decide(tmp_path, capsys):
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import workflow

    assert acceptance.PLANS == workflow.PLANS and acceptance.FLOW_FILE == "plan_execute_trajectory.json"
    rows = registered_rows(PE_PORT, run="run")
    run = write_run(tmp_path / "run", rows)
    for n, r in enumerate(rows):
        folder = run / r["episode"]
        folder.mkdir()
        plans = [{"review": {"status": "Invalid"}}] * (n % 2) + [{"review": {"status": "Valid"}}]
        history = [{"agent_name": "IoT Data Download"}] * 3 + [{"agent_name": "SummarizationAgent"}]
        (folder / "plan_execute_trajectory.json").write_text(json.dumps({"plans": plans, "history": history}), encoding="utf-8")
    code = acceptance.main([str(run), "--port", PE_PORT, "--out", str(tmp_path / "out")])
    text = (tmp_path / "out" / "acceptance_plan_execute_paper.md").read_text(encoding="utf-8")  # beside the other port's file
    assert code == 0 and text == capsys.readouterr().out.rstrip("\n") + "\n" and not (tmp_path / "out" / "acceptance.md").exists()
    assert text.startswith("# Acceptance of the paper-era Plan-Execute, granite-test\n\nVerdict: PASS. 120 episodes over 120 scenarios, 0 of them ended by an error.")
    assert "| Task kind | Paper steps | Port mean [95% interval] | Half-width | Mean minus paper | Passes the steps rule |" in text
    single = acceptance.check(rows, [run_record(rows)], PE_PORT)["steps"]["single"]
    assert (f"| single-agent | 5.2 | 5.50 [{single['low']:.2f}, {single['high']:.2f}] | {single['half_width']:.2f} | +0.30 | True |"
            in text)
    assert "| multi-agent | 5.1 | 5.00 [5.00, 5.00] | 0.00 | -0.10 | True |" in text
    assert ("The steps rule: the 95 percent interval of the port's mean steps lies within 1 step of the paper's value on either side, "
            "an equivalence, so a wide interval misses where an overlap would let it pass.") in text
    assert "| Task completion | 22/84 (26.19) | 0.0 |" in text and "one judge call at temperature 0" in text
    assert "Plan length: mean 4.00 [4.00, 4.00] steps, beside the paper's 5.56 planned steps over all paradigms." in text
    assert f"Reviewer rounds per episode: mean 1.50, at most 2, {0} of {len(rows)} episodes ran the last plan after 5 invalid reviews." in text
    assert "History entries per episode, the executed plan steps that returned and the summarization step: mean 4.00 over 120 episodes." in text
    assert "The steps check alone decides the verdict." in text and "decision" not in text.lower()


def test_the_check_exits_with_0_for_a_pass_1_for_a_miss_and_3_when_the_runs_cannot_decide(tmp_path, capsys):
    cases = {"pass": (registered_rows(PE_PORT), {}, 0, "PASS"),
             "miss": (registered_rows(PE_PORT, single=(2,), multi=(2,)), {}, 1, "MISS"),
             "pilot": ([r for r in registered_rows(PE_PORT) if int(r["scenario_id"]) in scenario_sets.PILOT], {"set": "pilot"}, 3, "NOT DECIDABLE")}
    for name, (rows, changes, expected, word) in cases.items():
        run = write_run(tmp_path / name / "run", [dict(r, run="run") for r in rows], **changes)
        code = acceptance.main([str(run), "--port", PE_PORT, "--out", str(tmp_path / name / "out")])
        text = (tmp_path / name / "out" / "acceptance_plan_execute_paper.md").read_text(encoding="utf-8")
        assert code == expected and text == capsys.readouterr().out.rstrip("\n") + "\n"
        assert text.splitlines()[2].startswith(f"Verdict: {word}.")
    assert acceptance.NOT_DECIDABLE == 3
    # the report of runs that cannot decide names every condition that failed before it prints a value
    assert "The runs cannot decide the check" in text and "- run run is no run of a registered set: set pilot" in text
    assert text.index("- run run is no run of a registered set") < text.index("| Task kind |")


def test_the_reproduction_job_checks_each_port_that_aob_agents_names_and_prints_its_exit_code():
    script = (PACKAGE / "frameworks" / "AssetOpsBench" / "cluster_jobs" / "run_reproduction.sh").read_text(encoding="utf-8")
    step = '-m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check "$RUN_DIR" --port plan_execute_paper --out "$OUT/results"; echo "PE_ACCEPTANCE_EXIT $?"'
    assert step in script and '*" plan_execute_paper "*)' in script
    assert '-m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check "$RUN_DIR" --out "$OUT/results"; echo "ACCEPTANCE_EXIT $?"' in script  # the Agent-As-Tool step unchanged
