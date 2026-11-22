"""The Plan-Execute wiring of the driver: --agent plan_execute plays the paper-era port over AssetOpsBench's servers behind the
metering layer, with a scripted stand-in for SGLang's completion routes in place of a model. The planner writes a plan, the
reviewer accepts it, each step runs on its agent's tool on the row, the run stops when the environment ends the episode, and
the driver records the run with the columns the tables read and the files the training and GEPA read: the model calls on the
completion route, plan.txt, trajectory.json and the record of the run, which a run that raised leaves as well. Both ports also
play one scripted episode on each of the repair rules row, the deterioration row and the healthy control, every call in the form
their examples teach.
No network, no model."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import agent_runs  # noqa: E402

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.tests.test_plan_execute_paper import VALID, plan  # noqa: E402

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online  # noqa: E402
from CausalDiagnosis.training import rejection_sampling_examples as rft_data  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import react_tool_adapter, sub_agent_registry  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.pinned_clock import pinned_preset  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.tests.test_agent_as_tool_driver import _no_calls, _without_entry  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts import split_receipt  # noqa: E402

MODEL = "granite-stand-in"
END_TASK = react_tool_adapter.agent_name("finish")  # the name under which the agents see the task tool finish
PLAN = plan(("List the IoT sites.", "IoT Data Download", "None"), ("End the task with the summary no repair was made.", "TaskAgent", "None"))
SCRIPT = [PLAN, VALID, *step("I need the sites.", "sites", "{}"), *finish("The sites are MAIN"), REVIEW_OK,
          *step("I need to end the task.", END_TASK, "summary=no repair was made")]


def _rows(run_dir):
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _play(tmp_path, monkeypatch, script, args, agent="plan_execute"):
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with ScriptedEndpoint(script, model=MODEL) as ep:
        run_episodes.main([*args, "--agent", agent, "--model", MODEL, "--base-url", ep.base_url, "--seeds", "1", "--no-grade",
                           "--out", str(tmp_path / "out")])
    run_dir = next(p for p in (tmp_path / "out").iterdir() if p.is_dir())
    return run_dir, _rows(run_dir), read_json(run_dir / "run.json"), ep


def test_the_port_plays_the_base_row_and_records_what_the_tables_the_training_and_gepa_read(tmp_path, monkeypatch):
    run_dir, rows, meta, ep = _play(tmp_path, monkeypatch, SCRIPT, ["--config", "base"])
    assert meta["agent"] == "plan_execute" and meta["agent_label"] == f"plan_execute_{MODEL}" and meta["servers"][-1] == "causal_diagnosis"
    assert meta["model_id"] == MODEL and meta["agent_as_tool"] is None
    block = meta["plan_execute"]
    assert block["form"] == "paper_era" and block["plans"] == 5 and block["step_agent_max_steps"] == 6 and block["reflections"] == 1
    assert block["agents"] == ["IoT Data Download", "TaskAgent", "Failure Mode and Sensor Relevancy Expert for Industrial Asset",
                               "Work Order Management", "CausalDiagnosisAgent"]
    # the example start of the run, and the planner lists the question of the diagnosis example written for it
    example = block["diagnosis_example"]
    named = ", ".join(f"{c} {v}" for c, v in example["levels"].items())
    assert example["seed"] == 0 and f"submit the diagnosis {named} with the evidence" in ep.completions()[0]["prompt"]
    assert len(rows) == 1 and rows[0]["error"] == "", rows[0]
    row = rows[0]
    assert row["agent"] == f"plan_execute_{MODEL}" and row["end_reason"] == "finish" and row["plan_steps"] == "2"
    assert row["tool_calls"] == "2" and row["causal_calls"] == "0" and row["react_steps"] == ""
    assert row["llm_calls"] == str(len(SCRIPT)) == str(len(ep.completions()))
    ep_dir = run_dir / "seed000000"
    assert [(t["agent"], t["server"], t["tool"]) for t in read_json(ep_dir / "trajectory.json")] == [
        ("IoTAgent", "iot", "sites"), ("TaskAgent", "task", "finish")]
    assert (ep_dir / "plan.txt").read_text(encoding="utf-8") == PLAN
    record = read_json(ep_dir / "plan_execute_trajectory.json")
    assert record["end"] == "episode_over" and [p["review"]["status"] for p in record["plans"]] == ["Valid"] and record["plan"] == PLAN
    # both plan steps were started, and the second ended the episode before it returned, so the history holds the first alone
    assert (record["executed_steps"], record["plan_steps"]) == (2, 2) and [h["agent_name"] for h in record["history"]] == ["IoT Data Download"]
    assert record["question"].startswith((ep_dir / "question.txt").read_text(encoding="utf-8")[:40])
    calls = read_json(ep_dir / "llm_calls.json")
    assert [c["prompt"] for c in calls] == [b["prompt"] for b in ep.completions()] and all("reply" in c for c in calls)
    # GEPA's card reads the plan, the training data takes every call on the completion route, and the tables count the cells
    assert gepa_online.agent_outputs(ep_dir).startswith("plan:\n" + PLAN)
    examples, dropped = rft_data.call_examples(ep_dir, "plan_execute")
    assert {e["route"] for e in examples} == {"completion"} and [e["prompt"] for e in examples] == [c["prompt"] for c in calls] and not dropped
    episode = {**row, "episode_dir": ep_dir, "run_dir": run_dir}
    # the trajectory is always written, also by a run that raised
    assert aggregate._trajectory(episode) is not None
    assert aggregate.tool_invocations(episode) == 2.0 and aggregate.planned_steps(episode) == 2.0


def test_a_planning_round_takes_the_no_call_step_of_the_driver(tmp_path, monkeypatch):
    """The planning round calls no tool of the row, so it ends an iteration without a counted call, which takes the episode's
    no_call step through the entry the driver hands the runner, one step of the row's step cap, as for every design. On the row
    with the CausalDiagnosis MCP the Finish of the IoT step agent, a ReAct step without a counted call, takes one more. The control
    is the same play with the runner built without that entry, which takes none."""
    run_dir, rows, _, _ = _play(tmp_path / "bare", monkeypatch, SCRIPT, ["--config", "base", "--no-diagnostics"])
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == "finish" and _no_calls(run_dir) == 1, rows[0]
    run_dir, rows, _, _ = _play(tmp_path / "entry", monkeypatch, SCRIPT, ["--config", "base"])
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == "finish" and _no_calls(run_dir) == 2, rows[0]
    runner_class = agent_runs.plan_execute_runner_class()
    monkeypatch.setattr(agent_runs, "plan_execute_runner_class", lambda: _without_entry(runner_class))
    run_dir, rows, _, _ = _play(tmp_path / "control", monkeypatch, SCRIPT, ["--config", "base"])
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == "finish" and _no_calls(run_dir) == 0, rows[0]


def test_without_the_diagnostics_server_the_planner_has_no_causal_diagnosis_agent(tmp_path, monkeypatch):
    finish_only = plan(("End the task with the summary nothing found.", "TaskAgent", "None"))
    script = [finish_only, VALID, *step("I need to end the task.", END_TASK, "summary=nothing found")]
    _, rows, meta, ep = _play(tmp_path, monkeypatch, script, ["--config", "base", "--no-diagnostics"])
    assert "CausalDiagnosisAgent" not in meta["plan_execute"]["agents"]
    block = meta["plan_execute"]
    assert (block["form"], block["step_agent_max_steps"], block["decoding"]["max_tokens"]) == ("paper_era", 6, 500) and "no_call_steps" not in block
    assert all(b["max_tokens"] == 500 for b in ep.completions())
    assert "CausalDiagnosisAgent" not in ep.completions()[0]["prompt"] and rows[0]["end_reason"] == "finish" and rows[0]["plan_steps"] == "1"


class RaisingRunner:
    """A runner that planned, made one tool call and raised, as an API error that outlasts the client's retries would."""

    def __init__(self, server_paths, base_url, api_key, model, temperature=None, log_path=None, example_start=None, on_idle=None,
                 example_asset=None):
        self.calls = [{"prompt": "plan", "reply": "## Step 1", "agent": "planner"}]
        self.trajectory = [{"step": 1, "turn": 1, "agent": "IoTAgent", "task": "List the IoT sites.", "server": "iot", "tool": "sites",
                            "tool_args": {}, "response": "MAIN", "error": None, "duration_ms": 3.0}]  # the record shape of the runtime

    async def run(self, question):
        raise ConnectionError("the endpoint went away")

    def record(self, result=None):
        return {"end": "no_result", "plan": PLAN, "plan_steps": 2, "plans": [{"plan": PLAN, "review": {"status": "Valid"}}], "history": [],
                "executed_steps": 1, "step_agents": []}


class RaisedBeforeItsWorkflow(RaisingRunner):
    """The same failure before the runner built its workflow, so that it holds no record."""

    def record(self, result=None):
        return None


def test_a_run_that_raised_still_writes_its_calls_its_trajectory_and_its_record(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_runs, "plan_execute_runner_class", lambda: RaisingRunner)
    run_dir, rows, _, _ = _play(tmp_path / "planned", monkeypatch, [], ["--config", "base"])
    ep_dir = run_dir / "seed000000"
    assert "ConnectionError: the endpoint went away" in rows[0]["error"] and rows[0]["llm_calls"] == "1"
    assert read_json(ep_dir / "trajectory.json") == RaisingRunner(None, None, None, None).trajectory
    record = read_json(ep_dir / "plan_execute_trajectory.json")
    assert record["end"] == "no_result" and record["question"] is None and (record["executed_steps"], record["plan_steps"]) == (1, 2)
    assert not (ep_dir / "plan.txt").exists() and not (ep_dir / "answer.txt").exists()
    # the row carries the plan length of the run's own record, and the tables read the cells of the row: the trajectory exists,
    # so the count of the released runner's plan parser over the call log never stands in for a port
    assert rows[0]["plan_steps"] == "2"
    episode = {**rows[0], "episode_dir": ep_dir, "run_dir": run_dir}
    assert aggregate._trajectory(episode) is not None and aggregate.planned_steps(episode) == 2.0
    # a run that raised before it built its workflow has no record and no plan length
    monkeypatch.setattr(agent_runs, "plan_execute_runner_class", lambda: RaisedBeforeItsWorkflow)
    run_dir, rows, _, _ = _play(tmp_path / "unplanned", monkeypatch, [], ["--config", "base"])
    ep_dir = run_dir / "seed000000"
    assert "ConnectionError" in rows[0]["error"] and rows[0]["plan_steps"] == "" and not (ep_dir / "plan_execute_trajectory.json").exists()
    assert read_json(ep_dir / "trajectory.json") == RaisingRunner(None, None, None, None).trajectory


def test_the_port_refuses_the_rigs_action_server(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "stand-in-key")
    with pytest.raises(SystemExit, match="paper-era Plan-Execute over AssetOpsBench's servers") as refused:
        run_episodes.main(["--config", "static", "--agent", "plan_execute", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1",
                           "--seeds", "1", "--out", str(tmp_path / "rig")])
    assert "decision" not in str(refused.value)


def test_plan_execute_without_an_api_key_is_refused_with_the_sentence_of_the_paper_era_runners(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(SystemExit, match="the paper-era runners send it to the endpoint of --base-url"):
        run_episodes.main(["--config", "base", "--agent", "plan_execute", "--model", "m", "--base-url", "http://127.0.0.1:9/v1", "--seeds", "1",
                           "--out", str(tmp_path / "nokey")])


# ----- both ports on the rows above the real-time row: one scripted episode each, every call in the form that the
# examples teach, under the pinned clock, so that every call and every wait takes one cycle and an episode depends on its calls alone

PORTS = ("agent_as_tool", "plan_execute")
RUN_RECORD = {"agent_as_tool": "react_trajectory.json", "plan_execute": "plan_execute_trajectory.json"}  # where a runner records its end


def _waits(first, count):
    """count waits of one cycle. The thought and the timeout differ from wait to wait, since the engine stops an agent that repeats
    one action with one input three times in a row."""
    return [(f"The rig has not notified the close yet, wait {n}.", "wait_for_notification", f"timeout={n}") for n in range(first, first + count)]


# per task: the request, the sub-agent of Agent-As-Tool, the agent of the planner, the steps of that agent, and its answer, which the
# task whose call ends the episode does not reach
REPAIR = [("read the asset profile", "TaskAgent", "TaskAgent", [("I need the profile.", "get_asset_profile", "{}")], "The asset is hyd_1"),
          ("estimate the effect of the action replace:cooler", "CausalDiagnosisAgent", "CausalDiagnosisAgent",
           [("I need the predicted effect.", "estimate_intervention_effect", "action=replace:cooler")], "The prediction is pred/1"),
          ("submit the diagnosis cooler 3, valve 100, pump 0, accumulator 130 with the evidence pred/1", "TaskAgent", "TaskAgent",
           [("I submit the diagnosis.", "submit_diagnosis", "cooler=3, valve=100, pump=0, accumulator=130, evidence_ids=pred/1")],
           "The diagnosis is submitted"),
          ("create a work order to replace the cooler on its child asset hyd_1_cooler at site MAIN", "WOAgent", "Work Order Management",
           [("I create the work order.", "generate_work_order", "description=replace the cooler, asset_num=hyd_1_cooler, site_id=MAIN")],
           "Work order wo-1 is created"),
          ("wait until work order wo-1 closes", "TaskAgent", "TaskAgent", _waits(1, 4), "Work order wo-1 on the cooler closed"),
          ("verify the outcome of prediction pred/1", "CausalDiagnosisAgent", "CausalDiagnosisAgent",
           [("I need the verification.", "verify_outcome", "prediction_id=pred/1")], "The outcome of pred/1 is read"),
          ("end the task with the summary the cooler was replaced", "TaskAgent", "TaskAgent",
           [("I need to end the task.", END_TASK, "summary=the cooler was replaced")], None)]
REPAIR_CALLS = [("TaskAgent", "task", "get_asset_profile"), ("CausalDiagnosisAgent", "causal_diagnosis", "estimate_intervention_effect"),
                ("TaskAgent", "task", "submit_diagnosis"), ("WOAgent", "wo", "generate_work_order"),
                *[("TaskAgent", "task", "wait_for_notification")] * 4, ("CausalDiagnosisAgent", "causal_diagnosis", "verify_outcome"),
                ("TaskAgent", "task", "finish")]
FAILURE = [("wait for the next notification five times", "TaskAgent", "TaskAgent", _waits(1, 5), "Nothing was notified in five waits"),
           ("wait for the next notification five more times", "TaskAgent", "TaskAgent", _waits(6, 5), None)]


def _script(agent, tasks):
    """The model's replies for a port: for Agent-As-Tool the supervisor's delegation before each task, for Plan-Execute the plan of
    the tasks and its review, then the steps of the task's agent with its Finish and the review of ReActXen."""
    replies = []
    if agent == "plan_execute":
        replies += [plan(*[(f"{request[0].upper()}{request[1:]}.", planned, "None") for request, _, planned, _, _ in tasks]), VALID]
    for request, sub_agent, _planned, steps, answer in tasks:
        if agent == "agent_as_tool":
            replies += step(f"I need the {sub_agent} to {request}.", sub_agent, f"request={request}")
        for thought, tool, arguments in steps:
            replies += step(thought, tool, arguments)
        if answer is not None:
            replies += [*finish(answer), REVIEW_OK]
    return replies


def _stopped_at_the_end(ep_dir, agent, ep, script):
    """The runner ended with the episode, at the call that ended it, and no model call followed that call."""
    assert read_json(ep_dir / RUN_RECORD[agent])["end"] == "episode_over"
    assert ep.script == [] and len(ep.completions()) == len(script)


@pytest.mark.parametrize("agent", PORTS)
def test_each_port_plays_an_order_through_a_visit_on_the_repair_rules_row(tmp_path, monkeypatch, agent):
    script = _script(agent, REPAIR)
    run_dir, rows, _, ep = _play(tmp_path, monkeypatch, script, ["--config", pinned_preset(tmp_path, "repair_rules")], agent)
    ep_dir = run_dir / "seed000000"
    trajectory = read_json(ep_dir / "trajectory.json")
    # a call that the wrapper refuses never reaches the trajectory, and the row accepted every call that did
    assert [(t["agent"], t["server"], t["tool"]) for t in trajectory] == REPAIR_CALLS
    assert [t["error"] for t in trajectory] == [None] * len(REPAIR_CALLS)
    # on this row, with the CausalDiagnosis MCP, the Finish of the WO agent is a ReAct step without a counted call and takes a no_call
    # step, one cycle of the pinned clock, so the third wait brings the close of the order and the fourth times out
    closes = [i for i, t in enumerate(trajectory) if "Work order wo-1 on the cooler closed." in t["response"]]
    assert closes == [6] and [t["tool"] for t in trajectory[4:8]] == ["wait_for_notification"] * 4, closes
    _stopped_at_the_end(ep_dir, agent, ep, script)
    row = rows[0]
    assert row["agent"] == f"{agent}_{MODEL}" and row["error"] == "" and row["end_reason"] == "finish" and row["refusals"] == "0", row
    assert all(row[column] != "" for column in ("pass", "recovered", "faulty_set_right", "cost")), row


@pytest.mark.parametrize("agent", PORTS)
def test_each_port_stops_when_the_failure_event_ends_the_episode(tmp_path, monkeypatch, agent):
    # each no_call step is one cycle of the pinned clock, and a run meets the failure one wait earlier for each: the planning round of
    # Plan-Execute, an iteration without a counted call, and, on this row with the CausalDiagnosis MCP, the Finish of the TaskAgent
    # after its first five waits, a ReAct step without a counted call
    no_calls = (agent == "plan_execute") + 1
    tasks = [FAILURE[0], (*FAILURE[1][:3], _waits(6, 5 - no_calls), None)]
    script = _script(agent, tasks)
    run_dir, rows, _, ep = _play(tmp_path, monkeypatch, script, ["--config", pinned_preset(tmp_path, "deterioration")], agent)
    ep_dir = run_dir / "seed000000"
    trajectory = read_json(ep_dir / "trajectory.json")
    waits = sum(len(steps) for _request, _sub_agent, _planned, steps, _answer in tasks)
    assert [t["tool"] for t in trajectory] == ["wait_for_notification"] * waits and _no_calls(run_dir) == no_calls
    receipt = json.loads(trajectory[-1]["response"].rsplit("\n", 1)[1])  # the receipt of the last wait shows the end of the episode
    assert [(n["event"], n.get("reason")) for n in receipt["notifications"] if n["event"] == "end"] == [("end", "failure")]
    _stopped_at_the_end(ep_dir, agent, ep, script)
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == "failure", rows[0]


def _healthy_tasks():
    """The two tasks of the healthy control on the asset of the rows: the diagnosis that names every component at its healthy level with no evidence
    to cite, in the engine's form for no value, and the end of the task."""
    healthy = get_profile(DEFAULT_ASSET).healthy
    named = ", ".join(f"{c} {v}" for c, v in healthy.items())
    arguments = ", ".join(f"{c}={v}" for c, v in healthy.items())
    return [(f"submit the diagnosis {named} with no evidence", "TaskAgent", "TaskAgent",
             [("I submit the diagnosis with no evidence to cite.", "submit_diagnosis", f"{arguments}, evidence_ids=None")],
             "The diagnosis is submitted with no evidence"),
            ("end the task with the summary no component is faulty", "TaskAgent", "TaskAgent",
             [("I need to end the task.", END_TASK, "summary=no component is faulty")], None)]


@pytest.mark.parametrize("agent", PORTS)
def test_a_start_with_no_fault_is_a_valid_run_of_the_port(tmp_path, monkeypatch, agent):
    """On the healthy control the TaskAgent submits every component at its healthy level without evidence and ends the task. No call
    is refused, the diagnosis stands with an empty evidence list, no order is created, and the row counts no faulty component."""
    script = _script(agent, _healthy_tasks())
    # the control is no row of the environment table, so the driver gives it the rig's server unless the run names the ports' surface
    run_dir, rows, _, ep = _play(tmp_path, monkeypatch, script, ["--config", "healthy_control", "--servers", "aob"], agent)
    ep_dir = run_dir / "seed000000"
    trajectory = read_json(ep_dir / "trajectory.json")
    assert [(t["agent"], t["server"], t["tool"]) for t in trajectory] == [("TaskAgent", "task", "submit_diagnosis"), ("TaskAgent", "task", "finish")]
    assert [t["error"] for t in trajectory] == [None, None] and trajectory[0]["tool_args"]["evidence_ids"] == ""
    assert json.loads(trajectory[0]["response"])["evidence_ids"] == []  # the base row answers without a receipt line
    _stopped_at_the_end(ep_dir, agent, ep, script)
    row = rows[0]
    assert row["error"] == "" and row["end_reason"] == "finish" and row["n_faulty"] == "0" and row["refusals"] == "0", row


# ----- a notice inside the no_call step of an IBM agent: the repair rules row with the token clock, one cycle per full
# think_seconds_per_cycle of generated tokens, where only the long thought of the WO agent's Finish fills cycles

TOKEN_ROW = "repair_rules_token_clock"
ORDERED = "Work order wo-1 is created"  # the answer of the WO agent, whose Finish thinks long


def _order_then_read(agent, think_words):
    """The replies of a port that diagnoses, orders the cooler, reads the asset profile and ends the task, the Finish of the WO
    agent with a thought of think_words words, which the token clock charges as so many generated tokens."""
    tasks = [("submit the diagnosis cooler 3, valve 100, pump 0, accumulator 130 with no evidence", "TaskAgent", "TaskAgent",
              [("I submit the diagnosis.", "submit_diagnosis", "cooler=3, valve=100, pump=0, accumulator=130, evidence_ids=None")],
              "The diagnosis is submitted"),
             ("create a work order to replace the cooler on its child asset hyd_1_cooler at site MAIN", "WOAgent", "Work Order Management",
              [("I create the work order.", "generate_work_order", "description=replace the cooler, asset_num=hyd_1_cooler, site_id=MAIN")],
              ORDERED),
             ("read the asset profile", "TaskAgent", "TaskAgent", [("I need the profile.", "get_asset_profile", "{}")], "The asset is hyd_1"),
             ("end the task with the summary the cooler was replaced", "TaskAgent", "TaskAgent",
              [("I need to end the task.", END_TASK, "summary=the cooler was replaced")], None)]
    script = _script(agent, tasks)
    at = script.index(finish(ORDERED)[1])
    script[at - 1] = finish(ORDERED, " ".join(["weighing"] * think_words))[0]
    return script


def _closes(response):
    """The closes of work orders that the receipt of a recorded answer carries, as (id, cycle)."""
    receipt = split_receipt(response)[1] or {}
    return [(n.get("id"), n.get("t")) for n in receipt.get("notifications", []) if n.get("event") == "work_order_closed"]


@pytest.mark.parametrize("diagnostics", [True, False], ids=["with_mcp", "control_without_mcp"])
@pytest.mark.parametrize("agent", PORTS)
def test_a_close_inside_the_no_call_step_of_an_ibm_agent_reaches_its_next_call(tmp_path, monkeypatch, agent, diagnostics):
    """With the CausalDiagnosis MCP the Finish of the WO agent is a ReAct step without a counted call and takes a no_call step,
    whose thinking runs past the close of the order just made. A no_call step answers no agent, so the close stays pending and
    the next counted call, the TaskAgent's read of the profile, carries it with its cycle, and no later call carries it again.
    The control without the MCP: the paper's rule charges that thinking to the read itself, which carries the close too, and no
    no_call step holds it."""
    _name, config = load_config(TOKEN_ROW)
    per_cycle = config.think_seconds_per_cycle / config.seconds_per_token  # the generated tokens of one cycle of thinking
    durations = config.durations
    think = int((durations["lead_time"]["cooler"] + durations["repair_downtime"]["cooler"] + 2) * per_cycle)
    script = _order_then_read(agent, think)
    run_dir, rows, _, ep = _play(tmp_path, monkeypatch, script, ["--config", TOKEN_ROW, *([] if diagnostics else ["--no-diagnostics"])], agent)
    ep_dir = run_dir / "seed000000"
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == "finish", rows[0]
    log = read_json(ep_dir / "hidden" / "log.json")
    close = next(e for e in log if e.get("event") == "work_order_closed")
    trajectory = read_json(ep_dir / "trajectory.json")
    tools = [t["tool"] for t in trajectory]
    assert tools == ["submit_diagnosis", "generate_work_order", "get_asset_profile", "finish"], tools
    assert _closes(trajectory[2]["response"]) == [(close["id"], close["t_end"])], trajectory[2]["response"][-600:]
    assert _closes(trajectory[3]["response"]) == []
    stepped = [e for e in log if e.get("step") is not None and e.get("event") != "end"]
    held = [e for e in stepped if e["event"] == "no_call" and max((s["t_end"] for s in stepped if s["step"] < e["step"]), default=-1) < close["t_end"] <= e["t_end"]]
    assert len(held) == int(diagnostics), held
