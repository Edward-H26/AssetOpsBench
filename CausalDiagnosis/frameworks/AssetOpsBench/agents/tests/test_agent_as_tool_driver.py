"""The Agent-As-Tool wiring of the driver: the port of the paper era supervisor plays rows of the environment
table over AssetOpsBench's servers behind the metering layer, with a scripted stand-in for SGLang's completion routes
in place of a model. The supervisor calls its sub-agents, each sub-agent's tool runs on the row, the run stops when the
environment ends the episode on finish, the step cap, the budget or the horizon, and the driver records the run with the
columns the tables read plus the supervisor's ReAct counter. No network, no model."""
from CausalDiagnosis.utils.paths import CONFIGS, PACKAGE, preset_path
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import agent_runs  # noqa: E402

PORT_TESTS = PACKAGE / "frameworks" / "AssetOpsBench" / "agents" / "Agent_As_Tool_Paper" / "tests"
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import react_tool_adapter, sub_agent_registry  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import REPLY_ROOM  # noqa: E402

END_TASK = react_tool_adapter.agent_name("finish")  # the name under which the agents see the task tool finish

MODEL = "granite-stand-in"


def _rows(run_dir):
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _play(tmp_path, monkeypatch, script, args, config=None, context_length=131072):
    """One episode of the port on a row. config, when given, is a preset with some fields replaced, under the preset's name, and
    context_length the context the stand-in endpoint serves."""
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    if config:
        name, fields = config
        preset = json.loads(preset_path(name).read_text(encoding="utf-8"))
        preset.update(fields)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(preset), encoding="utf-8")
        args = ["--config", str(path), *args]
    with ScriptedEndpoint(script, model=MODEL, context_length=context_length) as ep:
        run_episodes.main([*args, "--agent", "agent_as_tool", "--model", MODEL, "--base-url", ep.base_url, "--seeds", "1", "--no-grade",
                           "--out", str(tmp_path / "out")])
    run_dir = next(p for p in (tmp_path / "out").iterdir() if p.is_dir())
    return run_dir, _rows(run_dir), read_json(run_dir / "run.json"), ep


def _history_args():
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import default_sensor_groups

    sensors = list(map(get_profile("hyd_1").field_name, default_sensor_groups()["tier1"]))
    return json.dumps({"site_name": "MAIN", "asset_id": "hyd_1", "sensors": sensors, "limit": 5})


def _no_calls(run_dir):
    """The no_call steps of the episode of a run of one seed, as the environment logs them."""
    return sum(1 for e in read_json(run_dir / "seed000000" / "hidden" / "log.json") if e["event"] == "no_call")


def _without_entry(runner_class):
    """The runner class built without the episode's no_call entry, the control of the tests of that entry."""
    class WithoutEntry(runner_class):
        def __init__(self, *args, on_idle=None, **kwargs):
            super().__init__(*args, **kwargs)
    return WithoutEntry


def test_the_port_plays_the_base_row_and_is_recorded_with_the_columns_the_tables_read(tmp_path, monkeypatch):
    script = [*step("I need the asset profile.", "TaskAgent", "request=read the asset profile"),
              *step("I need the profile.", "get_asset_profile", "{}"), *finish("The profile is read"), REVIEW_OK,
              *step("I need the sites.", "IoTAgent", "request=list all the sites"),
              *step("I need the sites tool.", "sites", "{}"), *finish("The sites are MAIN"), REVIEW_OK,
              *step("I should end the task.", "TaskAgent", "request=end the task with the summary no repair was made"),
              *step("I need to end the task.", END_TASK, "summary=no repair was made")]
    run_dir, rows, meta, ep = _play(tmp_path, monkeypatch, script, ["--config", "base"])
    assert meta["agent"] == "agent_as_tool" and meta["agent_label"] == f"agent_as_tool_{MODEL}" and meta["servers"][-1] == "causal_diagnosis"
    block = meta["agent_as_tool"]
    assert block["form"] == sub_agent_registry.MCP_FORM and block["supervisor_max_steps"] is None and block["reflections"] == 1
    assert block["sub_agents"] == ["IoTAgent", "TaskAgent", "FMSRAgent", "WOAgent", "CausalDiagnosisAgent"]
    # the example start of the run, a development start drawn by the registered seed, and the TaskAgent's prompt shows its levels
    example = block["diagnosis_example"]
    assert example["seed"] == 0 and example["n_faulty"] == sum(level != ok for level, ok in zip(example["levels"].values(), (100, 100, 0, 130)))
    shown = ", ".join(f"{c}={v}" for c, v in example["levels"].items())
    assert f"Action Input 1: {shown}, evidence_ids=" in ep.completions()[2]["prompt"]
    assert block["decoding"] == {"temperature": 0.0, "max_tokens": REPLY_ROOM, "min_tokens": 1, "no_stop_trim": True}
    # the row lists the CausalDiagnosis MCP, so every request asks for the room its prompt leaves in the served context, recorded
    assert meta["served"] == {"context_length": 131072, "cache_report": None} and meta["decoding"]["max_tokens"] == REPLY_ROOM
    assert [b["max_tokens"] for b in ep.completions()] == [131072 - len(b["prompt"].split()) - 2 for b in ep.completions()]
    assert "react_steps" in COLUMNS and len(rows) == 1 and rows[0]["error"] == "", rows[0]
    row = rows[0]
    assert row["agent"] == f"agent_as_tool_{MODEL}" and row["end_reason"] == "finish" and row["react_steps"] == "4"
    assert row["tool_calls"] == "3" and row["causal_calls"] == "0" and row["plan_steps"] == ""
    assert row["llm_calls"] == str(len(script)) == str(len(ep.completions()))
    ep_dir = run_dir / "seed000000"
    trajectory = read_json(ep_dir / "trajectory.json")
    assert [(t["agent"], t["server"], t["tool"], t["turn"]) for t in trajectory] == [
        ("TaskAgent", "task", "get_asset_profile", 1), ("IoTAgent", "iot", "sites", 2), ("TaskAgent", "task", "finish", 3)]
    calls = read_json(ep_dir / "llm_calls.json")
    assert [c["prompt"] for c in calls] == [b["prompt"] for b in ep.completions()] and all("reply" in c for c in calls)
    assert read_json(ep_dir / "react_trajectory.json")["task"] == (ep_dir / "question.txt").read_text(encoding="utf-8")
    assert not (ep_dir / "turns.json").exists() and not (ep_dir / "plan.txt").exists()


def test_without_the_diagnostics_server_the_supervisor_has_no_causal_diagnosis_sub_agent(tmp_path, monkeypatch):
    script = [*step("I should end the task.", "TaskAgent", "request=end the task with the summary nothing found"),
              *step("I need to end the task.", END_TASK, "summary=nothing found")]
    _, rows, meta, ep = _play(tmp_path, monkeypatch, script, ["--config", "base", "--no-diagnostics"])
    assert meta["agent_as_tool"]["sub_agents"] == ["IoTAgent", "TaskAgent", "FMSRAgent", "WOAgent"]
    # the paper's form: its step limits and 500 tokens on every request, and the served context recorded as on every run
    block = meta["agent_as_tool"]
    assert (block["form"], block["supervisor_max_steps"], block["decoding"]["max_tokens"]) == ("paper_era", 10, 500) and "no_call_steps" not in block
    assert all(b["max_tokens"] == 500 for b in ep.completions()) and meta["served"] == {"context_length": 131072, "cache_report": None}
    prompt = ep.completions()[0]["prompt"]
    # the 23 examples, the line of the format that names the question, and the task's own question
    assert "CausalDiagnosisAgent" not in prompt and sum(1 for line in prompt.splitlines() if line.startswith("Question")) == 23 + 2
    assert rows[0]["end_reason"] == "finish" and rows[0]["react_steps"] == "2"


@pytest.mark.parametrize("row, fields, reason", [("base", {"max_steps": 2}, "max_steps"), ("priced", {"switches": {"budget": 1}}, "budget"),
                                                 ("real_time", {"horizon": 3}, "horizon")])
def test_the_step_cap_the_budget_and_the_horizon_of_the_row_still_end_the_episode(tmp_path, monkeypatch, row, fields, reason):
    if reason == "horizon":
        work = [*step("I wait for the rig.", "wait_for_notification", "timeout=10")]
        agent = "TaskAgent"
    elif reason == "budget":
        work = [*step("I read the history.", "history", _history_args())]
        agent = "IoTAgent"
    else:
        work = [*step("I need the sites tool.", "sites", "{}"), *step("I need the assets.", "assets", "site_name=MAIN")]
        agent = "IoTAgent"
    # after the ending call the sub-agent tries one more call, which the row refuses, and the run stops there
    script = [*step("I need to work on the asset.", agent, "request=work on the asset"), *work,
              *step("I continue.", "sites" if agent == "IoTAgent" else "get_asset_profile", "{}"), *finish("never reached"), REVIEW_OK]
    _, rows, _, ep = _play(tmp_path, monkeypatch, script, [], config=(row, fields))
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == reason, rows[0]
    # the run stops at the answer that ended the episode, or at the refusal of the next call where that answer does not say so
    assert len(ep.completions()) in (2 + len(work), 2 + len(work) + 2) and ep.script[-3:] == [*finish("never reached"), REVIEW_OK]
    assert rows[0]["react_steps"] == "2"


def test_a_supervisor_step_without_a_counted_call_takes_the_no_call_step_of_the_driver(tmp_path, monkeypatch):
    """A supervisor step whose sub-agent finishes without a tool call is an iteration without a counted call, which takes the
    episode's no_call step through the entry the driver hands the runner, one step of the row's step cap, as for every design. The
    control is the same play with the runner built without that entry, which takes none."""
    script = [*step("I need the sites.", "IoTAgent", "request=list the sites"), *finish("No site is listed yet"), REVIEW_OK,
              *step("I should end the task.", "TaskAgent", "request=end the task with the summary no repair was made"),
              *step("I need to end the task.", END_TASK, "summary=no repair was made")]
    (tmp_path / "entry").mkdir()
    run_dir, rows, _, _ = _play(tmp_path / "entry", monkeypatch, script, ["--config", "base"])
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == "finish" and _no_calls(run_dir) == 1, rows[0]
    runner_class = agent_runs.agent_as_tool_runner_class()
    monkeypatch.setattr(agent_runs, "agent_as_tool_runner_class", lambda: _without_entry(runner_class))
    (tmp_path / "control").mkdir()
    run_dir, rows, _, _ = _play(tmp_path / "control", monkeypatch, script, ["--config", "base"])
    assert rows[0]["error"] == "" and rows[0]["end_reason"] == "finish" and _no_calls(run_dir) == 0, rows[0]


def test_the_port_refuses_the_rigs_action_server(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "stand-in-key")
    with pytest.raises(SystemExit, match="AssetOpsBench"):
        run_episodes.main(["--config", "static", "--agent", "agent_as_tool", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1",
                           "--seeds", "1", "--out", str(tmp_path / "rig")])


def test_agent_as_tool_without_an_api_key_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(SystemExit):
        run_episodes.main(["--config", "base", "--agent", "agent_as_tool", "--model", "m", "--base-url", "http://127.0.0.1:9/v1", "--seeds", "1",
                           "--out", str(tmp_path / "nokey")])



@pytest.mark.parametrize("context_length, diagnostics, refused", [(32768, True, True), (49152, True, False), (32768, False, False)],
                         ids=["with the MCP below 48k", "control: with the MCP at 48k", "control: without the MCP below 48k"])
def test_a_row_with_the_mcp_refuses_a_served_context_below_48k_before_its_run_folder(tmp_path, monkeypatch, context_length, diagnostics, refused):
    script = [*step("I should end the task.", "TaskAgent", "request=end the task with the summary nothing found"),
              *step("I need to end the task.", END_TASK, "summary=nothing found")]
    args = ["--config", "base", *([] if diagnostics else ["--no-diagnostics"])]
    if refused:
        with pytest.raises(SystemExit, match=f"needs a served context of at least {sub_agent_registry.SERVED_CONTEXT_MIN} tokens, and the "
                                             f"endpoint serves a context of {context_length} tokens"):
            _play(tmp_path, monkeypatch, script, args, context_length=context_length)
        assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
        return
    _, rows, meta, _ = _play(tmp_path, monkeypatch, script, args, context_length=context_length)
    assert rows[0]["end_reason"] == "finish" and meta["served"] == {"context_length": context_length, "cache_report": None}


def test_the_served_record_and_the_refusal_read_the_context_that_the_endpoint_names():
    """served_facts reads max_model_len of the model's card in the endpoint's listing, and None where the card names none or no
    endpoint answers. A row with the CausalDiagnosis MCP refuses an unnamed context as it refuses one below 49152 tokens, and a row
    without it is never refused, the control."""
    from CausalDiagnosis.experiments.episode_runners.run_episodes import runner_setup
    from CausalDiagnosis.utils import frameworks

    with ScriptedEndpoint([], model=MODEL, cards=[{"id": MODEL, "object": "model"}]) as ep:
        assert runner_setup.served_facts(ep.base_url, MODEL) == {"context_length": None, "cache_report": None}
    with ScriptedEndpoint([], model=MODEL, context_length=sub_agent_registry.SERVED_CONTEXT_MIN) as ep:
        assert runner_setup.served_facts(ep.base_url, MODEL) == {"context_length": sub_agent_registry.SERVED_CONTEXT_MIN, "cache_report": None}
    assert runner_setup.served_facts("http://127.0.0.1:9/v1", MODEL) == runner_setup.served_facts(None, MODEL) == {"context_length": None, "cache_report": None}
    with_mcp, without = ("iot", "fmsr", "wo", "task", "causal_diagnosis"), ("iot", "fmsr", "wo", "task")
    for agent in ("agent_as_tool", "plan_execute"):
        paper = frameworks.code("AGENTS", agent)
        assert paper.served_refusal(with_mcp, {"context_length": None}).endswith("the endpoint names no context length")
        assert paper.served_refusal(with_mcp, {"context_length": 49151}).endswith("the endpoint serves a context of 49151 tokens")
        assert paper.served_refusal(with_mcp, {"context_length": 49152}) is None
        assert all(paper.served_refusal(without, served) is None for served in ({"context_length": None}, {"context_length": 1024}, None))
