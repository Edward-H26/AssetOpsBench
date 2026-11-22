"""The paper-era Plan-Execute port without a model: a scripted stand-in for SGLang's completion routes answers every model
call and the stand-in MCP servers of the Agent-As-Tool port's tests answer every tool call. The tests check the paper era
behaviour the port keeps: IBM's prompts byte for byte, the planner and its reviewer with the invalid plan fed back, the
steps on their agents with the outputs of their dependencies, the summarization step with its retries, the agents and
their examples, the stop when the environment ends the episode, the step count and the record of a run without a result."""
from CausalDiagnosis.utils.paths import PACKAGE
import asyncio
import dataclasses
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
AAT_TESTS = PACKAGE / "frameworks" / "AssetOpsBench" / "agents" / "Agent_As_Tool_Paper" / "tests"

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests import stub_servers  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import in_context_examples as examples, react_tool_adapter, sub_agent_registry  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import CompletionRoute, RunStopped, activated  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.runner import PlanExecutePaperRunner
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import prompts
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.runner import run_settings
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import workflow  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import runner as runner_module  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.react.agents import ReactReflectAgent  # noqa: E402

PAPER = json.loads((HERE / "data" / "paper_prompts.json").read_text(encoding="utf-8"))
VALID = '{"status": "Valid", "reasoning": "The plan covers the question.", "suggestions": "None."}\n(END OF RESPONSE)'
INVALID = '{"status": "Invalid", "reasoning": "Step 1 names no site.", "suggestions": "Name the site MAIN."}\n(END OF RESPONSE)'
END = "\n(END OF RESPONSE)"
EXAMPLE_START = stub_servers.EXAMPLE_START  # the example start of the runs over the stand-in servers


def plan(*steps):
    """A plan in the planner's format from (task, agent, dependency) triples."""
    lines = []
    for n, (task, agent, dependency) in enumerate(steps, start=1):
        lines += [f"## Step {n}", f"#Task{n}: {task}", f"#Agent{n}: {agent}", f"#Dependency{n}: {dependency}", f"#ExpectedOutput{n}: the answer", ""]
    return "\n".join(lines)


def agent_step(thought, tool, arguments, answer):
    """The calls of one step agent: one tool call, Finish and ReActXen's review."""
    return [*step(thought, tool, arguments), *finish(answer), REVIEW_OK]


def run(launchers, script, tmp_path, question="What IoT sites are available?"):
    with ScriptedEndpoint(script) as ep:
        runner = PlanExecutePaperRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in",  # pragma: allowlist secret
                                        log_path=tmp_path / "react_log.txt", example_start=EXAMPLE_START)
        result = asyncio.run(runner.run(question))
    return runner, result, ep


def test_the_planner_reviewer_and_summarizer_prompts_are_ibms_byte_for_byte():
    assert workflow.planner_prompt(PAPER["question"], PAPER["descriptions"], "", "") == PAPER["planner_first"]
    assert workflow.planner_prompt(PAPER["question"], PAPER["descriptions"], PAPER["plan"], PAPER["prev_review"]) == PAPER["planner_after_invalid"]
    route = CompletionRoute("http://127.0.0.1:9/v1", "local", "m")
    seen = []
    route.generate = lambda prompt, **kw: seen.append((prompt, kw)) or {"generated_text": VALID}
    with activated(route):
        workflow.review(PAPER["question"], PAPER["descriptions"], PAPER["plan"], "m")
        route.generate = lambda prompt, **kw: seen.append((prompt, kw)) or {"generated_text": "MAIN"}
        workflow.summarize(PAPER["user_input"], "m")
    assert seen[0][0] == PAPER["reviewer"] and seen[0][1]["stop"] == [END]
    assert seen[1][0] == PAPER["summarizer"] and seen[1][1]["stop"] == [END] and seen[1][1]["temperature"] == 0.0


def test_a_valid_plan_runs_its_step_on_the_agent_and_the_summary_answers(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("List the IoT sites.", "IoT Data Download", "None")), VALID,
              *agent_step("I need the sites.", "sites", "{}", "The sites are MAIN"), "MAIN"]
    runner, result, ep = run(launchers, script, tmp_path)
    assert result.error is None and result.end == "finish" and result.answer == "MAIN"
    assert result.plan_steps == 1 and len(result.plans) == 1 and result.plans[0]["review"]["status"] == "Valid"
    assert [(h["task_number"], h["agent_name"], h["response"]) for h in result.history] == [
        (1, "IoT Data Download", "The sites are MAIN"), (2, "SummarizationAgent", "MAIN")]
    assert [(r["agent"], r["server"], r["tool"]) for r in result.trajectory] == [("IoTAgent", "iot", "sites")]
    assert [c["agent"] for c in runner.calls] == ["planner", "reviewer", "IoTAgent", "IoTAgent", "IoTAgent", "IoTAgent", "IoTAgent", "summarizer"]
    prompts = ep.completions()
    # the planner lists the agents of the row under AgentHive's names with their task examples, the Question lines of their few-shots
    assert "(1) Agent name: IoT Data Download" in prompts[0]["prompt"] and "(2) Agent name: TaskAgent" in prompts[0]["prompt"]
    assert "Tasks that agent can solve:\n1. " + workflow.task_examples("\n\n".join(examples.IOT_RIG))[0] in prompts[0]["prompt"]
    # the step agent carries the few-shot examples of the Agent-As-Tool port's IoT sub-agent
    assert examples.IOT_RIG[0].splitlines()[0] in prompts[2]["prompt"]
    assert "List the IoT sites.\n" in prompts[2]["prompt"]
    # the summary reads the question and every step's output
    assert prompts[-1]["prompt"].endswith("Question:\nWhat IoT sites are available?\n\nContext:\nThe sites are MAIN\n\n\nAnswer:\n")


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the answer as served"])
def test_a_step_agent_reads_the_answer_of_a_server_on_one_line(tmp_path, monkeypatch, on_one_line):
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "one_line", lambda text: text)
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("List the assets of MAIN.", "IoT Data Download", "None")), VALID,
              *agent_step("I need the assets of MAIN.", "assets", "site_name=MAIN", "The assets are CHL-A100"), "CHL-A100"]
    _, result, ep = run(launchers, script, tmp_path)
    served = result.trajectory[0]["response"]
    assert "\n" in served  # FastMCP indents the JSON of an answer, which the trajectory keeps as served
    shown = json.dumps(json.loads(served)) if on_one_line else served
    assert f"\nObservation 1: {shown}\nThought 2:" in ep.completions()[4]["prompt"]


def test_an_invalid_plan_goes_back_to_the_planner_with_the_reason_and_the_suggestion(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    first = plan(("List the sites.", "IoT Data Download", "None"))
    second = plan(("List the sites of MAIN.", "IoT Data Download", "None"))
    script = [first, INVALID, second, VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"), "MAIN"]
    _, result, ep = run(launchers, script, tmp_path)
    assert [p["review"]["status"] for p in result.plans] == ["Invalid", "Valid"] and result.plan == second
    replanned = ep.completions()[2]["prompt"]
    assert "Invalid plan:\n" + first in replanned
    assert "Reason why this plan is invalid: Step 1 names no site." in replanned and "Suggestion for improvement: Name the site MAIN." in replanned


def test_a_review_without_a_status_keeps_the_plan_of_its_round_as_the_archive_does(tmp_path):
    """A reviewer reply without a status fails its round when the status is read, after the plan became prev_plan, so the next
    planner prompt shows that plan as the invalid one, as PlanningReviewWorkflow.generate_steps of the archive does."""
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    first = plan(("List the sites.", "IoT Data Download", "None"))
    second = plan(("List the sites of MAIN.", "IoT Data Download", "None"))
    no_status = '{"reasoning": "Step 1 names no site.", "suggestions": "Name the site MAIN."}' + END
    script = [first, no_status, second, VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"), "MAIN"]
    _, result, ep = run(launchers, script, tmp_path)
    assert result.plans[0]["plan"] == first and result.plans[1]["error"].startswith("KeyError") and result.plan == second
    assert "Invalid plan:\n" + first in ep.completions()[2]["prompt"]


def test_after_five_invalid_plans_the_last_one_runs(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    plans = [plan((f"List the sites, try {n}.", "IoT Data Download", "None")) for n in range(5)]
    script = [x for p in plans for x in (p, INVALID)] + [*agent_step("I need the sites.", "sites", "{}", "MAIN"), "MAIN"]
    _, result, _ = run(launchers, script, tmp_path)
    assert len(result.plans) == workflow.PLANS == 5 and result.plan == plans[-1] and result.end == "finish"


def test_a_step_reads_the_outputs_of_its_dependencies_and_an_unknown_agent_falls_back_to_the_first(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task", "fmsr"))
    steps = plan(("List the sites.", "IoT Data Download", "None"), ("Read the failure modes.", "No Such Agent", "[#S1]"))
    script = [steps, VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"),
              *agent_step("I need the assets.", "assets", "site_name=MAIN", "CHL-A100"), "CHL-A100 at MAIN"]
    _, result, ep = run(launchers, script, tmp_path)
    assert [h["agent_name"] for h in result.history] == ["IoT Data Download", "IoT Data Download", "SummarizationAgent"]
    second_step = ep.completions()[7]["prompt"]
    assert "Read the failure modes.\n\nContext:\nMAIN\n" in second_step


def test_the_summary_retries_at_a_higher_temperature_until_it_answers(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("List the sites.", "IoT Data Download", "None")), VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"),
              "Not enough information available.", "Not enough information available.", "MAIN"]
    _, result, ep = run(launchers, script, tmp_path)
    assert [c["temperature"] for c in ep.completions()[-3:]] == [0.0, 0.1, 0.2] and result.answer == "MAIN"


def test_a_summary_that_never_answers_leaves_the_last_step_as_the_answer(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("List the sites.", "IoT Data Download", "None")), VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"),
              *["Not enough information available."] * 3]
    _, result, _ = run(launchers, script, tmp_path)
    assert result.history[-1]["response"] == "" and result.answer == "MAIN"


def test_the_reviewer_reads_a_reply_by_hand_when_it_is_not_json():
    reply = 'Here it is: {"status": "Valid", "reasoning": "fine", "suggestions": "none",}'
    assert workflow.parse_review(reply)["status"] == "Error"
    assert workflow.parse_review_by_hand(reply) == {"status": "Valid", "reasoning": "fine", "suggestions": "none"}


def test_the_run_stops_when_the_environment_ends_the_episode(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("End the task with the summary done.", "TaskAgent", "None"), ("List the sites.", "IoT Data Download", "None")), VALID,
              *step("I need to end the task.", react_tool_adapter.agent_name("finish"), "summary=done")]
    runner, result, ep = run(launchers, script, tmp_path)
    assert result.end == "episode_over" and result.error is None
    assert [(r["server"], r["tool"]) for r in result.trajectory] == [("task", "finish")]
    assert len(ep.completions()) == 4 and len(runner.calls) == 4  # nothing after the call that ended the episode


def test_the_agents_and_their_examples_are_those_of_the_agent_as_tool_port():
    settings = run_settings(("iot", "fmsr", "wo", "task", "causal_diagnosis"))
    assert settings["agents"] == ["IoT Data Download", "TaskAgent", "Failure Mode and Sensor Relevancy Expert for Industrial Asset",
                                  "Work Order Management", "CausalDiagnosisAgent"]
    assert settings["plans"] == 5 and settings["step_agent_max_steps"] is None and settings["form"] == sub_agent_registry.MCP_FORM
    bare = run_settings(("iot", "fmsr", "wo", "task"))
    assert "causal_diagnosis" not in str(bare["agents"])
    assert (bare["plans"], bare["step_agent_max_steps"], bare["form"], bare["decoding"]["max_tokens"]) == (5, 6, "paper_era", 500)


class RecordingAgent:
    """A stand-in for the step agent that keeps the keywords it was built with and answers without a model call."""
    seen: list = []

    def __init__(self, **keywords):
        RecordingAgent.seen.append(keywords)
        self.answer, self.per_round_info, self.step_n, self.metric = "The sites are MAIN", [], 1, None

    def run(self):
        return None


def test_a_step_agent_runs_with_the_settings_of_the_archived_wrapper(tmp_path, monkeypatch):
    options = runner_module.STEP_AGENT_OPTIONS
    assert options == {"max_steps": 6, "num_reflect_iteration": 1, "handle_context_length_overflow": True,
                       "apply_loop_detection_check": True, "log_structured_messages": True, "early_stop": True}
    # Agent-Ask is not passed, and the engine leaves it off
    assert inspect.signature(ReactReflectAgent.__init__).parameters["enable_agent_ask"].default is False
    RecordingAgent.seen = []
    monkeypatch.setattr(runner_module, "ReactReflectAgent", RecordingAgent)
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    _, result, _ = run(launchers, [plan(("List the IoT sites.", "IoT Data Download", "None")), VALID, "MAIN"], tmp_path)
    assert result.answer == "MAIN" and len(RecordingAgent.seen) == 1
    keywords = RecordingAgent.seen[0]
    assert {name: keywords[name] for name in options} == options
    # beside the wrapper's keywords, the switch that ReActXen adds: the step agent stops at its first step after the episode ended
    assert set(keywords) == set(options) | {"question", "key", "cbm_tools", "react_llm_model_id", "reflect_llm_model_id", "react_example",
                                            "stop_when"}
    assert keywords["stop_when"].__func__ is runner_module.Runtime.episode_ended


def test_the_planner_knows_an_agent_by_the_archive_only_when_it_carries_the_papers_description(tmp_path, monkeypatch):
    specs = {s.name: s for s in sub_agent_registry.registration(("iot", "task"), "rig")}
    assert runner_module.plan_agent(specs["IoTAgent"]) == runner_module.AGENT_HIVE["IoTAgent"]
    assert runner_module.plan_agent(specs["TaskAgent"]) == ("TaskAgent", specs["TaskAgent"].description)
    other = dataclasses.replace(specs["IoTAgent"], description="Measures the unit on request.")
    assert runner_module.plan_agent(other) == ("IoTAgent", "Measures the unit on request.")
    # a run over that agent shows the planner the agent's own name and description, and the plan names it so
    monkeypatch.setattr(sub_agent_registry, "registration", lambda servers, surface: [other, specs["TaskAgent"]])
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("List the sites.", "IoTAgent", "None")), VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"), "MAIN"]
    _, result, ep = run(launchers, script, tmp_path)
    planner = ep.completions()[0]["prompt"]
    assert "(1) Agent name: IoTAgent\nAgent description: Measures the unit on request." in planner and "IoT Data Download" not in planner
    assert result.error is None and [h["agent_name"] for h in result.history] == ["IoTAgent", "SummarizationAgent"]


def test_the_steps_of_a_run_are_the_plan_steps_it_reached(tmp_path):
    def play(name, script):
        """One run on servers and a log folder of its own, so that the end of one episode does not reach the next run."""
        folder = tmp_path / name
        folder.mkdir()
        return run(stub_servers.write_launchers(folder / "servers", ("iot", "task")), script, folder)[1]

    two = plan(("List the sites.", "IoT Data Download", "None"), ("List the sites again.", "IoT Data Download", "None"))
    result = play("returned", [two, VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"),
                               *agent_step("I need the sites.", "sites", "{}", "MAIN"), "MAIN"])
    assert (result.executed_steps, result.plan_steps, len(result.history)) == (2, 2, 3)
    assert result.history[-1]["agent_name"] == workflow.SUMMARY_AGENT
    # the environment ends the episode inside the first of two steps: that step counts, and nothing was logged
    result = play("ended", [plan(("End the task with the summary done.", "TaskAgent", "None"), ("List the sites.", "IoT Data Download", "None")),
                            VALID, *step("I need to end the task.", react_tool_adapter.agent_name("finish"), "summary=done")])
    assert result.end == "episode_over" and (result.executed_steps, result.plan_steps, result.history) == (1, 2, [])
    # a plan without a readable step: no step, and the summarization alone
    result = play("no_step", ["I cannot plan this question.", VALID, "MAIN"])
    assert (result.executed_steps, result.plan_steps) == (0, 0) and [h["agent_name"] for h in result.history] == [workflow.SUMMARY_AGENT]


def test_a_run_without_a_result_leaves_its_record_and_its_count(tmp_path, monkeypatch):
    # a step that is cut counts as reached and logs nothing
    def cut(user_input):
        raise RunStopped("cancelled")

    route = CompletionRoute("http://127.0.0.1:9/v1", "local", "m")
    flow = workflow.Workflow("q", [workflow.PlanAgent("A", "first", [], lambda user_input: "one"), workflow.PlanAgent("B", "second", [], cut)], "m", route)
    flow.plan = plan(("Do the first thing.", "A", "None"), ("Do the second thing.", "B", "None"))
    flow.steps = workflow.plan_steps(flow.plan, flow.agents)
    with pytest.raises(RunStopped):
        flow.execute()
    assert flow.reached == 2 and [h["agent_name"] for h in flow.history] == ["A"]
    # before a run the runner holds neither a count nor a record
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    fresh = PlanExecutePaperRunner(launchers, base_url="http://127.0.0.1:9/v1", api_key="local", model="m")
    assert fresh.steps_reached() is None and fresh.record() is None
    # a run that fails after its workflow returns no result and still leaves both
    def fail(**keywords):
        raise OSError("the trajectory folder is gone")

    monkeypatch.setattr(runner_module, "persist_trajectory", fail)
    script = [plan(("List the IoT sites.", "IoT Data Download", "None")), VALID, *agent_step("I need the sites.", "sites", "{}", "The sites are MAIN"), "MAIN"]
    with ScriptedEndpoint(script) as ep:
        runner = PlanExecutePaperRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", log_path=tmp_path / "react_log.txt",  # pragma: allowlist secret
                                        example_start=EXAMPLE_START)
        with pytest.raises(OSError):
            asyncio.run(runner.run("What IoT sites are available?"))
    record = runner.record()
    assert runner.steps_reached() == 1 and record["end"] == "no_result" and (record["executed_steps"], record["plan_steps"]) == (1, 1)
    assert record["plans"][0]["review"]["status"] == "Valid" and [h["agent_name"] for h in record["history"]] == ["IoT Data Download", "SummarizationAgent"]
    assert [a["agent"] for a in record["step_agents"]] == ["IoTAgent"] and record["plan"].startswith("## Step 1")


def test_the_fixture_names_its_source():
    source = PAPER["source"]
    assert source["repository"] == "IBM/AssetOpsBench" and source["branch"] == "archive/legacy-tmp" and source["license"] == "Apache-2.0"
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"]) and source["commit"] in prompts.__doc__
    assert len(source["files"]) == 3 and all(name.startswith("src/tmp/agent_hive/") for name in source["files"])


# ----- the written examples and the client of the route -----

def test_the_plan_execute_port_reads_the_same_written_examples(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("List the IoT sites.", "IoT Data Download", "None")), VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"), "MAIN"]
    runner, result, ep = run(launchers, script, tmp_path)
    assert result.error is None and result.end == "finish"
    written = examples.diagnosis_example(["alpha", "beta", "evidence_ids"], EXAMPLE_START)
    question = written.splitlines()[0][len("Question:"):].strip()
    # the planner lists the question line of the written diagnosis example among the TaskAgent's task examples
    planner = ep.completions()[0]["prompt"]
    assert "(2) Agent name: TaskAgent" in planner and question in planner and "compressor=" not in planner
    # a step agent of the TaskAgent carries the written example, the same text that the Agent-As-Tool port's TaskAgent reads
    script = [plan(("End the task with the summary done.", "TaskAgent", "None")), VALID,
              *step("I need to end the task.", react_tool_adapter.agent_name("finish"), "summary=done")]
    (tmp_path / "second").mkdir()
    _, ended, ep = run(launchers, script, tmp_path / "second")
    assert ended.end == "episode_over" and written in ep.completions()[2]["prompt"]


def test_a_runner_over_a_diagnosis_tool_without_an_example_start_stops_before_its_first_model_call(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    with ScriptedEndpoint([]) as ep:
        runner = PlanExecutePaperRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in")
        with pytest.raises(ValueError, match="example start"):
            asyncio.run(runner.run("What IoT sites are available?"))
        assert ep.completions() == [] and runner.route._client.is_closed  # no model call, and the client of the route closed


def test_the_route_closes_its_client_when_the_run_ends(tmp_path, monkeypatch):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [plan(("List the IoT sites.", "IoT Data Download", "None")), VALID, *agent_step("I need the sites.", "sites", "{}", "MAIN"), "MAIN"]
    runner, result, _ = run(launchers, script, tmp_path)
    assert result.end == "finish" and runner.route._client.is_closed

    def fail(**keywords):
        raise OSError("the trajectory folder is gone")

    monkeypatch.setattr(runner_module, "persist_trajectory", fail)
    with ScriptedEndpoint(script) as ep:
        failed = PlanExecutePaperRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", example_start=EXAMPLE_START)
        assert not failed.route._client.is_closed
        with pytest.raises(OSError):
            asyncio.run(failed.run("What IoT sites are available?"))
    assert failed.route._client.is_closed


class Silent:
    """An endpoint that takes every connection and never answers, a hung model service."""

    def __init__(self):
        import socket

        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(64)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.socket.getsockname()[1]}/v1"

    def close(self):
        self.socket.close()


def test_a_run_whose_model_never_answers_returns_at_the_rows_deadline_and_leaves_no_thread(tmp_path):
    """No client of the port has a time limit of its own: on a service that never answers the run ends at the row's deadline
    (CausalDiagnosis.utils.model_request_retry), its client closes and no thread of the run is left running a call."""
    import threading
    import time

    from CausalDiagnosis.utils import model_request_retry as retry

    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    silent = Silent()
    try:
        before = {t.ident for t in threading.enumerate()}
        deadline = time.monotonic() + 3.0
        runner = PlanExecutePaperRunner(launchers, base_url=silent.base_url, api_key="local", model="granite-stand-in",  # pragma: allowlist secret
                                        example_start=EXAMPLE_START)
        started = time.monotonic()
        with retry.episode_deadline(lambda: deadline):
            result = asyncio.run(runner.run("What IoT sites are available?"))
        ended = time.monotonic()
    finally:
        silent.close()
    assert result.end == "error" and ended >= deadline and ended - started < 3.0 + 2.0
    assert runner.route._client.is_closed
    assert not [t for t in threading.enumerate() if t.ident not in before and t.is_alive() and t.name.startswith("asyncio_")]


def test_each_planning_round_goes_to_the_drivers_no_call_entry(monkeypatch):
    """A planning round calls no tool of the rig, so each round, a valid one included, goes to the driver's
    no_call entry; the archive's five rounds and its reviewer stay as they are."""
    import contextlib

    verdicts = iter([{"status": "Invalid", "reasoning": "r", "suggestions": "s"}] * 2 + [{"status": "Valid", "reasoning": "r", "suggestions": "s"}])
    monkeypatch.setattr(workflow, "watsonx_llm", lambda prompt, **_: {"generated_text": "#Task1: read\n#Agent1: IoTAgent\n#Dependency1: None\n#ExpectedOutput1: sites"})
    monkeypatch.setattr(workflow, "review", lambda *args, **kwargs: next(verdicts))
    route = type("Route", (), {"agent": staticmethod(lambda name: contextlib.nullcontext())})()
    ended = []
    agent = workflow.PlanAgent("IoTAgent", "IoT data", "", lambda request: "")
    flow = workflow.Workflow("Which sites?", [agent], "granite-stand-in", route, round_ended=lambda: ended.append(len(ended) + 1))
    flow.make_plan()
    assert ended == [1, 2, 3] and len(flow.plans) == 3 and workflow.PLANS == 5


@pytest.mark.parametrize("calls, entries_made, over", [
    ((), 2, False),  # one planning round and the pass, neither with a counted call
    (((sub_agent_registry.CAUSAL_SERVER, "ranked"),), 2, False),  # a call of the CausalDiagnosis MCP is no counted call
    ((("iot", "sites"),), 1, False),  # the control: a pass with a call of a server of the environment takes no step
    ((), 2, True),  # the step of the pass ends the episode: the run ends with the end the environment gave
], ids=["no call", "MCP alone", "action server", "pass ends the episode"])
def test_the_pass_of_the_step_agents_without_a_counted_call_takes_one_step(monkeypatch, calls, entries_made, over):
    """Each planning round takes its step as built, and the pass of the step agents with the summary,
    one iteration of its own in IBM's flow, goes to the driver's no_call entry when its whole subtree made no counted call."""
    class Route:
        stop_reason = None

        def stop(self, reason):
            self.stop_reason = self.stop_reason or reason

        def stopped(self):
            return self.stop_reason is not None

    entries = []
    runtime = runner_module.Runtime(Route(), None, {}, None, None, on_idle=lambda: entries.append(1) or (over and len(entries) == 2))

    class Flow:
        def __init__(self, question, agents, model, route, round_ended=None):
            self.round_ended, self.steps, self.plan, self.reached, self.history, self.plans = round_ended, [], "", 0, [], []

        def make_plan(self):
            self.round_ended()

        def execute(self):
            runtime.trajectory.extend({"response": response, "server": server} for server, response in calls)

        def answer(self):
            return ""

    monkeypatch.setattr(runner_module.workflow, "Workflow", Flow)
    runner = object.__new__(runner_module.PlanExecutePaperRunner)
    runner.step_agents, runner._model, runner.route = [], "granite-stand-in", runtime.route
    runner._agents = lambda runtime, step_agents: []
    result = runner._play("Which sites?", runtime)
    assert len(entries) == entries_made
    assert result.end == ("episode_over" if over else "finish") and runtime.route.stop_reason == ("episode_over" if over else None)



# ----- the limits of a run with the CausalDiagnosis MCP -----

ROWS = {"with the MCP": ("iot", "causal_diagnosis"), "control: without it": ("iot",)}
RIG_ASSET, RIG_START = "hyd_1", {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130}  # a start with one faulty component


def run_on(servers, script, tmp_path):
    """A run over the stand-in servers of a row with its no_call entries counted: (runner, result, endpoint, entries)."""
    launchers = stub_servers.write_launchers(tmp_path / "servers", servers)
    entries = []
    with ScriptedEndpoint(script) as ep:
        runner = PlanExecutePaperRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", log_path=tmp_path / "react_log.txt",  # pragma: allowlist secret
                                        example_start=RIG_START, example_asset=RIG_ASSET, on_idle=lambda: entries.append(1) and False)
        result = asyncio.run(runner.run("What IoT sites are available?"))
    return runner, result, ep, entries


@pytest.mark.parametrize("servers", ROWS.values(), ids=ROWS.keys())
def test_on_a_run_with_the_mcp_the_step_agents_have_no_step_limit_and_every_step_without_a_counted_call_is_charged(tmp_path, servers, monkeypatch):
    """One planning round, one step agent that calls sites, a counted call, and writes Finish, and the summary. With the MCP the
    round and the step agent's Finish take a no_call step each, and the pass with the summary none, since it took steps; every
    request asks for the room the served context leaves its prompt, and the step agent has no step limit of its own. Without it the
    round alone is charged, every request carries 500 and the step agent keeps AgentHive's six steps."""
    built = []
    real = runner_module.ReactReflectAgent
    monkeypatch.setattr(runner_module, "ReactReflectAgent", lambda **keywords: built.append(keywords["max_steps"]) or real(**keywords))
    script = [plan(("List the IoT sites.", "IoT Data Download", "None")), VALID, *agent_step("I need the sites.", "sites", "{}", "The sites are MAIN"), "MAIN"]
    runner, result, ep, entries = run_on(servers, script, tmp_path)
    mcp = "causal_diagnosis" in servers
    assert result.error is None and result.end == "finish" and [r["tool"] for r in result.trajectory] == ["sites"]
    assert entries == ([1, 1] if mcp else [1])
    assert built == [sub_agent_registry.NO_STEP_CAP if mcp else runner_module.STEP_AGENT_STEPS]
    bodies = ep.completions()
    assert [b["max_tokens"] for b in bodies] == ([131072 - len(b["prompt"].split()) - 2 for b in bodies] if mcp else [500] * len(bodies))
