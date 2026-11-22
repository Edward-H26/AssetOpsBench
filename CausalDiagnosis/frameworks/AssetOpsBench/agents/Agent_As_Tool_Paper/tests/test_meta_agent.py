"""The Agent-As-Tool port without a model: a scripted stand-in for SGLang's completion routes answers every model call
and stand-in MCP servers answer every tool call. The tests check the paper era behaviour the port keeps: the request of
watsonx's raw generation, the supervisor calling a sub-agent whose tool runs on the servers of the row, ReActXen's step
counter, Finish ending the run, the registration of the sub-agents per server, the supervisor prompt and examples of IBM's
own sample, and the stop when the environment ends the episode."""
import asyncio
import json
import sys
import threading
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests import engine_forms  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests import stub_servers  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.supervisor_runner import MetaAgentRunner  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import agent_builders, react_tool_adapter, sub_agent_registry  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import sub_agent_runtime, supervisor_runner  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import CompletionRoute, activated  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import text_completion_route as completion  # noqa: E402

THOUGHT_STOP = ["\nAction", "\nObservation", "\nQuestion"]
EXAMPLE_START = stub_servers.EXAMPLE_START  # the example start of the runs over the stand-in servers


def questions(text):
    """ReActXen's count of examples: the lines that start with Question."""
    return sum(1 for line in text.splitlines() if line.strip().startswith("Question"))


def test_the_route_sends_the_request_of_watsonx_raw_generation_and_keeps_the_matched_stop():
    with ScriptedEndpoint([" I need the list of sites.\nAction 1: IoTAgent"]) as ep:
        route = CompletionRoute(ep.base_url, "local", "granite-stand-in")
        out = route.generate("Question: what sites are there\nThought 1:", temperature=0.0, max_tokens=500, stop=THOUGHT_STOP)
        assert route.count_tokens("one two three") == 3 and route.context_length() == 131072
    assert out["generated_text"] == " I need the list of sites.\nAction" and out["input_token_count"] == 7
    body = ep.completions()[0]
    assert body["prompt"] == "Question: what sites are there\nThought 1:" and body["model"] == "granite-stand-in"
    assert body["temperature"] == 0.0 and body["max_tokens"] == 500 and body["min_tokens"] == 1
    assert body["stop"] == THOUGHT_STOP and body["no_stop_trim"] is True
    record = route.calls[0]
    assert record["prompt"] == body["prompt"] and record["reply"] == out["generated_text"]
    assert record["agent"] == "supervisor" and record["kind"] == "thought" and record["stop"] == THOUGHT_STOP
    assert record["finish_reason"] == "stop" and record["matched_stop"] == "\nAction" and record["temperature"] == 0.0
    assert {"input_tokens", "output_tokens", "seconds", "max_tokens", "model"} <= set(record)
    assert isinstance(record["started_at"], float) and record["started_at"] <= record["ended_at"]  # wall time


def test_a_sampling_temperature_replaces_the_papers_on_every_call_and_is_recorded():
    with ScriptedEndpoint([" A thought\nAction"]) as ep:
        route = CompletionRoute(ep.base_url, "local", "granite-stand-in", temperature=0.7)
        route.generate("Thought 1:", temperature=0.0, max_tokens=500, stop=THOUGHT_STOP)
    assert ep.completions()[0]["temperature"] == 0.7 and route.calls[0]["temperature"] == 0.7


def test_the_context_length_is_the_served_one_even_after_a_token_count():
    """SGLang's /v1/tokenize reports the tokenizer's model_max_length, 2^63 - 1 for granite, as max_model_len."""
    with ScriptedEndpoint([], tokenizer_max_len=2**63 - 1) as ep:
        route = CompletionRoute(ep.base_url, "local", "granite-stand-in")
        assert route.count_tokens("one two") == 2 and route.context_length() == 131072


@pytest.mark.parametrize("adapter_first", [False, True])
def test_a_fine_tuned_run_reads_the_context_length_of_its_base_model(adapter_first):
    """SGLang v0.5.19 lists each adapter as a card of its own without a context length, beside the base model's, and a
    fine-tuned run names its model base:adapter."""
    base = {"id": "granite-stand-in", "object": "model", "max_model_len": 131072}
    adapter = {"id": "lora-stand-in", "object": "model", "parent": "granite-stand-in", "max_model_len": None}
    with ScriptedEndpoint([], cards=[adapter, base] if adapter_first else [base, adapter]) as ep:
        route = CompletionRoute(ep.base_url, "local", "granite-stand-in:lora-stand-in")
        assert route.context_length() == 131072


def run(launchers, script, tmp_path, surface="rig", question="What IoT sites are available?"):
    with ScriptedEndpoint(script) as ep:
        runner = MetaAgentRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", surface=surface,  # pragma: allowlist secret
                                 log_path=tmp_path / "react_log.txt", example_start=EXAMPLE_START)
        result = asyncio.run(runner.run(question))
    return runner, result, ep


def test_the_supervisor_calls_a_sub_agent_whose_tool_runs_and_finish_ends_the_run_at_the_papers_counter(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [*step("I need the list of sites.", "IoTAgent", "request=list all the sites"),
              *step("I need to call the sites tool.", "sites", "{}"),
              *finish("The sites are MAIN"), REVIEW_OK,
              *finish("The sites are MAIN"), REVIEW_OK]
    runner, result, ep = run(launchers, script, tmp_path)
    assert result.error is None and result.end == "finish" and result.answer == "The sites are MAIN"
    assert result.react_steps == 3  # one sub-agent call and Finish end at 3, as in IBM's sample mistral-large 0001
    assert [(r["agent"], r["server"], r["tool"], r["step"], r["turn"]) for r in result.trajectory] == [("IoTAgent", "iot", "sites", 1, 1)]
    assert stub_servers.calls(tmp_path / "servers") == [{"server": "iot", "tool": "sites", "arguments": {}}]
    assert [(c["agent"], c["kind"]) for c in runner.calls] == [
        ("supervisor", "thought"), ("supervisor", "action"), ("IoTAgent", "thought"), ("IoTAgent", "action"), ("IoTAgent", "thought"),
        ("IoTAgent", "action"), ("IoTAgent", "review"), ("supervisor", "thought"), ("supervisor", "action"), ("supervisor", "review")]
    # the supervisor reads the sub-agent's answer as the paper's wrappers return it, a JSON string of the JSON reply
    assert 'Observation 1: "{\\n  \\"answer\\": \\"The sites are MAIN' in ep.completions()[7]["prompt"]
    # the sub-agent's prompt lists the tools of its server in ReActXen's form
    assert "(1) sites[] : List the IoT sites." in ep.completions()[2]["prompt"]
    assert result.react_trajectory["overall_metric"]["per_round_info"][-1]["step"] == 3
    assert [s["agent"] for s in result.sub_agents] == ["IoTAgent"] and result.sub_agents[0]["react_steps"] == 3
    assert (tmp_path / "react_log.txt").exists()


def test_an_invalid_action_and_a_second_call_each_count_one_step(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [" I should look around.\nAction", " look around somehow\nObservation",
              *step("I need the list of sites.", "IoTAgent", "request=list all the sites"),
              *step("I need to call the sites tool.", "sites", "{}"), *finish("The sites are MAIN"), REVIEW_OK,
              *finish("The sites are MAIN"), REVIEW_OK]
    _, result, _ = run(launchers, script, tmp_path)
    assert result.error is None and result.end == "finish" and result.react_steps == 4


def test_the_causal_diagnosis_sub_agent_is_registered_only_with_its_server():
    bare = sub_agent_registry.registration(("iot", "fmsr", "wo", "task"), "rig")
    served = sub_agent_registry.registration(("iot", "fmsr", "wo", "task", "causal_diagnosis"), "rig")
    assert [s.name for s in bare] == ["IoTAgent", "TaskAgent", "FMSRAgent", "WOAgent"]
    assert [s.name for s in served] == ["IoTAgent", "TaskAgent", "FMSRAgent", "WOAgent", "CausalDiagnosisAgent"]
    assert questions(sub_agent_registry.merged_examples(bare)) == 23 and questions(sub_agent_registry.merged_examples(served)) == 28


def test_a_registered_surface_is_served_and_the_built_in_ones_keep_their_specs():
    built_in = {name: tuple(specs) for name, specs in sub_agent_registry.SURFACES.items()}
    with pytest.raises(ValueError, match="unknown surface 'made_up': rig or repro$"):
        sub_agent_registry.registration((), "made_up")
    spec = sub_agent_registry.SURFACES["rig"][0]
    try:
        assert sub_agent_registry.register_surface("made_up", (spec,), unused=("vibration",)) == (spec,)
        assert sub_agent_registry.registration(("iot", "vibration"), "made_up") == [spec]
        with pytest.raises(ValueError, match="the surface 'rig' is already registered"):
            sub_agent_registry.register_surface("rig", ())
        with pytest.raises(ValueError, match="unknown surface 'other': rig or repro or made_up$"):
            sub_agent_registry.registration((), "other")
    finally:
        sub_agent_registry.SURFACES.pop("made_up", None)
        sub_agent_registry.UNUSED.pop("made_up", None)
    assert {name: tuple(specs) for name, specs in sub_agent_registry.SURFACES.items()} == built_in


def test_the_rig_refuses_a_server_no_paper_sub_agent_maps_to():
    with pytest.raises(ValueError, match="hydraulic_rig"):
        sub_agent_registry.registration(("hydraulic_rig", "causal_diagnosis"), "rig")


def _supervisor_for_inspection(specs, question, base_url, model):
    """The supervisor as a run builds it, without servers or a run, so that its prompt can be read."""
    with activated(CompletionRoute(base_url, "local", model)):
        return agent_builders.build_supervisor(question, specs, None, model)


def test_the_supervisor_of_the_reproduction_has_the_prompt_and_the_28_examples_of_ibms_sample(tmp_path):
    paper = json.loads((HERE / "data" / "paper_supervisor_prompt.json").read_text(encoding="utf-8"))
    specs = sub_agent_registry.registration(("iot", "utilities", "fmsr", "tsfm", "wo"), "repro")
    assert [s.name for s in specs] == ["IoTAgent", "TSFMAgent", "FMSRAgent", "WOAgent", "RuleLogicAgent", "jsonreader"]
    assert sub_agent_registry.merged_examples(specs) == paper["demonstration"] and questions(paper["demonstration"]) == 28
    with ScriptedEndpoint([]) as ep:
        supervisor = _supervisor_for_inspection(specs, "What IoT sites are available?", ep.base_url, "granite-stand-in")
        assert supervisor._get_system_prompt() == paper["system_prompt"]
        assert supervisor.max_steps == 10 and supervisor.num_reflect_iteration == 1


def test_the_run_stops_when_the_environment_ends_the_episode(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [*step("I should end the task.", "TaskAgent", "request=end the task with the summary done"),
              *step("I need to end the task.", react_tool_adapter.agent_name("finish"), "summary=done"),
              *finish("The task is ended"), REVIEW_OK, *finish("Done"), REVIEW_OK]
    runner, result, ep = run(launchers, script, tmp_path)
    assert result.end == "episode_over" and result.error is None
    assert [(r["server"], r["tool"]) for r in result.trajectory] == [("task", "finish")]
    assert result.react_steps == 2  # the supervisor's step 1 counts as done, as ReActXen's counter would after it
    assert len(ep.completions()) == 4 and len(runner.calls) == 4  # nothing after the call that ended the episode


SITES_RUN = [*step("I need the list of sites.", "IoTAgent", "request=list all the sites"),
             *step("I need to call the sites tool.", "sites", "{}"), *finish("The sites are MAIN"), REVIEW_OK,
             *finish("The sites are MAIN"), REVIEW_OK]


def test_a_run_without_a_result_keeps_the_supervisors_counter(tmp_path, monkeypatch):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    fresh = MetaAgentRunner(launchers, base_url="http://127.0.0.1:9/v1", api_key="local", model="m")
    assert fresh.steps_reached() is None  # no supervisor was built

    # a run that fails after its workflow returns no result and keeps the counter the workflow ended with
    def fail(**keywords):
        raise OSError("the trajectory folder is gone")

    with monkeypatch.context() as patched, ScriptedEndpoint(SITES_RUN) as ep:
        patched.setattr(supervisor_runner, "persist_trajectory", fail)
        failed = MetaAgentRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", log_path=tmp_path / "failed.txt",  # pragma: allowlist secret
                                 example_start=EXAMPLE_START)
        with pytest.raises(OSError):
            asyncio.run(failed.run("What IoT sites are available?"))
    assert failed.steps_reached() == failed.react_steps == 3  # the count of the same run with a result

    # a run that is cut during its first supervisor step, as a time limit cuts it: the first model call answers only once
    # the cut has stopped the route, the thread ends at its next call, and the step in flight counts as done
    reached = threading.Event()

    def held_until_the_cut(body):
        reached.set()
        cut.route._stopped.wait(60)
        return " I need the list of sites.\nAction"

    async def cut_at_the_first_model_call():
        task = asyncio.ensure_future(cut.run("What IoT sites are available?"))
        while not reached.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await task

    with ScriptedEndpoint([held_until_the_cut, *SITES_RUN[1:]]) as ep:
        cut = MetaAgentRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", log_path=tmp_path / "cut.txt",  # pragma: allowlist secret
                              example_start=EXAMPLE_START)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(cut_at_the_first_model_call())
        assert len(ep.completions()) == 1  # no model call after the cut
    assert cut.react_steps == 2 and cut.steps_reached() == 2

    # a thread that outlives the join has stored no counter yet: the counter of the supervisor in flight plus one
    cut.react_steps, cut.supervisor = None, types.SimpleNamespace(step_n=4)
    assert cut.steps_reached() == 5


def test_the_settings_name_the_step_limit_of_every_sub_agent():
    """The paper's limits without the CausalDiagnosis MCP; with it the form of its rows, no limit of an agent's own, the room of
    every request as max_tokens and the rule of the no_call steps."""
    bare = supervisor_runner.run_settings(("iot", "fmsr", "wo", "task"))
    assert list(bare["sub_agent_max_steps"]) == ["IoTAgent", "TaskAgent", "FMSRAgent", "WOAgent"] == bare["sub_agents"]
    assert bare["sub_agent_max_steps"] == {"IoTAgent": 15, "TaskAgent": 10, "FMSRAgent": 10, "WOAgent": 20}
    assert (bare["form"], bare["supervisor_max_steps"], bare["decoding"]["max_tokens"]) == ("paper_era", 10, 500) and "no_call_steps" not in bare
    served = supervisor_runner.run_settings(("iot", "fmsr", "wo", "task", "causal_diagnosis"))
    assert served["sub_agent_max_steps"] == dict.fromkeys(["IoTAgent", "TaskAgent", "FMSRAgent", "WOAgent", "CausalDiagnosisAgent"])
    assert (served["form"], served["supervisor_max_steps"]) == (sub_agent_registry.MCP_FORM, None) != (bare["form"], bare["supervisor_max_steps"])
    assert served["decoding"]["max_tokens"] == completion.REPLY_ROOM and served["no_call_steps"] == sub_agent_registry.NO_CALL_RULE
    # on the reproduction surface the four paper agents have a limit, and the stand-in rule agent and the JSON reader have none
    repro = supervisor_runner.run_settings(("iot", "utilities", "fmsr", "tsfm", "wo"), "repro")
    assert repro["sub_agent_max_steps"] == {"IoTAgent": 15, "TSFMAgent": 6, "FMSRAgent": 10, "WOAgent": 20}
    assert repro["sub_agents"] == ["IoTAgent", "TSFMAgent", "FMSRAgent", "WOAgent", "RuleLogicAgent", "jsonreader"]


# ----- the glue of the two ports -----

def test_a_tool_name_shown_to_an_agent_holds_no_word_that_the_engine_reads_as_its_own():
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.react.utils import extract_action_and_input

    assert react_tool_adapter.agent_name("finish") == "end_episode" and react_tool_adapter.RESERVED == {"finish": "end_episode"}
    # a tool whose shown name would hold a word of the engine and has no entry in RESERVED is refused, and the message names it
    for name, word in (("finish_episode_now", "finish"), ("get final answer", "final answer"), ("self-ask_tool", "self-ask"),
                       ("agent-ask_tool", "agent-ask")):
        assert word in react_tool_adapter.ENGINE_WORDS
        with pytest.raises(ValueError, match=repr(word)):
            react_tool_adapter.agent_name(name)
    # every tool of the stand-in servers passes (the tools of the rig row pass in the replay of test_in_context_examples.py)
    for name in ("sites", "assets", "history", "get_asset_profile", "submit_diagnosis", "finish", "get_failure_modes", "list_workorders",
                 "rank_root_causes", "current_date_time", "get_ai_tasks"):
        assert react_tool_adapter.agent_name(name)
    # the engine's own action reader: a one-line step that holds the word finish is read as the agent's own Finish
    stop = ["\nObservation", "\nThought", "\nQuestion"]
    assert extract_action_and_input("finish_episode(summary=done)", stop)["action"] == "Finish"
    assert extract_action_and_input("end_episode(summary=done)", stop)["action"] != "Finish"


def test_the_copied_pair_pattern_stands_in_the_engine():
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.react import agents as engine_agents

    source = Path(engine_agents.__file__).read_text(encoding="utf-8")
    assert engine_forms.ENGINE_PAIRS in source
    assert 'value = match[1].split(", ")' in source and "value.strip()" in source  # a key=[...] pair is stripped as a list


def test_an_object_argument_written_as_key_value_reaches_the_tool_as_an_object():
    got = {}
    schema = {"properties": {"description": {"type": "string"}, "aob_source": {"anyOf": [{"type": "object"}, {"type": "null"}], "default": None}},
              "required": ["description"]}
    tool = react_tool_adapter.mcp_tool("generate_work_order", "", schema, lambda arguments: got.update(arguments) or "ok")
    text = 'description=x, aob_source={"prediction_id": "pred/1", "evidence_ids": ["obs/2025-04-02T00:15:00/Supply Temperature"]}'
    tool.run(engine_forms.engine_split(text))
    assert got["aob_source"] == {"prediction_id": "pred/1", "evidence_ids": ["obs/2025-04-02T00:15:00/Supply Temperature"]}
    with pytest.raises(Exception, match="valid dictionary"):
        tool.run(engine_forms.engine_split("description=x, aob_source=not an object"))


def test_a_text_argument_written_as_no_value_or_as_a_list_goes_out_as_text():
    got = {}
    schema = {"properties": {"cooler": {"type": "integer"}, "evidence_ids": {"type": "string"},
                             "status": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}},
              "required": ["cooler", "evidence_ids"]}
    tool = react_tool_adapter.mcp_tool("submit_diagnosis", "", schema, lambda arguments: got.update(arguments) or "ok")

    def sent(written, key):
        got.clear()
        tool.run(engine_forms.engine_split(written) if not written.startswith("{") else json.loads(written))
        return got[key]

    # a required text: the engine's forms for no value go out as the empty text, a list as its items joined by commas. A list
    # reaches the wrapper only inside a JSON object input, since the engine refuses a key=[...] pair before any tool runs
    for written, expected in (("cooler=3, evidence_ids=None", ""), ('cooler=3, evidence_ids=""', ""), ('{"cooler": 3, "evidence_ids": []}', ""),
                              ('{"cooler": 3, "evidence_ids": null}', ""), ('{"cooler": 3, "evidence_ids": ["a", "b"]}', "a,b"),
                              ("cooler=3, evidence_ids=obs/1/Supply Temperature_mean", "obs/1/Supply Temperature_mean")):
        assert sent(written, "evidence_ids") == expected, written
    # an optional text left at no value stays no value, so that the server's default holds, and a list is its items
    for written, expected in (("cooler=3, evidence_ids=a, status=None", None), ('cooler=3, evidence_ids=a, status=""', None),
                              ('{"cooler": 3, "evidence_ids": "a", "status": []}', None), ('{"cooler": 3, "evidence_ids": "a", "status": ["OPEN", "APPR"]}', "OPEN,APPR")):
        assert sent(written, "status") == expected, written
    with pytest.raises(AttributeError):
        engine_forms.engine_split("cooler=3, evidence_ids=[a, b]")
    # a required text that the model did not write still fails, also when the engine dropped an empty pair
    for written in ("cooler=3", "cooler=3, evidence_ids="):
        with pytest.raises(Exception, match="Field required"):
            tool.run(engine_forms.engine_split(written))


HISTORY_ROWS = 200  # the rows of the answer below, so that the answer alone outgrows the served context of its run


def _sub_agent_run(answer, script, room=None):
    """An IoTAgent built as the port builds a sub-agent, over one tool of the adapter that answers ``answer``, or that calls it when
    it is a function of the arguments, run on a scripted endpoint. With room, the served context holds the agent's first prompt and room words more, so that a long answer outgrows
    it and the engine's overflow handler cuts the scratchpad after the call (agents.py, ReactAgent.run)."""
    schema = {"properties": {"site_name": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, "required": ["site_name"]}
    tool = react_tool_adapter.mcp_tool("history", "Return the telemetry of a site.", schema, answer if callable(answer) else lambda arguments: answer)
    spec = sub_agent_registry.SubAgentSpec("IoTAgent", "", "", ("iot",), max_steps=3)

    def build():
        return agent_builders.build_sub_agent(spec, "read the history of MAIN", [tool], "granite-stand-in", "")

    with ScriptedEndpoint(script) as ep:
        if room is not None:
            with activated(CompletionRoute(ep.base_url, "local", "granite-stand-in")):
                ep.cards[0]["max_model_len"] = len(build()._build_agent_prompt().split()) + room
        with activated(CompletionRoute(ep.base_url, "local", "granite-stand-in")):
            agent = build()
            agent.run()
    return agent


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the answer as served"])
def test_an_answer_over_many_lines_reaches_the_agent_on_one_line_so_the_overflow_handler_can_cut_it(monkeypatch, on_one_line):
    """An answer that outgrows the served context, as a page of telemetry does. The overflow handler of the engine cuts only the
    lines that open with Observation (agents/react/utils.py, truncate_scratchpad). On one line the answer is cut and the agent
    goes on to its Finish. As served, over many lines, the handler cuts the one line that opens the answer, is still over the
    context and raises IndexError, which ends the episode."""
    rows = [{"timestamp": f"2025-04-02T00:{n // 60:02d}:{n % 60:02d}", "value": n} for n in range(HISTORY_ROWS)]
    answer = json.dumps({"observations": rows}, indent=2)
    script = [*step("I need the history.", "history", "site_name=MAIN"), *finish("The history is read"), REVIEW_OK]
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "one_line", lambda text: text)
        with pytest.raises(IndexError, match="pop from empty list"):
            _sub_agent_run(answer, script, room=60)
        return
    agent = _sub_agent_run(answer, script, room=60)
    [shown] = [entry["raw_observation_output"] for entry in agent.json_log if entry.get("raw_observation_output")]
    assert "\n" not in shown and json.loads(shown) == json.loads(answer)
    assert "\nObservation 1: [truncated excerpt]\nThought 2:" in agent.scratchpad and agent.answer == "The history is read"


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the answer as served"])
def test_an_answer_with_a_receipt_line_reaches_the_agent_as_its_json_values_on_one_line(monkeypatch, on_one_line):
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "one_line", lambda text: text)
    body = json.dumps({"site_name": "MAIN", "observations": [{"timestamp": "2025-04-02T00:15:00", "value": 6.8}]}, indent=2)
    receipt = json.dumps({"price": 0, "budget_left": 10, "steps_left": 3}, separators=(",", ":"))
    tool = react_tool_adapter.mcp_tool("history", "", {"properties": {"site_name": {"type": "string"}}, "required": ["site_name"]},
                                       lambda arguments: body + "\n" + receipt)
    shown = tool.run({"site_name": "MAIN"})
    one_line = json.dumps(json.loads(body)) + " " + json.dumps(json.loads(receipt))
    assert shown == (one_line if on_one_line else body + "\n" + receipt)
    # an answer that is no JSON keeps its text with its line breaks replaced
    text = react_tool_adapter.mcp_tool("sites", "", {}, lambda arguments: "Error executing tool sites:\n  no database").run({})
    assert (text == "Error executing tool sites:   no database") == on_one_line


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the refusal as pydantic writes it"])
def test_a_refusal_of_the_argument_model_reaches_the_agent_on_one_line(monkeypatch, on_one_line):
    """The engine writes the text of a refused call into the scratchpad as its observation, and pydantic writes a refusal over
    several lines."""
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "one_line", lambda text: text)
    script = [*step("I need the history.", "history", "site_name=MAIN, limit=many"), *finish("No history"), REVIEW_OK]
    agent = _sub_agent_run("{}", script)
    [line] = [line for line in agent.scratchpad.split("\n") if line.startswith("Observation 1:")]
    assert "validation error for history_inputs" in line
    assert ("For further information visit" in line) == on_one_line


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the paper's indented text"])
def test_the_failure_reply_of_a_sub_agent_tool_reaches_the_supervisor_on_one_line(monkeypatch, on_one_line):
    if not on_one_line:
        monkeypatch.setattr(agent_builders, "one_line", lambda text: text)
    spec = sub_agent_registry.SubAgentSpec("RuleLogicAgent", "", "", kind="stub")
    reply = agent_builders.SubAgentTool(name="RuleLogicAgent", description="", spec=spec)._run("a rule")
    answer = json.loads(reply)
    assert answer["status"] == "failure" and answer["message"] == agent_builders.FAILURE_MESSAGE
    assert ("\n" not in reply) == on_one_line


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the answer as served"])
def test_the_supervisors_sub_agent_reads_the_answer_of_a_server_on_one_line(tmp_path, monkeypatch, on_one_line):
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "one_line", lambda text: text)
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [*step("I need the assets of MAIN.", "IoTAgent", "request=list the assets of MAIN"),
              *step("I need the assets tool.", "assets", "site_name=MAIN"), *finish("The assets are CHL-A100"), REVIEW_OK,
              *finish("The assets are CHL-A100"), REVIEW_OK]
    _, result, ep = run(launchers, script, tmp_path)
    served = result.trajectory[0]["response"]
    assert "\n" in served  # FastMCP indents the JSON of an answer, which the trajectory keeps as served
    shown = json.dumps(json.loads(served)) if on_one_line else served
    assert f"\nObservation 1: {shown}\nThought 2:" in ep.completions()[4]["prompt"]


@pytest.mark.parametrize("unquoted", [True, False], ids=["layer by layer", "control: the quotes kept"])
def test_a_text_argument_loses_its_outer_quotes_layer_by_layer(monkeypatch, unquoted):
    if not unquoted:
        monkeypatch.setattr(react_tool_adapter, "_unquoted", lambda value: value)
    got = {}
    schema = {"properties": {"work_type": {"type": "string"}, "evidence_ids": {"type": "string"}, "priority": {"type": "integer", "default": 3},
                             "site_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}},
              "required": ["work_type", "evidence_ids"]}
    tool = react_tool_adapter.mcp_tool("generate_work_order", "", schema, lambda arguments: got.update(arguments) or "ok")

    def sent(given):
        got.clear()
        tool.run(given)
        return got

    split = engine_forms.engine_split
    # double and single quotes, two layers, the items of a list given to a text, an optional text, any tool's text
    for given, key, expected in ((split('work_type="CM", evidence_ids=a'), "work_type", "CM"),
                                 (split("work_type='PM', evidence_ids=a"), "work_type", "PM"),
                                 (split('work_type=""CM"", evidence_ids=a'), "work_type", "CM"),
                                 ({"work_type": '"EM"', "evidence_ids": ['"obs/1/a"', "'obs/2/b'"]}, "evidence_ids", "obs/1/a,obs/2/b"),
                                 (split('work_type=CM, evidence_ids=a, site_id="MAIN"'), "site_id", "MAIN")):
        assert (sent(given)[key] == expected) == unquoted, (given, key)
    # a text without a matching outer pair passes unchanged, and the engine's forms for no value keep their reading
    for given, key, expected in ((split('work_type="CM, evidence_ids=a'), "work_type", '"CM'),
                                 (split('work_type=say "hi" now, evidence_ids=a'), "work_type", 'say "hi" now'),
                                 (split('work_type=CM, evidence_ids=a, site_id=""'), "site_id", None),
                                 (split('work_type=CM, evidence_ids=a, priority=2'), "priority", 2)):
        assert sent(given)[key] == expected, given
    # a plain string input reaches the first field as that field reads it
    modes = react_tool_adapter.mcp_tool("get_failure_modes", "", {"properties": {"asset_class": {"type": "string"}}, "required": ["asset_class"]},
                                        lambda arguments: got.update(arguments) or "ok")
    got.clear()
    modes.run('"chiller"')
    assert (got["asset_class"] == "chiller") == unquoted


@pytest.mark.parametrize("unquoted", [True, False], ids=["one pair off", "control: the quotes kept"])
def test_a_work_order_written_with_quotes_passes_the_work_order_checks_of_the_metering_layer(monkeypatch, unquoted):
    """The listed schema of AssetOpsBench's generate_work_order and the checks with which the metering layer refuses an order
    before the engine sees it (metering_layer._validated and _order_problem)."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import metering_layer as layer

    if not unquoted:
        monkeypatch.setattr(react_tool_adapter, "_unquoted", lambda value: value)
    schema = layer.SERVERS["wo"].mcp._tool_manager.get_tool("generate_work_order").parameters
    got = {}
    tool = react_tool_adapter.mcp_tool("generate_work_order", "", schema, lambda arguments: got.update(arguments) or "{}")
    tool.run(engine_forms.engine_split('description="inspect the pump", asset_num="hyd_1_pump", site_id="MAIN", priority=2, work_type="CM"'))
    assert (got["asset_num"], got["site_id"]) == (("hyd_1_pump", "MAIN") if unquoted else ('"hyd_1_pump"', '"MAIN"'))
    assert layer._order_problem(layer._validated("wo", "generate_work_order", got)) == (None if unquoted else "work_type")


def _quotes_off_before_peeling(value):
    """The control of the test below, _unquoted as it stood before the quotes were peeled layer by layer: one pair off when the
    first and the last character are the same quote, or, when that quote also stands inside and the first and the last comma
    item each carry a pair of their own, one pair off each item whose quote does not stand inside it."""
    def item_off(item):
        core = item.strip()
        if len(core) >= 2 and core[0] == core[-1] and core[0] in "\"'" and core[0] not in core[1:-1]:
            return item.replace(core, core[1:-1], 1)
        return item

    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
            items = stripped.split(",")
            if stripped[0] in stripped[1:-1] and all(item_off(item) != item for item in (items[0], items[-1])):
                return ",".join(item_off(item) for item in items)
            return stripped[1:-1]
    return value


# every text whose sent form the peeling changes, with that form: layers of either kind, a comma item peeled where the whole
# text keeps its pair or has none, and a free text that opens with a quoted word
PEELED = (('""CM""', "CM"), ("\"'CM'\"", "CM"), ("'\"CM\"'", "CM"), ('""""', ""), ('"obs/1/a", obs/2/b', "obs/1/a, obs/2/b"),
          ('obs/1/a, "obs/2/b"', "obs/1/a, obs/2/b"), ("'obs/1/a', \"obs/2/b\"", "obs/1/a, obs/2/b"), ('"go", he said', "go, he said"),
          ('"replace the seal, "urgent", then test"', '"replace the seal, urgent, then test"'), ('"pump" is "bad", "x"', '"pump" is "bad", x'),
          ("\"replace 'seal', 'gasket'\"", "replace 'seal', gasket"))
# every text that the peeling sends as written, since a quote of each of its layers also stands inside it in another way, and
# that the rule before it changed
AS_WRITTEN = ('"pump" is "bad"', "'the pump's seal'", '"replace the "seal", then test "pressure""', "'\"it's\"'", "\"'\"x'y\"'\"")
# the texts that both rules send alike
ALIKE = (('"CM"', "CM"), ("'CM'", "CM"), ("CM", "CM"), ('"the pump\'s seal"', "the pump's seal"), ("\"'it's'\"", "'it's'"), ('"a, b"', "a, b"),
         ('"worn, then cracked"', "worn, then cracked"), ('"obs/1/a", "obs/2/b"', "obs/1/a, obs/2/b"), ('"obs/1/a", obs/2/b, "wo/3"', "obs/1/a, obs/2/b, wo/3"),
         ('"a", "pump" is "bad", "x"', 'a, "pump" is "bad", x'), ('say "hi" now', 'say "hi" now'), ('"', '"'), ('"CM', '"CM'))


@pytest.mark.parametrize("rule", ["peeled", "before"], ids=["layer by layer", "control: the rule before the peeling"])
def test_a_text_and_each_comma_item_lose_their_quotes_layer_by_layer_unless_the_quote_stands_inside(monkeypatch, rule):
    """The pairs of the same quote that wrap a text one inside the other go layer by layer, as many as leave none of their
    quotes in what is left, so ""CM"" gives CM, and a pair whose quote also stands inside the text in another way stays, as in
    "pump" is "bad". Each comma item of what is left loses its quotes by the same rule, so a comma text of several quoted items,
    as a model can write evidence_ids on the rows with the CausalDiagnosis MCP, reaches the action server as its ids
    (Environment_Actions_MCP evidence_list)."""
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import evidence_list

    if rule == "before":
        monkeypatch.setattr(react_tool_adapter, "_unquoted", _quotes_off_before_peeling)
    got = {}
    schema = {"properties": {"evidence_ids": {"type": "string"}, "cooler": {"type": "integer"},
                             "diagnosis": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}},
              "required": ["evidence_ids", "cooler"]}
    tool = react_tool_adapter.mcp_tool("submit_diagnosis", "", schema, lambda arguments: got.update(arguments) or "ok")

    def sent(text):
        """The text as a required and as an optional text argument of the JSON object form."""
        got.clear()
        tool.run({"evidence_ids": text, "cooler": 3, "diagnosis": text})
        return got["evidence_ids"], got["diagnosis"]

    for text, expected in (*PEELED, *((text, text) for text in AS_WRITTEN)):
        assert (sent(text) == (expected, expected)) == (rule == "peeled"), text
    for text, expected in ALIKE:
        assert sent(text) == (expected, expected), text
    # the engine's key=value form, the items of a list and the engine's form for no value
    got.clear()
    tool.run(engine_forms.engine_split('evidence_ids="obs/1/a", "obs/2/b", cooler=3'))
    assert got["evidence_ids"] == "obs/1/a, obs/2/b" and evidence_list(got["evidence_ids"]) == ["obs/1/a", "obs/2/b"]
    got.clear()
    tool.run({"evidence_ids": ['""CM""', '"obs/1/a"', "'obs/2/b'"], "cooler": 3})
    assert (got["evidence_ids"] == "CM,obs/1/a,obs/2/b") == (rule == "peeled")
    assert sent('""') == ("", None)


class _TransportFailure(Exception):
    """A failure of the call itself, as a transport error of a session, not a refusal of the argument model."""


def _raising(failure):
    def call(arguments):
        raise failure
    return call


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the exception as raised"])
def test_an_exception_of_the_call_reaches_the_engine_on_one_line_and_keeps_its_type(monkeypatch, on_one_line):
    """The engine writes str() of an exception that a tool raises into the scratchpad (agents/react/agents.py, the call of a tool),
    and in its JSON branch it reads a TypeError apart, so the text goes on one line and the type stays."""
    import pydantic
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import RunStopped

    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "_failure_on_one_line", lambda failure: failure)
    schema = {"properties": {"site_name": {"type": "string"}}, "required": ["site_name"]}

    def raised(failure):
        with pytest.raises(BaseException) as caught:
            react_tool_adapter.mcp_tool("sites", "", schema, _raising(failure)).run({"site_name": "MAIN"})
        return caught.value

    for failure in (_TransportFailure("the session closed\n  while the answer was read"), TypeError("unexpected argument\n  site")):
        text = str(failure)
        got = raised(failure)
        assert type(got) is type(failure) and (str(got) == react_tool_adapter.one_line(text) and "\n" not in str(got)) == on_one_line
    # a class whose text comes from fields of its own goes on one line as a ValueError
    with pytest.raises(pydantic.ValidationError) as refusal:
        pydantic.TypeAdapter(int).validate_python("many")
    assert "\n" in str(refusal.value)
    got = raised(refusal.value)
    assert (type(got) is ValueError and str(got) == react_tool_adapter.one_line(str(refusal.value))) == on_one_line
    # an exception whose text is already one line, and the port's stop of a run, a BaseException, pass as raised
    for failure in (KeyError("site"), _TransportFailure("closed"), RunStopped("episode_over")):
        args = failure.args
        assert raised(failure) is failure and failure.args == args


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the refusal as pydantic writes it"])
def test_a_refusal_of_a_sub_agent_tools_argument_model_reaches_the_supervisor_on_one_line(tmp_path, monkeypatch, on_one_line):
    """A sub-agent tool of the supervisor takes one argument, request, so an input under another key is refused by its argument
    model before any sub-agent runs, and the engine writes the refusal into the supervisor's scratchpad."""
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "one_line", lambda text: text)
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    script = [*step("I need the list of sites.", "IoTAgent", "question=list all the sites"), *finish("No sites"), REVIEW_OK]
    runner, result, _ = run(launchers, script, tmp_path)
    [line] = [line for line in runner.supervisor.scratchpad.split("\n") if line.startswith("Observation 1:")]
    assert "validation error for IoTAgentInputs" in line and result.sub_agents == [] and result.answer == "No sites"
    assert ("For further information visit" in line) == on_one_line


ENGINE_FAILURE = "Execution failed while calling the Tool. Details: "  # the engine's text for an exception of a tool


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the exception as raised"])
@pytest.mark.parametrize("written", ["site_name=MAIN", '{"site_name": "MAIN"}', "MAIN"], ids=["key=value", "JSON object", "plain string"])
def test_an_exception_of_the_call_reaches_the_agents_scratchpad_on_one_line_in_every_input_form(monkeypatch, on_one_line, written):
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "_failure_on_one_line", lambda failure: failure)
    text = "the session closed while the answer was read:\n  connection reset\n  by the server"
    script = [*step("I need the history.", "history", written), *finish("No history"), REVIEW_OK]
    agent = _sub_agent_run(_raising(_TransportFailure(text)), script)
    [line] = [line for line in agent.scratchpad.split("\n") if line.startswith("Observation 1:")]
    assert line.endswith(ENGINE_FAILURE + (react_tool_adapter.one_line(text) if on_one_line else text.split("\n")[0]))
    assert (f"\n{line}\nThought 2:" in agent.scratchpad) == on_one_line and agent.answer == "No history"


@pytest.mark.parametrize("on_one_line", [True, False], ids=["one line", "control: the exception as raised"])
def test_a_long_exception_of_the_call_reaches_the_agent_on_one_line_so_the_overflow_handler_can_cut_it(monkeypatch, on_one_line):
    """An exception whose text outgrows the served context, as a transport error that carries the page it was reading. On one
    line the overflow handler cuts it as it cuts an answer (agents/react/utils.py, truncate_scratchpad) and the agent goes on to
    its Finish. Over many lines the handler cuts the one line that opens it, is still over the context and raises IndexError."""
    rows = [{"timestamp": f"2025-04-02T00:{n // 60:02d}:{n % 60:02d}", "value": n} for n in range(HISTORY_ROWS)]
    failure = _TransportFailure("the session closed while the answer was read:\n" + json.dumps({"observations": rows}, indent=2))
    script = [*step("I need the history.", "history", "site_name=MAIN"), *finish("No history"), REVIEW_OK]
    if not on_one_line:
        monkeypatch.setattr(react_tool_adapter, "_failure_on_one_line", lambda failure: failure)
        with pytest.raises(IndexError, match="pop from empty list"):
            _sub_agent_run(_raising(failure), script, room=60)
        return
    agent = _sub_agent_run(_raising(failure), script, room=60)
    assert "\nObservation 1: [truncated excerpt]\nThought 2:" in agent.scratchpad and agent.answer == "No history"


def test_one_supervisor_example_written_for_the_port_chains_two_delegations_and_the_counts_stay():
    import re

    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import in_context_examples as examples

    bare = sub_agent_registry.registration(("iot", "fmsr", "wo", "task"), "rig")
    served = sub_agent_registry.registration(("iot", "fmsr", "wo", "task", "causal_diagnosis"), "rig")
    assert questions(sub_agent_registry.merged_examples(bare)) == 23 and questions(sub_agent_registry.merged_examples(served)) == 28
    agents = {s.name for s in served}
    chained = []
    for text in (examples.TASK_SUPERVISOR, examples.CAUSAL_SUPERVISOR):
        for example in (e for e in text.split("\n\n") if e.strip()):
            actions = re.findall(r"^Action (\d+): (.+)$", example, re.M)
            if len([a for _, a in actions if a in agents]) >= 2:
                chained.append(example)
    assert len(chained) == 1
    example = chained[0]
    assert "Finish" not in example and example.rstrip().splitlines()[-1].startswith("Observation 2:")
    question = re.search(r"^Question: (.*)$", example, re.M).group(1)
    first = json.loads(re.search(r"^Observation 1: (.*)$", example, re.M).group(1))
    second_request = re.search(r"^Action Input 2: (.*)$", example, re.M).group(1)
    passed = [site for site in first["sites"] if site in second_request and site not in question]
    assert passed, (second_request, question)


def test_a_run_stops_before_its_first_model_call_when_the_example_start_does_not_fit_the_tool_contract(tmp_path):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    for start, message in (({"alpha": 4, "gamma": 1}, "gamma"), (None, "example start")):
        with ScriptedEndpoint([]) as ep:
            runner = MetaAgentRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", example_start=start)
            with pytest.raises(ValueError, match=message):
                asyncio.run(runner.run("What IoT sites are available?"))
            assert ep.completions() == [] and runner.route._client.is_closed  # no model call, and the client of the route closed
    # with a fitting start the TaskAgent's prompt holds the written diagnosis example
    script = [*step("I need to record the diagnosis.", "TaskAgent", "request=submit the diagnosis alpha 4, beta 7"),
              *step("I need to submit the diagnosis.", "submit_diagnosis", "alpha=4, beta=7, evidence_ids=None"),
              *finish("The diagnosis is submitted"), REVIEW_OK, *finish("The diagnosis is submitted"), REVIEW_OK]
    runner, result, ep = run(launchers, script, tmp_path)
    assert result.error is None and result.end == "finish"
    assert "Action Input 1: alpha=4, beta=7, evidence_ids=obs/2025-04-02T00:15:00/Supply Temperature" in ep.completions()[2]["prompt"]
    # the diagnosis without evidence went out with the empty text, and the stand-in recorded it
    assert {"server": "task", "tool": "submit_diagnosis", "arguments": {"alpha": 4, "beta": 7, "evidence_ids": ""}} in stub_servers.calls(tmp_path / "servers")


def test_the_route_closes_its_client_when_the_run_ends(tmp_path, monkeypatch):
    launchers = stub_servers.write_launchers(tmp_path / "servers", ("iot", "task"))
    runner, result, _ = run(launchers, SITES_RUN, tmp_path)
    assert result.end == "finish" and runner.route._client.is_closed

    # also after a run that raised once its agents' thread had ended
    def fail(**keywords):
        raise OSError("the trajectory folder is gone")

    with monkeypatch.context() as patched, ScriptedEndpoint(SITES_RUN) as ep:
        patched.setattr(supervisor_runner, "persist_trajectory", fail)
        failed = MetaAgentRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", example_start=EXAMPLE_START)
        assert not failed.route._client.is_closed
        with pytest.raises(OSError):
            asyncio.run(failed.run("What IoT sites are available?"))
    assert failed.route._client.is_closed


def test_the_port_hands_its_model_the_same_text_for_every_answer():
    """The runtime reads the end of an episode with the reader of
    CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts. On a table of answers (no receipt, a receipt with
    steps left, none left, the end event, the refusal after the end, an accepted finish and a refused one) the runtime hands the
    tool of the port the answer text byte for byte where the episode goes on, which the tool puts on one line for its model
    (react_tool_adapter.one_line), and stops the run on each answer that ends the episode, so a verdict that differs from the
    table on one answer fails the test."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import metering_receipts as receipts
    body = json.dumps({"time": 2})
    receipt = {"steps_left": 3, "budget_left": 10}
    answers = [("read", body, False, False),
               ("read", body + "\n" + json.dumps(receipt), False, False),
               ("read", body + "\n" + json.dumps(dict(receipt, steps_left=0)), False, True),
               ("wait", body + "\n" + json.dumps(dict(receipt, notifications=[{"event": "end", "reason": "budget"}])), False, True),
               ("read", json.dumps({"error": receipts.EPISODE_OVER + "; this call changed nothing"}), True, True),
               ("finish", json.dumps({"finished": True}), False, True),
               ("finish", json.dumps({"error": "a diagnosis comes first"}), True, False)]

    class Route:
        stop_reason = None

        def check(self):
            pass

        def stop(self, reason):
            self.stop_reason = reason

    class Answer:
        def __init__(self, text, is_error):
            self.reply = (text, is_error)

        async def call(self, server, tool, arguments, timeout_s=None):
            return self.reply

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        for tool, text, is_error, over in answers:
            runtime = sub_agent_runtime.Runtime(Route(), Answer(text, is_error), {}, loop, None)
            if over:
                with pytest.raises(sub_agent_runtime.RunStopped):
                    runtime.call_tool("supervisor", "server", tool, {})
            else:
                assert runtime.call_tool("supervisor", "server", tool, {}) == text
            assert runtime.route.stop_reason == ("episode_over" if over else None), text
            assert (runtime.trajectory[-1]["response"], runtime.trajectory[-1]["error"]) == (text, text if is_error else None)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
    assert sub_agent_runtime.split_receipt is receipts.split_receipt and sub_agent_runtime.ended is receipts.ended


def test_a_supervisor_step_without_a_counted_call_goes_to_the_drivers_no_call_entry():
    """A supervisor step whose sub-agent runs made no counted call goes to the driver's no_call entry, a step with
    one does not, and when the entry reports that the episode ended the run stops with the end episode_over."""
    class Route:
        stop_reason = None

        def stop(self, reason):
            self.stop_reason = self.stop_reason or reason

        def stopped(self):
            return self.stop_reason is not None

    entries = []
    runtime = sub_agent_runtime.Runtime(Route(), None, {}, None, None, on_idle=lambda: entries.append(1) or len(entries) == 2)
    runtime.supervisor = types.SimpleNamespace(step_n=1)
    assert runtime.supervisor_turn() is False and entries == []  # the first step ends no iteration
    runtime.supervisor.step_n = 2
    runtime.trajectory.append({"response": "MAIN", "server": "iot"})
    assert runtime.supervisor_turn() is False and entries == []  # the step before made a counted call
    runtime.supervisor.step_n = 3
    assert runtime.supervisor_turn() is False and entries == [1]  # no call: one entry, the episode goes on
    runtime.supervisor.step_n = 4
    assert runtime.supervisor_turn() is True and runtime.route.stop_reason == "episode_over"


def test_a_supervisor_step_whose_sub_agent_called_only_the_causal_mcp_takes_its_step():
    """A call of the CausalDiagnosis MCP is no counted call, so a supervisor step whose sub-agent only
    called it goes to the driver's no_call entry, as a step without any call does. The control: a step whose sub-agent called a
    server of the environment goes to no entry, and a call that got no answer counts as none."""
    class Route:
        stop_reason = None

        def stop(self, reason):
            self.stop_reason = self.stop_reason or reason

        def stopped(self):
            return self.stop_reason is not None

    entries = []
    runtime = sub_agent_runtime.Runtime(Route(), None, {}, None, None, on_idle=lambda: entries.append(1) and False)
    runtime.supervisor = types.SimpleNamespace(step_n=1)
    runtime.supervisor_turn()
    for step_n, record, charged in ((2, {"response": "ranked", "server": sub_agent_registry.CAUSAL_SERVER}, [1]),
                                    (3, {"response": "sites", "server": "iot"}, [1]),
                                    (4, {"response": None, "server": "wo"}, [1, 1])):
        runtime.trajectory.append(record)
        runtime.supervisor.step_n = step_n
        assert runtime.supervisor_turn() is False and entries == charged, (step_n, record)
    assert sub_agent_registry.CAUSAL_SERVER in [s for spec in sub_agent_registry.SURFACES["rig"] if spec.name == "CausalDiagnosisAgent" for s in spec.servers]


@pytest.mark.parametrize("waits, delivered", [((0.05, 5.0), True), ((0.05,), False)], ids=["deadline moved on", "control: deadline passed"])
def test_a_call_in_flight_waits_on_while_the_deadline_read_again_lies_ahead(monkeypatch, waits, delivered):
    """The wait of a counted call reads the deadline again at each expiry (retry.waits_to_deadline): an answer that arrives
    after the first read ran out, while a credit moved the deadline on, is delivered; when the deadline read again has passed,
    the call is given up with DeadlineReached and its record keeps no answer."""
    class Route:
        stop_reason = None

        def check(self):
            pass

        def stop(self, reason):
            self.stop_reason = reason

    class Slow:
        async def call(self, server, tool, arguments, timeout_s=None):
            await asyncio.sleep(0.3)
            return "sites", False

    monkeypatch.setattr(sub_agent_runtime.retry, "waits_to_deadline", lambda: iter(waits))
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        runtime = sub_agent_runtime.Runtime(Route(), Slow(), {}, loop, None)
        if delivered:
            assert runtime.call_tool("IoTAgent", "iot", "sites", {}) == "sites"
            assert runtime.trajectory[-1]["response"] == "sites"
        else:
            with pytest.raises(sub_agent_runtime.DeadlineReached):
                runtime.call_tool("IoTAgent", "iot", "sites", {})
            assert runtime.trajectory[-1]["response"] is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()



# ----- the limits of a run with the CausalDiagnosis MCP -----

# the runs over the stand-in servers with and without the CausalDiagnosis MCP, on the asset of the rig with a start that holds one
# faulty component, which the work order examples of a run with the MCP name (sub_agent_registry.mcp_row)
ROWS = {"with the MCP": ("iot", "causal_diagnosis"), "control: without it": ("iot",)}
RIG_ASSET, RIG_START = "hyd_1", {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130}


def run_on(servers, script, tmp_path, context_length=131072):
    """A run of the stand-in servers of a row, with its no_call entries counted: (runner, result, endpoint, entries)."""
    launchers = stub_servers.write_launchers(tmp_path / "servers", servers)
    entries = []
    with ScriptedEndpoint(script, context_length=context_length) as ep:
        runner = MetaAgentRunner(launchers, base_url=ep.base_url, api_key="local", model="granite-stand-in", log_path=tmp_path / "react_log.txt",  # pragma: allowlist secret
                                 example_start=RIG_START, example_asset=RIG_ASSET, on_idle=lambda: entries.append(1) and False)
        result = asyncio.run(runner.run("What IoT sites are available?"))
    return runner, result, ep, entries


def words(text):
    return len(text.split())  # the token count of the scripted endpoint


@pytest.mark.parametrize("servers", ROWS.values(), ids=ROWS.keys())
def test_on_a_run_with_the_mcp_every_request_asks_for_the_room_the_served_context_leaves_its_prompt(tmp_path, servers):
    """The completion route reads an absent max_tokens as 16, so a request that caps no reply names the room: the served context
    less the tokenize count of its prompt less 2, the most the engine generates (CompletionRoute.room_of). Without the MCP every
    request carries the paper's 500. Each model call records what its request sent."""
    runner, result, ep, _ = run_on(servers, list(SITES_RUN), tmp_path)
    bodies = ep.completions()
    assert result.end == "finish" and len(bodies) == len(runner.calls) == 10
    expected = [131072 - words(b["prompt"]) - 2 for b in bodies] if "causal_diagnosis" in servers else [500] * len(bodies)
    assert [b["max_tokens"] for b in bodies] == expected == [c["max_tokens"] for c in runner.calls]
    tokenized = [b["prompt"] for path, b in ep.requests if path.endswith("/v1/tokenize")]
    assert all(b["prompt"] in tokenized for b in bodies) == ("causal_diagnosis" in servers)


def test_a_prompt_that_leaves_no_room_ends_the_run_as_context_without_a_request():
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import CONTEXT_END, RunStopped

    with ScriptedEndpoint([" ok\nAction"], context_length=6) as ep:
        route = CompletionRoute(ep.base_url, "local", "granite-stand-in", reply_room=True)
        assert route.room_of("a b c") == 1  # 6 less 3 less 2
        with pytest.raises(RunStopped, match=CONTEXT_END):
            route.generate("a b c d", stop=THOUGHT_STOP)
        assert route.stop_reason == CONTEXT_END and ep.completions() == []
        control = CompletionRoute(ep.base_url, "local", "granite-stand-in")  # without the room the request goes out with the paper's 500
        control.generate("a b c d", stop=THOUGHT_STOP)
        assert [b["max_tokens"] for b in ep.completions()] == [500]


@pytest.mark.parametrize("servers", ROWS.values(), ids=ROWS.keys())
def test_on_a_run_with_the_mcp_every_react_step_without_a_counted_call_takes_one_no_call_step(tmp_path, servers):
    """The supervisor delegates to IoTAgent, which calls sites, a counted call, and writes Finish, and the supervisor writes
    Finish. With the MCP the sub-agent's Finish takes a no_call step and so does the supervisor's, whose subtree took none, while
    the supervisor step that delegated is charged nothing, since its subtree took steps: two entries. Without it the paper's rule
    charges the supervisor's Finish alone: one entry. No agent has a step limit of its own with the MCP."""
    runner, result, _, entries = run_on(servers, list(SITES_RUN), tmp_path)
    mcp = "causal_diagnosis" in servers
    assert result.end == "finish" and [r["tool"] for r in result.trajectory] == ["sites"]
    assert entries == ([1, 1] if mcp else [1])
    assert runner.supervisor.max_steps == (sub_agent_registry.NO_STEP_CAP if mcp else sub_agent_registry.SUPERVISOR_STEPS)
    assert result.sub_agents[0]["metric"]["max_steps"] == (sub_agent_registry.NO_STEP_CAP if mcp else 15)


def test_a_react_step_is_charged_once_however_often_its_agent_asks_to_stop():
    """ReActXen asks stop_when before each step and once after its round: the charge reads the agent's step counter, so a step is
    charged once, the step with a counted call is charged nothing, and after the episode ends nothing is charged."""
    class Route:
        stop_reason = None

        def stop(self, reason):
            self.stop_reason = self.stop_reason or reason

        def stopped(self):
            return self.stop_reason is not None

    entries = []
    runtime = sub_agent_runtime.Runtime(Route(), None, {"iot": [], sub_agent_registry.CAUSAL_SERVER: []}, None, None,
                                        on_idle=lambda: entries.append(1) or len(entries) == 3)
    charge = runtime.step_charge()
    charge.agent = types.SimpleNamespace(step_n=1)
    assert charge() is False and entries == []  # before the first step
    runtime.trajectory.append({"response": "sites", "server": "iot"})
    charge.agent.step_n = 2
    assert charge() is False and charge() is False and entries == []  # step 1 made a counted call, asked twice
    charge.agent.step_n = 3
    assert charge() is False and charge() is False and entries == [1]  # step 2 made none, charged once
    charge.agent.step_n = 5
    charge.close()  # two steps since, the second ends the episode
    assert entries == [1, 1, 1] and runtime.route.stop_reason == "episode_over" and runtime.acted_since((1, 0))
    charge.agent.step_n = 6
    assert charge() is True and entries == [1, 1, 1]
    assert sub_agent_runtime.Runtime(Route(), None, {"iot": []}, None, None).step_charge() is None  # control: without the MCP no charge
