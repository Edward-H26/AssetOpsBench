"""What the two ports record for the harness, without a model: a tool call without an answer, an error answer of a tool, a
run that its caller cuts, a failed request of the route, an error that crosses the sessions, and the request settings they
report. A scripted stand-in answers the model calls. The iot server is a stand-in that the test writes, whose one tool
answers, hangs or raises, and the task server is the stand-in of this folder."""
import asyncio
import inspect
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests import stub_servers  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.tests.scripted_endpoint import REVIEW_OK, ScriptedEndpoint, finish, step  # noqa: E402

from CausalDiagnosis.agents.utils import mcp_sessions  # noqa: E402
from CausalDiagnosis.utils import model_request_retry as retry  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.supervisor_runner import MetaAgentRunner
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import text_completion_route as completion  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import sub_agent_runtime  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.runner import PlanExecutePaperRunner  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import runner as plan_execute_runner  # noqa: E402

IOT_STAND_IN = '''
import asyncio

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stand-in-iot", log_level="ERROR")


@mcp.tool()
async def sites() -> dict:
    """List the IoT sites."""
    if MODE == "hang":
        await asyncio.sleep(60)
    if MODE == "raise":
        raise ValueError("the sites table is locked")
    return {"sites": ["MAIN"]}


mcp.run(transport="stdio")
'''
QUESTION = "What IoT sites are available?"
# the supervisor asks its IoT agent, which calls its one tool, answers and is reviewed, and the supervisor answers and is reviewed
SUPERVISED = [*step("I need the list of sites.", "IoTAgent", "request=list all the sites"), *step("I need to call the sites tool.", "sites", "{}"),
              *finish("The sites are MAIN"), REVIEW_OK, *finish("The sites are MAIN"), REVIEW_OK]
# the planner's one step, its review, the step agent with one tool call, and the summary
PLANNED = ["## Step 1\n#Task1: List the IoT sites.\n#Agent1: IoT Data Download\n#Dependency1: None\n#ExpectedOutput1: the sites\n",
           '{"status": "Valid", "reasoning": "The plan covers the question.", "suggestions": "None."}\n(END OF RESPONSE)',
           *step("I need the sites.", "sites", "{}"), *finish("The sites are MAIN"), REVIEW_OK, "MAIN"]


def launchers(folder, mode="answer"):
    """{server: launcher} of a run: the iot stand-in in the mode given and the task stand-in of this folder."""
    paths = stub_servers.write_launchers(folder, ("task",))
    paths["iot"] = Path(folder) / "iot_server.py"
    paths["iot"].write_text(f"MODE = {mode!r}\n{IOT_STAND_IN}", encoding="utf-8")
    return {"iot": paths["iot"], "task": paths["task"]}


def play(runner_class, paths, script, tmp_path, **settings):
    with ScriptedEndpoint(script) as endpoint:
        runner = runner_class(paths, base_url=endpoint.base_url, api_key="local", model="granite-stand-in", log_path=tmp_path / "react_log.txt",  # pragma: allowlist secret
                              example_start=stub_servers.EXAMPLE_START, **settings)
        result = asyncio.run(runner.run(QUESTION))
    return runner, result, endpoint


def route_on(handler):
    """A completion route whose client answers from a function of the request, without a socket."""
    return completion.CompletionRoute("http://endpoint.example/v1", "local", "granite-stand-in", client=httpx.Client(transport=httpx.MockTransport(handler)))


def _given_up_at_the_deadline(record, sent_at, deadline):
    """Whether the record of a tool call says that the call was given up at the deadline: no answer, the error of the session's
    wait (ToolTransportError) or of the port's (DeadlineReached), each naming the deadline, after at least the time that was
    left when the call was sent."""
    error = record["error"] or ""
    return (record["response"] is None and error.startswith(("ToolTransportError", "DeadlineReached")) and "deadline" in error
            and record["duration_ms"] / 1000 >= deadline - sent_at - 0.01)


def test_a_tool_call_without_an_answer_is_given_up_at_the_rows_deadline_and_ends_the_run(tmp_path, monkeypatch):
    """No tool call of a port has a time limit of its own: a call that never answers is given up when the deadline of the row
    passes, after the time that was left when it was sent, and the next model request, which no time is left for, ends the run.
    Outside the harness's cut both waits of the call read the deadline, the session's and the port's, and either may give the
    call up first. The control: an answered call, and a call given up before its time, are no call given up at the deadline."""
    sent = []
    send = mcp_sessions.EpisodeSessions.call

    async def call(self, server, tool, arguments):
        sent.append(time.monotonic())
        return await send(self, server, tool, arguments)

    monkeypatch.setattr(mcp_sessions.EpisodeSessions, "call", call)

    # the deadline falls one second after the call is sent, whatever the start of the servers took before it
    def deadline():
        return sent[0] + 1.0 if sent else time.monotonic() + 60.0

    with retry.episode_deadline(deadline):
        runner, result, endpoint = play(MetaAgentRunner, launchers(tmp_path / "servers", "hang"), SUPERVISED, tmp_path)
    (record,) = result.trajectory
    assert record["tool"] == "sites" and _given_up_at_the_deadline(record, sent[0], deadline()) and time.monotonic() >= deadline()
    assert result.end == "error" and result.error.startswith("DeadlineReached")
    sent.clear()
    answered_by = time.monotonic() + 60.0
    with retry.episode_deadline(lambda: answered_by):
        runner, answered, endpoint = play(MetaAgentRunner, launchers(tmp_path / "answered", "answer"), SUPERVISED, tmp_path / "answered")
    (record,) = answered.trajectory
    assert record["response"] is not None and not _given_up_at_the_deadline(record, sent[0], answered_by)
    early = {**record, "response": None, "duration_ms": 10.0,
             "error": "ToolTransportError: sites of iot gave no answer before the deadline of the episode"}
    assert not _given_up_at_the_deadline(early, sent[0], sent[0] + 3.0)


def test_the_servers_refusal_of_a_prompt_past_the_context_ends_the_run_and_is_not_retried():
    """SGLang answers a prompt that, with its 500 tokens, passes the served context with 400 BadRequest. The request
    is not retried and the run ends with the end "context"; a 400 of another kind stays an error of the run."""
    asked = []

    def refusing(request):
        asked.append(request.url.path)
        return httpx.Response(400, json={"object": "error", "type": "BadRequest", "code": 400, "message":
                                         "Requested token count exceeds the model's maximum context length of 4096 tokens."})

    route = route_on(refusing)
    with pytest.raises(completion.RunStopped, match="^context$"):
        route.generate("Thought 1:", stop=["\nAction"])
    assert asked == ["/v1/completions"] and route.stop_reason == "context"
    other = route_on(lambda request: httpx.Response(400, json={"object": "error", "message": "unknown field n"}))
    with pytest.raises(httpx.HTTPStatusError):
        other.generate("Thought 1:", stop=["\nAction"])
    assert other.stop_reason is None


def test_an_error_answer_of_a_tool_is_recorded_with_its_text_and_returned(tmp_path):
    runner, result, endpoint = play(MetaAgentRunner, launchers(tmp_path / "servers", "raise"), SUPERVISED, tmp_path)
    (record,) = result.trajectory
    assert record["response"] == record["error"] and "the sites table is locked" in record["error"]
    assert not record["error"].startswith("ToolTransportError")  # an error answer is an answer and no failure of the transport
    assert "the sites table is locked" in endpoint.completions()[4]["prompt"] and "Execution failed" not in endpoint.completions()[4]["prompt"]
    assert result.end == "finish" and result.error is None


@pytest.mark.parametrize("runner_class, module, script", [(MetaAgentRunner, sys.modules[MetaAgentRunner.__module__], SUPERVISED), (PlanExecutePaperRunner, plan_execute_runner, PLANNED)])
def test_both_ports_open_their_servers_through_the_session_module_once_per_run(tmp_path, monkeypatch, runner_class, module, script):
    opened = []

    def spy(server_paths):
        opened.append(dict(server_paths))
        return mcp_sessions.open_sessions(server_paths)

    monkeypatch.setattr(module, "open_sessions", spy)
    paths = launchers(tmp_path / "servers")
    runner, result, endpoint = play(runner_class, paths, script, tmp_path)
    assert opened == [paths] and result.end == "finish" and result.error is None
    assert [(r["server"], r["tool"], r["error"]) for r in result.trajectory] == [("iot", "sites", None)]


@pytest.mark.parametrize("runner_class, script", [(MetaAgentRunner, SUPERVISED), (PlanExecutePaperRunner, PLANNED)])
def test_an_error_that_crosses_the_sessions_reaches_the_caller_under_its_own_type(tmp_path, runner_class, script):
    """The model list names no context length, which the first agent of a run reads while the sessions are open. The
    Agent-As-Tool port builds its supervisor outside the guard of its play, so the error leaves run(). The Plan-Execute port
    builds its first step agent inside that guard, which keeps every error of a run as the run's error."""
    paths = launchers(tmp_path / "servers")
    cards = [{"id": "granite-stand-in", "object": "model", "max_model_len": None}]
    with ScriptedEndpoint(list(script), cards=cards) as endpoint:
        runner = runner_class(paths, base_url=endpoint.base_url, api_key="local", model="granite-stand-in", example_start=stub_servers.EXAMPLE_START)
        if runner_class is MetaAgentRunner:
            with pytest.raises(completion.RouteSetupError, match="names no context length for granite-stand-in"):
                asyncio.run(runner.run(QUESTION))
        else:
            result = asyncio.run(runner.run(QUESTION))
            assert (result.end, result.error) == ("error", f"RouteSetupError: {endpoint.base_url}/models names no context length for granite-stand-in")


def test_a_cut_run_ends_as_a_time_out_and_keeps_its_records(tmp_path):
    paths = launchers(tmp_path / "servers", "hang")

    async def cut(runner):
        run = asyncio.ensure_future(runner.run(QUESTION))
        while not runner.trajectory and not run.done():  # until the tool call that never answers is in flight
            await asyncio.sleep(0.05)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(run, 0.2)
        return asyncio.current_task().cancelling()

    with ScriptedEndpoint(list(SUPERVISED)) as endpoint:
        runner = MetaAgentRunner(paths, base_url=endpoint.base_url, api_key="local", model="granite-stand-in", log_path=tmp_path / "react_log.txt",  # pragma: allowlist secret
                                 example_start=stub_servers.EXAMPLE_START)
        started = time.perf_counter()
        assert asyncio.run(cut(runner)) == 0
        assert time.perf_counter() - started < 60  # the cut did not wait for the limit of the tool call
        requests = len(endpoint.completions())
    assert [(c["agent"], c["kind"]) for c in runner.calls] == [("supervisor", "thought"), ("supervisor", "action"), ("IoTAgent", "thought"),
                                                                 ("IoTAgent", "action")]
    assert requests == 4  # no model call after the cut
    (record,) = runner.trajectory
    deadline = time.perf_counter() + 5
    while record["error"] is None and time.perf_counter() < deadline:  # the thread of the agents writes it as it is woken
        time.sleep(0.01)
    # the call in flight was given up with the run, which is no failure of the transport
    assert (record["tool"], record["response"], record["error"]) == ("sites", None, "RunStopped: cancelled")


def test_a_failed_token_count_leaves_a_record_with_its_error():
    route = route_on(lambda request: httpx.Response(500, json={"error": "the tokenizer is down"}))
    with route.agent("IoTAgent"), pytest.raises(httpx.HTTPStatusError):
        route.count_tokens("one two three")
    (record,) = route.calls
    assert (record["agent"], record["kind"], record["prompt"], record["model"]) == ("IoTAgent", "tokenize", "one two three", "granite-stand-in")
    assert record["error"].startswith("HTTPStatusError: ") and "500" in record["error"] and "reply" not in record
    assert record["started_at"] <= record["ended_at"] and record["seconds"] >= 0
    # a count that succeeds leaves none, as before
    counted = route_on(lambda request: httpx.Response(200, json={"count": 3}))
    assert counted.count_tokens("one two three") == 3 and counted.calls == []


def test_a_model_list_without_a_context_length_raises_a_route_setup_error():
    route = route_on(lambda request: httpx.Response(200, json={"data": [{"id": "granite-stand-in", "max_model_len": None}]}))
    with pytest.raises(completion.RouteSetupError, match="^http://endpoint.example/v1/models names no context length for granite-stand-in$"):
        route.context_length()
    assert issubclass(completion.RouteSetupError, RuntimeError)


def test_the_request_settings_of_the_completion_route_are_what_it_sends(monkeypatch):
    settings = sub_agent_runtime.request_settings()
    decoding, transport = settings["decoding"], settings["transport"]
    assert set(settings) == {"decoding", "transport"}
    assert set(decoding) == {"route", "max_tokens", "min_tokens", "temperature", "stop", "no_stop_trim", "extra_body"}
    assert set(transport) == {"retries", "retry_status", "retry_delay_s", "timeout_s", "connect_timeout_s", "deadline", "sessions", "tool_timeout_s"}
    # a generation, as the engine asks for one: the body holds the recorded settings and goes to the recorded route
    seen = []

    def answer(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"text": " ok", "finish_reason": "stop", "matched_stop": None}], "usage": {}})

    route = route_on(answer)
    stop = next(list(strings) for strings, kind in completion.KINDS.items() if kind == "thought")
    route.generate("Thought 1:", stop=stop)
    body = json.loads(seen[0].content)
    assert seen[0].url.path == decoding["route"] == "/v1/completions"
    sent = ("max_tokens", "min_tokens", "temperature", "no_stop_trim")
    assert [body[key] for key in sent] == [decoding[key] for key in sent]
    assert body["stop"] == decoding["stop"]["thought"] and decoding["extra_body"] is None
    assert decoding["stop"] == {kind: list(strings) for strings, kind in completion.KINDS.items()}
    assert sub_agent_runtime.request_settings(temperature=0.7)["decoding"]["temperature"] == 0.7
    # the retries (CausalDiagnosis.utils.model_request_retry): an endpoint without an answer or with status 429, 502 or 503 is asked again with no count of
    # attempts, after the registered wait, while its health probe answers, until an attempt is answered
    from CausalDiagnosis.utils import model_request_retry as retry

    waits, probed = [], []
    monkeypatch.setattr(retry.time, "sleep", waits.append)

    def unreachable(request):
        raise httpx.ConnectError("no answer")

    with retry.health_check(lambda: probed.append(True) or True):
        for failure in (lambda request: httpx.Response(503), lambda request: httpx.Response(502), lambda request: httpx.Response(429), unreachable):
            asked = []
            failing = route_on(lambda request, failure=failure: asked.append(request) or (failure(request) if len(asked) <= 7 else answer(request)))
            failing.generate("Thought 1:", stop=stop)
            assert len(asked) == 8  # seven failures, past any former count, then the answer
    assert len(probed) == 28 and transport["retries"] is None and set(waits) == {transport["retry_delay_s"]}
    assert transport["retry_status"] == "no answer, 429, 502, 503"
    # outside an episode with no health probe named, a generation that hangs past the read timeout is sent once, since nothing
    # tells a dead endpoint from a live one; with a probe that answers, the same endpoint is asked again and serves the reply
    asked = []

    def hung(request):
        raise httpx.ReadTimeout("no reply within the limit")

    slow = route_on(lambda request: asked.append(request) or hung(request))
    with pytest.raises(httpx.ReadTimeout):
        slow.generate("Thought 1:", stop=stop)
    assert len(asked) == 1
    asked = []
    late = route_on(lambda request: asked.append(request) or (hung(request) if len(asked) == 1 else answer(request)))
    with retry.health_check(lambda: True):
        late.generate("Thought 1:", stop=stop)
    assert len(asked) == 2
    monkeypatch.undo()
    asked = []
    refused = route_on(lambda request: asked.append(request) or httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        refused.generate("Thought 1:", stop=stop)
    assert len(asked) == 1  # a status outside the recorded ones is not asked again
    # no time limit of the route's client or of a tool call of the ports: the row's deadline bounds every call
    built = completion.CompletionRoute("http://endpoint.example/v1", "local", "granite-stand-in")
    assert (built._client.timeout.read, built._client.timeout.connect) == (None, None) == (transport["timeout_s"], transport["connect_timeout_s"])
    built.close()
    assert transport["sessions"] == mcp_sessions.SESSIONS and transport["tool_timeout_s"] is None
    for runner_class in (MetaAgentRunner, PlanExecutePaperRunner):
        assert "tool_timeout_s" not in inspect.signature(runner_class.__init__).parameters
