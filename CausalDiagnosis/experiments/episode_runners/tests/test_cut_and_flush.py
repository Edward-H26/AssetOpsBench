"""The facts of a cut that run_limited takes at the deadline,
the cached tokens of an episode's first answered request, and the flush of the model server's cache before an episode of a
preset with the wall clock, written into the state."""
import asyncio
import json
import time
import types
from contextlib import nullcontext

from pathlib import Path
from unittest.mock import ANY

import pytest

from CausalDiagnosis.experiments.episode_runners import cache_flush
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.episode_runners import wall_cut
from CausalDiagnosis.agents.utils import mcp_sessions
from CausalDiagnosis.utils import model_request_retry as retry
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import read_rows
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
from CausalDiagnosis.utils import model_failures


def _cut_facts(tmp_path, runner, limit_s=0.2):
    with token_clock.episode(tmp_path), retry.episode_deadline(time.monotonic() + limit_s):
        _result, error, _seconds = run_episodes.run_limited(runner)
    assert error.startswith(run_episodes.WALL_LIMIT)
    return run_episodes.CUT_FACTS.pop(str(tmp_path))


def test_a_model_request_in_flight_at_the_deadline_is_open_at_the_cut(tmp_path):
    async def runner():
        return await retry.apost_with_retries(lambda timeout: asyncio.sleep(5))
    assert _cut_facts(tmp_path, runner) == {"model_request_open": True, "failed_span_open": False, "tool_call_open": False,
                                            "at_wall": ANY}  # the deadline in wall time, which a test of its own checks


def test_a_wait_after_a_failed_attempt_is_an_open_failed_span_at_the_cut(tmp_path):
    class Busy:
        status_code = 503

    async def send(timeout):
        return Busy()

    async def runner():
        return await retry.apost_with_retries(send, delay_s=5)
    assert _cut_facts(tmp_path, runner)["failed_span_open"] is True


def test_a_cut_while_the_agent_thinks_between_calls_finds_nothing_open(tmp_path):
    async def runner():
        await retry.apost_with_retries(lambda timeout: asyncio.sleep(0))
        await asyncio.sleep(5)
    assert _cut_facts(tmp_path, runner) == {"model_request_open": False, "failed_span_open": False, "tool_call_open": False,
                                            "at_wall": ANY}


def test_a_tool_call_without_its_exit_is_open_at_the_cut(tmp_path):
    from CausalDiagnosis.mcp_servers.utils import transcript
    transcript.append(tmp_path, "enter", server="rig", pid=1, tool="read_sensors", wall=0, mono=0, thread_cpu=0)
    assert _cut_facts(tmp_path, lambda: asyncio.sleep(5))["tool_call_open"] is True


class _Busy:
    status_code = 503


class _Answered:
    status_code = 200


def _runs_out(timeout):
    """An attempt that holds its thread for the time left and ends with the timeout of its read, as an HTTP client does."""
    time.sleep(timeout)
    raise TimeoutError("the read timeout of the attempt")


async def _holds_a_wait_after_a_503():
    return retry.post_with_retries(lambda timeout: _Busy(), delay_s=5)


async def _holds_a_request_to_its_timeout():
    return retry.post_with_retries(_runs_out)


async def _holds_the_loop_after_an_answer():
    retry.post_with_retries(lambda timeout: _Answered())
    time.sleep(0.4)  # the agent thinks on the thread of the event loop past the deadline, then asks again
    return retry.post_with_retries(lambda timeout: _Answered())


def _facts_and_end(tmp_path, runner):
    """The facts of the cut of runner without the wall time, and the end that wall_cut gives them without a probe."""
    facts = {key: value for key, value in _cut_facts(tmp_path, runner).items() if key != "at_wall"}
    return facts, wall_cut.cut_end(facts["failed_span_open"], facts["tool_call_open"], facts["model_request_open"])["end"]


@pytest.mark.parametrize("runner, open_kind, end", [(_holds_a_wait_after_a_503, "failed_span_open", failures.SERVICE),
                                                     (_holds_a_request_to_its_timeout, "model_request_open", failures.SERVICE),
                                                     (_holds_the_loop_after_an_answer, None, failures.WALL)])
def test_a_runner_that_reads_the_deadline_first_and_ends_at_it_is_cut_with_its_facts(tmp_path, runner, open_kind, end):
    """The order in which the runner reads the deadline first, forced by a runner that holds the event loop as a model call of
    the causal agent does, so the harness wakes only after the runner ended: the cut is noted all the same, with the facts of
    the deadline. A failed attempt whose wait the deadline cut short leaves its failed span open (service), and an attempt
    given the time left that ran out at the deadline leaves its model request open (the probe decides; none is sent here). The
    control: every call answered before the deadline and the next one refused by it leaves nothing open (wall)."""
    facts, ended = _facts_and_end(tmp_path, runner)
    assert {kind for kind, is_open in facts.items() if is_open} == ({open_kind} if open_kind else set()) and ended == end


def test_a_harness_that_reads_the_deadline_first_cuts_a_wait_that_outlives_it_with_the_failed_span_open(tmp_path, monkeypatch):
    """The order in which the harness reads the deadline first, forced by a wait after a failed attempt that is not cut short at
    the deadline: the cut takes the failed span open (service). The control: an answered request, then thinking past the
    deadline, leaves nothing open (wall)."""
    async def busy(timeout):
        return _Busy()

    async def answered(timeout):
        return _Answered()

    async def waits_past_the_deadline():
        return await retry.apost_with_retries(busy, delay_s=5)

    async def thinks_past_the_deadline():
        await retry.apost_with_retries(answered)
        await asyncio.sleep(5)

    monkeypatch.setattr(retry, "_wait", lambda deadline, delay_s: delay_s)
    (tmp_path / "busy").mkdir()
    (tmp_path / "thinks").mkdir()
    facts, ended = _facts_and_end(tmp_path / "busy", waits_past_the_deadline)
    assert facts == {"model_request_open": False, "failed_span_open": True, "tool_call_open": False} and ended == failures.SERVICE
    facts, ended = _facts_and_end(tmp_path / "thinks", thinks_past_the_deadline)
    assert facts == {"model_request_open": False, "failed_span_open": False, "tool_call_open": False} and ended == failures.WALL


class _Late:
    """A server whose answer comes delay_s after the call."""

    def __init__(self, delay_s):
        self.delay_s = delay_s

    async def call_tool(self, tool, arguments):
        await asyncio.sleep(self.delay_s)
        return types.SimpleNamespace(content=[types.SimpleNamespace(text="answered")], isError=False)


def _answer_past_the_deadline(under_the_drivers_cut):
    """The answer of one call that comes 0.2 s after the deadline that the call itself reads, or the error that ended it."""
    sessions = mcp_sessions.EpisodeSessions({"rig": _Late(0.4)}, {"rig": ["read_sensors"]})

    async def call():
        at = time.monotonic() + 0.2  # a fixed instant, so the call finds it passed when its wait expires
        with retry.episode_deadline(lambda: at), (retry.driver_cut() if under_the_drivers_cut else nullcontext()):
            try:
                return await sessions.call("rig", "read_sensors", {})
            except mcp_sessions.ToolTransportError as exc:
                return exc

    return asyncio.run(call())


def test_an_answer_that_comes_back_after_the_call_read_the_deadline_is_delivered_under_the_drivers_cut():
    """The order in which the call reads the deadline before the harness and finds it passed while its answer is on the way:
    under the driver's cut the call waits on, and the answer that comes back is delivered, never read as unanswered. The
    control: outside the driver's cut the call gives itself up at the deadline."""
    assert _answer_past_the_deadline(True) == ("answered", False)
    assert isinstance(_answer_past_the_deadline(False), mcp_sessions.ToolTransportError)


def test_the_drivers_cut_ends_a_call_that_never_answers_with_the_call_open(tmp_path):
    """A call that waits on past the deadline under the driver's cut is ended by the cut in bounded time, and the cut finds the
    tool call open (service)."""
    from CausalDiagnosis.mcp_servers.utils import transcript
    transcript.append(tmp_path, "enter", server="rig", pid=1, tool="read_sensors", wall=0, mono=0, thread_cpu=0)
    sessions = mcp_sessions.EpisodeSessions({"rig": _Late(30)}, {"rig": ["read_sensors"]})
    started = time.monotonic()
    facts, ended = _facts_and_end(tmp_path, lambda: sessions.call("rig", "read_sensors", {}))
    assert facts["tool_call_open"] and ended == failures.SERVICE and time.monotonic() - started < 10


def test_the_first_answered_request_gives_the_cached_tokens_of_the_episode():
    def reply(cached):
        details = None if cached is None else types.SimpleNamespace(cached_tokens=cached)
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens_details=details))

    with retry.episode_deadline(time.monotonic() + 60):
        assert retry.first_cached_tokens() is None
        retry.post_with_retries(lambda timeout: reply(None))  # SGLang leaves the details out when nothing was reused
        retry.post_with_retries(lambda timeout: reply(40))
        assert retry.first_cached_tokens() == 0
    with retry.episode_deadline(time.monotonic() + 60):  # the next episode starts its own record
        retry.post_with_retries(lambda timeout: reply(12))
        assert retry.first_cached_tokens() == 12


def test_a_reply_without_usage_reports_no_cached_tokens():
    class Response:
        def json(self):
            return {"choices": []}
    assert retry.reply_cached_tokens(Response()) is None
    assert retry.reply_cached_tokens(types.SimpleNamespace(usage={"prompt_tokens_details": {"cached_tokens": 7}})) == 7


def _runner(clock_mode, agent="causal"):
    runner = object.__new__(run_episodes.EpisodeRunner)
    runner.agent, runner.base_url = agent, "http://127.0.0.1:1/v1"
    return runner, types.SimpleNamespace(clock_mode=clock_mode)


def _state_file(tmp_path):
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
    state = tmp_path / run_episodes.STATE_FILE
    state.write_text(json.dumps({"format": episode.STATE_FORMAT}), encoding="utf-8")
    return state


def test_an_episode_of_the_wall_clock_starts_after_a_flush_written_into_the_state(tmp_path, monkeypatch):
    state = _state_file(tmp_path)
    monkeypatch.setattr(cache_flush, "flush_before_episode", lambda base_url, api_key=None: {
        "flush_seconds": 0.01, "status": 200, "running": 0.0, "queued": 0.0, "first_request_cached_tokens": None})
    runner, config = _runner("generation_time")
    flushed = runner.flush_cache(tmp_path, config)
    assert flushed["status"] == 200 and isinstance(flushed["started_at"], float)
    assert json.loads(state.read_text(encoding="utf-8"))["cache_flush"] == flushed


@pytest.mark.parametrize("clock_mode, agent", [("declared", "causal"), ("generated_tokens", "causal"), ("generation_time", "scripted")])
def test_no_flush_on_another_clock_or_without_a_model(tmp_path, monkeypatch, clock_mode, agent):
    _state_file(tmp_path)
    monkeypatch.setattr(cache_flush, "flush_before_episode", lambda *a, **k: pytest.fail("flushed"))
    runner, config = _runner(clock_mode, agent)
    assert runner.flush_cache(tmp_path, config) is None


def test_a_flush_that_fails_stops_the_episode(tmp_path, monkeypatch):
    _state_file(tmp_path)

    def refused(base_url, api_key=None):
        raise cache_flush.CacheNotFlushed("flush_cache answered 400")
    monkeypatch.setattr(cache_flush, "flush_before_episode", refused)
    runner, config = _runner("generation_time")
    with pytest.raises(cache_flush.CacheNotFlushed):
        runner.flush_cache(tmp_path, config)


def test_the_finish_after_a_cut_is_stamped_at_the_deadline_and_not_after_the_teardown(tmp_path):
    started_wall = time.time()
    facts = _cut_facts(tmp_path, lambda: asyncio.sleep(5))
    # the deadline fell 0.2 s after the start; the cut's own instant in wall time is that deadline, before this line ran
    assert started_wall + 0.15 <= facts["at_wall"] <= time.time()
    stamp, source = run_episodes.finish_stamp([{"ended_at": started_wall}], facts["at_wall"], True)
    assert stamp == facts["at_wall"] and source == "the cut at the wall limit"


def test_a_failed_flush_makes_the_attempt_the_harness_and_writes_no_failure_ledger(tmp_path, monkeypatch):
    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")

    def refused(self, ep_dir, episode_config):
        raise cache_flush.CacheNotFlushed("flush_cache answered 400")
    monkeypatch.setattr(run_episodes.EpisodeRunner, "flush_cache", refused)
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", lambda *a, **k: pytest.fail("the agent played after a failed flush"))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setattr(failures, "endpoint_health", lambda *a, **k: {"ok": True, "status": 200, "error": None, "seconds": 0.0})
    run_dir = run_episodes.main(["--config", "static", "--agent", "causal", "--seeds", "1", "--no-grade", "--out", str(tmp_path / "runs")])
    row = read_rows(run_dir)[0]
    assert row["error_class"] == failures.HARNESS_ERROR and "CacheNotFlushed" in row["error"]
    assert not list(Path(run_dir).rglob(model_failures.NAME))  # the flush goes through no retry policy and no ledger


def test_an_agent_as_tool_episode_on_the_wall_clock_records_the_cached_tokens_of_its_first_completion(tmp_path, monkeypatch):
    """The completion route of the paper-era ports sends its model listing and its tokenize count as no model reply, as
    CausalDiagnosis.utils.model_reply_room does, so an Agent-As-Tool episode on a preset with the wall clock, whose agents ask both before their first
    completion, records the cached tokens of that completion, where it once recorded the None of the listing. The control:
    the listing and the count are sent and answered in the episode before its first completion, and the later completions,
    which report 5 cached tokens, leave the record at the first one's 0."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    model, seen = "vendor/stand-in-1.0", []

    class Endpoint(BaseHTTPRequestHandler):
        """The routes a port calls on SGLang: the model listing, the tokenize count and the two completion routes, each
        completion with a usage block whose cached tokens are 0 for the first completion and 5 after it."""

        def _answer(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - the http.server hook name
            seen.append(f"GET {self.path}")
            if self.path == "/server_info":  # SGLang's server arguments, the cache report among them
                return self._answer({"enable_cache_report": True, "api_key": "server-secret"})  # pragma: allowlist secret
            self._answer({"object": "list", "data": [{"id": model, "max_model_len": 100000}]})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if self.path.endswith("/tokenize"):
                seen.append(f"POST {self.path}")
                return self._answer({"count": 11})
            cached = 5 if any(s.endswith("completions") for s in seen) else 0
            seen.append(f"POST {self.path}")
            usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "prompt_tokens_details": {"cached_tokens": cached}}
            if self.path.endswith("/chat/completions"):
                return self._answer({"id": "c", "object": "chat.completion", "created": 1, "model": body.get("model"), "usage": usage,
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": "Final Answer: no fault found."}}]})
            return self._answer({"id": "c", "object": "text_completion", "created": 1, "model": body.get("model"), "usage": usage,
                                 "choices": [{"index": 0, "text": " The readings suffice.\nFinal Answer: no fault found.",
                                              "finish_reason": "stop", "matched_stop": None}]})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(cache_flush, "flush_before_episode", lambda base_url, api_key=None: {
        "flush_seconds": 0.01, "status": 200, "running": 0.0, "queued": 0.0, "first_request_cached_tokens": None})
    try:
        run_dir = run_episodes.main(["--config", "real_time", "--agent", "agent_as_tool", "--model", model, "--base-url",
                                     f"http://127.0.0.1:{server.server_address[1]}/v1", "--seeds", "1", "--starts", "seeds",
                                     "--out", str(tmp_path / "runs")])
    finally:
        server.shutdown()
    flush = json.loads(next(Path(run_dir).glob("seed*/episode.json")).read_text(encoding="utf-8"))["cache_flush"]
    first = next(i for i, request in enumerate(seen) if request.endswith("completions"))
    assert {"GET /v1/models", "POST /v1/tokenize"} <= set(seen[:first]) and len(seen) > first + 1, seen
    assert flush["first_request_cached_tokens"] == 0, (flush, seen[:first + 2])
    # the flush record carries the server's cache report from run.json served, so the overlap check reads the first 0 as a clean
    # cache; the control of a server without the report stands in test_run_checks_overlap.py
    from CausalDiagnosis.experiments.cluster_runs import run_health_checks

    assert flush["cache_report"] is True and run_health_checks.flush_problem(next(Path(run_dir).glob("seed*"))) is None, flush
    # llm_calls.json holds the count of every completion as the endpoint sent it, the first completion's 0 and 5 after it
    calls = json.loads(next(Path(run_dir).glob("seed*/llm_calls.json")).read_text(encoding="utf-8"))
    assert len(calls) > 1 and [call["cached_tokens"] for call in calls] == [0] + [5] * (len(calls) - 1), calls
