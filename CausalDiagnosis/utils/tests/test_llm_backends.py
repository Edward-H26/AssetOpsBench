"""The chat route of the benchmark (llm_backends.EndpointChatBackend): no reply cap of the harness and a reply that ends by length
checked against the engine's room, the engine's refusal of a prompt past the context as ContextEnd, an endpoint bound to the
backend, extra_body, a client that sends each request once while CausalDiagnosis.utils.model_request_retry attempts a failed request again, the read timeout
outside an episode and the row's deadline inside one, the reasoning text as the reply, the reason for the end of a reply, and
the settings of the route for the record of a run."""
import json
import sys
import time
import types

import pytest

from CausalDiagnosis.utils import llm_backends
from CausalDiagnosis.utils import model_request_retry as retry
from CausalDiagnosis.utils.model_reply_room import ContextEnd, EarlyCut, NoReplyRoom, ReplyRoom
from CausalDiagnosis.utils.llm_backends import ChatResult, EndpointChatBackend

CLIENT_KEYS = ("base_url", "api_key", "client")  # what the stub keeps of the client, beside the request
BOUND = {"base_url": "http://agent.example/v1", "api_key": "k"}


def words(prompt):
    return len(prompt.split())


def _install_fake_openai(monkeypatch, captured: dict, message=None, finish_reason="stop", usage=(3, 2), errors=(), details=None):
    """Install a stub ``openai`` module that records the call kwargs and the arguments the client was built with. Each call
    first raises the next of errors, then answers, with details as the prompt_tokens_details of its usage; captured["calls"]
    counts the calls."""
    pending = list(errors)

    def create(**kwargs):
        captured["calls"] = captured.get("calls", 0) + 1
        captured.update(kwargs)
        if pending:
            raise pending.pop(0)
        choice = types.SimpleNamespace(message=message or types.SimpleNamespace(content="hi"), finish_reason=finish_reason)
        return types.SimpleNamespace(choices=[choice], usage=types.SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1],
                                                                                  prompt_tokens_details=details))

    class OpenAI:
        def __init__(self, base_url=None, api_key=None, **client):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["client"] = client
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    fake = types.ModuleType("openai")
    fake.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)


def request_of(captured):
    return {k: v for k, v in captured.items() if k not in (*CLIENT_KEYS, "calls", "timeout")}


def test_reasoning_content_is_the_reply_when_content_is_empty(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured, types.SimpleNamespace(content=None, reasoning_content="1. read"), "length")
    # a reply that ended by length is checked against the room: 7 - 3 - 2 = 2 tokens, the reply's own count
    result = EndpointChatBackend("tokenrouter/MiniMax-M3", room=ReplyRoom(7, words), **BOUND).generate_with_usage("plan")
    assert isinstance(result, ChatResult)
    assert result.text == "1. read" and result.finish_reason == "length" and result.output_tokens == 2


def test_the_thinking_that_a_reasoning_parser_moved_out_of_the_text_is_kept_in_the_reasoning_field(monkeypatch):
    """A server with a reasoning parser returns the thinking outside the text, in reasoning_content or reasoning, and the result
    keeps it in its reasoning field, which the judge's count of thinking reads. The control: a reply without one has none."""
    for message, reasoning in ((types.SimpleNamespace(content="yes", reasoning_content="the criteria first"), "the criteria first"),
                               (types.SimpleNamespace(content="yes", reasoning="the criteria first"), "the criteria first"),
                               (types.SimpleNamespace(content="yes"), None)):
        _install_fake_openai(monkeypatch, {}, message)
        result = EndpointChatBackend("tokenrouter/granite", **BOUND).generate_with_usage("review")
        assert result.text == "yes" and result.reasoning == reasoning, (message, result)


def test_the_result_keeps_the_cached_prompt_tokens_that_the_reply_reports_and_none_without_a_count(monkeypatch):
    """The chat route passes the cached prompt tokens of usage.prompt_tokens_details on in its result, a zero included, for
    llm_calls.json. The control: a reply without details, as SGLang answers a request that reused nothing and every server
    without the cache report answers every request, keeps None and never a zero."""
    for details, cached in ((types.SimpleNamespace(cached_tokens=12), 12), (types.SimpleNamespace(cached_tokens=0), 0), (None, None)):
        _install_fake_openai(monkeypatch, {}, details=details)
        result = EndpointChatBackend("tokenrouter/granite", **BOUND).generate_with_usage("plan")
        assert result.cached_tokens == cached and (result.input_tokens, result.output_tokens) == (3, 2), (details, result)


def test_an_endpoint_bound_to_the_backend_wins_over_the_environment(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "judge-key")
    EndpointChatBackend("tokenrouter/granite", base_url="http://agent.example/v1", api_key="agent-key").generate_with_usage("hello")  # pragma: allowlist secret
    assert captured["base_url"] == "http://agent.example/v1" and captured["api_key"] == "agent-key"  # pragma: allowlist secret
    captured.clear()
    EndpointChatBackend("tokenrouter/granite").generate_with_usage("hello")  # neither: the endpoint of the environment
    assert captured["base_url"] == "https://judge.example/v1" and captured["api_key"] == "judge-key"  # pragma: allowlist secret


def test_extra_body_is_sent_only_when_the_backend_has_one(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello")
    assert "extra_body" not in captured  # the agent's request is unchanged
    no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}
    EndpointChatBackend("tokenrouter/judge", base_url="http://judge.example/v1", api_key="k", extra_body=no_thinking).generate("hello")
    assert captured["extra_body"] == no_thinking


def test_the_chat_request_carries_no_max_tokens_and_a_callers_cap_moves_only_its_field(monkeypatch):
    """The engine gives the reply the rest of the context by its own count, so the chat request names no max_tokens; a caller's
    own cap is sent as it is."""
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    expected = {"model": "granite", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.0}
    EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello")
    assert json.dumps(request_of(captured), sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert "max_completion_tokens" not in captured
    captured.clear()
    EndpointChatBackend("tokenrouter/granite", max_tokens=8192, **BOUND).generate("hello")
    assert request_of(captured) == {**expected, "max_tokens": 8192}
    assert EndpointChatBackend("tokenrouter/granite", **BOUND).request_settings()["decoding"]["max_tokens"] is None


def test_a_reply_cut_before_the_engine_room_raises(monkeypatch):
    """A reply that ends by length must have used the context less the prompt less 2; one cut earlier raises EarlyCut. A reply
    that stopped by itself, or one under a caller's own cap, is not checked."""
    room = ReplyRoom(100, words)
    _install_fake_openai(monkeypatch, {}, finish_reason="length", usage=(40, 58))
    assert EndpointChatBackend("tokenrouter/granite", room=room, **BOUND).generate_with_usage("x").output_tokens == 58
    _install_fake_openai(monkeypatch, {}, finish_reason="length", usage=(40, 30))
    with pytest.raises(EarlyCut):
        EndpointChatBackend("tokenrouter/granite", room=room, **BOUND).generate("x")
    EndpointChatBackend("tokenrouter/granite", room=room, max_tokens=30, **BOUND).generate("x")
    _install_fake_openai(monkeypatch, {}, finish_reason="stop", usage=(40, 30))
    EndpointChatBackend("tokenrouter/granite", room=room, **BOUND).generate("x")


def test_a_prompt_past_the_context_is_the_end_of_the_context_and_is_sent_once(monkeypatch):
    """SGLang answers a prompt past the context with HTTP 400 and its text; that is ContextEnd, never a failed attempt, and any
    other 400 is raised as it came."""

    class BadRequest(Exception):
        status_code = 400

    captured: dict = {}
    refusal = BadRequest("The input (131070 tokens) is longer than the model's context length (131072 tokens).")
    _install_fake_openai(monkeypatch, captured, errors=[refusal])
    with pytest.raises(ContextEnd):
        EndpointChatBackend("tokenrouter/granite", **BOUND).generate("x")
    assert captured["calls"] == 1
    _install_fake_openai(monkeypatch, {}, errors=[BadRequest("unknown field")])
    with pytest.raises(BadRequest):
        EndpointChatBackend("tokenrouter/granite", **BOUND).generate("x")


def test_a_request_fits_while_it_leaves_room_for_the_longest_reply_so_far(monkeypatch):
    """Before any reply a prompt fits while one token is left; after a reply of 2 tokens it fits while 2 are left."""
    _install_fake_openai(monkeypatch, {}, usage=(1, 2))
    b = EndpointChatBackend("tokenrouter/granite", room=ReplyRoom(5, words), **BOUND)
    assert b.request_fits("a b c d") and not b.request_fits("a b c d e")
    b.generate("hi")
    assert b.longest_reply == 2 and not b.request_fits("a b c d") and b.request_fits("a b c")


def test_a_prompt_is_under_the_trigger_of_compaction_while_its_tokens_alone_are_at_most_nine_tenths_of_the_context(monkeypatch):
    """The trigger of compaction of the CausalDiagnosis agent: the prompt's tokens alone, no reply term, against
    floor(context * 9 / 10) of the same room as request_fits."""
    _install_fake_openai(monkeypatch, {})
    b = EndpointChatBackend("tokenrouter/granite", room=ReplyRoom(100, words), **BOUND)
    assert b.request_under_trigger(" ".join(["w"] * 89)) and b.request_under_trigger(" ".join(["w"] * 90))
    assert not b.request_under_trigger(" ".join(["w"] * 91))


def test_the_room_is_read_from_the_bound_endpoint_with_the_template_arguments(monkeypatch):
    seen = {}

    def endpoint_room(base_url, api_key, model, chat=True, extra=None):
        seen.update(base_url=base_url, api_key=api_key, model=model, chat=chat, extra=extra)
        return ReplyRoom(9, words)

    monkeypatch.setattr(llm_backends, "endpoint_room", endpoint_room)
    extra = {"chat_template_kwargs": {"enable_thinking": False}, "separate_reasoning": True}
    b = EndpointChatBackend("tokenrouter/granite", extra_body=extra, **BOUND)
    assert b.room().context_length == 9 and b.room() is b.room()  # built once
    assert seen == {"base_url": "http://agent.example/v1", "api_key": "k", "model": "granite", "chat": True,
                    "extra": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_a_backend_with_no_endpoint_cannot_tell_its_room(monkeypatch):
    monkeypatch.delenv("TOKENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("TOKENROUTER_API_KEY", raising=False)
    with pytest.raises(NoReplyRoom):
        EndpointChatBackend("tokenrouter/granite").request_fits("x")


def test_outside_an_episode_a_503_is_sent_again_while_the_named_probe_answers_and_a_dead_endpoint_ends_with_service_down(monkeypatch):
    """Outside an episode the route's request follows the one retry policy with the health probe its caller names
    (CausalDiagnosis.utils.model_request_retry.health_check): a 503 is sent again while the probe answers, and a lost connection under a probe that no longer
    answers ends with ServiceDown after one attempt. The control: with no probe named the 503 is raised after one call."""
    import httpx

    monkeypatch.setattr(retry, "_wait", lambda deadline, delay_s: 0.0)

    class Unavailable(Exception):
        status_code = 503  # as the SDK raises an answer of 503

    captured: dict = {}
    _install_fake_openai(monkeypatch, captured, errors=[Unavailable()])
    with retry.health_check(lambda: True):
        assert EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello") == "hi"
    assert captured["calls"] == 2
    captured = {}
    _install_fake_openai(monkeypatch, captured, errors=[httpx.ConnectError("refused")])
    with retry.health_check(lambda: False), pytest.raises(retry.ServiceDown):
        EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello")
    assert captured["calls"] == 1
    captured = {}
    _install_fake_openai(monkeypatch, captured, errors=[Unavailable()])
    with pytest.raises(Unavailable):
        EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello")
    assert captured["calls"] == 1


def test_the_client_sends_each_request_once_with_the_pinned_time_limits(monkeypatch):
    import httpx

    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")
    for backend in (EndpointChatBackend("tokenrouter/granite", **BOUND), EndpointChatBackend("tokenrouter/granite")):
        captured.clear()
        backend.generate("hello")
        assert set(captured["client"]) == {"max_retries", "timeout"}
        assert captured["client"]["max_retries"] == llm_backends.CLIENT_MAX_RETRIES == 0  # CausalDiagnosis.utils.model_request_retry retries, not the client
        timeout = captured["client"]["timeout"]
        assert isinstance(timeout, httpx.Timeout) and timeout.connect == llm_backends.CLIENT_CONNECT_TIMEOUT_S == 5.0
        assert timeout.read == timeout.write == timeout.pool == llm_backends.CLIENT_TIMEOUT_S == retry.READ_TIMEOUT_S == 600.0


def test_outside_an_episode_an_attempt_has_the_read_timeout_and_a_timed_out_attempt_is_attempted_again(monkeypatch):
    """Outside an episode each attempt carries the read timeout, and an attempt that it ends is a failed attempt of
    the rule: it is sent again while the endpoint's health probe answers, and with no probe named it is the request's last;
    inside an episode the attempt carries the time left before the row's deadline."""
    import httpx

    monkeypatch.setattr(retry, "_wait", lambda deadline, delay_s: 0.0)
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured, errors=[httpx.ReadTimeout("read timed out")])
    with retry.health_check(lambda: True):
        assert EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello") == "hi"
    assert captured["calls"] == 2 and captured["timeout"] == retry.READ_TIMEOUT_S
    captured.clear()
    _install_fake_openai(monkeypatch, captured, errors=[httpx.ReadTimeout("read timed out")])
    with pytest.raises(httpx.ReadTimeout):
        EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello")
    assert captured["calls"] == 1
    captured.clear()
    _install_fake_openai(monkeypatch, captured)
    EndpointChatBackend("tokenrouter/granite", deadline=time.monotonic() + 50.0, **BOUND).generate("hello")
    assert 0.0 < captured["timeout"] <= 50.0


def test_a_failed_attempt_is_attempted_again_and_a_dead_service_ends_the_program_outside_an_episode(monkeypatch):
    """A 503 is a failed attempt; outside an episode the health hook then decides: a live endpoint is attempted again, a dead
    one ends with ServiceDown."""

    class Unavailable(Exception):
        status_code = 503

    monkeypatch.setattr(retry, "_wait", lambda deadline, delay_s: 0.0)
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured, errors=[Unavailable("busy")])
    with retry.health_check(lambda: True):
        assert EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello") == "hi"
    assert captured["calls"] == 2
    _install_fake_openai(monkeypatch, {}, errors=[Unavailable("gone")])
    with retry.health_check(lambda: False), pytest.raises(retry.ServiceDown):
        EndpointChatBackend("tokenrouter/granite", **BOUND).generate("hello")


def test_the_request_settings_of_the_chat_route_are_what_it_sends(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    backend = EndpointChatBackend("tokenrouter/granite", extra_body=extra, max_tokens=4096, **BOUND)
    settings = backend.request_settings(0.7)
    backend.generate("hello", 0.7)
    decoding, transport = settings["decoding"], settings["transport"]
    assert set(settings) == {"decoding", "transport"}
    assert set(decoding) == {"route", "max_tokens", "min_tokens", "temperature", "stop", "no_stop_trim", "extra_body"}
    assert set(transport) == {"retries", "retry_status", "retry_delay_s", "timeout_s", "connect_timeout_s"}
    assert (decoding["max_tokens"], decoding["temperature"], decoding["extra_body"]) == (captured["max_tokens"], captured["temperature"], captured["extra_body"])
    assert decoding["route"] == "/v1/chat/completions" and (captured["max_tokens"], captured["temperature"]) == (4096, 0.7)
    # what the route does not send is recorded as None, and the request holds no such field
    assert [decoding[key] for key in ("min_tokens", "stop", "no_stop_trim")] == [None, None, None]
    assert not {"min_tokens", "stop", "no_stop_trim"} & set(captured)
    assert transport["retries"] is None and captured["client"]["max_retries"] == 0  # no count of attempts (CausalDiagnosis.utils.model_request_retry)
    assert (transport["timeout_s"], transport["connect_timeout_s"]) == (captured["client"]["timeout"].read, captured["client"]["timeout"].connect)
    assert EndpointChatBackend("tokenrouter/granite", **BOUND).request_settings()["decoding"] == {
        "route": "/v1/chat/completions", "max_tokens": None, "min_tokens": None, "temperature": 0.0, "stop": None, "no_stop_trim": None,
        "extra_body": None}


def test_the_recorded_retry_rule_is_the_one_policy_of_the_benchmark():
    """The record of a run states the policy of CausalDiagnosis.utils.model_request_retry: no answer, 429, 502 and 503, after the registered wait."""
    assert llm_backends.CLIENT_RETRY_STATUS == "no answer, 429, 502, 503" and retry.STATUSES == (429, 502, 503)
    assert llm_backends.CLIENT_RETRY_DELAY_S == retry.RETRY_DELAY_S == 2.0


def test_half_an_endpoint_is_refused(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")
    for half in ({"base_url": "http://agent.example/v1"}, {"api_key": "k"}, {"base_url": "http://agent.example/v1", "api_key": ""},
                 {"base_url": "", "api_key": "k"}):
        with pytest.raises(ValueError, match="^base_url and api_key are given together or not at all$"):
            EndpointChatBackend("tokenrouter/granite", **half)
    EndpointChatBackend("tokenrouter/granite")  # neither: the endpoint of the environment, as before
    EndpointChatBackend("tokenrouter/granite", **BOUND)


def test_the_longest_reply_counts_only_replies_that_ended_on_their_own(monkeypatch):
    """Replies of 100 tokens (stop), 50,000 (length) and 200 (stop) leave the
    longest reply at 200, and a prompt that fits beside 200 tokens but not beside 50,000 fits."""
    b = EndpointChatBackend("tokenrouter/granite", room=ReplyRoom(50_100, words), **BOUND)
    for finish, tokens in (("stop", 100), ("length", 50_000), ("stop", 200)):
        _install_fake_openai(monkeypatch, {}, finish_reason=finish, usage=(100, tokens))
        if finish == "length":
            b._room = ReplyRoom(50_102, words)  # 50,102 - 100 - 2 = 50,000: the reply used its room, so no EarlyCut
            b.generate("x")
            b._room = ReplyRoom(50_100, words)
        else:
            b.generate("x")
    assert b.longest_reply == 200
    prompt = " ".join(["w"] * 49_900)
    assert b.request_fits(prompt) and 49_900 + 50_000 > 50_100
