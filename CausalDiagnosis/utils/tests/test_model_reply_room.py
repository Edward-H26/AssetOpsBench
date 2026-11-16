"""The room of a reply (CausalDiagnosis.utils.model_reply_room): the completion route sends the context less the tokenize count,
the context comes from the endpoint's model listing, and the prompt count posts what the route renders. The chat route's
use of the room lives with its backend (CausalDiagnosis/utils/tests/test_llm_backends.py)."""

from __future__ import annotations

import pytest

from CausalDiagnosis.utils.model_reply_room import ContextEnd, NoReplyRoom, ReplyRoom, endpoint_room, served_context, tokenize_counter


def words(prompt):
    return len(prompt.split())


def test_the_room_is_read_from_the_endpoint():
    """The context is max_model_len of GET {base_url}/models; the chat count posts the same messages and template arguments to
    {base_url}/tokenize, the completion count the raw prompt."""
    calls = []

    def get(url, key):
        calls.append(("get", url))
        return {"data": [{"id": "granite", "max_model_len": 131072}]}

    def post(url, key, body):
        calls.append(("post", url, body))
        return {"count": 7}

    room = endpoint_room("http://agent.example/v1/", "k", "granite", extra={"chat_template_kwargs": {"enable_thinking": False}}, get=get, post=post)
    assert room.context_length == 131072 and room.count_tokens("hello") == 7
    assert calls[0] == ("get", "http://agent.example/v1/models")
    assert calls[1] == ("post", "http://agent.example/v1/tokenize", {"model": "granite", "messages": [{"role": "user", "content": "hello"}],
                                                                     "chat_template_kwargs": {"enable_thinking": False}})
    tokenize_counter("http://agent.example/v1", "k", "granite", chat=False, post=post)("raw text")
    assert calls[-1] == ("post", "http://agent.example/v1/tokenize", {"model": "granite", "prompt": "raw text"})
    with pytest.raises(NoReplyRoom):
        served_context("http://agent.example/v1", "k", "other", get=lambda url, key: {"data": [{"id": "a"}, {"id": "b"}]})


def test_the_completion_route_sends_the_context_less_the_prompt():
    """On the completion route an absent max_tokens is 16 tokens, so the request names the context less the prompt's count;
    a prompt that fills the context is the end of the context."""
    room = ReplyRoom(10, words)
    assert room.max_tokens("a b c") == 7
    with pytest.raises(ContextEnd):
        room.max_tokens(" ".join("a" * 10))


def test_a_prompt_under_ninety_percent_of_the_context_is_under_the_trigger_and_one_above_is_not():
    room = ReplyRoom(100, len)
    assert room.request_under_trigger("x" * 89) is True
    assert room.request_under_trigger("x" * 90) is True
    assert room.request_under_trigger("x" * 91) is False
    assert ReplyRoom(7, len).request_under_trigger("x" * 6) is True  # the trigger is floor(7 * 9 / 10) = 6
    assert ReplyRoom(7, len).request_under_trigger("x" * 7) is False


def test_the_requests_of_the_room_go_through_the_retry_policy(tmp_path, monkeypatch):
    """A lost connection or a 503 of the listing or the tokenize route is a failed attempt sent again under CausalDiagnosis.utils.model_request_retry, and the
    answer is no model reply, so it never stands for the episode's first reply."""
    import httpx

    from CausalDiagnosis.utils import model_request_retry as retry, model_reply_room as room
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock

    request = httpx.Request("POST", "http://agent.example/v1/tokenize")
    outcomes = [httpx.ConnectError("no answer", request=request), httpx.Response(503, request=request),
                httpx.Response(200, json={"count": 7}, request=request)]
    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: _next(outcomes))
    with token_clock.episode(tmp_path), retry.episode_deadline(None):
        assert room.tokenize_counter("http://agent.example/v1", "k", "granite")("hello") == 7
        assert retry.first_cached_tokens() is None and outcomes == []


def _next(outcomes):
    outcome = outcomes.pop(0)
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome
