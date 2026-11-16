"""attempt_error_classes.py: the kind of every error type the recorders write, read from real exceptions of the installed clients, the class
of an attempt with the one health check, and the stop of a run after three harness errors in a row. No server, no model."""
import httpx
import openai
import pytest

from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
from CausalDiagnosis.agents.utils.mcp_sessions import ToolTransportError
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import RouteSetupError

REQUEST = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")


def _response(status):
    return httpx.Response(status, request=REQUEST)


def _recorded(exc):
    """An error as the recorders write it into llm_calls.json, trajectory.json and the error cell."""
    return f"{type(exc).__name__}: {exc}"


def test_the_kind_of_every_listed_error_type_is_read_from_a_real_exception():
    """Each exception is built from the installed openai and httpx packages, so a class that a package renames fails here."""
    cases = [
        (openai.APIConnectionError(request=REQUEST), failures.CONNECTION),
        (httpx.ConnectError("refused", request=REQUEST), failures.CONNECTION),
        (httpx.ConnectTimeout("no connection in time", request=REQUEST), failures.CONNECTION),
        (httpx.ReadError("reset by peer", request=REQUEST), failures.CONNECTION),
        (httpx.RemoteProtocolError("closed without a response", request=REQUEST), failures.CONNECTION),
        (openai.InternalServerError("Error code: 500", response=_response(500), body=None), failures.SERVER),
        (openai.RateLimitError("Error code: 429", response=_response(429), body=None), failures.SERVER),
        (openai.ConflictError("Error code: 409", response=_response(409), body=None), failures.SERVER),
        (openai.APIResponseValidationError(response=_response(200), body=None), failures.SERVER),
        (httpx.HTTPStatusError("Server error '503 Service Unavailable' for url 'x'", request=REQUEST, response=_response(503)), failures.SERVER),
        (httpx.HTTPStatusError("Client error '408 Request Timeout' for url 'x'", request=REQUEST, response=_response(408)), failures.SERVER),
        (openai.AuthenticationError("Error code: 401", response=_response(401), body=None), failures.SETUP),
        (openai.PermissionDeniedError("Error code: 403", response=_response(403), body=None), failures.SETUP),
        (openai.NotFoundError("Error code: 404", response=_response(404), body=None), failures.SETUP),
        (httpx.HTTPStatusError("Client error '404 Not Found' for url 'x'", request=REQUEST, response=_response(404)), failures.SETUP),
        (RouteSetupError("the model list names no context length"), failures.SETUP),
        (openai.APITimeoutError(request=REQUEST), failures.TIMEOUT),
        (httpx.ReadTimeout("no answer in time", request=REQUEST), failures.TIMEOUT),
        (openai.BadRequestError("Error code: 400", response=_response(400), body=None), failures.REQUEST),
        (openai.UnprocessableEntityError("Error code: 422", response=_response(422), body=None), failures.REQUEST),
        (httpx.HTTPStatusError("Client error '422 Unprocessable Entity' for url 'x'", request=REQUEST, response=_response(422)), failures.REQUEST),
        (ToolTransportError("the call passed 120.0 s"), failures.TOOL),
        (ValueError("a reply the runner could not use"), failures.OTHER),
    ]
    for exc, kind in cases:
        assert failures.kind_of(_recorded(exc)) == kind, (type(exc).__name__, str(exc))
    assert failures.kind_of("KeyError: 'x'", harness=True) == failures.HARNESS
    assert failures.error_type("not an error text") == "" and failures.kind_of("") == failures.OTHER


def test_a_request_the_endpoint_refuses_as_too_long_is_the_agents():
    """A context overflow is a request the endpoint refused, so the attempt is the agent's and is not replayed."""
    text = "BadRequestError: Error code: 400 - {'error': {'message': \"This model's maximum context length is 32768 tokens\"}}"
    assert failures.kind_of(text) == failures.REQUEST
    assert failures.attempt_class([text], healthy=lambda: pytest.fail("no health check without a timeout or a cut"))[0] == failures.AGENT


def test_one_failed_call_of_the_infrastructure_makes_the_attempt_infrastructure_whatever_the_runner_returned():
    """The wrapper of a ported sub-agent turns a failed call into a sentence the supervisor reads, so one failed call of
    the infrastructure in the call records makes the attempt infrastructure even when the runner returned an answer."""
    calls = ["", "", "InternalServerError: Error code: 500", ""]
    assert failures.attempt_class(calls)[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class(["", "ToolTransportError: the session is closed"])[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class([], harness_error="FileNotFoundError: the store")[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class(["", "ValueError: bad plan"])[0] == failures.AGENT and failures.attempt_class(["", ""])[0] == ""


def test_a_timeout_is_the_agents_while_the_endpoint_answers_and_infrastructure_when_it_does_not():
    """The health function is called once, and only for a timeout or a cut."""
    calls = []

    def check(ok):
        def healthy():
            calls.append(ok)
            return {"ok": ok}
        return healthy

    assert failures.attempt_class(["APITimeoutError: Request timed out."], healthy=check(True))[0] == failures.AGENT
    assert failures.attempt_class(["ReadTimeout: no answer in time"], healthy=check(False))[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class([], cut=True, healthy=check(True))[0] == failures.AGENT
    assert failures.attempt_class([], cut=True, healthy=check(False))[0] == failures.INFRASTRUCTURE
    assert calls == [True, False, True, False]
    # an error of the infrastructure decides without the check
    assert failures.attempt_class(["APITimeoutError: x", "ConnectError: refused"], healthy=check(True))[0] == failures.INFRASTRUCTURE and len(calls) == 4


def test_three_harness_errors_in_a_row_are_due_and_two_with_a_good_row_between_are_not():
    bad, good = {"error_class": failures.HARNESS_ERROR}, {"error_class": ""}
    assert failures.stop_due([good, bad, bad, bad], 3) and not failures.stop_due([bad, bad, good, bad, bad], 3)
    assert not failures.stop_due([bad, bad], 3) and not failures.stop_due([], 3)


def test_the_health_check_of_an_endpoint_that_does_not_answer_says_so():
    record = failures.endpoint_health("http://127.0.0.1:9/v1", "key", 1)
    assert record["ok"] is False and record["status"] is None and record["error"] and record["seconds"] >= 0
