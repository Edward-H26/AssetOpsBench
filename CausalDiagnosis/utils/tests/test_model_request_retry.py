"""The one retry policy of the model clients (CausalDiagnosis.utils.model_request_retry) and the ledger of failed attempts (CausalDiagnosis.utils.model_failures):
no count of attempts, a failure is no connection or a status of 429, 502 or 503, each attempt is given the time left before the
deadline and the policy raises once it has passed, and every failed attempt lands in the episode's ledger as the span from its
start to the start of the next attempt."""
import asyncio
import json
import time
import types

import httpx
import pytest

from CausalDiagnosis.utils import model_request_retry as retry
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
from CausalDiagnosis.utils import model_failures


class Reply:
    def __init__(self, status_code):
        self.status_code = status_code


def sender(outcomes, seen=None):
    """send(timeout) that hands out the outcomes in order, raising those that are exceptions, and keeps every timeout."""
    outcomes = list(outcomes)

    def send(timeout):
        if seen is not None:
            seen.append(timeout)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    return send


REQUEST = httpx.Request("POST", "http://endpoint.example/v1/completions")


@pytest.mark.parametrize("failure, status", [(httpx.ConnectError("no answer", request=REQUEST), retry.CONNECT), (Reply(429), 429),
                                             (Reply(502), 502), (Reply(503), 503)])
def test_each_failure_is_attempted_again_and_written_to_the_ledger(tmp_path, failure, status):
    with token_clock.episode(tmp_path):
        before = time.monotonic()
        answer = retry.post_with_retries(sender([failure, Reply(200)]), delay_s=0.01)
        after = time.monotonic()
    assert answer.status_code == 200
    lines = [json.loads(line) for line in (tmp_path / model_failures.NAME).read_text().splitlines()]
    assert [line["status"] for line in lines] == [status]
    start, next_start = lines[0]["start"], lines[0]["next_start"]
    assert before <= start < next_start <= after and next_start - start >= 0.01  # the span holds the registered wait


def test_there_is_no_count_of_attempts(tmp_path):
    with token_clock.episode(tmp_path):
        answer = retry.post_with_retries(sender([Reply(503)] * 40 + [Reply(200)]), delay_s=0.0)
    assert answer.status_code == 200
    spans, _cursor = model_failures.failed_spans_since(tmp_path, 0)
    assert len(spans) == 40 and all(a <= b for a, b in spans) and all(b <= c for (_a, b), (c, _d) in zip(spans, spans[1:]))


@pytest.mark.parametrize("status", [400, 404, 408, 409, 500, 504])
def test_an_answer_that_is_no_failure_goes_back_to_the_caller_once(tmp_path, status):
    seen = []
    with token_clock.episode(tmp_path):
        answer = retry.post_with_retries(sender([Reply(status), Reply(200)], seen), delay_s=0.0)
    assert answer.status_code == status and len(seen) == 1 and not (tmp_path / model_failures.NAME).exists()


def test_an_error_that_is_no_failure_is_raised_at_once():
    """A reply the client cannot read is raised at once; a read timeout inside an episode is raised too, since there it comes
    only at the deadline."""
    seen = []
    with pytest.raises(ValueError):
        retry.post_with_retries(sender([ValueError("a reply the client cannot read"), Reply(200)], seen), delay_s=0.0)
    assert len(seen) == 1
    seen.clear()
    with retry.episode_deadline(time.monotonic() + 60), pytest.raises(httpx.ReadTimeout):
        retry.post_with_retries(sender([httpx.ReadTimeout("slow", request=REQUEST), Reply(200)], seen), delay_s=0.0)
    assert len(seen) == 1


def test_outside_an_episode_each_attempt_has_the_read_timeout_and_a_timed_out_attempt_is_attempted_again():
    """Outside an episode an attempt is given READ_TIMEOUT_S (the 600 s the client had), and an attempt that the read timeout
    ends is a failed attempt: it is attempted again while the endpoint's health probe answers, and a dead endpoint ends the call."""
    seen = []
    with retry.health_check(lambda: True):
        reply = retry.post_with_retries(sender([httpx.ReadTimeout("slow", request=REQUEST), Reply(200)], seen), delay_s=0.0)
    assert reply.status_code == 200 and seen == [retry.READ_TIMEOUT_S, retry.READ_TIMEOUT_S] == [600.0, 600.0]
    with retry.health_check(lambda: False), pytest.raises(retry.ServiceDown):
        retry.post_with_retries(sender([httpx.ReadTimeout("slow", request=REQUEST), Reply(200)]), delay_s=0.0)


def test_an_sdk_error_is_read_by_its_status_and_a_timeout_is_no_failure():
    sdk = types.SimpleNamespace(**{n: type(n, (Exception,), {}) for n in ("APIConnectionError",)})
    timeout_error = type("APITimeoutError", (sdk.APIConnectionError,), {})
    status_error = type("APIStatusError", (Exception,), {"status_code": 503})
    assert retry.failure_status(sdk.APIConnectionError()) == retry.CONNECT
    assert retry.failure_status(timeout_error()) is None
    assert retry.failure_status(status_error()) == 503 and retry.failure_status(Reply(500)) is None


def test_each_attempt_has_the_time_left_and_the_deadline_ends_the_retrying(tmp_path):
    seen = []
    deadline = time.monotonic() + 0.3
    with token_clock.episode(tmp_path), pytest.raises(retry.DeadlineReached):
        retry.post_with_retries(sender([Reply(502)] * 1000, seen), deadline=deadline, delay_s=0.05)
    assert seen and all(0 < left <= 0.3 for left in seen) and seen == sorted(seen, reverse=True)
    spans, _cursor = model_failures.failed_spans_since(tmp_path, 0)
    assert len(spans) == len(seen) and spans[-1][1] <= deadline + 0.05  # the last wait is cut at the deadline


def test_a_deadline_given_as_a_callable_is_read_at_every_attempt_and_may_move_later():
    """The row's wall limit moves while the episode plays, by the credited time of the first starts of the tool servers and of the counted calls' service, so the deadline may
    be a callable: each attempt reads it again, an extension lets the retrying go on, and the policy stops once the moved
    deadline has passed."""
    start = time.monotonic()
    moved = {"at": start + 0.2}
    seen = []

    def send(timeout):
        seen.append(timeout)
        if len(seen) == 2:
            moved["at"] += 0.3  # a credit lands during the retrying and extends the limit
        return Reply(503)

    with pytest.raises(retry.DeadlineReached):
        retry.post_with_retries(send, deadline=lambda: moved["at"], delay_s=0.05)
    assert retry.deadline_at(lambda: 7.0) == 7.0 and retry.deadline_at(None) is None and retry.deadline_at(3.0) == 3.0
    assert seen[2] > seen[1] and time.monotonic() - start >= 0.45  # the attempt after the extension had more time left
    with retry.episode_deadline(lambda: moved["at"]):
        assert retry.deadline_at(retry.current_deadline()) == moved["at"]
        with pytest.raises(retry.DeadlineReached):
            retry.deadline_left()  # the moved deadline has passed by now
    assert retry.deadline_left() is None  # outside an episode
    with retry.episode_deadline(lambda: time.monotonic() + 5):
        assert 4 < retry.deadline_left() <= 5


def test_the_deadline_the_driver_names_holds_for_a_call_that_passes_none():
    seen = []
    with retry.episode_deadline(time.monotonic() + 5):
        retry.post_with_retries(sender([Reply(200)], seen))
    assert 0 < seen[0] <= 5 and retry.current_deadline() is None
    seen.clear()
    retry.post_with_retries(sender([Reply(200)], seen))
    assert seen == [retry.READ_TIMEOUT_S]  # outside an episode the read timeout of the client that every request had


def test_outside_an_episode_no_ledger_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with retry.health_check(lambda: True):
        assert retry.post_with_retries(sender([Reply(503), Reply(200)]), delay_s=0.0).status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_the_async_policy_is_the_same(tmp_path):
    outcomes = [httpx.ConnectError("no answer", request=REQUEST), Reply(429), Reply(200)]

    async def send(timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    with token_clock.episode(tmp_path):
        answer = asyncio.run(retry.apost_with_retries(send, delay_s=0.0))
    assert answer.status_code == 200
    statuses = [json.loads(line)["status"] for line in (tmp_path / model_failures.NAME).read_text().splitlines()]
    assert statuses == [retry.CONNECT, 429]


def test_the_ledger_reads_the_spans_since_a_cursor(tmp_path):
    assert model_failures.failed_spans_since(tmp_path, 0) == ([], 0)
    assert model_failures.failed_spans_since(tmp_path, 17) == ([], 0)  # a missing ledger
    model_failures.add_failed(tmp_path, 1.0, 3.0, 503)
    model_failures.add_failed(tmp_path, 3.0, 5.5, retry.CONNECT)
    spans, cursor = model_failures.failed_spans_since(tmp_path, 0)
    assert spans == [(1.0, 3.0), (3.0, 5.5)] and cursor == (tmp_path / model_failures.NAME).stat().st_size
    assert model_failures.failed_spans_since(tmp_path, cursor) == ([], cursor)
    model_failures.add_failed(tmp_path, 6.0, 8.0, 429)
    with (tmp_path / model_failures.NAME).open("a", encoding="utf-8") as ledger:
        ledger.write('{"start": 9.0, "next')  # a line still being written is read by a later call
    later, cursor2 = model_failures.failed_spans_since(tmp_path, cursor)
    assert later == [(6.0, 8.0)] and cursor2 > cursor


def test_the_completion_route_sends_through_the_policy(tmp_path):
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import CompletionRoute

    answers = [httpx.Response(503, request=REQUEST), httpx.Response(200, json={"choices": [{"text": "x"}]}, request=REQUEST)]
    posts = []

    class Client:
        def request(self, method, url, json=None, timeout=None):
            posts.append(timeout)
            return answers.pop(0)
    route = CompletionRoute("http://endpoint.example/v1", "k", "m", client=Client())
    with token_clock.episode(tmp_path):
        assert route._post("completions", {}) == {"choices": [{"text": "x"}]}
    assert len(posts) == 2 and [json.loads(line)["status"] for line in (tmp_path / model_failures.NAME).read_text().splitlines()] == [503]


def test_outside_an_episode_a_dead_endpoint_ends_with_service_down_and_a_live_one_is_attempted_again():
    """With no deadline named, a failed attempt asks the named health probe; a probe that does not answer ends the
    call with ServiceDown (SERVICE_DOWN) after that one attempt, a probe that answers lets the policy attempt again, and inside an
    episode the deadline bounds the wait and the probe is not asked."""
    asked = []

    def probe(answer):
        def check():
            asked.append(answer)
            return answer
        return check

    seen = []
    with retry.health_check(probe(False)):
        with pytest.raises(retry.ServiceDown) as down:
            retry.post_with_retries(sender([Reply(503), Reply(200)], seen), delay_s=0.0)
    assert down.value.code == "SERVICE_DOWN" and len(seen) == 1 and asked == [False]
    asked.clear()
    with retry.health_check(probe(True)):
        assert retry.post_with_retries(sender([Reply(503), Reply(502), Reply(200)]), delay_s=0.0).status_code == 200
    assert asked == [True, True]
    asked.clear()
    with retry.health_check(probe(False)), retry.episode_deadline(time.monotonic() + 5):
        assert retry.post_with_retries(sender([Reply(503), Reply(200)]), delay_s=0.0).status_code == 200
    assert asked == []
    with retry.health_check(probe(False)):
        with pytest.raises(retry.ServiceDown):
            asyncio.run(retry.apost_with_retries(async_sender([Reply(429), Reply(200)]), delay_s=0.0))


def test_outside_an_episode_without_a_probe_a_failed_attempt_is_the_last_and_with_a_probe_or_a_deadline_it_is_sent_again(
        monkeypatch):
    """Outside an episode with no deadline and no health probe named, nothing tells a dead endpoint from a live one, so a failed
    attempt is not sent again and no wait follows it: no connection and a read timeout are raised as they came, and a status of
    429, 502 or 503 goes back to the caller. The controls: with a probe that answers and with a deadline named, the same
    failures are sent again after the registered wait, as they are inside an episode (the first tests of this file)."""
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    for failure in (httpx.ConnectError("refused", request=REQUEST), httpx.ReadTimeout("slow", request=REQUEST)):
        seen = []
        with pytest.raises(type(failure)):
            retry.post_with_retries(sender([failure, Reply(200)], seen))
        assert seen == [retry.READ_TIMEOUT_S]
    seen = []
    assert retry.post_with_retries(sender([Reply(503), Reply(200)], seen)).status_code == 503 and len(seen) == 1
    with pytest.raises(httpx.ConnectError):
        asyncio.run(retry.apost_with_retries(async_sender([httpx.ConnectError("refused", request=REQUEST), Reply(200)])))
    assert waits == []
    failing = [httpx.ConnectError("refused", request=REQUEST), Reply(503), Reply(200)]
    with retry.health_check(lambda: True):
        assert retry.post_with_retries(sender(failing)).status_code == 200
    assert retry.post_with_retries(sender(failing), deadline=time.monotonic() + 60).status_code == 200
    assert waits == [retry.RETRY_DELAY_S] * 4


def test_the_waits_of_a_call_end_at_the_deadline_except_under_the_drivers_cut():
    """A call gives itself up once the deadline has passed (waits_to_deadline ends), except while the driver cuts the episode
    itself (driver_cut): then the last wait is one without bound and the cut ends the call, so an answer that comes back before
    the cut is delivered. The control: outside the driver's cut the waits end at the deadline as before."""
    passed = time.monotonic() - 1.0
    assert list(retry.waits_to_deadline(passed)) == []
    with retry.driver_cut():
        assert list(retry.waits_to_deadline(passed)) == [None]
        assert 59.0 < next(retry.waits_to_deadline(time.monotonic() + 60.0)) <= 60.0
    assert list(retry.waits_to_deadline(passed)) == []


def async_sender(outcomes):
    send = sender(outcomes)

    async def asend(timeout):
        return send(timeout)
    return asend


def test_inside_an_episode_without_a_wall_limit_an_attempt_has_no_read_timeout(tmp_path):
    """A model request inside an episode carries no read timeout of its own; on a row without a wall limit the
    attempt is sent with no read bound (None), and a timeout there is no failed attempt of the rule, so it is raised. Outside
    an episode the read timeout stays."""
    seen = []
    with token_clock.episode(tmp_path):
        assert retry.attempt_timeout() is None
        reply = retry.post_with_retries(sender([Reply(200)], seen), delay_s=0.0)
        assert reply.status_code == 200 and seen == [None]
        with pytest.raises(httpx.ReadTimeout):
            retry.post_with_retries(sender([httpx.ReadTimeout("slow", request=REQUEST), Reply(200)]), delay_s=0.0)
    assert retry.attempt_timeout() == retry.READ_TIMEOUT_S



def test_the_reported_cached_tokens_are_the_count_a_reply_carries_and_none_without_one():
    """reported_cached_tokens reads usage.prompt_tokens_details.cached_tokens of an SDK object and of the JSON of a reply alike, a
    count of zero included, and gives None where the reply carries no count, since SGLang omits the details of a reply that
    reused nothing and every server without its flag omits them always. The control: reply_cached_tokens, the reader of the
    cache flush check, reads the same replies without a count as 0, so only the strict reader tells no count from a zero."""
    def sdk(details):
        return types.SimpleNamespace(prompt_tokens=9, prompt_tokens_details=details)

    for usage, count in ((sdk(types.SimpleNamespace(cached_tokens=12)), 12), (sdk(types.SimpleNamespace(cached_tokens=0)), 0),
                         (sdk(types.SimpleNamespace(cached_tokens=None)), None), (sdk(None), None), (types.SimpleNamespace(prompt_tokens=9), None),
                         ({"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": 12}}, 12),
                         ({"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": 0}}, 0), ({"prompt_tokens": 9, "prompt_tokens_details": {}}, None),
                         ({"prompt_tokens": 9, "prompt_tokens_details": None}, None), ({"prompt_tokens": 9}, None), ({}, None), (None, None)):
        assert retry.reported_cached_tokens(usage) == count, usage
    for usage in (sdk(None), {"prompt_tokens": 9}, {"prompt_tokens": 9, "prompt_tokens_details": {}}):
        assert retry.reported_cached_tokens(usage) is None and retry.reply_cached_tokens(types.SimpleNamespace(usage=usage)) == 0, usage
