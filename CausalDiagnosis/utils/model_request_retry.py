"""The one retry policy of the benchmark's model clients. A model request
that the service fails is attempted again, with no count of attempts, after the registered wait between attempts. A failure is
no connection, or a status of 429, 502 or 503; a model's reply cannot trigger one. Inside an episode each attempt is given the
time left before the deadline of the row's wall limit, and the policy raises DeadlineReached once that deadline has passed;
on a row without a wall limit an attempt inside an episode has no read timeout, since the episode's own end stops it as it
stops a tool call; outside an episode (the judge, the reflection model of training) each attempt has the read
timeout, an attempt that it ends is a failed attempt, and a failed attempt is sent again only while the health probe named
for its endpoint answers (health_check): a probe that does not answer ends the request with ServiceDown, and with no probe
named nothing tells a dead endpoint from a live one, so the failed attempt, bounded by the read timeout, is the request's
last. Every failed attempt is written to the episode's ledger of failed spans (CausalDiagnosis.utils.model_failures), from the
start of the attempt to the start of the next one, so that the thinking clock does not charge it.

The deadline is a monotonic instant, or a zero-argument callable that returns it, since the row's wall limit moves while the
episode plays: it is extended by the credited time of each first start of a tool server and of each counted call's service. A caller passes it, or the driver names it
for the time of an episode (episode_deadline), as it names the episode folder for the ledgers (CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.token_clock.episode).
deadline_at reads either form, and every reader of the deadline goes through it. Where the driver cuts the episode at the
deadline itself (driver_cut), the wait of a call does not end at its own read of the deadline: the cut ends it, so the facts of
the cut are taken while the call is open, and an answer that comes back before the cut is delivered."""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager

# The two constants of the policy, each stated with its source; registered_analysis.json states both under "model_request", and a
# test fails when a registered value and its constant differ. No reader of the registered file sits in this layer.
# the wait between attempts: watsonx's ModelInference(delay_time=2) of the paper's agents (AssetOpsBench)
RETRY_DELAY_S = 2.0
# the read timeout of an attempt outside an episode: the 600 seconds of the openai client's default timeout that every request
# had before; inside an episode the row's deadline is the only bound
READ_TIMEOUT_S = 600.0
STATUSES = (429, 502, 503)
CONNECT = "connect"  # the status of an attempt that got no connection
TIMEOUT = "timeout"  # the status of an attempt outside an episode that the read timeout ended

_DEADLINE = None
_DRIVER_CUT = False
_LOCK = threading.Lock()


def _fresh_episode():
    """What one episode of the driver has open (model requests in flight, waits after a failed attempt), the monotonic instant
    each kind last closed, and the cached prompt tokens its first answered model request reported."""
    return {"open": {"request": 0, "wait": 0}, "closed": {"request": None, "wait": None}, "replied": False, "first_cached_tokens": None}


_EPISODE = _fresh_episode()
_HEALTH = None


class DeadlineReached(TimeoutError):
    """The deadline of the row's wall limit passed before the model request was answered."""


class ServiceDown(RuntimeError):
    """Outside an episode, a failed attempt met an endpoint whose health probe does not answer: the program ends with SERVICE_DOWN
    instead of waiting for ever on a dead service."""

    code = "SERVICE_DOWN"


@contextmanager
def health_check(check):
    """Name the health probe of the endpoint (a zero-argument callable, true while the endpoint answers) for every model request
    made outside an episode, such as those of the judge, the rescoring, the reproduction scorer, GEPA or run_checks. After a failed
    attempt with no deadline named, the policy asks the probe and ends with ServiceDown when it is false; outside an episode with
    no probe named, a failed attempt is not sent again (_sent_again)."""
    global _HEALTH
    before, _HEALTH = _HEALTH, check
    try:
        yield
    finally:
        _HEALTH = before


def _sent_again(deadline):
    """Whether a failed attempt is sent again: always with a deadline named, which bounds the retrying, and inside an episode,
    whose clock and cut govern; outside an episode with neither, only while a health probe is named, since nothing else tells a
    dead endpoint from a live one, so without a probe the failed attempt is the request's last and no request waits for ever."""
    return deadline_at(deadline) is not None or _HEALTH is not None or in_episode()


def _still_served(deadline):
    """After a failed attempt that is sent again: with a deadline named the deadline bounds the wait; without one the named
    health probe decides, and inside an episode with neither the policy attempts again."""
    if deadline_at(deadline) is None and _HEALTH is not None and not _HEALTH():
        raise ServiceDown("the endpoint's health probe does not answer after a failed attempt and no episode deadline is named")


@contextmanager
def episode_deadline(deadline):
    """Name the deadline of the episode the driver plays (a monotonic instant or a callable returning it), for every model
    request of the episode. The outermost naming starts the episode's record of open requests and first reply."""
    global _DEADLINE, _EPISODE
    if _DEADLINE is None:
        _EPISODE = _fresh_episode()
    before, _DEADLINE = _DEADLINE, deadline
    try:
        yield
    finally:
        _DEADLINE = before


@contextmanager
def driver_cut():
    """Name the driver's cut at the deadline for the runner it plays (run_episodes.run_limited), in every thread of the runner:
    while it is named, a wait that the deadline would end (waits_to_deadline) waits on past the deadline until the answer comes
    or the cut ends the call, so whichever reads the deadline first, the driver takes the facts of the cut while the call is
    open, and an answer that came back is never read as unanswered."""
    global _DRIVER_CUT
    before, _DRIVER_CUT = _DRIVER_CUT, True
    try:
        yield
    finally:
        _DRIVER_CUT = before


def current_deadline():
    """The deadline the driver named, as it was named (an instant, a callable or None); read it with deadline_at."""
    return _DEADLINE


def deadline_at(deadline):
    """The monotonic instant of a deadline given as an instant, a zero-argument callable or None (no deadline)."""
    return deadline() if callable(deadline) else deadline


@contextmanager
def _phase(kind):
    """Count a model request ("request") or the wait after a failed attempt ("wait") as open while it runs."""
    episode = _EPISODE
    with _LOCK:
        episode["open"][kind] += 1
    try:
        yield
    finally:
        with _LOCK:
            episode["open"][kind] -= 1
            episode["closed"][kind] = time.monotonic()


def open_at(instant):
    """{"model_request_open", "failed_span_open"} of the episode at a monotonic instant that has just passed, the deadline of a
    cut: a kind counts as open when one is open now or one closed at or after the instant, since a request given the time left
    before the deadline, or a wait cut short at it, closes at the deadline itself."""
    with _LOCK:
        def was_open(kind):
            closed = _EPISODE["closed"][kind]
            return _EPISODE["open"][kind] > 0 or (closed is not None and closed >= instant)
        return {"model_request_open": was_open("request"), "failed_span_open": was_open("wait")}


def reported_cached_tokens(usage):
    """The cached prompt tokens that the usage block of a reply reports in prompt_tokens_details.cached_tokens (an SDK object or
    the JSON of the reply), and None when it reports no count: no usage, no details or no cached_tokens. None is no zero, since
    SGLang leaves the details out of a reply that reused no cached token under --enable-cache-report and out of every reply
    without that flag, and vLLM out of every reply without --enable-prompt-tokens-details. llm_calls.json records it for every
    model call."""
    if usage is None:
        return None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    return None if cached is None else int(cached)


def reply_cached_tokens(outcome):
    """The cached prompt tokens an answered request reports (reported_cached_tokens of an SDK object or of the JSON of an httpx
    response): 0 when the usage holds no count, as SGLang with --enable-cache-report answers a request that reused nothing,
    None when the reply holds no usage."""
    if hasattr(outcome, "usage"):
        usage = outcome.usage
    else:
        try:
            usage = (outcome.json() or {}).get("usage")
        except Exception:  # noqa: BLE001 - a body that is no JSON object reports no usage
            usage = None
    if usage is None:
        return None
    cached = reported_cached_tokens(usage)
    return 0 if cached is None else cached


def _note_reply(outcome):
    tokens = reply_cached_tokens(outcome)
    with _LOCK:
        if not _EPISODE["replied"]:
            _EPISODE["replied"], _EPISODE["first_cached_tokens"] = True, tokens


def first_cached_tokens():
    """The cached prompt tokens of the episode's first answered model request, None before one answered or when it reported no
    usage."""
    return _EPISODE["first_cached_tokens"]


def failure_status(outcome, statuses=STATUSES):
    """The status of a failed attempt, or None: no connection (httpx.ConnectError or the openai SDK's APIConnectionError that is
    no timeout), or a response or an SDK error whose status_code is one of statuses. A timeout is no failure here: _timed_out
    makes one outside an episode a failed attempt, and inside one it comes only at the deadline."""
    names = [k.__name__ for k in type(outcome).__mro__]
    if "ConnectError" in names or ("APIConnectionError" in names and "APITimeoutError" not in names):
        return CONNECT
    code = getattr(outcome, "status_code", None)
    return code if code in statuses else None


def _record(start, next_start, status):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
    from CausalDiagnosis.utils import model_failures  # at the failure, so that importing a client loads no server file

    folder = token_clock.episode_folder()
    if folder is not None:
        model_failures.add_failed(folder, start, next_start, status)


def deadline_left(deadline=None):
    """The seconds left before a deadline (an instant, a callable or None), the one the driver named when none is given: None
    without a deadline, DeadlineReached once it has passed. Every blocking call of an episode reads its time through it."""
    deadline = deadline_at(_DEADLINE if deadline is None else deadline)
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left <= 0:
        raise DeadlineReached("the deadline of the episode passed before the call was answered")
    return left


def waits_to_deadline(deadline=None):
    """The waits of one call bounded by a deadline (an instant, a callable or None, the one the driver named when none is
    given): the seconds left, read again before every wait, so a credit that moves the deadline while the call is open lets
    the wait go on, as run_limited does. Without a deadline one None, a wait without bound; the waits end once the deadline
    has passed, and the caller then gives the call up, except under the driver's cut (driver_cut), where the last wait is one
    None, without bound, and the cut ends the call."""
    while True:
        at = deadline_at(_DEADLINE if deadline is None else deadline)
        if at is None:
            yield None
            return
        left = at - time.monotonic()
        if left <= 0:
            if _DRIVER_CUT:
                yield None
            return
        yield left


def in_episode():
    """Whether the driver plays an episode now: it names the episode's folder for the ledgers (CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.token_clock)."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock  # at the call, so that importing a client loads no environment file

    return token_clock.episode_folder() is not None


def attempt_timeout(deadline=None):
    """The read bound of one attempt: the seconds left before the deadline of the row's wall limit; inside an episode on a row
    without one None, no read timeout; outside an episode the read timeout. A client sends None as a read without
    a bound, with its own connect timeout. The rule covers the agent's own work inside an episode, its model requests and its
    tool calls, which carry no timeout of their own; the harness's own calls (the probe at the cut, the cache flush, the
    opening check, the state reads) keep their declared timeouts, and a failed one is the harness's attempt, never an end of
    the agent. The episode is the scope the driver opens around the agent alone (token_clock.episode)."""
    left = deadline_left(deadline)
    if left is not None:
        return left
    return None if in_episode() else READ_TIMEOUT_S


def _left(deadline):
    return attempt_timeout(deadline)


def _timed_out(exc, deadline):
    """Whether an attempt outside an episode ended at its read timeout, which is a failed attempt of the rule; inside an episode
    a timeout comes only at the deadline, which the next attempt reports as DeadlineReached."""
    names = [k.__name__ for k in type(exc).__mro__]
    return (deadline_at(deadline) is None and not in_episode()
            and any(n in names for n in ("TimeoutException", "APITimeoutError", "TimeoutError")))


def _wait(deadline, delay_s):
    """The registered wait, cut short where the deadline comes first."""
    deadline = deadline_at(deadline)
    return delay_s if deadline is None else min(delay_s, max(0.0, deadline - time.monotonic()))


async def _asleep(seconds):
    """asyncio.sleep that ends no earlier than its bound, as time.sleep does: the event loop runs a timer up to its clock
    resolution early, and a wait that the deadline cut short must still be open at the deadline (open_at)."""
    until = time.monotonic() + seconds
    await asyncio.sleep(seconds)
    while (left := until - time.monotonic()) > 0:
        await asyncio.sleep(left)


def post_with_retries(send, *, deadline=None, delay_s=RETRY_DELAY_S, statuses=STATUSES, model_reply=True):
    """send(timeout) once per attempt, timeout the seconds left before the deadline (None without one), until an attempt is no
    failure or is a failure that is not sent again (_sent_again); returns that attempt's response, or raises its exception when
    it is an error that is no failure or a failure that is not sent again. model_reply False
    marks a request to the model's endpoint that is no model reply (the model listing, the tokenize count of CausalDiagnosis.utils.model_reply_room): it is
    attempted and recorded by the same rule, and its answer never stands for the episode's first reply."""
    deadline = _DEADLINE if deadline is None else deadline
    while True:
        start = time.monotonic()
        timeout = _left(deadline)
        try:
            with _phase("request"):
                outcome = send(timeout)
        except Exception as exc:  # noqa: BLE001 - classified below, raised again unless it is a failure sent again
            status = failure_status(exc, statuses) or (TIMEOUT if _timed_out(exc, deadline) else None)
            if status is None or not _sent_again(deadline):
                raise
        else:
            status = failure_status(outcome, statuses)
            if status is None:
                if model_reply:
                    _note_reply(outcome)
                return outcome
            if not _sent_again(deadline):
                return outcome  # the failed attempt is the request's last: its status goes back to the caller
        with _phase("wait"):
            time.sleep(_wait(deadline, delay_s))
        _record(start, time.monotonic(), status)
        _still_served(deadline)


async def apost_with_retries(send, *, deadline=None, delay_s=RETRY_DELAY_S, statuses=STATUSES):
    """post_with_retries for an async send(timeout)."""
    deadline = _DEADLINE if deadline is None else deadline
    while True:
        start = time.monotonic()
        timeout = _left(deadline)
        try:
            with _phase("request"):
                outcome = await send(timeout)
        except Exception as exc:  # noqa: BLE001 - classified below, raised again unless it is a failure sent again
            status = failure_status(exc, statuses) or (TIMEOUT if _timed_out(exc, deadline) else None)
            if status is None or not _sent_again(deadline):
                raise
        else:
            status = failure_status(outcome, statuses)
            if status is None:
                _note_reply(outcome)
                return outcome
            if not _sent_again(deadline):
                return outcome  # the failed attempt is the request's last: its status goes back to the caller
        with _phase("wait"):
            await _asleep(_wait(deadline, delay_s))
        _record(start, time.monotonic(), status)
        _still_served(deadline)
