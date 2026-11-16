"""The deadline of an episode: measured from the handover, moved by the credited seconds of the state,
read by every runner and every tool call of the episode, and never by a call outside one."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.agents.utils import mcp_sessions
from CausalDiagnosis.utils import model_request_retry as retry


def _state(path, **fields):
    path.write_text(json.dumps(fields), encoding="utf-8")


def test_the_limit_counts_from_the_handover_and_moves_with_credits(tmp_path):
    state = tmp_path / "state.json"
    _state(state, handover_mono=100.0, credited_seconds={"starts": 0.0, "calls": 0.0})
    wall = run_episodes.WallLimit(state, limit_s=30)
    assert wall.deadline() == 130.0
    _state(state, handover_mono=100.0, credited_seconds={"starts": 2.5, "calls": 10.0})
    assert wall.deadline() == 142.5  # a credit written after the start moves the deadline


def test_a_state_without_a_handover_counts_from_now(tmp_path):
    before = time.monotonic()
    wall = run_episodes.WallLimit(tmp_path / "missing.json", limit_s=30)
    assert before + 30 <= wall.deadline() <= time.monotonic() + 30


def test_a_credit_that_arrives_while_the_runner_plays_delays_the_cut():
    moved = {"credit": 0.0}
    start = time.monotonic()
    given = lambda: start + 0.2 + moved["credit"]  # noqa: E731

    async def runner():
        await asyncio.sleep(0.1)
        moved["credit"] = 0.4  # the servers credit their start while the agent plays
        await asyncio.sleep(0.3)
        return None

    with retry.episode_deadline(given):
        _result, error, seconds = run_episodes.run_limited(runner)
    assert error is None and seconds >= 0.4  # without the credit the cut would come at 0.2 s


def test_the_runner_is_cut_at_the_deadline():
    start = time.monotonic()
    with retry.episode_deadline(start + 0.1):
        _result, error, seconds = run_episodes.run_limited(lambda: asyncio.sleep(5))
    assert error.startswith(run_episodes.WALL_LIMIT) and seconds < 1


def test_the_runner_reads_the_deadline_in_its_own_threads():
    seen = []

    async def runner():
        await asyncio.to_thread(lambda: seen.append(retry.deadline_at(retry.current_deadline()) - time.monotonic()))  # a port's thread

    start = time.monotonic()
    with retry.episode_deadline(lambda: start + 60):
        run_episodes.run_limited(runner)
    assert seen and 0 < seen[0] <= 60
    assert retry.current_deadline() is None  # nothing outside an episode reads a deadline


class _SlowSession:
    async def call_tool(self, tool, arguments):
        await asyncio.sleep(5)


def _call_with_deadline(seconds_left):
    sessions = mcp_sessions.EpisodeSessions({"rig": _SlowSession()}, {"rig": ["read_sensors"]})

    async def call():
        at = time.monotonic() + seconds_left  # a fixed instant: the wait reads the deadline again, so it must not move by itself
        with retry.episode_deadline(lambda: at):
            return await sessions.call("rig", "read_sensors", {})

    return asyncio.run(call())


def test_a_tool_call_after_the_deadline_is_not_sent():
    with pytest.raises(mcp_sessions.ToolTransportError, match="deadline of the episode has passed"):
        _call_with_deadline(-1)


def test_the_deadline_bounds_the_wait_of_a_call():
    started = time.monotonic()
    with pytest.raises(mcp_sessions.ToolTransportError):
        _call_with_deadline(0.2)  # a call has no limit of its own, the deadline ends the wait
    assert time.monotonic() - started < 2


class _CreditedSession:
    """A server whose call writes its own credit before the reply reaches the caller (the meter writes the line of the call,
    then the reply travels): the deadline moves from the first instant to the second while the call is open."""

    def __init__(self, deadline, credited_to, reply_after):
        self.deadline, self.credited_to, self.reply_after = deadline, credited_to, reply_after

    async def call_tool(self, tool, arguments):
        await asyncio.sleep(0.05)
        self.deadline[0] = self.credited_to
        await asyncio.sleep(self.reply_after - 0.05)
        return SimpleNamespace(content=[SimpleNamespace(text="answered")], isError=False)


def _credited_call(first_s, credited_s, reply_after):
    """(answer or the error, seconds waited) of one call whose credit moves the deadline from first_s to credited_s after its
    send, its reply coming reply_after seconds after the send."""
    start = time.monotonic()
    deadline = [start + first_s]
    sessions = mcp_sessions.EpisodeSessions({"rig": _CreditedSession(deadline, start + credited_s, reply_after)}, {"rig": ["repair"]})

    async def call():
        with retry.episode_deadline(lambda: deadline[0]):
            try:
                return await sessions.call("rig", "repair", {})
            except mcp_sessions.ToolTransportError as exc:
                return exc

    return asyncio.run(call()), time.monotonic() - start


def test_an_answer_that_arrives_inside_the_credit_of_its_own_call_is_delivered():
    """The deadline at the send lies 0.3 s ahead, the call's own credit moves it to 1.0 s, and the reply comes at 0.6 s: the
    wait reads the deadline again at its first expiry and the agent gets the answer."""
    outcome, waited = _credited_call(0.3, 1.0, 0.6)
    assert outcome == ("answered", False) and waited >= 0.6


def test_the_control_a_reply_past_the_credited_deadline_still_ends_the_wait_at_that_deadline():
    """The same call with its credit to 0.5 s and its reply at 1.5 s: the wait ends at the moved deadline, not at the first one
    and not at the reply, as a transport error."""
    outcome, waited = _credited_call(0.3, 0.5, 1.5)
    assert isinstance(outcome, mcp_sessions.ToolTransportError) and "before the deadline of the episode" in str(outcome)
    assert 0.5 <= waited < 1.4


def test_a_module_whose_file_is_no_absolute_path_names_no_loaded_file(monkeypatch, tmp_path):
    import sys
    import types
    fake = types.ModuleType("fake_namespace")
    fake.__file__ = "_ops.py"  # the name torch gives torch.ops
    monkeypatch.setitem(sys.modules, "fake_namespace", fake)
    folder = run_episodes.ROOT / "CausalDiagnosis" / "experiments" / "episode_runners"
    monkeypatch.chdir(folder)
    assert folder.relative_to(run_episodes.ROOT).as_posix() + "/_ops.py" not in run_episodes.loaded_paths()
