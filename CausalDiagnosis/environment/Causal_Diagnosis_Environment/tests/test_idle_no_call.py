"""The idle hook sends one no_call step only for an iteration that made no counted call, nothing outside an episode, and
the environment counts the no_call as a step that charges nothing and logs the event no_call. A no_call step answers no
agent, so the notices of its thinking reach the next counted call."""
import asyncio

import pytest

from CausalDiagnosis.agents.utils import idle_hook as idle
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import events, order, read, submit_truth, wait


def test_the_hook_steps_only_for_an_idle_iteration_and_only_inside_an_episode():
    sent = []
    assert idle.iteration_end(False) is False  # no episode bound: nothing is sent
    with idle.bound(lambda: sent.append(1) or False):
        assert idle.iteration_end(True) is False and sent == []
        assert idle.iteration_end(False) is False and sent == [1]
        assert asyncio.run(idle.aiteration_end(False)) is False and sent == [1, 1]
    assert idle.iteration_end(False) is False and sent == [1, 1]


def test_the_hook_reports_the_end_the_environment_gave():
    with idle.bound(lambda: True):
        assert idle.iteration_end(False) is True


def test_a_no_call_step_counts_a_step_charges_nothing_and_is_logged(make_env):
    env = make_env()
    env.reset(0)
    steps = env.steps
    _envelope, cost, _done, info = env.step({"type": "no_call"})
    assert env.steps == steps + 1 and cost == 0 and "refused" not in info
    assert env.log[-1]["event"] == "no_call" and env.log[-1]["cost"] == 0


def test_a_no_call_whose_thinking_reaches_the_horizon_is_logged_as_a_no_call_and_ends_the_episode(make_env):
    env = make_env(clock_mode="generation_time")
    env.reset(0)
    _envelope, cost, done, info = env.step({"type": "no_call"}, think_cycles=env.config.horizon + 10)
    # the no_call pays no price: what the step charged is the downtime and the penalties of the cycles its thinking played
    assert done and env.prices_paid == 0 and cost == env.ledger["downtime"] + env.ledger["penalties"] and "refused" not in info
    assert [e["event"] for e in env.log if e.get("event") in ("no_call", "refused")] == ["no_call"]
    assert env.log[-1]["event"] == "end" and env.log[-1]["reason"] == "horizon"


# ----- the notices of a no_call step -----

def _closed_inside_a_no_call(make_env):
    """An order on the cooler, then a no_call step whose thinking runs past the close of the order: the environment, the answer of
    the no_call step and the close event, which lies inside that thinking."""
    env = make_env(clock_mode="generation_time")
    env.reset(0)
    submit_truth(env)
    order(env, "cooler")
    created = env.time
    durations = env.config.durations
    think = durations["lead_time"]["cooler"] + durations["repair_downtime"]["cooler"] + 2
    no_call, _cost, _done, _info = env.step({"type": "no_call"}, think_cycles=think)
    closed = events(env.log, "work_order_closed")
    assert len(closed) == 1 and created < closed[0]["t_end"] <= no_call["time"], closed
    return env, no_call, closed[0]


def _closes(envelope):
    return [(n["id"], n["t"]) for n in envelope["notifications"] if n.get("event") == "work_order_closed"]


@pytest.mark.parametrize("after", ["read", "wait"])
def test_the_notices_of_a_no_call_step_reach_the_next_call(make_env, after):
    """A no_call step answers no agent, so the close of a work order inside its thinking stays pending: the next call, a read or a
    wait, carries it with the cycle of the close, a wait returns at once on it, and the call after that does not carry it again.
    The answer of the no_call step lists it too, as the record of that step."""
    env, no_call, close = _closed_inside_a_no_call(make_env)
    assert _closes(no_call) == [(close["id"], close["t_end"])]
    before = env.time
    answer = read(env)[0] if after == "read" else wait(env, 5)[0]
    assert _closes(answer) == [(close["id"], close["t_end"])], answer["notifications"]
    if after == "wait":
        assert answer["result"]["ended_by"] == "notification" and answer["result"]["waited"] == 0 and env.time == before
    assert _closes(read(env)[0]) == []


def test_every_no_call_step_in_a_row_lists_the_pending_notices_and_the_next_call_carries_them_once(make_env):
    """No_call steps in a row: the answer of each lists every notice pending since the last counted call, the next counted call
    carries them once, and the log holds the event once."""
    env, first, close = _closed_inside_a_no_call(make_env)
    second = env.step({"type": "no_call"})[0]
    assert _closes(first) == _closes(second) == [(close["id"], close["t_end"])]
    assert _closes(read(env)[0]) == [(close["id"], close["t_end"])]
    assert _closes(read(env)[0]) == []
    assert len(events(env.log, "work_order_closed")) == 1


def test_a_counted_call_still_hands_its_notices_over_once(make_env):
    """The control of the rule: the same close inside the thinking of a counted call reaches that call's answer and no later one."""
    env = make_env(clock_mode="generation_time")
    env.reset(0)
    submit_truth(env)
    order(env, "cooler")
    durations = env.config.durations
    answer = env.step({"type": "read_sensors", "sensor_group": "tier1", "cycles": 1},
                      think_cycles=durations["lead_time"]["cooler"] + durations["repair_downtime"]["cooler"] + 2)[0]
    close = events(env.log, "work_order_closed")[0]
    assert _closes(answer) == [(close["id"], close["t_end"])]
    assert _closes(read(env)[0]) == []
