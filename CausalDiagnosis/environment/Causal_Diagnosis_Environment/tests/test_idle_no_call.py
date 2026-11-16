"""The idle hook sends one no_call step only for an iteration that made no counted call, nothing outside an episode, and
the environment counts the no_call as a step that charges nothing and logs the event no_call."""
import asyncio

from CausalDiagnosis.agents.utils import idle_hook as idle


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
