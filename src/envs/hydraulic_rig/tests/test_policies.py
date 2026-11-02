"""The reference policies (src/envs/hydraulic_rig/policies.py and greedy.py) on the v2 action set: the oracle recovers
every start and matches the oracle set, do nothing ends at the horizon, random is seeded and legal, greedy is
deterministic and keeps budget for its orders. Data-dependent assertions are relations, never values."""
import json
import random

import pytest

from envs.hydraulic_rig.classifiers import TierClassifiers
from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY
from envs.hydraulic_rig.greedy import GreedyPolicy, ServedClassifiers
from envs.hydraulic_rig.policies import DoNothingPolicy, OraclePolicy, RandomPolicy, run_episode

from .conftest import events, faulty_components

TIMED = {"schedule": {"mode": "seeded", "components": 1, "first_drop": [5, 10], "interval": 10},
         "external_events": {"mode": "seeded", "probability": 0.5, "window": [5, 30]}}
HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]


def passes(env, log):
    """The oracle's own criteria: the end is a finish, the rig is healthy, the orders equal the oracle set, none unnecessary."""
    created = events(log, "work_order_created")
    ordered = sorted(e["component"] for e in created)
    expected = sorted(o["component"] for o in env.oracle()["work_orders"])
    return (log[-1]["reason"] == "finish" and log[-1]["final"] == HEALTHY_LIST and ordered == expected
            and not any(e["unnecessary"] for e in created) and not events(log, "refused"))


def test_the_oracle_recovers_every_candidate_start_at_the_price_of_its_orders_plus_downtime(make_env, world):
    env = make_env()
    for start in env.candidates:
        single = make_env(allowed=[list(start)])
        log = run_episode(single, OraclePolicy(single), 0)
        assert passes(single, log), start
        orders = [e["component"] for e in events(log, "work_order_created")]
        assert sorted(orders) == sorted(faulty_components(start))
        assert log[-1]["cost"] == sum(single.config.order_price(c) for c in orders) + log[-1]["downtime_total"]
        assert events(log, "submit_diagnosis")[0]["diagnosis"] == dict(zip(COMPONENTS, start))
        assert events(log, "submit_diagnosis")[0]["step"] < min(e["step"] for e in events(log, "work_order_created"))


def test_the_oracle_follows_the_schedule_and_leaves_the_technician_s_component_alone(make_env):
    env = make_env(**TIMED)
    seen_schedule = seen_external = False
    for seed in range(30):
        log = run_episode(env, OraclePolicy(env), seed)
        assert passes(env, log), seed
        seen_schedule |= bool(log[0]["schedule"])
        seen_external |= log[0]["external_events"] is not None
        if log[0]["external_events"] is not None:
            replaced = log[0]["external_events"]["component"]
            assert all(e["component"] != replaced or e["t_start"] < log[0]["external_events"]["t"] for e in events(log, "work_order_created"))
            assert any(e["id"] == "wo-ext-1" for e in events(log, "work_order_closed"))
    assert seen_schedule and seen_external


def test_the_oracle_finishes_before_a_drop_it_could_not_repair_and_orders_what_the_technician_would_fail(make_env):
    late = make_env(schedule={"mode": "seeded", "components": 1, "first_drop": [56, 58], "interval": 10})
    for seed in range(10):
        log = run_episode(late, OraclePolicy(late), seed)
        assert passes(late, log), seed
        drop = log[0]["schedule"][0]
        assert log[-1]["time"] <= drop["t"] and not events(log, "scheduled_drop")  # finished before the drop the horizon could not fit
        assert drop["component"] not in {o["component"] for o in late.oracle()["work_orders"]}
    patience = EpisodeConfig().failure["patience"]
    worst_start = [list(k) for k in make_env().candidates if k[0] == 3]  # the cooler at its worst level
    slow = make_env(allowed=worst_start, external_events={"mode": "seeded", "probability": 1.0, "window": [patience + 5, patience + 5]})
    for seed in range(10):
        log = run_episode(slow, OraclePolicy(slow), seed)
        assert passes(slow, log), seed
        replaced = log[0]["external_events"]["component"]
        if replaced == "cooler":
            assert any(e["component"] == "cooler" for e in events(log, "work_order_created")) and not events(log, "failure")


def test_do_nothing_advances_to_the_horizon_in_one_step(make_env):
    env = make_env()
    log = run_episode(env, DoNothingPolicy(), 0)
    assert log[-1]["reason"] == "horizon" and log[-1]["time"] == env.config.horizon and env.steps == 1
    assert [e["event"] for e in log if e["step"] is not None] == ["advance"] and log[-1]["final"] == list(env.initial)
    assert log[-1]["downtime_total"] > 0  # a faulty rig accrues downtime while nothing is done


def test_random_is_seeded_apart_from_the_environment_and_stays_within_the_rules(make_env):
    assert RandomPolicy(5).rng.getstate() != random.Random(5).getstate()
    env = make_env()
    for seed in range(10):
        first = run_episode(env, RandomPolicy(seed), seed)
        assert json.dumps(first) == json.dumps(run_episode(env, RandomPolicy(seed), seed))
        assert first[-1]["reason"] in ("finish", "budget", "horizon", "max_steps")
        for refusal in events(first, "refused"):
            assert refusal["reason"] != "unknown action type"


@pytest.fixture(scope="module")
def served_classifiers(world):
    cycles, blocks = world
    return ServedClassifiers(TierClassifiers(cycles, blocks), cycles)


def test_greedy_episodes_are_deterministic_legal_and_keep_budget_for_the_orders(make_env, served_classifiers):
    env = make_env()
    policy = GreedyPolicy(served_classifiers, use_max_faulty=True)
    first = [json.dumps(run_episode(env, policy, seed)) for seed in range(8)]
    assert first == [json.dumps(run_episode(env, policy, seed)) for seed in range(8)]
    for text in first:
        log = json.loads(text)
        assert not events(log, "refused") and log[-1]["reason"] == "finish" and log[-1]["cost"] <= env.config.budget
        diagnosis = events(log, "submit_diagnosis")[0]
        assert diagnosis["evidence_ids"] and all(i.startswith("obs/") for i in diagnosis["evidence_ids"])
        assert events(log, "read") and all(e["step"] < diagnosis["step"] for e in events(log, "read"))
        diagnosed_faulty = {c for c, level in diagnosis["diagnosis"].items() if level != HEALTHY[c]}
        assert {e["component"] for e in events(log, "work_order_created")} <= diagnosed_faulty


def test_greedy_prior_takes_the_fault_rule_from_the_profile(make_env, served_classifiers):
    from envs.hydraulic_rig.classifiers import COMBOS
    healthy = tuple(HEALTHY[c] for c in COMPONENTS)
    for low, high in ((1, 1), (2, 2), (1, 2)):
        env = make_env(fault_rule={"min_faulty": low, "max_faulty": high})
        for use_rule, support in ((False, set(COMBOS) - {healthy}), (True, set(env.candidates))):
            policy = GreedyPolicy(served_classifiers, use_max_faulty=use_rule)
            policy.start(env.reset(0))
            assert {h for h, b in zip(COMBOS, policy.belief) if b > 0} == support


def test_a_harness_may_hand_the_reset_envelope_to_act_without_calling_start(make_env, served_classifiers):
    env = make_env()
    for policy in (OraclePolicy(env), DoNothingPolicy(), RandomPolicy(0), GreedyPolicy(served_classifiers)):
        obs = env.reset(1)
        while not env.done:
            obs, *_ = env.step(policy.act(obs))
        assert env.log[-1]["reason"] in ("finish", "horizon", "budget", "max_steps")
    env.reset(1)
    with pytest.raises(ValueError, match="asset profile"):
        DoNothingPolicy().act(env.step({"type": "advance_time", "cycles": 1})[0])
