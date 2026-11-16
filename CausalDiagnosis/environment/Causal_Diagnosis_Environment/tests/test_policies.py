"""The reference policies (CausalDiagnosis/environment/reference_policies/policies.py and greedy_policy.py) on the v2 action set: the oracle recovers
every start and matches the oracle set, do nothing ends at the horizon, random is seeded and legal, greedy is
deterministic and keeps budget for its orders. Data-dependent assertions are relations, never values."""
import json
import random

import pytest

from CausalDiagnosis.environment.reference_policies.tier_classifiers import TierClassifiers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.environment.reference_policies.greedy_policy import GreedyPolicy, ServedClassifiers
from CausalDiagnosis.environment.reference_policies.policies import DeclarePolicy, DoNothingPolicy, OraclePolicy, RandomPolicy, run_episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import events, faulty_components

TIMED = {"schedule": {"mode": "seeded", "components": 1, "first_drop": [5, 10], "interval": 10}}
HEALTHY_LIST = [HEALTHY[c] for c in COMPONENTS]


@pytest.mark.parametrize("read, cheapest", [({"tier1": 5, "tier2": 1, "tier3": 10}, "tier2"), ({"tier1": 0, "tier2": 0, "tier3": 0}, "tier1")],
                         ids=["priced", "free, the first name"])
def test_the_declare_policy_reads_the_cheapest_group_once_cites_that_reading_declares_and_finishes(make_env, read, cheapest):
    env = make_env(prices={**EpisodeConfig().prices, "read": read}, budget=None)
    log = run_episode(env, DeclarePolicy(), 0)
    assert [e["event"] for e in log if e.get("step") is not None] == ["read", "declare_insufficient_evidence", "finish"]
    reading = events(log, "read")[0]
    assert reading["sensor_group"] == cheapest and reading["cycles"] == 1 and len(reading["served"]) == 1
    declaration = events(log, "declare_insufficient_evidence")[0]
    assert declaration["separating_test"] == {"sensor_group": cheapest, "cycles": 1}
    assert declaration["evidence_ids"] == [f"obs/{reading['served'][0]['n']}/{field}" for field in reading["served"][0]["values"]] != []
    assert log[-1]["reason"] == "finish" and not events(log, "work_order_created")
    verdict = verify(log, released_from_env(env), env.oracle(), 10.0, grade={"separable": False, "e18": {"separable": False}})
    assert verdict["abstention_cites_reading"] and verdict["abstention_correct"] and verdict["success"] and not verdict["pass"]


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
        assert len({e["visit"] for e in events(log, "work_order_created")}) == 1  # every order in one cycle, one visit
        assert log[-1]["cost"] == sum(single.config.order_price(c) for c in orders) + single.config.prices["visit"] + log[-1]["downtime_total"]
        assert events(log, "submit_diagnosis")[0]["diagnosis"] == dict(zip(COMPONENTS, start))
        assert events(log, "submit_diagnosis")[0]["step"] < min(e["step"] for e in events(log, "work_order_created"))


def test_every_static_verdict_scores_the_start_since_nothing_changes_the_rig_before_the_write(make_env, world):
    cycles, blocks = world
    env = make_env()
    greedy = ServedClassifiers(TierClassifiers(cycles, blocks), cycles)
    for make in (lambda: OraclePolicy(env), DoNothingPolicy, lambda: RandomPolicy(3), lambda: GreedyPolicy(greedy)):
        for seed in range(6):
            log = run_episode(env, make(), seed)
            v = verify(log, released_from_env(env), env.oracle(), None)
            assert v["truth_at_write"] in (None, v["hidden_start"]), seed


def test_the_oracle_follows_the_schedule(make_env):
    env = make_env(**TIMED)
    seen_schedule = False
    for seed in range(30):
        log = run_episode(env, OraclePolicy(env), seed)
        assert passes(env, log), seed
        seen_schedule |= bool(log[0]["schedule"])
    assert seen_schedule


def test_the_oracle_finishes_before_a_drop_it_could_not_repair(make_env):
    late = make_env(schedule={"mode": "seeded", "components": 1, "first_drop": [56, 58], "interval": 10})
    for seed in range(10):
        log = run_episode(late, OraclePolicy(late), seed)
        assert passes(late, log), seed
        drop = log[0]["schedule"][0]
        assert log[-1]["time"] <= drop["t"] and not events(log, "scheduled_drop")  # finished before the drop the horizon could not fit
        assert drop["component"] not in {o["component"] for o in late.oracle()["work_orders"]}


def test_the_oracle_finishes_before_a_scheduled_drop_that_would_come_after_its_orders_close(make_env):
    """A drop at or after the cycle at which the oracle's orders close never happens, since the oracle finishes first, so it
    neither waits for the drop nor orders the component the drop would degrade, which the oracle set leaves out.
    The drops fall after the orders of the start close and early enough that their repair would fit before the horizon."""
    later = make_env(schedule={"mode": "seeded", "components": 1, "first_drop": [20, 25], "interval": 10})
    for seed in range(10):
        log = run_episode(later, OraclePolicy(later), seed)
        assert passes(later, log), seed
        drop = log[0]["schedule"][0]
        assert log[-1]["time"] <= drop["t"] and not events(log, "scheduled_drop"), seed
        assert drop["component"] not in {e["component"] for e in events(log, "work_order_created")}, seed


def test_do_nothing_only_waits_up_to_the_horizon_and_waits_again_after_each_notification(make_env):
    env = make_env()
    log = run_episode(env, DoNothingPolicy(), 0)
    waits = [e for e in log if e["step"] is not None]
    assert log[-1]["reason"] == "horizon" and log[-1]["time"] == env.config.horizon and env.steps == len(waits)
    assert {e["event"] for e in waits} == {"wait"} and log[-1]["final"] == list(env.initial)
    # every wait but the last ended at a notification of its last cycle, here the failure of a component at its worst level
    assert all(e["ended_by"] == "notification" and any(f["event"] == "failure" and f["t_end"] == e["t_end"] - 1 for f in log)
               for e in waits[:-1]) and waits[-1]["ended_by"] == "timeout"
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
    policy = GreedyPolicy(served_classifiers)
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


def test_greedy_starts_uniform_over_every_combination_whatever_the_fault_rule_of_the_starts(make_env, served_classifiers):
    from CausalDiagnosis.environment.reference_policies.tier_classifiers import COMBOS
    for low, high in ((1, 1), (2, 2), (1, 2)):
        env = make_env(fault_rule={"min_faulty": low, "max_faulty": high})
        policy = GreedyPolicy(served_classifiers)
        policy.start(env.reset(0))
        assert policy.belief == [1 / len(COMBOS)] * len(COMBOS)  # the healthy combination and three or four faults included


def test_greedy_keeps_the_repair_plan_of_its_most_probable_combination_and_one_visit_charge(make_env, served_classifiers):
    from CausalDiagnosis.environment.reference_policies.tier_classifiers import COMBOS
    env = make_env()
    policy = GreedyPolicy(served_classifiers)
    policy.start(env.reset(0))
    prices = env.config.prices
    for top in ((3, 100, 0, 130), (3, 73, 2, 90), tuple(HEALTHY[c] for c in COMPONENTS)):
        policy.belief = [1.0 if h == top else 0.0 for h in COMBOS]
        plan = [prices["parts"][c] + prices["repair_charge"] for c, level in zip(COMPONENTS, top) if level != HEALTHY[c]]
        assert policy._order_reserve() == (sum(plan) + prices["visit"] if plan else 0)


def test_a_harness_may_hand_the_reset_envelope_to_act_without_calling_start(make_env, served_classifiers):
    env = make_env()
    for policy in (OraclePolicy(env), DoNothingPolicy(), RandomPolicy(0), GreedyPolicy(served_classifiers)):
        obs = env.reset(1)
        while not env.done:
            obs, *_ = env.step(policy.act(obs))
        assert env.log[-1]["reason"] in ("finish", "horizon", "budget", "max_steps")
    env.reset(1)
    with pytest.raises(ValueError, match="asset profile"):
        DoNothingPolicy().act(env.step({"type": "wait_for_notification", "timeout": 1})[0])
