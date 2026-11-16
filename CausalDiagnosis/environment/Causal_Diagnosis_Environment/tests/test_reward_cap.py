"""The cost term of the reward is capped: the reward charges the cost ratio up to COST_CAP and the delay
up to the horizon, so every penalty lies between 0 and w_c * COST_CAP + w_t, and while that sum stays below the smallest gap
between two outcome levels, a pass earns more than a correct declaration and a correct declaration more than a wrong answer,
whatever either episode cost or how long it took. The reward refuses weights that break the bound."""
import random

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import ABSTENTION_CREDIT, COST_CAP, outcome_gap, reward

HORIZON = 120
COMPONENTS = ("cooler", "valve", "pump", "accumulator")


def verdict(outcome, cost, delay):
    """A verified verdict with the given outcome ('pass', 'declaration', 'wrong' or 'made_up'), normalized cost and largest delay."""
    delays = dict.fromkeys(COMPONENTS)
    delays["pump"] = delay
    return {"verified": True, "cost_normalized": cost, "pass": outcome == "pass", "abstention_correct": outcome == "declaration",
            "checks": {"provenance": {"ok": outcome != "made_up", "unmatched": []}},
            "detection_delay": dict(delays), "repair_delay": dict(delays), "unrestored": []}


def draws(rng, n):
    """Costs from zero to far above any cap and delays from zero to the horizon, with the corners always present."""
    corners = [(0.0, 0), (0.0, HORIZON), (1e9, 0), (1e9, HORIZON), (COST_CAP, HORIZON)]
    return corners + [(rng.choice([rng.uniform(0, 3), rng.uniform(0, 1e4)]), rng.randint(0, HORIZON)) for _ in range(n)]


@pytest.mark.parametrize("credit", [ABSTENTION_CREDIT, 0.0])
def test_a_higher_outcome_level_earns_more_whatever_either_episode_cost(credit):
    rng = random.Random(0)
    levels = ["pass", "declaration", "wrong"] if credit > 0 else ["pass", "wrong"]
    for (c_hi, d_hi) in draws(rng, 300):
        for (c_lo, d_lo) in draws(rng, 20):
            rewards = [reward(verdict(o, c, d), HORIZON, declaration_credit=credit)
                       for o, (c, d) in zip(levels, [(c_hi, d_hi)] + [(c_lo, d_lo)] * (len(levels) - 1))]
            assert all(a > b for a, b in zip(rewards, rewards[1:])), (rewards, c_hi, d_hi, c_lo, d_lo)
            worst = reward(verdict(levels[-2], c_hi, d_hi), HORIZON, declaration_credit=credit)
            assert worst > reward(verdict(levels[-1], c_lo, d_lo), HORIZON, declaration_credit=credit)


def test_the_cost_ratio_is_charged_up_to_the_cap_and_no_further():
    at_cap = reward(verdict("pass", COST_CAP, 3), HORIZON)
    assert reward(verdict("pass", 10 * COST_CAP, 3), HORIZON) == at_cap == reward(verdict("pass", 1e9, 3), HORIZON)
    assert reward(verdict("pass", 1.0, 3), HORIZON) > at_cap  # below the cap the reward still falls with the cost


def test_the_default_weights_keep_the_bound_and_weights_that_break_it_are_refused():
    assert 0.2 * COST_CAP + 0.1 < outcome_gap(ABSTENTION_CREDIT)
    assert outcome_gap(ABSTENTION_CREDIT) == min(ABSTENTION_CREDIT, 1.0 - ABSTENTION_CREDIT) and outcome_gap(0.0) == 1.0
    with pytest.raises(ValueError):
        reward(verdict("pass", 1.0, 0), HORIZON, w_c=0.3, w_t=0.1)  # 0.3 * 1.5 + 0.1 = 0.55 is not below 0.5
    assert reward(verdict("pass", 1.0, 0), HORIZON, w_c=0.3, w_t=0.1, declaration_credit=0.0) == 1.0 - 0.3  # the gap is 1 there


def test_negative_weights_are_refused_even_where_they_pass_the_bound():
    """w_c = 0.4 with w_t = -0.2 passes the bound (0.4 * 1.5 - 0.2 = 0.4 < 0.5) and would put a correct declaration above a
    pass; the reward refuses it, and the registered weights pass."""
    for w_c, w_t in ((0.4, -0.2), (-0.1, 0.1)):
        with pytest.raises(ValueError, match="nonnegative"):
            reward(verdict("pass", 1.0, 0), HORIZON, w_c=w_c, w_t=w_t)
    assert reward(verdict("pass", 1.0, 0), HORIZON, w_c=0.2, w_t=0.1) == 1.0 - 0.2


def test_a_fault_never_restored_takes_the_delay_at_the_horizon():
    """An episode that leaves a fault in place takes the full delay term whatever delays the verdict measured, so doing
    nothing is never cheaper in delay than a slow repair."""
    left = {**verdict("wrong", 1.0, None), "unrestored": ["pump"]}
    assert reward(left, HORIZON) == 0.0 - 0.2 * 1.0 - 0.1
    slow = verdict("wrong", 1.0, HORIZON - 1)
    assert reward(slow, HORIZON) > reward(left, HORIZON)
    declared = {**verdict("declaration", 1.0, None), "unrestored": ["pump"]}
    assert reward(declared, HORIZON) == ABSTENTION_CREDIT - 0.2 - 0.1 > reward(left, HORIZON)


def test_a_made_up_citation_still_scores_no_more_than_an_honest_failure_under_the_cap():
    for cost, delay in [(0.5, 2), (COST_CAP, HORIZON), (1e6, 7)]:
        assert reward(verdict("made_up", cost, delay), HORIZON) == reward(verdict("wrong", cost, delay), HORIZON) < 0.0
