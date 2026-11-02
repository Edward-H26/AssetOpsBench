"""E7 to E12 of design v2: the decision rule that replaces the fixed confidence cut (docs/hydraulic_rig/causal-diagnosis-server.md).

The expected loss of acting now on the top candidate's faulty set is compared with the expected loss of first reading
one cycle of a sensor group and then acting on the updated posterior, the one step lookahead of Heckerman, Breese and
Rommelse 1995 written for the rig's prices, durations and downtime rates and for more than one fault. The evidence is
insufficient when some test lowers the expected loss, and the test that lowers it most is the separating test. The
lookahead integrates over the predictive mixture of the next reading by Monte Carlo with a declared sample count and
seed. Every quantity below is a price, a duration times a downtime rate, or a posterior probability.
"""
from collections import namedtuple

import numpy as np

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY

from .effect_model import GROUP_FIELDS

MONTE_CARLO_SAMPLES = 512  # draws of the next reading per candidate test, a design parameter printed in every result
MONTE_CARLO_SEED = 0  # the lookahead is seeded so that a replayed call answers as the first one did

Costs = namedtuple("Costs", "unnecessary missed test_price test_duration remaining rates")
Decision = namedtuple("Decision", "insufficient_evidence separating_test expected_saving expected_loss_act expected_loss_test "
                                  "standard_error test_cost samples seed lookahead")


def costs(rules, time):
    """E7 from the public rules at the current cycle. An unnecessary replacement costs the part, the repair charge and
    the stopped rate over the repair downtime; a missed fault costs its downtime rate over the remaining horizon; a test
    costs its read price for one cycle plus one cycle of the current downtime, added when the posterior is known."""
    prices, durations, rates = rules["prices"], rules["durations"], rules["downtime_rate"]
    remaining = max(rules["horizon"] - time, 0)
    unnecessary = {c: prices["parts"][c] + prices["repair_charge"] + rates["stopped"] * durations["repair_downtime"][c] for c in COMPONENTS}
    level_rates = {c: {int(level): rate for level, rate in rates[c].items()} for c in COMPONENTS}
    missed = {c: {level: rate * remaining for level, rate in level_rates[c].items()} for c in COMPONENTS}
    groups = sorted(rules["sensor_groups"])
    return Costs(unnecessary, missed, {g: prices["read"][g] for g in groups}, {g: durations["read_per_cycle"] for g in groups},
                 remaining, level_rates)


def downtime_rate(combo, cost):
    return sum(cost.rates[c][combo[j]] for j, c in enumerate(COMPONENTS))


def expected_downtime_rate(current, probabilities, cost):
    return float(sum(p * downtime_rate(h, cost) for h, p in zip(current, probabilities)))


def test_cost(group, cost, rate):
    """C_t(a) = pi(a) + d(a) times the downtime rate of the current state, for one cycle of the group."""
    return cost.test_price[group] + cost.test_duration[group] * rate


def loss_matrix(current, cost, components=COMPONENTS):
    """L[a, h]: the loss of acting on the faulty set of combination a when h is true, summed over the given components:
    an unnecessary replacement where a is faulty and h healthy, a missed fault where a is healthy and h faulty."""
    current = np.asarray(current)
    n = len(current)
    loss = np.zeros((n, n))
    for c in components:
        j = COMPONENTS.index(c)
        a_faulty = (current[:, j] != HEALTHY[c])[:, None]
        h_faulty = (current[:, j] != HEALTHY[c])[None, :]
        missed = np.array([cost.missed[c][level] for level in current[:, j]])[None, :]
        loss += (a_faulty & ~h_faulty) * cost.unnecessary[c]
        loss += (~a_faulty & h_faulty) * missed
    return loss


def expected_loss_act(probabilities, loss, top):
    """E8: the posterior expected loss of acting on the faulty set of the top candidate."""
    return float((probabilities * loss[top]).sum())


def expected_loss_test(updated, loss):
    """E9 without the test cost: for each hypothetical reading, the expected loss of acting on its new top candidate,
    averaged over the draws; also the standard error of the average."""
    tops = updated.argmax(axis=1)
    losses = (updated * loss[tops]).sum(axis=1)
    return float(losses.mean()), float(losses.std(ddof=1) / np.sqrt(len(losses))) if len(losses) > 1 else 0.0


def decide(post, cost, samples=MONTE_CARLO_SAMPLES, seed=MONTE_CARLO_SEED):
    """E10: insufficient evidence when the cheapest test lowers the expected loss below acting now; the separating test
    is that test and the expected saving the difference. The candidate tests are one cycle of each sensor group."""
    current = post.current
    loss = loss_matrix(current, cost)
    top = post.order()[0]
    act = expected_loss_act(post.probabilities, loss, top)
    rate = expected_downtime_rate(current, post.probabilities, cost)
    rng = np.random.default_rng(seed)
    lookahead, observe, errors, prices = {}, {}, {}, {}
    for group in sorted(cost.test_price):
        fields = GROUP_FIELDS[group]
        updated = post.update_with(fields, post.sample_next(fields, samples, rng))
        mean, error = expected_loss_test(updated, loss)
        prices[group] = test_cost(group, cost, rate)
        observe[group] = prices[group] + mean
        errors[group] = error
        lookahead[group] = updated
    best = min(observe, key=lambda g: (observe[g], g))
    return Decision(bool(observe[best] < act), best, act - observe[best], act, observe, errors, prices, samples, seed, lookahead)


def unseparated(post, cost, decision):
    """E12: a component is unseparated when the decision to replace it differs between the top and the second candidate
    and some test lowers the expected loss of that decision alone. Returns {component: fields of the argmin test}."""
    order = post.order()
    if len(order) < 2:
        return {}
    current = post.current
    top, second = current[order[0]], current[order[1]]
    out = {}
    for j, c in enumerate(COMPONENTS):
        if (top[j] != HEALTHY[c]) == (second[j] != HEALTHY[c]):
            continue
        loss = loss_matrix(current, cost, [c])
        act = expected_loss_act(post.probabilities, loss, order[0])
        observe = {g: decision.test_cost[g] + expected_loss_test(updated, loss)[0] for g, updated in decision.lookahead.items()}
        best = min(observe, key=lambda g: (observe[g], g))
        if observe[best] < act:
            out[c] = list(GROUP_FIELDS[best])
    return out


def one_component_threshold(c_test, c_unnecessary):
    """E11: with one uncertain component, one resolving test and a replacement that is unnecessary with probability 1 - p,
    acting now is cheaper than testing when p >= 1 - C_t / C_u. It shows what a fixed confidence cut stands for."""
    return 1 - c_test / c_unnecessary
