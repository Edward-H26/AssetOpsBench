"""E7 to E12: the decision rule that replaces a fixed confidence cut, over the combinations now (version 7,
CausalDiagnosis/docs/causal-diagnosis-mcp.md, section E7 to E12).

The expected loss of acting now on the top combination, which repairs its faulty set and diagnoses it, is compared with
the expected loss of first reading one cycle of a sensor group and then acting on the updated belief, the one step
rule of Heckerman, Breese and Rommelse 1995 written for the rig's prices, durations and downtime rates and for more
than one fault. A diagnosis off the truth in any level costs the price of a wrong diagnosis, as the score
counts it wrong, so a reading that separates the levels of one faulty set can be worth its cost. The predictive of the
reading is the belief it meets (Filter.next_reading), one predict step ahead where a read takes cycles and the belief now
where a read takes none. E10 says read first when some test
lowers the expected loss, and the test that lowers it most is the separating test. The long forms of the ranking report
E10, and the rule of the server, which the compact form and the advice state, reads the record once before a diagnosis
(tools/), with the information gain per unit cost (read_choice.py) reported beside it. E9 integrates over the predictive
mixture of the next reading by Monte Carlo with a declared sample count and seed. Every quantity below is a price, a
duration times a downtime rate, or a posterior probability.

On a row whose rules price nothing the loss follows what the verifier scores: a diagnosis off the truth in any level, a
work order on a healthy component and a fault left unrepaired each void the pass, so an act that differs from the truth
costs the calls left before the step cap, what a failed episode costs, once. The best act is then the most probable
combination. A reading there lowers neither Top-1 nor pass, so the rule of the server weighs no call against that loss
and reads while the row can still serve a reading (served_recordings.py, tools/), while E10 stays the comparison of the long
forms. A declaration is never its advice, since both measures give a declaration nothing, and declare_above states for
each measure of the verifier when one would pay.
"""
from collections import namedtuple

import numpy as np

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import STOPPED, cost_unit  # the environment's rule of the cost unit of a row, kept in one place
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import ABSTENTION_CREDIT  # the public constant of the reward for a correct declaration; no verdict and no label is read

MONTE_CARLO_SAMPLES = 512  # draws of the next reading per candidate test, a design parameter printed in every result
MONTE_CARLO_SEED = 0  # the draws of E9 are seeded so that a replayed call answers as the first one did
# What a correct declaration earns under each measure of the verifier, where a diagnosis of the true combination earns one:
# Top-1 and pass give a declaration nothing, success counts it, and the reward gives it the verifier's credit.
DECLARATION_WORTH = {"top1": 0.0, "pass": 0.0, "success": 1.0, "reward": ABSTENTION_CREDIT}

Costs = namedtuple("Costs", "components healthy unnecessary missed test_price test_duration remaining rates unit misdiagnosis",
                   defaults=("money", 0.0))
Decision = namedtuple("Decision", "read_first separating_test expected_saving expected_loss_act expected_loss_test "
                                  "standard_error test_cost samples seed masses_after draws")


def costs(view, time, steps_left=None):
    """E7 from the public rules of the view at the current cycle, in the unit of the episode's row. In money an
    unnecessary replacement costs the part, the repair charge and the stopped rate over the repair downtime; a missed
    fault costs its downtime rate over the remaining horizon; a test costs its read price for one cycle plus one cycle of
    the current downtime, added when the belief is known, where the read plays its cycle. On a row that reads a record a
    read of a recorded cycle takes no time, so the duration of every test is zero there and a test costs its price alone.
    A wrong diagnosis costs prices.misdiagnosis, zero
    when the rules do not carry it. On a row whose rules price nothing every cost counts calls: a test costs one call
    and no downtime is charged. A missed fault, an unnecessary order and a wrong diagnosis each cost the calls left before
    the step cap (steps_left, the step cap when it is not known), as the remaining horizon does in money, since each voids
    the pass and a failed episode has spent its calls for nothing."""
    rules = view.rules
    components = tuple(view.components)
    read_duration = 0 if view.reads_record else rules["durations"]["read_per_cycle"]
    if cost_unit(rules["prices"]) == "calls":
        left = max(int(rules["max_steps"] if steps_left is None else steps_left), 0)
        free = {c: {int(level): 0 for level in view.levels[c]} for c in components}
        missed = {c: {int(level): 0 if int(level) == view.healthy[c] else left for level in view.levels[c]} for c in components}
        groups = sorted(view.sensor_groups)
        return Costs(components, dict(view.healthy), {c: float(left) for c in components}, missed, {g: 1.0 for g in groups},
                     {g: read_duration for g in groups}, left, free, "calls", float(left))
    prices, durations, rates = rules["prices"], rules["durations"], rules["downtime_rate"]
    remaining = max(rules["horizon"] - time, 0)
    unnecessary = {c: prices["parts"][c] + prices["repair_charge"] + rates[STOPPED] * durations["repair_downtime"][c] for c in components}
    level_rates = {c: {int(level): rate for level, rate in rates[c].items()} for c in components}
    missed = {c: {level: rate * remaining for level, rate in level_rates[c].items()} for c in components}
    groups = sorted(view.sensor_groups)
    return Costs(components, dict(view.healthy), unnecessary, missed, {g: prices["read"][g] for g in groups},
                 {g: read_duration for g in groups}, remaining, level_rates,
                 misdiagnosis=float(prices.get("misdiagnosis", 0)))


def downtime_rate(combo, cost):
    return sum(cost.rates[c][combo[j]] for j, c in enumerate(cost.components))


def expected_downtime_rate(combos, probabilities, cost):
    return float(sum(p * downtime_rate(h, cost) for h, p in zip(combos, probabilities)))


def test_cost(group, cost, rate):
    """C_t(a) = pi(a) + d(a) times the downtime rate of the current state, for one cycle of the group."""
    return cost.test_price[group] + cost.test_duration[group] * rate


def _component_losses(combos, cost, components):
    """Per component, the unnecessary and the missed part of the repair loss of acting on a when h is true."""
    combos = np.asarray(combos)
    for c in components or cost.components:
        j = cost.components.index(c)
        faulty = combos[:, j] != cost.healthy[c]
        a_faulty, h_faulty = faulty[:, None], faulty[None, :]
        missed = np.array([cost.missed[c][level] for level in combos[:, j]])[None, :]
        yield (a_faulty & ~h_faulty) * cost.unnecessary[c], (~a_faulty & h_faulty) * missed


def repair_losses(combos, cost, components=None):
    """The two parts of the repair loss of acting on the faulty set of combination a when h is true, summed over the given
    components: the unnecessary part, a replacement where a is faulty and h healthy, and the missed part, a fault where a is
    healthy and h faulty. They are blind to levels, which diagnosis_loss prices."""
    n = len(combos)
    unnecessary, missed = np.zeros((n, n)), np.zeros((n, n))
    for u, m in _component_losses(combos, cost, components):
        unnecessary += u
        missed += m
    return unnecessary, missed


def loss_matrix(combos, cost, components=None):
    """L[a, h], the repair part of E8's loss: the sum of the two parts of repair_losses, added component by component."""
    n = len(combos)
    loss = np.zeros((n, n))
    for u, m in _component_losses(combos, cost, components):
        loss += u
        loss += m
    return loss


def loss_beyond_outlay(combos, cost):
    """What acting on combination a can cost beyond the outlay of its repair plan when h is true. The outlay of the plan
    holds every replacement of a, the unnecessary ones included, so in money it is the missed part of the repair loss plus
    the price of a wrong diagnosis. On a row whose rules price nothing it is decision_loss, the cost of a failed episode,
    counted once."""
    if cost.unit == "calls":
        return decision_loss(combos, cost)
    return repair_losses(combos, cost)[1] + diagnosis_loss(combos, cost)


def diagnosis_loss(combos, cost):
    """C_d 1[a != h]: the price of a wrong diagnosis wherever the diagnosed combination a differs from the
    true h in any level, as the score of a diagnosis counts it: the price of the rules on a row with prices, zero where
    they do not carry it, and the calls left on a row whose rules price nothing."""
    combos = np.asarray(combos)
    return cost.misdiagnosis * (combos[:, None, :] != combos[None, :, :]).any(axis=-1)


def decision_loss(combos, cost):
    """The loss of E8 to E10 and of the troubleshooting plans: acting on combination a repairs its faulty set and diagnoses a.
    On a row with prices it is the repair loss plus the price of a wrong diagnosis. On a row whose rules price nothing the
    act fails as a whole or not at all, since one level off, one unnecessary order or one fault left voids the pass, so the
    loss is the cost of a failed episode wherever a differs from h, counted once whatever the number of errors."""
    if cost.unit == "calls":
        return diagnosis_loss(combos, cost)
    return loss_matrix(combos, cost) + diagnosis_loss(combos, cost)


def expected_loss_act(probabilities, loss, top):
    """E8: the expected loss of acting on the top combination, its row of the loss times the belief."""
    return float((probabilities * loss[top]).sum())


def expected_loss_test(updated, loss):
    """E9 without the test cost: for each hypothetical reading, the expected loss of acting on its new top combination,
    averaged over the draws; also the standard error of the average."""
    tops = updated.argmax(axis=1)
    losses = (updated * loss[tops]).sum(axis=1)
    return float(losses.mean()), float(losses.std(ddof=1) / np.sqrt(len(losses))) if len(losses) > 1 else 0.0


def open_to(serving, group):
    """Whether the row can still serve a reading of the sensor group. serving is the count of serving.left_to_serve, and a
    caller that hands none does not say what the row serves, so every group counts as open."""
    return serving is None or serving[group].cycles > 0


def decide(post, cost, samples=MONTE_CARLO_SAMPLES, seed=MONTE_CARLO_SEED, ahead=None, margin=0.0, serving=None):
    """E10: read first when the best test lowers the expected loss below acting now; the separating test is that
    test and the expected saving the difference. post is the belief now, ahead the belief when a reading of one cycle
    arrives (post itself when no hidden process runs). The draws and updated masses of every test are kept for E12, E20
    and E21. margin, zero by default, asks the best test to lower the expected loss by more than margin times its cost, the
    e10_margin of a decision threshold card. serving is what the row can still serve to a reading of each sensor group
    (serving.left_in_episode), which the server hands on every row: a group the row can serve nothing more
    draws nothing and keeps the belief, so its expected loss after the test is its cost plus the loss of acting and its
    gain is zero. Without it every group draws as before."""
    ahead = post if ahead is None else ahead
    loss = decision_loss(post.candidates, cost)
    top = post.order()[0]
    act = expected_loss_act(post.probabilities, loss, top)
    rate = expected_downtime_rate(post.candidates, post.probabilities, cost)
    rng = np.random.default_rng(seed)
    masses_after, draws, observe, errors, prices = {}, {}, {}, {}, {}
    for group in sorted(cost.test_price):
        fields = post.space.groups[group]
        if open_to(serving, group):
            x = ahead.sample_next(fields, samples, rng)
            updated = ahead.update_with(fields, x)
        else:  # a reading the row cannot serve leaves the belief where it is
            x, updated = np.empty((0, len(fields))), ahead.probabilities[None, :]
        mean, error = expected_loss_test(updated, loss)
        prices[group] = test_cost(group, cost, rate)
        observe[group] = prices[group] + mean
        errors[group] = error
        masses_after[group] = updated
        draws[group] = x
    best = min(observe, key=lambda g: (observe[g], g))
    return Decision(bool(observe[best] + margin * prices[best] < act), best, act - observe[best], act, observe, errors, prices, samples, seed,
                    masses_after, draws)


def unseparated(post, cost, decision):
    """E12: a component is unseparated when the decision to replace it differs between the top and the second combination
    and some test lowers the expected loss of that decision alone. The loss is the repair part alone, so a component whose
    level alone differs between the two is not listed. Returns {component: fields of the argmin test}."""
    order = post.order()
    if len(order) < 2 or post.probabilities[order[1]] <= 0:
        return {}
    top, second = post.candidates[order[0]], post.candidates[order[1]]
    out = {}
    for j, c in enumerate(cost.components):
        if (top[j] != cost.healthy[c]) == (second[j] != cost.healthy[c]):
            continue
        loss = loss_matrix(post.candidates, cost, [c])
        act = expected_loss_act(post.probabilities, loss, order[0])
        observe = {g: decision.test_cost[g] + expected_loss_test(updated, loss)[0] for g, updated in decision.masses_after.items()}
        best = min(observe, key=lambda g: (observe[g], g))
        if observe[best] < act:
            out[c] = list(post.space.groups[best])
    return out


def declare_above(mass):
    """{measure: threshold} for a first candidate of the given mass: the chance that the start is one that both separability
    labels mark inseparable above which a declaration earns more in expectation than a diagnosis of that candidate, None
    where no chance does. A diagnosis earns its mass under every measure of DECLARATION_WORTH and a declaration earns its
    worth on such a start alone, so the threshold is the mass over the worth wherever that lies below one, and a worth of
    zero gives None without a division. The worth under the reward is the verifier's public constant ABSTENTION_CREDIT;
    the server reads no verdict and no label."""
    return {measure: mass / worth if mass < worth else None for measure, worth in DECLARATION_WORTH.items()}


def one_component_threshold(c_test, c_unnecessary, c_misdiagnosis=0.0):
    """E11: with one uncertain component, one resolving test, a replacement that is unnecessary with probability 1 - p
    and a wrong diagnosis priced C_d, acting now is cheaper than testing when
    p >= 1 - C_t / (C_u + C_d). It shows what a fixed confidence cut stands for."""
    return 1 - c_test / (c_unnecessary + c_misdiagnosis)
