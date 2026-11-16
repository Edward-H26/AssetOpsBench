"""E21 and E22: the score of a read and the batch of reads whose choice no outcome can change (version 7,
CausalDiagnosis/docs/causal-diagnosis-mcp.md, section Information gain and the read batch).

A read a is chosen by its information gain per unit cost, IG(a) / C_t(a), with IG(a) = H(b_a) - E_y[H(b_{a,y})], the
expected drop of log N(b), b_a the belief when the reading of one cycle arrives and b_{a,y} that belief updated with the
reading y drawn from its predictive, from the same seeded draws as E9. The gain is never negative in expectation, which is
why it chooses the read: the expected elimination EE(a) = N(b_a) - E_y[N(b_{a,y})], with N(b) = exp(H(b))
the effective number of hypotheses, is not monotone and turns negative for the read that tells a concentrated belief's top
combination from the rest, so it stays a reported number beside the gain. When
the belief is uniform over n combinations and an outcome falls exactly on groups of m_o of them, EE equals n minus the sum of
m_o squared over n, the expected number of combinations the outcome rules out, EE per cost is the rule of generalized binary
search with costs, and the gain per cost picks the same read when every read has two outcomes and one cost. C_t(a) is the
cost in the unit of the row, the test cost of E7 from the priced row on and one call on the free base row. E10 stays the stop rule
and the gain per cost, an informed heuristic without an approximation guarantee in this setting, chooses the read. The batch puts first the affordable read of largest gain per cost above
IG_EPSILON, and a further read joins when it has the largest gain per cost, above IG_EPSILON, under every simulated outcome of
the reads before it, so that waiting for those outcomes could not change the choice, while the read prices still fit the
budget left after the price of the top repair plan and the reads leave the cycles of that plan before the horizon. The
batch holds the belief fixed between its reads, a stated approximation over the few cycles a batch spans.

ec2 is one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause and Ray, not their
algorithm: the expected weight of the edges between the plan classes of the repair loss of E8
that one reading cuts. It is reported beside the gain for the third arm of the read choice ablation Ag7 and ranks
nothing here.
"""
from collections import namedtuple

import numpy as np

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import loss_matrix
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import effective_number, entropy
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import Left

BATCH_OUTCOMES = 16  # simulated outcomes of the reads already in the batch against which a further read is tested
BATCH_INNER_SAMPLES = 64  # draws per group for the rates under one simulated outcome
ROUNDING = 1e-9  # relative size below which a drop is the rounding of a belief that the reading cannot move
IG_EPSILON = 1e-9  # information gain per unit cost, in nats, that a read must exceed to be chosen, so that a gain left at zero by ROUNDING never heads a batch
EC2_DRAWS = 16  # readings of every sensor group drawn under each combination with positive mass, the noise drawn into the hypotheses of the edge cut
Scores = namedtuple("Scores", "effective_before effective_after elimination standard_error information_gain")


def dropped(before, after):
    """before minus the mean of after over the draws, zero when the difference is rounding."""
    value = before - float(np.mean(after))
    return 0.0 if abs(value) <= ROUNDING * max(abs(before), 1.0) else value


def gained(entropy_before, entropies_after):
    """The information gain from the entropy of the belief a reading meets and the entropies after its draws, in nats, held at
    zero when a Monte Carlo mean falls below it."""
    return max(dropped(entropy_before, entropies_after), 0.0)


def eliminations(ahead, decision):
    """{group: Scores} from the decision's masses after each drawn reading: the effective number before and its mean after,
    the expected elimination with its Monte Carlo standard error, and the information gain in nats."""
    before, h_before = float(effective_number(ahead.probabilities)), float(entropy(ahead.probabilities))
    out = {}
    for group, updated in decision.masses_after.items():
        after = effective_number(updated)
        error = float(after.std(ddof=1) / np.sqrt(len(after))) if len(after) > 1 else 0.0
        out[group] = Scores(before, float(after.mean()), dropped(before, after), error, gained(h_before, entropy(updated)))
    return out


def read_units(decision, cost):
    """({group: C_t of E21}, unit): the test cost of E7 in the unit of the row, "money" from the priced row on and "call"
    on the free base row, where every read costs one call."""
    return dict(decision.test_cost), ("call" if cost.unit == "calls" else "money")


def per_cost(values, units):
    """{group: value over C_t}, zero for a read that costs nothing, whose gain then stands in information_gain alone, since a
    rate without a finite value cannot be written in a reply."""
    return {g: (v / units[g] if units[g] > 0 else 0.0) for g, v in values.items()}


def best_group(rate, cost, allowed):
    """The group of largest rate among the allowed ones, ties to the cheaper read, then the name."""
    candidates = [g for g in allowed if g in rate]
    if not candidates:
        return None
    return min(candidates, key=lambda g: (-rate[g], cost.test_price[g], g))


def scenario_rates(post, groups, units, samples, rng):
    """The information gain per cost of every group under one simulated belief, from its own draws."""
    before = float(entropy(post.probabilities))
    out = {}
    for group in groups:
        fields = post.space.groups[group]
        updated = post.update_with(fields, post.sample_next(fields, samples, rng))
        out[group] = gained(before, entropy(updated)) / units[group] if units[group] > 0 else 0.0
    return out


def read_batch(ahead, cost, decision, budget_left, cycles_left, reserve_price, reserve_cycles, seed, *, calls_left, cap=None,
               reserve_calls=0):
    """E22: the ordered reads of one batch, with no count of its own: the budget, the horizon and the step cap end it beside
    the reserve, and cap, the batch_cap of the decision threshold card, bounds it only where a card sets one. calls_left, the
    calls the step cap leaves, has no default, so no caller can drop the step cap from the fit. reserve_price,
    reserve_cycles and reserve_calls are the price, the cycles to a restored rig and the calls (branches.plan_completion: the
    diagnosis, the orders, the waits, the verifying read and finish) of the repair plans the batch must leave affordable before
    the horizon and the step cap, the top branch's and, under the card's reserve_plans 2, the cheapest plan of another combination of the credible set."""
    units, _unit = read_units(decision, cost)
    rate = per_cost({g: s.information_gain for g, s in eliminations(ahead, decision).items()}, units)
    spent, cycles = 0.0, 0

    def fits(group):
        return (spent + cost.test_price[group] <= budget_left - reserve_price
                and cycles + cost.test_duration[group] + reserve_cycles < cycles_left  # strict: the episode ends at t >= T before a finish
                and len(batch_so_far) + 1 + reserve_calls <= calls_left)

    batch_so_far = []

    head = best_group(rate, cost, [g for g in rate if rate[g] > IG_EPSILON and fits(g)])
    if head is None:
        return []
    batch = batch_so_far
    batch.append(head)
    spent += cost.test_price[head]
    cycles += cost.test_duration[head]
    fields = ahead.space.groups[head]
    scenarios = [ahead.updated(fields, x) for x in decision.draws[head][:BATCH_OUTCOMES]]
    rng = np.random.default_rng([seed, len(ahead.candidates), 22])
    while cap is None or len(batch) < cap:
        choices, informative = set(), True
        for scenario in scenarios:
            scenario_rate = scenario_rates(scenario, sorted(cost.test_price), units, BATCH_INNER_SAMPLES, rng)
            choice = best_group(scenario_rate, cost, sorted(cost.test_price))
            choices.add(choice)
            informative &= choice is not None and scenario_rate[choice] > IG_EPSILON
            if len(choices) > 1 or not informative:
                break
        if len(choices) != 1 or not informative:
            break
        group = choices.pop()
        if not fits(group):
            break
        batch.append(group)
        spent += cost.test_price[group]
        cycles += cost.test_duration[group]
        fields = ahead.space.groups[group]
        scenarios = [s.updated(fields, s.sample_next(fields, 1, rng)[0]) for s in scenarios]
    return batch


def plan_losses(candidates, cost):
    """The repair loss of E8, without the price of a wrong diagnosis, of every distinct faulty set of the candidates, taken
    as a repair plan, under every candidate: one row per plan, in the order the plans first occur."""
    loss = loss_matrix(candidates, cost)
    first = {}
    for i, h in enumerate(candidates):
        first.setdefault(tuple(h[j] != cost.healthy[c] for j, c in enumerate(cost.components)), i)
    return loss[list(first.values())]


def readings_per_hypothesis(post, sequence, draws, rng):
    """(readings, combination of every row): `draws` readings of the cycles in `sequence` under every combination with positive
    mass, the cycles sharing the block offset and drawing their own within noise, as the predictive of several cycles does."""
    terms, k = post.predictive(sequence)
    noise = rng.standard_normal((len(terms.index), draws, k))
    readings = terms.mean[:, None, :] + noise @ terms.cholesky.transpose(0, 2, 1)
    return readings.reshape(-1, k), np.repeat(terms.index, draws)


def edge_weight(masses):
    """The weight of the edges between hypotheses of different classes, for each row of class masses: half the squared total
    less the sum of the squares."""
    return (masses.sum(axis=-1) ** 2 - (masses ** 2).sum(axis=-1)) / 2


def ec2_cuts(ahead, cost, seed):
    """{group: one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause and Ray, not
    their algorithm, the expected weight of the edges between the plan classes of the repair loss of E8 that one reading
    of the group cuts}, in squared probability. Every combination h with positive mass b(h) carries EC2_DRAWS seeded
    readings, one cycle of every group, each of weight b(h) / EC2_DRAWS, which draws the noise into the hypotheses. The
    outcome of a group's reading is the repair plan of least expected repair loss after it, a few outcomes per group:
    bins of every summary, or the combination of largest mass, let the reading noise split the draws of one combination
    into many outcomes, so that every read cut nearly every edge on the rig whatever it measured. The readings with the
    same outcome on every group form one hypothesis, whose class is the repair plan of least expected repair loss given
    those outcomes. The cut of a group is the edge weight now less its expectation over the group's outcomes, zero for a
    read whose reading alone never changes the plan, the one step blindness that E10 shares."""
    groups = sorted(cost.test_price)
    sequence = [tuple(ahead.space.groups[g]) for g in groups]
    x, which = readings_per_hypothesis(ahead, sequence, EC2_DRAWS, np.random.default_rng([seed, len(ahead.candidates), 21]))
    weight = ahead.probabilities[which] / EC2_DRAWS
    bounds = np.cumsum([0] + [len(fields) for fields in sequence])
    plans = plan_losses(ahead.candidates, cost)
    outcomes = np.stack([(ahead.update_with(fields, x[:, bounds[i]:bounds[i + 1]]) @ plans.T).argmin(axis=1) for i, fields in enumerate(sequence)],
                        axis=1)
    vectors, vector_of = np.unique(outcomes, axis=0, return_inverse=True)
    vector_of = vector_of.reshape(-1)  # the inverse of a unique over rows is one dimensional on every numpy release
    joint = np.zeros((len(vectors), len(ahead.candidates)))
    np.add.at(joint, (vector_of, which), weight)
    klass = (joint @ plans.T).argmin(axis=1)[vector_of]
    total = float(edge_weight(np.bincount(klass, weights=weight, minlength=len(plans))))
    cuts = {}
    for i, group in enumerate(groups):
        cell = np.zeros((len(plans), len(plans)))
        np.add.at(cell, (outcomes[:, i], klass), weight)
        cuts[group] = max(total - float((cell.sum(axis=1) * edge_weight(cell)).sum()), 0.0)
    return cuts


ACT_CALLS = 2  # the diagnosis and the finish, the calls of an act beside the orders of its repair plan


def one_read(group, cycles, cost, calls_left, cycles_left, reserve_calls, reserve_cycles):
    """E22 on a row whose rules price nothing: the batch is one read, the separating test over every cycle the row can still
    serve it, since there no read waits for the outcome of another. Returns [(sensor group, cycles)], or nothing when the
    call does not fit the calls left after the calls of finishing the reserved repair plan (reserve_calls, plan_completion: the
    diagnosis, the orders and the finish), or when its cycles do not leave the cycles of that plan before the horizon, as
    read_batch asks of every read."""
    fits = (cost.test_price[group] + reserve_calls <= calls_left
            and cycles * cost.test_duration[group] + reserve_cycles < cycles_left)
    return [(group, cycles)] if cycles > 0 and fits else []


def rule_group(serving, losses):
    """The sensor group of the rule on a row whose rules price nothing: among the groups the row can still serve, the one that
    can still show the most summaries not yet shown (serving.left_to_serve), ties to the lower expected loss after its test
    (E9, losses), then to the name. None when the row can serve no group."""
    open_groups = [g for g in serving if serving[g].cycles > 0]
    return min(open_groups, key=lambda g: (-serving[g].summaries, losses[g], g)) if open_groups else None


def record_read(serving, cost, budget_left, reserve, calls_left, cycles_left):
    """E22 on a row whose rules hold a price: one read of the record before a diagnosis. The group is the one that can still
    show the most summaries not yet shown (serving.left_to_serve), ties to the lower price of one cycle, then to the name,
    among the groups the row can still serve whose one cycle the budget left pays beside the reserve, (price, calls, cycles)
    of the repair of every component. Its cycles are those the row can still serve it, at most as many as the money left
    beside the reserve pays, all of them for a group without a price. The call must fit the calls left beside the orders of
    the reserve, the diagnosis and the finish, and the cycles of the read with those of the reserve the cycles left before the
    horizon, else the next group in the same order is tried. Returns [(sensor group, cycles)] or nothing. The rule reads no
    belief, so no outcome of a reading changes it."""
    reserve_price, reserve_calls, reserve_cycles = reserve  # reserve_calls: plan_completion, the diagnosis and the finish included
    money = budget_left - reserve_price
    if 1 + reserve_calls > calls_left:
        return []
    for group in sorted((g for g in cost.test_price if serving[g].cycles > 0), key=lambda g: (-serving[g].summaries, cost.test_price[g], g)):
        price = cost.test_price[group]
        cycles = serving[group].cycles if price <= 0 or money == float("inf") else min(serving[group].cycles, int(money // price))
        if cycles >= 1 and cycles * cost.test_duration[group] + reserve_cycles < cycles_left:
            return [(group, cycles)]
    return []


def reads_in_turn(first, view, serving, spend):
    """E22: the reads that the rule of the row names one after another. first(serving, spent) is the rule, one_read or
    record_read with its other arguments bound, given what the row can still serve and what the reads before it spent,
    and returns [(sensor group, cycles)] or nothing. spend(read) is what one read spends, (money, calls, cycles). After each
    read the summaries of its group count as shown on its cycles: a group all of whose summaries the reads so far hold is
    closed, another group loses the summaries it shares with them, and the rule is asked again, until it names no read, which
    comes after at most one read per sensor group, since a group the rule names shows a field not yet shown and its read
    shows every field of it. The rule reads no belief, so no outcome of a read changes the reads after it, and on nested groups
    the list is the one read of the rule. The marking by summaries is exact when the first read takes every cycle that the
    later groups can still be shown, which holds unless the money ends inside a read."""
    reads, spent, left, shown = [], (0.0, 0, 0), dict(serving), set()
    while True:
        named = first(left, spent)
        if not named:
            break
        read = named[0]
        reads.append(read)
        spent = tuple(a + b for a, b in zip(spent, spend(read)))
        newly = set(view.group_fields(read[0])) - shown
        shown |= newly
        left = {g: Left(0, 0) if set(view.group_fields(g)) <= shown else
                Left(entry.cycles, max(entry.summaries - entry.cycles * len(newly & set(view.group_fields(g))), 0))
                for g, entry in left.items()}
    return reads
