"""E21 and E22 on a synthetic grid of two components whose readings are exact up to a tiny noise: when the belief is
uniform over n combinations and the reading falls on classes of sizes m_o, the expected elimination of a group equals n
minus the sum of squared class sizes over n and its information gain equals ln n minus the sum of m_o over n times ln m_o,
a group whose summaries no hypothesis moves eliminates nothing and gains nothing, both scores divide by a read cost that
exists on every row of the environment table, money from the priced row on and one call on the free base
row, the batch keeps a read that every outcome of the reads before it still ranks first and stops at a read whose rank
depends on their outcome, under a concentrated belief the decisive read heads the batch though its expected elimination is
negative, and one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause
and Ray, not their algorithm, prefers the read that decides the repair plan where the gain prefers the read that
separates levels."""

import math

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import decide
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import (ACT_CALLS, ec2_cuts, eliminations, one_read, per_cost, read_batch, read_units,
                                                  record_read, rule_group)
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior, entropy
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import Left
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.synthetic import TINY, effect_with_means, table_of, tiny_costs

SAMPLES = 2048


def uniform(model):
    return Posterior.uniform(model, block=False)


def both_rates_divide_by(scores, units):
    """The expected elimination and the information gain of every group over the same C_t of E21."""
    return all(abs(per_cost({g: getattr(s, name)}, units)[g] - getattr(s, name) / units[g]) < 1e-12
               for g, s in scores.items() for name in ("elimination", "information_gain"))


def test_expected_elimination_counts_the_combinations_an_exact_reading_rules_out():
    # g1 reads the level of A (3 classes of 4), g2 tells B at level 0 from the other three (classes of 3 and 9), g3 reads
    # nothing, g4 reads both components exactly (12 classes of 1)
    model = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * (h["B"] == 0), 0.0, 10.0 * h["A"] + 100.0 * h["B"]]), noise=0.01)
    post = uniform(model)
    decision = decide(post, tiny_costs(), samples=SAMPLES, seed=0)
    scores = eliminations(post, decision)
    n = len(TINY.combinations)
    expected = {"g1": n - 3 * 4 ** 2 / n, "g2": n - (3 ** 2 + 9 ** 2) / n, "g3": 0.0, "g4": n - n * 1 / n}
    gains = {"g1": math.log(n) - math.log(4), "g2": math.log(n) - (3 * math.log(3) + 9 * math.log(9)) / n, "g3": 0.0, "g4": math.log(n)}
    for group, value in expected.items():
        s = scores[group]
        assert abs(s.effective_before - n) < 1e-9 and abs(s.effective_before - s.effective_after - s.elimination) < 1e-12
        assert abs(s.elimination - value) <= 4 * s.standard_error + 1e-9, (group, s.elimination, value, s.standard_error)
        after = entropy(decision.masses_after[group])
        gain_error = float(after.std(ddof=1) / math.sqrt(len(after)))
        assert abs(s.information_gain - gains[group]) <= 4 * gain_error + 1e-9, (group, s.information_gain, gains[group], gain_error)
    assert abs(scores["g3"].elimination) < 1e-9 and scores["g3"].standard_error < 1e-9 and scores["g3"].information_gain == 0
    units, unit = read_units(decision, tiny_costs())
    assert unit == "money" and units == decision.test_cost and both_rates_divide_by(scores, units)


def test_the_read_cost_exists_on_every_row_of_the_environment_table():
    """Both rates divide by the test cost of E7, the price plus the declared duration times the expected
    downtime rate from the priced row on, with or without a clock, and one call on the free base row."""
    model = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 0.0, 0.0]), noise=0.01)
    post = uniform(model)
    priced = tiny_costs(prices={"g1": 2.0, "g2": 4.0, "g3": 1.0, "g4": 8.0})
    instant = priced._replace(test_duration={g: 0 for g in priced.test_duration})  # a priced row whose reads take no cycle
    healthy = dict(zip(TINY.components, TINY.healthy))
    base = priced._replace(unnecessary={c: 30.0 for c in TINY.components}, test_price={g: 1.0 for g in priced.test_price},
                           missed={c: {lv: 0 if lv == healthy[c] else 30 for lv in levels} for c, levels in zip(TINY.components, TINY.levels)},
                           rates={c: {lv: 0 for lv in levels} for c, levels in zip(TINY.components, TINY.levels)}, remaining=30, unit="calls",
                           misdiagnosis=30.0)
    for cost, unit in ((priced, "money"), (instant, "money"), (base, "call")):
        decision = decide(post, cost, samples=256, seed=0)
        units, name = read_units(decision, cost)
        assert name == unit and units == decision.test_cost
        assert both_rates_divide_by(eliminations(post, decision), units)
    assert decide(post, instant, samples=64, seed=0).test_cost == {g: 2.0 if g == "g1" else 4.0 if g == "g2" else 1.0 if g == "g3" else 8.0 for g in priced.test_price}
    assert all(v > priced.test_price[g] for g, v in decide(post, priced, samples=64, seed=0).test_cost.items())
    assert set(decide(post, base, samples=64, seed=0).test_cost.values()) == {1.0}


def test_a_dominated_read_joins_the_batch_and_a_contested_read_ends_it():
    """In the dominated world g2 reads the four levels of B and g1 the three of A, so g2 comes first and after any outcome
    of it the best read is g1, which joins. In the contested world g1 reads A, g2 reads B only when A is at level 0 and g3
    only otherwise, at a higher price, so g1 comes first and the read after it depends on its outcome, which ends the
    batch with g1. g3 gains more information than g1 at its higher price, which leaves a narrow margin in gain per cost,
    so E9 draws SAMPLES readings."""
    costs = tiny_costs(prices={"g1": 1.0, "g2": 1.0, "g3": 1.5, "g4": 50.0})
    dominated = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 0.0, 0.0]), noise=0.01)
    contested = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"] * (h["A"] == 0), 10.0 * h["B"] * (h["A"] != 0), 0.0]),
                                  noise=0.01)
    batches = {}
    for name, model in (("dominated", dominated), ("contested", contested)):
        post = uniform(model)
        decision = decide(post, costs, samples=SAMPLES, seed=0)
        batches[name] = read_batch(post, costs, decision, budget_left=100, cycles_left=40, reserve_price=0, reserve_cycles=0, seed=0, calls_left=100)
    assert batches["dominated"][:2] == ["g2", "g1"]
    assert batches["contested"] == ["g1"]


def test_the_batch_leaves_the_reserve_and_the_horizon_and_reads_only_what_eliminates():
    costs = tiny_costs(prices={"g1": 5.0, "g2": 5.0, "g3": 5.0, "g4": 5.0})
    model = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 0.0, 0.0]), noise=0.01)
    post = uniform(model)
    decision = decide(post, costs, samples=256, seed=0)
    common = {"seed": 0, "calls_left": 100}
    assert read_batch(post, costs, decision, budget_left=60, cycles_left=40, reserve_price=56, reserve_cycles=0, **common) == []
    assert read_batch(post, costs, decision, budget_left=60, cycles_left=40, reserve_price=51, reserve_cycles=0, **common) == ["g2"]
    assert read_batch(post, costs, decision, budget_left=60, cycles_left=6, reserve_price=0, reserve_cycles=4, **common) == ["g2"]
    batch = read_batch(post, costs, decision, budget_left=100, cycles_left=40, reserve_price=0, reserve_cycles=0, **common)
    assert "g3" not in batch and "g4" not in batch
    flat = effect_with_means(TINY, table_of(TINY, lambda h: [0.0, 0.0, 0.0, 0.0]), noise=0.01)
    flat_post = uniform(flat)
    assert read_batch(flat_post, costs, decide(flat_post, costs, samples=64, seed=0), budget_left=100, cycles_left=40, reserve_price=0,
                      reserve_cycles=0, **common) == []


def test_a_decisive_read_under_a_concentrated_belief_heads_the_batch():
    """With nine tenths of the mass on one combination and the
    rest uniform over the eleven others, the read that tells that combination from the others has a negative expected
    elimination, N(b) minus 0.9 times 1 minus 0.1 times 11, while its information gain, H(b) minus 0.1 ln 11, is positive.
    Selection by information gain per unit cost puts the decisive read first, where a rule that admits only positive
    expected elimination would leave the batch empty."""
    top = (1, 2)
    model = effect_with_means(TINY, table_of(TINY, lambda h: [0.0, 10.0 * ((h["A"], h["B"]) == top), 0.0, 0.0]), noise=0.01)
    post = uniform(model)
    k = len(TINY.combinations) - 1
    post.override([0.9 if h == top else 0.1 / k for h in TINY.combinations])
    costs = tiny_costs()
    decision = decide(post, costs, samples=SAMPLES, seed=0)
    scores = eliminations(post, decision)
    h_before = -0.9 * math.log(0.9) - 0.1 * math.log(0.1 / k)
    assert abs(scores["g2"].effective_before - math.exp(h_before)) < 1e-9
    assert scores["g2"].elimination < 0 and abs(scores["g2"].elimination - (math.exp(h_before) - 0.9 - 0.1 * k)) <= 4 * scores["g2"].standard_error + 0.05
    assert abs(scores["g2"].information_gain - (h_before - 0.1 * math.log(k))) < 0.05
    assert all(scores[g].information_gain == 0 for g in ("g1", "g3", "g4"))
    assert read_batch(post, costs, decision, budget_left=100, cycles_left=40, reserve_price=0, reserve_cycles=0, seed=0, calls_left=100)[:1] == ["g2"]


def test_ec2_prefers_the_read_that_decides_the_repair_plan_where_the_gain_prefers_levels():
    """With A healthy and B uniform over its four levels, g1 tells levels 2 and 3 apart but merges the healthy
    level with level 1, g2 tells healthy from faulty B, g3 reads nothing and g4 reads every level at a high price. The
    plan after every outcome of g1 is to repair B, so g1 cuts no edge between the plan classes, while g2 cuts all of
    them, the weight of the edges between healthy B and faulty B, a quarter times three quarters. One step edge cutting
    per cost thus picks g2 and the information gain per cost picks g1, and the cut is zero for every read when every
    combination with mass shares one plan, though the gain is not."""
    costs = tiny_costs(prices={"g1": 1.0, "g2": 1.0, "g3": 1.0, "g4": 50.0})
    model = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * max(h["B"], 1), 10.0 * (h["B"] != 0), 0.0, 10.0 * h["B"] + 100.0 * h["A"]]),
                              noise=0.01)
    post = Posterior.uniform(model, block=False, support=[(0, b) for b in range(4)])
    decision = decide(post, costs, samples=SAMPLES, seed=0)
    units, _unit = read_units(decision, costs)
    cuts = ec2_cuts(post, costs, seed=0)
    edges = 0.25 * 0.75
    assert all(abs(cuts[g] - value) < 1e-9 for g, value in {"g1": 0.0, "g2": edges, "g3": 0.0, "g4": edges}.items()), cuts
    assert cuts == ec2_cuts(post, costs, seed=0)
    ec2_rate = per_cost(cuts, units)
    ig_rate = per_cost({g: s.information_gain for g, s in eliminations(post, decision).items()}, units)
    assert max(ec2_rate, key=ec2_rate.get) == "g2" and max(ig_rate, key=ig_rate.get) == "g1"
    one_plan = Posterior.uniform(model, block=False, support=[(0, b) for b in (1, 2, 3)])
    assert set(ec2_cuts(one_plan, costs, seed=0).values()) == {0.0}
    assert eliminations(one_plan, decide(one_plan, costs, samples=256, seed=0))["g1"].information_gain > 0


def test_the_card_cap_bounds_the_batch():
    """The batch_cap of the decision threshold card stops E22 at that many reads: in the dominated world the batch of two
    reads keeps its first read alone under a cap of one, and without a card the batch has no count of its own."""
    costs = tiny_costs(prices={"g1": 1.0, "g2": 1.0, "g3": 1.5, "g4": 50.0})
    post = uniform(effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 0.0, 0.0]), noise=0.01))
    decision = decide(post, costs, samples=SAMPLES, seed=0)
    common = {"budget_left": 100, "cycles_left": 40, "reserve_price": 0, "reserve_cycles": 0, "seed": 0, "calls_left": 100}
    full = read_batch(post, costs, decision, **common)
    assert len(full) >= 2 and read_batch(post, costs, decision, cap=None, **common) == full
    assert read_batch(post, costs, decision, cap=1, **common) == full[:1]


def test_without_a_count_the_calls_left_bound_the_batch_beside_the_reserve():
    """The batch has no count of its own: the module holds no cap, and a batch that the money and the cycles would let
    grow ends where the calls left before the step cap no longer pay a further read beside the calls of the reserved plan."""
    import inspect
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import read_choice as elimination
    assert not hasattr(elimination, "BATCH_CAP") and inspect.signature(read_batch).parameters["cap"].default is None
    costs = tiny_costs(prices={"g1": 1.0, "g2": 1.0, "g3": 1.5, "g4": 50.0})
    post = uniform(effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 0.0, 0.0]), noise=0.01))
    decision = decide(post, costs, samples=SAMPLES, seed=0)
    common = {"budget_left": 100, "cycles_left": 40, "reserve_price": 0, "reserve_cycles": 0, "seed": 0}
    full = read_batch(post, costs, decision, calls_left=100, **common)
    assert len(full) >= 2
    assert read_batch(post, costs, decision, calls_left=3, reserve_calls=2, **common) == full[:1]
    assert read_batch(post, costs, decision, calls_left=2, reserve_calls=2, **common) == []


def test_the_reads_of_the_rule_end_by_themselves_after_one_read_per_group():
    """Without a count the reads of the rule end when it names no read: eight groups that share no field give eight reads, each
    group once, more than the six the server once allowed."""
    import inspect
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import reads_in_turn
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import Left
    assert "cap" not in inspect.signature(reads_in_turn).parameters
    groups = {f"g{i}": (f"field{i}",) for i in range(8)}

    class View:
        def group_fields(self, group):
            return groups[group]

    def first(left, spent):
        open_groups = sorted(g for g, entry in left.items() if entry.cycles > 0)
        return [(open_groups[0], 1)] if open_groups else []

    reads = reads_in_turn(first, View(), {g: Left(5, 5) for g in groups}, lambda read: (0.0, 1, 0))
    assert [g for g, _cycles in reads] == sorted(groups)


def test_without_a_count_a_batch_passes_six_reads_where_the_limits_pay_them(monkeypatch):
    """A planted world where every simulated outcome names the same informative group: the batch takes a further read of it
    until the calls left before the step cap stop it, ten reads where the calls pay ten, more than the six the server once
    allowed, and a card's cap still bounds it."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import read_choice as elimination
    costs = tiny_costs(prices={"g1": 1.0, "g2": 1.0, "g3": 1.5, "g4": 50.0})
    post = uniform(effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 0.0, 0.0]), noise=0.01))
    decision = decide(post, costs, samples=SAMPLES, seed=0)
    monkeypatch.setattr(elimination, "scenario_rates", lambda post, groups, units, samples, rng: {g: (1.0 if g == "g2" else 0.0) for g in groups})
    common = {"budget_left": 1000, "cycles_left": 1000, "reserve_price": 0, "reserve_cycles": 0, "seed": 0}
    batch = read_batch(post, costs, decision, calls_left=10, **common)
    assert len(batch) == 10 and set(batch[1:]) == {"g2"}
    assert len(read_batch(post, costs, decision, calls_left=10, cap=6, **common)) == 6


def test_the_one_read_of_a_free_row_fits_the_calls_and_the_cycles_left():
    """E22 on a row whose rules price nothing is one read of the separating test over the cycles the row can still serve it,
    when its call fits the calls left beside the orders of the reserved plan, the diagnosis and the finish, and its cycles
    leave the cycles of that plan before the horizon."""
    cost = tiny_costs()._replace(test_price={g: 1.0 for g in TINY.groups}, unit="calls")
    # the reserve calls are those of finishing the plan (branches.plan_completion): its orders, the diagnosis and the finish
    orders, plan_cycles = 2, 4
    reserve = orders + ACT_CALLS
    calls = 1 + reserve
    assert ACT_CALLS == 2
    assert one_read("g1", 5, cost, calls, 30, reserve, plan_cycles) == [("g1", 5)]
    assert one_read("g1", 5, cost, calls - 1, 30, reserve, plan_cycles) == []
    assert one_read("g1", 0, cost, calls, 30, reserve, plan_cycles) == []
    assert one_read("g1", 5, cost, calls, 5 + plan_cycles + 1, reserve, plan_cycles) == [("g1", 5)]
    assert one_read("g1", 5, cost, calls, 5 + plan_cycles, reserve, plan_cycles) == []
    assert one_read("g1", 3, cost, 1 + ACT_CALLS, 30, ACT_CALLS, 0) == [("g1", 3)]  # a plan with no order, a start with no fault, keeps the act alone


def test_the_read_of_the_record_is_the_richest_group_the_money_pays():
    """E22 on a row whose rules hold a price is one read of the record: the group that can still show the most summaries,
    ties to the lower price of one cycle, then to the name, for the cycles the row can still serve it and the money left
    beside the reserve pays, all of them for a group without a price. A group of which the money pays no cycle, or whose
    cycles do not fit before the horizon beside the reserve, gives way to the next in the same order, a group with nothing
    left is never read, and no read is named when the call does not fit beside the reserve, the diagnosis and the finish."""
    cost = tiny_costs(prices={"g1": 1.0, "g2": 3.0, "g3": 10.0, "g4": 1.0})
    serving = {"g1": Left(5, 4), "g2": Left(5, 8), "g3": Left(5, 4), "g4": Left(0, 0)}
    reserve = (50.0, 2 + ACT_CALLS, 4)  # the price, the calls of finishing it (its orders, the diagnosis, finish) and the cycles of the repair of every component
    calls, cycles = 1 + 2 + ACT_CALLS, 60
    assert record_read(serving, cost, 1000.0, reserve, calls, cycles) == [("g2", 5)]
    assert record_read(serving, cost, float("inf"), reserve, calls, cycles) == [("g2", 5)]  # a row with prices and no budget
    assert record_read(serving, cost, 50.0 + 9, reserve, calls, cycles) == [("g2", 3)]
    assert record_read(serving, cost, 50.0 + 2, reserve, calls, cycles) == [("g1", 2)]
    assert record_read(serving, cost, 50.0 + 0.5, reserve, calls, cycles) == []
    free = cost._replace(test_price={**cost.test_price, "g2": 0.0})
    assert record_read(serving, free, 50.0, reserve, calls, cycles) == [("g2", 5)]
    assert record_read({**serving, "g2": Left(0, 0)}, cost, 1000.0, reserve, calls, cycles) == [("g1", 5)]
    assert record_read({g: Left(0, 0) for g in serving}, cost, 1000.0, reserve, calls, cycles) == []
    assert record_read(serving, cost, 1000.0, reserve, calls - 1, cycles) == []
    assert record_read(serving, cost, 1000.0, reserve, calls, 5 + 4 + 1) == [("g2", 5)]
    assert record_read({**serving, "g1": Left(3, 4)}, cost, 1000.0, reserve, calls, 5 + 4) == [("g1", 3)]
    assert record_read(serving, cost, 1000.0, reserve, calls, 5 + 4) == []


def test_the_group_of_the_rule_shows_the_most_summaries_left():
    """On a row whose rules price nothing the rule reads the group that can still show the most summaries, ties to the lower
    expected loss after its test and then to the name, among the groups the row can still serve, and none when none is left."""
    losses = {"g1": 3.0, "g2": 1.0, "g3": 2.0, "g4": 0.5}
    serving = {"g1": Left(5, 8), "g2": Left(2, 8), "g3": Left(5, 4), "g4": Left(0, 0)}
    assert rule_group(serving, losses) == "g2"
    assert rule_group({**serving, "g2": Left(2, 3)}, losses) == "g1"
    assert rule_group({**serving, "g1": Left(5, 8), "g2": Left(5, 8)}, {**losses, "g1": 1.0}) == "g1"
    assert rule_group({g: Left(0, 0) for g in serving}, losses) is None


def test_the_reads_of_the_rule_follow_one_another_on_groups_that_are_not_nested():
    """A view with two sensor groups that share no channel and a third inside the first: the rule reads the first group for
    its cycles, then the second, and the third never, since the first holds its summaries. With money for one read alone the
    list ends after it. On the nested groups of the rig the list is the one read of the rule."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import LIVE_WINDOW, profile_view
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import costs
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import reads_in_turn
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import left_to_serve

    rig = profile_view()
    channels = sorted({c for group in rig.sensor_groups.values() for c in group})
    groups = {"wide": channels[:6], "other": channels[6:10], "inside": channels[:3]}
    view = profile_view({**rig.rules, "sensor_groups": groups, "prices": {**rig.rules["prices"], "read": {"wide": 3, "other": 2, "inside": 1}}})
    cost = costs(view, 0)
    reserve = (0.0, 0, 0)

    def turn(view, cost, budget):
        def first(serving, spent):
            return record_read(serving, cost, budget - spent[0], reserve, 60 - spent[1], 60 - spent[2])

        def spend(read):
            group, cycles = read
            return cost.test_price[group] * cycles, 1, cycles * cost.test_duration[group]

        return reads_in_turn(first, view, left_to_serve(view, [], LIVE_WINDOW), spend)

    assert turn(view, cost, 1000.0) == [("wide", LIVE_WINDOW), ("other", LIVE_WINDOW)]
    assert turn(view, cost, 3.0 * LIVE_WINDOW + 1) == [("wide", LIVE_WINDOW)]
    assert turn(rig, costs(rig, 0), 1000.0) == [(max(rig.sensor_groups, key=lambda g: len(rig.group_fields(g))), LIVE_WINDOW)]


def test_a_read_whose_reserve_ends_at_the_horizon_is_not_bought():
    """The environment ends an episode at t >= T before a finish can run, so a read is bought
    only when its cycles and the reserve's end strictly before the horizon. A planted state with cycles_left equal to the
    read's duration plus the reserve's cycles buys nothing in read_batch, one_read and record_read; one cycle more buys it."""
    costs = tiny_costs(prices={"g1": 5.0, "g2": 5.0, "g3": 5.0, "g4": 5.0})
    model = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 0.0, 0.0]), noise=0.01)
    post = uniform(model)
    decision = decide(post, costs, samples=256, seed=0)
    duration, reserve_cycles = costs.test_duration["g2"], 4
    edge = duration + reserve_cycles
    assert read_batch(post, costs, decision, budget_left=60, cycles_left=edge, reserve_price=0, reserve_cycles=reserve_cycles, seed=0, calls_left=100) == []
    assert read_batch(post, costs, decision, budget_left=60, cycles_left=edge + 1, reserve_price=0, reserve_cycles=reserve_cycles, seed=0, calls_left=100) == ["g2"]
    assert one_read("g2", 1, costs, 10, edge, 0, reserve_cycles) == []
    assert one_read("g2", 1, costs, 10, edge + 1, 0, reserve_cycles) == [("g2", 1)]
    serving = {g: Left(1, 1) for g in costs.test_price}
    assert record_read(serving, costs, 1000.0, (0.0, 0, reserve_cycles), 10, duration + reserve_cycles) == []
    assert record_read(serving, costs, 1000.0, (0.0, 0, reserve_cycles), 10, duration + reserve_cycles + 1) != []
