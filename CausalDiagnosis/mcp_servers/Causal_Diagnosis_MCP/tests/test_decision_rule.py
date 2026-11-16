"""E7 to E12 on the default rules of the asset and a synthetic effect model: the costs, the loss matrix, the degenerate
case, the seeding of the draws of E9, the predict step inside E9, the one component threshold, the unseparated
components, and the price of a wrong diagnosis on a split between two levels of one faulty set."""
import numpy as np

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import cost_unit
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import ABSTENTION_CREDIT
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import decision_rule as de
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import eliminations
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import Left
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import with_misdiagnosis
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.synthetic import effect_with_means
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import SPACE, random_model

VIEW = profile_view()
LEVEL_SPLIT = [(100, 80, 0, 90), (100, 73, 0, 100)]  # one faulty set, the valve and the accumulator, at two levels each
PRICE = VIEW.visit_charge()  # the price of a wrong diagnosis on the priced presets, the price table's default visit charge


def priced(view, price):
    return profile_view(with_misdiagnosis(view.rules, price))

def free_view():
    """The default view of the asset on the free base row of the environment table, where the rules price nothing."""
    rules = dict(profile_view().rules)
    prices = rules["prices"]
    free = {**prices, "read": {g: 0 for g in prices["read"]}, "parts": {c: 0 for c in prices["parts"]}, "repair_charge": 0, "visit": 0}
    return profile_view({**rules, "prices": free})


def test_the_free_base_row_counts_every_cost_in_calls():
    """On the base row, where the rules price nothing, a test costs one call, an unnecessary order, a missed fault and a
    wrong diagnosis each cost the calls left before the step cap, and no downtime is charged, while the priced rows keep
    their money costs."""
    view = free_view()
    assert cost_unit(VIEW.rules["prices"]) == de.costs(VIEW, time=3).unit == "money"
    assert cost_unit(view.rules["prices"]) == de.costs(view, time=3).unit == "calls"  # the environment's rule decides the unit
    cost = de.costs(view, time=3, steps_left=17)
    assert cost.unit == "calls" and cost.remaining == 17 and de.costs(view, time=3).remaining == view.rules["max_steps"]
    assert all(cost.unnecessary[c] == 17 for c in view.components) and cost.misdiagnosis == 17 and set(cost.test_price.values()) == {1.0}
    assert all(cost.missed[c][lv] == (0 if lv == view.healthy[c] else 17) for c in view.components for lv in view.levels[c])
    assert all(rate == 0 for c in view.components for rate in cost.rates[c].values())
    post = Posterior.uniform(random_model(np.random.default_rng(3)))
    decision = de.decide(post, de.costs(profile_view({**view.rules, "max_steps": 60}), 0), samples=64, seed=0)
    assert set(decision.test_cost.values()) == {1.0} and decision.expected_loss_act > 0


def test_costs_trace_to_prices_durations_and_downtime_rates():
    cost = de.costs(VIEW, time=10)
    prices, durations, rates = VIEW.rules["prices"], VIEW.rules["durations"], VIEW.rules["downtime_rate"]
    for c in VIEW.components:
        assert cost.unnecessary[c] == prices["parts"][c] + prices["repair_charge"] + rates["stopped"] * durations["repair_downtime"][c]
        assert cost.missed[c] == {int(level): rate * (VIEW.rules["horizon"] - 10) for level, rate in rates[c].items()}
    assert cost.remaining == VIEW.rules["horizon"] - 10 and cost.test_price == prices["read"]
    assert de.costs(VIEW, time=VIEW.rules["horizon"] + 5).remaining == 0
    assert de.test_cost("tier1", cost, 2.5) == prices["read"]["tier1"] + durations["read_per_cycle"] * 2.5


def test_loss_matrix_charges_unnecessary_replacements_and_missed_faults():
    cost = de.costs(VIEW, time=0)
    current = [(100, 100, 0, 130), (100, 100, 1, 130), (3, 100, 0, 130)]
    loss = de.loss_matrix(current, cost)
    horizon = VIEW.rules["horizon"]
    assert loss[0].tolist() == [0.0, 1 * horizon, 2 * horizon]
    assert loss[1].tolist() == [cost.unnecessary["pump"], 0.0, cost.unnecessary["pump"] + 2 * horizon]
    assert loss[2].tolist() == [cost.unnecessary["cooler"], cost.unnecessary["cooler"] + 1 * horizon, 0.0]
    assert de.loss_matrix(current, cost, ["pump"])[2].tolist() == [0.0, 1 * horizon, 0.0]


def test_a_single_combination_acts_and_the_draws_of_e9_are_seeded():
    model = random_model()
    cost = de.costs(VIEW, time=0)
    single = Posterior.uniform(model, support=[(100, 80, 0, 90)])
    decision = de.decide(single, cost, samples=64)
    assert decision.expected_loss_act == 0 and not decision.read_first and decision.expected_saving <= 0
    assert all(value >= decision.test_cost[g] for g, value in decision.expected_loss_test.items())
    post = Posterior.uniform(model)
    first, second = de.decide(post, cost, samples=64, seed=0), de.decide(post, cost, samples=64, seed=0)
    assert first.expected_loss_test == second.expected_loss_test and first.separating_test == second.separating_test
    assert first.read_first == (min(first.expected_loss_test.values()) < first.expected_loss_act)
    assert abs(first.expected_saving - (first.expected_loss_act - min(first.expected_loss_test.values()))) < 1e-12
    assert set(first.expected_loss_test) == set(cost.test_price) and first.samples == 64 and first.seed == 0
    assert set(first.masses_after) == set(first.draws) == set(cost.test_price)


def test_the_reading_is_drawn_from_the_belief_the_reading_meets():
    """E9 acts on the belief now and draws the reading from the belief the reading meets: when that belief has moved to
    one combination, every draw comes from it, so the expected loss after the test is the loss of acting on it."""
    model = random_model(noise=0.01)
    cost = de.costs(VIEW, time=0)
    now = Posterior.uniform(model)
    ahead = Posterior.uniform(model, support=[(100, 100, 0, 130)])
    decision = de.decide(now, cost, samples=64, seed=0, ahead=ahead)
    healthy_loss = de.loss_matrix(now.candidates, cost)[now.candidates.index((100, 100, 0, 130))]
    assert all(np.allclose(updated[:, now.candidates.index((100, 100, 0, 130))], 1.0) for updated in decision.masses_after.values())
    assert all(abs(decision.expected_loss_test[g] - decision.test_cost[g] - healthy_loss[now.candidates.index((100, 100, 0, 130))]) < 1e-9
               for g in cost.test_price)


def test_an_informative_reading_resolves_an_uncertain_prior():
    model = random_model(noise=0.01)
    cost = de.costs(VIEW, time=0)
    post = Posterior.uniform(model)
    decision = de.decide(post, cost, samples=128)
    # a reading at noise far below the effects tells the combinations apart, so testing first beats acting on a uniform belief
    assert decision.read_first and decision.separating_test in cost.test_price
    fields = SPACE.groups["tier3"]
    resolved = post.updated(fields, model.mu((100, 80, 0, 90))[SPACE.field_indices(fields)])
    assert not de.decide(resolved, cost, samples=128).read_first


def test_one_component_threshold_follows_the_prices():
    assert de.one_component_threshold(1, 40) == 0.975 and de.one_component_threshold(10, 40) == 0.75
    # the default prices of E11 with the price of a wrong diagnosis
    assert round(de.one_component_threshold(1, 40, 15), 3) == 0.982 and round(de.one_component_threshold(10, 40, 15), 3) == 0.818


def test_unseparated_needs_a_decision_change_between_the_top_two():
    model = random_model(noise=0.01)
    cost = de.costs(VIEW, time=0)
    pair = Posterior.uniform(model, support=[(100, 80, 0, 90), (100, 80, 0, 130)])
    separating = de.unseparated(pair, cost, de.decide(pair, cost, samples=128))
    assert list(separating) == ["accumulator"] and separating["accumulator"] in [list(f) for f in SPACE.groups.values()]
    same_faults = Posterior.uniform(model, support=[(100, 80, 0, 90), (100, 73, 0, 100)])
    assert de.unseparated(same_faults, cost, de.decide(same_faults, cost, samples=64)) == {}


def test_a_priced_wrong_diagnosis_makes_a_split_between_levels_worth_a_read():
    """Two combinations with one faulty set cost nothing to act on under the repair loss alone, so E10 found
    the evidence sufficient while the score asks for the levels. Under the price of a wrong diagnosis acting now costs
    that price times one minus the top mass, and a reading that tells the levels apart lowers it, so the evidence is
    insufficient."""
    pair = Posterior.uniform(random_model(noise=0.01), support=LEVEL_SPLIT)
    unpriced = de.decide(pair, de.costs(priced(VIEW, None), 0), samples=128)
    assert unpriced.expected_loss_act == 0 and not unpriced.read_first
    cost = de.costs(priced(VIEW, PRICE), 0)
    decision = de.decide(pair, cost, samples=128)
    assert PRICE > 0 and cost.misdiagnosis == PRICE
    assert abs(decision.expected_loss_act - PRICE * (1 - max(pair.probabilities))) < 1e-12
    assert decision.read_first and decision.separating_test in cost.test_price


def test_the_price_buys_no_read_that_cannot_tell_the_levels_apart():
    """When both levels give every summary the same mean, no reading moves the split, so under the price of a wrong
    diagnosis every test costs its price on top of the loss of acting now and the evidence stays sufficient."""
    means = np.random.default_rng(0).normal(size=(len(SPACE.combinations), len(SPACE.fields)))
    first, second = (SPACE.combinations.index(h) for h in LEVEL_SPLIT)
    means[second] = means[first]
    pair = Posterior.uniform(effect_with_means(SPACE, means, noise=0.01), support=LEVEL_SPLIT)
    decision = de.decide(pair, de.costs(priced(VIEW, PRICE), 0), samples=128)
    assert abs(decision.expected_loss_act - PRICE / 2) < 1e-12 and not decision.read_first
    assert all(decision.expected_loss_test[g] >= decision.expected_loss_act + decision.test_cost[g] - 1e-9 for g in decision.test_cost)


def test_rules_without_the_price_keep_the_repair_loss_exactly():
    """Rules without prices.misdiagnosis, as released before the price of a wrong diagnosis existed, give the repair loss of
    E8 alone, so every decision field equals the one that loss gives."""
    post = Posterior.uniform(random_model(noise=0.01))
    plain = de.costs(priced(VIEW, None), 0)
    assert plain.misdiagnosis == 0
    assert np.array_equal(de.decision_loss(post.candidates, plain), de.loss_matrix(post.candidates, plain))
    decision = de.decide(post, plain, samples=64, seed=0)
    loss = de.loss_matrix(post.candidates, plain)
    assert decision.expected_loss_act == de.expected_loss_act(post.probabilities, loss, post.order()[0])
    assert all(decision.expected_loss_test[g] == decision.test_cost[g] + de.expected_loss_test(decision.masses_after[g], loss)[0]
               for g in plain.test_price)


def test_the_card_margin_turns_away_a_test_that_saves_little_and_a_zero_margin_is_the_plain_call():
    """The e10_margin of the decision threshold card asks the best test to lower the expected loss by more than the margin
    times its cost, so a margin just above the saving per unit cost turns the evidence sufficient, one just below keeps it
    insufficient, and a margin of zero gives every field of the call without it."""
    post = Posterior.uniform(random_model(noise=0.01))
    cost = de.costs(priced(VIEW, None), 0)
    plain = de.decide(post, cost, samples=64, seed=0)
    zero = de.decide(post, cost, samples=64, seed=0, margin=0.0)
    fields = ("read_first", "separating_test", "expected_saving", "expected_loss_act", "expected_loss_test", "test_cost")
    assert all(getattr(zero, name) == getattr(plain, name) for name in fields)
    assert plain.read_first
    ratio = plain.expected_saving / plain.test_cost[plain.separating_test]
    assert de.decide(post, cost, samples=64, seed=0, margin=ratio * 0.99).read_first
    assert not de.decide(post, cost, samples=64, seed=0, margin=ratio * 1.01).read_first


# ----- the rule of a row whose rules price nothing -----

def free_costs(steps_left=30):
    return de.costs(free_view(), time=0, steps_left=steps_left)


def serving_of(left):
    """A count as serving.left_to_serve gives it, with the given cycles left per sensor group and every summary of a group
    still lacking on each of them."""
    view = free_view()
    return {g: Left(left.get(g, 0), left.get(g, 0) * len(view.group_fields(g))) for g in view.sensor_groups}


def test_on_a_row_that_prices_nothing_an_act_off_the_truth_costs_one_failed_episode_once():
    """A level off, a work order too many, a fault left and a disjoint faulty set each void the pass, so each act costs the
    calls left, and an act with two of the errors costs them once."""
    truth = (100, 80, 0, 90)
    acts = [truth, (100, 73, 0, 90), (3, 80, 0, 90), (100, 80, 0, 130), (3, 100, 2, 130), (3, 73, 0, 90)]
    loss = de.decision_loss(acts, free_costs(steps_left=30))
    assert [loss[a, 0] for a in range(len(acts))] == [0.0, 30.0, 30.0, 30.0, 30.0, 30.0]


def test_on_a_row_that_prices_nothing_the_best_act_is_the_most_probable_combination():
    loss = de.decision_loss(SPACE.combinations, free_costs())
    rng = np.random.default_rng(7)
    for _ in range(20):
        p = rng.dirichlet(np.full(len(SPACE.combinations), 0.3))
        assert int(np.argmin([de.expected_loss_act(p, loss, a) for a in range(len(p))])) == int(np.argmax(p))
    larger, smaller = SPACE.combinations.index((3, 80, 0, 90)), SPACE.combinations.index((100, 80, 0, 90))
    p = np.zeros(len(SPACE.combinations))
    p[larger], p[smaller] = 0.51, 0.49  # a smaller faulty set that holds nearly as much does not win
    assert int(np.argmin([de.expected_loss_act(p, loss, a) for a in range(len(p))])) == larger


def test_a_split_between_two_levels_reads_on_a_row_that_prices_nothing():
    """Two combinations of one faulty set at one half each: acting costs half the calls left, where the repair loss alone
    gave nothing, so E10 asks for a read that tells them apart while a group can be served, and for none when no group is
    left."""
    pair = Posterior.uniform(random_model(noise=0.01), support=LEVEL_SPLIT)
    cost = free_costs(steps_left=30)
    decision = de.decide(pair, cost, samples=64, serving=serving_of({g: 5 for g in cost.test_price}))
    assert abs(decision.expected_loss_act - 15.0) < 1e-12 and decision.read_first
    assert not de.decide(pair, cost, samples=64, serving=serving_of({})).read_first


def test_a_group_the_row_can_serve_nothing_more_draws_nothing_and_e10_never_names_it():
    """Given what the row can still serve, a group with nothing left draws nothing and keeps the belief: its gain is zero and
    its expected loss after the test is its call plus the loss of acting, so E10 names another group while one lowers the
    loss and asks for no read when no group is left. A count with every group open draws as a call without one."""
    model = random_model(noise=0.01)
    cost = free_costs(steps_left=30)
    groups = sorted(cost.test_price)
    uniform = Posterior.uniform(model)
    some = serving_of({groups[0]: 5, groups[1]: 2})
    decision = de.decide(uniform, cost, samples=32, serving=some)
    closed = groups[2]
    assert decision.draws[closed].shape[0] == 0
    assert abs(decision.expected_loss_test[closed] - (decision.test_cost[closed] + decision.expected_loss_act)) < 1e-9
    assert eliminations(uniform, decision)[closed].information_gain == 0.0
    assert decision.read_first and decision.separating_test != closed
    for post in (uniform, Posterior.uniform(model, support=[(100, 80, 0, 90)])):
        assert not de.decide(post, cost, samples=32, serving=serving_of({})).read_first
    plain, counted = de.decide(uniform, cost, samples=32, seed=4), de.decide(uniform, cost, samples=32, seed=4, serving=serving_of({g: 5 for g in groups}))
    assert all(np.array_equal(plain.draws[g], counted.draws[g]) for g in groups) and plain.expected_loss_test == counted.expected_loss_test


def test_a_declaration_pays_above_the_mass_over_its_worth_and_never_under_top1_and_pass():
    assert de.DECLARATION_WORTH == {"top1": 0.0, "pass": 0.0, "success": 1.0, "reward": ABSTENTION_CREDIT}
    for mass in (0.0, ABSTENTION_CREDIT / 2, ABSTENTION_CREDIT, (1 + ABSTENTION_CREDIT) / 2, 1.0):
        above = de.declare_above(mass)
        assert above["top1"] is None and above["pass"] is None
        assert above["success"] == (mass if mass < 1 else None)
        assert above["reward"] == (mass / ABSTENTION_CREDIT if mass < ABSTENTION_CREDIT else None)


# ----- what a row serves, on every row, and the outlay of a plan -----

def record_view(view):
    """The view of the same rules on a row that reads a record: the released tool table lists the read of recorded cycles."""
    return profile_view(dict(view.rules, tools=[{"name": "history", "server": "iot"}]))


def test_a_read_of_a_recorded_cycle_costs_its_price_alone():
    """On a row that reads a record a read of a recorded cycle takes no time, so with a declared read duration of one cycle
    every test duration is zero there and the test cost is the read price at any downtime rate, on both cost units, while
    the same rules on a row that plays a recording per read charge the duration."""
    timed = profile_view(dict(VIEW.rules, durations={**VIEW.rules["durations"], "read_per_cycle": 1}))
    for view in (timed, free_view()):
        cost = de.costs(record_view(view), time=0)
        assert set(cost.test_duration.values()) == {0}
        assert all(de.test_cost(g, cost, rate) == cost.test_price[g] for g in cost.test_price for rate in (0.0, 3.5))
    played = de.costs(timed, time=0)
    assert set(played.test_duration.values()) == {1} and de.test_cost("tier1", played, 3.5) == played.test_price["tier1"] + 3.5


def test_a_group_with_nothing_left_is_no_test_on_a_row_with_prices():
    """decide on the money unit with a serving that closes one group: that group draws nothing and keeps the belief, so its
    expected loss after the test is its cost plus the loss of acting and its gain is zero, while the open groups are drawn
    and scored. One seeded stream serves the groups in the order of their names, so the draws of the groups after a closed
    one move."""
    model = random_model(noise=0.01)
    post = Posterior.uniform(model)
    cost = de.costs(VIEW, time=0)
    groups = sorted(cost.test_price)
    closed = groups[0]
    serving = {g: Left(0, 0) if g == closed else Left(5, 5 * len(VIEW.group_fields(g))) for g in groups}
    decision = de.decide(post, cost, samples=32, serving=serving)
    assert cost.unit == "money" and len(decision.draws[closed]) == 0
    assert np.array_equal(decision.masses_after[closed], post.probabilities[None, :])
    assert abs(decision.expected_loss_test[closed] - (decision.test_cost[closed] + decision.expected_loss_act)) < 1e-9
    assert eliminations(post, decision)[closed].information_gain == 0.0
    assert all(len(decision.draws[g]) == 32 and np.isfinite(decision.expected_loss_test[g]) for g in groups[1:])


def test_the_loss_beyond_the_outlay_holds_no_replacement_twice():
    """repair_losses splits the repair loss of E8 into its unnecessary and its missed part, which sum to loss_matrix, and
    the loss beyond the outlay of a plan is the missed part plus the price of a wrong diagnosis in money, since the outlay
    holds every replacement of the plan, and the loss of a failed episode, once, where the rules price nothing."""
    combos = list(VIEW.combinations)
    for cost in (de.costs(priced(VIEW, PRICE), time=0), de.costs(VIEW, time=0)):
        unnecessary, missed = de.repair_losses(combos, cost)
        np.testing.assert_allclose(unnecessary + missed, de.loss_matrix(combos, cost), rtol=1e-12, atol=0)
        assert (unnecessary >= 0).all() and (missed >= 0).all()
        np.testing.assert_array_equal(de.loss_beyond_outlay(combos, cost), missed + de.diagnosis_loss(combos, cost))
    free = free_costs(steps_left=30)
    np.testing.assert_array_equal(de.loss_beyond_outlay(combos, free), de.decision_loss(combos, free))
