"""E7 to E12 on the placeholder rules and a synthetic effect model: the costs, the loss matrix, the degenerate case, the
seeding of the lookahead, the one component threshold and the unseparated components."""
import numpy as np

from envs.hydraulic_rig.data import COMPONENTS
from servers.causal_diagnosis import decision as de
from servers.causal_diagnosis import effect_model as em
from servers.causal_diagnosis import posterior as po
from servers.causal_diagnosis.tests.fixture_episode import PLACEHOLDER_RULES
from servers.causal_diagnosis.tests.test_posterior import RULES, reading, synthetic_model

RULES_FULL = {**PLACEHOLDER_RULES, "sensor_groups": {"tier1": [], "tier2": [], "tier3": []}}


def test_costs_trace_to_prices_durations_and_downtime_rates():
    cost = de.costs(RULES_FULL, time=10)
    prices, durations, rates = RULES_FULL["prices"], RULES_FULL["durations"], RULES_FULL["downtime_rate"]
    for c in COMPONENTS:
        assert cost.unnecessary[c] == prices["parts"][c] + prices["repair_charge"] + rates["stopped"] * durations["repair_downtime"][c]
        assert cost.missed[c] == {int(level): rate * (RULES_FULL["horizon"] - 10) for level, rate in rates[c].items()}
    assert cost.remaining == RULES_FULL["horizon"] - 10 and cost.test_price == prices["read"]
    assert de.costs(RULES_FULL, time=RULES_FULL["horizon"] + 5).remaining == 0
    assert de.test_cost("tier1", cost, 2.5) == prices["read"]["tier1"] + durations["read_per_cycle"] * 2.5


def test_loss_matrix_charges_unnecessary_replacements_and_missed_faults():
    cost = de.costs(RULES_FULL, time=0)
    current = [(100, 100, 0, 130), (100, 100, 1, 130), (3, 100, 0, 130)]
    loss = de.loss_matrix(current, cost)
    horizon = RULES_FULL["horizon"]
    assert loss[0].tolist() == [0.0, 1 * horizon, 2 * horizon]
    assert loss[1].tolist() == [cost.unnecessary["pump"], 0.0, cost.unnecessary["pump"] + 2 * horizon]
    assert loss[2].tolist() == [cost.unnecessary["cooler"], cost.unnecessary["cooler"] + 1 * horizon, 0.0]
    only_pump = de.loss_matrix(current, cost, ["pump"])
    assert only_pump[2].tolist() == [0.0, 1 * horizon, 0.0]


def test_a_single_candidate_acts_and_the_lookahead_is_seeded():
    model = synthetic_model()
    cost = de.costs(RULES_FULL, time=0)
    single = po.Posterior(model, RULES, [], [], time=0, starts=[(100, 80, 0, 90)])
    decision = de.decide(single, cost, samples=64)
    assert decision.expected_loss_act == 0 and not decision.insufficient_evidence and decision.expected_saving <= 0
    assert all(value >= decision.test_cost[g] for g, value in decision.expected_loss_test.items())
    post = po.Posterior(model, RULES, [], [], time=0)
    first, second = de.decide(post, cost, samples=64, seed=0), de.decide(post, cost, samples=64, seed=0)
    assert first.expected_loss_test == second.expected_loss_test and first.separating_test == second.separating_test
    assert first.insufficient_evidence == (min(first.expected_loss_test.values()) < first.expected_loss_act)
    assert abs(first.expected_saving - (first.expected_loss_act - min(first.expected_loss_test.values()))) < 1e-12
    assert set(first.expected_loss_test) == set(cost.test_price) and first.samples == 64 and first.seed == 0


def test_an_informative_reading_resolves_an_uncertain_prior():
    model = synthetic_model(noise=0.01)
    cost = de.costs(RULES_FULL, time=0)
    post = po.Posterior(model, RULES, [], [], time=0)
    decision = de.decide(post, cost, samples=128)
    # a reading at noise far below the effects tells the combinations apart, so testing first beats acting on a uniform prior
    assert decision.insufficient_evidence and decision.separating_test in cost.test_price
    resolved = po.Posterior(model, RULES, [reading(model, 1, 1, "tier3", (100, 80, 0, 90))], [], time=1)
    assert not de.decide(resolved, cost, samples=128).insufficient_evidence


def test_one_component_threshold_follows_the_prices():
    assert de.one_component_threshold(1, 40) == 0.975 and de.one_component_threshold(10, 40) == 0.75


def test_unseparated_needs_a_decision_change_between_the_top_two():
    model = synthetic_model(noise=0.01)
    cost = de.costs(RULES_FULL, time=0)
    pair = po.Posterior(model, RULES, [], [], time=0, starts=[(100, 80, 0, 90), (100, 80, 0, 130)])
    decision = de.decide(pair, cost, samples=128)
    separating = de.unseparated(pair, cost, decision)
    assert list(separating) == ["accumulator"] and separating["accumulator"] in [list(f) for f in em.GROUP_FIELDS.values()]
    same_faults = po.Posterior(model, RULES, [], [], time=0, starts=[(100, 80, 0, 90), (100, 73, 0, 100)])
    assert de.unseparated(same_faults, cost, de.decide(same_faults, cost, samples=64)) == {}
