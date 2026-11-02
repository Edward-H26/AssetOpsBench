"""E2 to E6 on a synthetic effect model: the prior over the rule, the update by a reading, the current combination after
closed orders, the merging of repeated cycles, the discount of the block variant, the stability over resampled fits and
the predictive sampling."""
import numpy as np
import pytest

from envs.hydraulic_rig.classifiers import COMBOS
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY
from servers.causal_diagnosis import effect_model as em
from servers.causal_diagnosis import posterior as po

RULES = {"fault_rule": {"min_faulty": 1, "max_faulty": 2}}
D = len(em.SUMMARY_FIELDS)


def synthetic_model(seed=0, noise=0.05, block=0.0, resamples=3, jitter=0.0):
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=(1 + len(em.MAIN_TERMS), D))
    drawn = [(beta + jitter * rng.normal(size=beta.shape), noise ** 2 * np.eye(D)) for _ in range(resamples)]
    calibration = {"n_blocks": 10, "scores": np.abs(rng.normal(size=(50, D))), "weights": np.full(50, 0.2)}
    return em.EffectModel([], beta, noise ** 2 * np.eye(D), block ** 2 * np.eye(D), drawn, np.zeros(D), np.ones(D), calibration, {})


def reading(model, n, t, group, combo, shift=0.0):
    fields = em.GROUP_FIELDS[group]
    mu = model.mu(combo)
    return po.Reading(n, t, group, {f: float(mu[em.FIELD_INDEX[f]] + shift) for f in fields})


def test_without_readings_the_posterior_is_uniform_over_the_rule():
    model = synthetic_model()
    post = po.Posterior(model, RULES, [], [], time=0)
    assert len(post.candidates) == sum(1 for h in COMBOS if 1 <= em.n_faulty(h) <= 2) == 47
    assert np.allclose(post.probabilities, 1 / 47) and post.current == post.candidates
    one = po.Posterior(model, {"fault_rule": {"min_faulty": 1, "max_faulty": 1}}, [], [], time=0)
    assert len(one.candidates) == 10
    assert all(abs(sum(levels.values()) - 1) < 1e-9 for levels in post.level_probability().values())


def test_a_reading_at_the_model_mean_puts_its_combination_on_top():
    model = synthetic_model()
    truth = (100, 80, 0, 90)
    post = po.Posterior(model, RULES, [reading(model, 1, 1, "tier3", truth)], [], time=1)
    assert post.candidates[post.order()[0]] == truth and post.probabilities[post.order()[0]] > 0.99
    marginals = post.component_marginals()
    assert marginals["valve"] > 0.99 and marginals["accumulator"] > 0.99 and marginals["cooler"] < 0.01 and marginals["pump"] < 0.01
    assert post.current_marginals() == marginals


def test_closed_orders_set_the_current_combination_and_read_later_cycles_under_it():
    model = synthetic_model()
    truth, repaired = (100, 80, 0, 90), (100, 100, 0, 90)
    orders = [po.ClosedOrder(10, "valve", "agent", "wo-1")]
    reads = [reading(model, 1, 1, "tier3", truth), reading(model, 2, 12, "tier3", repaired)]
    post = po.Posterior(model, RULES, reads, orders, time=12)
    assert [len(s.observations) for s in post.segments] == [1, 1] and post.segments[1].orders == tuple(orders)
    assert post.candidates[post.order()[0]] == truth and post.current[post.order()[0]] == repaired
    assert post.current_marginals()["valve"] == 0 and post.component_marginals()["valve"] > 0.99
    # a reading at the closing cycle is already served from the new block
    assert po.orders_before(orders, 10) == tuple(orders) and po.orders_before(orders, 9) == ()
    after, addresses, recovery = post.post_repair("accumulator")
    assert all(h[COMPONENTS.index("accumulator")] == HEALTHY["accumulator"] for h in after) and abs(sum(after.values()) - 1) < 1e-9
    assert abs(addresses - post.current_marginals()["accumulator"]) < 1e-12 and recovery > 0.99


def test_repeated_cycles_merge_within_a_segment_and_not_across_segments():
    model = synthetic_model()
    truth = (100, 80, 0, 90)
    first = reading(model, 1, 1, "tier1", truth)
    full = reading(model, 2, 2, "tier3", truth)
    again = po.Reading(3, 3, "tier3", dict(full.values))
    other = reading(model, 4, 4, "tier3", truth, shift=0.01)
    merged = po.merge_observations([first, full, again, other])
    assert [len(o.fields) for o in merged] == [D, D] and merged[0].numbers == (1, 2, 3) and merged[1].numbers == (4,)
    orders = [po.ClosedOrder(2, "cooler", "external", "wo-ext-1")]
    segments = po.segments([first, again], orders)
    assert [len(s.observations) for s in segments] == [1, 1]


def test_the_block_variant_discounts_repeated_readings_while_the_literal_model_counts_them():
    literal, block = synthetic_model(noise=0.1), synthetic_model(noise=0.1, block=0.3)
    a, b = (100, 80, 0, 90), (100, 80, 0, 100)
    one = [reading(literal, 1, 1, "tier3", a)]
    three = one + [po.Reading(n, n, "tier3", dict(one[0].values)) for n in (2, 3)]
    three = [one[0], reading(literal, 2, 2, "tier3", a, shift=0.001), reading(literal, 3, 3, "tier3", a, shift=0.002)]

    def ratio(model, reads):
        post = po.Posterior(model, RULES, reads, [], time=len(reads))
        i, j = post.candidates.index(a), post.candidates.index(b)
        return post.log_weights[i] - post.log_weights[j]

    assert abs(ratio(literal, three) - 3 * ratio(literal, one)) < 0.05 * abs(ratio(literal, one))
    assert ratio(block, three) < 3 * ratio(block, one)
    assert ratio(block, three) > ratio(block, one) > 0


def test_stability_and_resampled_posteriors():
    same = synthetic_model(resamples=4, jitter=0.0)
    truth = (3, 100, 0, 130)
    post = po.Posterior(same, RULES, [reading(same, 1, 1, "tier3", truth)], [], time=1, resampled=True)
    assert post.resampled_probabilities.shape == (4, len(post.candidates))
    stability = post.stability()
    assert stability["top_component"] == "cooler" and stability["resamples"] == 4 and stability["top_share"] == 1.0
    assert stability["interval_low"] <= stability["top_marginal"] <= stability["interval_high"]
    jittered = synthetic_model(resamples=4, jitter=0.5)
    post = po.Posterior(jittered, RULES, [reading(jittered, 1, 1, "tier1", truth)], [], time=1, resampled=True)
    stability = post.stability()
    assert 0 <= stability["interval_low"] <= stability["interval_high"] <= 1 and 0 <= stability["top_share"] <= 1
    recovery = post.resampled_recovery("cooler")
    assert recovery.shape == (4,) and np.all((0 <= recovery) & (recovery <= 1))
    assert po.Posterior(jittered, RULES, [], [], time=0).stability() is None


def test_predictive_sampling_and_hypothetical_updates():
    model = synthetic_model()
    post = po.Posterior(model, RULES, [], [], time=0)
    rng = np.random.default_rng(0)
    fields = em.GROUP_FIELDS["tier2"]
    x = post.sample_next(fields, 16, rng)
    assert x.shape == (16, len(fields))
    updated = post.update_with(fields, x)
    assert updated.shape == (16, len(post.candidates)) and np.allclose(updated.sum(axis=1), 1)
    exact = np.array([[model.mu(post.candidates[5])[em.FIELD_INDEX[f]] for f in fields]])
    assert post.update_with(fields, exact)[0].argmax() == 5


def test_override_replaces_the_probabilities_and_keeps_the_candidates():
    model = synthetic_model()
    post = po.Posterior(model, RULES, [], [], time=0, resampled=True)
    weights = np.arange(1, len(post.candidates) + 1, dtype=float)
    post.override(weights)
    assert np.allclose(post.probabilities, weights / weights.sum()) and post.resampled_probabilities is None
    assert post.candidates[post.order()[0]] == post.candidates[-1]


def test_a_posterior_over_given_starts():
    model = synthetic_model()
    starts = [(100, 100, 0, 90), em.HEALTHY_COMBO]
    post = po.Posterior(model, RULES, [reading(model, 1, 1, "tier3", em.HEALTHY_COMBO)], [], time=1, starts=starts)
    assert post.candidates == starts and post.current_posterior()[em.HEALTHY_COMBO] > 0.99
