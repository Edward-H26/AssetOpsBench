"""The collapsed belief of version 7 on synthetic effect models: a uniform belief over every combination of the profile, a
reading at a combination's mean puts it on top, the grouped predictive equals the density of each combination computed
alone, a reading of two cycles in a row equals two updates, each with covariances of its own per hypothesis as the clean fits
give them, the effective number of a uniform belief is its support, the stability over resampled fits and the
override of another ranking model."""
import numpy as np
import pytest

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import EffectSpace
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import log_density
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior, effective_number
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.synthetic import effect_with_means

SPACE = EffectSpace.from_view(profile_view())


def random_model(seed=0, noise=0.05, block=0.0, resamples=0, jitter=0.0):
    rng = np.random.default_rng(seed)
    return effect_with_means(SPACE, rng.normal(size=(len(SPACE.combinations), len(SPACE.fields))), noise=noise, block=block,
                             resamples=resamples, jitter=jitter, seed=seed)


def per_hypothesis_covariances(rng, scale):
    """One positive definite matrix per combination, scaled to about scale squared."""
    d = len(SPACE.fields)
    return np.array([scale ** 2 * (a @ a.T / d + 0.1 * np.eye(d)) for a in rng.normal(size=(len(SPACE.combinations), d, d))])


def test_without_readings_the_belief_is_uniform_over_every_combination():
    post = Posterior.uniform(random_model())
    assert len(post.candidates) == len(SPACE.combinations) == 144 and np.allclose(post.probabilities, 1 / 144)
    assert abs(post.effective_hypotheses() - 144) < 1e-9
    assert all(abs(sum(levels.values()) - 1) < 1e-9 for levels in post.level_probability().values())
    healthy = Posterior.uniform(random_model(), support=[SPACE.healthy_combination])
    assert healthy.current_posterior() == {SPACE.healthy_combination: 1.0} and abs(healthy.effective_hypotheses() - 1) < 1e-12


def test_a_reading_at_a_mean_puts_its_combination_on_top():
    model = random_model(noise=0.01)
    truth = (100, 80, 0, 90)
    fields = SPACE.groups["tier3"]
    post = Posterior.uniform(model).updated(fields, model.mu(truth)[SPACE.field_indices(fields)])
    assert post.candidates[post.order()[0]] == truth and post.probabilities[post.order()[0]] > 0.99
    marginals = post.current_marginals()
    assert marginals["valve"] > 0.99 and marginals["accumulator"] > 0.99 and marginals["cooler"] < 0.01 and marginals["pump"] < 0.01


def test_the_grouped_predictive_equals_the_density_of_each_combination_alone():
    model = random_model(block=0.2)
    rng = np.random.default_rng(3)
    d = len(SPACE.fields)
    covariance, model.sigma_within = per_hypothesis_covariances(rng, 1.0), per_hypothesis_covariances(rng, 0.05)
    post = Posterior(model, rng.dirichlet(np.ones(len(SPACE.combinations))), 0.1 * rng.normal(size=(len(SPACE.combinations), d)), covariance)
    fields = SPACE.groups["tier2"]
    idx = SPACE.field_indices(fields)
    x = post.sample_next(fields, 7, rng)
    grouped = post.log_density(fields, x)
    for h in range(len(SPACE.combinations)):
        cov = covariance[h][np.ix_(idx, idx)] + model.sigma_within[h][np.ix_(idx, idx)]
        alone = log_density(x - (model.mu_table[h, idx] + post.mean[h, idx]), np.linalg.cholesky(cov))
        assert np.allclose(grouped[:, h], alone, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("own_covariances", [False, True], ids=("shared", "per hypothesis"))
def test_a_reading_of_two_cycles_equals_two_updates_in_a_row(own_covariances):
    model = random_model(block=0.3)
    rng = np.random.default_rng(4)
    if own_covariances:
        model.sigma_block, model.sigma_within = per_hypothesis_covariances(rng, 0.3), per_hypothesis_covariances(rng, 0.05)
    first, second = SPACE.groups["tier1"], SPACE.groups["tier2"]
    post = Posterior.uniform(model)
    x = post.sample_next([first, second], 1, rng)[0]
    joint = post.update_with([first, second], x[None])[0]
    stepwise = post.updated(first, x[:len(first)]).updated(second, x[len(first):]).probabilities
    assert np.allclose(joint, stepwise, rtol=1e-9, atol=1e-12)
    assert np.allclose(post.updated(first, x[:len(first)]).probabilities, post.update_with(first, x[None, :len(first)])[0])


def test_the_effective_number_counts_a_uniform_support():
    for m in (1, 3, 12, 144):
        p = np.zeros(144)
        p[:m] = 1 / m
        assert abs(effective_number(p) - m) < 1e-9


def test_stability_over_resampled_fits_and_the_override_of_another_ranking_model():
    model = random_model(resamples=4, jitter=0.0)
    truth = (3, 100, 0, 130)
    fields = SPACE.groups["tier3"]
    updated = Posterior.uniform(model).updated(fields, model.mu(truth)[SPACE.field_indices(fields)])
    post = Posterior(model, updated.probabilities, updated.mean, updated.covariance, resampled=np.repeat(updated.probabilities[None], 4, axis=0))
    stability = post.stability()
    assert stability["top_component"] == "cooler" and stability["resamples"] == 4 and stability["top_share"] == 1.0
    assert stability["interval_low"] <= stability["top_marginal"] <= stability["interval_high"]
    assert Posterior.uniform(model).stability() is None
    weights = np.arange(1, 145, dtype=float)
    post.override(weights)
    assert np.allclose(post.probabilities, weights / weights.sum()) and post.resampled_probabilities is None
    assert post.candidates[post.order()[0]] == post.candidates[-1]


def test_a_belief_whose_masses_sum_to_zero_is_refused_by_name():
    """A support that holds no combination, or an override by a ranking model whose masses are all zero, leaves nothing to
    normalize, so the belief names it and holds no nan."""
    model = random_model()
    with pytest.raises(ValueError, match="sum to zero"):
        Posterior.uniform(model, support=[])
    with pytest.raises(ValueError, match="sum to zero"):
        Posterior.uniform(model).override(np.zeros(len(SPACE.combinations)))
