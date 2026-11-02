"""E1: the design, the weighted fit, the shrinkage, the recovery of a synthetic additive world and the fit on the facts
file. No test states a value computed from the rig data; the data dependent checks are identities and ranks."""
import json
from pathlib import Path

import numpy as np
import pytest

from envs.hydraulic_rig.classifiers import COMBOS
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, SENSORS, Cycle, build_blocks, default_facts_path, load_cycles
from servers.causal_diagnosis import effect_model as em
from servers.causal_diagnosis.fit import split
from servers.hydraulic_rig.names import HYD1_FIELDS

needs_facts = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")


def facts_name(summary_field):
    """PS1_Pressure_bar_100Hz_mean -> Pressure_PS1_mean, the naming of the facts file."""
    channel, statistic = em.channel_field(summary_field), em.statistic_of(summary_field)
    sensor = next(s for s in SENSORS if HYD1_FIELDS[s] == channel)
    return f"{channel.split('_')[1]}_{sensor}_{statistic}"


def synthetic_world(rng, cycles_per_block=6, noise=0.1):
    """Cycles of every combination drawn from an additive model with known main effects and independent noise."""
    beta = rng.normal(size=(1 + len(em.MAIN_TERMS), len(em.SUMMARY_FIELDS)))
    design = em.design_matrix(COMBOS, [])
    cycles, index = [], 0
    for row, combo in zip(design, COMBOS):
        for _ in range(cycles_per_block):
            x = row @ beta + noise * rng.normal(size=len(em.SUMMARY_FIELDS))
            cycles.append(Cycle(index, combo, True, {facts_name(f): float(v) for f, v in zip(em.SUMMARY_FIELDS, x)}))
            index += 1
    return beta, cycles, build_blocks(cycles)


def test_design_matrix_has_an_intercept_main_effects_and_the_selected_interactions():
    x = em.design_matrix(COMBOS, [])
    assert x.shape == (len(COMBOS), 1 + len(em.MAIN_TERMS)) and len(em.MAIN_TERMS) == sum(len(LEVELS[c]) - 1 for c in COMPONENTS)
    healthy = list(COMBOS).index(em.HEALTHY_COMBO)
    assert x[healthy].tolist() == [1.0] + [0.0] * len(em.MAIN_TERMS)
    one_fault = [h for h in COMBOS if em.n_faulty(h) == 1]
    assert all(x[list(COMBOS).index(h)].sum() == 2 for h in one_fault)
    full = em.design_matrix(COMBOS, list(em.PAIRS))
    assert full.shape[1] == 1 + len(em.MAIN_TERMS) + len(em.terms_of(em.PAIRS)) and len(em.terms_of(em.PAIRS)) == 37
    assert em.full_rank(full)


def test_weighted_fit_on_block_means_equals_least_squares_over_cycles():
    rng = np.random.default_rng(0)
    beta, cycles, blocks = synthetic_world(rng)
    pool = [h for h in COMBOS if em.n_faulty(h) != 2]
    data = em.PoolData(cycles, blocks, pool)
    fitted = em.weighted_fit(em.design_matrix(data.combos, []), data.means, data.counts)
    rows = np.concatenate([np.repeat(em.design_matrix([h], [])[0][None, :], len(x), axis=0) for h, x in zip(data.pool, data.arrays)])
    stacked = np.concatenate(data.arrays)
    direct = np.linalg.lstsq(rows, stacked, rcond=None)[0]
    assert np.allclose(fitted, direct)


def test_ledoit_wolf_shrinks_off_diagonals_only_and_keeps_the_weight_in_the_unit_interval():
    rng = np.random.default_rng(1)
    deviations = rng.normal(size=(50, 5)) @ np.array([[1, 0.5, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0.3], [0, 0, 0, 0, 1]])
    deviations -= deviations.mean(axis=0)
    shrunk, weight = em.ledoit_wolf_diagonal(deviations, len(deviations) - 1)
    sample = deviations.T @ deviations / (len(deviations) - 1)
    assert 0 <= weight <= 1 and np.allclose(np.diag(shrunk), np.diag(sample))
    off = ~np.eye(5, dtype=bool)
    assert np.allclose(shrunk[off], (1 - weight) * sample[off])


def test_a_synthetic_additive_world_is_recovered_from_the_pool_without_the_evaluation_combinations():
    rng = np.random.default_rng(2)
    beta, cycles, blocks = synthetic_world(rng)
    pool, evaluation = split()
    model = em.fit_effect_model(cycles, blocks, pool, interactions="none", resamples=5, seed=0)
    assert model.pairs == [] and np.allclose(model.beta, beta, atol=0.2)
    truth = em.design_matrix(evaluation, []) @ beta
    assert np.allclose(model.mu_rows(evaluation), truth, atol=0.2)
    assert np.allclose(model.sigma_within, 0.01 * np.eye(len(em.SUMMARY_FIELDS)), atol=0.01)
    assert len(model.resampled) == 5 and model.calibration["n_blocks"] == len(pool)
    assert abs(model.calibration["weights"].sum() - len(pool)) < 1e-9
    scores = model.anomaly_scores({f: float(v) for f, v in zip(em.SUMMARY_FIELDS, model.healthy_median)})
    assert set(scores) == set(em.SUMMARY_FIELDS) and all(v == 0 for v in scores.values())


def test_selection_refuses_pairs_a_training_fold_cannot_identify():
    rng = np.random.default_rng(3)
    beta, cycles, blocks = synthetic_world(rng)
    pool, _evaluation = split()
    data = em.PoolData(cycles, blocks, pool)
    fold_of = {h: i % 4 for i, h in enumerate(data.pool)}
    value = em.cross_fitted_log_likelihood(data, [], fold_of)
    assert np.isfinite(value)
    assert em.cross_fitted_log_likelihood(data, list(em.PAIRS), fold_of) == -np.inf
    selected, log = em.select_interactions(data, fold_of)
    assert log[0]["step"] == "main effects" and all(pair in em.PAIRS for pair in selected)


@needs_facts
def test_fit_on_the_facts_file_is_identified_deterministic_and_round_trips():
    cycles = load_cycles(default_facts_path())
    blocks = build_blocks(cycles)
    pool, evaluation = split()
    assert not (set(pool) & set(evaluation)) and len(pool) + len(evaluation) == len(COMBOS) and em.HEALTHY_COMBO in pool
    first = em.fit_effect_model(cycles, blocks, pool, resamples=3, seed=0)
    again = em.fit_effect_model(cycles, blocks, pool, resamples=3, seed=0)
    assert np.array_equal(first.beta, again.beta) and all(np.array_equal(a[0], b[0]) for a, b in zip(first.resampled, again.resampled))
    other = em.fit_effect_model(cycles, blocks, pool, resamples=3, seed=1)
    assert not all(np.array_equal(a[0], b[0]) for a, b in zip(first.resampled, other.resampled))
    assert em.full_rank(first.design_all) and np.allclose(first.mu_table, first.design_all @ first.beta)
    assert np.allclose(first.sigma_within, first.sigma_within.T) and np.all(np.linalg.eigvalsh(first.sigma_within) > 0)
    assert np.all(np.linalg.eigvalsh(first.sigma_block) > -1e-12)
    assert first.sigma_for(em.GROUP_FIELDS["tier1"]).shape == (len(em.GROUP_FIELDS["tier1"]),) * 2
    assert abs(first.calibration["weights"].sum() - len(pool)) < 1e-9
    back = em.EffectModel.from_json(json.loads(json.dumps(first.to_json())))
    assert np.allclose(back.mu_table, first.mu_table) and np.allclose(back.sigma_within, first.sigma_within) and back.pairs == first.pairs
    none = em.fit_effect_model(cycles, blocks, pool, interactions="none", resamples=0)
    assert none.pairs == [] and len(none.resampled) == 0


def test_component_terms_split_interactions_in_halves():
    pairs = [(0, 1)]
    beta = np.zeros((1 + len(em.MAIN_TERMS) + len(em.terms_of(pairs)), len(em.SUMMARY_FIELDS)))
    beta[1:1 + len(em.MAIN_TERMS)] = 1.0
    beta[1 + len(em.MAIN_TERMS):] = 2.0
    model = em.EffectModel(pairs, beta, np.eye(34), np.zeros((34, 34)), [], np.zeros(34), np.ones(34), {"n_blocks": 1, "scores": np.zeros((1, 34)), "weights": np.ones(1)}, {})
    combo = (3, 73, 0, 130)
    terms = model.component_terms(combo)
    assert np.allclose(terms[0], 2.0) and np.allclose(terms[1], 2.0) and np.allclose(terms[2], 0.0) and np.allclose(terms[3], 0.0)
    assert np.allclose(model.component_terms(em.HEALTHY_COMBO), 0.0)
