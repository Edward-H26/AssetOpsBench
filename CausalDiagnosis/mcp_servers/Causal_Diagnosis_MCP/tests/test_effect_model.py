"""E1 of version 7: the design over the space of the asset, the weighted fit, the shrinkage and
its sums over blocks, the deletion formulas against explicit refits without one and two blocks, every hypothesis of an episode
scored by an explicit refit that leaves out two blocks, the start's and its own or the second block, also on a grid with
unrecorded combinations, the second block as a hash of the start's levels among its neighbors, the resampled tables of E6 under the rule of the
means, no mean that reads its own block, no fit of the start that reads its second block, the blocks a caller leaves out with
the start's, the refusal of blocks that alone identify a term, the refusal of pairs a fold cannot identify, the halves of an
interaction in E4, and the fits on the facts file. No test states a value computed from the rig data; the data dependent
checks are identities and ranks."""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import folds_of, profile_view, recorded_blocks
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import SPACE

needs_facts = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
FOLD_OF = folds_of(profile_view(), em.FOLDS, em.FOLD_SEED)
PAIR = [(0, 3)]  # cooler and accumulator, an interaction whose cells every grid row of one component level crosses


def synthetic_blocks(rng, cycles_per_block=6, noise=0.1, pairs=(), uneven=False):
    """Blocks of every combination drawn from a model with known main effects and the given interactions, with independent
    noise, and with lengths that differ by block when uneven, so that the weights of the fit differ."""
    design = em.design_matrix(SPACE, SPACE.combinations, list(pairs))
    beta = rng.normal(size=(design.shape[1], len(SPACE.fields)))
    lengths = cycles_per_block + (rng.integers(0, 4, size=len(design)) if uneven else np.zeros(len(design), dtype=int))
    blocks = {h: row @ beta + noise * rng.normal(size=(n, len(SPACE.fields))) for row, h, n in zip(design, SPACE.combinations, lengths)}
    return beta, blocks


def refit(recorded, pairs, rows):
    """The coefficients of a fresh weighted fit on the recorded blocks of rows."""
    design = em.design_matrix(SPACE, recorded.combos, pairs)
    return em.weighted_fit(design[rows], recorded.means[rows], recorded.counts[rows])


def pooled_within(recorded, rows):
    """The within covariance of the blocks of rows from their cycles, each block centered on its own mean."""
    deviations = np.concatenate([recorded.arrays[i] - recorded.arrays[i].mean(axis=0) for i in rows])
    return em.ledoit_wolf_diagonal(deviations, len(deviations) - len(rows))[0]


def loo_misfits(recorded, pairs, rows):
    """The misfit of every block of rows against a fresh fit on the other blocks of rows."""
    design = em.design_matrix(SPACE, recorded.combos, pairs)
    return np.array([recorded.means[i] - design[i] @ refit(recorded, pairs, rows[rows != i]) for i in rows])


def test_design_matrix_has_an_intercept_main_effects_and_the_selected_interactions():
    x = em.design_matrix(SPACE, SPACE.combinations, [])
    assert x.shape == (144, 1 + len(SPACE.main_terms)) and len(SPACE.main_terms) == sum(len(levels) - 1 for levels in SPACE.levels)
    assert x[SPACE.combo_index[SPACE.healthy_combination]].tolist() == [1.0] + [0.0] * len(SPACE.main_terms)
    assert all(x[SPACE.combo_index[h]].sum() == 2 for h in SPACE.combinations if SPACE.n_faulty(h) == 1)
    full = em.design_matrix(SPACE, SPACE.combinations, list(SPACE.pairs))
    assert full.shape[1] == 1 + len(SPACE.main_terms) + len(em.terms_of(SPACE, SPACE.pairs)) and len(em.terms_of(SPACE, SPACE.pairs)) == 37
    assert em.model_terms.full_rank(full)


def test_weighted_fit_on_block_means_equals_least_squares_over_cycles():
    beta, blocks = synthetic_blocks(np.random.default_rng(0), uneven=True)
    data = em.PoolData(SPACE, blocks, [h for h in SPACE.combinations if SPACE.n_faulty(h) != 2])
    fitted = em.weighted_fit(em.design_matrix(SPACE, data.combos, []), data.means, data.counts)
    rows = np.concatenate([np.repeat(em.design_matrix(SPACE, [h], [])[0][None, :], len(x), axis=0) for h, x in zip(data.pool, data.arrays)])
    assert np.allclose(fitted, np.linalg.lstsq(rows, np.concatenate(data.arrays), rcond=None)[0])


def test_ledoit_wolf_shrinks_off_diagonals_only_and_its_sums_over_blocks_give_the_covariance_without_a_block():
    rng = np.random.default_rng(1)
    deviations = rng.normal(size=(50, 5)) @ np.array([[1, 0.5, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0.3], [0, 0, 0, 0, 1]])
    deviations -= deviations.mean(axis=0)
    shrunk, weight = em.ledoit_wolf_diagonal(deviations, len(deviations) - 1)
    sample = deviations.T @ deviations / (len(deviations) - 1)
    off = ~np.eye(5, dtype=bool)
    assert 0 <= weight <= 1 and np.allclose(np.diag(shrunk), np.diag(sample)) and np.allclose(shrunk[off], (1 - weight) * sample[off])
    _beta, blocks = synthetic_blocks(rng, uneven=True)
    recorded = em.RecordedBlocks(SPACE, blocks)
    rows = recorded.pool([SPACE.combinations[4]])
    cross, squares = recorded.cross[rows].sum(axis=0), recorded.squares[rows].sum(axis=0)
    cycles = recorded.counts[rows].sum()
    np.testing.assert_allclose(em.shrunk_toward_diagonal(cross, squares, cycles, cycles - len(rows))[0], pooled_within(recorded, rows), rtol=1e-9)
    i, rest = rows[10], rows[rows != rows[10]]
    without = em.shrunk_toward_diagonal(cross - recorded.cross[i], squares - recorded.squares[i], cycles - recorded.counts[i],
                                        cycles - recorded.counts[i] - len(rest))[0]
    np.testing.assert_allclose(without, pooled_within(recorded, rest), rtol=1e-9)


def test_the_deletion_formulas_equal_refits_without_one_and_two_blocks():
    _beta, blocks = synthetic_blocks(np.random.default_rng(2), uneven=True, pairs=PAIR)
    recorded = em.RecordedBlocks(SPACE, blocks)
    design = em.design_matrix(SPACE, recorded.combos, PAIR)
    rows = recorded.pool([SPACE.combinations[5]])
    fit = em.fit_pool(design, recorded.means, recorded.counts, rows)
    loo = em.leave_one_out(fit, recorded.means)
    for j in (0, 17, len(rows) - 1):
        others = np.delete(rows, j)
        np.testing.assert_allclose(loo[j], design[rows[j]] @ refit(recorded, PAIR, others), rtol=1e-8, atol=1e-9)
        misfits = em.leave_two_out_misfits(fit, recorded.counts, j)
        assert misfits.shape == (len(others), len(SPACE.fields))
        for k in (0, 40, len(others) - 1):
            i = others[k]
            np.testing.assert_allclose(misfits[k], recorded.means[i] - design[i] @ refit(recorded, PAIR, others[others != i]), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("pairs", [[], PAIR], ids=("main effects", "one pair"))
def test_every_hypothesis_is_scored_by_the_refit_without_the_start_s_block_and_its_own(pairs):
    """The mean of a hypothesis is the prediction of the fit on the blocks other than the start's and
    its own, its within covariance pools the same blocks, its block covariance is that of the leave one block out misfits among
    them, its E4 terms come from the same coefficients, the start is scored by the fit without its block and the second block,
    so that every hypothesis is scored by a fit on all recorded blocks but two, E3 takes the clean prediction of the healthy
    combination with its within sd, and the calibration scores every other block against the fit without it."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(3), uneven=True, pairs=pairs)
    recorded = em.RecordedBlocks(SPACE, blocks)
    start = SPACE.combinations[7]
    episode = em.fit_episode(recorded, pairs, start, resamples=0, seed=0)
    effect, grid = episode.effect, em.design_matrix(SPACE, SPACE.combinations, pairs)
    pool = recorded.pool([start])
    assert len(pool) == len(recorded.combos) - 1 and recorded.index[start] not in pool and episode.pool == tuple(recorded.combos[i] for i in pool)
    assert episode.second == em.recorded_blocks.second_block(start, episode.pool) and episode.second in episode.pool
    second = recorded.index[episode.second]
    for h in (start, episode.second, SPACE.healthy_combination, SPACE.combinations[30], SPACE.combinations[143]):
        rows = pool[pool != (second if h == start else recorded.index[h])]
        assert len(rows) == len(recorded.combos) - 2
        beta, index = refit(recorded, pairs, rows), SPACE.combo_index[h]
        np.testing.assert_allclose(effect.mu(h), grid[index] @ beta, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(effect.sigma_within[index], pooled_within(recorded, rows), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(effect.sigma_block[index], em.ledoit_wolf_diagonal(loo_misfits(recorded, pairs, rows), len(rows))[0], rtol=1e-7, atol=1e-12)
        explicit = em.EffectModel(SPACE, pairs, effect.mu_table, effect.sigma_within, effect.sigma_block, effect.healthy_reference, effect.healthy_scale,
                                  coefficients=lambda _combo, beta=beta: beta)
        np.testing.assert_allclose(effect.component_terms(h), explicit.component_terms(h), rtol=1e-8, atol=1e-9)
    healthy = SPACE.combo_index[SPACE.healthy_combination]
    assert np.array_equal(effect.healthy_reference, effect.mu_table[healthy])
    assert np.array_equal(effect.healthy_scale, np.sqrt(np.diag(effect.sigma_within[healthy])))
    sd = np.sqrt(np.diag(pooled_within(recorded, pool)))
    np.testing.assert_allclose(episode.sigma, sd, rtol=1e-9)
    loo = recorded.means[pool] - loo_misfits(recorded, pairs, pool)
    np.testing.assert_allclose(episode.scores, np.concatenate([np.abs(recorded.arrays[i] - m) / sd for i, m in zip(pool, loo)]), rtol=1e-7, atol=1e-9)
    assert episode.n_blocks == len(pool) and abs(episode.weights.sum() - len(pool)) < 1e-9


def test_no_mean_reads_its_own_block():
    """A shift of a combination's own block leaves its mean as it was and moves the means of the others."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(6))
    start = SPACE.combinations[11]
    base = em.fit_episode(em.RecordedBlocks(SPACE, blocks), [], start, resamples=0, seed=0).effect
    for h in (SPACE.healthy_combination, (100, 80, 0, 90), (3, 73, 2, 90)):
        shifted = em.fit_episode(em.RecordedBlocks(SPACE, {**blocks, h: blocks[h] + 5.0}), [], start, resamples=0, seed=0).effect
        assert np.allclose(shifted.mu(h), base.mu(h), rtol=0, atol=1e-9)
        others = [g for g in SPACE.combinations if g not in (h, start)]
        assert not np.allclose(shifted.mu_rows(others), base.mu_rows(others))


def test_a_resampled_table_predicts_every_row_without_its_own_draws_and_never_draws_the_start():
    _beta, blocks = synthetic_blocks(np.random.default_rng(4), uneven=True, pairs=PAIR)
    recorded = em.RecordedBlocks(SPACE, blocks)
    start = SPACE.combinations[9]
    episode = em.fit_episode(recorded, PAIR, start, resamples=3, seed=5)
    effect = episode.effect
    tables = effect.tables(3)
    assert tables.shape == (4, len(SPACE.combinations), len(SPACE.fields)) and np.array_equal(tables[0], effect.mu_table)
    again = em.fit_episode(recorded, PAIR, start, resamples=3, seed=5).effect
    assert np.array_equal(again.tables(2), tables[:3])  # one order in every process, whatever the number asked for
    pool, design, grid = recorded.pool([start]), em.design_matrix(SPACE, recorded.combos, PAIR), em.design_matrix(SPACE, SPACE.combinations, PAIR)
    multiplicity = effect._resampler._draw(1)[0]
    assert len(multiplicity) == len(pool)  # a draw picks among the pool's blocks alone, so the start's block is never drawn
    weights = recorded.counts[pool] * multiplicity
    drawn = weights > 0
    second = episode.pool.index(episode.second)
    without_second = drawn.copy()
    without_second[second] = False
    beta = em.weighted_fit(design[pool][without_second], recorded.means[pool][without_second], weights[without_second])
    np.testing.assert_allclose(tables[2][SPACE.combo_index[start]], grid[SPACE.combo_index[start]] @ beta, rtol=1e-8, atol=1e-9)
    for k in np.flatnonzero(drawn)[:3]:
        kept = drawn.copy()
        kept[k] = False
        beta_k = em.weighted_fit(design[pool][kept], recorded.means[pool][kept], weights[kept])
        h = recorded.combos[pool[k]]
        np.testing.assert_allclose(tables[2][SPACE.combo_index[h]], grid[SPACE.combo_index[h]] @ beta_k, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("kept, message", [(1, "a recorded block alone identifies"), (2, "two recorded blocks alone identify")])
def test_blocks_that_alone_identify_a_term_are_refused(kept, message):
    """With one recorded block at a level, no fit without it exists; with two, the fit of one of them without the other does not."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(7))
    at_level = [h for h in SPACE.combinations if h[0] == SPACE.levels[0][-1]]
    recorded = em.RecordedBlocks(SPACE, {h: x for h, x in blocks.items() if h not in at_level[kept:]})
    with pytest.raises(ValueError, match=message):
        em.fit_episode(recorded, [], SPACE.healthy_combination, resamples=0, seed=0)


def test_selection_refuses_pairs_a_training_fold_cannot_identify():
    _beta, blocks = synthetic_blocks(np.random.default_rng(3))
    data = em.PoolData(SPACE, blocks, [h for h in SPACE.combinations if SPACE.n_faulty(h) != 2])
    fold_of = {h: i % 4 for i, h in enumerate(data.pool)}
    assert np.isfinite(em.cross_fitted_log_likelihood(SPACE, data, [], fold_of))
    assert em.cross_fitted_log_likelihood(SPACE, data, list(SPACE.pairs), fold_of) == -np.inf
    selected, log = em.select_interactions(SPACE, data, fold_of)
    assert log[0]["step"] == "main effects" and all(pair in SPACE.pairs for pair in selected)


@needs_facts
def test_the_fits_of_an_episode_on_the_facts_file_are_identified_deterministic_and_positive_definite():
    view = profile_view()
    blocks, _records = recorded_blocks(view)
    assert set(blocks) == set(view.combinations)
    recorded, start = em.RecordedBlocks(SPACE, blocks), (100, 80, 0, 90)
    pairs, log = em.select_pool_pairs(SPACE, blocks, [start], FOLD_OF)
    assert log[0]["step"] == "main effects" and all(pair in SPACE.pairs for pair in pairs)
    first, again = (em.fit_episode(recorded, pairs, start, resamples=3, seed=0) for _ in range(2))
    assert np.array_equal(first.effect.tables(3), again.effect.tables(3)) and np.array_equal(first.scores, again.scores)
    other = em.fit_episode(recorded, pairs, start, resamples=3, seed=1)
    assert np.array_equal(other.effect.mu_table, first.effect.mu_table) and not np.array_equal(other.effect.tables(3)[1:], first.effect.tables(3)[1:])
    for within, block in zip(first.effect.sigma_within, first.effect.sigma_block):
        assert np.allclose(within, within.T) and np.all(np.linalg.eigvalsh(within) > 0) and np.all(np.linalg.eigvalsh(block) > -1e-12)
    assert first.n_blocks == len(blocks) - 1 and abs(first.weights.sum() - first.n_blocks) < 1e-9
    none, none_log = em.select_pool_pairs(SPACE, blocks, [start], FOLD_OF, interactions="none")
    assert none == [] and [step["step"] for step in none_log] == ["main effects"]


def test_component_terms_split_interactions_in_halves():
    pairs = [(0, 1)]
    terms = 1 + len(SPACE.main_terms) + len(em.terms_of(SPACE, pairs))
    beta = np.zeros((terms, len(SPACE.fields)))
    beta[1:1 + len(SPACE.main_terms)] = 1.0
    beta[1 + len(SPACE.main_terms):] = 2.0
    d = len(SPACE.fields)
    model = em.EffectModel(SPACE, pairs, np.zeros((144, d)), np.eye(d), np.zeros((d, d)), np.zeros(d), np.ones(d), coefficients=lambda _combo: beta)
    terms_of_combo = model.component_terms((3, 73, 0, 130))
    assert np.allclose(terms_of_combo[0], 2.0) and np.allclose(terms_of_combo[1], 2.0) and np.allclose(terms_of_combo[2:], 0.0)
    assert np.allclose(model.component_terms(SPACE.healthy_combination), 0.0)


UNRECORDED = (SPACE.combinations[5], SPACE.combinations[77], SPACE.combinations[140])  # a grid whose recordings miss three cells


def test_every_hypothesis_is_scored_by_a_fit_that_leaves_out_two_blocks_on_a_grid_with_unrecorded_combinations():
    """For any asset: a hypothesis with a block in the pool is scored without the start's block and its own,
    and every other hypothesis, the start and the combinations that have no recording, without the start's block and the
    second block, so every mean, within covariance, block covariance and E4 term comes from a fit on all recorded blocks but
    two."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(8), uneven=True, pairs=PAIR)
    recorded = em.RecordedBlocks(SPACE, {h: x for h, x in blocks.items() if h not in UNRECORDED})
    start = SPACE.combinations[60]
    episode = em.fit_episode(recorded, PAIR, start, resamples=0, seed=0)
    effect = episode.effect
    grid, pool = em.design_matrix(SPACE, SPACE.combinations, PAIR), recorded.pool([start])
    second = recorded.index[em.recorded_blocks.second_block(start, [recorded.combos[i] for i in pool])]
    assert episode.second == recorded.combos[second] and episode.second not in (start, *UNRECORDED)
    for index, h in enumerate(SPACE.combinations):
        own = recorded.index.get(h)
        rows = pool[pool != (own if own is not None and own in pool else second)]
        assert len(rows) == len(recorded.combos) - 2
        beta = refit(recorded, PAIR, rows)
        np.testing.assert_allclose(effect.mu(h), grid[index] @ beta, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(effect.sigma_within[index], pooled_within(recorded, rows), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(effect.sigma_block[index], em.ledoit_wolf_diagonal(loo_misfits(recorded, PAIR, rows), len(rows))[0], rtol=1e-7, atol=1e-12)
        explicit = em.EffectModel(SPACE, PAIR, effect.mu_table, effect.sigma_within, effect.sigma_block, effect.healthy_reference, effect.healthy_scale,
                                  coefficients=lambda _combo, beta=beta: beta)
        np.testing.assert_allclose(effect.component_terms(h), explicit.component_terms(h), rtol=1e-7, atol=1e-9)


def test_a_resampled_row_on_a_grid_with_unrecorded_combinations_leaves_out_its_own_draws_or_the_second_block_s():
    """E6 on a grid that misses recordings: the row of a pooled combination is the fit on the drawn
    blocks without its own copies, at that combination's place in the grid, and every other row the fit without the copies of
    the second block."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(9), uneven=True, pairs=PAIR)
    recorded = em.RecordedBlocks(SPACE, {h: x for h, x in blocks.items() if h not in UNRECORDED})
    start = SPACE.combinations[60]
    effect = em.fit_episode(recorded, PAIR, start, resamples=2, seed=3).effect
    table = effect.tables(2)[2]
    pool = recorded.pool([start])
    combos = [recorded.combos[i] for i in pool]
    design, grid = em.design_matrix(SPACE, recorded.combos, PAIR)[pool], em.design_matrix(SPACE, SPACE.combinations, PAIR)
    weights = recorded.counts[pool] * effect._resampler._draw(1)[0]
    second = combos.index(em.recorded_blocks.second_block(start, combos))
    for index, h in enumerate(SPACE.combinations):
        kept = weights > 0
        kept[combos.index(h) if h in combos else second] = False
        beta = em.weighted_fit(design[kept], recorded.means[pool][kept], weights[kept])
        np.testing.assert_allclose(table[index], grid[index] @ beta, rtol=1e-7, atol=1e-8)


def test_the_second_block_is_keyed_on_the_start_s_levels_and_not_on_an_order_or_a_process():
    """The second block is a stable hash of LEAVE_OUT_SEED, the start's levels and the candidate's, so it
    is one of the candidates, never the start, the same for the candidates in any order and in a fresh process with another
    hash seed, and it differs between starts."""
    candidates = list(SPACE.combinations[1:])
    start = SPACE.combinations[0]
    chosen = em.recorded_blocks.second_block(start, candidates)
    assert chosen in candidates and chosen != start and em.recorded_blocks.second_block(start, candidates[::-1]) == chosen
    code = ("from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em; from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import SPACE; "
            "print(em.combo_key(em.second_block(SPACE.combinations[0], list(SPACE.combinations[1:]))))")
    fresh = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": "7"})
    assert fresh.stdout.strip() == em.combo_key(chosen)
    seconds = {em.recorded_blocks.second_block(h, [g for g in SPACE.combinations if g != h]) for h in SPACE.combinations}
    assert len(seconds) > len(SPACE.combinations) // 4


def test_the_start_and_its_second_block_share_one_fit_and_no_other_hypothesis_does():
    """On a grid where every combination is recorded: the start and its second block are scored by the one
    fit without those two blocks, so their coefficients and their covariances are equal, and every other hypothesis is scored
    by a fit of its own."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(10), uneven=True)
    recorded = em.RecordedBlocks(SPACE, blocks)
    start = SPACE.combinations[33]
    episode = em.fit_episode(recorded, [], start, resamples=0, seed=0)
    effect, index = episode.effect, SPACE.combo_index
    assert episode.second != start and episode.second == em.recorded_blocks.second_block(start, [h for h in recorded.combos if h != start])
    sharing = [h for h in SPACE.combinations if np.array_equal(effect.sigma_within[index[h]], effect.sigma_within[index[start]])]
    assert sharing == sorted((start, episode.second), key=index.get)
    assert np.array_equal(effect.sigma_block[index[start]], effect.sigma_block[index[episode.second]])
    assert np.array_equal(effect._coefficients(start), effect._coefficients(episode.second))
    assert not any(np.array_equal(effect._coefficients(h), effect._coefficients(start)) for h in SPACE.combinations if h not in sharing)


def test_no_fit_of_the_start_reads_its_second_block():
    """Moving the recordings of the second block moves no part of the start's row, its mean, its covariances, its E4 terms and
    its resampled means, and moves the rows of the other hypotheses, which read that block."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(11), uneven=True, pairs=PAIR)
    start = SPACE.combinations[50]
    base = em.fit_episode(em.RecordedBlocks(SPACE, blocks), PAIR, start, resamples=2, seed=1)
    second, index = base.second, SPACE.combo_index[start]
    moved = em.fit_episode(em.RecordedBlocks(SPACE, {**blocks, second: 3.0 * blocks[second] + 5.0}), PAIR, start, resamples=2, seed=1)
    assert moved.second == second
    for name in ("mu_table", "sigma_within", "sigma_block"):
        np.testing.assert_allclose(getattr(moved.effect, name)[index], getattr(base.effect, name)[index], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(moved.effect.component_terms(start), base.effect.component_terms(start), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(moved.effect.tables(2)[:, index], base.effect.tables(2)[:, index], rtol=1e-7, atol=1e-9)
    others = [h for h in SPACE.combinations if h not in (start, second)]
    assert not np.allclose(moved.effect.mu_rows(others), base.effect.mu_rows(others))
    assert not np.allclose(moved.effect.sigma_within[[SPACE.combo_index[h] for h in others]], base.effect.sigma_within[[SPACE.combo_index[h] for h in others]])


def test_blocks_left_out_with_the_start_leave_every_fit_and_the_default_changes_nothing():
    """The keyword of fit_episode that leaves further blocks out with the start's: no fit reads them, so moving their recordings
    moves nothing, the second block is drawn among the rest, and every hypothesis is still scored by a fit that leaves out
    the same number of blocks. Without the keyword every part of the fits equals the call that omits it."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(12), uneven=True, pairs=PAIR)
    recorded = em.RecordedBlocks(SPACE, blocks)
    start, extra = SPACE.combinations[20], (SPACE.combinations[21], SPACE.combinations[99])
    parts = ("mu_table", "sigma_within", "sigma_block")

    def same(a, b):
        return (all(np.array_equal(getattr(a.effect, name), getattr(b.effect, name)) for name in parts) and a.second == b.second
                and np.array_equal(a.effect.tables(2), b.effect.tables(2)) and np.array_equal(a.scores, b.scores) and a.pool == b.pool)

    assert same(em.fit_episode(recorded, PAIR, start, 2, 4), em.fit_episode(recorded, PAIR, start, 2, 4, also_left_out=()))
    episode = em.fit_episode(recorded, PAIR, start, 2, 4, also_left_out=extra)
    assert episode.n_blocks == len(recorded.combos) - 3 and not set(episode.pool) & {start, *extra} and episode.second in episode.pool
    assert not same(episode, em.fit_episode(recorded, PAIR, start, 2, 4))
    moved = em.RecordedBlocks(SPACE, {**blocks, **{h: 2.0 * blocks[h] - 1.0 for h in extra}})
    assert same(episode, em.fit_episode(moved, PAIR, start, 2, 4, also_left_out=extra))
    grid, pool = em.design_matrix(SPACE, SPACE.combinations, PAIR), recorded.pool([start, *extra])
    second = recorded.index[episode.second]
    for h in (start, extra[0], extra[1], episode.second, SPACE.combinations[143]):
        own = recorded.index[h]
        rows = pool[pool != (own if own in pool else second)]
        assert len(rows) == len(recorded.combos) - 4
        np.testing.assert_allclose(episode.effect.mu(h), grid[SPACE.combo_index[h]] @ refit(recorded, PAIR, rows), rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(episode.effect.sigma_within[SPACE.combo_index[h]], pooled_within(recorded, rows), rtol=1e-8, atol=1e-12)


def test_a_pool_that_does_not_identify_a_term_is_refused_by_name():
    """A level that only the start's block holds leaves a term of the model that no block of the pool identifies, so no fit
    of the episode exists, and the refusal names the cause where the inverse of the fit would meet a singular matrix."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(7))
    at_level = [h for h in SPACE.combinations if h[0] == SPACE.levels[0][-1]]
    recorded = em.RecordedBlocks(SPACE, {h: x for h, x in blocks.items() if h not in at_level[1:]})
    with pytest.raises(ValueError, match="the recorded blocks of the fit do not identify every term of the model") as refused:
        em.fit_episode(recorded, [], at_level[0], resamples=0, seed=0)
    assert not isinstance(refused.value, np.linalg.LinAlgError)


def test_a_summary_that_does_not_vary_within_the_blocks_is_refused_by_name():
    """A summary that takes one value within every recorded block has no within variance, so no density of a reading of it
    exists. The selection of the pairs and the fits of an episode refuse such recordings by the name of that summary and of
    no other, under every registered estimator. The refusal reads the recorded values: a summary that varies in one block
    alone is accepted on the blocks with that block and refused on the blocks without it, so the fit of that block's own
    hypothesis refuses it."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(16), uneven=True)
    field = SPACE.fields[5]
    k = SPACE.field_index[field]
    flat = {h: x.copy() for h, x in blocks.items()}
    for h, x in flat.items():
        x[:, k] = 23.7 + h[0] / 7  # one value within each block, whose mean over the block a float need not hold exactly
    named = re.escape(f"the summaries {field} take one value")
    start = SPACE.combinations[3]
    with pytest.raises(ValueError, match=named) as refused:
        em.select_pool_pairs(SPACE, flat, [start], FOLD_OF)
    assert not isinstance(refused.value, np.linalg.LinAlgError)
    recorded = em.RecordedBlocks(SPACE, flat, window=3)
    for estimator in em.COVARIANCE_ESTIMATORS:
        with pytest.raises(ValueError, match=named):
            em.fit_episode(recorded, [], start, 0, 0, covariance=estimator)
    varied = SPACE.combinations[40]
    recorded = em.RecordedBlocks(SPACE, {**flat, varied: blocks[varied]}, window=3)
    em.refuse_constant_summaries(SPACE, recorded.sums(np.arange(len(recorded.combos))).varying)
    with pytest.raises(ValueError, match="^" + named):
        em.refuse_constant_summaries(SPACE, recorded.sums(recorded.pool([varied])).varying)
    with pytest.raises(ValueError, match=named):
        em.fit_episode(recorded, [], start, 0, 0)


def test_the_second_block_is_a_neighbor_of_the_start_when_one_is_recorded():
    """The second block that a fit leaves out besides the start's is drawn by the hash of the start's levels and its own among
    the recorded combinations one component away from the start, and among the whole pool by the same hash when the pool holds
    no neighbor. The draw does not move with the order of the pool, and the fits of an episode take it."""
    import hashlib

    _beta, blocks = synthetic_blocks(np.random.default_rng(8))
    recorded = list(blocks)

    def away(h, start):
        return sum(a != b for a, b in zip(h, start))

    def by_hash(start, pool):
        return min(pool, key=lambda h: hashlib.sha256(f"{em.LEAVE_OUT_SEED}|{em.combo_key(start)}|{em.combo_key(h)}".encode()).digest())

    for start in recorded:
        pool = [h for h in recorded if h != start]
        second = em.recorded_blocks.second_block(start, pool)
        assert second in pool and away(second, start) == 1
        assert second == by_hash(start, [h for h in pool if away(h, start) == 1]) == em.recorded_blocks.second_block(start, pool[::-1])
        far = [h for h in pool if away(h, start) != 1]
        assert em.recorded_blocks.second_block(start, far) == by_hash(start, far)
    for start in (SPACE.healthy_combination, SPACE.combinations[7], SPACE.combinations[143]):
        assert away(em.fit_episode(em.RecordedBlocks(SPACE, blocks), [], start, resamples=0, seed=0).second, start) == 1


def test_a_resample_that_does_not_identify_the_model_is_drawn_again_with_no_count(monkeypatch):
    """A block resample that leaves the model unidentified is drawn again with
    no count of attempts, past the former bound of 100, and the redraws are counted; the draws that do identify the model and
    their order from the seed are those of a resampler whose first draws all identify it."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(4), uneven=True, pairs=PAIR)
    recorded = em.RecordedBlocks(SPACE, blocks)
    start = SPACE.combinations[9]
    plain = em.fit_episode(recorded, PAIR, start, resamples=2, seed=5).effect
    refused = {"left": 150}
    full_rank = em.model_terms.full_rank

    def first_draws_refused(x):
        if refused["left"] > 0:
            refused["left"] -= 1
            return False
        return full_rank(x)

    late = em.fit_episode(recorded, PAIR, start, resamples=2, seed=5).effect  # fitted first, its draws are made on demand
    monkeypatch.setattr(em.model_terms, "full_rank", first_draws_refused)
    late.tables(2)
    monkeypatch.undo()
    assert not hasattr(em, "RESAMPLE_ATTEMPTS") and late._resampler.redraws >= 150 and refused["left"] == 0
    assert plain.tables(2).shape == late.tables(2).shape


def test_a_pool_where_one_block_alone_supports_a_term_is_refused_before_any_redraw():
    """The redraws end with probability 1 only if the draw that takes every block once is accepted. A planted pool
    whose third block alone supports the second term (leverage 1) would be drawn again forever, so the resampler refuses it at
    construction with ValueError, as the old bound of 100 attempts did; a pool where two blocks share that term is accepted."""
    alone = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    means, counts = np.zeros((3, 1)), np.ones(3)
    with pytest.raises(ValueError, match="no resample would ever be accepted"):
        em.Resampler(alone, alone, means, counts, np.arange(3), np.arange(3), 0, 1, 0)
    shared = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    resampler = em.Resampler(shared, shared, np.zeros((4, 1)), np.ones(4), np.arange(4), np.arange(4), 0, 1, 0)
    assert resampler.table(0).shape == (4, 1)
