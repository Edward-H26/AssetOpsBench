"""The form of the model of the CausalDiagnosis MCP, chosen before any episode:
the exact log density of a block's rows with one shared offset, the registered covariance estimators of one cycle from rows and
from the stored sums of the recorded blocks, a block leaving those sums by a subtraction, the fits of an episode under a named
estimator, a form that reads the blocks it is given and no other, the estimator of the largest held out within block density of
window means, the fallback when the held out windows are too few, an estimator that cannot be evaluated, drift within blocks,
the window level report of the block effects, the test of whether development blocks identify every pair, and the rule that chooses a
part of the form on the development blocks when they identify it and once per start when they do not. Every recording here is
synthetic. The test on the rig reads its grid, its folds and its split, and no recording."""
import math

import numpy as np
import pytest

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import development_blocks, profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_effect_model import FOLD_OF, PAIR, loo_misfits, synthetic_blocks
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import SPACE

DEVELOPMENT = [h for h in SPACE.combinations if SPACE.n_faulty(h) not in (1, 2)]  # the healthy block and the blocks of three or four faults
TEST = [h for h in SPACE.combinations if h not in DEVELOPMENT]
WINDOW = 3
# a small grid for the rule of the forms, which fits once per start
SMALL_FIELDS = ("u_mean", "u_std", "v_mean", "v_std")
SMALL = em.EffectSpace(("a", "b", "c"), ((0, 1, 2), (0, 1, 2), (0, 1, 2)), (0, 0, 0), SMALL_FIELDS, {"g1": SMALL_FIELDS[:2], "g2": SMALL_FIELDS[2:]})
# three folds, each without one block of every line of the grid: a training fold identifies the main effects and any one pair,
# and not every pair at once, since it holds fewer blocks than the model with every pair has terms
LATIN_FOLDS = {h: sum(h) % 3 for h in SMALL.combinations}
# nine folds of three blocks that differ in every level, so a training fold with one more block left out still holds two blocks
# of every line and identifies every pair
DIAGONAL_FOLDS = {h: 3 * ((h[0] + h[1]) % 3) + (h[0] + h[2]) % 3 for h in SMALL.combinations}
MULTI = [h for h in SMALL.combinations if SMALL.n_faulty(h) != 1]  # no block of one fault, so these blocks do not identify every pair
OUTSIDE = (SMALL.combinations[1], SMALL.combinations[9])  # two blocks of one fault, starts outside MULTI
WIDE = em.EffectSpace(("a", "b", "c", "d"), ((0, 1, 2),) * 4, (0, 0, 0, 0), SMALL_FIELDS, {"g1": SMALL_FIELDS[:2], "g2": SMALL_FIELDS[2:]})
COVARIANCE_KEYS = ("covariance", "fallback", "held_out_window_dof", "forms", "block_effects")


def spd(rng, d, scale=1.0):
    a = rng.normal(size=(d, d))
    return scale * (a @ a.T / d + np.eye(d))


def small_blocks(seed, pairs=((0, 1),)):
    """Blocks of six to eight cycles for every combination of the small grid, drawn from a model with the given interactions."""
    rng = np.random.default_rng(seed)
    design = em.design_matrix(SMALL, SMALL.combinations, list(pairs))
    beta = rng.normal(size=(design.shape[1], len(SMALL.fields)))
    return {h: row @ beta + 0.1 * rng.normal(size=(6 + i % 3, len(SMALL.fields))) for i, (row, h) in enumerate(zip(design, SMALL.combinations))}


def moved(blocks, *combinations):
    """The recordings with the blocks of the given combinations given other values."""
    return {**blocks, **{h: 3.0 * blocks[h] + 1.0 for h in combinations}}


def rows_covariance(arrays, estimator):
    """The covariance of one cycle of a set of blocks from their rows, by the definition of each registered estimator."""
    deviations = np.concatenate([x - x.mean(axis=0) for x in arrays])
    dof = len(deviations) - len(arrays)
    if estimator == "ledoit_wolf_diagonal":
        return em.ledoit_wolf_diagonal(deviations, dof)[0]
    if estimator == "diagonal":
        return np.diag((deviations ** 2).sum(axis=0) / dof)
    means = [em.window_means(x, WINDOW) for x in arrays]
    of_windows = np.concatenate([w - w.mean(axis=0) for w in means])
    return WINDOW * em.ledoit_wolf_diagonal(of_windows, sum(len(w) - 1 for w in means))[0]


def window_criterion(blocks, pool, fold_of, estimator):
    """The criterion that chooses the covariance estimator, from its definition: over the folds and the held out blocks, the
    log density, up to a constant, of the scatter of a block's window means under the covariance of one cycle of the training
    blocks over the window."""
    total = 0.0
    for fold in sorted({fold_of[h] for h in pool}):
        scale = rows_covariance([blocks[h] for h in pool if fold_of[h] != fold], estimator) / WINDOW
        logdet = np.linalg.slogdet(scale)[1]
        for h in (h for h in pool if fold_of[h] == fold):
            means = em.window_means(blocks[h], WINDOW)
            scatter = (means - means.mean(axis=0)).T @ (means - means.mean(axis=0))
            total += -0.5 * ((len(means) - 1) * logdet + np.trace(np.linalg.solve(scale, scatter)))
    return total


# ----- the densities and the estimators -----

def test_the_block_log_density_is_the_normal_density_of_the_stacked_rows():
    rng = np.random.default_rng(0)
    n, d = 5, 3
    within, block = spd(rng, d), spd(rng, d, 0.5)
    residuals = rng.normal(size=(n, d))
    stacked = np.kron(np.eye(n), within) + np.kron(np.ones((n, n)), block)
    vec = residuals.reshape(-1)
    sign, logdet = np.linalg.slogdet(stacked)
    expected = -0.5 * (n * d * math.log(2 * math.pi) + logdet + vec @ np.linalg.solve(stacked, vec))
    assert sign > 0 and abs(em.block_log_density(residuals, within, block) - expected) < 1e-9
    independent = em.gaussian_log_density(residuals, np.linalg.cholesky(within)).sum()
    assert abs(em.block_log_density(residuals, within, np.zeros((d, d))) - independent) < 1e-9


def test_the_window_scatter_density_differs_from_the_density_of_the_window_contrasts_by_a_term_no_scale_changes():
    rng = np.random.default_rng(10)
    d, m = 3, 4
    means = rng.normal(size=(m, d))
    scatter = (means - means.mean(axis=0)).T @ (means - means.mean(axis=0))
    basis = np.linalg.qr(np.column_stack([np.ones(m), np.eye(m)[:, : m - 1]]))[0][:, 1:]  # orthonormal contrasts of m rows
    contrasts = basis.T @ means
    gaps = [em.window_scatter_log_density(scatter, m, scale) - em.gaussian_log_density(contrasts, np.linalg.cholesky(scale)).sum()
            for scale in (spd(rng, d), spd(rng, d, 2.0))]
    assert abs(gaps[0] - gaps[1]) < 1e-9 and abs(gaps[0] - 0.5 * (m - 1) * d * math.log(2 * math.pi)) < 1e-9


@pytest.mark.parametrize("estimator", em.COVARIANCE_ESTIMATORS)
def test_every_registered_estimator_agrees_from_rows_and_from_the_stored_sums_of_the_recorded_blocks(estimator):
    _beta, blocks = synthetic_blocks(np.random.default_rng(1), uneven=True)
    recorded = em.RecordedBlocks(SPACE, blocks, window=WINDOW)
    rows = np.arange(0, len(recorded.combos), 2)
    stored = em.one_cycle_covariance(recorded.sums(rows), estimator, WINDOW)
    np.testing.assert_allclose(stored, rows_covariance([recorded.arrays[i] for i in rows], estimator), rtol=1e-9, atol=1e-12)
    assert np.allclose(stored, stored.T) and np.all(np.linalg.eigvalsh(stored) > 0)
    assert (np.count_nonzero(stored - np.diag(np.diag(stored))) == 0) == (estimator == "diagonal")
    assert em.COVARIANCE_ESTIMATORS[0] == em.DEFAULT_COVARIANCE == "ledoit_wolf_diagonal"
    assert np.array_equal(em.window_means(np.arange(14.0).reshape(7, 2), 3), [[2.0, 3.0], [8.0, 9.0]])  # the seventh cycle is left out
    with pytest.raises(ValueError, match="covariance estimator must be one of"):
        em.one_cycle_covariance(recorded.sums(rows), "sample", WINDOW)
    one_window = em.RecordedBlocks(SPACE, {h: x[:WINDOW] for h, x in blocks.items()}, window=WINDOW)
    with pytest.raises(ValueError, match="needs a block of at least two windows"):
        em.one_cycle_covariance(one_window.sums(rows), "window_means", WINDOW)
    with pytest.raises(ValueError, match="needs the width of the window"):
        em.one_cycle_covariance(recorded.sums(rows), "window_means")


@pytest.mark.parametrize("estimator", em.COVARIANCE_ESTIMATORS)
def test_removing_a_block_s_sums_equals_the_covariance_of_the_other_blocks(estimator):
    """The deletion that the fits of an episode use: the sums of a pool less the sums of one block give the covariance of the
    other blocks, under every registered estimator."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(13), uneven=True)
    recorded = em.RecordedBlocks(SPACE, blocks, window=WINDOW)
    rows = recorded.pool([SPACE.combinations[4]])
    pool = recorded.sums(rows)
    assert (pool.blocks, pool.cycles) == (len(rows), recorded.counts[rows].sum())
    for j in (0, 60, len(rows) - 1):
        rest, without = np.delete(rows, j), pool.minus(recorded.sums(rows[j]))
        assert (without.blocks, without.cycles) == (len(rest), recorded.counts[rest].sum())
        np.testing.assert_allclose(em.one_cycle_covariance(without, estimator, WINDOW),
                                   rows_covariance([recorded.arrays[i] for i in rest], estimator), rtol=1e-8, atol=1e-12)


def test_the_fits_of_an_episode_take_the_named_estimator_and_keep_their_means():
    """The estimator of the start's form sets the within covariance of every hypothesis, over the blocks its fit reads, and the
    scale of the conformal scores. The block covariance takes the same estimator, or the first registered one under the window
    estimator, since the mean of a block has no windows. The means do not move with the estimator."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(2), uneven=True, pairs=PAIR)
    recorded, start = em.RecordedBlocks(SPACE, blocks, window=WINDOW), SPACE.combinations[7]
    default = em.fit_episode(recorded, PAIR, start, 0, 0)
    pool = recorded.pool([start])
    second = recorded.index[default.second]
    for estimator in em.COVARIANCE_ESTIMATORS:
        episode = em.fit_episode(recorded, PAIR, start, 0, 0, covariance=estimator)
        assert np.array_equal(episode.effect.mu_table, default.effect.mu_table) and episode.second == default.second
        for h in (SPACE.combinations[30], start):
            rows = pool[pool != (second if h == start else recorded.index[h])]
            index, misfits = SPACE.combo_index[h], loo_misfits(recorded, PAIR, rows)
            np.testing.assert_allclose(episode.effect.sigma_within[index], rows_covariance([recorded.arrays[i] for i in rows], estimator), rtol=1e-8, atol=1e-12)
            block = np.diag((misfits ** 2).sum(axis=0) / len(rows)) if estimator == "diagonal" else em.ledoit_wolf_diagonal(misfits, len(rows))[0]
            np.testing.assert_allclose(episode.effect.sigma_block[index], block, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(episode.sigma, np.sqrt(np.diag(rows_covariance([recorded.arrays[i] for i in pool], estimator))), rtol=1e-9)
    assert np.array_equal(default.effect.sigma_within, em.fit_episode(recorded, PAIR, start, 0, 0, covariance=em.DEFAULT_COVARIANCE).effect.sigma_within)
    with pytest.raises(ValueError, match="needs the width of the window"):
        em.fit_episode(em.RecordedBlocks(SPACE, blocks), PAIR, start, 0, 0, covariance="window_means")


# ----- the form that a set of blocks chooses -----

def test_a_form_reads_the_blocks_it_is_given_and_no_other():
    """The choice of the form and its logs move with a block of the pool and never with a block outside it, so a form chosen on
    development blocks reads no block of a test start."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(3), uneven=True, pairs=PAIR)
    form = em.select_form(SPACE, blocks, DEVELOPMENT, FOLD_OF, WINDOW)
    assert em.select_form(SPACE, moved(blocks, *TEST), DEVELOPMENT, FOLD_OF, WINDOW) == form
    changed = em.select_form(SPACE, moved(blocks, DEVELOPMENT[20]), DEVELOPMENT, FOLD_OF, WINDOW)
    assert changed["forms"] != form["forms"] and changed["block_effects"] != form["block_effects"]


def test_the_estimator_is_the_registered_one_of_the_largest_held_out_window_density_and_the_block_effects_choose_nothing(monkeypatch):
    _beta, blocks = synthetic_blocks(np.random.default_rng(4), cycles_per_block=9, uneven=True, pairs=PAIR)
    form = em.select_form(SPACE, blocks, DEVELOPMENT, FOLD_OF, WINDOW)
    data = em.PoolData(SPACE, blocks, DEVELOPMENT)
    recorded = em.RecordedBlocks(SPACE, {h: blocks[h] for h in DEVELOPMENT}, WINDOW)
    folds_of_blocks = {h: FOLD_OF[h] for h in DEVELOPMENT}
    folds = em.folds_of_pool(DEVELOPMENT, folds_of_blocks)
    assert [f["covariance"] for f in form["forms"]] == list(em.COVARIANCE_ESTIMATORS) and not form["fallback"]
    for f in form["forms"]:
        assert f["window_log_likelihood"] == pytest.approx(window_criterion(blocks, DEVELOPMENT, FOLD_OF, f["covariance"]), rel=1e-9)
    assert form["covariance"] == max(form["forms"], key=lambda f: f["window_log_likelihood"])["covariance"]
    assert form["held_out_window_dof"] == sum(len(blocks[h]) // WINDOW - 1 for h in DEVELOPMENT) == em.held_out_window_dof(recorded)
    assert form["pairs"] == em.select_interactions(SPACE, data, folds_of_blocks)[0]
    assert [e["block_effect"] for e in form["block_effects"]] == list(em.BLOCK_EFFECTS) and "block_effect" not in form
    for e in form["block_effects"]:
        value = em.model_form.window_block_likelihood(SPACE, data, recorded, form["pairs"], folds, form["covariance"], e["block_effect"] == "joint_block")
        assert e["window_log_likelihood"] == value and math.isfinite(value)
    # a tie goes to the earlier registered estimator, and a report of the block effects moves no choice
    monkeypatch.setattr(em.model_form, "window_block_likelihood", lambda *_args: 0.0)
    assert em.select_form(SPACE, blocks, DEVELOPMENT, FOLD_OF, WINDOW)["covariance"] == form["covariance"]
    monkeypatch.setattr(em.model_form, "form_window_likelihood", lambda *_args: 1.0)
    assert em.select_form(SPACE, blocks, DEVELOPMENT, FOLD_OF, WINDOW)["covariance"] == em.COVARIANCE_ESTIMATORS[0]


def test_too_few_held_out_windows_keep_the_default_estimator_and_say_so():
    """With fewer held out degrees of freedom of the window means than summary fields the estimators are not told apart, so the
    form keeps DEFAULT_COVARIANCE and records the fallback, whichever estimator the criterion names, and with enough of them the
    criterion decides. The rule counts the windows of the blocks it is given, so it holds for any asset."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(15), cycles_per_block=6)  # two windows a block, independent summaries
    few, enough = DEVELOPMENT[:len(SPACE.fields) - 4], DEVELOPMENT[:len(SPACE.fields) + 6]

    def named_by_the_criterion(form):
        assert all(f["window_log_likelihood"] is not None for f in form["forms"])
        return max(form["forms"], key=lambda f: f["window_log_likelihood"])["covariance"]

    form = em.select_form(SPACE, blocks, few, FOLD_OF, WINDOW)
    assert form["held_out_window_dof"] == len(few) < len(SPACE.fields) and named_by_the_criterion(form) != em.DEFAULT_COVARIANCE
    assert form["fallback"] and form["covariance"] == em.DEFAULT_COVARIANCE
    form = em.select_form(SPACE, blocks, enough, FOLD_OF, WINDOW)
    assert form["held_out_window_dof"] == len(enough) >= len(SPACE.fields) and not form["fallback"]
    assert form["covariance"] == named_by_the_criterion(form) != em.DEFAULT_COVARIANCE
    one_window = em.select_form(SPACE, {h: x[:WINDOW + 1] for h, x in blocks.items()}, DEVELOPMENT, FOLD_OF, WINDOW)
    assert one_window["held_out_window_dof"] == 0 and one_window["fallback"] and one_window["covariance"] == em.DEFAULT_COVARIANCE


def test_an_estimator_that_cannot_be_evaluated_is_left_out_and_the_degrees_of_freedom_stay_those_of_the_windows():
    """Blocks whose windows repeat one pattern have window means that do not vary, so the window estimator gives a covariance
    that cannot be factored. It is left out, the other registered estimators are compared, and the degrees of freedom are still
    those of the windows of every block, so the form does not fall back."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(14), cycles_per_block=WINDOW)
    repeating = {h: np.tile(x, (2 + 2 * (i % 2), 1)) for i, (h, x) in enumerate(blocks.items())}  # two or four equal windows
    form = em.select_form(SPACE, repeating, DEVELOPMENT, FOLD_OF, WINDOW)
    criterion = {f["covariance"]: f["window_log_likelihood"] for f in form["forms"]}
    assert criterion["window_means"] is None and criterion["ledoit_wolf_diagonal"] is not None and criterion["diagonal"] is not None
    assert form["held_out_window_dof"] == sum(len(repeating[h]) // WINDOW - 1 for h in DEVELOPMENT) >= len(SPACE.fields)
    assert not form["fallback"] and form["covariance"] == max(("ledoit_wolf_diagonal", "diagonal"), key=criterion.get)


def test_drift_within_blocks_is_found_by_the_window_estimator_and_a_shared_offset_by_the_block_report():
    """Blocks whose cycles drift within the block are fitted better by the window estimator, and blocks with a large shared
    offset per block are fitted better at window level with the offset integrated out than without it."""
    rng = np.random.default_rng(6)
    _beta, blocks = synthetic_blocks(rng, cycles_per_block=12, noise=0.1)
    assert em.select_form(SPACE, blocks, DEVELOPMENT, FOLD_OF, WINDOW)["covariance"] != "window_means"
    drifting = {h: x + np.cumsum(rng.normal(scale=0.08, size=x.shape), axis=0) for h, x in blocks.items()}
    assert em.select_form(SPACE, drifting, DEVELOPMENT, FOLD_OF, WINDOW)["covariance"] == "window_means"
    offset = {h: x + rng.normal(scale=1.0, size=x.shape[1]) for h, x in blocks.items()}
    effects = {e["block_effect"]: e["window_log_likelihood"] for e in em.select_form(SPACE, offset, DEVELOPMENT, FOLD_OF, WINDOW)["block_effects"]}
    assert effects["joint_block"] > effects["joint"]


# ----- the rule that gives every start its form -----

def test_development_blocks_identify_the_pairs_only_when_every_pool_of_the_rule_does():
    """The pairs are chosen on the development blocks when the model with every pair has full column rank on every training
    fold of every pool the rule uses: all of them, for a start outside them, and all of them less one, for each start among
    them. The check reads the grid and the folds and no recording."""
    everything = list(SMALL.combinations)
    assert em.identifies_every_pair(SMALL, everything, DIAGONAL_FOLDS) and em.model_form.development_identifies(SMALL, everything, DIAGONAL_FOLDS)
    assert not em.identifies_every_pair(SMALL, everything, LATIN_FOLDS) and not em.model_form.development_identifies(SMALL, everything, LATIN_FOLDS)
    # one block outside: all of the others identify every pair, and less one more they lose a line of the grid in some fold
    less_one = everything[1:]
    assert em.identifies_every_pair(SMALL, less_one, DIAGONAL_FOLDS) and not em.model_form.development_identifies(SMALL, less_one, DIAGONAL_FOLDS)
    assert not em.identifies_every_pair(SMALL, MULTI, DIAGONAL_FOLDS) and not em.model_form.development_identifies(SMALL, MULTI, DIAGONAL_FOLDS)
    assert not em.identifies_every_pair(SMALL, [], DIAGONAL_FOLDS) and not em.model_form.development_identifies(SMALL, [], DIAGONAL_FOLDS)
    # a grid whose lines hold nine blocks: two blocks outside and one more left out break no training fold
    wide_folds = {h: sum(h) % 3 for h in WIDE.combinations}
    inside = [h for h in WIDE.combinations if h not in (WIDE.combinations[1], WIDE.combinations[30])]
    assert em.model_form.development_identifies(WIDE, inside, wide_folds)
    assert not em.model_form.development_identifies(WIDE, [h for h in WIDE.combinations if WIDE.n_faulty(h) not in (1, 2)], wide_folds)


def test_an_asset_without_development_blocks_keeps_the_default_estimator_and_selects_its_pairs_per_start():
    blocks = small_blocks(20)
    selections, parts = em.model_forms(SMALL, blocks, [], LATIN_FOLDS, WINDOW)
    assert parts == {"pairs": {"chosen": "per_start", "development_identifies": False}, "covariance": {"chosen": "development"},
                     "block_effect": {"chosen": "registered"}}
    assert list(selections) == list(SMALL.combinations)
    for start, form in selections.items():
        second = em.recorded_blocks.second_block(start, [h for h in SMALL.combinations if h != start])
        assert form["left_out"] == [start, second] and form["covariance_left_out"] == [] and form["report_pairs"] == []
        assert (form["pairs"], form["log"]) == em.select_pool_pairs(SMALL, blocks, [start, second], LATIN_FOLDS)
        assert (form["covariance"], form["fallback"], form["held_out_window_dof"]) == (em.DEFAULT_COVARIANCE, True, 0)
        assert all(f["window_log_likelihood"] is None for f in form["forms"] + form["block_effects"])
    assert any(form["pairs"] for form in selections.values())


def test_a_part_the_development_blocks_identify_is_chosen_on_them():
    """Development blocks that identify every pair in every training fold choose the pairs, for a start among them on the
    others, so the pairs of a start read no recording of its block."""
    blocks = small_blocks(21)
    everything = list(SMALL.combinations)
    selections, parts = em.model_forms(SMALL, blocks, everything, DIAGONAL_FOLDS, WINDOW)
    assert parts["pairs"] == {"chosen": "development", "development_identifies": True}
    for start in everything[::5]:
        form = em.select_form(SMALL, blocks, [h for h in everything if h != start], DIAGONAL_FOLDS, WINDOW)
        chosen = selections[start]
        assert (chosen["pairs"], chosen["log"], chosen["report_pairs"]) == (form["pairs"], form["log"], form["pairs"])
        assert chosen["left_out"] == chosen["covariance_left_out"] == [start]
    assert any(form["pairs"] for form in selections.values())
    again, _parts = em.model_forms(SMALL, moved(blocks, everything[5]), everything, DIAGONAL_FOLDS, WINDOW)
    assert again[everything[5]] == selections[everything[5]] and again[everything[10]]["log"] != selections[everything[10]]["log"]


def test_a_start_outside_development_blocks_that_identify_the_pairs_takes_the_form_they_chose(monkeypatch):
    """Where the development blocks identify every pair, a start outside them takes the pairs and the estimator chosen on all
    of them, and its form moves with no block outside them. The test of identification is given here, since on a grid this
    small one block outside already breaks it, as the test above shows."""
    blocks = small_blocks(24)
    inside = [h for h in SMALL.combinations if h not in OUTSIDE]
    monkeypatch.setattr(em.model_form, "development_identifies", lambda *_args: True)
    selections, parts = em.model_forms(SMALL, blocks, inside, LATIN_FOLDS, WINDOW)
    shared = em.select_form(SMALL, blocks, inside, LATIN_FOLDS, WINDOW)
    assert parts["pairs"] == {"chosen": "development", "development_identifies": True} and shared["pairs"]
    for start in OUTSIDE:
        assert selections[start] == {**shared, "left_out": [], "covariance_left_out": [], "report_pairs": shared["pairs"]}
    for start in inside[::6]:
        form = em.select_form(SMALL, blocks, [h for h in inside if h != start], LATIN_FOLDS, WINDOW)
        assert selections[start] == {**form, "left_out": [start], "covariance_left_out": [start], "report_pairs": form["pairs"]}
    assert em.model_forms(SMALL, moved(blocks, *OUTSIDE), inside, LATIN_FOLDS, WINDOW)[0] == selections


def test_a_part_the_development_blocks_do_not_identify_is_chosen_per_start_without_the_two_blocks_its_fit_leaves_out():
    """Development blocks that do not identify every pair leave the pairs to a selection per start, on every
    recorded block but the start's and its second block, the two that the fit of the start's own hypothesis leaves out. So the
    pairs of a start move with neither of those blocks, and a selection that left out fewer blocks would."""
    blocks = small_blocks(22)
    assert not em.model_form.development_identifies(SMALL, MULTI, LATIN_FOLDS)
    selections, parts = em.model_forms(SMALL, blocks, MULTI, LATIN_FOLDS, WINDOW)
    assert parts["pairs"] == {"chosen": "per_start", "development_identifies": False}
    for start in (MULTI[0], MULTI[7], *OUTSIDE):
        form = selections[start]
        second = em.recorded_blocks.second_block(start, [h for h in SMALL.combinations if h != start])
        assert form["left_out"] == [start, second]
        assert (form["pairs"], form["log"]) == em.select_pool_pairs(SMALL, blocks, [start, second], LATIN_FOLDS)
        # the second block of the fits of an episode is the one the selection left out, whatever the pairs (main effects here,
        # since a pair cell of this small grid holds three blocks and the fits of an episode need more)
        assert em.fit_episode(em.RecordedBlocks(SMALL, blocks, WINDOW), [], start, 0, 0).second == second
        changed = moved(blocks, start, second)
        assert em.select_pool_pairs(SMALL, changed, [start, second], LATIN_FOLDS) == (form["pairs"], form["log"])
        assert em.select_pool_pairs(SMALL, changed, [start], LATIN_FOLDS) != em.select_pool_pairs(SMALL, blocks, [start], LATIN_FOLDS)
        assert em.select_pool_pairs(SMALL, changed, [], LATIN_FOLDS) != em.select_pool_pairs(SMALL, blocks, [], LATIN_FOLDS)
    assert any(form["pairs"] for form in selections.values())
    moved_two, _parts = em.model_forms(SMALL, moved(blocks, MULTI[0], selections[MULTI[0]]["left_out"][1]), MULTI, LATIN_FOLDS, WINDOW)
    assert (moved_two[MULTI[0]]["pairs"], moved_two[MULTI[0]]["log"]) == (selections[MULTI[0]]["pairs"], selections[MULTI[0]]["log"])


def test_every_start_takes_the_estimator_of_the_development_blocks_less_its_own():
    """The covariance estimator is always chosen on the development blocks: for a start among them without its own block, for
    any other start on all of them. So no start's estimator reads its own block, and no estimator moves with a block outside the
    development blocks."""
    blocks = small_blocks(23)
    selections, parts = em.model_forms(SMALL, blocks, MULTI, LATIN_FOLDS, WINDOW)
    shared = em.select_form(SMALL, blocks, MULTI, LATIN_FOLDS, WINDOW)
    assert parts["covariance"] == {"chosen": "development"} and parts["block_effect"] == {"chosen": "registered"}
    for start in OUTSIDE:
        assert {k: selections[start][k] for k in COVARIANCE_KEYS} == {k: shared[k] for k in COVARIANCE_KEYS}
        assert selections[start]["covariance_left_out"] == [] and selections[start]["report_pairs"] == shared["pairs"]
    for start in MULTI:
        form = em.select_form(SMALL, blocks, [h for h in MULTI if h != start], LATIN_FOLDS, WINDOW)
        assert {k: selections[start][k] for k in COVARIANCE_KEYS} == {k: form[k] for k in COVARIANCE_KEYS}
        assert selections[start]["covariance_left_out"] == [start] and selections[start]["forms"] != shared["forms"]
    outside_moved, _parts = em.model_forms(SMALL, moved(blocks, *OUTSIDE), MULTI, LATIN_FOLDS, WINDOW)
    assert all({k: outside_moved[s][k] for k in COVARIANCE_KEYS} == {k: selections[s][k] for k in COVARIANCE_KEYS} for s in SMALL.combinations)
    own_moved, _parts = em.model_forms(SMALL, moved(blocks, MULTI[3]), MULTI, LATIN_FOLDS, WINDOW)
    assert own_moved[MULTI[3]]["forms"] == selections[MULTI[3]]["forms"] and own_moved[MULTI[4]]["forms"] != selections[MULTI[4]]["forms"]


def test_the_development_blocks_of_the_rig_do_not_identify_every_pair():
    """The worked instance: the development blocks of the rig are the healthy block and the blocks of three or four faults. They
    identify the main effects in every training fold and not every pair, so the rig takes the branch that chooses the pairs once
    per start. The check reads the grid, the folds and the split of the rig, and no recording."""
    development = development_blocks(profile_view(), SPACE.combinations)
    assert development == DEVELOPMENT and SPACE.healthy_combination in development
    assert not em.identifies_every_pair(SPACE, development, FOLD_OF) and not em.model_form.development_identifies(SPACE, development, FOLD_OF)
    design = em.design_matrix(SPACE, development, [])
    assert all(em.model_terms.full_rank(design[[i for i, h in enumerate(development) if FOLD_OF[h] != fold]]) for fold in set(FOLD_OF.values()))
    assert em.identifies_every_pair(SPACE, SPACE.combinations, FOLD_OF)
