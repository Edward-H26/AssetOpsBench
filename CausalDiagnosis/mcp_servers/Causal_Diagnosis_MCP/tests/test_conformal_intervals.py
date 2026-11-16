"""E14, E16 and E17: on hand made calibration sets the finite sample quantile, the block weights and the combined score with
its p value, and on the rig two checks, that the combined score rejects a reading of the block
one cooler level away from the predicted combination, the cooler alone, and that the leave one block out calibration of the
episode of a development start holds its exact bounds: at most alpha (n + 1) - 1 units of calibration weight found
inconsistent, at least (1 - alpha)(n + 1) units inside every interval, and a level that too few blocks cannot reach said so."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import conformal_intervals as co

needs_facts = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")


def test_the_quantile_is_the_finite_sample_order_statistic_with_unit_weights():
    n = 19
    scores = np.arange(1, n + 1, dtype=float)[:, None]
    calibration = co.Calibration(scores, np.ones(n), n_blocks=n)
    for alpha in (0.05, 0.1, 0.5):
        q, capped = calibration.quantile(0, alpha)
        assert not capped and q == math.ceil((1 - alpha) * (n + 1))
    q, capped = calibration.quantile(0, 0.001)
    assert capped and q == n
    low, high = co.interval(10.0, 2.0, 3.0)
    assert (low, high) == (4.0, 16.0)


def test_blocks_share_one_unit_of_weight_whatever_their_length():
    scores = np.array([[1.0]] * 10 + [[5.0]])
    weights = np.array([0.1] * 10 + [1.0])
    calibration = co.Calibration(scores, weights, n_blocks=2)
    q, capped = calibration.quantile(0, 0.5)
    assert q == 5.0 and not capped  # the long block of low scores counts once, so the median block score is the second block's
    assert calibration.combined_p_value([0], [5.0]) == (5.0, (1 + 1.0) / 3)
    assert calibration.combined_p_value([0], [1.0]) == (1.0, (1 + 2.0) / 3)
    assert calibration.combined_p_value([0], [9.0]) == (9.0, 1 / 3)


def test_the_combined_score_is_the_largest_deviation_over_the_columns_read_and_its_p_value_counts_calibration_cycles_scored_alike():
    """One score per reading, the largest standardized deviation over the columns it holds, against the same
    statistic of the calibration cycles over the same columns, one unit of weight per block; the verdict is one test at alpha."""
    scores = np.array([[0.1, 3.0], [0.2, 0.5], [2.5, 0.4], [0.3, 0.6]])
    calibration = co.Calibration(scores, np.array([0.5, 0.5, 1.0, 1.0]), n_blocks=3)
    assert calibration.combined_p_value([0, 1], [1.0, 2.0]) == (2.0, (1 + 0.5 + 1.0) / 4)  # rows 0 (3.0) and 2 (2.5) lie at or above 2.0
    assert calibration.combined_p_value([0], [1.0, 2.0][:1]) == (1.0, (1 + 1.0) / 4)  # over the first column alone row 2 (2.5) does
    assert calibration.combined_p_value([0, 1], [0.1, 0.1]) == (0.1, 1.0)
    assert calibration.combined_p_value([0, 1], [1.0, 2.0], keep=np.array([False, True, True, True])) == (2.0, (1 + 1.0) / 3)
    assert co.verdict(0.06, 0.05) is True and co.verdict(0.05, 0.05) is False and co.verdict(1 / 4, 0.05) is True


def test_the_prediction_is_the_most_probable_combination_s_with_ties_broken_in_the_candidate_order():
    """The predicted mean of every summary under a belief is the mean of its most probable combination, not the
    belief weighted mean, with the conformal interval of the summary about it, and a tie goes to the combination that comes
    first in the candidate order."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.synthetic import TINY, effect_with_means, table_of

    effect = effect_with_means(TINY, table_of(TINY, lambda h: [10.0 * h["A"], 10.0 * h["B"], 1.0, 2.0]))
    models = SimpleNamespace(effect=effect, calibration=co.Calibration(np.full((19, 4), 1.5), np.ones(19), n_blocks=19), sigma=np.full(4, 2.0))
    first, second = (1, 0), (2, 3)
    predicted = co.predict_summaries(models, {second: 0.4, first: 0.6}, 0.05)
    assert [p.mean for p in predicted] == list(effect.mu(first)) and all((p.low, p.high) == (p.mean - 3.0, p.mean + 3.0) for p in predicted)
    assert co.most_probable({second: 0.5, first: 0.5}, TINY) == (first, 0.5)
    assert [p.mean for p in co.predict_summaries(models, {second: 0.5, first: 0.5}, 0.05)] == list(effect.mu(first))


def rig_episode():
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view, recorded_blocks

    models = load_models()
    view = profile_view()
    return models, view, recorded_blocks(view)[0], models.without((100, 80, 0, 90))


@needs_facts
def test_the_combined_score_rejects_a_reading_of_the_block_one_cooler_level_away_from_the_predicted_combination():
    """A reading of a recorded block whose combination differs from the predicted one by one level
    of the cooler is inconsistent at alpha 0.05 on at least nine cycles in ten, for every tier and for a healthy and a faulty
    predicted combination. The check is about the cooler alone. No test asserts a rejection for a step of the valve, the pump
    or the accumulator."""
    models, view, blocks, episode = rig_episode()
    space, alpha = models.space, co.DEFAULT_ALPHA
    for predicted in (space.healthy_combination, (100, 100, 0, 90)):
        mu = episode.effect.mu(predicted)
        levels = space.levels[0]
        for level in levels:
            if level == predicted[0]:
                continue
            if abs(levels.index(level) - levels.index(predicted[0])) != 1:
                continue
            neighbour = (level,) + tuple(predicted[1:])
            for group in ("tier1", "tier2", "tier3"):
                idx = space.field_indices(view.group_fields(group))
                rejected = 0
                cycles = np.asarray(blocks[neighbour], dtype=float)
                for cycle in cycles:
                    _score, p = episode.calibration.combined_p_value(idx, np.abs(cycle[idx] - mu[idx]) / episode.sigma[idx])
                    rejected += not co.verdict(p, alpha)
                assert rejected >= 0.9 * len(cycles), (predicted, neighbour, group, rejected, len(cycles))


def development_episode():
    """The fits of an episode of the first development start of the rig, and the view and models it comes from."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view, start_sets

    models, view = load_models(), profile_view()
    start = start_sets(view, models.blocks)[0][0]
    return models, view, models.without(start)


@needs_facts
def test_the_verdict_finds_at_most_its_exact_bound_of_the_calibration_weight_inconsistent():
    """Every calibration cycle scored by its own p value against the calibration set: the block weight of the cycles found
    inconsistent at alpha is at most alpha (n + 1) - 1, for every sensor group of the episode of a development start. On n
    blocks of one cycle with distinct scores the count is exactly the whole part of alpha (n + 1) less one, which a p value
    over n in place of n + 1 would miss."""
    models, view, episode = development_episode()
    calibration, alpha = episode.calibration, co.DEFAULT_ALPHA
    bound = alpha * (calibration.n_blocks + 1) - 1
    for group in sorted(view.sensor_groups):
        idx = models.space.field_indices(view.group_fields(group))
        found = [calibration.combined_p_value(idx, calibration.rows[r, idx])[1] <= alpha for r in range(len(calibration.rows))]
        assert float(calibration.row_weights[np.array(found)].sum()) <= bound + 1e-9, group
    for n in (19, 50, 99, 120):
        hand = co.Calibration(np.arange(1, n + 1, dtype=float)[:, None], np.ones(n), n_blocks=n)
        found = sum(hand.combined_p_value([0], [s])[1] <= alpha for s in hand.rows[:, 0])
        assert found == math.floor(alpha * (n + 1) + 1e-9) - 1, n


@needs_facts
def test_every_interval_holds_at_least_its_exact_share_of_the_calibration_weight():
    """Per summary, the calibration weight at or below the quantile of E14 is at least (1 - alpha)(n + 1) units, on the episode of
    a development start whose blocks reach the level."""
    models, _view, episode = development_episode()
    calibration, alpha = episode.calibration, co.DEFAULT_ALPHA
    assert calibration.level_reachable(alpha)
    target = (1 - alpha) * (calibration.n_blocks + 1)
    for k in range(len(models.space.fields)):
        q = calibration.quantile(k, alpha)[0]
        assert float(calibration.row_weights[calibration.rows[:, k] <= q].sum()) >= target - 1e-9, models.space.fields[k]


def test_too_few_blocks_are_reported_as_a_level_that_cannot_be_reached():
    """With alpha 0.05 eighteen blocks cannot carry the level: the quantile is the largest score and no reading reaches a p value
    at or below alpha. Nineteen blocks can."""
    alpha = 0.05
    short = co.Calibration(np.arange(1, 19, dtype=float)[:, None], np.ones(18), n_blocks=18)
    assert not short.level_reachable(alpha)
    assert short.quantile(0, alpha) == (18.0, True)
    assert min(short.combined_p_value([0], [s])[1] for s in (0.5, 18.0, 1000.0)) > alpha
    enough = co.Calibration(np.arange(1, 20, dtype=float)[:, None], np.ones(19), n_blocks=19)
    assert enough.level_reachable(alpha) and enough.quantile(0, alpha) == (19.0, False)
