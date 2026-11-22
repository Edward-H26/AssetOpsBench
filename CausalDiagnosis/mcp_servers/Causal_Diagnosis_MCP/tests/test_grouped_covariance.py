"""The grouping of the covariances (offline/covariance_grouping.py, episode_fit.group_tables): every hypothesis's grouped
covariances leave its own block out, or the second block, and reproduce the centred sample covariances of fresh fits; the pooled
form and the weight one give the fits of the version before the grouping, and E3 reads the grouped within covariance of the
healthy combination; the start's block moves no grouped part while another block moves them; the rule takes the pooled form on
a tie and within one standard error and the best candidate beyond it; the score of a record is the filter's belief. No test states a
value computed from the rig data."""
import math

import numpy as np

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import covariance_grouping as cg
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import SPACE
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_clean_fits import PAIRS, START, moved, needs_facts
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_effect_model import synthetic_blocks

COMPONENT = 0


def blocks_of(seed):
    return synthetic_blocks(np.random.default_rng(seed), uneven=True, pairs=PAIRS)[1]


def test_the_grouped_covariances_leave_the_hypothesis_s_own_block_out_and_equal_those_of_fresh_fits():
    blocks = blocks_of(0)
    recorded = em.RecordedBlocks(SPACE, blocks)
    within, block = em.group_tables(recorded, PAIRS, START, COMPONENT)
    pool = [h for h in SPACE.combinations if h in blocks and h != START]
    design = em.design_matrix(SPACE, pool, PAIRS)
    means = np.array([np.asarray(blocks[h]).mean(axis=0) for h in pool])
    counts = np.array([len(blocks[h]) for h in pool], dtype=float)
    h = pool[7]
    members = [p for p, g in enumerate(pool) if g[COMPONENT] == h[COMPONENT] and g != h]
    deviations = np.concatenate([np.asarray(blocks[pool[p]]) - means[p] for p in members])
    assert np.allclose(within[SPACE.combo_index[h]], deviations.T @ deviations / (len(deviations) - len(members)))
    misfits = []
    for p in members:
        keep = [q for q in range(len(pool)) if q not in (p, pool.index(h))]
        beta = em.weighted_fit(design[keep], means[keep], counts[keep])
        misfits.append(means[p] - design[p] @ beta)
    assert np.allclose(block[SPACE.combo_index[h]], np.cov(np.array(misfits).T))
    with_own = [p for p, g in enumerate(pool) if g[COMPONENT] == h[COMPONENT]]
    deviations_own = np.concatenate([np.asarray(blocks[pool[p]]) - means[p] for p in with_own])
    assert not np.allclose(within[SPACE.combo_index[h]], deviations_own.T @ deviations_own / (len(deviations_own) - len(with_own)))


def test_the_pooled_form_and_the_weight_one_give_the_former_fits_and_e3_reads_the_grouped_healthy_covariance():
    recorded = em.RecordedBlocks(SPACE, blocks_of(1))
    former = em.fit_episode(recorded, PAIRS, START, 0, 0).effect
    for grouping in (None, (COMPONENT, 1.0)):
        effect = em.fit_episode(recorded, PAIRS, START, 0, 0, grouping=grouping).effect
        assert np.array_equal(effect.sigma_within, former.sigma_within) and np.array_equal(effect.sigma_block, former.sigma_block)
    grouped = em.fit_episode(recorded, PAIRS, START, 0, 0, grouping=(COMPONENT, 0.1)).effect
    healthy = SPACE.combo_index[SPACE.healthy_combination]
    assert not np.array_equal(grouped.sigma_within, former.sigma_within)
    assert np.array_equal(grouped.healthy_scale, np.sqrt(np.diag(grouped.sigma_within[healthy])))


def test_the_start_s_block_moves_no_grouped_part_and_another_block_moves_them():
    blocks = blocks_of(2)

    def tables(changed):
        return em.group_tables(em.RecordedBlocks(SPACE, changed), PAIRS, START, COMPONENT)
    stored = tables(blocks)
    assert all(np.array_equal(a, b) for a, b in zip(stored, tables(moved(blocks, START))))
    assert not all(np.array_equal(a, b) for a, b in zip(stored, tables(moved(blocks, SPACE.combinations[40]))))


def test_the_rule_takes_the_pooled_form_on_a_tie_and_within_one_standard_error_and_the_best_beyond_it():
    rng = np.random.default_rng(0)
    rows = list(range(20))
    pooled = rng.normal(size=20)
    clear = {r: [pooled[r], pooled[r] + 1.0 + 0.1 * rng.normal(), pooled[r] - 1.0] for r in rows}
    assert cg.choose(clear, rows)[0] == 1
    noisy = {r: [pooled[r], pooled[r] + (0.05 if r % 2 else -0.04) * 10, pooled[r] - 1.0] for r in rows}
    index, gain, se = cg.choose(noisy, rows)
    assert gain > 0 and gain <= se and index == 0
    tie = {r: [pooled[r], pooled[r], pooled[r]] for r in rows}
    assert cg.choose(tie, rows)[0] == 0
    singular = {r: [pooled[r], -math.inf, pooled[r] - 1.0] for r in rows}
    assert cg.choose(singular, rows)[0] == 0


@needs_facts
def test_the_score_of_a_record_is_the_belief_of_the_filter():
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import asset_view, served_segments, start_sets
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Episode, Filter, Reading
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.offline_table import released

    models, view = load_models(), asset_view()
    space, index = models.space, models.space.field_index
    development, _test = start_sets(view, models.blocks)
    for start in [tuple(s) for s in development][:2]:
        effect, block = models.without(start).effect, models.blocks[start]
        for positions in served_segments()[:2]:
            readings = [Reading(n + 1, n - len(positions), "history", released(index, block[p], view.summary_fields), recorded=True) for n, p in enumerate(positions)]
            post = Filter(view, effect, Episode(0, readings, [], None), resampled=False, block=True).posterior()
            record = np.array([[released(index, block[p], view.summary_fields)[f] for f in space.fields] for p in positions])
            ll = cg.record_log_likelihoods(effect, record)
            assert np.allclose(post.probabilities, np.exp(ll - ll.max()) / np.exp(ll - ll.max()).sum(), rtol=0, atol=1e-10)
