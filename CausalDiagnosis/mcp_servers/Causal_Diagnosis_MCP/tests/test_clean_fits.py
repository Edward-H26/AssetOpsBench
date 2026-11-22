"""The clean fits end to end: no part of the fits of an episode reads the start's block and each leak of that block put back moves
the part it leaks into, the block the driver names at launch is all the launch input changes, the server reads no hidden input
at a call, every reader of a mean reads the fits of the episode, and the resampling spread of the start is the widest no more
often than chance makes it. No test states a value computed from the rig data; the data dependent checks are identities,
ranks and bounds that chance sets."""
import builtins
import io
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HIDDEN_FIELDS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import main as server_main
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools, Models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import folds_of, profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import most_probable
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import RANKING_MODELS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_effect_model import synthetic_blocks
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import SPACE

START = (100, 80, 0, 90)
OTHER = (3, 100, 0, 130)
PAIRS = [(0, 3)]
RESAMPLES = 3
NOWHERE = tuple(-1 for _ in SPACE.components)  # no recorded combination, so the fits that leave it out read every block
FOLD_OF = folds_of(profile_view(), em.FOLDS, em.FOLD_SEED)
PARTS = ("means", "within covariances", "block covariances", "healthy reference", "attribution terms", "resampled tables", "calibration")
GROUPS = ("tier1", "tier2", "tier3")
needs_facts = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")


def moved(blocks, combination, seed=0):
    """The recordings with the block of one combination given other values and three more recordings."""
    rng = np.random.default_rng(seed)
    block = np.asarray(blocks[combination], dtype=float)
    return {**blocks, combination: np.concatenate([block + rng.normal(size=block.shape) * block.std(axis=0), block[:3] * 1.5])}


# ----- the parts of the fits of an episode, on synthetic recordings -----

def parts_of(episode):
    effect = episode.effect
    return {"means": effect.mu_table, "within covariances": effect.sigma_within, "block covariances": effect.sigma_block,
            "healthy reference": np.concatenate([effect.healthy_reference, effect.healthy_scale]),
            "attribution terms": np.array([effect.component_terms(h) for h in SPACE.combinations]),
            "resampled tables": effect.tables(RESAMPLES)[1:],
            "calibration": np.concatenate([episode.scores.ravel(), episode.weights, episode.sigma, [episode.n_blocks]])}


def moved_parts(fit, blocks, changed):
    """The parts of the fits of an episode of START that differ when the block of changed moves."""
    before = parts_of(fit(em.RecordedBlocks(SPACE, blocks), PAIRS, START, RESAMPLES, 0))
    after = parts_of(fit(em.RecordedBlocks(SPACE, moved(blocks, changed)), PAIRS, START, RESAMPLES, 0))
    return {part for part in PARTS if before[part].shape != after[part].shape or not np.array_equal(before[part], after[part])}


def with_leak(part):
    """fit_episode with one part taken from the fits that also read the start's block, as the per hypothesis fit of the earlier server
    did, each of whose fits left out the scored combination's block alone."""
    def fit(recorded, pairs, left_out, resamples, seed):
        clean, leaky = (em.fit_episode(recorded, pairs, out, resamples, seed) for out in (left_out, NOWHERE))
        effect, other = clean.effect, leaky.effect
        if part == "means":
            effect.mu_table = other.mu_table
        elif part == "within covariances":
            effect.sigma_within = other.sigma_within
        elif part == "block covariances":
            effect.sigma_block = other.sigma_block
        elif part == "healthy reference":
            effect.healthy_reference, effect.healthy_scale = other.healthy_reference, other.healthy_scale
        elif part == "attribution terms":
            effect._coefficients = other._coefficients
        elif part == "resampled tables":
            effect._resampler = other._resampler
        else:
            clean.scores, clean.weights, clean.sigma, clean.n_blocks = leaky.scores, leaky.weights, leaky.sigma, leaky.n_blocks
        return clean
    return fit


def test_no_part_of_the_fits_of_an_episode_moves_with_the_start_s_block_and_every_part_moves_with_another_block():
    _beta, blocks = synthetic_blocks(np.random.default_rng(0), uneven=True, pairs=PAIRS)
    assert moved_parts(em.fit_episode, blocks, START) == set()
    assert moved_parts(em.fit_episode, blocks, SPACE.combinations[40]) == set(PARTS)


@pytest.mark.parametrize("part", PARTS)
def test_each_leak_of_the_start_s_block_put_back_moves_the_part_it_leaks_into(part):
    _beta, blocks = synthetic_blocks(np.random.default_rng(1), uneven=True, pairs=PAIRS)
    assert moved_parts(with_leak(part), blocks, START) == {part}


def test_the_selection_without_the_start_does_not_move_with_its_block_and_a_selection_over_every_block_does():
    """The pairs of a start are selected without the two blocks that the fit of its own hypothesis leaves out,
    the start's and its second block, so they move with neither, and a selection that reads either block moves with it."""
    _beta, blocks = synthetic_blocks(np.random.default_rng(2), uneven=True, pairs=PAIRS)
    second = em.recorded_blocks.second_block(START, [h for h in SPACE.combinations if h != START])
    changed, only_second = moved(moved(blocks, START), second, seed=1), moved(blocks, second, seed=1)
    assert em.select_pool_pairs(SPACE, blocks, [START, second], FOLD_OF) == em.select_pool_pairs(SPACE, changed, [START, second], FOLD_OF)
    assert em.select_pool_pairs(SPACE, blocks, [START], FOLD_OF) != em.select_pool_pairs(SPACE, only_second, [START], FOLD_OF)
    assert em.select_pool_pairs(SPACE, blocks, [], FOLD_OF) != em.select_pool_pairs(SPACE, moved(blocks, START), [], FOLD_OF)


# ----- the server on the rig's recordings -----

@pytest.fixture(scope="module")
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


@pytest.fixture(scope="module")
def models():
    return load_models()


def raw_with(models, blocks, selections=None):
    """models.json with the given recordings in place of the stored ones, and the given selections or the stored ones."""
    raw = {**models.raw, "blocks": {em.combo_key(h): np.asarray(x).tolist() for h, x in blocks.items()},
           "folds": {key: fold for key, fold in models.raw["folds"].items() if em.parse_combo(key) in blocks}}
    return {**raw, "selections": selections} if selections is not None else raw


def released_copies(world, folder, names):
    """One episode of START that read every group, with its released evidence copied into a folder per name."""
    episode = FixtureEpisode(Path(folder) / "played", *world, START, seed=0)
    for group in GROUPS:
        episode.read(group)
    paths = {}
    for name in names:
        (Path(folder) / name).mkdir()
        paths[name] = Path(shutil.copy(episode.released_path, Path(folder) / name / "released.json"))
    return paths


def answers(episode_models, path, ranking_model):
    """Every tool's answer, as JSON, of a server with the fits of an episode over the released evidence at path."""
    tools = CausalTools(episode_models, path, ranking_model=ranking_model)
    effect = tools.estimate_intervention_effect("replace:valve")
    replies = (tools.get_episode_evidence("all"), tools.rank_root_causes(3, "summary"), effect, tools.verify_outcome(effect.prediction_id))
    return [json.dumps(reply.model_dump(), sort_keys=True) for reply in replies]


@needs_facts
@pytest.mark.parametrize("ranking_model", RANKING_MODELS)
def test_the_launch_input_changes_nothing_but_the_block_it_leaves_out(ranking_model, world, models, tmp_path, monkeypatch):
    """The start's block, moved, leaves every answer of every ranking model as it was; two launches
    with different starts answer alike once neither start's block is recorded and both draw one second block, so the start
    enters only as the blocks it removes, its own and its second block; and a server whose fits read the start's block, put back
    as a check of this test, answers differently on the moved block."""
    paths = released_copies(world, tmp_path, ("stored", "moved", "first", "second", "leaky", "leaky_moved"))
    changed = Models(raw_with(models, moved(models.blocks, START)))
    assert answers(models.without(START), paths["stored"], ranking_model) == answers(changed.without(START), paths["moved"], ranking_model)
    absent = {h: x for h, x in models.blocks.items() if h not in (START, OTHER)}
    pairs, log = em.select_pool_pairs(models.space, absent, [], models.fold_of)
    selection = {"pairs": [[models.space.components[a], models.space.components[b]] for a, b in pairs], "log": log,
                 "covariance": models.covariance_without(START)}
    without_both = Models(raw_with(models, absent, {em.combo_key(START): selection, em.combo_key(OTHER): selection}))
    assert em.recorded_blocks.second_block(START, list(absent)) != em.recorded_blocks.second_block(OTHER, list(absent))
    one_second = em.recorded_blocks.second_block(START, list(absent))
    with monkeypatch.context() as patch:
        patch.setattr(em.recorded_blocks, "second_block", lambda _start, _pool: one_second)
        assert answers(without_both.without(START), paths["first"], ranking_model) == answers(without_both.without(OTHER), paths["second"], ranking_model)
    monkeypatch.setattr(adapter.fitted_models, "fit_episode", lambda recorded, pairs, _start, resamples, seed, **options:
                        em.fit_episode(recorded, pairs, NOWHERE, resamples, seed, **options))
    assert answers(models.without(START), paths["leaky"], ranking_model) != answers(changed.without(START), paths["leaky_moved"], ranking_model)


@needs_facts
def test_the_server_reads_no_hidden_input_at_a_call_and_takes_no_other_launch_value(world, models, tmp_path, monkeypatch):
    """A call opens released.json, and predictions.json beside it with the lock file of its stores, and nothing else, reads no
    environment variable, and no answer or file it writes names the start; a launch value of the served or the unread table of
    old, the entry offset or the start combination, is refused, and the environment keeps no hidden field for it."""
    path = released_copies(world, tmp_path, ("calls",))["calls"]
    tools = CausalTools(models.without(START), path)
    opened, read_keys, real_open = [], [], builtins.open

    def recording_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            opened.append(Path(file).resolve())
        return real_open(file, *args, **kwargs)

    class RecordingEnviron(dict):
        def __getitem__(self, key):
            read_keys.append(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            read_keys.append(key)
            return super().get(key, default)

    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(io, "open", recording_open)
    monkeypatch.setattr(os, "environ", RecordingEnviron(os.environ))
    effect = tools.estimate_intervention_effect("replace:valve")
    replies = (tools.get_episode_evidence("all"), tools.rank_root_causes(3, "summary"), effect, tools.verify_outcome(effect.prediction_id))
    monkeypatch.undo()
    assert opened and {p.name for p in opened} <= {"released.json", "predictions.json", "predictions.json.lock"}
    assert {p.parent for p in opened} == {path.parent.resolve()}
    assert read_keys == []
    held = [json.dumps(reply.model_dump()) for reply in replies] + [path.with_name("predictions.json").read_text(encoding="utf-8")]
    assert all(em.combo_key(START) not in text and json.dumps(list(START)) not in text and "left_out" not in text for text in held)
    for hidden in ({"entry_offset": 3}, {"start_combination": START}):
        with pytest.raises(TypeError):
            server_main.build_server(None, **hidden)
    assert "start_combination" not in HIDDEN_FIELDS


@needs_facts
def test_the_notes_of_two_starts_are_one_text(world, models, tmp_path):
    """The notes of every result are one text whatever the start, so that no note tells the block the fits leave out."""
    path = released_copies(world, tmp_path, ("notes",))["notes"]
    view = CausalTools(models.without(START), path)._released().view
    notes = CausalTools(models.without(START), path)._model_assumptions(view)
    assert notes == CausalTools(models.without(OTHER), path)._model_assumptions(view)
    assert len(notes) == 6 and notes[0] == adapter.A_RULE and adapter.A_BLOCK in notes
    assert "the server never reads a seed or the split, and no result names the start" in " ".join(notes)


@needs_facts
def test_every_reader_of_a_mean_reads_the_fits_of_the_episode(world, models, tmp_path):
    """The predicted summaries of E14, those of the most probable combination after the action, the attribution of E4 and the
    anomaly scores of E3 against the clean prediction of the healthy combination read the fits of the episode, and the ranking's
    belief and the recovery check after a repair read the fits of the belief, which also leave out the block of the repaired
    combination once a reading of it is released."""
    cycles, blocks = world
    episode_models = models.without(START)
    effect = episode_models.effect
    played = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0)
    played.read("tier3")
    tools = CausalTools(episode_models, played.released_path)
    _flt, post = tools._belief(tools._released())
    assert post.effect is tools._belief_backend.models.effect and np.array_equal(post.effect.mu_table, effect.mu_table)  # START's block alone served
    estimate = tools.estimate_intervention_effect("replace:accumulator")
    record = next(p for p in json.loads(played.released_path.with_name("predictions.json").read_text(encoding="utf-8"))
                  if p["id"] == estimate.prediction_id)
    named, mass = most_probable({em.parse_combo(key): p for key, p in record["after"].items()}, models.space)
    assert (record["predicted_combination"], record["predicted_mass"]) == (em.combo_key(named), mass)
    np.testing.assert_allclose([s.mean for s in estimate.predicted], effect.mu(named), rtol=1e-12)
    healthy = models.space.combo_index[models.space.healthy_combination]
    assert np.array_equal(effect.healthy_reference, effect.mu_table[healthy])
    assert np.array_equal(effect.healthy_scale, np.sqrt(np.diag(effect.sigma_within[healthy])))
    ranking = tools.rank_root_causes(1, "summary")
    top = tuple(ranking.candidates[0].combination[c] for c in COMPONENTS)
    shares = np.abs(effect.component_terms(top))
    explained = {item.component: item.explained_anomaly for item in ranking.attribution}
    values, _anomaly, _combo = post.attribution(post.order()[0], effect)
    assert explained == {c: values[c]["explained_anomaly"] for c in COMPONENTS} and shares.shape == (len(COMPONENTS), len(models.space.fields))
    played.submit_diagnosis(dict(zip(COMPONENTS, START)))
    played.create_work_order("accumulator", prediction_id=estimate.prediction_id)
    played.advance(estimate.duration)
    played.read("tier3")
    verification = tools.verify_outcome(estimate.prediction_id)
    _flt, now = tools._belief(tools._released())  # the recovery check reports the belief of the filter, whose fits leave the repaired block out
    repaired = tuple(HEALTHY[c] if c == "accumulator" else level for c, level in zip(COMPONENTS, START))
    assert now.effect is tools._belief_backend.models.effect and verification.status == "verified"
    assert np.array_equal(now.effect.mu_table, models.without(START, [repaired]).effect.mu_table) and not np.array_equal(now.effect.mu_table, effect.mu_table)
    assert verification.recovery_posterior == float(now.probabilities[now.candidates.index(models.space.healthy_combination)])


# ----- the uncertainty of every hypothesis by one rule -----

def start_spread_ranks(models, starts):
    """For each start, the rank of its E6 spread among every combination, 1 for the widest: the standard deviation of its mean over
    the resampled tables of the episode in within standard deviations, averaged over the summaries."""
    space, ranks = models.space, []
    for start in starts:
        episode = models.without(start)
        spread = (episode.effect.tables(models.fit["resamples"])[1:].std(axis=0) / episode.sigma).mean(axis=1)
        ranks.append(1 + int((spread > spread[space.combo_index[start]]).sum()))
    return np.array(ranks)


def spread_like_chance(ranks, hypotheses):
    """At most one widest start in the starts, which chance exceeds with a probability about 0.04 for 47 starts and 144
    hypotheses, and a mean rank no further than three standard errors toward the widest from the middle rank of chance."""
    middle, standard_error = (hypotheses + 1) / 2, math.sqrt((hypotheses ** 2 - 1) / 12 / len(ranks))
    return int((ranks == 1).sum()) <= 1 and ranks.mean() >= middle - 3 * standard_error


@needs_facts
def test_the_widest_resampling_spread_names_the_start_no_more_often_than_chance(models, monkeypatch):
    """Every row of a resampled table is predicted without its own block's draws and the start is never drawn, so the
    extrapolated truth carries no wider spread than any other combination, over the 47 starts of one or two faults. A resampled
    table whose rows keep their own draws, which would let the start's spread name it, fails the check."""
    space = models.space
    starts = [h for h in space.combinations if 1 <= space.n_faulty(h) <= 2]
    assert len(starts) == 47 and spread_like_chance(start_spread_ranks(models, starts), len(space.combinations))
    monkeypatch.setattr(em.Resampler, "table", lambda self, r: self.grid @ em.weighted_fit(self.design[self.rows], self.means[self.rows], self._draw(r)[1]))
    assert not spread_like_chance(start_spread_ranks(models, starts), len(space.combinations))


@needs_facts
def test_the_selection_without_each_recorded_combination_is_one_pair_set(models):
    """The number of distinct pair sets among the stored selections of models.json, each made without the start's block and its
    second block. It pins the stored file and no text of the docs."""
    selections = models.raw["selections"]
    assert len(selections) == len(models.recorded.combos) and len({json.dumps(s["pairs"]) for s in selections.values()}) == 1
