"""The fits of the belief after a change of the hidden combination (CausalDiagnosis/docs/causal-diagnosis-mcp.md, Repairs and drops):
the blocks that the released readings come from, which Models.served_blocks finds by looking the values of each reading up in
the models file, leave the fits that weigh a reading, and nothing else. Every reading finds its own block, and a reading whose
values two blocks hold names both; a served block enters only with its reading; every number that names a combination stays on
the fits without the start's block; the fits of the belief are built once per change of the released readings whatever the
blocks, and each build is padded to the time of the start's fits; a set of blocks never turns into a refusal; the yardstick of
the tables replays the server's belief. No test states a value computed from the rig data."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS
from CausalDiagnosis.evaluation.metrics import realized_read_elimination as read_elimination
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools, Models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import causal_tools as tools_module
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import SPACE
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_clean_fits import PAIRS, PARTS, moved_parts, needs_facts
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_effect_model import synthetic_blocks

START = (100, 80, 0, 90)
REPAIRED = (100, 80, 0, 130)  # START after the repair of the accumulator


@pytest.fixture(scope="module")
def world():
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles

    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


@pytest.fixture(scope="module")
def models():
    return load_models()


def repaired_episode(world, folder, component="accumulator"):
    """An episode of START that read tier3, diagnosed, repaired a component and waited for the close, before any reading after it."""
    episode = FixtureEpisode(folder, *world, START, seed=0)
    episode.read("tier3")
    episode.submit_diagnosis(dict(zip(COMPONENTS, START)))
    episode.create_work_order(component)
    episode.advance(4)
    return episode


def observations(episode):
    return json.loads(episode.released_path.read_text(encoding="utf-8"))["observations"]


def test_no_part_of_the_belief_fits_moves_with_a_served_block_and_every_part_of_the_start_fits_does():
    _beta, blocks = synthetic_blocks(np.random.default_rng(0), uneven=True, pairs=PAIRS)
    served = SPACE.combinations[40]

    def belief_fit(recorded, pairs, left_out, resamples, seed):
        return em.fit_episode(recorded, pairs, left_out, resamples, seed, also_left_out=[served])
    assert moved_parts(belief_fit, blocks, served) == set()
    assert moved_parts(em.fit_episode, blocks, served) == set(PARTS)


@needs_facts
def test_a_reading_without_a_field_of_the_space_names_no_block(models):
    assert models.served_blocks([{"values": {}}]) == [] and models.served_blocks([{"values": {"no_such_field": 1.0}}]) == []


@needs_facts
def test_a_set_of_served_blocks_never_turns_into_a_refusal(models):
    """The fits leave out the longest prefix of the blocks whose remaining blocks still fit the model; the whole set, which
    takes every block of one level of the cooler, is refused by the fit itself."""
    one_level = [h for h in models.recorded.combos if h[0] == 3 and h != START]
    with pytest.raises(ValueError):
        models.without(START, one_level)
    k = len(one_level)
    while True:
        try:
            expected = models.without(START, one_level[:k])
            break
        except ValueError:
            k -= 1
    fits = models.without_longest(START, [START, *one_level])
    assert 0 < k < len(one_level) and np.array_equal(fits.effect.mu_table, expected.effect.mu_table)


@needs_facts
def test_every_reading_finds_its_own_block_and_one_that_two_blocks_hold_names_both(world, models, tmp_path):
    episode = repaired_episode(world, tmp_path / "played")
    assert models.served_blocks(observations(episode)) == [START]  # the repair has closed, no reading of its block yet
    episode.read("tier3")
    assert models.served_blocks(observations(episode)) == [START, REPAIRED]
    other = (3, 73, 2, 90)
    raw = copy.deepcopy(models.raw)
    raw["blocks"][em.combo_key(other)] = [*raw["blocks"][em.combo_key(other)], raw["blocks"][em.combo_key(REPAIRED)][0]]
    twice = Models(raw)  # one recorded cycle of the repaired block copied into another block
    reading = {"values": {f: float(f"{v:.6g}") for f, v in zip(models.space.fields, models.blocks[REPAIRED][0])}}
    assert models.served_blocks([reading]) == [REPAIRED] and twice.served_blocks([reading]) == sorted([REPAIRED, other])


@needs_facts
def test_no_number_that_names_a_combination_moves_when_a_new_block_is_served_and_the_belief_does(world, models, tmp_path):
    episode = repaired_episode(world, tmp_path)
    tools = CausalTools(models.without(START), episode.released_path)

    def first_reading_scores():
        return [(item.field, item.anomaly_score) for item in tools.get_episode_evidence("all").items if item.n == 1]
    before = first_reading_scores()
    _flt, post_before = tools._belief(tools._released())
    episode.read("tier3")
    assert first_reading_scores() == before
    _flt, post_after = tools._belief(tools._released())
    assert not np.array_equal(post_after.effect.mu_table, post_before.effect.mu_table)
    values = observations(episode)[0]["values"]
    assert post_after.effect.anomaly_scores(values) != tools.models.effect.anomaly_scores(values)  # the belief's fits would have moved them


@needs_facts
def test_the_belief_fits_are_built_once_per_change_of_the_released_readings_whatever_the_blocks(world, models, tmp_path, monkeypatch):
    """A repair that changes the combination and a repair of a healthy component, which changes nothing, give the same number of
    builds, the reference build that leaves out the start's block alone first; only the first repair leaves the repaired block out."""
    builds = {}
    real = Models.without_longest
    for variant, component in (("changed", "accumulator"), ("unchanged", "cooler")):
        record = builds.setdefault(variant, [])
        monkeypatch.setattr(Models, "without_longest", lambda self, start, also, record=record: record.append(tuple(also)) or real(self, start, also))
        episode = repaired_episode(world, tmp_path / variant, component)
        tools = CausalTools(models.without(START), episode.released_path)
        tools.rank_root_causes(1)
        tools.rank_root_causes(1)
        episode.read("tier3")
        tools.rank_root_causes(1)
        tools.rank_root_causes(1)
    assert len(builds["changed"]) == len(builds["unchanged"]) == 3 and builds["changed"][0] == builds["unchanged"][0] == (START,)
    assert builds["changed"][-1] == (START, REPAIRED) and builds["unchanged"][-1] == (START,)


@needs_facts
def test_every_build_of_the_belief_fits_is_padded_to_one_budget(world, models, tmp_path, monkeypatch):
    """The first build of a process is preceded by a reference build on the same path that leaves out the start's block alone,
    the fewest blocks; every build then sleeps up to that time times one plus PAD_MARGIN, and a slower build raises the budget, with
    the resampled tables of E6 drawn inside the build, so that the time of a call does not tell how many blocks the belief leaves out."""
    episode = repaired_episode(world, tmp_path)
    episode.read("tier3")
    slept = []
    clock = iter([0.0, 0.050, 1.0, 1.030, 2.0, 2.090, 3.0, 3.020])  # the reference build, then three rebuilds
    monkeypatch.setattr(tools_module, "time", SimpleNamespace(perf_counter=lambda: next(clock), sleep=slept.append))
    tools = CausalTools(models.without(START), episode.released_path)
    for _ in range(3):
        tools._belief_key = None
        tools._belief_model(tools._released())
    budget = 0.050 * (1 + tools_module.PAD_MARGIN)
    assert budget < 0.090 and slept == [pytest.approx(budget - 0.030), pytest.approx(0.0), pytest.approx(0.090 - 0.020)]
    effect = tools._belief_backend.models.effect
    assert effect.resample_count > 0 and len(effect._resampled) == effect.resample_count  # E6 drawn inside the padded build


@needs_facts
def test_the_yardstick_replays_the_belief_of_the_server_after_a_change(world, models, tmp_path):
    episode = repaired_episode(world, tmp_path)
    episode.read("tier3")
    served = CausalTools(models.without(START), episode.released_path).rank_root_causes(1).effective_hypotheses
    released = json.loads(episode.released_path.read_text(encoding="utf-8"))
    last = read_elimination.read_records(released, left_out=START)[-1]
    assert abs(last["effective_after"] - served) <= 1e-9 * served
    stick = read_elimination.yardstick(left_out=START)
    start_only = stick._backend(()).replay(stick.view(released), read_elimination.released_before(released, last["index"], last["t_end"], with_read=True)).posterior().effective_hypotheses()
    assert abs(start_only - served) > 1e-6 * served
