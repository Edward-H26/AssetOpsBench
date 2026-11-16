"""Every briefing states how the replay serves readings: while the condition levels stay the same, the asset replays the
same SEGMENT recorded cycles in a loop. The sentence stands once in the briefing of the rig's own row and of every
AssetOpsBench row at every guidance level, it names the length of the live window of CausalDiagnosis/datasets/dataset_loader/evidence_source.py, and the
recordings the clock plays repeat with exactly that period. On an AssetOpsBench row where a cycle of the episode can
run, the telemetry holds live cycles beside the cycles recorded before the episode, and there the sentence says that
those earlier cycles are other recordings, which the engine of these rows keeps true."""
import dataclasses
import json
from pathlib import Path

import pytest

import CausalDiagnosis.datasets
from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, recordings as records
from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT, RecordingSource
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.profile_registry import register_profile, unregister_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import HISTORY_CYCLES, RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing as aob_briefing
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import ROWS, TASK_SERVER, tools_for
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import asset_text, briefing, n_cycles, readings_sentence, replay_sentence

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config

LEVELS = (1, 2, 3)
TINY_FILE = Path(CausalDiagnosis.datasets.__file__).resolve().parent / "tests" / "tiny_asset.json"
BEFORE = "recorded before the episode started are other recordings outside that loop"


def cycles_run(config):
    """Whether a cycle of the episode can run on a row: its clock runs, or its task server lists the wait."""
    return config.has_clock() or "wait_for_notification" in tools_for(TASK_SERVER, config)


@pytest.mark.parametrize("level", LEVELS)
def test_the_rig_briefing_states_the_loop_once_at_every_guidance_level(world, level):
    cycles, blocks = world
    config = EpisodeConfig(guidance_level=level)
    env = HydraulicRigEnv(cycles, blocks, config)
    env.reset(7)
    profile = env.public_profile()
    text = briefing(profile, config)
    assert text.count(replay_sentence(asset_text(profile)["unit_noun"], SEGMENT)) == 1  # the noun of the rows on AssetOpsBench too
    assert BEFORE not in text  # the rig's own surface serves no cycle from before the episode


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("level", LEVELS)
def test_every_aob_row_briefing_states_the_loop_once_at_every_guidance_level(world, row, level):
    """Where a cycle of the episode can run, the sentence carries the clause on the cycles recorded before the episode,
    with their number. On a row where none runs it carries no such clause, and the next sentence says that none runs."""
    cycles, blocks = world
    config = dataclasses.replace(preset_config(row), guidance_level=level)
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(7)
    profile = env.public_profile()
    text, noun = aob_briefing(profile, config), asset_text(profile)["unit_noun"]
    assert text.count("Every reading is a recorded cycle") == 1
    if cycles_run(config):
        assert text.count(replay_sentence(noun, SEGMENT, HISTORY_CYCLES)) == 1
        assert text.count(f", and the {n_cycles(HISTORY_CYCLES)} {BEFORE}.") == 1
    else:
        assert text.count(replay_sentence(noun, SEGMENT)) == 1 and BEFORE not in text
        assert "No cycle runs in this episode" in text


def test_the_rows_where_a_cycle_can_run_are_the_rows_with_a_clock():
    """Both kinds of rows exist on the ladder, so the test above reads the clause and its absence: a cycle can run on the
    rows with a clock and on no other row."""
    running = {row for row in ROWS if cycles_run(preset_config(row))}
    assert running == {row for row in ROWS if preset_config(row).has_clock()}
    assert running and running != set(ROWS)


@pytest.mark.parametrize("row", ROWS)
def test_the_cycles_recorded_before_the_episode_are_other_recordings_than_the_loop(world, row):
    """What the clause says of the engine of these rows: at reset the record holds HISTORY_CYCLES cycles, all before
    cycle 0, and none of their recordings is among those that the live window plays, however long it loops."""
    cycles, blocks = world
    env = RecordedRigEnv(cycles, blocks, preset_config(row))
    env.reset(7)
    assert env.recorded_cycles() == list(range(-HISTORY_CYCLES, 0))
    before = set(env.source.prehistory(env.initial))
    assert len(before) == HISTORY_CYCLES
    live = {index for index, _repeated in (env.source.play() for _ in range(3 * SEGMENT))}
    assert len(live) == SEGMENT and not live & before


def test_the_sentence_names_the_window_length_and_the_played_recordings_repeat_with_that_period(world):
    assert f"replays the same {n_cycles(SEGMENT)} in a loop" in replay_sentence("rig", SEGMENT)
    cycles, blocks = world
    env = HydraulicRigEnv(cycles, blocks, EpisodeConfig())
    env.reset(7)
    source = env.source
    played = [source.play() for _ in range(3 * SEGMENT)]
    indices = [index for index, _repeated in played]
    assert all(indices[i] == indices[i + SEGMENT] for i in range(2 * SEGMENT))
    assert len(set(indices[:SEGMENT])) == SEGMENT  # one loop holds SEGMENT distinct recordings
    assert [repeated for _index, repeated in played] == [False] * SEGMENT + [True] * (2 * SEGMENT)
    assert isinstance(source, RecordingSource)


def test_the_sentence_is_built_from_the_noun_and_the_two_lengths_of_the_kind_for_any_asset():
    """No word of the rig is written into the sentence: another asset's noun, another window length and another number of
    cycles recorded before the episode give their own, and the briefings take the two lengths from the loop of the evidence
    kind of the asset."""
    text = replay_sentence("compressor", 7)
    assert "the compressor replays the same 7 cycles in a loop" in text and "every 7 cycles" in text
    assert "rig" not in text and " 5 " not in text and BEFORE not in text
    with_history = replay_sentence("compressor", 7, 3)
    assert with_history == text[:-1] + f", and the 3 cycles {BEFORE}."
    assert replay_sentence("compressor", 7, 1).endswith(f", and the 1 cycle {BEFORE}.")
    assert "the same 1 cycle in a loop" in replay_sentence("motor", 1)
    kind = kinds.register_kind(kinds.EvidenceKind("made_up_loop", loader=records.load_recordings, source=RecordingSource,
                                                  record_keys=("label_keys", "stable_flag_stable"), loop=lambda profile: (7, 3)))
    data = json.loads(TINY_FILE.read_text(encoding="utf-8"))
    profile = register_profile(AssetProfile.from_json(dict(data, evidence_kind=kind.name)))
    try:
        assert readings_sentence({"asset_id": profile.asset_id}, "compressor") == text
        assert readings_sentence({"asset_id": profile.asset_id}, "compressor", with_history=True) == with_history
    finally:
        unregister_profile(profile.asset_id)
        kinds.unregister_kind(kind.name)
