"""The token clock on the action server: a call of the generated_tokens mode reads the total of the
episode's ledger of generated tokens, charges the tokens since the previous call at seconds_per_token by the rule of
generation_time, refuses an episode without a ledger by name, keeps the tokens and the total in the state and the total in
the transcript, and a replay of the transcript serves each call under its recorded total with no ledger; the briefing states
the rule in that mode alone. No test states a value computed from the rig data."""
import dataclasses
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import TOKEN_MODE
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing

needs_facts = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
START = 100.0
TIME = {"type": "get_current_time"}
RATE, THINK = 0.0625, 10  # a rate that binary floating point holds exactly, so that every charge below is exact
TOKENS = (160, 240, 400)  # 10, 15 and 25 seconds at RATE: one cycle, one cycle with 5 seconds carried, three cycles


def token_config(**changes):
    return dataclasses.replace(EpisodeConfig(), clock_mode=TOKEN_MODE, seconds_per_token=RATE, think_seconds_per_cycle=THINK, **changes)


def started(folder, config, ledger=True):
    path = folder / "episode.json"
    if ledger:
        token_clock.open_ledger(folder)
    with episode.recorded_clock(START, 5.0):
        episode.start(path, 3, config, default_facts_path())
        episode.hand_over(path)
    return path


def served(path, wall):
    return episode.call(path, "get_current_time", {}, TIME, now=wall, monotonic=wall)


def play(folder, gap=1.0):
    path = started(folder, token_config())
    texts = []
    for k, tokens in enumerate(TOKENS, start=1):
        token_clock.add(folder, tokens)
        texts.append(served(path, START + k * gap))
    return path, texts


@needs_facts
def test_the_token_clock_charges_the_generated_tokens_since_the_previous_call_at_the_asset_rate(tmp_path):
    path, _texts = play(tmp_path)
    state = episode.load_state(path)
    assert state["think_tokens"] == list(TOKENS) and state["generated_tokens_read"] == sum(TOKENS)
    assert state["think_seconds"] == [t * RATE for t in TOKENS] == [10.0, 15.0, 25.0]
    assert state["think_cycles"] == [1, 1, 3] and state["think_seconds_carried"] == 0.0
    assert state["think_seconds_monotonic"] == [0.0, 0.0, 0.0]  # the wall seconds between the calls are kept and not charged
    calls = [line for line in transcript.read(tmp_path) if line["kind"] == "call"]
    assert [line["generated_tokens"] for line in calls] == [160, 400, 800]


@needs_facts
def test_the_token_clock_charges_the_same_cycles_whatever_the_wall_seconds_between_calls(tmp_path):
    runs = []
    for gap in (1.0, 1000.0):
        folder = tmp_path / f"gap_{int(gap)}"
        folder.mkdir()
        path, texts = play(folder, gap)
        state = episode.load_state(path)
        runs.append((state["think_cycles"], state["think_seconds"], texts))
    assert runs[0] == runs[1]


@needs_facts
def test_a_token_clock_episode_whose_folder_has_no_ledger_is_refused_by_name(tmp_path):
    path = started(tmp_path, token_config(), ledger=False)
    token_clock.add(tmp_path, 160)  # an addition to a folder without a ledger leaves the folder as it is
    assert not (tmp_path / token_clock.NAME).exists()
    with pytest.raises(ValueError, match="charges the tokens that the harness of the agent writes to generated_tokens.jsonl"):
        served(path, START + 1)
    assert episode.load_state(path)["actions"] == []


@needs_facts
def test_a_transcript_replay_serves_each_token_call_under_its_recorded_total(tmp_path):
    played, again = tmp_path / "played", tmp_path / "again"
    played.mkdir()
    again.mkdir()
    path, _texts = play(played)
    calls = [line for line in transcript.read(played) if line["kind"] == "call"]
    replayed = started(again, token_config(), ledger=False)
    for line in calls:
        with episode.recorded_clock(line["wall"], line["wall"], generated=line["generated_tokens"]):
            episode.call(replayed, line["tool"], line["arguments"], TIME)
    assert not (again / token_clock.NAME).exists()
    assert episode.load_state(replayed)["think_cycles"] == episode.load_state(path)["think_cycles"]
    assert [line["answer_sha256"] for line in transcript.read(again) if line["kind"] == "call"] == [line["answer_sha256"] for line in calls]


@needs_facts
def test_the_briefing_states_the_token_rule_in_the_token_mode_alone(tmp_path):
    for mode, stated in ((TOKEN_MODE, True), ("generation_time", False), ("declared", False)):
        folder = tmp_path / mode
        folder.mkdir()
        config = dataclasses.replace(token_config(), clock_mode=mode, guidance_level=2)
        path = started(folder, config)
        profile = episode.replay(episode.load_state(path)).public_profile()
        text = briefing(profile, config)
        rule = f"and by the tokens you generate between two calls, each counted as {RATE} seconds, whose seconds add up"
        assert (rule in text) is stated and ("seconds_per_cycle, think_seconds_per_cycle and seconds_per_token" in text) is stated, mode
        assert ("seconds_per_token" in text) is stated, mode
