"""The Gymnasium wrapper (envs.hydraulic_rig.gym_env): determinism, the reward order of the oracle and the do
nothing policy, truncation at the horizon and at the step cap, the information barrier of the observation text,
and the import without gymnasium."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import default_facts_path
from envs.hydraulic_rig.env import HIDDEN_FIELDS
from envs.hydraulic_rig.gym_env import GYMNASIUM_AVAILABLE, HydraulicRigGym, parse_action
from envs.hydraulic_rig.policies import DoNothingPolicy, OraclePolicy
from envs.hydraulic_rig.verifier import released_from_env

from .conftest import keys_of

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

FACTS = default_facts_path()
SRC = Path(__file__).resolve().parents[3]
LABELS = ("cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure")  # the label keys of the facts file
# the seeded schedule and the seeded technician give the hidden processes something to leak
TIMED = replace(
    EpisodeConfig(),
    schedule=dict(EpisodeConfig().schedule, mode="seeded"),
    external_events=dict(EpisodeConfig().external_events, mode="seeded", probability=1.0),
)
SCRIPTED = [
    {"type": "get_asset_profile"},
    {"type": "read_sensors", "sensor_group": "tier3", "cycles": 2},
    {"type": "advance_time", "cycles": 12},
    {"type": "summarize_changes_since", "cycle": 0},
    {"type": "get_failure_modes", "component": ""},
    {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1},
    {"type": "finish", "summary": "scripted"},
]


def play(gym, policy, seed):
    """One episode through the wrapper: the observation texts, the rewards, the final flags and the final info."""
    text, _info = gym.reset(seed)
    texts, rewards = [text], []
    while True:
        text, reward, terminated, truncated, info = gym.step(policy.act(gym.observation))
        texts.append(text)
        rewards.append(reward)
        if terminated or truncated:
            return texts, rewards, (terminated, truncated), info


def observation_json(gym, text):
    """The JSON part of an observation text: a step text is the envelope, the reset text is the briefing followed by
    the profile. The hidden field names are checked as keys, since the briefing prose may use the words in another
    sense and the contract's parameter table (skipped by keys_of) names an argument cycle."""
    if text.startswith(gym.task_text + "\n\n"):
        return json.loads(text[len(gym.task_text) + 2:])
    return json.loads(text)


def assert_no_hidden_field(gym, text):
    """No hidden field and no label of the facts file as a key; a failure mode code may contain a label's words."""
    assert not keys_of(observation_json(gym, text)) & set(HIDDEN_FIELDS), text
    assert '"seed":' not in text and not any(f'"{label}":' in text for label in LABELS)


def test_same_seed_gives_the_same_texts_and_the_same_terminal_reward():
    a = HydraulicRigGym(EpisodeConfig(), FACTS)
    b = HydraulicRigGym(EpisodeConfig(), FACTS)
    texts_a, rewards_a, flags_a, info_a = play(a, OraclePolicy(a.env), 5)
    texts_b, rewards_b, flags_b, info_b = play(b, OraclePolicy(b.env), 5)
    assert texts_a == texts_b and rewards_a == rewards_b and flags_a == flags_b
    assert info_a["end_reason"] == info_b["end_reason"] and info_a["cost_oracle"] == info_b["cost_oracle"]
    # the same instance reproduces the episode after another episode ran on it
    play(a, DoNothingPolicy(), 6)
    texts_c, rewards_c, _flags, _info = play(a, OraclePolicy(a.env), 5)
    assert texts_c == texts_a and rewards_c == rewards_a


def test_seed_offset_shifts_the_seed_and_a_missing_seed_counts_episodes():
    gym = HydraulicRigGym(EpisodeConfig(), FACTS, seed_offset=100)
    _text, info = gym.reset(3)
    assert info["seed"] == 103
    _text, first = gym.reset()
    _text, second = gym.reset()
    assert (first["seed"], second["seed"]) == (101, 102)


def test_reward_is_zero_before_the_end_and_the_oracle_beats_do_nothing():
    gym = HydraulicRigGym(EpisodeConfig(), FACTS)
    for seed in range(3):
        _texts, oracle_rewards, _flags, oracle_info = play(gym, OraclePolicy(gym.env), seed)
        _texts, idle_rewards, _flags, idle_info = play(gym, DoNothingPolicy(), seed)
        assert all(r == 0 for r in oracle_rewards[:-1]) and all(r == 0 for r in idle_rewards[:-1])
        assert oracle_rewards[-1] > idle_rewards[-1]
        assert oracle_info["reward"] == oracle_rewards[-1] and idle_info["reward"] == idle_rewards[-1]
        assert "verdict" in oracle_info and "verdict" in idle_info


def test_advancing_to_the_horizon_truncates_the_episode():
    config = EpisodeConfig()
    gym = HydraulicRigGym(config, FACTS)
    gym.reset(0)
    _text, _reward, terminated, truncated, info = gym.step({"type": "advance_time", "cycles": config.horizon})
    assert (terminated, truncated) == (False, True) and info["end_reason"] == "horizon"
    assert gym.env.time >= config.horizon


def test_the_step_cap_truncates_the_episode_and_refused_steps_count():
    gym = HydraulicRigGym(replace(EpisodeConfig(), max_steps=3), FACTS)
    gym.reset(0)
    first = gym.step("not an action")
    second = gym.step('{"type": "get_asset_profile"}')
    third = gym.step("still not an action")
    assert "refused" in json.loads(first[0]) and "refused" not in json.loads(second[0])
    assert (first[2], first[3]) == (False, False) and (second[2], second[3]) == (False, False)
    assert (third[2], third[3]) == (False, True) and third[4]["end_reason"] == "max_steps"


def test_malformed_text_is_refused_logged_and_counted():
    gym = HydraulicRigGym(EpisodeConfig(), FACTS)
    gym.reset(1)
    _text, _reward, _terminated, _truncated, info = gym.step({"type": "get_asset_profile"})
    steps_left = gym.observation["steps_left"]
    assert "verdict" not in info
    text, reward, terminated, truncated, _info = gym.step("read the sensors please")
    answer = json.loads(text)
    assert "refused" in answer and reward == 0 and not terminated and not truncated
    assert answer["steps_left"] == steps_left - 1
    assert gym.env.log[-1]["event"] == "refused"
    assert gym.env.log[-1]["action"] == {"type": "malformed", "text": "read the sensors please"}


def test_parse_action_decodes_json_objects_and_wraps_everything_else():
    assert parse_action({"type": "finish", "summary": "done"}) == {"type": "finish", "summary": "done"}
    assert parse_action('{"type": "advance_time", "cycles": 2}') == {"type": "advance_time", "cycles": 2}
    assert parse_action("[1, 2]") == {"type": "malformed", "text": "[1, 2]"}
    assert parse_action("{broken") == {"type": "malformed", "text": "{broken"}
    assert parse_action(42) == {"type": "malformed", "text": "42"}


def test_no_hidden_field_appears_in_any_observation_text_or_in_the_released_evidence():
    gym = HydraulicRigGym(TIMED, FACTS)
    for seed in range(4):
        for policy in (OraclePolicy(gym.env), DoNothingPolicy()):
            texts, _rewards, _flags, _info = play(gym, policy, seed)
            for text in texts:
                assert_no_hidden_field(gym, text)
        text, _info = gym.reset(seed)
        assert_no_hidden_field(gym, text)
        for action in SCRIPTED:
            text, _reward, terminated, truncated, _info = gym.step(action)
            assert_no_hidden_field(gym, text)
            if terminated or truncated:
                break
        # the released evidence holds the public events after reset, so the seed and the drawn processes never enter it
        released = released_from_env(gym.env)
        assert released["events"] and all(e["event"] != "reset" for e in released["events"])
        assert not keys_of(released) & set(HIDDEN_FIELDS)


def test_wrapper_imports_and_runs_without_gymnasium():
    # sys.modules[name] = None makes import gymnasium raise ImportError whether or not it is installed
    code = "\n".join([
        "import sys",
        "sys.modules['gymnasium'] = None",
        "from envs.hydraulic_rig.config import EpisodeConfig",
        "from envs.hydraulic_rig.data import default_facts_path",
        "import envs.hydraulic_rig.gym_env as gym_env",
        "assert gym_env.GYMNASIUM_AVAILABLE is False",
        "gym = gym_env.HydraulicRigGym(EpisodeConfig(), default_facts_path())",
        "text, info = gym.reset(0)",
        "assert isinstance(text, str) and 'seed' in info",
        "text, reward, terminated, truncated, info = gym.step({'type': 'get_asset_profile'})",
        "assert reward == 0 and not terminated and not truncated",
    ])
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(SRC.parent), env=dict(os.environ, PYTHONPATH=str(SRC)))


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium is not installed")
def test_gymnasium_spaces_contain_the_texts():
    import gymnasium

    gym = HydraulicRigGym(EpisodeConfig(), FACTS)
    assert isinstance(gym, gymnasium.Env)
    text, _info = gym.reset(0)
    assert gym.observation_space.contains(text)
    action = json.dumps({"type": "read_sensors", "sensor_group": "tier1", "cycles": 1})
    assert gym.action_space.contains(action)
    text, _reward, _terminated, _truncated, _info = gym.step(action)
    assert gym.observation_space.contains(text)
