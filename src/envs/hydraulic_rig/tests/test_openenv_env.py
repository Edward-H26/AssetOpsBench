"""The OpenEnv interface (envs.hydraulic_rig.openenv_env): reset, step and state against the Gymnasium wrapper,
the typed models, the state contract, the containment of the verdict, and the import without openenv."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import default_facts_path
from envs.hydraulic_rig.env import HIDDEN_FIELDS
from envs.hydraulic_rig.gym_env import HydraulicRigGym
from envs.hydraulic_rig.openenv_env import (
    OPENENV_AVAILABLE,
    HydraulicRigAction,
    HydraulicRigObservation,
    HydraulicRigOpenEnv,
    HydraulicRigState,
)
from envs.hydraulic_rig.policies import OraclePolicy

from .conftest import keys_of

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

FACTS = default_facts_path()
SRC = Path(__file__).resolve().parents[3]
SCRIPTED = [
    ("get_asset_profile", {}),
    ("read_sensors", {"sensor_group": "tier1", "cycles": 1}),
    ("summarize_changes_since", {"cycle": 0}),
]


def tool_call(action):
    """The OpenEnv action of an environment action dict, as a policy of policies.py emits it."""
    return HydraulicRigAction(tool_name=action["type"], arguments={k: v for k, v in action.items() if k != "type"})


def play_oracle(env, seed):
    """The oracle policy through the OpenEnv interface; returns the terminal observation."""
    env.reset(seed=seed)
    policy = OraclePolicy(env.gym.env)
    while True:
        observation = env.step(tool_call(policy.act(env.gym.observation)))
        if observation.done:
            return observation


def play_oracle_gym(gym, seed):
    """The oracle policy through the Gymnasium wrapper; returns the terminal reward and info."""
    gym.reset(seed)
    policy = OraclePolicy(gym.env)
    while True:
        _text, reward, terminated, truncated, info = gym.step(policy.act(gym.observation))
        if terminated or truncated:
            return reward, info


def test_reset_and_steps_give_the_texts_of_the_gymnasium_wrapper():
    env = HydraulicRigOpenEnv(EpisodeConfig(), FACTS)
    gym = HydraulicRigGym(EpisodeConfig(), FACTS)
    observation = env.reset(seed=3)
    text, _info = gym.reset(3)
    assert isinstance(observation, HydraulicRigObservation)
    assert observation.text == text and observation.done is False and observation.reward is None
    for tool, arguments in SCRIPTED:
        observation = env.step(HydraulicRigAction(tool_name=tool, arguments=arguments))
        text, _reward, _terminated, _truncated, info = gym.step(dict(arguments, type=tool))
        assert observation.text == text and observation.done is False and observation.reward == 0
        assert observation.metadata == {"cost": info["cost"], "time": info["time"]}
    assert env.state.step_count == len(SCRIPTED) and env.state.time == gym.env.time


def test_the_terminal_observation_carries_the_reward_and_the_state_the_verdict():
    env = HydraulicRigOpenEnv(EpisodeConfig(), FACTS)
    env.reset(seed=4)
    assert env.state.verdict is None and env.state.done is False
    observation = play_oracle(env, 4)
    reward, info = play_oracle_gym(HydraulicRigGym(EpisodeConfig(), FACTS), 4)
    assert observation.done is True and observation.reward == reward
    assert env.state.done is True and env.state.end_reason == info["end_reason"] and env.state.verdict is not None
    assert "verdict" not in observation.metadata and "verdict" not in observation.text
    assert not keys_of(json.loads(observation.text)) & set(HIDDEN_FIELDS) and not set(observation.metadata) & set(HIDDEN_FIELDS)


def test_a_dict_action_is_accepted_and_an_unknown_tool_is_refused_and_counted():
    env = HydraulicRigOpenEnv(EpisodeConfig(), FACTS)
    env.reset(seed=0)
    observation = env.step({"tool_name": "get_asset_profile", "arguments": {}})
    assert '"refused":' not in observation.text and env.state.step_count == 1
    observation = env.step(HydraulicRigAction(tool_name="fly"))
    assert '"refused":' in observation.text and observation.done is False and env.state.step_count == 2


def test_the_action_model_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        HydraulicRigAction(tool_name="finish", summary="done")


def test_state_keeps_the_episode_id_and_names_the_seed_by_default():
    env = HydraulicRigOpenEnv(EpisodeConfig(), FACTS)
    env.reset(seed=7, episode_id="pilot-7")
    assert isinstance(env.state, HydraulicRigState)
    assert env.state.episode_id == "pilot-7" and env.state.step_count == 0 and env.state.seed == 7
    env.step(HydraulicRigAction(tool_name="get_asset_profile"))
    assert env.state.step_count == 1
    env.reset(seed=7)
    again = HydraulicRigOpenEnv(EpisodeConfig(), FACTS)
    again.reset(seed=7)
    assert env.state.episode_id == again.state.episode_id and env.state.step_count == 0


def test_the_class_builds_without_arguments_for_the_openenv_server():
    env = HydraulicRigOpenEnv()
    assert env.gym.facts_path == str(FACTS) and env.gym.config == EpisodeConfig()


def test_wrapper_imports_and_runs_without_openenv():
    # sys.modules[name] = None makes import openenv raise ImportError whether or not it is installed
    code = "\n".join([
        "import sys",
        "sys.modules['openenv'] = None",
        "import envs.hydraulic_rig.openenv_env as openenv_env",
        "assert openenv_env.OPENENV_AVAILABLE is False",
        "env = openenv_env.HydraulicRigOpenEnv()",
        "observation = env.reset(seed=0)",
        "assert observation.done is False and observation.reward is None",
        "observation = env.step({'tool_name': 'get_asset_profile', 'arguments': {}})",
        "assert observation.reward == 0 and env.state.step_count == 1",
    ])
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(SRC.parent), env=dict(os.environ, PYTHONPATH=str(SRC)))


@pytest.mark.skipif(not OPENENV_AVAILABLE, reason="openenv is not installed")
def test_core_types_are_used_when_openenv_is_installed():
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

    assert issubclass(HydraulicRigOpenEnv, Environment) and issubclass(HydraulicRigAction, Action)
    assert issubclass(HydraulicRigObservation, Observation) and issubclass(HydraulicRigState, State)
