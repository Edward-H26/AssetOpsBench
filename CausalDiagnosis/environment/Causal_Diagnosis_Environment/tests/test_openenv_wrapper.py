"""The OpenEnv interface (CausalDiagnosis.environment.Causal_Diagnosis_Environment.openenv_wrapper): reset, step and state against the Gymnasium wrapper,
the typed models, the state contract, the containment of the verdict, and the import without openenv."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HIDDEN_FIELDS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import HydraulicRigGym
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.openenv_wrapper import (
    OPENENV_AVAILABLE,
    HydraulicRigAction,
    HydraulicRigObservation,
    HydraulicRigOpenEnv,
    HydraulicRigState,
)
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy

from CausalDiagnosis.datasets.tests.conftest import session_tables

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import keys_of

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

FACTS = default_facts_path()
from CausalDiagnosis.utils.paths import SRC
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
        assert observation.metadata == {"time": info["time"]}
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
    env.reset(seed=7, episode_id="episode-7")
    assert isinstance(env.state, HydraulicRigState)
    assert env.state.episode_id == "episode-7" and env.state.step_count == 0 and env.state.seed == 7
    env.step(HydraulicRigAction(tool_name="get_asset_profile"))
    assert env.state.step_count == 1
    env.reset(seed=7)
    again = HydraulicRigOpenEnv(EpisodeConfig(), FACTS)
    again.reset(seed=7)
    assert env.state.episode_id == again.state.episode_id and env.state.step_count == 0


def test_the_class_builds_without_arguments_for_the_openenv_server():
    """The server builds a session without arguments, which plays EpisodeConfig() on the development starts of its asset."""
    from CausalDiagnosis.datasets.splits.development_split import development_starts, starts_of_test_set
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import played_config

    env = HydraulicRigOpenEnv()
    assert env.gym.facts_path == str(FACTS) and env.gym.starts == "development"
    assert env.gym.config == played_config(EpisodeConfig(), FACTS, "development")
    world = env.gym.env
    assert {tuple(c) for c in env.gym.config.allowed} == set(development_starts(world.profile, set(world.blocks)))
    env.reset(seed=0)
    assert tuple(world.initial) not in starts_of_test_set(world.profile, world.blocks)


def test_wrapper_imports_and_runs_without_openenv():
    # sys.modules[name] = None makes import openenv raise ImportError whether or not it is installed
    code = "\n".join([
        "import sys",
        "sys.modules['openenv'] = None",
        *session_tables(),
        "import CausalDiagnosis.environment.Causal_Diagnosis_Environment.openenv_wrapper as openenv_env",
        "assert openenv_env.OPENENV_AVAILABLE is False",
        "env = openenv_env.HydraulicRigOpenEnv()",
        "observation = env.reset(seed=0)",
        "assert observation.done is False and observation.reward is None",
        "observation = env.step({'tool_name': 'get_asset_profile', 'arguments': {}})",
        "assert observation.reward == 0 and env.state.step_count == 1",
    ])
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(SRC.parent), env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC.parent), str(SRC)])))


@pytest.mark.skipif(not OPENENV_AVAILABLE, reason="openenv is not installed (the package group rl installs it)")
def test_core_types_are_used_when_openenv_is_installed():
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

    assert issubclass(HydraulicRigOpenEnv, Environment) and issubclass(HydraulicRigAction, Action)
    assert issubclass(HydraulicRigObservation, Observation) and issubclass(HydraulicRigState, State)


def test_no_observation_field_follows_the_hidden_downtime():
    """A free wait of one cycle charges the downtime of the hidden condition to the hidden ledger. Two episodes whose starts
    differ in that downtime answer the same wait with the same metadata, the time alone, so no field of an observation tells
    the hidden condition."""
    config = EpisodeConfig()
    seeds = {}
    for seed in range(40):
        env = HydraulicRigOpenEnv(config, FACTS)
        env.reset(seed=seed)
        hidden = env.gym.env.hidden
        seeds.setdefault(sum(config.level_rate(c, level) for c, level in zip(config.components, hidden)), seed)
        if len(seeds) == 2:
            break
    assert len(seeds) == 2, "the first seeds hold starts with two downtime rates"
    observed = []
    for seed in seeds.values():
        env = HydraulicRigOpenEnv(config, FACTS)
        env.reset(seed=seed)
        observation = env.step(HydraulicRigAction(tool_name="wait_for_notification", arguments={"timeout": 1}))
        assert set(observation.metadata) == {"time"}
        observed.append(observation.metadata)
    assert observed[0] == observed[1]
