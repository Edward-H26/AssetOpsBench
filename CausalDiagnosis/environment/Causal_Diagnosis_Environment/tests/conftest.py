"""Shared fixtures of the environment tests: the recorded cycles, the stable blocks and environment builders. The data file
is the release file of hyd_1, and in a clone that states that it holds no data the tests that need it are skipped."""
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

HAS_DATA = Path(default_facts_path()).exists()


@pytest.fixture(scope="session")
def world():
    if not HAS_DATA:
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


@pytest.fixture
def make_env(world):
    """An environment over the recordings with EpisodeConfig(**overrides)."""
    cycles, blocks = world

    def build(**overrides):
        return HydraulicRigEnv(cycles, blocks, EpisodeConfig(**overrides))
    return build


def truth(env):
    return dict(zip(COMPONENTS, env.initial))


def faulty_components(combination):
    return [c for i, c in enumerate(COMPONENTS) if combination[i] != HEALTHY[c]]


def repaired(combination, component):
    fixed = list(combination)
    fixed[COMPONENTS.index(component)] = HEALTHY[component]
    return tuple(fixed)


def submit_truth(env, evidence=()):
    return env.step({"type": "submit_diagnosis", "diagnosis": truth(env), "evidence_ids": list(evidence)})


def order(env, component, prediction_id="", evidence=()):
    return env.step({"type": "create_work_order", "component": component, "prediction_id": prediction_id, "evidence_ids": list(evidence)})


def read(env, group="tier1", cycles=1):
    return env.step({"type": "read_sensors", "sensor_group": group, "cycles": cycles})


def wait(env, timeout=None):
    """One call of wait_for_notification, which ends at the first notification or after timeout cycles."""
    return env.step({"type": "wait_for_notification", "timeout": timeout})


def advance(env, cycles):
    """Let the rig run the given cycles, waiting again after every notification as an agent would, and return the last
    answer with the summed charge, so that a test of the clock reads one stretch of time as one call."""
    target, charged = env.time + cycles, 0
    while True:
        obs, cost, done, info = wait(env, target - env.time)
        charged += cost
        if done or env.time >= target:
            return obs, charged, done, info


def summary(env, since):
    return env.step({"type": "summarize_changes_since", "cycle": since})


def finish(env, text="done"):
    return env.step({"type": "finish", "summary": text})


def keys_of(obj, skip=("action", "parameters", "processes")):
    """Every key of a nested JSON-like object, without the agent's own refused actions, the tool parameter tables of the
    contract, which carry argument names such as cycle, and the processes block of the rules, whose parameters sit
    under the name schedule that the hidden draw carries too."""
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(k)
            if k not in skip:
                found |= keys_of(v, skip)
    elif isinstance(obj, list):
        for v in obj:
            found |= keys_of(v, skip)
    return found


def events(log, name):
    return [e for e in log if e["event"] == name]
