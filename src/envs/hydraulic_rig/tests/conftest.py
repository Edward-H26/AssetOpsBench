"""Shared fixtures of the environment tests: the recorded cycles, the stable blocks and environment builders. The data file
is named by HYD_FACTS_PATH; the tests that need it are skipped without it."""
from pathlib import Path

import pytest

from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import HydraulicRigEnv

HAS_DATA = Path(default_facts_path()).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")


@pytest.fixture(scope="session")
def world():
    if not HAS_DATA:
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
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


def advance(env, cycles):
    return env.step({"type": "advance_time", "cycles": cycles})


def summary(env, since):
    return env.step({"type": "summarize_changes_since", "cycle": since})


def finish(env, text="done"):
    return env.step({"type": "finish", "summary": text})


def keys_of(obj, skip=("action", "parameters")):
    """Every key of a nested JSON-like object, without the agent's own refused actions and the tool parameter tables of the
    contract, which carry argument names such as cycle."""
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
