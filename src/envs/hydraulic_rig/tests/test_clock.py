"""The clock: measurement and repair durations, advance_time, and one hidden wear event."""
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import DEFAULT_CLOCK, TinyHydraulicEnv
from envs.hydraulic_rig.policies import OraclePolicy, run_episode
from envs.hydraulic_rig.scoring import score

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)


def env_with(degrade_at=None, **kwargs):
    return TinyHydraulicEnv(CYCLES, BLOCKS, max_steps=1_000, clock={**DEFAULT_CLOCK, "degrade_at": degrade_at}, **kwargs)


def test_without_a_clock_nothing_changes_and_seeds_start_the_same():
    plain, timed = TinyHydraulicEnv(CYCLES, BLOCKS), env_with([20, 200])
    for seed in range(20):
        a, b = plain.reset(seed), timed.reset(seed)
        assert plain.initial == timed.initial and plain._block == timed._block and plain._pos == timed._pos
        assert "clock" not in a and b["clock"] == 0 and "wear_at" not in str(b)
    _, cost, done, info = plain.step({"type": "advance_time", "cycles": 5})
    assert info == {"refused": "this episode has no clock"} and cost == 0 and not done


def test_measurements_and_advance_time_move_the_clock():
    env = env_with()
    env.reset(0)
    env.step({"type": "measure", "tier": 1})
    env.step({"type": "measure", "tier": 3})
    obs, *_ = env.step({"type": "advance_time", "cycles": 30})
    assert obs["clock"] == 1 + 5 + 30 and obs["history"][-1] == {"event": "advance_time", "cycles": 30, "clock": 6}
    for bad in (0, 501, True, "3", None):
        _, cost, _, info = env.step({"type": "advance_time", "cycles": bad})
        assert info["refused"].startswith("cycles must be") and cost == 0
    assert env.clock == 36


def test_the_wear_event_is_hidden_fires_on_time_and_changes_the_recordings():
    env = env_with([10, 10])
    env.reset(1)
    start = tuple(env.hidden)
    env.step({"type": "advance_time", "cycles": 9})
    assert tuple(env.hidden) == start
    env.step({"type": "advance_time", "cycles": 1})
    wear = next(e for e in env.log if e["event"] == "degrade")
    i = COMPONENTS.index(wear["component"])
    assert wear["clock"] == 10 and wear["from"] == start[i] and env.hidden[i] == wear["to"] != start[i]
    assert all(env.cycles[idx].combo == tuple(env.hidden) for idx in env._block)
    assert not any(e["event"] == "degrade" for e in env.observation()["history"])


def test_the_diagnosis_is_scored_against_the_state_at_submission():
    env = env_with([5, 5])
    env.reset(2)
    env.step({"type": "advance_time", "cycles": 5})
    now = tuple(env.hidden)
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, now))})
    env.step({"type": "finish"})
    s = score(env.log)
    assert now != s["initial"] and s["diagnosis_all_correct"] and s["diagnosis_target"] == now
    assert s["degradations"] == 1 and s["cycles"] == 5 and s["steps"] == 3


def test_a_replacement_takes_repair_cycles():
    env = env_with()
    env.reset(4)
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})
    faulty = next(c for i, c in enumerate(COMPONENTS) if env.initial[i] != HEALTHY[c])
    env.step({"type": "replace", "component": faulty})
    assert env.clock == DEFAULT_CLOCK["repair_cycles"]


def test_wear_stays_inside_allowed():
    start = next(k for k in BLOCKS if sum(k[i] != HEALTHY[c] for i, c in enumerate(COMPONENTS)) == 1)
    env = env_with([1, 1], allowed={start})
    env.reset(0)
    env.step({"type": "advance_time", "cycles": 5})
    assert tuple(env.hidden) == start and not any(e["event"] == "degrade" for e in env.log)


def test_the_oracle_still_recovers_the_rig_with_the_clock():
    env = env_with()
    for seed in range(10):
        s = score(run_episode(env, OraclePolicy(env), seed))
        assert s["recovered"] and s["diagnosis_all_correct"]
