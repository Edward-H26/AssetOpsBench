"""The clock: tool durations, advance_time, the hidden degradation schedule and replace as a work order."""
import copy
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import DEFAULT_CLOCK, HEALTHY_COMBO, TinyHydraulicEnv, worse_level
from envs.hydraulic_rig.policies import OraclePolicy, run_episode
from envs.hydraulic_rig.scoring import score

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)


def clock(**changes):
    cfg = copy.deepcopy(DEFAULT_CLOCK)
    cfg.update(changes)
    return cfg


def env_with(**changes):
    return TinyHydraulicEnv(CYCLES, BLOCKS, budget=10_000, max_steps=1_000, clock=clock(**changes))


def test_worse_level_moves_away_from_healthy():
    assert worse_level("cooler", 100) == 20 and worse_level("cooler", 3) is None
    assert worse_level("pump", 0) == 1 and worse_level("pump", 2) is None
    assert worse_level("accumulator", 130) == 115 and worse_level("valve", 73) is None


def test_without_a_clock_nothing_changes():
    plain, timed = TinyHydraulicEnv(CYCLES, BLOCKS), env_with(degradation=None)
    for seed in range(20):
        a, b = plain.reset(seed), timed.reset(seed)
        assert plain.initial == timed.initial and plain._block == timed._block and plain._pos == timed._pos
        assert "clock" not in a and b["clock"] == 0 and b["clock_rules"]["degradation"] is False
    obs, cost, done, info = plain.step({"type": "advance_time", "cycles": 5})
    assert info == {"refused": "this episode has no clock"} and cost == 0 and not done


def test_measurements_take_their_duration_and_advance_time_moves_the_clock():
    env = env_with(degradation=None)
    env.reset(0)
    env.step({"type": "measure", "tier": 1})
    env.step({"type": "measure", "tier": 3})
    assert env.clock == 1 + 5
    obs, *_ = env.step({"type": "advance_time", "cycles": 30})
    assert obs["clock"] == 36 and obs["history"][-1] == {"event": "advance_time", "cycles": 30, "clock": 6}
    assert [e["clock"] for e in obs["history"] if e["event"] == "measure"] == [0, 1]
    for bad in (0, -1, 501, True, 2.0, "3", None):
        _, cost, done, info = env.step({"type": "advance_time", "cycles": bad})
        assert info["refused"].startswith("cycles must be") and cost == 0 and not done
    assert env.clock == 36


def test_the_schedule_is_hidden_and_seeded():
    env = env_with()
    obs = env.reset(3)
    assert "schedule" not in str(obs)
    first = list(env.log[0]["schedule"])
    env.reset(3)
    assert env.log[0]["schedule"] == first and len(first) == 1 and 20 <= first[0][0] <= 200


def test_a_degradation_fires_on_time_and_changes_the_served_recordings():
    env = env_with(degradation={"events": 1, "min_cycle": 10, "max_cycle": 10})
    env.reset(1)
    start = tuple(env.hidden)
    env.step({"type": "advance_time", "cycles": 9})
    assert tuple(env.hidden) == start
    env.step({"type": "advance_time", "cycles": 1})
    drop = next(e for e in env.log if e["event"] == "degrade")
    assert drop["clock"] == 10 and drop["component"] is not None
    i = COMPONENTS.index(drop["component"])
    assert drop["from"] == start[i] and drop["to"] == worse_level(drop["component"], start[i]) == env.hidden[i]
    assert all(env.cycles[idx].combo == tuple(env.hidden) for idx in env._block)
    assert not any(e["event"] == "degrade" for e in env.observation()["history"])


def test_the_diagnosis_is_scored_against_the_combination_at_the_submission():
    env = env_with(degradation={"events": 1, "min_cycle": 5, "max_cycle": 5})
    env.reset(2)
    initial = env.initial
    env.step({"type": "advance_time", "cycles": 5})
    now = tuple(env.hidden)
    assert now != initial
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, now))})
    env.step({"type": "finish"})
    s = score(env.log)
    assert s["diagnosis_all_correct"] and s["diagnosis_target"] == now and s["initial"] == initial
    assert s["degradations"] == 1 and s["cycles"] == 5 and s["steps"] == 3


def test_replace_is_a_work_order_with_lead_time_and_downtime():
    env = env_with(degradation=None, lead_time=10, repair_cycles=5)
    env.reset(4)
    combo = env.initial
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, combo))})
    faulty = [c for i, c in enumerate(COMPONENTS) if combo[i] != HEALTHY[c]]
    obs, cost, *_ = env.step({"type": "replace", "component": faulty[0]})
    assert env.clock == 15 and obs["history"][-1]["clock"] == 0
    assert cost == env.prices["parts"][faulty[0]] + env.prices["downtime"]
    env.step({"type": "finish"}) if not env.done else None
    s = score(env.log)
    assert s["downtime_cycles"] == 5 and s["fault_cycles"] == 10 and s["cycles"] == 15


def test_fault_cycles_cost_money_and_can_end_the_episode():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, budget=50, clock=clock(degradation=None, fault_cost_per_cycle=1))
    env.reset(0)
    obs, cost, done, info = env.step({"type": "advance_time", "cycles": 40})
    assert cost == 40 and not done and obs["budget_left"] == 10
    obs, cost, done, info = env.step({"type": "advance_time", "cycles": 40})
    assert done and info["reason"] == "budget"
    assert score(env.log)["cost"] == env.cost == 80


def test_degradation_stays_inside_allowed():
    start = next(k for k in BLOCKS if sum(k[i] != HEALTHY[c] for i, c in enumerate(COMPONENTS)) == 1)
    env = TinyHydraulicEnv(CYCLES, BLOCKS, budget=10_000, max_steps=1_000, allowed={start},
                           clock=clock(degradation={"events": 3, "min_cycle": 1, "max_cycle": 3}))
    env.reset(0)
    env.step({"type": "advance_time", "cycles": 5})
    assert tuple(env.hidden) == start
    assert [e["component"] for e in env.log if e["event"] == "degrade"] == [None, None, None]


def test_the_oracle_still_recovers_the_rig_with_the_clock():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, clock=clock(degradation=None))
    for seed in range(10):
        log = run_episode(env, OraclePolicy(env), seed)
        s = score(log)
        assert s["recovered"] and s["diagnosis_all_correct"] and s["final"] == HEALTHY_COMBO


@pytest.mark.parametrize("bad", [
    {"lead_time": -1}, {"max_advance": 0}, {"fault_cost_per_cycle": float("nan")},
    {"durations": {"measure": {"1": 1}}}, {"degradation": {"events": 1, "min_cycle": 9, "max_cycle": 3}}, {"extra": 1},
])
def test_a_bad_clock_raises(bad):
    with pytest.raises(ValueError, match="invalid clock"):
        TinyHydraulicEnv(CYCLES, BLOCKS, clock=clock(**bad))
