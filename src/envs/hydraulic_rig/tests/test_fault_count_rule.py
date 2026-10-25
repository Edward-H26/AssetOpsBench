from pathlib import Path

import pytest

from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import TinyHydraulicEnv, n_faulty

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)
RULES = {(1, 1): 10, (2, 2): 37, (1, 2): 47}  # fault-count rule -> start combinations


def test_the_observation_states_the_fault_count_rule():
    assert TinyHydraulicEnv(CYCLES, BLOCKS).reset(0)["min_faulty"] == 1  # the default rule is one to two
    for low, high in RULES:
        obs = TinyHydraulicEnv(CYCLES, BLOCKS, min_faulty=low, max_faulty=high).reset(0)
        assert (obs["min_faulty"], obs["max_faulty"]) == (low, high)


def test_the_start_combinations_follow_the_fault_count_rule():
    assert len(TinyHydraulicEnv(CYCLES, BLOCKS).candidates) == 47
    for (low, high), count in RULES.items():
        env = TinyHydraulicEnv(CYCLES, BLOCKS, min_faulty=low, max_faulty=high)
        assert len(env.candidates) == count and all(low <= n_faulty(k) <= high for k in env.candidates)
        for seed in range(30):
            env.reset(seed)
            assert low <= n_faulty(env.initial) <= high


@pytest.mark.parametrize("low, high", [(2, 1), (3, 2), (0, 2)])
def test_a_fault_count_rule_outside_one_to_max_is_refused(low, high):
    with pytest.raises(ValueError, match="min_faulty"):
        TinyHydraulicEnv(CYCLES, BLOCKS, min_faulty=low, max_faulty=high)
