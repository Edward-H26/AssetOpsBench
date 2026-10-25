from pathlib import Path

import pytest

from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import TinyHydraulicEnv
from envs.hydraulic_rig.start_states import seed_table, start_state, start_states

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)


def replays(env, table):
    for seed, combo, w, e in table:
        env.reset(seed)
        if start_state(env) != (combo, w, e):
            return False
    return True


def test_a_seed_table_lists_every_start_state_once_and_each_seed_replays_its_start_state():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)  # the 47 combinations with one or two faults
    table = seed_table(env)
    assert [(c, w, e) for _, c, w, e in table] == sorted(start_states(env)) and replays(env, table)
    per_combination = [sum(1 for _, c, _, _ in table if c == combo) for combo in env.candidates]
    cap = env.block_cap  # a block of n stable cycles gives n - cap + 1 windows with cap entries each
    assert per_combination == [(len(BLOCKS[combo]) - cap + 1) * cap for combo in env.candidates]


def test_a_record_environment_has_its_own_seed_table():
    for combo in ((100, 100, 0, 90), (3, 100, 0, 130), (3, 73, 0, 130)):
        env = TinyHydraulicEnv(CYCLES, BLOCKS, allowed={combo})
        table = seed_table(env)
        assert {c for _, c, _, _ in table} == {combo} and len(table) == len(start_states(env)) and replays(env, table)
