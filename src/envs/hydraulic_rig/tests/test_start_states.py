"""Start states and the seed table (src/envs/hydraulic_rig/start_states.py) on the v2 environment."""
from envs.hydraulic_rig.env import BLOCK_CAP
from envs.hydraulic_rig.start_states import seed_table, start_state, start_states


def replays(env, table):
    for seed, combo, w, e in table:
        env.reset(seed)
        if start_state(env) != (combo, w, e):
            return False
    return True


def test_a_seed_table_lists_every_start_state_once_and_each_seed_replays_its_start_state(make_env, world):
    cycles, blocks = world
    env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 1})
    table = seed_table(env)
    assert [(c, w, e) for _, c, w, e in table] == sorted(start_states(env)) and replays(env, table)
    per_combination = [sum(1 for _, c, _, _ in table if c == combo) for combo in env.candidates]
    assert per_combination == [(len(blocks[combo]) - BLOCK_CAP + 1) * BLOCK_CAP for combo in env.candidates]


def test_a_record_environment_has_its_own_seed_table(make_env):
    for combo in ((100, 100, 0, 90), (3, 100, 0, 130), (3, 73, 0, 130)):
        env = make_env(allowed=[list(combo)])
        table = seed_table(env)
        assert {c for _, c, _, _ in table} == {combo} and len(table) == len(start_states(env)) and replays(env, table)
