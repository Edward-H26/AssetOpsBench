"""Start states of an environment (docs/hydraulic_rig/environment.md): the (combination, window start, entry offset)
triples that env.reset can draw. A stable block of n cycles gives n - block_cap + 1 windows with block_cap entries each.
A seed table maps every start state to the smallest seed whose reset
reaches it, so an episode from any start state replays through the public reset(seed); for a scenario record, an
environment with `allowed` set to its one combination, the table lists the record's own seeds."""
MAX_SEED = 1_000_000  # far beyond the seeds that the start states of a scenario class need


def start_state(env):
    """(combination, window start, entry offset) of the current episode. env.py keeps the window and the offset only as
    the served slice and the offset, without an accessor, so they are read from those two attributes."""
    return env.initial, env.blocks[env.initial].index(env._block[0]), env._pos


def start_states(env):
    """Every (combination, window start, entry offset) that env.reset can draw."""
    states = []
    for combo in env.candidates:
        length = len(env.blocks[combo])
        cap = min(env.block_cap or length, length)
        states += [(combo, w, e) for w in range(length - cap + 1) for e in range(cap)]
    return states


def seed_table(env):
    """[(seed, combination, window start, entry offset)] in start-state order, each with the smallest seed reaching it."""
    wanted = set(start_states(env))
    first = {}
    for seed in range(MAX_SEED):
        env.reset(seed)
        first.setdefault(start_state(env), seed)
        if len(first) == len(wanted):
            return [(first[s], *s) for s in sorted(first)]
    raise RuntimeError(f"seeds 0 to {MAX_SEED - 1} reach {len(first)} of the {len(wanted)} start states")
