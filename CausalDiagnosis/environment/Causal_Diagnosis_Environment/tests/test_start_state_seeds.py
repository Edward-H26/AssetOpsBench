"""Start states and the seed table (CausalDiagnosis/environment/Causal_Diagnosis_Environment/start_state_seeds.py) on the v2 environment."""
import json
import random
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.tests.conftest import tiny, tiny_data, tiny_recordings  # noqa: F401
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as sampler
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import BLOCK_CAP, HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import EVALUATION_POOL, combination_seeds, sampled_combination_seeds, seed_table, start_state, start_states

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.test_diagnosability_label import ObservationModel

# The start combinations that reset draws for seeds 0 to 19 under the default fault rule, recorded before the
# repair extensions. They depend on the seed and the candidate list only, not on a recorded value, and every run folder and
# the sampled evaluation set depend on them, so a change to the draws at reset shows here first.
FIRST_STARTS = [(100, 80, 0, 90), (3, 100, 2, 130), (3, 100, 0, 90), (20, 100, 0, 130), (20, 100, 0, 130), (100, 100, 1, 90),
                (100, 100, 0, 90), (100, 73, 0, 115), (20, 100, 0, 115), (100, 80, 2, 130), (100, 100, 0, 90), (100, 80, 1, 130),
                (100, 90, 0, 90), (20, 100, 1, 130), (3, 100, 0, 130), (20, 100, 0, 100), (100, 73, 2, 130), (100, 90, 0, 130),
                (20, 90, 0, 130), (100, 100, 2, 90)]


def replays(env, table):
    for seed, combo, w, e in table:
        env.reset(seed)
        if start_state(env) != (combo, w, e):
            return False
    return True


def test_the_start_combinations_of_seeds_0_to_19_stay_as_recorded(make_env):
    env = make_env()  # the static preset changes no field that the draw of the start reads
    drawn = []
    for seed in range(len(FIRST_STARTS)):
        env.reset(seed)
        drawn.append(env.initial)
    assert drawn == FIRST_STARTS


def test_a_seed_table_lists_every_start_state_once_and_each_seed_replays_its_start_state(make_env, world):
    cycles, blocks = world
    env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 1})
    table = seed_table(env)
    assert [(c, w, e) for _, c, w, e in table] == sorted(start_states(env)) and replays(env, table)
    per_combination = [sum(1 for _, c, _, _ in table if c == combo) for combo in env.candidates]
    assert per_combination == [BLOCK_CAP for _combo in env.candidates]  # every window starts at 0, whatever the length of the block


def test_a_record_environment_has_its_own_seed_table(make_env):
    for combo in ((100, 100, 0, 90), (3, 100, 0, 130), (3, 73, 0, 130)):
        env = make_env(allowed=[list(combo)])
        table = seed_table(env)
        assert {c for _, c, _, _ in table} == {combo} and len(table) == len(start_states(env)) and replays(env, table)


def test_combination_seeds_draws_every_candidate_the_asked_number_of_times_with_its_smallest_seeds(make_env):
    env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 2})
    seeds = combination_seeds(env, 2)
    drawn = []
    for seed in seeds:
        env.reset(seed)
        drawn.append(env.initial)
    assert len(seeds) == 2 * len(env.candidates) and sorted(set(seeds)) == sorted(seeds)
    assert drawn == [combo for combo in env.candidates for _ in range(2)]
    smaller = [s for s in range(max(seeds) + 1) if s not in seeds]
    for seed in smaller:  # every skipped seed draws a combination that already had its two smaller seeds
        env.reset(seed)
        assert sum(1 for s in seeds if s < seed and drawn[seeds.index(s)] == env.initial) == 2


def test_sampled_combination_seeds_draws_every_candidate_once_from_its_pool_without_the_reserved_seed_and_replays(make_env):
    env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 2})
    seeds = sampled_combination_seeds(env, master_seed=42, pool=4)
    assert len(seeds) == len(env.candidates)
    pools = {combo: [] for combo in env.candidates}
    for seed in range(max(seeds) + 1):
        env.reset(seed)
        if len(pools[env.initial]) < 4:
            pools[env.initial].append(seed)
    development = combination_seeds(env, 1)
    for seed, combo, dev in zip(seeds, env.candidates, development):
        env.reset(seed)
        assert env.initial == combo and seed in pools[combo][1:] and seed != dev == pools[combo][0]
    assert sampled_combination_seeds(env, master_seed=42, pool=4) == seeds
    assert sampled_combination_seeds(env, master_seed=0, pool=4) != seeds
    clashing = [combo for combo in env.candidates if _state(env, pools[combo][1]) == _state(env, pools[combo][0])]
    assert clashing  # with one usable seed per pool some start could only replay the episode of its reserved seed
    with pytest.raises(ValueError, match="episode of its reserved seed"):
        sampled_combination_seeds(env, master_seed=42, pool=2)
    with pytest.raises(ValueError, match="at least two seeds"):
        sampled_combination_seeds(env, master_seed=42, pool=1)


def test_combination_seeds_after_the_evaluation_pool_take_the_next_smallest_seeds_of_every_candidate(make_env):
    """The rollouts of the separability label play seeds outside the evaluation pool of every start, the smallest after the
    pool, so no label reads the recordings of an evaluation episode."""
    env = make_env(fault_rule={"min_faulty": 1, "max_faulty": 2})
    pool = combination_seeds(env, EVALUATION_POOL)
    after = combination_seeds(env, 2, after=EVALUATION_POOL)
    assert len(after) == 2 * len(env.candidates) and not set(after) & set(pool)
    for i, combo in enumerate(env.candidates):
        own, own_pool = after[2 * i:2 * i + 2], pool[EVALUATION_POOL * i:EVALUATION_POOL * (i + 1)]
        assert min(own) > max(own_pool)
        for seed in own:
            env.reset(seed)
            assert env.initial == combo
        # every seed between the pool and the two draws another combination
        for seed in range(max(own_pool) + 1, max(own)):
            env.reset(seed)
            assert seed in own or env.initial != combo
    assert combination_seeds(env, 2, after=0) == combination_seeds(env, 2)


# The draw of the master seed 42 on the default configuration, the test set with one episode per start as it stood before the
# replicates. It depends on the seed and the candidate list only, as FIRST_STARTS does, so a change of the
# sampler shows here first.
TEST_SEEDS_OF_42 = [520, 104, 57, 327, 66, 91, 164, 160, 304, 82, 515, 308, 179, 39, 375, 114, 77, 52, 115, 280, 131, 374, 199, 76, 210, 148,
                    315, 517, 251, 500, 189, 83, 502, 191, 163, 287, 313, 101, 487, 264, 307, 384, 407, 158, 241, 106, 283]
# The starts where that draw enters at the start state of the episode of the reserved seed, with the drawn seed and the seed the first
# replicate takes instead, the first seed of the pool off that start state. Seed 4 of
# (20, 100, 0, 130) became the smallest inseparable seed of the evaluation set, so fixed_starts.json moved with it.
MOVED_OFF_DEVELOPMENT = {(3, 100, 0, 115): (91, 121), (3, 100, 0, 130): (164, 183), (20, 100, 0, 130): (114, 4),
                         (100, 80, 2, 130): (500, 40), (100, 90, 1, 130): (163, 94), (100, 90, 2, 130): (287, 35)}


def _state(env, seed):
    env.reset(seed)
    return start_state(env)


def test_the_first_replicate_is_the_draw_of_the_master_seed_42_as_it_stood_unless_the_draw_replays_the_episode_of_the_reserved_seed(make_env):
    """Replicate one is the test set as it stood, the draw of the master seed 42 from the eight smallest seeds of every start
    less its reserved seed, except on a start where that draw enters at the start state of the episode of the reserved seed,
    which on a row without hidden processes is that episode itself: there it takes the first seed of the pool that does
    not, and the draws of every other start stay as they were."""
    env = make_env()
    first = sampled_combination_seeds(env, 42, 8)
    assert first == sampled_combination_seeds(env) == sampler.replicate_seeds(env, 1) and len(first) == len(TEST_SEEDS_OF_42)
    pools = dict(zip(env.candidates, (combination_seeds(env, 8)[8 * i:8 * i + 8] for i in range(len(env.candidates)))))
    for seed, drawn, combo in zip(first, TEST_SEEDS_OF_42, env.candidates):
        development = _state(env, pools[combo][0])
        if _state(env, drawn) != development:
            assert seed == drawn  # the draw as it stood
        else:
            assert seed == next(s for s in pools[combo][1:] if _state(env, s) != development)
    assert {combo: (drawn, seed) for seed, drawn, combo in zip(first, TEST_SEEDS_OF_42, env.candidates) if seed != drawn} == MOVED_OFF_DEVELOPMENT


def test_every_start_plays_its_replicates_at_distinct_start_states_in_replicate_order_and_they_replay(make_env):
    """The five replicates of a start are five distinct start states of its block, one entry offset each, flat
    in replicate order (every start under the first replicate, then every start under the second), the first replicate the
    draw of the master seed, and replicate_seeds names the seeds of the one replicate that one run of a grid line plays."""
    env = make_env()
    n = len(env.candidates)
    five = sampled_combination_seeds(env, 42, 8, sampler.REPLICATES)
    assert sampler.REPLICATES == 5 and len(five) == 5 * n and five[:n] == sampled_combination_seeds(env, 42, 8)
    assert [sampler.replicate_seeds(env, k) for k in range(1, 6)] == [five[(k - 1) * n:k * n] for k in range(1, 6)]
    assert sampled_combination_seeds(env, 42, 8, 3) == five[:3 * n] and sampled_combination_seeds(env, 42, 8, 5) == five
    for i, combo in enumerate(env.candidates):
        states = [_state(env, seed) for seed in five[i::n]]
        assert {state[0] for state in states} == {combo} and len(set(states)) == 5
    with pytest.raises(ValueError, match="replicates"):
        sampled_combination_seeds(env, 42, 8, 6)


def test_no_replicate_is_a_rollout_of_the_separability_label_or_replays_the_episode_of_the_reserved_seed(make_env):
    """No seed of a replicate is a rollout seed of the start's separability label, the ROLLOUTS seeds after the
    evaluation pool in the label's static configuration, and no replicate enters at the start state of the episode of the reserved
    seed, the start's smallest seed, so no label rollout and no episode of a reserved seed is a test episode. Every further replicate is the
    smallest seed of its start in seed order that keeps both rules and enters at a start state that no replicate before it plays."""
    env = make_env()
    n = len(env.candidates)
    five = sampled_combination_seeds(env, 42, 8, 5)
    label_env = HydraulicRigEnv(sorted(env.cycles.values(), key=lambda c: c.index), env.blocks, separability.static_config(env.config))
    labels = combination_seeds(label_env, separability.ROLLOUTS, after=EVALUATION_POOL)
    rollouts = {combo: labels[i * separability.ROLLOUTS:(i + 1) * separability.ROLLOUTS] for i, combo in enumerate(label_env.candidates)}
    development = combination_seeds(env, 1)
    states = [_state(env, seed) for seed in range(max(five) + 1)]
    for i, combo in enumerate(env.candidates):
        own = five[i::n]
        assert not set(own) & set(rollouts[combo])
        assert states[development[i]] not in {states[seed] for seed in own}
        played, expected = {states[development[i]], states[own[0]]}, []
        for seed, state in enumerate(states):
            if len(expected) < 4 and state[0] == combo and seed not in rollouts[combo] and state not in played:
                expected.append(seed)
                played.add(state)
        assert expected == own[1:]


def test_the_rollout_seeds_the_replicates_avoid_are_those_the_label_table_plays_where_the_config_allows_fewer_starts(world):
    """The label of a configuration is computed in its static counterpart with every start of the fault rule, so where the
    configuration allows fewer starts the seeds that draw a start differ, and label_seeds names the seeds that the label table
    records for every start, which the replicates of the allowed starts leave out."""
    cycles, blocks = world
    config = EpisodeConfig(fault_rule={"min_faulty": 1, "max_faulty": 1}, allowed=[[3, 100, 0, 130], [100, 100, 0, 90], [100, 73, 0, 130]])
    table = separability.compute_table(config, cycles, blocks, classifiers=ObservationModel(cycles, blocks, sharp=False))
    env = HydraulicRigEnv(cycles, blocks, config)
    labels = sampler.label_seeds(env)
    assert len(labels) == len(table["labels"]) > len(env.candidates)
    assert all(labels[combo] == set(table["labels"][separability.combo_key(combo)]["seeds"]) for combo in labels)
    five = sampled_combination_seeds(env, 42, 8, 5)
    assert all(not set(five[i::len(env.candidates)]) & labels[combo] for i, combo in enumerate(env.candidates))


def test_training_plays_the_start_states_of_its_fold_that_no_replicate_of_the_test_set_plays(make_env):
    """The one exclusion that the stream of the online loop of GEPA and the fine tuning share: the start states of
    the combinations that the model of a fold trains on that no replicate of the test set plays, five of the ten of every block, 70
    per model on its 14 training combinations, the episodes of the reserved seeds among them, since no test number scores the
    episode of a reserved seed or a label rollout. The split is the rule with K = 2 and seed 0, the rig's registered file."""
    env = make_env()
    n = len(env.candidates)
    five = sampled_combination_seeds(env, 42, 8, sampler.REPLICATES)
    tested = {_state(env, seed) for seed in five}
    folds = sampler.combination_folds(env, 2, 0)
    development = combination_seeds(env, 1)
    for fold in sampler.split_folds(folds):
        trains = [combo for combo in env.candidates if fold in folds[combo][1]]
        every = {state for state in start_states(env) if state[0] in trains}
        training = sampler.training_start_states(env, folds, fold)
        assert training == sorted(training) and len(training) == len(set(training)) == (BLOCK_CAP - sampler.REPLICATES) * len(trains) == 70
        assert not set(training) & tested and set(training) | (tested & every) == every
        assert {_state(env, development[i]) for i, combo in enumerate(env.candidates) if combo in trains} <= set(training)
        assert sampler.training_start_states(env, folds, fold, 42, 8) == training
    assert len(five) == sampler.REPLICATES * n



# ----- the start state of a seed without a reset, and the scans that read it -----

from CausalDiagnosis.utils.paths import CONFIGS, preset_files
SEEDS_CHECKED = 300
RESTRICTED = {"fault_rule": {"min_faulty": 1, "max_faulty": 1}, "allowed": [[3, 100, 0, 130], [100, 100, 0, 90], [100, 73, 0, 130]]}


def _starts_agree(build):
    """start_of on an environment before any reset, on one after the reset of another seed and after its own reset gives
    the state that reset(seed) reaches, for the first SEEDS_CHECKED seeds."""
    drawn, reached, mixed = build(), build(), build()
    before = [drawn.start_of(seed) for seed in range(SEEDS_CHECKED)]
    states = []
    for seed in range(SEEDS_CHECKED):
        reached.reset(seed)
        states.append(start_state(reached))
    after = []
    for seed in range(SEEDS_CHECKED):
        mixed.reset((seed + 1) % SEEDS_CHECKED)
        after.append(mixed.start_of(seed))
    return before == states == after == [reached.start_of(seed) for seed in range(SEEDS_CHECKED)]


def test_the_start_of_a_seed_is_the_start_that_its_reset_reaches(world, tiny, tiny_recordings):
    cycles, blocks = world
    presets = sorted(preset_files(CONFIGS).values())
    assert presets
    for path in presets:
        config = layers.compose(json.loads(path.read_text(encoding="utf-8")))
        assert _starts_agree(lambda: HydraulicRigEnv(cycles, blocks, config)), path.name
    tiny_cycles = records.load_cycles(tiny_recordings, tiny)
    assert _starts_agree(lambda: HydraulicRigEnv(tiny_cycles, records.build_blocks(tiny_cycles), tiny.default_config()))


def test_two_environments_keep_their_own_start_states(make_env):
    """The same seed on two configurations with different start sets reaches two states, each the state of its own reset,
    so a store of start states shared between environments fails."""
    full, few = make_env(), make_env(**RESTRICTED)
    differing = [seed for seed in range(50) if full.start_of(seed) != few.start_of(seed)]
    assert differing
    for env in (full, few):
        for seed in differing:
            env.reset(seed)
            assert env.start_of(seed) == start_state(env)


# The scans as they stood before the start state of a seed was read without a reset, the test's own copies, every one
# reading the start state through a reset.

def _reset_state(env, seed):
    env.reset(seed)
    return start_state(env)


def _base_combination_seeds(env, per_combination=1, after=0):
    found = {combo: [] for combo in env.candidates}
    drawn = dict.fromkeys(env.candidates, 0)
    for seed in range(sampler.MAX_SEED):
        env.reset(seed)
        drawn[env.initial] += 1
        seeds = found[env.initial]
        if after < drawn[env.initial] and len(seeds) < per_combination:
            seeds.append(seed)
            if all(len(s) == per_combination for s in found.values()):
                return [seed for combo in env.candidates for seed in found[combo]]
    raise RuntimeError("the seeds ran out")


def _base_sampled_combination_seeds(env, master_seed=42, pool=EVALUATION_POOL, replicates=1):
    found = {combo: [] for combo in env.candidates}
    for seed in range(sampler.MAX_SEED):
        env.reset(seed)
        seeds = found[env.initial]
        if len(seeds) < pool:
            seeds.append(seed)
            if all(len(s) == pool for s in found.values()):
                break
    rng = random.Random(master_seed)
    first = [_base_off_reserved(env, found[combo], rng.choice(found[combo][1:])) for combo in env.candidates]
    if replicates == 1:
        return first
    further = _base_further_replicates(env, first, replicates - 1)
    return first + [further[combo][k] for k in range(replicates - 1) for combo in env.candidates]


def _base_off_reserved(env, pool, seed):
    reserved = _reset_state(env, pool[0])
    if _reset_state(env, seed) != reserved:
        return seed
    for other in pool[1:]:
        if _reset_state(env, other) != reserved:
            return other
    raise ValueError("every seed of the pool replays the start state of the episode of its reserved seed")


def _base_further_replicates(env, first, count):
    labels = _base_label_seeds(env)
    played = {}
    for seed in (*_base_combination_seeds(env, 1), *first):
        state = _reset_state(env, seed)
        played.setdefault(state[0], set()).add(state)
    out = {combo: [] for combo in env.candidates}
    for seed in range(sampler.MAX_SEED):
        if all(len(seeds) == count for seeds in out.values()):
            return out
        state = _reset_state(env, seed)
        combo = state[0]
        if len(out[combo]) < count and seed not in labels[combo] and state not in played[combo]:
            out[combo].append(seed)
            played[combo].add(state)
    raise RuntimeError("the seeds ran out")


def _base_label_seeds(env):
    label_env = HydraulicRigEnv(sorted(env.cycles.values(), key=lambda c: c.index), env.blocks, separability.static_config(env.config))
    seeds = _base_combination_seeds(label_env, separability.ROLLOUTS, after=EVALUATION_POOL)
    rollouts = separability.ROLLOUTS
    return {combo: set(seeds[i * rollouts:(i + 1) * rollouts]) for i, combo in enumerate(label_env.candidates)}


def _base_seed_table(env):
    wanted = set(start_states(env))
    first = {}
    for seed in range(sampler.MAX_SEED):
        env.reset(seed)
        first.setdefault(start_state(env), seed)
        if len(first) == len(wanted):
            return [(first[s], *s) for s in sorted(first)]
    raise RuntimeError("the seeds ran out")


def _base_training_start_states(env, split, fold, master_seed=42, pool=EVALUATION_POOL):
    tested = {_reset_state(env, seed) for seed in _base_sampled_combination_seeds(env, master_seed, pool, sampler.REPLICATES)}
    trains = {combination for combination, (_f, models) in split.items() if fold in models}
    return [state for state in sorted(start_states(env)) if state[0] in trains and state not in tested]


def _base_reserved_seeds(env):
    return set(_base_combination_seeds(env, 1)) | {seed for seeds in _base_label_seeds(env).values() for seed in seeds}


def _base_training_seeds(env, split, sample, fold):
    states = _base_training_start_states(env, split, fold)
    reserved = _base_reserved_seeds(env)
    found = {state: [] for state in states}
    for seed in range(sampler.MAX_SEED):
        if all(len(seeds) == sample for seeds in found.values()):
            return [found[state][-1] for state in states]
        if seed in reserved:
            continue
        state = _reset_state(env, seed)
        if state in found and len(found[state]) < sample:
            found[state].append(seed)
    raise RuntimeError("the seeds ran out")


def _base_training_stream(env, split, count, fold):
    states = set(_base_training_start_states(env, split, fold))
    reserved = _base_reserved_seeds(env)
    seeds = []
    for seed in range(sampler.MAX_SEED):
        if len(seeds) == count:
            break
        if seed not in reserved and _reset_state(env, seed) in states:
            seeds.append(seed)
    return seeds


@pytest.mark.parametrize("overrides", [{}, RESTRICTED], ids=["default", "restricted"])
def test_the_seed_tables_are_the_tables_of_the_base(make_env, overrides):
    """combination_seeds, sampled_combination_seeds with five replicates, seed_table, training_seeds for three samples and
    training_stream equal the scans that read every start state through a reset, on the default configuration and on one
    with a restricted start set, each scan on a fresh environment and the reference on another."""
    split = sampler.combination_folds(make_env(), 2, 0)
    fold = sampler.split_folds(split)[0]
    pairs = [(lambda env: combination_seeds(env, 2), lambda env: _base_combination_seeds(env, 2)),
             (lambda env: combination_seeds(env, 2, after=EVALUATION_POOL), lambda env: _base_combination_seeds(env, 2, after=EVALUATION_POOL)),
             (lambda env: sampled_combination_seeds(env, 42, EVALUATION_POOL, sampler.REPLICATES),
              lambda env: _base_sampled_combination_seeds(env, 42, EVALUATION_POOL, sampler.REPLICATES)),
             (seed_table, _base_seed_table),
             (lambda env: [sampler.training_seeds(env, split, sample, fold) for sample in (1, 2, 3)],
              lambda env: [_base_training_seeds(env, split, sample, fold) for sample in (1, 2, 3)]),
             (lambda env: sampler.training_stream(env, split, 40, fold), lambda env: _base_training_stream(env, split, 40, fold))]
    for scan, base in pairs:
        assert scan(make_env(**overrides)) == base(make_env(**overrides))
