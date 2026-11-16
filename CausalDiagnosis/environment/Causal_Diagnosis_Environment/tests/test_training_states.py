"""The folds of the training split and their training seeds on the hydraulic rig, the instance with K = 2 of the general rule
(start_states.combination_folds, training_start_states, training_seeds,
training_stream, fold_replicate_seeds): the split is the registered file of the rig, a model is named by the fold it holds out,
the model of fold h trains on the single fault combinations of the other fold and on the two fault combinations whose faults
both lie there, no training start of a model reaches by one repair a combination it is judged on, and the seeds replay no
episode of a reserved seed and no label rollout. The test of the label map pins that the models are the models of the first fold convention with
their labels swapped, h = 3 minus the label of that convention."""
from CausalDiagnosis.utils.paths import CONFIGS
import json
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.splits.development_split import development_starts, starts_of_test_set
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as sampler

FOLDS_FILE = CONFIGS / "training" / "hyd_1" / "folds.json"


def _registered():
    return json.loads(FOLDS_FILE.read_text(encoding="utf-8"))


def _split():
    """{combination: (fold, models that train on it)} of the rig's registered file."""
    return {tuple(row["combination"]): (row["fold"], tuple(row["models"])) for row in _registered()["combinations"]}


def _state(env, seed):
    env.reset(seed)
    return sampler.start_state(env)


def _faulty(env, combination):
    return [c for c, level in zip(env.config.components, combination) if level != env.config.healthy[c]]


def _one_repair_children(env, combination):
    """The combinations a start reaches when one of its faulty components is repaired."""
    return [tuple(env.config.healthy[c] if c == component else level for c, level in zip(env.config.components, combination))
            for component in _faulty(env, combination)]


def _judged_reached(env, split, model):
    """The training starts of a model that are, or reach by one repair, a combination the model is judged on, those of the fold
    it holds out."""
    judged = {combination for combination, (f, _models) in split.items() if f == model}
    return [combination for combination, (_f, models) in split.items() if model in models
            and ({combination, *_one_repair_children(env, combination)} & judged)]


def test_the_registered_split_is_the_rule_on_the_real_data(make_env):
    """configs/training/hyd_1/folds.json is the rule's output with the K and the seed it records, fixed before any run, and the
    rule replays from its seed alone."""
    env = make_env()
    registered = _registered()
    k, seed = registered["k"], registered["seed"]
    assert registered["components"] == list(env.config.components) and registered["rule"].startswith("CausalDiagnosis.datasets.splits.train_test_folds.fault_folds")
    assert sampler.combination_folds(env, k, seed) == _split()
    assert sampler.combination_folds(env, k, seed + 1) != _split()
    assert len(_split()) == len(env.candidates) and sampler.split_folds(_split()) == tuple(range(1, k + 1))


def test_a_model_is_named_by_the_fold_it_holds_out_and_the_rig_labels_are_those_of_d95_swapped(make_env):
    """The model of fold h holds out fold h and plays its test starts. Under the first fold convention the model of
    fold f trained on the combinations of fold f that train and played the other fold, so the model of fold h now is the model
    of fold 3 - h then: the same training starts, the same test seeds, only the label swapped."""
    env = make_env()
    split = _split()
    for h in sampler.split_folds(split):
        old = 3 - h
        old_trains = {c for c, (f, models) in split.items() if f == old and models}
        assert {s[0] for s in sampler.training_start_states(env, split, h)} == old_trains
        assert all(split[_state(env, seed)[0]][0] == h for seed in sampler.fold_replicate_seeds(env, split, 1, h))
        assert not any(split[_state(env, seed)[0]][0] == old for seed in sampler.fold_replicate_seeds(env, split, 1, h))


def test_each_model_trains_on_the_single_faults_outside_its_fold_and_the_two_fault_combinations_whose_parts_lie_there(make_env):
    env = make_env()
    split = _split()
    for h in sampler.split_folds(split):
        trains = [c for c, (_f, models) in split.items() if h in models]
        single = [c for c in trains if len(_faulty(env, c)) == 1]
        pairs = [c for c in trains if len(_faulty(env, c)) == 2]
        assert len(single) == 5 and len(pairs) == 9 and all(split[c][0] != h for c in trains)
        # every component keeps a single fault combination outside each held-out fold, so each model sees every component faulty
        assert {_faulty(env, c)[0] for c in single} == set(env.config.components)
        for pair in pairs:
            assert all(h in split[part][1] for part in _one_repair_children(env, pair))
    evaluation_only = [c for c, (_f, models) in split.items() if not models]
    assert sorted(sum(1 for c in evaluation_only if split[c][0] == h) for h in sampler.split_folds(split)) == [9, 10]
    assert all(len(_faulty(env, c)) == 2 for c in evaluation_only)


def test_no_training_start_of_a_model_reaches_by_one_repair_a_combination_it_is_judged_on(make_env):
    """Every training start of a model and its one-repair children lie outside the fold it holds out. The same check flags a split
    in which a start of the held-out fold trains the model that holds it out."""
    env = make_env()
    split = _split()
    assert all(_judged_reached(env, split, h) == [] for h in sampler.split_folds(split))
    mixed = next(c for c, (_f, models) in split.items() if not models)
    leaky = {**split, mixed: (split[mixed][0], (split[mixed][0],))}
    assert _judged_reached(env, leaky, split[mixed][0]) == [mixed]


def test_the_training_states_of_a_model_are_its_training_combinations_at_offsets_no_test_replicate_plays(make_env):
    env = make_env()
    split = _split()
    tested = {_state(env, seed) for seed in sampler.sampled_combination_seeds(env, 42, 8, sampler.REPLICATES)}
    for h in sampler.split_folds(split):
        states = sampler.training_start_states(env, split, h)
        trains = {c for c, (_f, models) in split.items() if h in models}
        assert states == sorted(states) and len(states) == len(set(states)) == 70 and not set(states) & tested
        assert {s[0] for s in states} == trains and all(sum(1 for s in states if s[0] == c) == 5 for c in trains)
        assert sampler.training_start_states(env, split, h, 42, 8) == states
    with pytest.raises(ValueError, match="registered split names the folds"):
        sampler.training_start_states(env, split, len(sampler.split_folds(split)) + 1)


def test_the_seeds_of_a_sample_enter_their_states_and_replay_no_reserved_seed(make_env):
    env = make_env()
    split = _split()
    reserved = sampler.reserved_seeds(env)
    test = set(sampler.sampled_combination_seeds(env, replicates=sampler.REPLICATES))
    assert set(sampler.combination_seeds(env, 1)) <= reserved
    assert all(seeds <= reserved for seeds in sampler.label_seeds(env).values())
    for h in sampler.split_folds(split):
        states = sampler.training_start_states(env, split, h)
        first, second = sampler.training_seeds(env, split, 1, h), sampler.training_seeds(env, split, 2, h)
        assert [_state(env, seed) for seed in first] == states == [_state(env, seed) for seed in second]
        assert all(a < b for a, b in zip(first, second))  # the second sample of a state is its next free seed
        assert not (set(first) | set(second)) & (reserved | test)
    # the registered samples per state of configs/training/hyd_1/rft.json reach every state of every model
    samples = json.loads((FOLDS_FILE.parent / "rft.json").read_text(encoding="utf-8"))["samples_per_state"]
    last = sampler.training_seeds(env, split, samples, 1)
    assert len(last) == 70 and not set(last) & (reserved | test)
    with pytest.raises(ValueError, match="count from 1"):
        sampler.training_seeds(env, split, 0, 1)


def test_the_stream_runs_in_seed_order_over_the_training_states_of_its_model(make_env):
    env = make_env()
    split = _split()
    reserved = sampler.reserved_seeds(env)
    for h in sampler.split_folds(split):
        states = set(sampler.training_start_states(env, split, h))
        stream = sampler.training_stream(env, split, 60, h, first_seed=5)
        assert len(stream) == 60 and stream == sorted(stream) and stream[0] >= 5
        assert all(_state(env, seed) in states and seed not in reserved for seed in stream)
        skipped = [seed for seed in range(stream[0], stream[-1]) if seed not in stream]
        assert all(seed in reserved or _state(env, seed) not in states for seed in skipped)


def test_the_stream_passes_over_the_first_seeds_of_every_state(make_env):
    """skip_per_state of 0 gives the stream of before. With 2 the stream passes over the seeds of the samples 1 and 2 of every
    state, counted from seed 0, also with a first_seed above 0, and goes on with the next seed of each state in seed order."""
    env = make_env()
    split = _split()
    for h in sampler.split_folds(split):
        assert sampler.training_stream(env, split, 60, h, first_seed=5, skip_per_state=0) == sampler.training_stream(env, split, 60, h, first_seed=5)
        states = sampler.training_start_states(env, split, h)
        sampled = set(sampler.training_seeds(env, split, 1, h)) | set(sampler.training_seeds(env, split, 2, h))
        third = dict(zip(states, sampler.training_seeds(env, split, 3, h)))
        for first_seed in (0, 5):
            stream = sampler.training_stream(env, split, 60, h, first_seed=first_seed, skip_per_state=2)
            assert len(stream) == 60 and stream == sorted(stream) and stream[0] >= first_seed and not set(stream) & sampled
            firsts = {}
            for seed in stream:
                firsts.setdefault(_state(env, seed), seed)
            assert firsts and all(firsts[state] == third[state] for state in firsts if third[state] >= first_seed)


def test_each_model_plays_the_replicate_seeds_of_the_fold_it_holds_out(make_env):
    env = make_env()
    split = _split()
    for replicate in (1, 5):
        whole = sampler.replicate_seeds(env, replicate)
        parts = {h: sampler.fold_replicate_seeds(env, split, replicate, h) for h in sampler.split_folds(split)}
        assert sorted(sum(parts.values(), [])) == sorted(whole) and sum(len(p) for p in parts.values()) == len(set(whole))
        assert sorted(len(seeds) for seeds in parts.values()) == [23, 24]
        for h, seeds in parts.items():
            assert all(split[_state(env, seed)[0]][0] == h for seed in seeds)
    # a configuration with a start that the registered split does not hold is refused by name
    partial = {combination: folds for combination, folds in split.items() if combination != env.candidates[0]}
    with pytest.raises(ValueError, match="is no start of the registered split"):
        sampler.fold_replicate_seeds(env, partial, 1, 1)


def test_the_development_starts_of_the_rig_are_its_recorded_three_and_four_fault_combinations(make_env, world):
    """The development split on the rig: the test set stays the 47 combinations its fault rule allows, nothing moves, and the
    development starts are the recorded combinations outside it, which cover every faulty level of every component."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

    env = make_env()
    _cycles, blocks = world
    profile = get_profile(env.config.asset_id)
    development = development_starts(profile, set(blocks))
    assert starts_of_test_set(profile, set(blocks)) == tuple(sorted(env.candidates, key=list(profile.combinations()).index))
    assert len(env.candidates) == 47 and not set(development) & set(env.candidates)
    assert {len(_faulty(env, c)) for c in development} == {3, 4}
    assert set(development) == {c for c in blocks if len(_faulty(env, c)) >= 3}
    for i, component in enumerate(env.config.components):
        assert {c[i] for c in development} >= {level for level in env.config.levels[component] if level != env.config.healthy[component]}
