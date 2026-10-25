import random
from pathlib import Path

import pytest

from envs.hydraulic_rig.classifiers import (COMBOS, FOLD_SEEDS, N_FOLDS, TIERS, TierClassifiers, cross_fitted_predictions, lda_fit,
                                    lda_predict, scatter, solve_spd, stratified_folds)
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, build_blocks, default_facts_path, load_cycles, tier_features
from envs.hydraulic_rig.env import TinyHydraulicEnv
from envs.hydraulic_rig.greedy import GreedyPolicy, expected_information_gain
from envs.hydraulic_rig.policies import run_episode
from envs.hydraulic_rig.scoring import score

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)
BY_INDEX = {c.index: c for c in CYCLES}
CLF = TierClassifiers(CYCLES, BLOCKS)


def test_linear_algebra_and_lda_on_hand_computed_cases():
    assert all(abs(a - b) < 1e-12 for a, b in zip(solve_spd([[4.0, 2.0], [2.0, 3.0]], [[2.0, 1.0]])[0], [0.5, 0.0]))
    rows = [[-1.0], [-1.2], [-0.8], [1.0], [1.2], [0.8]]
    model = lda_fit(scatter(rows), [[-3.0], [3.0]], [3, 3])
    assert [lda_predict(model, [v]) for v in (-0.3, 0.3, -5.0, 5.0)] == [0, 1, 0, 1]


def test_stratified_folds_put_every_level_in_every_training_set():
    for n_folds in (2, 4, N_FOLDS):
        size = len(COMBOS) // n_folds
        for seed in FOLD_SEEDS:
            fold = stratified_folds(n_folds, seed)
            assert set(fold) == set(COMBOS)
            assert [sum(1 for c in COMBOS if fold[c] == f) for f in range(n_folds)] == [size] * n_folds
            for f in range(n_folds):
                for j, comp in enumerate(COMPONENTS):
                    assert {c[j] for c in COMBOS if fold[c] != f} == set(LEVELS[comp]), (n_folds, seed, f, comp)
                    for level in LEVELS[comp]:  # stratified: every level within 2 of its share of a fold
                        assert abs(sum(1 for c in COMBOS if fold[c] == f and c[j] == level) - size / len(LEVELS[comp])) <= 2
    assert CLF.fold == stratified_folds(N_FOLDS, 0) != stratified_folds(N_FOLDS, 1)


def test_cross_fitting_refuses_a_split_whose_training_set_lacks_a_level():
    by_sorted_index = {combo: h % 4 for h, combo in enumerate(COMBOS)}  # one accumulator level per fold
    with pytest.raises(ValueError, match="accumulator"):
        cross_fitted_predictions(CLF.features[1], CLF.cycle_combo, by_sorted_index)


def test_held_out_predictions_come_from_classifiers_fitted_without_the_fold():
    for tier, fold in ((1, 0), (3, 5)):
        train = [BY_INDEX[i] for combo in COMBOS if CLF.fold[combo] != fold for i in BLOCKS[combo]]
        test = [BY_INDEX[i] for combo in COMBOS if CLF.fold[combo] == fold for i in BLOCKS[combo]]
        names = sorted(tier_features(train[0], tier))
        rows = [[tier_features(c, tier)[n] for n in names] for c in train]
        for j, comp in enumerate(COMPONENTS):
            k = len(LEVELS[comp])
            sums, counts = [[0.0] * len(names) for _ in range(k)], [0] * k
            for c, r in zip(train, rows):
                level = LEVELS[comp].index(c.combo[j])
                counts[level] += 1
                sums[level] = [a + b for a, b in zip(sums[level], r)]
            model = lda_fit(scatter(rows), sums, counts)
            for c in test:
                features = tier_features(c, tier)
                assert lda_predict(model, [features[n] for n in names]) == CLF.predict(tier, features)[j]


def test_observation_model_excludes_the_hypothesis_own_block():
    tier, j = 3, COMPONENTS.index("valve")
    for h in (5, 70, 140):
        level = COMBOS[h][j]
        counts = [0] * len(LEVELS["valve"])
        for combo, idx in BLOCKS.items():
            if combo[j] == level and combo != COMBOS[h]:
                for i in idx:
                    counts[CLF.predict(tier, tier_features(BY_INDEX[i], tier))[j]] += 1
        expected = [(c + 1) / (sum(counts) + len(counts)) for c in counts]
        assert all(abs(a - b) < 1e-12 for a, b in zip(CLF.base[tier][j][h], expected))
    for t in TIERS:
        for j in range(len(COMPONENTS)):
            assert all(abs(sum(row) - 1.0) < 1e-12 for row in CLF.base[t][j]) and CLF.alpha[t][j] > 0


def test_expected_information_gain_is_bounded_by_the_belief_entropy():
    belief = [0.25] * 4
    certain = [[[1.0 if o == h else 0.0 for o in range(4)]] for h in range(4)]
    assert abs(expected_information_gain(belief, certain) - 2.0) < 1e-12
    assert abs(expected_information_gain(belief, [[[1 / 3] * 3] for _ in range(4)])) < 1e-12
    rng = random.Random(0)
    noisy = []
    for _ in range(4):
        w = [rng.random() for _ in range(5)]
        noisy.append([[x / sum(w) for x in w]])
    assert 0.0 <= expected_information_gain(belief, noisy) <= 2.0


def test_prior_takes_the_fault_count_rule_from_the_observation():
    healthy = tuple(HEALTHY[c] for c in COMPONENTS)
    for min_faulty, max_faulty in ((1, 1), (2, 2), (1, 2)):
        env = TinyHydraulicEnv(CYCLES, BLOCKS, min_faulty=min_faulty, max_faulty=max_faulty)
        for use_rule, support in ((False, set(COMBOS) - {healthy}), (True, set(env.candidates))):
            policy = GreedyPolicy(CLF, use_max_faulty=use_rule)
            policy.act(env.reset(0))
            assert {h for h, b in zip(COMBOS, policy.belief) if b > 0} == support


def test_greedy_episodes_are_deterministic_legal_and_keep_budget_for_repairs():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    policy = GreedyPolicy(CLF)
    first = [score(run_episode(env, policy, seed)) for seed in range(20)]
    assert first == [score(run_episode(env, policy, seed)) for seed in range(20)]
    for s in first:
        assert s["submitted"] and s["refusals"] == 0 and s["end_reason"] != "budget"
        assert s["cost"] <= env.budget and s["repairs_match_diagnosis"]
        assert s["diagnosis_per_component"]["cooler"]  # the held-out cooler predictions are correct at every tier
