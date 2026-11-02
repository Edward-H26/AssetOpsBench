"""The per-tier classifiers of the greedy policy (src/envs/hydraulic_rig/classifiers.py): the linear algebra, the stratified
folds, the cross-fit, the observation model, the information gain, and the lookup keyed by served values."""
import random
from pathlib import Path

import pytest

from envs.hydraulic_rig.classifiers import (COMBOS, FOLD_SEEDS, N_FOLDS, TIERS, TierClassifiers, cross_fitted_predictions, lda_fit,
                                    lda_predict, scatter, solve_spd, stratified_folds)
from envs.hydraulic_rig.data import COMPONENTS, LEVELS, TIER_SENSORS, build_blocks, default_facts_path, load_cycles, tier_features
from envs.hydraulic_rig.greedy import ServedClassifiers, expected_information_gain
from envs.hydraulic_rig.env import served_values

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


def test_served_values_key_every_held_out_prediction_without_collisions():
    served = ServedClassifiers(CLF, CYCLES)
    for tier in TIERS:
        assert len(served.lookup[tier]) == len(CLF.predicted[tier])
        cycle = BY_INDEX[next(iter(CLF.predicted[tier]))]
        assert served.predict(tier, served_values(cycle, TIER_SENSORS[tier])) == CLF.predict(tier, tier_features(cycle, tier))
    with pytest.raises(KeyError):
        served.predict(1, {"not": "served"})
