"""The general split of training from testing (CausalDiagnosis/datasets/splits/train_test_folds.py) on a made-up asset that is not
the hydraulic rig: three components, K = 3, starts of one, two and three faults. Each model holds out one fold, trains on no
start that reaches an atom of that fold by repairs, and every start is tested by exactly one model that never trained on it
and cannot reach it by a repair from any of its training starts."""
from itertools import combinations as subsets

import pytest

from CausalDiagnosis.datasets.splits.train_test_folds import fault_folds

COMPONENTS = ["motor", "bearing", "seal"]
FAULTS = {"motor": ("winding", "rotor bar"), "bearing": ("outer race", "inner race", "cage"), "seal": ("leak",)}
ATOMS = [(c, f) for c in COMPONENTS for f in FAULTS[c]]


def _starts():
    """Every single fault, every pair of faults on two components and one start with a fault on every component."""
    singles = [(a,) for a in ATOMS]
    pairs = [(a, b) for a, b in subsets(ATOMS, 2) if a[0] != b[0]]
    return {f"s{i:02d}": atoms for i, atoms in enumerate([*singles, *pairs, (("motor", "winding"), ("bearing", "cage"), ("seal", "leak"))])}


def _removal_only(atoms):
    """The starts that repairs reach on an asset whose repair only removes a fault."""
    return [frozenset(sub) for n in range(len(atoms)) for sub in subsets(atoms, n)]


def test_every_start_is_tested_once_by_a_model_that_never_trained_on_it_or_reaches_it_by_repair():
    starts = _starts()
    split, atom_fold = fault_folds(starts, COMPONENTS, 3, seed=0)
    assert set(split) == set(starts) and set(atom_fold) == set(ATOMS) and set(atom_fold.values()) == {1, 2, 3}
    assert any(len(atoms) == 3 for atoms in starts.values())
    for key, (test_fold, trains) in split.items():
        own = {atom_fold[a] for a in starts[key]}
        assert test_fold in own and test_fold not in trains  # the testing model holds out a fold of the start's own faults
        assert set(trains) == {1, 2, 3} - own  # a model trains on every start without a fault of its held-out fold
    for model in (1, 2, 3):
        training = [starts[key] for key, (_f, trains) in split.items() if model in trains]
        tested = {frozenset(starts[key]) for key, (f, _t) in split.items() if f == model}
        reached = {state for atoms in training for state in [frozenset(atoms), *_removal_only(atoms)]}
        assert training and tested and not reached & tested
        assert not {atom_fold[a] for atoms in training for a in atoms} & {model}


def test_the_atoms_are_balanced_by_component_and_the_rule_replays_from_its_seed():
    starts = _starts()
    split, atom_fold = fault_folds(starts, COMPONENTS, 3, seed=0)
    for component in COMPONENTS:
        per_fold = [sum(1 for (c, _f), fold in atom_fold.items() if c == component and fold == h) for h in (1, 2, 3)]
        assert max(per_fold) - min(per_fold) <= 1
    folds = [sum(1 for fold in atom_fold.values() if fold == h) for h in (1, 2, 3)]
    assert max(folds) - min(folds) <= 1
    assert fault_folds(starts, COMPONENTS, 3, seed=0) == (split, atom_fold)
    assert fault_folds(starts, COMPONENTS, 3, seed=1) != (split, atom_fold)


def test_a_repair_that_turns_one_fault_into_another_closes_the_training_set_over_it():
    """A scripted asset whose repair of the inner race leaves an outer race fault: a start with the inner race reaches the
    outer race by a repair, so it trains no model that holds out the outer race's fold."""
    starts = _starts()
    plain, atom_fold = fault_folds(starts, COMPONENTS, 3, seed=0)
    inner, outer = ("bearing", "inner race"), ("bearing", "outer race")
    if atom_fold[inner] == atom_fold[outer]:
        pytest.skip("the seed put both bearing faults in one fold, so the transition changes nothing")

    def reach(atoms):
        return [*atoms, outer] if inner in atoms else list(atoms)

    closed, _ = fault_folds(starts, COMPONENTS, 3, seed=0, repair_reach=reach)
    key = next(k for k, atoms in starts.items() if atoms == (inner,))
    assert atom_fold[outer] in plain[key][1] and atom_fold[outer] not in closed[key][1]
    assert {k: f for k, (f, _t) in closed.items()} == {k: f for k, (f, _t) in plain.items()}  # the test folds do not move


def test_a_component_with_fewer_faults_than_k_is_faulty_in_that_many_folds_only():
    """Case 1: the seal has one fault and the motor two, so with K = 3 the seal is faulty in one fold and the motor in two, and
    every fold still holds a fault and a test start."""
    starts = _starts()
    split, atom_fold = fault_folds(starts, COMPONENTS, 3, seed=0)
    for component in COMPONENTS:
        assert len({fold for (c, _f), fold in atom_fold.items() if c == component}) == min(len(FAULTS[component]), 3)
    assert {fold for fold, _models in split.values()} == {1, 2, 3}


def test_k_beyond_the_single_faults_or_a_fold_without_a_test_start_is_refused():
    """Case 2: six single faults admit at most six folds. A fold whose faults occur only inside starts tested by another fold
    would leave its model nothing to test, so such a split is refused, while every split the rule returns tests in every fold.
    The fault of the seal occurs only beside the winding, so its fold tests a start on the seeds that give it the start of two
    faults, and on every seed each model keeps a training start, so no other refusal stands in the way."""
    with pytest.raises(ValueError, match="exceeds the 6 single faults"):
        fault_folds(_starts(), COMPONENTS, 7, seed=0)
    starts = {"a": (("motor", "winding"),), "b": (("motor", "winding"), ("seal", "leak")), "c": (("bearing", "cage"),)}
    refused = 0
    for seed in range(20):
        try:
            split, _atoms = fault_folds(starts, COMPONENTS, 3, seed)
        except ValueError as e:
            assert "without a test start" in str(e)
            refused += 1
        else:
            assert {fold for fold, _models in split.values()} == {1, 2, 3}
    assert 0 < refused < 20


def test_a_split_that_leaves_a_model_without_a_training_start_is_refused():
    """Four components with one fault each and starts of three and four faults: under K = 2 every start holds a fault of both
    folds, so neither model has a training start and the split is refused by name, on every seed. With the single fault starts
    added the same call returns a split in which every model trains."""
    components = ["fan", "belt", "coil", "duct"]
    atoms = [(component, "fault") for component in components]
    starts = {f"s{i}": faults for i, faults in enumerate([*subsets(atoms, 3), tuple(atoms)])}
    for seed in range(20):
        with pytest.raises(ValueError, match="the split leaves the models of folds 1, 2 of 2 without a training start"):
            fault_folds(starts, components, 2, seed)
    with_singles = {**starts, **{f"single{i}": (atom,) for i, atom in enumerate(atoms)}}
    split, _atom_fold = fault_folds(with_singles, components, 2, seed=0)
    assert all(any(model in models for _fold, models in split.values()) for model in (1, 2))


def test_an_asset_of_single_faults_alone_tests_each_in_its_fold_and_trains_every_other_model():
    """Case 3: without starts of two or more faults every start is tested by the model that holds out its fault's fold and
    trains every other model."""
    singles = {f"s{i}": (atom,) for i, atom in enumerate(ATOMS)}
    split, atom_fold = fault_folds(singles, COMPONENTS, 3, seed=0)
    for key, (fold, models) in split.items():
        assert fold == atom_fold[singles[key][0]] and set(models) == {1, 2, 3} - {fold}


def test_a_start_of_any_number_of_faults_is_tested_by_one_model_among_its_own_folds():
    """Case 4: a start of three faults on four components, one of four faults and the singles under K = 2, 3 and 4: each start
    is tested once by a model that holds out one of its own folds and trained on by the models of the other folds only."""
    components = [*COMPONENTS, "shaft"]
    atoms = [*ATOMS, ("shaft", "crack")]
    starts = {**{f"s{i}": (atom,) for i, atom in enumerate(atoms)},
              "three": (("motor", "winding"), ("bearing", "cage"), ("seal", "leak")),
              "four": (("motor", "rotor bar"), ("bearing", "outer race"), ("seal", "leak"), ("shaft", "crack"))}
    for k in (2, 3, 4):
        split, atom_fold = fault_folds(starts, components, k, seed=0)
        for key, (fold, models) in split.items():
            own = {atom_fold[a] for a in starts[key]}
            assert fold in own and set(models) == set(range(1, k + 1)) - own


def test_fewer_than_two_folds_or_an_unknown_component_is_refused():
    with pytest.raises(ValueError, match="two folds"):
        fault_folds(_starts(), COMPONENTS, 1, seed=0)
    with pytest.raises(ValueError, match="component"):
        fault_folds({"x": (("gearbox", "tooth"),)}, COMPONENTS, 2, seed=0)


def test_two_starts_with_the_same_single_fault_deal_its_atom_once():
    """Two starts may hold the same single fault, such as one combination at two entry offsets. Its atom is dealt once, so the
    split of K = 2 over two atoms puts one atom in each fold, and every fold has a test start."""
    starts = {"a": (("motor", "rotor"),), "b": (("motor", "winding"),), "c": (("motor", "winding"),)}
    split, atom_fold = fault_folds(starts, ["motor"], 2, seed=0)
    assert sorted(atom_fold.values()) == [1, 2] and {fold for fold, _models in split.values()} == {1, 2}
    assert split["b"] == split["c"]


def test_a_start_without_a_fault_is_refused_by_name():
    """A start without a fault belongs to no fold, so no model would hold it out or be judged on it: the rule refuses it by name."""
    starts = {"healthy": (), "a": (("motor", "rotor"),), "b": (("motor", "winding"),)}
    with pytest.raises(ValueError, match="every start holds at least one fault"):
        fault_folds(starts, ["motor"], 2, seed=0)
