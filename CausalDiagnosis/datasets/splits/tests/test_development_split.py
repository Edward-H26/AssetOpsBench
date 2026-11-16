"""The development starts on made-up assets (CausalDiagnosis/datasets/splits/development_split.py): the recorded fault combinations
outside the test set, and when they miss a faulty level of a component, the fewest test combinations that complete the coverage,
chosen by the registered seed, move to development and leave the test set. The search for those combinations has a bound, and a
profile gives both sets in grid order."""
from itertools import product
from types import SimpleNamespace

import pytest

from CausalDiagnosis.datasets.splits import development_split as module
from CausalDiagnosis.datasets.splits.development_split import development_split, development_starts, example_start, faulty_levels, starts_of_test_set
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import world  # noqa: F401 - the recorded cycles, skipped in a clone that states that it holds no data

LEVELS = [(0, 1, 2), (0, 1), (0, 1, 2)]  # three components, level 0 healthy
HEALTHY = (0, 0, 0)
ALL = [c for c in product(*LEVELS)]


def _faults(c):
    return sum(level != 0 for level in c)


def _covered(development):
    return set().union(*(faulty_levels(c, HEALTHY) for c in development))


NEEDED = {(i, level) for i, levels in enumerate(LEVELS) for level in levels if level}


def test_when_the_combinations_outside_the_test_set_cover_every_level_nothing_moves():
    test = [c for c in ALL if 1 <= _faults(c) <= 2]
    development, kept = development_split(ALL, test, HEALTHY, LEVELS, seed=0)
    assert set(development) == {c for c in ALL if _faults(c) == 3} and set(kept) == set(test)
    assert _covered(development) == NEEDED


def test_a_level_missing_outside_the_test_set_moves_the_fewest_test_combinations_that_cover_it():
    """Only the three-fault combinations at level 1 of the first component are recorded outside the test set, so level 2 of the
    first component has no development start: one test combination holding it moves, chosen by the seed among several."""
    test = [c for c in ALL if 1 <= _faults(c) <= 2]
    recorded = [*test, *(c for c in ALL if _faults(c) == 3 and c[0] == 1)]
    development, kept = development_split(recorded, test, HEALTHY, LEVELS, seed=0)
    moved = set(test) - set(kept)
    assert len(moved) == 1 and moved <= set(development) and _covered(development) == NEEDED
    assert not set(kept) & set(development) and set(kept) | moved == set(test)
    assert development_split(recorded, test, HEALTHY, LEVELS, seed=0) == (development, kept)
    others = {development_split(recorded, test, HEALTHY, LEVELS, seed=s)[0] for s in range(12)}
    assert len(others) > 1  # several covers of one combination exist, and the seed picks among them


def test_an_empty_complement_takes_its_development_starts_from_the_test_set():
    """Every recorded fault combination is a test start, so development is the fewest test combinations that hold every faulty
    level, here two."""
    recorded = [c for c in ALL if _faults(c) >= 1]
    development, kept = development_split(recorded, recorded, HEALTHY, LEVELS, seed=3)
    assert len(development) == 2 and _covered(development) == NEEDED and set(kept) == set(recorded) - set(development)


def test_a_level_without_any_recording_or_a_test_start_outside_the_recordings_is_refused():
    recorded = [c for c in ALL if 1 <= _faults(c) and c[1] != 1]
    with pytest.raises(ValueError, match="no recorded combination"):
        development_split(recorded, recorded, HEALTHY, LEVELS, seed=0)
    with pytest.raises(ValueError, match="every test combination must be a recorded combination"):
        development_split(ALL[:3], ALL, HEALTHY, LEVELS, seed=0)


def test_the_search_for_the_fewest_covers_is_refused_beyond_its_bound(monkeypatch):
    """The exact search tries the covers by size, a count that grows without bound with the asset, so a split whose search passes
    MAX_COVERS_TRIED candidate covers is refused. Here every recorded combination is a single fault test start, so the only cover
    is all five of them, the last of the 31 candidate covers."""
    singles = [c for c in ALL if _faults(c) == 1]
    monkeypatch.setattr(module, "MAX_COVERS_TRIED", 30)
    with pytest.raises(ValueError, match="30 candidate covers"):
        development_split(singles, singles, HEALTHY, LEVELS, seed=0)
    monkeypatch.setattr(module, "MAX_COVERS_TRIED", 31)
    with pytest.raises(module.SplitRefused, match="without a start in the test set"):
        development_split(singles, singles, HEALTHY, LEVELS, seed=0)  # the cover moves every test start, so no fault is left to score


def test_a_split_that_leaves_a_faulty_level_without_a_test_start_is_refused():
    """A made-up asset of one component and two levels whose fault rule admits the healthy start: its one faulty combination
    is a test combination and must also be its development start, which would leave its faulty level without a test start, so
    the split is refused by name, while the environment's sampler moves nothing for such recordings."""
    one = SimpleNamespace(components=["press"], levels={"press": [0, 1]}, healthy={"press": 0},
                          sampler={"fault_rule": {"min_faulty": 0, "max_faulty": 1}}, combinations=lambda: [(0,), (1,)])
    with pytest.raises(module.SplitRefused, match=r"the faulty levels \[\(0, 1\)\] \(component index and level\) without a start"):
        development_split([(0,), (1,)], [(0,), (1,)], [0], [[0, 1]], seed=0)
    for request in (development_starts, starts_of_test_set):
        with pytest.raises(ValueError, match="without a start in the test set"):
            request(one, [(0,), (1,)])
    assert module.moved_to_development(one, [(0,), (1,)]) == frozenset()


def test_a_profile_gives_its_test_set_and_its_development_starts_in_grid_order():
    """development_starts and starts_of_test_set read the fault rule, the levels and the grid of a profile alone. Both come in
    grid order and share no start, and a test combination that moves to development is no test start any more."""
    names = ("a", "b", "c")
    profile = SimpleNamespace(components=names, levels=dict(zip(names, LEVELS)), healthy=dict(zip(names, HEALTHY)),
                              sampler={"fault_rule": {"min_faulty": 1, "max_faulty": 2}}, combinations=lambda: tuple(ALL))
    assert starts_of_test_set(profile, ALL) == tuple(c for c in ALL if 1 <= _faults(c) <= 2)
    assert development_starts(profile, ALL) == tuple(c for c in ALL if _faults(c) == 3)
    # level 2 of the first component has no recording outside the test set, so one test combination that holds it moves
    recorded = [c for c in ALL if _faults(c) <= 2 or c[0] == 1]
    test, development = starts_of_test_set(profile, recorded), development_starts(profile, recorded)
    moved = [c for c in development if _faults(c) <= 2]
    assert len(moved) == 1 and moved[0][0] == 2 and not set(test) & set(development)
    assert set(test) | set(development) == {c for c in recorded if _faults(c)}
    assert list(test) == sorted(test, key=ALL.index) and list(development) == sorted(development, key=ALL.index)


def _example_holds(profile, recorded):
    start = example_start(profile, recorded)
    healthy = tuple(profile.healthy[c] for c in profile.components)
    development, test = development_starts(profile, recorded), starts_of_test_set(profile, recorded)
    assert start in development and start not in test and start != healthy
    assert len(faulty_levels(start, healthy)) == min(len(faulty_levels(c, healthy)) for c in development)
    assert example_start(profile, recorded) == start  # the same on every call
    return start


def test_the_example_start_is_a_development_start_with_the_fewest_faults_drawn_by_the_registered_seed():
    names = ("a", "b", "c")
    profile = SimpleNamespace(components=names, levels=dict(zip(names, LEVELS)), healthy=dict(zip(names, HEALTHY)),
                              sampler={"fault_rule": {"min_faulty": 1, "max_faulty": 2}}, combinations=lambda: tuple(ALL))
    # every development start holds three faults
    assert _faults(_example_holds(profile, ALL)) == 3
    # the split moves one test combination with fewer faults, which is then the one start with the fewest
    recorded = [c for c in ALL if _faults(c) <= 2 or c[0] == 1]
    moved = [c for c in development_starts(profile, recorded) if _faults(c) <= 2]
    assert _example_holds(profile, recorded) == moved[0]
    # the seed draws among the starts with the fewest faults
    drawn = {example_start(profile, ALL, seed=s) for s in range(20)}
    assert len(drawn) > 1 and all(_faults(c) == 3 for c in drawn)
    assert module.EXAMPLE_SEED == 0


def test_the_example_start_of_the_registered_asset_of_the_recordings_holds(world):
    """The registered asset whose recordings the tests read, on its recorded combinations: the example start is a development
    start, no test start and not the healthy combination, with the fewest faults of the development starts. A registered asset is
    one with a file CausalDiagnosis/datasets/<asset>/asset.json."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, get_profile

    _cycles, blocks = world
    registered = sorted(path.parent.name for path in ASSETS_DIR.glob("*/asset.json"))
    recorded = set(blocks)
    played = [asset_id for asset_id in registered if set(get_profile(asset_id).combinations()) >= recorded]
    assert played, registered
    for asset_id in played:
        _example_holds(get_profile(asset_id), recorded)
