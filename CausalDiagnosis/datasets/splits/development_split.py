"""The development starts of any asset: the recorded fault combinations outside the test set, a start set
of their own beside the test set.

Every faulty level of every component must appear among the development starts. When the recorded combinations outside the test
set miss a level, the fewest test combinations that complete the coverage move to development and leave the test set, and among
the covers of that size the registered seed picks one. A development episode can still reach the recordings of a test
combination after a repair, as every episode reaches the healthy recordings.

The module knows no asset. development_starts, starts_of_test_set and moved_to_development read an asset profile and its
recorded combinations, and on_starts is the one way a configuration is narrowed to a start set, for the drivers and the
wrappers alike."""
import dataclasses
import random
from itertools import combinations as subsets
from itertools import islice

DEVELOPMENT_SEED = 0  # the registered seed that picks the test combinations moved to development
EXAMPLE_SEED = 0  # the registered seed that draws the example start of the two reference designs
# the search for the fewest moved combinations tries the covers by size, a count that grows without bound with the asset, so
# a split whose search passes this many candidate covers is refused
MAX_COVERS_TRIED = 1_000_000


class SplitRefused(ValueError):
    """The recorded combinations of an asset admit no development split, for the reason the message names."""


def faulty_levels(combination, healthy):
    """The (component index, level) of every component of a combination off its healthy level."""
    return {(i, level) for i, (level, ok) in enumerate(zip(combination, healthy)) if level != ok}


def coverage_order(items, levels_of, seed):
    """(covering, rest): the items in an order that the seed shuffles, first every item that holds a level of levels_of that no
    item taken before it holds, then the others in the same order. The one coverage rule of the capped development starts
    (run_episodes.capped_starts) and of the cut look of the test set (start_states.cut_combinations)."""
    order = list(items)
    random.Random(seed).shuffle(order)
    covering, covered = [], set()
    for item in order:
        levels = levels_of(item)
        if levels - covered:
            covering.append(item)
            covered |= levels
    return covering, [item for item in order if item not in covering]


def development_split(recorded, test, healthy, levels, seed):
    """(development, test) as sorted tuples. recorded holds every recorded combination, test the combinations of the test set
    (all recorded), healthy the healthy level of each component and levels the levels of each component, both in component
    order. The development starts are the recorded faulty combinations outside test, then the fewest test combinations that
    give every faulty level of every component a development start, chosen by seed among the covers of that size."""
    recorded = {tuple(c) for c in recorded}
    test = {tuple(c) for c in test}
    if not test <= recorded:
        raise ValueError("every test combination must be a recorded combination")
    needed = {(i, level) for i, component_levels in enumerate(levels) for level in component_levels if level != healthy[i]}
    development = {c for c in recorded - test if faulty_levels(c, healthy)}
    if not needed <= set().union(*(faulty_levels(c, healthy) for c in recorded)):
        raise SplitRefused("a faulty level of the asset has no recorded combination, so no split can give it a development start")
    tested = set().union(*(faulty_levels(c, healthy) for c in test))
    missing = needed - set().union(*(faulty_levels(c, healthy) for c in development))
    if missing:
        candidates = sorted(c for c in test if faulty_levels(c, healthy) & missing)
        random.Random(seed).shuffle(candidates)
        moved = _fewest_cover(candidates, missing, healthy)
        development |= set(moved)
        test -= set(moved)
    # a faulty level that the test set showed keeps a start in it, else the test set could not score a fault of that level
    lost = sorted(tested - set().union(*(faulty_levels(c, healthy) for c in test)))
    if lost:
        raise SplitRefused(f"the split would leave the faulty levels {lost} (component index and level) without a start in the test set, "
                           "since every test combination that shows them moves to development, so the split is refused: record a "
                           "combination outside the test set that shows each of them")
    return tuple(sorted(development)), tuple(sorted(test))


def _fewest_cover(candidates, missing, healthy):
    """The first of the smallest sets of candidates that hold every missing level, in the order of the candidates. Every missing
    level has a candidate, so a cover exists, and the search ends without one only at its bound."""
    covers = (cover for size in range(1, len(candidates) + 1) for cover in subsets(candidates, size))
    for cover in islice(covers, MAX_COVERS_TRIED):
        if missing <= set().union(*(faulty_levels(c, healthy) for c in cover)):
            return cover
    raise SplitRefused(f"the search for the fewest test combinations that give every faulty level a development start passed "
                     f"{MAX_COVERS_TRIED} candidate covers, so the split is refused: record more combinations outside the test set")


def _profile_split(profile, recorded, seed):
    rule = profile.sampler["fault_rule"]
    healthy = [profile.healthy[c] for c in profile.components]
    recorded = [tuple(c) for c in recorded]
    test = [c for c in recorded if rule["min_faulty"] <= len(faulty_levels(c, healthy)) <= rule["max_faulty"]]
    development, test = development_split(recorded, test, healthy, [profile.levels[c] for c in profile.components], seed)
    order = {tuple(c): i for i, c in enumerate(profile.combinations())}
    return tuple(sorted(development, key=order.get)), tuple(sorted(test, key=order.get))


def development_starts(profile, recorded, seed=DEVELOPMENT_SEED):
    """The development starts of an asset, in grid order: the recorded fault combinations outside the test set, which holds the
    recorded combinations that the profile's fault rule allows, plus the fewest test combinations that give every faulty level of
    every component a development start, chosen by the seed. recorded holds the combinations with a stable block of recordings."""
    return _profile_split(profile, recorded, seed)[0]


def starts_of_test_set(profile, recorded, seed=DEVELOPMENT_SEED):
    """The start combinations of the test set of an asset, in grid order: the recorded combinations its fault rule allows less
    those that development_starts moved to development. The name does not open with test, so pytest never collects it."""
    return _profile_split(profile, recorded, seed)[1]


def moved_to_development(profile, recorded, seed=DEVELOPMENT_SEED):
    """The combinations that development_starts moved out of the test set, which the start sampler of the environment
    leaves out, so that the sampler and the folds read one test set. None move where the recorded combinations admit no
    split (SplitRefused), so that an environment can still be built on such recordings, while every request for the
    development starts or the folds of the asset is refused by name."""
    recorded = [tuple(c) for c in recorded]
    healthy = [profile.healthy[c] for c in profile.components]
    rule = profile.sampler["fault_rule"]
    allowed = {c for c in recorded if rule["min_faulty"] <= len(faulty_levels(c, healthy)) <= rule["max_faulty"]}
    try:
        return frozenset(allowed - set(starts_of_test_set(profile, recorded, seed)))
    except SplitRefused:
        return frozenset()


def on_starts(config, starts):
    """The configuration with its fault rule and its allowed starts replaced by the range of fault counts and the combinations
    of starts, so that reset draws these starts alone."""
    healthy = [config.healthy[c] for c in config.components]
    faults = [len(faulty_levels(c, healthy)) for c in starts]
    replaced = dataclasses.replace(config, fault_rule={"min_faulty": min(faults), "max_faulty": max(faults)},
                                   allowed=[list(c) for c in starts])
    return type(config).from_json(replaced.to_json())


def example_start(profile, recorded, seed=EXAMPLE_SEED):
    """The start whose levels the one diagnosis example of the two reference designs shows: among the development starts of the
    asset, in grid order, those with the fewest components off their healthy level, and of them the one that the registered seed
    draws. So it is never a test start and never the healthy combination. The combination as a tuple in component order."""
    healthy = [profile.healthy[c] for c in profile.components]
    starts = development_starts(profile, recorded)
    if not starts:
        raise ValueError("the asset has no development start, so no example start can be drawn")
    fewest = min(len(faulty_levels(c, healthy)) for c in starts)
    return tuple(random.Random(seed).choice([c for c in starts if len(faulty_levels(c, healthy)) == fewest]))
