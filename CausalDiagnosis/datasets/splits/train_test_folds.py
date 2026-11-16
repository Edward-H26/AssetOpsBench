"""The split of training from testing for any asset: leave-one-fold-out cross-fitting over fault
atoms, closed under repair.

An atom is one faulty component at one faulty level, or one declared fault of a scripted case, and a start is the set of its
atoms, whatever their number. The atoms are dealt into K folds component by component, each component's atoms shuffled
under the seed and dealt cyclically from the fold that holds the fewest atoms so far, a seeded draw breaking a tie, so every
fold holds each component's atoms within one of the others. K is at most the number of atoms, and a component with fewer
atoms than K is faulty in that many folds only. The deal does not promise every fold a test start for K above 2, so a seed
whose split leaves a fold without one is refused below, and another seed is taken. Model h holds out fold h: it trains on every start from which no
sequence of repairs reaches an atom of fold h, which under repairs that only remove atoms is every start without an atom of
fold h. Every start is tested by exactly one model, the one that holds out a fold of its own atoms: a start whose atoms lie
in one fold by that fold's model, and a start whose atoms span folds by the one of its folds that holds the fewest test
starts of its number of faults so far, a seeded draw breaking a tie. So no test start is a training start of its model or
reachable from one by a repair, and only a fault the episode adds (the layer 5 hazard of the hydraulic rig) can reach it.
A split that leaves a fold without a test start is refused, since that fold's model would play no test episode.

A split that leaves a model without a training start is refused, as a split that leaves a fold without a test start is.

The module knows no asset. start_states.combination_folds is the rule on the start combinations of the hydraulic rig, for any K."""
import random


def fault_folds(starts, components, k, seed, repair_reach=None):
    """{start: (test fold, models that train on it)} and {atom: fold} for starts given as {start: tuple of atoms}, an atom being
    (component, fault). Folds and models count from 1, model h holding out fold h. components fixes the dealing order. With
    repair_reach, a function of a start's atoms to every atom a sequence of repairs can reach from it, a start trains model h
    only when none of those atoms lies in fold h; without it a repair only removes atoms, so the start's own atoms decide."""
    if k < 2:
        raise ValueError(f"cross-fitting needs at least two folds, got {k}")
    order = {c: i for i, c in enumerate(components)}
    if any(c not in order for atoms in starts.values() for c, _fault in atoms):
        raise ValueError("an atom names a component the asset does not list")
    if any(not atoms for atoms in starts.values()):
        # a start without a fault belongs to no fold, so no model holds it out and no trained variant plays it
        raise ValueError("every start holds at least one fault")
    n_atoms = len({atom for atoms in starts.values() for atom in atoms})
    if k > n_atoms:
        raise ValueError(f"K = {k} exceeds the {n_atoms} single faults of the starts, so a fold would hold no fault and its model "
                         "would test nothing")
    keys = sorted(starts)
    rng = random.Random(seed)
    atom_fold, held = {}, [0] * k

    # atoms of single fault starts first, in start order within a component, then atoms that occur only inside larger starts
    per_component = {c: [] for c in components}
    for key in keys:
        # an atom is dealt once, however many single fault starts hold it
        if len(starts[key]) == 1 and starts[key][0] not in per_component[starts[key][0][0]]:
            per_component[starts[key][0][0]].append(starts[key][0])
    for key in keys:
        for atom in starts[key]:
            if atom not in per_component[atom[0]]:
                per_component[atom[0]].append(atom)
    for component in components:
        atoms = list(per_component[component])
        if not atoms:
            continue
        # the singles are shuffled together and dealt from the least loaded fold, the extra atoms after them alike
        singles = [a for a in atoms if any(starts[key] == (a,) for key in keys)]
        for group in (singles, [a for a in atoms if a not in singles]):
            if not group:
                continue
            rng.shuffle(group)
            first = _least(held, range(k), rng)
            for j, atom in enumerate(group):
                atom_fold[atom] = (first + j) % k
                held[atom_fold[atom]] += 1

    out, tested = {}, {}
    strata = {}
    for key in keys:
        strata.setdefault(tuple(c for c, _fault in starts[key]), []).append(key)
    for stratum in sorted(strata, key=lambda s: (len(s), [order[c] for c in s])):
        members = list(strata[stratum])
        n = len(stratum)
        counts = tested.setdefault(n, [0] * k)
        if n == 1:
            for key in members:
                fold = atom_fold[starts[key][0]]
                out[key] = fold
            continue
        rng.shuffle(members)
        mixed = []
        for key in members:
            own = sorted({atom_fold[a] for a in starts[key]})
            if len(own) == 1:
                out[key] = own[0]
                counts[own[0]] += 1
            else:
                mixed.append((key, own))
        for key, own in mixed:
            fold = _least(counts, own, rng)
            out[key] = fold
            counts[fold] += 1

    empty = [h + 1 for h in range(k) if h not in set(out.values())]
    if empty:
        raise ValueError(f"the split leaves fold {', '.join(map(str, empty))} of {k} without a test start, so its model would play "
                         "no test episode")
    result = {}
    for key in keys:
        reach = set(repair_reach(starts[key]) if repair_reach else starts[key]) | set(starts[key])
        touched = {atom_fold[a] for a in reach if a in atom_fold}
        result[key] = (out[key] + 1, tuple(h + 1 for h in range(k) if h not in touched))
    idle = [h for h in range(1, k + 1) if not any(h in models for _fold, models in result.values())]
    if idle:
        which = f"model of fold {idle[0]}" if len(idle) == 1 else f"models of folds {', '.join(map(str, idle))}"
        raise ValueError(f"the split leaves the {which} of {k} without a training start, so {'it' if len(idle) == 1 else 'they'} "
                         "would train on nothing")
    return result, {atom: fold + 1 for atom, fold in atom_fold.items()}


def _least(counts, candidates, rng):
    """The candidate fold with the smallest count, a seeded draw among those tied."""
    candidates = list(candidates)
    low = min(counts[f] for f in candidates)
    tied = [f for f in candidates if counts[f] == low]
    return tied[0] if len(tied) == 1 else tied[rng.randrange(len(tied))]
