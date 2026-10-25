"""Checks behind the split statements of docs/hydraulic_rig/environment.md; prints its results.
usage: python3 -I benchmarks/hydraulic_rig/split_checks.py [--facts PATH]
1. Labelled data behind an episode: for each reset candidate and fold seed, how many combinations of each fault count
   the classifiers that predict its cycles were fitted on (the 126 combinations outside its fold).
2. Training only on the combinations that no v0 episode can reach (three or four faulty components): valve accuracy at
   tier 3 on the cycles of the reset candidates.
3. Tier-1 valve accuracy held out by combination against held out by valve family (the four combinations that share
   the cooler, pump and accumulator levels)."""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import COMBOS, FOLD_SEEDS, N_FOLDS, TierClassifiers, lda_fit, lda_predict, scatter, stratified_folds  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS, LEVELS, build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv, n_faulty  # noqa: E402

VALVE = COMPONENTS.index("valve")


def valve_accuracy(x, cycle_combo, train, test):
    """Valve accuracy on the `test` cycles of one LDA fitted on the `train` cycles (feature rows x)."""
    k = len(LEVELS["valve"])
    sums, counts = [[0.0] * len(x[train[0]]) for _ in range(k)], [0] * k
    for i in train:
        level = LEVELS["valve"].index(cycle_combo[i][VALVE])
        counts[level] += 1
        sums[level] = [a + b for a, b in zip(sums[level], x[i])]
    model = lda_fit(scatter([x[i] for i in train]), sums, counts)
    return sum(lda_predict(model, x[i]) == LEVELS["valve"].index(cycle_combo[i][VALVE]) for i in test), len(test)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    a = ap.parse_args(argv)
    cycles = load_cycles(a.facts)
    blocks = build_blocks(cycles)
    candidates = set(TinyHydraulicEnv(cycles, blocks).candidates)

    folds = [stratified_folds(N_FOLDS, seed) for seed in FOLD_SEEDS]
    for faults in (1, 2):
        seen = {n: [] for n in range(5)}
        for fold in folds:
            for g in (h for h in candidates if n_faulty(h) == faults):
                training = [n_faulty(h) for h in COMBOS if fold[h] != fold[g]]
                for n in seen:
                    seen[n].append(training.count(n))
        print(f"1. episodes from a {faults}-fault combination, combinations in the training set by fault count, mean (range) over "
              f"fold seeds and starts: " + ", ".join(f"{n} faulty {statistics.mean(v):.1f} ({min(v)} to {max(v)})" for n, v in seen.items()))

    clf = TierClassifiers(cycles, blocks)
    x3, x1, combo_of = clf.features[3], clf.features[1], clf.cycle_combo
    right, total = valve_accuracy(x3, combo_of, [i for i in x3 if n_faulty(combo_of[i]) >= 3], [i for i in x3 if combo_of[i] in candidates])
    print(f"2. trained on the {sum(1 for h in COMBOS if n_faulty(h) >= 3)} unreachable combinations, valve accuracy at tier 3 on the "
          f"{total} reset-candidate cycles: {right / total:.3f}")

    def family(h):
        return h[:VALVE] + h[VALVE + 1:]
    for label, held_out in (("by combination", lambda g, h: h == g), ("by valve family", lambda g, h: family(h) == family(g))):
        right = total = 0
        for g in COMBOS:
            r, t = valve_accuracy(x1, combo_of, [i for i in x1 if not held_out(g, combo_of[i])], [i for i in x1 if combo_of[i] == g])
            right, total = right + r, total + t
        print(f"3. tier-1 valve accuracy over the {total} stable cycles, held out {label}: {right / total:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
