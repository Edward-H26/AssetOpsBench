"""Held-out evaluation of the per-tier classifiers of the greedy reference policy over the fold seeds, with the split
recorded (docs/hydraulic_rig/environment.md).
usage: python3 -I benchmarks/hydraulic_rig/eval_classifiers.py [--facts PATH] [--out-dir benchmarks/hydraulic_rig/results]
Writes v0_classifiers.csv (per tier and component: held-out accuracy on all stable cycles and on the reset candidates,
and the fitted concentration, each as mean, range and value under each fold seed) and v0_classifier_folds.csv (the
fold of each combination under each fold seed)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import COMBOS, FOLD_SEEDS, N_FOLDS, SPLIT_RULE, TIERS, TierClassifiers  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS, build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv  # noqa: E402
from envs.hydraulic_rig.reporting import seed_columns, write_csv  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args(argv)
    cycles = load_cycles(a.facts)
    blocks = build_blocks(cycles)
    candidates = set(TinyHydraulicEnv(cycles, blocks).candidates)
    n_all = sum(len(blocks[c]) for c in COMBOS)
    n_reset = sum(len(blocks[c]) for c in candidates)
    fits = [TierClassifiers(cycles, blocks, seed=seed) for seed in FOLD_SEEDS]
    split = {"fold_seeds": " ".join(map(str, FOLD_SEEDS)), "split": f"{N_FOLDS} {SPLIT_RULE}"}
    rows = []
    for tier in TIERS:
        acc_all = [clf.accuracy(tier) for clf in fits]
        acc_reset = [clf.accuracy(tier, candidates) for clf in fits]
        for j, comp in enumerate(COMPONENTS):
            rows.append({"tier": tier, "component": comp, "held_out_cycles": n_all,
                         **seed_columns("accuracy", [acc[comp] for acc in acc_all]),
                         "reset_candidate_cycles": n_reset,
                         **seed_columns("accuracy_reset_candidates", [acc[comp] for acc in acc_reset]),
                         **seed_columns("dm_concentration", [clf.alpha[tier][j] for clf in fits]), **split})
        print(f"tier {tier}, held-out accuracy on {n_all} stable cycles, mean (range) over fold seeds {split['fold_seeds']}: "
              + ", ".join(f"{r['component']} {r['accuracy_mean']:.3f} ({r['accuracy_min']:.3f} to {r['accuracy_max']:.3f})" for r in rows[-4:]))
    out = Path(a.out_dir)
    write_csv(out / "v0_classifiers.csv", rows)
    write_csv(out / "v0_classifier_folds.csv", [{"combination": combo, "stable_cycles": len(blocks[combo]), "reset_candidate": combo in candidates,
                                                 **{f"fold_seed_{clf.seed}": clf.fold[combo] for clf in fits}} for combo in COMBOS])
    print(f"split: {split['split']}")
    print(f"wrote {out / 'v0_classifiers.csv'} and {out / 'v0_classifier_folds.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])
