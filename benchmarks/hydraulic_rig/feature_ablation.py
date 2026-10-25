"""Which statistics and channels carry each fault: held-out accuracy of the greedy policy's classifiers on feature
sets other than the one the environment serves (docs/hydraulic_rig/environment.md). Analysis only: the environment
still serves the mean, std, min and max of the tier's channels. Same LDA, stratified 8-fold splits and fold seeds as
v0_classifiers.csv; the extra statistics come from the IndustryAssetEQA facts file.
usage: python3 -I benchmarks/hydraulic_rig/feature_ablation.py [--facts PATH] [--out benchmarks/hydraulic_rig/results/v0_feature_ablation.csv]"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import (COMBOS, FOLD_SEEDS, N_FOLDS, SPLIT_RULE, cross_fitted_predictions,  # noqa: E402
                                    held_out_accuracy, stratified_folds)
from envs.hydraulic_rig.data import COMPONENTS, SENSORS, TIER_SENSORS, build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.reporting import seed_columns, write_csv  # noqa: E402

SERVED = ("mean", "std", "min", "max")
ALL_NINE = SERVED + ("median", "ptp", "trend", "spec_peak_freq", "spec_peak_power")
PRESSURES = ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6")
FEATURE_SETS = (
    ("served statistics", 1, TIER_SENSORS[1], SERVED),
    ("served statistics", 2, TIER_SENSORS[2], SERVED),
    ("served statistics", 3, SENSORS, SERVED),
    ("all nine statistics", 1, TIER_SENSORS[1], ALL_NINE),
    ("all nine statistics", 2, TIER_SENSORS[2], ALL_NINE),
    ("all nine statistics", 3, SENSORS, ALL_NINE),
    ("means only", 3, SENSORS, ("mean",)),
    ("served statistics without PS1 to PS6", 3, tuple(s for s in SENSORS if s not in PRESSURES), SERVED),
)


def select(feature_names, channels, stats):
    """Names of the per-cycle features of `channels` that end in one of `stats` (same channel match as data.tier_features)."""
    patterns = [re.compile(rf"_{s}(_|$)") for s in channels]
    suffixes = tuple("_" + st for st in stats)
    return sorted(n for n in feature_names if n.endswith(suffixes) and any(p.search(n) for p in patterns))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "v0_feature_ablation.csv"))
    a = ap.parse_args(argv)
    cycles = load_cycles(a.facts)
    blocks = build_blocks(cycles)
    by_index = {c.index: c for c in cycles}
    cycle_combo = {i: combo for combo in COMBOS for i in blocks[combo]}
    folds = [stratified_folds(N_FOLDS, seed) for seed in FOLD_SEEDS]
    split = {"fold_seeds": " ".join(map(str, FOLD_SEEDS)), "split": f"{N_FOLDS} {SPLIT_RULE}"}
    rows = []
    for label, tier, channels, stats in FEATURE_SETS:
        names = select(next(iter(by_index.values())).features, channels, stats)
        x = {i: [by_index[i].features[n] for n in names] for i in cycle_combo}
        accs = [held_out_accuracy(cross_fitted_predictions(x, cycle_combo, fold), cycle_combo) for fold in folds]
        row = {"features": label, "tier_channels": tier, "values_per_cycle": len(names)}
        for c in COMPONENTS:
            row.update(seed_columns(c, [acc[c] for acc in accs]))
        rows.append({**row, **split})
        print(f"tier {tier} channels, {label} ({len(names)} values), mean (range) over fold seeds: "
              + ", ".join(f"{c} {row[c + '_mean']:.3f} ({row[c + '_min']:.3f} to {row[c + '_max']:.3f})" for c in COMPONENTS), flush=True)
    write_csv(a.out, rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
