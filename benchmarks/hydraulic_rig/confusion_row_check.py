"""Does the greedy policy's observation model gain from the episode's own combination? (docs/hydraulic_rig/environment.md,
remaining leakage). In an episode on combination g, the shipped confusion rows count g's held-out
predictions in the rows of the hypotheses that share a level with g, and the held-out predictions on the other folds
come from classifiers fitted with g's cycles. Two variants build the observation model without g: "strict" drops
g's block from every row and from the concentration fit, and "nested" takes every row from inner cross-fits on the
other folds, by classifiers that never saw g's fold. All three read the served cycles through the same held-out
table. Analysis only: both variants choose their model by the hidden g, so neither is a policy.
usage: python3 -I benchmarks/hydraulic_rig/confusion_row_check.py [--facts PATH] [--episodes N] [--out benchmarks/hydraulic_rig/results/v0_confusion_row_check.csv]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import FOLD_SEEDS, TIERS, TierClassifiers, cross_fitted_predictions, observation_model  # noqa: E402
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv  # noqa: E402
from envs.hydraulic_rig.greedy import GreedyPolicy  # noqa: E402
from envs.hydraulic_rig.policies import run_episode  # noqa: E402
from envs.hydraulic_rig.reporting import seed_columns, write_csv  # noqa: E402
from envs.hydraulic_rig.scoring import score  # noqa: E402

POLICIES = (("greedy_prior_1to2", {"use_max_faulty": True}), ("greedy", {"use_max_faulty": False}))
VARIANTS = ("shipped", "strict", "nested")


class ObservationView:
    """The classifiers' held-out lookup with another observation model."""

    def __init__(self, classifiers, models):
        self.predict = classifiers.predict
        self.base = {t: models[t][0] for t in TIERS}
        self.alpha = {t: models[t][1] for t in TIERS}


def strict_view(clf, g):
    return ObservationView(clf, {t: observation_model({i: p for i, p in clf.predicted[t].items() if clf.cycle_combo[i] != g},
                                                      clf.cycle_combo) for t in TIERS})


def nested_view(clf, fold):
    models = {}
    for t in TIERS:
        outside = {i: v for i, v in clf.features[t].items() if clf.fold[clf.cycle_combo[i]] != fold}
        models[t] = observation_model(cross_fitted_predictions(outside, clf.cycle_combo, clf.fold), clf.cycle_combo)
    return ObservationView(clf, models)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "v0_confusion_row_check.csv"))
    a = ap.parse_args(argv)
    cycles = load_cycles(a.facts)
    blocks = build_blocks(cycles)
    env = TinyHydraulicEnv(cycles, blocks)
    starts = []  # the hidden starting combination of every episode seed
    for seed in range(a.episodes):
        env.reset(seed)
        starts.append(env.initial)
    results = {(name, v): [] for name, _ in POLICIES for v in VARIANTS}
    for fold_seed in FOLD_SEEDS:
        clf = TierClassifiers(cycles, blocks, seed=fold_seed)
        views = {"shipped": {g: clf for g in set(starts)},
                 "strict": {g: strict_view(clf, g) for g in set(starts)}}
        nested = {f: nested_view(clf, f) for f in {clf.fold[g] for g in starts}}
        views["nested"] = {g: nested[clf.fold[g]] for g in set(starts)}
        for name, settings in POLICIES:
            outcome = {v: [score(run_episode(env, GreedyPolicy(views[v][g], **settings), seed))
                           for seed, g in enumerate(starts)] for v in VARIANTS}
            for v in VARIANTS:
                top1 = [s["diagnosis_all_correct"] for s in outcome[v]]
                base = [s["diagnosis_all_correct"] for s in outcome["shipped"]]
                results[(name, v)].append({
                    "top1": sum(top1) / a.episodes,
                    "recovered": sum(s["recovered"] for s in outcome[v]) / a.episodes,
                    "mean_cost": sum(s["cost"] for s in outcome[v]) / a.episodes,
                    "top1_gain_points": 100 * (sum(top1) - sum(base)) / a.episodes,
                    "episodes_gained": sum(1 for x, y in zip(base, top1) if y and not x),
                    "episodes_lost": sum(1 for x, y in zip(base, top1) if x and not y)})
            print(f"fold seed {fold_seed} {name}: " + ", ".join(f"{v} Top-1 {results[(name, v)][-1]['top1']:.3f}" for v in VARIANTS), flush=True)
    rows = []
    for (name, v), per_seed in results.items():
        row = {"policy": name, "variant": v, "episodes": a.episodes, "fold_seeds": " ".join(map(str, FOLD_SEEDS))}
        for statistic in per_seed[0]:
            row.update(seed_columns(statistic, [r[statistic] for r in per_seed], digits=4))
        rows.append(row)
        print(f"{name} {v}: Top-1 gain over shipped, points: mean {row['top1_gain_points_mean']}, range {row['top1_gain_points_min']} to {row['top1_gain_points_max']}")
    write_csv(a.out, rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
