"""The greedy reference policy under the classifiers of every fold seed (docs/hydraulic_rig/environment.md).
usage: python3 -I benchmarks/hydraulic_rig/greedy_fold_seeds.py [--facts PATH] [--episodes N] [--out benchmarks/hydraulic_rig/results/v0_greedy_fold_seeds.csv]
For each configuration, N seeded episodes (default 100, the seeds of run_policies.py) run under the classifiers of each
fold seed; every statistic is written as its mean, range and value under each fold seed, one row per configuration and
statistic. Configurations: the two greedy rows of run_policies.py (greedy_prior_1to2, the reference, and greedy, the
ablation without the generator rule) and the reference with other minimum rates and confidence thresholds."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import FOLD_SEEDS, TierClassifiers  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS, build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv  # noqa: E402
from envs.hydraulic_rig.greedy import GreedyPolicy  # noqa: E402
from envs.hydraulic_rig.policies import run_episode  # noqa: E402
from envs.hydraulic_rig.reporting import seed_summary, write_csv  # noqa: E402
from envs.hydraulic_rig.scoring import score  # noqa: E402

REFERENCE = {"use_max_faulty": True, "min_gain_per_price": 0.05, "confidence": 0.95}
CONFIGS = (
    ("greedy_prior_1to2", REFERENCE),
    ("greedy", {**REFERENCE, "use_max_faulty": False}),
    *(("greedy_prior_1to2", {**REFERENCE, "min_gain_per_price": rate}) for rate in (0.01, 0.02, 0.1)),
    *(("greedy_prior_1to2", {**REFERENCE, "confidence": level}) for level in (0.8, 0.9, 0.99)),
)


def episode_statistics(env, classifiers, settings, episodes):
    """Summary statistics of `episodes` seeded episodes of GreedyPolicy(classifiers, **settings)."""
    scores, beliefs, tiers, repeated = [], [], [], 0
    for seed in range(episodes):
        policy = GreedyPolicy(classifiers, **settings)
        log = run_episode(env, policy, seed)
        scores.append(score(log))
        beliefs.append(max(policy.belief))  # the belief at submission: the policy measures only before it submits
        measures = [e for e in log if e["event"] == "measure"]
        tiers.append([e["tier"] for e in measures])
        repeated += sum(e["repeated"] for e in measures)
    n = len(scores)
    after_tier3 = [t[t.index(3) + 1:] for t in tiers if 3 in t]
    return {
        "top1": sum(s["diagnosis_all_correct"] for s in scores) / n,
        "recovered": sum(s["recovered"] for s in scores) / n,
        "mean_cost": sum(s["cost"] for s in scores) / n,
        "unnecessary_replacements": sum(s["unnecessary_replacements"] for s in scores),
        "measurements_per_episode": sum(len(t) for t in tiers) / n,
        **{f"tier{k}_measurements_per_episode": sum(s[f"measures_tier{k}"] for s in scores) / n for k in (1, 2, 3)},
        "cost_before_submit": sum(s["cost_before_submit"] for s in scores) / n,
        "largest_belief_at_submission": sum(beliefs) / n,
        "episodes_first_measuring_tier1": sum(1 for t in tiers if t[:1] == [1]),
        "episodes_reaching_tier3": len(after_tier3),
        "episodes_returning_below_tier3": sum(1 for rest in after_tier3 if any(k < 3 for k in rest)),
        "repeated_cycles": repeated,
        "refusals": sum(s["refusals"] for s in scores),
        "episodes_repairs_match_diagnosis": sum(s["repairs_match_diagnosis"] for s in scores),
        **{f"{c}_correct": sum(s["diagnosis_per_component"][c] for s in scores) for c in COMPONENTS},
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "v0_greedy_fold_seeds.csv"))
    a = ap.parse_args(argv)
    cycles = load_cycles(a.facts)
    blocks = build_blocks(cycles)
    env = TinyHydraulicEnv(cycles, blocks)
    per_seed = {i: [] for i in range(len(CONFIGS))}
    for fold_seed in FOLD_SEEDS:
        classifiers = TierClassifiers(cycles, blocks, seed=fold_seed)
        for i, (_, settings) in enumerate(CONFIGS):
            per_seed[i].append(episode_statistics(env, classifiers, settings, a.episodes))
        print(f"fold seed {fold_seed}: " + ", ".join(f"{name} Top-1 {per_seed[i][-1]['top1']:.2f}" for i, (name, _) in enumerate(CONFIGS[:2])), flush=True)
    rows = []
    for i, (name, settings) in enumerate(CONFIGS):
        for statistic in per_seed[i][0]:
            rows.append({"policy": name, **settings, "episodes": a.episodes, "fold_seeds": " ".join(map(str, FOLD_SEEDS)),
                         "statistic": statistic, **seed_summary([s[statistic] for s in per_seed[i]])})
        top1 = seed_summary([s["top1"] for s in per_seed[i]])
        print(f"{name} {settings}: Top-1 mean {top1['mean']:.3f} (range {top1['min']:.2f} to {top1['max']:.2f})")
    write_csv(a.out, rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
