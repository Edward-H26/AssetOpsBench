"""Run the reference policies on seeded episodes and write per-episode scores to CSV.
usage: python3 -I benchmarks/hydraulic_rig/run_policies.py [--facts PATH] [--episodes N] [--out benchmarks/hydraulic_rig/results/v0_scores.csv]"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import TierClassifiers  # noqa: E402
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv  # noqa: E402
from envs.hydraulic_rig.greedy import GreedyPolicy  # noqa: E402
from envs.hydraulic_rig.policies import OraclePolicy, RandomPolicy, run_episode  # noqa: E402
from envs.hydraulic_rig.scoring import score  # noqa: E402


def policy_table(env, classifiers):
    """Policy name to constructor taking the episode seed; the tests import this table, so a CSV row reruns the
    same way it was produced."""
    return {
        "random": lambda seed: RandomPolicy(seed),
        "oracle": lambda seed: OraclePolicy(env),
        "greedy_prior_1to2": lambda seed: GreedyPolicy(classifiers, use_max_faulty=True),
        "greedy": lambda seed: GreedyPolicy(classifiers),
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "v0_scores.csv"))
    a = ap.parse_args(argv)
    cycles = load_cycles(a.facts)
    blocks = build_blocks(cycles)
    env = TinyHydraulicEnv(cycles, blocks)
    classifiers = TierClassifiers(cycles, blocks)
    rows = []
    t0 = time.time()
    for name, make in policy_table(env, classifiers).items():
        for seed in range(a.episodes):
            policy = make(seed)  # one policy per episode, so any single row of the CSV can be rerun on its own
            s = score(run_episode(env, policy, seed))
            s = {k: (v if not isinstance(v, (dict, tuple)) else str(v)) for k, v in s.items()}
            rows.append({"policy": name, "seed": seed, **s})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for name in policy_table(env, classifiers):
        rs = [r for r in rows if r["policy"] == name]
        n = len(rs)
        print(f"{name}: episodes {n}, diagnosis all correct {sum(r['diagnosis_all_correct'] for r in rs)/n:.2f}, recovered {sum(r['recovered'] for r in rs)/n:.2f}, mean cost {sum(r['cost'] for r in rs)/n:.1f}, unnecessary replacements {sum(r['unnecessary_replacements'] for r in rs)}, insufficient evidence {sum(r['insufficient_evidence'] for r in rs)/n:.2f}")
    print(f"wrote {a.out}; {len(rows)} episodes in {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
