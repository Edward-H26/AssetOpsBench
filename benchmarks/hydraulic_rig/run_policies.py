"""The reference policies in process on seeded episodes of one configuration, each episode verified from its log, as
one CSV row per policy and seed, with the ordering check of OpenEnv RFC 008 printed: the oracle's reward above the do
nothing policy's on every seed (docs/hydraulic_rig/benchmarks.md).
usage: python benchmarks/hydraulic_rig/run_policies.py [--config static] [--seeds 100] [--first-seed 0] [--facts PATH]
       [--out benchmarks/hydraulic_rig/results/policies_<config>.csv]

No server runs here, so the released evidence the verifier reads is composed by verifier.released_from_env from the
environment's public view of its log.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import SCORE_COLUMNS, flat, load_config, load_world, write_rows  # noqa: E402
from envs.hydraulic_rig import verifier  # noqa: E402
from envs.hydraulic_rig.classifiers import TierClassifiers  # noqa: E402
from envs.hydraulic_rig.env import HydraulicRigEnv  # noqa: E402
from envs.hydraulic_rig.greedy import GreedyPolicy, ServedClassifiers  # noqa: E402
from envs.hydraulic_rig.policies import DoNothingPolicy, OraclePolicy, RandomPolicy, run_episode  # noqa: E402

POLICIES = ("oracle", "do_nothing", "random", "greedy")


def policy_table(cycles, blocks):
    """Policy name to a constructor taking the environment and the episode seed, so a CSV row reruns the same way it
    was produced. The greedy policy's classifiers are cross fitted once per table."""
    classifiers = ServedClassifiers(TierClassifiers(cycles, blocks), cycles)
    return {
        "oracle": lambda env, seed: OraclePolicy(env),
        "do_nothing": lambda env, seed: DoNothingPolicy(),
        "random": lambda env, seed: RandomPolicy(seed),
        "greedy": lambda env, seed: GreedyPolicy(classifiers, use_max_faulty=True),
    }


def play(cycles, blocks, config, make, seed):
    """One verified episode: the verdict as a dict and the reward."""
    env = HydraulicRigEnv(cycles, blocks, config)
    log = run_episode(env, make(env, seed), seed)
    reference = HydraulicRigEnv(cycles, blocks, config)
    run_episode(reference, OraclePolicy(reference), seed)
    verdict = verifier.verify(log, verifier.released_from_env(env), env.oracle(), reference.cost, grade=None, scope=config.verifier_scope)
    return verdict, verifier.reward(verdict, config.horizon)


def rows_of(cycles, blocks, config, seeds):
    rows = []
    for name, make in policy_table(cycles, blocks).items():
        for seed in seeds:
            verdict, reward = play(cycles, blocks, config, make, seed)
            row = {"policy": name, "seed": seed, **{k: flat(verdict.get(k)) for k in SCORE_COLUMNS}}
            row.update(reward=reward, verified=verdict.get("verified"))
            rows.append(row)
    return rows


def ordering_holds(rows):
    """RFC 008: on every seed the oracle's reward exceeds the do nothing policy's."""
    by = {(r["policy"], r["seed"]): r["reward"] for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    return all(by.get(("oracle", s)) is not None and by.get(("do_nothing", s)) is not None and by[("oracle", s)] > by[("do_nothing", s)] for s in seeds)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="static")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--facts", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    _facts, cycles, blocks = load_world(a.facts)
    name, config = load_config(a.config)
    out = Path(a.out) if a.out else HERE / "results" / f"policies_{name}.csv"
    started = time.perf_counter()
    rows = rows_of(cycles, blocks, config, list(range(a.first_seed, a.first_seed + a.seeds)))
    write_rows(out, rows, ["policy", "seed", *SCORE_COLUMNS, "verified"])
    print(f"wrote {out}: {len(rows)} episodes of {len(POLICIES)} policies in {time.perf_counter() - started:.1f} s; "
          f"RFC 008 ordering (oracle above do_nothing on every seed): {'holds' if ordering_holds(rows) else 'FAILS'}")


if __name__ == "__main__":
    main(sys.argv[1:])
