"""The reference policies in process on seeded episodes of one configuration, each episode verified from its log, as
one CSV row per policy and seed, with the ordering check of OpenEnv RFC 008 printed: the oracle's reward above the do
nothing policy's on every seed (CausalDiagnosis/docs/experiments.md).
usage: python -m CausalDiagnosis.experiments.episode_runners.run_reference_policies [--config static] [--asset ID] [--seeds 100] [--first-seed 0] [--facts PATH]
       [--out CausalDiagnosis/results/runs/policies_<config>.csv]

No server runs here, so the released evidence the verifier reads is composed by verifier.released_from_env from the
environment's public view of its log, and the separability of every start comes from the label that the asset of the
configuration declares (CausalDiagnosis.datasets.dataset_loader.dataset_packages.start_label), on the rig the independent label with the server's E18 grade of
start_grade of CausalDiagnosis.environment.reference_policies.diagnosability_label. --asset composes the preset over another registered asset, whose rows go
to CausalDiagnosis/results/runs/<asset_id>/ unless --out names a file. The rows are measured values, so they go to the runs folder of
their asset, which a rule of .gitignore leaves out of the repository. The greedy policy runs the reference classifiers of
CausalDiagnosis.environment.reference_policies.tier_classifiers, fitted on the recordings of the rig alone, so the table of another asset leaves it out.
"""
from CausalDiagnosis.utils.paths import RUNS
import argparse
import sys
import time
from pathlib import Path


from CausalDiagnosis.experiments.episode_runners.run_folder_layout import SCORE_COLUMNS, episode_reward
from CausalDiagnosis.utils.csv_rows import flat, write_rows
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402
from CausalDiagnosis.environment.reference_policies.tier_classifiers import PROFILE as CLASSIFIED_ASSET, TierClassifiers  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.reference_policies.greedy_policy import GreedyPolicy, ServedClassifiers  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost  # noqa: E402
from CausalDiagnosis.environment.reference_policies.policies import DeclarePolicy, DoNothingPolicy, OraclePolicy, RandomPolicy, run_episode  # noqa: E402

POLICIES = ("oracle", "do_nothing", "random", "greedy", "declare")


def policy_table(cycles, blocks, asset_id=None):
    """Policy name to a constructor taking the environment and the episode seed, so a CSV row reruns the same way it was
    produced. The greedy policy's classifiers are cross fitted once per table. They are the reference classifiers of
    CausalDiagnosis.environment.reference_policies.tier_classifiers, fitted on the recordings of one asset, so the table of another asset holds no greedy policy;
    asset_id None is the engine's default asset."""
    table = {
        "oracle": lambda env, seed: OraclePolicy(env),
        "do_nothing": lambda env, seed: DoNothingPolicy(),
        "random": lambda env, seed: RandomPolicy(seed),
    }
    if (asset_id or DEFAULT_ASSET) == CLASSIFIED_ASSET.asset_id:
        classifiers = ServedClassifiers(TierClassifiers(cycles, blocks), cycles)
        table["greedy"] = lambda env, seed: GreedyPolicy(classifiers)
    table["declare"] = lambda env, seed: DeclarePolicy()
    return table


def play(cycles, blocks, config, make, seed, label_of=None):
    """One verified episode: the verdict as a dict and the reward. label_of(env), called after reset, gives the separability
    of the start that the verifier's success and the reward read, the label with E18 as start_grade gives it, or None when
    unknown."""
    env = HydraulicRigEnv(cycles, blocks, config)
    env.reset(seed)
    label = label_of(env) if label_of else None
    log = run_episode(env, make(env, seed), seed)
    # the verifier's own cost of the oracle episode is the normalizer, as in the drivers and the gym, so the base row, whose
    # cost runs in calls while every price is zero, gets a reward
    cost_oracle = oracle_cost(HydraulicRigEnv(cycles, blocks, config), seed)
    verdict = verifier.verify(log, verifier.released_from_env(env), env.oracle(), cost_oracle, grade=label, scope=config.verifier_scope)
    return verdict, episode_reward(verdict, config.horizon)


def rows_of(cycles, blocks, config, seeds, label_of=None):
    rows = []
    for name, make in policy_table(cycles, blocks, config.asset_id).items():
        for seed in seeds:
            verdict, reward = play(cycles, blocks, config, make, seed, label_of)
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
    ap.add_argument("--asset", default=None,
                    help=f"the registered asset the preset is composed over (default: the asset the preset names, {DEFAULT_ASSET} when it names none)")
    ap.add_argument("--facts", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    name, config = load_config(a.config, a.asset)
    _facts, cycles, blocks = load_world(a.facts, config.asset_id)
    out = Path(a.out) if a.out else default_out(config.asset_id, name)
    started = time.perf_counter()
    rows = rows_of(cycles, blocks, config, list(range(a.first_seed, a.first_seed + a.seeds)), ports.start_label)
    write_rows(out, rows, ["policy", "seed", *SCORE_COLUMNS, "verified"])
    print(f"wrote {out}: {len(rows)} episodes of {len({row['policy'] for row in rows})} policies in {time.perf_counter() - started:.1f} s; "
          f"RFC 008 ordering (oracle above do_nothing on every seed): {'holds' if ordering_holds(rows) else 'FAILS'}")


def default_out(asset_id, name):
    """The file of the rows without --out: the runs folder of the asset, the rig's beside this script and a port's beside its run
    folders, which .gitignore covers, since the rows are measured values and the tracked results folder holds only generated tables."""
    folder = RUNS if asset_id == DEFAULT_ASSET else RUNS / asset_id
    return folder / f"policies_{name}.csv"


if __name__ == "__main__":
    main(sys.argv[1:])
