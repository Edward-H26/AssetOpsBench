"""Admission anchors of the first two scenario classes (docs/hydraulic_rig/environment.md).
usage: python3 -I benchmarks/hydraulic_rig/scenario_anchors.py [--facts PATH] [--out-dir benchmarks/hydraulic_rig/results] [--classes S1 S2]

A class states its exact fault-count rule (min_faulty = max_faulty = 1 for the 10 combinations of S1, 2 for the 37 of
S2), and each of its combinations is one record. Every policy plays every start state of the class environment at the
default prices, each from the smallest seed whose reset reaches it (src/envs/hydraulic_rig/start_states.py), and
each record weighs 1/R of the class, shared equally by its start states, so a class number is the mean over records of
the record means.

Enumeration with record weights is chosen over seeds stratified by entry position because it
is the exact expectation over env.reset's own draw on the class environment (a combination uniformly, then a window,
then an entry), so no anchor depends on which seeds were drawn: a finite set of seeds can miss start states and whole
combinations, and even when stratified by entry it leaves the combination counts to chance. The record
weights matter as well: a record whose stable block is longer than a window has many overlapping windows, and so many
start states, and the weights keep it from dominating the class numbers. Intervals come from
a bootstrap that resamples records, since the start states of one record share the hidden combination and most of their
cycles. Policy randomness (random, replace-everything) and the window drawn after a repair still come from the episode
seed, one draw per start state. Seeds of the class environment, rather than of one-record environments (`allowed` set
to one combination, as a scenario record runs), give every episode its own policy stream: every record whose block is
one window long reaches its entries from the same seeds, so one-record environments would replay the same policy draws
in every record.

Policies: the oracle; do-nothing, which submits the healthy combination and finishes; replace-everything, which submits
a diagnosis drawn from the class without evidence and replaces components in a seeded random order until the repairs
run out; random; the greedy reference row (greedy_prior_1to2, told the class rule by the observation) under each of the
10 fold seeds; and symptom-only, which chooses its diagnosis from the task text and the public rules without measuring
and then replaces only the components it calls faulty. v0's task text carries no fault information, so symptom-only
submits the heaviest combination of the class on the start-state table; every combination weighs 1/R there, so it takes
the first in sorted order and resolves exactly 1/R.

Columns: the Top-1, recovered and resolved rates, the mean cost over all episodes, and the cost per
resolved episode, which divides the total cost over all episodes by the resolved episodes and by the mean reference
cost (blank when nothing resolves). Record weighting, which every rate, mean and interval uses: a start state of record
r weighs 1/(R n_r), where R is the number of records of the class and n_r the number of start states of r, so a class
number is the mean over records of the record means; episodes and
episodes_at_reference_cost are plain counts, and the weighting column states this definition on every row. Intervals
are 95% percentile intervals over 10,000 record resamples (the interval of the cost per resolved episode is blank when
some resample resolves nothing). Writes scenario_anchors_<class>.csv with one row per anchor (per fold seed for anchor
5)."""
import argparse
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import COMBOS, FOLD_SEEDS, N_FOLDS, SPLIT_RULE, TierClassifiers  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv, n_faulty  # noqa: E402
from envs.hydraulic_rig.greedy import GreedyPolicy  # noqa: E402
from envs.hydraulic_rig.policies import OraclePolicy, RandomPolicy, run_episode  # noqa: E402
from envs.hydraulic_rig.reporting import cluster_resamples, percentile_interval, write_csv  # noqa: E402
from envs.hydraulic_rig.scoring import score  # noqa: E402
from envs.hydraulic_rig.start_states import seed_table  # noqa: E402

CLASSES = {"S1": 1, "S2": 2}  # components off their healthy level in every start combination, told to the agent exactly
GREEDY = "greedy_prior_1to2"
RESAMPLES = 10_000
RECORD_MEANS = ("top1", "recovered", "resolved", "cost", "reference_cost", "unnecessary_replacements", "steps", "refusals")
INTERVALS = ("top1", "recovered", "resolved", "cost_per_resolved_episode")
WEIGHTING = ("record-weighted: a start state of record r weighs 1/(R n_r), with R the records of the class "
             "(start_combinations) and n_r the start states of record r, so every rate, mean and interval is the mean "
             "over records of the record means; episodes and episodes_at_reference_cost are plain counts")


class DoNothingPolicy:
    """Submits the healthy combination, then finishes (anchor 2). A finish before the submission is refused."""

    def act(self, obs):
        if obs["phase"] == "investigate":
            return {"type": "submit", "diagnosis": dict(obs["healthy"])}
        return {"type": "finish"}


class ReplaceEverythingPolicy:
    """Submits a diagnosis drawn from the class's combinations, then replaces components in a seeded random order
    until the repairs run out (anchor 3). It draws from its own stream, so its choices are independent of the
    hidden combination."""

    def __init__(self, seed, combinations):
        rng = random.Random(f"replace-everything:{seed}")
        self.diagnosis = dict(zip(COMPONENTS, rng.choice(sorted(combinations))))
        self.order = rng.sample(COMPONENTS, len(COMPONENTS))

    def act(self, obs):
        if obs["phase"] == "investigate":
            return {"type": "submit", "diagnosis": dict(self.diagnosis)}
        replaced = sum(1 for e in obs["history"] if e["event"] == "replace")
        return {"type": "replace", "component": self.order[replaced]}


class SymptomOnlyPolicy:
    """Submits a diagnosis chosen without measuring, replaces the components that diagnosis calls faulty in component
    order, and finishes if the rig has not recovered by then (anchor 6)."""

    def __init__(self, combination):
        self.diagnosis = dict(zip(COMPONENTS, combination))
        self.faulty = [c for c in COMPONENTS if self.diagnosis[c] != HEALTHY[c]]

    def act(self, obs):
        if obs["phase"] == "investigate":
            return {"type": "submit", "diagnosis": dict(self.diagnosis)}
        replaced = sum(1 for e in obs["history"] if e["event"] == "replace")
        return {"type": "replace", "component": self.faulty[replaced]} if replaced < len(self.faulty) else {"type": "finish"}


def class_combinations(name):
    return {h for h in COMBOS if n_faulty(h) == CLASSES[name]}


def class_env(name, cycles, blocks):
    """The environment of a class at the default prices, budget and repair limit, with the class's exact rule."""
    faults = CLASSES[name]
    env = TinyHydraulicEnv(cycles, blocks, min_faulty=faults, max_faulty=faults, allowed=class_combinations(name))
    assert set(env.candidates) == class_combinations(name)
    return env


def reference_cost(prices, initial):
    """The oracle's cost: the replacement price (part plus downtime) of every component that starts off healthy."""
    return sum(prices["parts"][c] + prices["downtime"] for c, level in zip(COMPONENTS, initial) if level != HEALTHY[c])


def start_state_table(name, cycles, blocks):
    """{combination: [(seed, window start, entry offset)]}: every start state of the class environment, grouped by
    record, each with the smallest seed whose reset reaches it."""
    table = {}
    for seed, combo, window, entry in seed_table(class_env(name, cycles, blocks)):
        table.setdefault(combo, []).append((seed, window, entry))
    return table


def episode_numbers(env, policy, seed):
    """One episode's statistics; resolved as in docs/hydraulic_rig/environment.md."""
    s = score(run_episode(env, policy, seed))
    ref = reference_cost(env.prices, s["initial"])
    return {"top1": s["diagnosis_all_correct"], "recovered": s["recovered"],
            "resolved": s["diagnosis_all_correct"] and s["recovered"] and s["unnecessary_replacements"] == 0,
            "cost": s["cost"], "reference_cost": ref, "at_reference_cost": s["cost"] == ref,
            "unnecessary_replacements": s["unnecessary_replacements"], "steps": s["steps"], "refusals": s["refusals"]}


def class_statistics(records):
    """Class numbers from per-record means, every record weighing the same (exact on fractions, also used on floats)."""
    mean = {k: sum(r[k] for r in records) / len(records) for k in RECORD_MEANS}
    return {"top1": mean["top1"], "recovered": mean["recovered"], "resolved": mean["resolved"],
            "mean_cost": mean["cost"], "mean_reference_cost": mean["reference_cost"],
            "cost_per_resolved_episode": mean["cost"] / mean["resolved"] / mean["reference_cost"] if mean["resolved"] else None,
            "mean_unnecessary_replacements": mean["unnecessary_replacements"], "mean_steps": mean["steps"],
            "mean_refusals": mean["refusals"]}


def policy_numbers(env, table, make_policy):
    """One policy on every start state of the class (one episode each): exact per-record means, the number of
    episodes and of episodes at the reference cost, and the class numbers."""
    records, counts = [], {"episodes": 0, "episodes_at_reference_cost": 0}
    for seeds in table.values():
        episodes = [episode_numbers(env, make_policy(seed), seed) for seed, _, _ in seeds]
        records.append({k: Fraction(sum(e[k] for e in episodes), len(episodes)) for k in RECORD_MEANS})
        counts["episodes"] += len(episodes)
        counts["episodes_at_reference_cost"] += sum(e["at_reference_cost"] for e in episodes)
    return {"records": records, **counts, **class_statistics(records)}


def class_numbers(name, cycles, blocks, fold_classifiers):
    """Every policy's numbers on the class; the greedy row's numbers are keyed by fold seed."""
    env = class_env(name, cycles, blocks)
    table = start_state_table(name, cycles, blocks)
    combinations = class_combinations(name)
    guess = min(table)  # every record weighs 1/R on the table, so the heaviest combination is a tie, broken by sorted order
    makers = {
        "oracle": lambda seed: OraclePolicy(env),
        "do_nothing": lambda seed: DoNothingPolicy(),
        "replace_everything": lambda seed: ReplaceEverythingPolicy(seed, combinations),
        "random": lambda seed: RandomPolicy(seed),
        "symptom_only": lambda seed: SymptomOnlyPolicy(guess),
    }
    numbers = {p: policy_numbers(env, table, make) for p, make in makers.items()}
    numbers["symptom_only"]["diagnosis"] = guess
    numbers[GREEDY] = {clf.seed: policy_numbers(env, table, lambda seed, clf=clf: GreedyPolicy(clf, use_max_faulty=True))
                       for clf in fold_classifiers}
    return numbers


def anchor_checks(numbers):
    """The six anchors of docs/hydraulic_rig/environment.md as (anchor, policy, fold seed, requirement, holds, condition)."""
    o, d, rep, rnd, sym = (numbers[p] for p in ("oracle", "do_nothing", "replace_everything", "random", "symptom_only"))
    checks = [
        (1, "oracle", None, "Top-1 1.00, recovery 1.00 and resolved 1.00, every episode at the reference cost",
         o["top1"] == 1 and o["recovered"] == 1 and o["resolved"] == 1 and o["episodes_at_reference_cost"] == o["episodes"],
         f"Top-1 {o['top1']:.2f}, recovered {o['recovered']:.2f}, resolved {o['resolved']:.2f}, {o['episodes_at_reference_cost']} of "
         f"{o['episodes']} episodes at the reference cost (mean {o['mean_reference_cost']:.1f}), cost per resolved episode {o['cost_per_resolved_episode']:.2f}"),
        (2, "do_nothing", None, "submit the healthy combination, then finish: Top-1 0, recovery 0 and resolved 0",
         d["top1"] == 0 and d["recovered"] == 0 and d["resolved"] == 0,
         f"Top-1 {d['top1']:.2f}, recovered {d['recovered']:.2f}, resolved {d['resolved']:.2f}"),
        (3, "replace_everything", None, "resolved at most 0.20",
         rep["resolved"] <= Fraction(1, 5),
         f"resolved {rep['resolved']:.3f} (Top-1 {rep['top1']:.3f}, recovered {rep['recovered']:.3f}, "
         f"{rep['mean_unnecessary_replacements']:.2f} unnecessary replacements per episode)"),
        (4, "random", None, "Top-1 at most 0.10 and resolved at most 0.05; recovery reported, not anchored",
         rnd["top1"] <= Fraction(1, 10) and rnd["resolved"] <= Fraction(1, 20),
         f"Top-1 {rnd['top1']:.3f}, resolved {rnd['resolved']:.3f} (recovered {rnd['recovered']:.3f}, not anchored)"),
    ]
    for fold_seed, g in numbers[GREEDY].items():
        checks.append((5, GREEDY, fold_seed, "strictly between random and oracle on Top-1 and on the resolved rate, for every fold seed",
                       rnd["top1"] < g["top1"] < o["top1"] and rnd["resolved"] < g["resolved"] < o["resolved"],
                       f"fold seed {fold_seed}: Top-1 random {rnd['top1']:.3f} < greedy {g['top1']:.3f} < oracle {o['top1']:.3f}; "
                       f"resolved random {rnd['resolved']:.3f} < greedy {g['resolved']:.3f} < oracle {o['resolved']:.3f}"))
    checks.append((6, "symptom_only", None, "resolved at most 0.20",
                   sym["resolved"] <= Fraction(1, 5),
                   f"resolved {sym['resolved']:.3f}: it submits {sym['diagnosis']}, the first of the {len(sym['records'])} combinations, "
                   f"which all weigh 1/{len(sym['records'])} on the start-state table, and replaces its faulty components (Top-1 {sym['top1']:.3f}, "
                   f"recovered {sym['recovered']:.3f})"))
    return checks


def intervals(records, resamples):
    """{statistic: (low, high)}: 95% percentile intervals of the INTERVALS statistics over record resamples, blank where
    a resample resolves nothing."""
    floats = [{k: float(v) for k, v in r.items()} for r in records]
    draws = [class_statistics([floats[i] for i in idx]) for idx in resamples]
    out = {}
    for stat in INTERVALS:
        values = [d[stat] for d in draws]
        out[stat] = ("", "") if None in values else tuple(round(v, 4) for v in percentile_interval(values))
    return out


def rounded(value):
    return "" if value is None else round(float(value), 4)


def anchor_rows(name, numbers, resamples):
    rows = []
    for anchor, policy, fold_seed, requirement, holds, condition in anchor_checks(numbers):
        n = numbers[policy][fold_seed] if fold_seed is not None else numbers[policy]
        ci = intervals(n["records"], resamples)
        row = {"scenario": name, "start_combinations": len(n["records"]), "episodes": n["episodes"],
               "min_faulty": CLASSES[name], "max_faulty": CLASSES[name], "fault_count_rule": f"{CLASSES[name]} to {CLASSES[name]}",
               "anchor": anchor, "requirement": requirement, "policy": policy, "fold_seed": "" if fold_seed is None else fold_seed}
        for stat in ("top1", "recovered", "resolved", "mean_cost", "mean_reference_cost", "cost_per_resolved_episode"):
            row[stat] = rounded(n[stat])
            if stat in ci:
                row[f"{stat}_ci_low"], row[f"{stat}_ci_high"] = ci[stat]
        row.update({"episodes_at_reference_cost": n["episodes_at_reference_cost"],
                    "mean_unnecessary_replacements": rounded(n["mean_unnecessary_replacements"]), "mean_steps": rounded(n["mean_steps"]),
                    "mean_refusals": rounded(n["mean_refusals"]), "condition": condition, "holds": holds,
                    "greedy_split": f"{N_FOLDS} {SPLIT_RULE}", "weighting": WEIGHTING})
        rows.append(row)
    return rows


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--classes", nargs="+", default=list(CLASSES))
    a = ap.parse_args(argv)
    cycles = load_cycles(a.facts)
    blocks = build_blocks(cycles)
    fold_classifiers = [TierClassifiers(cycles, blocks, seed=seed) for seed in FOLD_SEEDS]
    for name in a.classes:
        numbers = class_numbers(name, cycles, blocks, fold_classifiers)
        resamples = cluster_resamples(len(class_combinations(name)), RESAMPLES, f"record-bootstrap:{name}")
        rows = anchor_rows(name, numbers, resamples)
        out = Path(a.out_dir) / f"scenario_anchors_{name}.csv"
        write_csv(out, rows)
        print(f"{name}: {rows[0]['start_combinations']} records, {rows[0]['episodes']} start states, fault-count rule {rows[0]['fault_count_rule']}")
        for r in rows:
            cost = r["cost_per_resolved_episode"]
            print(f"  anchor {r['anchor']} {r['policy']}{'' if r['fold_seed'] == '' else ' fold seed ' + str(r['fold_seed'])}: {'holds' if r['holds'] else 'FAILS'}; "
                  f"Top-1 {r['top1']}, recovered {r['recovered']}, resolved {r['resolved']} ({r['resolved_ci_low']} to {r['resolved_ci_high']}), "
                  f"mean cost {r['mean_cost']} (reference {r['mean_reference_cost']}), cost per resolved episode {cost if cost != '' else 'none resolved'}, "
                  f"steps {r['mean_steps']}")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
