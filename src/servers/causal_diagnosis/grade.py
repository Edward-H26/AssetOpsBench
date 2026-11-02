"""E18 of design v2: the diagnosability grade of a start combination under a budget (docs/hydraulic_rig/causal-diagnosis-server.md).

The grade is the cost of the cheapest test sequence within the budget after which the decision rule of E10 commits to the
correct faulty set, in expectation over resampled cycles of the true combination. The search is the greedy rule of the
environment's reference policy, the reading with the largest expected information gain per unit price, evaluated under
the joint model. A start is inseparable under the budget when no rollout commits to the correct set within it. Because
the grade and the declaration use one rule, an abstention is correct exactly on the starts the grade marks inseparable.
grades.json beside models.json holds the grades of every evaluation combination under the rules it was computed for; it
is generated on first use and ignored by git, like models.json.

usage: HYD_FACTS_PATH=<facts file> python -m servers.causal_diagnosis.grade [--ranking-model joint_block] [--budget B] [--force]
       regenerates the table of the default episode configuration (envs.hydraulic_rig.config.EpisodeConfig).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from servers.json_files import write_json_atomically

from .decision import costs, decide, downtime_rate
from .effect_model import FIELD_INDEX, GROUP_FIELDS, PoolData, combo_key, parse_combo
from .posterior import DEFAULT_RANKING_MODEL, Posterior, Reading

GRADES_PATH = Path(__file__).resolve().with_name("grades.json")
GRADES_FORMAT = 1
ROLLOUTS = 10  # rollouts over resampled cycles of the true block per start, a design parameter
GRADE_SAMPLES = 128  # Monte Carlo draws per test inside the grade's decisions and information gains
GRADE_SEED = 0
RULE_KEYS = ("prices", "durations", "downtime_rate", "horizon", "fault_rule", "budget")


def faulty_set(combo):
    return frozenset(c for j, c in enumerate(COMPONENTS) if combo[j] != HEALTHY[c])


def entropy_bits(p):
    p = np.clip(p, 1e-300, None)
    return -(p * np.log2(p)).sum(axis=-1)


def expected_information_gain(post, fields, samples, rng):
    """Mutual information in bits between the start combination and one reading of the given summaries, by Monte Carlo over
    the predictive mixture: the entropy of the posterior minus the expected entropy after the reading."""
    updated = post.update_with(fields, post.sample_next(fields, samples, rng))
    return float(entropy_bits(post.probabilities) - entropy_bits(updated).mean())


def rollout(start, budget, effect, rules, block, rng, samples=GRADE_SAMPLES, block_variance=False):
    """One rollout from the start: read the group with the largest information gain per price, charging the price and one
    cycle of the true downtime, until E10 commits to the true faulty set or the budget allows no further reading. A
    commitment to a wrong set does not end the search, since the grade asks for the cheapest sequence after which the
    rule commits to the correct set. Returns whether that happened, the cost spent and the number of readings."""
    readings, spent, t = [], 0.0, 0
    truth = faulty_set(start)
    while True:
        cost = costs(rules, t)
        post = Posterior(effect, rules, readings, [], time=t, block_variance=block_variance)
        decision = decide(post, cost, samples, seed=int(rng.integers(2 ** 31)))
        if not decision.insufficient_evidence and faulty_set(post.candidates[post.order()[0]]) == truth:
            return True, spent, len(readings)
        rate = downtime_rate(start, cost)
        affordable = [g for g in sorted(cost.test_price) if spent + cost.test_price[g] + rate <= budget]
        if not affordable:
            return False, spent, len(readings)
        gains = {g: expected_information_gain(post, GROUP_FIELDS[g], samples, rng) / cost.test_price[g] for g in affordable}
        group = max(gains, key=lambda g: (gains[g], g))
        t += 1
        cycle = block[rng.integers(len(block))]
        readings.append(Reading(len(readings) + 1, t, group, {f: float(f"{cycle[FIELD_INDEX[f]]:.6g}") for f in GROUP_FIELDS[group]}))
        spent += cost.test_price[group] + rate


def grade(start, budget, effect, rules, block, rng, rollouts=ROLLOUTS, samples=GRADE_SAMPLES, block_variance=False):
    """E18 for one start: None when no rollout commits to the true faulty set within the budget, else the mean cost of the
    committing rollouts with the share of rollouts that committed."""
    outcomes = [rollout(start, budget, effect, rules, block, rng, samples, block_variance) for _ in range(rollouts)]
    committed = [spent for ok, spent, _ in outcomes if ok]
    if not committed:
        return None
    return {"start": list(start), "budget": budget, "cost": float(np.mean(committed)), "commit_share": len(committed) / rollouts,
            "readings": float(np.mean([n for ok, _, n in outcomes if ok])), "rollouts": rollouts}


def rules_key(rules, budget, ranking_model=DEFAULT_RANKING_MODEL):
    """One grade table per price table, duration table, downtime table, horizon, fault rule, budget and ranking model."""
    relevant = {k: rules[k] for k in RULE_KEYS if k in rules}
    relevant.update({"budget": budget, "ranking_model": ranking_model})
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def compute_grades(models, rules, budget, ranking_model=DEFAULT_RANKING_MODEL, rollouts=ROLLOUTS, samples=GRADE_SAMPLES, seed=GRADE_SEED, facts_path=None):
    """Grades of every evaluation combination of the models' pool under the rules, the budget and the ranking model (joint
    or joint_block, the two models with a Gaussian predictive)."""
    if ranking_model not in ("joint", "joint_block"):
        raise ValueError("the grade is computed under joint or joint_block")
    cycles = load_cycles(facts_path or default_facts_path())
    blocks = build_blocks(cycles)
    evaluation = [parse_combo(key) for key in models.pool["evaluation"]]
    data = PoolData(cycles, blocks, evaluation)
    rng = np.random.default_rng(seed)
    grades = {}
    for start, block in zip(data.pool, data.arrays):
        grades[combo_key(start)] = grade(start, budget, models.effect, rules, block, rng, rollouts, samples, ranking_model == "joint_block")
    return {"rules_key": rules_key(rules, budget, ranking_model), "budget": budget, "ranking_model": ranking_model, "rollouts": rollouts,
            "samples": samples, "seed": seed,
            "rule": "mean cost of the rollouts in which E10 commits to the true faulty set within the budget; null when none does",
            "grades": grades}


def ensure_grades(models, rules, budget=None, ranking_model=DEFAULT_RANKING_MODEL, path=GRADES_PATH, **options):
    """The grade table for the rules, the budget and the ranking model, computed on first use and kept beside models.json."""
    budget = rules["budget"] if budget is None else budget
    key = rules_key(rules, budget, ranking_model)
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"format": GRADES_FORMAT, "tables": {}}
    if stored.get("format") != GRADES_FORMAT:
        stored = {"format": GRADES_FORMAT, "tables": {}}
    if key not in stored["tables"]:
        stored["tables"][key] = compute_grades(models, rules, budget, ranking_model, **options)
        write_json_atomically(path, stored, indent=1)
    return stored["tables"][key]


def grade_of(start, models, rules, budget=None, ranking_model=DEFAULT_RANKING_MODEL, path=GRADES_PATH):
    """The grade of a start under the rules, or None when the start is inseparable under the budget."""
    return ensure_grades(models, rules, budget, ranking_model, path)["grades"].get(combo_key(start))


def default_rules():
    """The released rules of the default episode configuration, the single source of the placeholder tables."""
    from envs.hydraulic_rig.config import EpisodeConfig  # imported here so that the module does not depend on the environment package

    config = EpisodeConfig().to_json()
    return {key: config[key] for key in RULE_KEYS + ("sensor_groups",)}


def main(argv=None):
    from .adapter import load_models

    parser = argparse.ArgumentParser(description="regenerate the E18 grade table of the default configuration beside models.json")
    parser.add_argument("--ranking-model", choices=("joint", "joint_block"), default=DEFAULT_RANKING_MODEL)
    parser.add_argument("--budget", type=float, default=None, help="defaults to the configuration's budget")
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--samples", type=int, default=GRADE_SAMPLES)
    parser.add_argument("--seed", type=int, default=GRADE_SEED)
    parser.add_argument("--force", action="store_true", help="recompute the table even when grades.json holds one for these rules")
    parser.add_argument("--out", type=Path, default=GRADES_PATH)
    args = parser.parse_args(argv)
    rules = default_rules()
    budget = rules["budget"] if args.budget is None else args.budget
    if args.force and args.out.exists():
        stored = json.loads(args.out.read_text(encoding="utf-8"))
        stored.get("tables", {}).pop(rules_key(rules, budget, args.ranking_model), None)
        write_json_atomically(args.out, stored, indent=1)
    table = ensure_grades(load_models(), rules, budget, args.ranking_model, args.out, rollouts=args.rollouts, samples=args.samples, seed=args.seed)
    print(f"{args.out}: table for ranking_model {args.ranking_model}, budget {budget:g}, {table['rollouts']} rollouts, {table['samples']} samples, "
          f"seed {table['seed']}, {len(table['grades'])} starts")


if __name__ == "__main__":
    main()
