"""E18: the diagnosability grade of a combination under a budget, the second separability label (version 8,
CausalDiagnosis/docs/causal-diagnosis-mcp.md, section E18).

The grade asks whether the rule of the server, played on the recordings that a row serves of the true combination, commits
to it, every level, as the score of a diagnosis asks, in expectation over rollouts. A rollout starts from the recordings
that the row records before cycle 0 at a drawn entry offset, as an episode does, under the fits of an episode of that
start, which leave the start's block out of every fit as the server does, on the belief of a static episode: a uniform
prior over every combination of the profile updated by the joint_block likelihood, which is the filter when no rule card
fires before the diagnosis. It reads that record as the rule of the server does, without charge, as the independent label
of the verifier reads it: every summary on a row whose rules price nothing, and on a row with prices the reads of the
record that the budget pays beside the repair of every component (record_reads). Wherever no clock runs, and on every row
with prices, the rule then diagnoses the most probable combination, so the grade is binary there: a combination has one,
at no cost and with no reading, or has none. On a row that prices nothing and has a clock the rollout goes on over the live
window of the same entry offset in play order, as the environment serves it, a cycle served twice entering once, while the
flag of decide is on. A combination on which no rollout commits to the truth has no grade. A rollout also records whether
its first commitment names the truth, and the binary label takes the rule of the independent label of the verifier, which
comes from the environment: a start is separable when more than half of the rollouts commit first to the truth
(`separable`). This grade is the second label beside it.

The budget counts the read prices alone, as the environment's budget counter counts prices only.

grades.json beside models.json holds one table per rules key, which covers the sensor groups, the prices, the durations,
the downtime rates, the horizon, the budget, whether a clock runs and the text of the rule, so that a table graded under
another rule is never read, and neither the fault rule of the starts nor the ranking model. The file names the SHA-256 of
the data file that its models were fitted on under facts_sha256, and a file graded on another data file is refused. A table
grades a combination the first time it is asked for it, with a generator seeded by the combination's place in the grid,
so its entries do not depend on the order of the requests, and it writes under a lock on the file, so that the processes
of one clone that grade at once keep each other's grades. The file is generated on use and ignored by git, like models.json,
and CausalDiagnosis/experiments/cluster_runs/precompute_grid_tables.py fills it for every rules key of a grid before the grid's lines start.

usage: python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade [--budget B] [--rollouts R] [--samples S] [--force]
       grades every recorded combination under the default rules of the asset.
       python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade --asset-id ID --models PATH --out PATH [options] grades another registered
       asset from its own models file into a grades file of its own.
"""
import argparse
import hashlib
import json
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from CausalDiagnosis.utils.atomic_writes_and_locks import locked, refuse_writing, write_json_atomically
from CausalDiagnosis.datasets.dataset_loader.dataset_packages import table_path

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import asset_view, default_facts_path, profile_view, recorded_blocks, require_same_data, served_pairs
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import cost_unit, has_clock

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import plan_completion, plan_terms
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import costs, decide
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import combo_key
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import record_read
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import REFIT, code_digest, facts_sha256, refuse_default_files
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import left_of

GRADES_PATH = table_path(Path(__file__).resolve().with_name("grades.json"))
GRADES_FORMAT = 9  # moves with the fits and the rule a rollout plays, which the rules key does not hold: 9 since the second block is a neighbor of the start
ROLLOUTS = 10  # rollouts over the true block per combination, a design parameter
GRADE_SAMPLES = 128  # Monte Carlo draws per test inside the grade's decisions and information gains
GRADE_SEED = 0
RULE_KEYS = ("sensor_groups", "prices", "durations", "downtime_rate", "horizon", "budget")
RULE_TEXT = ("the rule of the server on what the row serves: the recordings before cycle 0 read without charge, every summary "
             "where the rules price nothing and the reads that the budget pays beside the repair of every component where they "
             "hold prices, then the diagnosis of the most probable combination, or where a clock runs and nothing is priced the "
             "live window read by gain per call while the flag of decide is on; the mean cost of the rollouts that commit to "
             "the true combination, every level, null when none does, and binary at no cost where the live window is not read")
HEADER_KEYS = ("rules_key", "budget", "rollouts", "samples", "seed", "rule", "code")  # the settings a stored table keeps beside its grades
REGRADE = "Remove it and grade again with `python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables`."
Rollout = namedtuple("Rollout", "committed cost prices readings first_right")


def entropy_bits(p):
    p = np.clip(p, 1e-300, None)
    return -(p * np.log2(p)).sum(axis=-1)


def expected_information_gain(post, fields, samples, rng):
    """Mutual information in bits between the combination and one reading of the given summaries, by Monte Carlo over the
    predictive mixture: the entropy of the belief minus the expected entropy after the reading."""
    updated = post.update_with(fields, post.sample_next(fields, samples, rng))
    return float(entropy_bits(post.probabilities) - entropy_bits(updated).mean())


def static_belief(effect, served, block_variance):
    """The belief after the served recordings {key: (fields, values)}, each recording once with the union of its summaries."""
    post = Posterior.uniform(effect, block=block_variance)
    for fields, values in served.values():
        post = post.updated(fields, values)
    return post


def served_of(space, recording, summaries):
    """(fields, values) of one recording for the summaries shown of it that the effect space knows, in its field order and
    at the six significant digits that the environment serves."""
    fields = tuple(f for f in space.fields if f in summaries)
    return fields, np.array([float(f"{recording[space.field_index[f]]:.6g}") for f in fields])


def record_reads(view, prehistory, budget):
    """{position: summaries} of the recordings before cycle 0 that the rule of the server reads: every summary of every
    recording on a row whose rules price nothing, and on a row with prices the reads that elimination.record_read names in
    turn for what the record can still show each group (serving.left_of), the money of the budget left beside the repair of
    every component, the calls and the cycles of the row, until it names none. A read of n cycles takes the n newest
    recordings that do not show every summary of the group yet."""
    window = len(prehistory)
    if cost_unit(view.rules["prices"]) == "calls":
        every = set().union(*(view.group_fields(g) for g in view.sensor_groups))
        return {p: every for p in range(window)}
    cost = costs(view, 0, steps_left=view.rules["max_steps"])
    price, cycles = plan_terms(view, list(view.components))
    reserve = (price, plan_completion(view, list(view.components)).calls, cycles)
    shown, money, calls, cycles_left = {}, float("inf") if budget is None else budget, view.rules["max_steps"], view.rules["horizon"]
    while True:
        read = record_read(left_of(view, list(shown.values()), window), cost, money, reserve, calls, cycles_left)
        if not read:
            return shown
        (group, count), = read
        fields = set(view.group_fields(group))
        for p in [p for p in reversed(range(window)) if not fields <= shown.get(p, set())][:count]:
            shown[p] = shown.get(p, set()) | fields
        money, calls, cycles_left = money - count * cost.test_price[group], calls - 1, cycles_left - count * cost.test_duration[group]


def rollout(view, effect, start, prehistory, live, budget, rng, samples=GRADE_SAMPLES, block_variance=True):
    """One rollout from the start: the belief after the recordings before cycle 0 that the rule reads (record_reads), read
    without charge. Wherever no clock runs and on every row with prices the rule then diagnoses, so the rollout commits to
    the most probable combination, at no cost and with no reading. On a row that prices nothing and has a clock it goes on
    over the live window in play order: it reads the sensor group of largest information gain per call among those that
    the window can still serve, one call each, until the flag of decide is off on the true combination, every level, or no
    read is left within the budget. A commitment to a wrong combination does not end that search, since the grade asks for
    the cheapest sequence after which the rule commits to the true one, and first_right says whether the first commitment
    of the rollout, which an agent acting on the rule would have taken, named the true one."""
    space = effect.space
    served = {("before", p): served_of(space, prehistory[p], summaries) for p, summaries in sorted(record_reads(view, prehistory, budget).items())}
    served = {key: entry for key, entry in served.items() if entry[0]}
    post = static_belief(effect, served, block_variance)
    if cost_unit(view.rules["prices"]) == "money" or not has_clock(view.rules["clock_mode"], view.rules["durations"]):
        right = tuple(post.candidates[post.order()[0]]) == tuple(start)
        return Rollout(right, 0.0, 0.0, 0, right)
    shown, calls, t = {}, 0.0, 0
    committed_elsewhere = False
    while True:
        cost = costs(view, t, steps_left=view.rules["max_steps"] - t)
        left = left_of(view, list(shown.values()), len(live))
        decision = decide(post, cost, samples, seed=int(rng.integers(2 ** 31)), serving=left)
        if not decision.read_first:
            if tuple(post.candidates[post.order()[0]]) == tuple(start):
                return Rollout(True, calls, calls, t, not committed_elsewhere)
            committed_elsewhere = True
        affordable = [g for g in sorted(cost.test_price) if left[g].cycles > 0 and calls + cost.test_price[g] <= budget]
        if not affordable:
            return Rollout(False, calls, calls, t, False)
        gains = {g: expected_information_gain(post, space.groups[g], samples, rng) / cost.test_price[g] for g in affordable}
        group = max(gains, key=lambda g: (gains[g], g))
        index = t % len(live)
        before = shown.get(index, set())
        shown[index] = before | set(view.group_fields(group))
        if shown[index] != before:
            fields, values = served_of(space, live[index], shown[index])
            if ("live", index) in served:  # a recording shown again through more summaries enters once, with the union
                served[("live", index)] = (fields, values)
                post = static_belief(effect, served, block_variance)
            elif fields:
                served[("live", index)] = (fields, values)
                post = post.updated(fields, values)
        t += 1
        calls += cost.test_price[group]


def serving_of(episode, block):
    """serving(rng) of grade: the fits of an episode of the start, and the recordings before cycle 0 and the live window that
    the evidence source serves at a drawn entry offset."""
    pairs = served_pairs()

    def serving(rng):
        before, live = pairs[int(rng.integers(len(pairs)))]
        return episode.effect, block[before], block[live]
    return serving


def grade(view, serving, start, budget, rng, rollouts=ROLLOUTS, samples=GRADE_SAMPLES):
    """E18 for one combination: None when no rollout commits to it within the budget, else the mean cost of
    the committing rollouts with the share of rollouts that committed and the share whose first commitment named it. serving(rng)
    gives the fits, the recordings before cycle 0 and the live window of a rollout."""
    outcomes = []
    for _ in range(rollouts):
        effect, prehistory, live = serving(rng)
        outcomes.append(rollout(view, effect, start, prehistory, live, budget, rng, samples))
    committed = [o for o in outcomes if o.committed]
    if not committed:
        return None
    return {"start": list(start), "budget": budget, "cost": float(np.mean([o.cost for o in committed])), "commit_share": len(committed) / rollouts,
            "first_share": sum(o.first_right for o in outcomes) / rollouts, "readings": float(np.mean([o.readings for o in committed])),
            "rollouts": rollouts}


def separable(entry):
    """The binary label of E18, the majority rule of the independent label: more than half of the rollouts commit first to
    the true combination. A combination without a grade, on which no rollout commits to the truth, is not separable."""
    return entry is not None and entry["first_share"] > 0.5


def rules_key(rules, budget):
    """One grade table per sensor group table, price table, duration table, downtime table, horizon, budget, clock and rule
    text, so that two rows that differ only in whether a clock runs, and so in what they serve, grade apart."""
    relevant = {k: rules[k] for k in RULE_KEYS if k in rules}
    relevant["budget"] = budget
    relevant["clock"] = has_clock(rules["clock_mode"], rules["durations"])
    relevant["rule"] = RULE_TEXT
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def _data_of(models):
    """The SHA-256 of the data file that the models were fitted on, None without models."""
    return None if models is None else models.raw["facts_sha256"]


def _stored(path, facts_sha256):
    """The content of the grades file, an empty one of the current format for a missing file or one of another format. A file
    of the current format graded on another data file than facts_sha256 names, or naming none, is refused."""
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if stored.get("format") != GRADES_FORMAT:
        return {"format": GRADES_FORMAT, "facts_sha256": facts_sha256, "tables": {}}
    require_same_data(path, stored.get("facts_sha256"), facts_sha256, REGRADE)
    return stored


class GradeTable(Mapping):
    """The grades of one rules key over every recorded combination of the profile, each computed on first access and
    written to grades.json at once, so that a later process reads it instead of computing it again."""

    def __init__(self, models, rules, header, path, facts_path=None):
        self.models, self.rules, self.header, self.path, self.facts_path = models, rules, header, Path(path), facts_path
        self.view = profile_view(rules)
        self._blocks = None
        stored = _stored(self.path, _data_of(models))["tables"].get(header["rules_key"], {})
        # a table written by other code is graded again, so a grade never outlives a change of the server's code
        self._grades = dict(stored.get("grades", {})) if stored.get("code") == header.get("code") else {}

    def blocks(self):
        """The recorded blocks of the data file, read only when the models were fitted on it."""
        if self._blocks is None:
            path = self.facts_path or default_facts_path(self.view.asset_id)
            require_same_data("The models file of this grade table", _data_of(self.models), facts_sha256(path), REFIT)
            self._blocks = recorded_blocks(self.view, path)[0]
        return self._blocks

    def recorded(self):
        """The keys of the combinations of the grid that have a recorded block, in grid order."""
        return [combo_key(h) for h in self.view.combinations if h in self.blocks()]

    def __iter__(self):
        return iter(self.recorded())

    def __len__(self):
        return len(self.recorded())

    def __contains__(self, key):
        return key in self._grades or key in set(self.recorded())

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        if key not in self._grades:
            # the read of the file, the grade and the write run under one lock, so two processes that grade at once keep both
            # grades, and a process that waited for the lock reads the grade the other wrote instead of grading again
            with locked(self.path):
                stored = _stored(self.path, _data_of(self.models))
                table = stored["tables"].get(self.header["rules_key"])
                if table is None or table.get("code") != self.header.get("code"):
                    table = stored["tables"][self.header["rules_key"]] = {**self.header, "grades": {}}
                if key not in table["grades"]:
                    refuse_writing(f"the grade of the combination {key} of the grade table {self.header['rules_key'][:12]}")
                    start = tuple(int(v) for v in key.split(","))
                    rng = np.random.default_rng([self.header["seed"], self.view.combinations.index(start)])
                    serving = serving_of(self.models.without(start), self.blocks()[start])
                    table["grades"][key] = grade(self.view, serving, start, self.header["budget"], rng, self.header["rollouts"], self.header["samples"])
                    write_json_atomically(self.path, stored, indent=1)
                self._grades[key] = table["grades"][key]
        return self._grades[key]


def ensure_grades(models, rules, budget=None, ranking_model=None, path=None, rollouts=ROLLOUTS, samples=GRADE_SAMPLES, seed=GRADE_SEED,
                  facts_path=None):
    """The grade table for the rules and the budget, whose grades are computed on first use and kept beside models.json.
    ranking_model is accepted for the drivers' call and enters nothing: E18 runs the joint_block belief whatever ranking a
    run serves. The settings of a stored table win over the arguments, since its entries were computed with them. Without a
    path the table is GRADES_PATH as the module holds it at the call, which the tests point at a copy."""
    path = GRADES_PATH if path is None else path
    budget = budget_of(rules) if budget is None else budget
    key = rules_key(rules, budget)
    stored = _stored(Path(path), _data_of(models))["tables"].get(key)
    if stored is not None and stored.get("code") != code_digest():  # a table of other code is graded anew with today's settings
        stored = None
    header = {k: stored[k] for k in HEADER_KEYS} if stored else \
        {"rules_key": key, "budget": budget, "rollouts": rollouts, "samples": samples, "seed": seed, "rule": RULE_TEXT,
         "code": code_digest()}
    return {**header, "grades": GradeTable(models, rules, header, path, facts_path)}


def table_problems(models, rules, budget=None, path=None, facts_path=None):
    """What a check of the tables finds wrong with the grade table of the rules: missing, written by other code, or short of a
    recorded combination. Reads the file and writes nothing."""
    path = Path(GRADES_PATH if path is None else path)
    budget = budget_of(rules) if budget is None else budget
    key = rules_key(rules, budget)
    stored = _stored(path, _data_of(models))["tables"].get(key)
    if stored is None:
        return [f"grades {key[:12]}: missing"]
    if stored.get("code") != code_digest():
        return [f"grades {key[:12]}: of other code"]
    table = GradeTable(models, rules, {k: stored[k] for k in HEADER_KEYS}, path, facts_path)
    short = [combination for combination in table.recorded() if combination not in stored["grades"]]
    return [f"grades {key[:12]}: {len(short)} recorded combinations without a grade"] if short else []


def budget_of(rules):
    """The budget of the rules, or on the free base row, which has none, the step cap in calls."""
    budget = rules.get("budget")
    return rules["max_steps"] if budget is None and cost_unit(rules["prices"]) == "calls" else budget


def grade_of(start, models, rules, budget=None, path=None):
    """The grade of a combination under the rules, or None when no rollout commits to it under the budget."""
    return ensure_grades(models, rules, budget, path=path)["grades"].get(combo_key(start))


def default_rules(asset_id=None):
    """The public rules of the default episode of an asset, the default asset when asset_id is None, the single source of
    the default tables."""
    return dict(asset_view(asset_id).rules)


def main(argv=None):
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models

    parser = argparse.ArgumentParser(description="grade every recorded combination under the default rules of the asset, beside models.json")
    parser.add_argument("--asset-id", help="a registered asset profile, the default asset when omitted")
    parser.add_argument("--models", type=Path, help="the models file of the asset, models.json of the default asset when omitted")
    parser.add_argument("--budget", type=float, default=None, help="defaults to the configuration's budget")
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--samples", type=int, default=GRADE_SAMPLES)
    parser.add_argument("--seed", type=int, default=GRADE_SEED)
    parser.add_argument("--force", action="store_true", help="recompute the table even when grades.json holds one for these rules")
    parser.add_argument("--out", type=Path, default=GRADES_PATH)
    args = parser.parse_args(argv)
    refuse_default_files(parser, args, models=None, out=GRADES_PATH)
    rules = default_rules(args.asset_id)
    budget = budget_of(rules) if args.budget is None else args.budget
    models = load_models(args.models)
    if args.force and args.out.exists():
        # the drop of the table reads, changes and writes the file that grading processes add to, so it holds their lock
        with locked(args.out):
            stored = _stored(args.out, _data_of(models))
            stored["tables"].pop(rules_key(rules, budget), None)
            write_json_atomically(args.out, stored, indent=1)
    table = ensure_grades(models, rules, budget, path=args.out, rollouts=args.rollouts, samples=args.samples, seed=args.seed)
    graded = {key: table["grades"][key] for key in table["grades"]}
    print(f"{args.out}: budget {budget:g}, {table['rollouts']} rollouts, {table['samples']} samples, seed {table['seed']}, "
          f"{len(graded)} combinations, {sum(1 for v in graded.values() if separable(v))} separable")


if __name__ == "__main__":
    main()
