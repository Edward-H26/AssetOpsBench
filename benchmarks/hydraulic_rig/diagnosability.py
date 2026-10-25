"""Study B: how much evidence separates each start state's fault from its look-alikes, read from the server's posterior.

usage, from the fork root in the runtime of requirements-smoke.txt (no model, no MCP; the tools run in process):
  uv run --no-project --python 3.12 --with-requirements benchmarks/hydraulic_rig/requirements-smoke.txt \\
      python benchmarks/hydraulic_rig/diagnosability.py [--scenarios S1 S2] [--per-record K] [--max-records N]
      [--out-dir benchmarks/hydraulic_rig/results] [--keep DIR]

Every start state of a scenario class climbs a ladder of evidence: one reading of tier 1; tiers 1 and 2; all three tiers;
then nine more readings of each tier, ordered so that every tier reads every cycle of the served window of ten once:
the environment serves the window's cycles in turn whatever the tier, and a cycle read again at the same tier adds
nothing, so the last rung is all the evidence a measurement can buy. After each rung the server ranks the candidates. A start state is graded by the cheapest rung at which the true combination is the top candidate at the
server's confidence of 0.95; a state that never gets there within the ladder is inseparable by measurement alone, and
its unseparated components at the last rung name the look-alikes. The grade is the server's reading of the recorded
cycles, not a property proved about the rig: the models were fitted on the pool outside S2, so on S2 the grade also
measures how the models carry over to combinations they never saw.
"""
import argparse
import json
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (CLASSES, load_world, record_sizes_line, record_weighted_mean, scenario_env_kwargs, scenario_episodes, scenario_task_text,  # noqa: E402
                    start_episode)
from envs.hydraulic_rig.classifiers import TIERS  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS  # noqa: E402
from envs.hydraulic_rig.reporting import write_csv  # noqa: E402
from servers.causal_diagnosis.adapter import CONFIDENCE, CausalTools, load_models  # noqa: E402
from servers.causal_diagnosis.fit import ensure_models  # noqa: E402
from servers.hydraulic_rig import episode  # noqa: E402

BLOCK = 10  # cycles of the served window; the eleventh reading of an episode, at any tier, serves the first one again
FIRST_RUNGS = (("tier 1", [1]), ("tiers 1 and 2", [2]), ("tiers 1 to 3", [3]))


def _rest_of_window(earlier, tiers=TIERS, window=BLOCK):
    """The tiers of the readings after `earlier`, the tier of every reading so far, such that every tier reads every
    cycle of the window once. Reading k of an episode serves cycle k of the window, modulo its length, whatever the
    tier, so each further reading goes to the lowest tier that has not yet read that cycle."""
    unread = {t: set(range(window)) - {k % window for k, tier in enumerate(earlier) if tier == t} for t in tiers}
    plan = []
    for k in range(len(earlier), window * len(tiers)):
        tier = next(t for t in tiers if k % window in unread[t])
        unread[tier].discard(k % window)
        plan.append(tier)
    assert not any(unread.values()), "the readings cannot cover the window at every tier"
    return plan


LADDER = (*FIRST_RUNGS, ("tiers 1 to 3, ten readings each", _rest_of_window([t for _, rung in FIRST_RUNGS for t in rung])))
INSEPARABLE = "inseparable within the ladder"
CSV_NAME = "diagnosability.csv"
SUMMARY_NAME = "diagnosability_summary.md"


def _measure(state_path, tier):
    return json.loads(episode.call(state_path, "measure", {"tier": tier}, {"type": "measure", "tier": tier}))


def one_state(ep_dir, seed, env_kwargs, facts_path, scenario, models):
    """The rows of one start state, one per rung, all carrying the state's grade."""
    state_path = start_episode(ep_dir, seed, env_kwargs, facts_path, task_text=scenario_task_text(scenario))
    tools = CausalTools(models, episode.released_path(state_path))
    hidden = episode.replay(episode.load_state(state_path)).initial
    rows, readings, cost = [], 0, 0
    for rung, tiers in LADDER:
        for tier in tiers:
            answer = _measure(state_path, tier)
            readings += 1
            cost += answer["cost"]
        ranked = tools.rank_root_causes(3)
        top = ranked.candidates[0]
        top_combo = tuple(top.combination[c] for c in COMPONENTS)
        rows.append({"scenario": scenario, "record": json.dumps(list(hidden)), "seed": seed, "rung": rung, "readings": readings, "cost": cost,
                     "top_candidate": json.dumps(list(top_combo)), "top_probability": round(top.probability, 6), "top_is_truth": top_combo == hidden,
                     "insufficient_evidence": ranked.insufficient_evidence, "unseparated": " ".join(ranked.unseparated),
                     "truth_probability": round(_truth_probability(ranked, hidden), 6)})
    grade = next((r["rung"] for r in rows if r["top_is_truth"] and not r["insufficient_evidence"]), INSEPARABLE)
    for r in rows:
        r["grade"] = grade
    return rows


def _truth_probability(ranked, hidden):
    """The posterior mass on the true combination, or, when it is not among the returned candidates, the product of its
    per-component level probabilities, an approximation that treats the components as independent."""
    for candidate in ranked.candidates:
        if tuple(candidate.combination[c] for c in COMPONENTS) == hidden:
            return candidate.probability
    product = 1.0
    for c, level in zip(COMPONENTS, hidden):
        product *= ranked.level_probability[c][str(level)]
    return product


def study(scenarios, per_record=None, max_records=None, work_dir=None, facts=None, progress=None):
    facts_path, cycles, blocks = load_world(facts)
    ensure_models()
    models = load_models()
    rows = []
    for scenario in scenarios:
        env_kwargs = scenario_env_kwargs(scenario)
        for seed, _combo in scenario_episodes(scenario, cycles, blocks, per_record=per_record, max_records=max_records):
            state_rows = one_state(Path(work_dir) / scenario / f"seed{seed:06d}", seed, env_kwargs, facts_path, scenario, models)
            rows += state_rows
            if progress:
                progress(state_rows)
    return rows


def _fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def summarise(rows):
    out = ["# Diagnosability of the start states", "",
           f"A start state's grade is the cheapest rung at which the server's top candidate is the true combination at confidence "
           f"{CONFIDENCE}; the rungs are {', '.join(name for name, _ in LADDER)}. The posterior comes from the server's fixed-pool "
           "models over the released readings, so a grade is the server's reading of the recorded cycles. The S1 combinations belong "
           "to the pool on which the classifiers were fitted, while the S2 combinations are held out. Records hold unequal numbers "
           "of start states, so the shares of the grades and of the top candidates that are the truth are also given with each "
           "record weighted equally.", ""]
    rungs = [name for name, _ in LADDER]
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == scenario]
        states = {}
        for r in rs:
            states.setdefault((r["record"], r["seed"]), r["grade"])
        grades = Counter(states.values())
        out += [f"## {scenario}: {len(states)} start states of {len({r['record'] for r in rs})} records", "",
                "| grade | start states | share | share, records weighted equally |", "|---|---|---|---|"]
        state_rows = [{"record": record, "grade": grade} for (record, _seed), grade in states.items()]
        for name in [*rungs, INSEPARABLE]:
            out.append(f"| {name} | {grades.get(name, 0)} | {_fmt(grades.get(name, 0) / len(states))} | "
                       f"{_fmt(record_weighted_mean(state_rows, lambda r: r['grade'] == name))} |")
        out += ["", record_sizes_line(r["record"] for r in state_rows), "",
                "| rung | readings | cost | top is truth | top is truth, records weighted equally | insufficient | mean top probability | "
                "mean probability of the truth |", "|---|---|---|---|---|---|---|---|"]
        for name in rungs:
            at = [r for r in rs if r["rung"] == name]
            out.append(f"| {name} | {at[0]['readings']} | {at[0]['cost']} | {_fmt(statistics.mean(r['top_is_truth'] for r in at))} | "
                       f"{_fmt(record_weighted_mean(at, lambda r: r['top_is_truth']))} | "
                       f"{_fmt(statistics.mean(r['insufficient_evidence'] for r in at))} | {_fmt(statistics.mean(r['top_probability'] for r in at))} | "
                       f"{_fmt(statistics.mean(r['truth_probability'] for r in at))} |")
        last = [r for r in rs if r["rung"] == rungs[-1] and r["grade"] == INSEPARABLE]
        if last:
            unseparated = Counter(c for r in last for c in r["unseparated"].split())
            out += ["", f"Components left unseparated at the last rung over the {len(last)} inseparable start states: "
                    + ", ".join(f"{c} {n}" for c, n in unseparated.most_common()) + "."]
            wrong = Counter(r["top_candidate"] for r in last if not r["top_is_truth"])
            if wrong:
                out.append("Top candidate other than the truth at the last rung, by combination: " + "; ".join(f"{k} {n}" for k, n in wrong.most_common(8)) + ".")
        by_record = defaultdict(Counter)
        for (record, _seed), grade in states.items():
            by_record[record][grade] += 1
        out += ["", "| record | start states | " + " | ".join(rungs) + f" | {INSEPARABLE} |", "|---|---|" + "---|" * (len(rungs) + 1)]
        for record, counter in sorted(by_record.items()):
            out.append(f"| {record} | {sum(counter.values())} | " + " | ".join(str(counter.get(name, 0)) for name in [*rungs, INSEPARABLE]) + " |")
        out.append("")
    return "\n".join(out)


def main(argv):
    ap = argparse.ArgumentParser(description="grade every start state by the evidence that separates its fault, read from the causal tools")
    ap.add_argument("--scenarios", nargs="+", choices=list(CLASSES), default=list(CLASSES))
    ap.add_argument("--per-record", type=int, default=None)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--facts", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--keep", default=None, help="keep the episode folders under this directory (default: a temporary directory, deleted)")
    a = ap.parse_args(argv)
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="diagnosability_"))
    try:
        rows = study(a.scenarios, a.per_record, a.max_records, work, a.facts,
                     progress=lambda state_rows: print(f"{state_rows[0]['scenario']} seed {state_rows[0]['seed']} {state_rows[0]['record']}: {state_rows[0]['grade']}", flush=True))
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    out_dir = Path(a.out_dir)
    write_csv(out_dir / CSV_NAME, rows)
    (out_dir / SUMMARY_NAME).write_text(summarise(rows), encoding="utf-8")
    print(f"wrote {out_dir / CSV_NAME} ({len(rows)} rows) and {out_dir / SUMMARY_NAME}")


if __name__ == "__main__":
    main(sys.argv[1:])
