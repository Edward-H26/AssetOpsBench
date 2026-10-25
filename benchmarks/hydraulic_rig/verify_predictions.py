"""Study A: the causal-diagnosis server predicts what a replacement will do, and the recorded cycles of the rig verify it.

usage, from the fork root in the runtime of requirements-smoke.txt (no model, no MCP; the tools run in process):
  uv run --no-project --python 3.12 --with-requirements benchmarks/hydraulic_rig/requirements-smoke.txt \\
      python benchmarks/hydraulic_rig/verify_predictions.py [--scenarios S1 S2] [--per-record K] [--max-records N]
      [--out-dir benchmarks/hydraulic_rig/results] [--keep DIR]

For every start state of a scenario class and every component, one episode: the agent side measures once at each tier,
ranks the candidates, asks the server what replacing each component would do (four predictions), submits the top
candidate as the diagnosis, replaces the one component, measures the full tier once more where the episode continues,
and asks the server to verify the prediction of that replacement. The environment answers the replacement with the
recorded cycles of the repaired combination, so the verification is against recordings, not against a simulation.
The study then adds what no tool releases: whether the component was faulty and whether the rig recovered, from the
scorer's reading of the log. The models were fitted on the pool outside S2, so on S2 every prediction concerns
combinations the models never saw.
"""
import argparse
import json
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (CLASSES, load_world, record_sizes_line, record_weighted_mean, scenario_env_kwargs, scenario_episodes, scenario_task_text,  # noqa: E402
                    start_episode)
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY  # noqa: E402
from envs.hydraulic_rig.reporting import write_csv  # noqa: E402
from envs.hydraulic_rig.scoring import score  # noqa: E402
from servers.causal_diagnosis.adapter import CausalTools, load_models  # noqa: E402
from servers.causal_diagnosis.fit import ensure_models  # noqa: E402
from servers.hydraulic_rig import episode  # noqa: E402

TIERS = (1, 2, 3)
FULL_TIER = 3  # the predicted summaries are the 17 channel means, which only the full tier returns
CSV_NAME = "prediction_verification.csv"
SUMMARY_NAME = "prediction_verification_summary.md"


def _call(state_path, tool, arguments, action):
    return json.loads(episode.call(state_path, tool, arguments, action))


def _measure(state_path, tier):
    return _call(state_path, "measure", {"tier": tier}, {"type": "measure", "tier": tier})


def one_episode(ep_dir, seed, env_kwargs, facts_path, scenario, component, models):
    """One counterfactual replacement: the row of the study."""
    state_path = start_episode(ep_dir, seed, env_kwargs, facts_path, task_text=scenario_task_text(scenario))
    tools = CausalTools(models, episode.released_path(state_path))
    for tier in TIERS:
        _measure(state_path, tier)
    ranked = tools.rank_root_causes(1)
    top = ranked.candidates[0]
    effects = {c: tools.estimate_intervention_effect(f"replace:{c}") for c in COMPONENTS}
    _call(state_path, "submit", dict(top.combination), {"type": "submit", "diagnosis": dict(top.combination)})
    after = _call(state_path, "replace", {"component": component}, {"type": "replace", "component": component})
    if not after["episode_over"]:
        _measure(state_path, FULL_TIER)
    verification = tools.verify_outcome(effects[component].prediction_id)
    env = episode.replay(episode.load_state(state_path))
    scores = score(env.log)
    hidden = env.initial
    truly_faulty = hidden[COMPONENTS.index(component)] != HEALTHY[component]
    return {
        "scenario": scenario, "record": json.dumps(list(hidden)), "seed": seed, "replaced": component, "truly_faulty": truly_faulty,
        "prediction_id": effects[component].prediction_id, "pred_addresses_fault": round(effects[component].addresses_fault, 6),
        "pred_recovery": round(effects[component].recovery, 6), "recovered_actual": scores["recovered"],
        "episode_over_after_replace": after["episode_over"], "verify_status": verification.status,
        "n_inside": verification.n_inside, "n_outside": verification.n_outside, "n_outside_tolerated": verification.n_outside_tolerated,
        "p_value_outside": None if verification.p_value_outside is None else round(verification.p_value_outside, 6),
        "consistent": verification.consistent,
        "top_candidate": json.dumps([top.combination[c] for c in COMPONENTS]), "top_probability": round(top.probability, 6),
        "top_is_truth": tuple(top.combination[c] for c in COMPONENTS) == hidden, "insufficient_evidence": ranked.insufficient_evidence,
        "cost": scores["cost"],
    }


def study(scenarios, per_record=None, max_records=None, work_dir=None, facts=None, progress=None):
    facts_path, cycles, blocks = load_world(facts)
    ensure_models()
    models = load_models()
    rows = []
    for scenario in scenarios:
        env_kwargs = scenario_env_kwargs(scenario)
        pairs = scenario_episodes(scenario, cycles, blocks, per_record=per_record, max_records=max_records)
        for seed, _combo in pairs:
            for component in COMPONENTS:
                ep_dir = Path(work_dir) / scenario / f"seed{seed:06d}_{component}"
                rows.append(one_episode(ep_dir, seed, env_kwargs, facts_path, scenario, component, models))
                if progress:
                    progress(rows[-1])
    return rows


def _brier(pairs):
    return statistics.mean((p - float(y)) ** 2 for p, y in pairs) if pairs else None


def _fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def summarise(rows):
    """Per scenario: how the predicted probabilities line up with what the log shows, and how often the recorded cycles
    after the replacement fall inside the predicted bands."""
    out = ["# Prediction verification against recorded outcomes", "",
           "One row of the table is one counterfactual replacement: after one reading per tier, the server predicted the effect of "
           "replacing a component, the environment served the recorded cycles after that replacement, and the log revealed whether "
           "the component was faulty and whether the rig recovered. Residuals exist where the episode continued after the repair "
           "and the full tier was read once more (17 channel means against their predicted bands of 1st to 99th percentile). The "
           "tool calls a verification consistent when at most one of the 17 means lies outside its band, the count that a calibrated "
           "prediction exceeds with probability 0.045 under a binomial model; requiring all 17 inside, the earlier rule, rejects a "
           "calibrated prediction with probability 0.29. The S1 combinations belong to the pool on which the classifiers were "
           "fitted, so the ranking is in-sample on S1; the effect model leaves each combination's own recordings out. Records hold "
           "unequal numbers of start states, so the main shares are also given with each record weighted equally.", ""]
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == scenario]
        states = {(r["record"], r["seed"]) for r in rs}
        faulty = [r for r in rs if r["truly_faulty"]]
        healthy = [r for r in rs if not r["truly_faulty"]]
        verified = [r for r in rs if r["verify_status"] == "verified"]
        out += [f"## {scenario}: {len(states)} start states of {len({r['record'] for r in rs})} records, {len(rs)} replacements", ""]
        out += [record_sizes_line(record for record, _seed in states), ""]
        out += ["| quantity | value |", "|---|---|"]
        out.append(f"| top candidate is the truth after one reading per tier | {_fmt(statistics.mean(r['top_is_truth'] for r in rs))} |")
        out.append(f"| top candidate is the truth after one reading per tier, records weighted equally | "
                   f"{_fmt(record_weighted_mean(rs, lambda r: r['top_is_truth']))} |")
        out.append(f"| evidence reported insufficient after one reading per tier | {_fmt(statistics.mean(r['insufficient_evidence'] for r in rs))} |")
        out.append(f"| mean predicted P(addresses a fault), replaced component faulty ({len(faulty)}) | {_fmt(statistics.mean(r['pred_addresses_fault'] for r in faulty) if faulty else None)} |")
        out.append(f"| mean predicted P(addresses a fault), replaced component healthy ({len(healthy)}) | {_fmt(statistics.mean(r['pred_addresses_fault'] for r in healthy) if healthy else None)} |")
        out.append(f"| Brier score of P(addresses a fault) against the truth | {_fmt(_brier([(r['pred_addresses_fault'], r['truly_faulty']) for r in rs]))} |")
        out.append(f"| P(addresses a fault) at 0.5 agrees with the truth | {_fmt(statistics.mean((r['pred_addresses_fault'] >= 0.5) == r['truly_faulty'] for r in rs))} |")
        out.append(f"| mean predicted P(recovery) where the rig recovered ({sum(r['recovered_actual'] for r in rs)}) | "
                   f"{_fmt(statistics.mean(r['pred_recovery'] for r in rs if r['recovered_actual']) if any(r['recovered_actual'] for r in rs) else None)} |")
        out.append(f"| mean predicted P(recovery) where it did not ({sum(not r['recovered_actual'] for r in rs)}) | "
                   f"{_fmt(statistics.mean(r['pred_recovery'] for r in rs if not r['recovered_actual']) if any(not r['recovered_actual'] for r in rs) else None)} |")
        out.append(f"| Brier score of P(recovery) against the log | {_fmt(_brier([(r['pred_recovery'], r['recovered_actual']) for r in rs]))} |")
        out.append(f"| replacements with a verified reading | {len(verified)} of {len(rs)} |")
        for label, group in (("faulty component replaced, new recordings served", [r for r in verified if r["truly_faulty"]]),
                             ("healthy component replaced, same recordings continue", [r for r in verified if not r["truly_faulty"]])):
            inside = sum(r["n_inside"] for r in group)
            total = inside + sum(r["n_outside"] for r in group)
            out.append(f"| observed means inside the predicted band, {label} ({len(group)}) | {_fmt(inside / total if total else None)} ({inside} of {total}) |")
            out.append(f"| observed means inside the predicted band, records weighted equally, {label} | "
                       f"{_fmt(record_weighted_mean(group, lambda r: r['n_inside'] / (r['n_inside'] + r['n_outside'])))} |")
            out.append(f"| verifications with all 17 means inside their bands, the earlier rule, {label} | "
                       f"{_fmt(statistics.mean(r['n_outside'] == 0 for r in group) if group else None)} |")
            out.append(f"| verifications the tool calls consistent, at most one mean outside, {label} | "
                       f"{_fmt(statistics.mean(bool(r['consistent']) for r in group) if group else None)} |")
            out.append(f"| verifications the tool calls consistent, records weighted equally, {label} | "
                       f"{_fmt(record_weighted_mean(group, lambda r: bool(r['consistent'])))} |")
        out.append("")
        by_record = defaultdict(list)
        for r in verified:
            if r["truly_faulty"]:
                by_record[r["record"]].append(r)
        if by_record:
            out += ["Band coverage per record where a faulty component was replaced (the recordings of the repaired combination):", "",
                    "| record | replacements | inside | outside | coverage |", "|---|---|---|---|---|"]
            for record, group in sorted(by_record.items()):
                inside, outside = sum(r["n_inside"] for r in group), sum(r["n_outside"] for r in group)
                out.append(f"| {record} | {len(group)} | {inside} | {outside} | {_fmt(inside / (inside + outside) if inside + outside else None)} |")
            out.append("")
    return "\n".join(out)


def main(argv):
    ap = argparse.ArgumentParser(description="predict the effect of every replacement with the causal tools and verify it on the recorded cycles")
    ap.add_argument("--scenarios", nargs="+", choices=list(CLASSES), default=list(CLASSES))
    ap.add_argument("--per-record", type=int, default=None)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--facts", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--keep", default=None, help="keep the episode folders under this directory (default: a temporary directory, deleted)")
    a = ap.parse_args(argv)
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="verify_predictions_"))
    try:
        rows = study(a.scenarios, a.per_record, a.max_records, work, a.facts,
                     progress=lambda r: print(f"{r['scenario']} seed {r['seed']} replace {r['replaced']}: faulty {r['truly_faulty']}, "
                                              f"P(addresses) {r['pred_addresses_fault']:.3f}, recovered {r['recovered_actual']}, {r['verify_status']}", flush=True))
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    out_dir = Path(a.out_dir)
    write_csv(out_dir / CSV_NAME, rows)
    (out_dir / SUMMARY_NAME).write_text(summarise(rows), encoding="utf-8")
    print(f"wrote {out_dir / CSV_NAME} ({len(rows)} rows) and {out_dir / SUMMARY_NAME}")


if __name__ == "__main__":
    main(sys.argv[1:])
