"""Section 5.6 of design v2: the calibration test of the component marginals (docs/hydraulic_rig/causal-diagnosis-server.md).

For every evaluation combination, rollouts of readings drawn from its recorded block feed each ranking model, and the fault
probabilities of E5 after one to several readings are scored by their Brier score and their calibration curve against the
true faulty set; the share of rollouts whose top candidate equals the truth by number of readings is the symptom of the
misfit of 22 October, and the coverage of the conformal intervals of E14 on the same cycles checks E14. The report is
written as Markdown under benchmarks/hydraulic_rig/results/, which git ignores. Nothing from it enters code, tests or docs:
the evaluation combinations are never in the fitting pool, so any revision of the model this report prompts has to be
declared as a decision taken after reading it.

usage: HYD_FACTS_PATH=<facts file> python -m servers.causal_diagnosis.calibration [--readings N] [--rollouts R] [--seed S] [--out DIR]
"""
import argparse
import datetime
from pathlib import Path

import numpy as np

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles

from .adapter import load_models
from .conformal import DEFAULT_ALPHA
from .discriminant_urn import urn_probabilities
from .effect_model import FIELD_INDEX, GROUP_FIELDS, PoolData, SUMMARY_FIELDS, parse_combo
from .posterior import RANKING_MODELS, Posterior, Reading
from .tests.fixture_episode import PLACEHOLDER_RULES

RESULTS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "hydraulic_rig" / "results"
READINGS, ROLLOUTS, SEED, BINS = 10, 20, 0, 10


def probabilities(models, ranking_model, rules, readings):
    if ranking_model == "discriminant_urn":
        return np.array(urn_probabilities(models.urn, rules, readings, [])), None
    post = Posterior(models.effect, rules, readings, [], time=len(readings), block_variance=ranking_model == "joint_block")
    return post.probabilities, post


def evaluate(models, rules, cycles, blocks, readings=READINGS, rollouts=ROLLOUTS, seed=SEED, ranking_models=RANKING_MODELS, alpha=DEFAULT_ALPHA):
    """Records per (ranking model, number of readings): predicted fault probabilities with their truth, top candidate hits;
    and per number of readings the share of summaries inside the conformal interval of the true combination."""
    evaluation = [parse_combo(key) for key in models.pool["evaluation"]]
    data = PoolData(cycles, blocks, evaluation)
    rng = np.random.default_rng(seed)
    starts = Posterior(models.effect, rules, [], [], time=0).candidates
    records = {(m, n): {"p": [], "y": [], "top": []} for m in ranking_models for n in range(1, readings + 1)}
    coverage = {n: [] for n in range(1, readings + 1)}
    quantiles = np.array([models.calibration.quantile(k, alpha)[0] for k in range(len(SUMMARY_FIELDS))])
    for start, block in zip(data.pool, data.arrays):
        truth = np.array([start[j] != HEALTHY[c] for j, c in enumerate(COMPONENTS)], dtype=float)
        mu = models.effect.mu(start)
        faulty = np.array([[h[j] != HEALTHY[c] for j, c in enumerate(COMPONENTS)] for h in starts], dtype=float)
        for _ in range(rollouts):
            drawn = [block[rng.integers(len(block))] for _ in range(readings)]
            reads = [Reading(n + 1, n + 1, "tier3", {f: float(f"{cycle[FIELD_INDEX[f]]:.6g}") for f in GROUP_FIELDS["tier3"]}) for n, cycle in enumerate(drawn)]
            for n in range(1, readings + 1):
                inside = np.abs(drawn[n - 1] - mu) <= quantiles * models.sigma
                coverage[n].append(float(inside.mean()))
                for m in ranking_models:
                    p, _post = probabilities(models, m, rules, reads[:n])
                    marginals = np.clip(p @ faulty, 0, 1)
                    records[(m, n)]["p"].extend(marginals.tolist())
                    records[(m, n)]["y"].extend(truth.tolist())
                    records[(m, n)]["top"].append(starts[int(np.argmax(p))] == tuple(start))
    return records, coverage


def brier(record):
    p, y = np.array(record["p"]), np.array(record["y"])
    return float(((p - y) ** 2).mean())


def curve(record, bins=BINS):
    p, y = np.array(record["p"]), np.array(record["y"])
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high) if high < 1 else (p >= low) & (p <= high)
        rows.append((low, high, int(mask.sum()), float(p[mask].mean()) if mask.any() else None, float(y[mask].mean()) if mask.any() else None))
    return rows


def report(records, coverage, models, rules, readings, rollouts, seed, ranking_models, alpha):
    lines = [f"# Calibration report of the diagnostics server, {datetime.date.today().isoformat()}", "",
             "Section 5.6 of design v2. This file is generated under benchmarks/hydraulic_rig/results/, which git ignores. Nothing in it "
             "enters code, tests or docs; a revision of the model prompted by it is a declared decision.", "",
             "## Protocol", "",
             f"- Evaluation combinations: {len(models.pool['evaluation'])} starts of the default fault rule, none of them in the fitting pool "
             f"({len(models.pool['combinations'])} combinations).",
             f"- Readings: {readings} per rollout, drawn with replacement from the start's recorded block, all at the full sensor group; "
             f"{rollouts} rollouts per start; seed {seed}.",
             f"- Ranking models: {', '.join(ranking_models)}. Prices and durations: the placeholder table of the build plan.",
             f"- Conformal coverage: the share of the 34 summaries of each drawn cycle inside the interval of E14 at alpha = {alpha:g} "
             "for the true combination, which is the known combination case the interval is calibrated for.",
             f"- Effect model: {len(models.effect.pairs)} interaction pair(s) selected; {len(models.effect.resampled)} resampled fits.", "",
             "## Brier score of the component fault probabilities by number of readings", "",
             "| readings | " + " | ".join(ranking_models) + " |", "|---|" + "---|" * len(ranking_models)]
    for n in range(1, readings + 1):
        lines.append(f"| {n} | " + " | ".join(f"{brier(records[(m, n)]):.4f}" for m in ranking_models) + " |")
    lines += ["", "## Share of rollouts whose top candidate equals the true start, by number of readings", "",
              "| readings | " + " | ".join(ranking_models) + " |", "|---|" + "---|" * len(ranking_models)]
    for n in range(1, readings + 1):
        lines.append(f"| {n} | " + " | ".join(f"{np.mean(records[(m, n)]['top']):.3f}" for m in ranking_models) + " |")
    lines += ["", f"## Conformal coverage of E14 at alpha = {alpha:g} on the evaluation cycles", "", "| readings | mean share of summaries inside |", "|---|---|"]
    for n in range(1, readings + 1):
        lines.append(f"| {n} | {np.mean(coverage[n]):.4f} |")
    for m in ranking_models:
        for n in (1, readings // 2 or 1, readings):
            lines += ["", f"## Calibration curve, {m}, {n} reading(s)", "", "| bin | count | mean predicted | observed frequency |", "|---|---|---|---|"]
            for low, high, count, predicted, observed in curve(records[(m, n)]):
                cell = lambda v: "" if v is None else f"{v:.3f}"
                lines.append(f"| {low:.1f} to {high:.1f} | {count} | {cell(predicted)} | {cell(observed)} |")
    return "\n".join(lines) + "\n"


def run(out_dir=RESULTS_DIR, readings=READINGS, rollouts=ROLLOUTS, seed=SEED, ranking_models=RANKING_MODELS, alpha=DEFAULT_ALPHA, facts_path=None):
    models = load_models()
    cycles = load_cycles(facts_path or default_facts_path())
    blocks = build_blocks(cycles)
    rules = {**PLACEHOLDER_RULES, "sensor_groups": {g: list(f) for g, f in GROUP_FIELDS.items()}}
    records, coverage = evaluate(models, rules, cycles, blocks, readings, rollouts, seed, ranking_models, alpha)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration_{datetime.date.today().isoformat()}.md"
    path.write_text(report(records, coverage, models, rules, readings, rollouts, seed, ranking_models, alpha), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="calibration report of the diagnostics server's ranking models on the evaluation combinations")
    parser.add_argument("--readings", type=int, default=READINGS)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    path = run(args.out, args.readings, args.rollouts, args.seed)
    print(f"wrote {path} ({args.readings} readings, {args.rollouts} rollouts per start, seed {args.seed}); nothing from it enters code, tests or docs")


if __name__ == "__main__":
    main()
