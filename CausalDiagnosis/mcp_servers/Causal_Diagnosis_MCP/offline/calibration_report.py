"""The calibration test of the component marginals (CausalDiagnosis/docs/causal-diagnosis-mcp.md, section Calibration).

For every combination of the start set, the development blocks of the asset by default (asset_model.development_blocks)
or every recorded combination with --starts all, rollouts of readings drawn from its block feed each ranking model, and the fault
probabilities of E5 after one to several readings are scored by their Brier score and their calibration curve against the
true faulty set; the share of rollouts whose top combination equals the truth by number of readings is the symptom of
a misfit of the ranking model, and the coverage of the conformal intervals of E14 on the same cycles checks E14. A rollout
draws distinct cycles of the live segment of a drawn entry offset, as the environment serves them, under the static belief, a
uniform prior over every combination of the profile, scored by the fits of an episode of that combination, which leave its
block out of every fit, so the intervals of E14 of the true combination come from a fit that never read the
block either. The report also scores every cycle of the healthy block, per sensor group, by the verdict of E17 and the
intervals of E14 of an episode of each start, a check on one named combination, for which the calibration promises no rate.
The report is written as Markdown under CausalDiagnosis/results/, where
git ignores the calibration reports. Nothing from it enters code, tests or docs, and any revision of the model this
report prompts has to be declared as a decision taken after reading it.

usage: python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.calibration_report [--starts development|all] [--readings N] [--rollouts R]
       [--seed S] [--out DIR]
       python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.calibration_report --asset-id ID --models PATH --out DIR [options] reports on another
       registered asset from its own models file into a folder of its own.
"""
from CausalDiagnosis.utils.paths import RESULTS
import argparse
import datetime
from pathlib import Path

import numpy as np

from CausalDiagnosis.utils.atomic_writes_and_locks import replaced_atomically

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.diagnostics_model import EffectDiagnostics
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import asset_view, default_facts_path, development_blocks, recorded_blocks, require_same_data, served_segments
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import DEFAULT_ALPHA, predict_summaries
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.discriminant_urn_ranking import urn_probabilities
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Reading
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import MODELS_PATH, REFIT, facts_sha256, refuse_default_files
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import RANKING_MODELS, Posterior

RESULTS_DIR = RESULTS
READINGS, ROLLOUTS, SEED, BINS = 5, 20, 0, 10  # a rollout reads at most the recordings of one segment
START_SETS = ("development", "all")  # the starts the report scores: the development blocks, or every recorded combination


def rich_group(view):
    """The sensor group that reads every summary, the one each rollout reads."""
    return max(view.sensor_groups, key=lambda g: (len(view.group_fields(g)), g))


def probabilities(episode, ranking_model, readings):
    """The masses over the combinations after the readings under one ranking model and the fits of an episode, as a static
    episode serves them."""
    effect = episode.effect
    if ranking_model == "discriminant_urn":
        masses = urn_probabilities(episode.urn, effect.space, readings, [])
        return np.array([masses.get(h, 0.0) for h in effect.space.combinations])
    post = Posterior.uniform(effect, block=ranking_model == "joint_block")
    for r in readings:
        fields = tuple(f for f in effect.space.fields if f in r.values)
        post = post.updated(fields, np.array([r.values[f] for f in fields]))
    return post.probabilities


def starts_of(view, blocks, starts):
    """The combinations the report scores, in grid order: the development blocks of the asset, or every recorded one."""
    if starts not in START_SETS:
        raise ValueError(f"starts must be one of {', '.join(START_SETS)}")
    chosen = set(development_blocks(view, blocks)) if starts == "development" else set(blocks)
    return [h for h in view.combinations if h in blocks and h in chosen]


def score_healthy_block(episode, models, view, alpha, tallies):
    """Adds to tallies, per sensor group, the cycles of the healthy block that the verdict of E17 of the episode's fits calls
    inconsistent, the cycles scored, and the share of the group's summaries inside their intervals of E14 of the healthy
    combination, one share per cycle."""
    healthy, index = view.healthy_combination(), models.space.field_index
    model = EffectDiagnostics(episode, block=True, resampled=False)
    summaries = predict_summaries(episode, {healthy: 1.0}, alpha)
    prediction = {"alpha": alpha, "predicted": {s.field: [s.mean, s.low, s.high, s.quantile] for s in summaries}}
    for group in sorted(view.sensor_groups):
        tally = tallies.setdefault(group, {"inconsistent": 0, "scored": 0, "shares": []})
        fields = view.group_fields(group)
        for cycle in models.blocks[healthy]:
            checks, _combined, _p, consistent = model.outcome_check(prediction, {f: float(f"{cycle[index[f]]:.6g}") for f in fields})
            tally["inconsistent"] += int(not consistent)
            tally["scored"] += 1
            tally["shares"].append(float(np.mean([low <= observed <= high for _f, _m, observed, low, high, _s in checks])))


def evaluate(models, view, blocks, readings=READINGS, rollouts=ROLLOUTS, seed=SEED, ranking_models=RANKING_MODELS, alpha=DEFAULT_ALPHA,
             starts=None):
    """Records per (ranking model, number of readings): predicted fault probabilities with their truth, top combination
    hits; per number of readings the share of summaries inside the conformal interval of the true combination; and per
    sensor group {group: (inconsistent, scored, share inside)} of the healthy block under the fits of each episode
    (score_healthy_block). starts are the combinations scored, every recorded one when None."""
    space = models.space
    group = rich_group(view)
    fields = view.group_fields(group)
    rng = np.random.default_rng(seed)
    grid = space.combinations
    segments = served_segments()
    faulty = np.array([[h[j] != space.healthy[j] for j in range(len(space.components))] for h in grid], dtype=float)
    records = {(m, n): {"p": [], "y": [], "top": []} for m in ranking_models for n in range(1, readings + 1)}
    coverage = {n: [] for n in range(1, readings + 1)}
    tallies = {}
    for start in grid:
        if start not in blocks or (starts is not None and start not in starts):
            continue
        block = blocks[start]
        truth = np.array([start[j] != space.healthy[j] for j in range(len(space.components))], dtype=float)
        episode = models.without(start)
        score_healthy_block(episode, models, view, alpha, tallies)
        mu = episode.effect.mu(start)
        quantiles = np.array([episode.calibration.quantile(k, alpha)[0] for k in range(len(space.fields))])
        for _ in range(rollouts):
            served = block[segments[int(rng.integers(len(segments)))]]
            drawn = served[rng.permutation(len(served))[:readings]]
            reads = [Reading(n + 1, n + 1, group, {f: float(f"{cycle[space.field_index[f]]:.6g}") for f in fields}) for n, cycle in enumerate(drawn)]
            for n in range(1, len(drawn) + 1):
                inside = np.abs(drawn[n - 1] - mu) <= quantiles * episode.sigma
                coverage[n].append(float(inside.mean()))
                for m in ranking_models:
                    p = probabilities(episode, m, reads[:n])
                    records[(m, n)]["p"].extend(np.clip(p @ faulty, 0, 1).tolist())
                    records[(m, n)]["y"].extend(truth.tolist())
                    records[(m, n)]["top"].append(grid[int(np.argmax(p))] == tuple(start))
    healthy = {g: (t["inconsistent"], t["scored"], float(np.mean(t["shares"])) if t["shares"] else float("nan")) for g, t in tallies.items()}
    return records, coverage, healthy


def brier(record):
    p, y = np.array(record["p"]), np.array(record["y"])
    return float(((p - y) ** 2).mean()) if len(p) else float("nan")


def curve(record, bins=BINS):
    p, y = np.array(record["p"]), np.array(record["y"])
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high) if high < 1 else (p >= low) & (p <= high)
        rows.append((low, high, int(mask.sum()), float(p[mask].mean()) if mask.any() else None, float(y[mask].mean()) if mask.any() else None))
    return rows


def report(records, coverage, models, view, blocks, readings, rollouts, seed, ranking_models, alpha, starts=None, healthy=None):
    """The report as Markdown; starts names the start set and its combinations, healthy the counts of the healthy block."""
    label, scored = starts if starts is not None else ("recorded", [h for h in view.combinations if h in blocks])
    lines = [f"# Calibration report of the diagnostics server, {datetime.date.today().isoformat()}", "",
             "This file is generated under CausalDiagnosis/results/, where git ignores the calibration reports. Nothing in it enters code, tests or "
             "docs; a revision of the model prompted by it is a declared decision.", "",
             "## Protocol", "",
             f"- Combinations: the {len(scored)} combinations of the start set {label} ({models.fit['rule']}).",
             f"- Readings: up to {readings} distinct cycles per rollout, drawn from the live segment of a drawn entry offset, all at the sensor group "
             f"{rich_group(view)}; {rollouts} rollouts per combination, scored by the fits of an episode of that combination, which leave its block "
             f"out; seed {seed}; a uniform prior over every combination.",
             f"- Ranking models: {', '.join(ranking_models)}. Prices and durations: the default rules of the asset.",
             f"- Conformal coverage: the share of the summaries of each drawn cycle inside the interval of E14 at alpha = {alpha:g} "
             "for the true combination, which is the known combination case the interval is calibrated for.",
             f"- Effect model: the interaction pairs of every start's form, {len({tuple(map(tuple, s['pairs'])) for s in models.raw['selections'].values()})} distinct set(s).", "",
             "## Brier score of the component fault probabilities by number of readings", "",
             "| readings | " + " | ".join(ranking_models) + " |", "|---|" + "---|" * len(ranking_models)]
    for n in range(1, readings + 1):
        lines.append(f"| {n} | " + " | ".join(f"{brier(records[(m, n)]):.4f}" for m in ranking_models) + " |")
    lines += ["", "## Share of rollouts whose top combination equals the true one, by number of readings", "",
              "| readings | " + " | ".join(ranking_models) + " |", "|---|" + "---|" * len(ranking_models)]
    for n in range(1, readings + 1):
        lines.append(f"| {n} | " + " | ".join(f"{np.mean(records[(m, n)]['top']) if records[(m, n)]['top'] else float('nan'):.3f}" for m in ranking_models) + " |")
    lines += ["", f"## Conformal coverage of E14 at alpha = {alpha:g} on the recorded cycles", "", "| readings | mean share of summaries inside |", "|---|---|"]
    for n in range(1, readings + 1):
        lines.append(f"| {n} | {np.mean(coverage[n]) if coverage[n] else float('nan'):.4f} |")
    if healthy is not None:
        lines += ["", f"## The healthy block under the verdict of E17 and the intervals of E14 at alpha = {alpha:g}", "",
                  "Every cycle of the healthy block, scored per sensor group by the fits of an episode of each start of the set.", "",
                  "| sensor group | cycles found inconsistent | cycles scored | mean share of summaries inside |", "|---|---|---|---|"]
        for group, (inconsistent, count, share) in healthy.items():
            lines.append(f"| {group} | {inconsistent} | {count} | {share:.4f} |")
    for m in ranking_models:
        for n in (1, readings // 2 or 1, readings):
            lines += ["", f"## Calibration curve, {m}, {n} reading(s)", "", "| bin | count | mean predicted | observed frequency |", "|---|---|---|---|"]
            for low, high, count, predicted, observed in curve(records[(m, n)]):
                cell = lambda v: "" if v is None else f"{v:.3f}"
                lines.append(f"| {low:.1f} to {high:.1f} | {count} | {cell(predicted)} | {cell(observed)} |")
    return "\n".join(lines) + "\n"


def run(out_dir=RESULTS_DIR, readings=READINGS, rollouts=ROLLOUTS, seed=SEED, ranking_models=RANKING_MODELS, alpha=DEFAULT_ALPHA, facts_path=None,
        asset_id=None, models_path=MODELS_PATH, starts=START_SETS[0]):
    models = load_models(models_path)
    view = asset_view(asset_id)
    path = facts_path or default_facts_path(asset_id)
    require_same_data("The models file of this report", models.raw["facts_sha256"], facts_sha256(path), REFIT)
    blocks, _records = recorded_blocks(view, path)
    chosen = starts_of(view, blocks, starts)
    records, coverage, healthy = evaluate(models, view, blocks, readings, rollouts, seed, ranking_models, alpha, starts=set(chosen))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration_{datetime.date.today().isoformat()}.md"
    text = report(records, coverage, models, view, blocks, readings, rollouts, seed, ranking_models, alpha, starts=(starts, chosen),
                  healthy=healthy)
    with replaced_atomically(path) as f:
        f.write(text)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="calibration report of the diagnostics server's ranking models on the recorded combinations")
    parser.add_argument("--asset-id", help="a registered asset profile, the default asset when omitted")
    parser.add_argument("--models", type=Path, default=MODELS_PATH, help="the models file of the asset")
    parser.add_argument("--starts", choices=START_SETS, default=START_SETS[0],
                        help="the combinations scored: the development blocks, or every recorded combination")
    parser.add_argument("--readings", type=int, default=READINGS)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    refuse_default_files(parser, args, models=MODELS_PATH, out=RESULTS_DIR)
    path = run(args.out, args.readings, args.rollouts, args.seed, asset_id=args.asset_id, models_path=args.models, starts=args.starts)
    print(f"wrote {path} ({args.starts} starts, {args.readings} readings, {args.rollouts} rollouts per combination, seed {args.seed}); nothing "
          f"from it enters code, tests or docs")


if __name__ == "__main__":
    main()
