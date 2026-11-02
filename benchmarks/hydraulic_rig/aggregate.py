"""One table per ablation A1 to A12 of design v2 (docs/hydraulic_rig/benchmarks.md) from the run folders of
run_episodes.py, as CSV and Markdown.

usage: python benchmarks/hydraulic_rig/aggregate.py [--runs benchmarks/hydraulic_rig/runs] [--out benchmarks/hydraulic_rig/results/aggregate]
       [--readme benchmarks/hydraulic_rig/results/README.md] [--published benchmarks/hydraulic_rig/results/runs] [--cluster "Columbia Insomnia"]
       [--resamples 10000] [--alpha 0.05] [--eqa-gold-direction "no change"] [--eqa-gold-action "open a work order"]

Every table is computed only from the run folders that exist; without any folder the Markdown states the design of each
table (what varies, the hypothesis, the metric, the reading) with the header row and empty cells, and every CSV holds its
header alone. Beside the tables the script writes the results README (--readme) for the published results folder: the
models, the cluster, the dates and the presets of the runs, every table with a one sentence reading rule, one example
episode with its tool calls and verdict, and the commands that reproduce the runs; before any run exists it says so. Differences between levels of a factor are paired by episode (the same seed, scenario and every other
factor) with a bootstrap interval clustered by recorded block, and the p values of a table's primary metric are
corrected by the Holm step down (stats.py).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import DEFAULT_RUNS, as_bool, as_float, episode_dirs, read_json, read_rows, run_dirs, write_rows  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS  # noqa: E402
from stats import cluster_bootstrap, cohens_kappa, holm, paired_cluster_bootstrap, spearman, wilson  # noqa: E402

PROPORTIONS = frozenset({"pass", "success", "separable", "commitment_on_inseparable", "diagnosis_all_correct", "recovered", "abstained",
                         "abstention_correct", "judge_pass", "provenance_ok", "static_faulty_set_correct", "static_repairs_match_oracle"})
FACTORS = ("config", "agent", "model", "clock_mode", "guidance_level", "diagnostics_server", "prices", "verifier_scope", "ranking_model")
IDENTITY = ("seed", "scenario_id")
BINS = 10  # calibration bins of A12 over the posterior of the top candidate


class Table:
    def __init__(self, id, varies, hypothesis, metric, reading, columns, build, rule=""):
        self.id, self.varies, self.hypothesis, self.metric, self.reading, self.columns, self.build = id, varies, hypothesis, metric, reading, columns, build
        self.rule = rule  # the one sentence reading rule of the results README


# ----- episodes -----

def load_episodes(runs_root):
    """Every row of every scores.csv under the root, with the run's name and folder and the per episode folder beside
    it, so a builder can open the hidden files of an episode when a table needs more than scores.csv holds."""
    episodes = []
    for run_dir in run_dirs(runs_root):
        meta = read_json(run_dir / "run.json", {})
        folders = {p.name: p for p in episode_dirs(run_dir)}
        for row in read_rows(run_dir):
            episode = dict(row)
            episode["run"] = row.get("run") or run_dir.name
            episode["run_dir"] = run_dir
            episode["episode_dir"] = folders.get(row.get("episode", ""), None) or _episode_folder(run_dir, row)
            episode["mode"] = meta.get("mode", "gym")
            episode["abstention_rule"] = meta.get("abstention_rule", "decision")
            episodes.append(episode)
    return episodes


def _episode_folder(run_dir, row):
    for p in episode_dirs(run_dir):
        if p.name.endswith(f"seed{int(row['seed']):06d}") and (not row.get("scenario_id") or f"scenario{row['scenario_id']}_" in p.name):
            return p
    return None


def value(episode, metric):
    """The metric of an episode as a number: a flag as 1 or 0, an empty cell as None."""
    cell = episode.get(metric, "")
    if metric in PROPORTIONS:
        flag = as_bool(cell)
        return None if flag is None else float(flag)
    return as_float(cell)


def with_metric(episodes, metric):
    return [e for e in episodes if value(e, metric) is not None]


def groups(episodes, *fields):
    """{(field values): [episodes]} in sorted key order."""
    out = {}
    for e in episodes:
        out.setdefault(tuple(e.get(f, "") for f in fields), []).append(e)
    return dict(sorted(out.items(), key=lambda kv: tuple(str(k) for k in kv[0])))


def summary(episodes, metric, ctx):
    """Mean of a metric with its interval: Wilson for a proportion, a cluster bootstrap of the mean otherwise."""
    eps = with_metric(episodes, metric)
    if not eps:
        return {metric: "", f"{metric}_low": "", f"{metric}_high": "", f"{metric}_n": 0}
    values = [value(e, metric) for e in eps]
    if metric in PROPORTIONS:
        w = wilson(int(sum(values)), len(values))
        return {metric: _r(w["proportion"]), f"{metric}_low": _r(w["low"]), f"{metric}_high": _r(w["high"]), f"{metric}_n": len(values)}
    boot = paired_cluster_bootstrap(values, [0.0] * len(values), [e.get("block", "") for e in eps], ctx["resamples"], ctx["seed"])
    return {metric: _r(boot["difference"]), f"{metric}_low": _r(boot["low"]), f"{metric}_high": _r(boot["high"]), f"{metric}_n": len(values)}


def paired(level_eps, reference_eps, factor, metric, ctx):
    """The paired difference of a metric between a level and the reference level of a factor: episodes match on their
    identity and on every other factor; clusters are the recorded blocks of the start."""
    key_fields = [f for f in (*IDENTITY, *FACTORS) if f != factor]

    def key(e):
        return tuple(e.get(f, "") for f in key_fields)

    reference = {key(e): e for e in with_metric(reference_eps, metric)}
    matched = [(e, reference[key(e)]) for e in with_metric(level_eps, metric) if key(e) in reference]
    if not matched:
        return {f"{metric}_diff": "", f"{metric}_diff_low": "", f"{metric}_diff_high": "", f"{metric}_p": "", "pairs": 0, "clusters": 0}
    boot = paired_cluster_bootstrap([value(a, metric) for a, _ in matched], [value(b, metric) for _, b in matched],
                                    [a.get("block", "") for a, _ in matched], ctx["resamples"], ctx["seed"])
    return {f"{metric}_diff": _r(boot["difference"]), f"{metric}_diff_low": _r(boot["low"]), f"{metric}_diff_high": _r(boot["high"]),
            f"{metric}_p": _r(boot["p_value"], 6), "pairs": boot["episodes"], "clusters": boot["clusters"]}


def _r(x, digits=4):
    return "" if x is None else round(float(x), digits)


def add_holm(rows, metric, alpha):
    """Holm over the p values of the primary metric of one table; rows without a p value stay unmarked."""
    indexed = [(i, as_float(r.get(f"{metric}_p"))) for i, r in enumerate(rows)]
    tested = [(i, p) for i, p in indexed if p is not None]
    corrected = holm([p for _, p in tested], alpha) if tested else {"adjusted": [], "reject": []}
    for (i, _), adjusted, reject in zip(tested, corrected["adjusted"], corrected["reject"]):
        rows[i]["holm_p"] = _r(adjusted, 6)
        rows[i]["reject"] = reject
    return rows


# ----- factor tables (A4, A5, A6, A9, A10, A11) -----

def factor_table(factor, reference, metrics, paired_metrics, level_of=None):
    """Rows per agent, model and level of the factor with the mean of every metric, and the paired difference to the
    reference level for the paired metrics; the first paired metric is the table's primary test."""

    def build(episodes, ctx):
        rows = []
        for (agent, model), eps in groups(episodes, "agent", "model").items():
            levels = {}
            for e in eps:
                levels.setdefault(level_of(e) if level_of else e.get(factor, ""), []).append(e)
            for level, level_eps in sorted(levels.items(), key=lambda kv: str(kv[0])):
                row = {"agent": agent, "model": model, factor: level, "n": len(level_eps)}
                for m in metrics:
                    row.update(summary(level_eps, m, ctx))
                for m in paired_metrics:
                    if str(level) != str(reference) and reference in levels:
                        row.update(paired(level_eps, levels[reference], factor, m, ctx))
                    else:
                        row.update({f"{m}_diff": "", f"{m}_diff_low": "", f"{m}_diff_high": "", f"{m}_p": "", "pairs": "", "clusters": ""})
                rows.append(row)
        return add_holm(rows, paired_metrics[0], ctx["alpha"]) if paired_metrics else rows

    columns = ["agent", "model", factor, "n"]
    for m in metrics:
        columns += [m, f"{m}_low", f"{m}_high", f"{m}_n"]
    for m in paired_metrics:
        columns += [f"{m}_diff", f"{m}_diff_low", f"{m}_diff_high", f"{m}_p"]
    if paired_metrics:
        columns += ["pairs", "clusters", "holm_p", "reject"]
    return columns, build


def clock_level(e):
    """A6: the clock mode, with the suffix zero when the configuration charges no duration at all."""
    config = read_json(Path(e["run_dir"]) / "run.json", {}).get("config", {}) if e.get("run_dir") else {}
    durations = config.get("durations", {})
    flat = [durations.get("read_per_cycle"), durations.get("advance_per_cycle"),
            *durations.get("lead_time", {}).values(), *durations.get("repair_downtime", {}).values()]
    zero = flat and all(v == 0 for v in flat)
    return f"{e.get('clock_mode', '')}{':zero' if zero else ''}"


# ----- A1, A2, A3, A7, A8, A12 -----

def table_a1(episodes, ctx):
    rows = []
    for (agent, model), eps in groups(episodes, "agent", "model").items():
        judged = [e for e in eps if value(e, "judge_pass") is not None and value(e, "pass") is not None]
        if not judged:
            continue
        j = [bool(value(e, "judge_pass")) for e in judged]
        p = [bool(value(e, "pass")) for e in judged]
        both = sum(1 for a, b in zip(j, p) if a and b)
        judge_only = sum(1 for a, b in zip(j, p) if a and not b)
        recorded_only = sum(1 for a, b in zip(j, p) if b and not a)
        neither = len(judged) - both - judge_only - recorded_only
        false_pass = wilson(judge_only, judge_only + neither)
        kappa = cluster_bootstrap(lambda pairs: cohens_kappa([a for a, _ in pairs], [b for _, b in pairs]), list(zip(j, p)),
                                  [e.get("block", "") for e in judged], ctx["resamples"], ctx["seed"])
        rows.append({"agent": agent, "model": model, "n": len(judged), "both_pass": both, "judge_pass_only": judge_only,
                     "recorded_pass_only": recorded_only, "neither": neither, "false_pass_rate": _r(false_pass["proportion"]),
                     "false_pass_low": _r(false_pass["low"]), "false_pass_high": _r(false_pass["high"]), "kappa": _r(kappa["estimate"]),
                     "kappa_low": _r(kappa["low"]), "kappa_high": _r(kappa["high"])})
    return rows


def table_a2(episodes, ctx):
    """The released hydraulic gold of IndustryAssetEQA gives one answer to every counterfactual question and one to
    every action question (the lessons report, section 5, items 26 and 27); both are parameters here. The recorded side
    comes from the hidden log: a work order on a faulty component improves the asset, one on a healthy component changes
    nothing, and the oracle's work order set says whether a work order was needed."""
    seen, orders, needed = set(), [], []
    for e in episodes:
        folder = e.get("episode_dir")
        if folder is None or (e["run"], folder.name) in seen:
            continue
        seen.add((e["run"], folder.name))
        log = read_json(Path(folder) / "hidden" / "log.json", [])
        oracle = read_json(Path(folder) / "hidden" / "oracle.json", {})
        orders += ["no change" if ev.get("unnecessary") else "improved" for ev in log if ev.get("event") == "work_order_created"]
        if oracle:
            needed.append("open a work order" if oracle.get("work_orders") else "no work order")
    rows = []
    for question, gold, recorded in (("counterfactual", ctx["eqa_gold_direction"], orders), ("action", ctx["eqa_gold_action"], needed)):
        agree = sum(1 for r in recorded if r == gold)
        w = wilson(agree, len(recorded))
        rows.append({"question_type": question, "model_gold": gold, "n": len(recorded), "agreement": _r(w["proportion"]),
                     "agreement_low": _r(w["low"]), "agreement_high": _r(w["high"])})
    return rows if seen else []


def table_a3(episodes, ctx):
    rows = []
    static = groups([e for e in episodes if e.get("mode") == "static"], "model")
    gym = groups([e for e in episodes if e.get("mode") != "static" and e.get("mode") != "gepa" and e.get("model")], "model")
    for (model,) in sorted(set(static) | set(gym)):
        row = {"model": model, "static_n": len(static.get((model,), [])), "gym_n": len(gym.get((model,), []))}
        for metric, prefix in (("judge_pass", "static_judge_pass"), ("static_faulty_set_correct", "static_faulty_set_correct")):
            s = summary(static.get((model,), []), metric, ctx)
            row.update({prefix: s[metric], f"{prefix}_low": s[f"{metric}_low"], f"{prefix}_high": s[f"{metric}_high"]})
        recorded = summary(gym.get((model,), []), "pass", ctx)
        row.update({"gym_recorded_pass": recorded["pass"], "gym_recorded_pass_low": recorded["pass_low"], "gym_recorded_pass_high": recorded["pass_high"]})
        row.update({"gym_cost": summary(gym.get((model,), []), "cost", ctx)["cost"]})
        rows.append(row)
    return rows


def table_gepa(episodes, ctx):
    """GEPA against the untuned planner prompt: per GEPA run (run.json mode gepa, gepa_result.json beside it) the mean
    validation reward of the seed candidate and of the best candidate, the episodes the optimization played (validation
    rollouts counted), and the untuned prompt's mean reward over the same number of episodes from the plain plan_execute
    runs of the same model and config, when that many exist."""
    rows = []
    seen = set()
    for e in episodes:
        run_dir = e.get("run_dir")
        if e.get("mode") != "gepa" or run_dir in seen:
            continue
        seen.add(run_dir)
        result = read_json(Path(run_dir) / "gepa_result.json", {})
        meta = read_json(Path(run_dir) / "run.json", {})
        played = result.get("episodes_played")
        plain = [p for p in episodes if p.get("mode") == "gym" and p.get("model") == meta.get("model") and p.get("config") == meta.get("config_name")
                 and p.get("agent", "").startswith("plan_execute")]
        plain = sorted(plain, key=lambda p: (p["run"], as_float(p.get("seed")) or 0))[:played or 0]
        untuned = summary(plain, "reward", ctx)
        for candidate in ("seed", "best"):
            scores = result.get(f"{candidate}_val_scores") or []
            mean = sum(scores) / len(scores) if scores else None
            rows.append({"run": Path(run_dir).name, "model": meta.get("model", ""), "config": meta.get("config_name", ""), "candidate": candidate,
                         "val_seeds": len(scores), "val_mean_reward": _r(mean), "episodes_played": played,
                         "untuned_mean_reward_same_episodes": untuned["reward"] if candidate == "best" else "",
                         "untuned_episodes_available": len(plain) if candidate == "best" else ""})
    return rows


def table_a7(episodes, ctx):
    rows, by_scope = [], {"writes": {}, "all": {}}
    for (agent, model), eps in groups(episodes, "agent", "model").items():
        row = {"agent": agent, "model": model}
        for scope in ("writes", "all"):
            s = summary([e for e in eps if e.get("verifier_scope") == scope], "pass", ctx)
            row.update({f"pass_{scope}": s["pass"], f"pass_{scope}_low": s["pass_low"], f"pass_{scope}_high": s["pass_high"], f"n_{scope}": s["pass_n"]})
            if s["pass"] != "":
                by_scope[scope][(agent, model)] = s["pass"]
        rows.append(row)
    shared = sorted(set(by_scope["writes"]) & set(by_scope["all"]))
    if rows:
        rho = spearman([by_scope["writes"][k] for k in shared], [by_scope["all"][k] for k in shared]) if len(shared) > 1 else None
        rows.append({"agent": "all agents", "model": "", "rank_correlation": _r(rho), "agents_in_both_scopes": len(shared)})
    return rows


def table_a8(episodes, ctx):
    rows = []
    for (agent, model, rule), eps in groups(episodes, "agent", "model", "abstention_rule").items():
        scored = [e for e in eps if value(e, "abstained") is not None]
        if not scored:
            continue
        abstained = [e for e in scored if value(e, "abstained") == 1]
        correct = sum(1 for e in abstained if value(e, "abstention_correct") == 1)
        inseparable = [e for e in scored if e.get("grade", "") == ""]
        precision, recall = wilson(correct, len(abstained)), wilson(correct, len(inseparable))
        rows.append({"agent": agent, "model": model, "abstention_rule": rule, "n": len(scored), "abstentions": len(abstained),
                     "inseparable_starts": len(inseparable), "precision": _r(precision["proportion"]), "precision_low": _r(precision["low"]),
                     "precision_high": _r(precision["high"]), "recall": _r(recall["proportion"]), "recall_low": _r(recall["low"]),
                     "recall_high": _r(recall["high"])})
    return rows


def rankings(episode):
    """(readings so far, top candidate as a list, its posterior) of every rank_root_causes call of an episode, read from
    the trajectory the runner wrote; the policies write no trajectory and contribute nothing."""
    folder = episode.get("episode_dir")
    if folder is None:
        return []
    out, readings = [], 0
    for step in read_json(Path(folder) / "trajectory.json", []):
        if step.get("error"):
            continue
        if step.get("tool") == "read_sensors":
            readings += 1
        if step.get("tool") != "rank_root_causes":
            continue
        try:
            answer = json.loads(step.get("response", ""))
            top = answer["candidates"][0]
            out.append((readings, list(top["combination"]), float(top["probability"])))
        except (ValueError, KeyError, IndexError, TypeError):
            continue
    return out


def _brier(points):
    return sum((p - hit) ** 2 for _, _, hit, p in points) / len(points) if points else None


def table_a12(episodes, ctx):
    """One row per agent, model, ranking model and number of readings (and one over all readings): the Brier score of
    the top candidate's posterior with a bootstrap interval clustered by the episode's block, and the top candidate rate
    with a Wilson interval. The calibration bins go to A12_curves.csv."""
    rows = []
    ctx["a12_curves"] = []
    for (agent, model, ranking), eps in groups(episodes, "agent", "model", "ranking_model").items():
        points = []  # (block, readings, hit, posterior)
        for e in eps:
            oracle = read_json(Path(e["episode_dir"]) / "hidden" / "oracle.json", {}) if e.get("episode_dir") else {}
            truth = [oracle.get("diagnosis", {}).get(c) for c in COMPONENTS]
            points += [(e.get("block", ""), readings, float(top == truth), p) for readings, top, p in rankings(e)]
        if not points:
            continue
        for readings in [*sorted({r for _, r, _, _ in points}), "all"]:
            subset = [pt for pt in points if readings == "all" or pt[1] == readings]
            brier = cluster_bootstrap(_brier, subset, [pt[0] for pt in subset], ctx["resamples"], ctx["seed"])
            top1 = wilson(int(sum(hit for _, _, hit, _ in subset)), len(subset))
            rows.append({"agent": agent, "model": model, "ranking_model": ranking, "readings": readings, "calls": len(subset),
                         "brier": _r(brier["estimate"]), "brier_low": _r(brier["low"]), "brier_high": _r(brier["high"]),
                         "top_candidate_correct": _r(top1["proportion"]), "top1_low": _r(top1["low"]), "top1_high": _r(top1["high"])})
            for b in range(BINS):
                lo, hi = b / BINS, (b + 1) / BINS
                inside = [(hit, p) for _, _, hit, p in subset if lo <= p < hi or (b == BINS - 1 and p == 1.0)]
                if inside:
                    ctx["a12_curves"].append({"agent": agent, "model": model, "ranking_model": ranking, "readings": readings,
                                              "bin_lo": lo, "bin_hi": hi, "predicted_mean": _r(sum(p for _, p in inside) / len(inside)),
                                              "observed_frequency": _r(sum(h for h, _ in inside) / len(inside)), "n": len(inside)})
    return rows


# ----- the twelve tables -----

def tables():
    a4 = factor_table("diagnostics_server", "False", ["pass", "diagnosis_all_correct", "cost"], ["pass", "cost"])
    a5 = factor_table("config", "static", ["pass", "cost", "cost_normalized", "detection_delay"], ["pass", "cost_normalized"])
    a6 = factor_table("clock_mode", "declared", ["pass", "detection_delay", "repair_delay"], ["pass"], level_of=clock_level)
    a9 = factor_table("prices", "placeholder", ["cost", "cost_normalized", "pass"], ["cost_normalized", "pass"])
    a10 = factor_table("guidance_level", "1", ["pass"], ["pass"])
    a11 = factor_table("n_faulty", "1", ["pass", "unnecessary"], [])
    return [
        Table("A1", "the scoring head: the AssetOpsBench judge on the closing answer against the recorded pass, on the same trajectories",
              "the judge passes episodes whose recordings show no recovery or an unnecessary replacement",
              "the two by two table of judge pass against recorded pass, the false pass rate (the share of episodes without a recorded pass that the judge passes) with a Wilson interval, Cohen's kappa",
              "the size of the gap a gym closes over answer scoring",
              ["agent", "model", "n", "both_pass", "judge_pass_only", "recorded_pass_only", "neither", "false_pass_rate", "false_pass_low", "false_pass_high", "kappa", "kappa_low", "kappa_high"], table_a1,
              rule="Read kappa against 1 for agreement between the judge and the recordings, and the false pass rate as the share of recorded failures the judge still passes."),

        Table("A2", "the source of the gold for the counterfactual and action questions of IndustryAssetEQA on this rig: its risk model against the recordings",
              "the risk model's answers, one for every counterfactual question and one for every action question on this asset, disagree with the recorded consequence",
              "agreement of the model gold with the recorded direction of every agent work order (improved when the component was faulty, no change when it was healthy) and with the recorded need for a work order (the oracle's work order set), with Wilson intervals; the gold answers are the parameters --eqa-gold-direction and --eqa-gold-action",
              "whether a learned risk model can stand in for the machine on this asset",
              ["question_type", "model_gold", "n", "agreement", "agreement_low", "agreement_high"], table_a2,
              rule="Read the agreement as the share of recorded consequences that match the one answer the released gold gives every question of that type."),

        Table("A3", "the mode of answering the same question: statically from the question, the facts of the cycle and the failure modes in one model call (runs of --agent static, mode static in run.json), or executed in the gym with tools and recordings",
              "static answers are graded correct more often than their recorded consequence warrants",
              "the judge pass rate of the static answers and the share whose parsed faulty set equals the recorded start, against the recorded pass rate and the cost of the gym episodes, per model, with Wilson intervals",
              "the first evidence for why a gym is needed, on the questions the mentor's group already poses",
              ["model", "static_n", "static_judge_pass", "static_judge_pass_low", "static_judge_pass_high", "static_faulty_set_correct", "static_faulty_set_correct_low", "static_faulty_set_correct_high", "gym_n", "gym_recorded_pass", "gym_recorded_pass_low", "gym_recorded_pass_high", "gym_cost"], table_a3,
              rule="Read the static judge pass rate beside the share of static answers whose faulty set matches the recorded start, against the recorded pass rate of the gym episodes."),

        Table("A4", "the diagnostics server on or off (diagnostics_server of run.json)",
              "the server raises the right component repair rate and lowers the cost to a correct diagnosis",
              "pass, diagnosis accuracy and cost per level, the paired difference of pass and cost against the server off, Holm over the pass differences",
              "the value of the helper tools, candidate A of 26 October", *a4,
              rule="Read the pass difference of the server on against off with its interval; an interval above zero means the helper tools raise the pass rate."),

        Table("A5", "static against timed episodes with the same hidden start (config static against timed)",
              "timed episodes lower pass and raise cost through downtime, and detection delay separates agents that static episodes rank alike",
              "pass, cost, normalized cost and detection delay per config, the paired difference of pass and normalized cost against static, Holm over the pass differences",
              "whether the dynamics the mentor asked for change conclusions", *a5,
              rule="Read the timed row against the static row: a lower pass or a higher cost under timed shows what the dynamics cost the agent."),

        Table("A6", "the clock mode: declared durations, declared durations set to zero (level declared:zero), instant, generation time",
              "charging time changes which agents pass, as the instant mode changed the Time split of Gaia2",
              "pass, detection delay and repair delay per mode, the paired difference of pass against declared, Holm over the pass differences",
              "what the time cost measures beyond the price", *a6,
              rule="Read pass per clock mode; a mode whose interval excludes the declared mode's pass changes who passes."),

        Table("A7", "the verifier scope: writes only against all actions (verifier_scope of run.json)",
              "scoring reads changes the ranking of agents without changing their outcomes",
              "the pass rate of every agent under the two scopes and Spearman's rank correlation of the agents' pass rates between the scopes",
              "the justification of design v2 section 4",
              ["agent", "model", "n_writes", "pass_writes", "pass_writes_low", "pass_writes_high", "n_all", "pass_all", "pass_all_low", "pass_all_high", "rank_correlation", "agents_in_both_scopes"], table_a7,
              rule="Read the rank correlation: near 1 means scoring reads leaves the ranking of agents as the write only scope has it."),

        Table("A8", "the abstention rule of the diagnostics server: the fixed posterior cut (abstention_rule fixed_cut of run.json, with abstention_cut) against the decision rule E10 (abstention_rule decision, the default)",
              "the decision rule declares on inseparable episodes and commits on separable ones more often than the constant",
              "abstention precision (correct declarations among declarations) and recall (correct declarations among the starts the grade E18 marks inseparable, the episodes whose verdict carries no grade) with Wilson intervals",
              "whether the threshold should follow prices",
              ["agent", "model", "abstention_rule", "n", "abstentions", "inseparable_starts", "precision", "precision_low", "precision_high", "recall", "recall_low", "recall_high"], table_a8,
              rule="Read precision and recall of the declarations against the grade; the better rule declares on inseparable starts and commits on separable ones."),

        Table("A9", "the price table: placeholder against the values of the subject matter experts (sme) and against the full tier priced far above the rest (full_tier_high)",
              "conclusions on cost hold under the alternative tables",
              "cost, normalized cost and pass per table, the paired difference of normalized cost and pass against placeholder, Holm over the normalized cost differences",
              "the sensitivity the preregistration promised", *a9,
              rule="Read whether the ordering of agents by normalized cost holds under every price table."),

        Table("A10", "the guidance level L1 to L3 (guidance_level of run.json)",
              "the level an agent needs to pass is a measured quantity and differs between agents",
              "pass per level, the paired difference against L1, Holm over the pass differences",
              "the information a prompt supplies", *a10,
              rule="Read pass per guidance level; the lowest level at which an agent's interval reaches the next level's pass is the information its prompt needs."),

        Table("A11", "one against two faulty components at the start (n_faulty, from the hidden start of every episode)",
              "two faults lower pass and raise unnecessary replacements",
              "pass and unnecessary replacements per number of faults with intervals; the two levels are different episodes, so no paired difference is formed",
              "difficulty", *a11,
              rule="Read pass and unnecessary replacements of two fault starts against one fault starts."),

        Table("A12", "the ranking model: the discriminants with the urn (ranking_model discriminant_urn) against the joint interventional likelihood of E1 and E2 (ranking_model joint)",
              "the joint model is calibrated where the discriminants were not, and its top candidate does not fall with more readings",
              "per number of readings and over all readings: the Brier score of the top candidate's posterior over the rank_root_causes calls with a bootstrap interval clustered by block, and the share of calls whose top candidate is the hidden start with a Wilson interval; A12_curves.csv holds the calibration bins (ten posterior bins with their bounds, the mean posterior and the observed frequency)",
              "the resolution of the misfit of 22 October",
              ["agent", "model", "ranking_model", "readings", "calls", "brier", "brier_low", "brier_high", "top_candidate_correct", "top1_low", "top1_high"], table_a12,
              rule="Read the Brier score against zero and the top candidate rate by readings; a rate that falls with more readings is the misfit of 22 October."),

        Table("GEPA", "the planner prompt: the untuned prompt of the Plan-Execute runner against GEPA's best candidate (gepa_optimize.py, mode gepa in run.json)",
              "a reflective prompt optimizer with the verifier's reward as its score and the verdict's checks as its feedback raises the reward of the prompted agent at a matched rollout budget",
              "the mean validation reward of the seed and the best candidate, the episodes the optimization played (validation rollouts counted), and the untuned prompt's mean reward over the same number of plan_execute episodes of the same model and config",
              "whether prompt optimization closes the gap a trained policy would close, at the same number of episodes",
              ["run", "model", "config", "candidate", "val_seeds", "val_mean_reward", "episodes_played", "untuned_mean_reward_same_episodes", "untuned_episodes_available"], table_gepa,
              rule="Read the best candidate's validation reward against the seed candidate's and against the untuned prompt at the same number of episodes."),

    ]


# ----- output -----

def table_markdown(table, rows):
    """The header row and the rows of one table, or the header and one empty row."""
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    if rows:
        lines += ["| " + " | ".join(_cell(r.get(c, "")) for c in table.columns) + " |" for r in rows]
    else:
        lines += ["| " + " | ".join("" for _ in table.columns) + " |"]
    return lines


def _tool_sequence(ep_dir):
    """The tool calls of an episode in order, one line each, from the trajectory or the action server's own record."""
    trajectory = read_json(Path(ep_dir) / "trajectory.json", [])
    if trajectory:
        return [f"{i}. {step.get('server')}.{step.get('tool')}({json.dumps(step.get('tool_args', {}), separators=(',', ':'))})"
                + (f" error: {step['error']}" if step.get("error") else "") for i, step in enumerate(trajectory, 1)]
    state = read_json(Path(ep_dir) / "episode.json", {})
    return [f"{i}. hydraulic_rig.{c['tool']}({json.dumps(c.get('arguments', {}), separators=(',', ':'))})" for i, c in enumerate(state.get("tool_calls", []), 1)]


def _verdict_summary(ep_dir):
    verdict = read_json(Path(ep_dir) / "hidden" / "verdict.json", {})
    shown = {k: v for k, v in verdict.items() if not isinstance(v, (dict, list))}
    shown["checks"] = {name: (c.get("ok") if isinstance(c, dict) else c) for name, c in (verdict.get("checks") or {}).items()}
    return json.dumps(shown, indent=1, default=str)


def readme_runs(published_root, runs_root):
    """The run folders the README describes, one per run name: the published copy when it exists, else the synced one,
    so a run that exists in both places counts once."""
    folders = {}
    for root in (published_root, runs_root):
        for run_dir in run_dirs(root):
            folders.setdefault(run_dir.name, run_dir)
    return [folders[name] for name in sorted(folders)]


def example_episode(run_folders):
    """The first episode with a report whose tool calls are on record (a trajectory or the action server's own list),
    else the first episode with a report, as (run folder, episode folder)."""
    with_report = [(run_dir, ep_dir) for run_dir in run_folders for ep_dir in episode_dirs(run_dir) if (ep_dir / "report.md").exists()]
    for run_dir, ep_dir in with_report:
        if (ep_dir / "trajectory.json").exists() or read_json(ep_dir / "episode.json", {}).get("tool_calls"):
            return run_dir, ep_dir
    return with_report[0] if with_report else (None, None)


def results_readme(tables_rows, ctx, readme_path, runs_root, published_root, cluster):
    """The README of the published results folder."""
    readme_path = Path(readme_path)
    run_folders = readme_runs(published_root, runs_root)
    metas = [read_json(r / "run.json", {}) for r in run_folders]
    out = ["# Results of the hydraulic rig gym", ""]
    if metas:
        models = sorted({m.get("model") or m.get("agent_label", "") for m in metas if m.get("model") or m.get("agent_label")})
        presets = sorted({m.get("config_name", "") for m in metas if m.get("config_name")})
        started = sorted(m.get("started", "") for m in metas if m.get("started"))
        ended = sorted(m.get("ended", "") for m in metas if m.get("ended"))
        out += [f"Models and agents: {', '.join(models)}. Cluster: {cluster}. Runs from {started[0][:10] if started else '?'} to "
                f"{ended[-1][:10] if ended else '?'}. Config presets: {', '.join(presets)}. Run folders: {len(metas)}; episodes read: {ctx['episodes']}.",
                "Every table below is `aggregate/<id>.csv` of this folder; the published run folders under `runs/` hold each episode's report, "
                "trajectory, released evidence, verdict and oracle. Intervals are Wilson for proportions and a bootstrap drawn by recorded block otherwise; "
                f"paired differences are corrected by Holm at alpha {ctx['alpha']}.", ""]
    else:
        out += [f"No run exists yet. The tables below show the design of every ablation with empty cells; the values come from the runs on {cluster}, "
                "published here after they exist.", ""]
    for table, rows in tables_rows:
        out += [f"## {table.id}: {table.varies}", "", f"Hypothesis: {table.hypothesis}", "", f"Reading rule: {table.rule}", ""]
        out += table_markdown(table, rows) + [""]
    run_dir, ep_dir = example_episode(run_folders)
    out += ["## Example episode", ""]
    if ep_dir is None:
        out += ["No episode has been played yet; after the runs this section links one episode's report and shows its tool calls and its verdict.", ""]
    else:
        link = os.path.relpath(ep_dir / "report.md", readme_path.parent)
        out += [f"Episode `{ep_dir.name}` of run `{run_dir.name}`: [report.md]({link}).", "", "Tool calls in order:", "", "```"]
        out += _tool_sequence(ep_dir) or ["(no tool call)"]
        out += ["```", "", "Verifier verdict (flat fields and the outcome of every check):", "", "```json", _verdict_summary(ep_dir), "```", ""]
    out += ["## How to reproduce", "",
            "On the Mac, a wiring check without a model (the fork's virtual environment first on PATH, HYD_FACTS_PATH set to the rig data file):", "",
            "```", "python benchmarks/hydraulic_rig/run_episodes.py --config static --agent scripted --seeds 1",
            "python benchmarks/hydraulic_rig/run_episodes.py --config static --agent oracle --seeds 3", "```", "",
            "On Columbia's cluster, the jobs of `CausalDiagnosis/Code/hpc/README.md` in order: `setup_env.sbatch`, `download_model.sbatch`, "
            "`serve_model.sbatch` for the agent model and for the judge model, `run_pilot.sbatch`, `run_grid.sbatch` (a job array over `grid.txt`), "
            "`run_gepa.sbatch`; then on the Mac `bash Code/hpc/sync_runs.sh`, which copies the run folders, publishes their subset under `runs/` "
            "with `benchmarks/hydraulic_rig/publish_runs.py` and rebuilds the tables and this file with:", "",
            "```", "python benchmarks/hydraulic_rig/aggregate.py --runs <synced run folders> --out benchmarks/hydraulic_rig/results/aggregate", "```", ""]
    return "\n".join(out)


def markdown(tables_rows, ctx):
    runs = run_dirs(ctx["runs"])
    out = [f"# Ablation tables A1 to A12 ({datetime.now():%Y-%m-%d %H:%M})", "",
           f"Run folders under `{ctx['runs']}`: {len(runs)}; episodes read: {ctx['episodes']}. "
           f"Intervals: Wilson for proportions, otherwise a bootstrap of {ctx['resamples']} resamples drawn by recorded block; paired "
           f"differences match episodes on seed, scenario and every other factor; Holm at alpha {ctx['alpha']} over the primary metric of a table.", ""]
    if not runs:
        out += ["No run folder exists yet, so every table below states its design with empty cells; the values come from the runs on Columbia's cluster.", ""]
    for table, rows in tables_rows:
        out += [f"## {table.id}", "", f"- What varies: {table.varies}", f"- Hypothesis: {table.hypothesis}", f"- Metric: {table.metric}",
                f"- What a result would mean: {table.reading}", ""]
        out += table_markdown(table, rows)
        if not rows:
            out += ["", "No episode contributes to this table yet."]
        out.append("")
    return "\n".join(out)


def _cell(v):
    return "" if v is None else str(v).replace("|", "/")


def aggregate(runs_root, out_dir, resamples=10_000, alpha=0.05, seed=0, eqa_gold_direction="no change", eqa_gold_action="open a work order",
              readme_path=None, published_root=None, cluster="Columbia Insomnia"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = Path(readme_path) if readme_path else out_dir.parent / "README.md"
    published_root = Path(published_root) if published_root else out_dir.parent / "runs"
    episodes = load_episodes(runs_root)
    ctx = {"runs": str(runs_root), "resamples": resamples, "alpha": alpha, "seed": seed, "episodes": len(episodes),
           "eqa_gold_direction": eqa_gold_direction, "eqa_gold_action": eqa_gold_action}
    tables_rows = []
    for table in tables():
        rows = table.build(episodes, ctx) if episodes else []
        write_rows(out_dir / f"{table.id}.csv", rows, table.columns)
        tables_rows.append((table, rows))
    write_rows(out_dir / "A12_curves.csv", ctx.get("a12_curves", []),
               ["agent", "model", "ranking_model", "readings", "bin_lo", "bin_hi", "predicted_mean", "observed_frequency", "n"])
    (out_dir / "aggregate.md").write_text(markdown(tables_rows, ctx), encoding="utf-8")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(results_readme(tables_rows, ctx, readme_path, Path(runs_root), published_root, cluster), encoding="utf-8")
    return tables_rows


def main(argv):
    ap = argparse.ArgumentParser(description="one table per ablation A1 to A12 from the run folders of run_episodes.py")
    ap.add_argument("--runs", default=str(DEFAULT_RUNS), help="the folder that holds the run folders")
    ap.add_argument("--out", default=str(HERE / "results" / "aggregate"))
    ap.add_argument("--readme", default=str(HERE / "results" / "README.md"), help="the README of the published results folder")
    ap.add_argument("--published", default=str(HERE / "results" / "runs"), help="the published run subsets, where the example episode is linked")
    ap.add_argument("--cluster", default="Columbia Insomnia, one NVIDIA H100 per served model")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap draws")
    ap.add_argument("--eqa-gold-direction", default="no change", help="A2: the released gold direction of every counterfactual question")
    ap.add_argument("--eqa-gold-action", default="open a work order", help="A2: the released gold answer of every action question")
    a = ap.parse_args(argv)
    tables_rows = aggregate(a.runs, a.out, a.resamples, a.alpha, a.seed, a.eqa_gold_direction, a.eqa_gold_action, a.readme, a.published, a.cluster)
    print(f"wrote {a.out}: {len(tables_rows)} tables, {sum(len(rows) for _, rows in tables_rows)} rows from {len(run_dirs(a.runs))} run folders")


if __name__ == "__main__":
    main(sys.argv[1:])
