from pathlib import Path
from statistics import median
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import context_or_api_error, input_tokens_per_call  # noqa: E402
from CausalDiagnosis.utils.csv_rows import as_bool, as_float  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import cell_statistics


# ----- one row per run folder (RUNS), the table that leaderboard.py reads -----
RUN_SETTINGS = ("agent", "model", "config", "mode", "diagnostics_server", "clock_mode", "surface", "guidance_level", "verifier_scope", "prices",
                "ranking_model", "abstention_rule", "training_variant", "lora_adapters")
RUN_META = ("starts", "master_seed", "replicate", "started", "episodes_planned", "judge_model")  # fields of run.json
RUN_SCORES = ("pass", "scope_pass", "judge_pass", "cost_normalized", "success", "reward")
RUN_COLUMNS = (["run", *RUN_SETTINGS, *RUN_META, "n", "judge_verdicts"] + [f"{m}{part}" for m in RUN_SCORES for part in ("", "_low", "_high", "_n")]
               + ["delay_median", "delay_n", "context_errors", "input_tokens_per_call", "status"])


def episode_delay(episode):
    """The delay that the reward charges an episode (verifier.reward): the larger of its detection and its repair delay,
    each already the largest over the components in scores.csv, or None when neither was measured."""
    delays = [d for d in (as_float(episode.get("detection_delay")), as_float(episode.get("repair_delay"))) if d is not None]
    return max(delays, default=None)


def run_lines(episodes):
    """The runs of every row of RUNS as [(run folder, run.json, episodes)]: a run folder alone, or the replicate runs of one line
    of the grid on the test set in replicate order, which share their condition (results_index.condition_of), their set of
    episodes and the rule of their replicates, one run per replicate, so the leaderboard reads a line over
    every replicate played."""
    lines = {}
    for _, eps in cell_statistics.groups(episodes, "run_dir").items():
        run_dir = Path(eps[0]["run_dir"])
        meta = read_json(run_dir / "run.json", {})
        key = run_dir if meta.get("replicate") is None else (
            pooling_rules.condition_of(meta), *(str(meta.get(k)) for k in ("starts", "master_seed", "seed_pool", "replicate_rule")))
        lines.setdefault(key, []).append((run_dir, meta, eps))
    return [sorted(runs, key=lambda run: (run[1].get("replicate") or 0, run[0].name)) for runs in lines.values()]


def pooled_meta(key, metas):
    """A field of run.json over the runs of a row of RUNS: the episodes planned add up, the start is the earliest, so that a
    cutoff of the leaderboard counts a line once every run of it started after the cutoff, and any other field is the value
    the runs share, or their values in replicate order."""
    values = [meta.get(key, "") for meta in metas]
    if len(values) == 1:
        return values[0]
    if key == "episodes_planned":
        return sum(values)
    if key == "started":
        return min(values)
    return values[0] if len({str(v) for v in values}) == 1 else " ".join(str(v) for v in values)


def table_runs(episodes, ctx):
    """One row per run folder, named by the folder, since two runs of one label started in the same second share the run
    column of scores.csv, and one row per line of the test set over its replicate runs (run_lines), named by their folders
    in replicate order: the settings of its episodes, the set of episodes, the replicates, the start, the judge model of run.json or,
    when run.json names none, the judge models of the records of its episodes,
    the pass rate, the pass rate with the read gate under the verifier scope all (scope_pass), the judge's pass rate, the mean
    cost over the cost of the oracle policy on the same seed (cost_normalized), the verifier's success rate and the mean reward,
    each with a bootstrap interval clustered by block, which the leaderboard prints for the declaring
    policy, the median delay over the episodes that have one, and, so that a context overflow of the served model shows per row,
    the share of episodes whose error cell names a context overflow or an error of the endpoint and the mean
    input tokens per model call of llm_calls.json."""
    rows = []
    for runs in run_lines(episodes):
        eps = [e for _, _, run_eps in runs for e in run_eps]
        metas = [meta for _, meta, _ in runs]
        row = {"run": " ".join(run_dir.name for run_dir, _, _ in runs), **{k: eps[0].get(k, "") for k in RUN_SETTINGS},
               **{k: pooled_meta(k, metas) for k in RUN_META}, "n": len(eps)}
        row["judge_model"] = row["judge_model"] or " ".join(sorted({e["judge_record_model"] for e in eps if e.get("judge_record_model")}))
        # the episodes with a verdict of the judge, a record or a judge cell of scores.csv, which the leaderboard waits for, since a missing
        # verdict reads as a failure once a judge is named
        row["judge_verdicts"] = sum(1 for e in eps if e.get("judge_record_protocol") or as_bool(e.get("judge_pass")) is not None)
        for m in RUN_SCORES:
            row.update(cell_statistics.summary(eps, m, ctx))
        delays = [d for d in map(episode_delay, eps) if d is not None]
        tokens, _ = input_tokens_per_call([e["episode_dir"] for e in eps if e.get("episode_dir") is not None])
        row.update({"delay_median": _r(median(delays)) if delays else "", "delay_n": len(delays),
                    "context_errors": _r(sum(1 for e in eps if context_or_api_error(e.get("error"))) / len(eps)), "input_tokens_per_call": _r(tokens),
                    "status": cell_statistics.row_status(eps, ctx)})
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows
