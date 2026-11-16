"""The paper's granite numbers that the acceptance check of the two paper-era ports compares with (AssetOpsBench, arXiv
2506.03828v4), and the reader of scored run folders with the statistics acceptance_check.py takes from it
(CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md).

load_runs reads scores.csv (written by score_runs.py) and run.json of every run folder and pools their episodes, which must come
from one model and one telemetry. Every mean carries a 95 percent interval from a bootstrap that resamples whole scenarios
(stats.cluster_bootstrap of the hydraulic rig), since every episode played on one scenario shares its text and its data.
"""
from pathlib import Path

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_folder_names import RUN_FILE, SCORES_FILE

from CausalDiagnosis.evaluation.metrics import table_statistics as stats  # noqa: E402  the hydraulic rig's statistics: cluster_bootstrap
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import read_rows  # noqa: E402

# Printed bar labels of Figures 5a and 5b for granite-3-3-8b-instruct: TC, DRA, RV. Agent-As-Tool in percent over 141 scenarios.
# Plan-Execute over 84, inferred: v4 prints 26, 32 and 31, and v1 page 8 prints two decimals, all six on multiples of 1/84 and
# on no other denominator up to 141. The paper states no such denominator and names no such set.
FIGURE5 = {"agent_as_tool": (35, 43, 37), "plan_execute": ("22/84 (26.19)", "27/84 (32.14)", "26/84 (30.95)")}
# Table 3 granite-3-3-8b: (steps, runtime in seconds) per paradigm and task kind, over 99 single and 42 multi-agent tasks.
TABLE3 = {("agent_as_tool", "single"): ("5.3", "197"), ("agent_as_tool", "multi"): ("6.6", "228"),
          ("plan_execute", "single"): ("5.2", "413"), ("plan_execute", "multi"): ("5.1", "433")}
# granite-3-3-8b-instruct: mean, standard deviation, minimum and maximum of planned steps, plans without a step, tasks
TABLE12_PLANNED = ("5.56", "2.44", 2, 30, 0, 141)
JUDGED = (("task_completion", "Task completion"), ("data_retrieval_accuracy", "Data retrieval accuracy"),
          ("generalized_result_verification", "Result verification"))
PAPER_SCENARIOS = 141


def rows_files(metas):
    """The SHA-256 prefix of the optional rows file of each run, None for a run on public readings alone."""
    return {((m.get("layer_b") or {}).get("recovered_rows") or {}).get("sha256", "")[:12] or None for m in metas}


def load_runs(run_dirs):
    """(run.json of every run, the scores.csv rows of all runs). Refuses runs without scores, runs of different models and
    runs with and without one rows file."""
    metas, rows = [], []
    for run_dir in map(Path, run_dirs):
        meta = read_json(run_dir / RUN_FILE)
        if meta is None or not (run_dir / SCORES_FILE).exists():
            raise SystemExit(f"{run_dir} needs {RUN_FILE} and {SCORES_FILE}. Run score_runs.py on it first")
        metas.append(meta)
        rows += read_rows(run_dir)
    if len(rows_files(metas)) > 1:
        raise SystemExit("the runs mix telemetry, with and without one rows file, and these tables are for one")
    models = sorted({m["model"] for m in metas})
    if len(models) != 1:
        raise SystemExit(f"the runs mix models {models}, and the tables are for one model")
    return metas, rows


def number(row, field):
    value = row.get(field)
    if value in (None, ""):
        return None
    if value in ("True", "False"):
        return float(value == "True")
    return float(value)


def mean_interval(rows, field):
    """Mean of a field over the episodes that have it, with a 95 percent interval from a bootstrap over scenarios."""
    pairs = [(number(r, field), r["scenario_id"]) for r in rows if number(r, field) is not None]
    if not pairs:
        return None
    values, clusters = zip(*pairs)
    return stats.cluster_bootstrap([[v, 1.0] for v in values], list(clusters), stats.ratio)  # the mean as a sum over a count


def select(rows, agent=None, family=None, kind=None):
    return [r for r in rows if (agent is None or r["agent"] == agent) and (family is None or r["family"] == family)
            and (kind is None or r["task_kind"] == kind)]


def scenario_count(rows):
    return len({r["scenario_id"] for r in rows})


def paper_era_field(agent, field):
    """The column of scores.csv under what the paper-era code takes and the paper's printed numbers lie on, multiples of one
    over the number of scenarios: the majority of five calls for Agent-As-Tool, one call at temperature 0 for Plan-Execute."""
    return f"paper_{field}_majority" if agent == "agent_as_tool" else f"single_{field}"
