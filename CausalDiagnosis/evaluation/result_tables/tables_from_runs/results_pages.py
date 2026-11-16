import json
from datetime import date
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_dirs  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.utils.public_rules.visits import is_no_call  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402


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
    return [f"{i}. hydraulic_rig.{c['tool']}({json.dumps(c.get('arguments', {}), separators=(',', ':'))})" for i, c in enumerate((c for c in state.get("tool_calls", []) if not is_no_call(c)), 1)]


def _verdict_summary(ep_dir):
    verdict = read_json(Path(ep_dir) / "hidden" / "verdict.json", {})
    shown = {k: v for k, v in verdict.items() if not isinstance(v, (dict, list))}
    shown["checks"] = {name: (c.get("ok") if isinstance(c, dict) else c) for name, c in (verdict.get("checks") or {}).items()}
    return json.dumps(shown, indent=1, default=str)


def readme_runs(published_root, runs_root):
    """The run folders the README describes, one per run name: the published copy when it exists, else the synced one,
    so a run that exists in both places counts once, and never a run that the tables refuse."""
    folders = {}
    for root in (published_root, runs_root):
        for run_dir in pooling_rules.folders_read(root):
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


def results_readme(tables_rows, ctx, runs_root, published_root, cluster):
    """The README of the results folder, which names its run folders and links none, since they stay out of the repository."""
    run_folders = readme_runs(published_root, runs_root)
    metas = [read_json(r / "run.json", {}) for r in run_folders]
    out = ["# Results of the hydraulic rig gym", ""]
    if metas:
        models = sorted({m.get("model") or m.get("agent_label", "") for m in metas if m.get("model") or m.get("agent_label")})
        presets = sorted({m.get("config_name", "") for m in metas if m.get("config_name")})
        started = sorted(m.get("started", "") for m in metas if m.get("started"))
        ended = sorted(m.get("ended", "") for m in metas if m.get("ended"))
        out += [f"Models and agents: {', '.join(models)}. {f'Cluster: {cluster}. ' if cluster else ''}Runs from {started[0][:10] if started else '?'} to "
                f"{ended[-1][:10] if ended else '?'}. Config presets: {', '.join(presets)}. Run folders: {len(metas)}; episodes read: {ctx['episodes']}.",
                "Every table below is `tables/<id>.csv` of this folder, and `tables/RUNS.csv` names the run folder behind every row of the "
                "leaderboard that `leaderboard.py` writes beside this file. The run folders stay out of the repository, since they hold sensor "
                "readings of single cycles. Every interval comes from a bootstrap drawn by recorded block, proportions included, and "
                f"paired differences are corrected by Holm at alpha {ctx['alpha']}.", ""]
    else:
        out += ["No run has been played on the current code, so the tables below show the design of every ablation with empty cells. "
                f"The cells stay empty until the test set is played{f' on {cluster}' if cluster else ''}.", ""]
    out += ["`MAIN_TABLES.md` beside this file sets out the result tables, first those with the metrics of the AssetOpsBench paper, then the verified tables that paper lacks, "
            "the tables of training and the strata, and `main_tables.tex` holds them in LaTeX.", ""]
    for table, rows in tables_rows:
        out += [f"## {table.id}: {table.varies}", "", f"Hypothesis: {table.hypothesis}", "", f"Reading rule: {table.rule}", ""]
        out += table_markdown(table, rows) + [""]
    run_dir, ep_dir = example_episode(run_folders)
    out += ["## Example episode", ""]
    if ep_dir is None:
        out += ["No episode has been played yet. After the runs this section names one episode and shows its tool calls and its verdict.", ""]
    else:
        out += [f"Episode `{ep_dir.name}` of the run folder `{run_dir.name}`, which stays out of the repository with the other run folders.", "",
                "Tool calls in order:", "", "```"]
        out += _tool_sequence(ep_dir) or ["(no tool call)"]
        out += ["```", "", "Verifier verdict (flat fields and the outcome of every check):", "", "```json", _verdict_summary(ep_dir), "```", ""]
    out += ["## How to reproduce", "",
            "On a laptop, a wiring check without a model (the fork's virtual environment first on PATH, the data file built once by "
            "`python -m CausalDiagnosis.datasets.hyd_1.download_and_build --download`):", "",
            "```", "python -m CausalDiagnosis.experiments.episode_runners.run_episodes --config static --agent scripted --seeds 1",
            "python -m CausalDiagnosis.experiments.episode_runners.run_episodes --config static --agent oracle --seeds 3", "```", "",
            "On the cluster the runs follow the run order of `CausalDiagnosis/cluster_jobs/README.md`, from the two gates to the grid "
            "files and the training of the training table with `gepa_prompt_optimization.py`, the online loop of GEPA, then on a laptop `bash CausalDiagnosis/cluster_jobs/4_after_run/sync_runs.sh <remote runs folder>`, which copies the run folders to the laptop, where they stay. "
            "The tables, this file and the leaderboard are rebuilt from those copies with:", "",
            "```", "python -m CausalDiagnosis.evaluation.result_tables.tables_from_runs --runs <synced run folders> --out CausalDiagnosis/results/tables",
            "python -m CausalDiagnosis.evaluation.result_tables.leaderboard --since <agent>=<first valid start> --runs <synced run folders>", "```", ""]
    return "\n".join(out)


def markdown(tables_rows, ctx):
    runs = pooling_rules.folders_read(ctx["runs"])
    out = [f"# Ablation and result tables ({date.today().isoformat()})", "",
           f"Run folders under `{Path(ctx['runs']).name}`: {len(runs)}; episodes read: {ctx['episodes']}. "
           f"Intervals: a bootstrap of {ctx['resamples']} resamples drawn by recorded block, proportions included; paired "
           f"differences match episodes on seed, scenario and every other factor; Holm at alpha {ctx['alpha']} over the primary metric of a table.", ""]
    if not runs:
        out += ["No run has been played on the current code, so every table below states its design with empty cells. "
                "The cells stay empty until the test set is played on Columbia's cluster.", ""]
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
