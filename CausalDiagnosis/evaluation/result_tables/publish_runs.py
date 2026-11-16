"""Copy the readable subset of the run folders that the tables read into one folder with an index (CausalDiagnosis/docs/experiments.md).

usage: python -m CausalDiagnosis.evaluation.result_tables.publish_runs --runs <folder of synced run folders> [--out CausalDiagnosis/results/published_runs]
       [--llm-cap-bytes 200000]

For every run folder the published copy holds run.json, scores.csv, summary.md, of a run of the online loop of GEPA
gepa_online.json and the texts of every candidate under candidates/ (its reflection calls under reflection/ stay out),
the aggregate tables beside them, and per episode the report, the trajectory, the released evidence, the
question, the plan, the answer, the static prompt, and under hidden/ the verdict, the oracle, the scores, the static
record and the judge record, since the runs are over when they are published. llm_calls.json comes along only when it is
at most --llm-cap-bytes long; a larger one is replaced by a note with its size. The full log, the state file and the
launcher files stay out. Every file is copied byte for byte: the drivers write run folders in their public form
(public_record.py), so no endpoint, host, job or path of a machine is left to remove. The default folder results/published_runs
is listed in .gitignore, since the copy holds recorded readings, so it reaches no repository. A run folder that no table
reads stays unpublished and INDEX.md names it with the reason of results_index.refused: a run on the memory backend,
a run of the retired mode gepa or a model agent on a row of the environment table over the rig's action server.
"""
from CausalDiagnosis.utils.paths import PUBLISHED_RUNS
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, episode_dirs, run_dirs
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402

RUN_FILES = ("run.json", "scores.csv", "summary.md", "gepa_online.json")
RUN_FOLDERS = ("candidates",)  # the texts of every candidate of a run of the online loop of GEPA, copied byte for byte
EPISODE_FILES = ("report.md", "trajectory.json", "released.json", "question.txt", "plan.txt", "plans.json", "answer.txt", "prompt.txt")
HIDDEN_FILES = ("verdict.json", "oracle.json", "score.json", "static.json", "judge.json")
LLM_CAP_BYTES = 200_000


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def publish_run(run_dir, out_dir, llm_cap=LLM_CAP_BYTES):
    """One run folder's published subset; returns the number of episodes published."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    for name in RUN_FILES:
        if (run_dir / name).exists():
            copy_file(run_dir / name, out_dir / name)
    for folder in RUN_FOLDERS:
        for source in sorted(p for p in (run_dir / folder).glob("*") if p.is_file()):
            (out_dir / folder).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, out_dir / folder / source.name)
    for table in sorted((run_dir / "aggregate").glob("*")) if (run_dir / "aggregate").exists() else []:
        if table.suffix in (".csv", ".md"):
            copy_file(table, out_dir / "aggregate" / table.name)
    episodes = episode_dirs(run_dir)
    for ep_dir in episodes:
        target = out_dir / ep_dir.name
        for name in EPISODE_FILES:
            if (ep_dir / name).exists():
                copy_file(ep_dir / name, target / name)
        for name in HIDDEN_FILES:
            if (ep_dir / HIDDEN / name).exists():
                copy_file(ep_dir / HIDDEN / name, target / HIDDEN / name)
        calls = ep_dir / "llm_calls.json"
        if calls.exists():
            size = calls.stat().st_size
            if size <= llm_cap:
                copy_file(calls, target / "llm_calls.json")
            else:
                target.mkdir(parents=True, exist_ok=True)
                (target / "llm_calls.omitted.txt").write_text(f"llm_calls.json has {size} bytes, above the published cap of {llm_cap} bytes; "
                                                              "the full file stays in the synced run folder.\n", encoding="utf-8")
    return len(episodes)


def publish(runs_root, out_root, llm_cap=LLM_CAP_BYTES):
    """Every run folder under the root that the tables read (results_index.folders_read), plus an index that also names the
    folders left out with the reason; returns {run name: episodes published}. A root of development runs is never published."""
    if pooling_rules.root_kind(runs_root) == "development":
        raise SystemExit(f"{runs_root} holds development runs, which are not results, so nothing of it is published")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    published, left_out = {}, {}
    read, pooled_out = {p.name for p in pooling_rules.folders_read(runs_root)}, pooling_rules.pooling_left_out(runs_root)
    for run_dir in run_dirs(runs_root):
        reason = pooling_rules.refused(run_dir) or pooled_out.get(run_dir.name)
        if reason is None and run_dir.name in read:
            published[run_dir.name] = publish_run(run_dir, out_root / run_dir.name, llm_cap)
        else:
            left_out[run_dir.name] = reason
    lines = ["# Published run folders", "",
             "Each folder is the readable subset of a run folder, every file byte for byte: run.json, scores.csv, summary.md, "
             "gepa_online.json and candidates/ of a run of the online loop of GEPA, "
             "and per episode the report, the trajectory, the released evidence, the question, the plan and the answer, with the verdict, the "
             f"oracle and the scores under hidden/; llm_calls.json only up to {llm_cap} bytes. The full run folders stay on the machine that "
             "synced them.", "", "| run | episodes | agent | model | config |", "|---|---|---|---|---|"]
    for name, count in published.items():
        meta = read_json(out_root / name / "run.json", {})
        lines.append(f"| [{name}]({name}/summary.md) | {count} | {meta.get('agent_label', meta.get('agent', ''))} | {meta.get('model') or ''} | {meta.get('config_name', '')} |")
    if left_out:
        lines += ["", "No table reads the run folders below, so they are not published.", "", "| run | reason |", "|---|---|"]
        lines += [f"| {name} | {reason} |" for name, reason in left_out.items()]
    (out_root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return published


def main(argv):
    ap = argparse.ArgumentParser(description="copy the readable subset of the run folders that the tables read")
    ap.add_argument("--runs", required=True, help="the folder that holds the synced run folders")
    ap.add_argument("--out", default=str(PUBLISHED_RUNS))
    ap.add_argument("--llm-cap-bytes", type=int, default=LLM_CAP_BYTES)
    a = ap.parse_args(argv)
    published = publish(a.runs, a.out, a.llm_cap_bytes)
    print(f"published {len(published)} run folders with {sum(published.values())} episodes to {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
