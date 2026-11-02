"""Publish the subset of run folders that goes to GitHub with the results (docs/hydraulic_rig/benchmarks.md).

usage: python benchmarks/hydraulic_rig/publish_runs.py --runs <folder of synced run folders> [--out benchmarks/hydraulic_rig/results/runs]
       [--llm-cap-bytes 200000]

For every run folder the published copy holds run.json, scores.csv, summary.md (and gepa_result.json and best_prompt.txt
of a GEPA run), the aggregate tables beside them, and per episode the report, the trajectory, the released evidence, the
question, the plan, the answer, the static prompt, and under hidden/ the verdict, the oracle, the scores, the static
record and the judge record, since the runs are over when they are published. llm_calls.json comes along only when it is
at most --llm-cap-bytes long; a larger one is replaced by a note with its size. The full log, the state file and the
launcher files stay out. Every endpoint field is set to null and every path field reduced to its basename, so no endpoint,
host or key reaches the published copy; the key never enters a run folder in the first place.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import HIDDEN, episode_dirs, read_json, run_dirs, write_json  # noqa: E402

RUN_FILES = ("run.json", "scores.csv", "summary.md", "gepa_result.json", "best_prompt.txt")
EPISODE_FILES = ("report.md", "trajectory.json", "released.json", "question.txt", "plan.txt", "answer.txt", "prompt.txt")
HIDDEN_FILES = ("verdict.json", "oracle.json", "score.json", "static.json", "judge.json")
SECRET_KEYS = ("endpoint", "judge_endpoint", "base_url", "reflection_endpoint", "judge_base_url")
PATH_KEYS = ("facts_path",)
LLM_CAP_BYTES = 200_000


def scrubbed(value):
    """The value with every endpoint field set to null and every path field reduced to its basename, recursively."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in SECRET_KEYS:
                out[k] = None
            elif k in PATH_KEYS and isinstance(v, str):
                out[k] = Path(v).name
            else:
                out[k] = scrubbed(v)
        return out
    if isinstance(value, list):
        return [scrubbed(v) for v in value]
    return value


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        write_json(target, scrubbed(read_json(source)))
    else:
        shutil.copyfile(source, target)


def publish_run(run_dir, out_dir, llm_cap=LLM_CAP_BYTES):
    """One run folder's published subset; returns the number of episodes published."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    for name in RUN_FILES:
        if (run_dir / name).exists():
            copy_file(run_dir / name, out_dir / name)
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
    """Every run folder under the root, plus an index; returns {run name: episodes published}."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    published = {}
    for run_dir in run_dirs(runs_root):
        published[run_dir.name] = publish_run(run_dir, out_root / run_dir.name, llm_cap)
    lines = ["# Published run folders", "",
             "Each folder is the subset of a run folder that goes with the results: run.json (endpoints removed), scores.csv, summary.md, "
             "and per episode the report, the trajectory, the released evidence, the question, the plan and the answer, with the verdict, the "
             f"oracle and the scores under hidden/; llm_calls.json only up to {llm_cap} bytes. The full run folders stay on the machine that "
             "synced them.", "", "| run | episodes | agent | model | config |", "|---|---|---|---|---|"]
    for name, count in published.items():
        meta = read_json(out_root / name / "run.json", {})
        lines.append(f"| [{name}]({name}/summary.md) | {count} | {meta.get('agent_label', meta.get('agent', ''))} | {meta.get('model') or ''} | {meta.get('config_name', '')} |")
    (out_root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return published


def main(argv):
    ap = argparse.ArgumentParser(description="publish the subset of run folders that goes with the results")
    ap.add_argument("--runs", required=True, help="the folder that holds the synced run folders")
    ap.add_argument("--out", default=str(HERE / "results" / "runs"))
    ap.add_argument("--llm-cap-bytes", type=int, default=LLM_CAP_BYTES)
    a = ap.parse_args(argv)
    published = publish(a.runs, a.out, a.llm_cap_bytes)
    print(f"published {len(published)} run folders with {sum(published.values())} episodes to {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
