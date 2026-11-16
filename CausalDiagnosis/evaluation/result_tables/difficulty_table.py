"""The difficulty table of the datasheet: how the model-free reference policies fare on the development set, read from the run
folders of one reference job on the base preset, and printed in CausalDiagnosis/docs/dataset.md.

usage: python3 -m CausalDiagnosis.evaluation.result_tables.difficulty_table <run folder> [<run folder> ...]

The job plays the reference policies of design.REFERENCE_POLICIES with --starts development. Each row counts the episodes of one
policy, the binary pass and Top-1 (diagnosis_all_correct) as counts, and the graded pass and graded Top-1 of the verifier as the
means of their shares per episode. An episode that the verifier did not score counts as a failure on every outcome, zero on a graded
one, the registered counting.unscored, and the row prints how many there were. The datasheet prints markdown of the same rows, with empty cells until the job has run. The
outcomes on the test set wait until the agents have been played there, so the table holds the development set alone.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.utils.csv_rows import as_bool, as_float
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import read_rows  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AGENTS_PRESET, REFERENCE_POLICIES  # noqa: E402

START_SETS = (("development", "Development set"),)
COUNTS = ("pass", "top1")  # binary outcomes, counted
SHARES = ("graded_pass", "graded_top1")  # graded outcomes, summed over episodes and shown as their mean
CELL_HEADS = ("pass", "Top-1", "graded pass", "graded Top-1", "unscored")


def difficulty_table(run_dirs, preset=AGENTS_PRESET):
    """[{policy, starts, episodes, pass, top1, graded_pass, graded_top1, unscored}], one row per reference policy and start set in the
    order of REFERENCE_POLICIES and START_SETS, every row present with 0 episodes when no run played it. A run of another
    start set, another preset or an agent that is not a reference policy is refused, so the table never mixes in other runs."""
    policies = [label for label, _ in REFERENCE_POLICIES]
    sets = [s for s, _ in START_SETS]
    rows = {(p, s): {"policy": p, "starts": s, "episodes": 0, "pass": 0, "top1": 0, "graded_pass": 0.0, "graded_top1": 0.0, "unscored": 0}
            for p in policies for s in sets}
    for run_dir in run_dirs:
        meta = read_json(Path(run_dir) / "run.json", {})
        starts = meta.get("starts")
        if starts not in sets:
            raise ValueError(f"{run_dir}: start set {starts!r}, the table reads {' and '.join(sets)} only")
        if meta.get("config_name") != preset:
            raise ValueError(f"{run_dir}: preset {meta.get('config_name')!r}, the table reads the {preset} preset only")
        for r in read_rows(run_dir):
            if r["agent"] not in policies:
                raise ValueError(f"{run_dir}: agent {r['agent']!r} is no reference policy of {', '.join(policies)}")
            row = rows[(r["agent"], starts)]
            row["episodes"] += 1
            if as_bool(r.get("pass")) is None:  # unscored: a failure on every outcome, counted on its own
                row["unscored"] += 1
                continue
            row["pass"] += bool(as_bool(r["pass"]))
            row["top1"] += bool(as_bool(r["diagnosis_all_correct"]))
            for field in SHARES:
                row[field] += as_float(r[field]) or 0.0
    return list(rows.values())


def _cells(row):
    n = row["episodes"]
    if n == 0:
        return [""] * len(CELL_HEADS)
    return [f"{row[c]} of {n}" for c in COUNTS] + [f"{row[s] / n:.2f}" for s in SHARES] + [str(row["unscored"])]


def markdown(rows):
    """The datasheet's table: one line per reference policy, the four outcomes on each start set."""
    names = dict(REFERENCE_POLICIES)
    by = {(r["policy"], r["starts"]): r for r in rows}
    head = ["Policy"] + [f"{label}, {h}" for _, label in START_SETS for h in CELL_HEADS]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for policy, _ in REFERENCE_POLICIES:
        cells = [names[policy]] + [c for s, _ in START_SETS for c in _cells(by[(policy, s)])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", type=Path, help="the run folders of the reference job")
    print(markdown(difficulty_table(ap.parse_args(argv).runs)))


if __name__ == "__main__":
    main()
