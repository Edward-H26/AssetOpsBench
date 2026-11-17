from CausalDiagnosis.utils.paths import RESULTS
import argparse
import sys
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import DEFAULT_RUNS  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402  the fractions of the test set (fraction_plan, fraction_text, fraction_value)
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from . import table_declarations, write_tables


def main(argv):
    ap = argparse.ArgumentParser(description="one table per supporting ablation, the GEPA tables, the tables of the six rows, the environment, training and the fleet and the table of run folders from the run folders of run_episodes.py")
    ap.add_argument("--runs", default=str(DEFAULT_RUNS), help="the folder that holds the run folders")
    ap.add_argument("--out", default=str(RESULTS / "tables"))
    ap.add_argument("--readme", default=None, help="the README of the results folder (default: README.md beside INDEX.json, in the parent of --out)")
    ap.add_argument("--published", default=None, help="the published run subsets, which git ignores, searched first for the example episode (default: the folder of published runs beside INDEX.json, in the parent of --out)")
    ap.add_argument("--cluster", default=None, help="the cluster the runs were played on, which the page names only when it is given")
    ap.add_argument("--resamples", type=int, default=None, help="the resamples of the bootstrap (default: those of the registered file)")
    ap.add_argument("--alpha", type=float, default=None, help="the level of the tests (default: the level of the registered file)")
    ap.add_argument("--registered", default=None, help="the registered analysis file (default: registered_analysis.json beside this script)")
    ap.add_argument("--seed", type=int, default=None, help="seed of the bootstrap draws (default: the seed of the registered file)")
    ap.add_argument("--index", default=None, help="the results index INDEX.json (default: beside the README, in the parent of --out)")
    ap.add_argument("--departure", action="store_true", help="build the tables of registered runs with other resamples or another seed, which INDEX.json states")
    ap.add_argument("--commit", default=None, help="the commit whose runs of record the tables read when the root holds more than one group")
    ap.add_argument("--fraction", default=start_states.DEFAULT_FRACTION,
                    help="the fraction of the test set the tables read: a whole fraction r/5 reads the replicates 1 to r, a smaller one the cut "
                         "look tagged with it; below the fraction of record no registered test runs (default: %(default)s)")
    ap.add_argument("--table", default=None, help="build this table alone into --out, with no index and no page (ids: --list)")
    ap.add_argument("--list", action="store_true", help="print the id of every table of record and what varies in it, then stop")
    a = ap.parse_args(argv)
    if a.list:
        for table in table_declarations.tables():
            print(f"{table.id}\t{table.varies}")
        return
    if a.table:
        rows = write_tables.build_table(a.table, a.runs, a.out, a.resamples, a.alpha, a.seed, a.registered, a.departure, a.commit, fraction=a.fraction)
        print(f"wrote {Path(a.out) / (a.table + '.csv')}: {len(rows)} rows")
        print_chosen_starts(a.runs)
        return
    tables_rows = write_tables.aggregate(a.runs, a.out, a.resamples, a.alpha, a.seed, a.readme, a.published, a.cluster, a.index, registered_path=a.registered,
                            departure=a.departure, commit=a.commit, fraction=a.fraction)
    print(f"wrote {a.out}: {len(tables_rows)} tables, {sum(len(rows) for _, rows in tables_rows)} rows from {len(pooling_rules.folders_read(a.runs))} run folders")
    for name, reason in sorted(pooling_rules.pooling_left_out(a.runs).items()):  # the runs no table read, also in INDEX.json excluded
        print(f"left out {name}: {reason}", file=sys.stderr)
    print_chosen_starts(a.runs)


def print_chosen_starts(runs_root):
    """Prints on the standard error how many runs of chosen starts no table read, and which, which INDEX.json excluded names one by one."""
    if chosen := pooling_rules.chosen_starts_left_out(runs_root):
        print(f"left out {pooling_rules.chosen_starts_line(chosen)}", file=sys.stderr)
