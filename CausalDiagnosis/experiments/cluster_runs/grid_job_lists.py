"""The grid files of CausalDiagnosis/cluster_jobs/, written from the design of the tables (experiment_tables.py) and of the training (training_variants/), so that no run
line of the cluster is written by hand and a row, a preset or a variant renamed there changes every grid with it.

    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists               the files that differ from the design, exit 1 when one does
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists write         writes the five files into CausalDiagnosis/cluster_jobs/
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists presets FILE  the presets a grid file names, which the reference policies play
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists policies      the agents of the reference runs of CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists wiring        the agents among them of the wiring check, which play the first
                                                                replicate of the test set alone
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists agents [--model NAME] RUNS   the finished runs of the six rows of the main table under run folders,
                                                                played on the served model NAME
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists clocked FILE  the lines of a grid file whose clock charges the wall seconds of
                                                                thinking, and unclocked FILE the other lines
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists second_round AGENT FOLD  the sampling lines of the second round of sampling
                                                                for one agent and fold, the next samples of the configuration
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists folds         the folds of the registered split, which CausalDiagnosis/cluster_jobs/3_run/run_gepa_online.sh loops over
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists fillers FILE  the arguments of the filler run of every line of a grid file, in its order
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists part PART     one part of the grid as a grid file, PART one of PARTS
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists references PART  the reference runs beside one part of the grid as a grid file
    python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists row ID        a grid file of one line, the row ID of design.ROWS_BY_ID on the preset of
                                                                the main table, for one development row

A grid file holds one run of run_episodes/ per line after a comment line that names the table row or the ablation it serves,
as CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch reads it. grid_agents.txt holds the runs that need no training: the six rows of the main table on layer 0
and the healthy control, the arm of En9 that stands beside the main table, the environment ladder with the three agents without
the CausalDiagnosis MCP, whose base rung is three lines of the main table and whose top rung gives the baselines of the training
table, and the supporting ablations where design.ABLATION_PLACES puts them, each after the reference line of a row that no other
line plays on its preset. grid_pilot.txt and grid_gate.txt hold development runs that no table reads, the optional pilot that measures
the thinking rate against the value the asset file declares and the development gate. grid_training_sampling.txt and grid_training_evaluation.txt hold the sampling runs of the
fine tuning and the evaluation runs of every variant of the training table. The commands part and row print one part of the grid,
or one development row, as a grid file for a job to write into a folder of its own; part layers holds the rungs between layer 0
and all six big layers alone, the lines of the ladder that no table of the agents plays, for a job that plays the layer study.
grid_agents.txt ends with the same rows under the clock that charges the tokens the agent's model generated in place of the wall
seconds, a second condition of the clock: part environment_token, the six rows on the token twin of all six big layers, and part
layers_token, the rungs of part layers on their token twins (design.TOKEN_PRESETS). Then part misdiagnosis_price, the ablation L1
with the three agents without the CausalDiagnosis MCP, and last part environment, the three rows with the CausalDiagnosis MCP
on all six big layers as released, each after every other line, so that adding them left the place of every earlier line
unchanged. Every part plays on a server without LoRA adapters but the part training, which grid_training_evaluation.txt holds
and which plays beside the adapters (ADAPTER_PARTS), so the two cells of every registered comparison play on one serving setup:
the agents table reads the rows with the MCP on the full environment from part environment, and the training table reads the
same three lines without training from part training, on the server of its trained variants.
"""
import sys
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import training_variants as training  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs.pending_grid_lines import grid_lines  # noqa: E402
from . import run_lines, grid_files


# the options of a line that a filler run never plays, with their values
FILLER_DROPPED = ("--starts", "--replicate", "--fraction", "--start-cap", "--training-variant", "--training-fold", "--training-sample",
                  "--training-split", "--judge", "--resume")


def filler_arguments(line):
    """The arguments of the filler run in the slot of a run of a grid line (run_episodes/ --filler): the line without the options
    of FILLER_DROPPED, in the form --option value or --option=value, so that the filler plays the line's preset, clock, agent and
    switches as released, on the development starts that CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch adds, and judges nothing."""
    kept, skip = [], False
    for word in line.split():
        if skip:
            skip = False
        elif word.split("=", 1)[0] in FILLER_DROPPED:
            skip = "=" not in word
        else:
            kept.append(word)
    return " ".join(kept)


def show(lines):
    """One line of output per entry, and no line for none, so that a file written from it is empty."""
    if lines:
        print(*lines, sep="\n")
    return 0


def main(argv):
    command = argv[0] if argv else "check"
    if command == "write":
        for name, wanted in grid_files.files().items():
            (grid_files.HPC / name).write_text(wanted, encoding="utf-8")
            print(f"wrote CausalDiagnosis/cluster_jobs/grids/{name}, {sum(1 for line in wanted.splitlines() if not line.startswith('#'))} run lines")
        return 0
    if command == "presets" and len(argv) == 2:
        print(*grid_files.presets(argv[1]))
        return 0
    if command == "policies":
        print(*run_lines.REFERENCE_AGENTS)
        return 0
    if command == "wiring":
        print(*pending_lines.WIRING_AGENTS)
        return 0
    if command == "share" and len(argv) == 3:
        # the replicates and the option of a cut look that run_grid.sbatch plays, from GRID_FRACTION and GRID_REPLICATES
        played, option = pending_lines.play_plan(argv[1] or None, [int(r) for r in argv[2].split()])
        print(*played)
        print(option)
        return 0
    if command in ("clocked", "unclocked") and len(argv) == 2:
        return show(grid_files.by_clock(argv[1])[command == "clocked"])
    if command == "fillers" and len(argv) == 2:
        return show([filler_arguments(line) for line in grid_lines(argv[1])])
    if command == "agents" and len(argv) > 1:
        roots, model = argv[1:], None
        if roots[0] == "--model" and len(roots) > 2:
            roots, model = roots[2:], roots[1]
        return show(grid_files.agent_runs(roots, model))
    if command == "second_round" and len(argv) == 3:
        header = (f"the second round of sampling for {argv[1]} on fold {argv[2]}, the next samples of the training "
                  "configuration, played with GRID_STARTS=training after a GATE line of rejection_sampling_examples.py says second round due, written by "
                  "CausalDiagnosis/experiments/cluster_runs/grid_job_lists.py and never by hand")
        print(grid_files.text(header, run_lines.second_round(argv[1], int(argv[2]))), end="")
        return 0
    if command == "folds" and len(argv) == 1:
        print(*training.held_out_folds.fold_ids())
        return 0
    if command == "part" and len(argv) == 2:
        if argv[1] not in run_lines.PARTS:
            print(f"no part {argv[1]}; the parts are {', '.join(run_lines.PARTS)}", file=sys.stderr)
            return 2
        print(grid_files.text(f"the part {argv[1]} of the run grid, {run_lines.replicates(argv[1])} run(s) per line, written by "
                   "CausalDiagnosis/experiments/cluster_runs/grid_job_lists.py and never by hand", run_lines.part(argv[1])), end="")
        return 0
    if command == "references" and len(argv) == 2:
        if argv[1] not in run_lines.PARTS:
            print(f"no part {argv[1]}; the parts are {', '.join(run_lines.PARTS)}", file=sys.stderr)
            return 2
        print(grid_files.text(f"the reference runs beside the part {argv[1]} of the run grid, {run_lines.replicates(argv[1])} run(s) per line, written by "
                   "CausalDiagnosis/experiments/cluster_runs/grid_job_lists.py and never by hand", grid_files.reference_lines(run_lines.part(argv[1]))), end="")
        return 0
    if command == "row" and len(argv) == 2:
        if argv[1] not in design.ROWS_BY_ID:
            print(f"no row {argv[1]}; the rows are {', '.join(design.ROWS_BY_ID)}", file=sys.stderr)
            return 2
        print(grid_files.text(f"one development row, {argv[1]} on the preset of the main table, written by CausalDiagnosis/experiments/cluster_runs/grid_job_lists.py "
                   "and never by hand", run_lines.row_grid(argv[1])), end="")
        return 0
    if command != "check":
        raise SystemExit(__doc__)
    stale = grid_files.differing()
    for name in stale:
        print(f"CausalDiagnosis/cluster_jobs/grids/{name} differs from the design: python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists write")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
