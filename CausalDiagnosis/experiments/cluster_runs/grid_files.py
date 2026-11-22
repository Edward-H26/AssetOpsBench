from CausalDiagnosis.utils.paths import GRIDS, ROOT, SRC
import sys
import textwrap
from pathlib import Path
from . import run_lines


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC)]
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import training_variants as training  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs.pending_grid_lines import grid_lines  # noqa: E402


HPC = GRIDS
FILES = ("grid_agents.txt", "grid_pilot.txt", "grid_gate.txt", "grid_training_sampling.txt", "grid_training_evaluation.txt")


def text(header, entries):
    """A grid file: the header as comment lines, then every run line after the comment that names what it serves."""
    lines = [f"# {line}" for line in textwrap.wrap(header, 118)]
    for comment, line in entries:
        lines += [f"# {comment}", line]
    return "\n".join(lines) + "\n"


def files():
    """{file name: text} of the five grid files of CausalDiagnosis/cluster_jobs/."""
    tokens = run_lines.environment_token() + run_lines.layers_token()
    last = run_lines.misdiagnosis_price() + run_lines.environment() + run_lines.follower()
    faulty = run_lines.agents() + run_lines.companion() + run_lines.ladder() + run_lines.ablations() + tokens + last
    main = run_lines.main_part() + run_lines.companion() + run_lines.ladder() + run_lines.ablations() + tokens + last
    episodes = (len(faulty) * run_lines.STARTS + len(run_lines.healthy_control()) * run_lines.HEALTHY_STARTS) * run_lines.REPLICATES
    written = "written by CausalDiagnosis/experiments/cluster_runs/grid_job_lists.py and never by hand"
    development = ("played on the development starts alone, GRID_STARTS=development and GRID_SEEDS=1 with GRID_EXTRA=--start-cap=N, N the "
                   "count that cluster_run_checks.py cap prints, the registered share of the development starts, never on a start of the test set")
    return {
        "grid_agents.txt": text(
            f"The run grid of the main table, of the cumulative layer study and of the supporting ablations of the paper, {written}, from "
            f"experiment_tables.py: {len(main)} run lines, one run per line and replicate of GRID_REPLICATES, {run_lines.REPLICATES} replicates, {run_lines.STARTS} "
            f"episodes per run on a preset with faulty starts and {run_lines.HEALTHY_STARTS} on the healthy control ({episodes} episodes), played "
            "through run_grid.sbatch with GRID_STARTS=sampled, the test set. Every run line follows one comment line with the row or the "
            f"ablation it serves. Lines 1 to {len(run_lines.agents())} are the main table, whose runs the rescoring of En4 scores again, and the "
            f"next {len(run_lines.healthy_control())} the healthy control on the start with no faulty component. Then the arm of En9 that stands "
            "beside the main table. The cumulative layer study plays the three rows with the CausalDiagnosis MCP: its first layer "
            "is their lines of the main table and its last their lines in the environment at the end of this file. After its middle "
            "layers, three lines play the rows without the MCP on all six big layers, the baselines of the training table and the "
            "references of h1_top, h2_prime and the arms of layers 2, 3 and 5. A supporting ablation plays where "
            "design.ABLATION_PLACES puts it, after the reference line of each of its rows that no other line plays on its preset, and "
            "an arm that is an option of layers 2, 3 and 5 plays a preset of its own, all six big layers with one setting "
            f"changed, against the preset of training. Then {len(tokens)} lines play the six rows on all six big layers and the "
            "layer study on the rungs with a clock under the clock that charges the tokens the agent's model generated in place of "
            f"the wall seconds, a second condition of the clock, then {len(run_lines.misdiagnosis_price())} lines the ablation L1, the misdiagnosis "
            f"price at the budget, then {len(run_lines.environment())} lines the three rows with the CausalDiagnosis MCP on all six big "
            f"layers as released, and the last {len(run_lines.follower())} lines the MCP follower on every layer, a policy without a model whose "
            "runs enter the table of the reference policies and no other table that selects agents, each part after every other line, so "
            "that adding it left the place of "
            "every earlier line unchanged. "
            "Every line of this file plays on a server without LoRA adapters, the server of the rows that the agents table compares, and "
            "the part environment is the row of the agents table that grid_training_evaluation.txt plays again beside the adapters.",
            main),
        "grid_pilot.txt": text(
            f"The optional pilot that measures the thinking rate against think_seconds_per_cycle of the asset file, which every run of "
            f"record keeps (CausalDiagnosis/docs/environment.md), {written}: the "
            f"three agents with the CausalDiagnosis MCP on the full environment, {development}.",
            run_lines.pilot()),
        "grid_gate.txt": text(
            f"The development gate, {written}: the CausalDiagnosis agent and the paper-era Plan-and-Execute, both without "
            f"the CausalDiagnosis MCP, on layer 0, {development}. cluster_run_checks.py gate compares the two runs.",
            run_lines.gate()),
        "grid_training_sampling.txt": text(
            f"The sampling runs of the fine tuning of the training table, {written}, from training_variants.py and "
            f"{training.CONFIG_FILE.relative_to(ROOT).as_posix()}: "
            f"{len(run_lines.sampling())} run lines, one run each, played with GRID_STARTS=training and without a judge, since the filter is the "
            "verifier's pass. A run plays the training start states of the model of its fold once.",
            run_lines.sampling()),
        "grid_training_evaluation.txt": text(
            f"The evaluation runs of the training table, {written}, from training_variants.py and experiment_tables.py: {len(run_lines.evaluation())} run lines on "
            "the test set, GRID_STARTS=sampled, one run per line and replicate, on a server started with SERVE_ADAPTERS=1 and without a "
            "judge, which runs after the grid over the folders of these runs in a serving job of the judge alone (run_judge.sh), as "
            "for every other grid file. An agent without training plays every start of a replicate, and the model of a fold plays the "
            "starts of the fold it holds out. The lines without training serve the training table alone, whose comparisons read every "
            "variant on this server; the agents table reads the same rows from the part environment of grid_agents.txt, on a server "
            "without the adapters.",
            run_lines.evaluation()),
    }


def presets(path):
    """The presets that the lines of a grid file name, each once in file order, with its clock mode after a colon when a line
    overrides it, the form of GRID_REFERENCE_CONFIGS of CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch."""
    return presets_of(grid_lines(path))


def presets_of(lines):
    """The presets that run lines name, each once in order, with its clock mode after a colon when a line overrides it."""
    found = []
    for line in lines:
        words = line.split()
        entry = words[words.index("--config") + 1]
        if "--clock-mode" in words:
            entry += f":{words[words.index('--clock-mode') + 1]}"
        if entry not in found:
            found.append(entry)
    return found


def by_clock(path):
    """{True: the lines of a grid file whose run charges the wall seconds of thinking, False: the others}, in file order. The
    first play on the node and at design.RUNS_IN_FLIGHT, and the others on any node."""
    split = {True: [], False: []}
    for line in grid_lines(path):
        split[pending_lines.expected(line)["clock_mode"] == run_lines.WALL_CLOCK].append(line)
    return split


def agent_runs(roots, model=None):
    """The finished runs of the six rows of the agents table under the roots, every replicate, which the rescoring of En4 scores again.
    A run is found by what it played, as pending_grid_lines.py matches a run to a line, so the file its job read does not matter, and on
    model, the model its job served (pending_lines.expected). A run that rescore_under_verifier_scope.py --own-scope scored again under the roots
    gives way to its copy, which the tables read in its place."""
    wanted = [pending_lines.expected(f"{line} --replicate {r}", model=model) for _comment, line in run_lines.agents() for r in pending_lines.REPLICATES]
    found = {folder: meta for meta, folder in pending_lines.finished_runs(roots) if any(pending_lines.matches(meta, fields, {}) for fields in wanted)}
    replaced = {meta["rescored"]["from"] for meta in found.values() if pooling_rules.own_scope_copy(meta)}
    return sorted(folder for folder in found if folder.name not in replaced)


def differing():
    """The grid files of CausalDiagnosis/cluster_jobs/ that are missing or differ from what the design implies."""
    return [name for name, wanted in files().items() if not (HPC / name).exists() or (HPC / name).read_text(encoding="utf-8") != wanted]


def reference_lines(lines):
    """(comment, line) of the reference runs beside run lines: every agent of REFERENCE_AGENTS on every preset that the lines name,
    with the clock mode a line sets, the presets in order and the agents in the order of experiment_tables.py, the runs that
    CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch plays from the commands presets and policies."""
    out = []
    for entry in presets_of([line for _, line in lines]):
        preset, _, mode = entry.partition(":")
        setting = design.LADDER_NAMES[design.LADDER.index(preset)] if preset in design.LADDER else preset
        for agent in run_lines.REFERENCE_AGENTS:
            out.append((f"reference run, {agent}, {setting}" + (f", clock {mode}" if mode else ""),
                        f"--config {preset}" + (f" --clock-mode {mode}" if mode else "") + f" --agent {agent}"))
    return out
