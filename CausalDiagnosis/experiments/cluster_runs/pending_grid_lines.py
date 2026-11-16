"""The runs of a grid file that still lack a finished run, for a job that continues a grid after an allocation ended.

On the test set, the start set that run_grid.sbatch plays by default, a line of the grid is one run per replicate (--replicate of
run_episodes/), so the unit is the pair of a line and a replicate, and a line of another start set is one run.
A pair counts as done when a run folder under one of the given roots carries run.json (written only when a run ends), its
episodes are complete (one row of scores.csv per planned episode), its configuration, agent and switches equal the line's
(as run_episodes/ resolves them, under the start set of --starts), on the test set also its replicate, the rule of its
replicates, its master seed and its seed pool, under every start set its training variant and its fold and on the training
starts its training sample, inside a serving job its serving setup that of the job (the LoRA adapters of HYD_ENDPOINT_FILE or of
ENDPOINT_FILE, so that the lines without training of the part training never count as runs of the part environment), and it
started at or after the cutoff given for its agent (--since agent=ISO time), so that runs of code that a later fix replaced do
not count. The pending pairs are printed in replicate order, the first
replicate of every line first, each as the line with its --replicate and one per line, ready for run_grid.sbatch's GRID_FILE,
which plays a line that names its replicate once. A pending pair that a run in progress plays, a run that a job left partly
played, is printed with --resume and the run folder, so that the job that plays it takes the run up from its last finished
episode (run_episodes/ --resume).

A grid line names no model: the job adds the model it serves after the line (run_grid.sbatch), so --model names that model, which
a finished run of a model agent records, and run_lines.sh passes the served name of its job.

    python -m CausalDiagnosis.experiments.cluster_runs.pending_grid_lines --grid <grid file> --runs runs/<run folder> [--starts sampled] [--fraction 0.05 | --replicates 1 2 3 4 5] --since <agent>=<ISO time> [--model <served name>]

On the test set the share of --fraction (start_states.DEFAULT_FRACTION when neither option is given) names the replicates: a
whole share r/5 the replicates 1 to r, a cut look below one replicate the first, each unit with --fraction (play_plan).
"""
from CausalDiagnosis.utils.paths import SRC
import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC)]
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import DEFAULT_FRACTION, REPLICATES as REPLICATE_COUNT, REPLICATE_RULE  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import WIRING_AGENTS, fraction_plan, fraction_text, fraction_value  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import fold_line_empty  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import is_marker  # noqa: E402


# what a run.json without the key played, the default of its switch
ABSENT = {**{key: run_episodes.build_parser().get_default(key) for key in ("abstention_rule", "training_variant",
                                                                             "training_fold")}, "lora_adapters": 0, "fraction": None}
DEFAULT_STARTS = "sampled"  # the start set of the grid, GRID_STARTS of run_grid.sbatch, which plays the test set unless told otherwise
REPLICATES = tuple(range(1, REPLICATE_COUNT + 1))  # the replicates of the test set, GRID_REPLICATES of run_grid.sbatch


def grid_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def play_plan(fraction=None, replicates=None):
    """(replicates, option) of a grid on the test set: the replicates its lines play and the option that every unit carries, from
    the share (DEFAULT_FRACTION when neither is given) or the whole replicates named by hand. A whole share r/5 plays the
    replicates 1 to r with no option, a cut look the first with --fraction, and both given is refused: the one rule of this
    script and of run_grid.sbatch (grid_job_lists.py share)."""
    if fraction is not None and replicates:
        raise SystemExit("a share and replicates both name the part of the test set to play (GRID_FRACTION and GRID_REPLICATES): set one")
    if replicates:
        outside = [r for r in replicates if r not in REPLICATES]
        if outside:
            raise SystemExit(f"the test set holds the replicates {REPLICATES[0]} to {REPLICATES[-1]}, not {outside[0]}")
        return tuple(replicates), ""
    try:
        kind, value = fraction_plan(DEFAULT_FRACTION if fraction is None else fraction)
    except ValueError as e:
        raise SystemExit(f"--fraction: {e}") from None
    if kind == "whole":
        return REPLICATES[:value], ""
    return REPLICATES[:1], f"--fraction {fraction_text(value)}"


def units(lines, starts=DEFAULT_STARTS, replicates=REPLICATES, option=""):
    """The runs that the lines of a grid play: on the test set every line once per replicate, as the line with its --replicate,
    in replicate order, so that the first replicate of every line comes first, a line of the wiring check on the first replicate
    of the test set alone, and a line that names its replicate, as this script prints the pairs, or a line of another start set
    once."""
    out = []
    for r in (replicates if starts == "sampled" else replicates[:1]):
        for line in lines:
            if starts == "sampled" and "--replicate" not in line.split():
                if r == REPLICATES[0] or not wiring_line(line):
                    out.append(f"{line} --replicate {r}" + (f" {option}" if option else ""))
            elif r == replicates[0]:
                out.append(line)
    return out


def wiring_line(line):
    """Whether a run line names an agent of the wiring check after --agent."""
    words = line.split()
    return "--agent" in words[:-1] and words[words.index("--agent") + 1] in WIRING_AGENTS


def expected(line, starts=DEFAULT_STARTS, model=None):
    """The run.json fields a finished run of the line carries under the start set of the grid, on the test set those of the
    replicate the line names, the first when it names none. model is the model the job serves, which run_grid.sbatch passes
    after the line, so it stands in for the model of the line of a model agent; None keeps the model of the line."""
    args = line.split()
    if "--agent" not in args:
        args += ["--agent", "plan_execute"]
    a = run_episodes.build_parser().parse_args(args + ["--starts", starts])
    config_name, config, price_table = run_episodes.configure(a)
    fields = {"config_name": config_name, "agent": a.agent, "clock_mode": config.clock_mode, "guidance_level": config.guidance_level,
              "diagnostics_server": config.diagnostics_server, "verifier_scope": config.verifier_scope, "ranking_model": a.ranking_model,
              "starts": starts, "prices": price_table["name"] if price_table else "default",
              "abstention_rule": a.abstention_rule,
              "abstention_cut": a.abstention_cut if a.abstention_rule == "fixed_cut" else None}
    if starts == "sampled":
        fields.update(replicate=run_episodes.replicate_of(a, config), replicate_rule=REPLICATE_RULE, master_seed=a.master_seed, seed_pool=a.seed_pool,
                      fraction=None if a.fraction is None else fraction_value(a.fraction))
    fields["training_variant"], fields["training_fold"] = a.training_variant, a.training_fold
    if starts == "training":  # the 34 sampling lines of an agent differ by their sample and their fold alone
        fields.update(training_sample=run_episodes.training_sample_of(a), master_seed=a.master_seed, seed_pool=a.seed_pool)
    if a.agent == "causal":
        fields["causal_options"] = run_episodes.causal_options(a)
    # the model the grid serves, the tool surface and the store of the AssetOpsBench servers
    fields["model"] = (model or a.model) if a.agent in run_episodes.MODEL_AGENTS else None
    if a.agent in run_episodes.MODEL_AGENTS and (setup := job_setup()) is not None:
        fields["lora_adapters"] = setup
    surface = a.servers or run_episodes.default_servers(a.agent, config_name)
    fields["surface"] = surface
    fields["backend_kind"] = a.store if surface in frameworks.surfaces() else None
    return fields


def job_setup():
    """The serving setup of the job that asks, the LoRA adapters on the line lora_adapters of the endpoint file of its server
    (HYD_ENDPOINT_FILE, else ENDPOINT_FILE of serve_model.sbatch), or None outside a serving job or for a file without the line."""
    fields = ledger.endpoint_file(os.environ.get("HYD_ENDPOINT_FILE") or os.environ.get("ENDPOINT_FILE")) or {}
    value = str(fields.get("lora_adapters", ""))
    return int(value) if value.isdigit() else None


IDENTITY_ERA_KEYS = ("model", "surface", "backend_kind")  # compared only on a run that records an identity


def observed(meta, key):
    """The field of a run.json that expected() names: the surface and the store are read from servers and backend."""
    if key == "surface":
        return pooling_rules.surface_of(meta)
    if key == "backend_kind":
        return (meta.get("backend") or {}).get("kind")
    return {**ABSENT, **meta}.get(key)


_IDENTITY = {}


def tree_identity(agent):
    """The identity of an agent in the tree that asks, once per agent and process (ledger.code_record)."""
    if agent not in _IDENTITY:
        _IDENTITY[agent] = ledger.code_record(agent)["identity"]
    return _IDENTITY[agent]


def finished_runs(roots):
    """(run.json, folder) of every finished and complete run folder under the roots."""
    for root in roots:
        for meta_path in sorted(Path(root).glob("*/run.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            scores = meta_path.parent / "scores.csv"
            # a filler run holds the runs in flight of its job on the development starts and covers no line
            if not meta.get("ended") or not scores.exists() or meta.get("filler"):
                continue
            with open(scores, newline="", encoding="utf-8") as f:
                rows = sum(1 for _ in csv.DictReader(f))
            # a run that planned no episode counts only as the marker of a fold without an adapter, which the code makes and no option asks for
            if rows == meta.get("episodes_planned") and (rows or is_marker(meta.get("adapter"))):
                yield meta, meta_path.parent


def matches(meta, fields, since):
    """Whether a finished run counts for a line: its fields equal the line's, it started at or after the cutoff of its agent, and
    its identity is that of its agent in the tree that asks, so a run of earlier code is pending again. A run written before the
    identity existed is compared on the fields of that time and its cutoff alone, as before."""
    identity = pooling_rules.identity_of(meta)
    if any(observed(meta, k) != v for k, v in fields.items() if identity is not None or k not in IDENTITY_ERA_KEYS):
        return False
    if identity is not None and identity != tree_identity(meta.get("agent")):
        return False
    cutoff = since.get(meta.get("agent"))
    return cutoff is None or (meta.get("started") or "") >= cutoff


def runs_in_progress(roots, starts=DEFAULT_STARTS):
    """(ledger, folder, fields) of every run in progress under the roots, a run that a job left partly played (ledger.in_progress),
    apart from a filler, which is never taken up, with the fields that expected() reads from the command its ledger recorded,
    None for a command that no longer parses."""
    for root in roots:
        for path in sorted(Path(root).glob(f"*/{ledger.LEDGER}")):
            if not ledger.in_progress(path.parent):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            argv = (record.get("command") or {}).get("argv") or []
            if "--filler" in argv:
                continue
            try:
                fields = expected(" ".join(argv), starts)
            except SystemExit:
                fields = None
            yield record, path.parent, fields


def resumable(record, recorded, fields, since):
    """Whether a run in progress, whose command gives the fields recorded, plays the pair of fields: the two are equal, it started
    at or after the cutoff of its agent, and every code file its ledger hashed is unchanged in the tree that asks."""
    cutoff = since.get(record.get("agent"))
    if recorded != fields or (cutoff is not None and (record.get("started") or "") < cutoff):
        return False
    return all(ledger.sha256_of(ledger.run_data.ROOT / path) == digest for path, digest in ((record.get("code") or {}).get("files") or {}).items())


def empty_cut(unit, starts=DEFAULT_STARTS):
    """Whether a unit is a fold line of a cut look whose fold receives no start combination, which plays no run and is not pending."""
    words = unit.split()
    if starts != "sampled" or "--fraction" not in words or "--training-fold" not in words:
        return False
    a = run_episodes.build_parser().parse_args(words + ["--starts", starts])
    return fold_line_empty(a.config, a.training_fold, a.fraction, a.master_seed, a.training_split)


def pending(lines, roots, since, starts=DEFAULT_STARTS, replicates=REPLICATES, model=None, option=""):
    runs = [meta for meta, _ in finished_runs(roots)]
    progress = list(runs_in_progress(roots, starts))
    out, taken = [], set()
    for unit in units(lines, starts, replicates, option):
        if empty_cut(unit, starts):
            continue
        fields = expected(unit, starts, model)  # once per run of the grid, which five replicates per line make many
        if any(matches(meta, fields, since) for meta in runs):
            continue
        # of the runs in progress that play the pair, the one with the most finished episodes, each taken up by one pair alone
        candidates = [(len(record.get("episodes") or {}), str(folder)) for record, folder, recorded in progress
                      if str(folder) not in taken and resumable(record, recorded, fields, since)]
        if candidates:
            folder = max(candidates)[1]
            taken.add(folder)
            out.append(f"{unit} --resume {folder}")
        else:
            out.append(unit)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", required=True, help="the grid file of run_grid.sbatch")
    ap.add_argument("--runs", nargs="+", required=True, help="folders that hold run folders, such as runs/grid_<job>")
    ap.add_argument("--since", nargs="*", default=[], help="agent=ISO time: runs of that agent count only from that start on")
    ap.add_argument("--starts", choices=run_episodes.START_SETS, default=DEFAULT_STARTS,
                    help="the start set the grid plays, GRID_STARTS of run_grid.sbatch: a run of another set does not count")
    ap.add_argument("--fraction", default=None,
                    help=f"the share of the test set the grid plays, GRID_FRACTION of run_grid.sbatch ({DEFAULT_FRACTION} when neither this nor "
                         "--replicates is given): a whole share of the replicates, or a cut look below one replicate")
    ap.add_argument("--replicates", type=int, nargs="+", choices=REPLICATES, default=None,
                    help="whole replicates of the test set named by hand in place of the share, GRID_REPLICATES of run_grid.sbatch")
    ap.add_argument("--model", default=None,
                    help="the model the job serves, SERVED_NAME of the serving job, which run_grid.sbatch adds after every line of a model agent")
    a = ap.parse_args(argv)
    since = dict(item.split("=", 1) for item in a.since)
    replicates, option = play_plan(a.fraction, a.replicates)
    for unit in pending(grid_lines(a.grid), a.runs, since, a.starts, replicates, a.model, option):
        print(unit)


if __name__ == "__main__":
    main(sys.argv[1:])
