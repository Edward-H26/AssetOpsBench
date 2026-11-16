"""The tables a grid reads, computed once before its lines start (CausalDiagnosis/cluster_jobs/README.md section 9): the models of the diagnostics
server, the separability label table of every rules key the lines play (CausalDiagnosis/environment/reference_policies/diagnosability_label.py) and the E18
grade table of every rules key of their released rules (CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/diagnosability_grade.py), for the lines of the grid
files, for the named presets and for the asset-default EpisodeConfig(), which the gym and the OpenEnv wrapper default to, so
that a fresh clone pays the computation once and no line of a grid computes a table at
first use while another line writes it. The tables job of the cluster (precompute_tables.sbatch) is the one process that
writes them on a clone; every other job, and a registered run, reads them, and run_grid.sbatch calls this with --check before
its first line, which writes nothing and stops the job on a missing table. A launch with its own start set or run options
(GRID_STARTS, the words of GRID_EXTRA such as --start-cap) passes them here, and every line and preset is resolved by the
driver's own functions, so a table has the rules key of the run that reads it.

A call computes the tables of one asset, that of --asset, over which every named preset is composed and which every line of
the grid files plays: the models of the diagnostics server where its model code can fit the asset, and the label table and
the grade table where the asset declares the label of diagnosability_label.py, which reads both. The label of another asset
computes the tables that its package declares on first use, by its own code.

    python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables --grid CausalDiagnosis/cluster_jobs/grids/grid_agents.txt --config real_time --config base
"""
from CausalDiagnosis.utils.paths import SRC
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC)]
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs.grid_files import presets as grid_presets  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs.pending_grid_lines import grid_lines  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET, EpisodeConfig  # noqa: E402

# the configuration the gym and the OpenEnv wrapper default to, computed after the grid lines and the presets whether or not
# one of them shares its rules key
ASSET_DEFAULTS = ("EpisodeConfig()", EpisodeConfig())
# the label of a start that reads the label table of diagnosability_label.py and the grade table of the diagnostics server, so that a
# call computes both for an asset that declares it
SEPARABILITY_LABEL = "CausalDiagnosis.environment.reference_policies.diagnosability_label:start_grade"


def asset_defaults(asset_id):
    """(name, configuration) of the defaults of an asset, which the gym and the OpenEnv wrapper default to: ASSET_DEFAULTS on
    the engine's default asset, the default configuration of the profile of any other."""
    if asset_id == DEFAULT_ASSET:
        return ASSET_DEFAULTS
    return (f"the defaults of {asset_id}", get_profile(asset_id).default_config())


def played(args, blocks):
    """The configuration that run_episodes/ plays for the command line args, through the driver's own functions: the preset
    with its overrides (configure), then the start set and the run options (played_config), so that a table computed here has
    the rules key of the run."""
    a = run_episodes.build_parser().parse_args(args)
    _name, config, _prices = run_episodes.configure(a)
    return run_episodes.played_config(a, config, blocks)


def preset_words(entry):
    """The words of a reference run of a preset entry, preset[:clock mode], as run_grid.sbatch starts it."""
    preset, _colon, mode = entry.partition(":")
    return ["--config", preset, *(["--clock-mode", mode] if mode else [])]


def configurations(grids, presets, starts=None, blocks=None, asset_id=None, run_options=()):
    """(name, configuration) of every line of the grid files and of every named preset, preset[:clock mode] as the reference
    policies play it, as run_episodes/ plays it on the start set starts with the run options of the launch (the words of
    GRID_EXTRA, such as --start-cap or --training-split): each command line is resolved by played(), the driver's own
    functions, so a line on the development starts with a start cap has the table of its run. The tables of a call are those
    of one asset, asset_id or else the engine's default asset, so a line or a preset that plays another asset stops by name."""
    asset = asset_id or DEFAULT_ASSET
    tail = [*(["--asset", asset_id] if asset_id else []), *(["--starts", starts] if starts else []), *run_options]

    def resolved(name, words):
        if "--agent" not in words:
            words = [*words, "--agent", "plan_execute"]
        config = played([*words, *tail], blocks)
        if config.asset_id != asset:
            raise SystemExit(f"{name} plays the asset {config.asset_id}, and this call computes the tables of the asset {asset} "
                             "(--asset): compute the tables of each asset by a call of its own")
        return name, config

    found = [resolved(f"{grid}: {line}", line.split()) for grid in grids for line in grid_lines(grid)]
    return found + [resolved(entry, preset_words(entry)) for entry in presets]


def problems(found, cycles, blocks, facts, asset, labels_path, grades_path):
    """What --check finds for the configurations found: the models file of the asset that is missing or of other code, and for
    every configuration a label table and a grade table that is missing, of other code, or short of a recorded combination,
    each with the name of its line. Reads the files and writes nothing."""
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit, diagnosability_grade as grade
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models

    models_path = fit.models_path(asset)
    found_problems = [f"models: {reason}" for reason in fit.models_problems(models_path, asset, facts)]
    models = None if found_problems else load_models(models_path)
    for name, config in found:
        if separability.current_label_table(config, getattr(cycles, "facts_sha256", None), labels_path) is None:
            found_problems.append(f"labels {separability.rules_key(separability.rules_of(config))[:12]}: missing or of other code, for {name}")
        if models is not None:
            rules = HydraulicRigEnv(cycles, blocks, config).public_profile()
            found_problems += [f"{reason}, for {name}" for reason in grade.table_problems(models, rules, path=grades_path, facts_path=facts)]
    return list(dict.fromkeys(found_problems))


def main(argv=None):
    ap = argparse.ArgumentParser(description="compute the tables a grid reads before its lines start")
    ap.add_argument("--grid", action="append", default=[], help="a grid file of run lines, as run_grid.sbatch reads it (repeatable)")
    ap.add_argument("--config", action="append", default=[],
                    help="a preset of configs/ with an optional clock mode after a colon, preset[:mode], as the reference policies play it (repeatable)")
    ap.add_argument("--asset", default=None,
                    help=f"the registered asset whose tables the call computes, over whose defaults every preset is composed and "
                         f"which every line plays (default: {DEFAULT_ASSET})")
    ap.add_argument("--facts", default=None,
                    help="the data file of the asset (default: the file that the variable of its asset file names, else its release file)")
    ap.add_argument("--labels", type=Path, default=None, help="the label table file (default: the one the verifier reads)")
    ap.add_argument("--grades", type=Path, default=None, help="the grade table file (default: the one the drivers read)")
    ap.add_argument("--grid-presets", action="store_true",
                    help="add the presets that the lines of each --grid file name, as the reference runs of its grid job play them")
    ap.add_argument("--check", action="store_true",
                    help="write nothing: print every table the launch needs that is missing or of other code and exit 3, else exit 0")
    ap.add_argument("--starts", choices=run_episodes.START_SETS, default=None,
                    help="the start set the lines play, GRID_STARTS of the grid job: development resolves every configuration to the development starts")
    # every other option is a run option of the launch, the words of GRID_EXTRA, which each line and preset plays with
    a, run_options = ap.parse_known_args(argv)
    from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit, diagnosability_grade as grade
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models

    asset = a.asset or DEFAULT_ASSET
    facts, cycles, blocks = load_world(a.facts, asset)
    presets = list(dict.fromkeys([*a.config, *(entry for grid in a.grid for entry in grid_presets(grid) if a.grid_presets)]))
    found = configurations(a.grid, presets, a.starts, blocks, a.asset, run_options) + [asset_defaults(asset)]
    report = {"labels": [], "grades": []}
    if a.check:
        # the check of a launch before its first line: reads the files, writes nothing, and exit status 3 names what is missing
        if ports.declared(asset, "LABELS") != SEPARABILITY_LABEL:
            print(f"TABLES_READY no label table and no grade table for {asset}, whose label is not the label of separability.py", flush=True)
            return report
        found_problems = problems(found, cycles, blocks, facts, asset, a.labels or separability.LABELS_PATH, a.grades or grade.GRADES_PATH)
        for line in found_problems:
            print(f"TABLES_MISSING {line}", flush=True)
        if found_problems:
            raise SystemExit(3)
        print(f"TABLES_READY {len(found)} configurations of {asset}", flush=True)
        return report
    # the models where the model code of the server can fit the asset, fitted as the driver fits them for a row with the server
    run_episodes.diagnostics_models(asset, (), blocks, facts, fit_models=True)
    unfit = fit.unfit_reasons(asset, set(blocks))
    if ports.declared(asset, "LABELS") != SEPARABILITY_LABEL:
        print(f"no label table and no grade table for {asset}, whose label is not the label of diagnosability_label.py; models of the "
              f"diagnostics server: {'fitted' if not unfit else 'none, since ' + '; '.join(unfit)}", flush=True)
        return report
    if unfit:
        raise SystemExit(f"the asset {asset} declares the label of diagnosability_label.py, which nests the grade of the diagnostics server, "
                         f"and the model code of the server fits none for it: {'; '.join(unfit)}")
    models = load_models(fit.models_path(asset))
    labels_path = a.labels or separability.LABELS_PATH
    grades_path = a.grades or grade.GRADES_PATH
    for name, config in found:
        table = separability.ensure_labels(config, cycles, blocks, labels_path)
        if table["rules_key"] not in report["labels"]:
            report["labels"].append(table["rules_key"])
            print(f"labels {table['rules_key'][:12]}, {len(table['labels'])} starts, for {name}", flush=True)
        # the released rules of an episode of the configuration, which the drivers hand to the grade table
        grades = grade.ensure_grades(models, HydraulicRigEnv(cycles, blocks, config).public_profile(), path=grades_path, facts_path=facts)
        if grades["rules_key"] not in report["grades"]:
            graded = [grades["grades"][key] for key in grades["grades"]]
            report["grades"].append(grades["rules_key"])
            print(f"grades {grades['rules_key'][:12]}, {len(graded)} combinations, for {name}", flush=True)
    return report


if __name__ == "__main__":
    main()
