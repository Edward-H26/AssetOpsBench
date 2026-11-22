"""The one command entry of CausalDiagnosis: python -m CausalDiagnosis <group> <command> [arguments].

A group is a top-level folder of CausalDiagnosis/, and a command is the file name of the program it starts, with - for _.
Two groups are not folders: tables, for the result tables, and one group per framework under frameworks/, which the
framework declares. The table only dispatches. It imports nothing until a command is chosen, then starts the program's
module as python -m <module> would, with the arguments after the command, so every option and subcommand of the program
works as before and a run records the same command whichever way it was started. Each program keeps its main in its own
module, and the direct form python -m CausalDiagnosis.<package>.<module> keeps working.

    python -m CausalDiagnosis                    the groups
    python -m CausalDiagnosis <group>            the commands of a group, one line each
    python -m CausalDiagnosis <group> <command> --help
"""
import sys

P = "CausalDiagnosis."
COMMANDS = {
    "datasets": {
        "download": ("datasets.<asset_id>.download_and_build",
                     "fetch the raw data of a dataset and build its data file (the dataset id first, such as hyd_1)"),
        "new-dataset-scaffold": ("datasets.new_dataset_scaffold", "write a first asset.json for a new dataset of recorded levels"),
        "build-features-and-blocks": ("datasets.hyd_1.build_features_and_blocks",
                                      "recompute the per-cycle features and stable blocks of hyd_1 from its raw files"),
        "verify-data-facts": ("datasets.hyd_1.verify_data_facts", "check the data facts of docs/environment.md on the hyd_1 files"),
    },
    "environment": {
        "diagnosability-label": ("environment.reference_policies.diagnosability_label",
                                 "fill the diagnosability label of every start with the greedy reference policy"),
    },
    "mcp_servers": {
        "fit-models": ("mcp_servers.Causal_Diagnosis_MCP.offline.fit_models", "fit models.json of the CausalDiagnosis MCP"),
        "diagnosability-grade": ("mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade",
                                 "grade every recorded fault combination under a budget (grades.json)"),
        "offline-table": ("mcp_servers.Causal_Diagnosis_MCP.offline.offline_table", "the offline table of the CausalDiagnosis MCP"),
        "calibration-report": ("mcp_servers.Causal_Diagnosis_MCP.offline.calibration_report",
                               "the calibration report of the ranking model"),
    },
    "agents": {
        "model-adapters": ("agents.utils.model_adapters", "rewrite the chat template of the served model before serving"),
    },
    "experiments": {
        "run-episodes": ("experiments.episode_runners.run_episodes",
                         "play episodes with an agent or a reference policy and verify each one"),
        "run-reference-policies": ("experiments.episode_runners.run_reference_policies",
                                   "play the reference policies in process, without the servers"),
        "grid-job-lists": ("experiments.cluster_runs.grid_job_lists", "write, check and query the job lists of the tables"),
        "precompute-grid-tables": ("experiments.cluster_runs.precompute_grid_tables",
                                   "compute the tables a job list reads before its lines start"),
        "pending-grid-lines": ("experiments.cluster_runs.pending_grid_lines", "the lines of a job list that lack a finished run"),
        "cluster-run-checks": ("experiments.cluster_runs.cluster_run_checks", "the checks of the cluster run order"),
        "write-training-split": ("experiments.table_definitions.write_training_split",
                                 "write the registered training split of a dataset"),
        "run-ledger": ("experiments.run_verification.run_ledger", "the ledger of a run folder, the platform record, the record check and the serving file of record"),
        "replay-check": ("experiments.run_verification.replay_check", "replay a finished run on a clone of its commit and certify it"),
        "verify-results-folder": ("experiments.run_verification.verify_results_folder",
                                  "check that the tables of a results folder follow from their runs"),
        "recompute-verdicts": ("experiments.run_verification.recompute_verdicts",
                               "recompute the verdict of every episode under a runs folder"),
        "rescore-under-verifier-scope": ("experiments.run_verification.rescore_under_verifier_scope",
                                         "score finished runs again under the verifier scope all"),
        "invariant-digests": ("experiments.run_verification.invariant_digests",
                              "digests of every text an agent reads and of the invariants of the environment"),
        "check-grid-complete": ("experiments.run_verification.check_grid_complete", "check that a finished job list ran in full"),
        "check-synced-runs": ("experiments.run_verification.check_synced_runs",
                              "the laptop side of a sync of run folders from the cluster"),
        "upload-manifest": ("experiments.run_verification.upload_manifest",
                            "the manifest of an upload to the cluster and the commit it matches"),
        "publish-check": ("experiments.run_verification.publish_check",
                          "check run folders and results folders before they go to the branch of the runs"),
    },
    "evaluation": {
        "leaderboard": ("evaluation.result_tables.leaderboard", "the leaderboard page"),
        "difficulty-table": ("evaluation.result_tables.difficulty_table", "the difficulty table of the datasheet"),
        "datasheet-statistics": ("evaluation.result_tables.datasheet_statistics", "the counts and statistics of docs/dataset.md"),
        "publish-runs": ("evaluation.result_tables.publish_runs", "copy the readable part of the runs the tables read into one folder"),
    },
    "training": {
        "rejection-sampling-examples": ("training.rejection_sampling_examples",
                                        "select the good episodes and write the fine-tuning examples"),
        "train-lora": ("training.train_lora", "LoRA fine-tuning of the agent model"),
        "gepa-prompt-optimization": ("training.gepa_prompt_optimization", "the online GEPA loop over one block of prompt text"),
    },
    "episode_viewer": {
        "walkthrough": ("episode_viewer.walkthrough", "the illustrated walkthrough of one episode"),
        "step-through-page": ("episode_viewer.step_through_page", "the step-through page of one episode, one slide per tool call"),
        "run-review-pack": ("episode_viewer.run_review_pack", "the review pack of a run folder and the fixed starts of the walkthroughs"),
    },
    "examples": {
        "run-one-episode": ("examples.run_one_episode", "play one episode of the oracle and of the do-nothing policy, no model needed"),
    },
    "utils": {
        "paths": ("utils.paths", "print a location of the repository by its name"),
    },
    "performance": {
        "profile-table": ("performance.profile_table", "the stage profile of the episodes of run folders, per episode, cell and job"),
        "tracking-export": ("performance.tracking_export",
                            "send finished run folders and training folders to Weights and Biases, after a dry run that prints it all"),
    },
}
TABLES = P + "evaluation.result_tables.tables_from_runs"
TABLES_HELP = (("list", "the name of every result table and what varies in it"),
               ("<table name>", "build one table alone into a folder outside results/ (--runs, --out)"),
               ("all", "build every table, the index and the pages"))


def framework_commands():
    """{group: {command: module}} of the frameworks, read from their declarations without importing a framework."""
    from CausalDiagnosis.utils import frameworks
    return frameworks.commands()


def first_sentence(module):
    """The first sentence of a module's docstring, read from its file without running it."""
    import ast
    import importlib.util
    spec = importlib.util.find_spec(module)
    path = spec.origin if spec.submodule_search_locations is None else spec.submodule_search_locations[0] + "/__main__.py"
    with open(path, encoding="utf-8") as f:
        doc = ast.get_docstring(ast.parse(f.read())) or ""
    return " ".join(doc.split(". ")[0].split())


def usage(lines, status):
    print("\n".join(lines), file=sys.stderr if status else sys.stdout)
    return status


def list_groups(status):
    groups = [*COMMANDS, "tables", *framework_commands()]
    return usage(["usage: python -m CausalDiagnosis <group> <command> [arguments]", "", "groups:",
                  *(f"  {group}" for group in groups), "", "python -m CausalDiagnosis <group> lists the commands of a group"],
                 status)


def list_commands(group, rows, status):
    width = max(len(name) for name, _ in rows)
    return usage([f"usage: python -m CausalDiagnosis {group} <command> [arguments]", "", "commands:",
                  *(f"  {name.ljust(width)}  {line}" for name, line in rows)], status)


def start(module, argv):
    """Start a module as python -m <module> [argv] starts it: sys.argv[0] is the module's file, then the arguments."""
    import importlib.util
    import runpy
    if importlib.util.find_spec(module) is None:
        return usage([f"no module {module}"], 2)
    sys.argv = [sys.argv[0], *argv]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        return list_groups(0 if argv else 2)
    group, rest = argv[0], argv[1:]
    asked = not rest or rest[0] in ("-h", "--help")
    if group == "tables":
        if asked:
            return list_commands(group, TABLES_HELP, 0 if rest else 2)
        command, rest = rest[0], rest[1:]
        flags = {"all": [], "list": ["--list"]}.get(command, ["--table", command])
        return start(TABLES, [*flags, *rest])
    if group in COMMANDS:
        commands = {name: (P + module, line) for name, (module, line) in COMMANDS[group].items()}
    else:
        declared = framework_commands()
        if group not in declared:
            return list_groups(2)
        commands = {name: (module, None) for name, module in declared[group].items()}
    if asked:
        rows = [(name, line if line is not None else first_sentence(module)) for name, (module, line) in commands.items()]
        return list_commands(group, rows, 0 if rest else 2)
    command, rest = rest[0], rest[1:]
    if command not in commands:
        return usage([f"{group} has no command {command!r}; python -m CausalDiagnosis {group} lists them"], 2)
    module = commands[command][0]
    if "<asset_id>" in module:
        if not rest or not rest[0].isidentifier():
            return usage([f"usage: python -m CausalDiagnosis {group} {command} <asset_id> [arguments]"], 2)
        module, rest = module.replace("<asset_id>", rest[0]), rest[1:]
    return start(module, rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
