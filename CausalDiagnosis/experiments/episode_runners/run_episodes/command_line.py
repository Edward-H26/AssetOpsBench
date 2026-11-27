import argparse
import dataclasses
import os
import sys
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import RANKING_MODELS  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import apply_price_table, load_config, load_price_table, load_world  # noqa: E402
from CausalDiagnosis.utils.csv_rows import write_rows  # noqa: E402
from CausalDiagnosis.environment.reference_policies.tier_classifiers import PROFILE as CLASSIFIED_ASSET  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import CLOCK_MODES  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import STOP_AFTER_FAILURES  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import scenario_set  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import AS_RELEASED, REGISTERED, SPLITS, TRAINING_VARIANTS, is_marker  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES  # noqa: E402
from . import agent_kinds, static_agent, run_record, play_one_episode, planned_episodes, start_check, runner_setup


FILLER_STARTS = "--filler plays the development starts, so it needs --starts development"
FILLER_JUDGE = "--filler judges nothing, so it needs --judge none"
FILLER_OUT = "--filler writes beside the grid folders and never into one, and --out lies in"
RESUME_FILLER = "--resume takes up a run that a job left partly played, and a filler never resumes"


def scenario_id_list(text):
    """The ids of --scenario-ids, one or several joined by /, one shell word that a grid line and GRID_EXTRA carry as it is, as
    sorted distinct integers, so that two spellings of one set of scenarios record one value."""
    words = text.split("/")
    if not all(word.isdigit() for word in words):
        raise argparse.ArgumentTypeError(f"{text!r} is no scenario id and no ids joined by /, such as 2001/2003")
    return sorted({int(word) for word in words})


def build_parser():
    ap = argparse.ArgumentParser(description="agents and reference policies on hydraulic rig episodes through the MCP servers")
    ap.add_argument("--config", default="static", help="a preset of CausalDiagnosis/configs/ or a JSON path")
    ap.add_argument("--asset", default=None,
                    help="the registered asset the episodes play, over whose defaults the preset is composed (default: the asset the "
                         f"preset names, {DEFAULT_ASSET} when it names none)")
    ap.add_argument("--agent", choices=agent_kinds.AGENTS, default="scripted")
    ap.add_argument("--model", default=agent_kinds.DEFAULT_MODEL, help="the model name the model agents send to the endpoint")
    ap.add_argument("--base-url", default=agent_kinds.ROUTER_BASE_URL, help="OpenAI compatible endpoint of the model agents (default: TokenRouter)")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (per scenario for the anchored preset)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--starts", choices=start_check.START_SETS, default="seeds",
                    help="seeds plays the seeds from --first-seed as they are; combinations plays every start combination the "
                         "config allows under its --seeds smallest seeds, the same seeds in every line of a grid; sampled plays "
                         "every combination once at the start state of one replicate of the evaluation set, --replicate; training plays "
                         "every training start state of the model of --training-fold once under the seed of sample --training-sample, at "
                         "the sampling temperature; development plays every development start under its --seeds smallest seeds")
    ap.add_argument("--start-cap", type=int, default=None,
                    help="with --starts development: play this many of the development starts, drawn by the registered seed of the development "
                         "split so that every faulty level of a component that they hold is covered")
    ap.add_argument("--start-state", default=None, metavar="LEVELS@OFFSET",
                    help="with --starts seeds: play every seed at this start state, as a preset whose field start names it plays: the level "
                         "of each component of the asset in their order joined by /, then @ and the entry offset (3/100/0/130@2 on the rig), "
                         "refused when the fault rule of the preset does not allow the combination or the data hold no recording of it at "
                         "that entry offset, and, as the field start of a preset, not held to the allowed starts of the preset; run.json and "
                         "the ledger record it, and no cell of a table reads such a run")
    ap.add_argument("--scenario-ids", type=scenario_id_list, default=None, metavar="ID/ID",
                    help="with the anchored preset: play the scenarios of these ids alone, one id or several joined by /, refused for an id "
                         "that the scenario file does not hold; run.json and the ledger record them, and no cell of a table reads such a run")
    ap.add_argument("--filler", action="store_true",
                    help="a filler run of a grid job, which plays the development starts in the slot of a finished run while a run "
                         "on the wall clock still plays, so that the job keeps its runs in flight (CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch); run.json "
                         "records filler and no table reads the run")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR",
                    help="take up the run that a job left partly played in RUN_DIR: its finished episodes stay, the folder of an "
                         "episode cut in mid play moves into interrupted/ of the run folder and the episode is played again, and the "
                         "run folder moves under --out; a folder that holds no run in progress of the same run is left as it is and "
                         "the line plays from its first episode; pending_lines.py prints a pending pair with it")
    ap.add_argument("--training-sample", type=int, default=None,
                    help="the sample of the training starts that --starts training plays, 1 by default, one run of a sampling line each")
    ap.add_argument("--training-variant", choices=TRAINING_VARIANTS, default=AS_RELEASED,
                    help="the variant of the training table that the agent plays: without training, with the GEPA block, with "
                         "the block of the control loop that read the score alone, with the fine tuned adapter, with both, or with the "
                         "adapter of the control, fine-tuning on unfiltered trajectories, read under HYD_TRAINING")
    ap.add_argument("--training-fold", type=int, default=None,
                    help="the fold that the model of a trained variant holds out and whose test episodes it plays, or whose model's training "
                         "start states a sampling run of --starts training plays, one of the folds of the split of --training-split")
    ap.add_argument("--training-split", choices=SPLITS, default=REGISTERED,
                    help="the split whose folds --training-fold names: registered, that of the test set, or development, that of a trial "
                         "before the runs of record, which plays the preset widened to the development starts as --starts development does, "
                         "never a test combination, and is stamped")
    ap.add_argument("--replicate", type=int, choices=range(1, start_states.REPLICATES + 1), default=None,
                    help="the replicate of the evaluation set that --starts sampled plays, 1 by default: the first is drawn with --master-seed "
                         "from the --seed-pool smallest seeds of every combination, and each further one enters every combination at another "
                         "start state (start_states.sampled_combination_seeds), so a line of the grid is one run per replicate")
    ap.add_argument("--fraction", default=None,
                    help="a cut look of the test set with --starts sampled: a share below one replicate, which plays replicate 1 on "
                         "the start combinations of start_states.cut_combinations and records fraction in run.json; a whole share is "
                         "played as its replicates (pending_lines.py --fraction), so the driver takes no whole value")
    ap.add_argument("--master-seed", type=int, default=42, help="the generator seed of the first replicate of --starts sampled (42 everywhere)")
    ap.add_argument("--seed-pool", type=int, default=8, help="how many smallest seeds per combination the first replicate of --starts sampled draws from")
    ap.add_argument("--environment-mode", choices=("static", "dynamic"), default=None,
                    help="explicit development mode: frozen time and immediate repairs, or evolving time; omitted preserves the preset")
    ap.add_argument("--clock-mode", choices=CLOCK_MODES, default=None, help="override the preset's clock mode")
    ap.add_argument("--guidance", type=int, choices=(1, 2, 3), default=None, help="override the preset's guidance level")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="register the action servers without the CausalDiagnosis MCP, and the causal agent then plans on its own candidates and the normal ranges")
    ap.add_argument("--servers", choices=("rig", *frameworks.surfaces()), default=None,
                    help="rig: the rig's action server; aob: AssetOpsBench's iot, fmsr and wo servers behind the metering layer; "
                         "without the option aob for a model agent on a row of the environment table or an arm of its rows, rig otherwise")
    ap.add_argument("--store", choices=tuple(dict.fromkeys(store for entry in frameworks.surfaces().values() for store in entry["stores"])),
                    default="memory",
                    help="the database of the AssetOpsBench servers: the memory store, or the CouchDB that COUCHDB_URL, "
                         "COUCHDB_USERNAME and COUCHDB_PASSWORD name")
    ap.add_argument("--prices", default=None, help="a price table of configs/prices/ or a JSON path (default: the preset's prices)")
    ap.add_argument("--verifier-scope", choices=("writes", "all"), default=None)
    ap.add_argument("--judge", choices=("none", *sorted(frameworks_judges())), default="none",
                    help="a judge that a framework declares (utils/frameworks.py, field JUDGES); local: AssetOpsBench's judge")
    ap.add_argument("--judge-protocol", choices=("single", "paper"), default="paper",
                    help="the paper's five calls at 0.3, the default, or single, one call at temperature 0 for development")
    ap.add_argument("--judge-model", default=None, help="the judge's model name (default: --model); the fork's evaluator refuses a judge equal to the agent model")
    ap.add_argument("--judge-base-url", default=None, help="the judge's OpenAI compatible endpoint (default: --base-url)")
    ap.add_argument("--ranking-model", choices=RANKING_MODELS, default=DEFAULT_RANKING_MODEL, help="the ranking model of the diagnostics server (ablation A12); the default is the server's own")
    ap.add_argument("--abstention-rule", choices=agent_kinds.ABSTENTION_RULES, default="decision", help="the abstention rule of the diagnostics server (ablation A8)")
    ap.add_argument("--abstention-cut", type=float, default=0.95, help="the posterior cut of the fixed_cut rule")
    ap.add_argument("--no-grade", action="store_true", help="skip the diagnosability grade lookup, so every start counts as separable and no declaration as correct")
    ap.add_argument("--causal-no-branches", action="store_true", help="causal agent: hide the troubleshooting plans of the ranking (ablation branches off)")
    ap.add_argument("--causal-single-hypothesis", action="store_true", help="causal agent: show the troubleshooting plan of the top candidate only")
    ap.add_argument("--causal-plan-length", type=int, default=None,
                    help="causal agent: a count of steps per plan, none by default (the budget, the horizon and the step cap bound a plan); "
                         "1 is the reactive form")
    ap.add_argument("--causal-no-expectations", action="store_true", help="causal agent: run every plan to its end without expectation checks (the Plan-Execute form)")
    ap.add_argument("--causal-e10-reads", action="store_true",
                    help="causal agent: the advice names the separating test of E10 as the read, in place of the server's read batch by information gain per unit cost (ablation Ag7)")
    ap.add_argument("--causal-ec2-reads", action="store_true",
                    help="causal agent: the advice names the group with the largest expected cut of repair plan classes per unit cost, one step edge cutting after the EC2 objective, in place of the batch (ablation Ag7)")
    ap.add_argument("--causal-no-backtrack", action="store_true",
                    help="causal agent: keep a refuted repair plan a surprise only, without recording its set and advising the cheapest other plan (ablation Ag8)")
    ap.add_argument("--causal-sequential", action="store_true",
                    help="causal agent: run every read step alone with a ranking after it and show no schedule of the candidate plans on the working memory, "
                         "the agent without the schedule of candidate plans (ablation Ag9)")
    ap.add_argument("--causal-no-structured-output", action="store_true",
                    help="causal agent: send the planning and the orientation requests without the JSON schema of their reply, so the engine "
                         "decodes them unconstrained, as the two reimplemented designs decode (ablation Ag10)")
    ap.add_argument("--causal-expectation-tolerance", type=float, default=0.05, help="causal agent: a mass or a recovery this far below an expectation is not a surprise")
    ap.add_argument("--facts", default=None,
                    help="the data file of the asset (default: the file that the variable of its asset file names, else its release file)")
    ap.add_argument("--out", default=None,
                    help="the root of the run folders (default: CausalDiagnosis/results/runs for the engine's default asset, "
                         "CausalDiagnosis/results/runs/<asset_id> for any other)")
    return ap


def configure(a):
    """The validated configuration of a run from the command line: the preset, the overrides, the price table."""
    mode_option = getattr(a, "environment_mode", None)
    config_name, config = (load_config(a.config, getattr(a, "asset", None), environment_mode=mode_option)
                           if mode_option is not None else load_config(a.config, getattr(a, "asset", None)))
    mode = getattr(a, "environment_mode", None) or config.environment_mode
    expected_clock = "declared" if mode == "static" else "generation_time"
    if mode and a.clock_mode is not None and a.clock_mode != expected_clock:
        raise SystemExit(f"--environment-mode {mode} requires --clock-mode {expected_clock}")
    overrides = {k: v for k, v in (("clock_mode", a.clock_mode), ("guidance_level", a.guidance), ("verifier_scope", a.verifier_scope)) if v is not None}
    if a.no_diagnostics:
        overrides["diagnostics_server"] = False
    if overrides:
        config = type(config).from_json(dataclasses.replace(config, **overrides).to_json())
    price_table = load_price_table(a.prices) if a.prices else None
    if price_table:
        if mode:
            config = dataclasses.replace(config, environment_mode=None)
        config = apply_price_table(config, price_table)
    if mode:
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.runtime_mode import apply
        config = apply(config, mode)
    return config_name, config, price_table


def frameworks_judges():
    """The names of the judges the frameworks declare, read from their declarations as data (no framework module is imported)."""
    from CausalDiagnosis.utils import frameworks

    return frameworks.judges()


def hold_tables(reasons):
    """A registered run, one whose start check gave no reason, reads the generated tables as they are and computes none: it
    sets CD_TABLES_READ_ONLY=1 for itself and for every server process it starts, which inherits it, so that a missing table
    stops the run by name instead of changing the data under it (CausalDiagnosis.utils.atomic_writes_and_locks.refuse_writing)."""
    if not reasons:
        os.environ[READ_ONLY_TABLES] = "1"


def refuse_bound_parts(a, config):
    """Refuses at launch, before any table is fitted and before the run folder exists, a part of the drivers that its module binds
    to the recordings of one asset when the run plays another: the greedy reference policy and the read ceiling of the verifier
    scope all run the reference classifiers of CausalDiagnosis.environment.reference_policies.tier_classifiers, and the anchored preset maps the AssetOpsBench
    scenarios onto the dates of the recordings of the asset that a framework declares them for (run_setup.scenario_set). The static agent shows the recorded fields of the
    first cycle that a read serves, which the records of the kind STATIC_KIND hold, so it refuses an asset of another kind."""
    asset = config.asset_id
    for on, part in ((a.agent == "greedy", "--agent greedy"), (config.verifier_scope == "all", "the verifier scope all")):
        if on and asset != CLASSIFIED_ASSET.asset_id:
            raise SystemExit(f"{part} runs the reference classifiers of CausalDiagnosis.environment.reference_policies.tier_classifiers, fitted on the recordings of the "
                             f"asset {CLASSIFIED_ASSET.asset_id} alone, and this run plays the asset {asset}")
    if config.anchors is not None and scenario_set(asset) is None:
        raise SystemExit(f"the anchored preset maps the scenarios of a scenario set onto the dates of the recordings of the asset "
                         f"that a framework declares the set for, and no framework declares one for the asset {asset} that this run plays")
    kind = play_one_episode.get_profile(asset).evidence_kind
    if a.agent == "static" and kind != static_agent.STATIC_KIND:
        raise SystemExit(f"the static agent shows the recorded fields of the first cycle that a read serves, which the records of the kind "
                         f"{static_agent.STATIC_KIND} hold, and this run plays the asset {asset} of the kind {kind}")


def main(argv):
    a = build_parser().parse_args(argv)
    # checked here and not in configure(), which also resolves the lines that pending_grid_lines.py prints, each with its --replicate
    if a.replicate is not None and a.starts != "sampled":
        raise SystemExit("--replicate names a replicate of the evaluation set, so it needs --starts sampled")
    planned_episodes.fraction_checks(a)
    if a.seeds < 1 and a.starts in start_check.SEED_STARTS:
        raise SystemExit(f"--seeds {a.seeds} plans no episode, and --starts {a.starts} plays --seeds seeds of each start")
    if a.start_cap is not None and a.starts != "development":
        raise SystemExit("--start-cap caps the development starts, so it needs --starts development")
    if a.filler and a.starts != "development":
        raise SystemExit(FILLER_STARTS)
    if a.filler and a.judge != "none":
        raise SystemExit(FILLER_JUDGE)
    if a.filler and (inside := [part for part in Path(a.out or "").parts if part.startswith("grid_")]):
        raise SystemExit(f"{FILLER_OUT} {inside[-1]}")
    if a.resume and a.filler:
        raise SystemExit(RESUME_FILLER)
    config_name, config, price_table = configure(a)
    refuse_bound_parts(a, config)
    facts_path, cycles, blocks = load_world(a.facts, config.asset_id)  # the world of the asset that the configuration names
    planned_episodes.training_checks(a, config)
    planned_episodes.selector_checks(a, config)
    config = planned_episodes.played_config(a, config, blocks)
    if a.start_state is not None:
        planned_episodes.recorded_start(a, config, cycles, blocks)
    if a.fraction is not None and planned_episodes.replicate_of(a, config) is None:
        raise SystemExit("--fraction cuts a replicate of the test set, and an anchored preset plays its scenarios")
    a.seeds_list = planned_episodes.chosen_seeds(a, config, cycles, blocks)
    if a.fraction is not None and not a.seeds_list:
        # a fold that the prefix leaves empty plays nothing, which a first start of a grid meets as well as pending_grid_lines.py
        print(f"NOTHING_TO_PLAY the cut look {a.fraction} gives the fold {a.training_fold} no start combination", file=sys.stderr)
        raise SystemExit(0)
    plan = planned_episodes.episode_plan(config, a.seeds_list, cycles, blocks, a.scenario_ids)
    # the start check, before any table is fitted and before the run folder exists
    reasons = start_check.registration(a, config, config_name, facts_path)
    if a.starts == "training":
        start_check.refuse_registered_training(a, reasons, "--starts training")
    hold_tables(reasons)
    runner, judge_options = runner_setup.build_runner(a, config_name, config, price_table, cycles, blocks, facts_path, registered=not reasons)
    runner.unregistered = reasons
    if is_marker(runner.adapter):
        # the fold kept too few episodes for an adapter, so the run is a marker that plays no episode and records the
        # gate and the alias, and the tables read the runs without training of the fold in its place, replicate by replicate
        a.seeds_list, plan = [], []
        print(f"no adapter for {a.agent} on fold {a.training_fold}: a marker run, the tables read {runner.adapter['alias']}")
    run_dir, meta, rows = (run_record.resume_run if a.resume else run_record.start_run)(a, runner, config, price_table, plan, judge_options, reasons, argv)
    # the episodes a run taken up again keeps are a prefix of its plan, and the rest plays as in an uninterrupted run
    for played, (name, seed, scenario, episode_config) in enumerate(plan[len(rows):]):
        row = runner.play(run_dir / name, name, seed, scenario, episode_config, meta)
        rows.append(row)
        print(f"{name}: ended {row['end_reason']!r} after {row['tool_calls']} tool calls, verified {row['pass'] != ''}, {row['wall_s']} s"
              + (f", error {row['error']}" if row["error"] else ""), flush=True)
        # the rows so far, so that a run cut short by the end of an allocation keeps every finished episode; run.json is
        # written at the end only, so a folder without it is a run that did not finish
        write_rows(run_dir / "scores.csv", rows, COLUMNS)
        ledger.ledger_file.add_episode(run_dir, name, row)
        if played == 0:
            run_record.refuse_unlisted(run_dir)  # an episode loads the modules of its agent, so an unlisted one shows after the first
            run_record.refuse_untracked(run_dir, meta)
        if failures.stop_due(rows, STOP_AFTER_FAILURES):
            # the infrastructure is down: the run writes no run.json, so its line stays pending and is played again as a whole
            ledger.append_event(run_dir, "run_episodes.py", failures.STOP_EVENT)
            print(f"{run_dir}: {failures.STOP_EVENT}, {STOP_AFTER_FAILURES} episodes of the class {failures.HARNESS_ERROR}; no run.json is written",
                  file=sys.stderr, flush=True)
            raise SystemExit(75)
    run_record.refuse_unlisted(run_dir)
    run_record.refuse_untracked(run_dir, meta)
    meta["loaded_files"] = run_record.loaded_files()
    meta["load_averages"] = runner.load_averages  # at check_servers_once
    run_record.write_run(run_dir, meta, rows)
    ledger.finish(run_dir)
    print(f"wrote {run_dir}")
    return run_dir
