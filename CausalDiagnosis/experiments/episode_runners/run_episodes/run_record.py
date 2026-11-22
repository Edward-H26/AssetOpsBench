import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, episode_name, read_rows  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT  # noqa: E402
from CausalDiagnosis.utils.csv_rows import write_rows  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json, write_json  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import DEVELOPMENT_SEED  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.run_verification.public_record import public_value  # noqa: E402
from CausalDiagnosis.experiments.run_verification.run_identity_file_lists import excluded, in_roots, lists_of  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import TRAINING_FOLD_KEY, TRAINING_MODE, TRAINING_SPLIT_KEY, TRAINING_VARIANT_KEY  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.audit_reports import run_summary  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import BRIEFING_VERSION  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.agents.utils.mcp_sessions import SESSIONS  # noqa: E402
from . import agent_kinds


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def loaded_paths(root=ROOT):
    """The paths relative to the root of the tree of every file that a module of this process was loaded from, sorted. A
    __file__ that is no absolute path names no file (torch gives its namespaces torch.ops and torch.classes "_ops.py" and
    "_classes.py"), and resolving it against the working folder would name a file of the tree that was never loaded."""
    root, paths = Path(root).resolve(), set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if isinstance(name, str) and Path(name).is_absolute() and Path(name).resolve().is_relative_to(root):
            paths.add(Path(name).resolve().relative_to(root).as_posix())
    return sorted(paths)


def loaded_files(root=ROOT):
    """run.json loaded_files: the loaded files that lie in a list of experiment_tables.py, so that a later change of a listed file counts
    against a run only when its driver process had loaded it."""
    return [path for path in loaded_paths(root) if lists_of(path)]


def loaded_outside_lists(root=ROOT):
    """The loaded files under the roots of the lists of experiment_tables.py that lie in no list, which the identity of a run would miss."""
    return [path for path in loaded_paths(root) if in_roots(path) and not excluded(path) and not lists_of(path)]


def refuse_unlisted(run_dir):
    """Stops a run whose driver process loaded a file under the roots of the lists of experiment_tables.py that lies in no list, which the
    identity of the run would miss, before run.json is written."""
    outside = loaded_outside_lists()
    if outside:
        raise SystemExit(f"{run_dir}: the driver process loaded {', '.join(outside)}, which lies in no list of experiment_tables.py and so in no "
                         "identity; no run.json is written")


# the settings of the process that the AssetOpsBench servers read, recorded with their values in run.json server_env
SERVER_ENV_RECORDED = ("AOB_READONLY", "FMSR_MODEL_ID", "IOT_DBNAME", "ASSET_DBNAME", "WO_DBNAME", "FAILURE_CODE_DBNAME", "FAILURE_MODE_DBNAME")


def refuse_untracked(run_dir, meta):
    """Stops a run of record whose driver process loaded a file that git does not track at the commit (rule 3), before run.json is
    written, since the commit would not hold what the run played."""
    if meta.get("stamp"):
        return
    untracked = ledger.code_identity.untracked_loaded(loaded_files())
    if untracked:
        raise SystemExit(f"{run_dir}: the driver process loaded {', '.join(untracked)}, which git does not track at the commit of the run; "
                         "no run.json is written")


def clock_totals(rows):
    """run.json clock: the sums over the episode rows of the clock cells."""
    def summed(key, part=None):
        total = 0.0
        for row in rows:
            value = row.get(key)
            value = json.loads(value) if isinstance(value, str) and value.startswith("{") else value
            if isinstance(value, dict):
                value = value.get(part, 0) if part else sum(value.values())
            total += float(value) if value not in (None, "") else 0.0
        return round(total, 3)
    return {"model_failures": {"count": int(summed("failed_model_attempts")), "seconds": summed("failed_model_s")},
            "credited_seconds": {"starts": summed("credited_s", "starts"), "calls": summed("credited_s", "calls")},
            "uncredited_seconds": {"starts": summed("uncredited_s", "starts"), "calls": summed("uncredited_s", "calls")},
            "uncharged_after_last_seconds": summed("uncharged_after_last_s"),
            "cuts": {end: sum(1 for row in rows if row.get("cut_end") == end) for end in (failures.SERVICE, failures.WALL)}}


# the variables of Slurm that name what the job was granted, by the key of run.json granted
GRANTED_VARIABLES = {"cpus": "SLURM_CPUS_ON_NODE", "memory_per_node_mb": "SLURM_MEM_PER_NODE", "memory_per_cpu_mb": "SLURM_MEM_PER_CPU"}


def granted():
    """run.json granted: the processors and the memory that the scheduler granted the job of the run, an integer each or None for a
    value the job was not granted that way, under the scheduler slurm. A run outside a Slurm job, as on a laptop, records the scheduler
    None with every value None, so that its group key on a wall-clock preset differs from that of every run of the cluster."""
    if not os.environ.get("SLURM_JOB_ID"):
        return {"scheduler": None, **{key: None for key in GRANTED_VARIABLES}}
    values = {key: os.environ.get(variable) for key, variable in GRANTED_VARIABLES.items()}
    return {"scheduler": "slurm", **{key: int(value) if value and value.isdigit() else None for key, value in values.items()}}


RESUME_NONE = "no run in progress, a folder with a ledger of the status running, no run.json and no stop after harness errors in a row"


def run_label(config_name, agent_label):
    # the label names one folder: the slashes of a router model id such as z-ai/glm-5.3 become underscores, and so do its
    # dots, the form of the name of a run folder that the docs state and the tools that read run folders know
    return f"{config_name}_{agent_label}_{datetime.now():%Y%m%d-%H%M%S}".replace(".", "_").replace("/", "_")


def run_base(out, name):
    """The folder of a run of that name under --out, which a relative --out names from the root of the repository. A launcher of an
    episode starts as a file wherever its folder lies (episode_launchers.py), so a dot in a folder name is no reason to refuse it."""
    return (ROOT / out / name).resolve()


def run_folder(out, label):
    base = run_base(out, label)
    # created here, atomically, so two runs of one label started in the same second (parallel grid lines) get distinct folders
    run_dir = base
    for n in itertools.count(2):
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = base.with_name(f"{base.name}-{n}")


def write_run(run_dir, meta, rows):
    meta["ended"] = datetime.now().isoformat(timespec="seconds")
    meta["clock"] = clock_totals(rows)
    ledger.hash_samplers(meta.get("serving"))  # the GPU sampler of every endpoint as it stands at the end of the run
    write_json(run_dir / "run.json", public_value(meta))  # the public form: no endpoint, host, job, card id or machine path
    write_rows(run_dir / "scores.csv", rows, COLUMNS)
    (run_dir / "summary.md").write_text(run_summary(run_dir), encoding="utf-8")


def agent_settings(a, runner):
    """The blocks decoding, transport and limits of run.json, with the same keys for every agent: the settings each route
    sends with every request and how its client and the tool sessions carry it, from the accessors of the routes, and the own
    step limits of the agent as its package names them. An agent without a model has no decoding."""
    sessions = {"sessions": SESSIONS, "tool_timeout_s": None}  # a tool call has no limit of its own, the row's deadline bounds it
    limits = None
    if a.agent in agent_kinds.PAPER_ERA_AGENTS:
        paper = frameworks.code("AGENTS", a.agent)
        settings = paper.request_settings(runner.temperature)
        limits = paper.settings(runner.servers, runner.temperature)
    elif a.agent == "causal":
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import CausalLoopOptions
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import request_settings
        routed = hasattr(getattr(runner.backend, "inner", runner.backend), "request_settings")  # a stand-in planner sends no request
        settings = request_settings(runner.backend, temperature=runner.temperature) if routed else {"decoding": None, "transport": sessions}
        limits = CausalLoopOptions(**runner.causal_options).to_json()
    elif runner.backend is not None and hasattr(runner.backend.inner, "request_settings"):
        settings = runner.backend.request_settings()  # the static agent on the chat route, with the sessions added here
        settings = {**settings, "transport": {**(settings.get("transport") or {}), **sessions}}
    else:  # the scripted check and the reference policies call no model
        settings = {"decoding": None, "transport": sessions}
    return {"decoding": settings.get("decoding"), "transport": settings.get("transport"), "limits": limits}


def run_meta(a, runner, config, price_table, plan, judge_options, reasons=None):
    """run.json of a run as it starts. reasons are those of registration(): a run with a reason carries the stamp development and
    its reasons, and a run without one, a registered run, carries neither."""
    reasons = list(runner.unregistered or []) if reasons is None else list(reasons)
    stamp = {"stamp": start_check.STAMP, "stamp_reasons": reasons} if reasons else {}
    return {
        "label": runner.label, "config_name": runner.config_name, "config": config.to_json(),
        "mode": "static" if a.agent == "static" else TRAINING_MODE if a.starts == "training" else "gym",
        "prices": runner.prices_name, "price_table_source": price_table["source"] if price_table else None,
        "agent": a.agent, "agent_label": runner.agent_label, "servers": list(runner.servers),
        # the runner's own dict, which the listing of the servers in the first episode fills before this record is written
        "tool_listings": runner.tool_listings,
        "backend": {k: v for k, v in runner.store.items() if k in ("kind", "url")} if runner.store else None,
        "model": a.model if a.agent in agent_kinds.MODEL_AGENTS else None,
        "model_id": runner.backend.model_id if runner.backend else (runner.served_model if a.agent in agent_kinds.PAPER_ERA_AGENTS else None),
        # one block per declared port, present with null for every other agent
        **{name: {**frameworks.code("AGENTS", name).settings(runner.servers, runner.temperature), "diagnosis_example": runner.diagnosis_example}
           if a.agent == name else None
           for name in sorted(agent_kinds.PAPER_ERA_AGENTS)},
        "endpoint": a.base_url if a.agent in agent_kinds.MODEL_AGENTS else None, "judge": a.judge,
        "judge_protocol": judge_options["protocol"] if judge_options else None, "judge_model": judge_options["model"] if judge_options else None,
        "judge_endpoint": judge_options["base_url"] if judge_options else None, "grade_lookup": not a.no_grade,
        "ranking_model": a.ranking_model, "abstention_rule": a.abstention_rule, "abstention_cut": a.abstention_cut if a.abstention_rule == "fixed_cut" else None,
        "causal_options": runner.causal_options,
        "clock_mode": config.clock_mode, "guidance_level": config.guidance_level, "diagnostics_server": config.diagnostics_server,
        "verifier_scope": config.verifier_scope, "briefing": runner.surface.briefing_version if runner.store else BRIEFING_VERSION, "starts": a.starts, "master_seed": a.master_seed if a.starts in ("sampled", "training") else None,
        "seed_pool": a.seed_pool if a.starts in ("sampled", "training") else None,
        "replicate": planned_episodes.replicate_of(a, config), "replicate_rule": start_states.REPLICATE_RULE if planned_episodes.replicate_of(a, config) is not None else None,
        # a cut look alone carries the share, so the run.json of a whole replicate is that of every run before the share
        **({"fraction": start_states.fraction_value(a.fraction), "fraction_rule": start_states.FRACTION_RULE}
           if getattr(a, "fraction", None) is not None else {}),
        TRAINING_VARIANT_KEY: a.training_variant, TRAINING_FOLD_KEY: a.training_fold, TRAINING_SPLIT_KEY: a.training_split,
        "adapter": runner.adapter, "task_block": runner.task_block_record,
        "training_sample": planned_episodes.training_sample_of(a), "sampling_temperature": runner.temperature,
        "seeds": list(a.seeds_list), "episodes_planned": len(plan),
        "anchored": planned_episodes.anchored_note(config, plan), "facts_path": str(runner.facts_path),
        "facts_sha256": hashlib.sha256(Path(runner.facts_path).read_bytes()).hexdigest()[:12],
        "code": ledger.code_record(a.agent), "on_origin": bool(ledger.git_lines(ROOT, "branch", "-r", "--contains", "HEAD")),
        "server_env": {name: os.environ.get(name) for name in SERVER_ENV_RECORDED}, "python": platform.python_version(), "mcp": _version("mcp"), "openai": _version("openai"),
        "asset_id": config.asset_id, "platform": ledger.platform_of(), "data": ledger.run_data.data_record(runner.facts_path, config.asset_id),
        "serving": serving_of(a, judge_options), "lora_adapters": lora_adapters_of(a),
        "runs_in_flight": start_check.runs_in_flight(), "granted": granted(), "start_cap": getattr(a, "start_cap", None),
        "start_cap_seed": DEVELOPMENT_SEED if getattr(a, "start_cap", None) is not None else None,
        # the start state and the scenarios of a run of a selector, which a run without one does not carry
        **({"start_state": planned_episodes.start_state_of(a, config)} if getattr(a, "start_state", None) is not None else {}),
        **({"scenario_ids": a.scenario_ids} if getattr(a, "scenario_ids", None) is not None else {}),
        # the prices the run charged, so that two runs of one price label and other prices never pool (the label of a run that names
        # no table is default whatever prices its preset holds)
        "prices_sha256": ledger.canonical_sha256(config.to_json().get("prices")), **stamp,
        # the key on a filler alone, which results_index.refused and pending_lines.finished_runs read
        **({"filler": True} if getattr(a, "filler", False) else {}),
        **agent_settings(a, runner), "harness": harness_rule.harness_settings(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def serving_of(a, judge_options):
    """run.json serving: what the run records of the model endpoint of a model agent and of the judge endpoint of a run that
    judges in the run, each from the endpoint file that the grid job names for it (ledger.serving_record), None for an endpoint
    that the run does not reach. write_run adds the hash of each sampler file at the end of the run."""
    agent = ledger.serving_record(ledger.served(a.model, a.base_url, "HYD_ENDPOINT_FILE")) if a.agent in agent_kinds.MODEL_AGENTS else None
    judge = (ledger.serving_record(ledger.served(judge_options["model"], judge_options["base_url"], "HYD_JUDGE_ENDPOINT_FILE"))
             if judge_options else None)
    return {"agent": agent, "judge": judge}


def lora_adapters_of(a):
    """run.json lora_adapters, the serving setup of a run: the LoRA adapters that the server of a model agent's endpoint holds beside
    the model, the line lora_adapters of the endpoint file of its serving job (CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch) when that file names the
    run's endpoint, and 0 for a reference policy, an endpoint without that file or a file written before the line. The condition
    of a run holds it (results_index.CONDITION_KEYS), so one line played on a server with adapters and on one without is two
    conditions and never one pair."""
    if a.agent not in agent_kinds.MODEL_AGENTS:
        return 0
    fields = ledger.endpoint_file(os.environ.get("HYD_ENDPOINT_FILE")) or {}
    value = str(fields.get("lora_adapters", "")) if fields.get("base_url") == a.base_url else ""
    return int(value) if value.isdigit() else 0


def serving_started(base_url):
    """The start of the serving job of the model endpoint, from the endpoint file that the grid job names in HYD_ENDPOINT_FILE, or
    None without one and for a file of another endpoint, which a fine tuned variant then records as not compared."""
    record = ledger.endpoint_file(os.environ.get("HYD_ENDPOINT_FILE"))
    return record.get("started") if record and record.get("base_url") == base_url else None


def default_out(asset_id):
    """The root of the run folders of a run whose command line names none: the runs folder of the rig for the engine's default
    asset, and its folder under it, CausalDiagnosis/results/runs/<asset_id>, for any other asset, which the code lists leave out, so
    that the runs of two assets never share a root unless a command line asks for it."""
    return "CausalDiagnosis/results/runs" if asset_id == DEFAULT_ASSET else f"CausalDiagnosis/results/runs/{asset_id}"


def start_run(a, runner, config, price_table, plan, judge_options, reasons, argv):
    """(run folder, run.json fields, rows) of a run that starts: its folder under --out, its fields and its ledger, no row yet."""
    run_dir = run_folder(a.out or default_out(config.asset_id), runner.label)
    meta = run_meta(a, runner, config, price_table, plan, judge_options, reasons)
    ledger.start(run_dir, meta, argv)
    return run_dir, meta, []


def intact_rows(run_dir):
    """The rows of scores.csv that the file holds whole: the longest prefix of its rows that, written again, gives the first bytes of
    the file. The driver rewrites the file after every episode, so a job cut during the write leaves the rows before the cut whole
    and one row at most half written, which is never kept."""
    rows = read_rows(run_dir)
    data = (Path(run_dir) / "scores.csv").read_bytes() if rows else b""
    with tempfile.TemporaryDirectory() as scratch:
        for n in range(len(rows), 0, -1):
            try:
                write_rows(Path(scratch) / "scores.csv", rows[:n], COLUMNS)
            except ValueError:  # a row with cells beyond the header
                continue
            if data.startswith((Path(scratch) / "scores.csv").read_bytes()):
                return rows[:n]
    return []


def set_aside_unfinished(run_dir, names, kept):
    """[(episode, folder under ledger.INTERRUPTED)] of every folder of a planned episode that the run does not keep, and of the
    scratch folder of the tool listing, each moved into ledger.INTERRUPTED of the run folder, where a second copy of a name takes
    the suffix -2, -3 and so on; nothing is deleted."""
    aside = []
    for name in [*(name for name in names if name not in kept), harness_rule.SERVER_CHECK_FOLDER]:
        folder = Path(run_dir) / name
        if not folder.exists():
            continue
        moved, n = name, 2
        while (Path(run_dir) / ledger.INTERRUPTED / moved).exists():
            moved, n = f"{name}-{n}", n + 1
        (Path(run_dir) / ledger.INTERRUPTED).mkdir(exist_ok=True)
        shutil.move(str(folder), str(Path(run_dir) / ledger.INTERRUPTED / moved))
        aside.append((name, moved))
    return aside


def resume_run(a, runner, config, price_table, plan, judge_options, reasons, argv):
    """(run folder, run.json fields, kept rows) of the run in progress that --resume names, taken up by this job when the command
    plays the same run (ledger.resume_refusal): the run keeps its label, which its episodes name, and its start time; its folder
    moves under --out; it keeps the longest prefix of its plan whose episodes are finished and whose rows scores.csv holds whole
    (ledger.kept_episodes, intact_rows); every other planned folder moves aside (set_aside_unfinished); the ledger records the job
    (ledger.record_resume); and scores.csv keeps the rows of the kept episodes alone. Any other folder is left as it is, and the
    line plays from its first episode as a run that starts (start_run)."""
    source = Path(a.resume).resolve()
    label, out = runner.label, a.out or default_out(config.asset_id)
    refusal, target = RESUME_NONE, run_base(out, source.name)
    if ledger.in_progress(source):
        record = read_json(source / ledger.LEDGER)
        runner.label = record["label"]
        meta = run_meta(a, runner, config, price_table, plan, judge_options, reasons)
        meta["started"] = record["started"]
        refusal = ledger.resume_refusal(source, meta, reasons)
        if refusal is None and target != source and target.exists():
            refusal = f"{target} exists already"
    if refusal is not None:
        runner.label = label
        print(f"--resume {source}: {refusal}; the line plays from its first episode", flush=True)
        return start_run(a, runner, config, price_table, plan, judge_options, reasons, argv)
    if target != source:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    names = [name for name, _seed, _scenario, _config in plan]
    recorded = {episode_name(row["seed"], row.get("scenario_id"), row.get("gepa_call")): row for row in intact_rows(target)}
    kept = ledger.kept_episodes(target, names, recorded)
    aside = set_aside_unfinished(target, names, kept)
    ledger.record_resume(target, argv, kept, aside, None if target == source else str(source), serving=meta["serving"],
                         granted=granted())
    rows = [recorded[name] for name in kept]
    write_rows(target / "scores.csv", rows, COLUMNS)
    print(f"{target}: taken up again after {len(kept)} of {len(names)} planned episodes"
          + (f", set aside {', '.join(name for name, _moved in aside)}" if aside else ""), flush=True)
    return target, meta, rows


# Imported after the definitions: this module uses them only when a function runs, and they may load
# this module first (the order of the package is in __init__.py).
from . import harness_rule, planned_episodes, start_check
