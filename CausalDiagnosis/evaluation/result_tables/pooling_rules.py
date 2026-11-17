from CausalDiagnosis.utils.paths import RESULTS
import functools
import json
from pathlib import Path
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_ABSTENTION_RULE
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import RIG_SERVER
from CausalDiagnosis.experiments.episode_runners.run_configuration import ROW_PRESETS
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import run_dirs
from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.experiments.run_verification.run_identity_file_lists import AGENT_LIST
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AS_RELEASED, TRAINING_FOLD_KEY, TRAINING_MODE, TRAINING_VARIANT_KEY
from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of  # the one rule of the asset of a run, which the modules of the tables and the training share


HERE = Path(__file__).resolve().parent
STAMP = "development"  # the run.json stamp of a run that the start check of run_episodes/ did not register
RESCORE_TOOL = "rescore_scope.py"  # the tool of the event that rescore_under_verifier_scope.py writes into the ledger of its copy
ARCHIVE_TOOL = "archive"  # run_ledger/ event RUN_DIR archive WHY: the run of a pair of one registered condition that leaves the tables
DATA_CHANGED = "data files changed while the run was in progress"  # the event of ledger.finish that keeps a run out of a registered root
# the keys of the serving identity of run.json that a group of the tables shares: what the tracked serving file declares and the
# start check compares (served model, weights commit, launch dtype, launch arguments, engine version) held again as a second guard, and
# what only the serving job reports (the reported dtype, quantization, kv cache dtype, image and template hashes), bound here alone;
# the GPU card joins them in serving_key
BOUND_SERVING = ("model", "weights_commit", "dtype_launch", "launch_args_sha256", "engine_version",
                 "dtype", "quantization", "kv_cache_dtype", "image_sha256", "template_sha256")
WALL_CLOCK = "generation_time"  # the clock mode that charges the wall seconds between calls, whose runs in flight bind
DEVELOPMENT_BANNER = "Development runs. Not results."  # the first line of every page that a table tool writes from a development root
MEMORY_BACKEND = ("the memory backend of AssetOpsBench's servers (run.json backend.kind memory), while every number of the paper "
                  "comes from the real servers over CouchDB")
RETIRED_GEPA_MODE = ("a run of the offline GEPA optimizer that left the fork (run.json mode gepa, which no driver of the fork writes), "
                     "while the GEPA results of the paper come from the online loop")
TRAINING_SAMPLE = "a run that samples trajectories for fine tuning (run.json mode training), whose episodes are training data and enter no result table"
RIG_SURFACE_ROW = ("a model agent on a row of the environment table, or on an arm of its rows, over the rig's action server, while every row runs on "
                   "AssetOpsBench's iot, fmsr and wo servers with the task server")
FILLER_RUN = ("a filler run, which played the development starts to keep the runs in flight of its grid job (run.json filler), "
              "while a table reads the runs of the grid lines alone")
CHOSEN_STARTS = ("a run of chosen starts, one start state or named scenarios of its preset (run.json start_state or scenario_ids), "
                 "while a cell of a table reads the start set of its preset")


def stamped(meta):
    """Whether a run.json carries the stamp of a run that the start check did not register."""
    return meta.get("stamp") == STAMP


def chosen_starts(record):
    """Whether a run played chosen starts, the start state of --start-state or the scenarios of --scenario-ids, as its run.json
    records them under start_state and scenario_ids, and the seeds block of its ledger while it runs (ledger_file.run_fields)."""
    return record.get("start_state") is not None or record.get("scenario_ids") is not None


def chosen_starts_line(names, quote=""):
    """The words in which a table tool says how many runs of chosen starts it left out, and which."""
    return (f"{len(names)} {'run' if len(names) == 1 else 'runs'} of chosen starts (run.json start_state or scenario_ids), which no "
            f"cell of a table reads: {', '.join(f'{quote}{name}{quote}' for name in names)}")


def surface_of(meta):
    """The tool surface of a run from the servers of its run.json: rig when the rig's action server serves it, aob when
    AssetOpsBench's servers behind the metering layer do. A run.json without the field predates the AssetOpsBench rows."""
    servers = meta.get("servers") or [RIG_SERVER]
    return "rig" if RIG_SERVER in servers else next(name for name, entry in frameworks.surfaces().items() if set(entry["servers"]) & set(servers))


def refused(run_dir):
    """The reason no table reads a finished run folder, or None. The memory backend, which the driver writes into run.json
    as backend.kind for --servers aob --store memory, serves local tests and debugging only. A run of
    the retired mode gepa stays outside the paper, as does a run that samples trajectories for fine tuning. A model agent on a row preset, or on an arm of the rows (ROW_PRESETS of
    episode_launchers.py, run_configuration.py, run_folder_layout.py, servers_of_a_run.py, CausalDiagnosis/utils/csv_rows.py and CausalDiagnosis/utils/json_files.py), over the rig's action server plays a surface no row of the paper has, since the action server serves the reference policies, the gym and training,
    while a reference policy there stays a reference and the static agent plays no tool. A run of chosen starts (chosen_starts)
    plays one start state or named scenarios and not the start set that a cell reads."""
    meta = read_json(Path(run_dir) / "run.json", {})
    if meta.get("filler"):
        return FILLER_RUN
    if chosen_starts(meta):
        return CHOSEN_STARTS
    if (meta.get("backend") or {}).get("kind") == "memory":
        return MEMORY_BACKEND
    if meta.get("mode") == "gepa":
        return RETIRED_GEPA_MODE
    if meta.get("mode") == TRAINING_MODE:
        return TRAINING_SAMPLE
    if meta.get("mode", "gym") == "gym" and meta.get("model") and meta.get("config_name") in ROW_PRESETS and surface_of(meta) == "rig":
        return RIG_SURFACE_ROW
    return None


# the fields of run.json that name the condition of a run, those of CONDITION in tables_from_runs/ with the agent label for the agent
# and the gepa block, which tells the two online loops of GEPA apart, and what a run.json without a key played (load_episodes)
# (run.json key, episode field, whether a row of the tables keeps it apart), the one list of the condition: aggregate.CONDITION is
# ROW_FIELDS in this order, and two runs of one condition share every key
CONDITION_FIELDS = (
    ("agent_label", "agent", True), ("model", "model", True), ("config_name", "config", True), ("clock_mode", "clock_mode", True),
    ("servers", "surface", True), ("diagnostics_server", "diagnostics_server", True), ("guidance_level", "guidance_level", True),
    ("verifier_scope", "verifier_scope", True), ("prices", "prices", True), ("ranking_model", "ranking_model", True),
    ("abstention_rule", "abstention_rule", True), (TRAINING_VARIANT_KEY, "training_variant", True), ("lora_adapters", "lora_adapters", True),
    ("abstention_cut", "abstention_cut", True), ("prices_sha256", "prices_sha256", True),
    ("fraction", "fraction", True),  # the share of a cut look of the test set, a set of rows of its own
    ("mode", "mode", False),  # every table selects the mode it reads
    ("gepa", "gepa", False),  # the GEPA tables alone read the block of a loop
    ("asset_id", "asset_id", False),  # a table set is one asset: load_episodes refuses two
    (TRAINING_FOLD_KEY, "training_fold", False),  # the trained rows pool their folds by cross-fitting
    # the episode set of a line: a registered root plays one set per line, and RUNS and the leaderboard keep a line's set apart
    ("master_seed", "master_seed", False), ("seed_pool", "seed_pool", False), ("replicate_rule", "replicate_rule", False),
    ("start_cap", "start_cap", False),
)
CONDITION_KEYS = tuple(key for key, _, _ in CONDITION_FIELDS)
ROW_FIELDS = tuple(field for _, field, row in CONDITION_FIELDS if row)
# a run.json written before the serving setup played on a server without LoRA adapters, as every run before the training table
ABSENT = {"abstention_rule": DEFAULT_ABSTENTION_RULE, TRAINING_VARIANT_KEY: AS_RELEASED, "lora_adapters": 0}
# the condition keys that a run.json holds only for the runs they name, so the condition of every other run reads as before them
PRESENT_ONLY = ("fraction",)


def condition_of(meta):
    """The condition of a run as its run.json names it, on which two runs that share a seed would pool in every table."""
    full = {**ABSENT, **meta}
    return json.dumps({key: full.get(key) for key in CONDITION_KEYS if key not in PRESENT_ONLY or full.get(key) is not None},
                      sort_keys=True, default=str)


def identity_of(meta):
    return (meta.get("code") or {}).get("identity")


def card_of(meta):
    """The GPU card of the model endpoint of a run as its run.json records it under serving, None for a run without a model and
    for a run.json written before the key (ledger.card_of)."""
    return ledger.card_of((meta.get("serving") or {}).get("agent"))


def _events(run_dir):
    record = ledger.load(run_dir) if (Path(run_dir) / ledger.LEDGER).exists() else None
    return (record or {}).get("events", []), record


def archived(run_dir):
    """Whether the ledger of a run holds an archive event, which takes it out of a pair of one registered condition."""
    return any(event.get("tool") == ARCHIVE_TOOL for event in _events(run_dir)[0])


def serving_key(serving):
    """The bound keys of the serving identity of one endpoint record (BOUND_SERVING) with its GPU card, or None without a record."""
    if not serving:
        return None
    return {**{key: serving.get(key) for key in BOUND_SERVING}, "card": ledger.card_of(serving)}


def registered_refusal(meta, run_dir):
    """The reason a run of record is left out, or None: its tree differed from its commit or from its upload, its ledger names
    data files that changed while it played, or a job that took it up (ledger.record_resume) recorded another serving of the model
    endpoint, its card among it, than run.json records, so that the run played on two endpoints and is two identities, or, on a
    wall-clock preset, a job that took it up under other processors or memory than run.json records."""
    code = meta.get("code") or {}
    if code.get("dirty"):
        return f"its tree differed from its commit in {', '.join(code['dirty'][:3])}"
    if code.get("differs_from_upload"):
        return f"its tree differed from its upload in {', '.join(code['differs_from_upload'][:3])}"
    events, _record = _events(run_dir)
    if any(event.get("what") == DATA_CHANGED for event in events):
        return "its ledger names data files that changed while the run was in progress"
    first = serving_key((meta.get("serving") or {}).get("agent"))
    for event in events:
        if event.get("tool") == ledger.RESUME_TOOL and "serving" in event:
            later = serving_key((event.get("serving") or {}).get("agent"))
            if later != first:
                differing = sorted(key for key in set(first or {}) | set(later or {}) if (first or {}).get(key) != (later or {}).get(key))
                return f"the job that took it up at {event.get('at')} served its model otherwise than its first job, in {', '.join(differing) or 'its record'}"
        if (event.get("tool") == ledger.RESUME_TOOL and "granted" in event and meta.get("clock_mode") == WALL_CLOCK
                and event["granted"] != meta.get("granted")):
            return f"the job that took it up at {event.get('at')} was granted other processors or memory than its first job on the wall clock"
    return None


def group_key(meta):
    """The group key of a run of record, {part: hash or value}: lists, the SHA-256 over the lists of the role identity that are no
    agent's own (ledger.identity_lists), with the data and the generated tables of run.json data; serving, the bound serving of the
    model endpoint with its card (None for a run without a model); runs_in_flight, the runs in flight of a wall-clock preset (None
    on any other clock); granted, the SHA-256 of the processors and the memory granted to the job of a wall-clock preset (None on
    any other clock and for a run.json without the key); agent, the SHA-256 over the agent's own lists. Only lists of the role
    identity enter, so a later change of a bound or free list, the code of the tables and the cluster scripts among them, splits no
    run."""
    files = (meta.get("code") or {}).get("files") or {}
    identity = ledger.identity_lists(meta.get("agent"))
    shared = sorted([name, path, digest] for name in identity if not name.startswith(AGENT_LIST) for path, digest in (files.get(name) or {}).items())
    own = sorted([name, path, digest] for name in identity if name.startswith(AGENT_LIST) for path, digest in (files.get(name) or {}).items())
    serving = serving_key((meta.get("serving") or {}).get("agent"))
    return {"lists": ledger.canonical_sha256({"lists": shared, "data": meta.get("data")}),
            "serving": None if serving is None else ledger.canonical_sha256(serving),
            "runs_in_flight": meta.get("runs_in_flight") if meta.get("clock_mode") == WALL_CLOCK else None,
            "granted": ledger.canonical_sha256(meta["granted"]) if meta.get("clock_mode") == WALL_CLOCK and meta.get("granted") is not None else None,
            "agent": ledger.canonical_sha256(own)}


def kind_of(meta):
    """record, development or legacy: a run of record, which the start check registered, carries an identity and no stamp, a run it did
    not register carries the stamp, and a run.json written before the identity existed, a test fixture among them, carries
    neither and reads as runs read before it."""
    return "development" if stamped(meta) else "record" if identity_of(meta) else "legacy"


def commit_of(meta):
    return (meta.get("code") or {}).get("commit")


def _later_start_counts(order, metas, left_out):
    """Of two finished runs of one condition that share a seed, the later start alone counts, as the leaderboard chooses."""
    kept = []  # the later runs with a seeds list that count, latest first
    for name in reversed([n for n in order if n not in left_out and metas[n].get("seeds")]):
        meta = metas[name]
        counting = next((k for k in kept if condition_of(metas[k]) == condition_of(meta) and set(meta["seeds"]) & set(metas[k]["seeds"])), None)
        if counting is None:
            kept.append(name)
            continue
        shared = sorted(set(meta["seeds"]) & set(metas[counting]["seeds"]))
        left_out[name] = (f"an earlier finished run of the condition of {counting} (started {metas[counting].get('started')}), which plays "
                          f"{len(shared)} of its seeds again ({', '.join(str(s) for s in shared)}), so the later start alone counts, as the "
                          "leaderboard chooses, and no table pools two runs of one line")


def _legacy(order, metas, left_out):
    """Runs written before the identity: every run whose commit differs from the commit of the latest start is left out, then
    the later start counts."""
    newest = order[-1]
    newest_commit = commit_of(metas[newest])
    for name in order:
        commit = commit_of(metas[name])
        if commit != newest_commit:
            left_out[name] = (f"played commit {commit or 'none recorded'} while {newest} (the latest start, {metas[newest].get('started')}) played "
                              f"{newest_commit or 'none recorded'}, and runs of different commits are never pooled")
    _later_start_counts(order, metas, left_out)


def own_scope_copy(meta):
    """Whether a run folder is a copy that rescore_under_verifier_scope.py --own-scope scored again under the run's own scope."""
    return bool((meta.get("rescored") or {}).get("own_scope"))


def rescored_refusal(copy_meta, source_meta, copy_dir=None):
    """The reason a copy that rescore_under_verifier_scope.py --own-scope wrote is not read in place of its run, or None: every list but scoring
    is that of the run it came from, and, with its folder, its ledger holds the event of rescore_under_verifier_scope.py. The copy then enters the
    group of its root with the scoring list it was scored under (_grouped), so it is read only beside runs of that list."""
    if not own_scope_copy(copy_meta):
        return "it is no copy scored again under its own scope"
    lists, source_lists = copy_meta["code"]["files"], (source_meta.get("code") or {}).get("files") or {}
    differing = sorted(name for name in set(lists) | set(source_lists) if name != "scoring" and lists.get(name) != source_lists.get(name))
    if differing:
        return f"its lists {', '.join(differing)} differ from those of the run it came from"
    if copy_dir is not None and not any(event.get("tool") == RESCORE_TOOL for event in _events(copy_dir)[0]):
        return f"its ledger holds no event of {RESCORE_TOOL}"
    return None


# the parts of group_key shared by every run of record of a root, in the words of a message; the agent part is compared per agent
GROUP_PARTS = {"lists": "shared lists and data", "serving": "serving of the model endpoint", "runs_in_flight": "runs in flight",
               "granted": "processors and memory granted to the job"}
# the commit that a table tool keeps when the runs of record of a root fall into more than one group (tables_from_runs/ --commit)
_SELECTED_COMMIT = None


def select_commit(commit):
    """Keeps, in every root of runs of record that holds more than one group, the group of the runs played at commit (a prefix of at
    least 7 hex digits names it); None restores the rule that such a root stops the tool."""
    global _SELECTED_COMMIT
    _SELECTED_COMMIT = commit or None


def _grouped(order, metas, left_out):
    """The runs of record of one asset in one group: one value of each part of group_key over the runs that have it (a run without a
    model leaves serving open, a run off the wall clock leaves runs_in_flight and granted open) and one agent part per agent. More than one
    stops the tool and prints the groups with their commits, unless select_commit names a commit, whose group is kept and every
    other run left out and named."""
    def described(names):
        commits = sorted({str(commit_of(metas[n]) or "none")[:12] for n in names})
        agents = sorted({str(metas[n].get("agent")) for n in names})
        return f"commits {', '.join(commits)}, agents {', '.join(agents)}, {len(names)} runs ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})"

    keys = {name: group_key(metas[name]) for name in order if name not in left_out}
    conflicts = []
    for part, words in GROUP_PARTS.items():
        values = {}
        for name, key in keys.items():
            if key[part] is not None:
                values.setdefault(key[part], []).append(name)
        if len(values) > 1:
            conflicts.append((words, values))
    by_agent = {}
    for name, key in keys.items():
        by_agent.setdefault(metas[name].get("agent"), {}).setdefault(key["agent"], []).append(name)
    conflicts += [(f"own lists of the agent {agent}", values) for agent, values in sorted(by_agent.items(), key=lambda item: str(item[0])) if len(values) > 1]
    if not conflicts:
        return
    if _SELECTED_COMMIT is None:
        lines = [f"{part}: " + "; ".join(f"group {str(value)[:12]}: {described(names)}" for value, names in values.items()) for part, values in conflicts]
        raise SystemExit("the runs of record of this root fall into more than one group, and no table pools two: " + " | ".join(lines)
                         + "; name the commit whose runs the tables read (--commit SHA)")
    chosen = [name for name in keys if str(commit_of(metas[name]) or "").startswith(_SELECTED_COMMIT)]
    if not chosen:
        raise SystemExit(f"no run of record of this root played the commit {_SELECTED_COMMIT}")
    for part, values in conflicts:
        kept = {value for value, names in values.items() if set(names) & set(chosen)}
        for value, names in values.items():
            if value not in kept:
                for name in names:
                    left_out.setdefault(name, f"{part}: not those of the runs of the commit {_SELECTED_COMMIT} that the tables read")


def _registered(order, metas, dirs, left_out):
    """A root of runs of record: registered_refusal(), and for a copy that rescore_under_verifier_scope.py --own-scope wrote rescored_refusal() as
    well, a copy read leaving the run it came from out in its place; then the runs read share one group (_grouped), and a second
    finished run of one condition on a shared seed stops the tool unless an archive event in the ledger of one of them takes it out."""
    replaced = {}
    for name in order:
        meta = metas[name]
        reason = registered_refusal(meta, dirs[name])
        if reason is None and own_scope_copy(meta):
            source = meta["rescored"]["from"]
            reason = (rescored_refusal(meta, metas[source], dirs[name]) if source in metas
                      else f"the run it came from, {source}, is not under the root")
            if reason is None:
                replaced[source] = name
        if reason is not None:
            left_out[name] = reason
    for source, copy in replaced.items():
        left_out[source] = f"scored again under its own scope in {copy}, which is read in its place"
    _grouped([name for name in order if name not in left_out], metas, left_out)
    kept = [name for name in order if name not in left_out and metas[name].get("seeds")]
    for k, name in enumerate(kept):
        for other in kept[:k]:
            if other in left_out or name in left_out or condition_of(metas[other]) != condition_of(metas[name]):
                continue
            if not set(metas[name]["seeds"]) & set(metas[other]["seeds"]):
                continue
            if archived(dirs[name]) or archived(dirs[other]):
                out, kept_one = (name, other) if archived(dirs[name]) else (other, name)
                left_out[out] = f"archived by an event of its ledger, beside {kept_one} of the same registered condition on shared seeds"
                continue
            raise SystemExit(f"two finished runs of one registered condition share seeds, {other} and {name}, and no table pools them until "
                             "an archive event in the ledger of one of them (run_ledger.py event RUN_DIR archive WHY) takes it out")


def _development(order, metas, left_out):
    """A root of development runs: per agent, a run whose identity or GPU card is not that of the latest start of the agent is
    left out, a run without an identity keyed by its commit, then the later start counts."""
    def key(meta):
        return identity_of(meta) or ("commit", commit_of(meta))

    latest = {}
    for name in order:
        latest[metas[name].get("agent")] = name
    for name in order:
        newest = latest[metas[name].get("agent")]
        if name != newest and key(metas[name]) != key(metas[newest]):
            left_out[name] = (f"its identity {str(identity_of(metas[name]) or 'none recorded')[:12]} is not that of {newest}, the latest start of its "
                              f"agent ({metas[newest].get('started')}), and runs of two identities are never pooled")
        elif name != newest and card_of(metas[name]) != card_of(metas[newest]):
            left_out[name] = (f"it played on {ledger.card_text(card_of(metas[name]))} and {newest}, the latest start of its agent "
                              f"({metas[newest].get('started')}), on {ledger.card_text(card_of(metas[newest]))}, and runs of two GPU cards are never pooled")
    _later_start_counts(order, metas, left_out)


def pooling_refusals(run_dirs):
    """{run folder: reason} of the finished runs that the tables leave out so that no two runs pool that must not,
    per asset, since no rule reaches across assets (kind_of gives the kind of a run). A root that holds registered runs beside runs
    of another kind stops the tool and names one of each. In a root of registered runs a run is read only when registered_refusal()
    finds nothing, and a second finished run of one condition that shares a seed with the first stops the tool and names both
    until an archive event in the ledger of one of them takes it out. The runs of record of an asset share one group (_grouped). A copy that rescore_under_verifier_scope.py --own-scope
wrote is read in place of its run when rescored_refusal() finds nothing either. In a root of development runs, per agent, a run of another
    identity than the latest start of its agent is left out. Runs written before the identity, alone in a root, read by their
    commit as before. In both of the last two, of two finished runs of one condition on a shared seed the later start counts."""
    dirs = {run_dir.name: run_dir for run_dir in run_dirs}
    metas = {name: read_json(run_dir / "run.json", {}) for name, run_dir in dirs.items()}
    kinds = {}
    for name in sorted(metas):
        kinds.setdefault(kind_of(metas[name]), name)
    if "record" in kinds and len(kinds) > 1:
        other = next(kind for kind in kinds if kind != "record")
        raise SystemExit(f"a root holds runs of record and {other} runs, {kinds['record']} and {kinds[other]}, and no table tool reads such a root")
    left_out = {}
    by_asset = {}
    for name, meta in metas.items():
        by_asset.setdefault(asset_of(meta), []).append(name)
    for asset, names in sorted(by_asset.items()):
        order = sorted(names, key=lambda name: (metas[name].get("started") or "", name))
        if "record" in kinds:
            _registered(order, metas, dirs, left_out)
        elif "development" in kinds:
            _development(order, metas, left_out)
        else:
            _legacy(order, metas, left_out)
    return left_out


# {(function, resolved root): answer} while a call that reads_once decorates runs, None outside such a call
_READS = None


def reads_once(function):
    """Decorates a table tool so that, while one call of it runs, folders_read, pooling_left_out and root_kind compute their answer
    once per root: each pass reads every run.json and, on a root of runs of record, the ledgers. The answers
    are forgotten when the outermost decorated call returns, so none outlives the call that read the folders."""
    @functools.wraps(function)
    def call(*args, **kwargs):
        global _READS
        outermost = _READS is None
        if outermost:
            _READS = {}
        try:
            return function(*args, **kwargs)
        finally:
            if outermost:
                _READS = None
    return call


def _read_once(name, runs_root, compute):
    """The answer of compute for the root, kept for the decorated call that runs (reads_once), else computed now."""
    if _READS is None:
        return compute()
    key = (name, Path(runs_root).resolve())
    if key not in _READS:
        _READS[key] = compute()
    return _READS[key]


def root_kind(runs_root):
    """record, development, legacy or empty: the kind of the finished runs under a root that refused() lets through and of its runs
    of chosen starts, with development for a root that holds a stamped run. A run of chosen starts enters no cell and still gives
    its root its kind, so that a root of such runs alone opens its pages with the banner and never enters the results folder. A root
    that holds runs of record beside others stops the tool that asks, as pooling_refusals() does."""
    def compute():
        counted = [run_dir for run_dir in run_dirs(runs_root) if refused(run_dir) in (None, CHOSEN_STARTS)]
        kinds = {kind_of(read_json(run_dir / "run.json", {})) for run_dir in counted}
        if "record" in kinds and len(kinds) > 1:
            pooling_refusals(counted)  # stops and names one run of each kind
        return "empty" if not kinds else "record" if kinds == {"record"} else "development" if "development" in kinds else "legacy"
    return _read_once("root_kind", runs_root, compute)


def refuse_results_folder_of_kind(kind, *paths):
    """Stops a table tool that would write a page or a table of a development root into the results folder of the fork or below
    it: such a tool must name another folder."""
    if kind == "development":
        results = RESULTS
        for path in paths:
            resolved = Path(path).resolve()
            if resolved == results or results in resolved.parents:
                raise SystemExit(f"{path} lies in {results}, and the tables of development runs never enter the results folder: name another folder")


def refuse_results_folder(runs_root, *paths):
    """The kind of a root (root_kind), after refuse_results_folder_of_kind() for the paths a tool writes."""
    kind = root_kind(runs_root)
    refuse_results_folder_of_kind(kind, *paths)
    return kind


def head_lines(kind, record=None):
    """The lines a page or a table file of the tables opens with: the banner for a development root, and for a root of runs of
    record the commits of the runs read and the group key of each asset (record_of), so that a table of folders written before the
    identity never passes for one of record. A root of such folders, or an empty root, opens with nothing."""
    if kind == "development":
        return [DEVELOPMENT_BANNER, ""]
    if kind == "record":
        return [*(f"Runs of record of the asset {asset}, commits {', '.join(group['commits'])}, group {group['lists'][:12]}."
                  for asset, group in sorted((record or {}).items())), ""]
    return []


def pooling_left_out(runs_root):
    """{run folder: reason} of pooling_refusals over the finished run folders under the root that refused() lets through, the
    folders that excluded() lists with these reasons and that tables_from_runs/ names on stderr."""
    return dict(_read_once("pooling_left_out", runs_root,
                           lambda: pooling_refusals([run_dir for run_dir in run_dirs(runs_root) if refused(run_dir) is None])))


def chosen_starts_left_out(runs_root):
    """The finished run folders under the root that refused() leaves out as runs of chosen starts, in name order, which the table
    tools count in what they print (chosen_starts_line)."""
    return [run_dir.name for run_dir in run_dirs(runs_root) if refused(run_dir) == CHOSEN_STARTS]


def folders_read(runs_root):
    """The run folders under the root that tables_from_runs/ reads: every finished run folder that refused() lets through and that
    pooling_refusals() does not leave out."""
    def compute():
        left_out = pooling_left_out(runs_root)
        return [run_dir for run_dir in run_dirs(runs_root) if refused(run_dir) is None and run_dir.name not in left_out]
    return list(_read_once("folders_read", runs_root, compute))
