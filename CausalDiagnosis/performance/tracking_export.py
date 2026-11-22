"""The export of finished run folders and training folders to the experiment tracker, Weights and Biases, one tracked run per
folder (CausalDiagnosis/performance/README.md, the section on the tracking export).

    python -m CausalDiagnosis performance tracking-export FOLDER [FOLDER ...] [--dry-run]
    python -m CausalDiagnosis performance tracking-export --login

A FOLDER is a run folder whose ledger records it finished, the grid folder of a cluster job, which stands for every run folder in
it, or the folder of the checkpoints of one training. build_payload() is the one place that decides what leaves the machine: per
run folder its configuration from run.json and its ledger, its summary from scores.csv and from the stage profile, and one table
with a row per episode; per training folder the configuration of the trainer and the history of its loss from the trainer's own
state file. Every name the payload holds stands in the allow-list below, and nothing of the hidden folder of an episode, no
prompt or reply, no private file, no path and no job id is read into it. Each tracked run goes to the project of its own kind, a
run of record to the project of record, the only one that is ever opened, and any other run to the project of the development
runs, the kind of a run read from its own ledger. The dry run prints the payload whole, with the address of the service, the team
or user and the project of every tracked run, and sends nothing; so does a shell that declares no address or no team or user, or
that names one project for both kinds, and a run whose project is not named is printed and kept back. Otherwise the payload goes
to tracking_client.py, which uv runs with the one pinned version of the client in an environment of its own, with the variables
of client_environment() alone, and which takes nothing but the payload. The id of a tracked run is derived from the folder's own
record, so a folder exported again updates the same tracked run, and every key of a summary, a table or a history stands under
the kind of its run, record or development, so that no chart of a result holds a value of a development run."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from CausalDiagnosis.experiments.episode_runners.run_episodes.start_check import STAMP
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_folder, read_rows
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification.check_grid_complete import GRID
from CausalDiagnosis.experiments.table_definitions.training_variants import ADAPTER_MANIFEST, REGISTERED, TRAINING_SPLIT_KEY
from CausalDiagnosis.training.train_lora import SOURCE
from CausalDiagnosis.utils.json_files import read_json
from . import profile_table, run_files, tracking_client

RECORD = "record"  # the kind of a run of record, as RUN_KIND names it; every other run is a development run, STAMP of start_check.py
RUN, GRID_FOLDER, TRAINING = "run", "grid", "training"  # the kinds of a folder of the command
EPISODES = "episodes"  # the job type of the tracked run of a run folder, beside TRAINING for a training folder
ENTITY = "WANDB_ENTITY"  # the variable that names the team or user on the service, which the tree never names
# the variable that names the project of each kind of run: the project of record, the only one that is ever opened, and the
# project of the development runs, so that no development value reaches a project that is opened
PROJECTS = {RECORD: "WANDB_PROJECT", STAMP: "WANDB_PROJECT_DEVELOPMENT"}
# the variable that declares the address of the service's API, the client's own name for it: the export sends to no address that
# the person did not declare, and the tree names none
ADDRESS = "WANDB_BASE_URL"
START_CHECK = "start_check"  # the reasons of the start check that the ledger records at the start of a run, none for a registered run
ID_RECORD = ("run", "label", "started")  # the fields of ledger.json that name a run folder for good, which a resume keeps
FINISHED = "finished"  # the status of a run in its ledger once its run.json is written (ledger_file.py); before it the run may play
CLIENT = Path(tracking_client.__file__)
KEY_FILE = "NETRC"  # the variable of the client that names the file of its key, else .netrc of the home folder
# the variables of this process that the environment of the client keeps, every other one left out, so that no variable of the
# client that a shell holds, a sweep, a project, an id of a run, notes or tags among them, turns where or what it sends: the search
# path, the home and the temporary folder, the locale, the key and the file of the key, the proxy and the certificates of a
# connection, and every variable of uv, whose names start with UV_PREFIX
CLIENT_VARIABLES = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "WANDB_API_KEY", KEY_FILE, "HTTPS_PROXY", "https_proxy",
                    "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE")
UV_PREFIX = "UV_"
# what the export sets in the environment of the client: the recorder of the client's own errors and metrics off, which would
# otherwise send the paths of its files and the version of Python to the service, and the folders of its settings, its cache and
# its data inside the temporary folder of one start, so that no record names a folder of the home folder and nothing stays there
ERROR_REPORTING = {"WANDB_ERROR_REPORTING": "false"}
CLIENT_FOLDERS = {"WANDB_CONFIG_DIR": "config", "WANDB_CACHE_DIR": "cache", "WANDB_DATA_DIR": "data"}
# the state file that the trainer writes into each of its checkpoint folders checkpoint-<step> (transformers, TRAINER_STATE_NAME and
# PREFIX_CHECKPOINT_DIR), and the folders of one training: train_lora.py keeps the checkpoints of an adapter in checkpoints/<adapter>
# and the adapter with its manifest in adapters/<adapter> of the same training folder
TRAINER_STATE, CHECKPOINT, CHECKPOINTS, ADAPTERS = "trainer_state.json", "checkpoint-", "checkpoints", "adapters"
SHORT = 12  # the digits of a SHA-256 that name a cell, as cluster_jobs/functions.sh names a lock and a tree
TAG_LIMIT = 64  # the longest tag that the pinned client takes (wandb_settings.Settings.validate_run_tags)

# ----- the allow-list: every name that the payload holds, the one place that decides what of a folder leaves the machine. Each
# name stands here by itself, so that a column or a figure that a writer adds leaves only once it is named here -----
# the configuration of the tracked run of a run folder: per name, the file of the run folder and the keys that lead to its value
RUN_CONFIG = {
    "asset": (run_files.RUN_FILE, ("asset_id",)), "agent": (run_files.RUN_FILE, ("agent",)),
    "agent_label": (run_files.RUN_FILE, ("agent_label",)), "preset": (run_files.RUN_FILE, ("config_name",)),
    "clock_mode": (run_files.RUN_FILE, ("clock_mode",)), "model": (run_files.RUN_FILE, ("model",)),
    "served_name": (run_files.RUN_FILE, ("serving", "agent", "model")), "dtype": (run_files.RUN_FILE, ("serving", "agent", "dtype")),
    "engine": (run_files.RUN_FILE, ("serving", "agent", "engine")),
    "engine_version": (run_files.RUN_FILE, ("serving", "agent", "engine_version")),
    "launch_args_sha256": (run_files.RUN_FILE, ("serving", "agent", "launch_args_sha256")),
    "card": (run_files.RUN_FILE, ("serving", "agent", "gpu_name")), "card_mib": (run_files.RUN_FILE, ("serving", "agent", "gpu_mib")),
    "runs_in_flight": (run_files.RUN_FILE, ("runs_in_flight",)), "commit": (run_files.RUN_FILE, ("code", "commit")),
    "upload_manifest_sha256": (run_files.RUN_FILE, ("code", "upload_manifest_sha256")),
    # the processor of the node that started the run, where its ledger holds the record of the numeric configuration
    "node_kind": (ledger.LEDGER, ("numeric", "processor")),
}
RUN_DERIVED = ("kind", "cell")  # the kind of the run and the digest of its asset and cell, which the export derives
# the tags of a tracked run of a run folder beside its kind: the prefix, the name of RUN_CONFIG that it shows and how many characters
# of the value it keeps, a commit and the manifest of an upload by their first SHORT digits as tree_id of cluster_jobs/functions.sh
TAGS = (("asset", "asset", None), ("card", "card", None), ("commit", "commit", SHORT), ("upload", "upload_manifest_sha256", SHORT),
        ("preset", "preset", None), ("agent", "agent_label", None))
# the columns of the table of a run: the identity of an episode, its quantities of the stage profile, which are sent as numbers
# alone (the GPU note names files of the job and stays out), and the outcome columns of scores.csv
TABLE_IDENTITY = ("run", "episode", "seed", "agent", "config", "clock_mode", "model", "runs_in_flight", "card", "card_mib", "left_out",
                  "cell")
TABLE_PROFILE = (
    "total_s",
    # the seconds of the stages (stage_profile.SECONDS) and their shares of the total (stage_profile.SHARES)
    "creation_s", "cache_flush_s", "agent_part_s", "end_s", "model_s", "failed_model_s", "tool_service_s", "tool_service_cpu_s",
    "no_call_service_s", "server_starts_s", "env_tools_client_s", "diagnostics_client_s", "transport_hold_s", "rest_s",
    "share_creation", "share_cache_flush", "share_agent_part", "share_end", "share_model", "share_failed_model", "share_tool_service",
    "share_tool_service_cpu", "share_no_call_service", "share_server_starts", "share_env_tools_client", "share_diagnostics_client",
    "share_transport_hold", "share_rest",
    # the counts and the checks (stage_profile.COUNTS and CHECKS)
    "model_calls_without_span", "counted_calls", "counted_calls_without_end", "no_call_steps", "stamped_steps", "server_start_count",
    "env_tool_calls", "diagnostics_calls", "lines_in_model_spans", "rest_negative", "late_server_starts",
    # the thinking clock and the model calls (stage_profile.CLOCK and CALLS)
    "thinking_s", "thinking_cycles", "horizon_cycles", "think_seconds_per_cycle", "thinking_share_of_horizon", "credited_starts_s",
    "credited_calls_s", "uncredited_starts_s", "uncredited_calls_s", "failed_model_attempts", "uncharged_after_last_s",
    "model_calls", "failed_model_calls", "call_seconds_sum", "call_seconds_median", "input_tokens", "output_tokens",
    "output_tokens_per_call_second",
    # the GPU figures from the job's sampler and the card-seconds held alone (profile_table.GPU, the note left out)
    "sampler_rows", "gpu_busy_share", "gpu_energy_j", "gpu_period_s", "held_card_s",
)
# the outcome columns of scores.csv that the table of a run holds, one cell per episode. Left out are the columns that state a fact
# of the hidden state of an episode, of its scenario or of the oracle, and those from which such a fact follows alone or with a
# column that is sent, since a project on the service may be opened: graded_top1_n and graded_pass_n, the counts of the faulty
# components (episode_checks.py), separable, the separability of the start, cost_oracle, the cost of the oracle's answer,
# cost_normalized, which is cost over cost_oracle (episode_costs.py), reward, from which cost_normalized follows with the
# registered weights (episode_reward.py), commitment_on_inseparable and abstention_correct, each true on an inseparable start
# alone, success, which is pass or abstention_correct, failure_class, two of whose classes name an abstention on a separable start
# (failure_class.py), and graded_top1 and graded_pass, shares whose fractions give back the counts of the faulty components. The
# details of the verdict and of the grade are JSON and stay out with every cell that holds a JSON object or list.
TABLE_SCORES = ("pass", "diagnosis_all_correct", "faulty_set_right", "recovered", "unnecessary", "false_alarms", "cost",
                "downtime_total", "detection_delay", "repair_delay", "turns_to_correct_diagnosis", "abstained", "abstention_cites_reading",
                "end_reason", "time_used", "steps", "provenance_ok", "write_count_ok", "timing_ok", "ordering_ok", "read_gate",
                "judge_pass", "static_faulty_set_correct", "static_repairs_match_oracle")
MEAN_SCORES = TABLE_SCORES  # the outcome columns whose mean over the episodes of a run its summary holds
# the columns of scores.csv that the summary of a run sums over its episodes, each over the episodes whose cell holds a number
SCORE_TOTALS = ("wall_s", "input_tokens", "output_tokens", "cached_tokens")
# the summary of a run, five sections: the summary of a cell (cells.cell_summary) over the run's own episodes and over its cell,
# a * standing for one key of the stages, the checks and the reasons for an episode left out that the profile declares; the view
# of its job (jobs.job_view), its card-seconds and both rates; the mean of every outcome column of MEAN_SCORES; and the totals of
# SCORE_TOTALS
RUN_PARTS = ("episodes", "kept", "left_out/*", "checks/*/failing", "checks/*/checked", "seconds/*/sum", "seconds/*/share",
             "seconds/*/median", "gpu/sampler_rows", "gpu/gpu_busy_share", "gpu/gpu_energy_j_per_episode",
             "gpu/gpu_energy_j_per_passed_episode", "cost/pass_rate", "cost/held_card_s_per_episode",
             "cost/held_card_s_per_passed_episode")
CELL_PARTS = (*RUN_PARTS, "jobs_view/card_s_per_episode", "jobs_view/card_s_per_passed_episode", "jobs_view/window_s",
              "jobs_view/ending_per_card_hour", "jobs_view/shares_per_card_hour")
JOB_PARTS = ("window_s", "episodes_ending_inside", "episode_shares_inside", "ending_per_s", "shares_per_s", "ending_per_card_hour",
             "shares_per_card_hour", "card_s_per_episode", "card_s_per_passed_episode")
# a training folder: the configuration of the trainer, its progress and the history of its loss, from the trainer's state file
TRAINER_CONFIG = ("train_batch_size", "num_train_epochs", "max_steps", "logging_steps", "save_steps", "eval_steps")
TRAINER_SUMMARY = ("global_step", "epoch")
TRAINER_HISTORY = ("loss", "epoch")
NESTED = object()  # a cell of scores.csv that holds a JSON object or list, a detail of the verifier, whose column is not sent


def value_at(record, keys):
    """The value under a path of keys of a nested record, None where a key is missing."""
    for key in keys:
        record = record.get(key) if isinstance(record, dict) else None
    return record


def scalar(value):
    """A value as a configuration or an identity sends it: a text, a number, a flag or None; a nested value is not sent."""
    return value if value is None or isinstance(value, (str, int, float, bool)) else None


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def number(value):
    """A quantity of the profile as the table sends it: a finite number, else None, so that no text of a column reaches it."""
    return value if is_number(value) else None


def cell_value(text):
    """A cell of scores.csv as the payload holds it: None when empty, a flag, an integer or a finite number, the text otherwise, and
    NESTED for a JSON object or list, also after blanks, which a JSON reader passes over."""
    if text is None or text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    if text.lstrip().startswith(("{", "[")):
        return NESTED
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        return text
    return value if math.isfinite(value) else text


def short_digest(value):
    return ledger.canonical_sha256(value)[:SHORT]


def flat(record, prefix=""):
    """The leaves of a nested dict as {"a/b/c": leaf}."""
    if not isinstance(record, dict):
        return {prefix[:-1]: record}
    return {path: leaf for key, item in record.items() for path, leaf in flat(item, f"{prefix}{key}/").items()}


def part_of(path, parts):
    """Whether a path a/b/c is named by one of parts, a * of a part standing for any one key."""
    keys = path.split("/")
    return any(len(keys) == len(named) and all(want in ("*", key) for want, key in zip(named, keys))
               for named in (part.split("/") for part in parts))


def picked(record, parts):
    """The numbers of a nested summary whose path one of parts names."""
    return {path: leaf for path, leaf in flat(record).items() if is_number(leaf) and part_of(path, parts)}


def sectioned(kind, section, values):
    return {f"{kind}/{section}/{name}": value for name, value in values.items()}


def run_kind(record):
    """The kind of a run from its own ledger: record when the ledger records the start check of the run without a reason, a
    registered run, and development otherwise, a ledger that records no start check among them."""
    reasons = record.get(START_CHECK)
    return RECORD if isinstance(reasons, list) and not reasons else STAMP


def folder_kind(folder):
    """RUN for a run folder, with run.json or of a run that did not finish, GRID_FOLDER for a grid folder grid_<job id> of a cluster
    job with run folders in it, TRAINING for the checkpoint folder of a training, which holds the record of what its checkpoints are
    made from, else None."""
    folder = Path(folder)
    if (folder / run_files.RUN_FILE).is_file() or run_files.is_unfinished(folder):
        return RUN
    if GRID.match(folder.name) and folder.is_dir() and profile_table.grid_runs(folder):
        return GRID_FOLDER
    return TRAINING if (folder / SOURCE).is_file() else None


def run_folders(folders):
    """The run folders that the folders of the command stand for, given or in a given grid folder, those of runs that did not
    finish among them, so that each is named rather than passed over, each once, in order."""
    found = []
    for folder in map(run_files.absolute, folders):
        kind = folder_kind(folder)
        found += [folder] if kind == RUN else [run_files.absolute(path) for path in profile_table.grid_runs(folder)] if kind == GRID_FOLDER else []
    return list(dict.fromkeys(found))


def last_state(folder):
    """The trainer's state file of the latest checkpoint of a training folder, of the checkpoint folder with the highest step that
    holds one, the checkpoint that the trainer takes up after a requeue; None without one."""
    found = [(int(path.name[len(CHECKPOINT):]), path / TRAINER_STATE) for path in Path(folder).glob(f"{CHECKPOINT}*")
             if path.name[len(CHECKPOINT):].isdigit() and (path / TRAINER_STATE).is_file()]
    return max(found)[1] if found else None


def refusals(folders):
    """Why the folders of the command cannot be exported, one line each, [] when every one can: a folder of none of the three kinds,
    a run folder whose ledger lacks the record of its id or does not record the run finished, since a run that may still play
    would show values that its end changes, a training folder without the trainer's state file, and two run folders of one name,
    which would be one tracked run."""
    out = []
    for folder in map(Path, folders):
        kind = folder_kind(folder)
        if kind is None:
            out.append(f"{folder}: no {run_files.RUN_FILE}, no grid folder grid_<job id> with run folders and no {SOURCE} of a training")
        elif kind == TRAINING and last_state(folder) is None:
            out.append(f"{folder}: no checkpoint folder of the training holds the trainer's {TRAINER_STATE}")
    names = {}
    for run in run_folders(folders):
        record = read_json(run / ledger.LEDGER, {}) or {}
        missing = [key for key in ID_RECORD if not record.get(key)]
        if missing:
            out.append(f"{run}: its {ledger.LEDGER} lacks {', '.join(missing)}, from which the id of its tracked run is derived")
        if record.get("status") != FINISHED:
            out.append(f"{run}: its {ledger.LEDGER} records the status {record.get('status')!r}, not {FINISHED!r}, so the run may still "
                       "play; name the finished run folders alone")
        names.setdefault(run.name, []).append(run)
    out += [f"the run folders {', '.join(map(str, runs))} share one name, which names one tracked run" for runs in names.values() if len(runs) > 1]
    return out


def tags_of(kind, config, notes):
    """The tags of a tracked run: its kind, then each name of TAGS that its configuration holds, a tag longer than the client
    takes left out with a note."""
    tags = [kind, *(f"{prefix}:{str(config[name])[:kept]}" for prefix, name, kept in TAGS if config.get(name) not in (None, ""))]
    notes += [f"the tag {tag} is longer than the {TAG_LIMIT} characters the client takes and is not sent" for tag in tags if len(tag) > TAG_LIMIT]
    return [tag for tag in tags if len(tag) <= TAG_LIMIT]


def score_columns(scores):
    """The outcome columns of scores.csv that the table of a run holds, those that it has with no JSON object or list in a cell."""
    return [name for name in TABLE_SCORES if any(name in row for row in scores) and all(cell_value(row.get(name)) is not NESTED for row in scores)]


def numbers_of(scores, name):
    """The filled cells of a column of scores.csv when every one of them is a flag or a number, else None."""
    filled = [value for value in (cell_value(row.get(name)) for row in scores) if value is not None]
    return filled if filled and all(is_number(value) for value in filled) else None


def table_cell(name, row, score, cell):
    """The value of one column of the table for the profile row and the scores.csv row of an episode."""
    if name == "cell":
        return cell
    if name == "seed":
        return cell_value(row.get("seed"))
    if name in TABLE_IDENTITY:
        return scalar(row.get(name))
    if name in TABLE_PROFILE:
        return number(row.get(name))
    return cell_value(score.get(name))


def profiled(paths):
    """(rows, cells, jobs) of the stage profile of the run folders, the one call of the profile in this module: the rows of their
    episodes, the summary of every cell that one of them plays and the view of every job whose grid folder holds one of them. The
    values that build returns after these three, the figures of the calls joined to the engine's export and the summary of that
    join, are not sent, since the export gives the profile no export of the engine."""
    return profile_table.build(paths)[:3]


def run_entry(path, rows, cell, view, projects):
    """(entry, notes) of one run folder from its profile rows, the summary of its cell, the view of its job (None without one) and
    the project of each kind."""
    meta, record, notes = run_files.run_meta(path), read_json(path / ledger.LEDGER, {}) or {}, []
    kind = run_kind(record)
    digest = short_digest({"asset": meta.get("asset_id"), "cell": cell["key"]})
    config = {name: scalar(value_at(meta if source == run_files.RUN_FILE else record, keys)) for name, (source, keys) in RUN_CONFIG.items()}
    config.update(kind=kind, cell=digest)
    scores = read_rows(path)
    (own,) = profiled([path])[1]  # the summary of the run's cell over the run's own episodes alone
    summary = {**sectioned(kind, "run", picked(own, RUN_PARTS)), **sectioned(kind, "cell", picked(cell, CELL_PARTS)),
               **sectioned(kind, "job", picked(view or {}, JOB_PARTS)),
               **sectioned(kind, "outcome", {name: sum(found) / len(found) for name in MEAN_SCORES if (found := numbers_of(scores, name))}),
               **sectioned(kind, "totals", {name: sum(found) for name in SCORE_TOTALS if (found := numbers_of(scores, name))})}
    if view is None:
        notes.append("no view of a job that played the cell of the run alone, so the summary holds no rate of a job")
    columns = list(dict.fromkeys([*TABLE_IDENTITY, *TABLE_PROFILE, *score_columns(scores)]))
    by_episode = {episode_folder(path, row).name: row for row in scores}
    table = {"key": f"{kind}/{EPISODES}", "columns": columns,
             "rows": [[table_cell(name, row, by_episode.get(row["episode"]) or {}, digest) for name in columns] for row in rows]}
    entry = {"id": ledger.canonical_sha256({key: record[key] for key in ID_RECORD}), "name": path.name, "project": projects.get(kind),
             "group": f"{kind}:{digest}", "job_type": EPISODES, "tags": tags_of(kind, config, notes), "config": config,
             "summary": summary, "table": table, "history": []}
    return entry, notes


def run_entries(paths, projects):
    """(entries, notes) of the run folders of one asset: the profile of all of them at once, so that a cell and a job are each read
    over every run folder given of them, then each run folder with the summary of its cell and the view of its job."""
    rows, summaries, views = profiled(paths)
    entries, notes = [], {}
    for path in paths:
        cell = next(summary for summary in summaries if path.name in summary["runs"])
        grid = run_files.grid_of(path)[0]
        view = next((view for view in views if grid is not None and view["grid"] == grid.name and view.get("cell") == cell["cell"]), None)
        entry, notes[path.name] = run_entry(path, [row for row in rows if row["run"] == path.name], cell, view, projects)
        entries.append(entry)
    return entries, notes


def training_kind(folder):
    """record only when the adapter of the training has its manifest beside it, which train_lora.py writes after the last step, and
    the manifest records, each key present, no stamp, no smoke run and the registered split of the folds, on which a run of record
    plays an adapter; development otherwise, so that a training not shown to be of record is never read as one."""
    folder = Path(folder)
    manifest = read_json(folder.parent.parent / ADAPTERS / folder.name / ADAPTER_MANIFEST) if folder.parent.name == CHECKPOINTS else None
    shown = isinstance(manifest, dict) and all(key in manifest for key in ("stamp", "smoke", TRAINING_SPLIT_KEY))
    of_record = shown and manifest["stamp"] is None and manifest["smoke"] is False and manifest[TRAINING_SPLIT_KEY] == REGISTERED
    return RECORD if of_record else STAMP


def training_entry(folder, projects):
    """(entry, notes) of one training folder: the configuration of the trainer and its progress from the state file of its latest
    checkpoint, every row of its history that holds a loss, at the trainer's own step, and the project of its kind."""
    folder = Path(folder)
    state, kind = read_json(last_state(folder)), training_kind(folder)
    history = [{"step": row["step"], "values": {f"{kind}/{name}": row[name] for name in TRAINER_HISTORY if is_number(row.get(name))}}
               for row in state.get("log_history") or [] if is_number(row.get("loss")) and isinstance(row.get("step"), int)]
    entry = {"id": ledger.canonical_sha256({"training": folder.name, "source": read_json(folder / SOURCE)}), "name": folder.name,
             "project": projects.get(kind), "group": f"{kind}:{TRAINING}", "job_type": TRAINING, "tags": [kind, TRAINING],
             "config": {"kind": kind, **{name: scalar(state.get(name)) for name in TRAINER_CONFIG}},
             "summary": {f"{kind}/{name}": state[name] for name in TRAINER_SUMMARY if is_number(state.get(name))}, "table": None,
             "history": history}
    return entry, []


def place_of(environ):
    """The place on the service that the environment names: the address of its API as the client takes it, without a closing slash
    (wandb_settings.Settings.validate_base_url), the team or user, and the project of each kind, each None when unset."""
    return {"address": (environ.get(ADDRESS) or "").rstrip("/") or None, "entity": environ.get(ENTITY) or None,
            "projects": {kind: environ.get(name) or None for kind, name in PROJECTS.items()}}


def place_refusals(place):
    """Why nothing may be sent to the place, one line each, [] when it may: no declared address, so that the client never falls back
    on an address of its own, no team or user, and one project named for both kinds, compared without case, which would put the
    development runs into the project that is opened."""
    out = []
    if not place["address"]:
        out.append(f"{ADDRESS} is not set, which declares the address of the service")
    if not place["entity"]:
        out.append(f"{ENTITY} is not set, which names the team or user on the service")
    record, development = (place["projects"][kind] for kind in (RECORD, STAMP))
    if record and development and record.casefold() == development.casefold():
        out.append(f"{PROJECTS[RECORD]} and {PROJECTS[STAMP]} name one project, {record}, which would hold the runs of both kinds")
    return out


def build_payload(folders, place):
    """(payload, notes): the payload of the export, the one place that decides what leaves the machine, and per tracked run the
    notes on what of its folder the payload leaves out. The payload is {"address", "entity", "runs"}, the address and the team or
    user from place, with one entry per run folder, given or in a given grid folder, then per training folder, and each entry
    names the project of its own kind from place (place_of), None when the variable of that project is unset. The run folders are
    profiled together per asset, so that no cell and no job view pools the runs of two assets."""
    groups = {}
    for path in run_folders(folders):
        groups.setdefault(run_files.run_meta(path).get("asset_id"), []).append(path)
    entries, notes = [], {}
    for paths in groups.values():
        found, found_notes = run_entries(paths, place["projects"])
        entries += found
        notes.update(found_notes)
    for folder in (run_files.absolute(folder) for folder in folders if folder_kind(folder) == TRAINING):
        entry, notes[folder.name] = training_entry(folder, place["projects"])
        entries.append(entry)
    return {"address": place["address"], "entity": place["entity"], "runs": entries}, notes


def handed_part(payload):
    """The part of the payload that the client is handed: the entries whose project is named, so that each tracked run goes to the
    project of its own kind or nowhere."""
    return {**payload, "runs": [entry for entry in payload["runs"] if entry["project"]]}


def render(payload, notes):
    """The payload whole as readable lines: the address and the team or user, then per tracked run its id, name, project, group,
    job type, tags, configuration and summary, every row of its table and of its history, each value as the JSON it is sent as, and
    the notes on what its folder keeps back."""
    shown = json.dumps
    lines = [f"the payload of the tracking export: {len(payload['runs'])} tracked runs, sent with {', '.join(tracking_client.pinned())}",
             f"address of the service: {shown(payload['address'])}", f"team or user on the service: {shown(payload['entity'])}",
             "settings of the client: " + ", ".join(f"{name}={shown(value)}" for name, value in tracking_client.SETTINGS.items())]
    for number_of, entry in enumerate(payload["runs"], 1):
        project = shown(entry["project"]) if entry["project"] else f"none, since {PROJECTS.get(entry['config']['kind'], 'no variable')} is not set, so it is kept back"
        lines += ["", f"tracked run {number_of} of {len(payload['runs'])}: {entry['name']}", f"  id {entry['id']}", f"  project {project}",
                  f"  group {entry['group']}, job type {entry['job_type']}", f"  tags {', '.join(entry['tags'])}", "  config"]
        lines += [f"    {name} = {shown(value)}" for name, value in entry["config"].items()]
        lines += ["  summary", *(f"    {name} = {shown(value)}" for name, value in entry["summary"].items())]
        table = entry["table"]
        if table is not None:
            lines.append(f"  table {table['key']}: {len(table['rows'])} rows of {len(table['columns'])} columns")
            lines += [f"    row {n}: " + ", ".join(f"{name}={shown(value)}" for name, value in zip(table["columns"], row))
                      for n, row in enumerate(table["rows"], 1)]
        lines.append(f"  history: {len(entry['history'])} rows")
        lines += [f"    step {row['step']}: " + ", ".join(f"{name}={shown(value)}" for name, value in row["values"].items())
                  for row in entry["history"]]
        lines += [f"  kept back: {note}" for note in notes.get(entry["name"], [])]
    return "\n".join(lines)


def client_uv():
    """The uv that starts the client: UV_BIN, which env.sh points at the uv of the tree, else uv on PATH; None without one."""
    return shutil.which(os.environ.get("UV_BIN") or "uv")


def client_environment(folder):
    """The environment of the client and of its login: the variables of CLIENT_VARIABLES and of uv that this process holds and no
    other, PYTHONPATH of a job's shell, which points at the tree, among those left out, with ERROR_REPORTING and the folders of
    CLIENT_FOLDERS inside folder, the temporary folder of one start."""
    kept = {name: value for name, value in os.environ.items() if name in CLIENT_VARIABLES or name.startswith(UV_PREFIX)}
    return {**kept, **ERROR_REPORTING, **{name: str(Path(folder) / sub) for name, sub in CLIENT_FOLDERS.items()}}


def key_file():
    """The file in which the login of the client keeps the key, as the client finds it: the file that KEY_FILE names, else .netrc
    of the home folder, else _netrc of the home folder when only that one exists."""
    if os.environ.get(KEY_FILE):
        return Path(os.environ[KEY_FILE]).expanduser()
    unix, windows = Path("~/.netrc").expanduser(), Path("~/_netrc").expanduser()
    return windows if windows.exists() and not unix.exists() else unix


def open_key_file():
    """The key file when it exists and its group or others may read or write it, else None. The login of the client creates the
    file readable by its owner alone but keeps the mode of a file that exists, so a send or a login stops before it starts."""
    path = key_file()
    return path if path.is_file() and path.stat().st_mode & 0o077 else None


def refuse_open_key_file(path):
    print(f"the key file {path} is open to group or others: run chmod 600 on it, then start again", file=sys.stderr)
    return 2


def send(payload, uv):
    """Starts the client through uv with the payload as its one input, in a temporary folder of its own, with client_environment()
    of that folder, and returns its exit status."""
    with tempfile.TemporaryDirectory(prefix="tracking_export_") as folder:
        path = Path(folder) / "payload.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        command = [uv, "run", "--python", sys.executable, "--script", str(CLIENT), str(path)]
        return subprocess.run(command, cwd=folder, env=client_environment(folder)).returncode


def login(uv, address):
    """Runs the login of the pinned client through uv for the declared address, which asks for the key and keeps it in the key
    file under the host of that address, with client_environment() of a temporary folder, and returns its exit status."""
    command = [uv, "tool", "run", "--python", sys.executable, "--from", tracking_client.pinned()[0], tracking_client.COMMAND, "login",
               "--host", address]
    with tempfile.TemporaryDirectory(prefix="tracking_login_") as folder:
        return subprocess.run(command, cwd=folder, env=client_environment(folder)).returncode


def no_uv():
    print("no uv to start the client with: UV_BIN names the uv of the tree in env.sh, else uv stands on PATH", file=sys.stderr)
    return 2


def parser():
    ap = argparse.ArgumentParser(description="The export of finished run folders and training folders to Weights and Biases; the module "
                                             "docstring and CausalDiagnosis/performance/README.md hold what it sends and what it never sends.")
    ap.add_argument("folders", nargs="*", help="run folders, grid folders of cluster jobs and checkpoint folders of trainings")
    ap.add_argument("--dry-run", action="store_true", help="print the whole payload and send nothing, the first thing to run")
    ap.add_argument("--login", action="store_true", help="run the login of the pinned client once, which keeps the key in its own file")
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.login:
        address = place_of(os.environ)["address"]
        if not address:
            print(f"{ADDRESS} is not set, which declares the address of the service whose key the login keeps", file=sys.stderr)
            return 2
        if path := open_key_file():
            return refuse_open_key_file(path)
        uv = client_uv()
        return login(uv, address) if uv else no_uv()
    if not a.folders:
        ap.error("name at least one folder, or --login")
    refused = refusals(a.folders)
    if refused:
        print("\n".join(refused), file=sys.stderr)
        return 2
    place = place_of(os.environ)
    payload, notes = build_payload(a.folders, place)
    if a.dry_run:
        print(render(payload, notes))
        print("".join(f"\na send would be refused: {reason}" for reason in place_refusals(place)) + "\na dry run: nothing is sent")
        return 0
    if place_refusals(place):
        print(render(payload, notes))
        print("\n" + "\n".join(place_refusals(place)) + "\nso nothing is sent", file=sys.stderr)
        return tracking_client.NOTHING_SENT
    handed = handed_part(payload)
    kept_back = [entry for entry in payload["runs"] if not entry["project"]]
    if kept_back:
        print(render({**payload, "runs": kept_back}, notes))
        print(f"\n{len(kept_back)} tracked runs are kept back, since the variable of their project is not set", file=sys.stderr)
    if not handed["runs"]:
        return tracking_client.NOTHING_SENT
    if path := open_key_file():
        return refuse_open_key_file(path)
    uv = client_uv()
    if not uv:
        return no_uv()
    places = ", ".join(f"{handed['address']} {handed['entity']}/{name}" for name in sorted({entry["project"] for entry in handed["runs"]}))
    print(f"sending {len(handed['runs'])} tracked runs to {places} with {', '.join(tracking_client.pinned())}")
    status = send(handed, uv)
    if status == tracking_client.NOTHING_SENT:
        print(render(handed, notes))
    return status


if __name__ == "__main__":
    sys.exit(main())
