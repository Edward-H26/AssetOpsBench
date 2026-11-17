import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_dirs
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.experiments.run_verification.public_record import add_private, public_argv, public_path, public_value, read_private, resolve_recorded_path
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AS_RELEASED, TRAINING_VARIANT_KEY
from . import hashing, serving_endpoints, run_data, code_identity


LEDGER = "ledger.json"
LOCAL_EVENTS = "local_events.json"  # beside the ledger of a synced copy, which keeps the cluster's ledger byte for byte
FORMAT = 1
# the entries of the data record that a run reads as they were recorded (data_files): the recordings and the fetched scenario
# records. Every other entry is a generated table, the fitted models or a table that the package of the asset declares under
# TABLES, which the tree computes from the recordings and its own code
RECORDED_DATA = ("facts", "scenarios")
# what a run reads besides its arguments or what names where it came from, a fixed list so that no key enters the ledger
ENV_NAMES = ("HYD_FACTS_PATH", "HYD_GRID_FILE", "HYD_GRID_LINE", "HYD_ENDPOINT_FILE", "HYD_JUDGE_ENDPOINT_FILE", "MODEL",
             "SERVED_NAME", "JUDGE_MODEL", "JUDGE_SERVED_NAME", "FORK", "GRID_FILE", "GRID_STARTS", "GRID_SEEDS",
             "GRID_FIRST_SEED", "GRID_EXTRA", "GRID_PARALLEL", "GRID_REPLICATES", "GRID_REFERENCE",
             *(name for _framework, name in frameworks.names("JOB_ENVIRONMENT")))
JOB_ENV = {"slurm_job_id": "SLURM_JOB_ID", "slurm_array_job_id": "SLURM_ARRAY_JOB_ID",
           "slurm_array_task_id": "SLURM_ARRAY_TASK_ID", "slurm_restart_count": "SLURM_RESTART_COUNT"}


def now():
    return datetime.now().isoformat(timespec="seconds")


def run_kind(meta):
    """The kind of run that a framework declares under RUN_KINDS and a run.json records under kind, the object its entry names,
    or None for a run of the driver, whose run.json records no declared kind."""
    kind = meta.get("kind")
    return frameworks.code("RUN_KINDS", kind) if isinstance(kind, str) and kind in frameworks.run_kinds() else None


def load(run_dir):
    """The ledger of a run folder, with the events that a laptop wrote into local_events.json beside the ledger of a synced copy
    among its own, in time order."""
    ledger = read_json(Path(run_dir) / LEDGER)
    local = read_json(Path(run_dir) / LOCAL_EVENTS, None)
    if ledger is not None and local:
        ledger["events"] = sorted([*ledger.get("events", []), *local], key=lambda event: event.get("at") or "")
    return ledger


def _write(run_dir, ledger):
    ledger["updated"] = now()
    write_json(Path(run_dir) / LEDGER, public_value(ledger))


def private_start(run_dir, command, job, served_records, data):
    """Records in the private file of the run what a start holds that its public ledger does not: the job and its host, the
    command as the process saw it, the endpoint files as read and the paths of the data files that finish() hashes again."""
    return add_private(run_dir, "starts", {"at": now(), "job": job, "command": command, "served": served_records,
                                           "data": {name: entry.get("path") for name, entry in (data or {}).items()}})


def public_command(command):
    """The command of a start as the public ledger holds it: the program, the arguments and the grid file by the path rule of
    public_record, the environment of ENV_NAMES without FORK and with every value by that rule, the cwd only inside the tree."""
    cwd = command.get("cwd")
    inside = cwd is not None and Path(cwd).resolve().is_relative_to(run_data.ROOT.resolve())
    return {"python": public_path(command.get("python")), "program": public_path(command.get("program")),
            "argv": public_argv(command.get("argv") or []), "cwd": public_path(cwd) if inside else None,
            "grid_file": public_path(command.get("grid_file")), "grid_file_sha256": command.get("grid_file_sha256"),
            "grid_line": command.get("grid_line"),
            "env": {name: public_argv([value])[0] for name, value in (command.get("env") or {}).items() if name != "FORK"}}


def run_fields(meta, serve=serving_endpoints.served):
    """The fields of a ledger that a run of the rig driver and a run of a declared kind (run_kind) fill from different parts of
    run.json: the labels, the model and the judge as serve() finds them, the data and the configuration."""
    kind = run_kind(meta)
    if kind is not None:
        return kind.fields(meta, serve)
    return {"label": meta["label"], "agent": meta["agent"], "agent_label": meta["agent_label"],
            "model": serve(meta.get("model"), meta.get("endpoint"), "HYD_ENDPOINT_FILE"),
            "judge": serve(meta.get("judge_model"), meta.get("judge_endpoint"), "HYD_JUDGE_ENDPOINT_FILE"),
            "data": run_data.data_files(meta["facts_path"], meta.get("asset_id"), anchored=(meta.get("config") or {}).get("anchors") is not None),
            "configuration": {"config_name": meta.get("config_name"), "config_sha256": hashing.canonical_sha256(meta.get("config")),
                              "mode": meta.get("mode"), "prices": meta.get("prices"), "price_table_source": meta.get("price_table_source"),
                              # the leaderboard reads it from the ledger of a run in progress, which has no run.json yet
                              TRAINING_VARIANT_KEY: meta.get(TRAINING_VARIANT_KEY) or AS_RELEASED},
            # with the start state and the scenarios of a run that names them, which a job that takes the run up again must name
            # alike (resume_refusal) and by which the leaderboard knows a run of chosen starts in progress (pooling_rules.chosen_starts)
            "seeds": {**{key: meta.get(key) for key in ("starts", "master_seed", "seed_pool", "seeds", "episodes_planned")},
                      **{key: meta[key] for key in ("start_state", "scenario_ids") if meta.get(key) is not None}}}


def _record(run_dir, fields, started, command, job, code, after=None):
    record = {"format": FORMAT, "run": Path(run_dir).name, "label": fields["label"], "agent": fields["agent"],
              "agent_label": fields["agent_label"], "status": "running", "started": started, "updated": None, "ended": None}
    if after:
        record["written_after_run"] = after
    record.update({"command": command, "job": job, "code": code,
                   **{key: fields[key] for key in ("model", "judge", "data", "configuration", "seeds")},
                   "episodes": {}, "run_files": {}, "events": []})
    return record


# what a ledger records of a run, which a driver of another benchmark gives start() in place of the run.json of the rig's driver
START_FIELDS = ("label", "agent", "agent_label", "model", "judge", "data", "configuration", "seeds")


def start(run_dir, meta, argv, fields=None):
    """Writes the ledger of a run before its first episode, from the run.json fields that run_meta built, the command line
    and the job's environment, with the status running until finish(). CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch passes the grid file and
    the grid line, the index of the line among the file's lines that are neither comments nor blank, or reference for the
    reference policies of its first task. A driver of another benchmark passes the fields of START_FIELDS itself, which then
    stand in place of those that run_fields() reads from the run.json of the rig's driver or of the reproduction, and a
    missing field is refused by name before anything is written."""
    if fields is not None:
        missing = [name for name in START_FIELDS if name not in fields]
        if missing:
            raise ValueError(f"the start fields of a ledger lack {', '.join(missing)}")
    grid_file = os.environ.get("HYD_GRID_FILE")
    command = {"python": sys.executable, "program": sys.argv[0], "argv": [str(arg) for arg in argv], "cwd": os.getcwd(),
               "grid_file": grid_file, "grid_file_sha256": hashing.sha256_of(grid_file) if grid_file else None,
               "grid_line": os.environ.get("HYD_GRID_LINE"),
               "env": {name: os.environ[name] for name in ENV_NAMES if name in os.environ}}
    job = {**{field: os.environ.get(variable) for field, variable in JOB_ENV.items()}, "host": platform.node()}
    fields = run_fields(meta) if fields is None else fields
    private_start(run_dir, command, job, {role: fields.get(role) for role in ("model", "judge")}, fields.get("data"))
    ledger = _record(run_dir, fields, meta["started"], public_command(command), None, code_identity.code_files(meta))
    # the reasons of the start check of the job that starts the run, none for a registered run, which a later job that takes the
    # run up again must match (resume_refusal)
    ledger["start_check"] = list(meta.get("stamp_reasons") or [])
    _write(run_dir, ledger)
    return ledger


def add_episode(run_dir, name, row):
    """Records a finished episode: its seed, its start combination as scores.csv writes it, and the hash of every file of
    its folder. Returns False for a run without a ledger."""
    if not (Path(run_dir) / LEDGER).exists():
        return False
    ledger = read_json(Path(run_dir) / LEDGER)  # the file alone: these rewrite it, and local events stay beside it
    ledger["episodes"][name] = {"seed": row.get("seed"), "block": row.get("block"), "files": hashing.file_hashes(Path(run_dir) / name)}
    _write(run_dir, ledger)
    return True


INTERRUPTED = "interrupted"  # the folder of a run taken up again that holds what a cut job left of its unfinished episodes
RESUME_TOOL = "run_episodes.py --resume"  # the tool of the event of a job that took a run up again (record_resume)


def stopped(record):
    """Whether a ledger records the stop of its run after harness errors in a row, a run that is played again as a whole."""
    from CausalDiagnosis.experiments.episode_runners.attempt_error_classes import STOP_EVENT  # here and not at the top: failures belongs to the rig's driver, and the reproduction imports this module

    return any(event.get("what") == STOP_EVENT for event in record.get("events") or [])


def in_progress(run_dir):
    """Whether a run folder holds a run that a job left partly played and a later job may take up: a ledger with the status
    running, no run.json, and no stop after harness errors in a row (stopped)."""
    run_dir = Path(run_dir)
    if (run_dir / "run.json").exists() or not (run_dir / LEDGER).is_file():
        return False
    record = read_json(run_dir / LEDGER, {}) or {}
    return record.get("status") == "running" and not stopped(record)


def kept_episodes(run_dir, names, rows):
    """The planned episodes, in plan order, that a run taken up again keeps: the longest prefix of names whose episodes are
    finished, each with its row among rows (a mapping from an episode to its row of scores.csv), its ledger record, which the
    driver writes after the row, and the files of its folder equal to that record."""
    record = read_json(Path(run_dir) / LEDGER)
    kept = []
    for name in names:
        episode = record["episodes"].get(name)
        folder = Path(run_dir) / name
        if name not in rows or episode is None or not folder.is_dir() or hashing.file_hashes(folder) != episode["files"]:
            break
        kept.append(name)
    return kept


RESUMED_FIELDS = ("agent", "agent_label", "configuration", "seeds")  # the fields of a ledger that a command taking a run up must equal


def resume_refusal(run_dir, meta, reasons):
    """Why the command whose run.json fields are meta, with the reasons of its start check, may not take up the run in progress
    of run_dir, None when it plays the same run: the agent, its label, the configuration and the seeds that the ledger records,
    the hashes of its recorded data (RECORDED_DATA), the names of the model and of the judge, every code file that the ledger
    hashed, and the start check, which must register the run when it registered the first job's part and stamp it when it stamped
    that part, so that no part played outside a registered run enters one. A generated table is left out, since the tree computes
    it again from the recordings and the unchanged code and its file changes as other presets add their entries, and the names
    of the model and the judge alone are compared, since a later job serves the same model at another endpoint."""
    record = read_json(Path(run_dir) / LEDGER)
    fields = run_fields(meta, serve=lambda name, endpoint, variable: {"name": name} if name else None)
    for key in RESUMED_FIELDS:
        if record.get(key) != fields[key]:
            return f"the field {key} of its ledger differs from the command's"
    for name in RECORDED_DATA:
        if (record["data"].get(name) or {}).get("sha256") != (fields["data"].get(name) or {}).get("sha256"):
            return f"its data file {name} differs from the command's"
    for role in ("model", "judge"):
        recorded, wanted = (record.get(role) or {}).get("name"), (fields[role] or {}).get("name")
        if recorded != wanted:
            return f"its {role} {recorded} differs from the command's {wanted}"
    changed = sorted(path for path, digest in ((record.get("code") or {}).get("files") or {}).items() if hashing.sha256_of(run_data.ROOT / path) != digest)
    if changed:
        return f"{len(changed)} of its code files changed since it started: {', '.join(changed[:3])}"
    if "start_check" not in record:
        return "its ledger records no start check"
    if bool(record["start_check"]) != bool(reasons):
        first = "stamped" if record["start_check"] else "registered"
        return f"the start check {first} its first part and {'registers' if not reasons else 'stamps'} this one"
    return None


def record_resume(run_dir, argv, kept, aside, moved_from=None, serving=None, granted=None):
    """Records that a job took up the run in progress of run_dir: one event with the job and its host, the command, the episodes
    kept and those set aside, and the files moved into the folder INTERRUPTED with their hashes. The ledger records of the episodes
    not kept are dropped, so that the episodes played again record themselves anew. aside holds (episode, folder under
    INTERRUPTED) pairs, and moved_from the folder the run left, when the job moved it. serving, as run.json writes it, goes into the
    event, so that a part taken up on another server or card shows in its ledger, and so does granted, the processors and the memory
    granted to the job that took the run up (run_episodes.granted), when the caller gives it."""
    run_dir = Path(run_dir)
    record = read_json(run_dir / LEDGER)
    dropped = sorted(name for name in record["episodes"] if name not in kept)
    for name in dropped:
        del record["episodes"][name]
    files = {f"{INTERRUPTED}/{moved}/{path}": (None, entry["sha256"]) for _name, moved in aside
             for path, entry in hashing.file_hashes(run_dir / INTERRUPTED / moved).items()}
    what = (f"taken up again after {len(kept)} finished episodes" + (f", moved from {public_path(moved_from)}" if moved_from else "")
            + (f"; set aside into {INTERRUPTED}: {', '.join(f'{name} as {moved}' for name, moved in aside)}" if aside else "")
            + (f"; ledger records dropped: {', '.join(dropped)}" if dropped else ""))
    event = _event(RESUME_TOOL, what, files)
    # the job, its host and the raw command of the job that took the run up go to the private file, the arguments in public form here
    command = {"argv": [str(arg) for arg in argv], "grid_file": os.environ.get("HYD_GRID_FILE"), "grid_line": os.environ.get("HYD_GRID_LINE")}
    add_private(run_dir, "starts", {"at": event["at"], "resumed": True, "command": command,
                                    "job": {**{field: os.environ.get(variable) for field, variable in JOB_ENV.items()}, "host": platform.node()}})
    event["command"] = {"argv": public_argv(command["argv"]), "grid_file": public_path(command["grid_file"]), "grid_line": command["grid_line"]}
    event["serving"] = public_value(serving)
    if granted is not None:
        event["granted"] = granted
    record["events"].append(event)
    _write(run_dir, record)
    return event


def _event(tool, what, files, code=None):
    event = {"at": now(), "tool": tool, "what": what,
             "files": {path: {"old": old, "new": new} for path, (old, new) in sorted((files or {}).items())}}
    if code is not None:
        event["code"] = code
    return event


def finish(run_dir):
    """Closes the ledger after run.json: the status finished, the end time of run.json, the hash of every file outside the
    episode folders, and every episode folder hashed again. An episode file whose hash differs from the one recorded after
    its episode, and a data file whose hash differs from the one at the start (grid lines in parallel share the grade and
    label tables), each get an event, so the ledger states the files as the run left them."""
    run_dir = Path(run_dir)
    ledger = read_json(Path(run_dir) / LEDGER)  # the file alone: these rewrite it, and local events stay beside it
    names = {p.name for p in episode_dirs(run_dir)}
    files = hashing.file_hashes(run_dir, exclude={LEDGER})
    ledger["run_files"] = {path: entry for path, entry in files.items() if path.split("/", 1)[0] not in names}
    changed = {}
    for name in sorted(names):
        current = {path.split("/", 1)[1]: entry for path, entry in files.items() if path.split("/", 1)[0] == name}
        recorded = ledger["episodes"].setdefault(name, {"seed": None, "block": None, "files": current})["files"]
        for path in sorted(set(recorded) | set(current)):
            old, new = (recorded.get(path) or {}).get("sha256"), (current.get(path) or {}).get("sha256")
            if old != new:
                changed[f"{name}/{path}"] = (old, new)
        ledger["episodes"][name]["files"] = current
    if changed:
        ledger["events"].append(_event("ledger.finish", "episode files changed after their episode was recorded", changed))
    # an asset table registered without a folder is recorded with no path and no hash; the paths that the first start of the run
    # read are in the private file, and without it a path of the public ledger is resolved on this machine (a data file it cannot
    # find is not compared)
    paths = ((read_private(run_dir).get("starts") or [{}])[0].get("data") or {})
    found = {name: paths.get(name) or resolve_recorded_path(entry["path"]) for name, entry in ledger["data"].items() if entry["path"]}
    data_now = {name: hashing.sha256_of(path) for name, path in found.items() if path}
    data_changed = {entry["path"]: (entry["sha256"], data_now[name]) for name, entry in ledger["data"].items()
                    if name in data_now and data_now[name] != entry["sha256"]}
    if data_changed:
        ledger["events"].append(_event("ledger.finish", "data files changed while the run was in progress", data_changed))
    ledger["status"], ledger["ended"] = "finished", (read_json(run_dir / "run.json", {}) or {}).get("ended")
    _write(run_dir, ledger)
    return ledger


def append_event(run_dir, tool, what, files=None, code=None):
    """Appends one event to a run's ledger: the time, the tool, what it did, for every file it changed the old and the new
    SHA-256, None for a file it created or removed, and the SHA-256 of the code that made the change when given. Returns
    False for a run without a ledger, made before the ledger existed."""
    if not (Path(run_dir) / LEDGER).exists():
        return False
    local = Path(run_dir) / LOCAL_EVENTS
    if local.exists():  # a synced copy: its ledger stays the cluster's, byte for byte
        write_json(local, [*(read_json(local, []) or []), _event(tool, what, files, code)])
        return True
    ledger = read_json(Path(run_dir) / LEDGER)
    ledger["events"].append(_event(tool, what, files, code))
    _write(run_dir, ledger)
    return True


def snapshot(run_dir):
    """The SHA-256 of every file of a run folder but its ledger, the state that record_changes() compares with."""
    return hashing.file_hashes(run_dir, exclude={LEDGER})


def record_changes(run_dir, before, tool, what, code=()):
    """Appends one event naming every file of the run folder that changed since snapshot() returned before, with its old
    and new SHA-256, and the SHA-256 of the given code files as they are now, the code that made the change, as score_runs.py
    does after it judges an AssetOpsBench run. Returns False when no file changed or the run has no ledger."""
    after = snapshot(run_dir)
    changed = {}
    for path in sorted(set(before) | set(after)):
        old, new = (before.get(path) or {}).get("sha256"), (after.get(path) or {}).get("sha256")
        if old != new:
            changed[path] = (old, new)
    if not changed:
        return False
    return append_event(run_dir, tool, what, changed, {path: hashing.sha256_of(run_data.ROOT / path) for path in code})
