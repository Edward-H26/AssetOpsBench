"""The one reader of the files that this package reads: run.json, scores.csv and ledger.json of a run folder, the state file, the
transcript, llm_calls.json and trajectory.json of each episode folder, and in the grid folder of a cluster job its job record, its
record of the processes in flight and the copies of its Slurm log and of its GPU sampler file. Every other module of the package
reads through this file.

A run that did not finish has no run.json, which the driver writes at the end of a run alone: the end of its job cut it, at the
time limit or by a cancel, or the run stopped before its end. Its folder still records what the cell of a run reads: the start record of its ledger names the agent, the preset, the model and the environment of the job with its runs
in flight, the ledger keeps the endpoint record of the served model, and the state file of an episode holds the configuration with
its asset and its clock mode. ledger_meta reads those fields and leaves a field that the folder does not record empty.

The reader only reads. The run folders it reads are no longer played, so it takes no lock: the lock helper of the servers leaves a
lock file beside the file it guards, and the files of a run folder, which its ledger hashes, must not grow by a read."""
import os
from pathlib import Path

from CausalDiagnosis.experiments.cluster_runs.run_health_checks import JOB_RECORD
from CausalDiagnosis.experiments.episode_runners.run_episodes.agent_kinds import AGENTS, MODEL_AGENTS
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import STATE_FILE, episode_dirs, episode_folder, read_rows
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification.check_grid_complete import GRID, IN_FLIGHT, IN_FLIGHT_FIELDS, JOB_LOGS, LLM_CALLS, SAMPLER
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.utils.json_files import read_json

RUN_FILE, SCORES_FILE = "run.json", "scores.csv"
TRAJECTORY_FILE = "trajectory.json"  # every runner's list of its tool calls, written by the driver after the runner ends
# the sacct fields of a line of the job record in the order that write_job_record of cluster_jobs/functions.sh asks for them
JOB_RECORD_FIELDS = ("job", "state", "start", "end", "elapsed", "suspended", "nodes")


def run_meta(run_dir):
    """The run.json of a run folder, an empty dict without it, with unrecorded naming the agent when the record holds none, as
    ledger_meta names the fields that a folder lacks, so that no default stands in for the agent."""
    meta = read_json(Path(run_dir) / RUN_FILE, {}) or {}
    return {**meta, "unrecorded": [] if meta.get("agent") else ["agent"]}


def is_unfinished(run_dir):
    """Whether a folder is a run that did not finish: it holds the ledger or scores.csv of a run and no run.json."""
    run_dir = Path(run_dir)
    return not (run_dir / RUN_FILE).is_file() and ((run_dir / ledger.LEDGER).is_file() or (run_dir / SCORES_FILE).is_file())


def unfinished_runs(grid):
    """The folders of a grid folder that hold a run that did not finish, in name order."""
    return sorted(path for path in Path(grid).iterdir() if path.is_dir() and is_unfinished(path))


def agent_unknown(meta):
    """Why the kind of a run's agent is not known here, None when it is: the run records no agent, or the agent that it records is
    none of the kinds that the tree declares, which is no evidence that no model ran."""
    agent = meta.get("agent")
    if not agent or "agent" in (meta.get("unrecorded") or []):
        return "records no agent"
    return None if agent in AGENTS else f"records the agent {agent}, which is not known here"


def ledger_meta(run_dir):
    """The fields of run.json that the cell and the job view read, for a run without run.json, from what its folder records: the
    agent, its label, the preset and the model of the start record of its ledger, the runs in flight from GRID_PARALLEL of the
    environment that the start record copies, as the driver reads them, the serving identity of the endpoint record that the
    ledger keeps, and the asset and the clock mode of the configuration in the state file of an episode. Under unrecorded it names
    each field whose record the folder lacks, which stays None, so that no default stands in for it."""
    record = ledger.load(run_dir) or {}
    command = record.get("command")
    parallel = ((command or {}).get("env") or {}).get("GRID_PARALLEL") or ""
    config = next((state.get("config") or {} for state in (read_json(path / STATE_FILE, None) for path in episode_dirs(run_dir)) if state), {})
    model = record.get("model")
    unrecorded = ([] if record.get("agent") else ["agent"]) + ([] if record else ["preset"]) + \
                 ([] if command is not None else ["runs in flight"]) + \
                 ([] if "asset_id" in config else ["asset"]) + ([] if "clock_mode" in config else ["clock mode"]) + \
                 (["model", "serving identity"] if not model and (record.get("agent") in MODEL_AGENTS or record.get("agent") not in AGENTS) else [])
    return {"asset_id": config.get("asset_id"), "agent": record.get("agent"), "agent_label": record.get("agent_label"),
            "config_name": (record.get("configuration") or {}).get("config_name"), "clock_mode": config.get("clock_mode"),
            "model": (model or {}).get("name"), "runs_in_flight": int(parallel) if parallel.isdigit() else None,
            "serving": {"agent": ledger.serving_identity(model) if model else None, "judge": None}, "unrecorded": unrecorded}


def process_of(run_dir):
    """(line, replicate) of the process of the grid runner that played a run, from the start record of its ledger: the index of its
    line in the grid file and the replicate that its command names, "none" without one, as the runner names the process line_<line>
    _replicate_<replicate> in in_flight.tsv; None for a run that no line of a grid played."""
    command = (ledger.load(run_dir) or {}).get("command") or {}
    line, argv = command.get("grid_line"), list(command.get("argv") or [])
    if line is None or not str(line).isdigit():
        return None
    replicate = argv[argv.index("--replicate") + 1] if "--replicate" in argv[:-1] else "none"
    return str(line), str(replicate)


def episode_entries(run_dir):
    """(name, row, folder) of every episode of a run folder in name order: the row of scores.csv joined to the folder that the
    driver's naming rule gives it whenever that folder exists, so that a folder without its state file is named as such, and a
    folder that holds a state file and that no row names with the row None; a row whose folder is missing has the folder None. So
    the profile names each instead of reading part of a run as the whole run."""
    run_dir = Path(run_dir)
    named = {episode_folder(run_dir, row).name: row for row in read_rows(run_dir)}
    folders = {name: run_dir / name for name in named if (run_dir / name).is_dir()}
    folders.update({path.name: path for path in episode_dirs(run_dir) if path.name not in named})
    return [(name, named.get(name), folders.get(name)) for name in sorted(set(named) | set(folders))]


def episode_files(ep_dir):
    """The files of one episode folder that the profile reads, each None when the folder does not hold it: the state file, the lines
    of the transcript as the servers' reader gives them, with the reason when a line is torn, llm_calls.json and trajectory.json."""
    ep_dir = Path(ep_dir)
    lines, torn = None, None
    if (ep_dir / transcript.NAME).is_file():
        try:
            lines = transcript.read(ep_dir)
        except ValueError as exc:
            torn = str(exc)
    return {"state": read_json(ep_dir / STATE_FILE, None), "transcript": lines, "transcript_torn": torn,
            "llm_calls": read_json(ep_dir / LLM_CALLS, None), "trajectory": read_json(ep_dir / TRAJECTORY_FILE, None)}


def endpoint_started(run_dir):
    """The started line of the endpoint file of the run's model endpoint as its ledger keeps it, an ISO time with the UTC offset of
    the node that served the model, or None for a run without a served model or without a ledger."""
    record = ledger.load(run_dir) or {}
    return ((record.get("model") or {}).get("endpoint_file") or {}).get("started")


def absolute(path):
    """A folder as an absolute path with its . and .. resolved and its links kept, so that a folder reached through a link keeps the
    name it is reached by."""
    return Path(os.path.abspath(path))


def grid_of(run_dir):
    """(grid folder, job id) of a run folder that a cluster job wrote into its folder grid_<job id>, (None, None) for any other."""
    parent = absolute(run_dir).parent
    match = GRID.match(parent.name)
    return (parent, match.group(1)) if match else (None, None)


def sampler_copy(grid, job):
    """The copy of the GPU sampler file of the job in job_logs of its grid folder, which job_record.sh makes after the job."""
    return Path(grid) / JOB_LOGS / SAMPLER.format(job=job)


def slurm_log_lines(grid, job):
    """The lines of the copy of the job's Slurm log in job_logs, the one file *-<job id>.out, or None without exactly one."""
    logs = sorted((Path(grid) / JOB_LOGS).glob(f"*-{job}.out"))
    return logs[0].read_text(encoding="utf-8", errors="replace").splitlines() if len(logs) == 1 else None


def in_flight_events(grid):
    """(events, unread) of the record of the processes in flight of a grid folder: each line of the seven tab separated fields of
    IN_FLIGHT_FIELDS as a dict with its epoch as an integer, in the order of the record, and the count of lines that are not of
    that form; ([], 0) without the record."""
    path = Path(grid) / IN_FLIGHT
    if not path.is_file():
        return [], 0
    events, unread = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != len(IN_FLIGHT_FIELDS) or not fields[0].isdigit() or fields[1] not in ("start", "end"):
            unread += 1
            continue
        events.append({**dict(zip(IN_FLIGHT_FIELDS, fields)), "epoch": int(fields[0])})
    return events, unread


def job_record(grid):
    """The lines of the job record of a grid folder as dicts of JOB_RECORD_FIELDS, every start of a requeued job on a line of its
    own, [] without the record."""
    path = Path(grid) / JOB_RECORD
    if not path.is_file():
        return []
    return [dict(zip(JOB_RECORD_FIELDS, line.split("|"))) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
