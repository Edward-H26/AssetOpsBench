"""The check of a complete grid folder, run after a grid job ended and its logs were copied into the folder job_logs of its grid
folder: every run of every grid folder named against the list of a complete run, and the logs of the job. It writes no file.

    python -m CausalDiagnosis.experiments.run_verification.check_grid_complete [--starts development] GRID_FOLDER ...

A grid folder is a folder named grid_<job id>; any other path, such as the manifest or a part of the archive that the pack writes
beside the grid folders, is passed over. The output is one line per problem, "<grid>/<run>: <item>: <what>", then
CHECK_GRID_OK with the number of runs and grid folders and exit 0, or CHECK_GRID_FAILED with the number of problems and exit 1;
exit 2 when no argument is a grid folder.

The items, for any asset, agent, preset, clock and store:
- the grid folder holds job_record.txt and, in job_logs, one Slurm log of its job (*-<job id>.out); a grid folder with a run of a
  model agent also holds the log of the model server (sglang-<job id>.log) and the file of the GPU sampler (gpu-<job id>.csv);
- the ledger of every run finished and names every file of the run folder with its hash (verify_results.check_ledger);
- run.json holds code, data and platform, and for a model agent the serving record of its model endpoint with the GPU card and the
  sampler;
- a run of record, the default, carries no stamp, passes results_index.registered_refusal, and, when env.sh names the card of
  record (GPU_OF_RECORD, GPU_MIB_OF_RECORD), played on that card; its runs fall into one group of the tables (results_index.group_key),
  and the check prints one line per group with its runs; with --starts development a run carries the stamp development;
- the run holds its planned episode folders and one row of scores.csv each, and every episode folder the files of required_files();
- the copy of the sampler file in job_logs holds, as its first bytes, the bytes whose SHA-256 and length run.json records;
- a grid folder with a run on the wall clock holds in_flight.tsv, and in each loop of its job every slot that freed after the start
  of the last run and before the end of the last run on the wall clock took a filler run before that end, so that the runs in
  flight of the job stayed as many as at the start of the last run (in_flight_problems)."""
from CausalDiagnosis.utils.paths import SRC
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC)]
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.run_verification import replay_check  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs.run_health_checks import JOB_RECORD  # noqa: E402  the accounting record that CausalDiagnosis/cluster_jobs/4_after_run/job_record.sh writes into the grid folder
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, HIDDEN_FILES, RELEASED_FILE, STATE_FILE, episode_dirs, read_rows
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs.run_lines import WALL_CLOCK  # noqa: E402

# grid_<job id>, and grid_<job id>_attempt<n>, the folder of an earlier attempt that CausalDiagnosis/cluster_jobs/3_run/run_lines.sh moves aside on a requeue
GRID = re.compile(r"^grid_(\d+)(?:_attempt\d+)?$")
JOB_LOGS = "job_logs"  # the folder of the grid folder that holds the copies of the logs of its job
SERVER_LOG = "sglang-{job}.log"  # the log of the model server of a serving job, beside its Slurm log in LOGS
SAMPLER = "gpu-{job}.csv"  # the file of the GPU sampler of a serving job, beside its Slurm log in LOGS
TRANSCRIPT = "transcript.jsonl"
LLM_CALLS = "llm_calls.json"
IN_FLIGHT = "in_flight.tsv"  # the record of CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch, one line per start and per end of a process of the job
# the fields of a line: the epoch second, start or end, the loop of the job (reference or units), run or filler, the name of the
# process, its pid, and 1 when its configuration charges the wall seconds of thinking, else 0
IN_FLIGHT_FIELDS = ("epoch", "event", "loop", "kind", "name", "pid", "wall")


def model_agent(meta):
    """Whether the run's agent reaches a served model, so its run.json records the serving of its model endpoint."""
    return meta.get("agent") in run_episodes.MODEL_AGENTS


def required_files(meta):
    """The files that every episode folder of the run holds, the one place of the rule: the task, the state file, the released
    evidence, the report, the four files of hidden and the launcher of every server that run.json names; for a run that is not
    the static agent's, whose episodes call no server, the record of the harness and the transcript; for an agent that calls a
    model, the scripted check among them, the record of its model calls."""
    files = ["question.txt", STATE_FILE, RELEASED_FILE, "report.md", *(f"{HIDDEN}/{name}" for name in HIDDEN_FILES),
             *(f"{server}_server.py" for server in meta.get("servers") or [])]
    if meta.get("mode") != replay_check.STATIC_MODE:
        files += [run_episodes.HARNESS_FILE, TRANSCRIPT]
    if model_agent(meta) or meta.get("agent") == "scripted":
        files.append(LLM_CALLS)
    return files


def groups_of(run_dirs):
    """{group text: [run names]} of the run folders by results_index.group_key, each part a hash cut to 12 characters or a value."""
    groups = {}
    for run_dir in run_dirs:
        key = pooling_rules.group_key(read_json(Path(run_dir) / "run.json", None) or {})
        text = " ".join(f"{part}={str(value)[:12]}" for part, value in key.items())
        groups.setdefault(text, []).append(Path(run_dir).name)
    return groups


def check_run(run_dir, development, logs):
    """The problems of one run folder, each "<run>: <item>: <what>"."""
    run_dir, name, problems = Path(run_dir), Path(run_dir).name, []
    if ledger.in_progress(run_dir):
        record = read_json(run_dir / ledger.LEDGER, {}) or {}
        return [f"{name}: {ledger.LEDGER}: a run in progress, with {len(record.get('episodes') or {})} of its "
                f"{(record.get('seeds') or {}).get('episodes_planned')} planned episodes recorded; its part submitted again with "
                "GRID_PENDING=1 takes it up from its last finished episode"]
    state, found = verify_results.check_ledger(run_dir)
    if state != "checked":
        problems.append(f"{name}: {ledger.LEDGER}: {state}")
    problems += [f"{name}: {ledger.LEDGER}: {p.split('/', 1)[1] if p.startswith(name + '/') else p}" for p in found if state == "checked"]
    meta = read_json(run_dir / "run.json", None)
    if not isinstance(meta, dict):
        return problems + [f"{name}: run.json: missing or not a JSON object"]
    problems += [f"{name}: run.json: no {key}" for key in ("code", "data", "platform") if not meta.get(key)]
    serving = (meta.get("serving") or {}).get("agent") or {}
    sampler = serving.get("sampler") or {}
    if model_agent(meta) and not (ledger.card_of(serving) and sampler.get("sha256") and sampler.get("bytes") is not None):
        problems.append(f"{name}: run.json: no serving record of the model endpoint with its GPU card and its sampler")
    kind = pooling_rules.kind_of(meta)
    if development:
        if kind != "development":
            problems.append(f"{name}: run.json: a {kind} run, and --starts development checks development runs")
    else:
        if kind != "record":
            problems.append(f"{name}: run.json: a {kind} run: {'; '.join(meta.get('stamp_reasons') or []) or 'no identity'}")
        reason = pooling_rules.registered_refusal(meta, run_dir)
        if reason is not None:
            problems.append(f"{name}: registration: {reason}")
        if model_agent(meta) and (card := ledger.card_record_reason(ledger.card_of(serving))) is not None:
            problems.append(f"{name}: card: {card}")
    episodes, planned, rows = episode_dirs(run_dir), meta.get("episodes_planned"), read_rows(run_dir)
    if planned is not None and len(episodes) != planned:
        problems.append(f"{name}: episodes: {len(episodes)} episode folders, {planned} planned")
    if planned is not None and len(rows) != planned:
        problems.append(f"{name}: scores.csv: {len(rows)} rows, {planned} planned")
    wanted = required_files(meta)
    for ep_dir in episodes:
        problems += [f"{name}: {ep_dir.name}/{path}: missing" for path in wanted if not (ep_dir / path).is_file()]
    grid = GRID.match(Path(logs).parent.name)
    if sampler.get("sha256") and grid:  # run.json names no sampler path: the copy of the job's sampler is named by the job of the grid folder
        copy = Path(logs) / SAMPLER.format(job=grid.group(1))
        if not copy.is_file():
            problems.append(f"{name}: sampler: {copy.name} is not in {JOB_LOGS}")
        elif ledger.prefix_hash(copy, sampler.get("bytes"))[0] != sampler.get("sha256"):
            problems.append(f"{name}: sampler: the first {sampler.get('bytes')} bytes of {JOB_LOGS}/{copy.name} are not those whose SHA-256 run.json records")
    return problems


def in_flight_problems(folder, metas):
    """The problems of the record of the runs in flight of a grid folder with a run on the wall clock: the record is missing, a line
    is not IN_FLIGHT_FIELDS separated by tabs, or a slot of a loop took no filler (_unfilled). A grid folder without a run on the
    wall clock needs no record, since the load of a job moves no clock but that one."""
    folder = Path(folder)
    if not any(meta.get("clock_mode") == WALL_CLOCK for meta in metas):
        return []
    if not (folder / IN_FLIGHT).is_file():
        return [f"{folder.name}: {IN_FLIGHT}: missing, and a run of the grid folder charges the wall seconds of thinking"]
    problems, events = [], []
    for number, line in enumerate((folder / IN_FLIGHT).read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        event = dict(zip(IN_FLIGHT_FIELDS, fields))
        if (len(fields) != len(IN_FLIGHT_FIELDS) or not event["epoch"].isdigit() or event["event"] not in ("start", "end")
                or event["kind"] not in ("run", "filler") or event["wall"] not in ("0", "1")):
            problems.append(f"{folder.name}: {IN_FLIGHT}: line {number} is not {', '.join(IN_FLIGHT_FIELDS)} separated by tabs")
        else:
            events.append(event)
    for loop in dict.fromkeys(event["loop"] for event in events):
        problems += [f"{folder.name}: {IN_FLIGHT}: {p}" for p in _unfilled([e for e in events if e["loop"] == loop])]
    return problems


def _unfilled(events):
    """The slots of one loop that freed after the start of its last run and before the end of its last run on the wall clock and
    took no filler start before that end, ends and filler starts matched in the order of the record. An end in the second of that
    last end is passed over, since the job stops its fillers then."""
    runs = [i for i, event in enumerate(events) if event["kind"] == "run"]
    opened = max((i for i in runs if events[i]["event"] == "start"), default=None)
    closed = max((i for i in runs if events[i]["event"] == "end" and events[i]["wall"] == "1"), default=None)
    if opened is None or closed is None or closed < opened:
        return []
    last, freed = int(events[closed]["epoch"]), []
    for i in range(opened + 1, len(events)):
        event = events[i]
        if event["event"] == "end" and i < closed and int(event["epoch"]) < last:
            freed.append(event)
        elif event["event"] == "start" and event["kind"] == "filler" and freed and int(event["epoch"]) <= last:
            freed.pop(0)
    return [f"the slot that {e['name']} freed at {e['epoch']} in the loop {e['loop']} took no filler before the last run on the "
            f"wall clock ended at {last}" for e in freed]


def check_grid(folder, development):
    """(problems, runs) of one grid folder, each problem "<grid>/..."."""
    folder = Path(folder)
    job = GRID.match(folder.name).group(1)
    logs = folder / JOB_LOGS
    problems = [] if (folder / JOB_RECORD).is_file() else [f"{folder.name}: {JOB_RECORD}: missing"]
    slurm = sorted(logs.glob(f"*-{job}.out")) if logs.is_dir() else []
    if len(slurm) != 1:
        problems.append(f"{folder.name}: {JOB_LOGS}: {len(slurm)} Slurm logs *-{job}.out of the job, one wanted")
    runs = sorted(p for p in folder.iterdir() if p.is_dir() and ((p / "run.json").exists() or (p / ledger.LEDGER).exists()))
    if not runs:
        problems.append(f"{folder.name}: no run folder")
    metas = [read_json(run / "run.json", None) or {} for run in runs]
    if any(model_agent(meta) for meta in metas):
        problems += [f"{folder.name}: {JOB_LOGS}: {name} missing" for name in (SERVER_LOG.format(job=job), SAMPLER.format(job=job))
                     if not (logs / name).is_file()]
    for run in runs:
        problems += [f"{folder.name}/{p}" for p in check_run(run, development, logs)]
    if not development and len(groups := groups_of(runs)) > 1:
        problems.append(f"{folder.name}: groups: its runs fall into {len(groups)} groups of the tables, which pool none of them: "
                        + "; ".join(f"{text}: {', '.join(names)}" for text, names in groups.items()))
    problems += in_flight_problems(folder, metas)
    return problems, len(runs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="The check of a complete grid folder; the module docstring holds the items.")
    ap.add_argument("--starts", choices=("registered", "development"), default="registered",
                    help="registered (the default) checks the runs of record and prints their groups, development the runs of a trial")
    ap.add_argument("folders", nargs="+", help="grid folders grid_<job id> or grid_<job id>_attempt<n>; any other path is passed over")
    a = ap.parse_args(argv)
    grids = [Path(f) for f in a.folders if Path(f).is_dir() and GRID.match(Path(f).name)]
    for f in a.folders:
        if Path(f) not in grids:
            print(f"{f}: no grid folder, passed over")
    if not grids:
        print("no argument is a grid folder grid_<job id>")
        return 2
    problems, runs = [], 0
    for grid in grids:
        found, count = check_grid(grid, a.starts == "development")
        problems, runs = problems + found, runs + count
    if a.starts == "registered":
        run_dirs = [p for grid in grids for p in sorted(grid.iterdir()) if p.is_dir() and (p / "run.json").exists()]
        for text, names in groups_of(run_dirs).items():
            print(f"group {text}: {len(names)} runs ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})")
    for line in problems:
        print(line)
    if problems:
        print(f"CHECK_GRID_FAILED {len(problems)} problems")
        return 1
    print(f"CHECK_GRID_OK {runs} runs in {len(grids)} grid folders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
