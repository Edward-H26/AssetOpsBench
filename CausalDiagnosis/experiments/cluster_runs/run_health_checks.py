import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.experiments.cluster_runs import run_lines  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.run_verification.public_record import read_private  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_dirs
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402


MONOTONIC_GAPS = "think_seconds_monotonic"  # episode.json, the monotonic seconds between the entries of two tool calls
WALL_GAPS = "think_seconds_wall"  # episode.json, the wall seconds between the same two entries
WALL_CLOCK = run_lines.WALL_CLOCK  # the clock mode whose runs record the monotonic seconds, 0.0 in the other modes
JOB_RECORD = "job_record.txt"  # written by CausalDiagnosis/cluster_jobs/3_run/run_grid.sbatch beside the run folders: the sacct lines of the job
JOB_STATES_WITHOUT_NOTE = ("COMPLETED", "RUNNING")  # the states of a line of the job record that the suspect check does not name
SUSPENSION_FLOOR = 1.0  # seconds of wall time beyond the monotonic clock below which a run is not named


def call_seconds(run):
    """The seconds of every model call of a run folder, in rising order, a call that failed without seconds left out."""
    return sorted(call["seconds"] for path in sorted(run.glob("*/llm_calls.json")) for call in json.loads(path.read_text(encoding="utf-8"))
                  if call.get("seconds") is not None)


def call_tokens(run):
    """The generated tokens of every model call of a run folder, in rising order, a call without a count left out."""
    return sorted(call["output_tokens"] for path in sorted(run.glob("*/llm_calls.json")) for call in json.loads(path.read_text(encoding="utf-8"))
                  if call.get("output_tokens") is not None)


def latency_line(name, seconds):
    p90 = seconds[max(0, math.ceil(0.9 * len(seconds)) - 1)]  # the nearest rank
    return (f"CALL_LATENCY {name}: {len(seconds)} model calls, total {sum(seconds):.2f} s, median {statistics.median(seconds):.2f} s, "
            f"p90 {p90:.2f} s, longest {seconds[-1]:.2f} s")


def latency(root):
    """The generation time clock charges the wall seconds between the agent's tool calls as thinking time, so the latency of
    every model call moves the clock, and the node, the runs in flight and a judge on the same card move it with it. One line
    per run folder with model calls says against which latency a result reads, and the pooled line is the one a pilot reads."""
    runs = {run.name: call_seconds(run) for run in sorted(p for p in Path(root).iterdir() if (p / "run.json").exists())}
    runs = {name: seconds for name, seconds in runs.items() if seconds}
    for name, seconds in runs.items():
        print(latency_line(name, seconds))
    if len(runs) > 1:
        print(latency_line(f"pooled over {len(runs)} runs", sorted(s for seconds in runs.values() for s in seconds)))
    return 0


def duration_seconds(text):
    """The seconds of a duration as sacct prints it, [days-][hours:]minutes:seconds."""
    days, _, clock = text.rpartition("-")
    seconds = 0
    for part in clock.split(":"):
        seconds = seconds * 60 + float(part)
    return int(days or 0) * 86400 + seconds


def suspended_seconds(root):
    """{job id: seconds suspended} from the job record beside the run folders, the sacct lines of the job as JobID, State,
    Start, End, Elapsed, Suspended and NodeList separated by bars, every start of a requeued job on a line of its own."""
    record = Path(root) / JOB_RECORD
    found = {}
    if record.exists():
        for line in record.read_text(encoding="utf-8").splitlines():
            fields = line.split("|")
            if len(fields) >= 6 and fields[5]:
                found[fields[0]] = found.get(fields[0], 0) + duration_seconds(fields[5])
    return found


def job_record_notes(root):
    """The notes that the suspect check prints before the lines of the runs: a folder without the job record, whose suspended time
    is then unknown, and once each state of a line of the record that is neither COMPLETED nor RUNNING, a job that the time limit,
    a cancel or a failure of the node ended, whose lines suspended_seconds reads as any other."""
    record = Path(root) / JOB_RECORD
    if not record.exists():
        return [f"SUSPECT_NOTE no job record beside the runs of {root}, so the suspended time of the job is unknown"]
    states = (fields[1] for fields in (line.split("|") for line in record.read_text(encoding="utf-8").splitlines()) if len(fields) >= 2)
    return [f"SUSPECT_NOTE the job record of {root} ends in the state {state}"
            for state in dict.fromkeys(states) if state and state not in JOB_STATES_WITHOUT_NOTE]


def latest_job(run):
    """The job and host of the latest start of a run, from its private file (public_record), {} without one: the published ledger
    names no job and no host."""
    starts = read_private(run).get("starts") or [{}]
    return starts[-1].get("job") or {}


def suspect(root):
    """The clock charges the monotonic seconds between two counted calls that no start or failed model attempt credits as
    thinking time, and a held process advances that clock too, so a run that was held is suspect until the seconds of its
    calls are read. A run is named when its job was requeued (the restart count
    of its ledger), when its job was suspended (the job record, since a stopped process sees its wall clock and its monotonic
    clock advance alike), or when its wall seconds before the tool calls exceed the monotonic seconds, which a sleep of the
    machine or a step of the wall clock leaves behind. The monotonic seconds are compared on the generation time clock alone,
    where the driver records them (the other modes record 0.0), and a call whose entry is null is left out and counted. The notes of
    job_record_notes come first, and the exit status reads the runs alone."""
    suspended = suspended_seconds(root)
    for note in job_record_notes(root):
        print(note)
    runs = sorted(p for p in Path(root).iterdir() if (p / "ledger.json").exists())
    named = unrecorded = 0
    for run in runs:
        job = latest_job(run)
        meta = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").exists() else {}
        clocked = meta.get("clock_mode") == WALL_CLOCK
        beyond, longest, recorded = 0.0, (0.0, ""), True
        for path in sorted(run.glob("*/episode.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            wall, monotonic = state.get(WALL_GAPS) or [], state.get(MONOTONIC_GAPS)
            if max(wall, default=0.0) > longest[0]:  # the first episode that holds the longest gap
                longest = (max(wall), path.parent.name)
            if not clocked:
                continue
            pairs = [(w, m) for w, m in zip(wall, monotonic or []) if m is not None]
            recorded = recorded and monotonic is not None and len(pairs) == len(wall)
            beyond += sum(max(0.0, w - m) for w, m in pairs)
        unrecorded += clocked and not recorded
        reasons = []
        restarts = int(job.get("slurm_restart_count") or 0)
        if restarts:
            reasons.append(f"job {job.get('slurm_job_id')} restarted {restarts} time{'s' if restarts > 1 else ''}")
        if suspended.get(str(job.get("slurm_job_id")), 0) > 0:
            reasons.append(f"job {job['slurm_job_id']} suspended for {suspended[str(job['slurm_job_id'])]:.0f} s by its job record")
        if beyond > SUSPENSION_FLOOR:
            reasons.append(f"{beyond:.1f} s of wall time beyond the monotonic clock before its tool calls")
        if reasons:
            named += 1
            print(f"SUSPECT {run.name}: {', '.join(reasons)}; longest wall gap before a tool call {longest[0]:.1f} s ({longest[1]})")
    if unrecorded:
        print(f"SUSPECT_NOTE {unrecorded} of {len(runs)} runs record no monotonic seconds for all or some of their tool calls, so a sleep of the "
              "machine cannot be read from those calls")
    print(f"SUSPECT_RUNS {named} of {len(runs)} runs")
    return 1 if named else 0


def nodes(roots):
    """Every clocked run plays on one node type, one H100, and the node of every run is read afterwards from the
    host of its ledger, printed per run with its job and counted per node, so a reader sees whether the two H100 nodes split a
    table."""
    counts, total = {}, 0
    for root in roots:
        for run in sorted(p for p in Path(root).iterdir() if (p / "ledger.json").exists()):
            job = latest_job(run)
            host = job.get("host") or "unknown"
            counts[host] = counts.get(host, 0) + 1
            total += 1
            print(f"NODE {run.name}: {host}, job {job.get('slurm_job_id')}")
    print(f"NODES {total} runs on {len(counts)} nodes: " + ", ".join(f"{host} {n}" for host, n in sorted(counts.items())))
    return 0


def _epoch(text):
    """The epoch seconds of a time of run.json (ISO, local time of the node), None without one."""
    return datetime.fromisoformat(text).timestamp() if text else None


def episode_interval(ep_dir):
    """(start, end) in epoch seconds of one episode, or None when either is not recorded: from the flush of the model server's
    cache before it (episode.json cache_flush.started_at), else its start (episode_started_at), to its recorded end
    (episode_ended_at), else the latest of its last tool call (last_call_wall_time) and its last model call (llm_calls.json)."""
    state = read_json(Path(ep_dir) / "episode.json", {}) or {}
    start = (state.get("cache_flush") or {}).get("started_at") or state.get("episode_started_at")
    end = state.get("episode_ended_at")
    if end is None:
        calls = read_json(Path(ep_dir) / "llm_calls.json", []) or []
        ends = [call.get("ended_at") for call in calls if isinstance(call, dict)] + [state.get("last_call_wall_time")]
        end = max((float(t) for t in ends if t is not None), default=None)
    return None if start is None or end is None else (float(start), float(end))


def flush_problem(ep_dir):
    """Why the cache of the model server was not shown clean before one episode, None when it was: the record of the flush
    (episode.json cache_flush, cache_flush.flush_before_episode) is missing, the flush did not answer 200, or the first model
    request of the episode reused cached tokens (first_request_cached_tokens not 0, or not recorded)."""
    flush = (read_json(Path(ep_dir) / "episode.json", {}) or {}).get("cache_flush")
    if not flush:
        return "no record of the flush of the model server's cache before it"
    if flush.get("status") != 200:
        return f"the flush of the model server's cache answered {flush.get('status')}, not 200"
    if flush.get("first_request_cached_tokens") != 0:
        return f"its first model request reused {flush.get('first_request_cached_tokens')} cached tokens, not 0"
    return None


def run_attempts(run_dir, meta, record):
    """[(start, end)] in epoch seconds of the attempts of a run folder: one from its start (run.json or ledger started), and one more
    from the at of every resume event of its ledger, the job that took the run up again (ledger.record_resume). An attempt that
    another follows ends at the latest second that the episodes it began recorded, those that a cut left and the resume set aside
    into INTERRUPTED among them, or at its own start when they recorded none. The last ends at run.json ended, and for a run that
    its job left without run.json, killed or cut, at the latest second of the episodes it began. So a run is alive where a job
    played it and nowhere between two of its jobs, and a run without run.json is still placed."""
    started = _epoch(meta.get("started") or record.get("started"))
    if started is None:
        return []
    resumes = (_epoch(event.get("at")) for event in record.get("events") or [] if event.get("tool") == ledger.RESUME_TOOL)
    starts = [started, *sorted(t for t in resumes if t is not None)]
    folders = [*episode_dirs(run_dir), *sorted(p for p in (Path(run_dir) / ledger.INTERRUPTED).glob("*") if (p / "episode.json").exists())]
    seconds = []  # (first, last) second of every episode: its start, and the latest of its end, its calls and its model calls
    for ep_dir in folders:
        state = read_json(ep_dir / "episode.json", {}) or {}
        first = (state.get("cache_flush") or {}).get("started_at") or state.get("episode_started_at")
        if first is not None:
            interval = episode_interval(ep_dir)
            calls = [float(t) for t in state.get("called_at") or [] if isinstance(t, (int, float))]
            seconds.append((float(first), max([float(first), *(interval[1:] if interval else ()), *calls])))
    attempts = []
    for k, start in enumerate(starts):
        following = starts[k + 1] if k + 1 < len(starts) else math.inf
        latest = max((last for first, last in seconds if start <= first < following), default=start)
        attempts.append((start, _epoch(meta.get("ended")) if following == math.inf and meta.get("ended") else latest))
    return attempts


def overlap(root):
    """An episode of a run on the wall clock plays alone on its card on a clean cache: no other run of the job is
    alive between the flush before the episode and its end, since a second stream on the model server would slow the replies the
    clock charges, and for a run of a model agent (run.json serving.agent, the record of its model endpoint, which the public
    run.json keeps where it writes no endpoint URL) the flush is recorded with 200 and no cached token in the first request
    (flush_problem). An agent without a model has no model server to flush, so the driver records no flush for it and the overlap
    alone holds its run. Every run folder of the grid folder that holds run.json or a ledger is placed by its attempts
    (run_attempts), so a run that a later job took up again is alive where each job played it, and a run that its job left without
    run.json is alive up to the latest second it recorded. Such a run is judged as well, by the clock that the states of its
    episodes record and for the overlap alone, since a later job keeps its finished episodes and an episode cut in mid play holds no
    complete flush record. An episode of an earlier attempt of a run that a later job took up is also compared with the runs of the
    grid folder where that attempt played, the parent of the run folder that the take-up's --resume named in the private record of
    the run, so that the job that writes run.json names an overlap of an attempt that a job cut. Each overlap is printed with both
    runs and both intervals, an episode whose flush fails, whose interval is not recorded or whose earlier grid folder is not
    recorded or gone is named, as it cannot be shown alone on a clean cache, and the exit status is 1 on any."""
    def placed(folder):
        """(runs, run.json by run name, attempts by run name) of the run folders of a grid folder that hold run.json or a ledger."""
        runs = sorted(p for p in Path(folder).iterdir() if p.is_dir() and ((p / "run.json").exists() or (p / ledger.LEDGER).exists()))
        metas = {run.name: read_json(run / "run.json", {}) or {} for run in runs}
        return runs, metas, {run.name: run_attempts(run, metas[run.name], read_json(run / ledger.LEDGER, {}) or {}) for run in runs}

    runs, metas, alive = placed(root)
    elsewhere = {}  # by grid folder, the attempts of the runs of a folder where an earlier attempt of a run of root played
    found = unrecorded = unflushed = 0
    for run in runs:
        # the public ledger names the folder that a take-up moved the run from by its base name alone, so the grid folder of each
        # earlier attempt comes from the private record, by the second of the take-up that ended the attempt
        earlier = {}
        for entry in read_private(run).get("starts") or []:
            argv = (entry.get("command") or {}).get("argv") or []
            resumed = [value for flag, value in zip(argv, argv[1:]) if flag == "--resume"]
            if entry.get("resumed") and resumed:
                earlier[_epoch(entry.get("at"))] = Path(resumed[-1]).parent
        for ep_dir in episode_dirs(run):
            # a run left without run.json is judged by the clock that its episodes record, since a later job keeps them
            if (metas[run.name] or (read_json(ep_dir / "episode.json", {}) or {}).get("config") or {}).get("clock_mode") != WALL_CLOCK:
                continue
            if (metas[run.name].get("serving") or {}).get("agent") and (problem := flush_problem(ep_dir)) is not None:
                unflushed += 1
                print(f"OVERLAP_FLUSH {run.name}/{ep_dir.name}: {problem}")
            interval = episode_interval(ep_dir)
            if interval is None:
                unrecorded += 1
                print(f"OVERLAP_UNRECORDED {run.name}/{ep_dir.name}: no recorded start or end of the episode")
                continue
            start, end = interval
            beside = dict(alive)
            took_up = min((began for began, _ended in alive[run.name][1:] if began > start), default=None)
            if took_up is not None:
                folder = earlier.get(took_up)
                if folder is not None and folder.is_dir():
                    if folder not in elsewhere:
                        elsewhere[folder] = placed(folder)[2]
                    # a run that root holds as well is root's own, as after a take-up into the same folder or in a copy of it
                    beside.update((f"{folder.name}/{name}", attempts) for name, attempts in elsewhere[folder].items() if name not in alive)
                else:
                    unrecorded += 1
                    print(f"OVERLAP_UNRECORDED {run.name}/{ep_dir.name}: the grid folder of the job that played it is not recorded or gone, "
                          "so no run beside it is placed")
            for other, attempts in beside.items():
                if other == run.name:
                    continue
                for began, ended in attempts:
                    if began < end and ended > start:
                        found += 1
                        print(f"OVERLAP {run.name}/{ep_dir.name} [{start:.0f}, {end:.0f}] overlaps {other} [{began:.0f}, {ended:.0f}]")
    print(f"OVERLAP_EPISODES {found} overlaps, {unrecorded} episodes without a recorded interval, {unflushed} episodes without a clean flush")
    return 1 if found or unrecorded or unflushed else 0
