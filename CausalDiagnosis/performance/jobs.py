"""The view of one cluster job over its episodes: where the wall time of the job goes before its first episode, between its first
and its last episode and after its last, its card-seconds per episode and per passed episode, and the rate of episodes of a job
that keeps runs in flight (CausalDiagnosis/performance/README.md).

A serving job holds one card from its start to its end, the Elapsed of its job record. Its stamps place the parts of that time:
the start of the job in the record, the started line of its endpoint file once the server answers, the first line of the grid
runner in the job's log once the opening checks have passed, of which the timed opening call names its own seconds, and the
stamps of the episodes. sacct and the log print the local time of the cluster without its zone, so both are placed by the UTC
offset of the started line. A part whose stamps the grid folder does not hold stays empty and the note of the job says why.

The rate is read over the windows in which exactly N model runs of the job are alive by in_flight.tsv, N the runs in flight of
the job, so it measures the job at N streams of the served model, the time between two episodes of a stream included, and leaves
the start and the end of the job to the parts above. A run of an agent without the served model, as a baseline policy that a line
of the grid file plays, counts toward no window, so a window in which only such a run is alive is no window. It is given twice:
the episodes that end inside the windows over their length, whose two edges cancel in expectation, and the share of each
episode's span that lies inside the windows, summed over the episodes, over the same length, which has no edge term. The count of
the episodes that lie whole inside a window is not used: it drops at each edge the episode of every stream that straddles it, so
it reads low at N above one and not at one.

A run that did not finish has no run.json: the end of the job cut it, and then its process wrote no end line and its episode in
flight has no span, or it stopped before its end. Its finished episodes are read from what its folder records, and its process in
in_flight.tsv is found by the line and the replicate of its ledger, the name that the grid runner gives it. The window still open at the end of the record closes at the earliest last episode
end among the runs alive in it, so that no episode of it lacks its span and the second figure keeps no edge term there. A window
in which a process without a run folder was alive, or a run whose folder does not record a field of its cell, gives no rate, and
the notes of the job name each.

The parts around the episodes read the model episodes alone, so the time of a run that plays no model episode, a run of a
baseline policy or one whose agent is not known here, lies in the part before the first model episode, between the first and
the last, or after the last, as its episodes do, and the notes of the job name each such run with the parts that hold its
episodes. Its time also lies in the Elapsed that the card-seconds divide, and the notes say so. A run whose folder records no
agent, or records an agent of none of the kinds that the tree declares, may be a model run, so then the card-seconds of the job
are not divided."""
import re
from datetime import datetime

from CausalDiagnosis.experiments.cluster_runs.run_health_checks import duration_seconds
from CausalDiagnosis.utils.csv_rows import as_bool
from . import gpu_sampler, run_files
from .stage_profile import NO_ROW

LINES_LOOP = "units"  # the loop of run_grid.sbatch that plays the lines of the grid file; the reference agents play under its loop reference
PROCESS = "line_{line}_replicate_{replicate}"  # the name that run_grid.sbatch gives the process of a line of the grid file (line_name)
NOT_ENDED = "Unknown"  # the End that sacct prints for a job that still runs, as when the grid runner writes the record itself
# the first line of the grid runner in the log of a serving job (job_header of cluster_jobs/functions.sh), printed once the opening
# checks of run_lines.sh have passed, and the line of the timed opening call among those checks
RUNNER_HEADER = re.compile(r"^### \S+ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \S+ fork .* grid .+, runs in flight \d+$")
OPENING_CALL = re.compile(r"^OPENING_CALL .* seconds ([0-9.]+) cycles ")
PARTS = ("server_start_s", "opening_s", "opening_check_s", "opening_call_s", "to_first_episode_s", "episodes_span_s",
         "after_last_episode_s")
CARDS_PER_JOB = 1  # a serving job serves its model on one card (cluster_jobs/2_serve_model/serve_model.sbatch)


def local_epoch(text, zone):
    """The epoch seconds of a local time without its zone, as sacct and the log of a job print it, placed by zone; None for a text
    that is no such time."""
    try:
        return datetime.fromisoformat(text.strip()).replace(tzinfo=zone).timestamp()
    except (AttributeError, ValueError):
        return None


def log_stamps(lines, zone):
    """(runner start, opening call seconds) from the lines of a job's log: the epoch of the grid runner's first line, None when the
    log holds no such line or more than one, as a requeued job's log does, and the seconds of its timed opening call."""
    headers = [match.group(1) for match in map(RUNNER_HEADER.match, lines or []) if match]
    calls = [float(match.group(1)) for match in map(OPENING_CALL.match, lines or []) if match]
    return (local_epoch(headers[0], zone) if len(headers) == 1 else None), (calls[0] if len(calls) == 1 else None)


def alive_windows(events, n):
    """The windows (a, b, names, cut) in which exactly n processes of the loop of the lines are alive by the record of the processes
    in flight, read in the order of the record, each with the names of those processes; the window still open at the end of the
    record, which a run that the end of the job cut leaves, has b None and cut True."""
    alive, since, opened, windows = [], None, frozenset(), []
    for event in (event for event in events if event["loop"] == LINES_LOOP):
        before = len(alive)
        if event["event"] == "start":
            alive.append(event["name"])
        elif event["name"] in alive:
            alive.remove(event["name"])
        if before == n and len(alive) != n and event["epoch"] > since:
            windows.append((since, event["epoch"], opened, False))
        if len(alive) == n and before != n:
            since, opened = event["epoch"], frozenset(alive)
    if len(alive) == n and since is not None:
        windows.append((since, None, opened, True))
    return windows


def close_open(windows, last_ends, final=True):
    """(windows, notes): the windows with an end, and the window still open at the end of the record closed at the earliest last
    episode end among the runs alive in it, last_ends naming per process that end of its run; an open window that holds a process
    without a finished episode in a run folder, or whose runs finished no episode inside it, stays out. final says whether the
    record of the job is the one written after it, so that the note names the end of the job as the cut only then."""
    out, notes = [], []
    for a, b, names, cut in windows:
        if not cut:
            out.append((a, b, names, cut))
            continue
        lacking = sorted(name for name in names if last_ends.get(name) is None)
        close = min((last_ends[name] for name in names if name not in lacking), default=None)
        if lacking or close is None or close <= a:
            why = f"{', '.join(lacking)} has no finished episode in a run folder" if lacking else "a run alive in it finished no episode inside it"
            notes.append(f"the window open at the end of in_flight.tsv from {a} gives no rate, since {why}")
        else:
            out.append((a, close, names, cut))
            why = "the end of the job cut them" if final else "the record was read while they played"
            notes.append(f"the window open at the end of in_flight.tsv from {a} closes at {close}, the earliest last episode end among "
                         f"its runs, since {why}")
    return out, notes


def unblocked(windows, blocked):
    """(windows, notes): the windows in which no process of blocked was alive, blocked naming per process why the rate cannot read
    it, and one note per such process with the count of the windows it took out."""
    taken = {}
    for _a, _b, names, _cut in windows:
        for name in names & set(blocked):
            taken[name] = taken.get(name, 0) + 1
    notes = [f"no rate from the {count} windows in which the process {name} was alive, since {blocked[name]}" for name, count in sorted(taken.items())]
    return [window for window in windows if not window[2] & set(blocked)], notes


def window_rates(spans, windows):
    """(ending, shares, seconds) of episode spans over the windows (a, b, names, cut): the episodes whose end lies inside a window,
    the sum over the episodes of the share of the episode's span that lies inside the windows, and the length of the windows. A
    window that a cut closed at an episode end holds that end, so that the episode that sets its close is counted."""
    ending = sum(1 for _start, end in spans if any(a <= end < b or (cut and end == b) for a, b, _names, cut in windows))
    shares = sum(sum(max(0.0, min(end, b) - max(start, a)) for a, b, _names, _cut in windows) / (end - start) for start, end in spans if end > start)
    return ending, shares, sum(b - a for a, b, _names, _cut in windows)


def job_time(record):
    """(start, end, elapsed, note) of a job from its record: the local start of a job that started once, None for one that started
    more than once, the local end of its last start, the Elapsed of all its starts in seconds, and why the parts or the cost
    cannot be read."""
    if not record:
        return None, None, None, "the grid folder holds no job record; job_record.sh writes it after the job"
    if record[-1]["end"] == NOT_ENDED:
        return None, None, None, "the job record was written while the job ran; job_record.sh writes the final one after the job"
    elapsed = sum(duration_seconds(line["elapsed"]) for line in record)
    if len(record) > 1:
        return None, record[-1]["end"], elapsed, f"the job started {len(record)} times, so its parts are not read"
    return record[0]["start"], record[0]["end"], elapsed, ""


def parts(stamps, spans):
    """The parts of a job's time from its stamps and the spans of its model episodes, each None where a stamp it needs is missing:
    the server start until the server answers, the opening until the grid runner's first line with the opening call and the
    opening checks within it, the time to the first model episode, the span from the first model episode's start to the last
    model episode's end, and the time after the last model episode."""
    start, server, runner, call, end = stamps
    first = min((a for a, _b in spans), default=None)
    last = max((b for _a, b in spans), default=None)

    def between(a, b):
        return b - a if a is not None and b is not None else None
    found = {"server_start_s": between(start, server), "opening_s": between(server, runner), "opening_call_s": call,
             "to_first_episode_s": between(runner, first), "episodes_span_s": between(first, last),
             "after_last_episode_s": between(last, end if start is not None else None)}
    found["opening_check_s"] = between(call, found["opening_s"])
    return found


def episode_spans(rows):
    """The wall spans [start, start + total] of the episodes of profile rows that the profile kept; an episode it left out, a
    skewed one among them, has no span that compares with the stamps of the job."""
    return [(row["started_at"], row["started_at"] + row["total_s"]) for row in rows if not row["left_out"]]


def job_stamps(grid, job, model_runs, start_text, end_text):
    """(stamps, notes): the epoch of the job's start, of the server's answer and of the grid runner's first line, the seconds of the
    opening call and the epoch of the job's end, each None where the grid folder lacks it, with the notes that say which is
    missing."""
    started = next((stamp for stamp in map(run_files.endpoint_started, (run["dir"] for run in model_runs)) if stamp), None)
    zone = gpu_sampler.zone_of(started)
    if zone is None:
        return (None, None, None, None, None), ["no model run of the job has the started line of the endpoint file in its ledger, "
                                                "which places the local times of the job"]
    lines = run_files.slurm_log_lines(grid, job)
    runner, call = log_stamps(lines, zone)
    notes = [] if lines is not None else [f"job_logs holds no single Slurm log *-{job}.out"]
    if lines is not None and runner is None:
        notes.append("the Slurm log holds no single first line of the grid runner")
    start = local_epoch(start_text, zone) if start_text else None
    end = local_epoch(end_text, zone) if end_text else None
    return (start, datetime.fromisoformat(started).timestamp(), runner, call, end), notes


def job_cost(rows, elapsed):
    """The card-seconds of the job per model episode and per passed model episode over the rows of the scored episodes, the ones
    that the profile leaves out among them, since each held the card."""
    passed = sum(1 for row in rows if as_bool(row.get("pass")))
    held = elapsed * CARDS_PER_JOB if elapsed is not None else None
    return {"passed": passed, "card_s_per_episode": held / len(rows) if held is not None and rows else None,
            "card_s_per_passed_episode": held / passed if held is not None and passed else None}


def process_name(run):
    """The name of the process of the grid runner that played a run, None for a run that its ledger places on no line."""
    found = run["process"]
    return PROCESS.format(line=found[0], replicate=found[1]) if found else None


def rate_windows(events, n, runs, final=True):
    """(windows, notes): the windows of n model runs alive that the rate reads, less every window in which a process that no run
    folder records, or the process of a run whose folder does not record a field of its cell, was alive, and with the window open
    at the end of the record closed at the earliest last episode end of its runs. The process of a run whose agent reaches no
    served model counts toward no window; a process that no run folder records, or the process of a run whose agent is not known
    here, counts, since nothing says what it played, and the windows in which it was alive give no rate."""
    named = {process_name(run): run for run in runs if process_name(run)}
    without_model = {name for name, run in named.items() if not run["model_agent"] and run_files.agent_unknown(run["meta"]) is None}
    blocked = {name: "no run folder of the grid folder records that process" for name in
               {event["name"] for event in events if event["loop"] == LINES_LOOP} if name not in named}
    blocked.update({process_name(run): f"its run {run['dir'].name} records no {', '.join(run['missing'])}" for run in runs
                    if run["missing"] and process_name(run)})
    blocked.update({process_name(run): f"its run {run['dir'].name} {run_files.agent_unknown(run['meta'])}" for run in runs
                    if not run["missing"] and process_name(run) and run_files.agent_unknown(run["meta"])})
    windows, block_notes = unblocked(alive_windows([event for event in events if event["name"] not in without_model], n), blocked)
    last_ends = {name: max((b for _a, b in episode_spans(run["rows"])), default=None) for name, run in named.items()}
    windows, close_notes = close_open(windows, last_ends, final)
    return windows, block_notes + close_notes


def job_rate(events, unread, n, runs, spans, left_out, final=True):
    """(columns, notes) of the rate of a job at n runs in flight over the windows of its record of the processes in flight, every
    rate None without a window; final as close_open reads it."""
    notes = [f"{unread} lines of in_flight.tsv are not its seven fields"] if unread else []
    windows = []
    if not (isinstance(n, int) and n > 0 and events):
        notes.append("the rate needs in_flight.tsv and the runs in flight of run.json")
    else:
        windows, window_notes = rate_windows(events, n, runs, final)
        notes += window_notes
        if not windows:
            notes.append(f"no window of in_flight.tsv with exactly {n} model runs alive in the loop {LINES_LOOP} gives a rate")
    ending, shares, seconds = window_rates(spans, windows) if windows else (None, None, None)

    def per(count, scale):
        return count / seconds * scale / CARDS_PER_JOB if seconds else None
    return {"runs_in_flight": n, "window_s": seconds, "episodes_ending_inside": ending, "episode_shares_inside": shares,
            "ending_per_s": per(ending, 1), "shares_per_s": per(shares, 1), "ending_per_card_hour": per(ending, 3600),
            "shares_per_card_hour": per(shares, 3600), "rate_left_out": left_out}, notes


def other_run_note(run, model_spans):
    """The note of a run of the job that plays no model episode, a run whose agent reaches no served model or is not known here:
    the parts around the model episodes, of model_spans, that hold the spans of its episodes, so that no part holds the time of such
    a run unnamed."""
    first = min(a for a, _b in model_spans)
    last = max(b for _a, b in model_spans)
    why = run_files.agent_unknown(run["meta"]) or "reaches no served model"
    spans = episode_spans(run["rows"])
    held = [name for name, holds in (("to_first_episode_s", lambda a, b: a < first), ("episodes_span_s", lambda a, b: a < last and b > first),
                                     ("after_last_episode_s", lambda a, b: b > last)) if any(holds(a, b) for a, b in spans)]
    where = (f"its episodes lie in {' and '.join(held)}" if held else
             "it has no episode with a span, so no part is named for its time" if not spans else "no part of the job holds its episodes")
    return f"{run['dir'].name} {why}, and {where}"


def unfinished_note(run, ended=None):
    """The note of a run that did not finish: what ended it where the record of the processes in flight says it, ended naming the
    processes with an end line, and the fields of its cell that its folder does not record."""
    why = "" if ended is None else (", though its process ended, so it stopped before its end" if process_name(run) in ended else
                                    " and its process has no end line, so the end of the job cut it")
    return (f"{run['dir'].name} has no run.json{why}; its cell and its finished episodes are read from its ledger, its endpoint "
            "record and its state files" + (f", which record no {', '.join(run['missing'])}" if run["missing"] else ""))


def job_view(grid, job, runs):
    """The view of the job of a grid folder from the profile of every run of the folder; runs holds per run its folder, its run.json
    or what the folder of a run that did not finish records in its place, whether its agent reaches the served model, its cell, the fields of the cell
    that its folder does not record, its process in in_flight.tsv and its profile rows. The model episodes are the rows of the model
    runs, every one for the cost and those the profile kept for the spans and the rate. A job whose model runs play more than one
    cell gets no cost and no rate, since its card time is shared between the cells, and a model run whose cell is not recorded
    whole, or a run whose agent is not known here, leaves the card time undivided. Every run that plays no model episode is named
    in the notes with the parts that hold its episodes."""
    model_runs = [run for run in runs if run["model_agent"]]
    others = [run for run in runs if not run["model_agent"]]
    without_model = [run["dir"].name for run in others if run_files.agent_unknown(run["meta"]) is None]
    # an episode in flight at a cut has a folder and no row of scores.csv: it is no scored episode, and its card time stays in the
    # Elapsed of the job
    rows = [row for run in model_runs for row in run["rows"] if row["left_out"] != NO_ROW]
    complete = [run for run in model_runs if not run["missing"]]
    record = run_files.job_record(grid)
    start_text, end_text, elapsed, note = job_time(record)
    stamps, notes = job_stamps(grid, job, model_runs, start_text, end_text)
    events, unread = run_files.in_flight_events(grid)
    # a run without an end line was cut only once the job has ended; before the final record it may still be playing
    final = bool(record) and record[-1]["end"] != NOT_ENDED
    ended = {event["name"] for event in events if event["event"] == "end"} if final else None
    notes = ([note] if note else []) + notes + [unfinished_note(run, ended) for run in runs if run["unfinished"]]
    model_spans = episode_spans(rows)
    found = parts(stamps, model_spans)
    notes += [other_run_note(run, model_spans) for run in others] if model_spans else []
    view = {"grid": grid.name, "model_runs": len(model_runs), "runs_without_model": without_model, "model_episodes": len(rows),
            "model_folders_without_row": sum(len(run["rows"]) for run in model_runs) - len(rows), "elapsed_s": elapsed, **found,
            **{f"share_{name[:-2]}": (found[name] / elapsed if found[name] is not None and elapsed else None) for name in PARTS}}
    cells = {run["cell"] for run in complete}
    if len(cells) != 1:
        notes.append(f"the model runs of the job play {len(cells)} cells, so it gets no cost and no rate" if cells else
                     "no model run of the job records its cell whole, so it gets no cost and no rate" if model_runs else
                     "the job holds no run of a model agent, so it gets no cost and no rate")
        return {**view, "cells_played": len(cells), "cell": None, "notes": notes}
    spans = episode_spans([row for run in complete for row in run["rows"] if row["left_out"] != NO_ROW])
    rate, rate_notes = job_rate(events, unread, complete[0]["meta"].get("runs_in_flight"), runs, spans, len(rows) - len(episode_spans(rows)),
                                final)
    cost = job_cost(rows, elapsed)
    undivided = [why for why, holds in (("a model run of the job records its cell in part", len(complete) < len(model_runs)),
                                        ("a run of the job records no agent", any("agent" in run["missing"] for run in others))) if holds]
    undivided += [f"{run['dir'].name} {run_files.agent_unknown(run['meta'])}" for run in others
                  if "agent" not in run["missing"] and run_files.agent_unknown(run["meta"])]
    if undivided:
        cost = {**cost, "card_s_per_episode": None, "card_s_per_passed_episode": None}
        notes += [f"{why}, so the card-seconds of the job are not divided" for why in undivided]
    elif without_model and cost["card_s_per_episode"] is not None:
        notes.append(f"the card-seconds of the job divide its whole Elapsed, with the time of the runs without the served model "
                     f"({', '.join(without_model)})")
    return {**view, "cells_played": 1, "cell": next(iter(cells)), **cost, "card_s_undivided": bool(undivided), **rate,
            "notes": notes + rate_notes}
