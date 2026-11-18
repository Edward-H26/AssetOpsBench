"""The stage profile of one episode: where its wall time goes, from the files that the driver and the servers already write, one
row of the flat table per episode (CausalDiagnosis/performance/README.md gives every formula in words).

The episode total T is the wall_s cell of scores.csv, the attempt from before the state file exists to after the harness finish.
The state file stamps the start of the episode, the handover H of the task to the agent and the end of the agent's part, so T
splits into the creation C before H, the agent's part P after it and the end E, taken as T - C - P, which therefore also carries
the rounding of wall_s. Inside P the model time M is the union of the spans of the model calls, the client seconds of the tool
calls come from trajectory.json, and the starts of the server processes and the service of every counted call from the server
stamps of the transcript; the rest is P less the model time, the client seconds and the starts. The rest is a difference and no
measured interval, so it holds only when the parts it subtracts do not overlap, and three checks of every episode name the cases
in which they may: a counted call or a server start inside a model call, a rest below zero, and a server start outside the
opening of the episode, which a runner that starts a server process for every call makes.

The quantities of the thinking clock come from the columns of scores.csv, which the driver fills from the clock report at the
end of the attempt. The state file is written at the entry of each counted call, so the credit of the last counted call and a
failed model attempt after it reach the file only when a later step of the harness writes it. An episode folder that lacks the
transcript, llm_calls.json or trajectory.json, as the folder of a baseline policy lacks the last two, has a note that names the
file, and every quantity that needs it stays empty."""
import json
import statistics

from CausalDiagnosis.evaluation.result_tables.tables_from_runs.episode_fields import merged_spans
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER
from CausalDiagnosis.experiments.run_verification.check_grid_complete import LLM_CALLS
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.utils.csv_rows import as_bool, as_float
from CausalDiagnosis.utils.public_rules.visits import is_no_call
from .run_files import TRAJECTORY_FILE

TOKEN_COUNT = "tokenize"  # the record kind of a failed token count of the completion route, which generates nothing
STAMPS = ("episode_started_at", "handover_wall", "episode_ended_at")  # the wall stamps of the state file that split the total
# the seconds of the profile beside the total T, each also given as its share of T
SECONDS = ("creation_s", "cache_flush_s", "agent_part_s", "end_s", "model_s", "failed_model_s", "tool_service_s",
           "tool_service_cpu_s", "no_call_service_s", "server_starts_s", "env_tools_client_s", "diagnostics_client_s",
           "transport_hold_s", "rest_s")
SHARES = tuple(f"share_{name[:-2]}" for name in SECONDS)
COUNTS = ("model_calls_without_span", "counted_calls", "counted_calls_without_end", "no_call_steps", "stamped_steps",
          "server_start_count", "env_tool_calls", "diagnostics_calls")
CHECKS = ("lines_in_model_spans", "rest_negative", "late_server_starts")
CLOCK = ("thinking_s", "thinking_cycles", "horizon_cycles", "think_seconds_per_cycle", "thinking_share_of_horizon",
         "credited_starts_s", "credited_calls_s", "uncredited_starts_s", "uncredited_calls_s", "failed_model_attempts",
         "uncharged_after_last_s")
CALLS = ("model_calls", "failed_model_calls", "call_seconds_sum", "call_seconds_median", "input_tokens", "output_tokens",
         "output_tokens_per_call_second")
COLUMNS = ("note", "started_at", "total_s", *SECONDS, *SHARES, *COUNTS, *CHECKS, *CLOCK, *CALLS)
# the columns that each file of an episode folder gives, empty when the folder lacks the file
TRANSCRIPT = ("tool_service_s", "tool_service_cpu_s", "no_call_service_s", "server_starts_s", "counted_calls",
              "counted_calls_without_end", "no_call_steps", "stamped_steps", "server_start_count", "late_server_starts")
TRAJECTORY = ("env_tools_client_s", "diagnostics_client_s", "env_tool_calls", "diagnostics_calls")
FILES = (("transcript", transcript.NAME), ("llm_calls", LLM_CALLS), ("trajectory", TRAJECTORY_FILE))
# why an episode stays out of the profile, counted per cell under these names
NO_FOLDER, NO_ROW, NO_STATE, TORN, NO_TOTAL, SKEW = ("no episode folder", "no row in scores.csv", "no state file",
                                                     "torn transcript", "no wall_s", "clock_skew")


def left_out(row, folder, files):
    """Why an episode stays out of the profile, None when it enters: a row without its folder, a folder without its row, no state
    file, a torn transcript, a stamp of the state file missing, as for an episode whose attempt never reached its handover, no
    wall_s, or the clock skew flag of scores.csv, under which the wall stamps and the total cannot be compared."""
    if folder is None:
        return NO_FOLDER
    if row is None:
        return NO_ROW
    if files["state"] is None:
        return NO_STATE
    if files["transcript_torn"] is not None:
        return TORN
    missing = [key for key in STAMPS if files["state"].get(key) is None]
    if missing:
        return f"no {missing[0]} in the state file"
    if as_float(row.get("wall_s")) is None:
        return NO_TOTAL
    if as_bool(row.get("clock_skew")):
        return SKEW
    return None


def model_spans(calls):
    """(spans, unplaced): the (start, end) wall spans of the generation calls of llm_calls.json, a failed token count left out, and
    the count of generation calls whose record lacks one of the two stamps."""
    generation = [call for call in calls if call.get("kind") != TOKEN_COUNT]
    spans = [(float(call["started_at"]), float(call["ended_at"])) for call in generation
             if call.get("started_at") is not None and call.get("ended_at") is not None]
    return spans, len(generation) - len(spans)


def union_inside(spans, low, high):
    """The length of the union of spans inside [low, high], overlapping spans counted once."""
    return sum(max(0.0, min(end, high) - max(start, low)) for start, end in merged_spans(spans))


def counted_calls(lines):
    """The counted calls of a transcript, each (entry line, end line or None), in three lists: the agent's calls; the no_call steps
    that the driver's idle hook sends between two turns, which no runner records as a tool call; and the calls whose caller
    stamped them with a time of its own, the harness finish among them, whose end equals their entry (episode_state.end_fields),
    so they have no measured service."""
    ends = {line["enter_n"]: line for line in lines if line.get("kind") == "call" and line.get("end_wall") is not None}
    agent, idle, stamped = [], [], []
    for line in lines:
        if line.get("kind") != "enter":
            continue
        end = ends.get(line["n"])
        if end is not None and end["end_mono"] == line["mono"]:
            stamped.append((line, end))
        elif is_no_call(line):
            idle.append((line, end))
        else:
            agent.append((line, end))
    return agent, idle, stamped


def service(pairs, clock):
    """The summed service of counted calls from the entry stamp to the end stamp of one clock of the transcript, wall or
    thread_cpu; a call without an end line adds nothing."""
    return sum(end[f"end_{clock}"] - entry[clock] for entry, end in pairs if end is not None)


def start_interval(line):
    """The wall interval of a server start from its ready line: from the launcher's stamp s to the reading r, or the instant r for
    a process started without a launcher."""
    return (line["s_wall"] if line.get("s_wall") is not None else line["r_wall"]), line["r_wall"]


def client_seconds(trajectory, diagnostics):
    """(seconds, calls) of the tool calls of trajectory.json on the diagnostics server when diagnostics is true and on every other
    server otherwise, each call's duration_ms over 1000; a record without a tool or a duration is no timed tool call."""
    records = [record for record in trajectory if record.get("tool") and record.get("duration_ms") is not None
               and (record.get("server") == CAUSAL_SERVER) == diagnostics]
    return sum(float(record["duration_ms"]) for record in records) / 1000, len(records)


def overlaps(a, b, spans):
    """Whether the interval [a, b], or the instant a when b equals a, meets the inside of one of the spans."""
    return any(a < end and b > start for start, end in spans)


def lines_in_model_spans(agent_calls, starts, spans):
    """The agent's counted calls and the server starts whose interval meets the inside of a model call: a call's service from its
    entry to its end, or its entry alone without an end line, and a start from s to r."""
    intervals = [(entry["wall"], end["end_wall"] if end is not None else entry["wall"]) for entry, end in agent_calls]
    intervals += [start_interval(line) for line in starts]
    return sum(1 for a, b in intervals if overlaps(a, b, spans))


def late_server_starts(lines):
    """The server starts outside the opening of an episode: a ready line that is not the first of its server, or one that starts
    after the entry of the first counted call. The rest subtracts the starts and the client seconds of the tool calls apart, which
    holds when every server starts before the first call, as a runner that opens one session per server when the episode starts
    does; a runner that starts a server process for every call starts them inside its calls."""
    entries = [line["wall"] for line in lines if line.get("kind") == "enter"]
    first = min(entries) if entries else None
    seen, late = set(), 0
    for line in lines:
        if line.get("kind") != "ready":
            continue
        if line.get("server") in seen or (first is not None and start_interval(line)[0] > first):
            late += 1
        seen.add(line.get("server"))
    return late


def state_columns(row, state):
    """The seconds that the state file and the row of scores.csv give: the start stamp, the total, the creation with its flush, the
    agent's part, the end and the failed model attempts."""
    started, handover, ended = (float(state[key]) for key in STAMPS)
    total = float(row["wall_s"])
    return {"started_at": started, "total_s": total, "creation_s": handover - started,
            "cache_flush_s": (state.get("cache_flush") or {}).get("flush_seconds"), "agent_part_s": ended - handover,
            "end_s": total - (ended - started), "failed_model_s": as_float(row.get("failed_model_s"))}


def transcript_columns(lines):
    """The service of the counted calls and the starts of the server processes that the transcript stamps, their counts, the
    agent's entries without an end line, and the starts outside the opening."""
    agent, idle, stamped = counted_calls(lines)
    starts = [line for line in lines if line.get("kind") == "ready"]
    return {"tool_service_s": service(agent, "wall"), "tool_service_cpu_s": service(agent, "thread_cpu"),
            "no_call_service_s": service(idle, "wall"), "server_starts_s": sum(b - a for a, b in map(start_interval, starts)),
            "counted_calls": len(agent), "counted_calls_without_end": sum(1 for _entry, end in agent if end is None),
            "no_call_steps": len(idle), "stamped_steps": len(stamped), "server_start_count": len(starts),
            "late_server_starts": late_server_starts(lines)}


def trajectory_columns(trajectory):
    """The client seconds and the counts of the tool calls of trajectory.json, apart for the diagnostics server."""
    env_s, env_n = client_seconds(trajectory, diagnostics=False)
    mcp_s, mcp_n = client_seconds(trajectory, diagnostics=True)
    return {"env_tools_client_s": env_s, "diagnostics_client_s": mcp_s, "env_tool_calls": env_n, "diagnostics_calls": mcp_n}


def stage_columns(row, files):
    """The seconds, the counts and the checks of the profile of an episode that left_out lets in, with the note of the files that
    the episode folder lacks: a quantity that needs a missing file stays empty, and so does every difference and check over it, so
    that a missing file never reads as zero seconds."""
    state, lines, calls, trajectory = files["state"], files["transcript"], files["llm_calls"], files["trajectory"]
    out = state_columns(row, state)
    out.update(dict.fromkeys((*TRANSCRIPT, *TRAJECTORY, "model_s", "model_calls_without_span", "transport_hold_s", "rest_s", *CHECKS)))
    spans = None
    if calls is not None:
        spans, unplaced = model_spans(calls)
        handover, ended = float(state["handover_wall"]), float(state["episode_ended_at"])
        out.update(model_s=union_inside(spans, handover, ended), model_calls_without_span=unplaced)
    if lines is not None:
        out.update(transcript_columns(lines))
    if trajectory is not None:
        out.update(trajectory_columns(trajectory))
    if lines is not None and trajectory is not None:
        out["transport_hold_s"] = out["env_tools_client_s"] - out["tool_service_s"]
    if None not in (calls, lines, trajectory):
        out["rest_s"] = out["agent_part_s"] - out["model_s"] - out["env_tools_client_s"] - out["diagnostics_client_s"] - out["server_starts_s"]
        out["rest_negative"] = int(out["rest_s"] < 0)
    if calls is not None and lines is not None:
        agent = counted_calls(lines)[0]
        out["lines_in_model_spans"] = lines_in_model_spans(agent, [line for line in lines if line.get("kind") == "ready"], spans)
    out["note"] = "; ".join(f"no {name}" for key, name in FILES if files[key] is None)
    return out


def share_columns(seconds):
    """Each quantity of SECONDS over the episode total, empty where the quantity is."""
    total = seconds["total_s"]
    return {share: (seconds[name] / total if seconds[name] is not None and total else None) for name, share in zip(SECONDS, SHARES)}


def _cell_object(text):
    """A JSON object of a cell of scores.csv, such as credited_s with its keys starts and calls; {} for an empty cell."""
    return json.loads(text) if text else {}


def clock_columns(state, row):
    """The charged thinking of an episode against its horizon: the seconds and the cycles that the counted calls charged in the
    state file, the horizon and the thinking rate of its configuration, and the clock columns of scores.csv."""
    config = state.get("config") or {}
    cycles = sum(int(cycle or 0) for cycle in state.get("think_cycles") or [])
    horizon = config.get("horizon")
    credited, uncredited = _cell_object(row.get("credited_s")), _cell_object(row.get("uncredited_s"))
    return {"thinking_s": sum(float(seconds or 0) for seconds in state.get("think_seconds") or []), "thinking_cycles": cycles,
            "horizon_cycles": horizon, "think_seconds_per_cycle": config.get("think_seconds_per_cycle"),
            "thinking_share_of_horizon": cycles / horizon if horizon else None,
            "credited_starts_s": credited.get("starts"), "credited_calls_s": credited.get("calls"),
            "uncredited_starts_s": uncredited.get("starts"), "uncredited_calls_s": uncredited.get("calls"),
            "failed_model_attempts": as_float(row.get("failed_model_attempts")),
            "uncharged_after_last_s": as_float(row.get("uncharged_after_last_s"))}


def call_columns(calls):
    """(columns, seconds): the generation calls of llm_calls.json in the row, their count, the failed ones, the sum and the median
    of their seconds, their input and output tokens and the output tokens over the seconds of the calls that report both, a rate
    that holds the prefill; and the seconds of every call, which the summary of a cell spreads. Without llm_calls.json every
    column stays empty."""
    if calls is None:
        return dict.fromkeys(CALLS), []
    generation = [call for call in calls if call.get("kind") != TOKEN_COUNT]
    seconds = [float(call["seconds"]) for call in generation if call.get("seconds") is not None]
    timed = [call for call in generation if call.get("seconds") is not None and call.get("output_tokens") is not None]
    timed_seconds = sum(float(call["seconds"]) for call in timed)
    columns = {"model_calls": len(generation), "failed_model_calls": sum(1 for call in generation if call.get("error")),
               "call_seconds_sum": sum(seconds), "call_seconds_median": statistics.median(seconds) if seconds else None,
               "input_tokens": sum(int(call["input_tokens"]) for call in generation if call.get("input_tokens") is not None),
               "output_tokens": sum(int(call["output_tokens"]) for call in generation if call.get("output_tokens") is not None),
               "output_tokens_per_call_second": sum(int(call["output_tokens"]) for call in timed) / timed_seconds if timed_seconds else None}
    return columns, seconds


def episode_profile(row, folder, files):
    """(columns, call seconds) of one episode: the reason under left_out with every other column empty for an episode that stays
    out, else its stage seconds and their shares, counts and checks, its clock and its model calls."""
    reason = left_out(row, folder, files)
    if reason is not None:
        return {"left_out": reason, "note": "", "total_s": as_float((row or {}).get("wall_s"))}, []
    seconds = stage_columns(row, files)
    calls, call_seconds = call_columns(files["llm_calls"])
    return {"left_out": "", **seconds, **share_columns(seconds), **clock_columns(files["state"], row), **calls}, call_seconds
