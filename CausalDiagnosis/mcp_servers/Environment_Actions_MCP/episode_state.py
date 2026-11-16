"""One hydraulic rig episode behind a stateless tool server (CausalDiagnosis/docs/environment-actions-mcp.md).

AssetOpsBench's Plan-Execute runner starts a new server process for every tool call, so an episode lives in a JSON
state file rather than in server memory: the seed, the configuration, the data file, every action passed to env.step,
refused ones included because they count toward the step cap, and the thinking time charged before each action, in cycles
and in the seconds they came from, with the seconds carried to the next call, and beside those seconds the wall and the
monotonic seconds between the entries of the same two calls and the wall time of each call, which no replay reads. The
thinking time is the chess clock of servers/thinking_clock.py: the state keeps the handover, the cursor, the credited
segments that still matter and the service of the last counted call, and every counted call reads the transcript lines and
the failed model attempts that arrived since the call before it. Each call rebuilds the environment by
reset(seed) and replay, which is exact because every draw comes from the generator seeded at reset and the thinking cycles
are stored, applies the new action, and saves the file before it answers. Beside
the state file the server keeps the released evidence, everything the agent has been told and nothing else, for tools
that reason over the episode without touching the environment, such as the diagnostics server.
"""
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.dataset_loader.recordings import public_path, resolve_recorded_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import environment_factory as world
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET, EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import TOKEN_MODE
# the rules of the real-time clock, which both surfaces call
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.token_clock import token_seconds
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.wall_clock import monotonic_gap, think_cycles
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import RELEASED_FORMAT
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.utils import model_failures
from CausalDiagnosis.mcp_servers.utils import ready_line as ready
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import thinking_clock
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.utils.atomic_writes_and_locks import locked, write_json_atomically

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing

STATE_FORMAT = 4  # 4 since the chess clock of the thinking time: the handover, the cursor and the credited segments
RELEASED_NAME = "released.json"
EPISODE_OVER = "the episode is over; this call changed nothing"
# the three clocks of a reading, read when a call is served: the thread time is that of the calling thread, so a reading taken
# in the thread that serves a call measures that thread alone. A replay of a recorded episode sets them to the recorded
# readings (recorded_clock), and nothing in a served episode does
WALL_CLOCK = time.time
MONOTONIC_CLOCK = time.monotonic
THREAD_CLOCK = time.thread_time
_STAMPS = []  # the recorded readings (wall, mono, thread_cpu) that a replay hands to stamp, oldest first
NO_HANDOVER = ("the task of this episode was never handed over, so this call is outside a played episode; the driver, or a caller in "
               "its own process, hands the task over with episode.hand_over before the agent's first call")
SECOND_HANDOVER = "the task of this episode was handed over already"
BASELINE_FILE = "hidden/server_baseline.json"  # per server the launcher's SHA-256 and the thread count of its kind, as the driver measured them
# the total of the ledger of generated tokens of an episode folder, read when a call of the generated_tokens mode is served; a
# replay of a recorded episode sets it to the recorded total (recorded_clock)
GENERATED_TOKENS = token_clock.total
NO_TOKEN_LEDGER = (f"the clock mode {TOKEN_MODE} charges the tokens that the harness of the agent writes to {token_clock.NAME} "
                   "of the episode folder, and this episode has none; an agent whose harness does not report its tokens plays the "
                   "generation_time mode")


@contextmanager
def recorded_clock(wall, monotonic=None, generated=None, stamps=()):
    """Serve the calls inside the block under a recorded reading of the wall clock, of the monotonic clock where one was
    recorded and of the total of generated tokens that a call of the generated_tokens mode read, as a replay of a transcript
    does, so that a replay needs no ledger. stamps are the recorded readings (wall, mono, thread_cpu) of the entry and the end
    of the call, which stamp hands out in their order. The action server and the metering layer of the AssetOpsBench rows both
    read the clocks of this module, and the block restores them."""
    global WALL_CLOCK, MONOTONIC_CLOCK, THREAD_CLOCK, GENERATED_TOKENS
    saved = WALL_CLOCK, MONOTONIC_CLOCK, THREAD_CLOCK, GENERATED_TOKENS, list(_STAMPS)
    WALL_CLOCK, MONOTONIC_CLOCK, THREAD_CLOCK, GENERATED_TOKENS = (lambda: wall), (lambda: monotonic), (lambda: 0.0), (lambda folder: generated)
    _STAMPS[:] = [tuple(s) for s in stamps]
    try:
        yield
    finally:
        WALL_CLOCK, MONOTONIC_CLOCK, THREAD_CLOCK, GENERATED_TOKENS = saved[:4]
        _STAMPS[:] = saved[4]


def stamp(wall_clock=None, monotonic_clock=None):
    """(wall, mono, thread_cpu) read now in the calling thread, the given clocks in place of the module's where a caller
    stamps its call with a time of its own, or inside recorded_clock the next recorded reading."""
    if _STAMPS:
        return _STAMPS.pop(0)
    return (wall_clock or WALL_CLOCK)(), (monotonic_clock or MONOTONIC_CLOCK)(), THREAD_CLOCK()


def new_state(seed, config, facts_path, wall_time, monotonic_time=None):
    """A new state file. wall_time and monotonic_time are the two clocks at the start of the episode, time.time() and
    time.monotonic(); the monotonic clock does not advance while the machine sleeps and does not step when the wall clock
    is set, so the wall gap of a call minus its monotonic gap is the sleep or the clock step inside it."""
    return {
        "format": STATE_FORMAT,
        "seed": seed,
        "config": config.to_json(),
        "facts_path": public_path(facts_path),  # relative to the fork root, or the basename of a file outside it
        "actions": [],
        "think_cycles": [],
        "think_seconds": [],
        "think_seconds_carried": 0.0,
        "tool_calls": [],
        "last_call_wall_time": wall_time,
        "episode_started_at": wall_time,
        "called_at": [],
        "think_seconds_monotonic": [],
        "think_seconds_wall": [],
        "last_call_monotonic_time": monotonic_time,
        # the chess clock (servers/thinking_clock.py): the handover H, the cursor u, the byte offsets of the transcript and of the
        # ledger of failed model attempts read so far, the segments of B that end after the cursor, the servers whose first
        # start is credited, the service of the counted call in flight, the processes refused by their ready line, and the totals
        "handover_wall": None,
        "handover_mono": None,
        "charged_until_mono": None,
        "transcript_cursor": 0,
        "failures_cursor": 0,
        "credited_segments": [],
        "started_servers": [],
        "open_service": None,
        "refused_processes": {},
        "credited_seconds": {"starts": 0.0, "calls": 0.0},
        "uncredited_seconds": {"starts": 0.0, "calls": 0.0},
        "failed_model_seconds": 0.0,
        "failed_model_attempts": 0,
    }


def load_state(path):
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("format") != STATE_FORMAT:
        raise ValueError(f"unknown episode state format in {path}")
    return state


def save_state(path, state):
    """Write through a temporary file and a rename, so a server stopped mid-write never leaves a torn state file."""
    write_json_atomically(path, state, indent=1)


def released_path(state_path):
    """The released evidence beside the state file, which the diagnostics server reads."""
    return Path(state_path).with_name(RELEASED_NAME)


def write_released(state_path, env, compose=briefing):
    """Everything the agent has been told: the question, the public rules, the status, the public events and the served
    observations. Never the seed, the configuration, the data path, a recording index or the hidden state. The rules carry
    under "clock" the clock of the manifest, the mode, seconds_per_cycle, the declared durations and the
    order of the processes in one cycle, and under "tools" the price and the declared duration of every tool,
    so that the diagnostics server prices a read from what the agent was told.
    The processes block carries the parameters of the rules, and the fault rule of the starts never enters."""
    profile = env.public_profile()
    config = env.config
    events = env.public_events()
    status = {"time": env.time}
    if config.anchors is not None:
        status["day"] = config.date_of(env.time)
    if config.budget is not None:
        status["budget_left"] = env.budget_left()
    status.update({"steps_left": config.max_steps - env.steps, "open_work_orders": env.public_work_orders(), "episode_over": env.done})
    released = {
        "format": RELEASED_FORMAT,
        "asset_id": config.asset_id,
        "task_text": compose(profile, config),
        "guidance_level": config.guidance_level,
        "rules": {key: profile[key] for key in ("components", "levels", "healthy", "sensor_groups", "statistics", "prices", "durations",
                                                 "downtime_rate", "budget", "horizon", "max_steps", "processes", "clock_mode",
                                                 "clock", "tools", "normal_ranges")},
        "status": status,
        "events": events,
        "observations": [{"n": o["n"], "t": o["t"], "sensor_group": e["sensor_group"], "values": o["values"]}
                         for e in events if e["event"] == "read" for o in e["served"]],
    }
    write_json_atomically(released_path(state_path), released, indent=1)


def start(state_path, seed, config, facts_path, env_class=HydraulicRigEnv, compose=briefing):
    """Create an episode: its state file and its first released evidence (the question, the rules and no events). The
    thinking time counts from the handover (hand_over), never from here, so the driver's work between the two is not
    charged. env_class and compose let another engine and its own briefing, those of the AssetOpsBench rows, use the same
    state file and released evidence. The start line of the transcript records both clock readings of the start, from which
    the first call measures its wall and monotonic gaps, so that a replay starts the episode under the same readings."""
    state = new_state(seed, config, facts_path, WALL_CLOCK(), MONOTONIC_CLOCK())
    save_state(state_path, state)
    write_released(state_path, replay(state, env_class=env_class), compose=compose)
    transcript.begin(Path(state_path).parent, wall=state["episode_started_at"], mono=state["last_call_monotonic_time"],
                     released_sha256=transcript.file_sha256(released_path(state_path)))
    return state


def hand_over(state_path, wall=None, mono=None):
    """The handover H: the instant the task goes to the agent, the origin of the thinking clock and the start of its cursor.
    The driver calls it immediately before it runs the agent, and a caller that plays an episode in its own process calls it
    before the first call. It appends the handover line and records the two readings in the state under the state lock; a
    second handover is refused. Returns (wall, mono)."""
    with locked(state_path):
        state = load_state(state_path)
        if state.get("handover_mono") is not None:
            raise ValueError(SECOND_HANDOVER)
        if wall is None or mono is None:  # given readings are taken as they are, so a handover inside recorded_clock reads none
            now_wall, now_mono, _thread = stamp()
            wall = now_wall if wall is None else wall
            mono = now_mono if mono is None else mono
        state.update(handover_wall=wall, handover_mono=mono, charged_until_mono=mono)
        save_state(state_path, state)
        transcript.append(Path(state_path).parent, "handover", wall=wall, mono=mono)
    return wall, mono


def enter(folder, server, tool, wall_clock=None, monotonic_clock=None):
    """The entry e of a served call of a tool that steps the environment, read in the thread that serves it after the call
    holds the hold of its process, with its line in the transcript: (wall, mono, thread_cpu) and the line number n."""
    wall, mono, thread = stamp(wall_clock, monotonic_clock)
    n = transcript.append(folder, "enter", server=server, tool=tool, wall=wall, mono=mono, thread_cpu=thread)
    transcript.note(enter_n=n)  # a call that fails before its own line is written still names its entry in the line of the wrapper
    return {"wall": wall, "mono": mono, "thread": thread, "n": n}


def leave(folder, server, entered, outcome):
    """The exit line of a served call, written in the handler's finally: answered, cancelled or raised."""
    transcript.append(folder, "exit", server=server, enter_n=entered["n"], outcome=outcome)


def end_fields(entered, wall_clock=None, monotonic_clock=None):
    """The end f of the service of a stepped call, read immediately before its transcript line is appended, as the fields of
    that line. A call that a caller stamps with a time of its own has no measured service, so its end is its entry and its
    credit is zero."""
    if wall_clock is not None or monotonic_clock is not None:
        wall, mono, thread = entered["wall"], entered["mono"], entered["thread"]
    else:
        wall, mono, thread = stamp()
    return {"enter_n": entered["n"], "end_wall": wall, "end_mono": mono, "end_thread_cpu": thread}


def _baseline(folder, server):
    """The launcher's SHA-256 and the thread count of the kind of a server, as the driver recorded them, else nothing."""
    path = Path(folder) / BASELINE_FILE
    try:
        record = json.loads(path.read_text(encoding="utf-8")).get(server) or {}
    except (OSError, ValueError):
        return {}
    return {key: record[key] for key in ("launcher_sha256", "threads") if key in record}


def catch_up(state, folder):
    """Read what arrived since the last counted call into the clock of the state: the first ready line of each server credits
    its start, a ready line with a problem refuses its process, the transcript line of the counted call in service moves the
    cursor by its credit, and every failed model attempt credits its span. The segments that end before the cursor are
    dropped, since no later charge can reach them."""
    lines, state["transcript_cursor"] = transcript.read_since(folder, state["transcript_cursor"])
    segments = [tuple(s) for s in state["credited_segments"]]
    for line in lines:
        kind = line.get("kind")
        if kind == "ready":
            problem = ready.problem(line, **_baseline(folder, line.get("server")))
            if problem is not None:
                state["refused_processes"][str(line.get("pid"))] = problem
            if line.get("server") in state["started_servers"]:
                continue  # only the first start of a server is credited
            state["started_servers"].append(line.get("server"))
            segment = thinking_clock.start_segment(line)
            length = line["r_mono"] - line["s_mono"] if line.get("s_mono") is not None else 0.0
            credited = segment[1] - segment[0] if segment else 0.0
            state["credited_seconds"]["starts"] += credited
            state["uncredited_seconds"]["starts"] += max(0.0, length - credited)
            if segment:
                segments.append(segment)
        elif kind == "call" and state["open_service"] is not None and line.get("enter_n") == state["open_service"]["n"]:
            served = state["open_service"]
            length = line["end_mono"] - served["e"]
            g = thinking_clock.credit(length, line["end_thread_cpu"] - served["e_thread"])
            state["charged_until_mono"] = thinking_clock.advance(state["charged_until_mono"], served["e"], line["end_mono"], g)
            state["credited_seconds"]["calls"] += g
            state["uncredited_seconds"]["calls"] += max(0.0, length - g)
            state["open_service"] = None
    spans, state["failures_cursor"] = model_failures.failed_spans_since(folder, state["failures_cursor"])
    for begin, following in spans:
        segments.append((begin, following))
        state["failed_model_seconds"] += max(0.0, following - begin)
        state["failed_model_attempts"] += 1
    u = state["charged_until_mono"]
    state["credited_segments"] = thinking_clock.after(segments, u) if u is not None else [list(s) for s in thinking_clock.merged(segments)]


def accrued_thinking(state_path, now_mono=None):
    """(seconds, cycles) that a counted call entering at now_mono would be charged, |[u, now] \\ B| through the cycle rule with
    the carried rest, the monotonic clock read now when None. It moves neither the cursor nor the rest: the state file is read
    and not written."""
    state = load_state(state_path)
    if state.get("handover_mono") is None:
        return 0.0, 0
    folder = Path(state_path).parent
    catch_up(state, folder)
    config = EpisodeConfig.from_json(state["config"])
    if config.clock_mode == TOKEN_MODE:  # the token clock charges the tokens generated since the last counted call, as step does
        seconds = token_seconds(config, generated_total(config, folder) - state.get("generated_tokens_read", 0))
    else:
        now_mono = MONOTONIC_CLOCK() if now_mono is None else now_mono
        seconds = thinking_clock.charge(state["charged_until_mono"], now_mono, state["credited_segments"])
    cycles, _seconds, _carried = think_cycles(config, state["think_seconds_carried"], seconds)
    return seconds, cycles


def clock_report(state_path):
    """The clock of an episode for the reports, the state file read and not written: the credited and the uncredited seconds
    of the starts and of the counted calls, the failed model seconds and attempts, the seconds charged, and the cursor after
    the last counted call, from which the driver measures the time after that call that no call follows and nothing charges."""
    state = load_state(state_path)
    if state.get("handover_mono") is not None:
        catch_up(state, Path(state_path).parent)
    return {"credited_seconds": state["credited_seconds"], "uncredited_seconds": state["uncredited_seconds"],
            "failed_model_seconds": state["failed_model_seconds"], "failed_model_attempts": state["failed_model_attempts"],
            "charged_seconds": sum(state["think_seconds"]), "charged_until_mono": state["charged_until_mono"]}


def load_world(facts_path, asset_id=DEFAULT_ASSET):
    """The recorded cycles and the stable blocks of a data file of an asset, loaded once per process (world.load_world)."""
    return world.load_world(asset_id, str(facts_path))


def replay(state, env_class=HydraulicRigEnv):
    """The environment after reset(seed) and every stored action with its stored thinking cycles. env_class is
    HydraulicRigEnv or a subclass of it, such as the engine of the AssetOpsBench rows (recorded_history_env.py)."""
    config = EpisodeConfig.from_json(state["config"])
    cycles, blocks = load_world(resolve_recorded_path(state["facts_path"], get_profile(config.asset_id)) or state["facts_path"], config.asset_id)
    env = env_class(cycles, blocks, config)
    env.reset(state["seed"])
    for action, think in zip(state["actions"], state["think_cycles"]):
        if env.done:
            raise ValueError("the stored actions run past the end of the episode")
        env.step(action, think_cycles=think)
    return env


def opening(state):
    """The envelope a policy starts from: the current clock and counters with the asset profile as result."""
    env = replay(state)
    return env.envelope(env.public_profile())


def step(state, env, action, entered, folder, generated=None):
    """Apply one action to the stored episode after the thinking time since the previous call, the one rule of the action
    server and of the metering layer of the AssetOpsBench rows. The clock first reads what arrived since the call before
    (catch_up), and a call of an episode never handed over or of a process refused by its ready line is refused by name. The
    seconds charged are those of the chess clock, |[u, e] \\ B| for the call entering at e (servers/thinking_clock.py), or in
    the generated_tokens mode the tokens generated since the previous call, the total of the ledger generated less the total
    read before, times seconds_per_token; they go through think_cycles, then env.step, and the action, its cycles, the seconds
    charged and the seconds carried are kept in the state for replay, with the tokens charged and the total read in the
    generated_tokens mode. Beside them the state keeps the wall time of the entry, and the wall and the monotonic seconds since
    the previous entry where the wall seconds were charged (wall_clock.monotonic_gap), which no replay reads, and the service of this call,
    which the next call credits from the end in its transcript line. Returns what env.step returns."""
    catch_up(state, folder)
    if state.get("handover_mono") is None:
        raise ValueError(NO_HANDOVER)
    refusal = state["refused_processes"].get(str(os.getpid()))
    if refusal is not None:
        raise ValueError(f"{refusal}, so its calls are refused")
    if env.config.clock_mode == TOKEN_MODE:
        spent = token_clock.seconds_since_last_call(state, env.config, generated)
    else:
        spent = thinking_clock.charge(state["charged_until_mono"], entered["mono"], state["credited_segments"])
        # the cursor passes the entry with the charge, so a call whose line is never written is not charged again by the next
        # call; its service line, when it comes, advances from max(u, e) as from u
        state["charged_until_mono"] = max(state["charged_until_mono"], entered["mono"])
    cycles, seconds, carried = think_cycles(env.config, state["think_seconds_carried"], spent)
    state["called_at"].append(entered["wall"])
    state["think_seconds_monotonic"].append(monotonic_gap(env.config, state.get("last_call_monotonic_time"), entered["mono"]))
    state["think_seconds_wall"].append(monotonic_gap(env.config, state.get("last_call_wall_time"), entered["wall"]))
    answer = env.step(action, think_cycles=cycles)
    state["actions"].append(action)
    state["think_cycles"].append(cycles)
    state["think_seconds"].append(seconds)
    state["think_seconds_carried"] = carried
    state["open_service"] = {"n": entered["n"], "e": entered["mono"], "e_thread": entered["thread"]}
    return answer


def generated_total(config, folder):
    """The total of the ledger of generated tokens of an episode folder in the generated_tokens mode, None in every other mode,
    which reads no ledger. An episode of that mode whose folder holds no ledger is refused by name (NO_TOKEN_LEDGER)."""
    if config.clock_mode != TOKEN_MODE:
        return None
    total = GENERATED_TOKENS(folder)
    if total is None:
        raise ValueError(NO_TOKEN_LEDGER)
    return total


def ledger_fields(state):
    """The field of the transcript line of a stepped call that names the byte offset of the failure ledger its clock read, so
    that a replay feeds the ledger up to that offset before the call and reads the same failed attempts at the same call."""
    return {"failures_read": state["failures_cursor"]}


def generated_fields(generated):
    """The field of the transcript line of a call that read the ledger, the generated_tokens mode alone, so that the line of a
    call of every other mode keeps its bytes."""
    return {} if generated is None else {"generated_tokens": generated}


def call(state_path, tool, arguments, action, now=None, monotonic=None):
    """Apply one tool call to the stored episode and return the JSON text the agent sees.

    Every call reaches env.step, which refuses malformed or out-of-order actions itself and logs the reason, except a call
    after the episode has ended: env.step raises then, so the call is answered here as a refusal and only recorded. The
    answer is the envelope of the environment; never the info dict, which ends an episode with the full log and its hidden
    state. The released evidence is rewritten after every call, and the call then enters the transcript of the episode with
    the hash of its answer and, for a stepped call, the end of its service. The entry is read first, in the thread that serves
    the call, with its line in the transcript, and the exit line follows in every case. `now` and `monotonic` are the wall
    clock and the monotonic clock of the call for a caller that stamps it with a time of its own, read from the clocks of this
    module when None. The call holds the lock of the state file from its load to the line of the transcript, as
    StateFileSessions of the metering layer does, so that concurrent calls of one episode run one after the other and none
    loses the action of another."""
    folder = Path(state_path).parent
    clocks = {"wall_clock": None if now is None else (lambda: now), "monotonic_clock": None if monotonic is None else (lambda: monotonic)}
    entered = enter(folder, contract.ACTION_SERVER, tool, **clocks)
    outcome = "raised"
    try:
        with locked(state_path):
            state = load_state(state_path)
            env = replay(state)
            generated = generated_total(env.config, folder)
            stepped = not env.done
            if stepped:
                response, _cost, _done, _info = step(state, env, action, entered, folder, generated)
            else:
                response = env.envelope(refused=EPISODE_OVER)
            state["last_call_wall_time"] = entered["wall"]
            state["last_call_monotonic_time"] = entered["mono"]
            text = json.dumps(response, separators=(",", ":"))
            state["tool_calls"].append({"tool": tool, "arguments": arguments, "response": text})
            save_state(state_path, state)
            write_released(state_path, env)
            ended = {**end_fields(entered, **clocks), **ledger_fields(state)} if stepped else {"enter_n": entered["n"]}
            transcript.answered(folder, contract.ACTION_SERVER, tool, arguments, text, False, stepped=stepped, wall=entered["wall"],
                                released_sha256=transcript.file_sha256(released_path(state_path)), **generated_fields(generated), **ended)
        outcome = "answered"
        return text
    finally:
        leave(folder, contract.ACTION_SERVER, entered, outcome)
