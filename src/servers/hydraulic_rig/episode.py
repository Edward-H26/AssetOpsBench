"""One hydraulic rig episode behind a stateless tool server (docs/hydraulic_rig/action-server.md).

AssetOpsBench's Plan-Execute runner starts a new server process for every tool call, so an episode lives in a JSON
state file rather than in server memory: the seed, the configuration, the data file, every action passed to env.step,
refused ones included because they count toward the step cap, and the thinking time charged before each action. Each
call rebuilds the environment by reset(seed) and replay, which is exact because every draw comes from the generator
seeded at reset and the thinking cycles are stored, applies the new action, and saves the file before it answers. Beside
the state file the server keeps the released evidence, everything the agent has been told and nothing else, for tools
that reason over the episode without touching the environment, such as the diagnostics server.
"""
import json
import math
import time
from functools import lru_cache
from pathlib import Path

from envs.hydraulic_rig.config import EpisodeConfig
from envs.hydraulic_rig.data import build_blocks, load_cycles
from envs.hydraulic_rig.env import HydraulicRigEnv
from servers.json_files import write_json_atomically

from .briefing import briefing

STATE_FORMAT = 2
RELEASED_FORMAT = 2
RELEASED_NAME = "released.json"
EPISODE_OVER = "the episode is over; this call changed nothing"


def new_state(seed, config, facts_path, wall_time):
    return {
        "format": STATE_FORMAT,
        "seed": seed,
        "config": config.to_json(),
        "facts_path": str(facts_path),
        "actions": [],
        "think_cycles": [],
        "tool_calls": [],
        "last_call_wall_time": wall_time,
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


def write_released(state_path, env):
    """Everything the agent has been told: the question, the public rules, the status, the public events and the served
    observations. Never the seed, the configuration, the data path, a recording index or the hidden state."""
    profile = env.public_profile()
    config = env.config
    events = env.public_events()
    status = {"time": env.time}
    if config.anchors is not None:
        status["day"] = config.date_of(env.time)
    status.update({"budget_left": config.budget - env.cost, "steps_left": config.max_steps - env.steps,
                   "open_work_orders": env.public_work_orders(), "episode_over": env.done})
    released = {
        "format": RELEASED_FORMAT,
        "asset_id": config.asset_id,
        "task_text": briefing(profile, config),
        "guidance_level": config.guidance_level,
        "rules": {key: profile[key] for key in ("components", "levels", "healthy", "sensor_groups", "statistics", "prices", "durations",
                                                 "downtime_rate", "budget", "horizon", "max_steps", "fault_rule", "clock_mode", "healthy_bands")},
        "status": status,
        "events": events,
        "observations": [{"n": o["n"], "t": o["t"], "sensor_group": e["sensor_group"], "values": o["values"]}
                         for e in events if e["event"] == "read" for o in e["served"]],
    }
    write_json_atomically(released_path(state_path), released, indent=1)


def start(state_path, seed, config, facts_path):
    """Create an episode: its state file and its first released evidence (the question, the rules and no events). The
    thinking time of the first call counts from here."""
    state = new_state(seed, config, facts_path, time.time())
    save_state(state_path, state)
    write_released(state_path, replay(state))
    return state


@lru_cache(maxsize=2)
def load_world(facts_path):
    cycles = load_cycles(facts_path)
    return cycles, build_blocks(cycles)


def replay(state):
    """The environment after reset(seed) and every stored action with its stored thinking cycles."""
    cycles, blocks = load_world(state["facts_path"])
    env = HydraulicRigEnv(cycles, blocks, EpisodeConfig.from_json(state["config"]))
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


def think_cycles(config, last_wall_time, now):
    """The agent's thinking time since the last call in cycles, rounded up; charged in the generation_time mode only."""
    if config.clock_mode != "generation_time":
        return 0
    return max(0, math.ceil((now - last_wall_time) / config.seconds_per_cycle))


def call(state_path, tool, arguments, action, now=None):
    """Apply one tool call to the stored episode and return the JSON text the agent sees.

    Every call reaches env.step, which refuses malformed or out-of-order actions itself and logs the reason, except a call
    after the episode has ended: env.step raises then, so the call is answered here as a refusal and only recorded. The
    answer is the envelope of the environment; never the info dict, which ends an episode with the full log and its hidden
    state. The released evidence is rewritten after every call. `now` is the wall clock of the call, taken from
    time.time() when None."""
    state = load_state(state_path)
    env = replay(state)
    now = time.time() if now is None else now
    if env.done:
        response = env.envelope(refused=EPISODE_OVER)
    else:
        think = think_cycles(env.config, state["last_call_wall_time"], now)
        response, _cost, _done, _info = env.step(action, think_cycles=think)
        state["actions"].append(action)
        state["think_cycles"].append(think)
    state["last_call_wall_time"] = now
    text = json.dumps(response, separators=(",", ":"))
    state["tool_calls"].append({"tool": tool, "arguments": arguments, "response": text})
    save_state(state_path, state)
    write_released(state_path, env)
    return text
