"""One tiny world episode behind a stateless tool server (docs/hydraulic_rig/action-server.md).

AssetOpsBench's Plan-Execute runner starts a new server process for every tool call, so an episode lives in a JSON
state file rather than in server memory: the environment parameters, the seed and every action passed to env.step,
refused ones included because they count toward the step cap. Each call rebuilds the environment by reset(seed) and
replay, which is exact because every draw comes from the generator seeded at reset (docs/hydraulic_rig/environment.md), applies the new
action, and saves the file before it answers. Beside the state file the server keeps the released answers, everything
the agent has been told and nothing else, for tools that reason over the episode without touching the environment,
such as the causal-diagnosis server.
"""
import inspect
import json
from functools import lru_cache
from pathlib import Path

from envs.hydraulic_rig.data import build_blocks, load_cycles
from envs.hydraulic_rig.env import DEFAULT_PRICES, TinyHydraulicEnv
from servers.json_files import write_json_atomically

from .names import ASSET_ID, public_result, tier_fields

STATE_FORMAT = 1
RELEASED_FORMAT = 1
RELEASED_NAME = "released.json"
STATUS_KEYS = ("phase", "budget_left", "repairs_left", "steps_left")


def has_clock(state_path):
    """True when the episode of this state file runs with the clock, so the server offers advance_time."""
    return load_state(state_path)["env_params"].get("clock") is not None


def default_env_kwargs():
    """The environment's own defaults, so that a state file records every parameter explicitly."""
    params = inspect.signature(TinyHydraulicEnv.__init__).parameters
    kwargs = {name: p.default for name, p in params.items() if p.default is not inspect.Parameter.empty}
    kwargs["prices"] = kwargs["prices"] or DEFAULT_PRICES
    return kwargs


def encode_params(env_kwargs):
    """JSON form of TinyHydraulicEnv keyword arguments; decode_params inverts it."""
    out = dict(env_kwargs)
    prices = env_kwargs["prices"]
    out["prices"] = {"measure": {str(t): p for t, p in prices["measure"].items()}, "parts": dict(prices["parts"]), "downtime": prices["downtime"]}
    allowed = env_kwargs.get("allowed")
    out["allowed"] = None if allowed is None else sorted(list(c) for c in allowed)
    return out


def decode_params(params):
    """JSON turns the tier keys into strings and the combinations into lists; the environment needs ints and tuples."""
    out = dict(params)
    prices = params["prices"]
    out["prices"] = {"measure": {int(t): p for t, p in prices["measure"].items()}, "parts": dict(prices["parts"]), "downtime": prices["downtime"]}
    out["allowed"] = None if params["allowed"] is None else {tuple(c) for c in params["allowed"]}
    out.setdefault("clock", None)  # state files written before the clock existed have no clock
    return out


def new_state(seed, env_kwargs, facts_path):
    return {
        "format": STATE_FORMAT,
        "seed": seed,
        "env_params": encode_params(env_kwargs),
        "facts_path": str(facts_path),
        "actions": [],
        "tool_calls": [],
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
    """The released answers beside the state file, which the causal-diagnosis server reads."""
    return Path(state_path).with_name(RELEASED_NAME)


def write_released(state_path, env, state):
    """Everything the agent has been told: the public rules, the counters and the agent-facing history as the tool
    answers showed it. Never the seed, the allowed combinations, the facts path, a cycle number or the hidden state."""
    obs = env.observation()
    prices = obs["prices"]
    released = {
        "format": RELEASED_FORMAT,
        "asset_id": ASSET_ID,
        "task_text": state.get("task_text"),
        "rules": {
            "min_faulty": obs["min_faulty"],
            "max_faulty": obs["max_faulty"],
            "budget": env.budget,
            "max_repairs": env.max_repairs,
            "max_steps": env.max_steps,
            "prices": {"measure": {str(t): p for t, p in prices["measure"].items()}, "parts": dict(prices["parts"]), "downtime": prices["downtime"]},
            "components": obs["components"],
            "levels": obs["levels"],
            "healthy": obs["healthy"],
            "tiers": {str(t): tier_fields(t) for t in obs["tiers"]},
        },
        **({"clock_rules": obs["clock_rules"]} if "clock_rules" in obs else {}),
        "status": _status(env),
        "events": [_agent_view(e) for e in obs["history"]],
    }
    write_json_atomically(released_path(state_path), released, indent=1)


def start(state_path, seed, env_kwargs, facts_path, *, task_text=None):
    """Create an episode: its state file and its first released answers (the rules and an empty history)."""
    state = new_state(seed, env_kwargs, facts_path)
    if task_text is not None:
        state["task_text"] = task_text
    save_state(state_path, state)
    write_released(state_path, replay(state), state)
    return state


@lru_cache(maxsize=2)
def load_world(facts_path):
    cycles = load_cycles(facts_path)
    return cycles, build_blocks(cycles)


def replay(state):
    """The environment after reset(seed) and every stored action."""
    cycles, blocks = load_world(state["facts_path"])
    env = TinyHydraulicEnv(cycles, blocks, **decode_params(state["env_params"]))
    env.reset(state["seed"])
    for action in state["actions"]:
        if env.done:
            raise ValueError("the stored actions run past the end of the episode")
        env.step(action)
    return env


def call(state_path, tool, arguments, action):
    """Apply one tool call to the stored episode and return the JSON text the agent sees.

    Every call reaches env.step, which refuses malformed or out-of-order actions itself and logs the reason, except a
    call after the episode has ended: env.step raises then, so the call is answered here and only recorded. The answer
    carries what docs/hydraulic_rig/environment.md lets the tool server forward: the observation (the new history entry, which holds the
    refusal reason, and the counters), the cost and the done flag; never the info dict, which ends an episode with the
    full log and its hidden state. The released answers are rewritten after every call."""
    state = load_state(state_path)
    env = replay(state)
    if env.done:
        response = {**_status(env), "note": "the episode is over; this call changed nothing"}
    else:
        seen = len(env.observation()["history"])
        _obs, cost, _done, _info = env.step(action)
        state["actions"].append(action)
        new_events = env.observation()["history"][seen:]
        response = {"event": _agent_view(new_events[-1]) if new_events else None, "cost": cost, **_status(env)}
    text = json.dumps(response, separators=(",", ":"))
    state["tool_calls"].append({"tool": tool, "arguments": arguments, "response": text})
    save_state(state_path, state)
    write_released(state_path, env, state)
    return text


def _status(env):
    obs = env.observation()
    status = {key: obs[key] for key in STATUS_KEYS}
    if "clock" in obs:
        status["clock"] = obs["clock"]
    status["episode_over"] = env.done
    return status


def _agent_view(event):
    """The history entry of the environment, with the measurement as tool answers show it (names.public_result)."""
    if event["event"] == "measure":
        return dict(event, result=public_result(event["result"]))
    return event
