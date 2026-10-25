"""Run an agent or a reference policy on hydraulic-rig episodes through the two MCP servers and score every episode from
its log (benchmarks/hydraulic_rig/README.md).

usage, from the fork root in the runtime of benchmarks/hydraulic_rig/requirements-smoke.txt:
  uv run --no-project --python 3.12 --with-requirements benchmarks/hydraulic_rig/requirements-smoke.txt \\
      python benchmarks/hydraulic_rig/run_episodes.py [--scenario S1|S2|seeded] [--agent AGENT] [--per-record K]
      [--episodes N] [--first-seed S] [--model MiniMax-M3] [--base-url URL] [--out benchmarks/hydraulic_rig/runs]

Agents. plan_execute runs AssetOpsBench's Plan-Execute runner unchanged with both servers registered, the rig's five
action tools and the seven causal-diagnosis tools; plan_execute_rig_only registers the action server alone, the control
without diagnosis tools. Both reach the model through AssetOpsBench's own OpenAI-compatible backend, pointed inside this
process at TokenRouter, the router of AssetOpsBench's model panel, or at the endpoint --base-url names, with the key in
API_KEY, the only key the drivers read. scripted runs
the same runner with a fixed fifteen-step plan that calls every tool of both servers, a wiring check without a model. random, greedy and
greedy_prior_1to2 are the reference policies of benchmarks/hydraulic_rig/run_policies.py, each action sent as a tool call through
the action server by the runner's own stdio helper, so that their rows are comparable with a model's; oracle reads the
hidden state, which no tool releases, and runs in process.

Every episode folder holds the state file, the released answers, the predictions the causal tools stored, the replayed
log, the score, the plan, the model calls, the trajectory of both servers and an audit report (report.py). A Plan-Execute
run plans once, executes the plan and summarises, so an episode that the plan leaves open stays open and is scored as it
stands.
"""
import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (CAUSAL_SERVER, CAUSAL_TOOLS, RIG_SERVER, RIG_TOOLS, ROOT, SCENARIOS, SRC, answer_claims_recovery,  # noqa: E402
                    load_world, scenario_env_kwargs, scenario_episodes, scenario_task_text, start_episode, write_launchers)
from envs.hydraulic_rig.classifiers import TierClassifiers  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv  # noqa: E402
from envs.hydraulic_rig.policies import OraclePolicy, run_episode  # noqa: E402
from envs.hydraulic_rig.scoring import score  # noqa: E402
from report import episode_report, run_summary  # noqa: E402
from run_policies import policy_table  # noqa: E402
from servers.hydraulic_rig import episode  # noqa: E402
from servers.hydraulic_rig.briefing import BRIEFING_VERSION, briefing  # noqa: E402
from servers.hydraulic_rig.policy_client import PublicClassifiers, play, tool_call  # noqa: E402
from servers.causal_diagnosis.fit import ensure_models  # noqa: E402

ROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
DEFAULT_MODEL = "MiniMax-M3"
MODEL_AGENTS = ("plan_execute", "plan_execute_rig_only")
POLICIES = tuple(name for name in policy_table(None, None) if name != "oracle")
AGENTS = (*MODEL_AGENTS, "scripted", "oracle", *POLICIES)
TOOLS_NOTE_VERSION = "v1"
TOOLS_NOTE = (
    "\n\nA second server, causal_diagnosis, offers read-only diagnosis tools over the measurements released so far; they never "
    "change the rig: get_symptom_report() gives the task and the public rules; get_episode_evidence(sensor_group) lists the "
    "released values with their healthy bands, sensor_group being all, tier1, tier2, tier3 or a channel; "
    "rank_root_causes(top_k) ranks the candidate fault combinations and says whether the evidence is insufficient; "
    "estimate_intervention_effect(action) predicts what monitor or replace:<component> would do and returns a prediction id; "
    "verify_outcome(prediction_id) compares a prediction with the measurement taken after the action; get_asset_context(asset_id) "
    "and query_fmea_paths(asset_id, node) describe the asset hyd_1 and its failure modes."
)


def _register_runner_package():
    """src/agent/__init__.py imports every SDK runner with its packages, which the Plan-Execute modules do not need, so the
    package is registered by its path and its __init__ skipped; no bytecode is written, so the tree stays as checked out."""
    sys.dont_write_bytecode = True
    package = types.ModuleType("agent")
    package.__path__ = [str(SRC / "agent")]
    sys.modules.setdefault("agent", package)


_register_runner_package()
from agent.plan_execute.executor import _call_tool, _list_tools  # noqa: E402
from agent.plan_execute.runner import PlanExecuteRunner  # noqa: E402
from llm import LLMBackend, make_backend  # noqa: E402
from observability import set_run_context  # noqa: E402


class RecordingBackend(LLMBackend):
    """Keeps every prompt and reply of an episode with its token counts and duration."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def generate(self, prompt, temperature=0.0):
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(self, prompt, temperature=0.0):
        call = {"prompt": prompt}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            result = self.inner.generate_with_usage(prompt, temperature)
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            call["seconds"] = round(time.perf_counter() - started, 2)
        call.update(reply=result.text, input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        return result

    @property
    def model_id(self):
        return self.inner.model_id


class ScriptedBackend(LLMBackend):
    """Wiring check without a model: a fixed plan over both servers, fixed arguments and a fixed answer, returned in the
    order the runner asks for them (the plan, one argument reply per tool step, the summary). The diagnosis and the
    repair are fixed too, so the episode's score says nothing about diagnosis; the run checks that every tool of both
    servers answers through the runner and that the prediction stored by one call is read back by the next. The second
    submission is refused because the diagnosis is locked, which is the answer the wiring check expects."""

    STEPS = [
        ("Read the asset context", CAUSAL_SERVER, "get_asset_context", '{"asset_id": "hyd_1"}'),
        ("Read the failure-mode paths of the accumulator", CAUSAL_SERVER, "query_fmea_paths", '{"asset_id": "hyd_1", "node": "accumulator"}'),
        ("Read the complaint and the public rules", CAUSAL_SERVER, "get_symptom_report", "{}"),
        ("Read the cheapest sensor tier", RIG_SERVER, "measure", '{"tier": 1}'),
        ("Read the second sensor tier", RIG_SERVER, "measure", '{"tier": 2}'),
        ("Read the full sensor tier", RIG_SERVER, "measure", '{"tier": 3}'),
        ("List the released evidence with its healthy bands", CAUSAL_SERVER, "get_episode_evidence", '{"sensor_group": "all"}'),
        ("Rank the candidate fault combinations", CAUSAL_SERVER, "rank_root_causes", '{"top_k": 3}'),
        ("Submit a diagnosis", RIG_SERVER, "submit", '{"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130}'),
        ("Try to declare the evidence insufficient after the diagnosis", RIG_SERVER, "submit_insufficient_evidence", '{"next_action": "measure tier 3"}'),
        ("Predict the effect of replacing the pump", CAUSAL_SERVER, "estimate_intervention_effect", '{"action": "replace:pump"}'),
        ("Replace the pump", RIG_SERVER, "replace", '{"component": "pump"}'),
        ("Read the cheapest tier after the repair", RIG_SERVER, "measure", '{"tier": 1}'),
        ("Verify the prediction against the reading", CAUSAL_SERVER, "verify_outcome", '{"prediction_id": "pred/1"}'),
        ("End the episode", RIG_SERVER, "finish", "{}"),
    ]

    def __init__(self):
        self._model_id = "scripted"
        self._replies = iter(())

    def start_episode(self):
        plan = "\n\n".join(
            f"#Task{n}: {task}\n#Server{n}: {server}\n#Tool{n}: {tool}\n"
            f"#Dependency{n}: {'None' if n == 1 else f'#S{n - 1}'}\n#ExpectedOutput{n}: the tool's answer"
            for n, (task, server, tool, _) in enumerate(self.STEPS, 1)
        )
        self._replies = iter([plan, *(arguments for _, _, _, arguments in self.STEPS), "scripted run complete"])

    def generate(self, prompt, temperature=0.0):
        return next(self._replies)

    @property
    def model_id(self):
        return self._model_id


def router_backend(model, base_url=ROUTER_BASE_URL):
    key = os.environ.get("API_KEY")
    if not key:
        raise SystemExit("API_KEY is not set; configure a key or run with --agent scripted")
    os.environ["TOKENROUTER_BASE_URL"] = base_url
    os.environ["TOKENROUTER_API_KEY"] = key
    return make_backend(f"tokenrouter/{model}")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def code_versions():
    """The fork commit and the SHA-256 prefixes of the uncommitted code a run depends on."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = sorted([*(SRC / "envs" / "hydraulic_rig").glob("*.py"), *(SRC / "servers" / "hydraulic_rig").glob("*.py"),
                    *(SRC / "servers" / "causal_diagnosis").glob("*.py"), *(SRC / "servers" / "causal_diagnosis").glob("models.json"),
                    *HERE.glob("*.py")])
    return {"commit": commit, "sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in files}}


def prepare_episode(ep_dir, seed, env_kwargs, facts_path, scenario, servers):
    """The episode folder with its state file, released answers, launcher files and question; returns the launcher
    paths for the runner, the question and the observation at reset."""
    state_path = start_episode(ep_dir, seed, env_kwargs, facts_path, task_text=scenario_task_text(scenario))
    launchers = write_launchers(ep_dir, servers)
    observation = episode.replay(episode.load_state(state_path)).observation()
    question = briefing(observation) + (TOOLS_NOTE if CAUSAL_SERVER in servers else "")
    (ep_dir / "question.txt").write_text(question, encoding="utf-8")
    return launchers, question, observation


def check_servers(launchers):
    """List the tools of every registered server once before the first episode: a server that fails to start would
    otherwise only show up as unavailable inside the planner prompt."""
    expected = {RIG_SERVER: RIG_TOOLS, CAUSAL_SERVER: CAUSAL_TOOLS}
    for server, path in launchers.items():
        names = {tool["name"] for tool in asyncio.run(_list_tools(path))}
        if names != expected[server]:
            raise SystemExit(f"server {server} offers {sorted(names)}, expected {sorted(expected[server])}")


def run_plan_execute(backend, launchers, question, run_id, ep_dir):
    """One run of the unchanged runner. A plan with a malformed dependency raises out of run(), and so does an API error
    that outlasts the client's retries; either is recorded as the episode's error."""
    set_run_context(run_id=run_id, scenario_id=ep_dir.name)
    os.environ["AGENT_TRAJECTORY_DIR"] = str(ep_dir)
    runner = PlanExecuteRunner(llm=backend, server_paths=dict(launchers))
    started = time.perf_counter()
    try:
        result, error = asyncio.run(runner.run(question)), None
    except Exception as exc:  # noqa: BLE001 - a failed run is a result of the benchmark
        result, error = None, f"{type(exc).__name__}: {exc}"
    return result, error, time.perf_counter() - started


def run_policy(policy, server, observation):
    """One episode of a reference policy, each action sent as a tool call through the runner's own stdio helper."""

    def send(action):
        tool, arguments = tool_call(action)
        return asyncio.run(_call_tool(server, tool, arguments))

    started = time.perf_counter()
    try:
        play(policy, observation, send)
        error = None
    except Exception as exc:  # noqa: BLE001 - a failed episode is a result of the benchmark
        error = f"{type(exc).__name__}: {exc}"
    return error, time.perf_counter() - started


def run_oracle(ep_dir, seed, env_kwargs, cycles, blocks):
    """The oracle in process: it reads env.hidden, which no tool releases, so it cannot play through the servers."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    env = TinyHydraulicEnv(cycles, blocks, **env_kwargs)
    started = time.perf_counter()
    log = run_episode(env, OraclePolicy(env), seed)
    wall = time.perf_counter() - started
    scores = score(log)
    write_json(ep_dir / "log.json", log)
    write_json(ep_dir / "score.json", scores)
    return wall, {"tool_calls": "", "calls_after_end": "", "episode_ended": True, **_flat(scores)}


def _flat(scores):
    return {k: json.dumps(v) if isinstance(v, (dict, tuple)) else v for k, v in scores.items()}


def record(ep_dir):
    """Replay the stored actions, score the log, keep log and score beside the state file, and return the row."""
    state = episode.load_state(ep_dir / "episode.json")
    env = episode.replay(state)
    scores = score(env.log)
    write_json(ep_dir / "log.json", env.log)
    write_json(ep_dir / "score.json", scores)
    return {
        "tool_calls": len(state["tool_calls"]),
        "calls_after_end": sum("note" in json.loads(c["response"]) for c in state["tool_calls"]),
        "episode_ended": env.done,
        **_flat(scores),
    }


def record_plan_execute(ep_dir, result, calls, tool_calls):
    """What a failure analysis of a Plan-Execute run needs besides the log: every model call, the plan, the answer and
    the trajectory of both servers (the rig server records its own calls in the state file, the causal server records
    nothing but its predictions), plus the tool steps whose call reached no tool."""
    write_json(ep_dir / "llm_calls.json", calls)
    if result is not None:
        (ep_dir / "plan.txt").write_text(result.plan.raw, encoding="utf-8")
        (ep_dir / "answer.txt").write_text(result.answer or "", encoding="utf-8")
        write_json(ep_dir / "trajectory.json", [
            {"step": r.step_number, "task": r.task, "server": r.server, "tool": r.tool, "tool_args": r.tool_args,
             "response": r.response, "error": r.error, "duration_ms": r.duration_ms}
            for r in result.trajectory
        ])
    steps = result.plan.steps if result is not None else []
    trajectory = result.trajectory if result is not None else []
    tool_steps = sum(1 for r in trajectory if r.tool and r.tool.lower() not in ("none", "null"))
    rig_steps = sum(1 for r in trajectory if r.server == RIG_SERVER and r.tool and r.tool.lower() not in ("none", "null"))
    causal_calls = sum(1 for r in trajectory if r.server == CAUSAL_SERVER and r.success and r.tool)
    answer = result.answer if result is not None else ""
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
        "plan_steps": len(steps),
        "plan_tools": " ".join(f"{s.server}.{s.tool}" for s in steps),
        "tool_steps": tool_steps,
        "causal_calls": causal_calls,
        "calls_reaching_no_tool": rig_steps - tool_calls if result is not None else "",
        "answer_claims_recovery": answer_claims_recovery(answer),
    }


def main(argv):
    ap = argparse.ArgumentParser(description="agents and reference policies on hydraulic-rig episodes through the MCP servers")
    ap.add_argument("--scenario", choices=SCENARIOS, default="S1")
    ap.add_argument("--agent", choices=AGENTS, default="scripted")
    ap.add_argument("--per-record", type=int, default=None, help="start states per record of a class (default: all)")
    ap.add_argument("--max-records", type=int, default=None, help="only the first records of a class, for quick checks")
    ap.add_argument("--episodes", type=int, default=5, help="seeded scenario only: number of seeds")
    ap.add_argument("--first-seed", type=int, default=0, help="seeded scenario only: first seed")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=ROUTER_BASE_URL, help="OpenAI-compatible endpoint of the model agents (default: TokenRouter)")
    ap.add_argument("--facts", default=None, help="the rig data file (default: HYD_FACTS_PATH or the package's data folder)")
    ap.add_argument("--out", default="benchmarks/hydraulic_rig/runs")
    a = ap.parse_args(argv)

    facts_path, cycles, blocks = load_world(a.facts)
    env_kwargs = scenario_env_kwargs(a.scenario)
    pairs = scenario_episodes(a.scenario, cycles, blocks, a.per_record, a.episodes, a.first_seed, a.max_records)
    servers = (RIG_SERVER,) if a.agent == "plan_execute_rig_only" else (RIG_SERVER, CAUSAL_SERVER)
    if CAUSAL_SERVER in servers:
        ensure_models()  # fitted once here, before any server process starts and reads it
    backend = scripted = make_policy = None
    if a.agent in POLICIES:
        make_policy = policy_table(None, PublicClassifiers(TierClassifiers(cycles, blocks)))[a.agent]
        name, via = a.agent, "mcp"
    elif a.agent == "oracle":
        name, via = "oracle", "in_process"
    else:
        scripted = ScriptedBackend() if a.agent == "scripted" else None
        backend = RecordingBackend(scripted or router_backend(a.model, a.base_url))
        name, via = ("scripted" if scripted else f"{a.agent}_{a.model}"), "mcp"
    # a run folder inside the repository makes the runner start each server as `python -m <dotted path>`, which splits a
    # dot in any folder name, so the dots and slashes of a router model id such as z-ai/glm-5.3 become underscores here
    label = f"{a.scenario}_{name}_{datetime.now():%Y%m%d-%H%M%S}".replace(".", "_").replace("/", "_")
    run_dir = (ROOT / a.out / label).resolve()
    if run_dir.is_relative_to(ROOT) and any("." in part for part in run_dir.relative_to(ROOT).parts):
        raise SystemExit(f"--out {a.out}: a folder name inside the repository may not contain a dot, since the runner imports the "
                         "per-episode server launchers as modules of that path")
    meta = {
        "label": label, "scenario": a.scenario, "agent": a.agent, "servers": list(servers),
        "model": a.model if a.agent in MODEL_AGENTS else None, "model_id": backend.model_id if backend else None,
        "endpoint": a.base_url if a.agent in MODEL_AGENTS else None,
        "briefing": BRIEFING_VERSION, "tools_note": TOOLS_NOTE_VERSION if CAUSAL_SERVER in servers else None,
        "env_params": episode.encode_params(env_kwargs), "per_record": a.per_record, "max_records": a.max_records, "episodes": len(pairs),
        "facts_path": str(facts_path), "code": code_versions(), "python": platform.python_version(),
        "mcp": _version("mcp"), "openai": _version("openai"), "started": datetime.now().isoformat(timespec="seconds"),
    }
    rows = []
    for seed, combo in pairs:
        ep_dir = run_dir / f"seed{seed:06d}"
        if a.agent == "oracle":
            wall, outcome = run_oracle(ep_dir, seed, env_kwargs, cycles, blocks)
            error, extra = None, {}
        else:
            launchers, question, observation = prepare_episode(ep_dir, seed, env_kwargs, facts_path, a.scenario, servers)
            if not rows:
                check_servers(launchers)
            if make_policy:
                error, wall = run_policy(make_policy(seed), launchers[RIG_SERVER], observation)
                extra = {}
            else:
                backend.calls = []
                if scripted:
                    scripted.start_episode()
                result, error, wall = run_plan_execute(backend, launchers, question, f"{label}-seed{seed:06d}", ep_dir)
            outcome = record(ep_dir)
            if not make_policy:
                extra = record_plan_execute(ep_dir, result, backend.calls, outcome["tool_calls"])
        row = {"seed": seed, "scenario": a.scenario, "record": json.dumps(combo) if combo else "", "agent": name, "via": via,
               "wall_s": round(wall, 1), "error": error or "", **extra, **outcome}
        rows.append(row)
        (ep_dir / "report.md").write_text(episode_report(ep_dir, meta), encoding="utf-8")
        print(
            f"seed {seed}: ended {row['episode_ended']} ({row['end_reason']}), all correct {row['diagnosis_all_correct']}, "
            f"recovered {row['recovered']}, cost {row['cost']}, {row['wall_s']} s"
            + (f", plan [{row['plan_tools']}], causal calls {row['causal_calls']}, reaching no tool {row['calls_reaching_no_tool']}" if extra else "")
            + (f", error {row['error']}" if row["error"] else ""),
            flush=True,
        )
    meta["ended"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "run.json", meta)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(run_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "summary.md").write_text(run_summary(run_dir), encoding="utf-8")
    print(f"wrote {run_dir}")


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main(sys.argv[1:])
