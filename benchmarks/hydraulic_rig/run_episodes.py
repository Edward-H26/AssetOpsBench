"""Run an agent or a reference policy on hydraulic rig episodes through the two MCP servers and verify every episode from
its log (docs/hydraulic_rig/benchmarks.md).

usage, from the fork root with the fork's virtual environment first on PATH (the runner starts every tool server as python):
  python benchmarks/hydraulic_rig/run_episodes.py [--config static|timed|pilot|<json>] [--agent AGENT] [--seeds N] [--first-seed S]
      [--model NAME] [--base-url URL] [--clock-mode declared|generation_time|instant] [--guidance 1|2|3] [--no-diagnostics]
      [--prices placeholder|full_tier_high|sme|<json>] [--verifier-scope writes|all] [--judge none|local] [--judge-model NAME]
      [--judge-base-url URL] [--judge-protocol single|paper] [--ranking-model joint_block|joint|discriminant_urn]
      [--abstention-rule decision|fixed_cut] [--abstention-cut P] [--no-grade] [--out benchmarks/hydraulic_rig/runs]

Agents. plan_execute runs AssetOpsBench's Plan-Execute runner unchanged with the action server and, unless
--no-diagnostics, the diagnostics server registered; it reaches the model through AssetOpsBench's own OpenAI compatible
backend at the endpoint --base-url names (TokenRouter by default) with the key in API_KEY, the only key the drivers read.
scripted runs the same runner with a fixed plan that calls every tool of both servers, a wiring check without a model.
oracle, do_nothing and greedy are the reference policies of the environment, each action sent as a tool call through
the action server while a shadow environment in this process gives the oracle its hidden state; the served log is
compared with the shadow's at the end. The oracle against do_nothing is the ordering check of OpenEnv RFC 008. static
answers the question without the gym, in one model call over the question, the features of the recorded cycle the gym
would have served first (which the gym charges for) and the failure modes; its answer is judged by judge.py against the
oracle's reference answer and parsed for the components it calls faulty and repaired (ablation A3).

Every episode folder holds what the agent could see (the state file, the released evidence, the predictions, the
question, the trajectory, the model calls, the audit report) and under hidden/ what no agent facing tool reads (the
oracle, the verdict, the scores and the full log). The run folder gets run.json, scores.csv and summary.md, which counts
episodes and files and states no score. EpisodeRunner is the machinery behind main(); gepa_optimize.py drives it too.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (BOOKKEEPING_COLUMNS, CAUSAL_SERVER, COLUMNS, DEFAULT_RANKING_MODEL, HIDDEN, PREDICTIONS_FILE, RANKING_MODELS,  # noqa: E402
                    RELEASED_FILE, RIG_SERVER, ROOT,
                    SCORE_COLUMNS, SRC, STATE_FILE, apply_price_table, contract_tools, flat, load_config, load_price_table,
                    load_world, read_json, write_json, write_launchers, write_rows)
from envs.hydraulic_rig import verifier  # noqa: E402
from envs.hydraulic_rig.classifiers import TierClassifiers  # noqa: E402
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY  # noqa: E402
from envs.hydraulic_rig.env import HydraulicRigEnv  # noqa: E402
from envs.hydraulic_rig.greedy import GreedyPolicy, ServedClassifiers  # noqa: E402
from envs.hydraulic_rig.policies import DoNothingPolicy, OraclePolicy, run_episode  # noqa: E402
from pilot import anchors  # noqa: E402
from report import episode_report, run_summary  # noqa: E402
from servers.hydraulic_rig import episode  # noqa: E402
from servers.hydraulic_rig.briefing import BRIEFING_VERSION  # noqa: E402
from servers.hydraulic_rig.policy_client import play, tool_call  # noqa: E402

ROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
DEFAULT_MODEL = "MiniMax-M3"
MODEL_AGENTS = ("plan_execute", "static")
POLICY_AGENTS = ("oracle", "do_nothing", "greedy")
AGENTS = ("scripted", *MODEL_AGENTS, *POLICY_AGENTS)
CLOCK_MODES = ("declared", "generation_time", "instant")
ABSTENTION_RULES = ("decision", "fixed_cut")
STATIC_FACTS_GROUP = "tier3"  # the full sensor group, so the first served cycle is the one the static answer sees


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
    """Wiring check without a model: a fixed plan over both servers, fixed arguments and a fixed closing summary,
    returned in the order the runner asks for them (the plan, one argument reply per tool step, the summary). The
    diagnosis and the work order are fixed too, so the episode's verdict says nothing about diagnosis; the run checks
    that every tool of both servers answers through the runner, that the prediction stored by one call is read back by
    the next, that the work order closes after the declared lead time and downtime of the pump, and that a declaration
    after the diagnosis is refused, which is the answer the wiring check expects."""

    STEPS = [
        ("Read the asset profile", RIG_SERVER, "get_asset_profile", "{}"),
        ("Read the failure modes of the pump", RIG_SERVER, "get_failure_modes", '{"component": "pump"}'),
        ("Read one cycle of the cheapest sensor group", RIG_SERVER, "read_sensors", '{"sensor_group": "tier1", "cycles": 1}'),
        ("Read one cycle of the full sensor group", RIG_SERVER, "read_sensors", '{"sensor_group": "tier3", "cycles": 1}'),
        ("List the released evidence", CAUSAL_SERVER, "get_episode_evidence", '{"sensor_group": "all"}'),
        ("Rank the candidate fault combinations", CAUSAL_SERVER, "rank_root_causes", '{"top_k": 3}'),
        ("Summarize the changes since the start", RIG_SERVER, "summarize_changes_since", '{"cycle": 0}'),
        ("Submit a diagnosis", RIG_SERVER, "submit_diagnosis", '{"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130, "evidence_ids": ""}'),
        ("Predict the effect of replacing the pump", CAUSAL_SERVER, "estimate_intervention_effect", '{"action": "replace:pump"}'),
        ("Create a work order for the pump", RIG_SERVER, "create_work_order", '{"component": "pump", "prediction_id": "pred/1", "evidence_ids": ""}'),
        ("Advance time until the work order closes", RIG_SERVER, "advance_time", '{"cycles": 5}'),
        ("Read one cycle of the cheapest sensor group after the repair", RIG_SERVER, "read_sensors", '{"sensor_group": "tier1", "cycles": 1}'),
        ("Verify the prediction against the reading", CAUSAL_SERVER, "verify_outcome", '{"prediction_id": "pred/1"}'),
        ("Try to declare the evidence insufficient after the diagnosis", RIG_SERVER, "declare_insufficient_evidence", '{"sensor_group": "tier3", "cycles": 1, "evidence_ids": ""}'),
        ("End the episode", RIG_SERVER, "finish", '{"summary": "scripted run complete"}'),
    ]
    SUMMARY = "scripted run complete"

    def __init__(self, servers=(RIG_SERVER, CAUSAL_SERVER)):
        self._model_id = "scripted"
        self.steps = [s for s in self.STEPS if s[1] in servers]
        self._replies = iter(())

    def start_episode(self):
        plan = "\n\n".join(
            f"#Task{n}: {task}\n#Server{n}: {server}\n#Tool{n}: {tool}\n"
            f"#Dependency{n}: {'None' if n == 1 else f'#S{n - 1}'}\n#ExpectedOutput{n}: the tool's answer"
            for n, (task, server, tool, _) in enumerate(self.steps, 1)
        )
        self._replies = iter([plan, *(arguments for _, _, _, arguments in self.steps), self.SUMMARY])

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


def code_versions():
    """The fork commit and the SHA-256 prefixes of the code a run depends on, committed or not."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = sorted([*(SRC / "envs" / "hydraulic_rig").glob("*.py"), *(SRC / "servers" / "hydraulic_rig").glob("*.py"),
                    *(SRC / "servers" / "causal_diagnosis").glob("*.py"), *(SRC / "servers" / "causal_diagnosis").glob("models.json"),
                    *(SRC / "servers").glob("tool_contract_v2.json"), *HERE.glob("*.py"), *(HERE / "pilot").glob("*.py"),
                    *(HERE / "pilot").glob("*.json"), *(HERE / "configs").rglob("*.json")])
    return {"commit": commit, "sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in files}}


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


# ----- reference policies through the server -----

def make_policy(name, env, classifiers, inseparable=False):
    if name == "oracle":
        return OraclePolicy(env, inseparable=inseparable)
    if name == "do_nothing":
        return DoNothingPolicy()
    if name == "greedy":
        return GreedyPolicy(classifiers, use_max_faulty=True)
    raise ValueError(f"unknown policy {name}")


def play_policy(name, state_path, server_path, cycles, blocks, config, seed, classifiers=None, grade_lookup=None):
    """One episode of a reference policy through the action server, played by policy_client.play on the tool answers. A
    shadow environment in this process is reset with the same seed and configuration as the server's; the oracle reads
    the hidden state from it, and after every call the shadow steps the same action with the think cycles the server
    recorded, so the two environments stay identical. grade_lookup(env) gives the grade of the start, so the oracle
    declares instead of diagnosing on an inseparable start. Returns whether the served log equals the shadow's."""
    shadow = HydraulicRigEnv(cycles, blocks, config)
    shadow.reset(seed)
    grade = grade_lookup(shadow) if grade_lookup else None
    policy = make_policy(name, shadow, classifiers, inseparable=grade is not None and not grade["separable"])

    def send(action):
        tool, arguments = tool_call(action)
        text = asyncio.run(_call_tool(server_path, tool, arguments))
        state = episode.load_state(state_path)
        if not shadow.done:
            shadow.step(action, think_cycles=state["think_cycles"][-1] if state["think_cycles"] else 0)
        return text

    play(policy, episode.opening(episode.load_state(state_path)), send)
    served = episode.replay(episode.load_state(state_path))
    return served.log == shadow.log


def oracle_reference(cycles, blocks, config, seed):
    """The oracle policy's episode on the same seed and configuration: the verifier's own cost of it, which normalizes
    the agent's cost, and its read counts, which the verifier scope "all" compares with the agent's."""
    env = HydraulicRigEnv(cycles, blocks, config)
    log = run_episode(env, OraclePolicy(env), seed)
    verdict = verifier.verify(log, verifier.released_from_env(env), env.oracle(), None)
    return verdict["cost"], verifier.read_counts(log)


def grade_of_start(oracle, released, ranking_model):
    """The diagnosability grade E18 of the start under the released rules, from the diagnostics server's grade table, in
    the verifier's tri-state: {"separable": True, "cost": the grade's cost} when the table grades the start, {"separable":
    False, "cost": None} when the table marks it inseparable under the budget, and None when the table does not hold the
    start (an anchored pilot day outside the evaluation set), since absence is not inseparability. Returns the grade and
    its source."""
    from servers.causal_diagnosis.adapter import load_models
    from servers.causal_diagnosis.grade import combo_key, ensure_grades

    table = ensure_grades(load_models(), released["rules"], None, ranking_model)["grades"]
    key = combo_key([oracle["diagnosis"][c] for c in COMPONENTS])
    if key not in table:
        return None, "not in table"
    entry = table[key]
    if entry is None:
        return {"separable": False, "cost": None}, "table"
    return {"separable": True, "cost": entry.get("cost") if isinstance(entry, dict) else None}, "table"


# ----- the static agent (ablation A3) -----

FAULT_WORDS = re.compile(r"\b(fault\w*|degrad\w*|fail\w*|leak\w*|worn|abnormal\w*|critical\w*|severe\w*|reduced|low|lag\w*|"
                         r"below|off|defect\w*|malfunction\w*|insufficient|poor)\b", re.IGNORECASE)
HEALTHY_WORDS = re.compile(r"\b(healthy|normal\w*|nominal|ok|fine|intact|within (?:its |the )?(?:range|band)|no (?:fault|issue|problem))\b", re.IGNORECASE)
REPAIR_WORDS = re.compile(r"\b(replac\w*|repair\w*|work order\w*|servic\w*|overhaul\w*|fix\w*|swap\w*)\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(not|no|never|without|cannot|unable|isn't|aren't|wasn't|doesn't|don't)\b", re.IGNORECASE)
SENTENCES = re.compile(r"[.!?;\n]+")  # a semicolon ends a clause too, since answers list one component per clause


def parse_static_answer(answer, components=COMPONENTS):
    """A conservative reading of a free text answer: a component is called faulty when a clause names it with a fault
    word and no healthy word, and called repaired when a clause names it with a repair word; a clause with a negation
    counts for neither. The clauses behind every call are kept, so a reader can check the rule on the report."""
    faulty, repaired, evidence = [], [], {c: [] for c in components}
    for sentence in (s.strip() for s in SENTENCES.split(answer or "") if s.strip()):
        named = [c for c in components if re.search(rf"\b{c}\b", sentence, re.IGNORECASE)]
        if not named or NEGATION.search(sentence):
            continue
        is_faulty = bool(FAULT_WORDS.search(sentence)) and not HEALTHY_WORDS.search(sentence)
        is_repaired = bool(REPAIR_WORDS.search(sentence))
        for c in named:
            if is_faulty or is_repaired:
                evidence[c].append(sentence)
            if is_faulty and c not in faulty:
                faulty.append(c)
            if is_repaired and c not in repaired:
                repaired.append(c)
    return {"faulty": faulty, "repaired": repaired, "sentences": evidence}


def static_context(cycles, blocks, config, seed):
    """What the static answer sees besides the question: the features of the recorded cycle the gym would have served
    first (the facts of that cycle as the IndustryAssetEQA file supplies them, never its labels) and the failure modes
    the gym's get_failure_modes returns. Both are read from a throwaway environment, so the episode's own state file
    records no action."""
    env = HydraulicRigEnv(cycles, blocks, config)
    env.reset(seed)
    env.step({"type": "read_sensors", "sensor_group": STATIC_FACTS_GROUP, "cycles": 1})
    served = next(e for e in reversed(env.log) if e["event"] == "read")["served"][0]
    cycle = next(c for c in cycles if c.index == served["cycle"])
    envelope, _cost, _done, _info = env.step({"type": "get_failure_modes", "component": ""})
    return {"cycle_index": served["cycle"], "features": dict(sorted(cycle.features.items())), "failure_modes": envelope["result"]["failure_modes"]}


def static_prompt(question, context):
    facts = "\n".join(f"- {name}: {value}" for name, value in context["features"].items())
    modes = "\n".join(f"- {m['component']} at level {m['level']}: {m['code']} ({m['severity']}), recommended action {m['action']}, "
                      f"indicated by {', '.join(m['channels'])}" for m in context["failure_modes"])
    return (f"{question}\n\nYou cannot call any tool. Answer from the facts below.\n\n"
            f"Facts of the current operating cycle of the asset, as the IndustryAssetEQA facts file supplies them (feature: value):\n{facts}\n\n"
            f"Failure modes of the asset:\n{modes}\n\n"
            "Answer in prose: give the condition level of each component (cooler, valve, pump, accumulator), name the components "
            "you judge faulty, and name the work orders (component replacements) you would create.")


# ----- one run -----

class EpisodeRunner:
    """The settings of one run and the machinery that plays, verifies, judges and records one episode. main() builds it
    from the command line; gepa_optimize.py builds it for the plan_execute agent and plays episodes under a candidate
    planner prompt."""

    def __init__(self, config, config_name, agent, servers, cycles, blocks, facts_path, label, options, backend=None, scripted=None,
                 classifiers=None, model=None, ranking_model=DEFAULT_RANKING_MODEL, abstention=None, prices_name="placeholder"):
        self.config, self.config_name, self.agent, self.servers = config, config_name, agent, servers
        self.cycles, self.blocks, self.facts_path, self.label = cycles, blocks, facts_path, label
        self.options, self.backend, self.scripted, self.classifiers = options, backend, scripted, classifiers
        self.model, self.ranking_model, self.abstention, self.prices_name = model, ranking_model, abstention, prices_name
        self.agent_label = agent if agent in POLICY_AGENTS or agent == "scripted" else f"{agent}_{model}"
        self.checked_servers = False

    def start(self, ep_dir, seed, episode_config):
        """The episode folder with its state file, released evidence, launchers and question; returns the launchers."""
        ep_dir.mkdir(parents=True, exist_ok=True)
        episode.start(ep_dir / STATE_FILE, seed, episode_config, self.facts_path)
        launchers = write_launchers(ep_dir, self.servers, ranking_model=self.ranking_model, abstention=self.abstention)
        question = read_json(ep_dir / RELEASED_FILE)["task_text"]  # the briefing the server composed at the start
        (ep_dir / "question.txt").write_text(question, encoding="utf-8")
        if not self.checked_servers and self.agent != "static":
            check_servers(launchers)
            self.checked_servers = True
        return launchers, question

    def play(self, ep_dir, name, seed, scenario, episode_config, meta):
        """One episode: play it with the run's agent, verify it (or judge and parse the static answer), write the
        hidden files and the report, and return the row of scores.csv."""
        launchers, question = self.start(ep_dir, seed, episode_config)
        started = time.perf_counter()
        extra, error = {}, None
        if self.agent == "static":
            extra, error = self.play_static(ep_dir, question, episode_config, seed)
        elif self.agent in POLICY_AGENTS:
            try:
                extra["replay_matches"] = play_policy(self.agent, ep_dir / STATE_FILE, launchers[RIG_SERVER], self.cycles, self.blocks,
                                                      episode_config, seed, self.classifiers, self.grade_lookup(ep_dir))
            except Exception as exc:  # noqa: BLE001 - a failed episode is a result of the benchmark
                error = f"{type(exc).__name__}: {exc}"
        else:
            self.backend.calls = []
            if self.scripted:
                self.scripted.start_episode()
            result, error, _ = run_plan_execute(self.backend, launchers, question, f"{self.label}-{name}", ep_dir)
            extra = record_plan_execute(ep_dir, result, self.backend.calls)
        wall = time.perf_counter() - started
        if self.agent == "static":
            scores, combination = extra.pop("scores"), extra.pop("combination")
        else:
            try:
                scores, combination = verify_episode(ep_dir, episode_config, self.cycles, self.blocks, seed, self.options)
            except Exception as exc:  # noqa: BLE001 - a verifier that cannot run leaves the scores empty and the error in the row
                scores, combination = {k: None for k in SCORE_COLUMNS}, None
                error = (error + "; " if error else "") + f"verifier: {type(exc).__name__}: {exc}"
        row = {k: None for k in COLUMNS}
        row.update({
            "run": self.label, "seed": seed, "scenario_id": scenario["id"] if scenario else "", "date": scenario["date"] if scenario else "",
            "config": self.config_name, "agent": self.agent_label, "model": self.model if self.agent in MODEL_AGENTS else "",
            "clock_mode": self.config.clock_mode, "guidance_level": self.config.guidance_level, "diagnostics_server": self.config.diagnostics_server,
            "prices": self.prices_name, "verifier_scope": self.config.verifier_scope, "ranking_model": self.ranking_model,
            "block": combination, "n_faulty": None if combination is None else sum(1 for c, v in zip(COMPONENTS, combination) if v != HEALTHY[c]),
            "wall_s": round(wall, 1), "error": error or "",
        })
        row.update(scores)
        row.update(tool_counts(episode.load_state(ep_dir / STATE_FILE)))
        if scores.get("steps") is not None:
            row["refusals"] = read_json(ep_dir / HIDDEN / "verdict.json", {}).get("refusals", row["refusals"])
        row.update({k: v for k, v in extra.items() if k in BOOKKEEPING_COLUMNS})
        (ep_dir / "report.md").write_text(episode_report(ep_dir, meta), encoding="utf-8")
        return {k: flat(v) for k, v in row.items()}

    def grade_lookup(self, ep_dir):
        """The grade lookup the oracle policy uses before it plays, or None when grades are off or the agent is not the
        oracle; a lookup that fails leaves the grade unknown, as verify_episode does."""
        if self.agent != "oracle" or not self.options["grade"]:
            return None
        released = read_json(ep_dir / RELEASED_FILE, {})

        def lookup(env):
            try:
                return grade_of_start(env.oracle(), released, self.options["ranking_model"])[0]
            except Exception:  # noqa: BLE001 - an unknown grade counts as separable, as in the verifier
                return None
        return lookup

    def play_static(self, ep_dir, question, episode_config, seed):
        """The static answer of ablation A3 and its two scorings: the judge against the oracle's reference answer, and
        the parsed faulty and repaired sets against the recorded start and the oracle's work orders."""
        env = episode.replay(episode.load_state(ep_dir / STATE_FILE))
        oracle = env.oracle()
        context = static_context(self.cycles, self.blocks, episode_config, seed)
        prompt = static_prompt(question, context)
        (ep_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        self.backend.calls = []
        error, answer = None, ""
        try:
            answer = self.backend.generate(prompt) or ""
        except Exception as exc:  # noqa: BLE001 - a failed call is a result of the benchmark
            error = f"{type(exc).__name__}: {exc}"
        (ep_dir / "answer.txt").write_text(answer, encoding="utf-8")
        write_json(ep_dir / "llm_calls.json", self.backend.calls)
        parsed = parse_static_answer(answer)
        truth_faulty = [c for c in COMPONENTS if oracle["diagnosis"][c] != HEALTHY[c]]
        hidden = ep_dir / HIDDEN
        hidden.mkdir(exist_ok=True)
        write_json(hidden / "oracle.json", oracle)
        write_json(hidden / "log.json", env.log)
        judge = judge_static(ep_dir, question, oracle, answer, self.options["judge"]) if self.options["judge"] and answer else None
        static = {"mode": "static", "cycle_index": context["cycle_index"], "features_shown": len(context["features"]),
                  "failure_modes_shown": len(context["failure_modes"]), "answer": answer, "parsed": parsed, "truth_faulty": truth_faulty,
                  "oracle_work_orders": [o["component"] for o in oracle["work_orders"]],
                  "faulty_set_correct": set(parsed["faulty"]) == set(truth_faulty),
                  "per_component_correct": {c: (c in parsed["faulty"]) == (c in truth_faulty) for c in COMPONENTS},
                  "repairs_match_oracle": set(parsed["repaired"]) == {o["component"] for o in oracle["work_orders"]}}
        write_json(hidden / "static.json", static)
        scores = {k: None for k in SCORE_COLUMNS}
        scores.update({"end_reason": "static", "steps": 0, "static_faulty_set_correct": static["faulty_set_correct"],
                       "static_repairs_match_oracle": static["repairs_match_oracle"], "judge_pass": None if judge is None else judge.get("pass")})
        write_json(hidden / "score.json", {**scores, "judge": judge})
        extra = {"llm_calls": len(self.backend.calls), "input_tokens": sum(c.get("input_tokens", 0) for c in self.backend.calls),
                 "output_tokens": sum(c.get("output_tokens", 0) for c in self.backend.calls), "scores": scores,
                 "combination": [oracle["diagnosis"][c] for c in COMPONENTS]}
        return extra, error


def check_servers(launchers):
    """List the tools of every registered server once before the first episode: a server that fails to start would
    otherwise only show up as unavailable inside the planner prompt."""
    for server, path in launchers.items():
        names = {tool["name"] for tool in asyncio.run(_list_tools(path))}
        if names != set(contract_tools(server)):
            raise SystemExit(f"server {server} offers {sorted(names)}, expected {sorted(contract_tools(server))}")


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


def record_plan_execute(ep_dir, result, calls):
    """What a failure analysis of a Plan-Execute run needs besides the log: every model call, the plan, the closing
    answer and the trajectory of both servers."""
    write_json(ep_dir / "llm_calls.json", calls)
    if result is not None:
        (ep_dir / "plan.txt").write_text(result.plan.raw, encoding="utf-8")
        (ep_dir / "answer.txt").write_text(result.answer or "", encoding="utf-8")
        write_json(ep_dir / "trajectory.json", [
            {"step": r.step_number, "task": r.task, "server": r.server, "tool": r.tool, "tool_args": r.tool_args,
             "response": r.response, "error": r.error, "duration_ms": r.duration_ms}
            for r in result.trajectory
        ])
    trajectory = result.trajectory if result is not None else []
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
        "plan_steps": len(result.plan.steps) if result is not None else "",
        "causal_calls": sum(1 for r in trajectory if r.server == CAUSAL_SERVER and r.success and r.tool),
    }


def _check_ok(checks, name):
    value = (checks or {}).get(name)
    if isinstance(value, dict):
        return value.get("ok", value.get("passed"))
    return value


def _delay(value):
    """The largest delay over components, from a per component dict or a number; None when nothing was measured."""
    if isinstance(value, dict):
        measured = [v for v in value.values() if isinstance(v, (int, float))]
        return max(measured) if measured else None
    return value


def verdict_row(verdict, reward, cost_oracle, judge):
    """The verifier's verdict as the score columns of scores.csv; a key the verdict lacks leaves its cell empty."""
    row = {k: verdict.get(k) for k in SCORE_COLUMNS}
    row["detection_delay"] = _delay(verdict.get("detection_delay"))
    row["repair_delay"] = _delay(verdict.get("repair_delay"))
    row["provenance_ok"] = _check_ok(verdict.get("checks"), "provenance")
    row["reward"] = reward
    row["cost_oracle"] = cost_oracle
    row["judge_pass"] = None if judge is None else judge.get("pass")
    return row


def verify_episode(ep_dir, config, cycles, blocks, seed, options):
    """Replay the served episode, write the hidden files and return the score columns and the start combination. A log
    the verifier cannot read raises, so the caller records the error and leaves the scores empty. options holds the
    judge settings (or None), whether the grade is looked up, and the ranking model of the grade table."""
    env = episode.replay(episode.load_state(ep_dir / STATE_FILE))
    oracle = env.oracle()
    released = read_json(ep_dir / RELEASED_FILE, {})
    cost_oracle, oracle_reads = oracle_reference(cycles, blocks, config, seed)
    if config.verifier_scope == "all":
        oracle["reads"] = oracle_reads
    grade, grade_source = None, "disabled"
    if options["grade"]:
        try:
            grade, grade_source = grade_of_start(oracle, released, options["ranking_model"])
        except Exception as exc:  # noqa: BLE001 - a grade that cannot be looked up is recorded, not raised
            grade_source = f"error: {type(exc).__name__}: {exc}"
    verdict = verifier.verify(env.log, released, oracle, cost_oracle, grade=grade, scope=config.verifier_scope,
                              predictions=read_json(ep_dir / PREDICTIONS_FILE))
    hidden = ep_dir / HIDDEN
    hidden.mkdir(exist_ok=True)
    write_json(hidden / "log.json", env.log)
    write_json(hidden / "oracle.json", oracle)
    write_json(hidden / "verdict.json", verdict)
    if not verdict.get("verified"):
        raise RuntimeError(f"the verifier could not read the episode: {verdict.get('error')}")
    reward = verifier.reward(verdict, config.horizon)
    judge = run_judge(ep_dir, options["judge"]) if options["judge"] else None
    row = verdict_row(verdict, reward, cost_oracle, judge)
    write_json(hidden / "score.json", {**row, "scope": config.verifier_scope, "grade_source": grade_source, "judge": judge})
    return row, [oracle["diagnosis"][c] for c in COMPONENTS]


def run_judge(ep_dir, options):
    """The AssetOpsBench judge on the closing summary through judge.py (the verifier agent's adapter), which writes
    hidden/judge.json itself; a judge that cannot run leaves the pass empty rather than failing the episode."""
    import judge

    try:
        return judge.judge_episode(ep_dir, options["backend"], options["protocol"], options["agent_model"], options["base_url"])
    except Exception as exc:  # noqa: BLE001 - a judge failure is recorded, not raised
        return {"pass": None, "error": f"{type(exc).__name__}: {exc}"}


def judge_static(ep_dir, question, oracle, answer, options):
    """The judge on a static answer, through the same pieces of judge.py as an episode (its reference answer and its one
    call), since a static run has no finish event for judge_episode to read; the record lands in hidden/judge.json."""
    import judge

    settings = judge.PROTOCOLS[options["protocol"]]
    reference = judge.reference_answer(oracle)
    try:
        calls = [judge.judge_once(ep_dir.name, question, reference, answer, [], options["backend"], settings["temperature"], options["agent_model"])
                 for _ in range(settings["calls"])]
    except Exception as exc:  # noqa: BLE001 - a judge failure is recorded, not raised
        return {"pass": None, "error": f"{type(exc).__name__}: {exc}"}
    reviewed = [c for c in calls if c["error"] is None]
    pass_rate = sum(1 for c in reviewed if c["passed"]) / len(reviewed) if reviewed else None
    record = {"format": judge.JUDGE_FORMAT, "protocol": options["protocol"], "temperature": settings["temperature"], "calls_planned": settings["calls"],
              "judge_model": options["backend"].model_id, "base_url": options["base_url"], "agent_model": options["agent_model"], "mode": "static",
              "task_text": question, "reference_answer": reference, "answer": answer, "calls": calls, "pass_rate": pass_rate,
              "score_mean": sum(c["score"] for c in reviewed) / len(reviewed) if reviewed else None,
              "pass": None if pass_rate is None else pass_rate > 0.5}
    write_json(ep_dir / HIDDEN / judge.JUDGE_NAME, record)
    return record


def tool_counts(state):
    calls = state.get("tool_calls", [])
    refused = 0
    for call in calls:
        try:
            refused += "refused" in json.loads(call["response"])
        except (TypeError, ValueError, KeyError):
            pass
    return {"tool_calls": len(calls), "refusals": refused}


# ----- the plan of a run -----

def episode_plan(config, seeds, cycles, blocks):
    """[(folder name, seed, scenario or None, config of the episode)]. The pilot plays every scenario of
    pilot/scenarios_17.json under every seed, anchored at the scenario's date with the scenario text as the question;
    every other config plays the seeds as they are."""
    if config.anchors is None:
        return [(f"seed{seed:06d}", seed, None, config) for seed in seeds]
    date0 = config.anchors.get("date0", anchors.DATE0)
    plan = []
    for scenario in anchors.load_scenarios()["scenarios"]:
        anchored = anchors.anchor(scenario["date"], cycles, blocks, date0)
        episode_config = dataclasses.replace(config, start=anchors.start_of(anchored), question=scenario["text"])
        for seed in seeds:
            plan.append((f"scenario{scenario['id']}_seed{seed:06d}", seed, {**scenario, "anchor": anchored}, episode_config))
    return plan


def pilot_note(config, plan):
    """The two assumptions of the pilot and the anchor of every scenario, for run.json and the reports."""
    if config.anchors is None:
        return None
    return {"assumptions": anchors.ASSUMPTIONS, "date0": config.anchors.get("date0", anchors.DATE0),
            "anchors": {name: {k: scenario["anchor"][k] for k in ("date", "cycle_index", "stable", "anchored_cycle_index", "block_offset")}
                        for name, _, scenario, _ in plan}}


def run_label(config_name, agent_label):
    # a run folder inside the repository makes the runner start each server as `python -m <dotted path>`, which splits a
    # dot in any folder name, so the dots and slashes of a router model id such as z-ai/glm-5.3 become underscores here
    return f"{config_name}_{agent_label}_{datetime.now():%Y%m%d-%H%M%S}".replace(".", "_").replace("/", "_")


def run_folder(out, label):
    run_dir = (ROOT / out / label).resolve()
    if run_dir.is_relative_to(ROOT) and any("." in part for part in run_dir.relative_to(ROOT).parts):
        raise SystemExit(f"--out {out}: a folder name inside the repository may not contain a dot, since the runner imports the "
                         "per-episode server launchers as modules of that path")
    return run_dir


def write_run(run_dir, meta, rows):
    meta["ended"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "run.json", meta)
    write_rows(run_dir / "scores.csv", rows, COLUMNS)
    (run_dir / "summary.md").write_text(run_summary(run_dir), encoding="utf-8")


def build_parser():
    ap = argparse.ArgumentParser(description="agents and reference policies on hydraulic rig episodes through the MCP servers")
    ap.add_argument("--config", default="static", help="a preset of benchmarks/hydraulic_rig/configs/ or a JSON path")
    ap.add_argument("--agent", choices=AGENTS, default="scripted")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="the model name the model agents send to the endpoint")
    ap.add_argument("--base-url", default=ROUTER_BASE_URL, help="OpenAI compatible endpoint of the model agents (default: TokenRouter)")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (per scenario for the pilot)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--clock-mode", choices=CLOCK_MODES, default=None, help="override the preset's clock mode")
    ap.add_argument("--guidance", type=int, choices=(1, 2, 3), default=None, help="override the preset's guidance level")
    ap.add_argument("--no-diagnostics", action="store_true", help="register the action server alone")
    ap.add_argument("--prices", default=None, help="a price table of configs/prices/ or a JSON path (default: the preset's prices)")
    ap.add_argument("--verifier-scope", choices=("writes", "all"), default=None)
    ap.add_argument("--judge", choices=("none", "local"), default="none", help="local: the AssetOpsBench judge through judge.py")
    ap.add_argument("--judge-protocol", choices=("single", "paper"), default="single", help="one call at temperature 0, or the paper's five at 0.3")
    ap.add_argument("--judge-model", default=None, help="the judge's model name (default: --model); the fork's evaluator refuses a judge equal to the agent model")
    ap.add_argument("--judge-base-url", default=None, help="the judge's OpenAI compatible endpoint (default: --base-url)")
    ap.add_argument("--ranking-model", choices=RANKING_MODELS, default=DEFAULT_RANKING_MODEL, help="the ranking model of the diagnostics server (ablation A12); the default is the server's own")
    ap.add_argument("--abstention-rule", choices=ABSTENTION_RULES, default="decision", help="the abstention rule of the diagnostics server (ablation A8)")
    ap.add_argument("--abstention-cut", type=float, default=0.95, help="the posterior cut of the fixed_cut rule")
    ap.add_argument("--no-grade", action="store_true", help="skip the diagnosability grade lookup, so every declaration counts as correct")
    ap.add_argument("--facts", default=None, help="the rig data file (default: HYD_FACTS_PATH or the package's data folder)")
    ap.add_argument("--out", default="benchmarks/hydraulic_rig/runs")
    return ap


def configure(a):
    """The validated configuration of a run from the command line: the preset, the overrides, the price table."""
    config_name, config = load_config(a.config)
    overrides = {k: v for k, v in (("clock_mode", a.clock_mode), ("guidance_level", a.guidance), ("verifier_scope", a.verifier_scope)) if v is not None}
    if a.no_diagnostics:
        overrides["diagnostics_server"] = False
    if overrides:
        config = type(config).from_json(dataclasses.replace(config, **overrides).to_json())
    price_table = load_price_table(a.prices) if a.prices else None
    if price_table:
        config = apply_price_table(config, price_table)
    return config_name, config, price_table


def judge_settings(a, agent_label):
    """The judge options of a run, or None; a judge equal to the agent model is refused here, since the fork's evaluator
    refuses self judging on every episode."""
    if a.judge != "local":
        return None
    import judge

    judge_model, judge_base_url = a.judge_model or a.model, a.judge_base_url or a.base_url
    if a.agent in MODEL_AGENTS and judge_model == a.model:
        raise SystemExit("--judge local needs a judge model other than the agent model, since the fork's evaluator refuses self "
                         "judging; pass --judge-model (and --judge-base-url for a second endpoint)")
    return {"backend": judge.router_backend(judge_model, judge_base_url), "protocol": a.judge_protocol, "base_url": judge_base_url,
            "model": judge_model, "agent_model": a.model if a.agent in MODEL_AGENTS else agent_label}


def run_meta(a, runner, config, price_table, plan, judge_options):
    return {
        "label": runner.label, "config_name": runner.config_name, "config": config.to_json(), "mode": "static" if a.agent == "static" else "gym",
        "prices": runner.prices_name, "price_table_source": price_table["source"] if price_table else None,
        "agent": a.agent, "agent_label": runner.agent_label, "servers": list(runner.servers),
        "model": a.model if a.agent in MODEL_AGENTS else None, "model_id": runner.backend.model_id if runner.backend else None,
        "endpoint": a.base_url if a.agent in MODEL_AGENTS else None, "judge": a.judge,
        "judge_protocol": judge_options["protocol"] if judge_options else None, "judge_model": judge_options["model"] if judge_options else None,
        "judge_endpoint": judge_options["base_url"] if judge_options else None, "grade_lookup": not a.no_grade,
        "ranking_model": a.ranking_model, "abstention_rule": a.abstention_rule, "abstention_cut": a.abstention_cut if a.abstention_rule == "fixed_cut" else None,
        "clock_mode": config.clock_mode, "guidance_level": config.guidance_level, "diagnostics_server": config.diagnostics_server,
        "verifier_scope": config.verifier_scope, "briefing": BRIEFING_VERSION, "seeds": list(a.seeds_list), "episodes_planned": len(plan),
        "pilot": pilot_note(config, plan), "facts_path": str(runner.facts_path),
        "facts_sha256": hashlib.sha256(Path(runner.facts_path).read_bytes()).hexdigest()[:12],
        "code": code_versions(), "python": platform.python_version(), "mcp": _version("mcp"), "openai": _version("openai"),
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def build_runner(a, config_name, config, price_table, cycles, blocks, facts_path):
    """An EpisodeRunner for the command line settings; it fits the diagnostics models when a server or the grade needs them."""
    servers = (RIG_SERVER,) if not config.diagnostics_server else (RIG_SERVER, CAUSAL_SERVER)
    if CAUSAL_SERVER in servers or not a.no_grade:
        from servers.causal_diagnosis.fit import ensure_models
        ensure_models()  # fitted once here, before any server process starts and reads it; the grade table needs it too
    backend = scripted = classifiers = None
    if a.agent == "greedy":
        classifiers = ServedClassifiers(TierClassifiers(cycles, blocks), cycles)
    elif a.agent not in POLICY_AGENTS:
        scripted = ScriptedBackend(servers) if a.agent == "scripted" else None
        backend = RecordingBackend(scripted or router_backend(a.model, a.base_url))
    agent_label = a.agent if a.agent in POLICY_AGENTS or a.agent == "scripted" else f"{a.agent}_{a.model}"
    judge_options = judge_settings(a, agent_label)
    options = {"judge": judge_options, "grade": not a.no_grade, "ranking_model": a.ranking_model}
    abstention = (a.abstention_rule, a.abstention_cut) if a.abstention_rule != "decision" else None
    runner = EpisodeRunner(config, config_name, a.agent, servers, cycles, blocks, facts_path, run_label(config_name, agent_label), options,
                           backend=backend, scripted=scripted, classifiers=classifiers, model=a.model if a.agent in MODEL_AGENTS else None,
                           ranking_model=a.ranking_model, abstention=abstention, prices_name=price_table["name"] if price_table else "placeholder")
    return runner, judge_options


def main(argv):
    a = build_parser().parse_args(argv)
    facts_path, cycles, blocks = load_world(a.facts)
    config_name, config, price_table = configure(a)
    a.seeds_list = range(a.first_seed, a.first_seed + a.seeds)
    plan = episode_plan(config, a.seeds_list, cycles, blocks)
    runner, judge_options = build_runner(a, config_name, config, price_table, cycles, blocks, facts_path)
    run_dir = run_folder(a.out, runner.label)
    meta = run_meta(a, runner, config, price_table, plan, judge_options)
    rows = []
    for name, seed, scenario, episode_config in plan:
        row = runner.play(run_dir / name, name, seed, scenario, episode_config, meta)
        rows.append(row)
        print(f"{name}: ended {row['end_reason']!r} after {row['tool_calls']} tool calls, verified {row['pass'] != ''}, {row['wall_s']} s"
              + (f", error {row['error']}" if row["error"] else ""), flush=True)
    write_run(run_dir, meta, rows)
    print(f"wrote {run_dir}")
    return run_dir


if __name__ == "__main__":
    main(sys.argv[1:])
