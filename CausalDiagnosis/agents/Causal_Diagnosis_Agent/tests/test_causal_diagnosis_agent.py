"""The CausalDiagnosis agent through the driver: a scripted planner in the style of ScriptedBackend answers each turn from
the heading of the prompt and from the board it shows, so the same script holds on every seed, and the real servers answer
the agent's calls, in process through FastMCP for most tests (one process start per call would cost minutes) and through
the launchers over stdio for the driver tests, where a local endpoint hands every prompt to the scripted planner. The in
process diagnostics server adds the four ranking fields of version 7, derived from the real ranking, so that the board is
certain to show a read batch as advice. The model plans every read, the read that verifies a repair
included, and the same agent plays without the diagnostics server on its own candidates and the normal ranges. Every end
to end test asserts on the files written and on the row extras, never on a recorded value; the parser, the plan check, the
rule plan and the compaction are tested on a synthetic board without servers. Run folders go under pytest's tmp_path,
outside the repository."""
from CausalDiagnosis.utils.paths import ROOT
import contextlib
import dataclasses
import importlib.util
import threading
import time
import json
import math
import re
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, METERED_SERVERS, RIG_SERVER, contract_tools
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, HIDDEN, HIDDEN_FILES
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.episode_runners.episode_launchers import write_launchers
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent import agent_options, controller, episode_result  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import CausalLoopOptions  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board, band_verdict, channels_outside, n_cycles  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import (ESTIMATE, OPENING_KINDS, Plan, PlanError, Step, act_options, advice, batch_reads, check_plan,  # noqa: E402
                                    order_reserve, parse_plan, repair_options, rule_plan)
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import read_affordable  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import order_reserve_calls  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import ORIENTATION_HEADING, PLANNING_HEADING, RETRY_HEADING, SUMMARY_HEADING, orientation_turn, planning_turn  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import REPAIR_RULE, SERVER_ADVICE_RULE, system_prompt  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.causal_diagnosis_agent_surface import AobSurface  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import BRIEFING_VERSION as AOB_BRIEFING_VERSION  # noqa: E402
from CausalDiagnosis.agents.utils.mcp_sessions import SESSIONS, ToolTransportError  # noqa: E402
from llm import LLMBackend  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_episodes import EpisodeRunner, RecordingBackend  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server as build_causal_server  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import build_server as build_rig_server  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.tests.pinned_clock import pinned, pinned_preset  # noqa: E402


def separating_test_affordable(board, test):
    """Whether the separating test fits beside the completion of the reserve plan, as a read does (plan._fits_beside_reserve,
    the one rule of the tail)."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import _fits_beside_reserve
    if not test:
        return False
    price = test.get("price", board.read_price(test["sensor_group"], test.get("cycles", 1)))
    return _fits_beside_reserve(board, price, test.get("duration", board.read_duration(test.get("cycles", 1))))


ORIENTATION_REPLY = json.dumps({"differential": [
    {"component": "pump", "expected_symptoms": "lower flow and efficiency", "failure_mode_ids": ["fmea/industryasseteqa_hydraulic_facts/pump_internal_leakage_weak"]},
    {"component": "valve", "expected_symptoms": "pressure lag", "failure_mode_ids": ["fmea/industryasseteqa_hydraulic_facts/valve_switching_severe_lag"]}]})
SUMMARY_REPLY = "Scripted causal loop complete."
LONGEST_REPAIR = 5  # the longest default lead time plus repair downtime (the pump), the close of a visit of one order; the finish step waits for the later close of a visit of two
RANK_ONE = re.compile(r"^\| 1 \| ([^|]+) \| [^|]+ \| [^|]+ \|$", re.MULTILINE)  # rank, combination, codes, mass
EVIDENCE_ID = re.compile(r"^\| (obs/\d+/\S+) \|", re.MULTILINE)
ORDERED = re.compile(r"^\| wo-\d+ \| (\w+) \|", re.MULTILINE)
BRANCH_PREDICTION = re.compile(r"(\w+): (pred/b\d+)")
PLAN_PREDICTION = re.compile(r"every order of the plan cites (pred/\S+?)\b")
ADVISED = re.compile(r"Advised reads[^:]*: (\w+) x(\d+)")  # the head of the reads the board advises
STILL_TO_READ = re.compile(r"Still to read before a reading on this row, as steps of their own: ([a-z_, ]+)\.")
MODEL_SOURCES = ("model", "model_retry", "model_repaired")  # the sources of a plan the model wrote
VERIFYING_READ = {"kind": "read", "args": {"sensor_group": "tier3", "cycles": 1}, "expect": {}, "evidence": []}  # after the last close


def _section(prompt, heading):
    start = prompt.find(heading)
    if start < 0:
        return ""
    end = prompt.find("\n## ", start + len(heading))
    return prompt[start:end if end > 0 else None]


def top_faulty(prompt):
    """The components of the top candidate that are faulty now, as the board prints its current combination."""
    match = RANK_ONE.search(_section(prompt, "## Ranking"))
    assert match, "the board shows no ranking"
    levels = dict((part.rsplit(" ", 1)[0], int(part.rsplit(" ", 1)[1])) for part in match.group(1).strip().split(", "))
    return [c for c in COMPONENTS if levels[c] != HEALTHY[c]]


def branch_predictions(prompt):
    """The prediction identifier the orders of the top branch's repair plan cite, by component: the plan's one identifier when
    the board shows it, else the steps' own."""
    section = _section(prompt, "## Troubleshooting plans")
    first = next((line for line in section.splitlines() if line.startswith("| 1 |")), "")
    plan = PLAN_PREDICTION.search(first)
    if plan:
        return {component: plan.group(1) for component in top_faulty(prompt)}
    return dict(BRANCH_PREDICTION.findall(first))


def plan_json(steps, mode="act_now", branch=1):
    return json.dumps({"branch": branch, "mode": mode, "why": "scripted: the failure modes of the top candidate read as the cause", "steps": steps})


def opening_reply(prompt):
    """The plan of the opening that the model writes, or None once it is read: the profile, the failure modes of
    every component and the fix history on an empty board, then the reads the surface still needs before a reading."""
    if "No profile read yet" in prompt:
        return plan_json([{"kind": "profile"}, {"kind": "failure_modes", "args": {"component": ""}}, {"kind": "fix_history"}], mode="test_first")
    missing = STILL_TO_READ.search(prompt)
    if missing:
        return plan_json([{"kind": kind.strip()} for kind in missing.group(1).split(",")], mode="test_first")
    return None


def decision_plans(plans):
    """The plans of a plans.json that do more than read the opening, the first of them the first decision."""
    return [p for p in plans["plans"] if any(s["kind"] not in OPENING_KINDS for s in p["plan"]["steps"])]


def act_now_steps(prompt, prediction=None):
    """Diagnose the top candidate, order its faulty components that have no order yet, advance to the last close, finish."""
    steps = []
    if "Commitment: none" in prompt:
        steps.append({"kind": "diagnose", "args": {"rank": 1}, "expect": {"top_rank": 1}, "evidence": EVIDENCE_ID.findall(prompt)[:3]})
    ordered = set(ORDERED.findall(prompt))
    ids = branch_predictions(prompt) if prediction is None else {}
    for component in top_faulty(prompt):
        if component in ordered:
            continue
        pid = ids.get(component, prediction or ESTIMATE)
        steps.append({"kind": "order", "args": {"component": component, "prediction_id": pid},
                      "expect": {"prediction_id": pid, "recovery_at_least": 0.0}, "evidence": [pid] if pid != ESTIMATE else []})
    steps.append({"kind": "wait", "args": {"cycles": LONGEST_REPAIR}, "expect": {"no_new_alarm": True}, "evidence": []})
    steps.append(dict(VERIFYING_READ))  # the controller verifies a repair only on a reading a plan names
    steps.append({"kind": "finish", "args": {}, "expect": {}, "evidence": []})
    return steps


def advised_read(prompt):
    """A plan of one read, the head of the read batch the board advises while nothing is committed, the condition of the
    evidence guard, or None when the board advises none; the read expects nothing, so the advice alone moves the plan."""
    match = ADVISED.search(prompt)
    if "Commitment: none" not in prompt or match is None:
        return None
    read = {"kind": "read", "args": {"sensor_group": match.group(1), "cycles": int(match.group(2))}, "expect": {}, "evidence": EVIDENCE_ID.findall(prompt)[:2]}
    return plan_json([read], mode="test_first")


def act_now(prompt, call):
    """Read the head of the advised batch while the board advises one, else act on the top candidate, citing the branch's
    prediction when the board shows troubleshooting plans, else estimates."""
    return advised_read(prompt) or plan_json(act_now_steps(prompt, prediction=ESTIMATE) if "## Troubleshooting plans" not in prompt else act_now_steps(prompt))


def act_now_with_estimates(prompt, call):
    return advised_read(prompt) or plan_json(act_now_steps(prompt, prediction=ESTIMATE))


def reads_then_acts(prompt, call):
    """The first plan reads one more cycle of the cheapest group, a plan that tests before it commits; a later plan acts."""
    if call == 1:
        read = {"kind": "read", "args": {"sensor_group": "tier1", "cycles": 1}, "expect": {"top_rank": 1, "top_mass_at_least": 0.0},
                "evidence": EVIDENCE_ID.findall(prompt)[:2]}
        return plan_json([read], mode="test_first")
    return act_now(prompt, call)


class ScriptedPlanner(LLMBackend):
    """A planner without a model, in the style of ScriptedBackend: the orientation and the summary are fixed replies, and each
    planning turn is answered by `plans(prompt, call)` from the board in the prompt, so the script holds on every seed.
    `on_retry` answers a planning turn that quotes a parse error or the reasons of a refused plan."""

    def __init__(self, plans, on_retry=None, orientation=ORIENTATION_REPLY, summary=SUMMARY_REPLY):
        self._model_id = "scripted-causal"
        self.plans, self.on_retry, self.orientation, self.summary = plans, on_retry, orientation, summary
        self.prompts, self.plan_calls = [], 0

    def start_episode(self):
        self.prompts, self.plan_calls = [], 0

    def generate(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        if ORIENTATION_HEADING in prompt:
            return self.orientation
        if SUMMARY_HEADING in prompt:
            return self.summary
        assert PLANNING_HEADING in prompt
        opening = opening_reply(prompt)
        if opening is not None:  # the opening is a plan of its own, which no script counts among its calls
            return opening
        self.plan_calls += 1
        if RETRY_HEADING in prompt and self.on_retry is not None:
            return self.on_retry(prompt, self.plan_calls)
        return self.plans(prompt, self.plan_calls)

    @property
    def model_id(self):
        return self._model_id


class InProcessServers:
    """Both MCP servers called in process through FastMCP, their answers as the version 7 servers return them. force_read_first
    makes the first rankings say read first, with a separating test and a read batch made from the server's
    elimination rows when it names none, so a batch is certain to run; refute_first turns the first verified verdict
    inconsistent, so backtracking is certain to fire; promote_after_order puts a candidate with one more valve fault on top of
    every ranking from the first work order on and calls a verified plan consistent, so the belief after the verifying reading
    disagrees with a plan whose verdict stands."""

    def __init__(self, ep_dir, force_read_first=0, refute_first=False, promote_after_order=False):
        self.ep_dir = Path(ep_dir)
        self.force_read_first, self.refute_first, self.promote_after_order = force_read_first, refute_first, promote_after_order
        self.rig = build_rig_server(self.ep_dir / "episode.json")
        self.causal = None
        self.rankings = self.refuted = self.ordered = 0

    def diagnostics(self):
        """The diagnostics server, built once the episode has started, as the driver writes its launcher then and names the
        combination the episode started in, read from the reset without the actions played since."""
        if self.causal is None:
            state = {**episode.load_state(self.ep_dir / "episode.json"), "actions": [], "think_cycles": []}
            self.causal = build_causal_server(self.ep_dir / "released.json", left_out=tuple(episode.replay(state).initial))
        return self.causal

    def with_other_plan(self, ranking):
        """After the fake refuted a plan, one more candidate and branch whose repair plan is a valve replacement without a
        prediction, so that backtracking has another plan to repair whatever the belief says after the repair."""
        top = ranking["candidates"][0]
        rank = len(ranking["candidates"]) + 1
        combination = {**top["combination"], "valve": 90}
        current = {**(top.get("current_combination") or top["combination"]), "valve": 90}
        ranking["candidates"].append({**top, "rank": rank, "combination": combination, "current_combination": current, "probability": 0.0,
                                      "faults": [{"component": "valve", "level": 90, "code": "valve_switching_small_lag"}]})
        branches = ranking.get("branches") or []
        if branches:
            template = branches[0]
            branches.append({**template, "rank": rank, "ranks": [rank], "combination": combination, "current_combination": current, "probability": 0.0,
                             "faulty_set": ["valve"], "repair_plan": {"steps": [{"component": "valve", "part_price": 20, "repair_charge": 20, "lead_time": 3,
                                                                                 "repair_downtime": 1, "downtime_cost": 0.0, "prediction_id": ""}],
                                                                      "plan_prediction_id": None, "price": 40, "downtime_cost": 0.0, "cycles_to_restored": 5},
                             "cost_to_restored": {"act_now": 1.0, "test_first": 2.0}})
        return ranking

    def with_valve_on_top(self, ranking):
        """The candidate and branch of with_other_plan at rank 1 with most of the mass, so the top of the belief is a valve fault that
        the plan being verified does not predict. Every other row moves one rank down, so a branch keeps the rank of its candidate."""
        ranking = self.with_other_plan(ranking)
        added = len(ranking["candidates"])  # the rank with_other_plan gives the rows it appends
        for key in ("candidates", "branches"):
            rows = ranking.get(key) or []
            for row in rows:
                if row is rows[-1] and row["rank"] == added:
                    row["rank"], row["probability"] = 1, 0.9
                else:
                    row["rank"], row["probability"] = row["rank"] + 1, 0.1 * row.get("probability", 0.0)
                if "ranks" in row:  # a branch lists every rank of its faulty set, which move with the candidates
                    row["ranks"] = [1] if row["rank"] == 1 else [r + 1 for r in row["ranks"]]
            rows.sort(key=lambda row: row["rank"])
        return ranking

    async def call(self, path, tool, args):
        rig = Path(path).name.startswith("hydraulic_rig")
        server = self.rig if rig else self.diagnostics()
        result = await server.call_tool(tool, args)
        content = result[0] if isinstance(result, tuple) else result
        text = "\n".join(getattr(c, "text", str(c)) for c in content)
        if tool == "rank_root_causes":
            ranking = self.forced_read_first(json.loads(text))
            if self.refuted:
                ranking = self.with_other_plan(ranking)
            elif self.promote_after_order and self.ordered and ranking.get("candidates"):
                ranking = self.with_valve_on_top(ranking)
            text = json.dumps(ranking)
        if tool == "verify_outcome" and self.refute_first and self.refuted == 0:
            verdict = json.loads(text)
            if verdict.get("status") == "verified":
                verdict["consistent"], self.refuted = False, 1
                text = json.dumps(verdict)
        if tool == "verify_outcome" and self.promote_after_order:
            verdict = json.loads(text)
            if verdict.get("status") == "verified":
                verdict["consistent"] = True
                text = json.dumps(verdict)
        if tool == "create_work_order":
            self.ordered += 1
        return text

    def forced_read_first(self, ranking):
        if "error" in ranking:
            return ranking
        self.rankings += 1
        if self.rankings > self.force_read_first or ranking.get("read_first"):
            return ranking
        # the batch head goes by information gain per unit cost, by the elimination rate on a server without it
        rows = sorted(ranking.get("elimination") or [],
                      key=lambda r: (-(r.get("ig_per_cost") if r.get("ig_per_cost") is not None else r.get("elimination_per_cost") or 0), r.get("sensor_group")))
        forced = {**ranking, "read_first": True}
        if not ranking.get("separating_test") and rows:
            best = rows[0]
            forced["separating_test"] = {"sensor_group": best["sensor_group"], "cycles": 1, "price": best.get("price"), "duration": best.get("duration", 1),
                                         "cost": best.get("cost"), "expected_loss": (ranking.get("expected_loss_test") or {}).get(best["sensor_group"])}
        if not ranking.get("read_batch") and rows:
            forced["read_batch"] = [{"sensor_group": r["sensor_group"], "cycles": 1, "ig_per_cost": r.get("ig_per_cost")} for r in rows[:2]]
        return forced


def sessions_in_process(servers_of):
    """A stand-in of CausalDiagnosis.agents.utils.mcp_sessions.open_sessions for the controller: a block over the launchers of a run whose calls go, by
    the server's name, to the in-process servers that servers_of(launcher path) returns. An answer that raises in process comes
    back as the stdio server sends it, its text with the error mark, and a server the block did not open is a call without an
    answer, as in the module."""

    class Sessions:
        def __init__(self, server_paths):
            self.paths, self.tools = dict(server_paths), None  # no opening listing: the controller serves every tool (tools.Surface.listed)

        async def call(self, server, tool, arguments):
            if server not in self.paths:
                raise ToolTransportError(f"{tool} of {server} got no answer: the block opened no server of that name")
            path = self.paths[server]
            try:
                return await servers_of(path).call(path, tool, arguments), False
            except ToolError as exc:
                return str(exc), True

    @contextlib.asynccontextmanager
    async def open_sessions(server_paths):
        yield Sessions(server_paths)

    return open_sessions


@contextlib.contextmanager
def _in_process(ep_dir, **fake):
    """Route the runner's tool calls to InProcessServers through the stand-in of the session module, and skip the stdio tool
    listing of the driver."""
    servers = InProcessServers(ep_dir, **fake)
    with mock.patch.object(controller, "open_sessions", sessions_in_process(lambda path: servers)), \
            mock.patch.object(run_episodes.play_one_episode, "check_servers", lambda launchers, row=None: None):
        yield servers


def _play(tmp_path, planner, options, seed=0, config_name="static", **fake):
    """One episode of the causal agent through EpisodeRunner, as main() plays it, with the servers in process; returns the
    episode folder, the row and the recording backend."""
    facts_path, cycles, blocks = load_world()
    config_name, config = load_config(config_name)
    backend = RecordingBackend(planner)
    runner = EpisodeRunner(config, config_name, "causal", (RIG_SERVER, CAUSAL_SERVER), cycles, blocks, facts_path, "causal-test",
                           {"judge": None, "grade": True, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL}, backend=backend, scripted=planner,
                           model=planner.model_id, causal_options={**{"branches": True, "single_hypothesis": False, "plan_length": None, "expectations": True,
                                                                     "rule_only": False}, **options})
    name = f"seed{seed:06d}"
    ep_dir = tmp_path / name
    meta = {"config_name": config_name, "agent": "causal", "servers": [RIG_SERVER, CAUSAL_SERVER], "model": planner.model_id,
            "clock_mode": config.clock_mode, "guidance_level": config.guidance_level}
    with _in_process(ep_dir, **fake):
        row = runner.play(ep_dir, name, seed, None, config, meta)
    return ep_dir, row, backend


def _tools(ep_dir, server=None):
    trajectory = read_json(ep_dir / "trajectory.json")
    return [t["tool"] for t in trajectory if server is None or t["server"] == server]


def answer_context_route(handler, body, context=1_000_000, count=None):
    """The two routes a chat backend counts the context from (CausalDiagnosis.utils.model_reply_room): the model listing with max_model_len and the tokenize
    count, the words of the messages unless count fixes it. True when the request was one of them, which a stand-in then leaves
    out of its record of chat requests."""
    path = handler.path.rstrip("/")
    if handler.command == "GET" and path.endswith("/models"):
        reply = {"object": "list", "data": [{"id": "stand-in", "object": "model", "max_model_len": context}]}
    elif handler.command == "POST" and path.endswith("/tokenize"):
        text = " ".join(str(m.get("content", "")) for m in body.get("messages", [])) or str(body.get("prompt", ""))
        reply = {"count": len(text.split()) if count is None else count}
    else:
        return False
    data = json.dumps(reply).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return True


def _planner_endpoint(planner, context=1_000_000, count=None, hold=0.0, fail_first=False):
    """A local OpenAI compatible endpoint that hands the prompt of every chat completion to a scripted planner, so the driver
    plays the causal agent through the real client, request and response path without a model. It serves the routes of the
    context (answer_context_route) at a context of context tokens, every prompt counted as count tokens when count is given.
    Every completion stays open for hold seconds, and server.open_requests["most"] keeps the most completions open at once.
    fail_first answers 503 to the first attempt of every chat completion, so that the client's retry policy must attempt it
    again; server.stats counts the failed attempts and the most requests open at once, and lists the body of every attempt
    in the order of arrival."""
    requests = []
    stats, failed_bodies, lock = {"failed": 0, "open": 0, "most_open": 0, "arrivals": []}, set(), threading.Lock()
    open_lock = threading.Lock()
    open_requests = {"now": 0, "most": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the http.server hook name
            if not answer_context_route(self, None, context, count):
                self.send_error(404)

        def do_POST(self):  # noqa: N802 - the http.server hook name
            with lock:
                stats["open"] += 1
                stats["most_open"] = max(stats["most_open"], stats["open"])
            try:
                self._answer()
            finally:
                with lock:
                    stats["open"] -= 1

        def _answer(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if answer_context_route(self, body, context, count):
                return
            key = json.dumps(body, sort_keys=True)
            with lock:
                stats["arrivals"].append(key)
            if fail_first and key not in failed_bodies:
                failed_bodies.add(key)
                stats["failed"] += 1
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            with open_lock:
                requests.append((self.path, body))
                open_requests["now"] += 1
                open_requests["most"] = max(open_requests["most"], open_requests["now"])
            try:
                time.sleep(hold)
                content = planner.generate(body["messages"][-1]["content"])
            finally:
                with open_lock:
                    open_requests["now"] -= 1
            reply = {"id": f"planner-{len(requests)}", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                     "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.open_requests = open_requests
    server.stats = stats
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def _post_completion(base, prompt):
    body = json.dumps({"model": "vendor/stand-in-1.0", "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(base + "chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as answer:
        return json.loads(answer.read())


def _assert_every_read_planned(ep_dir, plans):
    """Every reading is a read step of a plan the model wrote, so the controller never reads on its own, and the
    board's read log holds one entry per reading at the plan and the step that named it."""
    trajectory = read_json(ep_dir / "trajectory.json")
    reads = [t for t in trajectory if t["tool"] in ("read_sensors", "history")]
    log = plans["board"]["reads"]
    assert len(log) == len(reads), (len(log), len(reads))
    by_index = {p["index"]: p for p in plans["plans"]}
    for entry, call in zip(log, reads):
        plan = by_index[entry["plan"]]
        step = plan["plan"]["steps"][entry["step"] - 1]
        assert plan["source"] in MODEL_SOURCES and step["kind"] == "read" and step["args"]["sensor_group"] == entry["sensor_group"], entry
    assert not [p for p in plans["plans"] if p["source"] == "read_batch" or p["source"].startswith("rule")]


def _assert_episode_files(ep_dir, row, server=True):
    for name in ("episode.json", "released.json", "trajectory.json", "llm_calls.json", "plans.json", "plan.txt", "answer.txt", "report.md", "question.txt"):
        assert (ep_dir / name).exists(), name
    for name in HIDDEN_FILES:
        assert (ep_dir / HIDDEN / name).exists(), name
    assert list(row) == list(COLUMNS) and row["error"] == "", row["error"]
    assert row["agent"].startswith("causal") and row["model"] == "scripted-causal"
    # the row names the entry offset of the hidden log, the launcher the start whose block the fits leave out, and the evidence the agent reads never does
    assert read_json(ep_dir / HIDDEN / "log.json")[0]["entry_offset"] == row["entry_offset"]
    assert (ep_dir / f"{CAUSAL_SERVER}_server.py").exists() is server
    if server:
        assert (ep_dir / "predictions.json").exists()
        assert f"left_out={tuple(read_json(ep_dir / HIDDEN / 'log.json')[0]['hidden'])!r}" in (ep_dir / f"{CAUSAL_SERVER}_server.py").read_text(encoding="utf-8")
    assert "entry_offset" not in (ep_dir / "released.json").read_text(encoding="utf-8")
    verdict = read_json(ep_dir / HIDDEN / "verdict.json")
    assert verdict["verified"] and row["pass"] == verdict["pass"]
    assert "## Plan" in (ep_dir / "report.md").read_text(encoding="utf-8")
    return verdict


def _assert_repairs_verified(ep_dir, verdict):
    """Every order cites a stored prediction, and every cited prediction was verified once the last order citing it closed:
    a plan's prediction (pred/b...) by a reading the server could compare, an estimate's by whatever the server could say."""
    predictions = {p["id"] for p in read_json(ep_dir / "predictions.json")}
    trajectory = read_json(ep_dir / "trajectory.json")
    orders = [t for t in trajectory if t["tool"] == "create_work_order" and "refused" not in t["response"]]
    verified = {t["tool_args"]["prediction_id"] for t in trajectory if t["tool"] == "verify_outcome"}
    verifications = read_json(ep_dir / "plans.json")["board"]["verifications"]
    assert orders, "the plan placed no work order"
    for order in orders:
        cited = order["tool_args"]["prediction_id"]
        assert cited in predictions and cited in verified and cited in verifications
        if cited.startswith("pred/b"):
            assert verifications[cited]["status"] == "verified" and verifications[cited]["consistent"] in (True, False), verifications[cited]
    assert verdict["checks"]["causality"]["ok"] and verdict["checks"]["provenance"]["ok"]
    assert len(verdict["prediction_verdict"]) == len(orders) and all(p["found"] for p in verdict["prediction_verdict"])
    assert verdict["end_reason"] == "finish"
    # the orders of one plan are created in one cycle, so they form one visit and pay its charge once (version 7)
    created = [e for e in read_json(ep_dir / "released.json")["events"] if e["event"] == "work_order_created"]
    by_cycle = {}
    for event in created:
        by_cycle.setdefault(event["t_start"], []).append(event)
    for events in by_cycle.values():
        assert len({e["visit"] for e in events}) == 1 and sum(1 for e in events if e["visit_charge"]) == 1
    return orders


def test_a_test_first_plan_reads_diagnoses_orders_advances_verifies_and_finishes(tmp_path):
    planner = ScriptedPlanner(reads_then_acts)
    ep_dir, row, backend = _play(tmp_path, planner, {})
    verdict = _assert_episode_files(ep_dir, row)
    _assert_repairs_verified(ep_dir, verdict)
    plans = read_json(ep_dir / "plans.json")
    assert row["first_plan_mode"] == plans["mode_of_first_plan"] == "test_first" and row["plans"] == len(plans["plans"]) >= 1
    first = decision_plans(plans)[0]  # the model plans every turn, the opening and the first read included
    assert first["source"] == "model" and first["plan"]["steps"][0]["kind"] == "read"
    assert first["plan"]["steps"][0]["expect"] == {"top_rank": 1, "top_mass_at_least": 0.0}
    # no batch reads before it any more, so the first read meets the ranking of the prior and may move its top
    assert first["steps"][0]["status"] in ("done", "surprise"), first["steps"][0]
    assert all(s["status"] in ("done", "surprise") for p in plans["plans"] for s in p["steps"])
    assert plans["differential"][0]["component"] == "pump" and plans["server"] is True
    rig = _tools(ep_dir, RIG_SERVER)
    assert rig[:7] == ["get_asset_profile", *["get_failure_modes"] * len(COMPONENTS), "get_fix_history", "read_sensors"] and rig[-1] == "finish"
    assert rig.count("read_sensors") >= 2 and rig.count("submit_diagnosis") == 1 and "summarize_changes_since" in rig
    _assert_every_read_planned(ep_dir, plans)
    # the controller waits event by event, so it learns of every close by its notification
    assert "wait_for_notification" in rig and "advance_time" not in rig
    notifications = read_json(ep_dir / "plans.json")["board"]["notifications"]
    assert [n["event"] for n in notifications].count("work_order_closed") == rig.count("create_work_order")
    # the controller ranks after every reading and again when the clock moved the belief, so the model never plans on a stale ranking
    assert _tools(ep_dir, CAUSAL_SERVER).count("rank_root_causes") >= rig.count("read_sensors")
    # one model call for the orientation, one per plan and the summary
    assert row["llm_calls"] == len(backend.calls) == 1 + sum(1 for p in backend.calls if PLANNING_HEADING in p["prompt"]) + 1
    assert row["causal_calls"] == len(_tools(ep_dir, CAUSAL_SERVER)) and row["revisions"] == plans["revisions"] and row["fallbacks"] == 0
    assert (ep_dir / "answer.txt").read_text(encoding="utf-8") == SUMMARY_REPLY
    finish = next(e for e in read_json(ep_dir / "released.json")["events"] if e["event"] == "finish")
    assert finish["summary"] == SUMMARY_REPLY
    assert f"Plan {first['index']} (model, branch 1, mode test_first)" in (ep_dir / "plan.txt").read_text(encoding="utf-8")


def test_an_act_now_plan_orders_the_repair_plan_of_the_top_branch_and_verifies_it_on_its_own_read(tmp_path):
    planner = ScriptedPlanner(act_now)
    ep_dir, row, backend = _play(tmp_path, planner, {})
    verdict = _assert_episode_files(ep_dir, row)
    _assert_repairs_verified(ep_dir, verdict)
    plans = read_json(ep_dir / "plans.json")
    _assert_every_read_planned(ep_dir, plans)
    acting = next(p for p in plans["plans"] if p["plan"]["mode"] == "act_now")
    assert acting["plan"]["steps"][0]["kind"] == "diagnose" and [s["kind"] for s in acting["plan"]["steps"]][-2:] == ["read", "finish"]
    trajectory = read_json(ep_dir / "trajectory.json")
    orders = [t for t in trajectory if t["tool"] == "create_work_order"]
    board_prompt = next(c["prompt"] for c in backend.calls if PLANNING_HEADING in c["prompt"] and "## Troubleshooting plans" in c["prompt"])
    assert "every order of the plan cites pred/b" in board_prompt
    # every order of the plan cites the plan's prediction, which the ranking registered, so no estimate call was needed and
    # one verification after the last close covers the plan
    cited = {o["tool_args"]["prediction_id"] for o in orders}
    assert len(cited) == 1 and next(iter(cited)).startswith("pred/b")
    assert "estimate_intervention_effect" not in _tools(ep_dir, CAUSAL_SERVER)
    assert [t["tool_args"]["prediction_id"] for t in trajectory if t["tool"] == "verify_outcome"] == list(cited)
    # the verification follows a read the plan named after the last wait, and no reading happens inside a wait or a finish
    last_wait = max(i for i, t in enumerate(trajectory) if t["tool"] == "wait_for_notification")
    verify = next(i for i, t in enumerate(trajectory) if t["tool"] == "verify_outcome")
    between = [t for t in trajectory[last_wait + 1:verify] if t["tool"] == "read_sensors"]
    assert between and between[0]["tool_args"] == VERIFYING_READ["args"]


def test_a_failed_expectation_is_a_surprise_that_triggers_a_second_plan(tmp_path):
    def plans(prompt, call):
        if call == 1:
            # the planned candidate is rank 5 while rank 1 is on top, so the check after the read fails; the read of the advised head
            # satisfies the evidence guard of the plan check
            head = ADVISED.search(prompt)
            read = [{"kind": "read", "args": {"sensor_group": head.group(1), "cycles": 1}, "expect": {}, "evidence": []}] if head else []
            return plan_json([*read, {"kind": "diagnose", "args": {"rank": 1}, "expect": {"top_rank": 5}, "evidence": []}])
        return act_now_with_estimates(prompt, call)
    planner = ScriptedPlanner(plans)
    ep_dir, row, backend = _play(tmp_path, planner, {})
    verdict = _assert_episode_files(ep_dir, row)
    _assert_repairs_verified(ep_dir, verdict)
    plans_file = read_json(ep_dir / "plans.json")
    first, second = decision_plans(plans_file)[:2]
    diagnose_step = len(first["plan"]["steps"])
    assert first["surprise"]["kind"] == "diagnose" and first["surprise"]["step"] == diagnose_step and "not the planned candidate" in first["surprise"]["reason"]
    assert first["steps"][-1]["status"] == "surprise" and second["after_surprise"] is True
    assert row["surprises"] == len(plans_file["surprises"]) >= 1 and row["revisions"] >= 1 and row["plans"] >= 2
    assert _tools(ep_dir, RIG_SERVER).count("submit_diagnosis") == 1  # the surprised diagnosis was never submitted
    # the surprise is on the board of the next planning turn
    assert any("not the planned candidate" in c["prompt"] for c in backend.calls if PLANNING_HEADING in c["prompt"])
    # the later plans asked for estimates, so the controller called estimate_intervention_effect before every order
    orders = [t for t in read_json(ep_dir / "trajectory.json") if t["tool"] == "create_work_order"]
    assert orders and all(re.fullmatch(r"pred/\d+", o["tool_args"]["prediction_id"]) for o in orders)
    assert _tools(ep_dir, CAUSAL_SERVER).count("estimate_intervention_effect") == len(orders)


def test_a_plan_the_check_refuses_goes_back_to_the_model_with_the_reasons(tmp_path):
    def plans(prompt, call):
        if call > 1:  # the first decision turn alone is refused; a later turn, after a read that the board advised, acts
            return act_now(prompt, call)
        return plan_json([{"kind": "order", "args": {"component": "pump", "prediction_id": ESTIMATE}, "expect": {}, "evidence": []},
                          {"kind": "finish", "args": {}, "expect": {}, "evidence": []}])

    def on_retry(prompt, call):
        if "Commitment: none" in prompt:  # before the commitment both reasons apply, a later turn's refusal names the close alone
            assert "a diagnosis or a declaration must precede every work order" in prompt
        assert "finish precedes the close" in prompt
        return act_now(prompt, call)
    planner = ScriptedPlanner(plans, on_retry=on_retry)
    ep_dir, row, backend = _play(tmp_path, planner, {})
    verdict = _assert_episode_files(ep_dir, row)
    _assert_repairs_verified(ep_dir, verdict)
    plans = read_json(ep_dir / "plans.json")
    # the refused turn runs nothing and makes no second request inside the turn; its reasons go to the next planning turn
    refused = [p for p in plans["plans"] if p["source"] == "no_plan_after_check_failure"]
    assert len(refused) == 1 and refused[0]["steps"] == [] and len(refused[0]["check_problems"]) == 1
    reasons = refused[0]["check_problems"][0]
    assert any("precede every work order" in r for r in reasons) and any("finish precedes the close" in r for r in reasons)
    first = decision_plans(plans)[0]
    assert first["source"] == "model" and first["plan"]["steps"][0]["kind"] in ("read", "diagnose") and row["fallbacks"] == 1
    planning = [c["prompt"] for c in backend.calls if PLANNING_HEADING in c["prompt"]]
    assert sum(RETRY_HEADING in p for p in planning) == 1 and len(planning) == len(plans["plans"])
    assert "refused by the plan check" in (ep_dir / "plan.txt").read_text(encoding="utf-8")


def test_the_system_prompt_tells_that_a_turn_without_a_call_takes_a_step():
    """Told equals done: the prompt states the no_call step and the end it brings, and no longer
    a count of retries."""
    text = system_prompt(None)
    assert "A turn that calls no tool of the" in text and "still takes one step" in text and "as the step cap allows end the episode" in text
    assert "refuses twice" not in text and "asked again with its reasons" in text
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import LOOP, NOUN_MARK
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import idle_turn_sentence
    assert idle_turn_sentence(NOUN_MARK) in LOOP  # the sentence that both briefings state beside the step cap, in the same words


def test_a_read_is_judged_on_the_faulty_set_on_top_and_its_mass_summed_over_the_set():
    """A reading that confirms the cooler splits the mass over two levels of it. The read holds, since the
    set on top is the cooler's and its mass as the server sums it over every combination of the set meets the expectation, although
    the top combination is the other level; with the valve's set on top the same read is a surprise."""
    import types
    runner = controller.CausalLoopRunner(llm=None, server_paths={CAUSAL_SERVER: "unused"})
    healthy = dict(HEALTHY)
    cooler, valve = COMPONENTS[0], COMPONENTS[1]

    class StandInBoard:
        profile = {"healthy": healthy}
        latest_cycles = {}
        faulty_set = Board.faulty_set
        top_set = Board.top_set

        def __init__(self, set_masses):
            self.ranking = {"set_masses": set_masses}

        def effective_hypotheses(self):
            return None
    low, high = {**healthy, cooler: "low"}, {**healthy, cooler: "high"}
    record = types.SimpleNamespace(candidates=[{"combination": low}, {"combination": {**healthy, valve: "low"}}])
    step = Step("read", args={"sensor_group": "tier1", "cycles": 1}, expect={"top_rank": 1, "top_mass_at_least": 0.6})
    top = {"combination": high, "probability": 0.35}  # the other level of the cooler holds the top combination
    runner.board = StandInBoard([{"faulty_set": [cooler], "mass": 0.7}, {"faulty_set": [valve], "mass": 0.3}])
    assert runner._judge_reading(step, record, {}, top).status == "done"
    runner.board = StandInBoard([{"faulty_set": [cooler], "mass": 0.4}, {"faulty_set": [valve], "mass": 0.6}])
    judged = runner._judge_reading(step, record, {}, top)
    assert judged.status == "surprise" and "faulty set on top" in judged.note


def test_the_summary_request_leaves_out_the_fewest_oldest_entries_that_make_it_fit():
    """The entries a summary replaces are ordered by the planning turn that wrote them, the
    bisection finds the fewest oldest entries to leave out of the summary request so that it fits the context, the request then
    says they will not return, the reply never enters the reply term, a reply cut by length is no summary, and a request that
    fits with nothing at all is no summary request."""
    import types
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import prompts
    board = _synthetic_board(read_first=False, second=True)
    board.mark_turn(1)
    board.notes = ["old note " + "y" * 2000]
    board.mark_turn(2)
    board.notes.append("new note")
    board.surprises = [{"plan": 1, "step": 1, "kind": "read", "reason": "z"}, {"plan": 2, "step": 1, "kind": "read", "reason": "w"}]
    entries = board.replaceable()
    turns = [e["turn"] for e in entries]
    assert turns == sorted(turns)
    notes = [e for e in entries if e["kind"] == "notes"]
    assert [e["index"] for e in notes] == [0, 1] and entries.index(notes[0]) < entries.index(notes[1])
    assert [e for e in entries if e["kind"] == "surprises"] == [{"kind": "surprises", "index": 0, "turn": 2}]
    copy_ = board.without(entries[:entries.index(notes[0]) + 1])
    assert copy_.notes == ["new note"] and len(board.notes) == 2

    class Llm:
        longest_reply = 7

        def __init__(self, limit, finish="stop"):
            self.limit, self.finish, self.asked = limit, finish, []

        def request_fits(self, prompt):
            return len(prompt) <= self.limit

        def generate_with_usage(self, prompt, temperature):
            self.asked.append(prompt)
            self.longest_reply = 10 ** 6
            return types.SimpleNamespace(text="the summary", finish_reason=self.finish)
    runner = controller.CausalLoopRunner(llm=None, server_paths={CAUSAL_SERVER: "unused"})
    runner.board, runner.model_calls, runner._cut = board, 0, False
    full = len(prompts.earlier_turns_turn(board, None, 0, surface=runner.surface))
    runner._llm = Llm(full - 1000)
    text, k = runner._summary_of_earlier_turns(board, entries)
    assert text == "the summary" and k == entries.index(notes[0]) + 1 and runner._llm.longest_reply == 7
    assert prompts.LEFT_OUT in runner._llm.asked[-1] and "old note" not in runner._llm.asked[-1]
    runner._llm = Llm(full - 1000, finish="length")
    assert runner._summary_of_earlier_turns(board, entries)[0] == ""
    runner._llm = Llm(10)
    assert runner._summary_of_earlier_turns(board, entries) is None and runner._llm.asked == []
    # a backend that counts the context but cannot read the trigger is refused, never judged against the whole context; one
    # that counts nothing, such as a scripted planner, takes every request as under the trigger
    runner._llm = Llm(10)
    with pytest.raises(TypeError, match="no request_under_trigger"):
        runner._request_under_trigger(board)
    # and the run refuses such a backend at its start, before any model call
    import asyncio
    refused = controller.CausalLoopRunner(llm=Llm(10), server_paths={CAUSAL_SERVER: "unused"})
    with pytest.raises(TypeError, match="no request_under_trigger"):
        asyncio.run(refused.run("q"))
    assert refused._llm.asked == []
    runner._llm = types.SimpleNamespace()
    assert runner._request_under_trigger(board) is True
    runner._llm = types.SimpleNamespace(request_under_trigger=lambda prompt: False, request_fits=lambda prompt: True)
    assert runner._request_under_trigger(board) is False


def test_a_recall_step_brings_collapsed_rows_back_on_the_next_board_only():
    """A recall {sensor_group, observations} is free and calls no tool; the full rows of the group's
    collapsed observations come back on the board of the next planning turn and leave after it; the plan check refuses a recall
    that brings nothing back, and a recall never makes a request too long for the context."""
    import asyncio
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import parse_step
    board = _synthetic_board(read_first=False, second=True)
    for n in range(2, 5):
        board.add_reading({"sensor_group": "tier3", "served": [{"n": n, "t": n - 1, "values": {"FS1_VolumeFlow_l_per_min_10Hz_mean": 30.0 + n}}]})
    board.status["time"] = 4
    board.set_ranking(board.ranking)
    board.compact()
    tier3 = [r for r in board.collapsed_rows if r["sensor_group"] == "tier3"]
    assert tier3 and all(r["evidence_id"] in board.known_ids() for r in tier3)
    first, last = tier3[0]["n"], tier3[-1]["n"]
    step, = parse_step({"kind": "recall", "args": {"sensor_group": "tier3", "observations": [first, last]}}, 1)
    with pytest.raises(PlanError):
        parse_step({"kind": "recall", "args": {"sensor_group": "tier3", "observations": [3, 2]}}, 1)
    assert not [p for p in check_plan(Plan(branch=None, mode="test_first", why="", steps=[step]), board) if "(recall)" in p]
    empty, = parse_step({"kind": "recall", "args": {"sensor_group": "tier2", "observations": [1, 1]}}, 1)
    assert any("nothing comes back" in p for p in check_plan(Plan(branch=None, mode="test_first", why="", steps=[empty]), board))
    runner = controller.CausalLoopRunner(llm=None, server_paths={CAUSAL_SERVER: "unused"})
    runner.board, runner.trajectory = board, []
    board.mark_turn(1)
    outcome = asyncio.run(runner._recall(step, None, 1))
    assert outcome.status == "done" and outcome.observed == {"rows": len(tier3)} and runner.trajectory == []
    assert "## Recalled readings" not in board.render()
    board.mark_turn(2)
    assert "## Recalled readings" in board.render() and tier3[0]["evidence_id"] in board.render().split("## Recalled readings")[1]
    board.mark_turn(3)
    assert "## Recalled readings" not in board.render()
    board.recall("tier3", first, last)
    board.mark_turn(4)
    board.drop_unfit_recall(lambda b: True)
    assert board.recalled and "## Recalled readings" in board.render()
    # at the start of the turn that shows them, the rows leave when the request with them would not fit the context
    board.drop_unfit_recall(lambda b: "## Recalled readings" not in b.render())
    assert board.recalled == [] and board.compaction_log[-1]["collapsed"] == [{"kind": "recall_dropped", "rows": len(tier3)}]
    # the switch recall off: the prompts drop the step and the plan check refuses it
    import types
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import RECALL_LINE, planning_turn
    assert RECALL_LINE in planning_turn(board, None) and "a seventh, recall" in planning_turn(board, None)
    board.recall_offered = False
    assert RECALL_LINE not in planning_turn(board, None) and "a seventh, recall" not in planning_turn(board, None)
    assert any("offers no recall step" in p for p in check_plan(Plan(branch=None, mode="test_first", why="", steps=[step]), board))
    options = agent_options.CausalLoopOptions(recall=False, summary_turn=False)
    assert options.label() == "causal_norecall_nosumturn" and options.to_json()["recall"] is False
    assert "recall" not in agent_options.CausalLoopOptions().to_json() and "summary_turn" not in agent_options.CausalLoopOptions().to_json()
    # the switch summary_turn off: the compaction gets no summary request, so a request that does not fit ends the run as context
    seen = {}
    runner.options = options
    runner.board = types.SimpleNamespace(compact=lambda fits, after_call=None, summarize=None, fits_context=None: seen.update(summarize=summarize))
    runner._compact()
    assert seen == {"summarize": None}


def test_an_entry_that_holds_an_identifier_the_summary_names_stays_word_for_word():
    """After the summary turn, a note, a notification, a surprise or a timeline event that holds an
    identifier the summary names stays on the board as it was; the other replaced entries leave."""
    board = _synthetic_board(read_first=False, second=True)
    board.orders["wo-7"] = {"id": "wo-7", "component": "pump", "status": "open", "prediction_id": None}
    board.notes = ["the plan of turn 1 placed wo/wo-7 on the pump", "a plain note"]
    board.mark_turn(1)
    entries = [e for e in board.replaceable() if e["kind"] == "notes"]
    record = board.apply_summary("We replaced the pump under wo/wo-7.", entries, 0)
    assert board.notes == ["the plan of turn 1 placed wo/wo-7 on the pump"]
    assert record["replaced"]["notes"] == 1 and record["kept_for_named_identifiers"] == {"notes": 1}


def test_an_entry_the_summary_names_stays_through_later_summaries_and_only_its_own_event_returns():
    """An entry kept because a summary names it leaves the replaceable set, so a later summary that names the same identifier
    finds nothing to replace; the timeline hides the events a summary replaced by their identifiers, so keeping one alarm brings
    back no newer replaced one; and an identifier is matched whole, so wo/1 keeps no entry that names only wo/12."""
    board = _synthetic_board(read_first=False, second=True)
    board.alarms = [{"first_t": t, "channel": "PS1"} for t in (5, 7, 9)]
    board.status["time"] = 10
    board.set_ranking(board.ranking)
    board.orders["1"] = {"id": "1", "component": "pump", "status": "open", "prediction_id": None}
    board.orders["12"] = {"id": "12", "component": "valve", "status": "open", "prediction_id": None}
    board.notes = ["the valve under wo/12", "a plain note"]
    board.mark_turn(1)
    entries = board.replaceable()
    record = board.apply_summary("The alarm alarm/7/PS1 moved us; the pump is under wo/1.", entries, 0)
    timeline = board._render_timeline()
    assert "alarm/7/PS1" in timeline and "alarm/5/PS1" not in timeline and "alarm/9/PS1" not in timeline
    assert board.notes == [] and record["kept_for_named_identifiers"] == {"timeline": 1}
    assert record["replaced"]["timeline"] == 2 and record["replaced"]["notes"] == 2
    assert board.replaceable() == []
    board.mark_turn(2)
    board.apply_summary("Still alarm/7/PS1.", board.replaceable(), 0)
    assert "alarm/7/PS1" in board._render_timeline()


def test_the_sentence_of_the_acting_kinds_holds_for_any_count_of_kinds():
    """No text a model reads may fail on a count the asset or the row brings: the sentence of the kinds that act names one
    kind, two, seven or thirteen as well, and the bytes for three, four and five kinds stay those of the old table."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import prompts
    old = {3: "Three", 4: "Four", 5: "Five"}
    for unserved in ((), ("declare",), ("declare", "wait")):
        kinds = [k for k in prompts.ACTING_KINDS if k not in unserved]
        assert prompts.acting_sentence(unserved) == f"{old[len(kinds)]} kinds act on the <NOUN>: {', '.join(kinds[:-1])} and {kinds[-1]}."
    assert prompts.acting_sentence(("read", "diagnose", "declare")) == "Two kinds act on the <NOUN>: order and wait."
    assert prompts.acting_sentence(("diagnose", "declare", "order", "wait")) == "One kind acts on the <NOUN>: read."
    seven = tuple(f"k{i}" for i in range(7))
    with mock.patch.object(prompts, "ACTING_KINDS", seven):
        assert prompts.acting_sentence(()) == "Seven kinds act on the <NOUN>: k0, k1, k2, k3, k4, k5 and k6."
    with mock.patch.object(prompts, "ACTING_KINDS", tuple(f"k{i}" for i in range(13))):
        assert prompts.acting_sentence(()).startswith("13 kinds act on the <NOUN>: k0, ")


def test_a_reply_without_json_runs_nothing_and_its_error_goes_to_the_next_turn(tmp_path):
    def prose_then_plan(prompt, call):
        # prose on the first decision turn alone; a later turn, after a read that the board advised, acts
        return "Let me think about the branches first." if call == 1 else act_now(prompt, call)
    ep_dir, row, backend = _play(tmp_path / "retry", ScriptedPlanner(prose_then_plan), {})
    verdict = _assert_episode_files(ep_dir, row)
    _assert_repairs_verified(ep_dir, verdict)
    plans = read_json(ep_dir / "plans.json")
    failed = [p for p in plans["plans"] if p["source"] == "no_plan_after_parse_failure"]
    assert len(failed) == 1 and failed[0]["parse_errors"] == ["no JSON object found in the reply"] and failed[0]["steps"] == []
    assert decision_plans(plans)[0]["source"] == "model" and row["fallbacks"] == 1
    assert any("could not be parsed as a plan: no JSON object found" in c["prompt"] for c in backend.calls)

    def prose(prompt, call):
        return "Still thinking."
    ep_dir, row, backend = _play(tmp_path / "idle", ScriptedPlanner(prose, orientation="No differential yet."), {}, seed=1)
    verdict = _assert_episode_files(ep_dir, row)
    plans = read_json(ep_dir / "plans.json")
    # every turn without a call takes the no_call step of the driver, so the step cap of the row ends the episode, with no count
    # of the controller's own; the differential is asked again in each planning turn with the parse error quoted
    opening, *failed = plans["plans"]
    assert [s["kind"] for s in opening["plan"]["steps"]] == ["profile", "failure_modes", "fix_history"] and opening["source"] == "model"
    max_steps = load_config("static")[1].max_steps
    opening_calls = sum(1 for t in read_json(ep_dir / "trajectory.json") if t["server"] == RIG_SERVER)  # the failure modes read one component a call
    assert [p["source"] for p in failed] == ["no_plan_after_parse_failure"] * (max_steps - opening_calls)
    assert all(p["steps"] == [] and p["plan"]["steps"] == [] and len(p["parse_errors"]) == 1 for p in failed)
    orientations = [c["prompt"] for c in backend.calls if ORIENTATION_HEADING in c["prompt"]]
    assert len(orientations) == len(failed) and all("could not be parsed as a differential" in o for o in orientations[1:])
    assert "could not be parsed as a differential" not in orientations[0]
    assert plans["differential"] == [] and row["fallbacks"] == plans["fallbacks"] == 2 * len(failed)
    assert "read_sensors" not in _tools(ep_dir) and verdict["end_reason"] == "max_steps"
    assert any("could not be parsed" in note for note in plans["board"]["notes"])


@pytest.mark.parametrize("config_name", ["static", "priced", "real_time"])
def test_the_batch_reserves_the_verifying_read_the_agent_reserves_whatever_it_was_told(config_name):
    """Told equals done: the price of the read that verifies a repair, which the reserve of the server's read batch holds,
    equals the one the agent's plan check holds (Board.verification_group) for every set of repaired components, whether the
    agent read no failure modes, those of every component or of one, or one answer of the repair manual. Pricing from every
    indicating channel of the profile instead, as the server did before, keeps less than the agent where it was told nothing."""
    import itertools
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import read_prices
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import verifying_read_price
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
    from CausalDiagnosis.utils.public_rules.repair_completion import verifies
    _facts, cycles, blocks = load_world()
    _name, config = load_config(config_name)
    told = {"nothing": [], "every mode": [{"type": "get_failure_modes", "component": ""}],
            "one mode": [{"type": "get_failure_modes", "component": COMPONENTS[0]}], "manual": [{"type": "lookup_fix", "component": COMPONENTS[1]}]}
    subsets = [list(s) for k in range(1, len(COMPONENTS) + 1) for s in itertools.combinations(COMPONENTS, k)]
    for name, actions in told.items():
        env = HydraulicRigEnv(cycles, blocks, config)
        env.reset(0)
        board = Board()
        board.set_profile(env.step({"type": "get_asset_profile"})[0]["result"])
        for action in actions:
            result = env.step(action)[0]["result"]
            board.add_failure_modes(result["failure_modes"]) if action["type"] == "get_failure_modes" else board.add_manual(result)
        released = verifier.released_from_env(env)
        view = profile_view(dict(released["rules"], asset_id=config.asset_id))
        prices = read_prices(board.profile)
        for components in subsets:
            agent = prices[board.verification_group(components)]
            assert verifying_read_price(view, components, released["events"]) == pytest.approx(agent), (name, components)
        if name == "nothing":
            whole = view.told_indicating_channels([{"event": "failure_modes", "component": ""}])  # the profile-wide pricing of before
            assert any(min(prices[g] for g, channels in view.sensor_groups.items() if verifies(channels, components, whole, view.sensor_groups))
                       < prices[board.verification_group(components)] for components in subsets)


@pytest.mark.parametrize("config_name", ["static", "priced", "real_time"])
def test_the_batch_reserves_the_verifying_read_of_every_repair_the_agent_waits_on(config_name):
    """Told equals done: the agent's plan check verifies the components of its open orders and of its
    closed repairs whose verifying reading is still to come beside the new ones (Board.repaired), and the reserve of the server's
    batch prices the read over the same set (adapter.repaired_components, from the released events). Equal while two orders are
    open, after both closed, and after a reading that verifies one of the two repairs; pricing the new components alone, as
    before, keeps less."""
    import itertools
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import read_prices
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import repaired_components, verifying_read_price
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
    _facts, cycles, blocks = load_world()
    _name, config = load_config(config_name)
    env = HydraulicRigEnv(cycles, blocks, config)
    env.reset(0)
    board = Board()
    board.set_profile(env.step({"type": "get_asset_profile"})[0]["result"])
    board.add_failure_modes(env.step({"type": "get_failure_modes", "component": ""})[0]["result"]["failure_modes"])
    repaired_two = COMPONENTS[:2]
    diagnosis = {c: (next(level for level in config.levels[c] if level != config.healthy[c]) if c in repaired_two else config.healthy[c]) for c in COMPONENTS}
    assert "refused" not in env.step({"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": []})[0]
    for component, prediction_id in zip(repaired_two, ("pred/1", "pred/2")):
        board.add_order(env.step({"type": "create_work_order", "component": component, "prediction_id": prediction_id, "evidence_ids": []})[0]["result"],
                        prediction_id)
    new_sets = [list(s) for k in range(len(COMPONENTS) + 1) for s in itertools.combinations(COMPONENTS, k)]
    shorter = []

    def same_reserve(state):
        released = verifier.released_from_env(env)
        view = profile_view(dict(released["rules"], asset_id=config.asset_id))
        prices = read_prices(board.profile)
        for new in new_sets:
            agent = sorted(board.repaired(new))
            assert repaired_components(view, released["events"], new) == agent, (state, new)
            price = verifying_read_price(view, agent, released["events"])
            assert price == pytest.approx(prices[board.verification_group(agent)] if agent else 0.0), (state, new)
            shorter.append(verifying_read_price(view, new, released["events"]) < price)

    same_reserve("open")
    while board.open_orders() and not env.done:
        env.step({"type": "wait_for_notification", "timeout": 1})
        board.add_changes({"work_orders_closed": [{"id": e["id"], "closed_t": e["t_end"]} for e in env.log if e["event"] == "work_order_closed"]})
    same_reserve("closed")
    group = next(g for g in board.profile["sensor_groups"] if board.verifies(g, repaired_two[:1]) and not board.verifies(g, repaired_two[1:]))
    served = env.step({"type": "read_sensors", "sensor_group": group, "cycles": 1})[0]["result"]["served"]
    board.record_read({"sensor_group": group, "t_last": max(s["t"] for s in served)})
    for key, closed_t in board.pending_verifications():  # the verdict the controller takes on the verifying reading (_verify_ready)
        if max(s["t"] for s in served) >= closed_t and board.verifies(group, [o["component"] for o in board.orders.values() if o["prediction_id"] == key]):
            board.verifications[key] = {"status": "verified"}
    same_reserve("one verified")
    assert board.repaired() == sorted(repaired_two[1:]) and any(shorter)


@pytest.mark.parametrize("config_name", ["static", "priced", "real_time"])
def test_the_rule_that_follows_the_reads_of_the_server_passes_the_plan_check(tmp_path, config_name):
    """Told equals done: the rule plans on the reads that the CausalDiagnosis MCP names, which no count bounds, and the repair it
    reserves, and the plan check of the agent refuses none of its plans on the preset."""
    planner = ScriptedPlanner(lambda prompt, call: pytest.fail("the rule only controller must not ask for a plan"))
    ep_dir, _row, _backend = _play(tmp_path, planner, {"rule_only": True}, config_name=config_name)
    plans = read_json(ep_dir / "plans.json")["plans"]
    assert plans and all(p["source"] == "rule_only" and not p["check_problems"] for p in plans)
    assert any(step["kind"] == "read" for p in plans for step in p["plan"]["steps"])


def test_the_rule_only_test_tool_plans_without_the_model_and_the_model_writes_the_summary_only(tmp_path):
    """The rule only controller left the experiments and stays an internal test tool: the rule plans every
    turn, the read that verifies its repair included, and the model writes the closing summary alone."""
    planner = ScriptedPlanner(lambda prompt, call: pytest.fail("the rule only controller must not ask for a plan"))
    ep_dir, row, backend = _play(tmp_path, planner, {"rule_only": True})
    verdict = _assert_episode_files(ep_dir, row)
    _assert_repairs_verified(ep_dir, verdict)
    plans = read_json(ep_dir / "plans.json")
    assert row["llm_calls"] == 1 and SUMMARY_HEADING in backend.calls[0]["prompt"]
    assert all(p["source"] == "rule_only" for p in plans["plans"]) and plans["differential"] == [] and row["fallbacks"] == 0
    assert row["first_plan_mode"] in ("test_first", "act_now") and row["agent"] == "causal_rule_scripted-causal"
    assert all(p["plan"]["why"].startswith("rule:") for p in plans["plans"])
    acting = next(p for p in plans["plans"] if any(s["kind"] == "order" for s in p["plan"]["steps"]))
    assert [s["kind"] for s in acting["plan"]["steps"]][-2:] == ["read", "finish"]


@pytest.mark.skipif("wait_for_notification" not in contract_tools(RIG_SERVER), reason="the real action server lacks the waiting tool")
def test_the_driver_runs_the_causal_agent_through_the_openai_compatible_client_with_the_switches(tmp_path, monkeypatch):
    planner = ScriptedPlanner(act_now)
    server, requests = _planner_endpoint(planner)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        with pytest.raises(SystemExit):  # the rule only controller left the driver
            run_episodes.build_parser().parse_args(["--agent", "causal", "--causal-rule-only"])
        run_episodes.main(["--config", "static", "--agent", "causal", "--causal-no-expectations", "--model", "vendor/stand-in-1.0",
                           "--base-url", base, "--seeds", "1", "--out", str(tmp_path / "run")])
        run_dir = next(p for p in (tmp_path / "run").iterdir() if p.is_dir())
        meta = read_json(run_dir / "run.json")
        import csv
        with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert "causal_noexp" in run_dir.name and meta["agent"] == "causal" and meta["agent_label"] == "causal_noexp_vendor/stand-in-1.0"
        assert "thresholds" not in meta
        assert meta["causal_options"] == {"branches": True, "single_hypothesis": False, "plan_length": None, "expectations": False,
                                          "reads": "gain", "backtracking": True, "expectation_tolerance": 0.05, "parallel": True}
        assert meta["servers"] == [RIG_SERVER, CAUSAL_SERVER] and meta["model"] == "vendor/stand-in-1.0"
        assert len(rows) == 1 and rows[0]["error"] == "" and rows[0]["agent"] == "causal_noexp_vendor/stand-in-1.0" and rows[0]["model"] == "vendor/stand-in-1.0"
        # every model call of the episode went through the endpoint: the orientation, every planning turn and the summary
        assert int(rows[0]["llm_calls"]) == len(requests) >= 3 and int(rows[0]["plans"]) >= 1
        ep_dir = run_dir / "seed000000"
        assert (ep_dir / "answer.txt").read_text(encoding="utf-8") == SUMMARY_REPLY
        assert "| plans.json | 1 |" in (run_dir / "summary.md").read_text(encoding="utf-8")
    finally:
        server.shutdown()


@pytest.mark.skipif("wait_for_notification" not in contract_tools(RIG_SERVER), reason="the real action server lacks the waiting tool")
def test_the_causal_agent_holds_one_model_request_open_and_every_failed_attempt_reaches_the_ledger(tmp_path, monkeypatch):
    """The guard of the failed attempts for the causal design: the endpoint fails the first attempt of every request, the
    client's one retry policy attempts each again, the episode ends without an error, no request opens while a failed span of
    the episode is open (from a failed attempt to the next attempt of the same request), and the ledger of failed spans holds
    one line per failed attempt, so that the thinking clock can credit each."""
    from CausalDiagnosis.utils import model_request_retry as retry
    from CausalDiagnosis.utils import model_failures

    monkeypatch.setattr(retry, "_wait", lambda deadline, delay_s: 0.0)  # the registered wait, not its length, is under test
    planner = ScriptedPlanner(act_now)
    server, requests = _planner_endpoint(planner, fail_first=True)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url", base, "--seeds", "1",
                           "--out", str(tmp_path / "run")])
        run_dir = next(p for p in (tmp_path / "run").iterdir() if p.is_dir())
        import csv
        with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and rows[0]["error"] == "", rows
        assert int(rows[0]["llm_calls"]) == len(requests) >= 3 and server.stats["failed"] >= 3
        arrivals = server.stats["arrivals"]
        assert server.stats["most_open"] == 1
        for key in set(arrivals):  # each request failed once, so its retry is the very next arrival
            first = arrivals.index(key)
            assert arrivals[first + 1] == key, "a model request opened inside the failed span of another"
        lines = (run_dir / "seed000000" / model_failures.NAME).read_text(encoding="utf-8").splitlines()
        assert len(lines) == server.stats["failed"]
        entries = [json.loads(line) for line in lines]
        assert all(e["status"] == 503 and e["next_start"] >= e["start"] for e in entries), entries
        spans, cursor = model_failures.failed_spans_since(run_dir / "seed000000", 0)
        assert len(spans) == len(entries) and model_failures.failed_spans_since(run_dir / "seed000000", cursor) == ([], cursor)
    finally:
        server.shutdown()


@pytest.mark.skipif("wait_for_notification" not in contract_tools(RIG_SERVER), reason="the real action server lacks the waiting tool")
def test_the_causal_agent_keeps_one_model_request_open_at_a_time_and_the_endpoint_sees_two_when_two_are_open(tmp_path, monkeypatch):
    """The causal agent with its parallel reads plays one episode through the driver against an endpoint that holds every
    completion open, so two requests of the episode in flight at once would overlap there. The control holds two completions
    open from two threads and the same counter sees both, so the count of one is the agent's and not a blind counter. The rule
    is one request open at the client: a resend of post_with_retries after an attempt timed out at the client, when a credit has
    moved the deadline, can leave the abandoned attempt open at the engine for a moment, which this counter does not reach."""
    server, requests = _planner_endpoint(ScriptedPlanner(act_now), hold=0.2)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url", base,
                           "--seeds", "1", "--out", str(tmp_path / "run")])
        assert len(requests) >= 3 and server.open_requests["most"] == 1
        control, _control_requests = _planner_endpoint(ScriptedPlanner(act_now), hold=0.5)
        control_base = f"http://127.0.0.1:{control.server_address[1]}/v1/"
        try:
            both = [threading.Thread(target=_post_completion, args=(control_base, ORIENTATION_HEADING)) for _ in range(2)]
            for thread in both:
                thread.start()
            for thread in both:
                thread.join()
            assert control.open_requests["most"] == 2
        finally:
            control.shutdown()
    finally:
        server.shutdown()


# ----- the parser, the plan check and the rule plan without servers -----

def _synthetic_board(read_first=True, branches=True, second=False):
    """A board after one reading with two ranked candidates and the preview of rank 1 (a pump fault). second makes rank 2 a
    valve fault and adds its preview, whose repair plan (pred/b2) costs 60 to a restored rig acting now against the pump's
    72, more than the tolerance of the cost choice below it."""
    board = Board()
    board.set_profile({
        "asset_id": "hyd_1", "components": list(COMPONENTS), "levels": {"cooler": [3, 20, 100], "valve": [73, 80, 90, 100], "pump": [0, 1, 2],
                                                                        "accumulator": [90, 100, 115, 130]},
        "healthy": dict(HEALTHY), "sensor_groups": {"tier1": ["CE_CoolingEfficiency_percent_1Hz"], "tier2": ["PS1_Pressure_bar_100Hz"], "tier3": ["FS1_VolumeFlow_l_per_min_10Hz"]},
        "prices": {"read": {"tier1": 1, "tier2": 3, "tier3": 10}, "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20}, "repair_charge": 20,
                   "visit": 15},
        "durations": {"read_per_cycle": 1, "advance_per_cycle": 1, "lead_time": {c: 3 for c in COMPONENTS}, "repair_downtime": {"cooler": 1, "valve": 1, "pump": 2, "accumulator": 1}},
        "downtime_rate": {}, "budget": 300, "horizon": 60, "max_steps": 60, "fault_rule": {"min_faulty": 1, "max_faulty": 2}, "clock_mode": "declared",
        "statistics": ["mean", "std"], "normal_ranges": {"CE_CoolingEfficiency_percent_1Hz_mean": {"low": 40.0, "high": 50.0}}})
    board.status = {"time": 1, "budget_left": 299, "steps_left": 57, "open_work_orders": [], "episode_over": False}
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 1, "t": 0, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 30.0, "CE_CoolingEfficiency_percent_1Hz_std": 0.1}}]})
    candidates = [
        {"rank": 1, "combination": {"cooler": 100, "valve": 100, "pump": 1, "accumulator": 130}, "probability": 0.6,
         "faults": [{"component": "pump", "level": 1, "code": "pump_internal_leakage_weak"}], "evidence_ids": ["obs/1/CE_CoolingEfficiency_percent_1Hz_mean"]},
        {"rank": 2, "combination": {"cooler": 100, "valve": 100, "pump": 2, "accumulator": 130}, "probability": 0.4,
         "faults": [{"component": "pump", "level": 2, "code": "pump_internal_leakage_severe"}], "evidence_ids": []},
    ]
    if second:
        candidates[1] = {"rank": 2, "combination": {"cooler": 100, "valve": 90, "pump": 0, "accumulator": 130}, "probability": 0.4,
                         "faults": [{"component": "valve", "level": 90, "code": "valve_switching_small_lag"}], "evidence_ids": []}
    ranking = {"ranking_model": "joint_block", "n_candidates": 47, "candidates": candidates, "read_first": read_first,
               "separating_test": {"sensor_group": "tier2", "cycles": 1, "price": 3, "duration": 1, "cost": 4, "expected_loss": 8} if read_first else None,
               "expected_loss_act": 12, "expected_saving": 4 if read_first else 0, "expected_loss_test": {"tier1": 11, "tier2": 8, "tier3": 9},
               "unseparated": [], "separating_channels": {}}
    if branches:
        ranking["branches"] = [{
            "rank": 1, "combination": candidates[0]["combination"], "faults": candidates[0]["faults"], "probability": 0.6, "faulty_set": ["pump"],
            "confirm_test": {"sensor_group": "tier3", "cycles": 1, "price": 10, "duration": 1, "cost": 11, "mass_if_true": 0.9, "mass_if_false": 0.2, "gap": 0.7, "standard_error": 0.01},
            "two_step": None,
            "repair_plan": {"steps": [{"component": "pump", "part_price": 30, "repair_charge": 20, "lead_time": 3, "repair_downtime": 2, "downtime_cost": 10, "prediction_id": "pred/b1"}],
                            "plan_prediction_id": "pred/b1", "price": 50, "downtime_cost": 10, "cycles_to_restored": 5},
            "predicted_outcome": {"predicted": [], "alpha": 0.05, "recovery_if_acted": 0.6, "restored_if_true": True},
            "expected_loss_if_acted": 12, "expected_loss_if_tested": 8, "cost_to_restored": {"act_now": 72, "test_first": 79}}]
        if second:
            ranking["branches"].append({
                "rank": 2, "combination": candidates[1]["combination"], "faults": candidates[1]["faults"], "probability": 0.4, "faulty_set": ["valve"],
                "confirm_test": {"sensor_group": "tier3", "cycles": 1, "price": 10, "duration": 1, "cost": 11, "mass_if_true": 0.8, "mass_if_false": 0.3, "gap": 0.5, "standard_error": 0.01},
                "two_step": None,
                "repair_plan": {"steps": [{"component": "valve", "part_price": 20, "repair_charge": 20, "lead_time": 3, "repair_downtime": 1, "downtime_cost": 8, "prediction_id": "pred/b2"}],
                                "plan_prediction_id": "pred/b2", "price": 40, "downtime_cost": 8, "cycles_to_restored": 5},
                "predicted_outcome": {"predicted": [], "alpha": 0.05, "recovery_if_acted": 0.4, "restored_if_true": True},
                "expected_loss_if_acted": 12, "expected_loss_if_tested": 20, "cost_to_restored": {"act_now": 60, "test_first": 68}})
    board.set_ranking(ranking)
    # the board knows, as answers of the repair manual would tell it, that the channel of every group indicates every component, so
    # that a read of any group verifies a repair, as the tests of this board assume (without a known channel only a
    # group of every channel verifies, and this board has none)
    board.manual_channels = {c: {f for fields in board.profile["sensor_groups"].values() for f in fields} for c in COMPONENTS}
    return board


def test_the_parser_takes_the_last_json_object_and_names_schema_errors():
    text = 'First {"branch": 2, "mode": "declare", "steps": [{"kind": "finish"}]} then the real one:\n{"branch": "1", "mode": "act_now", "why": "x", "steps": [{"kind": "read", "args": {"sensor_group": "tier2", "cycles": "2"}, "expect": {"top_rank": 1, "top_mass_at_least": "0.8"}}]}'
    plan = parse_plan(text, 6)
    assert plan.branch == 1 and plan.mode == "act_now" and plan.steps[0].args == {"sensor_group": "tier2", "cycles": 2}
    assert plan.steps[0].expect == {"top_rank": 1, "top_mass_at_least": 0.8}
    with pytest.raises(PlanError, match="no JSON object"):
        parse_plan("no plan here", 6)
    with pytest.raises(PlanError, match="mode must be one of"):
        parse_plan('{"steps": [{"kind": "finish"}]}', 6)
    with pytest.raises(PlanError, match="at most 1 are allowed"):
        parse_plan('{"mode": "act_now", "steps": [{"kind": "finish"}, {"kind": "finish"}]}', 1)
    with pytest.raises(PlanError, match="finish must be the last step"):
        parse_plan('{"mode": "act_now", "steps": [{"kind": "finish"}, {"kind": "advance", "args": {"cycles": 1}}]}', 6)
    with pytest.raises(PlanError, match="kind must be one of"):
        parse_plan('{"mode": "act_now", "steps": [{"kind": "repair"}]}', 6)
    # a wait of 0 cycles asks for the next event, so it reads as 1; a negative wait is still refused
    assert parse_plan('{"mode": "act_now", "steps": [{"kind": "wait", "args": {"cycles": 0}}]}', 6).steps[0].args == {"cycles": 1}
    with pytest.raises(PlanError, match="cycles must be an integer of at least 0"):
        parse_plan('{"mode": "act_now", "steps": [{"kind": "wait", "args": {"cycles": -1}}]}', 6)
    # the old step name advance is still read as a wait
    assert parse_plan('{"mode": "act_now", "steps": [{"kind": "advance", "args": {"cycles": 2}}]}', 6).steps[0].kind == "wait"
    default = parse_plan('{"mode": "repair_as_test", "steps": [{"kind": "order", "args": {"component": "pump"}}]}', 6)
    assert default.steps[0].args["prediction_id"] == ESTIMATE and default.steps[0].expect == {"prediction_id": ESTIMATE}


def test_the_plan_check_enforces_the_guards_and_the_rule_plan_follows_the_decision():
    board = _synthetic_board()

    def plan(*steps, branch=1, mode="act_now"):
        return Plan(branch=branch, mode=mode, why="", steps=list(steps))
    order = Step("order", {"component": "pump", "prediction_id": "pred/b1"}, {"prediction_id": "pred/b1"})
    problems = check_plan(plan(order, Step("finish")), board)
    assert any("must precede every work order" in p for p in problems) and any("finish precedes the close" in p for p in problems)
    # the check keeps no rule that comes from the server: an order may cite a prediction the board does not know,
    # which the controller replaces by an estimate, and a declaration may name any sensor group after a reading
    unknown = Step("order", {"component": "pump", "prediction_id": "pred/9"}, {"prediction_id": "pred/9"})
    assert check_plan(plan(Step("diagnose", {"rank": 1}, {"top_rank": 1}), unknown), board) == []
    for group in ("tier1", "tier2", "tier3"):
        assert check_plan(plan(Step("declare", {"sensor_group": group, "cycles": 1}), mode="declare"), board) == [], group
    assert any("unknown sensor group 'tier9'" in p for p in check_plan(plan(Step("declare", {"sensor_group": "tier9", "cycles": 1}), mode="declare"), board))
    good = plan(Step("diagnose", {"rank": 1}, {"top_rank": 1}), order, Step("wait", {"cycles": 5}), Step("read", {"sensor_group": "tier1", "cycles": 1}),
                Step("finish"))
    # the evidence guard is advice: while the ranking reports insufficient evidence and a read fits, the board advises
    # the reads before a diagnosis, and a plan that diagnoses first passes the check
    assert check_plan(good, board) == [] and check_plan(good, _synthetic_board(read_first=False)) == []
    assert advice(board)[0].startswith("Advised reads, the separating test of E10: tier2 x1.")
    assert not [line for line in advice(_synthetic_board(read_first=False)) if line.startswith("Advised reads")]
    wrong = plan(Step("read", {"sensor_group": "tier1", "cycles": 1}), Step("diagnose", {"rank": 1}, {"top_rank": 1}), mode="test_first")
    assert check_plan(wrong, board) == [] and check_plan(rule_plan(board, 6), board) == []
    expensive = plan(Step("read", {"sensor_group": "tier3", "cycles": 40}))
    assert any("exceed the budget left" in p for p in check_plan(expensive, board))
    late = plan(Step("wait", {"cycles": 59}), Step("finish"))
    assert any("ends at the horizon" in p for p in check_plan(late, board))
    assert any("not among the 2 ranked candidates" in p for p in check_plan(plan(Step("diagnose", {"rank": 3}, {"top_rank": 3})), board))
    # the rule reads the separating test while the evidence is insufficient and affordable, else acts on the top candidate
    reading = rule_plan(board, 6)
    assert reading.mode == "test_first" and [s.kind for s in reading.steps] == ["read"] and reading.steps[0].args["sensor_group"] == "tier2"
    assert reading.steps[0].evidence == ["obs/1/CE_CoolingEfficiency_percent_1Hz_mean"]  # the one channel mean outside its normal range
    acting = rule_plan(_synthetic_board(read_first=False), 6)
    assert [s.kind for s in acting.steps] == ["diagnose", "order", "wait", "read", "finish"] and acting.steps[1].args["prediction_id"] == "pred/b1"
    assert acting.steps[2].args["cycles"] == 5 and check_plan(acting, _synthetic_board(read_first=False)) == []
    # a finish follows a read at or after the last close where a cycle runs after a close, since the controller verifies a repair
    # only on a reading a plan names, and a row on which no cycle runs after a close needs none
    unread = plan(Step("diagnose", {"rank": 1}, {"top_rank": 1}), order, Step("wait", {"cycles": 5}), Step("finish"))
    refusal = [p for p in check_plan(unread, _synthetic_board(read_first=False)) if "verifies the repair" in p]
    assert len(refusal) == 1 and "read tier1 for 1 cycle at or after cycle 6" in refusal[0]
    no_clock = _synthetic_board(read_first=False)
    no_clock.clock_runs = False
    # where no cycle runs after a close no read verifies a repair, and no wait is possible either
    assert not [p for p in check_plan(unread, no_clock) if "verifies the repair" in p]
    assert "step 3 (wait): no clock runs on this row, so no wait is possible" in check_plan(unread, no_clock)
    assert [s.kind for s in rule_plan(_synthetic_board(read_first=False), 1).steps] == ["diagnose"]
    without = rule_plan(_synthetic_board(read_first=False, branches=False), 6)
    assert without.steps[1].args["prediction_id"] == ESTIMATE and _synthetic_board(branches=False).branches() == []
    # the board hides the troubleshooting plans under the ablation switches, and with them their prediction identifiers
    hidden = Board(show_branches=False)
    hidden.set_profile(board.profile)
    hidden.set_ranking({**board.ranking, "branches": [{"rank": 1, "repair_plan": {"steps": [{"component": "pump", "prediction_id": "pred/b1"}]}}]})
    assert hidden.branches() == [] and hidden.predictions == {} and "Troubleshooting plans" not in hidden.render()
    assert "Troubleshooting plans" in board.render() and "pred/b1" in board.render() and board.ranking_is_current()
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import branch_prediction_id
    assert branch_prediction_id(board.branch(1), "pump") == "pred/b1" and branch_prediction_id(board.branch(1), "valve") == "pred/b1"
    # a plan of two steps: every order cites the plan's prediction, the steps' own stay uncited and unknown to the check
    two = Board()
    two.set_profile(board.profile)
    two.set_ranking({**board.ranking, "branches": [{"rank": 1, "faulty_set": ["pump", "valve"], "repair_plan": {
        "steps": [{"component": "pump", "prediction_id": "pred/b2"}, {"component": "valve", "prediction_id": "pred/b3"}], "plan_prediction_id": "pred/b4"},
        "predicted_outcome": {"recovery_if_acted": 0.7}}]})
    assert branch_prediction_id(two.branch(1), "pump") == branch_prediction_id(two.branch(1), "valve") == "pred/b4"
    assert set(two.predictions) == {"pred/b4"} and two.predictions["pred/b4"]["action"] == "replace:pump+valve"
    assert "every order of the plan cites pred/b4" in two.render() and "pred/b2" not in two.render()
    # the server keys a branch by its faulty set and lists every candidate of the set in ranks, so each rank reads one plan
    keyed = Board()
    keyed.set_profile(board.profile)
    keyed.set_ranking({**board.ranking, "branches": [{"rank": 1, "ranks": [1, 3], "faulty_set": ["pump"], "repair_plan": {
        "steps": [{"component": "pump", "prediction_id": "pred/b5"}], "plan_prediction_id": "pred/b5"}}]})
    assert keyed.branch(1) is keyed.branch(3) is not None and keyed.branch(2) is None
    assert branch_prediction_id(keyed.branch(3), "pump") == "pred/b5"
    ordered = rule_plan(_synthetic_board(read_first=False), 6).steps[1]
    assert ordered.args["prediction_id"] == "pred/b1"
    # a plan's prediction is pending once every order citing it has closed, and a not yet observable verdict keeps it pending
    two.add_order({"work_order": {"id": "wo-1", "component": "pump", "created_t": 1, "expected_close_t": 6, "status": "open"}}, "pred/b4")
    two.add_order({"work_order": {"id": "wo-2", "component": "valve", "created_t": 1, "expected_close_t": 5, "status": "open"}}, "pred/b4")
    two.add_changes({"alarms": [], "work_orders_closed": [{"id": "wo-2", "component": "valve", "source": "agent", "closed_t": 5}]})
    assert two.pending_verifications() == []
    two.add_changes({"alarms": [], "work_orders_closed": [{"id": "wo-1", "component": "pump", "source": "agent", "closed_t": 6}]})
    assert two.pending_verifications() == [("pred/b4", 6)]
    two.add_verification({"prediction_id": "pred/b4", "status": "not yet observable: no reading since the action", "consistent": None})
    assert two.pending_verifications() == [("pred/b4", 6)]
    two.add_verification({"prediction_id": "pred/b4", "status": "verified", "consistent": True})
    assert two.pending_verifications() == []
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 2, "t": 1, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 45.0}}]})
    assert not board.ranking_is_current() and "STALE" in board.render()


def test_a_verified_prediction_names_the_most_probable_combination_after_the_repair_and_its_mass():
    """verify_outcome scores the reading after the repair against the most probable combination of the belief
    after it and names that combination with its mass, which the row of the prediction shows beside the recovery posterior."""
    board = _synthetic_board(read_first=False)

    def row(prediction_id):
        section = _section(board.render(), "## Predictions")
        return next(line for line in section.splitlines() if line.startswith(f"| {prediction_id} |"))
    assert "not verified" in row("pred/b1") and "predicted combination" not in row("pred/b1")
    board.add_verification({"prediction_id": "pred/b1", "status": "verified", "consistent": True, "recovery_predicted": 0.6,
                            "recovery_posterior": 0.93, "predicted_combination": {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130},
                            "predicted_mass": 0.912, "recovery_status": "", "message": "pred/b1 (replace:pump): verified"})
    assert ("posterior after the repair 0.930, predicted combination (cooler 100, valve 100, pump 0, accumulator 130), the most probable "
            "after the repair, with mass 0.912") in row("pred/b1")


def test_the_closing_summary_names_the_most_probable_combination_after_the_repair_and_its_mass():
    """The closing summary reports each verdict with the combination the reading was scored against and its
    mass, as the row of the prediction does."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import summary_turn
    board = _synthetic_board(read_first=False)
    board.add_order({"work_order": {"id": "wo-1", "component": "pump", "created_t": 1, "expected_close_t": 6, "status": "open"}}, "pred/b1")
    board.add_changes({"alarms": [], "work_orders_closed": [{"id": "wo-1", "component": "pump", "source": "agent", "closed_t": 6}]})
    board.add_verification({"prediction_id": "pred/b1", "status": "verified", "consistent": True, "recovery_predicted": 0.6,
                            "recovery_posterior": 0.93, "predicted_combination": {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130},
                            "predicted_mass": 0.912, "recovery_status": "", "message": "pred/b1 (replace:pump): verified"})
    assert ("posterior of the healthy combination after the repair 0.930, predicted combination (cooler 100, valve 100, pump 0, accumulator 130), "
            "the most probable after the repair, with mass 0.912.") in summary_turn(board, 6)


def test_consecutive_planning_requests_share_every_part_that_held_and_the_status_comes_last(tmp_path):
    """The board in the order of how often a part changes: the terms, the fix history and the
    failure modes first, then the evidence, whose heading carries no count, then the rest, and the status last. Over a run that
    reads one group per turn, each planning request repeats the one before it up to the end of that request's last evidence
    row, so the prefix cache of the serving engine computes the old evidence once."""
    groups = ("tier1", "tier2", "tier3")

    def plans(prompt, call, truth):
        start = {c: v for c, v in truth().items() if v != HEALTHY[c]}
        candidates = [{"levels": start, "failure_mode_ids": []}]
        steps = ([{"kind": "read", "args": {"sensor_group": groups[call % 3], "cycles": 1}, "expect": {}, "evidence": []}] if call <= 4
                 else [{"kind": "finish", "args": {}, "expect": {}, "evidence": []}])
        return json.dumps({"branch": 1, "mode": "test_first", "why": "scripted", "candidates": candidates, "steps": steps})
    _ep_dir, _row, backend = _play_without_server(tmp_path, plans)
    prompts = [p for p in backend.inner.prompts if PLANNING_HEADING in p and "## Evidence" in p and "| obs/" in p]
    assert len(prompts) >= 3
    for before, after in zip(prompts, prompts[1:]):
        shared = next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), min(len(before), len(after)))
        last_row = before.rindex("| obs/")
        assert shared > before.index("\n", last_row), (shared, last_row)
    board = prompts[-1]
    assert board.index("## Failure modes") < board.index("## Evidence (the normal range applies") < board.index("## Status")
    assert "values from" not in board.split("## Evidence", 1)[1].split("\n", 1)[0]
    assert board.index("## Status") > board.index("## Last plan") and "Evidence so far: " in board


def test_the_rule_acts_on_the_cheapest_branch_and_the_advice_without_the_server_names_rank_one():
    """The rule of E10, an internal test tool, acts on the previewed branch of the cheapest act now cost. The advice of the board
    has no cost rule: without the server's advice it names rank 1, and the cost choice, its switch and its flag
    are gone."""
    def plan(*steps, branch=1, mode="act_now"):
        return Plan(branch=branch, mode=mode, why="", steps=list(steps))

    def act(rank, component, prediction):
        order = Step("order", {"component": component, "prediction_id": prediction}, {"prediction_id": prediction})
        return plan(Step("diagnose", {"rank": rank}, {"top_rank": 1}), order, Step("wait", {"cycles": 5}), Step("read", {"sensor_group": "tier3", "cycles": 1}),
                    Step("finish"), branch=rank)
    board = _synthetic_board(read_first=False, second=True)
    # the options are the act now costs of the previewed branches, cheapest first; acting on rank 1 at 72 lies more than 10
    # percent above rank 2 at 60, so the rule acts on rank 2 and a plan acting on rank 1 is refused with the cheapest options
    assert act_options(board) == [(60, 2), (72, 1)]
    acting = rule_plan(board, 6)
    assert acting.branch == 2 and acting.mode == "act_now" and [s.kind for s in acting.steps] == ["diagnose", "order", "wait", "read", "finish"]
    assert acting.steps[0].args == {"rank": 2} and acting.steps[0].expect == {"top_rank": 1}  # rank 1 stays on top until a reading
    assert acting.steps[1].args == {"component": "valve", "prediction_id": "pred/b2"} and check_plan(acting, board) == []
    # the board advises rank 1 without the server's advice, and a plan acting on either rank passes the check
    assert "Advised repair plan, with no troubleshooting plan shown: act on rank 1, the top candidate." in advice(board)
    assert not any("cost choice" in line or "percent" in line for line in advice(board))
    assert check_plan(act(1, "pump", "pred/b1"), board) == [] and check_plan(act(2, "valve", "pred/b2"), board) == []
    # branches of equal cost go to the higher posterior rank
    board.branch(1)["cost_to_restored"]["act_now"] = 65
    assert check_plan(act(1, "pump", "pred/b1"), board) == []
    board.branch(1)["cost_to_restored"]["act_now"] = 60
    assert act_options(board) == [(60, 1), (60, 2)] and rule_plan(board, 6).steps[0].args == {"rank": 1}
    board.branch(1)["cost_to_restored"]["act_now"] = 72
    # an act_now plan diagnoses the branch it follows, and a read of the branch's confirm test makes the diagnosis a test
    mismatch = plan(Step("diagnose", {"rank": 1}, {"top_rank": 1}), branch=2)
    assert check_plan(mismatch, board) == []
    tested = plan(Step("read", {"sensor_group": "tier3", "cycles": 1}), Step("diagnose", {"rank": 1}, {"top_rank": 1}), mode="test_first")
    assert check_plan(tested, board) == []
    # while the evidence is insufficient the guard decides first: the rule reads the separating test and a plan that
    # diagnoses before it is refused, even on the cheapest branch
    guarded = _synthetic_board(read_first=True, second=True)
    guarded.parallel = False  # the rule without the schedule of candidate plans, since both plans share the confirm test tier3, which a schedule adds
    reading = rule_plan(guarded, 6)
    assert reading.mode == "test_first" and [s.kind for s in reading.steps] == ["read"] and reading.steps[0].args["sensor_group"] == "tier2"
    assert check_plan(act(2, "valve", "pred/b2"), guarded) == [] and advice(guarded)[0].startswith("Advised reads")
    # after the diagnosis of rank 2 the rule cites branch 2's prediction for what is left to order
    board.diagnosis = {"diagnosis": dict(board.candidate(2)["combination"]), "evidence_ids": []}
    assert act_options(board) == []
    after = rule_plan(board, 6)
    assert after.branch == 2 and after.steps[0].args == {"component": "valve", "prediction_id": "pred/b2"}
    board.diagnosis = None
    assert act_options(_synthetic_board(read_first=False, second=True, branches=False)) == []
    # the cost choice is gone: no option, no label suffix, no flag, and one form of the prompt, where a diagnosis expects the candidate on top
    assert not hasattr(CausalLoopOptions(), "cost_choice") and CausalLoopOptions().label() == "causal"
    assert "cost_choice" not in run_episodes.causal_options(run_episodes.build_parser().parse_args(["--agent", "causal"]))
    with pytest.raises(SystemExit):
        run_episodes.build_parser().parse_args(["--agent", "causal", "--causal-rank-choice"])
    prompt = planning_turn(board, 6)
    assert REPAIR_RULE in prompt and "<the same rank>" not in prompt and "10 percent" not in prompt


# ----- version 7: read batches, backtracking, one visit per plan, compaction, the reserve and the switches -----

def test_a_plan_that_declares_before_any_reading_is_refused_and_a_read_before_it_lifts_the_refusal():
    board = _synthetic_board()
    board.evidence, board.ranking["evidence_ids"] = [], []
    board.absorbed_ids.clear()
    for candidate in board.ranking["candidates"]:
        candidate["evidence_ids"] = []
    declare = Step("declare", {"sensor_group": "tier2", "cycles": 1})
    refusal = "a declaration counts only when it cites a reading"
    assert any(refusal in p for p in check_plan(Plan(branch=1, mode="declare", why="", steps=[declare]), board))
    read = Step("read", {"sensor_group": "tier1", "cycles": 1}, {"top_rank": 1})
    assert not any(refusal in p for p in check_plan(Plan(branch=1, mode="declare", why="", steps=[read, declare]), board))


def test_a_declaration_of_the_controller_cites_a_reading_even_when_the_top_candidate_bears_on_none(tmp_path, monkeypatch):
    """The ranking names no identifier for a top candidate with no fault, so the controller adds the last reading itself."""
    forced = InProcessServers.forced_read_first

    def without_bearing(self, ranking):
        ranking = forced(self, ranking)
        return {**ranking, "candidates": [{**c, "evidence_ids": []} for c in ranking.get("candidates", [])]}

    def declares(prompt, call):
        # the plan reads before it declares, since no read batch runs before the model's turn and a declaration
        # needs a reading to cite
        group = re.search(r"Separating test: (\w+) for", prompt).group(1)
        return plan_json([{"kind": "read", "args": {"sensor_group": group, "cycles": 1}, "expect": {}, "evidence": []},
                          {"kind": "declare", "args": {"sensor_group": group, "cycles": 1}, "expect": {}, "evidence": []},
                          {"kind": "finish", "args": {}, "expect": {}, "evidence": []}], mode="declare")

    monkeypatch.setattr(InProcessServers, "forced_read_first", without_bearing)
    ep_dir, _row, _backend = _play(tmp_path, ScriptedPlanner(declares), {}, force_read_first=99)
    declaration = next(e for e in read_json(ep_dir / HIDDEN / "log.json") if e["event"] == "declare_insufficient_evidence")
    assert [i for i in declaration["evidence_ids"] if i.startswith("obs/")]
    assert read_json(ep_dir / HIDDEN / "verdict.json")["abstention_cites_reading"]


def test_every_read_is_a_step_of_a_model_plan_and_the_board_shows_the_read_batch_as_advice(tmp_path):
    """While the first rankings report insufficient evidence, the board shows the server's read batch with its
    information gain per unit cost as advice, the controller runs no read of its own, and every reading is a read step of a
    plan the model wrote after a planning call, logged with the elimination the ranking expected against the realized one."""
    planner = ScriptedPlanner(act_now)
    ep_dir, row, backend = _play(tmp_path, planner, {}, force_read_first=1)
    verdict = _assert_episode_files(ep_dir, row)
    _assert_repairs_verified(ep_dir, verdict)
    plans = read_json(ep_dir / "plans.json")
    _assert_every_read_planned(ep_dir, plans)
    prompts = [c["prompt"] for c in backend.calls]
    assert "No profile read yet" in prompts[0] and ORIENTATION_HEADING in prompts[1] and PLANNING_HEADING in prompts[2]
    first = prompts[2]
    assert "## Advice of the CausalDiagnosis MCP" in first and ADVISED.search(first) and "the controller reads, diagnoses and orders only what your plan names" in first
    assert "Read batches run by the controller" not in first and "## Fix history" in first and "information gain per unit cost" in first
    # the model followed the advice: its first plan reads the head of the advised batch, after one planning call
    head = ADVISED.search(first).group(1)
    assert decision_plans(plans)[0]["source"] == "model" and decision_plans(plans)[0]["plan"]["steps"][0]["args"]["sensor_group"] == head
    log = plans["board"]["reads"]
    assert log[0]["advised"] is True and log[0]["sensor_group"] == head and all(r["realized_after"] is not None for r in log)
    assert log[0]["expected_after"] is not None and "expected_information_gain" in log[0] and "ig_per_cost" in log[0]
    # one model call for the orientation, one per planning turn, and the summary, so no reading ran without a model call
    assert row["llm_calls"] == 1 + sum(1 for p in prompts if PLANNING_HEADING in p) + 1
    assert row["causal_calls"] == sum(1 for t in read_json(ep_dir / "trajectory.json") if t["server"] == CAUSAL_SERVER)
    compactions = plans["board"]["compaction_log"]
    assert compactions and all(c["collapsed"] and isinstance(c["after_call"], int) for c in compactions)


def test_the_controller_ranks_before_the_first_planning_turn_and_the_model_reads_the_advised_head(tmp_path):
    """The first ranking comes before any reading and before the first planning turn, so the board of
    that turn advises the head of the server's read batch, chosen by information gain per unit cost, and the model reads it as a
    step of its plan rather than the controller reading it on its own."""
    planner = ScriptedPlanner(act_now)
    ep_dir, row, _backend = _play(tmp_path, planner, {})
    _assert_episode_files(ep_dir, row)
    trajectory = read_json(ep_dir / "trajectory.json")
    opening = {"get_asset_profile", "get_failure_modes", "get_fix_history"}
    first = next(i for i, t in enumerate(trajectory) if t["tool"] not in opening)
    assert (trajectory[first]["server"], trajectory[first]["tool"]) == (CAUSAL_SERVER, "rank_root_causes")
    ranking = json.loads(trajectory[first]["response"])
    assert ranking["read_first"] and ranking["read_batch"]
    head = ranking["read_batch"][0]["sensor_group"]
    reads = [i for i, t in enumerate(trajectory) if t["tool"] == "read_sensors"]
    assert reads and reads[0] > first and trajectory[reads[0]]["tool_args"]["sensor_group"] == head
    plans = read_json(ep_dir / "plans.json")
    assert decision_plans(plans)[0]["source"] == "model" and decision_plans(plans)[0]["plan"]["steps"][0]["args"]["sensor_group"] == head
    # one ranking before any reading and one after every reading
    assert _tools(ep_dir, CAUSAL_SERVER).count("rank_root_causes") >= 1 + len(reads)


def test_a_diagnosis_while_the_ranking_says_read_first_runs_with_the_advice_on_the_board(tmp_path):
    """The evidence guard is advice. A plan that diagnoses while every ranking says read
    first passes the plan check and the controller submits the diagnosis, while the board of that planning turn advises the
    reads to take before a diagnosis; the controller reads nothing on its own before it."""
    def diagnose_at_once(prompt, call):
        if "Commitment: none" in prompt:
            return plan_json([{"kind": "diagnose", "args": {"rank": 1}, "expect": {"top_rank": 1}, "evidence": []}])
        return plan_json([{"kind": "finish", "args": {}, "expect": {}, "evidence": []}])
    ep_dir, row, backend = _play(tmp_path, ScriptedPlanner(diagnose_at_once), {}, force_read_first=99)
    verdict = _assert_episode_files(ep_dir, row)
    first = decision_plans(read_json(ep_dir / "plans.json"))[0]
    assert first["source"] == "model" and first["check_problems"] == [] and [s["kind"] for s in first["plan"]["steps"]] == ["diagnose"]
    prompt = next(c["prompt"] for c in backend.calls
                  if PLANNING_HEADING in c["prompt"] and "Commitment: none" in c["prompt"] and "No profile read yet" not in c["prompt"])
    assert ADVISED.search(prompt) and "before a diagnosis" in _section(prompt, "## Advice of the CausalDiagnosis MCP")
    trajectory = read_json(ep_dir / "trajectory.json")
    tools = [t["tool"] for t in trajectory]
    diagnosis = tools.index("submit_diagnosis")
    assert "read_sensors" not in tools[:diagnosis] and "refused" not in trajectory[diagnosis]["response"]
    ranking = json.loads(next(t for t in reversed(trajectory[:diagnosis]) if t["tool"] == "rank_root_causes")["response"])
    assert ranking["read_first"] is True and verdict["end_reason"] == "finish"


def test_no_tool_call_on_the_rig_precedes_the_first_planning_call_whose_board_holds_the_question_verbatim(tmp_path, monkeypatch):
    """The model plans the opening too, so the first call of an episode is a planning call whose board holds the
    question alone and verbatim, the task message that a GEPA block extends, and the profile, the failure modes and the fix history
    run only as the steps of the plan it returns, before the orientation."""
    events = []
    call = InProcessServers.call

    async def recorded(self, path, tool, args):
        events.append(("tool", tool))
        return await call(self, path, tool, args)
    monkeypatch.setattr(InProcessServers, "call", recorded)
    planner = ScriptedPlanner(act_now)
    generate = planner.generate

    def generate_recorded(prompt, temperature=0.0):
        events.append(("model", prompt))
        return generate(prompt, temperature)
    planner.generate = generate_recorded
    ep_dir, row, _backend = _play(tmp_path, planner, {})
    _assert_episode_files(ep_dir, row)
    question = (ep_dir / "question.txt").read_text(encoding="utf-8")
    kind, prompt = events[0]
    assert kind == "model" and PLANNING_HEADING in prompt and "No profile read yet" in prompt and f"Question of the episode: {question}" in prompt
    second = next(i for i, (kind, _text) in enumerate(events) if kind == "model" and i > 0)
    opening = [text for kind, text in events[:second] if kind == "tool"]
    assert ORIENTATION_HEADING in events[second][1] and opening[0] == "get_asset_profile"
    assert set(opening) == {"get_asset_profile", "get_failure_modes", "get_fix_history"}
    plan = read_json(ep_dir / "plans.json")["plans"][0]
    assert plan["source"] == "model" and [s["kind"] for s in plan["plan"]["steps"]] == ["profile", "failure_modes", "fix_history"]


def test_a_refuted_verdict_leads_to_the_cheapest_other_plan_and_the_ablation_keeps_the_surprise_only(tmp_path):
    """The first verified verdict is turned inconsistent by the fake server: with backtracking the plan's component set is
    recorded as refuted and the rule's next plan repairs another previewed branch; without it the verdict stays a surprise
    and the rule finishes."""
    planner = ScriptedPlanner(lambda prompt, call: pytest.fail("the rule only controller must not ask for a plan"))
    ep_dir, row, backend = _play(tmp_path / "back", planner, {"rule_only": True}, refute_first=True)
    verdict = _assert_episode_files(ep_dir, row)
    plans = read_json(ep_dir / "plans.json")
    refuted = plans["board"]["refuted"]
    assert len(refuted) == 1 and refuted[0]["reason"].startswith("the reading after the last close is inconsistent")
    surprise = next(p for p in plans["plans"] if p["surprise"] and "was refuted" in p["surprise"]["reason"])
    assert surprise["surprise"]["observed"]["refuted"] == [refuted[0]["components"]]
    later = [p for p in plans["plans"] if p["index"] > surprise["index"]]
    assert later and later[0]["plan"]["why"].startswith("rule: the last repair plan was refuted, so repair rank")
    ordered_after = [s["args"]["component"] for p in later for s in p["plan"]["steps"] if s["kind"] == "order"]
    assert ordered_after == ["valve"] and sorted(set(ordered_after)) != refuted[0]["components"]
    assert any("Refuted repair plans" in c["prompt"] for c in backend.calls)  # the summary turn sees the refuted plan on the board
    assert verdict["end_reason"] == "finish"
    # the reading that verified the plan is the rule's read step after the wait, of a group whose channels indicate the repaired
    # components, and the refutation stopped the plan at that step
    verifying = surprise["plan"]["steps"][surprise["surprise"]["step"] - 1]
    assert surprise["surprise"]["kind"] == "read" and verifying["kind"] == "read" and verifying["args"]["sensor_group"] != "tier1"

    ep_dir, row, backend = _play(tmp_path / "noback", planner, {"rule_only": True, "backtracking": False}, refute_first=True, seed=1)
    _assert_episode_files(ep_dir, row)
    plans = read_json(ep_dir / "plans.json")
    assert plans["board"]["refuted"] == [] and row["agent"] == "causal_rule_noback_scripted-causal"
    surprise = next(p for p in plans["plans"] if p["surprise"] and "inconsistent with" in p["surprise"]["reason"])
    assert "was refuted" not in surprise["surprise"]["reason"] and not any("was refuted" in (p["surprise"] or {}).get("reason", "") for p in plans["plans"])
    later = [p for p in plans["plans"] if p["index"] > surprise["index"]]
    assert later and not any(s["kind"] == "order" for p in later for s in p["plan"]["steps"])


def test_a_manifest_without_prices_or_budget_prices_every_tool_at_zero_and_never_declares():
    """The base row prices nothing, runs no clock and sets no budget, so a price the manifest leaves out counts
    as zero, a budget, horizon or step cap it leaves out as unlimited, every read is affordable and the rule never declares
    insufficient evidence for want of an affordable read."""
    board = _synthetic_board()
    free = {k: v for k, v in board.profile.items() if k not in ("prices", "durations", "budget", "horizon", "max_steps", "clock_mode", "downtime_rate")}
    board.set_profile(free)
    board.status = {"time": 0, "open_work_orders": [], "episode_over": False}  # an envelope without budget_left or steps_left
    assert board.budget_left == math.inf and board.horizon == math.inf and board.steps_left == math.inf and not board.budget_limited
    assert board.read_price("tier3", 2) == 0 and board.order_price("pump") == 0 and board.visit_charge == 0 and board.plan_price(["pump", "valve"]) == 0
    assert board.order_duration("pump") == 0 and board.cheapest_group() == "tier1" and board.verification_group(["pump"]) in free["sensor_groups"]
    assert all(read_affordable(board, group, 3) for group in free["sensor_groups"]) and separating_test_affordable(board, board.separating_test())
    board.status["steps_left"] = 2  # the step cap is the budget there: a read must leave the calls of the reserve (the pump's order, its verification, finish)
    assert not read_affordable(board, "tier1") and not separating_test_affordable(board, board.separating_test())
    del board.status["steps_left"]
    plan = rule_plan(board, 6)
    assert plan.steps[0].kind == "read" and not any(s.kind == "declare" for s in plan.steps) and check_plan(plan, board) == []
    board.diagnosis = {"diagnosis": dict(board.top()["combination"]), "evidence_ids": []}
    order = Plan(branch=1, mode="act_now", why="x", steps=[Step("order", {"component": "pump", "prediction_id": "pred/b1"}, {}, []), Step("finish", {}, {}, [])])
    assert not [p for p in check_plan(order, board) if "budget" in p or "price" in p]
    text = board.render()
    assert "No budget limit" in text and "no horizon" in text and "No step cap" in text and "Read prices per cycle: tier1 0" in text
    prompt = orientation_turn("q", free, [], 6)
    assert "Budget unlimited" in prompt and "no horizon" in prompt and "no clock" in prompt


def test_a_read_pays_and_takes_only_the_cycles_not_yet_paid_or_recorded():
    """A re-read of a cycle the board has paid for is free, and a read takes time only on a surface that plays the cycles it
    reads. The hydraulic rig plays every cycle of a read. The AssetOpsBench servers serve recorded cycles at once, and their
    envelope names the record, which ends at the cycle before the clock, as AobSurface.parse writes it."""
    board = _synthetic_board()
    assert board.read_price("tier1", 3) == 3 and board.read_duration(3) == 3
    board.reads_take_time = False
    board.status.update(recorded_from=board.time - 3, recorded_until=board.time - 1)  # cycles -2 to 0, the board has paid for cycle 0 of tier1
    assert board.read_price("tier1", 3) == 2 and board.read_duration(3) == 0 and board.read_price("tier2", 3) == 9
    plan = Plan(branch=1, mode="test_first", why="x", steps=[Step("read", {"sensor_group": "tier1", "cycles": 3}, {"top_rank": 1}, []), Step("finish", {}, {}, [])])
    assert check_plan(plan, board) == []


def test_compaction_collapses_only_what_the_request_needs_to_fit_and_loses_no_refuted_set():
    """No budget and no keep counts: with the server the absorbed rows collapse whatever the size; while the
    request does not fit, the latest observation collapses and then the oldest notes, notifications and surprises leave one
    at a time, never a refuted plan."""
    board = _synthetic_board(read_first=False, second=True)
    board.record_refuted(["pump"], "the reading after the last close is inconsistent with the plan's prediction")
    for n in range(2, 12):
        board.add_reading({"sensor_group": "tier3", "served": [{"n": n, "t": n - 1, "values": {
            f"F{k}_VolumeFlow_l_per_min_10Hz_mean": 30.0 + k for k in range(17)} | {"CE_CoolingEfficiency_percent_1Hz_mean": 30.0}}]})
    board.status["time"] = 11
    board.set_ranking(board.ranking)  # the ranking absorbs every observation
    rows_before = len(board.evidence)
    full = len(board.render())
    collapsed = board.compact()
    # with room to spare the observations before the latest collapse to one summary row per sensor group and run
    assert collapsed and all(c["kind"] == "evidence" for c in collapsed) and len(board.evidence) == 18 < rows_before
    assert board.summaries and board.summaries[-1]["sensor_group"] == "tier3" and board.summaries[-1]["n_last"] == 10
    assert board.summaries[0]["outside"] == ["obs/1/CE_CoolingEfficiency_percent_1Hz_mean"]
    assert len(board.render()) < full and "Absorbed by the ranking and collapsed" in board.render()
    # past the trigger the latest observation collapses too, and one summary of the earlier turns replaces the notes and every
    # surprise but the latest; the refuted plans stay and every collapsed identifier stays citable
    board.surprises = [{"plan": i, "step": 1, "kind": "read", "reason": "x" * 50} for i in range(6)]
    board.notes = [f"note {i}" for i in range(5)]
    tight = len(board.render()) - 1500
    asked = []

    def summarize(b, entries):
        asked.append(len(entries))
        return "I read tier3 and set the pump aside.", 0
    collapsed = board.compact(lambda b: len(b.render()) <= tight, summarize=summarize)
    summary = next(c for c in collapsed if c["kind"] == "summary")
    assert board.evidence == [] and board.notes == [] and [s["plan"] for s in board.surprises] == [5]
    assert summary["replaced"]["notes"] == 5 and summary["replaced"]["surprises"] == 5 and len(asked) == 1 and asked[0] >= 10
    assert "## Your summary of earlier turns\nI read tier3 and set the pump aside." in board.render()
    assert not hasattr(__import__("CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board", fromlist=["x"]), "NOTES_KEPT")
    assert board.refuted and "Refuted repair plans" in board.render() and "pump" in board.render()
    assert "obs/5/CE_CoolingEfficiency_percent_1Hz_mean" in board.known_ids() and "obs/11/CE_CoolingEfficiency_percent_1Hz_mean" in board.known_ids()
    # nothing is left to replace, so a board still past the trigger sends its request without a second summary
    assert board.compact(lambda b: False, summarize=summarize) == [] and len(asked) == 1 and board.compactions == 2
    # a fixed part past the trigger is not summarized at every turn: with entries on the board, step 3 runs only when the board
    # without them is under the trigger or the request does not fit the context
    probe = __import__("copy").deepcopy(board)
    probe.notes = ["note a"]
    assert probe.compact(lambda b: False, summarize=summarize, fits_context=lambda b: True) == [] and len(asked) == 1
    probe.compact(lambda b: False, summarize=summarize, fits_context=lambda b: False)
    assert len(asked) == 2 and probe.notes == []
    assert board.to_json()["refuted"][0]["components"] == ["pump"] and board.to_json()["compactions"] == 2


def test_the_reserve_holds_the_top_plan_and_a_visit_charge_without_the_fault_rule():
    board = _synthetic_board(read_first=False, second=True)
    del board.profile["fault_rule"]
    assert "fault_rule" not in board.profile and board.visit_charge == 15
    # rank 1 repairs the pump: its order plus one visit charge
    assert order_reserve(board) == board.order_price("pump") + 15 + board.read_price(board.verification_group(["pump"]))  # with its verifying read
    assert planning_turn(board, 6).count("Fault rule") == 0
    # the branch costs account for the visit charge of the plan, and a refuted set leaves the options until released
    assert act_options(board) == [(60, 2), (72, 1)] and repair_options(board) == [(60, 2), (72, 1)]
    board.record_refuted(["valve"], "the reading after the last close is inconsistent with the plan's prediction")
    assert repair_options(board) == [(72, 1)] and "REFUTED" in board.render()
    board.diagnosis = {"diagnosis": dict(board.candidate(2)["combination"]), "evidence_ids": []}
    after = rule_plan(board, 6)
    assert after.branch == 1 and after.steps[0].args == {"component": "pump", "prediction_id": "pred/b1"} and after.why.startswith("rule: the last repair plan was refuted")
    board.diagnosis = None
    # a later ranking after a further reading that names the refuted set first releases it
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 2, "t": 1, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 45.0}}]})
    ranking = json.loads(json.dumps(board.ranking))
    ranking["candidates"][0], ranking["candidates"][1] = {**ranking["candidates"][1], "rank": 1}, {**ranking["candidates"][0], "rank": 2}
    board.set_ranking(ranking)
    assert board.refuted_last is None and board.ranking_basis == (2, 0, 1)
    # a profile without a visit price pays no visit charge
    plain = _synthetic_board(read_first=False)
    del plain.profile["prices"]["visit"]
    assert plain.visit_charge == 0 and order_reserve(plain) == plain.order_price("pump") + plain.read_price(plain.verification_group(["pump"]))


def test_the_advice_keeps_the_refuted_set_out_until_a_later_ranking_names_it_first():
    """A refuted plan is recorded and the advice names the cheapest other plan, and the plan check,
    which keeps no rule of the server, no longer refuses a plan that orders the refuted set again."""
    board = _synthetic_board(read_first=False, second=True)
    board.diagnosis = {"diagnosis": dict(board.candidate(2)["combination"]), "evidence_ids": []}
    board.record_refuted(["valve"], "the reading after the last close is inconsistent with the plan's prediction")

    def plan(component, prediction):
        order = Step("order", {"component": component, "prediction_id": prediction}, {"prediction_id": prediction})
        return Plan(branch=2, mode="act_now", why="", steps=[order, Step("wait", {"cycles": 5}), Step("read", {"sensor_group": "tier3", "cycles": 1}), Step("finish")])
    assert check_plan(plan("valve", "pred/b2"), board) == [] and check_plan(plan("pump", "pred/b1"), board) == []
    assert any(line.startswith("The last repair plan was refuted, so the advice is to repair rank 1, the cheapest other plan") for line in advice(board))
    assert "REFUTED" in board.render() and "Refuted repair plans" in board.render()
    board.backtracking = False  # Ag8: without backtracking the board advises no other plan
    assert not [line for line in advice(board) if "was refuted" in line]
    board.backtracking = True
    # a later ranking after a further reading that names the set first releases it, and the advice names it no more
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 2, "t": 1, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 45.0}}]})
    ranking = json.loads(json.dumps(board.ranking))
    ranking["candidates"][0], ranking["candidates"][1] = {**ranking["candidates"][1], "rank": 1}, {**ranking["candidates"][0], "rank": 2}
    board.set_ranking(ranking)
    assert board.refuted_last is None and not [line for line in advice(board) if "was refuted" in line]


def test_the_batch_rule_the_declaration_and_the_switches():
    board = _synthetic_board(read_first=True, second=True)
    board.parallel = False  # the rules without the schedule of candidate plans, since both plans share the confirm test tier3, which a schedule adds
    # without a read batch the controller reads the separating test, and with one it reads the affordable prefix in order
    assert batch_reads(board) == [("tier2", 1)] and batch_reads(board, reads="separating") == [("tier2", 1)]
    board.ranking["read_batch"] = [{"sensor_group": "tier3", "cycles": 1}, "tier1"]
    board.ranking["elimination"] = [{"sensor_group": "tier3", "cycles": 1, "price": 10, "duration": 1, "cost": 11, "effective_before": 2.0,
                                    "expected_effective_after": 1.2, "expected_elimination": 0.8, "elimination_per_cost": 0.07, "information_gain": 0.5, "ig_per_cost": 0.0454545, "standard_error": 0.0}]
    assert batch_reads(board) == [("tier3", 1), ("tier1", 1)] and batch_reads(board, reads="separating") == [("tier2", 1)]
    # the one step edge cutting arm reads the group with the largest expected cut of repair plan classes per unit cost, and
    # falls to the separating test when no reading can change the plan on its own (ec2 zero everywhere)
    board.ranking["elimination"][0].update({"ec2": 0.1, "ec2_per_cost": 0.1 / 11})
    board.ranking["elimination"].append({"sensor_group": "tier1", "cycles": 1, "price": 1, "duration": 1, "cost": 2, "effective_before": 2.0,
                                         "expected_effective_after": 1.9, "expected_elimination": 0.1, "elimination_per_cost": 0.05, "ec2": 0.05, "ec2_per_cost": 0.025})
    assert batch_reads(board, reads="ec2") == [("tier1", 1)] and rule_plan(board, 6, reads="ec2").why.endswith("(one step edge cutting)")
    assert "edge cut per unit cost" in board.render()
    for row in board.ranking["elimination"]:
        row["ec2"] = row["ec2_per_cost"] = 0.0
    assert batch_reads(board, reads="ec2") == [("tier2", 1)] and rule_plan(board, 6, reads="ec2").why.endswith("the separating test is affordable")
    board.ranking["elimination"].pop()
    plan = rule_plan(board, 6)
    assert [s.kind for s in plan.steps] == ["read", "read"] and plan.steps[0].expect == {"effective_at_most": 2.0}
    assert plan.steps[1].expect == {"top_rank": 1, "top_mass_at_least": 0.0} and plan.why.startswith("rule: the ranking says read first, so read the batch")
    assert check_plan(plan, board) == [] and "Read batch" in board.render() and "effective hypotheses" not in board.render()
    board.ranking["effective_hypotheses"], board.ranking["belief_time"] = 2.0, 1
    assert "effective hypotheses 2.000 at cycle 1" in board.render()
    # the advice names the batch before a diagnosis, and a plan that diagnoses first passes the check
    diagnose = Plan(branch=1, mode="act_now", why="", steps=[Step("diagnose", {"rank": 1}, {"top_rank": 1})])
    assert check_plan(diagnose, board) == [] and advice(board)[0].startswith("Advised reads, the read batch by information gain per unit cost: tier3 x1")
    # when no read is affordable the rule declares the separating test and repairs the cheapest plan after it
    board.status["budget_left"] = board.plan_price(["pump"]) + 1
    assert batch_reads(board) == []
    plan = rule_plan(board, 6)
    assert plan.mode == "declare" and [s.kind for s in plan.steps][:2] == ["declare", "order"] and plan.steps[0].args == {"sensor_group": "tier2", "cycles": 1}
    assert plan.steps[1].args["component"] == "valve" and plan.why.startswith("rule: the ranking says read first and no separating read is affordable")
    problems = check_plan(plan, board)
    assert problems == [] or all("exceed the budget left" in p for p in problems)
    # the two switches of version 7 land in the label and in the driver's options
    assert CausalLoopOptions(reads="separating").label() == "causal_e10" and CausalLoopOptions(backtracking=False).label() == "causal_noback"
    assert CausalLoopOptions(reads="ec2").label() == "causal_ec2"
    options = run_episodes.causal_options(run_episodes.build_parser().parse_args(["--agent", "causal", "--causal-e10-reads", "--causal-no-backtrack"]))
    assert options["reads"] == "separating" and options["backtracking"] is False
    assert run_episodes.agent_label_of("causal", "m", options) == "causal_e10_noback_m"
    ec2 = run_episodes.causal_options(run_episodes.build_parser().parse_args(["--agent", "causal", "--causal-ec2-reads"]))
    assert ec2["reads"] == "ec2" and run_episodes.agent_label_of("causal", "m", ec2) == "causal_ec2_m"
    with pytest.raises(SystemExit, match="two arms of one ablation"):
        run_episodes.causal_options(run_episodes.build_parser().parse_args(["--agent", "causal", "--causal-e10-reads", "--causal-ec2-reads"]))


# ----- the AssetOpsBench rows: the same controller on AssetOpsBench's servers behind the metering layer -----

AOB_EVIDENCE_ID = re.compile(r"^\| (obs/\S+) \|", re.MULTILINE)  # the citation form of the rows, obs/<timestamp>/<field>


def act_now_on_assetopsbench(prompt, call):
    """The act now plan on the AssetOpsBench rows: the head of the advised batch while the board advises one, else diagnose rank
    1, order the faulty components of the top branch citing the plan's prediction, and where a clock runs (the surface's tool
    line says when none does) wait one cycle for the close and read the cycle after it, which verifies the repair, then finish."""
    advised = advised_read(prompt)
    if advised:
        return advised
    steps = []
    if "Commitment: none" in prompt:
        steps.append({"kind": "diagnose", "args": {"rank": 1}, "expect": {"top_rank": 1}, "evidence": AOB_EVIDENCE_ID.findall(prompt)[:3]})
    ordered = set(ORDERED.findall(prompt))
    ids = branch_predictions(prompt)
    for component in top_faulty(prompt):
        if component in ordered:
            continue
        pid = ids.get(component, ESTIMATE)
        steps.append({"kind": "order", "args": {"component": component, "prediction_id": pid},
                      "expect": {"prediction_id": pid, "recovery_at_least": 0.0}, "evidence": [pid] if pid != ESTIMATE else []})
    if "no wait is possible" not in prompt:
        steps.append({"kind": "wait", "args": {"cycles": 1}, "expect": {"no_new_alarm": True}, "evidence": []})
        steps.append(dict(VERIFYING_READ))
    steps.append({"kind": "finish", "args": {}, "expect": {}, "evidence": []})
    return plan_json(steps)


def _aob_run(tmp_path, config_name, planner, options, seed=0, server=True, start=None, think_seconds=None):
    """One episode of the causal agent on an AssetOpsBench row through EpisodeRunner, as --servers aob plays it: the real
    metered servers over stdio on the memory store, with the diagnostics server or without it as --no-diagnostics, the
    launchers written by the driver, the verifier on the recorded engine; returns the episode folder, the row and the
    recording backend."""
    facts_path, cycles, blocks = load_world()
    _name, config = load_config(config_name)
    if think_seconds is None:
        config = pinned(config)  # the subject of these tests is the agent, not the speed of the machine
    else:  # the row's own clock mode with a thinking rate that no call of a test reaches, so thinking never moves the clock
        config = type(config).from_json(dataclasses.replace(config, think_seconds_per_cycle=think_seconds).to_json())
    if not server:
        config = type(config).from_json(dataclasses.replace(config, diagnostics_server=False).to_json())
    if start is not None:  # a start the test names in place of the draw of the seed
        config = type(config).from_json(dataclasses.replace(config, start=start).to_json())
    backend = RecordingBackend(planner)
    servers = (*METERED_SERVERS, CAUSAL_SERVER) if server else METERED_SERVERS
    runner = EpisodeRunner(config, config_name, "causal", servers, cycles, blocks, facts_path, "causal-aob",
                           {"judge": None, "grade": True, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL}, backend=backend, scripted=planner,
                           model=planner.model_id, causal_options={**{"branches": True, "single_hypothesis": False, "plan_length": None, "expectations": True,
                                                                     "rule_only": False}, **options}, store={"kind": "memory"})
    name = f"seed{seed:06d}"
    ep_dir = tmp_path / name
    meta = {"config_name": config_name, "agent": "causal", "servers": list(servers), "model": planner.model_id,
            "clock_mode": config.clock_mode, "guidance_level": config.guidance_level}
    row = runner.play(ep_dir, name, seed, None, config, meta)
    return ep_dir, row, backend


def _assert_aob_surface(ep_dir, row, clock, server=True):
    """What every episode on the AssetOpsBench rows shows: no rig server, the opening through the task server, fmsr, wo
    and the extent of the record, reads as history windows of whole cycles with the group's sensors, writes through the
    task server and wo on the child assets, the time tools on the clock rows only, a verdict the verifier accepts with
    the provenance of the cited evidence, and the store dropped after the export."""
    trajectory = read_json(ep_dir / "trajectory.json")
    assert row["error"] == "", row["error"]
    assert {t["server"] for t in trajectory} <= set(METERED_SERVERS) | {CAUSAL_SERVER} and not any(t["error"] for t in trajectory)
    tools = [(t["server"], t["tool"]) for t in trajectory]
    assert tools[0] == ("task", "get_asset_profile") and tools[-1] == ("task", "finish") and row["end_reason"] == "finish"
    assert tools.count(("fmsr", "get_failure_modes")) == len(COMPONENTS) and ("wo", "list_workorders") in tools and ("iot", "stream_extent") in tools
    assert ("task", "submit_diagnosis") in tools or ("task", "declare_insufficient_evidence") in tools
    histories = [t for t in trajectory if t["tool"] == "history"]
    closed = [t for t in histories if "end" in t["tool_args"]]  # a read with nothing unread asks for the open window after the newest cycle
    assert histories and all(set(t["tool_args"]) == {"site_name", "asset_id", "sensors", "start", "end", "limit"} for t in closed)
    assert all(set(t["tool_args"]) == {"site_name", "asset_id", "sensors", "start", "limit"} for t in histories if t not in closed)
    profile = json.JSONDecoder().raw_decode(trajectory[0]["response"])[0]  # the receipt line follows the body from the priced row on
    seconds = profile["clock"]["seconds_per_cycle"]
    read_before = {}
    for call in closed:
        args = call["tool_args"]
        span = (datetime.fromisoformat(args["end"]) - datetime.fromisoformat(args["start"])).total_seconds()
        assert span == args["limit"] * seconds and args["limit"] >= 1  # the window is exactly the cycles of the step
        group = next(g for g, fields in profile["sensor_groups"].items() if set(args["sensors"]) == {f for n in fields for f in (n, f"{n}_std")})
        assert group in profile["sensor_groups"]
        # a window never asks again for a cycle of its group that an earlier window read
        first = datetime.fromisoformat(args["start"])
        covered = {first + timedelta(seconds=seconds * i) for i in range(args["limit"])}
        assert not covered & read_before.setdefault(group, set()), f"{group} window from {args['start']} reads a cycle read before"
        read_before[group] |= covered
    orders = [t for t in trajectory if t["tool"] == "generate_work_order"]
    assert orders and all(t["tool_args"]["asset_num"] in profile["asset"]["children"].values() and t["tool_args"]["work_type"] == "CM"
                          and bool(t["tool_args"]["aob_source"]["prediction_id"]) is server for t in orders)
    if clock and server:  # the time tools are served on the clock rows at the level with the CausalDiagnosis MCP only
        assert ("task", "get_current_time") in tools and ("task", "wait_for_notification") in tools
    else:
        assert ("task", "get_current_time") not in tools and ("task", "wait_for_notification") not in tools
    verdict = read_json(ep_dir / HIDDEN / "verdict.json")
    assert verdict["verified"] and verdict["checks"]["provenance"]["ok"] and verdict["end_reason"] == "finish"
    assert not (ep_dir / "aob_store.json").exists() and (ep_dir / "aob_documents.json").exists()
    return trajectory, verdict


@pytest.mark.parametrize("config_name", ["base", "real_time"])
def test_the_causal_agent_plays_the_assetopsbench_rows_with_the_scripted_model(tmp_path, config_name):
    """The scripted model on the base row and on the real-time clock row through AssetOpsBench's servers: the plan acts on
    the top candidate, the orders cite the branch's prediction, and on the clock row the controller waits for the close,
    reads the recorded cycle after it and verifies the prediction."""
    planner = ScriptedPlanner(act_now_on_assetopsbench)
    ep_dir, row, _backend = _aob_run(tmp_path, config_name, planner, {})
    _assert_episode_files(ep_dir, row)
    trajectory, verdict = _assert_aob_surface(ep_dir, row, clock=config_name == "real_time")
    board = read_json(ep_dir / "plans.json")["board"]
    cited = [t for t in trajectory if t["tool"] == "generate_work_order"][0]["tool_args"]["aob_source"]["prediction_id"]
    assert cited in board["predictions"]
    if config_name == "real_time":
        # the order closed as the next cycle played, one reading after the close verified the plan's prediction
        verifications = board["verifications"]
        assert verifications and all(v["status"] == "verified" for v in verifications.values()), verifications
        wait = next(i for i, t in enumerate(trajectory) if t["tool"] == "wait_for_notification")
        assert any(t["tool"] == "history" for t in trajectory[wait:]) and any(t["tool"] == "verify_outcome" for t in trajectory[wait:])
        assert all(p["found"] for p in verdict["prediction_verdict"])
    else:
        # no cycle runs on the base row, so the orders close inside their call and nothing can verify them
        assert all(o["status"] == "closed" and o["closed_t"] == o["created_t"] for o in board["orders"].values())
        assert not board["verifications"] and any("No cycle runs after the close" in n for n in board["notes"])


@pytest.mark.parametrize("config_name", ["base", "real_time"])
def test_the_driver_runs_the_causal_agent_on_the_assetopsbench_rows(tmp_path, monkeypatch, config_name):
    """--agent causal --servers aob on both rows through run_episodes.main, every model call through the local endpoint: the run
    records the store and the briefing of the rows, and the episode holds what every AssetOpsBench episode of the agent holds."""
    planner = ScriptedPlanner(act_now_on_assetopsbench)
    server, requests = _planner_endpoint(planner)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_episodes.main(["--config", pinned_preset(tmp_path, config_name), "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url", base,
                           "--seeds", "1", "--servers", "aob", "--out", str(tmp_path / "run")])
    finally:
        server.shutdown()
    run_dir = next(p for p in (tmp_path / "run").iterdir() if p.is_dir())
    meta = read_json(run_dir / "run.json")
    import csv
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert meta["servers"] == [*METERED_SERVERS, CAUSAL_SERVER] and meta["backend"] == {"kind": "memory"} and meta["briefing"] == AOB_BRIEFING_VERSION
    assert len(rows) == 1 and int(rows[0]["llm_calls"]) == len(requests) >= 3
    ep_dir = run_dir / "seed000000"
    _assert_aob_surface(ep_dir, rows[0], clock=config_name == "real_time")
    assert (ep_dir / "answer.txt").read_text(encoding="utf-8") == SUMMARY_REPLY


# ----- robustness of the prompt and parser against joined component names, zero cycle waits, over long plans, echoed
# evidence lists, empty component openings and unnamed time tools, one test each -----

def _order(component, **extra):
    return {"kind": "order", "args": {"component": component, "prediction_id": "pred/b1"}, "expect": {"recovery_at_least": 0.5}, "evidence": ["pred/b1"], **extra}


def test_the_parser_splits_the_joined_components_of_an_order_step_into_one_order_each():
    finish = {"kind": "finish"}
    for joined, names in (("cooler+accumulator", ["cooler", "accumulator"]), ("valve, pump, accumulator", ["valve", "pump", "accumulator"]),
                          ("cooler and valve", ["cooler", "valve"]), ("pump/valve", ["pump", "valve"])):
        plan = parse_plan(json.dumps({"mode": "act_now", "steps": [_order(joined), finish]}), 6, COMPONENTS)
        assert [s.args.get("component") for s in plan.steps] == [*names, None]
        assert all(s.args["prediction_id"] == "pred/b1" and s.expect["prediction_id"] == "pred/b1" and s.evidence == ["pred/b1"] for s in plan.steps[:-1])
    # the split orders count against the step limit, and the refusal says so
    with pytest.raises(PlanError, match="once the joined components"):
        parse_plan(json.dumps({"mode": "act_now", "steps": [_order("cooler+valve"), *[{"kind": "wait", "args": {"cycles": 1}}] * 4, finish]}), 6,
                   COMPONENTS)
    # before the profile is read no name is split, so the order names what the model wrote
    joined = parse_plan(json.dumps({"mode": "act_now", "steps": [_order("cooler+valve"), finish]}), 6)
    assert [s.args.get("component") for s in joined.steps] == ["cooler+valve", None]


def test_a_wait_of_zero_cycles_reads_as_one_and_the_prompt_says_so():
    assert parse_plan('{"mode": "act_now", "steps": [{"kind": "wait", "args": {"cycles": "0"}}]}', 6).steps[0].args == {"cycles": 1}
    assert "0 is read as 1" in system_prompt(6)


def test_a_plan_over_the_step_limit_is_refused_and_the_prompt_states_the_limit():
    steps = [{"kind": "wait", "args": {"cycles": 1}}] * 7
    with pytest.raises(PlanError, match="the plan has 7 steps, at most 6 are allowed"):
        parse_plan(json.dumps({"mode": "act_now", "steps": steps}), 6)
    assert "at most 6 entries under steps, finish included" in system_prompt(6) and "at most 3 entries under steps" in system_prompt(3)


def test_without_a_count_a_plan_is_bounded_by_the_budget_the_horizon_and_the_step_cap_alone():
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import COUNTED_STEPS, PAID_STEPS
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import DEFAULT_PLAN_LENGTH
    assert DEFAULT_PLAN_LENGTH is None and CausalLoopOptions().plan_length is None and CausalLoopOptions().label() == "causal"
    assert CausalLoopOptions(plan_length=1).label() == "causal_len1"
    steps = [{"kind": "wait", "args": {"cycles": 1}}] * 9
    assert len(parse_plan(json.dumps({"mode": "act_now", "steps": steps}), None).steps) == 9
    # both system prompts, with and without the diagnostics server, state the bound in the same words, and with a count the
    # sentence of the count stays
    for server in (True, False):
        free, counted = system_prompt(None, server=server), system_prompt(6, server=server)
        assert free.count(PAID_STEPS) == 1 and "entries under steps, finish included" not in free.replace(PAID_STEPS, "")
        assert counted.count(COUNTED_STEPS.replace("<PLAN_LENGTH>", "6")) == 1 and PAID_STEPS not in counted
    assert PAID_STEPS == ("with as many entries under steps as the budget, the horizon and the step cap pay, finish included, and candidates "
                          "best first; a plan they cannot pay is refused and you are asked again.")
    assert "which count against the step limit" in system_prompt(None)


def test_the_planning_turn_names_a_count_only_when_one_is_set():
    board = Board(question="q")
    assert planning_turn(board, None).endswith("Return one plan as the JSON object described above, and nothing after it.")
    assert planning_turn(board, 6).endswith("Return one plan as the JSON object described above, with at most 6 steps and nothing after it.")


def test_a_step_keeps_every_identifier_it_cites_and_the_prompt_asks_for_few():
    """No count of cited identifiers: the parser keeps all of them, a plan may list any number of
    candidates, and the prompt asks for the few that moved the model without a number."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import plan as plan_module
    ids = [f"obs/1/F{i}_mean" for i in range(30)]
    plan = parse_plan(json.dumps({"mode": "act_now", "steps": [{"kind": "diagnose", "args": {"rank": 1}, "evidence": ids}]}), 6)
    assert plan.steps[0].evidence == ids and not hasattr(plan_module, "MAX_CITED") and not hasattr(plan_module, "MAX_CANDIDATES")
    many = [{"levels": {"pump": 3}, "failure_mode_ids": ids}] * 9
    assert len(plan_module.parse_candidates(many)) == 9
    for server in (True, False):
        text = system_prompt(6, server=server)
        assert "the few that moved you" in text and "at most 8" not in text and "at most 5 candidates" not in text
    assert "never copy the ranking's evidence list" in system_prompt(6)


def test_a_count_of_cycles_takes_the_singular_only_for_one():
    assert [n_cycles(n) for n in (0, 1, 10)] == ["0 cycles", "1 cycle", "10 cycles"]


def test_the_opening_asks_the_failure_modes_of_every_component_by_name(tmp_path):
    planner = ScriptedPlanner(act_now)
    ep_dir, row, backend = _play(tmp_path, planner, {})
    assert row["error"] == ""
    trajectory = read_json(ep_dir / "trajectory.json")
    assert [t["tool_args"] for t in trajectory if t["tool"] == "get_failure_modes"] == [{"component": c} for c in COMPONENTS]
    orientation = next(p for p in planner.prompts if ORIENTATION_HEADING in p)  # after the plan that read the opening
    assert "No profile read yet" in planner.prompts[0] and all(f"| {c} |" in orientation for c in COMPONENTS)


def test_the_board_names_the_time_tools_on_a_clock_row_only():
    board = _synthetic_board()
    assert "get_current_time" not in board.render()
    board.profile["tools"] = [{"name": "get_current_time"}, {"name": "wait_for_notification"}, {"name": "read_sensors"}]
    rendered = board.render()
    assert "get_current_time reads the cycle without moving the clock" in rendered and "wait_for_notification(timeout)" in rendered
    assert "get_current_time reads the cycle" in planning_turn(board, 6)


def test_a_consistent_verified_plan_stands_whatever_the_belief_top_shows_after_the_reading(tmp_path):
    """From the first work order on, the fake server puts a candidate with one more valve fault on top of every ranking and calls
    the verified plan consistent, so on any data the belief after the verifying reading disagrees with the plan. The verdict
    stands: the controller records no refuted plan, creates no work order after the verdict and finishes."""
    planner = ScriptedPlanner(lambda prompt, call: pytest.fail("the rule only controller must not ask for a plan"))
    ep_dir, row, _backend = _play(tmp_path, planner, {"rule_only": True}, promote_after_order=True)
    verdict = _assert_episode_files(ep_dir, row)
    board = read_json(ep_dir / "plans.json")["board"]
    assert board["verifications"] and all(v["status"] == "verified" and v["consistent"] is True for v in board["verifications"].values())
    trajectory = read_json(ep_dir / "trajectory.json")
    tools = [t["tool"] for t in trajectory]
    first_order, verified_at = tools.index("create_work_order"), tools.index("verify_outcome")
    tops = [json.JSONDecoder().raw_decode(t["response"].lstrip())[0]["candidates"][0]
            for t in trajectory[first_order:verified_at] if t["tool"] == "rank_root_causes"]
    assert tops and all(top["rank"] == 1 and [f["component"] for f in top["faults"]] == ["valve"] for top in tops)  # the premise holds
    assert board["refuted"] == [] and "create_work_order" not in tools[verified_at:]
    assert verdict["end_reason"] == "finish"


# ----- the agent without the diagnostics server, one switch, the same loop on its own candidates and the normal ranges -----

def test_a_summary_of_the_working_memory_keeps_every_cycle_it_holds():
    """Reads on the AssetOpsBench clock rows alternate between the newest cycle and cycles before the start, so the rows a
    ranking absorbed are not in cycle order. The summary they collapse into keeps each of their cycles, so a cycle read out
    of order stays paid and is never read again, and the board names the cycles as they are."""
    board = _synthetic_board()
    for n, t in enumerate((1, 2, -1, 3, -2), start=2):
        board.add_reading({"sensor_group": "tier1", "served": [{"n": n, "t": t, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 45.0}}]})
    board.ranking_basis = (board.observations, len(board.closed_orders), board.time)
    board.compact()
    assert board.summaries and board.paid_cycles("tier1") == {-2, -1, 0, 1, 2, 3}
    board.compact(lambda b: False)
    assert board.paid_cycles("tier1") == {-2, -1, 0, 1, 2, 3} and 4 not in board.paid_cycles("tier1")
    assert "| tier1 | 1 to 6 | -2 to 3 |" in board.render()


def test_a_history_window_covers_unread_cycles_in_one_run_and_never_a_read_one():
    """The window of an AssetOpsBench read is the newest run of unread cycles, at most the cycles asked for, so it never
    asks again for a cycle the working memory holds. When every recorded cycle is read the window is the open one after the
    newest known cycle, which serves what the record holds by then."""
    board = _synthetic_board()
    for n, t in ((2, 1), (3, 2)):
        board.add_reading({"sensor_group": "tier1", "served": [{"n": n, "t": t, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 45.0}}]})
    surface = AobSurface()
    surface.oldest, surface.newest = -2, 3
    board.status.update(recorded_from=-2, recorded_until=3)  # the envelope names the record the surface knows, as after its extent
    assert surface.read_window(board, "tier1", 1) == (3, 3)
    assert surface.read_window(board, "tier1", 3) == (3, 3)  # cycles 1 and 2 are read, so the run stops at 3
    surface.newest = board.status["recorded_until"] = 2
    assert surface.read_window(board, "tier1", 3) == (-2, -1)
    for n, t in ((4, -1), (5, -2)):
        board.add_reading({"sensor_group": "tier1", "served": [{"n": n, "t": t, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 45.0}}]})
    assert surface.read_window(board, "tier1", 1) == surface.read_window(board, "tier1", 4) == (3, None)


def test_a_read_of_a_group_with_every_recorded_cycle_read_makes_no_call(tmp_path):
    """On the base row of AssetOpsBench, where no clock runs and the record does not grow, a plan that reads the whole record
    of the cheapest group and then a plan that reads it again: the second read makes no history call and comes back as a
    surprise that says why, and the next plan acts (the model decides what follows)."""
    def plans(prompt, call):
        if call == 1:
            return plan_json([{"kind": "read", "args": {"sensor_group": "tier1", "cycles": 6}, "expect": {}, "evidence": []}], mode="test_first")
        if call == 2:
            return plan_json([{"kind": "read", "args": {"sensor_group": "tier1", "cycles": 1}, "expect": {}, "evidence": []}], mode="test_first")
        return act_now_on_assetopsbench(prompt, call)
    ep_dir, row, _backend = _aob_run(tmp_path, "base", ScriptedPlanner(plans), {})
    _assert_episode_files(ep_dir, row)
    _assert_aob_surface(ep_dir, row, clock=False)
    trajectory = read_json(ep_dir / "trajectory.json")
    tier1 = json.JSONDecoder().raw_decode(trajectory[0]["response"])[0]["sensor_groups"]["tier1"]
    reads = [t for t in trajectory if t["tool"] == "history" and set(t["tool_args"]["sensors"]) == {f for n in tier1 for f in (n, f"{n}_std")}]
    assert len(reads) == 1  # the first read took the whole record, at most the six cycles asked for, and nothing read it again
    second = decision_plans(read_json(ep_dir / "plans.json"))[1]
    assert second["steps"][0]["status"] == "surprise" and "every recorded cycle of tier1" in second["steps"][0]["note"]


def test_the_working_memory_names_every_prediction_of_the_mcp_as_a_prediction():
    """The values the CausalDiagnosis MCP predicts carry the word on the working memory, so the model never reads
    a predicted recovery, loss, cost or elimination as a measured one."""
    board = _synthetic_board(read_first=True)
    text = board.render()
    for label in ("predicted recovery if acted", "predicted loss if acted", "predicted loss if tested", "predicted cost to restored act now / test first",
                  "predicted mass if true 0.900, if false 0.200", "predicted expected loss of acting now", "| id | action | predicted recovery |"):
        assert label in text, label
    for bare in ("| recovery if acted |", "| loss if acted |", "| cost to restored", "; expected loss of acting now", ": price 10, mass if true"):
        assert bare not in text, bare


def bands_plans(order_first=None, wait=LONGEST_REPAIR, forward=None):
    """A scripted planner without the diagnostics server, which may read the hidden start, as a script may: read the full group,
    then list the start as the first candidate, or order_first(start) before it, diagnose rank 1, order the faulty components of
    that candidate, wait for the close, read the cycle after it and finish; after a refuted plan, list the start and repair what
    of it is not ordered yet. With forward, on a row that serves no wait, one read of that many cycles takes the place of the
    wait and the read, its window playing the cycles to the close; with wait 0 the plan finishes after its orders."""
    def plans(prompt, call, truth):
        if "No reading yet." in prompt:
            return plan_json([dict(VERIFYING_READ, expect={"outside": []})], mode="test_first")
        start = {c: v for c, v in truth().items() if v != HEALTHY[c]}
        ordered = set(ORDERED.findall(prompt))
        steps = []
        if "Commitment: none" in prompt:
            first = order_first(start) if order_first else start
            candidates = [{"levels": first, "failure_mode_ids": []}] + ([{"levels": start, "failure_mode_ids": []}] if order_first else [])
            steps.append({"kind": "diagnose", "args": {"rank": 1}, "expect": {}, "evidence": EVIDENCE_ID.findall(prompt)[:3]})
            faulty = list(first)
        else:
            candidates = [{"levels": start, "failure_mode_ids": []}]
            faulty = [c for c in start if c not in ordered]
        steps += [{"kind": "order", "args": {"component": c}, "expect": {}, "evidence": []} for c in faulty]
        if (faulty or ordered) and forward:
            steps.append(dict(VERIFYING_READ, args={"sensor_group": VERIFYING_READ["args"]["sensor_group"], "cycles": forward}))
        elif (faulty or ordered) and wait:  # wait=0: no tail, as on a row that serves no wait and plays no cycle by a read
            steps += [{"kind": "wait", "args": {"cycles": wait}, "expect": {"no_new_alarm": True}, "evidence": []},
                      dict(VERIFYING_READ)]
        steps.append({"kind": "finish", "args": {}, "expect": {}, "evidence": []})
        return json.dumps({"branch": 1, "mode": "act_now", "why": "scripted: the channels outside their normal ranges read as the start", "candidates": candidates,
                           "steps": steps})
    return plans


def _play_without_server(tmp_path, plans, seed=3, config_name="static", options=None):
    """One episode of the causal agent without the diagnostics server through EpisodeRunner, as --no-diagnostics plays it, the
    rig's action server in process; plans(prompt, call, truth) answers each planning turn, truth() giving the hidden start."""
    facts_path, cycles, blocks = load_world()
    _name, config = load_config(config_name)
    config = type(config).from_json(dataclasses.replace(config, diagnostics_server=False).to_json())
    name = f"seed{seed:06d}"
    ep_dir = tmp_path / name

    def truth():
        state = episode.load_state(ep_dir / "episode.json")
        return dict(zip(COMPONENTS, episode.replay({**state, "actions": [], "think_cycles": []}).hidden))
    planner = ScriptedPlanner(lambda prompt, call: plans(prompt, call, truth))
    backend = RecordingBackend(planner)
    runner = EpisodeRunner(config, config_name, "causal", (RIG_SERVER,), cycles, blocks, facts_path, "causal-normal-ranges",
                           {"judge": None, "grade": True, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL}, backend=backend, scripted=planner,
                           model=planner.model_id, causal_options=dict(options or {}))
    meta = {"config_name": config_name, "agent": "causal", "servers": [RIG_SERVER], "model": planner.model_id,
            "clock_mode": config.clock_mode, "guidance_level": config.guidance_level}
    with _in_process(ep_dir):
        row = runner.play(ep_dir, name, seed, None, config, meta)
    return ep_dir, row, backend


def test_without_the_server_the_agent_calls_no_diagnostics_tool_and_diagnoses_repairs_and_verifies_on_the_bands(tmp_path):
    """--no-diagnostics on the rig: the same loop runs without any tool of the diagnostics server, the board shows the failure modes,
    the channels outside their normal range and the candidates the model wrote in place of the ranking, the diagnosis submits the
    model's candidate, the orders cite no prediction, and the read after the close verifies the repair on the normal ranges."""
    ep_dir, row, backend = _play_without_server(tmp_path, bands_plans())
    verdict = _assert_episode_files(ep_dir, row, server=False)
    trajectory = read_json(ep_dir / "trajectory.json")
    assert {t["server"] for t in trajectory} == {RIG_SERVER} and row["causal_calls"] == 0 and _tools(ep_dir, CAUSAL_SERVER) == []
    plans = read_json(ep_dir / "plans.json")
    assert plans["server"] is False and plans["board"]["server"] is False
    _assert_every_read_planned(ep_dir, plans)
    prompts = [c["prompt"] for c in backend.calls if PLANNING_HEADING in c["prompt"] and "No profile read yet" not in c["prompt"]]
    for heading in ("## Failure modes", "## Channels outside their normal range"):
        assert all(heading in p for p in prompts), heading
    assert not any("## Ranking" in p or "## Advice of" in p or "Troubleshooting plans" in p for p in prompts)
    assert "rank_root_causes" not in prompts[0] and "verify_outcome" not in prompts[0]
    start = dict(zip(COMPONENTS, read_json(ep_dir / HIDDEN / "log.json")[0]["hidden"]))
    assert plans["board"]["candidates"][0]["combination"] == start  # the model's candidate, the healthy levels filled in
    diagnosis = next(t for t in trajectory if t["tool"] == "submit_diagnosis")
    assert {c: diagnosis["tool_args"][c] for c in COMPONENTS} == start
    orders = [t for t in trajectory if t["tool"] == "create_work_order"]
    assert orders and all(o["tool_args"]["prediction_id"] == "" for o in orders)
    verifications = plans["board"]["verifications"]
    assert len(verifications) == 1 and all(v["status"] == "verified by the normal ranges" and v["consistent"] is True for v in verifications.values())
    assert verdict["end_reason"] == "finish" and verdict["pass"] is True and row["fallbacks"] == 0


def test_without_the_server_a_repair_the_bands_refute_is_recorded_and_the_next_plan_repairs_another_candidate(tmp_path):
    """Seed 3 starts with the cooler at 20: the first candidate repairs the healthy pump, the reading after its close shows channel
    means outside their normal ranges, so the normal range verification refutes the plan, the controller records the pump, stops the plan
    at that read and the model's next plan repairs the cooler, which the next reading verifies."""
    ep_dir, row, backend = _play_without_server(tmp_path, bands_plans(order_first=lambda start: {"pump": 1}))
    _assert_episode_files(ep_dir, row, server=False)
    assert read_json(ep_dir / HIDDEN / "log.json")[0]["hidden"] == [20, 100, 0, 130]
    plans = read_json(ep_dir / "plans.json")
    refuted = plans["board"]["refuted"]
    assert [r["components"] for r in refuted] == [["pump"]] and refuted[0]["reason"] == "the reading after the last close shows a channel mean outside its normal range"
    surprise = next(p for p in plans["plans"] if p["surprise"] and "was refuted" in p["surprise"]["reason"])
    assert surprise["surprise"]["kind"] == "read" and surprise["surprise"]["observed"]["refuted"] == [["pump"]]
    later = [p for p in plans["plans"] if p["index"] > surprise["index"]]
    ordered_after = [s["args"]["component"] for p in later for s in p["plan"]["steps"] if s["kind"] == "order"]
    assert ordered_after == ["cooler"]
    verdicts = [v["consistent"] for v in plans["board"]["verifications"].values()]
    assert verdicts == [False, True] and "Refuted repair plans" in backend.calls[-1]["prompt"]
    replan = [c["prompt"] for c in backend.calls if PLANNING_HEADING in c["prompt"]][-1]
    assert "## Your candidates" in replan and "REFUTED" not in replan and "Refuted repair plans" in replan
    assert row["causal_calls"] == 0 and _tools(ep_dir, CAUSAL_SERVER) == []


def test_without_the_server_the_agent_plays_the_assetopsbench_clock_row_without_our_tools_on_the_bands(tmp_path):
    """The same agent without the diagnostics server on AssetOpsBench's servers of the advanced real-time row at the level without
    the CausalDiagnosis MCP, which serves no wait: no diagnostics call and no wait, the orders go through generate_work_order with an empty
    prediction, and one history read whose window plays the cycles to the close verifies the repair."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv

    def truth():
        state = episode.load_state(tmp_path / "seed000003" / "episode.json")
        return dict(zip(COMPONENTS, episode.replay({**state, "actions": [], "think_cycles": []}, env_class=RecordedRigEnv).hidden))
    plans = bands_plans(forward=2 * LONGEST_REPAIR + 1)  # a window past the present that reaches the close of a visit of two orders
    ep_dir, row, _backend = _aob_run(tmp_path, "advanced_real_time", ScriptedPlanner(lambda prompt, call: plans(prompt, call, truth)), {}, seed=3,
                                     server=False)
    _assert_episode_files(ep_dir, row, server=False)
    trajectory, verdict = _assert_aob_surface(ep_dir, row, clock=True, server=False)
    assert CAUSAL_SERVER not in {t["server"] for t in trajectory} and row["causal_calls"] == 0
    assert not {"wait_for_notification", "get_current_time", "declare_insufficient_evidence"} & {t["tool"] for t in trajectory}
    board = read_json(ep_dir / "plans.json")["board"]
    assert board["verifications"] and all(v["status"] == "verified by the normal ranges" for v in board["verifications"].values())
    assert verdict["pass"] is True


def test_without_the_server_the_agent_plays_the_assetopsbench_real_time_row_without_our_tools_and_finishes_unverified(tmp_path):
    """The real-time row at the level without the CausalDiagnosis MCP serves no wait, and its reads play no cycle, so after the diagnosis and the
    orders no step reaches the close: the agent finishes without a wait and without a verifying read, and the closing summary turn
    carries the note of the unverified finish; the verifier still scores the episode from its writes."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import NO_WAY_TO_THE_CLOSE
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv

    def truth():
        state = episode.load_state(tmp_path / "seed000003" / "episode.json")
        return dict(zip(COMPONENTS, episode.replay({**state, "actions": [], "think_cycles": []}, env_class=RecordedRigEnv).hidden))
    plans = bands_plans(wait=0)
    ep_dir, row, backend = _aob_run(tmp_path, "real_time", ScriptedPlanner(lambda prompt, call: plans(prompt, call, truth)), {}, seed=3, server=False)
    _assert_episode_files(ep_dir, row, server=False)
    trajectory, verdict = _assert_aob_surface(ep_dir, row, clock=True, server=False)
    tools = [t["tool"] for t in trajectory]
    assert "submit_diagnosis" in tools and "generate_work_order" in tools and tools[-1] == "finish"
    assert not {"wait_for_notification", "get_current_time", "declare_insufficient_evidence", "summarize_changes_since"} & set(tools)
    assert read_json(ep_dir / "plans.json")["board"]["verifications"] == {}
    assert NO_WAY_TO_THE_CLOSE in backend.calls[-1]["prompt"] and verdict["verified"]


def test_the_driver_plays_the_causal_agent_without_the_server_and_refuses_the_switches_that_need_it(tmp_path, monkeypatch):
    """--agent causal --no-diagnostics through run_episodes.main, the switch of the agents table that the released agents share: the run
    registers the action server alone, records diagnostics_server false and no diagnostics call, and keeps the agent label, while
    a switch that only changes what the server shows is refused before any episode."""
    def plans(prompt, call):
        if "No reading yet." in prompt:
            return plan_json([dict(VERIFYING_READ)], mode="test_first")
        return json.dumps({"branch": 1, "mode": "act_now", "why": "scripted", "candidates": [{"levels": {}, "failure_mode_ids": []}],
                           "steps": [{"kind": "diagnose", "args": {"rank": 1}, "expect": {}, "evidence": []}, {"kind": "finish"}]})
    server, requests = _planner_endpoint(ScriptedPlanner(plans))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        for switch in (["--causal-no-branches"], ["--causal-single-hypothesis"], ["--causal-e10-reads"], ["--causal-ec2-reads"]):
            with pytest.raises(SystemExit, match="CausalDiagnosis MCP"):
                run_episodes.main(["--config", "static", "--agent", "causal", "--no-diagnostics", *switch, "--model", "vendor/stand-in-1.0",
                                   "--base-url", base, "--out", str(tmp_path / "refused")])
        run_episodes.main(["--config", "static", "--agent", "causal", "--no-diagnostics", "--model", "vendor/stand-in-1.0", "--base-url", base,
                           "--seeds", "1", "--out", str(tmp_path / "run")])
    finally:
        server.shutdown()
    run_dir = next(p for p in (tmp_path / "run").iterdir() if p.is_dir())
    meta = read_json(run_dir / "run.json")
    import csv
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert meta["servers"] == [RIG_SERVER] and meta["config"]["diagnostics_server"] is False and meta["agent_label"] == "causal_vendor/stand-in-1.0"
    assert len(rows) == 1 and rows[0]["error"] == "" and rows[0]["diagnostics_server"] == "False" and rows[0]["causal_calls"] == "0"
    assert int(rows[0]["llm_calls"]) == len(requests) >= 3
    ep_dir = run_dir / "seed000000"
    assert read_json(ep_dir / "plans.json")["server"] is False and not (ep_dir / f"{CAUSAL_SERVER}_server.py").exists()


def _synthetic_board_without_server():
    """The synthetic board after one reading, without the diagnostics server: the failure modes of the pump and the cooler with
    their channels, a reading whose cooling efficiency lies below its normal range, and no ranking."""
    board = Board(server=False)
    board.set_profile(_synthetic_board().profile)
    board.set_failure_modes({"failure_modes": [
        {"component": "pump", "level": 1, "code": "pump_internal_leakage_weak", "severity": "minor", "action": "replace the pump",
         "channels": ["FS1_VolumeFlow_l_per_min_10Hz_mean"], "evidence_id": "fmea/src/pump_internal_leakage_weak"},
        {"component": "cooler", "level": 20, "code": "cooler_reduced_efficiency", "severity": "major", "action": "replace the cooler",
         "channels": ["CE_CoolingEfficiency_percent_1Hz_mean"], "evidence_id": "fmea/src/cooler_reduced_efficiency"}]})
    board.status = {"time": 1, "budget_left": 299, "steps_left": 57, "open_work_orders": [], "episode_over": False}
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 1, "t": 0, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 30.0, "CE_CoolingEfficiency_percent_1Hz_std": 0.1}}]})
    board.manual_channels = {c: {f for fields in board.profile["sensor_groups"].values() for f in fields} for c in COMPONENTS}  # as on _synthetic_board
    return board


def test_the_board_and_the_check_without_the_server_take_the_model_candidates_and_the_bands():
    board = _synthetic_board_without_server()
    text = board.render()
    assert "## Failure modes" in text and "fmea/src/cooler_reduced_efficiency" in text and "## Channels outside their normal range" in text
    assert "CE_CoolingEfficiency_percent_1Hz_mean below" in text and "1 of 1 channel means outside" in text
    assert "## Ranking" not in text and "Read advice" not in text and "## Predictions" not in text and "## Your candidates" not in text

    def plan(*steps, candidates=()):
        return Plan(branch=1, mode="act_now", why="", steps=list(steps), candidates=list(candidates))
    diagnose = Step("diagnose", {"rank": 1}, {})
    order = Step("order", {"component": "cooler", "prediction_id": ESTIMATE}, {})
    tail = [Step("wait", {"cycles": 4}), Step("read", {"sensor_group": "tier1", "cycles": 1}), Step("finish")]
    assert any("no candidate exists yet" in p for p in check_plan(plan(diagnose), board))
    cooler = {"levels": {"cooler": 20}, "failure_mode_ids": ["fmea/src/cooler_reduced_efficiency"]}
    assert check_plan(plan(diagnose, order, *tail, candidates=[cooler]), board) == []
    assert any("cooler has the levels 3, 20, 100, not 7" in p for p in check_plan(plan(diagnose, candidates=[{"levels": {"cooler": 7}, "failure_mode_ids": []}]), board))
    assert any("unknown component 'fan'" in p for p in check_plan(plan(diagnose, candidates=[{"levels": {"fan": 1}, "failure_mode_ids": []}]), board))
    assert any("rank 2 is not among the 1 candidates of yours" in p for p in check_plan(plan(Step("diagnose", {"rank": 2}, {}), candidates=[cooler]), board))
    # any sensor group may name a declaration after a reading, and the finish still needs the reading after the last close
    assert check_plan(plan(Step("declare", {"sensor_group": "tier3", "cycles": 1})), board) == []
    assert any("verifies the repair" in p for p in check_plan(plan(diagnose, order, Step("wait", {"cycles": 4}), Step("finish"), candidates=[cooler]), board))
    # the candidates the model listed become the board's, ranked in its order, with the healthy levels filled in
    board.set_candidates([cooler])
    assert board.candidates()[0]["combination"] == {"cooler": 20, "valve": 100, "pump": 0, "accumulator": 130} and "## Your candidates" in board.render()
    assert check_plan(plan(diagnose, order, *tail), board) == []
    bare = Board(server=False)
    bare.set_profile(board.profile)
    bare.status = {"time": 0, "open_work_orders": [], "episode_over": False}
    assert any("counts only when it cites a reading" in p for p in check_plan(plan(Step("declare", {"sensor_group": "tier3", "cycles": 1})), bare))


def _serve_bands(monkeypatch, make_band):
    """The rig's action server with every normal range of the profile replaced by make_band(low, high), so the test plays the agent on
    normal ranges the environment serves and not on any the agent could hold."""
    call = InProcessServers.call

    async def served(self, path, tool, args):
        text = await call(self, path, tool, args)
        if tool != "get_asset_profile":
            return text
        envelope = json.loads(text)  # the receipt fields and, under result, the profile
        profile = envelope["result"]
        assert profile.get("normal_ranges"), "the profile serves the normal ranges"
        profile["normal_ranges"] = {name: dict(zip(("low", "high"), make_band(band["low"], band["high"]))) for name, band in profile["normal_ranges"].items()}
        return json.dumps(envelope)
    monkeypatch.setattr(InProcessServers, "call", served)


def test_the_band_verification_follows_the_bands_the_environment_serves_and_not_a_copy_of_its_own(tmp_path, monkeypatch):
    """The agent judges a repair on the normal ranges the profile serves, so a change of the served normal ranges changes
    its verdicts with no change of the agent. With every normal range widened to cover any value the repair of the healthy pump on seed 3,
    which the served normal ranges refute, verifies as consistent; with every normal range narrowed to a point no reading can hit, the repair of
    the faulty cooler, which the served normal ranges verify, is refuted. An agent that held the normal ranges itself, the old percentiles or the
    new minimum to maximum, would give the served verdicts in both cases and fail here."""
    _serve_bands(monkeypatch, lambda low, high: (-1e9, 1e9))
    ep_dir, row, _backend = _play_without_server(tmp_path / "wide", bands_plans(order_first=lambda start: {"pump": 1}))
    plans = read_json(ep_dir / "plans.json")
    assert read_json(ep_dir / HIDDEN / "log.json")[0]["hidden"] == [20, 100, 0, 130] and row["causal_calls"] == 0
    assert plans["board"]["refuted"] == [] and plans["board"]["verifications"]
    assert [v["consistent"] for v in plans["board"]["verifications"].values()][0] is True
    assert plans["board"]["reads"] and all(outside == 0 for entry in plans["board"]["reads"] for _n, _t, outside, _banded in entry["cycles_outside"])
    _serve_bands(monkeypatch, lambda low, high: (1e9, 1e9))
    ep_dir, row, _backend = _play_without_server(tmp_path / "narrow", bands_plans())
    plans = read_json(ep_dir / "plans.json")
    verdicts = [v["consistent"] for v in plans["board"]["verifications"].values()]
    assert verdicts and verdicts[0] is False and [r["components"] for r in plans["board"]["refuted"]][0] == ["cooler"]
    assert all(outside == banded > 0 for entry in plans["board"]["reads"] for _n, _t, outside, banded in entry["cycles_outside"])


def test_the_agent_package_holds_no_band_of_its_own():
    """The normal ranges come from the profile alone (board.normal_ranges), so no module of the agent names a quantile, a percentile or the
    constant of CausalDiagnosis/datasets/dataset_loader/evidence_source.py that defines them, and a change of that constant reaches the agent through the profile."""
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"BAND_QUANTILES|quantile|percentile|from envs|import envs", text), path.name
    board = Board(server=False)
    assert board.normal_ranges() == {}
    board.set_profile({"components": [], "levels": {}, "healthy": {}, "normal_ranges": {"PS1_mean": {"low": 1.0, "high": 2.0}}})
    assert board.normal_ranges() == {"PS1_mean": {"low": 1.0, "high": 2.0}}


def test_the_band_verdict_takes_the_bands_and_refutes_a_repair_when_any_channel_mean_lies_outside():
    """The verification without the CausalDiagnosis MCP is one function of the normal ranges it is given. A reading is
    consistent with a restored rig when every channel mean with a normal range lies inside it, the ends included, in every cycle after the
    close; a field without a normal range, a std summary, is not judged; and other normal ranges give other verdicts with no change of the agent."""
    bands = {"PS1_mean": {"low": 1.0, "high": 2.0}, "TS1_mean": {"low": 30.0, "high": 40.0}}
    inside = {"PS1_mean": 1.5, "TS1_mean": 35.0, "PS1_std": 99.0}
    assert band_verdict(bands, [inside]) is True and band_verdict(bands, [inside, dict(inside, PS1_mean=1.0, TS1_mean=40.0)]) is True
    assert band_verdict(bands, [inside, dict(inside, PS1_mean=2.01)]) is False and band_verdict(bands, [dict(inside, TS1_mean=29.9)]) is False
    assert channels_outside(bands, dict(inside, PS1_mean=0.5, TS1_mean=41.0)) == [["PS1_mean", "below"], ["TS1_mean", "above"]]
    assert band_verdict(bands, []) is True and band_verdict({}, [{"PS1_mean": 100.0}]) is True
    assert band_verdict({**bands, "PS1_mean": {"low": 0.0, "high": 3.0}}, [dict(inside, PS1_mean=2.5)]) is True


def test_on_the_min_to_max_bands_every_healthy_recording_passes_and_every_recording_of_a_fault_is_refuted():
    """On every registered asset of the recorded levels kind the public normal range of a channel spans the smallest to the largest
    mean over the stable healthy recordings, so on every sensor group of the asset the normal range verification finds every served
    recording of the healthy combination consistent with a restored asset and refutes every served recording of a combination with
    a fault, on the values a reading returns."""
    from CausalDiagnosis.datasets.dataset_loader import evidence_source as evidence
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, get_profile

    assert evidence.BAND_QUANTILES == (0.0, 1.0), "the normal ranges span the smallest to the largest healthy mean"
    assets = [get_profile(path.parent.name) for path in sorted(ASSETS_DIR.glob("*/asset.json"))]
    assert [p for p in assets if p.evidence_kind == "recorded_levels"]
    for profile in assets:
        if profile.evidence_kind != "recorded_levels":
            continue
        _facts, cycles, blocks = load_world(asset_id=profile.asset_id)
        source = evidence.RecordingSource(profile, cycles, blocks)
        healthy = profile.healthy_state()
        for group, spec in profile.observation_kinds.items():
            for combination, block in source.blocks.items():
                verdicts = {band_verdict(source.normal_ranges, [source.served_values(i, tuple(spec["channels"]))]) for i in block[:evidence.BLOCK_CAP]}
                assert verdicts == {combination == healthy}, (profile.asset_id, group, combination, verdicts)

def test_the_rule_only_switch_left_the_driver_and_the_options_of_a_run():
    parser = run_episodes.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--agent", "causal", "--causal-rule-only"])
    options = run_episodes.causal_options(parser.parse_args(["--agent", "causal"]))
    assert "rule_only" not in options and CausalLoopOptions(**options).label() == "causal"
    assert CausalLoopOptions(rule_only=True).label() == "causal_rule"  # the internal test tool keeps its label


# ----- how a run ends, and what it records of a call without an answer -----

import asyncio  # noqa: E402


def _controller(tmp_path, planner, options=None, config=None, seed=0):
    """An episode folder as the driver starts it and a controller on its launchers, for a test that reads the controller's own
    result. Returns the folder, the controller and the question."""
    facts_path, cycles, blocks = load_world()
    config_name, base = load_config("static")
    config = config or base
    switches = {"branches": True, "single_hypothesis": False, "plan_length": None, "expectations": True, "rule_only": False, **(options or {})}
    runner = EpisodeRunner(config, config_name, "causal", (RIG_SERVER, CAUSAL_SERVER), cycles, blocks, facts_path, "causal-test",
                           {"judge": None, "grade": True, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL}, model=planner.model_id,
                           causal_options=switches)
    ep_dir = tmp_path / f"seed{seed:06d}"
    with mock.patch.object(run_episodes.play_one_episode, "check_servers", lambda launchers, row=None: None):
        started = runner.start(ep_dir, seed, config)  # the launchers and the question come first, whatever follows them
    episode.hand_over(ep_dir / run_episodes.STATE_FILE)  # as the driver's attempt does before run_agent
    planner.start_episode()
    loop_runner = controller.CausalLoopRunner(llm=planner, server_paths=dict(started[0]), options=CausalLoopOptions(**switches))
    return ep_dir, loop_runner, started[1]


@contextlib.contextmanager
def _idle_hook(stand_in):
    """The driver's idle hook replaced for a controller run without a driver: stand_in() answers whether the episode is over
    after a turn without a call; a turn that acted ends nothing. Yields the list of acted flags the hook was called with."""
    calls = []

    async def hook(acted):
        calls.append(acted)
        return False if acted else bool(stand_in())
    with mock.patch.object(controller, "aiteration_end", hook):
        yield calls


def test_the_result_names_how_the_run_ended(tmp_path):
    def played(folder, plans, options=None, config=None, on_idle=None, **planner):
        ep_dir, loop_runner, question = _controller(tmp_path / folder, ScriptedPlanner(plans, **planner), options, config)
        with _in_process(ep_dir), _idle_hook(on_idle or (lambda: False)):
            return asyncio.run(loop_runner.run(question))

    finished = played("finish", act_now)
    assert (finished.end, finished.end_reason, finished.error) == ("finish", "finish", None)
    # the controller has no limit of its own: a turn without a call goes to the driver's idle hook (the no_call step), and the run
    # stops when that entry reports the end of the episode (here a stand-in that ends it at its third call)
    idle_calls = []
    idle = played("idle", lambda prompt, call: "Still thinking.", orientation="No differential yet.",
                  on_idle=lambda: idle_calls.append(1) or len(idle_calls) == 3)
    assert (idle.end, idle.error, len(idle_calls)) == ("episode_over", None, 3) and len(idle.plans) == 4
    over = played("over", act_now, config=dataclasses.replace(load_config("static")[1], max_steps=2))
    assert (over.end, over.end_reason, over.error) == ("episode_over", "max_steps", None)

    def broken(prompt, call):
        raise RuntimeError("the planner broke")

    raised = played("error", broken)
    assert (raised.end, raised.end_reason, raised.error) == ("error", None, "RuntimeError: the planner broke")
    assert raised.plans and raised.trajectory  # what was recorded before the failure is kept


def test_a_cut_run_returns_what_it_recorded_with_the_end_cancelled(tmp_path):
    """The caller cuts the run while a reading is in flight. That the unwinding of the sessions hands a cut on as a cancellation
    is the rule of the session module and its tests."""
    ep_dir, loop_runner, question = _controller(tmp_path, ScriptedPlanner(reads_then_acts))

    async def play(servers):
        reached = asyncio.Event()

        class Hanging:
            async def call(self, path, tool, args):
                if tool != "read_sensors":
                    return await servers.call(path, tool, args)
                reached.set()
                await asyncio.sleep(60)  # the reading never answers

        with mock.patch.object(controller, "open_sessions", sessions_in_process(lambda path: Hanging())):
            task = asyncio.ensure_future(loop_runner.run(question))
            waiting = asyncio.ensure_future(reached.wait())
            await asyncio.wait({task, waiting}, return_when=asyncio.FIRST_COMPLETED)
            if not reached.is_set():  # the run ended before its reading: fail with its result rather than wait for ever
                waiting.cancel()
                pytest.fail(f"the run ended before the reading it should be cut in: {task.result()!r}")
            task.cancel()
            return await task, task.cancelling()

    with _in_process(ep_dir) as servers:
        result, cancelling = asyncio.run(play(servers))
    assert (result.end, result.error, result.answer, cancelling) == ("cancelled", None, "", 0)
    tools = [t["tool"] for t in result.trajectory]
    assert tools[0] == "get_asset_profile" and tools.count("read_sensors") == 1 and tools[-1] == "read_sensors"
    # the calls before the cut are kept as they were answered, and the one in flight is recorded as given up with the run
    assert [t["error"] for t in result.trajectory[:-1]] == [None] * (len(tools) - 1)
    assert (result.trajectory[-1]["error"], result.trajectory[-1]["response"]) == ("RunStopped: cancelled", "")
    assert result.plans and result.plans[-1]["steps"] == []  # the plan whose read was cut ran no step to its end


def test_a_call_without_an_answer_is_recorded_as_a_transport_error(tmp_path):
    ep_dir, loop_runner, question = _controller(tmp_path, ScriptedPlanner(reads_then_acts))

    gone = "get_fix_history of the stand-in got no answer: ConnectionError: the server is gone"
    with _in_process(ep_dir) as servers:
        class Gone:
            async def call(self, path, tool, args):
                if tool == "get_fix_history":
                    raise ToolTransportError(gone)
                return await servers.call(path, tool, args)

        with mock.patch.object(controller, "open_sessions", sessions_in_process(lambda path: Gone())):
            result = asyncio.run(loop_runner.run(question))
    failed = [t for t in result.trajectory if t["tool"] == "get_fix_history"]
    assert failed and all((t["error"], t["response"]) == (f"ToolTransportError: {gone}", "") for t in failed)
    assert all(t["error"] is None for t in result.trajectory if t["tool"] == "get_asset_profile")
    # the run went on, since the controller reads a call without an answer as a refused step
    assert result.error is None and result.end in ("finish", "episode_over")
    assert any(f"ToolTransportError: {gone}" in (s.get("note") or "") for p in result.plans for s in p["steps"])


def test_the_request_settings_of_the_controller_hold_every_key_and_the_values_of_its_backend():
    from CausalDiagnosis.utils.llm_backends import EndpointChatBackend

    backend = EndpointChatBackend("tokenrouter/granite", base_url="http://agent.example/v1", api_key="k", max_tokens=1024)
    settings, own = controller.request_settings(backend), backend.request_settings(controller.TEMPERATURE)
    assert set(settings) == {"decoding", "transport"}
    assert set(settings["decoding"]) == {"route", "max_tokens", "min_tokens", "temperature", "stop", "no_stop_trim", "extra_body"}
    assert set(settings["transport"]) == {"retries", "retry_status", "retry_delay_s", "timeout_s", "connect_timeout_s", "deadline", "sessions",
                                          "tool_timeout_s"}
    assert settings["decoding"] == own["decoding"] and (settings["decoding"]["temperature"], settings["decoding"]["max_tokens"]) == (0.0, 1024)
    assert settings["transport"] == {**own["transport"], "sessions": SESSIONS, "tool_timeout_s": None, "deadline": controller.DEADLINE_RULE}
    assert controller.request_settings(backend, temperature=0.7)["decoding"]["temperature"] == 0.7


# ----- the advice of the server and the arms of the agent's ablations (item Q of the agents design) -----

def _advised(advice=None, earlier=None, **settings):
    """The synthetic board whose ranking carries the advice of the server and, for a run under an arm, the block e10_rule."""
    board = _synthetic_board(read_first=True, second=True)
    board.ranking["elimination"] = [{"sensor_group": g, "cycles": 1, "price": p, "cost": p + 1, "ec2": cut, "ec2_per_cost": cut / (p + 1)}
                                    for g, p, cut in (("tier1", 1, 0.1), ("tier2", 3, 0.0), ("tier3", 10, 0.0))]
    if advice is not None:
        board.ranking["advice"] = advice
    if earlier is not None:
        board.ranking["e10_rule"] = earlier
    for name, value in settings.items():
        setattr(board, name, value)
    return board


ADVISED_READ = {"action": "read", "sensor_group": "tier3", "cycles": 5, "rank": None, "declare_above": {}, "text": "read tier3 for 5 cycles before a diagnosis"}
ADVISED_RANK_2 = {"action": "diagnose", "sensor_group": None, "cycles": None, "rank": 2, "declare_above": {},
                  "text": "diagnose rank 2 and create one work order for each of its faulty components (valve)"}
EARLIER_RULE = {"read_first": True, "margin": 0.0, "separating_test": {"sensor_group": "tier2", "cycles": 1, "price": 3, "duration": 1, "cost": 4,
                                                                       "expected_loss": 8},
                "read_batch": [], "batch_cap": None, "reserve_plans": 1, "text": "e10_rule, the earlier rule"}


def first_step(board, reads="gain"):
    step = rule_plan(board, 6, reads=reads).steps[0]
    return step.kind, dict(step.args)


def test_an_arm_keeps_its_own_choice_and_the_registered_agent_follows_the_advice():
    """A ranking that advises one sensor group for several cycles, while the separating test and the edge cutting read of the
    earlier rule name other reads: the main settings plan the advised read with its cycles, each read arm plans its own read
    and prints the advice of the server as one line. A ranking that advises rank 2 (a planted reply, the server advises rank 1):
    every setting diagnoses rank 2, a run with a read arm included."""
    assert first_step(_advised(ADVISED_READ)) == ("read", {"sensor_group": "tier3", "cycles": 5})
    for arm, own in (("separating", "tier2"), ("ec2", "tier1")):
        board = _advised(ADVISED_READ, EARLIER_RULE, read_advice=arm)
        assert first_step(board, reads=arm) == ("read", {"sensor_group": own, "cycles": 1})
        assert "Advice of the server: read tier3 for 5 cycles before a diagnosis" in advice(board)
    assert first_step(_advised(ADVISED_RANK_2)) == ("diagnose", {"rank": 2})
    no_read = dict(EARLIER_RULE, read_first=False, separating_test=None)
    assert first_step(_advised(ADVISED_RANK_2, no_read, read_advice="separating"), reads="separating") == ("diagnose", {"rank": 2})


def test_a_ranking_without_advice_is_planned_as_before_under_every_setting():
    """Without advice every setting plans the lines of before: here the separating test of the ranking, since the ranking
    lists no read batch, and no line quotes an advice that is not there."""
    for reads in ("gain", "separating", "ec2"):
        board = _synthetic_board(read_first=True, second=True)
        board.read_advice = reads
        assert first_step(board, reads=reads) == ("read", {"sensor_group": "tier2", "cycles": 1})
        assert not any(line.startswith("Advice of the server") for line in advice(board))


def test_the_board_shows_the_advice_of_the_server_and_the_rank_call_asks_for_its_form():
    """The board prints the advice of the server where the ranking carries it, the main settings ask for the compact form and an
    arm for the long form, and the binding names the form only when the controller gives one."""
    board = _advised(ADVISED_READ)
    assert "Advice of the server: read tier3 for 5 cycles before a diagnosis" in board.render()
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import ranking_detail
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import HYDRAULIC_RIG, bind
    assert ranking_detail(board) == "compact"
    assert ranking_detail(_advised(ADVISED_READ, read_advice="ec2")) == "summary"
    assert bind(HYDRAULIC_RIG, "rank")[2] == {}  # no top_k from the agent: the server's default of five applies
    assert bind(HYDRAULIC_RIG, "rank", branch_detail="compact")[2] == {"branch_detail": "compact"}


# ----- one session per server and episode -----

def test_the_controller_opens_its_servers_once_per_episode_through_the_session_module(tmp_path):
    ep_dir, loop_runner, question = _controller(tmp_path, ScriptedPlanner(reads_then_acts))
    opened, routed = [], []
    with _in_process(ep_dir) as servers:
        stand_in = sessions_in_process(lambda path: servers)

        @contextlib.asynccontextmanager
        async def spy(server_paths):
            opened.append(dict(server_paths))
            async with stand_in(server_paths) as sessions:
                call = sessions.call

                async def counted(server, tool, arguments):
                    routed.append((server, tool))
                    return await call(server, tool, arguments)

                sessions.call = counted
                yield sessions

        with mock.patch.object(controller, "open_sessions", spy):
            result = asyncio.run(loop_runner.run(question))
    assert opened == [dict(loop_runner._server_paths)] and result.error is None
    assert routed == [(t["server"], t["tool"]) for t in result.trajectory] and len(routed) > len(opened[0])


@pytest.mark.skipif("wait_for_notification" not in contract_tools(RIG_SERVER), reason="the real action server lacks the waiting tool")
def test_an_episode_over_stdio_starts_one_process_per_server(tmp_path, monkeypatch):
    """The driver plays the agent on the real launchers. Only the processes that the controller starts while it runs are
    counted, so the driver's own listing of the tools does not enter the count."""
    import mcp.client.stdio as stdio

    started, running, real_client, real_run = [], [], stdio.stdio_client, controller.CausalLoopRunner.run

    def counting(server, *args, **kwargs):
        if running:
            started.append(Path(server.args[-1]).name)
        return real_client(server, *args, **kwargs)

    async def run(self, question):
        running.append(True)
        try:
            return await real_run(self, question)
        finally:
            running.pop()

    monkeypatch.setattr(stdio, "stdio_client", counting)
    monkeypatch.setattr(controller.CausalLoopRunner, "run", run)
    planner = ScriptedPlanner(act_now)
    server, requests = _planner_endpoint(planner)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url", base, "--seeds", "1",
                           "--out", str(tmp_path / "run")])
    finally:
        server.shutdown()
    run_dir = next(p for p in (tmp_path / "run").iterdir() if p.is_dir())
    import csv
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    launchers = sorted(p.name for p in (run_dir / "seed000000").glob("*_server.py"))  # common.write_launchers names them so
    assert row["error"] == "" and int(row["tool_calls"]) > len(launchers) == 2
    assert sorted(started) == launchers  # one process per server of the row, where a process per call started one for every call


@pytest.mark.parametrize("breaks", ["at_the_opening", "under_the_caller"])
def test_a_transport_that_breaks_is_the_error_of_the_run(tmp_path, breaks):
    """The block of the session module raises ToolTransportError for a server that does not start and for a transport that
    breaks under the caller, which it hands on after the client cancelled the caller. The run ends with that error, and a call
    without an answer before it is a refused step (test_a_call_without_an_answer_is_recorded_as_a_transport_error)."""
    ep_dir, loop_runner, question = _controller(tmp_path, ScriptedPlanner(reads_then_acts))
    text = ("the server hydraulic_rig did not start: FileNotFoundError: no launcher" if breaks == "at_the_opening"
            else "the sessions broke under the caller: UnicodeDecodeError: the stream holds no text")
    with _in_process(ep_dir) as servers:
        stand_in = sessions_in_process(lambda path: servers)

        @contextlib.asynccontextmanager
        async def broken(server_paths):
            if breaks == "at_the_opening":
                raise ToolTransportError(text)
            async with stand_in(server_paths) as sessions:
                call = sessions.call

                async def cancels_on_a_reading(server, tool, arguments):
                    if tool == "read_sensors":
                        raise asyncio.CancelledError()  # the client cancels the caller when one of its own tasks fails
                    return await call(server, tool, arguments)

                sessions.call = cancels_on_a_reading
                try:
                    yield sessions
                except asyncio.CancelledError:
                    raise ToolTransportError(text) from None

        with mock.patch.object(controller, "open_sessions", broken):
            result = asyncio.run(loop_runner.run(question))
    assert (result.end, result.error) == ("error", f"ToolTransportError: {text}")
    if breaks == "at_the_opening":
        assert result.trajectory == []
    else:
        assert result.trajectory[0]["tool"] == "get_asset_profile" and result.trajectory[-1]["tool"] == "read_sensors"


# ----- the board's model of the surface and the model's own candidates -----

def test_an_order_stays_open_until_the_surface_says_it_closed():
    """A status that lists no open orders, as on the AssetOpsBench servers, closes no order: it stays open through any number
    of calls and closes with its notification. A status that lists the open orders closes the order it leaves out."""
    board = _synthetic_board(read_first=False)
    del board.status["open_work_orders"]  # the envelope of these servers names no open order
    board.add_order({"work_order": {"id": "wo-1", "component": "pump", "created_t": 1, "expected_close_t": None, "status": "open"}}, "pred/b1")
    for t in (2, 3, 4):
        board.update_status({"time": t, "budget_left": 250, "steps_left": 50})
    assert board.orders["wo-1"]["status"] == "open" and [o["id"] for o in board.open_orders()] == ["wo-1"]
    board.add_notifications([{"event": "work_order_closed", "id": "wo-1", "component": "pump", "t": 6}])
    assert board.orders["wo-1"]["status"] == "closed" and board.orders["wo-1"]["closed_t"] == 6
    listed = _synthetic_board(read_first=False)
    listed.add_order({"work_order": {"id": "wo-2", "component": "pump", "created_t": 1, "expected_close_t": 6, "status": "open"}}, "pred/b1")
    listed.update_status({"time": 2, "open_work_orders": [{"id": "wo-2", "component": "pump", "expected_close_t": 6}]})
    assert listed.orders["wo-2"]["status"] == "open"
    listed.update_status({"time": 3, "open_work_orders": []})
    assert listed.orders["wo-2"]["status"] == "closed"


def _asset_configs():
    """(profile as the agent reads it, EpisodeConfig) of every registered asset and of the tiny asset of the asset tests, each with
    its own priorities, lead times and repair downtimes, so that the rule of a visit is checked on more than one asset."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, load_profile
    from CausalDiagnosis.datasets.dataset_loader.profile_data import processes_block
    paths = sorted(ASSETS_DIR.glob("*/asset.json")) + [ASSETS_DIR / "tests" / "tiny_asset.json"]
    out = []
    for path in paths:
        config = load_profile(path).default_config()
        out.append(({"components": list(config.components), "durations": dict(config.durations), "processes": processes_block(config)}, config))
    return out


def test_the_close_of_a_visit_is_estimated_as_the_environment_schedules_it():
    """For every registered asset, every set of its components and both orders of creation, the closes the board estimates for
    the orders of one cycle are those of the environment's schedule of their visit."""
    import itertools
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import visit_closes
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import schedule_visit
    checked = 0
    for profile, config in _asset_configs():
        components = list(config.components)
        for size in range(1, len(components) + 1):
            for chosen in itertools.combinations(components, size):
                for members in (list(chosen), list(reversed(chosen))):
                    for created_t in (0, 7):
                        scheduled = {members[i]: close for i, _start, close in schedule_visit(config, created_t, [(c, "standard") for c in members])}
                        assert visit_closes(profile, [(c, created_t, None) for c in members]) == scheduled, (config.asset_id, members, created_t)
                        checked += 1
    assert checked and visit_closes(_asset_configs()[0][0], []) == {}


def test_a_read_takes_no_time_where_recorded_cycles_are_served():
    """On a board of the AssetOpsBench surface, which serves recorded cycles, a read takes no time: near the horizon a plan of two
    reads and a finish passes the check. The orders on either side of a read are created in one cycle, yet the read ends the run
    of orders, so the second order opens a visit of its own and pays its charge, on this surface as on one that plays its cycles
    (servers/public_rules/visits.py); without the read the two orders form one visit."""
    board = _synthetic_board(read_first=False)
    board.reads_take_time = False
    board.status["time"] = board.horizon - 2
    reads = Plan(branch=1, mode="test_first", why="x", steps=[Step("read", {"sensor_group": "tier1", "cycles": 1}), Step("read", {"sensor_group": "tier2", "cycles": 1}),
                                                              Step("finish")])
    assert board.read_duration(5) == 0 and check_plan(reads, board) == []
    board.status["time"] = 1
    board.diagnosis = {"diagnosis": dict(board.top()["combination"]), "evidence_ids": []}
    board.status["budget_left"] = board.plan_price(["pump", "valve"]) + board.read_price("tier1") + board.visit_charge - 1  # a second visit charge exceeds it
    visit = Plan(branch=1, mode="act_now", why="x", steps=[Step("order", {"component": "pump", "prediction_id": "pred/b1"}),
                                                          Step("read", {"sensor_group": "tier1", "cycles": 1}),
                                                          Step("order", {"component": "valve", "prediction_id": "pred/b1"})])
    assert any("exceed the budget left" in p for p in check_plan(visit, board))
    board.status["budget_left"] += board.visit_charge  # the second visit charge fits
    assert check_plan(visit, board) == []
    board.reads_take_time = True
    board.status["budget_left"] -= board.visit_charge
    assert any("exceed the budget left" in p for p in check_plan(visit, board))
    one_visit = Plan(branch=1, mode="act_now", why="x", steps=[visit.steps[0], visit.steps[2], visit.steps[1]])
    assert check_plan(one_visit, board) == []


def _recorded_surface(board, oldest, newest):
    """An AssetOpsBench surface after its opening, the record of the board's groups known from oldest to newest, and the
    board's status as the envelope of the extent leaves it."""
    surface = AobSurface()
    surface.site, surface.asset_id, surface.seconds_per_cycle = "MAIN", "hyd_1", 60
    surface.sensor_groups = {g: list(fields) for g, fields in board.profile["sensor_groups"].items()}
    surface.anchor, surface.oldest, surface.newest = (0, datetime(2026, 1, 1)), oldest, newest
    board.status.update(recorded_from=oldest, recorded_until=newest)
    return surface


def _history(surface, cycles, field="CE_CoolingEfficiency_percent_1Hz"):
    """The text of a history answer with one row for each of the cycles."""
    return json.dumps({"observations": [{"asset_id": surface.asset_id, "timestamp": surface.timestamp(c), field: 40.0 + c} for c in cycles]})


def test_a_cycle_without_a_record_is_not_offered_or_priced_again():
    """A window of three cycles comes back with one row: the two cycles without a row hold no record of the group, so the next
    window leaves them out, the envelope names them, the price of a read counts none of them, and new_cycles is zero once nothing
    else is unread."""
    board = _synthetic_board()  # the board has paid for cycle 0 of tier1
    surface = _recorded_surface(board, -2, 3)
    server, tool, args = surface.bind("read", board, sensor_group="tier1", cycles=3)
    assert (tool, args["limit"]) == ("history", 3) and surface.windows["tier1"] == (1, 3)
    data = surface.parse("read", tool, _history(surface, [3]), board, sensor_group="tier1", cycles=3)
    board.update_status(data)
    board.add_reading(data["result"])
    assert data["unrecorded"] == {"tier1": [1, 2]} and board.unrecorded("tier1") == {1, 2}
    assert board.new_cycles("tier1", 3) == 2 and board.read_price("tier1", 3) == 2 * board.read_price("tier1", 1)
    server, tool, args = surface.bind("read", board, sensor_group="tier1", cycles=3)
    assert (args["limit"], surface.windows["tier1"]) == (2, (-2, -1))  # past the paid cycle 0 and the unrecorded cycles 1 and 2
    data = surface.parse("read", tool, _history(surface, [-2, -1]), board, sensor_group="tier1", cycles=3)
    board.update_status(data)
    board.add_reading(data["result"])
    assert board.new_cycles("tier1", 3) == 0 and board.read_price("tier1", 3) == 0 and board.unrecorded("tier1") == {1, 2}


def _recorded_reader(clock, rows):
    """A controller on the AssetOpsBench surface without the diagnostics server whose board has read every recorded cycle of
    tier1, the record ending at cycle 0 with the clock at 1, and whose history tool answers the rows of the given cycles."""
    board = Board(server=False, reads_take_time=False, evidence_form=AobSurface.evidence_form)
    board.set_profile(_synthetic_board().profile)
    board.clock_runs = clock
    surface = _recorded_surface(board, 0, 0)
    surface.clock = clock
    board.status = {"time": 1, "recorded_from": 0, "recorded_until": 0}
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 1, "t": 0, "label": surface.timestamp(0), "values": {"CE_CoolingEfficiency_percent_1Hz": 40.0}}]})
    calls = []

    class Sessions:
        async def call(self, server, tool, arguments):
            calls.append((server, tool, dict(arguments)))
            return _history(surface, rows), False

    loop_runner = controller.CausalLoopRunner(llm=ScriptedPlanner(lambda prompt, call: ""), server_paths={"iot": "iot_server.py", "task": "task_server.py"},
                                                surface=surface)
    loop_runner.board, loop_runner.trajectory, loop_runner._served, loop_runner._sessions = board, [], {}, Sessions()
    return loop_runner, calls


def _read_one(loop_runner):
    step = Step("read", {"sensor_group": "tier1", "cycles": 1}, {}, [])
    return asyncio.run(loop_runner._read_step(step, episode_result.PlanRecord(index=1, source="model", plan={}, candidates=[]), 1))


def test_a_read_with_no_unread_cycle_known_is_answered_by_the_server_where_a_clock_runs():
    """The board's record ends at cycle 0 and every cycle of it is read. Where a clock runs the read is made on the open window
    after the newest known cycle and serves the cycle recorded since, and a read that serves nothing names the end of the record;
    either made a call, so the turn is not idle. Where no clock runs no cycle can have been recorded since, and no call is made."""
    import asyncio
    loop_runner, calls = _recorded_reader(True, [1])
    outcome = _read_one(loop_runner)
    assert [c[:2] for c in calls] == [("iot", "history")] and "end" not in calls[0][2] and calls[0][2]["start"] == loop_runner.surface.timestamp(1)
    assert outcome.status == "done" and loop_runner.board.observations == 2 and loop_runner.board.status["recorded_until"] == 1
    loop_runner, calls = _recorded_reader(True, [])
    outcome = _read_one(loop_runner)
    assert len(calls) == len(loop_runner.trajectory) == 1 and outcome.status == "surprise" and "the record ends at cycle 0" in outcome.note
    loop_runner, calls = _recorded_reader(False, [1])
    outcome = _read_one(loop_runner)
    assert calls == [] and outcome.status == "surprise" and "every recorded cycle of tier1" in outcome.note


def _ranked_rows(prompt):
    """The ranks the ranking of the board shows."""
    return [int(m) for m in re.findall(r"^\| (\d+) \|", _section(prompt, "## Ranking"), re.MULTILINE)]


OWN = {"cooler": 3, "valve": 73, "pump": 2, "accumulator": 90}  # every component faulty, a combination no ranking within the fault rule holds


def own_candidate_plans(rank_of_own):
    """A planner that lists OWN as a candidate of its own and diagnoses it at the rank rank_of_own(prompt) gives, then finishes."""
    def plans(prompt, call):
        if "Commitment: none" in prompt:
            rank = rank_of_own(prompt)
            return json.dumps({"branch": rank, "mode": "act_now", "why": "scripted: a candidate of its own", "candidates": [{"levels": OWN, "failure_mode_ids": []}],
                               "steps": [{"kind": "diagnose", "args": {"rank": rank}, "expect": {}, "evidence": []}, {"kind": "finish"}]})
        return plan_json([{"kind": "finish", "args": {}, "expect": {}, "evidence": []}])
    return plans


def test_with_the_server_a_plan_may_diagnose_a_candidate_of_its_own(tmp_path):
    """The model lists a combination the ranking does not hold and diagnoses its rank, the one after the ranked candidates: the
    diagnosis carries that combination, and the board keeps it among the model's candidates."""
    ep_dir, row, _backend = _play(tmp_path, ScriptedPlanner(own_candidate_plans(lambda prompt: len(_ranked_rows(prompt)) + 1)), {})
    _assert_episode_files(ep_dir, row)
    diagnosis = next(t for t in read_json(ep_dir / "trajectory.json") if t["tool"] == "submit_diagnosis")
    assert {c: diagnosis["tool_args"][c] for c in COMPONENTS} == OWN
    first = decision_plans(read_json(ep_dir / "plans.json"))[0]
    assert first["check_problems"] == [] and first["candidates"][-1].get("own") is True and first["candidates"][-1]["combination"] == OWN
    board = read_json(ep_dir / "plans.json")["board"]
    assert [c["combination"] for c in board["candidates"]] == [OWN] and board["diagnosis"] is not None


def test_a_diagnosis_is_possible_when_every_ranking_failed(tmp_path, monkeypatch):
    """The ranking tool answers an error on every call, so no ranking exists: the model's own candidate is the only one, rank 1,
    and its diagnosis is submitted."""
    call = InProcessServers.call

    async def no_ranking(self, path, tool, args):
        if tool == "rank_root_causes":
            return json.dumps({"error": "the ranking failed"})
        return await call(self, path, tool, args)
    monkeypatch.setattr(InProcessServers, "call", no_ranking)
    ep_dir, row, _backend = _play(tmp_path, ScriptedPlanner(own_candidate_plans(lambda prompt: 1)), {})
    assert row["error"] == "", row["error"]  # no ranking stored a prediction, so the episode holds no predictions.json
    trajectory = read_json(ep_dir / "trajectory.json")
    diagnosis = next(t for t in trajectory if t["tool"] == "submit_diagnosis")
    assert {c: diagnosis["tool_args"][c] for c in COMPONENTS} == OWN and "refused" not in diagnosis["response"]
    assert "rank_root_causes" in [t["tool"] for t in trajectory] and read_json(ep_dir / "plans.json")["board"]["notes"]


# ----- the threshold card left the agent, and the prompt says of the advice what the board shows -----

def test_the_options_of_the_agent_hold_no_threshold_card():
    """No switch that no registered run sets stays: the options of the agent and the switches of the server name no field of the
    threshold card, the label of the default options carries no suffix, the launcher writer takes no field of a card, and the
    package holds no module of it."""
    import dataclasses
    import inspect
    card_fields = {"cost_tolerance", "reserve_plans", "e10_margin", "batch_cap"}
    assert not {f.name for f in dataclasses.fields(CausalLoopOptions)} & card_fields and not set(agent_options.SERVER_SWITCHES) & card_fields
    assert importlib.util.find_spec("CausalDiagnosis.agents.Causal_Diagnosis_Agent.threshold_card") is None
    assert "thresholds" not in inspect.signature(write_launchers).parameters
    plain = run_episodes.causal_options(run_episodes.build_parser().parse_args(["--agent", "causal"]))
    assert not set(plain) & card_fields and run_episodes.agent_label_of("causal", "m", plain) == "causal_m"


def test_the_reserve_is_the_plan_of_the_first_candidate():
    """Guard of unchanged behavior: the order reserve keeps the completion of the repair plan of the first candidate: the
    diagnosis, the pump's order, the waits of its visit (it begins at cycle 4 and closes at 6, so one wait ends at 5 and one at
    6) with the change summary, the verifying read and finish, and their price. A reserve of two plans fails it."""
    board = _synthetic_board(read_first=False, second=True)
    assert order_reserve(board) == board.plan_price(["pump"]) + board.read_price(board.verification_group(["pump"]))
    assert order_reserve_calls(board) == 1 + 1 + 3 + 1 + 1


def test_the_prompt_says_of_the_advice_what_the_board_shows():
    """For the main settings and for each arm, the sentences of the system prompt on the advice hold on the board of that run:
    the advised reads are the read of the server or of the arm, the advised repair plan is the rank of the server, and the board
    quotes the advice of the server as one line."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import ARMS
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import READ_ARM_RULE, READ_RULE
    for reads in ("gain", "separating", "ec2"):
        prompt = system_prompt(6, reads=reads)
        read_rule = READ_RULE if reads == "gain" else READ_ARM_RULE.replace("<ARM>", ARMS[reads])
        assert read_rule in prompt and REPAIR_RULE in prompt and SERVER_ADVICE_RULE in prompt
        read_lines = advice(_advised(ADVISED_READ, EARLIER_RULE, read_advice=reads))
        rank_lines = advice(_advised(ADVISED_RANK_2, EARLIER_RULE, read_advice=reads))
        read_line = "Advised reads, the read of the server: tier3 x5." if reads == "gain" else f"Advised reads, {ARMS[reads]}: "
        assert any(line.startswith(read_line) for line in read_lines), read_lines
        rank_line = "Advised repair plan, the advice of the server: act on rank 2."
        assert rank_line in rank_lines, rank_lines
        assert f"Advice of the server: {ADVISED_READ['text']}" in read_lines and f"Advice of the server: {ADVISED_RANK_2['text']}" in rank_lines


# ----- the prompts, the board sections, the reply, the asset words, the receipt, the loop and the healthy start -----

def test_every_prompt_of_an_episode_holds_the_task_message_with_its_block(tmp_path, monkeypatch):
    """The task message of the episode, the briefing with a block appended, stands once in every prompt: the orientation, every
    planning turn, the last one included, and the summary."""
    block = "An appended block of the episode: keep every read short."
    monkeypatch.setattr(run_episodes.play_one_episode, "with_task_block", lambda question, _block: f"{question}\n\n{block}")
    planner = ScriptedPlanner(act_now)
    ep_dir, row, _backend = _play(tmp_path, planner, {})
    assert row["error"] == "", row["error"]
    task = f"{(ep_dir / 'question.txt').read_text(encoding='utf-8')}\n\n{block}"
    assert all(any(heading in p for p in planner.prompts) for heading in (ORIENTATION_HEADING, PLANNING_HEADING, SUMMARY_HEADING))
    assert all(prompt.count(task) == 1 for prompt in planner.prompts)


def test_the_board_with_the_server_shows_every_section_of_the_board_without_it():
    """One state of the asset, the failure modes and a reading outside its normal range, rendered without the server and with it: every
    section of the board without the server stands on the board with the server, the failure modes and the channels outside
    their normal range among them."""
    without = _synthetic_board_without_server()
    with_server = Board()
    with_server.set_profile(without.profile)
    with_server.set_failure_modes({"failure_modes": without.failure_modes})
    with_server.status = dict(without.status)
    with_server.add_reading({"sensor_group": "tier1", "served": [{"n": 1, "t": 0, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 30.0,
                                                                                          "CE_CoolingEfficiency_percent_1Hz_std": 0.1}}]})

    def headings(board):
        return {line.split(" (")[0] for line in board.render().splitlines() if line.startswith("## ")}
    assert {"## Failure modes", "## Channels outside their normal range"} <= headings(without) <= headings(with_server)
    assert "fmea/src/cooler_reduced_efficiency" in with_server.render() and "CE_CoolingEfficiency_percent_1Hz_mean below" in with_server.render()


def test_a_diagnosis_below_rank_one_without_an_expectation_expects_the_candidate_on_top(tmp_path):
    """A plan that diagnoses rank 2 and names no expectation: the step expects the candidate on top, which rank 1 is, so the
    diagnosis is submitted."""
    finish = {"kind": "finish", "args": {}, "expect": {}, "evidence": []}

    def rank_two(prompt, call):
        if call == 1:
            return plan_json([{"kind": "diagnose", "args": {"rank": 2}, "expect": {}, "evidence": []}, finish])
        return plan_json([finish])
    ep_dir, row, _backend = _play(tmp_path / "cost", ScriptedPlanner(rank_two), {})
    assert row["error"] == "" and "submit_diagnosis" in _tools(ep_dir)


def test_a_reply_cut_at_the_token_cap_is_not_parsed_and_the_retry_says_so(tmp_path):
    """The first plan of the episode comes back with the finish reason length although its JSON object is complete: nothing of it
    runs, the record names the cut, and the next prompt quotes the sentence of the cut."""
    from CausalDiagnosis.utils.llm_backends import ChatResult

    class CutOnce(ScriptedPlanner):
        cut = False

        def generate_with_usage(self, prompt, temperature=0.0):
            text = self.generate(prompt, temperature)
            first = PLANNING_HEADING in prompt and opening_reply(prompt) is None and RETRY_HEADING not in prompt and not self.cut
            self.cut = self.cut or first
            return ChatResult(text=text, finish_reason="length" if first else "stop")
    ep_dir, row, backend = _play(tmp_path, CutOnce(act_now), {})
    assert row["error"] == "", row["error"]
    at = next(i for i, c in enumerate(backend.calls) if c["finish_reason"] == "length")
    assert controller.CUT_REPLY in backend.calls[at + 1]["prompt"] and RETRY_HEADING in backend.calls[at + 1]["prompt"]
    assert controller.CUT_REPLY in (ep_dir / "plans.json").read_text(encoding="utf-8")


def test_an_incomplete_plan_object_is_reported_as_incomplete():
    """A reply whose plan object breaks off is an incomplete object and not a reply without JSON, also when a complete object
    without steps stands before it, and a complete plan still parses."""
    with pytest.raises(PlanError, match="the JSON object of the plan is not complete"):
        parse_plan('{"mode": "act_now", "steps": [{"kind": "read", "args": {"sensor_group": "tier1"', 6)
    with pytest.raises(PlanError, match="the JSON object of the plan is not complete"):
        parse_plan('{"note": "a first object"} {"mode": "act_now", "steps": [{"kind": "finish"', 6)
    assert parse_plan(json.dumps({"mode": "act_now", "steps": [{"kind": "finish"}]}), 6).steps[0].kind == "finish"


def test_the_texts_of_the_registered_asset_keep_their_words():
    """The driver builds the surface of either server from the text of the asset file, so the texts of the registered asset name
    it by its short noun and give its example channel, and a surface built without them names the asset and gives no example."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import RigSurface
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    config = load_config("static")[1]
    text = get_profile(config.asset_id).text
    for store, kind in ((None, RigSurface), ({"kind": "memory"}, AobSurface)):
        surface = run_episodes.causal_surface(config.asset_id, store)
        assert type(surface) is kind and (surface.noun, surface.example_tag) == (text["short_noun"], text.get("example_channel"))
    surface = run_episodes.causal_surface(config.asset_id)
    assert f"Five kinds act on the {text['short_noun']}:" in system_prompt(6, surface=surface)
    without = system_prompt(6, surface=surface, server=False)
    assert f"its tag such as {text['example_channel']}>" in without and f"healthy {text['short_noun']} shows no channel mean" in without
    plain = system_prompt(6, server=False)
    assert "its tag>" in plain and "healthy asset shows no channel value" in plain


def test_levels_and_component_names_are_taken_as_the_profile_spells_them():
    """A profile whose levels are words and one of whose component names holds a joiner: a level names the profile's level as a
    value or as its written form, a flag names none, an order of that component stays one order, and names of the profile
    joined in one order step still split."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import profile_level
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import parse_step
    profile = {"components": ["inlet/valve", "pump", "fan"], "levels": {"inlet/valve": ["open", "stuck"], "pump": [0, 1, 2], "fan": [0, 1]}}
    assert profile_level(profile, "inlet/valve", "stuck") == "stuck" and profile_level(profile, "pump", "1") == 1
    assert profile_level(profile, "pump", True) is None and profile_level(profile, "inlet/valve", "half") is None
    components = profile["components"]
    one = parse_step({"kind": "order", "args": {"component": "inlet/valve"}}, 1, components)
    two = parse_step({"kind": "order", "args": {"component": "pump and fan"}}, 1, components)
    assert [s.args["component"] for s in one] == ["inlet/valve"] and [s.args["component"] for s in two] == ["pump", "fan"]


def test_one_reader_of_the_receipt_serves_the_controller_and_the_port():
    """Guard: one table of answers read by CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts gives the body, the receipt and the end the base read from them, and the
    surface of the controller and the port read through the same functions."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import metering_receipts as receipts
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import tool_surfaces as surface_module
    from CausalDiagnosis.frameworks.AssetOpsBench.agents import causal_diagnosis_agent_surface as aob_surface_module
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import sub_agent_runtime
    body = json.dumps({"time": 2})
    receipt = {"steps_left": 3, "budget_left": 10}
    table = [("read", body, False, body, None, False),
             ("read", f"{body}\n{json.dumps(receipt)}", False, body, receipt, False),
             ("read", f"{body}\n{json.dumps(dict(receipt, steps_left=0))}", False, body, dict(receipt, steps_left=0), True),
             ("wait", f"{body}\n{json.dumps(dict(receipt, notifications=[{'event': 'end', 'reason': 'budget'}]))}", False, body,
              dict(receipt, notifications=[{"event": "end", "reason": "budget"}]), True),
             ("read", json.dumps({"error": receipts.EPISODE_OVER + "; this call changed nothing"}), True,
              json.dumps({"error": receipts.EPISODE_OVER + "; this call changed nothing"}), None, True),
             ("finish", json.dumps({"finished": True}), False, json.dumps({"finished": True}), None, True),
             ("finish", json.dumps({"error": "a diagnosis comes first"}), True, json.dumps({"error": "a diagnosis comes first"}), None, False)]
    for tool, text, is_error, want_body, want_receipt, over in table:
        assert receipts.split_receipt(text) == (want_body, want_receipt), text
        assert receipts.ended(tool, want_body, want_receipt, is_error) is over and sub_agent_runtime.episode_over(tool, text, is_error) is over, text
    assert aob_surface_module.split_receipt is receipts.split_receipt
    assert sub_agent_runtime.split_receipt is receipts.split_receipt and sub_agent_runtime.ended is receipts.ended


def test_the_server_names_agree_in_the_agent_the_port_and_the_drivers():
    """The three layers that may not import each other spell the server names of the rig alike: the surface of the agent, the
    sub-agents of the Agent-As-Tool port and the drivers."""
    from CausalDiagnosis.experiments.episode_runners import servers_of_a_run
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import tool_surfaces as surface_module
    from CausalDiagnosis.frameworks.AssetOpsBench.agents import causal_diagnosis_agent_surface as aob_surface_module
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import sub_agent_registry
    assert (surface_module.RIG_SERVER, surface_module.CAUSAL_SERVER) == (servers_of_a_run.RIG_SERVER, servers_of_a_run.CAUSAL_SERVER)
    assert aob_surface_module.AOB_SERVERS == servers_of_a_run.METERED_SERVERS
    assert {server for spec in sub_agent_registry.SURFACES["rig"] for server in spec.servers} == {*servers_of_a_run.METERED_SERVERS, servers_of_a_run.CAUSAL_SERVER}


def test_the_orientation_waits_for_the_failure_modes(tmp_path):
    """An opening that reads the profile alone and the failure modes in a later plan: the orientation is written once, at the first
    turn at which the profile and the failure modes are both on the board, so its prompt lists the failure modes."""
    class ProfileFirst(ScriptedPlanner):
        def generate(self, prompt, temperature=0.0):
            if PLANNING_HEADING in prompt and "No profile read yet" in prompt:
                self.prompts.append(prompt)
                return plan_json([{"kind": "profile"}], mode="test_first")
            return super().generate(prompt, temperature)
    planner = ProfileFirst(act_now)
    ep_dir, row, _backend = _play(tmp_path, planner, {})
    assert row["error"] == "", row["error"]
    orientations = [p for p in planner.prompts if ORIENTATION_HEADING in p]
    assert len(orientations) == 1 and "fmea/" in orientations[0].split(ORIENTATION_HEADING)[1]


def test_without_the_server_readings_collapse_only_when_the_request_does_not_fit_and_keep_their_side():
    """Without the server the rows the model planned on stay in full while the request fits, and once it does not a collapsed
    row keeps the side, below or above, of every value outside its normal range."""
    board = _synthetic_board_without_server()
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 2, "t": 1, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 55.0,
                                                                                    "CE_CoolingEfficiency_percent_1Hz_std": 0.1}}]})
    board.plan_basis = 2
    rows = len(board.evidence)
    assert board.compact() == [] and len(board.evidence) == rows
    assert board.compact(lambda b: False) and board.summaries[0]["sides"] == {"obs/1/CE_CoolingEfficiency_percent_1Hz_mean": "below",
                                                               "obs/2/CE_CoolingEfficiency_percent_1Hz_mean": "above"}


def test_a_group_counts_every_channel_the_board_holds_for_its_cycle():
    """A wider group read for a cycle whose narrower group an earlier read served: its line among the channels outside their normal ranges
    counts every channel of the group that the board holds for that cycle, the value of the earlier read included."""
    profile = json.loads(json.dumps(_synthetic_board().profile))
    profile["sensor_groups"]["tier12"] = ["CE_CoolingEfficiency_percent_1Hz", "PS1_Pressure_bar_100Hz"]
    profile["normal_ranges"]["PS1_Pressure_bar_100Hz_mean"] = {"low": 100.0, "high": 200.0}
    board = Board(server=False)
    board.set_profile(profile)
    board.status = {"time": 1, "budget_left": 299, "steps_left": 57, "open_work_orders": [], "episode_over": False}
    board.add_reading({"sensor_group": "tier1", "served": [{"n": 1, "t": 0, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 30.0,
                                                                                    "CE_CoolingEfficiency_percent_1Hz_std": 0.1}}]})
    board.add_reading({"sensor_group": "tier12", "served": [{"n": 2, "t": 0, "values": {"PS1_Pressure_bar_100Hz_mean": 150.0,
                                                                                     "PS1_Pressure_bar_100Hz_std": 0.1}}]})
    assert "1 of 2 channel means outside" in board.render()


def test_without_the_server_a_repair_is_not_judged_while_another_order_is_open():
    """Without the server the normal ranges judge the whole asset: a repair plan whose orders closed and whose reading came in is not judged
    while another order of the agent is open, and is judged once that order has closed as well."""
    loop_runner = controller.CausalLoopRunner(llm=ScriptedPlanner(act_now), server_paths={RIG_SERVER: "unused"}, options=CausalLoopOptions())
    board = loop_runner.board = _synthetic_board_without_server()
    del board.status["open_work_orders"]  # the orders close by their notifications alone
    for order, component, key in (("wo-1", "cooler", "plan/1"), ("wo-2", "pump", "plan/2")):
        board.add_order({"work_order": {"id": order, "component": component, "created_t": 1, "expected_close_t": None, "status": "open"}}, key)
    board.add_notifications([{"event": "work_order_closed", "id": "wo-1", "component": "cooler", "t": 4}])
    board.record_read({"plan": 2, "step": 1, "sensor_group": "tier1", "t_first": 5, "t_last": 5})
    judged = []
    loop_runner._band_verdict = lambda key, closed_t: judged.append(closed_t) or {"status": "verified by the normal ranges", "consistent": True}
    asyncio.run(loop_runner._verify_ready())
    assert judged == [] and "plan/1" not in board.verifications
    board.add_notifications([{"event": "work_order_closed", "id": "wo-2", "component": "pump", "t": 5}])
    asyncio.run(loop_runner._verify_ready())
    assert judged == [4, 5]


def healthy_plans(prompt, call):
    """List the combination with no faulty component as a candidate of its own and diagnose it, then finish without a work order."""
    finish = {"kind": "finish", "args": {}, "expect": {}, "evidence": []}
    if "Commitment: none" not in prompt:
        return plan_json([finish])
    rank = len(_ranked_rows(prompt)) + 1 if "## Ranking" in prompt else 1
    return json.dumps({"branch": rank, "mode": "act_now", "why": "scripted: a start with no fault", "candidates": [{"levels": {}, "failure_mode_ids": []}],
                       "steps": [{"kind": "diagnose", "args": {"rank": rank}, "expect": {}, "evidence": []}, finish]})


@pytest.mark.parametrize("server", [True, False])
def test_on_a_start_with_no_fault_the_agent_diagnoses_all_healthy_and_finishes_without_a_work_order(tmp_path, server):
    """On the control preset, whose starts hold no fault, a model that lists the combination with no faulty component diagnoses it
    and finishes without a work order, with the server and without it, under the unchanged briefing: the diagnosis holds every
    component at its healthy level, no work order is created, the episode ends with finish, and every planning turn allows a
    plan without a work order."""
    if server:
        ep_dir, row, backend = _play(tmp_path, ScriptedPlanner(healthy_plans), {}, config_name="healthy_control")
    else:
        ep_dir, row, backend = _play_without_server(tmp_path, lambda prompt, call, truth: healthy_plans(prompt, call), config_name="healthy_control")
    assert row["error"] == "", row["error"]
    trajectory = read_json(ep_dir / "trajectory.json")
    diagnosis = next(t for t in trajectory if t["tool"] == "submit_diagnosis")
    assert {c: diagnosis["tool_args"][c] for c in COMPONENTS} == dict(HEALTHY)
    assert "create_work_order" not in [t["tool"] for t in trajectory] and trajectory[-1]["tool"] == "finish"
    planning = [c["prompt"] for c in backend.calls if PLANNING_HEADING in c["prompt"]]
    assert planning and all("none when no component is faulty" in p for p in planning)


def test_a_candidate_without_a_fault_is_an_option_of_the_rule():
    """A previewed candidate under which no component is faulty is an option of the rule with its own cost: when it is the
    cheapest the rule plans a diagnosis and a finish without a work order."""
    board = _synthetic_board(read_first=False, second=True)
    board.ranking["candidates"][1].update(combination=dict(HEALTHY), faults=[])
    board.ranking["branches"][1].update(combination=dict(HEALTHY), faults=[], faulty_set=[], cost_to_restored={"act_now": 40, "test_first": 50},
                                        repair_plan={"steps": [], "plan_prediction_id": None, "price": 0, "downtime_cost": 0, "cycles_to_restored": 0})
    board.set_ranking(board.ranking)
    assert act_options(board)[0] == (40, 2)
    plan = rule_plan(board, 6)
    assert [s.kind for s in plan.steps] == ["diagnose", "finish"] and plan.steps[0].args == {"rank": 2}


def test_after_a_refuted_plan_a_candidate_without_a_fault_is_no_option_of_backtracking():
    """A refuted verdict says that the repair did not restore the asset, so after it a previewed candidate under which no
    component is faulty is no other plan: backtracking repairs the cheapest other branch that orders a component, though the
    candidate without a fault costs less to act on, and the advice names that branch."""
    board = _synthetic_board(read_first=False, second=True)
    board.ranking["candidates"].append({"rank": 3, "combination": dict(HEALTHY), "probability": 0.0, "faults": [], "evidence_ids": []})
    board.ranking["branches"].append({**board.ranking["branches"][1], "rank": 3, "combination": dict(HEALTHY), "probability": 0.0, "faults": [],
                                      "faulty_set": [], "cost_to_restored": {"act_now": 40, "test_first": 50},
                                      "repair_plan": {"steps": [], "plan_prediction_id": None, "price": 0, "downtime_cost": 0, "cycles_to_restored": 0}})
    board.set_ranking(board.ranking)
    assert act_options(board) == [(40, 3), (60, 2), (72, 1)]
    board.diagnosis = {"diagnosis": dict(board.candidate(1)["combination"]), "evidence_ids": []}
    board.record_refuted(["pump"], "the reading after the last close is inconsistent with the plan's prediction")
    assert repair_options(board) == [(60, 2)]
    assert any(line.startswith("The last repair plan was refuted, so the advice is to repair rank 2, the cheapest other plan") for line in advice(board))
    plan = rule_plan(board, 6)
    assert plan.branch == 2 and plan.why.startswith("rule: the last repair plan was refuted, so repair rank 2")
    assert [s.args["component"] for s in plan.steps if s.kind == "order"] == ["valve"]


# ----- the rows with lead times, the end of an episode, a refused forced finish and the texts of a further asset -----

ORDER_ROW = re.compile(r"^\| (wo-\d+) \| (\w+) \| (-?\d+) \| (-?\d+|None) \| (\w+) \|", re.MULTILINE)  # id, component, created, expected close, status
TIME_LINE = re.compile(r"^Time (-?\d+) of \d+ cycles", re.MULTILINE)
# seconds of thinking per cycle, more than any call of a test takes: on the rows of the real-time clock an order then takes no
# cycle, so the orders of one plan fall in one cycle and form one visit, which the instant mode of the pinned clock never allows
UNREACHED_THINKING = 10.0 ** 6


def _faulty(config, combination):
    """The (component, level) pairs of a combination whose level is not the healthy one, in the order of the components."""
    return [(c, level) for c, level in zip(config.components, combination) if level != config.healthy[c]]


def _first_start(config, blocks, rule):
    """The start that _aob_run takes in place of the draw of the seed: the first combination of the asset's blocks, in sorted
    order, whose faulty (component, level) pairs meet the rule, so that a test names no seed and no combination. Blocks without
    such a combination fail the test by name."""
    combination = next((c for c in sorted(blocks) if rule(_faulty(config, c))), None)
    if combination is None:
        pytest.fail("no combination of the blocks of the asset meets the rule of the test")
    return {"combination": list(combination), "entry_offset": 0}


def _visit_start(config_name):
    """The start of the visit tests on a row and its faulty components: the first combination of the blocks with two faulty
    components, neither at the worst level, the level at which the failure card of the deterioration row fails a component."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import worst_level
    _facts, _cycles, blocks = load_world()
    _name, config = load_config(config_name)
    assert config.fault_rule["min_faulty"] <= 2 <= config.fault_rule["max_faulty"]
    start = _first_start(config, blocks, lambda faulty: len(faulty) == 2 and all(level != worst_level(config, c) for c, level in faulty))
    return start, [c for c, _level in _faulty(config, start["combination"])]


def _visit_plans(ordered, idle_after_orders=False):
    """The scripted model on a row with lead times. While the board advises a read it reads; then it diagnoses rank 1 and orders
    the components it is given in one plan, so the orders fall in one cycle and form one visit, whatever the ranking shows. An
    order cites the prediction of the top branch's plan where the board shows one for its component and asks for an estimate
    otherwise. While an order is open it waits one cycle a plan, so a planning turn shows the board at every cycle of the visit,
    and once every order closed it waits one cycle more, reads the cycle recorded after the last close and finishes. With
    idle_after_orders the model writes no plan once the orders stand, so its turns make no call and the episode ends while
    they are open."""
    def plans(prompt, call):
        advised = advised_read(prompt)
        if advised:
            return advised
        if "Commitment: none" in prompt:
            ids = branch_predictions(prompt)
            steps = [{"kind": "diagnose", "args": {"rank": 1}, "expect": {"top_rank": 1}, "evidence": AOB_EVIDENCE_ID.findall(prompt)[:3]}]
            for component in ordered:
                pid = ids.get(component, ESTIMATE)
                steps.append({"kind": "order", "args": {"component": component, "prediction_id": pid},
                              "expect": {"prediction_id": pid, "recovery_at_least": 0.0}, "evidence": [pid] if pid != ESTIMATE else []})
            return plan_json(steps)
        if idle_after_orders:
            return "Still thinking."
        wait = {"kind": "wait", "args": {"cycles": 1}, "expect": {}, "evidence": []}
        if any(row[4] == "open" for row in ORDER_ROW.findall(_section(prompt, "## Work orders"))):
            return plan_json([wait])
        return plan_json([wait, dict(VERIFYING_READ), {"kind": "finish", "args": {}, "expect": {}, "evidence": []}])
    return plans


def _planning_prompts(planner):
    return [p for p in planner.prompts if PLANNING_HEADING in p and ORIENTATION_HEADING not in p and SUMMARY_HEADING not in p]


@pytest.mark.parametrize("config_name", ["repair_rules", "deterioration"])
def test_the_agent_repairs_two_components_in_one_visit_on_the_rows_with_lead_times(tmp_path, config_name):
    """On the two rows above the real-time row, a start with two faulty components that the test finds in the blocks of the asset,
    through AssetOpsBench's servers on the memory store with a thinking rate that no call reaches: the scripted model orders both
    in one plan, so both orders fall in one cycle and form one visit. Each order stays open on the board until its notification
    and shows meanwhile the close of the visit as the environment schedules it, the summary gives each close, the read after the
    last close asks for recorded cycles and serves them, no read comes back empty, the finish passes at its first attempt and the
    verifier accepts the episode."""
    start, faulty = _visit_start(config_name)
    planner = ScriptedPlanner(_visit_plans(faulty))
    ep_dir, row, _backend = _aob_run(tmp_path, config_name, planner, {}, start=start, think_seconds=UNREACHED_THINKING)
    trajectory, _verdict = _assert_aob_surface(ep_dir, row, clock=True)
    plans = read_json(ep_dir / "plans.json")
    orders = plans["board"]["orders"]
    assert sorted(o["component"] for o in orders.values()) == sorted(faulty) and len({o["created_t"] for o in orders.values()}) == 1, orders
    shown_open = set()
    for prompt in _planning_prompts(planner):
        rows = ORDER_ROW.findall(_section(prompt, "## Work orders"))
        if not rows:
            continue
        time = int(TIME_LINE.search(prompt).group(1))
        for order_id, _component, _created, expected, status in rows:
            close = orders[order_id]["closed_t"]
            if time < close:  # before its notification the order is open, at the close the visit schedules
                assert (status, expected) == ("open", str(close)), (order_id, time, close, status, expected)
                shown_open.add(order_id)
            elif time > close:
                assert status == "closed", (order_id, time, close, status)
    assert shown_open == set(orders)
    summary = next(p for p in planner.prompts if SUMMARY_HEADING in p)
    for o in orders.values():
        assert f"Work order {o['id']} on the {o['component']} created at cycle {o['created_t']}, closed at cycle {o['closed_t']}," in summary
    histories = [t for t in trajectory if t["tool"] == "history"]
    assert all(json.JSONDecoder().raw_decode(t["response"])[0].get("observations") for t in histories)  # no read comes back empty
    assert "end" in histories[-1]["tool_args"]  # the read after the last close asks for recorded cycles
    verifications = plans["board"]["verifications"]
    assert verifications and all(v["status"] == "verified" for v in verifications.values()), verifications
    assert [s["status"] for p in plans["plans"] for s in p["steps"] if s["kind"] == "finish"] == ["done"]


def test_the_summary_gives_an_open_order_as_open_with_the_close_of_its_visit(tmp_path):
    """The same start on the repair rules row, the model writing no plan once both orders stand: the controller finishes at its idle
    cap while both are open, and the summary gives each order as still open with the close of the visit as the environment
    schedules it from what the profile publishes."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import visit_closes
    start, faulty = _visit_start("repair_rules")
    planner = ScriptedPlanner(_visit_plans(faulty, idle_after_orders=True))
    ep_dir, row, _backend = _aob_run(tmp_path, "repair_rules", planner, {}, start=start, think_seconds=UNREACHED_THINKING)
    assert row["error"] == "", row["error"]
    orders = read_json(ep_dir / "plans.json")["board"]["orders"]
    created = {o["created_t"] for o in orders.values()}
    assert sorted(o["component"] for o in orders.values()) == sorted(faulty) and len(created) == 1, orders
    profile = json.JSONDecoder().raw_decode(read_json(ep_dir / "trajectory.json")[0]["response"])[0]
    closes = visit_closes(profile, [(o["component"], o["created_t"], None) for o in orders.values()])
    summary = next(p for p in planner.prompts if SUMMARY_HEADING in p)
    for o in orders.values():
        assert (f"Work order {o['id']} on the {o['component']} created at cycle {o['created_t']}, still open, expected to close at cycle "
                f"{closes[o['component']]},") in summary


def test_an_episode_that_a_failure_ends_reads_its_end_reason(tmp_path):
    """The deterioration row, whose failure card ends the episode, on a start with one component at its worst level and a model that
    only waits, through AssetOpsBench's servers: the component fails after its patience, the environment ends the episode with
    the reason failure, and the end reason of the row is the reason of the end event that the receipt carries."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts import split_receipt
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import worst_level
    _facts, _cycles, blocks = load_world()
    _name, config = load_config("deterioration")
    assert config.fault_rule["min_faulty"] <= 1 <= config.fault_rule["max_faulty"]
    start = _first_start(config, blocks, lambda faulty: len(faulty) == 1 and faulty[0][1] == worst_level(config, faulty[0][0]))
    wait = plan_json([{"kind": "wait", "args": {"cycles": 5}, "expect": {}, "evidence": []}])
    ep_dir, row, _backend = _aob_run(tmp_path, "deterioration", ScriptedPlanner(lambda prompt, call: wait), {}, start=start)
    receipts = [split_receipt(t["response"])[1] or {} for t in read_json(ep_dir / "trajectory.json")]
    ends = [n.get("reason") for r in receipts for n in r.get("notifications") or [] if isinstance(n, dict) and n.get("event") == "end"]
    verdict = read_json(ep_dir / HIDDEN / "verdict.json")
    assert ends[:1] == ["failure"] and row["end_reason"] == verdict["end_reason"] == "failure", (ends, row["end_reason"], verdict["end_reason"])


def test_an_episode_the_environment_ends_at_a_no_call_step_gets_one_summary_and_no_finish(tmp_path):
    """The controller forces no finish (it has no limit of its own): when the driver's no_call entry reports the end of the episode
    after a turn without a call, the run stops, the summary written then is the answer, and no finish is called."""
    planner = ScriptedPlanner(lambda prompt, call: "Still thinking.", orientation="No differential yet.")
    idle_calls = []
    ep_dir, loop_runner, question = _controller(tmp_path, planner)
    with _in_process(ep_dir), _idle_hook(lambda: idle_calls.append(1) or len(idle_calls) == 2):
        result = asyncio.run(loop_runner.run(question))
    assert (result.end, result.answer, sum(SUMMARY_HEADING in p for p in planner.prompts)) == ("episode_over", SUMMARY_REPLY, 1)
    assert not [t for t in result.trajectory if t["tool"] == "finish"] and len(idle_calls) == 2


def _tiny_ranking(profile, modes, group):
    """A ranking that names the components of the asset of the profile: rank 1 a faulty level of its first component with the failure
    mode the asset lists for it and a troubleshooting plan, rank 2 every component at its healthy level."""
    healthy = dict(profile["healthy"])
    component = profile["components"][0]
    level = next(v for v in profile["levels"][component] if v != healthy[component])
    code = next(m["code"] for m in modes if m["component"] == component and m["level"] == level)
    faulty, fault = {**healthy, component: level}, [{"component": component, "level": level, "code": code}]
    test = {"sensor_group": group, "cycles": 1, "price": 1, "duration": 1, "cost": 2, "expected_loss": 1}
    return {"ranking_model": "joint_block", "n_candidates": 2, "read_first": True, "separating_test": test,
            "candidates": [{"rank": 1, "combination": faulty, "probability": 0.7, "faults": fault, "evidence_ids": []},
                           {"rank": 2, "combination": healthy, "probability": 0.3, "faults": [], "evidence_ids": []}],
            "expected_loss_act": 3, "expected_saving": 1, "expected_loss_test": {g: 1 for g in profile["sensor_groups"]}, "unseparated": [],
            "separating_channels": {}, "set_masses": [{"faulty_set": [component], "mass": 0.7}, {"faulty_set": [], "mass": 0.3}],
            "branches": [{"rank": 1, "combination": faulty, "faults": fault, "probability": 0.7, "faulty_set": [component],
                          "confirm_test": {**test, "mass_if_true": 0.9, "mass_if_false": 0.2, "gap": 0.7, "standard_error": 0.01}, "two_step": None,
                          "repair_plan": {"steps": [{"component": component, "part_price": 1, "repair_charge": 1, "lead_time": 1, "repair_downtime": 1,
                                                     "downtime_cost": 1, "prediction_id": "pred/b1"}],
                                          "plan_prediction_id": "pred/b1", "price": 2, "downtime_cost": 1, "cycles_to_restored": 2},
                          "predicted_outcome": {"predicted": [], "alpha": 0.05, "recovery_if_acted": 0.7, "restored_if_true": True},
                          "expected_loss_if_acted": 3, "expected_loss_if_tested": 2, "cost_to_restored": {"act_now": 4, "test_first": 5}}]}


def test_the_texts_the_agent_builds_for_the_tiny_asset_hold_no_word_of_another_asset(tmp_path):
    """The texts that the functions of the agent's package return for a model, on the tiny asset of the asset tests read from its
    file: the system prompt of both modes, the orientation, a planning turn with a retry, the summary with a note, the board of both
    modes before and after the profile, the advice and the reasons of the plan check for plans that break its rules, on the surface
    of either server as the driver builds it. None names, as a whole word, a noun, the example channel, a component, a sensor
    group, a channel tag or the asset id of a registered asset that the tiny asset does not hold itself. The words are read from
    the asset files, so a further registered asset needs no change of the test."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import advice
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import summary_turn
    from CausalDiagnosis.datasets.dataset_loader import recordings as records
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, register_profile, unregister_profile
    from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
    from CausalDiagnosis.datasets.tests.conftest import TINY_FILE, write_recordings
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing

    def words(data):
        text = data["text"]
        nouns = {value for key, value in text.items() if key.endswith("noun") and isinstance(value, str)}
        example = {text["example_channel"]} if text.get("example_channel") else set()
        return nouns | example | {data["asset_id"], *data["components"], *data["observation_kinds"], *data["channels"]}
    tiny_data = json.loads(TINY_FILE.read_text(encoding="utf-8"))
    registered = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ASSETS_DIR.glob("*/asset.json"))]
    banned = set().union(*map(words, registered)) - words(tiny_data)
    assert registered and banned
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(banned, key=len, reverse=True))) + r")(?!\w)", re.IGNORECASE)
    tiny = register_profile(AssetProfile.from_json(tiny_data, str(TINY_FILE)))
    texts = []
    try:
        cycles = records.load_cycles(write_recordings(tmp_path / "tiny_cycles.jsonl", tiny), tiny)
        env = HydraulicRigEnv(cycles, records.build_blocks(cycles), tiny.default_config())
        profile = env.reset(0)["result"]
        question = briefing(profile, env.config)
        modes = env.step({"type": "get_failure_modes", "component": ""})[0]
        group = next(iter(profile["sensor_groups"]))
        reading = env.step({"type": "read_sensors", "sensor_group": group, "cycles": 1})[0]
        failure_modes = modes["result"]["failure_modes"]
        ranking = _tiny_ranking(profile, failure_modes, group)
        component = profile["components"][0]
        own = [{"levels": {component: ranking["candidates"][0]["combination"][component]}, "failure_mode_ids": []}]
        broken = [Plan(1, "test_first", "x", [Step("read", {"sensor_group": "no_such_group", "cycles": 1})]),
                  Plan(1, "act_now", "x", [Step("order", {"component": component, "prediction_id": "pred/b1"})]),
                  Plan(1, "act_now", "x", [Step("diagnose", {"rank": 9})]),
                  Plan(1, "act_now", "x", [Step("diagnose", {"rank": 1}), Step("order", {"component": component, "prediction_id": "pred/b1"}), Step("finish")]),
                  Plan(1, "test_first", "x", [Step("read", {"sensor_group": group, "cycles": 10 ** 6})]),
                  Plan(1, "act_now", "x", [Step("finish")], candidates=[{"levels": {"no_such_part": 1}, "failure_mode_ids": []}])]
        for store in (None, {"kind": "memory"}):
            surface = run_episodes.causal_surface(tiny.asset_id, store)
            for server in (True, False):
                texts.append(system_prompt(6, surface, server))
                texts.append(orientation_turn(question, profile, failure_modes, 6, surface=surface, server=server))
                board = Board(evidence_form=surface.evidence_form, reads_take_time=surface.reads_take_time, noun=surface.noun, question=question,
                              server=server)
                texts.append(board.render())
                board.set_profile(profile)
                board.set_failure_modes(modes["result"])
                board.update_status(modes)
                board.add_reading(reading["result"])
                board.update_status(reading)
                texts.append(board.render())
                if server:
                    board.set_ranking(ranking)
                else:
                    board.set_candidates(own)
                texts += [board.render(), planning_turn(board, 6, retry="The reply could not be parsed.", surface=surface),
                          summary_turn(board, 6, note="The episode ended.", surface=surface), *advice(board)]
                texts += [reason for plan in broken for reason in check_plan(plan, board)]
    finally:
        unregister_profile(tiny.asset_id)
    hits = sorted({m.group(0) for text in texts for m in pattern.finditer(text)})
    assert len(texts) > 40 and not hits, hits


def test_the_texts_of_the_agent_name_the_plain_statistic_the_asset_serves():
    """The texts of the agent name the values by the plain statistic of the asset, the served statistic listed ahead of the others,
    in the plural with an s appended: means on the rig, and values with no word of another statistic on a made-up port of the shape
    of the demo port of the asset tests, the tiny asset with value as its one recorded and served statistic. A read on
    AssetOpsBench's servers asks for the bare field of a sensor and the fields of the statistics served after the plain one, so the
    port's read asks for the bare fields alone. The words come from the asset file through the driver's surface and from the
    profile through the board, which the messages of the runner read too."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import summary_turn
    from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import register_profile, unregister_profile
    from CausalDiagnosis.datasets.tests.conftest import TINY_FILE

    def texts(asset_id, board):
        out = [board.render()]
        for store in (None, {"kind": "memory"}):
            surface = run_episodes.causal_surface(asset_id, store)
            out += [system_prompt(6, surface, server) for server in (True, False)]
            out += [orientation_turn("Which level does each part hold?", board.profile, board.failure_modes, 6, surface=surface),
                    planning_turn(board, 6, surface=surface), summary_turn(board, 6, surface=surface)]
        return out

    rig_board = _synthetic_board()
    rig_board.alarms.append({"channel": "CE_CoolingEfficiency_percent_1Hz", "first_t": 0})
    rig = texts("hyd_1", rig_board)
    assert rig_board.statistic == "mean" and all("channel mean" in t for t in rig)
    data = json.loads(TINY_FILE.read_text(encoding="utf-8"))
    data["statistics"] = {"recorded": ["value"], "served": ["value"]}
    port = register_profile(AssetProfile.from_json(data, str(TINY_FILE)))
    try:
        board = Board()
        board.set_profile({**_synthetic_board().profile, "statistics": ["value"], "sensor_groups": {"all": ["F1_Force_kN_1Hz"]},
                           "normal_ranges": {"F1_Force_kN_1Hz_value": {"low": 0.0, "high": 0.5}}})
        board.add_reading({"sensor_group": "all", "served": [{"n": 1, "t": 0, "values": {"F1_Force_kN_1Hz_value": 0.9}}]})
        board.alarms.append({"channel": "F1_Force_kN_1Hz", "first_t": 0})
        values = texts(port.asset_id, board)
        surfaces = {asset_id: run_episodes.causal_surface(asset_id, {"kind": "memory"}) for asset_id in ("hyd_1", port.asset_id)}
    finally:
        unregister_profile(port.asset_id)
    other = re.compile(r"(?<![A-Za-z0-9])(?:means?|std|standard deviations?)(?![A-Za-z0-9])", re.IGNORECASE)
    assert board.statistic == "value" and all("channel value" in t for t in values)
    assert "1 of 1 channel values outside" in values[0] and "the channel value left its normal range" in values[0]
    assert not sorted({m.group(0) for t in values for m in other.finditer(t)})
    assert surfaces["hyd_1"].summary_fields() == " and their _std fields" and surfaces[port.asset_id].summary_fields() == ""
    for asset_id, fields in (("hyd_1", ["CE_CoolingEfficiency_percent_1Hz", "CE_CoolingEfficiency_percent_1Hz_std"]),
                             (port.asset_id, ["CE_CoolingEfficiency_percent_1Hz"])):
        surface = _recorded_surface(rig_board, -2, 3)
        surface.statistics = surfaces[asset_id].statistics
        _server, tool, args = surface.bind("read", rig_board, sensor_group="tier1", cycles=1)
        assert (tool, args["sensors"]) == ("history", fields), (asset_id, args)


def test_the_agent_reads_the_flag_under_its_new_name_alone():
    """The board and the rule read the flag of the ranking as read_first alone: the ranking of the synthetic board reads the
    separating test first, while the same ranking with the flag under the old key, spelled here from two parts, advises no
    read and the rule acts on the top candidate."""
    old = "insufficient" + "_evidence"
    board = _synthetic_board(read_first=True)
    assert board.read_first() and rule_plan(board, 6).mode == "test_first"
    legacy = _synthetic_board(read_first=True)
    legacy.set_ranking({(old if key == "read_first" else key): value for key, value in legacy.ranking.items()})
    assert not legacy.read_first() and rule_plan(legacy, 6).mode == "act_now"
    assert not [line for line in advice(legacy) if line.startswith("Advised reads")]


def test_a_declaration_without_the_server_cites_the_newest_absorbed_reading():
    """Once every evidence row is absorbed, the declaration cites a value of the newest observation, compared as a number on the rig,
    where obs/9 sorts after obs/12 as text."""
    from types import SimpleNamespace

    board = SimpleNamespace(evidence=[], absorbed_ids={"obs/9/a", "obs/12/a", "obs/12/b"})
    assert controller.CausalLoopRunner._last_reading(SimpleNamespace(board=board)) == ["obs/12/b"]
    board.absorbed_ids = {"obs/2025-04-02T00:15:00/x", "obs/2025-04-02T00:30:00/x"}
    assert controller.CausalLoopRunner._last_reading(SimpleNamespace(board=board)) == ["obs/2025-04-02T00:30:00/x"]


def test_a_text_that_is_no_json_object_is_no_step_of_the_base_row():
    """On the base row, which appends no receipt, the surface counts the steps itself: an answer the meter wrote, an error object
    included, is a step, and a text that is no JSON object, which only a call that never reached the episode gives, is none, so
    the steps left stay those of the environment."""
    surface = AobSurface()
    surface.max_steps = 10
    unlisted = surface.parse("wait", "wait_for_notification", "'the connection closed before the answer'", None)
    assert "refused" in unlisted and unlisted["steps_left"] == 10 and surface.calls == 0
    refused = surface.parse("diagnose", "submit_diagnosis", json.dumps({"error": "a refusal of the engine"}), None)
    assert refused["refused"] == "a refusal of the engine" and refused["steps_left"] == 9 and surface.calls == 1


# ----- one rule for finishing a repair, the verifying read of the indicating group, and four texts -----

# each place of the tail and whether it must reach the rule itself (False: it is handed a completion made by its caller)
TAIL_PLACES = {"plan/repair_budget.py": {"read_affordable": True, "_fits_beside_reserve": True, "reserve_completion": True,
                                        "order_reserve": True, "order_reserve_calls": True, "_branch_costs": True},
               "plan/read_advice.py": {"exploration": True}, "plan/rule_controller.py": {"rule_plan": True},
               "plan/plan_check.py": {"_verification_problem": False, "_tail_fits": False},
               "candidate_plan_schedule.py": {"needs_of": True, "limit": True, "remaining": False}, "controller.py": {"_unverified_reason": True}}
TAIL_ARITHMETIC = {"read_price", "plan_price", "order_price", "visit_charge", "read_duration", "wait_cycles", "visit_closes", "first_cycle_after",
                   "order_duration", "visit_schedule", "visit_moments"}
TAIL_CALLERS = {"completion", "reserve_completion", "needs_of", "_fits_beside_reserve", "wait_calls"}


def _functions(path):
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _names(node):
    import ast
    return {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(node) if isinstance(n, (ast.Attribute, ast.Name))}


def test_every_place_that_counts_the_tail_of_a_repair_calls_the_one_rule():
    """Guard against a second copy: each place that needs what finishing a repair costs reaches it through Board.completion (or
    a function of plan.py that calls it) and names none of the unit prices, durations or visit rules itself, and the wait step of
    the plan check counts its calls with Board.wait_calls. A new place of the tail has to be named here."""
    package = Path(__file__).resolve().parents[1]
    for name, functions in TAIL_PLACES.items():
        found = _functions(package / name)
        for function, calls_rule in functions.items():
            names = _names(found[function])
            assert names & TAIL_CALLERS or not calls_rule, (name, function, "calls no function of the one rule")
            assert not names & TAIL_ARITHMETIC - ({"read_price", "read_duration"} if function == "read_affordable" else set()), (
                name, function, sorted(names & TAIL_ARITHMETIC))
    wait = next(n for n in __import__("ast").walk(_functions(package / "plan/plan_check.py")["check_plan"]) if isinstance(n, __import__("ast").If)
                and "wait" in __import__("ast").unparse(n.test) and "step.kind" in __import__("ast").unparse(n.test))
    helper = _functions(Path(__file__))["separating_test_affordable"]
    assert "_fits_beside_reserve" in _names(helper) and not _names(helper) & {"order_reserve", "order_reserve_calls", "order_duration"}
    body = set().union(*(_names(n) for n in wait.body))
    assert "wait_calls" in body and "closes" not in body
    finish = _functions(package / "plan/plan_check.py")["check_plan"]
    assert "tail_from" in _names(finish) and "completion" in _names(_functions(package / "plan/plan_check.py")["tail_from"])


def test_the_package_of_public_rules_reads_public_values_alone():
    """servers/public_rules, which the environment, the agent and the diagnostics server import, imports the standard library and
    its own modules alone (no engine, hidden state, verifier, server or agent module) and holds no module-level data beyond
    constants: a module states rules, never state."""
    import ast
    import sys as _sys
    from CausalDiagnosis.utils import public_rules as package
    files = sorted(Path(package.__file__).parent.glob("*.py"))
    assert len(files) >= 2
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        modules |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
        own = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level}
        assert modules <= set(_sys.stdlib_module_names) and own <= {f.stem for f in files}, (path.name, modules, own)
        assert all(n.level == 1 for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level), path.name
        constants = set()  # the module's constants above the node, which a constant may name, as RUN_KEEPING_STEPS names NO_CALL_ACTION
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                assert all(isinstance(t, ast.Name) and t.id.isupper() for t in targets), (path.name, ast.unparse(node))
                assert isinstance(node.value, (ast.Constant, ast.Tuple)) and all(
                    isinstance(e, ast.Constant) or (isinstance(e, ast.Name) and e.id in constants)
                    for e in ast.walk(node.value) if not isinstance(e, (ast.Tuple, ast.Load))), (path.name, ast.unparse(node))
                constants |= {t.id for t in targets}


def _charged(trajectory, start, end):
    """(calls on the action servers, cycles, price) that the environment charged for the calls start to end of a trajectory, read
    from the envelope or the receipt of the call before start and of the call end."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts import split_receipt

    def state(call):
        body, receipt = split_receipt(call["response"])
        if receipt is not None:
            return receipt.get("cycle"), receipt.get("budget_left"), receipt.get("steps_left")
        data = json.loads(body)
        return data.get("time"), data.get("budget_left"), data.get("steps_left")
    before, after = state(trajectory[start - 1]), state(trajectory[end])
    calls = sum(1 for t in trajectory[start:end + 1] if t["server"] != CAUSAL_SERVER)
    price = None if before[1] is None else before[1] - after[1]
    cycles = None if before[0] is None else after[0] - before[0]
    return calls, cycles, price


def _tail_planner(components_of):
    """A scripted model that diagnoses rank 1 and orders the components components_of(prompt) gives, then plans exactly the tail
    that Board.completion names on the board of its turn: the wait to the first cycle that verifies, the read of the
    verification group and finish. The board of each planning turn is kept by a spy of the plan check."""
    seen = {}

    def plans(prompt, call):
        board = seen["board"]
        if not board.committed:
            components = components_of(prompt)
            seen["components"], seen["tail"] = components, board.completion(components, commit=False)
            steps = [{"kind": "diagnose", "args": {"rank": 1}, "expect": {}, "evidence": EVIDENCE_ID.findall(prompt)[:2] or AOB_EVIDENCE_ID.findall(prompt)[:2]}]
            # every order asks for an estimate, so each repair carries its own prediction and is verified alone
            steps += [{"kind": "order", "args": {"component": c, "prediction_id": ESTIMATE}, "expect": {}, "evidence": []} for c in components]
            return plan_json(steps)
        tail = board.completion(commit=False)
        steps = []
        if tail.first is not None and tail.first > board.time:
            steps.append({"kind": "wait", "args": {"cycles": tail.first - board.time}, "expect": {}, "evidence": []})
        if tail.reads:
            steps.append({"kind": "read", "args": {"sensor_group": board.verification_group(board.repaired()), "cycles": 1}, "expect": {}, "evidence": []})
        return plan_json(steps + [{"kind": "finish", "args": {}, "expect": {}, "evidence": []}])
    check = controller.check_plan

    def spy(plan, board):
        seen["board"] = board
        return check(plan, board)
    return ScriptedPlanner(plans), seen, spy


@pytest.mark.parametrize("surface, config_name, two", [("rig", "repair_rules", False), ("rig", "timed", False), ("aob", "repair_rules", True),
                                                       ("aob", "deterioration", True), ("aob", "base", True)])
def test_the_completion_of_a_repair_is_what_the_environment_charges(tmp_path, monkeypatch, surface, config_name, two):
    """The environment is the truth: on the rig's action server and on AssetOpsBench's servers, rows with a clock and the base row
    without one, the completion that Board.completion gives before the orders (calls, cycles and price of the orders, the waits
    and the change summary, the verifying read and finish) is what the environment charged from the first order to finish, and
    the order reserve of the base row keeps that many calls."""
    planner_args = {}
    if surface == "aob" and two:
        start, faulty = _visit_start(config_name) if config_name != "base" else (None, None)
        components_of = (lambda prompt: list(faulty)) if faulty else (lambda prompt: top_faulty(prompt) or [COMPONENTS[0]])
        planner_args = {"start": start, "think_seconds": UNREACHED_THINKING} if start else {}
    else:
        components_of = lambda prompt: top_faulty(prompt) or [COMPONENTS[0]]  # noqa: E731
    planner, seen, spy = _tail_planner(components_of)
    monkeypatch.setattr(controller, "check_plan", spy)
    if surface == "rig":
        ep_dir, row, _backend = _play(tmp_path, planner, {}, config_name=config_name)
    else:
        ep_dir, row, _backend = _aob_run(tmp_path, config_name, planner, {}, **planner_args)
    assert row["error"] == "", row["error"]
    trajectory = read_json(ep_dir / "trajectory.json")
    first = next(i for i, t in enumerate(trajectory) if t["tool"] in ("create_work_order", "generate_work_order"))
    end = max(i for i, t in enumerate(trajectory) if t["tool"] == "finish")
    calls, cycles, price = _charged(trajectory, first, end)
    tail = seen["tail"]
    finish_cycles = 0  # finish takes no cycle on these rows
    assert calls == tail.calls, (calls, tail)
    if cycles is not None:
        assert cycles == tail.cycles + finish_cycles, (cycles, tail)
    if price is not None:
        assert price == tail.price, (price, tail)


def test_the_reserve_of_the_base_row_keeps_the_calls_the_environment_charges():
    """On a board without a clock an order closes inside its call and no read verifies it: the reserve keeps the diagnosis, the
    orders and finish, and a read that leaves those calls is affordable."""
    board = _synthetic_board()
    board.clock_runs = False
    assert order_reserve_calls(board) == 1 + 1 + 1  # the diagnosis, the pump's order, finish
    board.status["steps_left"] = 1 + order_reserve_calls(board)
    assert read_affordable(board, "tier1")
    board.status["steps_left"] -= 1
    assert not read_affordable(board, "tier1")


def _open_order_board():
    """The synthetic board after the diagnosis of rank 1 and one order on the pump at cycle 1, whose visit closes at cycle 6."""
    board = _synthetic_board(read_first=False)
    board.diagnosis = {"diagnosis": dict(board.candidate(1)["combination"]), "evidence_ids": []}
    board.add_order({"work_order": {"id": "wo-1", "component": "pump", "created_t": 1, "expected_close_t": 6, "status": "open"}}, "pred/b1")
    board.status["open_work_orders"] = [{"id": "wo-1", "component": "pump", "expected_close_t": 6}]
    return board


def test_near_the_step_cap_one_plan_passes_and_the_finish_guard_agrees():
    """With an open order, at every number of steps left either the plan that waits, reads and finishes or
    the finish alone passes the check, and the runner's finish guard refuses the finish exactly when the check does."""
    finish = Plan(branch=1, mode="act_now", why="", steps=[Step("finish")])
    full = Plan(branch=1, mode="act_now", why="", steps=[Step("wait", {"cycles": 5}), Step("read", {"sensor_group": "tier3", "cycles": 1}), Step("finish")])
    for steps_left in range(1, 12):  # finish alone needs one call
        board = _open_order_board()
        board.status["steps_left"] = steps_left
        passing = [check_plan(p, board) == [] for p in (finish, full)]
        assert any(passing), (steps_left, check_plan(finish, board), check_plan(full, board))
        loop_runner = controller.CausalLoopRunner(llm=ScriptedPlanner(act_now), server_paths={RIG_SERVER: "unused"})
        loop_runner.board = board
        assert (loop_runner._unverified_reason() is None) == passing[0], steps_left


def _modes_board(server=True):
    """The synthetic board with failure modes whose channels name the groups' fields as the rig serves them: the pump indicated
    by FS1 (tier3 only), the cooler by CE (every group)."""
    board = _synthetic_board(read_first=False) if server else _synthetic_board_without_server()
    board.manual_channels = {}
    board.set_failure_modes({"failure_modes": [
        {"component": "pump", "level": 1, "code": "pump_internal_leakage_weak", "channels": ["FS1_VolumeFlow_l_per_min_10Hz"], "evidence_id": "fmea/s/p"},
        {"component": "cooler", "level": 20, "code": "cooler_reduced_efficiency", "channels": ["CE_CoolingEfficiency_percent_1Hz"], "evidence_id": "fmea/s/c"}]})
    return board


@pytest.mark.parametrize("server", [True, False])
def test_only_a_read_of_a_group_that_indicates_the_repair_verifies_it(server):
    """On the plan check, with and without the server: after the close a read of a group that shows no indicating channel of
    the repaired pump does not verify it, so the finish after it is refused naming the verification group; a read of that group
    passes."""
    board = _modes_board(server)
    board.diagnosis = {"diagnosis": {}, "evidence_ids": []}
    assert board.verifies("tier3", ["pump"]) and not board.verifies("tier1", ["pump"]) and board.verification_group(["pump"]) == "tier3"
    order = Step("order", {"component": "pump", "prediction_id": "pred/b1"}, {})

    def plan(group):
        return Plan(branch=1, mode="act_now", why="", steps=[order, Step("wait", {"cycles": 5}), Step("read", {"sensor_group": group, "cycles": 1}), Step("finish")])
    refused = [p for p in check_plan(plan("tier1"), board) if "verifies the repair" in p]
    assert len(refused) == 1 and "read tier3 for 1 cycle" in refused[0]
    assert check_plan(plan("tier3"), board) == []


@pytest.mark.parametrize("server", [True, False])
def test_the_controller_verifies_a_repair_on_its_first_verifying_reading(server):
    """In the runner, with and without the server: a reading of a non indicating group after the close leaves the repair
    pending (no verify_outcome call, no normal range verdict); the later reading of the indicating group is the one judged."""
    board = _modes_board(server)
    del board.status["open_work_orders"]
    board.add_order({"work_order": {"id": "wo-1", "component": "pump", "created_t": 1, "expected_close_t": None, "status": "open"}}, "pred/b1")
    board.add_notifications([{"event": "work_order_closed", "id": "wo-1", "component": "pump", "t": 6}])
    loop_runner = controller.CausalLoopRunner(llm=ScriptedPlanner(act_now), server_paths={RIG_SERVER: "unused", **({CAUSAL_SERVER: "unused"} if server else {})})
    loop_runner.board, calls = board, []

    async def diag(kind, **fields):
        calls.append(kind)
        return {"prediction_id": fields.get("prediction_id"), "status": "verified", "consistent": True}

    async def rank():
        return None
    loop_runner._diag, loop_runner._rank = diag, rank
    loop_runner._served = {(2, 1): [[2, 6, {"CE_CoolingEfficiency_percent_1Hz_mean": 45.0}]], (3, 1): [[3, 7, {"FS1_VolumeFlow_l_per_min_10Hz_mean": 10.0}]]}
    board.record_read({"plan": 2, "step": 1, "sensor_group": "tier1", "t_first": 6, "t_last": 6})
    asyncio.run(loop_runner._verify_ready())
    assert "pred/b1" not in board.verifications and calls == []
    board.record_read({"plan": 3, "step": 1, "sensor_group": "tier3", "t_first": 7, "t_last": 7})
    verified, _refuted = asyncio.run(loop_runner._verify_ready())
    assert [key for key, _v in verified] == ["pred/b1"]
    assert calls == (["verify"] if server else []) and (server or board.verifications["pred/b1"]["reading"] == [3, 1])


def test_time_and_extent_wait_for_the_profile_on_every_surface():
    """A plan made before the profile is read that names time or extent is refused with the sentence the model already reads
    before the profile, also after a profile step in the same plan, so no surface ever calls a kind it does not bind."""
    board = Board()
    for kinds in (["profile", "extent", "finish"], ["profile", "time"], ["time"]):
        plan = Plan(branch=None, mode="test_first", why="", steps=[Step(k) for k in kinds])
        problems = check_plan(plan, board)
        assert problems and all("read the profile first, which names the components and the record" in p for p in problems), (kinds, problems)


def test_a_first_plan_with_an_extent_step_on_the_rig_ends_without_an_error(tmp_path):
    """End to end: the rig binds no extent kind, so the first plan [profile, extent, finish] is refused and the episode runs on."""
    def plans(prompt, call):
        return plan_json([{"kind": "finish", "args": {}, "expect": {}, "evidence": []}])

    class ExtentFirst(ScriptedPlanner):
        def generate(self, prompt, temperature=0.0):
            if PLANNING_HEADING in prompt and "No profile read yet" in prompt and RETRY_HEADING not in prompt:
                self.prompts.append(prompt)
                return plan_json([{"kind": "profile"}, {"kind": "extent"}, {"kind": "finish"}], mode="test_first")
            return super().generate(prompt, temperature)
    ep_dir, row, _backend = _play(tmp_path, ExtentFirst(plans), {})
    assert row["error"] == "", row["error"]
    first = read_json(ep_dir / "plans.json")["plans"][0]
    assert any("read the profile first" in p for p in first["check_problems"][0])


def test_a_wait_on_a_row_without_a_clock_is_refused_with_the_sentence_of_the_prompt():
    """No clock runs on the row, so the plan check refuses a wait step as the prompt of the row says it."""
    board = _synthetic_board(read_first=False)
    board.clock_runs = False
    problems = check_plan(Plan(branch=1, mode="act_now", why="", steps=[Step("wait", {"cycles": 1}), Step("finish")]), board)
    assert problems == ["step 1 (wait): no clock runs on this row, so no wait is possible"]
    surface = AobSurface()
    surface.site = "MAIN"  # a surface that has read the profile of a row without a clock
    assert "No clock runs on this row, so no wait is possible" in surface.prompt_tools


def test_the_closing_summary_words_the_limits_as_the_board_does():
    """On a row without a horizon, a budget or a step cap the closing summary says so in the board's words, never inf."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import summary_turn
    board = _synthetic_board()
    free = {k: v for k, v in board.profile.items() if k not in ("budget", "horizon", "max_steps")}
    board.set_profile(free)
    board.status = {"time": 0, "open_work_orders": [], "episode_over": False}
    text = summary_turn(board, 6)
    assert not re.findall(r"\binf\b", text) and board.limits_line() in text and board.limits_line() in board.render()
    assert "no horizon (no clock runs)" in text and "No budget limit (every tool is free)" in text and "No step cap." in text


def test_an_answer_that_is_not_json_keeps_its_text():
    """The diagnostics answer, the rig's answer and the answer of AssetOpsBench's servers that is not a JSON
    object reach the board with the start of their text."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import RigSurface, not_json
    message = "Error executing tool rank_root_causes: the fits are missing"
    loop_runner = controller.CausalLoopRunner(llm=ScriptedPlanner(act_now), server_paths={RIG_SERVER: "unused", CAUSAL_SERVER: "unused"})
    loop_runner.board = _synthetic_board()

    async def call(server, tool, args):
        return message, None
    loop_runner._call = call
    assert asyncio.run(loop_runner._diag("rank"))["error"] == not_json("rank_root_causes", message) and message in not_json("x", message)
    assert message in RigSurface().parse("read", "read_sensors", message, loop_runner.board)["refused"]
    aob = AobSurface()
    assert message in aob.parse("wait", "wait_for_notification", message, loop_runner.board, timeout=1)["refused"]


def test_the_thinking_clocks_of_the_board_are_those_of_the_environment():
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import THINKING_CLOCKS
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import THINKING_MODES
    assert set(THINKING_CLOCKS) == set(THINKING_MODES)


@pytest.mark.parametrize("row", ["deterioration", "repair_rules"])
def test_an_open_window_is_priced_at_most_what_thinking_lets_the_engine_charge(row):
    """Played on the engine of the AssetOpsBench rows: once every recorded cycle of a group is read, a read asks for the open
    window, and the engine serves and charges the cycles that thinking played during the call. The board prices that read at the
    most it may serve and does not call it useless, so the charge never exceeds what the plan check counted, for 0, 1 and 3
    thinking cycles."""
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory import load_world as engine_world
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness

    class Thinking(harness.MemorySessions):
        next_think = 0

        def step(self, action):
            think, self.next_think = self.next_think, 0
            self.actions.append((action, think))
            return self.engine.step(action, think_cycles=think)

    class Calls:
        def __init__(self, meter):
            self.meter = meter

        async def call(self, server, tool, args):
            try:
                reply = await self.meter.call(server, tool, args)
            except KeyError as exc:
                return json.dumps({"error": str(exc)}), True
            return reply.text, reply.is_error

    world = engine_world("hyd_1", str(default_facts_path()))
    for think in (0, 1, 3):
        meter, engine, store, _sessions = harness.memory_engine_episode(world, row, seed=1)
        sessions = Thinking(engine)
        meter._sessions = sessions
        surface = AobSurface("rig", "PS1", ("mean", "std"))
        runner = controller.CausalLoopRunner(llm=None, server_paths={"iot": "x", "fmsr": "x", "wo": "x", "task": "x"}, surface=surface)
        runner.board = board = Board(server=False, evidence_form=surface.evidence_form, reads_take_time=surface.reads_take_time, noun="rig",
                                     summary_call=surface.summary_is_call)
        runner.trajectory, runner.plans, runner.surprises, runner._served = [], [], [], {}
        runner._sessions = Calls(meter)

        async def play():
            board.set_profile((await runner._act("profile"))["result"])
            board.required_opening = list(surface.opening_kinds(board))
            board.clock_runs = surface.can_observe_after(board, board.time)
            for kind in board.required_opening:
                await runner._act(kind)
            await runner._read("tier1", max(1, len(board.unpaid_cycles("tier1", 999) or [])))
            predicted, affordable = board.read_price("tier1", 1), read_affordable(board, "tier1", 1)
            sessions.next_think = think
            budget = board.budget_left
            await runner._read("tier1", 1)
            return predicted, affordable, budget - board.budget_left
        with bound(store, harness.NAMES):
            predicted, affordable, charged = asyncio.run(play())
        assert board.record_grows and charged <= predicted and affordable, (row, think, predicted, charged, affordable)


def test_where_the_board_knows_no_indicating_channel_only_the_group_of_every_channel_verifies():
    """A read verifies a repair when its group covers the indicating channels the board knows for the repaired
    components; where it knows none, only a group that holds every channel of the asset verifies, on either surface; the channels of
    an answer of the repair manual count once the board holds it; and where no group holds every channel no read can verify, so the
    tail waits for the close and buys no read."""
    from CausalDiagnosis.utils.public_rules.repair_completion import verifies
    groups = {"a": ["f1"], "b": ["f2"], "all": ["f1", "f2"]}
    assert verifies(["f1", "f2"], ["pump"], {}, groups) and not verifies(["f1"], ["pump"], {}, groups)
    assert verifies(["f1"], ["pump"], {"pump": {"f1"}}, groups) and not verifies(["f2"], ["pump"], {"pump": {"f1"}}, groups)
    assert not verifies(["f1"], ["pump", "valve"], {"pump": {"f1"}}, groups) and verifies(["f1", "f2"], ["pump", "valve"], {"pump": {"f1"}}, groups)
    board = _synthetic_board(read_first=False)
    board.manual_channels = {}
    full = dict(board.profile["sensor_groups"], tier4=[f for fields in board.profile["sensor_groups"].values() for f in fields])
    board.profile["sensor_groups"] = full
    board.profile["prices"]["read"]["tier4"] = 20
    assert board.verification_group(["pump"]) == "tier4" and not board.verifies("tier1", ["pump"])
    board.add_manual({"component": "pump", "indicating_channels": ["FS1_VolumeFlow_l_per_min_10Hz"], "effect": "x"})
    assert board.verification_group(["pump"]) == "tier3" and board.verifies("tier3", ["pump"]) and not board.verifies("tier1", ["pump"])
    board.add_manual({"component": "valve", "effect": "an answer without the field"})
    assert board.verification_group(["valve"]) == "tier4"
    bare = _synthetic_board(read_first=False)  # no group holds every channel
    bare.manual_channels = {}
    tail = bare.completion(["pump"], commit=False)
    assert tail.reads == 0 and tail.read_price == 0 and tail.first is not None


def test_every_registered_asset_has_a_group_of_every_channel():
    """The fallback where the board knows no indicating channel needs one group that holds every channel of an asset; every registered asset and the tiny asset
    of the asset tests has one."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR
    for path in sorted(ASSETS_DIR.glob("*/asset.json")) + [ASSETS_DIR / "tests" / "tiny_asset.json"]:
        kinds = json.loads(path.read_text(encoding="utf-8"))["observation_kinds"]
        groups = {g: set(spec.get("channels") if isinstance(spec, dict) else spec) for g, spec in kinds.items()}
        every = set().union(*groups.values())
        assert any(channels >= every for channels in groups.values()), path


def test_a_second_order_on_a_component_is_refused_only_while_the_first_is_open():
    """The environment refuses an order on a component with an open order, not one whose order has closed; the plan check
    refuses the second order only while the first is open at that step, with that reason."""
    board = _synthetic_board(read_first=False)
    board.diagnosis = {"diagnosis": dict(board.candidate(1)["combination"]), "evidence_ids": []}
    order = Step("order", {"component": "pump", "prediction_id": "pred/b1"}, {"prediction_id": "pred/b1"})
    twice = Plan(branch=1, mode="act_now", why="", steps=[order, order])
    assert any("is still open at this step; the environment refuses an order on a component with an open order" in p for p in check_plan(twice, board))
    after_close = Plan(branch=1, mode="act_now", why="", steps=[order, Step("wait", {"cycles": 5}), order])
    assert not [p for p in check_plan(after_close, board) if "open order" in p]


def test_the_role_text_names_every_end_of_an_episode_and_the_lead_time_of_its_row():
    """The role names every end of an episode, each where the row sets it, and a lead time only where the row sets
    one; the prompt without the server says a fault usually shows a channel outside its normal range."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import ROLE
    for end in ("finish", "the horizon", "the step cap", "the budget", "the limit of wall time", "a component failure", "where the row sets it"):
        assert end in ROLE, end
    assert "after the lead time of its row, where the row sets one" in ROLE
    without = system_prompt(6, server=False)
    assert "outside its normal range, while a fault usually shows at least one" in without and "and a fault shows at least one" not in without


def test_the_steps_left_of_the_surface_equal_the_environment_after_every_call_of_the_base_row(tmp_path, monkeypatch):
    """The base row appends no receipt, so the surface counts the steps itself: after every call of a played episode, the call of
    the profile included, its steps left equal the environment's, max_steps less the calls the meter recorded as a step."""
    from CausalDiagnosis.mcp_servers.utils import transcript

    seen = []
    parse = AobSurface.parse

    def recording(self, kind, tool, text, board, **fields):
        data = parse(self, kind, tool, text, board, **fields)
        seen.append(data.get("steps_left"))
        return data

    monkeypatch.setattr(AobSurface, "parse", recording)
    ep_dir, row, _backend = _aob_run(tmp_path, "base", ScriptedPlanner(act_now_on_assetopsbench), {})
    assert row["error"] == "", row["error"]
    max_steps = json.JSONDecoder().raw_decode(read_json(ep_dir / "trajectory.json")[0]["response"])[0]["max_steps"]
    calls = [line for line in transcript.read(ep_dir) if line["kind"] == "call" and line["server"] != CAUSAL_SERVER]
    stepped = [sum(1 for line in calls[:i + 1] if line["stepped"]) for i in range(len(calls))]
    assert len(seen) == len(calls) and seen == [max_steps - n for n in stepped]


# ----- one rule for the visits of the work orders -----

def test_the_board_follows_the_run_of_orders_as_the_environment_does():
    """Board.note_step: an accepted standard order opens a run or continues it, every other call of the environment ends it, a
    refused order included; add_order puts an order in the visit of the open run or in a visit of its own."""
    board = _synthetic_board()
    accepted = lambda i, t=0: {"result": {"work_order": {"id": f"wo-{i}", "component": COMPONENTS[i % len(COMPONENTS)], "created_t": t,  # noqa: E731
                                                          "expected_close_t": None, "status": "open", "mode": "standard"}}}
    for i, (kind, data) in enumerate([("order", accepted(1)), ("order", accepted(2, 3)), ("read", {"result": {}}), ("order", accepted(3, 4)),
                                      ("order", {"refused": "budget"}), ("order", accepted(4, 5))], 1):
        board.note_step(kind, data)
        if kind == "order" and "refused" not in data:
            board.add_order(data["result"], "")
    assert [o["visit"] for o in board.orders.values()] == [1, 1, 2, 3], board.orders


def test_the_plan_check_charges_one_visit_per_run_of_orders():
    """check_plan prices a visit charge per run of consecutive orders: two orders in a row pay one, a read between them ends the run
    and the second pays its own, and a plan whose first step is an order joins the board's open run and pays none."""
    board = _synthetic_board()
    board.profile = dict(board.profile, prices=dict(board.profile.get("prices") or {}, visit=7))
    board.diagnosis = {"diagnosis": {}, "evidence_ids": []}
    order = lambda c: Step("order", {"component": c, "prediction_id": "p"}, {}, [])  # noqa: E731
    read = Step("read", {"sensor_group": next(iter(board.profile["sensor_groups"])), "cycles": 1}, {}, [])
    a, b = COMPONENTS[0], COMPONENTS[1]

    def visit_charges(steps, run=None):
        board.run = run
        problems = check_plan(Plan(branch=1, mode="act_now", why="", steps=list(steps)), board)
        money = re.search(r"the plan's prices \(([0-9.]+)\)", " ".join(problems))
        return problems, money
    budget = board.budget_left
    for steps, run, charges in (([order(a), order(b)], None, 1), ([order(a), read, order(b)], None, 2), ([order(b)], 1, 0)):
        board.status["budget_left"] = 0  # every plan exceeds it, so the problem line names the plan's prices
        problems, money = visit_charges(steps, run)
        reads = board.read_price(read.args["sensor_group"]) if read in steps else 0
        parts = sum(board.order_price(s.args["component"]) for s in steps if s.kind == "order")
        assert money and float(money.group(1)) == parts + reads + 7 * charges, (steps, run, problems)
    board.status["budget_left"] = budget


def test_no_place_groups_orders_by_their_creation_cycle():
    """Guard against a second copy of the visit rule: the engine (clock, repair rules, env), the diagnostics server's schedule
    and the agent's board and plan check name no grouping by creation cycle (visit_of, visit_at, a comparison of created_t) and
    reach utils/public_rules/visits.py. The board and the plan check are packages, so every module of each is read and the
    needed name stands in one of them."""
    import ast
    places = {"CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/episode_clock.py": "schedule_orders",
              "CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_4_repair_rules.py": "visit_schedule",
              "CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py": "joins",
              "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/advice/troubleshooting_plans.py": "schedule_of_visit",
              "CausalDiagnosis/agents/Causal_Diagnosis_Agent/situation_board/": "joins",
              "CausalDiagnosis/agents/Causal_Diagnosis_Agent/plan/": "visit_closes"}
    for rel, needed in places.items():
        trees = {path: ast.parse(path.read_text(encoding="utf-8"))
                 for path in (sorted((ROOT / rel).glob("*.py")) if rel.endswith("/") else [ROOT / rel])}
        assert trees and needed in set().union(*map(_names, trees.values())), rel
        for path, tree in trees.items():
            assert not _names(tree) & {"visit_of", "visit_at"}, path
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare) and any(isinstance(op, ast.Eq) for op in node.ops):
                    assert "created_t" not in ast.unparse(node), (path, ast.unparse(node))


@pytest.mark.parametrize("row", ["deterioration", "repair_rules"])
def test_two_orders_with_thinking_between_them_form_one_visit_and_cost_what_the_board_counted(row):
    """Played on the engine of the AssetOpsBench rows through the controller's calls: after the diagnosis the controller orders
    two components with 0, 1 or 3 thinking cycles before the second. The engine logs one visit and one visit charge, the board
    puts both orders in one visit, the budget drops by the price Board.completion counted for the orders before them, and the
    closes the board estimates are the engine's."""
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory import load_world as engine_world
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests import harness

    class Thinking(harness.MemorySessions):
        next_think = 0

        def step(self, action):
            think, self.next_think = self.next_think, 0
            self.actions.append((action, think))
            return self.engine.step(action, think_cycles=think)

    class Calls:
        def __init__(self, meter):
            self.meter = meter

        async def call(self, server, tool, args):
            try:
                reply = await self.meter.call(server, tool, args)
            except KeyError as exc:
                return json.dumps({"error": str(exc)}), True
            return reply.text, reply.is_error

    world = engine_world("hyd_1", str(default_facts_path()))
    for think in (0, 1, 3):
        meter, engine, store, _sessions = harness.memory_engine_episode(world, row, seed=1)
        sessions = Thinking(engine)
        meter._sessions = sessions
        surface = AobSurface("rig", "PS1", ("mean", "std"))
        runner = controller.CausalLoopRunner(llm=None, server_paths={"iot": "x", "fmsr": "x", "wo": "x", "task": "x"}, surface=surface)
        runner.board = board = Board(server=False, evidence_form=surface.evidence_form, reads_take_time=surface.reads_take_time, noun="rig",
                                     summary_call=surface.summary_is_call)
        runner.trajectory, runner.plans, runner.surprises, runner._served = [], [], [], {}
        runner._sessions = Calls(meter)
        first, second = engine.config.components[:2]

        async def play():
            board.set_profile((await runner._act("profile"))["result"])
            board.required_opening = list(surface.opening_kinds(board))
            board.clock_runs = surface.can_observe_after(board, board.time)
            for kind in board.required_opening:
                await runner._act(kind)
            await runner._read("tier1", 1)
            levels = dict(zip(engine.config.components, engine.hidden))
            assert "refused" not in await runner._act("diagnose", levels=levels, evidence=runner._last_reading())
            board.diagnosis = {"diagnosis": levels, "evidence_ids": []}
            tail, budget = board.completion([first, second]), board.budget_left
            for component, thinking in ((first, 0), (second, think)):
                sessions.next_think = thinking
                data = await runner._act("order", component=component, prediction_id="", evidence=[])
                assert "refused" not in data, data
                board.add_order(data["result"], "plan 1")
            return tail, budget - board.budget_left
        with bound(store, harness.NAMES):
            tail, charged = asyncio.run(play())
        created = [e for e in engine.log if e["event"] == "work_order_created"]
        assert len({e["visit"] for e in created}) == 1 and [bool(e["visit_charge"]) for e in created] == [bool(engine.config.prices["visit"]), False]
        assert len({o["visit"] for o in board.orders.values()}) == 1 and board.run is not None
        assert charged == tail.price - tail.read_price, (row, think, charged, tail)
        closes = {o.component: o.close_t for o in engine.clock.orders}
        assert {o["component"]: o["expected_close_t"] for o in board.orders.values()} == closes, (row, think, board.orders, closes)


def test_no_text_states_the_visit_of_one_cycle():
    """Every text an agent or a reader of the docs meets states the run rule of the visits: no file of the agents, the servers
    with their tool contracts, the engine, the presets or the docs says that the orders of one cycle form a visit (the oracle's
    and the greedy policy's own plans, whose orders of one cycle do follow each other, are words about a plan, not the rule)."""
    root = Path(__file__).resolve().parents[4]
    old = re.compile(r"(created|orders) (in|of) one cycle (form|as one|so that one|forming)", re.I)
    found = [f"{path.relative_to(root)}:{n}" for folder in ("src/agent", "src/servers", "src/envs", "docs", "CausalDiagnosis/configs")
             for path in sorted((root / folder).rglob("*")) if path.suffix in (".py", ".md", ".json") and "tests" not in path.parts
             for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1) if old.search(line)]
    assert not found, found
