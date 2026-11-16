import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import HYDRAULIC_RIG
from CausalDiagnosis.agents.utils.tool_answers import EPISODE_OVER
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS as AOB_SERVERS, CAUSAL_SERVER, RIG_SERVER
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, RELEASED_FILE, STATE_FILE, episode_name, read_rows
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.utils.public_rules.visits import is_no_call
from . import controller_walk


RUN_COLUMNS = ("config", "agent", "model", "prices", "ranking_model")  # settings of scores.csv, never its scores or its start
SERVER_NAMES = {RIG_SERVER: "action", CAUSAL_SERVER: "diagnostics"}
ACTION_SERVERS = (RIG_SERVER, *AOB_SERVERS)  # the servers whose answers carry the clock, the budget and the notifications
# the tool of every action kind of the controller on its two surfaces in CausalDiagnosis/agents/Causal_Diagnosis_Agent/tool_surfaces.py, the hydraulic rig's
# action server and AssetOpsBench's servers with the task server. On the latter the fix history is list_workorders, a read
# is history, an order is generate_work_order, the change summary is summarize_changes_since of the task server where the row serves
# it (the level with the CausalDiagnosis MCP) and no call elsewhere, and the opening reads the clock and the record
RIG_TOOLS = {kind: binding.tool for kind, binding in HYDRAULIC_RIG.items()}
AOB_TOOLS = {**RIG_TOOLS, "fix_history": "list_workorders", "read": "history", "order": "generate_work_order",
             "time": "get_current_time", "extent": "stream_extent"}


# ----- the files of an episode -----
@dataclass
class Call:
    """One tool call as the runner recorded it, with its answer parsed once."""

    server: str
    tool: str
    args: object
    text: str
    error: str = None
    task: str = None
    plan_step: object = None

    def __post_init__(self):
        self.answer = aob_envelope(self.text) if self.server in AOB_SERVERS else json_object(self.text)

    @property
    def result(self):
        result = (self.answer or {}).get("result")
        return result if isinstance(result, dict) else None


def json_object(text):
    """The JSON object a text holds, or None."""
    try:
        data = json.loads(text) if text else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def aob_envelope(text):
    """An answer of AssetOpsBench's servers or of the task server as the rig's envelope. From the priced row on the metering
    layer appends one JSON line to AssetOpsBench's text, with the price, the budget left, the steps left and, on the clock
    rows, the cycle after the call and the notifications. The text before it is the result, or the refusal when it is an
    error object, and stays text when it holds no JSON object, as an error of AssetOpsBench's own tools does."""
    head, _, last = (text or "").rstrip("\n").rpartition("\n")
    receipt = json_object(last) if head else None
    if receipt is None or "steps_left" not in receipt:
        head, receipt = text or "", {}
    data = json_object(head)
    envelope = {key: receipt[key] for key in ("budget_left", "steps_left", "notifications") if key in receipt}
    if "cycle" in receipt:
        envelope["time"] = receipt["cycle"]
    if data is None:
        envelope["text"] = head
    elif set(data) == {"error"}:
        envelope["refused"] = data["error"]
    else:
        envelope["result"] = data
    ended = any(isinstance(n, dict) and n.get("event") == "end" for n in envelope.get("notifications") or [])
    if ended or str(envelope.get("refused", "")).startswith(EPISODE_OVER):
        envelope["episode_over"] = True
    return envelope


def recorded_calls(folder, state):
    """The tool calls in order and, apart, the plan steps that named no tool, from the runner's trajectory of both servers when
    it wrote one, else the action server's own record, since the reference policies play through the action server alone."""
    trajectory = read_json(folder / "trajectory.json")
    if trajectory and all(isinstance(r, dict) and "tool" in r for r in trajectory):
        calls, toolless = [], []
        for r in trajectory:
            if r.get("tool") and str(r["tool"]).lower() not in ("none", "null"):
                response = r.get("response")
                text = json.dumps(response) if isinstance(response, (dict, list)) else str(response or "")
                calls.append(Call(r.get("server") or RIG_SERVER, r["tool"], r.get("tool_args"), text, r.get("error"), r.get("task"), r.get("step")))
            else:
                toolless.append(r)
        return calls, toolless
    return [Call(RIG_SERVER, c["tool"], c.get("arguments"), c.get("response") or "") for c in state.get("tool_calls", [])
            if not is_no_call(c)], []


class Episode:
    """The public files of one episode folder and the run settings of its run folder. hidden() reads the files under
    hidden/, which parts 3 and 4 alone call."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.state = read_json(self.folder / STATE_FILE)
        if self.state is None:
            raise SystemExit(f"{self.folder} holds no {STATE_FILE}, so it is not the episode folder of a run")
        self.released = read_json(self.folder / RELEASED_FILE)
        if not self.released:
            raise SystemExit(f"{self.folder} holds no {RELEASED_FILE}, so the episode has not begun or its files were cut off")
        self.config = self.state["config"]
        self.components = list(self.config["components"])
        question = self.folder / "question.txt"
        self.question = question.read_text(encoding="utf-8") if question.exists() else self.released.get("task_text") or ""
        self.run = read_json(self.folder.parent / "run.json") or {}
        self.row = self._run_columns()
        self.llm_calls = read_json(self.folder / "llm_calls.json") or []
        self.plans = read_json(self.folder / "plans.json")
        plan = self.folder / "plan.txt"
        self.plan_text = plan.read_text(encoding="utf-8") if plan.exists() else None
        self.calls, self.toolless = recorded_calls(self.folder, self.state)
        self.aob = any(call.server in AOB_SERVERS for call in self.calls)
        self.tools = AOB_TOOLS if self.aob else RIG_TOOLS
        self.kind = self._kind()
        self.bands = (self.released.get("rules") or {}).get("normal_ranges") or {}

    def hidden(self, name):
        return read_json(self.folder / HIDDEN / name)

    @cached_property
    def walk(self):
        """The plan step label of every call of the CausalDiagnosis controller and whether the walk placed them all."""
        return controller_walk.causal_walk(self.calls, (self.plans or {}).get("plans") or [], self.tools, (self.plans or {}).get("board"))

    @cached_property
    def children(self):
        """The component of every child asset of the profile, which the orders of the AssetOpsBench rows name."""
        asset = self.profile()[0].get("asset") or {}
        return {child: component for component, child in (asset.get("children") or {}).items()}

    def group_of(self, call):
        """The sensor group whose channels and their _std fields a history call asked for, else None."""
        sensors = call.args.get("sensors") if isinstance(call.args, dict) else None
        groups = (self.released.get("rules") or {}).get("sensor_groups") or {}
        return next((g for g, names in groups.items() if set(sensors or ()) == {f for n in names for f in (n, f"{n}_std")}), None)

    def setting(self, run_key, row_key=None):
        """A run setting from run.json, else from this episode's row of scores.csv, else None."""
        value = self.run.get(run_key)
        if value in (None, "") and row_key:
            value = self.row.get(row_key) or None
        return value

    def _run_columns(self):
        for row in read_rows(self.folder.parent):
            if episode_name(row["seed"], row.get("scenario_id"), row.get("gepa_call")) == self.folder.name:
                return {k: row.get(k, "") for k in RUN_COLUMNS}
        return {}

    def _kind(self):
        """causal, plan_execute (the scripted agent runs the same runner), policy, or the agent run.json names."""
        agent = self.run.get("agent") or ""
        if agent == "causal" or self.plans is not None:
            return "causal"
        if agent in ("plan_execute", "scripted") or (not agent and any(c.task for c in self.calls)):
            return "plan_execute"
        if agent in run_episodes.POLICY_AGENTS or (not agent and not self.llm_calls and not (self.folder / "trajectory.json").exists()):
            return "policy"
        return agent or "unknown"

    def label(self):
        return self.setting("agent_label", "agent") or self.kind

    def profile(self):
        """The asset profile the agent saw at step one and a sentence on its source, which is the answer of the agent's first
        get_asset_profile call, else the public rules of released.json with their tool table, the manifest of the episode's
        configuration that the envelope opening the episode hands a reference policy."""
        for n, call in enumerate(self.calls, 1):
            if call.tool == "get_asset_profile" and call.result is not None:
                return call.result, f"It is the answer of `get_asset_profile` at step {n}, the call that hands the agent the profile."
        profile = dict(self.released.get("rules") or {})
        profile.update(asset_id=self.released.get("asset_id"), seconds_per_cycle=self.config.get("seconds_per_cycle"),
                       think_seconds_per_cycle=self.config.get("think_seconds_per_cycle"))
        source = ("This agent never called `get_asset_profile`, since a reference policy receives the profile in the envelope that opens "
                  "the episode.")
        if profile.get("tools"):
            return profile, source + " The rules and the tool table below are those of `released.json`."
        return {**profile, "tools": []}, source + " The rules below are those of `released.json`, which holds no tool table."
