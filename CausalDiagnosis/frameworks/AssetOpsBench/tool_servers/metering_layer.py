"""The metering layer between an agent and AssetOpsBench's iot, fmsr and wo servers.

Every call first reaches the engine as one action, so it counts toward the step cap and pays its thinking time. The
engine moves the clock by the thinking time, charges a read for every cycle it returns that no group paid earlier for
that cycle covers, at the price of the cheapest group that covers the fields served, so a re-read is free, plays
the cycles a read reaches beyond the clock where reads have a declared duration, and accepts or refuses a work
order, which closes inside the order call on the rows without a clock. The producer then writes what
the step recorded, and only then does AssetOpsBench's own tool run, in this process and over the episode's databases,
so the database never holds a future row and the rows served are the rows charged. The agent reads AssetOpsBench's text
byte for byte, followed from the priced row on by one JSON line with the price, the budget left and the steps left, and
in the clock rows the cycle and the notifications. Every call appends one entry to the state file's tool_calls.
The task tools of the rig's action server run through the same lock, step, store writes, receipt and log: the asset
profile, one diagnosis or one declaration, finish and, from the real time clock row on, the two time tools.

The engine calls are those of RecordedRigEnv, which tests/engine_stub.py implements as their reference: a step
that takes an action and the thinking cycles and returns the engine's envelope, `read_recorded` with the read window in
cycles, `aob_read` for a free read, and `public_events` and `recorded_cycles` to read what is public.
"""
import asyncio
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents import class_keys
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_1_priced as priced
import servers.fmsr.main as fmsr_server
import servers.iot.main as iot_server
import servers.wo.main as wo_server
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract, episode_start_and_replay as rows
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import AOB_SERVERS, TASK_SERVER, WORK_TYPE_MODES, history_order_id
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import SOURCE_AGENTS, SOURCE_TRIGGERS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import NO_CALL_ACTION, UNLISTED_ACTION
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import action_of, build_server as build_rig_server
from servers.iot.telemetry_helper import _decode_history_cursor, _history_cursor_context
from CausalDiagnosis.utils.atomic_writes_and_locks import locked
from servers.wo import workorders

SERVERS = {"iot": iot_server, "fmsr": fmsr_server, "wo": wo_server}
# the AssetOpsBench tools every row lists, the hidden ones left out
TOOLS = {server: layer_contract.tools_of(server) for server in AOB_SERVERS}
READS = ("history", "latest_reading")
ASSET_ARGUMENTS = ("asset_id", "asset_num")  # the arguments of the AssetOpsBench tools that name an asset
DEFINITIONS_ONLY = Path(os.devnull)  # the rig's action server is built for its tool definitions alone, so no call reaches it


@dataclass(frozen=True)
class Row:
    name: str
    receipt: bool  # the agent sees the receipt line from the priced row on
    clock: bool  # the receipt adds the cycle and the notifications in the two clock rows


ROWS = {row.name: row for row in (Row("base", False, False), Row("priced", True, False), Row("real_time", True, True),
                                 Row("advanced_real_time", True, True), Row("repair_rules", True, True), Row("deterioration", True, True))}


@dataclass(frozen=True)
class MeterConfig:
    row: Row
    task_tools: tuple  # the task tools the configuration lists, layer_contract.tools_for
    site: str
    asset_id: str
    components: tuple
    field_channels: dict  # installed field name -> channel, the field of the plain statistic
    sensor_groups: dict  # group -> channels
    read_prices: dict  # group -> price per returned cycle
    timeline: object  # CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.cycle_timestamps.Timeline
    classes: dict  # fmsr class key -> component, as the documents of the asset file it
    statistics: tuple  # the served statistics, the plain one first
    episode: object = None  # the configuration of the episode, whose cards and clock the texts of the task tools follow
    asset_alias: str = None  # another id of the asset, the one the question of an anchored episode names (EpisodeConfig.anchors)

    def child_asset(self, component):
        return layer_contract.child_asset(self.asset_id, component)

    def served(self, arguments):
        """The arguments as AssetOpsBench serves them: an argument that names an asset by the alias names the asset, so that a
        call on the asset id of the question reads and writes as the same call on the asset's own id."""
        if self.asset_alias is None:
            return arguments
        return {key: self.asset_id if key in ASSET_ARGUMENTS and value == self.asset_alias else value for key, value in arguments.items()}

    def component_of(self, asset_num, site_id):
        """The rig component of a child asset at the episode's site, the site compared in upper case on both sides."""
        if str(site_id).upper() != self.site.upper():
            return None
        return next((c for c in self.components if self.child_asset(c) == asset_num), None)

    def profile_view(self, profile):
        """The asset profile as an agent on the AssetOpsBench rows reads it: the child asset of every component beside the
        asset, the normal range of the plain statistic of a channel under the record field that carries it, and no fault
        rule of the starts. The tool table and the clock are the layer's already, since the engine of the
        rows publishes them."""
        view = {key: value for key, value in profile.items() if key != "fault_rule"}
        view["asset"] = {**(profile.get("asset") or {}), "children": {c: self.child_asset(c) for c in self.components}}
        view["normal_ranges"] = {field.removesuffix(f"_{self.statistics[0]}"): band for field, band in profile["normal_ranges"].items()}
        return view

    def engine_field(self, field):
        """A record field as the engine names the served value: the bare field of a channel is its plain statistic, under
        `<field>_<plain>`, and `<field>_<statistic>` names every further served statistic."""
        if field in self.field_channels:
            return f"{field}_{self.statistics[0]}"
        if any(field.endswith(f"_{s}") and field[:-len(s) - 1] in self.field_channels for s in self.statistics[1:]):
            return field
        raise MeterError(f"field {field} belongs to no channel")

    def group_of(self, fields):
        """The sensor group a read of the fields is charged to, the rule of the priced layer."""
        channels = {self.field_channels[self.engine_field(f).rsplit("_", 1)[0]] for f in fields}
        return priced.cheapest_group(self.read_prices, self.sensor_groups, channels)


def meter_config(row, config, timeline, field_channels=None, task_tools=None):
    """The layer's view of an episode configuration of the engine: the row, the task tools that the configuration's big
    layers list unless the caller names them, the asset with its site, its components and their fmsr class keys, the
    sensor groups with their read prices, the served statistics, and the installed field of every channel, which the
    configuration's asset profile gives unless the caller names them."""
    if task_tools is None:
        task_tools = layer_contract.tools_for(TASK_SERVER, config)
    profile = get_profile(config.asset_id)
    if field_channels is None:
        field_channels = {profile.field_name(channel): channel for channel in profile.channels}
    classes = {key: component for component, key in class_keys(profile).items() if component is not None}
    return MeterConfig(row=ROWS[row], task_tools=tuple(task_tools), site=profile.text["fallback"]["site"], asset_id=config.asset_id,
                       components=tuple(config.components), field_channels=dict(field_channels),
                       sensor_groups={g: tuple(s) for g, s in config.sensor_groups.items()}, read_prices=dict(config.prices["read"]),
                       timeline=timeline, classes=classes, statistics=tuple(profile.statistics["served"]),
                       episode=config if isinstance(config, EpisodeConfig) else None,
                       asset_alias=(getattr(config, "anchors", None) or {}).get("asset_alias"))


class MeterError(RuntimeError):
    """The engine and AssetOpsBench's tool disagree, which is a bug of the layer or the engine, never an agent error."""


@dataclass(frozen=True)
class Reply:
    text: str
    is_error: bool = False


class Meter:
    def __init__(self, config, sessions, producer):
        self.config = config
        self._sessions = sessions
        self._producer = producer
        self._lock = asyncio.Lock()  # one process may serve concurrent calls (Agent-As-Tool), the state file lock serves the others

    def tools_of(self, server):
        """The tools a server lists in the episode: AssetOpsBench's on every row, and the task tools of the configuration,
        which the two time tools join where the real-time clock lists them and the repair manual where the repair rules run
        on a clock."""
        if server == TASK_SERVER:
            return self.config.task_tools
        return TOOLS.get(server, ())

    async def listed_tools(self, server):
        """The tool definitions of a server, names, descriptions and schemas unchanged, hidden ones out: AssetOpsBench's
        own, and for the task server those of the rig's action server, whose task tools it serves with their signatures."""
        app = (build_rig_server(DEFINITIONS_ONLY, self.config.asset_id, record=False, config=self.config.episode) if server == TASK_SERVER
               else SERVERS[server].mcp)
        return [t for t in await app.list_tools() if t.name in self.tools_of(server)]

    async def call(self, server, tool, arguments=None):
        """One call through the engine. The entry of the service is read after the lock is held, in the thread that serves
        the call, and the exit line follows in every case (episode.enter, episode.leave)."""
        async with self._lock:
            entered = self._sessions.enter(server, tool)
            outcome = "raised"
            try:
                with self._sessions(entered) as session:
                    reply = await self._call(session, server, tool, dict(arguments or {}))
                outcome = "answered"
                return reply
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                self._sessions.leave(server, entered, outcome)

    async def no_call(self):
        """The no_call step of an iteration of the agent that called no tool: an internal entry of
        the task process and no listed tool. It is a counted call of the thinking clock, entered, stamped and charged by the clock
        itself at its entry with no cycles passed, under the same lock as every metered step, and it goes through _step with the
        store sync on, so the cycles it plays reach the store as after any step. The engine counts it as one step of the step cap
        and logs the event no_call apart from the refusals; the engine ends the episode at the step cap as it does on the rig.
        Returns whether the episode is over after it."""
        async with self._lock:
            entered = self._sessions.enter(TASK_SERVER, NO_CALL_ACTION)
            outcome = "raised"
            try:
                with self._sessions(entered) as session:
                    before = session.engine.time
                    if session.engine.done:  # answered and recorded as a refusal, as a call after the end is on the rig (episode.call)
                        envelope, refusal = session.engine.envelope(refused=episode.EPISODE_OVER), episode.EPISODE_OVER
                    else:
                        envelope, refusal = await self._step(session, {"type": NO_CALL_ACTION})
                    self._close(session, TASK_SERVER, NO_CALL_ACTION, {}, before, envelope, Reply(json.dumps(envelope.get("result"))), 0, {}, refusal)
                    over = session.engine.done
                outcome = "answered"
                return over
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                self._sessions.leave(TASK_SERVER, entered, outcome)

    # ----- one call -----
    async def _call(self, session, server, tool, arguments):
        engine = session.engine
        before = engine.time
        if engine.done:
            envelope = engine.envelope(refused=episode.EPISODE_OVER)
            return self._close(session, server, tool, arguments, before, envelope, None, 0, {}, episode.EPISODE_OVER)
        if tool not in self.tools_of(server):
            # a tool the row does not list is a refused call of the engine, which counts it as a step as the briefing says
            envelope, refusal = await self._step(session, {"type": UNLISTED_ACTION, "server": server, "tool": tool})
            return self._close(session, server, tool, arguments, before, envelope, None, 0, {}, refusal)
        if server == TASK_SERVER:
            return await self._task(session, tool, arguments, before)
        if tool in READS:
            return await self._read(session, tool, arguments, before)
        if tool == "generate_work_order":
            return await self._order(session, arguments, before)
        if tool == "get_failure_modes":
            key = fmsr_server._asset_class_key(arguments.get("asset_class", ""))
            action = {"type": "get_failure_modes", "component": self.config.classes.get(key, "")}
        else:
            action = {"type": "aob_read", "server": server, "tool": tool}
            if server == "wo":
                shown = await self._past_orders_shown(session, tool, arguments)
                if shown:
                    action["fix_history"] = shown
        return await self._free(session, server, tool, arguments, before, action)

    async def _free(self, session, server, tool, arguments, before, action):
        envelope, refusal = await self._step(session, action)
        if refusal:
            return self._close(session, server, tool, arguments, before, envelope, None, 0, {}, refusal)
        with self._cycle_clock(session.engine.time):
            reply = await self._aob(server, tool, arguments)
        return self._close(session, server, tool, arguments, before, envelope, reply, 0, _count_summary(reply))

    async def _past_orders_shown(self, session, tool, arguments):
        """The engine identifiers of the past orders of the fix history that a wo read shows, from a first run of the read,
        which writes nothing. The producer writes the past orders at reset and never changes them, so the read shows the
        same ones after its step, and the engine logs them as the fix history the agent read."""
        with self._cycle_clock(session.engine.time):
            probe = await self._aob("wo", tool, arguments)
        try:
            data = json.loads(probe.text)
        except ValueError:
            return []
        return list(dict.fromkeys(i for i in map(history_order_id, _wonums(data)) if i is not None))

    async def _read(self, session, tool, arguments, before):
        """history and latest_reading. AssetOpsBench checks the arguments first, a check the clock does not change, and a
        call it answers with an error, or a call on another asset, is a free read."""
        probe = await self._aob("iot", tool, arguments)
        served = self.config.served(arguments)
        if (probe.is_error or _error_of(probe.text) is not None or arguments.get("site_name") != self.config.site
                or served.get("asset_id") != self.config.asset_id):
            return await self._free(session, "iot", tool, arguments, before, {"type": "aob_read", "server": "iot", "tool": tool})
        parsed = _validated("iot", tool, served)
        fields = self._served_fields(tool, parsed)
        action = {"type": "read_recorded", "tool": tool, "sensor_group": self.config.group_of(fields),
                  "fields": [self.config.engine_field(f) for f in fields]}
        if tool == "history":
            action.update(self._window(parsed))
        envelope, refusal = await self._step(session, action)
        if refusal:
            return self._close(session, "iot", tool, arguments, before, envelope, None, 0, {}, refusal)
        result = envelope["result"]
        with self._cycle_clock(session.engine.time):
            reply = await self._aob("iot", tool, arguments)
        returned = self._returned_cycles(tool, reply)
        if returned != result["cycles"]:
            raise MeterError(f"{tool} returned cycles {returned} while the engine served {result['cycles']}")
        cycles = result["cycles"]
        summary = {"rows": len(cycles), "cycle_span": [cycles[0], cycles[-1]] if cycles else None, "fields": fields,
                   "newly_paid": len(result["paid"]), "rows_read_again": len(cycles) - len(result["paid"]),
                   "new_observations": result["observations"]}
        return self._close(session, "iot", tool, arguments, before, envelope, reply, result["price"], summary)

    async def _order(self, session, arguments, before):
        """generate_work_order. A call AssetOpsBench or the work order database would refuse writes nothing and costs
        nothing, so it is a free read that returns AssetOpsBench's own error, and every other call is the engine's order."""
        parsed = _validated("wo", "generate_work_order", self.config.served(arguments))
        if parsed is None or _order_problem(parsed):
            return await self._free(session, "wo", "generate_work_order", arguments, before,
                                    {"type": "aob_read", "server": "wo", "tool": "generate_work_order"})
        source = parsed.get("aob_source") or {}
        component = self.config.component_of(parsed["asset_num"], parsed["site_id"])
        work_type = parsed["work_type"]
        action = {"type": "create_work_order", "component": component or parsed["asset_num"],
                  "mode": WORK_TYPE_MODES.get(work_type, work_type), "prediction_id": source.get("prediction_id", ""),
                  "evidence_ids": rewrite_evidence(source.get("evidence_ids", []), _observations(session.engine), self.config)}
        envelope, refusal = await self._step(session, action, sync=False)
        if refusal:
            await self._producer.sync(session.engine)
            return self._close(session, "wo", "generate_work_order", arguments, before, envelope, None, 0, {}, refusal)
        order = envelope["result"]["work_order"]
        with self._cycle_clock(order["created_t"]):
            reply = await self._aob("wo", "generate_work_order", {**arguments, "wonum": order["id"]})
        if reply.is_error or _error_of(reply.text) is not None:
            raise MeterError(f"the engine created {order['id']} but wo refused it: {reply.text}")
        await self._producer.sync(session.engine)  # the order's status changes, once its document exists
        return self._close(session, "wo", "generate_work_order", arguments, before, envelope, reply, envelope["result"]["price"],
                           {"work_order": order})

    async def _task(self, session, tool, arguments, before):
        """A task tool: the engine's action built as the rig's action server builds it, the evidence identifiers mapped as
        for an order, the cycles the step played written to the store, and the engine's result as the answer."""
        engine = session.engine
        action = action_of(tool, arguments, engine.config.components)
        if isinstance(action.get("evidence_ids"), list):
            action["evidence_ids"] = rewrite_evidence(action["evidence_ids"], _observations(engine), self.config)
        envelope, refusal = await self._step(session, action)
        if refusal:
            return self._close(session, TASK_SERVER, tool, arguments, before, envelope, None, 0, {}, refusal)
        result = envelope["result"]
        if tool == "get_asset_profile":
            result = self.config.profile_view(result)
        elif tool == "get_current_time":
            result = {**result, "current_datetime": self.config.timeline.timestamp(result["time"])}  # as ARE's System app
        reply = Reply(json.dumps(result, indent=2, ensure_ascii=False))
        written = layer_contract.entries(TASK_SERVER)[tool]["class"] == "write"
        return self._close(session, TASK_SERVER, tool, arguments, before, envelope, reply, 0, result if written else _count_summary(reply))

    # ----- parts -----
    async def _step(self, session, action, sync=True):
        envelope, _cost, _done, info = session.step(action)
        if sync:
            await self._producer.sync(session.engine)
        return envelope, info.get("refused")

    async def _aob(self, server, tool, arguments):
        try:
            result = await SERVERS[server].mcp.call_tool(tool, self.config.served(arguments))
        except ToolError as exc:  # what an MCP client receives as an error result
            return Reply(str(exc), True)
        blocks = result[0] if isinstance(result, tuple) else result
        return Reply("\n".join(block.text for block in blocks))

    @contextmanager
    def _cycle_clock(self, cycle):
        """AssetOpsBench's four readers of the wall clock see the cycle's time instead: wo's creation, KPIs and calendar
        through their own `now`, and the age of latest_reading."""
        now = self.config.timeline.utc(cycle)
        saved = [(workorders, name, getattr(workorders, name)) for name in ("create_workorder", "get_workorder_kpis", "get_schedule_calendar")]
        saved.append((iot_server, "_timestamp_age_seconds", iot_server._timestamp_age_seconds))
        for module, name, function in saved[:3]:
            setattr(module, name, partial(function, now=now))
        iot_server._timestamp_age_seconds = lambda moment: (now - (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc))).total_seconds()
        try:
            yield
        finally:
            for module, name, function in saved:
                setattr(module, name, function)

    def _served_fields(self, tool, arguments):
        chosen = arguments.get("sensors") if tool == "history" else ([arguments["sensor"]] if arguments.get("sensor") is not None else None)
        return list(dict.fromkeys(chosen)) if chosen is not None else iot_server.get_sensor_list(self.config.asset_id)

    def _window(self, arguments):
        """history's half-open window and page in cycles, for the engine to serve after the thinking time."""
        timeline = self.config.timeline
        bounds = {k: timeline.first_cycle_from(datetime.fromisoformat(arguments[k])) if arguments.get(k) is not None else None
                  for k in ("start", "end")}
        offset = 0
        if arguments.get("cursor") is not None:
            sensors = arguments.get("sensors")
            context = _history_cursor_context(arguments["site_name"], arguments["asset_id"], arguments.get("start"), arguments.get("end"),
                                              list(dict.fromkeys(sensors)) if sensors is not None else None)
            offset, _ = _decode_history_cursor(arguments["cursor"], context)
        return {**bounds, "offset": offset, "limit": arguments.get("limit", iot_server.PAGE_SIZE)}

    def _returned_cycles(self, tool, reply):
        data = json.loads(reply.text)
        stamps = [o["timestamp"] for o in data["observations"]] if tool == "history" else [data["timestamp"]]
        cycles = [self.config.timeline.cycle_of(datetime.fromisoformat(s)) for s in stamps]
        if any(c != round(c) for c in cycles):
            raise MeterError(f"{tool} returned a timestamp between two cycles")
        return [round(c) for c in cycles]

    def _close(self, session, server, tool, arguments, before, envelope, reply, price, summary, refusal=None):
        """The agent's text and the call's entry in tool_calls."""
        receipt = {"price": price, "budget_left": envelope.get("budget_left"), "steps_left": envelope["steps_left"]}
        if self.config.row.clock:
            receipt.update(cycle=envelope["time"], notifications=envelope["notifications"])
        line = json.dumps(receipt, separators=(",", ":"), ensure_ascii=False)
        if refusal is not None:
            reply = Reply(json.dumps({"error": refusal}, indent=2, ensure_ascii=False), True)
        entry = {"server": server, "tool": tool, "arguments": arguments, "summary": summary, "price": price, "time": envelope["time"],
                 "cycles": envelope["time"] - before, "budget_left": envelope.get("budget_left"), "steps_left": envelope["steps_left"],
                 "notifications": envelope["notifications"], "response": line}
        if refusal is not None:
            entry["refused"] = refusal
        session.log_call(entry)
        sent = Reply(f"{reply.text}\n{line}" if self.config.row.receipt else reply.text, reply.is_error)
        session.answered(server, tool, arguments, sent)  # the text the agent reads, for the transcript of the episode
        return sent


def rewrite_evidence(identifiers, observations, config):
    """obs/<timestamp>/<field>, as an agent reads it off a row, becomes obs/<n>/<field> of the value's first serving in the
    engine's naming, and wo/<wonum> of a past order of the fix history becomes the engine's hist/<n>. An
    identifier it cannot map passes unchanged, and the provenance check flags it."""
    first = {}
    for o in observations:
        for field in o["values"]:
            first.setdefault((o["t"], field), o["n"])
    rewritten = []
    for identifier in identifiers:
        parts = identifier.split("/", 2) if isinstance(identifier, str) else []
        mapped = None
        if len(parts) == 3 and parts[0] == "obs" and not parts[1].isdigit():
            try:
                cycle = config.timeline.cycle_of(datetime.fromisoformat(parts[1]))
                field = config.engine_field(parts[2])
            except (ValueError, MeterError):
                cycle = field = None
            if field is not None and cycle == round(cycle) and (round(cycle), field) in first:
                mapped = f"obs/{first[(round(cycle), field)]}/{field}"
        elif len(parts) == 2 and parts[0] == "wo":
            mapped = history_order_id(parts[1])
        rewritten.append(mapped or identifier)
    return rewritten


def _observations(engine):
    return [o for e in engine.public_events() if e["event"] == "read" for o in e["served"]]


def _wonums(data):
    """Every wonum a wo answer names, at any depth."""
    found = []
    if isinstance(data, dict):
        if isinstance(data.get("wonum"), str):
            found.append(data["wonum"])
        for value in data.values():
            found.extend(_wonums(value))
    elif isinstance(data, list):
        for value in data:
            found.extend(_wonums(value))
    return found


def _error_of(text):
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data.get("error") if isinstance(data, dict) and set(data) == {"error"} else None


def _count_summary(reply):
    """The item count of a read that sells nothing: the length of every list in the answer, or the error."""
    if reply.is_error or _error_of(reply.text) is not None:
        return {"error": _error_of(reply.text) or reply.text}
    data = json.loads(reply.text)
    lists = [v for v in data.values() if isinstance(v, list)] if isinstance(data, dict) else [data]
    return {"items": sum(len(v) for v in lists) if lists else 1}


def _validated(server, tool, arguments):
    """The arguments as FastMCP hands them to AssetOpsBench's function, a JSON string decoded and "3" read as 3, or
    None when FastMCP would refuse them before the function runs."""
    metadata = SERVERS[server].mcp._tool_manager.get_tool(tool).fn_metadata
    try:
        return metadata.arg_model.model_validate(metadata.pre_parse_json(arguments)).model_dump_one_level()
    except ValidationError:
        return None


def _order_problem(arguments):
    """What makes create_workorder answer a validation error or the work order database refuse the document, both
    without a write: the three required arguments, the priority, the work type, and aob_source's agent and trigger."""
    if not all(arguments.get(k) for k in ("description", "asset_num", "site_id")):
        return "required"
    if not 1 <= arguments["priority"] <= 5:
        return "priority"
    if arguments["work_type"] not in workorders.WORKTYPES:
        return "work_type"
    source = arguments.get("aob_source") or {}
    if source.get("agent") not in (None, "") and source["agent"] not in SOURCE_AGENTS:
        return "aob_source.agent"
    if source.get("trigger_type") not in (None, "") and source["trigger_type"] not in SOURCE_TRIGGERS:
        return "aob_source.trigger_type"
    return None


class StateFileSessions:
    """Episodes kept in the action server's state file: an exclusive lock on a file beside it, so the
    concurrent calls of Agent-As-Tool's three server processes run one at a time in the order the state file records,
    then replay, the call, and the save of the state and the released evidence before the lock is released. The replay
    and the released evidence are those of the AssetOpsBench rows unless a caller passes others."""

    def __init__(self, state_path, replay=rows.replay, write_released=rows.write_released, wall_clock=None, monotonic_clock=None):
        self.state_path = Path(state_path)
        self._replay = replay
        self._write_released = write_released
        # without clocks of its own a session reads the clocks of the action server's episode module when the call is served,
        # which a replay of a transcript sets to the recorded readings
        self._clocks = {"wall_clock": wall_clock, "monotonic_clock": monotonic_clock}

    def enter(self, server, tool):
        return episode.enter(self.state_path.parent, server, tool, **self._clocks)

    def leave(self, server, entered, outcome):
        episode.leave(self.state_path.parent, server, entered, outcome)

    @contextmanager
    def __call__(self, entered):
        with locked(self.state_path):
            state = episode.load_state(self.state_path)
            engine = self._replay(state)
            generated = episode.generated_total(engine.config, self.state_path.parent)
            session = _StateSession(state, engine, entered, self.state_path.parent, generated)
            yield session
            state["last_call_wall_time"] = entered["wall"]
            state["last_call_monotonic_time"] = entered["mono"]
            episode.save_state(self.state_path, state)
            self._write_released(self.state_path, session.engine)
            if session.answer is not None:  # still inside the lock, so the lines of the transcript follow the state file
                server, tool, arguments, reply = session.answer
                ended = ({**episode.end_fields(entered, **self._clocks), **episode.ledger_fields(session.state)} if session.stepped
                         else {"enter_n": entered["n"]})
                transcript.answered(self.state_path.parent, server, tool, arguments, reply.text, reply.is_error, stepped=session.stepped,
                                    wall=entered["wall"], released_sha256=transcript.file_sha256(episode.released_path(self.state_path)),
                                    **episode.generated_fields(session.generated), **ended)


class _StateSession:
    def __init__(self, state, engine, entered, folder, generated=None):
        self.state = state
        self.engine = engine
        self.entered = entered  # the entry of the call's service (episode.enter)
        self.folder = folder
        self.generated = generated  # the total of the ledger of generated tokens at the call, the generated_tokens mode alone
        self.stepped = False  # whether the call took a step of the engine
        self.answer = None  # (server, tool, arguments, reply) of the call once the meter has answered it

    def step(self, action):
        """The call's one action after the thinking time since the previous call, by the action server's own rule."""
        self.stepped = True
        return episode.step(self.state, self.engine, action, self.entered, self.folder, self.generated)

    def log_call(self, entry):
        self.state["tool_calls"].append(entry)

    def answered(self, server, tool, arguments, reply):
        self.answer = (server, tool, arguments, reply)
