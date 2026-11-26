"""How the CausalDiagnosis agent plays the servers of AssetOpsBench: the iot, fmsr and wo servers of AssetOpsBench and
the task server of the metering layer (tool_servers/), the surface every causal run of the rows plays. The core class
Surface and the bindings of the diagnostics server are the agent's own (agents/Causal_Diagnosis_Agent/tool_surfaces.py); this
module binds the agent's action kinds to these servers and reads their answers, receipt line included, into the one
envelope the controller consumes. The surface object of the framework (tool_servers/tool_surface.py) names AobSurface as the
class of the CausalDiagnosis agent on this surface.
"""
from datetime import datetime, timedelta

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import (DIAGNOSTICS_KINDS, HYDRAULIC_RIG, SUMMARY_TOOL, TIME_TOOL,
                                                                 WAIT_TOOL, RigSurface, Surface, _write, bind, not_json)
from CausalDiagnosis.agents.utils.tool_answers import json_object
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts import ended, split_receipt

IOT_SERVER, FMSR_SERVER, WO_SERVER, TASK_SERVER = "iot", "fmsr", "wo", "task"  # the servers of the AssetOpsBench rows, as the drivers register them
AOB_SERVERS = (IOT_SERVER, FMSR_SERVER, WO_SERVER, TASK_SERVER)
ORDER_PRIORITY, ORDER_WORK_TYPE = 2, "CM"  # a corrective order of normal priority, as the briefing of the rows asks for
RECORD_FIELDS = ("asset_id", "timestamp")  # the fields of a telemetry row that carry no value
# the clock of a row is learned from its answers, so the tool text states the row with a clock and the row without one from the first
# request on, and the status of the board says which holds: the system prompt is then the same in every request of an episode
WITHOUT_WAIT = (" Where a clock runs, time passes with your thinking, since this row serves no wait: a history window past the present plays "
                "the cycles it reaches where reads take time, and a work order closes after its lead time and downtime; where none runs, a "
                "work order closes as it is created.")
CLOCK_STATED = " The status of the board says whether a clock runs."


class AobSurface(Surface):
    """AssetOpsBench's iot, fmsr and wo servers and the task server of the metering layer. The surface learns
    the site, the asset, its child assets, the sensor groups, the seconds per cycle and whether the clock runs from the
    asset profile, anchors its cycle arithmetic on get_current_time where the row lists it and on the newest recorded row
    elsewhere (no cycle runs there, so that row is the cycle before the start), and bounds the record with stream_extent,
    a free registry read, so that no charged read is spent on the anchor."""

    action_servers = AOB_SERVERS
    evidence_form = "obs/<timestamp>/<field>"
    reads_take_time = False  # a history read serves cycles already recorded, so it takes no time

    @property
    def summary_is_call(self):
        """Whether the change summary is summarize_changes_since of the task server, a call the step cap counts, where the row serves
        it; elsewhere the surface answers it from the notifications the receipts carried, without a call (local)."""
        return self.serves(SUMMARY_TOOL)

    @property
    def prompt_changes(self):
        return RigSurface.prompt_changes if self.summary_is_call else ""

    def __init__(self, noun="asset", example_tag=None, statistics=("value",)):
        super().__init__(noun, example_tag, statistics)
        self.site = self.asset_id = None
        self.components, self.children, self.sensor_groups = (), {}, {}
        self.seconds_per_cycle = None
        self.read_per_cycle = 0  # the declared clock of one read cycle, from the profile
        self.max_steps = None
        self.clock = False  # a receipt carries the cycle, so the clock moves between calls and the record grows
        self.anchor = None  # (cycle, naive datetime) of one moment, from which every timestamp maps to a cycle
        self.oldest = self.newest = None  # the record the agent knows of, in cycles
        self.calls = 0  # calls on the action servers, which the step cap counts, for the base row that appends no receipt
        self.windows = {}  # sensor group -> (first, last) of the window its latest read asked for, last None for an open window
        self.unrecorded = {}  # sensor group -> the cycles a read of it asked for that came back without a row, which hold no record
        self.events = []  # the notifications the receipts carried, which answer the change summary

    @property
    def prompt_tools(self):
        clock = (" Where a clock runs, a wait calls wait_for_notification" + (" and the time is read with get_current_time" if self.serves(TIME_TOOL) else "")
                 + " of the task server; where none runs, no wait is possible and a work order closes as it is created."
                 if self.serves(WAIT_TOOL) else WITHOUT_WAIT) + CLOCK_STATED
        return ("\nOn this row the controller calls AssetOpsBench's servers for the kinds: read calls history of the iot server over the "
                f"newest recorded cycles the group has not paid for, with the group's sensors{self.summary_fields()}; order calls "
                "generate_work_order of the wo server on the child asset of the component, with work_type CM and the prediction and the "
                "evidence in aob_source; " + ("diagnose and finish" if "declare" in self.unserved_kinds() else "diagnose, declare and finish")
                + " call the task server." + clock
                + " The failure modes come from get_failure_modes of the fmsr server and the fix history from list_workorders of the wo server."
                + (" One more kind of the opening exists on this row, in a plan after the profile is read, and a reading needs it first: extent, no args, the "
                   "extent of the record by stream_extent of the iot server." if "time" in self.unserved_kinds() else
                   " Two more kinds of the opening exist on this row, in a plan after the profile is read, and a reading needs them first: time, no args, the clock by "
                   "get_current_time of the task server where the profile lists it, and extent, no args, the extent of the record by stream_extent "
                   "of the iot server."))

    # ----- the kinds -----
    def bind(self, kind, board, **fields):
        if kind in DIAGNOSTICS_KINDS:
            return bind(HYDRAULIC_RIG, kind, **fields)
        if kind == "changes":
            return (TASK_SERVER, SUMMARY_TOOL, {"cycle": fields["since"]}) if self.summary_is_call else None  # else from the notifications
        return getattr(self, f"_bind_{kind}")(board, **fields)

    def _bind_profile(self, board):
        # the level raw, the whole profile that _read_profile reads field by field
        return TASK_SERVER, "get_asset_profile", {"response_format": "raw"}

    def _bind_extent(self, board):
        return IOT_SERVER, "stream_extent", {"site_name": self.site, "asset_id": self.asset_id}

    def _bind_time(self, board):
        return TASK_SERVER, "get_current_time", {}

    def _bind_failure_modes(self, board, component):
        return FMSR_SERVER, "get_failure_modes", {"asset_class": component}

    def _bind_fix_history(self, board):
        return WO_SERVER, "list_workorders", {"site_id": self.site}

    def _bind_read(self, board, sensor_group, cycles):
        first, last = self.read_window(board, sensor_group, cycles)
        self.windows[sensor_group] = (first, last)
        fields = [f for name in self.sensor_groups.get(sensor_group, ()) for f in (name, *(f"{name}_{s}" for s in self.statistics[1:]))]
        window = ({"start": self.timestamp(first), "limit": cycles} if last is None  # an open window plays no cycle and serves what is recorded
                  else {"start": self.timestamp(first), "end": self.timestamp(last + 1), "limit": last - first + 1})
        return IOT_SERVER, "history", {"site_name": self.site, "asset_id": self.asset_id, "sensors": fields, **window}

    def _bind_diagnose(self, board, levels, evidence):
        return TASK_SERVER, "submit_diagnosis", _write(levels, evidence)

    def _bind_declare(self, board, sensor_group, cycles, evidence):
        return TASK_SERVER, "declare_insufficient_evidence", {"sensor_group": sensor_group, "cycles": cycles, "evidence_ids": ",".join(evidence)}

    def _bind_order(self, board, component, prediction_id, evidence):
        return WO_SERVER, "generate_work_order", {
            "description": f"restore the {component} of {self.asset_id} to its healthy level",
            "asset_num": self.children.get(component, f"{self.asset_id}_{component}"), "site_id": self.site,
            "priority": ORDER_PRIORITY, "work_type": ORDER_WORK_TYPE,
            "aob_source": {"prediction_id": prediction_id, "evidence_ids": list(evidence)}}

    def _bind_wait(self, board, timeout):
        return TASK_SERVER, "wait_for_notification", {"timeout": timeout}

    def _bind_finish(self, board, summary):
        return TASK_SERVER, "finish", {"summary": summary}

    def failure_mode_requests(self, board):
        return [{"component": c} for c in self.components]  # fmsr lists one component class per call

    def opening_kinds(self, board):
        return (["time"] if self.serves(TIME_TOOL) else []) + ["extent"]

    def can_observe_after(self, board, cycle):
        return self.clock  # on a row without a clock no cycle runs after a close

    @property
    def forward_reads(self):
        """Whether a read past the present plays its cycles, the way to a close on a clocked row that serves no wait."""
        return self.clock and not self.serves(WAIT_TOOL) and self.read_per_cycle > 0

    def local(self, kind, board, since):
        """The change summary of the rig's action server, answered from the notifications the receipts carried: the alarms
        and the closed orders since the cycle."""
        if kind != "changes":
            raise KeyError(f"{kind} has no local answer")
        alarms = [{"channel": n["channel"], "first_t": n.get("first_t", n.get("t"))} for n in self.events
                  if n.get("event") == "alarm" and n.get("channel")]
        closed = []
        for n in self.events:
            if n.get("event") == "work_order_closed":
                closed.append({"id": n.get("id"), "component": n.get("component"), "source": n.get("source", "agent"), "closed_t": n.get("t")})
        return {"result": {"since": since, "time": board.time,
                           "alarms": [a for a in alarms if a["first_t"] is None or a["first_t"] >= since],
                           "work_orders_closed": [c for c in closed if c["closed_t"] is None or c["closed_t"] >= since]}}

    # ----- cycles and timestamps -----
    def timestamp(self, cycle):
        """The timestamp of a cycle's row, in the naive ISO form the rows carry."""
        cycle0, moment0 = self.anchor
        return (moment0 + timedelta(seconds=(cycle - cycle0) * self.seconds_per_cycle)).isoformat()

    def cycle_of(self, stamp):
        cycle0, moment0 = self.anchor
        return round(cycle0 + (datetime.fromisoformat(stamp) - moment0).total_seconds() / self.seconds_per_cycle)

    def read_window(self, board, group, cycles):
        """The newest run of the group's unpaid recorded cycles (Board.unpaid_cycles), at most `cycles` of them, as (first,
        last). A window never spans a cycle the working memory holds or a cycle without a record of the group. When none is
        unread, the open window after the newest known cycle, (newest + 1, None), which the controller asks for only where the
        record may have grown since the latest answer, since where no clock runs or the clock runs by declaration no cycle can
        have been recorded since and the read stands (plan.standing)."""
        unread = board.unpaid_cycles(group, cycles)
        if self.forward_reads and (not unread or board.reads_forward(group)):
            return self.newest + 1, self.newest + cycles  # a bounded window plays its cycles to the close where no wait is served
        if not unread:
            return self.newest + 1, None
        run = [unread[0]]
        for cycle in unread[1:]:
            if len(run) == cycles or cycle != run[-1] - 1:
                break
            run.append(cycle)
        return run[-1], run[0]

    def _grow(self, cycle):
        if cycle is not None and (self.newest is None or cycle > self.newest):
            self.newest = cycle

    # ----- the answers -----
    def parse(self, kind, tool, text, board, **fields):
        """The envelope of a call from AssetOpsBench's text: the receipt line gives the budget left, the steps left and, on
        the clock rows, the cycle and the notifications; without a receipt the surface's own call count gives the steps
        left; the body becomes the result of the kind, or the refusal when it is an error object."""
        body, receipt = split_receipt(text)
        reply = json_object(body)
        # the meter answers every call that reached the episode with a JSON object, its refusals included, so an answer that is
        # none never reached the episode and is no step of the count
        if reply is not None:
            self.calls += 1
        data = {}
        if receipt is not None:
            data.update(budget_left=receipt.get("budget_left"), steps_left=receipt.get("steps_left"))
            if "cycle" in receipt:  # the meter writes the cycle on every call of a clocked row, whatever the row serves
                self.clock = True
                data["time"] = receipt["cycle"]
                self._grow(receipt["cycle"] - 1)
            notifications = [n for n in receipt.get("notifications") or [] if isinstance(n, dict)]
            data["notifications"] = notifications
            self.events += notifications
            progress = receipt.get("work_order_progress")
            if isinstance(progress, dict):
                data["open_work_orders"] = progress["open_work_orders"]
        if reply is None:
            data["refused"] = not_json(tool, body)
        elif set(reply) == {"error"}:
            data["refused"] = str(reply["error"])
        else:
            reader = getattr(self, f"_read_{kind}", None)
            data["result"] = reader(reply, board, data, **fields) if reader else reply
        if receipt is None and self.max_steps is not None:  # after the reply is read, so the profile's own call counts too
            data["steps_left"] = max(0, self.max_steps - self.calls)
        if ended(tool, body, receipt, False):  # an answer that reached parse is an answer, whatever its server marked
            data["episode_over"] = True
        if self.oldest is not None and self.newest is not None:
            data["recorded_from"], data["recorded_until"] = self.oldest, self.newest
        if self.unrecorded:
            data["unrecorded"] = {group: sorted(cycles) for group, cycles in self.unrecorded.items()}
        if data.get("steps_left") == 0:  # the surface's own count of the steps on the base row, which appends no receipt
            data["episode_over"] = True
        return data

    def _read_profile(self, reply, board, data):
        asset = reply.get("asset") or {}
        self.site, self.asset_id = asset.get("site"), reply.get("asset_id")
        self.components = tuple(reply.get("components") or ())
        self.children = dict(asset.get("children") or {})
        self.sensor_groups = {g: list(f) for g, f in (reply.get("sensor_groups") or {}).items()}
        self.seconds_per_cycle = (reply.get("clock") or {}).get("seconds_per_cycle")
        self.read_per_cycle = (reply.get("durations") or {}).get("read_per_cycle", 0) or 0
        self.max_steps = reply.get("max_steps")
        profiled = {t.get("name") for t in reply.get("tools") or []}
        if self.served is not None and profiled != self.served:
            # the profile's table and the opening listing are the one served list of the level; two lists are a fault of this project's code, never of the model
            raise ValueError(f"the profile and the opening listing differ in {sorted(profiled ^ self.served)}")
        return reply

    def _read_time(self, reply, board, data):
        self.anchor = (reply["time"], datetime.fromisoformat(reply["current_datetime"]))
        data["time"] = reply["time"]
        self._grow(reply["time"] - 1)
        return reply

    def _read_extent(self, reply, board, data):
        if self.anchor is None:  # no time tool: the newest recorded row is the cycle before the receipt's, the start without a clock
            self.anchor = (data.get("time", 0) - 1, datetime.fromisoformat(reply["end_time"]))
        self.oldest = self.cycle_of(reply["start_time"])
        self._grow(self.cycle_of(reply["end_time"]))
        return reply

    def _read_failure_modes(self, reply, board, data, component):
        source = reply.get("source")
        return {"failure_modes": [{"component": component, "level": None, "code": code, "severity": "", "action": "", "channels": [],
                                   "evidence_id": f"fmea/{source}/{code}"} for code in reply.get("failure_modes") or []]}

    def _read_fix_history(self, reply, board, data):
        """The work orders closed before the episode, which are every closed order at the opening."""
        children = {asset: component for component, asset in self.children.items()}
        return {"history": [{"id": o.get("wonum"), "component": children.get(o.get("assetnum"), o.get("assetnum")), "closed": o.get("actfinish"),
                             "description": o.get("description")} for o in reply.get("work_orders") or [] if o.get("status") == "COMP"]}

    def _read_read(self, reply, board, data, sensor_group, cycles):
        """The rows of a history as the served observations of the rig, one per row that shows a value the board has not
        seen, with the row's timestamp as the label of its evidence identifiers and its cycle from the anchor."""
        known = board.known_ids()
        served, rows = [], set()
        for row in reply.get("observations") or []:
            stamp = row.get("timestamp")
            cycle = self.cycle_of(stamp)
            rows.add(cycle)
            self._grow(cycle)
            values = {f: v for f, v in row.items() if f not in RECORD_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool)
                      and f"obs/{stamp}/{f}" not in known}
            if values:
                served.append({"n": board.observations + len(served) + 1, "t": cycle, "label": stamp, "values": values})
        first, last = self.windows.get(sensor_group, (None, None))
        if first is not None:  # the cycles the window asked for, up to the newest row an open window returned, that came back without a row
            end = last if last is not None else max(rows, default=first - 1)
            missing = {c for c in range(first, end + 1) if c not in rows}
            if missing:
                self.unrecorded.setdefault(sensor_group, set()).update(missing)
        return {"sensor_group": sensor_group, "served": served, "stopped": 0}

    def _read_order(self, reply, board, data, component, prediction_id, evidence):
        """The order as the rig's envelope describes one: created now, open until its notification where a clock runs, and closed
        inside its call where none runs. Shared maintenance feedback carries the public schedule;
        the older response shape still falls back to the board's estimate."""
        public = (reply.get("maintenance") or {}).get("work_order")
        if public is not None:
            return {"work_order": {**public, "source": "agent"}}
        created = data.get("time", board.time)
        return {"work_order": {"id": reply.get("wonum"), "component": component, "created_t": created, "expected_close_t": None if self.clock else created,
                               "status": "open" if self.clock else "closed", "source": "agent", "closed_t": None if self.clock else created}}

    def _read_wait(self, reply, board, data, timeout):
        if "time" in reply:
            data["time"] = reply["time"]
        return reply
