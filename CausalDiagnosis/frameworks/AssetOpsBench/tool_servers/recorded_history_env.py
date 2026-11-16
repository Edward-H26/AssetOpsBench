"""The engine of the AssetOpsBench rows: HydraulicRigEnv with the
recorded history that AssetOpsBench's iot server reads.

AssetOpsBench reads telemetry that is already recorded, so this engine keeps a record of the cycles the rig has played
and serves reads from it rather than playing new cycles for each read. At reset it records the source's prehistory of the
start, the segment at the entry offset, at cycles -HISTORY_CYCLES to -1, so that an agent finds data before its first
call, and the live cycles play the other segment and then wrap among it, so that no live cycle replays a
recording the prehistory showed and every cycle is released once. On the repair rules and deterioration
rows a level can return, when an emergency repair expires or a drop takes a repaired component back, and a
combination the rig played before then continues its window where the rig left it, the repeats marked once it wraps, so
the prehistory never comes back on the timeline. Only a combination met for the first time opens a window, on its live
segment. Every cycle the clock plays afterwards, through the thinking time, a wait or a read of the advanced
real time clock, joins the record unless a repair stops the rig. Two actions join HydraulicRigEnv's:

- read_recorded serves AssetOpsBench's history or latest_reading over the record, after the thinking time. Where a read
  takes time, a bounded history window first plays the cycles it reaches beyond the clock at their declared duration,
  and elsewhere it holds the cycles recorded so far. A served cycle costs the price of the read's sensor
  group unless a group paid earlier for that cycle covers the fields asked for, so a re-read is free, and
  the read logs one observation per served cycle that shows fields not released before. The price is charged before any
  cycle plays, as read_sensors charges, so a read the budget refuses moves nothing.
- aob_read logs a free read of AssetOpsBench's registry, failure codes or work orders, so that every call counts toward
  the step cap and pays its thinking time. A work order read that shows past orders of the fix history logs them as
  the rig's get_fix_history does, so that the diagnostics server and the verifier see the history the agent read.

read_sensors is refused, since the cycles it plays would bypass the record. On a row without a clock no cycle runs, so no
new reading follows a repair there, whether its order closes inside its call or with its visit at the next call that
is not an order. The asset profile publishes the tool table
and the clock of the AssetOpsBench rows from their contract, CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_tool_contract.json, which also classes
the two reads. The rig's own rows keep HydraulicRigEnv.
"""
import bisect
import copy

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract

from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT

from CausalDiagnosis.datasets.dataset_loader.profile_validation import is_int

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

RECORDED_READS = ("history", "latest_reading")
HISTORY_CYCLES = SEGMENT  # the recordings of the start that the prehistory shows, the same on every row
UNRECORDED_REASON = "read_sensors plays cycles outside the record, and the AssetOpsBench rows read with read_recorded"


class RecordedRigEnv(HydraulicRigEnv):
    def __init__(self, cycles, blocks, config=None, **kwargs):
        super().__init__(cycles, blocks, config, **kwargs)
        self._handlers.update(read_recorded=self._read_recorded, aob_read=self._aob_read, read_sensors=self._read_unrecorded)

    def reset(self, seed):
        self._recorded = {}  # cycle -> (recording index, repeated)
        self._order = []  # the recorded cycles, oldest first, which grows at its end: the prehistory, then every played cycle
        self._paid = {}  # cycle -> the fields that a group charged for the cycle covers
        self._released = {}  # cycle -> the fields released in observations of the cycle
        envelope = super().reset(seed)
        self._record_prehistory()
        return envelope

    def _record_prehistory(self):
        """The prehistory and the live cycles of a start come from disjoint segments: every
        record of the source's prehistory of the start is recorded, oldest first, at the cycles before 0 that end at -1,
        -HISTORY_CYCLES to -1 for the recorded levels kind, and the clock plays the live segment and then wraps among it,
        marked repeated as every wrap is. A combination met later for the first time enters its own live segment, as the
        rig's clock does, and a return to one played before resumes its window."""
        prehistory = self.source.prehistory(self.initial)
        for t, index in zip(range(-len(prehistory), 0), prehistory):
            self._recorded[t] = (index, False)
            self._order.append(t)

    def public_profile(self):
        """The rig's profile with the tool table and the clock of the AssetOpsBench rows on the configuration's row, the
        price and declared duration of every tool under this configuration, since this engine serves
        AssetOpsBench's tools and the task tools rather than the rig's action server."""
        profile = super().public_profile()
        table = layer_contract.public_table(self.config)
        profile["tools"] = table
        profile["clock"].update(durations=layer_contract.clock_durations(table),
                                notifications=layer_contract.load_contract()["clock"]["notifications"])
        return profile

    # ----- the record, for the producer of the AssetOpsBench documents -----
    def recorded_cycles(self):
        """The cycles with a record, oldest first: the prehistory, then every cycle the rig played and was not stopped."""
        return list(self._order)

    def record_values(self, cycle):
        """The served values of the recording played at a recorded cycle, every channel and statistic."""
        index, _repeated = self._recorded[cycle]
        return self.source.served_values(index, tuple(self.profile.channels))

    def _advance(self, cycles):
        advanced = super()._advance(cycles)
        for t, played in advanced:
            if played is not None:
                self._recorded[t] = played
                self._order.append(t)  # the clock plays each cycle once and in order, so the record stays in cycle order
        return advanced

    # ----- the two reads of the AssetOpsBench rows -----
    def _read_recorded(self, action, think):
        tool, group, fields = action.get("tool"), action.get("sensor_group"), action.get("fields")
        if tool not in RECORDED_READS:
            return self._refuse(action, f"tool must be one of {list(RECORDED_READS)}", think)
        if not isinstance(group, str) or group not in self.config.sensor_groups:
            return self._refuse(action, "unknown sensor group", think)
        granted = self._group_fields(group)
        # a field that is no string, such as a list, is refused like a field of another group and raises nothing
        if not isinstance(fields, list) or not fields or any(not isinstance(f, str) or f not in granted for f in fields):
            return self._refuse(action, f"fields must be a non-empty list of the served fields of {group}", think)
        window = (None, None, 0, 1) if tool == "latest_reading" else self._window(action)
        if window is None:
            return self._refuse(action, "start and end must be integer cycles or null, offset at least 0 and limit at least 1", think)
        start, end, offset, limit = window
        per_cycle = self.config.durations["read_per_cycle"]
        timed = tool == "history" and per_cycle > 0 and end is not None
        moves = 0 if timed else self._duration(0, 0)  # the one cycle of every call in the instant mode, none otherwise
        # the cycles a read plays, a timed window or the cycle of its call, decide the page it serves and so its price, so it
        # plays them first on a copy of the engine, and the engine itself moves only once the budget has paid
        priced = self._dry_run() if timed or moves else self
        if timed:
            priced._play_window(start, end, offset, limit, per_cycle)
        else:
            priced._advance(moves)
        price = self.config.read_price(group, len(self._unpaid(priced._page(tool, start, end, offset, limit), fields)))
        if not self._charge(action, price, think):
            return None, "budget"
        t_start, stopped = self.time, 0
        if timed:
            stopped = self._play_window(start, end, offset, limit, per_cycle)
        else:
            self._advance(moves)
        page = self._page(tool, start, end, offset, limit)
        paid = self._unpaid(page, fields)
        for c in paid:
            self._paid.setdefault(c, set()).update(granted)
        served = []
        for c in page:
            new = [f for f in fields if f not in self._released.get(c, set())]
            if new:
                served.append(self._recorded_observation(c, new))
                self._released.setdefault(c, set()).update(new)
        self._entry("read", t_start, think, price, tool=tool, sensor_group=group, cycles=len(page), stopped=stopped, served=served)
        return {"tool": tool, "sensor_group": group, "cycles": page, "paid": paid, "price": price,
                "observations": [o["n"] for o in served]}, None

    def _aob_read(self, action, think):
        server, tool = action.get("server"), action.get("tool")
        if not isinstance(server, str) or not isinstance(tool, str):
            return self._refuse(action, "server and tool must name the AssetOpsBench call", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        shown = action.get("fix_history") or []
        history = [copy.deepcopy(entry) for entry in self.fix_history if entry["id"] in shown]
        if history:
            self._entry("fix_history", t_start, think, 0, history=history, server=server, tool=tool)
            return {"server": server, "tool": tool, "history": history}, None
        self._entry("aob_read", t_start, think, 0, server=server, tool=tool)
        return {"server": server, "tool": tool}, None

    def _read_unrecorded(self, action, think):
        return self._refuse(action, UNRECORDED_REASON, think)

    def _create_work_order(self, action, think):
        """HydraulicRigEnv's order with its price in the result, for the receipt of the metering layer: the prices alone,
        never the downtime of the hidden ledger."""
        before = self.prices_paid
        result, refusal = super()._create_work_order(action, think)
        if result is not None:
            result["price"] = self.prices_paid - before
        return result, refusal

    # ----- parts -----
    def _play_window(self, start, end, offset, limit, per_cycle):
        """Play the cycles a timed history window reaches beyond the clock, at their declared duration,
        until the window holds the page, the clock reaches the window's end or the horizon, or the episode is cut short.
        Returns the cycles the rig was stopped for a repair."""
        stopped = 0
        while (len(self._recorded_in(start, end)) < offset + limit and self.time < min(end, self.config.horizon)
               and not self._cut_short()):
            stopped += sum(1 for t, _played in self._advance(self._duration(1, per_cycle)) if t in self.clock.stopped_cycles)
        return stopped

    def _page(self, tool, start, end, offset, limit):
        return self._order[-1:] if tool == "latest_reading" else self._recorded_in(start, end)[offset:offset + limit]

    def _unpaid(self, page, fields):
        """The cycles of a page that no group paid earlier for that cycle covers with the fields asked for."""
        return [c for c in page if not set(fields) <= self._paid.get(c, set())]

    def _group_fields(self, group):
        return {f"{self.profile.field_name(s)}_{statistic}" for s in self.config.sensor_groups[group]
                for statistic in self.profile.statistics["served"]}

    @staticmethod
    def _window(action):
        """(start, end, offset, limit) of a history read in cycles, or None unless they are integers in range."""
        start, end, offset, limit = (action.get(k) for k in ("start", "end", "offset", "limit"))
        if all(v is None or is_int(v) for v in (start, end)) and is_int(offset) and offset >= 0 and is_int(limit) and limit >= 1:
            return start, end, offset, limit
        return None

    def _recorded_in(self, start, end):
        """The recorded cycles from start on and before end, oldest first, cut from the record by bisection."""
        low = 0 if start is None else bisect.bisect_left(self._order, start)
        high = len(self._order) if end is None else bisect.bisect_left(self._order, end)
        return self._order[low:high]

    def _recorded_observation(self, cycle, fields):
        index, repeated = self._recorded[cycle]
        values = self.source.served_values(index, tuple(self.profile.channels))
        self._observations += 1
        return {"n": self._observations, "t": cycle, "cycle": index, "combination": list(self.clock.combination_at(cycle)),
                "repeated": repeated, "values": {f: values[f] for f in fields}}

    # the rig's contract classes the rig's actions, and the contract of the AssetOpsBench rows the two reads it adds
    def _entry(self, event, t_start, think, price, **fields):
        if event != "aob_read":
            return super()._entry(event, t_start, think, price, **fields)
        entry = {"event": event, "step": self.steps, "t_start": t_start, "t_end": self.time, "class": layer_contract.action_class(event),
                 "cost": price, "think_cycles": think, **fields}
        self.log.append(entry)
        return entry

    def _log_refusal(self, action, reason, t_start, think):
        kind = action.get("type")
        # an action type that is no string goes to the rig's refusal, which logs it as an unknown action
        if not isinstance(kind, str) or kind not in layer_contract.load_contract()["actions"]:
            return super()._log_refusal(action, reason, t_start, think)
        self.log.append({"event": "refused", "step": self.steps, "t_start": t_start, "t_end": self.time, "class": layer_contract.action_class(kind),
                         "cost": 0, "think_cycles": think, "action": copy.deepcopy(action), "reason": reason})
