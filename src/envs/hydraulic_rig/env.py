"""The hydraulic rig environment: a hidden combination of component conditions, a clock with priced and timed tools,
work orders that the recordings of the repaired rig answer, and a log for the verifier (docs/hydraulic_rig/environment.md).

An episode starts at cycle 0 with a hidden combination h(t0) drawn from the seed. The agent reads sensors at a price and
a duration, advances time, summarizes the changes since a cycle, looks the asset and its failure modes up, submits one
diagnosis or one declaration of insufficient evidence, creates work orders that a technician executes after a lead time,
and finishes. Three hidden processes run on the clock (clock.py). Every change of the combination is answered with the
recording of the new combination, so nothing is simulated.

Every malformed or out-of-order action that is a dict is refused (logged, no charge, counts a step, the episode
continues) rather than raised, so a language model agent that emits a bad action loses a step, not the episode; an
action that is not a dict is a programming error and raises. An action the budget cannot pay ends the episode. The log
keeps deep copies of everything the agent hands in and the agent facing views are copies, so no in-process policy can
rewrite its own diagnosis. Hidden fields live only in `log`; `public_event` and `public_profile` are the agent facing
views, and the oracle of oracle.py reads the hidden state.
"""
import copy
import random

from servers import contract
from servers.hydraulic_rig.names import HYD1_FIELDS, public_result

from . import knowledge
from .clock import RigClock, draw_external_event, draw_schedule
from .config import EpisodeConfig
from .data import COMPONENTS, HEALTHY, SENSORS, group_features, percentile
from .oracle import oracle

HEALTHY_COMBO = tuple(HEALTHY[c] for c in COMPONENTS)
SERVED_STATISTICS = ("mean", "std")  # the summaries a reading returns per channel (design v2, section 5)
BAND_QUANTILES = (0.01, 0.99)  # the public healthy band of a channel mean over the stable cycles of the healthy combination
BLOCK_CAP = 10  # cycles in a served window, so that block length reveals nothing about the combination
HIDDEN_FIELDS = ("cycle", "combination", "combination_after", "unnecessary", "hidden", "schedule", "external_events", "final", "seed", "config", "played")
HIDDEN_EVENTS = ("scheduled_drop",)  # the schedule is hidden, so a timed episode leaks nothing that a static one does not
WRITE_LOCK_REASON = "a diagnosis or a declaration was already submitted"
ORDER_BEFORE_WRITE_REASON = "a diagnosis or a declaration must precede a work order"
END_REASONS = ("finish", "budget", "horizon", "max_steps", "failure")


def n_faulty(combination):
    return sum(1 for i, c in enumerate(COMPONENTS) if combination[i] != HEALTHY[c])


def served_values(cycle, sensors):
    """What a reading returns for one recorded cycle: the mean and std of every channel of the group, under the public
    field names and at six significant digits."""
    return public_result(group_features(cycle, sensors, tuple(f"_{s}" for s in SERVED_STATISTICS)))


def mean_field(sensor):
    return f"{HYD1_FIELDS[sensor]}_mean"


def healthy_bands(cycles_by_index, blocks):
    """{public field of a channel mean: {"low", "high"}}: the 1st and 99th percentiles over the stable cycles of the
    healthy combination, computed on the served values so that the band and the alarm check see the same numbers."""
    served = [served_values(cycles_by_index[i], SENSORS) for i in blocks[HEALTHY_COMBO]]
    bands = {}
    for s in SENSORS:
        values = [v[mean_field(s)] for v in served]
        bands[mean_field(s)] = {"low": percentile(values, BAND_QUANTILES[0]), "high": percentile(values, BAND_QUANTILES[1])}
    return bands


class HydraulicRigEnv:
    def __init__(self, cycles, blocks, config=None, block_cap=BLOCK_CAP):
        """cycles and blocks come from data.load_cycles and data.build_blocks; config is an EpisodeConfig (default when
        None). The start candidates are the combinations within the fault rule, restricted to config.allowed."""
        self.cycles = {c.index: c for c in cycles}
        self.blocks = blocks
        self.config = config if config is not None else EpisodeConfig()
        self.block_cap = block_cap
        allowed = None if self.config.allowed is None else {tuple(k) for k in self.config.allowed}
        self.candidates = sorted(k for k in blocks if self.config.within_fault_rule(k) and (allowed is None or k in allowed))
        if not self.candidates:
            raise ValueError("no start combination satisfies the fault rule within the allowed set")
        self.healthy_bands = healthy_bands(self.cycles, blocks)
        self._handlers = {
            "get_asset_profile": self._get_asset_profile, "read_sensors": self._read_sensors, "advance_time": self._advance_time,
            "summarize_changes_since": self._summarize_changes_since, "get_failure_modes": self._get_failure_modes,
            "submit_diagnosis": self._submit_diagnosis, "declare_insufficient_evidence": self._declare_insufficient_evidence,
            "create_work_order": self._create_work_order, "finish": self._finish,
        }
        self.done = True

    # ----- episode control -----
    def reset(self, seed):
        """Draw the episode from the seed, in this order: the hidden combination (unless config.start names it), the
        served window, the degradation schedule and the external event. Returns the first envelope, whose result is the
        asset profile."""
        self._rng = random.Random(seed)
        if self.config.start is not None:
            hidden, offset = tuple(self.config.start["combination"]), self.config.start["block_offset"]
        else:
            hidden, offset = self._rng.choice(self.candidates), None
        self.initial = hidden
        self.clock = RigClock(self.config, self.blocks, self.block_cap, self._rng, hidden, offset)
        self.clock.schedule = draw_schedule(self._rng, self.config, hidden)
        self.clock.external = draw_external_event(self._rng, self.config, hidden)
        self.cost = 0
        self.steps = 0
        self.diagnosed = False
        self.declared = False
        self.done = False
        self._observations = 0
        self._strict_failure = False
        self._step_charges = 0
        self.log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0, "class": "system", "cost": 0, "seed": seed,
                     "hidden": list(hidden), "schedule": copy.deepcopy(self.clock.schedule),
                     "external_events": copy.deepcopy(self.clock.external), "config": self.config.to_json()}]
        return self.envelope(self.public_profile())

    @property
    def time(self):
        return self.clock.time

    @property
    def hidden(self):
        """The current combination as a list (hidden from agents; the oracle and the tests read it)."""
        return list(self.clock.combination)

    @property
    def window_start(self):
        return self.clock.window_start

    @property
    def entry_offset(self):
        return self.clock.entry_offset

    def combination_at(self, t):
        return self.clock.combination_at(t)

    def cycle_at(self, t):
        """The recording index the rig played during cycle t, or None when it was stopped or has not run that cycle (hidden)."""
        return dict(self.clock.played).get(t)

    def oracle(self):
        return oracle(self)

    # ----- agent facing views -----
    def public_profile(self):
        """The answer of get_asset_profile: everything public about the asset and the episode's rules."""
        c = self.config
        servers = [contract.ACTION_SERVER] + ([contract.DIAGNOSTICS_SERVER] if c.diagnostics_server else [])
        return {
            "asset_id": c.asset_id,
            "asset": knowledge.asset_profile(c.asset_id),
            "components": list(c.components),
            "levels": {k: list(v) for k, v in c.levels.items()},
            "healthy": dict(c.healthy),
            "sensor_groups": {g: [HYD1_FIELDS[s] for s in sensors] for g, sensors in c.sensor_groups.items()},
            "statistics": list(SERVED_STATISTICS),
            "prices": copy.deepcopy(c.prices),
            "durations": copy.deepcopy(c.durations),
            "downtime_rate": copy.deepcopy(c.downtime_rate),
            "budget": c.budget,
            "horizon": c.horizon,
            "max_steps": c.max_steps,
            "fault_rule": dict(c.fault_rule),
            "clock_mode": c.clock_mode,
            "seconds_per_cycle": c.seconds_per_cycle,
            "tools": contract.public_table(servers),
            "healthy_bands": copy.deepcopy(self.healthy_bands),
        }

    @staticmethod
    def public_event(entry):
        """The agent facing view of a log entry: without the recording indices, the combinations, the unnecessary flag,
        the hidden start, the drawn processes, the final combination, the seed and the configuration. A scheduled drop
        has no public view and gives None; a failure keeps its time but not its component, which the agent has to find."""
        if entry["event"] in HIDDEN_EVENTS:
            return None
        public = {k: copy.deepcopy(v) for k, v in entry.items() if k not in HIDDEN_FIELDS}
        if entry["event"] == "failure":
            public.pop("component", None)
        if "served" in public:
            public["served"] = [{k: v for k, v in obs.items() if k not in HIDDEN_FIELDS} for obs in public["served"]]
        return public

    def public_events(self):
        """The public views of every log entry after reset, in order, without the entries the agent never sees."""
        return [p for p in (self.public_event(e) for e in self.log[1:]) if p is not None]

    def public_work_orders(self, closed_since=None):
        """The open work orders, or with closed_since the orders closed at or after that cycle."""
        if closed_since is None:
            return [o.public() for o in self.clock.open_orders()]
        return [o.public() for o in self.clock.orders if o.status == "closed" and o.closed_t >= closed_since]

    def envelope(self, result=None, refused=None):
        """The answer of a call: the clock, the counters, the open work orders and the result, or a refusal without result."""
        envelope = {} if refused is None else {"refused": refused}
        envelope["time"] = self.time
        if self.config.anchors is not None:
            envelope["day"] = self.config.date_of(self.time)
        envelope.update({"budget_left": self.config.budget - self.cost, "steps_left": self.config.max_steps - self.steps,
                         "open_work_orders": self.public_work_orders(), "episode_over": self.done})
        if refused is None:
            envelope["result"] = copy.deepcopy(result)
        return envelope

    # ----- the step -----
    def step(self, action, think_cycles=0):
        """Apply one action. think_cycles is the agent's thinking time in cycles, charged before the action in the
        generation_time clock mode and ignored otherwise. Returns (envelope, cost, done, info), where cost is everything
        the step charged (price, downtime and penalties) and info holds `refused` after a refusal and the full log with
        its hidden state on the last step, which no tool server may forward."""
        if self.done:
            raise RuntimeError("episode is over; call reset")
        if not isinstance(action, dict):
            raise TypeError("action must be a dict")
        if not _is_count(think_cycles, 0):
            raise TypeError("think_cycles must be a non-negative integer")
        self.steps += 1
        self._step_charges = 0
        think = think_cycles if self.config.clock_mode == "generation_time" else 0
        self._advance(think)
        if self._cut_short():
            # the thinking time alone ran into the horizon or a strict failure, so the action never runs
            self._log_refusal(action, "the episode ended before the action ran", self.time, think)
            refusal = "the episode ended before the action ran"
            result = None
        else:
            kind = action.get("type")
            handler = self._handlers.get(kind, self._unknown) if isinstance(kind, str) else self._unknown
            result, refusal = handler(action, think)
        if not self.done:
            if self._strict_failure:
                self._end("failure")
            elif self.time >= self.config.horizon:
                self._end("horizon")
            elif self.steps >= self.config.max_steps:
                self._end("max_steps")
        info = {"refused": refusal} if refusal else {}
        if self.done:
            info.update(log=self.log, reason=self.log[-1]["reason"])
        return self.envelope(result, refusal), self._step_charges, self.done, info

    def _cut_short(self):
        return self._strict_failure or self.time >= self.config.horizon

    def _advance(self, cycles):
        """Move the clock by `cycles`, stopping at the horizon or at a failure under the strict rule. Returns the list of
        (t, played) of the advanced cycles, where played is (recording index, repeated) or None for a stopped cycle."""
        advanced = []
        for _ in range(min(cycles, self.config.horizon - self.time)):
            if self._strict_failure:
                break
            t = self.time
            events, played, charge = self.clock.advance_cycle()
            self._step_charges += charge
            self.cost += charge
            for e in events:
                cost = 0
                if e["event"] == "failure":
                    cost = self.config.failure["penalty"]
                    self._step_charges += cost
                    self.cost += cost
                    self._strict_failure = self._strict_failure or self.config.failure["strict"]
                fields = {k: v for k, v in e.items() if k not in ("event", "t")}
                self.log.append({"event": e["event"], "step": None, "t_start": e["t"], "t_end": e["t"], "class": "system", "cost": cost, **fields})
            advanced.append((t, played))
        return advanced

    def _duration(self, cycles, per_cycle):
        """The clock advance of a timed action in the current mode."""
        if self.config.clock_mode == "instant":
            return 1
        return cycles * per_cycle

    def _entry(self, event, t_start, think, price, **fields):
        """Append the log entry of an accepted action; the class comes from the contract."""
        entry = {"event": event, "step": self.steps, "t_start": t_start, "t_end": self.time, "class": contract.tool_class(_TOOL_OF_EVENT[event]),
                 "cost": price, "think_cycles": think, **fields}
        self.log.append(entry)
        return entry

    def _log_refusal(self, action, reason, t_start, think):
        kind = action.get("type")
        klass = contract.tool_class(kind) if isinstance(kind, str) and kind in self._handlers else "read"
        self.log.append({"event": "refused", "step": self.steps, "t_start": t_start, "t_end": self.time, "class": klass, "cost": 0,
                         "think_cycles": think, "action": copy.deepcopy(action), "reason": reason})

    def _refuse(self, action, reason, think):
        t_start = self.time
        if self.config.clock_mode == "instant":
            self._advance(1)  # in instant mode every agent step takes a cycle, a wasted one included
        self._log_refusal(action, reason, t_start, think)
        return None, reason

    def _charge(self, action, price, think):
        """Charge a price, or refuse for lack of budget and end the episode. Returns whether the price was charged."""
        if self.cost + price > self.config.budget:
            self._log_refusal(action, "budget", self.time, think)
            self._end("budget")
            return False
        self.cost += price
        self._step_charges += price
        return True

    def _end(self, reason):
        self.done = True
        self.log.append({"event": "end", "step": None, "t_start": self.time, "t_end": self.time, "class": "system", "reason": reason,
                         "time": self.time, "final": self.hidden, "cost": self.cost, "downtime_total": self.clock.downtime_total,
                         "downtime_by_cycle": copy.deepcopy(self.clock.downtime),
                         "played": [{"t": t, "cycle": index, "combination": list(self.combination_at(t))} for t, index in self.clock.played]})

    # ----- the nine actions -----
    def _unknown(self, action, think):
        return self._refuse(action, "unknown action type", think)

    def _get_asset_profile(self, action, think):
        t_start = self.time
        self._advance(self._duration(0, 0))
        result = self.public_profile()
        self._entry("profile", t_start, think, 0)
        return result, None

    def _read_sensors(self, action, think):
        group, cycles = action.get("sensor_group"), action.get("cycles")
        if not isinstance(group, str) or group not in self.config.sensor_groups:
            return self._refuse(action, "unknown sensor group", think)
        if not _is_count(cycles, 1):
            return self._refuse(action, "cycles must be a positive integer", think)
        price = self.config.read_price(group, cycles)
        if not self._charge(action, price, think):
            return None, "budget"
        sensors = self.config.sensor_groups[group]
        t_start = self.time
        served, stopped, remaining = [], 0, cycles
        # served cycle i is the recording the rig played during that cycle, so a change of the combination mid-read shows
        for t, played in self._advance(self._duration(cycles, self.config.durations["read_per_cycle"])):
            if remaining == 0:
                continue
            remaining -= 1
            if played is None:
                stopped += 1
            else:
                served.append(self._observation(t, played, sensors))
        # cycles the clock did not cover (instant mode, zero durations) come from the window at the current time
        while remaining > 0 and not self._cut_short():
            remaining -= 1
            if self.clock.stopped_at(self.time):
                stopped += 1
            else:
                served.append(self._observation(self.time, self.clock.next_cycle(), sensors))
        self._entry("read", t_start, think, price, sensor_group=group, cycles=cycles, stopped=stopped, served=served)
        return {"sensor_group": group, "served": [{k: v for k, v in o.items() if k not in HIDDEN_FIELDS} for o in served], "stopped": stopped}, None

    def _observation(self, t, played, sensors):
        index, repeated = played
        self._observations += 1
        return {"n": self._observations, "t": t, "cycle": index, "combination": list(self.clock.combination_at(t)),
                "repeated": repeated, "values": served_values(self.cycles[index], sensors)}

    def _advance_time(self, action, think):
        cycles = action.get("cycles")
        if not _is_count(cycles, 1):
            return self._refuse(action, "cycles must be a positive integer", think)
        t_start = self.time
        self._advance(self._duration(cycles, self.config.durations["advance_per_cycle"]))
        self._entry("advance", t_start, think, 0, cycles=cycles)
        return {"time": self.time}, None

    def _summarize_changes_since(self, action, think):
        since = action.get("cycle")
        if not _is_count(since, 0) or since > self.time:
            return self._refuse(action, "cycle must be an integer between 0 and the current time", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        alarms = self._alarms(since)
        closed = self.public_work_orders(closed_since=since)
        self._entry("summary", t_start, think, 0, since=since, alarms=alarms, work_orders_closed=closed)
        return {"since": since, "time": self.time, "cycles_elapsed": self.time - since, "alarms": alarms, "work_orders_closed": closed}, None

    def _alarms(self, since):
        """The channel means outside their public healthy band in any cycle the rig ran since `since`, with the first
        such cycle; computed from the recording the rig played, never from a label."""
        first = {}
        for t, index in self.clock.played:
            if t < since or index is None:
                continue
            values = served_values(self.cycles[index], SENSORS)
            for field, band in self.healthy_bands.items():
                if field not in first and not band["low"] <= values[field] <= band["high"]:
                    first[field] = t
        return [{"channel": field, "first_t": t} for field, t in sorted(first.items(), key=lambda kv: (kv[1], kv[0]))]

    def _get_failure_modes(self, action, think):
        component = action.get("component")
        if not isinstance(component, str) or (component and component not in self.config.components):
            return self._refuse(action, "unknown component", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        wanted = [component] if component else list(self.config.components)
        modes = [{"component": e["component"], "level": e["level"], "code": e["code"], "severity": e["severity"], "action": e["action"],
                  "channels": [HYD1_FIELDS[s] for s in SENSORS if e["component"] in knowledge.INDICATES.get(s, ())],
                  "evidence_id": f"fmea/{knowledge.FMEA_SOURCE}/{e['code']}"}
                 for e in knowledge.FAULT_CODES if e["component"] in wanted]
        self._entry("failure_modes", t_start, think, 0, component=component)
        return {"component": component or None, "association_source": knowledge.ASSOCIATION_SOURCE, "failure_modes": modes}, None

    def _submit_diagnosis(self, action, think):
        if self.diagnosed or self.declared:
            return self._refuse(action, WRITE_LOCK_REASON, think)
        diagnosis, evidence = action.get("diagnosis"), action.get("evidence_ids")
        if not isinstance(diagnosis, dict) or set(diagnosis) != set(self.config.components):
            return self._refuse(action, "diagnosis must give one level per component", think)
        # levels are ints; True, 1.0 or "100" are refused like a non-integer cycle count
        if any(not _is_count(diagnosis[c], None) or diagnosis[c] not in self.config.levels[c] for c in self.config.components):
            return self._refuse(action, "diagnosis level not in the component's levels", think)
        if not _is_evidence(evidence):
            return self._refuse(action, "evidence_ids must be a list of strings", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        self.diagnosed = True
        stored = {c: diagnosis[c] for c in self.config.components}
        self._entry("submit_diagnosis", t_start, think, 0, diagnosis=stored, evidence_ids=list(evidence))
        return {"diagnosis": stored, "evidence_ids": list(evidence)}, None

    def _declare_insufficient_evidence(self, action, think):
        if self.diagnosed or self.declared:
            return self._refuse(action, WRITE_LOCK_REASON, think)
        test, evidence = action.get("separating_test"), action.get("evidence_ids")
        if (not isinstance(test, dict) or test.get("sensor_group") not in self.config.sensor_groups
                or not _is_count(test.get("cycles"), 1)):
            return self._refuse(action, "separating_test must name a sensor group and a positive number of cycles", think)
        if not _is_evidence(evidence):
            return self._refuse(action, "evidence_ids must be a list of strings", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        self.declared = True
        stored = {"sensor_group": test["sensor_group"], "cycles": test["cycles"]}
        self._entry("declare_insufficient_evidence", t_start, think, 0, separating_test=stored, evidence_ids=list(evidence))
        return {"separating_test": stored, "evidence_ids": list(evidence)}, None

    def _create_work_order(self, action, think):
        if not (self.diagnosed or self.declared):
            return self._refuse(action, ORDER_BEFORE_WRITE_REASON, think)
        component, prediction, evidence = action.get("component"), action.get("prediction_id"), action.get("evidence_ids")
        if component not in self.config.components:
            return self._refuse(action, "unknown component", think)
        if not isinstance(prediction, str):
            return self._refuse(action, "prediction_id must be a string, empty when there is no prediction", think)
        if not _is_evidence(evidence):
            return self._refuse(action, "evidence_ids must be a list of strings", think)
        if self.clock.order_open_on(component):
            return self._refuse(action, f"a work order on the {component} is open", think)
        if not self._charge(action, self.config.order_price(component), think):
            return None, "budget"
        t_start = self.time
        self._advance(self._duration(0, 0))
        order = self.clock.create_order(component, prediction, evidence)
        self._entry("work_order_created", t_start, think, self.config.order_price(component), id=order.id, component=component,
                    prediction_id=prediction, evidence_ids=list(evidence), expected_close_t=order.close_t, unnecessary=order.unnecessary)
        return {"work_order": order.public()}, None

    def _finish(self, action, think):
        summary = action.get("summary")
        if not isinstance(summary, str):
            return self._refuse(action, "summary must be a string", think)
        t_start = self.time
        self._advance(self._duration(0, 0))
        self._entry("finish", t_start, think, 0, summary=summary)
        self._end("finish")
        return {"summary": summary}, None


_TOOL_OF_EVENT = {"profile": "get_asset_profile", "read": "read_sensors", "advance": "advance_time", "summary": "summarize_changes_since",
                  "failure_modes": "get_failure_modes", "submit_diagnosis": "submit_diagnosis",
                  "declare_insufficient_evidence": "declare_insufficient_evidence", "work_order_created": "create_work_order", "finish": "finish"}


def _is_count(value, minimum):
    """An int (not a bool) at or above the minimum; minimum None checks the type only."""
    return isinstance(value, int) and not isinstance(value, bool) and (minimum is None or value >= minimum)


def _is_evidence(value):
    return isinstance(value, list) and all(isinstance(v, str) for v in value)
