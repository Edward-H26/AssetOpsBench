"""The engine calls of the metering layer, as RecordedRigEnv adds them to HydraulicRigEnv, for the tests of the meter
and the producer. The stub keeps HydraulicRigEnv's step signature, envelope and public event shapes, and it is the
reference of these additions:

- a clock mode without a clock for the base and priced rows, with no thinking time and an order that closes in the step
  that creates it, so no new reading follows a repair there;
- the prehistory, the first recordings of the start window at cycles -W to -1, `recorded_cycles()`, the cycles that have a record,
  and `record_values(cycle)`, the served values of a recorded cycle, for the producer;
- `read_recorded`, the read of a window of recorded cycles. After the thinking time the engine takes the recorded cycles
  in [start, end), first plays the cycles not yet recorded that the page needs where reads have a declared duration,
  serves the page from offset to offset + limit (latest_reading the newest recorded cycle), charges every served
  cycle that no group paid earlier for that cycle covers at the group's price, so a re-read under a paid group is free,
  and logs a read event with one observation per served cycle that carries fields not released before, so the
  released evidence holds every value the agent has seen and nothing twice. The price comes from a copy that plays the
  window first, and the engine plays only once the budget has paid, so a read the budget refuses moves nothing;
- `aob_read`, a free action that logs the AssetOpsBench tool it stands for;
- the price of `read_recorded` and `create_work_order` in their result, for the receipt.

The stub plays one recorded cycle per cycle of read duration, stops the rig while an order is in repair, and knows no
schedule, hazard, visit or failure, which the real engine keeps.
"""
import copy
from dataclasses import dataclass, field

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import default_sensor_groups
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import EVIDENCE_IDS_REASON, unlisted_reason
from CausalDiagnosis.datasets.hyd_1.former_field_names import HYD1_FIELDS

NOTIFYING_EVENTS = ("work_order_executed", "work_order_closed")
ORDER_BEFORE_WRITE_REASON = "a diagnosis or a declaration must precede a work order"


@dataclass
class StubConfig:
    clock_mode: str = "none"  # "none" in the base and priced rows, "generation_time" in the two clock rows
    seconds_per_cycle: float = 60
    read_prices: dict = field(default_factory=lambda: {"tier1": 0, "tier2": 0, "tier3": 0})
    sensor_groups: dict = field(default_factory=default_sensor_groups)
    order_price: float = 0
    budget: float = None
    horizon: int = 60
    max_steps: int = 60
    prehistory: int = 10
    read_duration: int = 0  # cycles a played cycle takes in a read, nonzero in the advanced real time clock row
    repair_cycles: int = 0  # cycles from an order's creation to its close in the clock rows


class StubEngine:
    def __init__(self, config, values):
        self.config = config
        self._values = values  # cycle -> {engine field: value}, the recording the producer writes as well
        self.time = 0
        self.steps = 0
        self.done = False
        self.cost = 0
        self.diagnosed = False
        self.log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0}]
        self.fix_history = []  # the stub's episode has no fix history card
        self._recorded = list(range(-config.prehistory, 0))
        self._observations = 0
        self._paid = {}  # cycle -> the served fields of every group paid for that cycle
        self._released = {}  # cycle -> the fields released in observations
        self._orders = []
        self._notified = 1
        self._charged = 0

    # ----- public views -----
    def recorded_cycles(self):
        return list(self._recorded)

    def record_values(self, cycle):
        """The served values of the recording the rig played at a recorded cycle, every channel and statistic."""
        if cycle not in self._recorded:
            raise KeyError(f"cycle {cycle} has no record")
        return self._values(cycle)

    def public_events(self):
        return copy.deepcopy(self.log[1:])

    def public_work_orders(self):
        return [copy.deepcopy(o) for o in self._orders if o["status"] != "closed"]

    def budget_left(self):
        return self.config.budget - self.cost

    def envelope(self, result=None, refused=None, notifications=()):
        envelope = {} if refused is None else {"refused": refused}
        envelope["time"] = self.time
        if self.config.budget is not None:
            envelope["budget_left"] = self.budget_left()
        envelope.update({"steps_left": self.config.max_steps - self.steps, "open_work_orders": self.public_work_orders(),
                         "episode_over": self.done, "notifications": list(notifications)})
        if refused is None:
            envelope["result"] = copy.deepcopy(result)
        return envelope

    # ----- the step -----
    def step(self, action, think_cycles=0):
        if self.done:
            raise RuntimeError("episode is over; call reset")
        self.steps += 1
        self._charged = 0
        if self.config.clock_mode == "generation_time":
            self._advance(think_cycles)
        handler = {"read_recorded": self._read_recorded, "aob_read": self._aob_read, "get_failure_modes": self._get_failure_modes,
                   "submit_diagnosis": self._submit_diagnosis, "create_work_order": self._create_work_order}.get(action.get("type"))
        unknown = unlisted_reason(action) or "unknown action type"
        result, refused = handler(action) if handler else self._refuse(action, unknown)
        if not self.done and self.time >= self.config.horizon:
            self._end("horizon")
        elif not self.done and self.steps >= self.config.max_steps:
            self._end("max_steps")
        notifications = [{"event": e["event"], "id": e["id"], "t": e["t_end"]} for e in self.log[self._notified:] if e["event"] in NOTIFYING_EVENTS]
        self._notified = len(self.log)
        return self.envelope(result, refused, notifications), self._charged, self.done, ({"refused": refused} if refused else {})

    def _advance(self, cycles):
        for _ in range(min(cycles, self.config.horizon - self.time)):
            t = self.time
            stopped = False
            for order in self._orders:
                if order["status"] == "open" and t >= order["created_t"]:
                    order["status"] = "in_repair"
                    self._system("work_order_executed", t, order)
                if order["status"] == "in_repair" and t >= order["expected_close_t"]:
                    self._close(order, t)
                stopped = stopped or order["status"] == "in_repair"
            if not stopped:
                self._recorded.append(t)
            self.time += 1

    def _system(self, event, t, order):
        self.log.append({"event": event, "step": None, "t_start": t, "t_end": t, "id": order["id"], "component": order["component"]})

    def _close(self, order, t):
        order.update(status="closed", closed_t=t)
        self._system("work_order_closed", t, order)

    def _refuse(self, action, reason):
        self.log.append({"event": "refused", "step": self.steps, "t_start": self.time, "t_end": self.time, "action": copy.deepcopy(action), "reason": reason})
        return None, reason

    def _charge(self, action, price):
        if self.config.budget is not None and self.cost + price > self.config.budget:
            self._refuse(action, "budget")
            self._end("budget")
            return False
        self.cost += price
        self._charged += price
        return True

    def _end(self, reason):
        self.done = True
        self.log.append({"event": "end", "step": None, "t_start": self.time, "t_end": self.time, "reason": reason})

    def _entry(self, event, t_start, price, **fields):
        self.log.append({"event": event, "step": self.steps, "t_start": t_start, "t_end": self.time, "cost": price, **fields})

    # ----- actions -----
    def _read_recorded(self, action):
        group, fields = action.get("sensor_group"), action.get("fields")
        if group not in self.config.read_prices or not isinstance(fields, list):
            return self._refuse(action, "unknown sensor group")
        t_start = self.time
        price = len(self._unpaid(copy.deepcopy(self)._page(action), fields)) * self.config.read_prices[group]
        if not self._charge(action, price):
            return None, "budget"
        page = self._page(action)
        paid = self._unpaid(page, fields)
        granted = {f"{HYD1_FIELDS[s]}_{statistic}" for s in self.config.sensor_groups[group] for statistic in ("mean", "std")}
        served = []
        for cycle in page:
            if cycle in paid:
                self._paid.setdefault(cycle, set()).update(granted)
            new = [f for f in fields if f not in self._released.get(cycle, set())]
            if new:
                self._observations += 1
                served.append({"n": self._observations, "t": cycle, "values": {f: self._values(cycle)[f] for f in new}})
                self._released.setdefault(cycle, set()).update(new)
        self._entry("read", t_start, price, sensor_group=group, cycles=len(page), stopped=0, served=served)
        return {"cycles": page, "paid": paid, "price": price, "observations": [o["n"] for o in served]}, None

    def _page(self, action):
        """The cycles a read serves, after the cycles a timed window reaches have played."""
        if action.get("tool") == "latest_reading":
            return self._recorded[-1:]
        start, end = action.get("start"), action.get("end")
        offset, limit = action.get("offset", 0), action.get("limit", 1000)

        def window():
            return [c for c in self._recorded if (start is None or c >= start) and (end is None or c < end)]

        if self.config.read_duration > 0 and end is not None:
            while len(window()) < offset + limit and self.time < min(end, self.config.horizon):
                self._advance(self.config.read_duration)
        return window()[offset:offset + limit]

    def _unpaid(self, page, fields):
        return [c for c in page if not set(fields) <= self._paid.get(c, set())]

    def _aob_read(self, action):
        self._entry("aob_read", self.time, 0, server=action.get("server"), tool=action.get("tool"))
        return {}, None

    def _get_failure_modes(self, action):
        component = action.get("component")
        if not isinstance(component, str) or (component and component not in COMPONENTS):
            return self._refuse(action, "unknown component")
        self._entry("failure_modes", self.time, 0, component=component)
        return {"component": component or None, "failure_modes": []}, None

    def _submit_diagnosis(self, action):
        self.diagnosed = True
        self._entry("submit_diagnosis", self.time, 0, diagnosis=action.get("diagnosis"), evidence_ids=action.get("evidence_ids", []))
        return {"diagnosis": action.get("diagnosis")}, None

    def _create_work_order(self, action):
        if not self.diagnosed:
            return self._refuse(action, ORDER_BEFORE_WRITE_REASON)
        component, prediction, evidence, mode = (action.get(k) for k in ("component", "prediction_id", "evidence_ids", "mode"))
        if component not in COMPONENTS:
            return self._refuse(action, "unknown component")
        if not isinstance(prediction, str):
            return self._refuse(action, "prediction_id must be a string, empty when there is no prediction")
        if not (isinstance(evidence, list) and all(isinstance(e, str) for e in evidence)):
            return self._refuse(action, EVIDENCE_IDS_REASON)
        if mode not in ("standard", "emergency"):
            return self._refuse(action, "mode must be standard or emergency")
        if mode == "emergency":
            return self._refuse(action, "the emergency mode is off in this episode")
        if any(o["component"] == component and o["status"] != "closed" for o in self._orders):
            return self._refuse(action, f"a work order on the {component} is open")
        if not self._charge(action, self.config.order_price):
            return None, "budget"
        order = {"id": f"wo-{len(self._orders) + 1}", "component": component, "created_t": self.time,
                 "expected_close_t": self.time + self.config.repair_cycles, "status": "open", "source": "agent", "closed_t": None,
                 "mode": mode, "visit": None}
        self._orders.append(order)
        self._entry("work_order_created", self.time, self.config.order_price, id=order["id"], component=component, mode=mode,
                    prediction_id=prediction, evidence_ids=list(evidence), expected_close_t=order["expected_close_t"])
        if self.config.clock_mode == "none":  # without a clock the order is executed and closed in the step that creates it
            order["status"] = "in_repair"
            self._system("work_order_executed", self.time, order)
            self._close(order, self.time)
        return {"work_order": copy.deepcopy(order), "price": self.config.order_price}, None


def replay(state):
    """The stub after its stored actions, as episode.replay rebuilds HydraulicRigEnv, for StateFileSessions tests."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import hyd_1_sample_documents as fixture_documents
    engine = StubEngine(StubConfig(**state["stub_config"]), fixture_documents.engine_values)
    for action, think in zip(state["actions"], state["think_cycles"]):
        engine.step(action, think_cycles=think)
    return engine
