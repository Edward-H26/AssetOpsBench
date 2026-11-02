"""The clock of a hydraulic rig episode and the hidden processes that run on it (docs/hydraulic_rig/environment.md, section Time).

Time counts rig cycles from t0 = 0 and advances only inside an action. Each advanced cycle runs, in this order, the
scheduled drops, the external replacement, the work order lifecycle, the recording of the cycle, the downtime accrual and
the failure check; an order whose repair downtime ends with the cycle closes at its end. Every change of the hidden
combination switches the served recording to the stable block of the new combination through a seeded window, as the
environment has done since v1, so that nothing is simulated: the rig was recorded under all 144 combinations.

Three clock modes decide how far an action moves the clock. `declared` charges the declared duration of the action, so
that an episode is a function of its seed, its configuration and its actions. `generation_time` adds the measured
thinking time of the agent, in cycles, before every action. `instant` charges one cycle per agent step whatever the tool.
"""
from dataclasses import dataclass, field

from .config import STOPPED

EXTERNAL_ORDER_ID = "wo-ext-1"


def degradation_order(config, component):
    """The levels of a component from healthy to worst, which the schedule walks down. Ordering by distance from the
    healthy level gives cooler 100, 20, 3; valve 100, 90, 80, 73; pump 0, 1, 2; accumulator 130, 115, 100, 90."""
    healthy = config.healthy[component]
    return sorted(config.levels[component], key=lambda level: abs(level - healthy))


def worst_level(config, component):
    return degradation_order(config, component)[-1]


def draw_schedule(rng, config, hidden):
    """The scheduled drops of the episode: [{"t", "component", "to_level"}] in time order, empty in mode none.

    The components are drawn among those healthy at t0 first, so that a scheduled episode changes, and among the faulty
    ones only when more components are asked for than are healthy. Each drawn component gets a first drop cycle uniform
    in first_drop and then a drop every interval cycles until its worst level."""
    rule = config.schedule
    if rule["mode"] == "none" or rule["components"] == 0:
        return []
    levels = dict(zip(config.components, hidden))
    healthy = [c for c in config.components if levels[c] == config.healthy[c]]
    faulty = [c for c in config.components if levels[c] != config.healthy[c]]
    chosen = rng.sample(healthy, min(rule["components"], len(healthy)))
    chosen += rng.sample(faulty, rule["components"] - len(chosen))
    drops = []
    for c in chosen:
        first = rng.randint(*rule["first_drop"])
        order = degradation_order(config, c)
        below = order[order.index(levels[c]) + 1:]
        drops += [{"t": first + i * rule["interval"], "component": c, "to_level": level} for i, level in enumerate(below)]
    return sorted(drops, key=lambda d: (d["t"], config.components.index(d["component"])))


def draw_external_event(rng, config, hidden):
    """The external replacement of the episode, {"t", "component", "work_order_id"} or None: with the configured
    probability a technician replaces one component that is faulty at t0 at a cycle uniform in the window."""
    rule = config.external_events
    if rule["mode"] == "none":
        return None
    faulty = [c for c, level in zip(config.components, hidden) if level != config.healthy[c]]
    if rng.random() >= rule["probability"] or not faulty:
        return None
    return {"t": rng.randint(*rule["window"]), "component": rng.choice(faulty), "work_order_id": EXTERNAL_ORDER_ID}


@dataclass
class WorkOrder:
    id: str
    component: str
    created_t: int
    execute_t: int  # the technician arrives when the lead time has passed
    close_t: int  # the repair downtime has passed and the component is healthy
    source: str  # agent or external
    prediction_id: str = ""
    evidence_ids: list = field(default_factory=list)
    unnecessary: bool = False  # the component was healthy when the order was created (hidden)
    status: str = "open"  # open, in_repair, closed
    closed_t: int = None

    def public(self):
        """The work order as the agent sees it: never the unnecessary flag."""
        return {"id": self.id, "component": self.component, "created_t": self.created_t, "expected_close_t": self.close_t,
                "status": self.status, "source": self.source, "closed_t": self.closed_t}


class RigClock:
    """The hidden rig on the clock: the current levels, the served recording window, the work orders, the played cycles,
    the accrued downtime and the failure streaks. The environment drives it one cycle at a time."""

    def __init__(self, config, blocks, block_cap, rng, hidden, block_offset=None):
        self.config = config
        self.blocks = blocks
        self.block_cap = block_cap
        self.rng = rng
        self.levels = dict(zip(config.components, hidden))
        self.time = 0
        self.history = [(0, tuple(hidden))]  # (cycle, combination) at every change
        self.schedule = []
        self.external = None
        self.orders = []
        self.played = []  # (cycle, recording index or None when the rig was stopped), one per advanced cycle
        self.downtime = []  # {"t", "charge"} for every cycle with a positive charge
        self.downtime_total = 0
        self.streak = {c: 0 for c in config.components}  # consecutive cycles at the worst level without an open order
        self.failed = set()
        self._enter_window(0, block_offset)

    # ----- the served recording -----
    @property
    def combination(self):
        return tuple(self.levels[c] for c in self.config.components)

    @property
    def window_start(self):
        return self._window["start"]

    @property
    def entry_offset(self):
        return self._window["offset"]

    def _enter_window(self, t, block_offset=None):
        """Serve the stable block of the current combination through a window of block_cap cycles entered at a seeded
        offset, or at the given block offset for an anchored start, so that block length reveals nothing."""
        block = self.blocks[self.combination]
        cap = min(self.block_cap, len(block))
        if block_offset is None:
            start = self.rng.randrange(len(block) - cap + 1)
            offset = self.rng.randrange(cap)
        else:
            start = min(block_offset, len(block) - cap)
            offset = min(block_offset - start, cap - 1)
        self._window = {"t": t, "cycles": block[start:start + cap], "start": start, "offset": offset, "played": 0}

    def next_cycle(self):
        """Play the next recorded cycle of the window: (recording index, repeated), where repeated says that the window
        has wrapped and this cycle was played before in the current window."""
        w = self._window
        played = w["played"]
        w["played"] += 1
        return w["cycles"][(w["offset"] + played) % len(w["cycles"])], played >= len(w["cycles"])

    def combination_at(self, t):
        combination = self.history[0][1]
        for since, combo in self.history:
            if since <= t:
                combination = combo
        return combination

    def _set_level(self, component, level, t):
        if self.levels[component] == level:
            return
        self.levels[component] = level
        self.history.append((t, self.combination))
        self._enter_window(t)

    # ----- work orders -----
    def open_orders(self):
        return [o for o in self.orders if o.status != "closed"]

    def order_open_on(self, component):
        return any(o.component == component for o in self.open_orders())

    def stopped_at(self, t):
        """Whether the rig is stopped for a repair during cycle t."""
        return any(o.status != "closed" and o.execute_t <= t < o.close_t for o in self.orders)

    def create_order(self, component, prediction_id, evidence_ids):
        number = sum(1 for o in self.orders if o.source == "agent") + 1
        lead = self.config.durations["lead_time"][component]
        downtime = self.config.durations["repair_downtime"][component]
        order = WorkOrder(id=f"wo-{number}", component=component, created_t=self.time, execute_t=self.time + lead,
                          close_t=self.time + lead + downtime, source="agent", prediction_id=prediction_id,
                          evidence_ids=list(evidence_ids), unnecessary=self.levels[component] == self.config.healthy[component])
        self.orders.append(order)
        return order

    def _close(self, order, t):
        order.status = "closed"
        order.closed_t = t
        self._set_level(order.component, self.config.healthy[order.component], t)
        return {"event": "work_order_closed", "t": t, "id": order.id, "component": order.component, "source": order.source,
                "combination_after": list(self.combination)}

    # ----- one cycle of the clock -----
    def advance_cycle(self):
        """Run the processes of cycle `time` and move the clock to `time + 1`. Returns (events, played) where events are
        the system events of the cycle, each with its cycle `t`, and played is (recording index, repeated) or None when
        the rig was stopped for a repair during the cycle."""
        t = self.time
        events = []
        for drop in self.schedule:
            if drop["t"] == t:
                events.append({"event": "scheduled_drop", "t": t, "component": drop["component"],
                               "from_level": self.levels[drop["component"]], "to_level": drop["to_level"]})
                self._set_level(drop["component"], drop["to_level"], t)
        if self.external is not None and self.external["t"] == t:
            c = self.external["component"]
            events.append({"event": "external_replacement", "t": t, "component": c, "work_order_id": self.external["work_order_id"]})
            self._set_level(c, self.config.healthy[c], t)
            order = WorkOrder(id=self.external["work_order_id"], component=c, created_t=t, execute_t=t, close_t=t, source="external")
            self.orders.append(order)
            events.append(self._close(order, t))
        stopped = False
        for order in self.open_orders():
            if t >= order.close_t:
                events.append(self._close(order, t))  # a repair without downtime closes as the technician arrives
            elif t >= order.execute_t:
                if order.status == "open":
                    order.status = "in_repair"
                    events.append({"event": "work_order_executed", "t": t, "id": order.id})
                stopped = True
        played = None
        if not stopped:
            played = self.next_cycle()
        self.played.append((t, None if played is None else played[0]))
        charge = self._accrue(t, stopped)
        events += self._failure_check(t)
        self.time = t + 1
        for order in self.open_orders():
            if order.status == "in_repair" and self.time >= order.close_t:
                events.append(self._close(order, self.time))
        return events, played, charge

    def _accrue(self, t, stopped):
        """Downtime of the cycle: the stopped rate while a repair stops the rig, else the rate of every degraded component."""
        if stopped:
            charge = self.config.downtime_rate[STOPPED]
        else:
            charge = sum(self.config.level_rate(c, level) for c, level in self.levels.items())
        if charge:
            self.downtime.append({"t": t, "charge": charge})
            self.downtime_total += charge
        return charge

    def _failure_check(self, t):
        """A component at its worst level for `patience` consecutive cycles without an open work order fails once."""
        events = []
        for c in self.config.components:
            if self.levels[c] == worst_level(self.config, c) and not self.order_open_on(c):
                self.streak[c] += 1
            else:
                self.streak[c] = 0
            if self.streak[c] >= self.config.failure["patience"] and c not in self.failed:
                self.failed.add(c)
                events.append({"event": "failure", "t": t, "component": c})
        return events
