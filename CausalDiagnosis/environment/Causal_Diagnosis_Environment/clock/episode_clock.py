"""The clock of a hydraulic rig episode and the hidden processes that run on it (CausalDiagnosis/docs/environment.md, section Time).

Time counts rig cycles from t0 = 0 and advances only inside an action. Each advanced cycle runs, in this order, the
expiry of the emergency repairs, the scheduled drops or the drops of the stochastic hazard, the work order lifecycle, the
recording of the cycle, the downtime accrual and the failure check; an order whose repair downtime ends with the cycle
closes at its end. The standard orders created one after another with no other call between them form one visit, whose technician
repairs them back to back in priority order, and a repair takes only when the components it depends on are healthy at its close. A declared rule gives the levels that a repair
or a drop leaves, and every change of the hidden combination switches the served recording to the live segment of the stable
block of the new combination, so that no reading is simulated: the rig was recorded under all 144 combinations.

The rule of every process lives in the big layer that owns it (CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers): the expiry, the
visits and the close of an order in the repair rules, the drops and the failure check in deterioration and failure, the
downtime in the advanced real-time clock. RigClock keeps the state of the rig and the order of one cycle.

Four clock modes decide how far an action moves the clock. `declared` charges the declared duration of the action, so
that an episode is a function of its seed, its configuration and its actions. `generation_time` adds the thinking time of
the agent before every action, in the cycles that the servers charge from the seconds between calls, added up from call to
call with the rest carried (wall_clock.py). `generated_tokens` charges in the same way the tokens that the agent's model
generated between two calls, at seconds_per_token (token_clock.py). `instant` charges one cycle per agent step whatever
the tool.
"""
from dataclasses import dataclass, field

from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_3_advanced_real_time as advanced_real_time, layer_5_deterioration_and_failure as deterioration, layer_4_repair_rules as repair_rules
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import schedule_orders

# the processes of one cycle in the order RigClock.advance_cycle runs them, which the tool contract declares to the agent
CYCLE_ORDER = ("the emergency holds that end expire", "the scheduled drops", "the drops of the stochastic hazard",
               "the work orders close or start their repair, in priority order",
               "the rig plays one recorded cycle unless a repair stops it", "the downtime accrues", "the failure check",
               "the clock moves to the next cycle", "the repairs that end with the cycle close")


@dataclass
class WorkOrder:
    id: str
    component: str
    created_t: int
    execute_t: int  # the technician of the order's visit starts this repair
    close_t: int  # the repair downtime has passed
    source: str  # agent
    prediction_id: str = ""
    evidence_ids: list = field(default_factory=list)
    unnecessary: bool = False  # the component was healthy when the order was created (hidden)
    mode: str = "standard"  # standard or emergency
    visit: int = None  # the visit of the order
    status: str = "open"  # open, in_repair, closed
    closed_t: int = None

    def public(self):
        """The work order as the agent sees it: never the unnecessary flag."""
        return {"id": self.id, "component": self.component, "created_t": self.created_t, "expected_close_t": self.close_t,
                "status": self.status, "source": self.source, "closed_t": self.closed_t, "mode": self.mode, "visit": self.visit}


class RigClock:
    """The hidden rig on the clock: the current levels, the served recording window, the work orders, the emergency
    repairs that still hold, the played cycles, the accrued downtime and the failure streaks. The environment drives it one
    cycle at a time."""

    def __init__(self, config, source, rng, hidden, entry_offset=None):
        self.config = config
        self.source = source  # the evidence of the asset, an CausalDiagnosis.datasets.dataset_loader.evidence_source source
        self.rng = rng
        self.levels = dict(zip(config.components, hidden))
        self.time = 0
        self.history = [(0, tuple(hidden))]  # (cycle, combination) at every change
        self.schedule = []
        self.orders = []
        self.temporary = {}  # component: {"level", "expires_t", "id"} while an emergency repair holds
        self.hazard = None  # the uniforms of draw_hazard, set at reset
        self.hazard_drops = []  # {"t", "component", "to_level", "cause"} for every drop the hazard or a collateral damage made
        self.played = []  # (cycle, recording index or None when the rig was stopped), one per advanced cycle
        self.downtime = []  # {"t", "charge"} for every cycle with a positive charge
        self.downtime_total = 0
        self.streak = {c: 0 for c in config.components}  # consecutive cycles at the worst level, open orders or not
        self.failed = set()
        self.stopped_cycles = set()  # the cycles in which a repair stopped the asset
        # the records of a kind that replays no repair show no asset after a change, so none is played after the first close
        self.replays_repairs = kinds.kind_of(get_profile(config.asset_id, engine=False).evidence_kind).replays_repairs
        self.source.open(self.combination, 0, rng, entry_offset)

    # ----- the served recording -----
    @property
    def combination(self):
        return tuple(self.levels[c] for c in self.config.components)

    @property
    def _window(self):
        """The window the source serves, which the engine of the AssetOpsBench rows splits at reset."""
        return self.source.window

    @property
    def window_start(self):
        return self._window["start"]

    @property
    def entry_offset(self):
        return self._window["offset"]

    def next_cycle(self):
        """Play the next recorded cycle of the source: (recording index, repeated), or None for a cycle without a recording.
        For a kind that replays no repair that is every cycle from the close of the first work order of the episode on,
        whatever the order did, so that a missing reading never tells whether a repair took."""
        if not self.replays_repairs and any(o.status == "closed" for o in self.orders):
            return None
        return self.source.play()

    def combination_at(self, t):
        combination = self.history[0][1]
        for since, combo in self.history:
            if since <= t:
                combination = combo
        return combination

    def set_level(self, component, level, t):
        """Set a component's level at cycle t, a change of the combination opening the served window of the new one."""
        if self.levels[component] == level:
            return
        self.levels[component] = level
        self.history.append((t, self.combination))
        if self.replays_repairs:
            self.source.open(self.combination, t, self.rng)

    # ----- work orders -----
    def open_orders(self):
        return [o for o in self.orders if o.status != "closed"]

    def order_open_on(self, component):
        return any(o.component == component for o in self.open_orders())

    def stopped_at(self, t):
        """Whether the rig is stopped for a repair during cycle t."""
        return any(o.status != "closed" and o.execute_t <= t < o.close_t for o in self.orders)

    def create_order(self, component, prediction_id, evidence_ids, mode="standard", visit=None):
        """An order of the agent in the visit given, the visit of the order before it in a run of standard orders
        (servers/public_rules/visits.py), or in a visit of its own. The repairs of the visit that have begun keep their cycles and
        the others are scheduled again after them."""
        number = sum(1 for o in self.orders if o.source == "agent") + 1
        if visit is None:
            visit = 1 + max((o.visit for o in self.orders if o.visit is not None), default=0)
        # a component an emergency repair holds at its healthy level still needs its standard repair
        healthy = self.levels[component] == self.config.healthy[component] and component not in self.temporary
        order = WorkOrder(id=f"wo-{number}", component=component, created_t=self.time, execute_t=self.time, close_t=self.time,
                          source="agent", prediction_id=prediction_id, evidence_ids=list(evidence_ids), unnecessary=healthy,
                          mode=mode, visit=visit)
        self.orders.append(order)
        begun = [o for o in self.orders if o.visit == visit and o.status != "open"]
        waiting = [o for o in self.orders if o.visit == visit and o.status == "open"]
        after = max((o.close_t for o in begun), default=None)
        for i, execute_t, close_t in schedule_orders(self.config, [(o.component, o.mode, o.created_t) for o in waiting], after):
            waiting[i].execute_t, waiting[i].close_t = execute_t, close_t
        return order

    def _by_priority(self, orders):
        """Orders in priority order, ties in creation order, so that a dependency that closes in the same cycle closes first."""
        return sorted(orders, key=lambda o: self.config.priority[o.component])

    def mark_closed(self, order, t):
        """Mark an order closed at cycle t, its public event with the combination after the close."""
        order.status = "closed"
        order.closed_t = t
        return {"event": "work_order_closed", "t": t, "id": order.id, "component": order.component, "source": order.source,
                "combination_after": list(self.combination)}

    def settle_due_orders(self):
        """Close at the current cycle, without running it, every open order that is due by now, in priority order as the
        order step of a cycle closes them. On a row without a clock nothing runs the cycle in which an
        order of zero lead time and zero repair downtime closes, so the environment settles it: inside its own call while
        the repair card is off, and with the rest of its visit at the next call that is not an order while it is on. A finish
        on a row with a clock settles the same way once the clock has run to the last close."""
        events = []
        for order in self._by_priority(self.open_orders()):
            if order.close_t <= self.time:
                events += repair_rules.close(self, order, self.time)
        return events

    # ----- one cycle of the clock -----
    def advance_cycle(self):
        """Run the processes of cycle `time` and move the clock to `time + 1`. Returns (events, played, charge) where events
        are the system events of the cycle, each with its cycle `t`, played is (recording index, repeated) or None when
        the rig was stopped for a repair during the cycle, and charge is the downtime the cycle accrues."""
        t = self.time
        events = repair_rules.expire_holds(self, t)
        events += deterioration.drops(self, t)
        stopped = False
        for order in self._by_priority(self.open_orders()):
            if t >= order.close_t:
                events += repair_rules.close(self, order, t)  # a repair without downtime closes as the technician reaches it
            elif t >= order.execute_t:
                if order.status == "open":
                    order.status = "in_repair"
                    events.append({"event": "work_order_executed", "t": t, "id": order.id})
                stopped = True
        played = None
        if stopped:
            self.stopped_cycles.add(t)
        else:
            played = self.next_cycle()
        self.played.append((t, None if played is None else played[0]))
        charge = advanced_real_time.accrue(self, t, stopped)
        events += deterioration.failure_check(self, t)
        self.time = t + 1
        for order in self._by_priority(self.open_orders()):
            if order.status == "in_repair" and self.time >= order.close_t:
                events += repair_rules.close(self, order, self.time)
        return events, played, charge

