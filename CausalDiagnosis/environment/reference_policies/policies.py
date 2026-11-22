"""Reference policies over the v2 action set (CausalDiagnosis/docs/environment.md, section Baseline policies). The greedy
policy lives in greedy_policy.py with the per-tier classifiers.

A policy has start(first_envelope), called with the envelope that reset returns (its result is the asset profile), and
act(envelope), called with the answer to its previous action; it sees nothing else, which is exactly what a language
model agent gets through the tools. A harness that skips start may hand the reset envelope to act, which starts the
policy itself. OraclePolicy is the exception: it reads the hidden state through the environment.
"""
import itertools
import math
import random

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import degradation_order, hazard_rate, worst_level
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import schedule_visit


def profile_of(obs):
    """The asset profile when the envelope carries it, as the envelope that reset returns does, else None."""
    result = obs.get("result")
    return result if isinstance(result, dict) and "horizon" in result and "prices" in result else None


def _started(policy, obs):
    """Start the policy from the envelope when a harness hands it the reset envelope without calling start."""
    if getattr(policy, "started", False):
        return
    if profile_of(obs) is None:
        raise ValueError("the policy needs the envelope that reset returns, whose result is the asset profile, before any other envelope")
    policy.start(obs)


class OraclePolicy:
    """Reads the hidden state (the start combination, the current levels, the schedule and the table of the stochastic
    hazard), which no agent observes. It submits the current combination on every start, which the verifier
    scores at the cycle of the write, since a correct diagnosis with a recovered rig is a success whatever the label says,
    orders every faulty component once it will not drop further and its repair will take, in priority order
    so that one visit repairs a dependency before the component that needs it, waits until the orders close and finishes. Each
    repair has to close by the window end of its component, where the failure card would fail it (oracle.py), since an open
    order does not avert the failure: a component that waits for a later drop of its own is ordered from the latest cycle
    whose order, after the lead time and the repair downtimes of its visit, still closes by that end, unless the close could
    then come before the drop, and a cycle waited for the hazard never moves a close of the visit past its window end. When
    that schedule, played to the end on a copy of the environment, would still end with a failure, it plays the plan of
    order cycles of the faulty components that passes with no failure, with the earliest last cycle, the timing check met
    where it can be and the lowest cost (_plan). It waits for a drop only when the drop comes before the cycle at which it
    could finish without waiting, since
    a later drop never happens once the episode has ended, and a drop whose repair could not close before the
    horizon is not waited for either: the oracle set of such an episode never holds that component (oracle.py). Under the
    stochastic hazard it foresees the first drop of every healthy component and treats it as a scheduled one, so a repair
    that a dependency's drop would make ineffective waits for that drop, it places a visit a cycle later whenever the table
    would drop one of its repaired components again before the last of its orders closes, the orders after the drops it waits
    for included, and it repairs every degraded component at once and again after every drop. It is the reference for the
    cost of an episode (cost_oracle) and the best of the five reference policies on every ladder row and under both
    deterioration modes; only the tests, the drivers and the gym wrappers construct it, never an agent harness."""

    def __init__(self, env):
        self.env = env
        self.diagnosed = False
        self.planned, self.plan, self.ordered = False, None, set()

    def start(self, obs):
        self.diagnosed = False
        self.planned, self.plan, self.ordered = False, None, set()

    def act(self, obs):
        env, config, clock = self.env, self.env.config, self.env.clock
        if not self.diagnosed:
            self.diagnosed = True
            return {"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, env.hidden)), "evidence_ids": []}
        if not self.planned:
            self.planned = True
            self.plan = self._plan()
        planned = self._planned_act() if self.plan is not None else None
        if planned is not None:
            return planned
        t = env.time
        drops = self._foreseen_drops(t)
        # a drop at or after the cycle at which the oracle could finish never happens, since the episode ends first, so it
        # waits only for the drops before that cycle
        finish = self._finish_without_waiting(t, drops) if drops else config.horizon
        awaited = [d for d in drops if d["t"] < finish and self._repair_fits(d["t"] + 1, d["component"])]
        # a component whose order after its last drop would close past its window end waits for that drop no longer
        due, wakes = self._deadlines(t, awaited, drops)
        awaited = [d for d in awaited if d["component"] not in due]
        # an order of this cycle may have closed inside its own call on a row without a clock
        created = {o.component for o in clock.orders if o.source == "agent" and o.created_t == t}
        visit = self._visit(t, awaited, drops)
        if not created and self._undone(t, visit, awaited, drops):
            return {"type": "wait_for_notification", "timeout": 1}
        for c in visit:
            if not clock.order_open_on(c) and c not in created:
                return {"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}
        waits = [o.close_t for o in clock.open_orders()] + [d["t"] + 1 for d in awaited] + wakes
        if waits:
            # a notification may end the wait before the timeout, and the next act waits again from there
            return {"type": "wait_for_notification", "timeout": max(1, min(waits) - t)}
        return {"type": "finish", "summary": "Diagnosis submitted; every faulty component was repaired through a work order."}

    def _planned_act(self):
        """The action of the plan at the current cycle: its orders of this cycle in priority order, then a wait for the next
        cycle of the plan, or None once every cycle of the plan has passed, after which the one-visit schedule goes on."""
        config, clock, t = self.env.config, self.env.clock, self.env.time
        if t > max(self.plan.values()):
            return None
        for c in sorted(self.plan, key=lambda c: config.priority[c]):
            if self.plan[c] == t and c not in self.ordered and not clock.order_open_on(c):
                self.ordered.add(c)
                return {"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}
        later = [w for w in self.plan.values() if w > t]
        return {"type": "wait_for_notification", "timeout": max(1, min(later) - t)} if later else None

    def _simulate(self, plan):
        """(passes with no failure, (timing check missed, cost)) of the rest of the episode played on a copy of the environment,
        whose clock, schedule and table of uniforms equal those of the episode, with the given plan or, for None, the
        one-visit schedule alone."""
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

        twin = self.env._dry_run()
        policy = OraclePolicy(twin)
        policy.diagnosed, policy.planned, policy.plan = True, True, plan
        while not twin.done:
            twin.step(policy.act(None))
        verdict = verify(twin.log, released_from_env(twin), twin.oracle(), cost_oracle=None)
        passes = verdict["pass"] and not any(e["event"] == "failure" for e in twin.log)
        return passes, (not verdict["checks"]["timing"]["ok"], verdict["cost"])

    def _plan(self):
        """None while the one-visit schedule, played to the end on a copy of the environment, passes with no failure. Else the
        plan {component: order cycle} of the faulty components that does, the orders of one cycle created one after another
        so that they form one visit, one that meets the timing check before one that misses it and then the cheapest:
        each component is ordered from now up to the latest cycle whose order alone still closes by its window end, and one
        without a deadline up to the latest such cycle of the others; the plans with the earliest last cycle come first. None
        when no plan passes, and the one-visit schedule stays."""
        config, clock, t = self.env.config, self.env.clock, self.env.time
        if not config.failure_on() or self._simulate(None)[0]:
            return None
        faulty = [c for c in config.components if clock.levels[c] != config.healthy[c] and not clock.order_open_on(c)]
        drops = self._foreseen_drops(t)
        ends = {c: self._window_end(t, c, drops) for c in faulty}
        latest = {c: ends[c] - self._close_offset(c, [c]) for c in faulty if ends[c] < config.horizon}
        if not latest:
            return None
        reach = max(latest.values())
        bound = {c: max(t, latest.get(c, reach)) for c in faulty}
        for last in range(t, max(bound.values()) + 1):
            best = None
            for cycles in itertools.product(*[range(t, min(bound[c], last) + 1) for c in faulty]):
                if max(cycles) != last:
                    continue
                plan = dict(zip(faulty, cycles))
                passes, cost = self._simulate(plan)
                if passes and (best is None or cost < best[0]):
                    best = (cost, plan)
            if best is not None:
                return best[1]
        return None

    def _window_end(self, t, component, drops):
        """The cycle at which the failure card would fail the component if no repair of it closed before, the window end of
        oracle.py read forward from cycle t: patience cycles after the component reached its worst level, the cycles it has
        spent there already counted, or after the foreseen drop that takes it there, and the horizon when it never fails
        within the episode or has failed already."""
        config, clock = self.env.config, self.env.clock
        if not config.failure_on() or component in clock.failed:
            return config.horizon
        patience, worst = config.failure["patience"], worst_level(config, component)
        if clock.levels[component] == worst:
            return min(t + patience - clock.streak[component] - 1, config.horizon)
        # a foreseen drop of the hazard is the first drop of a healthy component, to the level below the healthy one
        below = degradation_order(config, component)[1]
        reach = [d["t"] for d in drops if d["component"] == component and d.get("to_level", below) == worst]
        return min(min(reach) + patience - 1, config.horizon) if reach else config.horizon

    def _close_offset(self, component, members):
        """The cycles from the creation of a standard order on the component to its close, in a visit of the given members."""
        members = list(dict.fromkeys([*members, component]))
        plan = schedule_visit(self.env.config, 0, [(m, "standard") for m in members])
        return next(close for i, _start, close in plan if members[i] == component)

    def _deadlines(self, t, awaited, drops):
        """(due, wakes): a faulty component without an open order that waits for a later drop of its own, while its order after
        the last of those drops would close after its window end, is due from the latest cycle at which an order on it still
        closes by that end in a visit of every faulty component without an open order, the most that can join it, and wakes
        holds the cycles at which such components fall due. A component whose repair, ordered from that cycle, could close
        before its last drop cannot be repaired by one lasting order before its window end, so it keeps waiting for the drop."""
        config, clock = self.env.config, self.env.clock
        waiting = {d["component"] for d in awaited}
        free = [c for c in config.components if clock.levels[c] != config.healthy[c] and not clock.order_open_on(c)]
        due, wakes = set(), []
        for c in free:
            end = self._window_end(t, c, drops)
            if c not in waiting or end >= config.horizon:
                continue
            last = max(d["t"] for d in awaited if d["component"] == c)
            offset = self._close_offset(c, free)
            earliest_close = end - offset + self._close_offset(c, [c])
            # a repair with a downtime closes at the end of the cycle before its close, after the drops of that cycle
            lasting = earliest_close - (1 if config.durations["repair_downtime"][c] else 0) >= last
            if last + 1 + offset <= end or not lasting:
                continue
            if t >= end - offset:
                due.add(c)
            else:
                wakes.append(end - offset)
        return due, wakes

    def _foreseen_drops(self, t):
        """The drops from cycle t on that the oracle foresees, {"t", "component"}: the schedule and, under the stochastic
        hazard, the first drop of every component healthy now, which the table of uniforms drawn at reset fixes, since a
        healthy component keeps its level until that drop whatever the oracle does, as it orders degraded components only."""
        config, clock = self.env.config, self.env.clock
        drops = [d for d in clock.schedule if d["t"] >= t]
        if config.hazard["mode"] == "stochastic":
            for i, c in enumerate(config.components):
                if clock.levels[c] != config.healthy[c]:
                    continue
                rate = hazard_rate(config, c, config.healthy[c])
                first = next((u for u in range(t, config.horizon) if clock.hazard["drop"][u][i] < rate), None)
                if first is not None:
                    drops.append({"t": first, "component": c})
        return drops

    def _finish_without_waiting(self, t, drops):
        """The cycle at which the oracle could finish if it waited for no drop: when the orders open now and the visit of
        cycle t have closed, or the horizon when a faulty component cannot be repaired now, since a drop has to come first."""
        config, clock = self.env.config, self.env.clock
        visit = self._visit(t, [], drops)
        if any(clock.levels[c] != config.healthy[c] and c not in visit and not clock.order_open_on(c) for c in config.components):
            return config.horizon
        closes = [o.close_t for o in clock.open_orders()]
        if visit:
            closes += [close for _i, _start, close in schedule_visit(config, t, [(c, "standard") for c in visit])]
        return max(closes, default=t)

    def _visit(self, t, awaited, drops):
        """The components of the visit of cycle t, smaller priority number first: the orders already created at t and every
        faulty component without an open order that will not drop further (one order after the last drop keeps the order
        set at one entry per component), less every component whose repair would not
        take. The check repeats until every component left takes, since dropping a dependency can block the component after
        it."""
        config, clock = self.env.config, self.env.clock
        faulty = [c for c in config.components if clock.levels[c] != config.healthy[c]]
        visit = [o.component for o in clock.orders if o.source == "agent" and o.created_t == t]
        visit += [c for c in faulty if not clock.order_open_on(c) and not any(d["component"] == c for d in awaited)]
        while True:
            closes = {visit[i]: close for i, _start, close in schedule_visit(config, t, [(c, "standard") for c in visit])} if visit else {}
            blocked = [c for c in visit if not clock.order_open_on(c) and not self._takes(c, t, closes, drops)]
            if not blocked:
                return sorted(visit, key=lambda c: config.priority[c])
            visit = [c for c in visit if c not in blocked]

    def _takes(self, component, t, closes, drops):
        """Whether an order on the component in the visit of cycle t would restore it: every component it depends on is in
        the visit, which repairs it first, or is healthy at the close through an earlier order, and no
        foreseen drop of a dependency falls between now and the close."""
        config, clock = self.env.config, self.env.clock
        close_t = closes[component]
        # the last cycle whose drops come before the close: a repair without downtime closes after the
        # processes of its own cycle, a repair with downtime at the end of the cycle before
        last = close_t if config.durations["repair_downtime"][component] == 0 else close_t - 1
        for d in config.depends_on(component):
            if any(drop["component"] == d and t <= drop["t"] <= last for drop in drops):
                return False
            if d in closes or clock.levels[d] == config.healthy[d]:
                continue
            earlier = [o for o in clock.open_orders() if o.component == d and o.close_t <= close_t]
            if not earlier:
                return False
        return True

    def _undone(self, t, visit, awaited, drops):
        """Whether the stochastic hazard would drop a component of the visit of cycle t again after its repair closes and
        before the last order closes, of the orders open now, of the visit and of the repairs placed right after the awaited
        drops, read from the table of uniforms at its healthy level, while waiting a cycle is safe: every repair of the visit
        would still close by the window end of its component, where the failure card would fail it, and before the horizon."""
        config, clock = self.env.config, self.env.clock
        if config.hazard["mode"] != "stochastic" or not visit:
            return False
        plan = schedule_visit(config, t, [(c, "standard") for c in visit])
        durations = config.durations
        after_drops = [d["t"] + 1 + durations["lead_time"][d["component"]] + durations["repair_downtime"][d["component"]] for d in awaited]
        last = max([o.close_t for o in clock.open_orders()] + [close for _i, _start, close in plan] + after_drops)
        undone = False
        for i, _start, close in plan:
            c, k = visit[i], config.components.index(visit[i])
            # a repair with downtime closes at the end of the cycle before close, one without it after the hazard of its cycle
            first = close if config.durations["repair_downtime"][c] else close + 1
            rate = hazard_rate(config, c, config.healthy[c])
            undone = undone or any(clock.hazard["drop"][u][k] < rate for u in range(first, min(last, config.horizon)))
        # a cycle waited moves every close of the visit one cycle later, and a close after the window end averts no failure
        ends = {c: self._window_end(t, c, drops) for c in visit}
        late = any(close > ends[visit[i]] for i, _start, close in schedule_visit(config, t + 1, [(c, "standard") for c in visit])
                   if ends[visit[i]] < config.horizon)
        return undone and not late and all(self._repair_fits(t + 1, c) for c in visit)

    def _repair_fits(self, order_t, component):
        """Whether an order created at order_t closes before the horizon, leaving a cycle for the finish."""
        durations = self.env.config.durations
        return order_t + durations["lead_time"][component] + durations["repair_downtime"][component] < self.env.config.horizon

class DoNothingPolicy:
    """Waits up to the horizon, waiting again after every notification, and finishes if the episode is still open: the
    lower reference of the verifier."""

    def __init__(self):
        self.started = False

    def start(self, obs):
        self.horizon = profile_of(obs)["horizon"]
        self.started = True

    def act(self, obs):
        _started(self, obs)
        if obs["time"] < self.horizon:
            return {"type": "wait_for_notification", "timeout": self.horizon - obs["time"]}
        return {"type": "finish", "summary": "No action was taken."}


class DeclarePolicy:
    """Reads the cheapest sensor group for one cycle, cites that reading, declares insufficient evidence and finishes, whatever
    the reading says: the most a constant declaration can earn, with no model and no hidden state."""

    def __init__(self):
        self.started = False

    def start(self, obs):
        prices = profile_of(obs)["prices"]["read"]
        self.group = min(prices, key=lambda g: (prices[g], g))
        self.read, self.declared = False, False
        self.started = True

    def act(self, obs):
        _started(self, obs)
        if not self.read:
            self.read = True
            return {"type": "read_sensors", "sensor_group": self.group, "cycles": 1}
        if not self.declared:
            self.declared = True
            cited = [f"obs/{served['n']}/{field}" for served in obs.get("result", {}).get("served", ()) for field in served["values"]]
            return {"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": self.group, "cycles": 1}, "evidence_ids": cited}
        return {"type": "finish", "summary": "Insufficient evidence was declared after one reading, and nothing was changed."}


class RandomPolicy:
    """A fixed mixture over legal looking actions, not a uniform draw over legal actions. Before the diagnosis it submits a
    random diagnosis with probability 0.15, or whenever no sensor group is affordable, and otherwise reads one cycle of a
    random affordable group. After the diagnosis it draws r uniformly: r < 0.4 orders a random component without an open
    order if its price and a visit charge are affordable (an unaffordable draw falls through), r < 0.7 waits one to five
    cycles for a notification, and it finishes otherwise. Without a budget every price is affordable."""

    def __init__(self, seed):
        # A stream separate from the environment's: the same integer seeds env.reset, and two generators seeded alike emit
        # the same numbers, which tied this policy's first decision to the hidden combination.
        self.rng = random.Random(f"random-policy:{seed}")
        self.started = False

    def start(self, obs):
        profile = profile_of(obs)
        self.profile = profile
        self.components = list(profile["components"])
        self.diagnosed = False
        self.started = True

    def act(self, obs):
        _started(self, obs)
        prices = self.profile["prices"]
        left = obs.get("budget_left", math.inf)
        groups = [g for g, p in prices["read"].items() if p <= left]
        if not self.diagnosed:
            if not groups or self.rng.random() < 0.15:
                self.diagnosed = True
                diagnosis = {c: self.rng.choice(self.profile["levels"][c]) for c in self.components}
                return {"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": []}
            return {"type": "read_sensors", "sensor_group": self.rng.choice(groups), "cycles": 1}
        r = self.rng.random()
        if r < 0.4:
            c = self.rng.choice(self.components)
            open_on = {o["component"] for o in obs["open_work_orders"]}
            if c not in open_on and prices["parts"][c] + prices["repair_charge"] + prices["visit"] <= left:
                return {"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}
        if r < 0.7:
            return {"type": "wait_for_notification", "timeout": self.rng.randint(1, 5)}
        return {"type": "finish", "summary": "Random policy finished."}


def run_episode(env, policy, seed):
    """Run one seeded episode to its end and return the log; the horizon and the step cap guarantee termination, so a
    policy that never diagnoses ends without a diagnosis instead of getting one written for it."""
    obs = env.reset(seed)
    policy.start(obs)
    while not env.done:
        obs, cost, done, info = env.step(policy.act(obs))
    return env.log
