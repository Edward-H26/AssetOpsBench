"""The v0 environment: hidden combination, priced measurements, submit-then-repair, replace-and-replay (docs/hydraulic_rig/environment.md).

Order rule: the diagnosis is submitted before the first repair, so repair outcomes cannot be
used to back out the diagnosis. Phase "investigate": measure and submit. Phase "repair": measure, replace, finish.

Every malformed or out-of-order action that is a dict is refused (logged, no charge, episode continues) rather than
raised, so a language-model agent that emits a bad action loses a step, not the episode; an action that is not a dict
is a programming error and raises. The step cap bounds every episode. The log keeps deep copies of everything the
agent hands in, and the agent-facing history is a deep copy, so no in-process policy can rewrite its own diagnosis.
"""
import copy
import math
import random

from .data import COMPONENTS, HEALTHY, LEVELS, TIER_SENSORS, tier_features

DEFAULT_PRICES = {
    "measure": {1: 1, 2: 3, 3: 10},
    "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20},
    "downtime": 20,
}
HEALTHY_COMBO = tuple(HEALTHY[c] for c in COMPONENTS)

# The clock (off unless an environment is built with clock=...): time is counted in 60-second rig cycles.
# durations: how many cycles a measurement at each tier takes. lead_time: cycles between a work order (replace) and
# the start of the repair, while the rig keeps running. repair_cycles: cycles the rig is stopped for the repair
# (downtime). max_advance: the largest advance_time call. degradation: a hidden schedule of `events` drops, each at a
# cycle drawn from min_cycle to max_cycle, where one component falls to its next recorded level; None turns it off.
# fault_cost_per_cycle: charged for every running cycle in which a component is off its healthy level.
DEFAULT_CLOCK = {
    "durations": {"measure": {"1": 1, "2": 2, "3": 5}},
    "lead_time": 10,
    "repair_cycles": 5,
    "max_advance": 500,
    "degradation": {"events": 1, "min_cycle": 20, "max_cycle": 200},
    "fault_cost_per_cycle": 0,
}


def worse_level(component, level):
    """The next recorded level of `component` away from its healthy level, or None at its worst level."""
    levels = LEVELS[component]
    step = -1 if HEALTHY[component] == levels[-1] else 1
    j = levels.index(level) + step
    return levels[j] if 0 <= j < len(levels) else None


def n_faulty(combo):
    return sum(1 for i, c in enumerate(COMPONENTS) if combo[i] != HEALTHY[c])


def reachable_by_repair(combo):
    """Every combination an episode starting at `combo` can enter by replacing components (each replacement sets one
    component to its healthy level), the start itself included."""
    seen, frontier = {tuple(combo)}, [tuple(combo)]
    while frontier:
        cur = frontier.pop()
        for i, c in enumerate(COMPONENTS):
            if cur[i] != HEALTHY[c]:
                nxt = cur[:i] + (HEALTHY[c],) + cur[i + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    return seen


def closed_under_repair(train, held_out):
    """True when no held-out combination can be entered from any training start by repairs (docs/hydraulic_rig/environment.md).
    The recommended split holds out the two-fault combinations: every single fault stays in training and an
    evaluation episode is a new combination of known faults."""
    reach = set()
    for combo in train:
        reach |= reachable_by_repair(combo)
    return not (reach & set(held_out))


class TinyHydraulicEnv:
    def __init__(self, cycles, blocks, prices=None, budget=150, max_repairs=2, min_faulty=1, max_faulty=2, max_steps=200, block_cap=10, allowed=None, clock=None):
        """min_faulty, max_faulty: the public fault-count rule; every episode starts with between min_faulty and
        max_faulty components off their healthy level, and the observation states both numbers.
        block_cap: every served block is a seeded window of exactly this many stable cycles, so block length
        carries no information about the combination, however long its stable block is.
        allowed: optional set of combinations the episode may start from (a train or evaluation split); a scheduled
        degradation never leaves it either.
        clock: None (no time, the v0 environment) or a clock configuration like DEFAULT_CLOCK."""
        if not 1 <= min_faulty <= max_faulty:
            raise ValueError(f"the fault-count rule needs 1 <= min_faulty <= max_faulty, got min_faulty={min_faulty}, max_faulty={max_faulty}")
        self.cycles = {c.index: c for c in cycles}
        self.blocks = blocks
        self.prices = _checked_prices(copy.deepcopy(prices or DEFAULT_PRICES))
        self.budget = budget
        self.max_repairs = max_repairs
        self.min_faulty = min_faulty
        self.max_faulty = max_faulty
        self.max_steps = max_steps
        self.block_cap = block_cap
        self.allowed = allowed
        self.clock_cfg = None if clock is None else _checked_clock(copy.deepcopy(clock))
        self.candidates = sorted(k for k in blocks if min_faulty <= n_faulty(k) <= max_faulty and (allowed is None or k in allowed))
        self._rng = None
        self.done = True

    # ----- episode control -----
    def reset(self, seed):
        self._rng = random.Random(seed)
        self.hidden = list(self._rng.choice(self.candidates))
        self.initial = tuple(self.hidden)
        self._enter_block()
        self.cost = 0
        self.repairs = 0
        self.steps = 0
        self.phase = "investigate"
        self.done = False
        self.log = [{"event": "reset", "seed": seed, "hidden": self.initial, "budget": self.budget, "max_repairs": self.max_repairs, "max_steps": self.max_steps, "prices": copy.deepcopy(self.prices)}]
        self.clock = 0
        self._schedule = []
        if self.clock_cfg:
            self._schedule = self._draw_schedule(seed)
            self.log[0].update(clock=copy.deepcopy(self.clock_cfg), schedule=[list(s) for s in self._schedule])
        return self.observation()

    # ----- the clock -----
    def _draw_schedule(self, seed):
        """The hidden degradation schedule: sorted (cycle, draw) pairs. The draw picks the component when the drop is
        due, among those that can still drop. It comes from a generator of its own, so a seed starts from the same
        combination and window with or without the clock."""
        deg = self.clock_cfg["degradation"]
        if not deg:
            return []
        rng = random.Random(f"clock-{seed}")
        return sorted((rng.randint(deg["min_cycle"], deg["max_cycle"]), rng.random()) for _ in range(deg["events"]))

    def _tick(self, cycles, running=True):
        """Move the clock by `cycles`. While the rig runs, due degradations fire, and a cycle with a component off its
        healthy level is a fault cycle, charged fault_cost_per_cycle. While it is stopped for a repair, the cycle is
        downtime and nothing degrades. Returns the time cost, which is already added to the episode cost."""
        if not cycles:
            return 0
        start, faults = self.clock, 0
        for _ in range(cycles):
            self.clock += 1
            if not running:
                continue
            while self._schedule and self._schedule[0][0] <= self.clock:
                self._degrade(self._schedule.pop(0)[1])
            if tuple(self.hidden) != HEALTHY_COMBO:
                faults += 1
        cost = faults * self.clock_cfg["fault_cost_per_cycle"]
        self.cost += cost
        self.log.append({"event": "time", "cost": cost, "from": start, "to": self.clock, "running": running, "fault_cycles": faults})
        return cost

    def _degrade(self, draw):
        """One scheduled drop: a component that can still fall (and stay inside `allowed`) falls to its next level, and
        the recordings of the new combination are served from then on. The agent is not told; it can only see the
        readings change."""
        eligible = []
        for i, c in enumerate(COMPONENTS):
            level = worse_level(c, self.hidden[i])
            if level is None:
                continue
            target = tuple(self.hidden[:i]) + (level,) + tuple(self.hidden[i + 1:])
            if target in self.blocks and (self.allowed is None or target in self.allowed):
                eligible.append((i, c, level))
        if not eligible:
            self.log.append({"event": "degrade", "cost": 0, "clock": self.clock, "component": None})
            return
        i, c, level = eligible[int(draw * len(eligible))]
        before = self.hidden[i]
        self.hidden[i] = level
        self._enter_block()
        self.log.append({"event": "degrade", "cost": 0, "clock": self.clock, "component": c, "from": before, "to": level, "combo_after": tuple(self.hidden)})

    def _after_time(self, cost):
        """The step result after an action that let time pass: a time cost can overrun the budget, which ends the episode."""
        if self.clock_cfg and self.cost > self.budget:
            return self._end("budget", cost)
        return self.observation(), cost, False, {}

    def _enter_block(self):
        """Serve a seeded window of `block_cap` cycles of the current combination's stable block, entered at a seeded offset."""
        block = self.blocks[tuple(self.hidden)]
        cap = min(self.block_cap or len(block), len(block))
        start = self._rng.randrange(len(block) - cap + 1)
        self._block = block[start:start + cap]
        self._pos = self._rng.randrange(cap)
        self._seen = 0

    def observation(self):
        obs = {
            "phase": self.phase,
            "components": list(COMPONENTS),
            "levels": {c: list(LEVELS[c]) for c in COMPONENTS},
            "healthy": dict(HEALTHY),
            "tiers": {t: list(s) for t, s in TIER_SENSORS.items()},
            "prices": copy.deepcopy(self.prices),
            "budget_left": self.budget - self.cost,
            "repairs_left": self.max_repairs - self.repairs,
            "min_faulty": self.min_faulty,  # the public fault-count rule: between min_faulty and max_faulty components
            "max_faulty": self.max_faulty,  # start off their healthy level
            "steps_left": self.max_steps - self.steps,
            "history": copy.deepcopy([self._public(e) for e in self.log if e["event"] in ("measure", "replace", "submit", "refused", "advance_time")]),
        }
        if self.clock_cfg:  # the public part of the clock: the rules and the time, never the schedule
            cfg = self.clock_cfg
            obs["clock"] = self.clock
            obs["clock_rules"] = {"durations": copy.deepcopy(cfg["durations"]), "lead_time": cfg["lead_time"], "repair_cycles": cfg["repair_cycles"],
                                  "max_advance": cfg["max_advance"], "fault_cost_per_cycle": cfg["fault_cost_per_cycle"], "degradation": bool(cfg["degradation"])}
        return obs

    @staticmethod
    def _public(e):
        """The agent-facing view of a log event: no cycle numbers (the UCI data is public and labelled),
        no combination after a repair, no unnecessary-replacement flag. The full event stays in the log for the scorer."""
        if e["event"] == "measure":
            return {k: e[k] for k in ("event", "tier", "cost", "result", "repeated", "clock") if k in e}
        if e["event"] == "replace":
            return {k: e[k] for k in ("event", "component", "cost", "clock") if k in e}
        if e["event"] == "advance_time":
            return {k: e[k] for k in ("event", "cycles", "clock")}
        if e["event"] == "submit":
            return {k: e[k] for k in ("event", "diagnosis", "insufficient_evidence", "next_action")}
        return {"event": e["event"], "action": e.get("action"), "reason": e.get("reason")}

    # ----- actions -----
    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over; call reset")
        if not isinstance(action, dict):
            raise TypeError("action must be a dict")
        self.steps += 1
        obs, cost, done, info = self._apply(action)
        if not done and self.steps >= self.max_steps:
            return self._end("step limit", cost)
        return obs, cost, done, info

    def _apply(self, action):
        kind = action.get("type")
        if kind == "measure":
            tier = action.get("tier")
            if not isinstance(tier, int) or isinstance(tier, bool) or tier not in self.prices["measure"]:
                return self._refuse(action, "unknown tier")  # tiers are the integer keys of the price list
            price = self.prices["measure"][tier]
            if not self._charge(price, action):
                return self._end("budget")
            idx = self._block[(self._pos + self._seen) % len(self._block)]
            repeated = self._seen >= len(self._block)
            self._seen += 1
            result = tier_features(self.cycles[idx], tier)
            event = {"event": "measure", "tier": tier, "cost": price, "cycle": idx, "repeated": repeated, "result": result}
            if self.clock_cfg:  # the reading is taken when the measurement starts; the measurement then takes its duration
                event["clock"] = self.clock
                self.log.append(event)
                return self._after_time(price + self._tick(self.clock_cfg["durations"]["measure"][str(tier)]))
            self.log.append(event)
            return self.observation(), price, False, {}
        if kind == "advance_time":
            if not self.clock_cfg:
                return self._refuse(action, "this episode has no clock")
            cycles, most = action.get("cycles"), self.clock_cfg["max_advance"]
            if not isinstance(cycles, int) or isinstance(cycles, bool) or not 1 <= cycles <= most:
                return self._refuse(action, f"cycles must be an integer from 1 to {most}")
            self.log.append({"event": "advance_time", "cost": 0, "cycles": cycles, "clock": self.clock})
            return self._after_time(self._tick(cycles))
        if kind == "submit":
            if self.phase != "investigate":
                return self._refuse(action, "already submitted")
            problem = self._check_submission(action)
            if problem:
                return self._refuse(action, problem)
            self.log.append({"event": "submit", "cost": 0, "diagnosis": copy.deepcopy(action.get("diagnosis")), "insufficient_evidence": bool(action.get("insufficient_evidence")), "next_action": action.get("next_action")})
            self.phase = "repair"
            if self.max_repairs == 0:
                return self._end("no repair phase")
            return self.observation(), 0, False, {}
        if kind == "replace":
            if self.phase != "repair":
                return self._refuse(action, "diagnosis must be submitted before the first repair")
            comp = action.get("component")
            if comp not in COMPONENTS:
                return self._refuse(action, "unknown component")
            price = self.prices["parts"][comp] + self.prices["downtime"]
            if not self._charge(price, action):
                return self._end("budget")
            ordered_at = self.clock
            if self.clock_cfg:  # a work order: the rig keeps running for the lead time, then stops for the repair
                price += self._tick(self.clock_cfg["lead_time"])
                price += self._tick(self.clock_cfg["repair_cycles"], running=False)
            i = COMPONENTS.index(comp)
            unnecessary = self.hidden[i] == HEALTHY[comp]
            self.hidden[i] = HEALTHY[comp]
            self.repairs += 1
            if not unnecessary:
                self._enter_block()  # the combination changed, so the recordings of the new combination are served
            event = {"event": "replace", "component": comp, "cost": self.prices["parts"][comp] + self.prices["downtime"], "unnecessary": unnecessary, "combo_after": tuple(self.hidden)}
            if self.clock_cfg:
                event["clock"] = ordered_at
            self.log.append(event)
            if tuple(self.hidden) == HEALTHY_COMBO:
                return self._end("recovered", price)
            if self.repairs >= self.max_repairs:
                return self._end("repairs used up", price)
            return self._after_time(price)
        if kind == "finish":
            if self.phase != "repair":
                return self._refuse(action, "finish is allowed after the diagnosis")
            self.log.append({"event": "finish", "cost": 0})  # an accepted finish is a step call too (a tool call in v0.1)
            return self._end("finish")
        return self._refuse(action, "unknown action type")

    @staticmethod
    def _check_submission(action):
        """A submission is either one valid level per component, or insufficient_evidence with a named next action."""
        diagnosis = action.get("diagnosis")
        if action.get("insufficient_evidence"):
            if diagnosis is not None:
                return "insufficient_evidence excludes a diagnosis"
            if not isinstance(action.get("next_action"), str) or not action["next_action"]:
                return "insufficient_evidence needs next_action"
            return None
        if not isinstance(diagnosis, dict) or set(diagnosis) != set(COMPONENTS):
            return "diagnosis must give one level per component"
        if any(type(diagnosis[c]) is not int or diagnosis[c] not in LEVELS[c] for c in COMPONENTS):
            return "diagnosis level not in the component's levels"  # levels are ints; True, 1.0 or "100" are refused like a non-integer tier
        return None

    def _charge(self, price, action):
        if self.cost + price > self.budget:
            self.log.append({"event": "refused", "action": copy.deepcopy(action), "cost": 0, "reason": "budget"})
            return False
        self.cost += price
        return True

    def _refuse(self, action, reason):
        self.log.append({"event": "refused", "action": copy.deepcopy(action), "cost": 0, "reason": reason})
        return self.observation(), 0, False, {"refused": reason}

    def _end(self, reason, cost=0):
        """End the episode; `cost` is what the final action was charged (0 for refusals and finish)."""
        self.done = True
        self.log.append({"event": "end", "cost": 0, "reason": reason})
        return self.observation(), cost, True, {"log": self.log, "reason": reason}


def _checked_clock(clock):
    """The clock configuration, checked and with the measure tiers as string keys (so it survives JSON). Like a bad
    price table, a bad clock is a configuration error and raises."""
    def count(value, low=0):
        return isinstance(value, int) and not isinstance(value, bool) and value >= low

    problems = []
    unknown = set(clock) - set(DEFAULT_CLOCK)
    if unknown:
        problems.append(f"unknown keys {sorted(unknown)}")
    missing = set(DEFAULT_CLOCK) - set(clock)
    if missing:
        problems.append(f"missing keys {sorted(missing)}")
    if problems:
        raise ValueError("invalid clock: " + "; ".join(problems))
    measure = {str(t): d for t, d in clock["durations"].get("measure", {}).items()}
    if set(measure) != {str(t) for t in TIER_SENSORS} or not all(count(d) for d in measure.values()):
        problems.append("durations.measure needs a whole number of cycles (0 or more) for every tier")
    clock["durations"] = {"measure": measure}
    for key in ("lead_time", "repair_cycles"):
        if not count(clock[key]):
            problems.append(f"{key} must be a whole number of cycles, 0 or more")
    if not count(clock["max_advance"], 1):
        problems.append("max_advance must be a whole number of cycles, 1 or more")
    rate = clock["fault_cost_per_cycle"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
        problems.append("fault_cost_per_cycle must be a finite number, 0 or more")
    deg = clock["degradation"]
    if deg is not None:
        if not isinstance(deg, dict) or set(deg) != {"events", "min_cycle", "max_cycle"}:
            problems.append("degradation must be None or have events, min_cycle and max_cycle")
        elif not (count(deg["events"]) and count(deg["min_cycle"], 1) and count(deg["max_cycle"], 1) and deg["min_cycle"] <= deg["max_cycle"]):
            problems.append("degradation needs events >= 0 and 1 <= min_cycle <= max_cycle")
    if problems:
        raise ValueError("invalid clock: " + "; ".join(problems))
    return clock


def _checked_prices(prices):
    """The price table, checked when the environment is built so that no measure or replace can count a step or
    charge the budget and then fail on a tier, part or downtime charge the table lacks, or on a price that is not a
    number, not finite or negative, or on a free measurement (a tier priced at zero would let an agent read the whole
    recording for nothing). A bad table is a configuration error, so it raises here instead of being refused like a
    bad action."""
    problems = []
    if "measure" not in prices:
        problems.append("no measure prices")
    unknown_tiers = [t for t in prices.get("measure", {}) if t not in TIER_SENSORS]
    if unknown_tiers:
        problems.append(f"measure tiers {unknown_tiers} have no sensors in TIER_SENSORS")
    unpriced = [c for c in COMPONENTS if c not in prices.get("parts", {})]
    if unpriced:
        problems.append(f"components {unpriced} have no part price")
    if "downtime" not in prices:
        problems.append("no downtime charge")
    by_name = {f"tier {t}": p for t, p in prices.get("measure", {}).items()}
    by_name.update({f"part {c}": p for c, p in prices.get("parts", {}).items()})
    if "downtime" in prices:
        by_name["downtime"] = prices["downtime"]
    # bool is a subclass of int, so True and False are excluded explicitly
    not_numbers = [name for name, p in by_name.items() if isinstance(p, bool) or not isinstance(p, (int, float))]
    if not_numbers:
        problems.append(f"prices {not_numbers} are not numbers")
    numbers = {name: p for name, p in by_name.items() if name not in not_numbers}
    not_finite = [name for name, p in numbers.items() if not math.isfinite(p)]
    if not_finite:
        problems.append(f"prices {not_finite} are not finite")
    negative = [name for name, p in numbers.items() if math.isfinite(p) and p < 0]
    if negative:
        problems.append(f"prices {negative} are negative")
    free_measures = [name for name, p in numbers.items() if name.startswith("tier ") and math.isfinite(p) and p <= 0]
    if free_measures:
        problems.append(f"measure {free_measures} are not positive")
    if problems:
        raise ValueError("invalid price table: " + "; ".join(problems))
    return prices
