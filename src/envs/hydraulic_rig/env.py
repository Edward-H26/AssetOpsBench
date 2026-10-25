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
    def __init__(self, cycles, blocks, prices=None, budget=150, max_repairs=2, min_faulty=1, max_faulty=2, max_steps=200, block_cap=10, allowed=None):
        """min_faulty, max_faulty: the public fault-count rule; every episode starts with between min_faulty and
        max_faulty components off their healthy level, and the observation states both numbers.
        block_cap: every served block is a seeded window of exactly this many stable cycles, so block length
        carries no information about the combination, however long its stable block is.
        allowed: optional set of combinations the episode may start from (a train or evaluation split)."""
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
        return self.observation()

    def _enter_block(self):
        """Serve a seeded window of `block_cap` cycles of the current combination's stable block, entered at a seeded offset."""
        block = self.blocks[tuple(self.hidden)]
        cap = min(self.block_cap or len(block), len(block))
        start = self._rng.randrange(len(block) - cap + 1)
        self._block = block[start:start + cap]
        self._pos = self._rng.randrange(cap)
        self._seen = 0

    def observation(self):
        return {
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
            "history": copy.deepcopy([self._public(e) for e in self.log if e["event"] in ("measure", "replace", "submit", "refused")]),
        }

    @staticmethod
    def _public(e):
        """The agent-facing view of a log event: no cycle numbers (the UCI data is public and labelled),
        no combination after a repair, no unnecessary-replacement flag. The full event stays in the log for the scorer."""
        if e["event"] == "measure":
            return {k: e[k] for k in ("event", "tier", "cost", "result", "repeated")}
        if e["event"] == "replace":
            return {k: e[k] for k in ("event", "component", "cost")}
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
            self.log.append({"event": "measure", "tier": tier, "cost": price, "cycle": idx, "repeated": repeated, "result": result})
            return self.observation(), price, False, {}
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
            i = COMPONENTS.index(comp)
            unnecessary = self.hidden[i] == HEALTHY[comp]
            self.hidden[i] = HEALTHY[comp]
            self.repairs += 1
            if not unnecessary:
                self._enter_block()  # the combination changed, so the recordings of the new combination are served
            self.log.append({"event": "replace", "component": comp, "cost": price, "unnecessary": unnecessary, "combo_after": tuple(self.hidden)})
            if tuple(self.hidden) == HEALTHY_COMBO:
                return self._end("recovered", price)
            if self.repairs >= self.max_repairs:
                return self._end("repairs used up", price)
            return self.observation(), price, False, {}
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
