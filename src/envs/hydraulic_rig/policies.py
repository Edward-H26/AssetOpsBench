"""Reference policies over the v2 action set (docs/hydraulic_rig/environment.md, section Reference policies). The greedy
policy lives in greedy.py with the per-tier classifiers.

A policy has start(first_envelope), called with the envelope that reset returns (its result is the asset profile), and
act(envelope), called with the answer to its previous action; it sees nothing else, which is exactly what a language
model agent gets through the tools. A harness that skips start may hand the reset envelope to act, which starts the
policy itself. OraclePolicy is the exception: it reads the hidden state through the environment.
"""
import random

from .clock import worst_level
from .data import COMPONENTS, HEALTHY


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
    """Reads the hidden state (the start combination, the current levels, the schedule and the external event), which no
    agent observes. It submits the true diagnosis (or, on a start the diagnosability grade marks inseparable, declares
    insufficient evidence with the cheapest sensor group as the separating test, see `inseparable`), orders every faulty
    component once it will not drop further, leaves a component to the technician when the technician comes before the
    component would fail, advances time until the orders close and finishes. A drop whose repair could not close before the horizon is not waited for: the policy
    finishes before it, and the oracle set of such an episode never holds that component (oracle.py). It is the reference
    for the cost of an episode (cost_oracle) and the upper bound of the verifier; only the tests, the drivers and the gym
    wrappers construct it, never an agent harness."""

    def __init__(self, env, inseparable=False):
        """inseparable: True when the grade of the start says that no test sequence within the budget separates it; the
        policy then declares insufficient evidence instead of submitting the diagnosis, since the verifier counts a
        commitment on such a start as no success, and it still repairs every faulty component, so it stays the reference
        of RFC 008 under the success definition."""
        self.env = env
        self.inseparable = inseparable
        self.diagnosed = False

    def start(self, obs):
        self.diagnosed = False

    def act(self, obs):
        env, config, clock = self.env, self.env.config, self.env.clock
        if not self.diagnosed:
            self.diagnosed = True
            if self.inseparable:
                cheapest = min(config.prices["read"], key=lambda g: (config.prices["read"][g], g))
                return {"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": cheapest, "cycles": 1}, "evidence_ids": []}
            return {"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, env.initial)), "evidence_ids": []}
        t = env.time
        awaited = [d for d in clock.schedule if t <= d["t"] < config.horizon and self._repair_fits(d["t"] + 1, d["component"])]
        technician = self._technician_in_time(t)
        for c in config.components:
            if clock.levels[c] == config.healthy[c] or clock.order_open_on(c):
                continue
            if any(d["component"] == c for d in awaited):
                continue  # one order after the last drop keeps the order set at one entry per component
            if technician is not None and technician["component"] == c:
                continue
            return {"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}
        waits = [o.close_t for o in clock.open_orders()] + [d["t"] + 1 for d in awaited]
        if technician is not None:
            waits.append(technician["t"] + 1)
        if waits:
            return {"type": "advance_time", "cycles": max(1, min(waits) - t)}
        opening = "Insufficient evidence declared" if self.inseparable else "Diagnosis submitted"
        return {"type": "finish", "summary": f"{opening}; every faulty component was repaired through a work order."}

    def _repair_fits(self, order_t, component):
        """Whether an order created at order_t closes before the horizon, leaving a cycle for the finish."""
        durations = self.env.config.durations
        return order_t + durations["lead_time"][component] + durations["repair_downtime"][component] < self.env.config.horizon

    def _technician_in_time(self, t):
        """The pending external event when waiting for it is safe: the technician comes before the horizon and, for a
        component at its worst level, before the failure event would fire."""
        config, clock = self.env.config, self.env.clock
        event = clock.external
        if event is None or not t <= event["t"] < config.horizon:
            return None
        c = event["component"]
        if clock.levels[c] == worst_level(config, c) and event["t"] > config.failure["patience"] - 1:
            return None
        return event


class DoNothingPolicy:
    """Advances time to the horizon and finishes if the episode is still open: the lower reference of the verifier."""

    def __init__(self):
        self.started = False

    def start(self, obs):
        self.horizon = profile_of(obs)["horizon"]
        self.started = True

    def act(self, obs):
        _started(self, obs)
        if obs["time"] < self.horizon:
            return {"type": "advance_time", "cycles": self.horizon - obs["time"]}
        return {"type": "finish", "summary": "No action was taken."}


class RandomPolicy:
    """A fixed mixture over legal looking actions, not a uniform draw over legal actions. Before the diagnosis it submits a
    random diagnosis with probability 0.15, or whenever no sensor group is affordable, and otherwise reads one cycle of a
    random affordable group. After the diagnosis it draws r uniformly: r < 0.4 orders a random component without an open
    order if affordable (an unaffordable draw falls through), r < 0.7 advances one to five cycles, and it finishes
    otherwise."""

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
        left = obs["budget_left"]
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
            if c not in open_on and prices["parts"][c] + prices["repair_charge"] <= left:
                return {"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []}
        if r < 0.7:
            return {"type": "advance_time", "cycles": self.rng.randint(1, 5)}
        return {"type": "finish", "summary": "Random policy finished."}


def run_episode(env, policy, seed):
    """Run one seeded episode to its end and return the log; the horizon and the step cap guarantee termination, so a
    policy that never diagnoses ends without a diagnosis instead of getting one written for it."""
    obs = env.reset(seed)
    policy.start(obs)
    while not env.done:
        obs, cost, done, info = env.step(policy.act(obs))
    return env.log


def faulty_components(combination):
    return [c for i, c in enumerate(COMPONENTS) if combination[i] != HEALTHY[c]]
