"""Greedy reference policy: the reading with the largest expected information gain per unit price.

The rule and its assumptions are written down in docs/hydraulic_rig/environment.md; this module follows it. The policy
keeps a belief over the 144 combinations, reads each served observation through the cross-fitted per-tier classifiers
(classifiers.py), buys one cycle of the affordable sensor group with the largest gain per unit price while keeping enough
budget for its work orders, submits the most probable combination at the confidence it is constructed with, creates one
work order per diagnosed fault, advances time until the orders close and finishes. It reads only the tool answers.
Standard library only.
"""
import math
from operator import add

from servers.hydraulic_rig.names import HYD1_FIELDS

from .classifiers import COMBOS, TIERS, signature
from .data import COMPONENTS, HEALTHY, LEVELS, TIER_SENSORS
from .env import n_faulty, served_values
from .policies import _started, profile_of

LEVEL_INDEX = [tuple(LEVELS[c].index(h[j]) for j, c in enumerate(COMPONENTS)) for h in COMBOS]


def entropy_bits(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def expected_information_gain(belief, rows):
    """Mutual information in bits between the hypothesis and the next outcome, IG = H(O) - sum_h b(h) H(O | h).

    rows[h] lists, per component, the distribution of that component's outcome under hypothesis h; the components
    are independent given h (assumption A1), so an outcome is a tuple in itertools.product order."""
    p_outcome, conditional = None, 0.0
    for w, comps in zip(belief, rows):
        if w <= 0:
            continue
        conditional += w * sum(entropy_bits(v) for v in comps)
        joint = [w]
        for v in comps:
            joint = [a * b for a in joint for b in v]
        p_outcome = joint if p_outcome is None else list(map(add, p_outcome, joint))
    return entropy_bits(p_outcome) - conditional


def uniform_prior(min_faulty, max_faulty):
    """Uniform belief over the combinations with min_faulty to max_faulty faulty components."""
    allowed = [h for h in COMBOS if min_faulty <= n_faulty(h) <= max_faulty]
    return [1.0 / len(allowed) if h in allowed else 0.0 for h in COMBOS]


class ServedClassifiers:
    """The held-out predictions of a TierClassifiers keyed by the values a reading serves (public names, mean and std at
    six significant digits), so that the policy decides on tool answers exactly as on the recordings. The constructor
    refuses a table in which two cycles of a tier serve the same values."""

    def __init__(self, classifiers, cycles):
        by_index = {c.index: c for c in cycles}
        self.base, self.alpha = classifiers.base, classifiers.alpha
        self.lookup = {}
        for tier, predicted in classifiers.predicted.items():
            table = {signature(served_values(by_index[i], TIER_SENSORS[tier])): p for i, p in predicted.items()}
            if len(table) != len(predicted):
                raise ValueError(f"two tier-{tier} cycles serve identical values")
            self.lookup[tier] = table

    def predict(self, tier, values):
        try:
            return self.lookup[tier][signature(values)]
        except KeyError:
            raise KeyError("served values not in the cross-fitted table; build ServedClassifiers from the environment's cycles") from None


class GreedyPolicy:
    def __init__(self, classifiers, use_max_faulty=False, confidence=0.95, min_gain_per_price=0.05):
        """classifiers: a ServedClassifiers. use_max_faulty: True, the reference, starts every episode from a uniform prior
        over the combinations within the public fault rule of the asset profile; False, the ablation without that rule,
        starts from a uniform prior over the 143 non-healthy combinations. min_gain_per_price: bits per price unit below
        which a reading is not worth buying."""
        self.clf = classifiers
        self.use_max_faulty = use_max_faulty
        self.confidence = confidence
        self.min_gain_per_price = min_gain_per_price
        self.started = False

    def start(self, obs):
        profile = profile_of(obs)
        self.profile = profile
        self.started = True
        rule = (profile["fault_rule"]["min_faulty"], profile["fault_rule"]["max_faulty"]) if self.use_max_faulty else (1, len(COMPONENTS))
        self.belief = uniform_prior(*rule)
        self.counts = {t: [[0] * len(LEVELS[c]) for c in COMPONENTS] for t in TIERS}
        self.measured = {t: set() for t in TIERS}
        # the classifiers are fitted per measurement tier, so a sensor group is read through the tier with the same channels
        tier_fields = {t: sorted(HYD1_FIELDS[s] for s in TIER_SENSORS[t]) for t in TIERS}
        self.tier_of_group = {g: t for g, fields in profile["sensor_groups"].items() for t in TIERS if sorted(fields) == tier_fields[t]}
        self.block_exhausted = False
        self.diagnosis = None
        self.evidence = []
        self.ordered = set()
        self.last_action = None

    def _next_rows(self, tier):
        """Per hypothesis and component, the distribution of the next tier-t prediction given this episode's earlier
        tier-t predictions (Polya urn around the held-out confusion row, assumption A3)."""
        rows = []
        for h in range(len(COMBOS)):
            comps = []
            for j in range(len(COMPONENTS)):
                alpha, n = self.clf.alpha[tier][j], self.counts[tier][j]
                total = alpha + sum(n)
                comps.append([(alpha * p + c) / total for p, c in zip(self.clf.base[tier][j][h], n)])
            rows.append(comps)
        return rows

    def _absorb(self, obs):
        """Update the belief with the observations the last reading served."""
        if self.last_action is None or self.last_action["type"] != "read_sensors" or "result" not in obs:
            return
        tier = self.tier_of_group[self.last_action["sensor_group"]]
        for served in obs["result"]["served"]:
            self.block_exhausted |= bool(served["repeated"])
            self.evidence += [f"obs/{served['n']}/{field}" for field in served["values"]]
            key = signature(served["values"])
            if key in self.measured[tier]:
                continue  # assumption A4: the same cycle again at the same tier carries no new information
            self.measured[tier].add(key)
            outcome = self.clf.predict(tier, served["values"])
            rows = self._next_rows(tier)
            self.belief = [b * math.prod(v[k] for v, k in zip(comps, outcome)) for b, comps in zip(self.belief, rows)]
            total = sum(self.belief)
            self.belief = [b / total for b in self.belief]
            for j, k in enumerate(outcome):
                self.counts[tier][j][k] += 1

    def _order_reserve(self):
        """Budget kept for the work orders: the most expensive orders the fault rule may need."""
        prices = self.profile["prices"]
        costs = sorted((prices["parts"][c] + prices["repair_charge"] for c in COMPONENTS), reverse=True)
        return sum(costs[: self.profile["fault_rule"]["max_faulty"]])

    def act(self, obs):
        _started(self, obs)
        self._absorb(obs)
        action = self._investigate(obs) if self.diagnosis is None else self._repair(obs)
        self.last_action = action
        return action

    def _investigate(self, obs):
        if max(self.belief) < self.confidence and not self.block_exhausted:
            spendable = obs["budget_left"] - self._order_reserve()
            rates = {}
            for group, price in self.profile["prices"]["read"].items():
                if price <= spendable and group in self.tier_of_group:
                    rates[group] = expected_information_gain(self.belief, self._next_rows(self.tier_of_group[group])) / price
            if rates:
                group = max(rates, key=rates.get)
                if rates[group] >= self.min_gain_per_price:
                    return {"type": "read_sensors", "sensor_group": group, "cycles": 1}
        best = max(range(len(COMBOS)), key=self.belief.__getitem__)
        self.diagnosis = dict(zip(COMPONENTS, COMBOS[best]))
        return {"type": "submit_diagnosis", "diagnosis": dict(self.diagnosis), "evidence_ids": list(self.evidence)}

    def _repair(self, obs):
        prices = self.profile["prices"]
        p_faulty = {c: sum(b for b, lv in zip(self.belief, LEVEL_INDEX) if lv[j] != LEVELS[c].index(HEALTHY[c]))
                    for j, c in enumerate(COMPONENTS)}
        todo = [c for c in COMPONENTS if self.diagnosis[c] != HEALTHY[c] and c not in self.ordered]
        for comp in sorted(todo, key=lambda c: -p_faulty[c]):
            if prices["parts"][comp] + prices["repair_charge"] <= obs["budget_left"]:
                self.ordered.add(comp)
                return {"type": "create_work_order", "component": comp, "prediction_id": "", "evidence_ids": list(self.evidence)}
        if obs["open_work_orders"]:
            closes = min(o["expected_close_t"] for o in obs["open_work_orders"])
            return {"type": "advance_time", "cycles": max(1, closes - obs["time"])}
        return {"type": "finish", "summary": f"Diagnosed {self.diagnosis} and ordered the repair of {sorted(self.ordered)}."}
