"""Greedy reference policy: the reading with the largest expected information gain per unit price.

The rule and its assumptions are written down in CausalDiagnosis/docs/environment.md; this module follows it. The policy
keeps a belief over the 144 combinations, starting uniform over all of them, reads each served observation through the
cross-fitted per-tier classifiers (tier_classifiers.py), buys one cycle of the affordable sensor group with the largest gain per
unit price while keeping the budget of the repair plan of its most probable combination, submits that combination at the
confidence it is constructed with, creates one work order per diagnosed fault, waits until the orders close and
finishes. On the free base row, where every price is zero, a cycle of any group costs one call, so the gain
per call ranks the groups and the calls left before the step cap take the place of the budget. It reads only the tool
answers and never the fault rule of the starts, which stays with the start sampler. Standard
library only.
"""
import math
from operator import add

from CausalDiagnosis.environment.reference_policies.tier_classifiers import COMBOS, COMPONENTS, LEVELS, PROFILE, TIERS, signature
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import cost_unit
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import served_values
from CausalDiagnosis.environment.reference_policies.policies import _started, profile_of

HEALTHY = PROFILE.healthy
TIER_SENSORS = {t: tuple(PROFILE.observation_kinds[f"tier{t}"]["channels"]) for t in TIERS}
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


def uniform_prior():
    """Uniform belief over every combination of the profile, the healthy one included."""
    return [1.0 / len(COMBOS)] * len(COMBOS)


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
    def __init__(self, classifiers, confidence=0.95, min_gain_per_price=0.05):
        """classifiers: a ServedClassifiers. confidence: the belief at which the policy submits its most probable combination.
        min_gain_per_price: bits per price unit below which a reading is not worth buying."""
        self.clf = classifiers
        self.confidence = confidence
        self.min_gain_per_price = min_gain_per_price
        self.started = False

    def start(self, obs):
        profile = profile_of(obs)
        self.profile = profile
        self.started = True
        self.belief = uniform_prior()
        self.counts = {t: [[0] * len(LEVELS[c]) for c in COMPONENTS] for t in TIERS}
        self.measured = {t: set() for t in TIERS}
        # the classifiers are fitted per measurement tier, so a sensor group is read through the tier with the same channels
        tier_fields = {t: sorted(PROFILE.field_name(s) for s in TIER_SENSORS[t]) for t in TIERS}
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
            self.evidence += [f"obs/{served['n']}/{field}" for field in served["values"]]
            self._absorb_reading(tier, served["values"])

    def read_history(self, group, readings):
        """Absorb readings of a sensor group that the episode served before the policy's first action, as the AssetOpsBench
        rows record the recordings of the start before cycle 0: the update of a read_sensors answer of the group, without
        evidence identifiers, since no call of the policy released them. The separability label reads the prehistory of a
        rollout through it (diagnosability_label.py)."""
        tier = self.tier_of_group[group]
        for values in readings:
            self._absorb_reading(tier, values)

    def _absorb_reading(self, tier, values):
        key = signature(values)
        if key in self.measured[tier]:
            # assumption A4: the same cycle again at the same tier carries no new information, and seeing one again
            # means the window of recorded cycles wrapped, which the served values tell without a public repeat flag
            self.block_exhausted = True
            return
        self.measured[tier].add(key)
        outcome = self.clf.predict(tier, values)
        rows = self._next_rows(tier)
        self.belief = [b * math.prod(v[k] for v, k in zip(comps, outcome)) for b, comps in zip(self.belief, rows)]
        total = sum(self.belief)
        self.belief = [b / total for b in self.belief]
        for j, k in enumerate(outcome):
            self.counts[tier][j][k] += 1

    def _order_reserve(self):
        """Budget kept for the work orders: the repair plan of the most probable combination, one order per faulty component at
        its part and repair charge, and the charge of the one visit that orders created one after another form."""
        prices = self.profile["prices"]
        top = COMBOS[max(range(len(COMBOS)), key=self.belief.__getitem__)]
        plan = [prices["parts"][c] + prices["repair_charge"] for c, level in zip(COMPONENTS, top) if level != HEALTHY[c]]
        return sum(plan) + prices["visit"] if plan else 0

    def _calls_reserve(self):
        """Calls kept on the free base row for the steps after the last read: the diagnosis, one order per faulty component
        of the most probable combination, a wait for the closes and the finish."""
        top = COMBOS[max(range(len(COMBOS)), key=self.belief.__getitem__)]
        return 3 + sum(1 for c, level in zip(COMPONENTS, top) if level != HEALTHY[c])

    def act(self, obs):
        _started(self, obs)
        self._absorb(obs)
        action = self._investigate(obs) if self.diagnosis is None else self._repair(obs)
        self.last_action = action
        return action

    def _investigate(self, obs):
        if max(self.belief) < self.confidence and not self.block_exhausted:
            money = cost_unit(self.profile["prices"]) == "money"
            if money:
                spendable = obs.get("budget_left", math.inf) - self._order_reserve()  # no budget, no limit
            else:
                spendable = obs["steps_left"] - self._calls_reserve()
            rates = {}
            for group, price in self.profile["prices"]["read"].items():
                cost = price if money else 1
                if cost <= spendable and group in self.tier_of_group:
                    rates[group] = expected_information_gain(self.belief, self._next_rows(self.tier_of_group[group])) / cost
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
        visit_charge = 0 if self.ordered else prices["visit"]  # the orders follow each other with no other call between, so one visit
        for comp in sorted(todo, key=lambda c: -p_faulty[c]):
            if prices["parts"][comp] + prices["repair_charge"] + visit_charge <= obs.get("budget_left", math.inf):
                self.ordered.add(comp)
                return {"type": "create_work_order", "component": comp, "prediction_id": "", "evidence_ids": list(self.evidence)}
        if obs["open_work_orders"]:
            closes = min(o["expected_close_t"] for o in obs["open_work_orders"])
            return {"type": "wait_for_notification", "timeout": max(1, closes - obs["time"])}
        return {"type": "finish", "summary": f"Diagnosed {self.diagnosis} and ordered the repair of {sorted(self.ordered)}."}
