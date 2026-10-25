"""Greedy reference policy: the measurement with the largest expected information gain per unit price.

The rule and its assumptions are written down in docs/hydraulic_rig/environment.md; this module follows it.
The policy keeps a belief over the 144 combinations, reads each measurement through the cross-fitted per-tier
classifiers (classifiers.py), buys the affordable tier with the largest gain per unit price while keeping enough
budget for its repairs, submits the most probable combination, and then replaces the components it diagnosed as
faulty. It reads only the observation. Standard library only.
"""
import math
from operator import add

from .classifiers import COMBOS, TIERS, signature
from .data import COMPONENTS, HEALTHY, LEVELS
from .env import n_faulty

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


class GreedyPolicy:
    def __init__(self, classifiers, use_max_faulty=False, confidence=0.95, min_gain_per_price=0.05):
        """use_max_faulty: True, the reference (row greedy_prior_1to2), starts every episode from a uniform prior over
        the combinations with min_faulty to max_faulty faulty components, the public fault-count rule as the first
        observation of the episode states it; False, the default and the ablation without that rule (row greedy),
        starts from a uniform prior over the 143 non-healthy combinations.
        min_gain_per_price: bits per price unit below which a measurement is not worth buying."""
        self.clf = classifiers
        self.use_max_faulty = use_max_faulty
        self.confidence = confidence
        self.min_gain_per_price = min_gain_per_price

    def _start_episode(self, obs):
        rule = (obs["min_faulty"], obs["max_faulty"]) if self.use_max_faulty else (1, len(COMPONENTS))
        self.belief = uniform_prior(*rule)
        self.counts = {t: [[0] * len(LEVELS[c]) for c in COMPONENTS] for t in TIERS}
        self.measured = {t: set() for t in TIERS}
        self.block_exhausted = False
        self.seen = 0
        self.diagnosis = None

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

    def _absorb(self, history):
        for event in history[self.seen:]:
            if event["event"] != "measure" or self.diagnosis is not None:
                continue
            tier = int(event["tier"])
            self.block_exhausted |= bool(event["repeated"])
            key = signature(event["result"])
            if key in self.measured[tier]:
                continue  # assumption A4: the same cycle again at the same tier carries no new information
            self.measured[tier].add(key)
            outcome = self.clf.predict(tier, event["result"])
            rows = self._next_rows(tier)
            self.belief = [b * math.prod(v[k] for v, k in zip(comps, outcome)) for b, comps in zip(self.belief, rows)]
            total = sum(self.belief)
            self.belief = [b / total for b in self.belief]
            for j, k in enumerate(outcome):
                self.counts[tier][j][k] += 1
        self.seen = len(history)

    @staticmethod
    def _repair_reserve(obs):
        prices = obs["prices"]
        costs = sorted((prices["parts"][c] + prices["downtime"] for c in COMPONENTS), reverse=True)
        return sum(costs[: obs["repairs_left"]])

    def act(self, obs):
        if not obs["history"]:
            self._start_episode(obs)
        self._absorb(obs["history"])
        if obs["phase"] == "investigate":
            return self._investigate(obs)
        return self._repair(obs)

    def _investigate(self, obs):
        if max(self.belief) < self.confidence and not self.block_exhausted:
            spendable = obs["budget_left"] - self._repair_reserve(obs)
            rates = {}
            for tier, price in obs["prices"]["measure"].items():
                if price <= spendable:
                    rates[int(tier)] = expected_information_gain(self.belief, self._next_rows(int(tier))) / price
            if rates:
                tier = max(rates, key=rates.get)
                if rates[tier] >= self.min_gain_per_price:
                    return {"type": "measure", "tier": tier}
        best = max(range(len(COMBOS)), key=self.belief.__getitem__)
        self.diagnosis = dict(zip(COMPONENTS, COMBOS[best]))
        return {"type": "submit", "diagnosis": dict(self.diagnosis)}

    def _repair(self, obs):
        replaced = {e["component"] for e in obs["history"] if e["event"] == "replace"}
        p_faulty = {c: sum(b for b, lv in zip(self.belief, LEVEL_INDEX) if lv[j] != LEVELS[c].index(HEALTHY[c]))
                    for j, c in enumerate(COMPONENTS)}
        todo = [c for c in COMPONENTS if self.diagnosis[c] != HEALTHY[c] and c not in replaced]
        for comp in sorted(todo, key=lambda c: -p_faulty[c]):
            price = obs["prices"]["parts"][comp] + obs["prices"]["downtime"]
            if obs["repairs_left"] > 0 and price <= obs["budget_left"]:
                return {"type": "replace", "component": comp}
        return {"type": "finish"}
