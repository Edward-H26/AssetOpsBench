"""Public ledger doubles for checks that must not implement an evidence kind."""

import copy

from CausalDiagnosis.utils.public_rules.plan_rules import StepCost


class Ledger:
    def __init__(self, **status):
        self.state = {"time": None, "steps_left": None, "budget_left": None, **status}
        self.recorded = []
        self.open = {}
        self.stands = {}
        self.public = {}
        self.prices = {}
        self.durations = {}
        self.reasons = {}
        self.end_reasons = ()

    def status(self):
        return copy.deepcopy(self.state)

    def ranges(self):
        return {"pressure": (1, 3)}

    def sections(self, vocabulary):
        return copy.deepcopy(self.public)

    def record(self, row, args, answer):
        self.recorded.append((row.name, args, answer))
        self.state.update(answer.get("status", {}))

    def walk(self):
        return Walk(self)

    def open_targets(self):
        return self.open


class Walk:
    def __init__(self, ledger):
        self.ledger = copy.deepcopy(ledger)
        self.price = self.time = self.calls = 0

    def step(self, row, args):
        stands = self.ledger.stands.get(row.name)
        cost = StepCost(self.ledger.prices.get(row.name, 1) if not stands else 0,
                        self.ledger.durations.get(row.name, 0), 0 if stands else 1, False,
                        self.ledger.reasons.get(row.name, ()), stands)
        self.price += cost.price
        self.calls += cost.calls
        self.time += cost.time
        return cost

    def totals(self):
        return self.price, self.time, self.calls

    def open_targets(self):
        return self.ledger.open_targets()

    def end(self):
        return self.ledger.end_reasons
