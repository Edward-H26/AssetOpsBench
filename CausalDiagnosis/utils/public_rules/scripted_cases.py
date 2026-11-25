"""Public state and counted-call walk for a scripted tool surface."""

import copy
import json

from .plan_rules import KindRules, Ledger, Walk, Needs, StepCost
from CausalDiagnosis.environment.scripted_cases.replay import condition


class ScriptedRules(KindRules):
    default_answer = "object"

    def __init__(self, tools):
        self.tools = copy.deepcopy(tools)

    def ledger(self, facts, flags, cards, task):
        if flags or cards:
            raise ValueError("this scripted surface declares no bindings or rule cards")
        public = json.loads(task) if isinstance(task, str) else task
        return ScriptedLedger(self.tools, public)


class ScriptedLedger(Ledger):
    def __init__(self, tools, task):
        self.tools = {item["name"]: copy.deepcopy(item) for item in tools}
        limits = task.get("limits", {})
        self.public = {"time": None, "time_limit": None, "over": False,
                       "steps_left": limits.get("max_tool_calls"), "budget_left": limits.get("budget"),
                       "state": copy.deepcopy(task.get("initial_states", {}))}
        self.observed = False

    def record(self, row, args, answer):
        self.public.update(copy.deepcopy(answer.get("status", {})))
        if "refused" not in answer and answer.get("observations"):
            self.observed = True

    def status(self):
        return copy.deepcopy(self.public)

    def limits(self):
        return tuple(key for key in ("steps_left", "budget_left") if self.public.get(key) is not None)

    def walk(self):
        return ScriptedWalk(copy.deepcopy(self))

    def open_targets(self):
        return {}

    def paid(self, series):
        return ()

    def pending(self):
        return []

    def verifies(self, row, args, targets):
        return False

    def verdict(self, pending, observations):
        return None

    def targets(self, candidate):
        return ()

    def verifying_call(self, targets):
        return None

    def ranges(self):
        return {}

    def sections(self, vocabulary):
        return [("Public state", "State returned by the tool server", copy.deepcopy(self.public.get("state", {}))),
                ("Call prices", "The server charges the matched action's price; the walk counts calls without guessing hidden prices.", [])]


class ScriptedWalk(Walk):
    def __init__(self, ledger):
        self.ledger, self.calls = ledger, 0

    def step(self, row, args):
        tool = self.ledger.tools[row.name]
        state = self.ledger.public.get("state", {})
        reasons = tuple(f"{row.name} requires {item}" for item in tool.get("requires", []) if not condition(item, state))
        self.calls += 1
        if not reasons:
            state.update(copy.deepcopy(tool.get("updates", {})))
            if row.role == "observe":
                self.ledger.observed = True
        return StepCost(0, 0, 1, False, reasons)

    def end(self):
        return () if self.ledger.observed else ("read an observation before submitting",)

    def needs(self, repairs):
        return Needs(0, 0, len(repairs))

    def totals(self):
        return 0, 0, self.calls
