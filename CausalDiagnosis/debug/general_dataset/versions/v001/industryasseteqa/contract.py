"""Public QA semantics for the general agent, separate from its controller."""

import copy

from CausalDiagnosis.agents.utils.declaration import Declaration, ToolRow, Vocabulary
from CausalDiagnosis.utils.public_rules.plan_rules import KindRules, Ledger, Walk, Needs, StepCost


class QARules(KindRules):
    def ledger(self, facts, flags, cards, task):
        if flags or cards:
            raise ValueError("QA declares no binding flags or rule cards")
        return QALedger()


class QALedger(Ledger):
    def __init__(self):
        self.observed = False
        self.over = False

    def record(self, row, args, answer):
        self.over = self.over or answer.get("status", {}).get("over") is True
        if "refused" not in answer and "result" in answer and row.role == "observe":
            self.observed = self.observed or bool(answer.get("observations"))

    def status(self):
        return dict(time=None, time_limit=None, budget_left=None, steps_left=None, over=self.over)

    def limits(self):
        return ()

    def walk(self):
        return QAWalk(copy.deepcopy(self))

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
        return []


class QAWalk(Walk):
    def __init__(self, ledger):
        self.ledger, self.calls = ledger, 0

    def step(self, row, args):
        reasons = ("the episode is over",) if self.ledger.over else ()
        if row.requires_observation and not self.ledger.observed:
            reasons += ("read the released evidence before submitting",)
        self.calls += 1
        if row.role == "observe":
            self.ledger.observed = True
        return StepCost(0, 0, 1, False, reasons)

    def end(self):
        return () if self.ledger.observed else ("read the released evidence before submitting",)

    def needs(self, repairs):
        return Needs(0, 0, 0)

    def totals(self):
        return 0, 0, self.calls


def declaration():
    rows = [ToolRow("get_task", "maintenance_qa", "knowledge", facts=True),
            ToolRow("read_fact", "maintenance_qa", "observe", after=("get_task",)),
            ToolRow("submit_answer", "maintenance_qa", "task", write="answer", commits=True,
                    ends=True, after=("read_fact",), requires_observation=True, candidate=("answer",))]
    return Declaration("industryasseteqa_pdm", "maintenance_qa", None,
                       Vocabulary("maintenance question", "question",
                                  identifiers=(("fact/<fact_id>", "released measurements"),)),
                       {row.name: row for row in rows}, {}, {}, (), {}, (), QARules(), None)
