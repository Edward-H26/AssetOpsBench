"""The check of a plan (CausalDiagnosis/docs/general-agent-plan.md, section The check): the walk of the plan on a copy of the
ledger, which reads the table with its row keys, the rule cards in force through the plan_rule of their kinds, the limits in the
status of the ledger and the schema of every tool. Stubs of the freeze; the check builds them."""
import copy
from dataclasses import dataclass, field

from . import rules
from .expectations import expectation_forms
from .plan_schema import PlanError, Step
from .preparation import candidates_of, expand_step, prepare_step
from .readers import copy_json


@dataclass(frozen=True)
class WalkState:
    """What a rule reads at a step: the table, the board, the copy of the ledger as the steps before this one left it (a Walk),
    those steps as a tuple, and the plan."""

    table: object
    board: object
    walk: object
    earlier: tuple
    plan: object


@dataclass
class CheckedStep:
    index: int
    step: object
    stood: object = None
    cost: object = None
    notes: tuple = ()
    problems: tuple = ()


@dataclass
class CheckResult:
    steps: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    def executable(self):
        if any(index is None for index, _reason in self.problems):
            return []
        return [entry for entry in self.steps if not entry.problems]


class _CheckWalk:
    """Extra checker context beside a kind walk, outside the frozen state fields."""

    def __init__(self, plan, board, ledger, table, modules):
        self.base = ledger.walk()
        self.ledger, self.modules = ledger, tuple(modules)
        self.forms = expectation_forms(table, self.modules)
        self.candidates = candidates_of(plan, board, self.modules)
        self.commitment = copy_json(board.commitment)
        if self.commitment:
            held, _notes = prepare_step(Step(self.commitment["tool"], self.commitment["args"], {}, []),
                                        table, board.facts, self.candidates)
            self.commitment = {"tool": held.tool, "args": held.args}
        self.observed = bool(board.observations) or bool(getattr(board, "collapsed_rows", ()))
        self.preview = self.cost = None

    def open_targets(self):
        method = getattr(self.base, "open_targets", None)
        return method() if method else self.ledger.open_targets()

    def end(self):
        return self.base.end()

    def totals(self):
        return self.base.totals()


def inspect_plan(plan, board, ledger, table, modules):
    """Inspect copies, returning an accepted prefix and whole-plan refusals."""
    context = _CheckWalk(plan, board, ledger, table, modules)
    state = WalkState(table, board, context, (), plan)
    result = CheckResult()
    if not plan.steps:
        result.problems.append((None, "the plan needs at least one step"))
        return result
    for rule in (rules.candidate_rank, rules.candidate_values):
        result.problems.extend((None, reason) for reason in rule(None, None, state))
    if result.problems:
        return result
    earlier = []
    for index, original in enumerate(plan.steps, 1):
        try:
            prepared, notes = prepare_step(original, table, board.facts, context.candidates)
            expanded = expand_step(prepared, table, board.facts)
        except PlanError as error:
            result.problems.append((index, str(error)))
            result.steps.append(CheckedStep(index, original, problems=(str(error),)))
            return result
        for step in expanded:
            state = WalkState(table, board, context, tuple(earlier), plan)
            context.cost = context.preview = None
            reasons = []
            for rule in (rules.known_tool, rules.arguments, rules.expectation, rules.candidate_rank):
                reasons.extend(rule(step, index, state))
            stood = None if reasons else rules.standing(step, index, state)
            if not reasons and stood is None:
                for rule in (rules.facts_first, rules.order, rules.single_commitment, rules.end_last,
                             rules.open_repair, rules.cites_observation, rules.recall_rows):
                    reasons.extend(rule(step, index, state))
                if not reasons and step.tool != "recall":
                    for rule in (rules.precondition, rules.end_guard):
                        reasons.extend(rule(step, index, state))
                    for module in context.modules:
                        reasons.extend(module.step_problems(step, index, state))
            reasons = tuple(dict.fromkeys(reasons))
            result.steps.append(CheckedStep(index, step, stood, context.cost, notes, reasons))
            if reasons:
                result.problems.extend((index, reason) for reason in reasons)
                return result
            if stood is None and step.tool != "recall":
                for rule in (rules.budget, rules.time_limit, rules.step_limit):
                    result.problems.extend((None, reason) for reason in rule(step, index, state))
                if result.problems:
                    return result
                context.base = context.preview
                row = table.tools[step.tool]
                if row.commits:
                    context.commitment = {"tool": step.tool, "args": copy_json(step.args)}
                if (row.role == "observe" or row.answer in ("text", "records")) and context.cost.calls:
                    context.observed = True
            earlier.append(step)
    return result


def step_problems(plan, board, ledger, table, modules):
    """The reasons of the check, each (step number or None, reason); a reason with step None belongs to the plan as a whole."""
    return inspect_plan(plan, board, ledger, table, modules).problems


def check_plan(plan, board, ledger, table, modules):
    """The reasons of step_problems as text, a step reason opened by "step k (t): "; an empty list for a plan that passes."""
    return [reason if index is None else f"step {index} ({plan.steps[index - 1].tool}): {reason}"
            for index, reason in step_problems(plan, board, ledger, table, modules)]
