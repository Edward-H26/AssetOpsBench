"""The check of a plan (CausalDiagnosis/docs/general-agent-plan.md, section The check): the walk of the plan on a copy of the
ledger, which reads the table with its row keys, the rule cards in force through the plan_rule of their kinds, the limits in the
status of the ledger and the schema of every tool. Stubs of the freeze; the check builds them."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkState:
    """What a rule reads at a step: the table, the board, the copy of the ledger as the steps before this one left it (a Walk),
    those steps as a tuple, and the plan."""

    table: object
    board: object
    walk: object
    earlier: tuple
    plan: object


def step_problems(plan, board, ledger, table, modules):
    """The reasons of the check, each (step number or None, reason); a reason with step None belongs to the plan as a whole."""
    raise NotImplementedError


def check_plan(plan, board, ledger, table, modules):
    """The reasons of step_problems as text, a step reason opened by "step k (t): "; an empty list for a plan that passes."""
    raise NotImplementedError
