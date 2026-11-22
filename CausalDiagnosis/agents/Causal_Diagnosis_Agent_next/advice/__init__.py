"""The call site of the advice modules (CausalDiagnosis/docs/general-agent-plan.md, section Advice modules). A module serves an
advice kind, evidence, ranking, plan_assessment or outcome_check; it never reads a tool name or writes an argument name, and the
key call of the row of its tool says which argument takes which input. Module, the method list of a module, and ModuleContext, what
a module reads and asks through, are protocols that no class inherits from. Stubs of the freeze; the loop and the modules build
them. Modules of this package, one per line:

ranking
warrants
"""
from typing import Protocol


class ModuleContext(Protocol):
    """What a module reads, the board and the ledger, and ask(kind, inputs), which calls the tool of that kind with the inputs
    mapped by the key call of its row, records the call in the trajectory, counts no step and returns the answer in the shape of
    agents/utils/answer.py."""

    board: object
    ledger: object

    async def ask(self, kind, inputs):
        ...


class Module(Protocol):
    """The method list of an advice module. tools is {kind: (tool name, advice record)}."""

    kinds: tuple  # the advice kinds it serves
    switches: tuple  # the switches it owns

    def __init__(self, tools, table, options):
        ...

    def blocks(self):
        """Its block of the system prompt and its sentence of BOARD."""

    def forms(self, row):
        """The expectation keys it judges for a step of this row, each with the reader of its value."""

    def step_problems(self, step, index, state):
        """Its rules of the check."""

    def candidates(self):
        """The candidates it ranks, which take the first ranks."""

    def sections(self, board):
        """What it prints on the board, each with its place."""

    def known_ids(self):
        ...

    def evidence_for(self, step, row):
        """The identifiers it adds to the evidence of a write."""

    async def before_turn(self, ctx):
        ...

    async def before_step(self, step, row, ctx):
        """The fills it writes (the warrant) and its notes, a dict."""

    async def after_batch(self, steps, answers, ctx):
        """The judgment of its forms, one per step. A standing observe step, which makes no call, comes alone with None as its
        answer and is judged on what the board holds."""

    async def on_pending(self, pending, observation, ctx):
        """Its verdict on a closed repair, a dict, or None."""

    def to_json(self):
        ...


def modules_for(table, options):
    """One module per advice kind for which the table lists a tool and a module exists."""
    raise NotImplementedError


def expectation_forms(table, modules):
    """{tool: {expectation key: reader}}, built once per turn: for every row the key of its answer form, no_event where the
    declaration lists events, and the keys of every module's forms(row)."""
    raise NotImplementedError
