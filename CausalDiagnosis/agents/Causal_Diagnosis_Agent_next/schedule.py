"""The schedule of the candidate plans on the board (CausalDiagnosis/docs/general-agent-plan.md, section The loop): the present
schedule of candidate plans over Walk.step and Walk.needs. The plans are those that a ranking module hands it, or the model's
candidates through Ledger.targets and Ledger.verifying_call where no such module is active; a kind whose candidates name no target
gets no schedule. A stub of the freeze; the loop builds it."""
from dataclasses import dataclass, field


@dataclass
class Schedule:
    """The schedule of one planning turn. differs is true when it says more than the board without it says."""

    # the observe calls of the batch, each with the ranks of the plans it serves and whether the advice names it
    reads: list = field(default_factory=list)
    cut: list = field(default_factory=list)  # the calls that the batch leaves out, each with the limit that leaves it out
    left_out: list = field(default_factory=list)  # [rank, limit] of the plans whose repair no longer fits after the calls
    remaining: list = field(default_factory=list)  # after a refuted plan, what each remaining plan still needs
    differs: bool = False

    def to_json(self):
        raise NotImplementedError


def schedule(board, ledger, plans, named, reserve, refuted):
    """The Schedule of a planning turn: the observe calls that the advice names and those whose test several plans share, each
    once with the plans it serves, cut where they would no longer leave the reserve for a repair, and after a refuted plan what
    each remaining plan still needs."""
    raise NotImplementedError
