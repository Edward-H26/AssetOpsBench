"""The rules of a kind of evidence, as functions over public values: the cost of a call, the ledger of what is open and what is
paid, the walk of a plan on a copy of the ledger, and the board sections of the kind (CausalDiagnosis/docs/general-agent-plan.md,
section Rules of a kind). One class per built in kind derives from KindRules; the declaration of a run instantiates the class
that the evidence kind names. Stubs of the freeze: the rules of each kind build them. The module imports the standard library
alone and holds no data at module level beyond constants and tuples of constants."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StepCost:
    """The cost of one step on the walk's copy of the ledger: its price, the time it takes, the counted calls it makes, whether
    the price or the time needs the facts, and the reasons of the kind's rules for this step. stands is "nothing_new" or
    "state_holds" where a call of the step could change nothing, else None."""

    price: float
    time: int
    calls: int
    needs_facts: bool
    reasons: tuple
    stands: str | None = None


@dataclass(frozen=True)
class Needs:
    """The price, the time and the counted calls of completing the named repair writes from a state of a walk."""

    price: float
    time: int
    calls: int


class KindRules:
    """The rules of one kind. refusal_marks are the phrases that mark an answer in text form as a refusal; default_answer is the
    form of an answer of an observe, knowledge or act tool whose row states none."""

    refusal_marks = ()
    default_answer = "object"

    def ledger(self, facts, flags, cards, task):
        """The ledger of one episode: facts None until the facts tool is answered, flags those of the binding ({} where none
        stands), cards the rule cards in force, task the task message of the episode."""
        raise NotImplementedError


class Ledger:
    """The public state of one episode that the rules of the kind read."""

    def record(self, row, args, answer):
        """After every answered call: what is open, what is paid, the clock. The answer of the facts tool hands over the facts."""
        raise NotImplementedError

    def status(self):
        """{time, time_limit, budget_left, steps_left, over}, each None where the kind has none or no answer has shown it yet."""
        raise NotImplementedError

    def limits(self):
        """The names of the limits that the episode has, of budget, time and steps; () while none is known."""
        raise NotImplementedError

    def walk(self):
        """A copy of the ledger for one walk of a plan."""
        raise NotImplementedError

    def open_targets(self):
        """target -> when its repair closes, None where unknown."""
        raise NotImplementedError

    def paid(self, series):
        """The units of a series that the episode has paid for, () where the kind counts none."""
        raise NotImplementedError

    def pending(self):
        """The repairs closed and not yet verified, each with its key, its targets and when it closed."""
        raise NotImplementedError

    def verifies(self, row, args, targets):
        """Whether an observation by this call can verify a repair of the targets."""
        raise NotImplementedError

    def verdict(self, pending, observations):
        """The kind's own verdict on a repair, a dict, or None where the kind has none."""
        raise NotImplementedError

    def targets(self, candidate):
        """What a candidate asks to repair, () where the kind cannot tell."""
        raise NotImplementedError

    def verifying_call(self, targets):
        """The cheapest observe call that can verify a repair of the targets, or None."""
        raise NotImplementedError

    def ranges(self):
        """field -> (low, high) of the public normal ranges, {} where the kind has none."""
        raise NotImplementedError

    def sections(self, vocabulary):
        """The sections that the board prints for the kind, each (heading, header, rows)."""
        raise NotImplementedError


class Walk:
    """One walk of a plan on a copy of the ledger."""

    def step(self, row, args):
        """The StepCost of a step: its price, time and calls, whether they need the facts, and the reasons of the kind's rules. A
        step for which the kind states stands has no price, no time and no call."""
        raise NotImplementedError

    def end(self):
        """What must precede the write that ends the episode, as reasons; none where the limits leave no way to it."""
        raise NotImplementedError

    def needs(self, repairs):
        """The Needs of completing these repair writes from here."""
        raise NotImplementedError

    def totals(self):
        """(price, time, calls) of the walk so far."""
        raise NotImplementedError
