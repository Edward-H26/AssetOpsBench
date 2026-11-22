"""The plan as the general agent reads it (CausalDiagnosis/docs/general-agent-plan.md, section The plan): a step is one call of
one tool of the table, with the arguments of the tool's schema, its expectation and the identifiers it cites; the one step that
calls no tool is recall."""
from dataclasses import dataclass

OWN_STEPS = ("recall",)  # the steps of the controller that call no tool


class PlanError(ValueError):
    """What a reader raises inside the reader of a reply; it never leaves a boundary function."""


@dataclass
class Step:
    """One step: the tool, its arguments, its expectation and the identifiers it cites, which a step holds only where its row
    fills an argument from evidence. candidate is the rank of the candidate whose arguments a step of a commit write with
    candidate arguments takes, lifted out of its arguments, or None."""

    tool: str
    args: dict
    expect: dict
    evidence: list
    candidate: int | None = None

    def to_json(self):
        data = {"tool": self.tool, "args": dict(self.args), "expect": dict(self.expect), "evidence": list(self.evidence)}
        if self.candidate is not None:
            data["candidate"] = self.candidate
        return data


@dataclass
class Plan:
    """A plan: the rank of the candidate it follows or None, one line of why, its steps, the model's candidates best first, each
    {argument that states the answer: value, ..., "cites": [identifier, ...]}, and the reply it was read from."""

    follow: int | None
    why: str
    steps: list
    candidates: list
    raw: str

    def to_json(self):
        data = {"follow": self.follow, "why": self.why, "steps": [step.to_json() for step in self.steps]}
        if self.candidates:
            data["candidates"] = [dict(candidate) for candidate in self.candidates]
        return data


@dataclass(frozen=True)
class Unusable:
    """The first step of a plan that the controller cannot use, by its number counted from 1, with its reasons, and the tools of
    it and of every step after it as the reply wrote them; that step and the steps after it do not run."""

    step: int
    reasons: tuple
    kinds: tuple


@dataclass(frozen=True)
class Refusal:
    """A reply that runs nothing, with its reason, which names for the model what to write. fault is None for a refusal by a reason
    of its own and holds the class name of the exception where the last guard of a boundary function made the refusal."""

    reason: str
    fault: str | None = None


@dataclass(frozen=True)
class Standing:
    """Why a step stands: its effect already stands on the board, so it makes no call and does not stop the plan. clause is one
    of episode_answer, surface_answer, nothing_new, same_commitment and state_holds; reason is the text that the next request
    names it with."""

    clause: str
    reason: str


@dataclass(frozen=True)
class Parsed:
    """What the reader of a reply returns: the plan with the first step that cannot be used, if any, or the refusal. Exactly one
    of plan and refusal is set."""

    plan: Plan | None
    unusable: Unusable | None
    refusal: Refusal | None

    def __post_init__(self):
        if (self.plan is None) == (self.refusal is None):
            raise ValueError("exactly one of plan and refusal is set")
