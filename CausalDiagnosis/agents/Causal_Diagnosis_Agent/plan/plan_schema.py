from dataclasses import dataclass, field


OPENING_KINDS = ("profile", "failure_modes", "fix_history", "time", "extent")  # the free reads of the opening, planned by the model
KINDS = ("read", "diagnose", "declare", "order", "wait", "finish", *OPENING_KINDS, "recall")
KIND_ALIASES = {"advance": "wait"}  # the old name of the wait step, still accepted from a model
MODES = ("test_first", "act_now", "repair_as_test", "declare")
ESTIMATE = "estimate"  # the prediction_id of an order that asks the controller to call estimate_intervention_effect first


class PlanError(ValueError):
    """A reply that holds no usable plan or differential; the message is quoted back to the model."""


@dataclass
class Step:
    kind: str
    args: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)

    def to_json(self):
        return {"kind": self.kind, "args": dict(self.args), "expect": dict(self.expect), "evidence": list(self.evidence)}


@dataclass
class Plan:
    branch: int
    mode: str
    why: str
    steps: list
    raw: str = ""
    candidates: list = field(default_factory=list)  # the model's own candidates, which only a plan without the server lists

    def to_json(self):
        data = {"branch": self.branch, "mode": self.mode, "why": self.why, "steps": [s.to_json() for s in self.steps]}
        if self.candidates:
            data["candidates"] = [{"levels": dict(c["levels"]), "failure_mode_ids": list(c["failure_mode_ids"])} for c in self.candidates]
        return data
