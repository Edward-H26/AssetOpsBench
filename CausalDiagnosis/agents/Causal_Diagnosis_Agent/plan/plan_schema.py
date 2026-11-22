from dataclasses import dataclass, field


EPISODE_KINDS = ("profile", "failure_modes", "fix_history")  # the reads of the opening whose answer holds for the whole episode
SURFACE_KINDS = ("time", "extent")  # the reads of the opening that a surface needs before a reading, whose answer moves only as cycles pass
OPENING_KINDS = (*EPISODE_KINDS, *SURFACE_KINDS)  # the free reads of the opening, planned by the model
COMMIT_KINDS = ("diagnose", "declare")  # the environment takes one commitment per episode
# the kinds whose handler hands the identifiers a step cites to its call; a step of another kind carries none, so the reader drops the key
EVIDENCE_KINDS = ("diagnose", "declare", "order")
# the words of the rule of the opening that the board, the plan check and the prompts share: what a plan made before the profile
# is read holds, in its order, and where every other step goes; the check also lets such a plan end with finish
OPENING_RULE = "a plan made before the profile is read holds only the reads of the opening"
OPENING_ORDER = "profile first, then failure_modes and fix_history"
NEXT_REQUEST = "in the next request, once the profile is on the board"
KINDS = ("read", "diagnose", "declare", "order", "wait", "finish", *OPENING_KINDS, "recall")
KIND_ALIASES = {"advance": "wait"}  # the old name of the wait step, still accepted from a model
MODES = ("test_first", "act_now", "repair_as_test", "declare")
ESTIMATE = "estimate"  # the prediction_id of an order that asks the controller to call estimate_intervention_effect first
LARGEST_INTEGER = 2 ** 53 - 1  # the largest integer on which every reader of a JSON number agrees exactly (RFC 8259, section 6)


# the form that the reader of a reply takes (reply_parser.parse_step, parse_plan, parse_candidates and parse_differential), as data
# from which reply_schema.py writes the JSON schema of a reply: per field its reader and whether a reply must hold it. A reader is
# text (a string with a character that is not blank), any text, a count of at least its least value, a count or null, a
# probability, a number, a flag, names (a list of text), a span (two observation numbers), the levels of a candidate (an object from
# component to level), a mode of MODES, the candidates or the steps
TEXT, ANY_TEXT, PROBABILITY, NUMBER, FLAG, NAMES, SPAN, LEVELS = "text", "any text", "probability", "number", "flag", "names", "span", "levels"
MODE, CANDIDATES, STEPS = "mode", "candidates", "steps"


def count(least, null=False):
    """The reader of an integer of at least least, or of null as well with null."""
    return ("count or null" if null else "count", least)


STEP_FORM = {
    "read": {"args": {"sensor_group": (TEXT, True), "cycles": (count(1), False)},
             "expect": {"top_rank": (count(1), False), "top_mass_at_least": (PROBABILITY, False), "effective_at_most": (NUMBER, False),
                        "outside": (NAMES, False)}},
    "diagnose": {"args": {"rank": (count(1), False)}, "expect": {"top_rank": (count(1), False)}},
    "declare": {"args": {"sensor_group": (TEXT, True), "cycles": (count(1), False)}, "expect": {}},
    "order": {"args": {"component": (TEXT, True), "prediction_id": (TEXT, False)},
              "expect": {"prediction_id": (TEXT, False), "recovery_at_least": (PROBABILITY, False)}},
    "wait": {"args": {"cycles": (count(0), False)}, "expect": {"no_new_alarm": (FLAG, False)}},
    "finish": {"args": {}, "expect": {}},
    "profile": {"args": {}, "expect": {}},
    "failure_modes": {"args": {"component": (ANY_TEXT, False)}, "expect": {}},
    "fix_history": {"args": {}, "expect": {}},
    "time": {"args": {}, "expect": {}},
    "extent": {"args": {}, "expect": {}},
    "recall": {"args": {"sensor_group": (TEXT, True), "observations": (SPAN, True)}, "expect": {}},
}
# the head of a plan, the free text of why first, so that a model that decodes under the schema writes its reasons before the fields
# that the schema constrains, and required, since a grammar may leave an optional first key out; the reader still reads a plan
# without it, as with the schema off
HEAD_FORM = {"why": (ANY_TEXT, True), "branch": (count(1, null=True), False), "mode": (MODE, True), "candidates": (CANDIDATES, False),
             "steps": (STEPS, True)}
CANDIDATE_FORM = {"levels": (LEVELS, True), "failure_mode_ids": (NAMES, False)}
DIFFERENTIAL_FORM = {"component": (TEXT, True), "expected_symptoms": (TEXT, False), "failure_mode_ids": (NAMES, False)}


class PlanError(ValueError):
    """A reply that holds no usable plan or differential; the message is quoted back to the model."""


@dataclass
class Refusal:
    """A reply that the reader reads nothing from (reply_parser.read_reply): the reason, which names for the model what to write,
    and the class of the exception that the last guard of the reader caught, None for a reply refused for a reason of its own."""
    reason: str
    fault: str = None


@dataclass
class Unusable:
    """The first step of a plan that the parser cannot use, a step that does not parse or a finish with steps after it: its place
    among the steps as the plan check numbers them, the parser's reasons, and the kinds of it and of every step after it as the
    reply wrote them, which do not run."""
    step: int
    reasons: list
    kinds: list


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
