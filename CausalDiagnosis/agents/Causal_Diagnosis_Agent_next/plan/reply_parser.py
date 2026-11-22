"""The reader of a reply of the model (CausalDiagnosis/docs/general-agent-plan.md, sections The plan and The check).
parse_usable_plan and parse_differential are the two boundary functions for a reply: each takes any text and returns, and one last
guard turns any exception into the refusal "the reply could not be read" with the class name of the exception. Stubs of the
freeze; the reader of a reply builds them."""


def last_json_object(text):
    """The last complete top level JSON object of a text, or None."""
    raise NotImplementedError


def parse_usable_plan(text, table, forms, plan_length):
    """The Parsed of a reply: the plan up to its first step that cannot be used, with that step as Unusable, or the Refusal of a
    reply without a plan head. forms is {tool: {expectation key: reader}} (expectation_forms of the advice package); plan_length
    is the count of steps that a plan may hold, or None. An argument that a row fills is dropped with a note. A step carries
    evidence only where its row fills an argument from evidence; on a step of another tool the key is not read. A rank written as
    "candidate" under the args of a step of a commit write with candidate arguments is lifted into Step.candidate, and a step of
    such a write that writes none of those arguments and no rank follows the candidate of rank 1, which what the reply itself
    writes decides before any omitted value is written. Then an argument that a row states under omitted and that a step or a
    candidate leaves out takes its value, before the check."""
    raise NotImplementedError


def parse_candidates(raw, table):
    """The model's own candidates of a plan, best first, each a value for each argument that states the answer and its cites."""
    raise NotImplementedError


def parse_differential(text):
    """(entries, Refusal or None): the differential of the orientation turn, each entry {"about", "expected", "cites"}."""
    raise NotImplementedError
