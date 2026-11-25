"""Shared text templates for the plan check and its prompt clauses."""

KNOWN_TOOL = "{tool} is no tool of this episode, and the tools are {names}"
FACTS_FIRST = "leave this step out and plan it in the next request, once {facts} is answered, since {detail} comes from that answer"
ORDER = "put {needed} before this step, since {tool} needs its answer"
END_LAST = "{tool} ends the episode, so it is the last step of a plan"
CITES_OBSERVATION = "{tool} counts only when it cites an observation, and none is on the board or earlier in the plan"
STANDS_STATE = "leave this step out, since {state} is already {value}"
REPLY_TOKEN_CAP = "the reply was cut at the token cap and was not parsed; return a shorter complete JSON object"

RULE_SENTENCES = (
    ("known_tool", "no step that names a tool outside the table"),
    ("arguments", "no step whose arguments its tool does not take"),
    ("facts_first", "before the facts tool is answered, leave out a step whose arguments or price need its answer"),
    ("order", "a tool follows every answer that its row requires"),
    ("single_commitment", "no commitment that differs from the one that stands"),
    ("end_last", "the tool that ends the episode is the last step of a plan"),
    ("cites_observation", "no write requiring an observation before an observation it can cite"),
    ("candidate_rank", "no follow or candidate rank outside the candidates shown"),
    ("candidate_values", "no candidate with a value that the table or public facts do not list"),
    ("recall_rows", "recall only observation rows that the board has collapsed"),
    ("expectation", "state only the expectations declared for the tool"),
    ("open_repair", "no repair write on a target whose repair is still open"),
    ("precondition", "no call while the state has a value that its tool does not allow"),
    ("end_guard", "finish after repair closes and verifying observations wherever the limits allow them"),
    ("budget", "no plan that the declared budget cannot pay"),
    ("time_limit", "finish before the declared time limit"),
    ("step_limit", "no plan that exceeds the declared counted calls left"),
)

from CausalDiagnosis.agents.utils.core_texts import (
    STANDS_EPISODE, STANDS_SURFACE, STANDS_COMMITMENT, STANDS_COLLAPSED,
    SECOND_COMMITMENT as SINGLE_COMMITMENT,
    STANDS_NOTHING_NEW, STANDS_NOTHING_RECORDED, OPEN_REPAIR,
    OPEN_REPAIR_UNTIL, STANDS_EXPECTATION, NOTHING_NEW_AFTER_CALL,
)
