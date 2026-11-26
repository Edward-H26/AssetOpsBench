"""The core texts of the prompt as named constants (CausalDiagnosis/docs/general-agent-plan.md, section The prompt): the blocks
ROLE, the head of TOOLS, EXPECTATIONS, EVIDENCE, CANDIDATES, BOARD, PLAN with RECALL and BATCH, the head of RULES and LOOP, and the
texts of the four turns, each a constant whose slots are nouns of the vocabulary, names of tools, names of limits or a clause
that the assembler renders. The texts that the present agent prints as well are imported from agents/utils/core_texts.py, and
this module holds the others. A slot is a name in braces, and only a slot that the assembler names is filled, so a brace of a
JSON form in a text stays as it is. No word of a dataset stands in a constant of this module."""
from CausalDiagnosis.agents.utils.core_texts import (BOARD, BOARD_HELD, BOARD_HELD_RECORD, EARLIER_TURNS_KEEPS, EVIDENCE,
                                                     EVIDENCE_CARRIED, EXPECTATIONS, LOOP, LOOP_STANDING, NOT_RUN_CLOSING,
                                                     NOT_RUN_HEADING, OBJECTIVE, READ_STANDS, ROLE, RULES_HEAD, STOOD_CLOSING,
                                                     STOOD_HEADING, STOOD_LINE)

__all__ = ["BOARD", "BOARD_HELD", "BOARD_HELD_RECORD", "EARLIER_TURNS_KEEPS", "EVIDENCE", "EVIDENCE_CARRIED", "EXPECTATIONS", "LOOP",
           "LOOP_STANDING", "NOT_RUN_CLOSING", "NOT_RUN_HEADING", "OBJECTIVE", "READ_STANDS", "ROLE", "RULES_HEAD", "STOOD_CLOSING",
           "STOOD_HEADING", "STOOD_LINE"]

# the two words of the plan form of this agent in the slots of the shared texts
STEP_KEY = "tool"  # the key of a step that names what it calls
PLAN_HEAD = "follow or candidates"  # the keys of the head of a plan that the check judges as a whole

# TOOLS: the head, the clause of each role in the order of the plan page, and the clause of each write class
TOOLS_HEAD = ("The tools. Each line below is one tool that a step of your plan can call, with its arguments. Every tool has a role: "
              "{roles}. {writes} An argument that the controller writes is marked so, and you leave it out.")
ROLE_CLAUSES = (
    ("observe", "observe for the tools that read the {short_noun}"),
    ("knowledge", "knowledge for what is known about it"),
    ("act", "act for the work on it"),
    ("task", "task for the tools that read or answer the task"),
)
WRITES_HEAD = "A tool that changes something is a write, and its line names its class: "
WRITE_CLAUSES = (
    ("repair", "a repair restores a part of the {short_noun}"),
    ("probe", "a probe changes it in order to test it"),
    ("state", "a state write sets a public state"),
    ("answer", "an answer records your answer to the task"),
)

# EXPECTATIONS: the sentence of the events, printed with a space before it where the declaration lists events
EVENTS = ('Any step may also state {"no_event": [<names>]}, with names among {event names}: a notification of one of these events '
          "during the step is a surprise.")

# EVIDENCE: the sentence that names the forms of the identifiers that a step may cite, each with what it names
IDENTIFIERS = "The identifiers take these forms: {forms}."
IDENTIFIER_FORM = "{form} for {names}"
FORM_RECORDS = ("obs/<label or n>/<field>", "a value of an observation")
FORM_TEXT = ("obs/<n>", "an observation")
FORM_KNOWLEDGE = ("<as an answer of {tool} gives it>", "an entry of that answer")

CANDIDATES = ('Candidates and your answer. A candidate is one possible answer to the task: a value for each argument that states the '
              'answer, {candidate_arguments}. List your candidates best first under "candidates", and name the one your plan follows '
              'by its rank under "follow". The board keeps your latest list. {commitment}')
CANDIDATE_VALUES = "each one of its values above"
CANDIDATE_OMITTED = "; one you leave out takes the value that {facts_tool} lists for it under {path}"
COMMIT_ONCE = "You commit your answer with {commit_tools}, once in an episode."
END_TOOL = "{end_tool} ends the episode."
COMMIT_ENDS = "You commit your answer with {tool}, which also ends the episode."

# BOARD: the clauses of {parts}, each printed where the table and the kind have its part, joined in this order
PART_FACTS = "what holds for the whole episode, which is the public facts once {facts_tool} is answered"
PART_FACTS_KNOWLEDGE = " and what the knowledge tools returned"
PART_KNOWLEDGE = "what holds for the whole episode, which is what the knowledge tools returned"
PART_SUMMARY = "your summary of earlier turns, when one was written"
PART_OBSERVATIONS = "the observations with their identifiers"
PART_CANDIDATES = "your candidates"
PART_OPEN_WORK = "the open work on the {short_noun}"
PART_SURPRISES = "the surprises"
PART_LAST_PLAN = "your last plan with the state of each step"

PLAN = ('The plan. Return\n'
        '{"follow": <rank of the candidate you follow, or null>, "why": "<one line>", "candidates": [{...}, ...], '
        '"steps": [{"tool": "<a tool>", "args": {...}, "expect": {...}}, ...]}\n'
        'with at least one entry under "steps"{count}. A step is one call of one tool with the arguments of its line. '
        '"candidates" is optional. The why line names the identifiers that moved you.')
COUNT_SET = " and at most {n} entries; a longer plan is refused and you are asked again"
COUNT_PAID = ", as many as {limits} pay; a plan they cannot pay is refused and you are asked again"

RECALL = ('One more step calls no tool: {"tool": "recall", "args": {"observations": [<first>, <last>]{series}}}, with no expectation. '
          "It costs nothing: the controller puts the full rows of those observations, which the board collapsed into a summary row, "
          "back on the next board, once. A plan of recalls alone makes no call and still takes one step.")
RECALL_SERIES_ARGUMENT = ', "series": <one {series term}>'
RECALL_SERIES = "Name one {series term} to bring back its rows alone."

BATCH = ("Consecutive steps of the role observe run as one batch: the controller makes the calls one after the other and checks every "
         "expectation after the last, so a batch pays for all of its calls before it sees any of them.")

# LOOP with the switch expectations off opens with this sentence in place of its first sentence
LOOP_WITHOUT_EXPECTATIONS = "The controller runs every step of your plan and calls you again when its steps have run."

# {limits}: the names of the limits that the episode has, from Ledger.limits()
LIMIT_WORDS = {"budget": "the budget", "time": "the {time_limit}", "steps": "the step limit"}
NO_LIMITS = "the limits of the episode"

# the rendered texts of a tool line
TOOL_LINE_HEAD = "- {name} ({role}{write}): args {args}."
TOOL_WRITE = ", {write} write"
ARGUMENT = '"{name}": <{text}>'
ARG_ANY = "any value"
ARG_ENUM = "one of {values}"
ARG_VALUES = "one of the {path} that {facts_tool} lists"
ARG_FILL = "written by the controller"
ARG_EACH = "or empty, for one call per value"
ARG_MINIMUM = "at least {n}"
ARG_WAITS = "0 is read as 1"
ARG_OMITTED_FIXED = "left out, it is {value}"
ARG_OMITTED_PATH = "left out, it takes the value that {facts_tool} lists for it under {path}"
KEY_SENTENCES = {
    "facts": "Its answer holds the public facts of the episode.",
    "commits": "It commits your answer.",
    "ends": "It ends the episode.",
    "sets": "It sets {state} to {value}.",
    "target": "It repairs what {argument} names.",
    "after": "Only after {x}.",
    "waits": "The controller repeats the call until the time asked has passed.",
    "words": "Its answer holds one of the words {words}.",
    "requires_observation": "It must cite an observation.",
    "candidate": 'Give these arguments, or "candidate": <rank> to take them from that candidate.',
}
KEY_ORDER = ("facts", "commits", "ends", "sets", "target", "after", "waits", "words", "requires_observation", "candidate")
STATE_WORD = "the state"  # {state} where the vocabulary of the kind names no state
NEEDS = "Needs {state} {value}."
NEEDS_WITH = "With {argument} {values}: needs {state} {value}."
EXPECT_RECORDS = ('expect {"outside": [<fields>]}: a listed field that the answer serves inside its normal range is a surprise; a name '
                  "stands for the field of that name or for every field whose name begins with it and an underscore{example}, and a "
                  "field that the answer did not serve is not judged.")
EXPECT_EXAMPLE = ", such as {example_field}"
EXPECT_TEXT = 'expect {"word": <one of its words>}: holds when the answer holds exactly that one of its words.'
EXPECT_MODULE = 'expect {"{key}": <{shape}>}.'

# the four turns
TASK = "Task of the episode: "
ORIENTATION_HEADING = "## Orientation"
PLANNING_HEADING = "## Planning turn"
EARLIER_TURNS_HEADING = "## Summary of your earlier turns"
SUMMARY_HEADING = "## Closing summary"
ORIENTATION_ASK = ("No observation has been made yet. Write the differential: for every cause that could explain the task, what you expect "
                   "to observe if it holds and the identifiers behind it. Set no probability. Return {\"differential\": [{\"about\": "
                   "\"<a cause, in the words of the tools and the board>\", \"expected\": \"<one line>\", \"cites\": [\"<identifier>\", "
                   "...]}, ...]} and nothing after it.")
BEFORE_FACTS_TURN = "Before {facts_tool} is answered this plan holds only the opening reads, and the next request asks for the rest."
# the sentence that the board prints before the facts (working_memory), kept beside the sentence of the turn that it matches
BEFORE_FACTS_BOARD = ("No answer of {facts_tool} yet, so the board holds nothing about the {short_noun}. A plan made before {facts_tool} is "
                      "answered holds only the opening reads: {facts_tool} first, then {the knowledge tools}. Plan every other step in the "
                      "next request, once its answer is on the board.")
RETRY_HEADING = "### Your previous reply was not usable"
RETRY_PLAN = "Return a plan that passes, as the JSON object only."
RETRY_DIFFERENTIAL = "Return the JSON object only."
NOT_RUN_LINE = "- step {number} ({step_name})"
NOT_RUN_REASON = "- {reason}"
EARLIER_TURNS_ASK = ("You are writing a summary of your own earlier planning turns, which will leave the board. " + EARLIER_TURNS_KEEPS
                     + " Write what those do not say: the progress so far and the decisions you made with their reasons, the evidence "
                     "identifiers that moved you and why, the candidates you set aside and why, and what remains to be done. Write no "
                     "probability of your own. Return the summary as plain text, with no JSON and no plan.")
LEFT_OUT = "The oldest entries of the board were left out of this request to make it fit, and they will not return."
WRITTEN_HEAD = "The writes of the episode:"
WRITE_LINE = "- {call}: {answer}"
NO_WRITE = "No write was made."
VERDICTS_HEAD = "The verdicts:"
VERDICT_LINE = "- {verdict}"
STATUS_LINE = "Limits: {limits}."
STATUS_WORDS = {"time": "{time_unit} {value}", "time_limit": "{time_limit} {value}", "budget_left": "budget left {value}",
                "steps_left": "steps left {value}"}
STATUS_OVER = "the episode is over"
TIME_WORD, TIME_LIMIT_WORD = "time", "time limit"  # {time_unit} and {time_limit} where the kind names none
CLOSING_ASK = ("Write the closing summary of the episode in a few sentences of plain prose: your answer and the evidence behind it, each "
               "write and what it returned, what each verification said, and the state of the {short_noun} as the answers show it. "
               "Claim only what an answer on the board supports. Return plain text, no JSON.")
