"""Core texts with declaration slots, kept here until shared texts are supplied."""

ROLE = "You plan the work on one {noun} for a controller. The controller keeps a working memory, the board, and calls the tools that your plan names, and no other tool of the {short_noun}. The task of the episode stands above the board in every request. Each turn you receive the task and the board, and you return one JSON object with nothing after it."
TOOLS = "The tools. Each line below is one tool that a step of your plan can call, with its arguments. Every tool has a role: {roles}. {writes}An argument that the controller writes is marked so, and you leave it out."
EXPECTATIONS = 'Expectations. A step may carry an expectation under "expect", a statement about its answer that you are willing to be stopped by. The controller checks it when the answer arrives, and an expectation that fails is a surprise. A call that the {short_noun} refuses is a surprise whatever the step expects. The line of a {step_key} names the expectations its steps can state, and a step without one is judged by its refusal alone.{events}'
EVENTS = ' Any step may also state {{"no_event": [<names>]}}, with names among {names}: a notification of one of these events during the step is a surprise.'
EVIDENCE = "Evidence. Every answer puts rows on the board, and every row has an identifier.{carried} Cite only identifiers that appear on the board; the controller drops any other and says so.{identifiers}"
EVIDENCE_CARRIED = ' A step of {tools} carries "evidence", a list of the identifiers from the board that justify it, the few that moved you, and no other step carries it.'
CANDIDATES = 'Candidates and your answer. A candidate is one possible answer to the task: a value for each argument that states the answer, {candidate_arguments}. List your candidates best first under "candidates", and name the one your plan follows by its rank under "follow". The board keeps your latest list. {commitment}'
NO_CANDIDATES = "Your answer. No tool in this table declares candidate arguments. Give each step its own required arguments. {commitment}"
BOARD = "The board. It shows, in this order: {parts}; and the status, with every limit that is left. Rows that you have planned on may collapse into one summary row, and a collapsed row stays citable by its identifier.\nThe status also states which answers the board holds for the whole episode and which calls were made more than once with the same arguments."
PLAN = 'The plan. Return\n{{"follow": {follow}, "why": "<one line>", "candidates": {candidate_list}, "steps": [{{"tool": "<a tool>", "args": {{...}}{expect}}}, ...]}}\nwith at least one entry under "steps"{count}. A step is one call of one tool with the arguments of its line. "candidates" is optional. The why line names the identifiers that moved you.'
RECALL = 'One more step calls no tool: {{"tool": "recall", "args": {{"observations": [<first>, <last>]{series}}}}}, with no expectation. It costs nothing: the controller puts the full rows of those observations, which the board collapsed into a summary row, back on the next board, once. A plan of recalls alone makes no call and still takes one step.{series_note}'
BATCH = "Consecutive steps of the role observe run as one batch: the controller makes the calls one after the other and checks every expectation after the last, so a batch pays for all of its calls before it sees any of them."
RULES = "The controller refuses, whatever you write, only what the {short_noun} or the plan itself would fail:\n{rules}."
LOOP = "The controller runs your steps while every expectation holds and calls you again when one fails, with the surprise written on the board, and when the steps of your plan have run, so plan several steps and attach an expectation you are willing to be stopped by. The steps of a plan before its first step that the controller cannot use, a step that does not parse or that it refuses, run as you wrote them, and that step and the steps after it do not run and come back to you with the reasons. A reply that does not parse as a plan, a plan whose first step cannot be used, a plan whose {plan_head} the controller refuses, and a plan that {limits} cannot pay run nothing, and you are asked again with the reasons. A turn that calls no tool of the {short_noun} still takes one step."
STANDING = "A step whose effect already stands on the board makes no call and does not stop the plan: its status in your last plan is standing, the steps after it run, and the next request names it with what the board already holds, so leave such a step out. An expectation on such a step is held against the board as it stands, and one that fails is a surprise."
LOOP_NO_EXPECTATIONS = "The controller runs your steps and calls you again when a call is refused, with the refusal written on the board, and when the steps of your plan have run. The steps before the first step that cannot be used run as written; that step and the steps after it do not run and come back with their reasons. A reply that does not parse, a plan whose first step cannot be used, a plan whose {plan_head} the controller refuses, and a plan that {limits} cannot pay run nothing. A turn that calls no tool of the {short_noun} still takes one step."
STANDING_NO_EXPECTATIONS = "A step whose effect already stands on the board makes no call and does not stop the plan: its status in your last plan is standing, the steps after it run, and the next request names it with what the board already holds, so leave such a step out."
BATCH_NO_EXPECTATIONS = "Consecutive steps of the role observe run as one batch: the controller makes the calls one after the other, stopping after a refused call."
PLANNING = "Objective: the task above, within the limits on the board. Return the plan alone, the one JSON object described above, with no text before it and nothing after it."
OPENING = "Before {facts_tool} is answered this plan holds only the opening reads, and the next request asks for the rest."
OPENING_BOARD = "No answer of {facts_tool} yet, so the board holds nothing about the {short_noun}. A plan made before {facts_tool} is answered holds only the opening reads: {opening}. Plan every other step in the next request, once its answer is on the board."
OPENING_CANDIDATES = "Candidates written before the facts were answered were not kept; list them again from the current board."
ORIENTATION = 'No observation has been made yet. Write the differential: for every cause that could explain the task, what you expect to observe if it holds and the identifiers behind it. Set no probability. Return {{"differential": [{{"about": "<a cause, in the words of the tools and the board>", "expected": "<one line>", "cites": ["<identifier>", ...]}}, ...]}} and nothing after it.'
NOT_RUN_HEADING = "### Steps of your previous plan that did not run"
NOT_RUN_END = "The steps before them ran as far as the board shows. Plan from the board as it stands now, as the JSON object only."
STOOD_HEADING = "### Steps of your previous plan that made no call"
STOOD_LINE = "- step {number} ({step_name}): {reason}"
STOOD_END = "Their effect already stood, so the controller made no call for them and went on with the steps after them. Plan from the board as it stands now, without these steps, as the JSON object only."
RETRY = "### Previous reply needs correction\n{reason}\nReturn the JSON object only."
EARLIER = "The board keeps {keeps}. Summarize the earlier turns in plain prose. Keep the identifiers behind the claims and distinguish accepted results from refused calls."
LEFT_OUT = "The oldest replaceable entries were left out of this request to fit the context; summarize only what is shown."
CLOSING = "Write the closing summary of the episode in a few sentences of plain prose: your answer and the evidence behind it, each write and what it returned, what each verification said, and the state of the {short_noun} as the answers show it. Claim only what an answer on the board supports. Return plain text, no JSON."
OUTSIDE = 'expect {{"outside": [<fields>]}}: a listed field that the answer serves inside its normal range is a surprise; a name stands for the field of that name or for every field whose name begins with it and an underscore{example}, and a field that the answer did not serve is not judged'
WORD = 'expect {{"word": <one of its words>}}: holds when the answer holds exactly that one of its words'
NOTHING_NEW = "Where no {time_unit} passes in the episode, a step of this {step_key} on a {series} whose every recorded {time_unit} is on the board makes no call and stands; where one passes, the record grows and the call is made."


def blocks(table, options):
    """Return the core text blocks for the current switch settings."""
    output = [("ROLE", ROLE)]
    if options.expectations:
        output.append(("EXPECTATIONS", EXPECTATIONS))
    output.extend((("EVIDENCE", EVIDENCE), ("CANDIDATES", CANDIDATES if any(row.candidate for row in table.tools.values()) else NO_CANDIDATES),
                   ("BOARD", BOARD), ("PLAN", PLAN)))
    if options.recall:
        output.append(("RECALL", RECALL))
    if options.parallel and table.of_role("observe"):
        output.append(("BATCH", BATCH if options.expectations else BATCH_NO_EXPECTATIONS))
    output.extend((("RULES", RULES), ("LOOP", LOOP if options.expectations else LOOP_NO_EXPECTATIONS),
                   ("STANDING", STANDING if options.expectations else STANDING_NO_EXPECTATIONS)))
    return output

from CausalDiagnosis.agents.utils.core_texts import (
    ROLE, EXPECTATIONS, LOOP,
    STOOD_HEADING, STOOD_LINE, STOOD_CLOSING as STOOD_END,
    NOT_RUN_HEADING, NOT_RUN_CLOSING as NOT_RUN_END,
    LOOP_STANDING as STANDING, OBJECTIVE as PLANNING,
    READ_STANDS as NOTHING_NEW,
)
