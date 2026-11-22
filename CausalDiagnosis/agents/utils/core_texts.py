"""The texts that both agents of the tree print: the CausalDiagnosis agent (prompts.py and its board) and the general agent
(prompt/core_protocol.py of CausalDiagnosis/docs/general-agent-plan.md, section The prompt). Each is a named constant whose
words of a dataset stand in slots, such as {noun}, {series} or {time_unit}, which the printer fills from the asset and the
words of its kind, so that no word of one dataset is typed here. A text that only one agent prints stays in that agent.
The module imports nothing, so that both agent packages can import it."""

# taken word for word from the section The prompt of the plan page
ROLE = "You plan the work on one {noun} for a controller. The controller keeps a working memory, the board, and calls the tools that your plan names, and no other tool of the {short_noun}. The task of the episode stands above the board in every request. Each turn you receive the task and the board, and you return one JSON object with nothing after it."
EXPECTATIONS = 'Expectations. A step may carry an expectation under "expect", a statement about its answer that you are willing to be stopped by. The controller checks it when the answer arrives, and an expectation that fails is a surprise. A call that the {short_noun} refuses is a surprise whatever the step expects. The line of a {step_key} names the expectations its steps can state, and a step without one is judged by its refusal alone.{events}'
BOARD = "The board. It shows, in this order: {parts}; and the status, with every limit that is left. Rows that you have planned on may collapse into one summary row, and a collapsed row stays citable by its identifier."
RULES_HEAD = "The controller refuses, whatever you write, only what the {short_noun} or the plan itself would fail:"
LOOP = "The controller runs your steps while every expectation holds and calls you again when one fails, with the surprise written on the board, and when the steps of your plan have run, so plan several steps and attach an expectation you are willing to be stopped by. The steps of a plan before its first step that the controller cannot use, a step that does not parse or that it refuses, run as you wrote them, and that step and the steps after it do not run and come back to you with the reasons. A reply that does not parse as a plan, a plan whose first step cannot be used, a plan whose {plan_head} the controller refuses, and a plan that {limits} cannot pay run nothing, and you are asked again with the reasons. A turn that calls no tool of the {short_noun} still takes one step."
NOT_RUN_HEADING = "### Steps of your previous plan that did not run"
NOT_RUN_CLOSING = "The steps before them ran as far as the board shows. Plan from the board as it stands now, as the JSON object only."
EARLIER_TURNS_KEEPS = "The board keeps the task, the public facts, what the knowledge tools returned, the open work, the advice, the refuted repairs, the latest observation of every series and the status."

# the sentences on a step whose effect already stands, its reasons, the status lines of the board and the mark of a note
# that repeats the reasons of earlier notes; a reason opens with the change to make, then names what the board holds
EVIDENCE = "Evidence. Every answer puts rows on the board, and every row has an identifier.{carried} Cite only identifiers that appear on the board; the controller drops any other and says so."
EVIDENCE_CARRIED = 'A step of {evidence_steps} carries "evidence", a list of the identifiers from the board that justify it, the few that moved you, and no other step carries it.'
OBJECTIVE = "Objective: the task above, within the limits on the board. Return the plan alone, the one JSON object described above, with no text before it and nothing after it."
LOOP_STANDING = "A step whose effect already stands on the board makes no call and does not stop the plan: its status in your last plan is standing, the steps after it run, and the next request names it with what the board already holds, so leave such a step out. An expectation on such a step is held against the board as it stands, and one that fails is a surprise."
READ_STANDS = "Where no {time_unit} passes in the episode, or where time passes only inside calls that take time and none came since the latest answer, a step of this {step_key} on a {series} whose every recorded {time_unit} is on the board makes no call and stands; elsewhere the record may have grown, and the call is made."
BOARD_HELD = "The status also states which answers the board holds for the whole episode and which calls were made more than once with the same arguments."
BOARD_HELD_RECORD = "For every {series} with a record, it states how much of the record the board holds."
STOOD_HEADING = "### Steps of your previous plan that made no call"
STOOD_LINE = "- step {number} ({step_name}): {reason}"
STOOD_CLOSING = "Their effect already stood, so the controller made no call for them and went on with the steps after them. Plan from the board as it stands now, without these steps, as the JSON object only."
STANDS_EPISODE = "leave this step out, since the board already holds the answer of {call}, which holds for the whole episode"
STANDS_SURFACE = "leave this step out, since the board already holds the answer of {call} and no time passes in this episode, so the answer cannot change"
STANDS_NOTHING_NEW = "leave this step out, since the board already holds every recorded {time_unit} of {series} {name} and no {time_unit} passes in this episode, so the call would return nothing new"
STANDS_NOTHING_RECORDED = "leave this step out, since the board already holds every recorded {time_unit} of {series} {name} and on this row the record grows only inside calls that take time, none of which came since the latest answer, so the call would return nothing new"
STANDS_COLLAPSED = "; its rows of observations {first} to {last} stand collapsed in a summary row, and a {recall_step} step brings them back"
STANDS_COMMITMENT = "leave this step out, since the board already holds this commitment ({commitment}) and the environment takes one"
STANDS_EXPECTATION = "{stands}; its expectation, held against the board as it stands, fails ({detail}), and no call of this step can change that, so state an expectation only on a step whose answer is new"
SECOND_COMMITMENT = "leave this step out, since another commitment already stands ({commitment}) and the environment takes one"
OPEN_REPAIR = "leave this step out, since the {repair} on the {target} {name} that an earlier call began is still open at this step and the environment takes no second one on it"
OPEN_REPAIR_UNTIL = "; it closes at {time_unit} {close}"
NOTHING_NEW_AFTER_CALL = "the call returned no {time_unit} that the board does not hold, since the record ends at {time_unit} {last}; plan this step again once the record has grown, or leave it out"
HELD_EPISODE = "On the board for the whole episode: {calls}."
HELD_SURFACE = "Answered, and unchanged since no time passes in this episode: {calls}."
HELD_HEAD = "What the record holds and what the board holds of it:"
HELD_SERIES = "- {series} {name}: recorded {time_unit} {first} to {last}; on the board {held} of {recorded}."
HELD_FIXED = "No {time_unit} passes in this episode, so the record stays as it is."
REPEATED = "Called more than once with the same arguments: {calls}."
REPEATED_ITEM = "{call} {count} times"
NOTE_REPEATED = " (seen before; earlier turns with the same reasons: {count})"
