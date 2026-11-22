"""The prompts of the CausalDiagnosis agent: one system prompt per mode and the builders of the three user turns, orientation,
planning and summary (CausalDiagnosis/docs/experiments.md, section The driver, agent causal).

One system prompt serves every guidance level and every model, since the design (preet2026simplicity, reddy2026hidden)
fixes one interface and one parser, and the prompt without the diagnostics server shares its role, its opening and the loop
of the controller and differs where the server's ranking, advice and predictions are missing. The system prompt
states the role, the step kinds with their arguments and expectations, the opening the model plans, the plan
schema, the rule that a step whose call records evidence cites evidence identifiers and every claim names a failure mode or a
timeline fact, and what the controller refuses whatever the model writes. The sentences that the general agent prints as well
come from the shared file agents/utils/core_texts.py, with the words of this agent in their slots. The evidence guard and the cost rule
are advice the board states: under the cost choice the prompt says the advice weighs the act now cost of a branch, and a
diagnosis expects the candidate on top rather than the one it submits, since a plan may act below rank 1; the rank choice
keeps the prompts whose advice names rank 1. The user turns carry the data: the question, the profile and the failure modes for the
orientation, the rendered board for a plan, the writes and the verdicts for the closing summary. The backend takes one string,
so a turn is the system prompt followed by the user turn.
"""
import re

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import (confirmation_text, durations_of, fmt, n_cycles, order_prices, read_prices,
                                                                          scored_combination_text, table)
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import ARMS
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.plan_schema import COMMIT_KINDS, EVIDENCE_KINDS, LARGEST_INTEGER
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board.board import KIND_WORDS
from CausalDiagnosis.agents.utils import core_texts
from CausalDiagnosis.agents.utils.core_texts import NOT_RUN_CLOSING, NOT_RUN_HEADING, OBJECTIVE, STOOD_CLOSING, STOOD_HEADING, STOOD_LINE
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import RigSurface

PLAN_LENGTH_MARK = "<PLAN_LENGTH>"
DIAGNOSE_TOP_RANK_MARK = "<DIAGNOSE_TOP_RANK>"
ADVICE_RULES_MARK = "<ADVICE_RULES>"  # what the advice names under the read choice and the repair choice of the run
ARM_MARK = "<ARM>"  # inside READ_ARM_RULE, the read of the arm
OBS_FORM_MARK, TOOLS_MARK, CHANGES_MARK = "<OBS_FORM>", "<TOOLS>", "<CHANGES>"  # what the surface the agent runs on fills in (tools/)
NOUN_MARK, EXAMPLE_TAG_MARK = "<NOUN>", "<EXAMPLE_TAG>"  # the noun of the asset and an example channel tag, from the surface
STATISTIC_MARK = "<STATISTIC>"  # the plain statistic of the served values, from the surface
# the kinds of the row whose call records evidence and the commitments it serves, from the surface (system_prompt)
EVIDENCE_STEPS_MARK, COMMITMENTS_MARK = "<EVIDENCE_STEPS>", "<COMMITMENTS>"
ORIENTATION_HEADING = "## Orientation"
PLANNING_HEADING = "## Planning turn"
SUMMARY_HEADING = "## Closing summary"
EARLIER_TURNS_HEADING = "## Summary of your earlier turns"  # the request of step 3 of the compaction
EARLIER_TURNS_ASK = ("You are writing a summary of your own earlier planning turns, which will leave the board. " + core_texts.EARLIER_TURNS_KEEPS
                     + " Write what those do not say: the progress so far and the decisions you made "
                     "with their reasons, the evidence identifiers that moved you and why, the candidates you set aside and why, and what "
                     "remains to be done. Write no probability of your own. Return the summary as plain text, with no JSON and no plan.")
LEFT_OUT = "The oldest entries of the board were left out of this request to make it fit, and they will not return."
RETRY_HEADING = "### Your previous reply was not usable"
# the sentence of the planning turn before the profile is read, which says that a second request follows the opening
OPENING_TURN = "Before the profile is read this plan holds only the reads of the opening, and the next request asks for the rest. "
PARALLEL_MARK = "<PARALLEL>"  # at the end of the read line: the batch of read steps and the schedule, empty with the switch off
# the size of a plan in both system prompts: the first sentence with a count of steps (--causal-plan-length), the second without one,
# where the budget, the horizon and the step cap bound a plan (plan.check_plan)
COUNTED_STEPS = 'with at least one and at most <PLAN_LENGTH> entries under steps, finish included, and candidates best first; a longer plan is refused and you are asked again.'
PAID_STEPS = ("with at least one entry under steps and as many as the budget, the horizon and the step cap pay, finish included, and "
              "candidates best first; a plan they cannot pay is refused and you are asked again.")
# the largest number of cycles, rank and branch that the parser reads (reply_parser._count), in both system prompts after the size of a plan
INTEGERS = f"Every number of cycles, rank and branch is an integer of at most {LARGEST_INTEGER}."

# the words of this agent in the slots of the shared texts: a step names its call by its kind, the check judges the branch, the
# mode and the candidates of a plan as a whole, the rows set the budget, the horizon and the step cap, and profile reads the facts
STEP_KEY, PLAN_HEAD, LIMITS, STEP_LIMIT, FACTS_STEP = "kind", "branch, mode or candidates", "the budget, the horizon or the step cap", "step cap", "profile"
# the clauses of {parts} of the shared BOARD, in the order of the plan page, every one of which this agent's board holds
BOARD_PARTS = ("what holds for the whole episode, which is the public facts once {facts_tool} is answered and what the knowledge tools returned",
               "your summary of earlier turns, when one was written", "the observations with their identifiers", "your candidates",
               "the open work on the {short_noun}", "the surprises", "your last plan with the state of each step")
COMMITMENT_NOUNS = {"diagnose": "diagnosis", "declare": "declaration"}  # the noun of each commit kind in the refusal of a second commitment
RULE_SECOND_COMMITMENT = "no {commitments} that differs from the commitment that stands;"
EVIDENCE_RANKED = "Never copy the ranking's evidence list, which the controller cites for you."
LOOP_TAIL = ("A differential that the controller cannot parse is asked for again with the reasons, and as many requests in a row without a call as "
             "the {step_limit} allows end the episode.")

ROLE = core_texts.ROLE.format(noun=NOUN_MARK, short_noun=NOUN_MARK)
# the rules of the row, the one commitment and the ends of an episode, printed after ROLE
ROW_RULES = """The asset runs under the rules the board states: a reading has the price and the duration of its row, downtime accrues while a component is degraded where the row charges it, and a technician executes work orders after the lead time of its row, where the row sets one. Exactly one diagnosis or one declaration of insufficient evidence is allowed per episode, a work order needs a prior diagnosis or declaration, and the episode ends with finish or when a limit of its row ends it: the horizon, the step cap, the budget, the limit of wall time or a component failure, each where the row sets it."""
# the status lines of this agent's board on the work orders (situation_board/board.py, _order_lines)
BOARD_ORDERS = ("Where a clock runs, the status also states the open work orders on schedule with their closes and how time passes on the row, the read "
                "that verifies a repair whose orders have closed, and, once every repair is verified, that a finish step ends the episode.")
BOARD = (core_texts.BOARD.format(parts="; ".join(BOARD_PARTS).format(facts_tool=FACTS_STEP, short_noun=NOUN_MARK)) + " " + core_texts.BOARD_HELD
         + " " + core_texts.BOARD_HELD_RECORD.format(**KIND_WORDS) + " " + BOARD_ORDERS)
EXPECTATIONS = core_texts.EXPECTATIONS.format(short_noun=NOUN_MARK, step_key=STEP_KEY, events="")  # a wait states its own expectation in its line
EVIDENCE = core_texts.EVIDENCE.format(carried=" " + core_texts.EVIDENCE_CARRIED.format(evidence_steps=EVIDENCE_STEPS_MARK))
READ_STANDS = core_texts.READ_STANDS.format(step_key=STEP_KEY, **KIND_WORDS)
RULES_HEAD = core_texts.RULES_HEAD.format(short_noun=NOUN_MARK)
OPENING_STEPS = """Step kinds. Three kinds read the opening, each free. profile, no args, reads the asset profile and comes first, since the board holds nothing about the asset before it. failure_modes, args {"component": <a component, or "" for every component>}, reads the failure modes the asset serves. fix_history, no args, reads the repairs recorded before the episode. Five kinds act on the <NOUN>: read, diagnose, declare, order and wait. A sixth, finish, ends the episode, and a seventh, recall, brings collapsed readings back to the board."""
# the recall step in the step kinds, its line and its refusal, which the switch recall off takes out (without_recall)
RECALL_CLAUSE = ", and a seventh, recall, brings collapsed readings back to the board"
RECALL_REFUSAL = " no recall of observations the board did not collapse;"
RECALL_LINE = ('- recall: args {"sensor_group": <a group>, "observations": [<first>, <last>]}; no expectation. It calls no tool and costs '
               "nothing: the controller puts the full rows of that group's observations first to last, which the board collapsed into a "
               "summary row, back on the next board, once. A plan of recalls alone makes no call and still takes one step.")
FINISH_STEP = "- finish: no args and no expectation; it is the last step of the plan. The controller writes the closing summary and ends the episode.\n" + RECALL_LINE
LOOP = (core_texts.LOOP.format(plan_head=PLAN_HEAD, limits=LIMITS, short_noun=NOUN_MARK) + " " + core_texts.LOOP_STANDING + " "
        + LOOP_TAIL.format(step_limit=STEP_LIMIT))

SYSTEM_PROMPT = ROLE + " " + ROW_RULES + """

""" + BOARD + """ Before the profile is read it holds nothing about the <NOUN> and names the reads of the opening. Evidence rows carry the identifiers <OBS_FORM>; the public normal range of every channel <STATISTIC> stands once in the legend above the evidence rows, a value outside it ends in low or high, and rows a ranking has absorbed collapse to one summary row per sensor group with the identifiers outside their normal range. The fix history lists the repairs recorded before the episode. The failure modes table lists the failure modes the asset serves with their identifiers fmea/<source>/<code>, and their level, severity, recommended action and indicating channels where this row gives them. Channels outside their normal range shows, for every sensor group read, its latest cycle with the channel <STATISTIC>s outside their normal ranges, below or above. The ranking lists the candidate combinations now with their failure mode codes and posterior mass, the effective number of hypotheses of the belief, and the decision fields of the rule: read_first, the separating test with its price and expected loss, the expected loss of acting now. The advice of the CausalDiagnosis MCP gives, per sensor group, the information gain of one cycle of it and that gain per unit cost, which ranks the reads, with the expected number of hypotheses the cycle eliminates beside them, the server's read batch, the reads best first by gain per unit cost whose choice no outcome can change, and its lines of advice: the reads to take while the ranking says read first, the repair plan to act on and, after a refuted plan, the other plan. It is advice: you decide every read, diagnosis and order, and the controller makes only those your plan names. A troubleshooting plan table, when present, walks each candidate forward: its confirm test (the sensor group that separates it best, with the expected mass of the candidate if it is true and if it is false), its repair plan (one work order per faulty component, created one after another as one technician visit with one visit charge, and the one prediction identifier that every order of the plan cites, the state after all of its replacements), the cycles to a restored <NOUN>, the predicted recovery if acted, the expected loss if acted now and if tested first, and the cost to a restored <NOUN> both ways. Refuted repair plans list the component sets whose verification failed, and the advice keeps the last of them out until a later ranking names it first. Your candidates lists the combinations of your latest plan that listed them, in your order, with the ranks after the ranked candidates. The timeline lists alarms (alarm/<t>/<channel>) and closed work orders (wo/<id>).

""" + OPENING_STEPS + """
- read: args {"sensor_group": <group>, "cycles": <n, at least 1>}; expect {"top_rank": <rank in the current ranking>, "top_mass_at_least": <mass>}. After the reading, the faulty set of the candidate at that rank should be the set on top of the new ranking, with at least that mass summed over every combination of the set. """ + READ_STANDS + """<PARALLEL>
- diagnose: args {"rank": <rank>}; expect {"top_rank": <DIAGNOSE_TOP_RANK>}. The controller submits the current combination of that candidate, scored at the cycle of the write, with the evidence you cite and the evidence the ranking names for it. A rank of your candidates submits that combination, the components it leaves out at their healthy level, with the evidence you cite and the channel <STATISTIC>s outside their normal ranges.
- declare: args {"sensor_group": <group>, "cycles": <n, at least 1>}; no expectation. The advice names the separating test of the ranking or the confirm test of the branch you follow. It must follow a reading, which the controller cites with the evidence you cite and the evidence the ranking names, since a declaration that cites none is scored as a failure.
- order: args {"component": <one component>, "prediction_id": <a prediction identifier from the branch table or the prediction list, or "estimate" to have the controller call estimate_intervention_effect for replace:<component> first>}; expect {"prediction_id": <the same>, "recovery_at_least": <probability>}. The controller places the order after a diagnosis or a declaration only, and an identifier the board does not know gets an estimate first. One component per order step: a step that joins several names is split into one order per component, which count against the step limit.
- wait: args {"cycles": <n, at least 1; 0 is read as 1>}; expect {"no_new_alarm": true}. The controller waits event by event (wait_for_notification with the cycles as the timeout, so the first notification ends each wait) and records every notification<CHANGES>. Once the last order citing a prediction has closed, plan a read, best of a group whose channels indicate the repaired components: on the first reading at or after the close the controller verifies that prediction with verify_outcome. A plan whose verification finds the reading inconsistent with its prediction is refuted: the controller records its component set and marks it on the board, and the advice names the cheapest other previewed branch. A consistent verified verdict stands whatever the ranking after that reading shows. The verdict also says whether the belief after that reading confirms the prediction, and names the belief's most probable combination with its mass when it does not; after a later reading the controller asks again, and the latest verdict counts. The status says that a finish step ends the episode only once the latest repair is confirmed, and a finish is never held back for it.
""" + FINISH_STEP + """

""" + EXPECTATIONS + """

""" + EVIDENCE + " " + EVIDENCE_RANKED + """<TOOLS>

The plan. Return
{"why": "<one line>", "branch": <rank of the candidate you follow>, "mode": "test_first" | "act_now" | "repair_as_test" | "declare", "candidates": [{"levels": {"<component>": <level>, ...}, "failure_mode_ids": ["fmea/<source>/<code>", ...]}, ...], "steps": [{"kind": ..., "args": {...}, "expect": {...}}, ...]}
with at least one and at most <PLAN_LENGTH> entries under steps, finish included, and candidates best first; a longer plan is refused and you are asked again. """ + INTEGERS + """ candidates is optional and lists combinations of your own: a candidate names the faulty components at levels of the profile, every other component staying at its healthy level, and your candidates take the ranks after the ranked candidates, in your order, so that a diagnosis may name one of them. A plan made before the profile is read lists no candidates, since a candidate names levels of the profile, and the controller keeps none that such a plan lists. test_first reads the confirm test or the separating test and commits when the expected mass arrives; act_now diagnoses and orders the branch's repair plan, then waits to the close and reads so that the controller verifies; repair_as_test orders the cheapest repair of a high mass branch when the confirm test is unaffordable or slower than the downtime it would save; declare names the separating test when no branch can be separated within the budget. The why line names the failure mode identifiers or the timeline facts that moved you beyond the numbers, and says whether the faulty set of the branch reads as the cause or as a symptom of what the evidence shows. You never write a probability of your own: masses, intervals and recoveries come from the board.

""" + RULES_HEAD + """ before the profile is read, a step that is neither a read of the opening nor a finish, or a failure_modes step before the profile step; no step that names a component or a sensor group that the profile does not hold, and no step or branch with a rank outside the candidates shown; no candidate of yours with a component or a level that the profile does not hold; no step of a kind this row does not serve or does not need, and no reading before the steps this row needs first; """ + RULE_SECOND_COMMITMENT.format(commitments=COMMITMENTS_MARK) + """ no work order on a component with an open order; no work order without a prior diagnosis or declaration; no declaration before a reading it can cite; no wait where no clock runs; no recall of observations the board did not collapse; no finish before the close of every work order and the reading that verifies the last order, unless the budget, the horizon or the step cap leaves no room for it; no plan that the budget, the horizon or the step cap cannot pay. Before a diagnosis or a declaration it ranks again on the current evidence. The advice asks you to read the advised reads, the separating test or the followed branch's confirm test before a diagnosis while the ranking says read first and such a read fits.<ADVICE_RULES> """ + LOOP

SYSTEM_PROMPT_WITHOUT_SERVER = ROLE + " " + ROW_RULES + """

""" + BOARD + """ Before the profile is read it holds nothing about the <NOUN> and names the reads of the opening. Evidence rows carry the identifiers <OBS_FORM>; the public normal range of every channel <STATISTIC> stands once in the legend above the evidence rows, a value outside it ends in low or high, and rows you have planned on collapse to one summary row per sensor group with the identifiers outside their normal range. A normal range spans the smallest to the largest <STATISTIC> of the channel over the stable recordings of the healthy <NOUN>, so a healthy <NOUN> shows no channel <STATISTIC> outside its normal range, while a fault usually shows at least one. The fix history lists the repairs recorded before the episode. The CausalDiagnosis MCP does not run on this row, so no ranking, advice or prediction exists: you form the candidates yourself. The failure modes table lists the failure modes the asset serves with their identifiers fmea/<source>/<code>, and their level, severity, recommended action and indicating channels where this row gives them. Channels outside their normal range shows, for every sensor group read, its latest cycle with the channel <STATISTIC>s outside their normal ranges, below or above. Your candidates lists the combinations of your latest plan that listed them, in your order. Refuted repair plans list the component sets whose verification on the normal ranges failed. The timeline lists alarms (alarm/<t>/<channel>) and closed work orders (wo/<id>).

""" + OPENING_STEPS + """
- read: args {"sensor_group": <group>, "cycles": <n, at least 1>}; expect {"outside": [<the channel <STATISTIC>s your followed candidate pushes outside their normal ranges, each by its field name or its tag<EXAMPLE_TAG>>]}. After the reading, a listed channel that the reading served within its normal range is a surprise. """ + READ_STANDS + """<PARALLEL>
- diagnose: args {"rank": <rank in your candidates>}; no expectation. The controller submits the full combination of that candidate, the components it leaves out at their healthy level, with the evidence you cite and the channel <STATISTIC>s outside their normal ranges.
- declare: args {"sensor_group": <the group you would read next to separate your candidates>, "cycles": <n, at least 1>}; no expectation. It must follow a reading, which the controller cites, since a declaration that cites none is scored as a failure.
- order: args {"component": <one component>}; no expectation, since no prediction exists on this row. The controller places the order after a diagnosis or a declaration only. One component per order step: a step that joins several names is split into one order per component, which count against the step limit.
- wait: args {"cycles": <n, at least 1; 0 is read as 1>}; expect {"no_new_alarm": true}. The controller waits event by event (wait_for_notification with the cycles as the timeout, so the first notification ends each wait) and records every notification<CHANGES>. The orders of one plan form one repair plan: once its last order has closed and no other order of yours is open, plan a read, and on the first reading at or after the close the controller verifies the repair plan on the normal ranges, which judge the whole <NOUN>. It is refuted when any channel <STATISTIC> of that reading lies outside its normal range: the controller records its component set and marks it on the board, and you repair another candidate.
""" + FINISH_STEP + """

""" + EXPECTATIONS + """

""" + EVIDENCE + """<TOOLS>

The plan. Return
{"why": "<one line>", "branch": <rank of the candidate you follow in your candidates>, "mode": "test_first" | "act_now" | "repair_as_test" | "declare", "candidates": [{"levels": {"<component>": <level>, ...}, "failure_mode_ids": ["fmea/<source>/<code>", ...]}, ...], "steps": [{"kind": ..., "args": {...}, "expect": {...}}, ...]}
with at least one and at most <PLAN_LENGTH> entries under steps, finish included, and candidates best first; a longer plan is refused and you are asked again. """ + INTEGERS + """ A candidate names the faulty components at levels of the profile, every other component staying at its healthy level, and a candidate with no faulty component states a healthy unit. A plan made before the profile is read lists no candidates, since a candidate names levels of the profile, and the controller keeps none that such a plan lists. candidates is needed before your first diagnosis and may be left out later, when the board keeps your last list. test_first reads to separate your candidates and commits when the readings support one; act_now diagnoses and orders the faulty components of the candidate you follow, then waits to the close and reads so that the controller verifies; repair_as_test orders the cheapest repair of a likely candidate when no affordable reading separates it; declare names a sensor group when no reading you can afford would separate your candidates. The why line names the failure mode identifiers or the readings that moved you. You never write a probability, since no mass exists on this row.

""" + RULES_HEAD + """ before the profile is read, a step that is neither a read of the opening nor a finish, or a failure_modes step before the profile step; no step that names a component or a sensor group that the profile does not hold, and no step or branch with a rank outside your candidates; no candidate of yours with a component or a level that the profile does not hold; no step of a kind this row does not serve or does not need, and no reading before the steps this row needs first; no diagnosis without your candidates; """ + RULE_SECOND_COMMITMENT.format(commitments=COMMITMENTS_MARK) + """ no work order on a component with an open order; no work order without a prior diagnosis or declaration; no declaration before a reading it can cite; no wait where no clock runs; no recall of observations the board did not collapse; no finish before the close of every work order and the reading that verifies the last repair plan, unless the budget, the horizon or the step cap leaves no room for it; no plan that the budget, the horizon or the step cap cannot pay. """ + LOOP


# what the advice names (plan.follows_advice): the step the server advises, and under an arm of Ag7 the arm's own read; the board
# quotes the server's advice
READ_RULE = " The advised reads are the read the server advises, a sensor group for its cycles."
READ_ARM_RULE = " The advised reads of this run are <ARM> and not the read the server advises."
REPAIR_RULE = " The advised repair plan is the rank the server advises to diagnose, with one work order for each of its faulty components."
SERVER_ADVICE_RULE = " The board quotes the advice of the server as one line, the step it advises before a diagnosis."
DIAGNOSE_TOP_RANK_V6 = "<the rank of the candidate on top when the diagnosis is made, 1 unless a read before it moves another candidate up>"
# the sentences of the mark with the switch on (candidate_plan_schedule.py), for the prompt with the diagnostics server and for the one without it
PARALLEL_SERVER = (" Consecutive read steps of one plan run as one batch: the controller makes the reads one after the other, ranks once after the last "
                   "and checks every expectation against that ranking, so a batch pays for all of its reads before it sees any of them. The schedule "
                   "on the board, when one is shown, lists the reads that the server names and the reads that several troubleshooting plans share, "
                   "each once with the plans it serves, and after a refuted plan what each remaining plan still needs.")
PARALLEL_WITHOUT_SERVER = (" Consecutive read steps of one plan run as one batch: the controller makes the reads one after the other and checks every "
                           "expectation after the last, so a batch pays for all of its reads before it sees any of them. The schedule on the board, when "
                           "one is shown, lists the reads that several of your candidates share, each once with the candidates it serves, and after a "
                           "refuted plan what each remaining candidate still needs.")


# the sentences that name a kind the row may not serve, with their form on such a row (tools/, Surface.unserved_kinds); the
# step line of the kind goes whole, and the sentence of the kinds that act names those served. Nothing changes on a row that serves them.
WITHOUT_KIND = {
    "declare": (
        ("Exactly one diagnosis or one declaration of insufficient evidence is allowed per episode, a work order needs a prior diagnosis or declaration,",
         "Exactly one diagnosis is allowed per episode, a work order needs a prior diagnosis,"),
        ("after a diagnosis or a declaration only", "after a diagnosis only"),
        (' | "declare"', ""),
        ("; declare names the separating test when no branch can be separated within the budget.", "."),
        ("; declare names a sensor group when no reading you can afford would separate your candidates.", "."),
        ("no work order without a prior diagnosis or declaration; no declaration before a reading it can cite;", "no work order without a prior diagnosis;"),
        ("Before a diagnosis or a declaration it ranks again", "Before a diagnosis it ranks again"),
    ),
    "wait": (
        ("then waits to the close and reads so that", "then reads after the close so that"),
        ("; no wait where no clock runs;", ";"),
    ),
}
ACTING_KINDS = ("read", "diagnose", "declare", "order", "wait")  # the kinds that act on the asset, in the order the prompts name them
ACTING_SENTENCE = "Five kinds act on the <NOUN>: read, diagnose, declare, order and wait."
COUNT_WORDS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve")  # a larger count is printed in digits, so no count of kinds fails
# the head of the wait line, which a row without a wait drops, keeping the sentences on the verifying read that follow it
WAIT_HEAD = re.compile(r"^- wait: .*?records every notification<CHANGES>\. ", re.MULTILINE)


def acting_sentence(unserved):
    """The sentence of the kinds that act on the asset, naming only those the row serves."""
    kinds = [k for k in ACTING_KINDS if k not in unserved]
    n = len(kinds)
    if not n:
        return "No kind acts on the <NOUN>."
    count = COUNT_WORDS[n] if n < len(COUNT_WORDS) else str(n)
    if n == 1:
        return f"{count} kind acts on the <NOUN>: {kinds[0]}."
    return f"{count} kinds act on the <NOUN>: {', '.join(kinds[:-1])} and {kinds[-1]}."


def _either(words):
    """Words joined with commas and a last or."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def without_kinds(text, unserved):
    """A prompt with every line and sentence on the kinds the row does not serve taken out or reworded; the text itself where it
    serves them all."""
    if not unserved:
        return text
    text = text.replace(ACTING_SENTENCE, acting_sentence(unserved))
    if "wait" in unserved:
        text = WAIT_HEAD.sub("", text)
    text = "\n".join(line for line in text.split("\n") if not any(line.startswith(f"- {kind}:") for kind in unserved))
    for kind in sorted(unserved):
        for old, new in WITHOUT_KIND.get(kind, ()):
            text = text.replace(old, new)
    return text


def advice_rules(reads="gain"):
    """The sentences on what the advice names under the read choice of a run, the server's step under its default and the arm's
    own choice under an arm, the repair plan the server advises, then the line of the board that quotes the advice of the server."""
    read = READ_RULE if reads == "gain" else READ_ARM_RULE.replace(ARM_MARK, ARMS[reads])
    return read + REPAIR_RULE + SERVER_ADVICE_RULE


def without_recall(text):
    """A prompt without the recall step: its clause in the step kinds, its step line and its refusal (the switch recall off)."""
    return text.replace(RECALL_CLAUSE, "").replace("\n" + RECALL_LINE, "").replace(RECALL_REFUSAL, "")


def system_prompt(plan_length, surface=None, server=True, reads="gain", parallel=True, recall=True):
    """The system prompt for a plan length and the read choice of the run, the surface the agent runs on, which
    gives the identifiers, the tools, the noun of the asset, an example channel and the plain statistic of the served values, and
    whether the diagnostics server runs;
    without a surface the hydraulic rig's action server. With parallel, consecutive read steps of one plan run as one batch."""
    surface = surface or RigSurface()
    text = SYSTEM_PROMPT_WITHOUT_SERVER if not server else (
        SYSTEM_PROMPT.replace(DIAGNOSE_TOP_RANK_MARK, DIAGNOSE_TOP_RANK_V6)
        .replace(ADVICE_RULES_MARK, advice_rules(reads)))
    tag = f" such as {surface.example_tag}" if surface.example_tag else ""
    unserved = surface.unserved_kinds()
    text = (without_kinds(text, unserved).replace(EVIDENCE_STEPS_MARK, f"the {STEP_KEY} " + _either([k for k in EVIDENCE_KINDS if k not in unserved]))
            .replace(COMMITMENTS_MARK, _either([COMMITMENT_NOUNS[k] for k in COMMIT_KINDS if k not in unserved])))
    if not recall:
        text = without_recall(text)
    if plan_length is None:
        text = text.replace(COUNTED_STEPS, PAID_STEPS)
    return (text.replace(PLAN_LENGTH_MARK, str(plan_length)).replace(OBS_FORM_MARK, surface.evidence_form)
            .replace(CHANGES_MARK, surface.prompt_changes).replace(TOOLS_MARK, surface.prompt_tools)
            .replace(PARALLEL_MARK, (PARALLEL_SERVER if server else PARALLEL_WITHOUT_SERVER) if parallel else "")
            .replace(NOUN_MARK, surface.noun).replace(EXAMPLE_TAG_MARK, tag).replace(STATISTIC_MARK, surface.statistic))


def _profile_text(profile):
    components = profile["components"]
    prices, orders, durations = read_prices(profile), order_prices(profile), durations_of(profile)
    charge, visit = (profile.get("prices") or {}).get("repair_charge", 0) or 0, (profile.get("prices") or {}).get("visit", 0) or 0
    rate = profile.get("downtime_rate") or {}
    lines = [
        f"Asset {profile['asset_id']}: components {', '.join(components)}.",
        "Condition levels: " + "; ".join(f"{c} {', '.join(map(str, profile['levels'][c]))} (healthy {profile['healthy'][c]})" for c in components) + ".",
        "Sensor groups and read price per cycle: " + "; ".join(f"{g} ({fmt(prices[g])}): {', '.join(fields)}"
                                                               for g, fields in profile["sensor_groups"].items()) + ".",
        "Work order price (part plus repair charge " + fmt(charge) + "): " + ", ".join(f"{c} {fmt(orders[c])}" for c in components)
        + f", plus one visit charge of {fmt(visit)} for the orders created one after another with no other call between them. Lead times: " + ", ".join(f"{c} {durations['lead_time'][c]}" for c in components)
        + ". Repair downtimes: " + ", ".join(f"{c} {durations['repair_downtime'][c]}" for c in components) + ".",
    ]
    if rate:
        lines.append("Downtime per cycle by level: " + "; ".join(f"{c} " + ", ".join(f"level {lv} {fmt(r)}" for lv, r in rate[c].items()) for c in components if c in rate)
                     + (f"; stopped for a repair {fmt(rate['stopped'])}." if "stopped" in rate else "."))
    # the base row prices nothing, runs no clock and sets no budget, so each limit is named only when the manifest sets it
    budget = f"Budget {fmt(profile['budget'])}" if profile.get("budget") is not None else "Budget unlimited (every tool is free)"
    horizon = f"horizon {profile['horizon']} cycles" if profile.get("horizon") is not None else "no horizon (no clock runs)"
    cap = f"step cap {profile['max_steps']} tool calls" if profile.get("max_steps") is not None else "no step cap"
    clock = f"clock mode {profile['clock_mode']}" if profile.get("clock_mode") else "no clock"
    lines.append(f"{budget}, {horizon}, {cap}, {clock}.")
    return "\n".join(lines)


def _failure_modes_text(modes):
    rows = [(m["component"], m["level"], m["code"], m["severity"], m["action"], ", ".join(m.get("channels") or []), m.get("evidence_id"))
            for m in modes]
    return table(["component", "level", "code", "severity", "recommended action", "indicating channels", "identifier"], rows)


def orientation_turn(question, profile, failure_modes, plan_length, retry=None, surface=None, server=True, reads="gain",
                     parallel=True, recall=True):
    """The orientation, once the opening put the profile and the failure modes on the board: the profile and the failure modes in, a
    differential out. The model sets no prior; the differential is scored for coverage and shown on every board."""
    text = (f"{system_prompt(plan_length, surface, server, reads, parallel, recall)}\n\n{ORIENTATION_HEADING}\n\nQuestion of the episode: {question}\n\n{_profile_text(profile)}\n\n"
            f"Failure modes served by the asset:\n{_failure_modes_text(failure_modes)}\n\n"
            "No reading has been taken yet. Write the differential: for every component that could be faulty, the symptoms its failure "
            f"modes lead you to expect in the channel {(surface or RigSurface()).statistic}s (which channels would leave their normal range) and the failure mode identifiers "
            "behind them. Set no probability. Return {\"differential\": [{\"component\": \"<component>\", \"expected_symptoms\": \"<one line>\", "
            "\"failure_mode_ids\": [\"fmea/<source>/<code>\", ...]}, ...]} and nothing after it.")
    if retry:
        text += f"\n\n{RETRY_HEADING}\n{retry}\nReturn the JSON object only."
    return text


def planning_turn(board, plan_length, retry=None, surface=None, not_run=None, stood=None):
    """The planning turn: the rendered board in, one plan out. retry carries the parse error or the reasons the plan check
    refused the previous plan, quoted so the model can repair it in the next turn; not_run names the steps of the previous plan
    that did not run, from its first step that the controller could not use, with the reasons; stood lists the steps of the
    previous plan that made no call, since their effect already stood, as (number, name, reason)."""
    text = (f"{system_prompt(plan_length, surface, board.server, board.read_advice, board.parallel, board.recall_offered)}\n\n{PLANNING_HEADING}\n\n"
            f"Question of the episode: {board.question}\n\n{board.render()}\n\n{OBJECTIVE}"
            + (f" {OPENING_TURN.rstrip()}" if board.profile is None else ""))
    if retry:
        text += f"\n\n{RETRY_HEADING}\n{retry}\nReturn a plan that passes, as the JSON object only."
    if not_run:
        text += f"\n\n{NOT_RUN_HEADING}\n{not_run}\n{NOT_RUN_CLOSING}"
    if stood:
        lines = "\n".join(STOOD_LINE.format(number=number, step_name=name, reason=reason) for number, name, reason in stood)
        text += f"\n\n{STOOD_HEADING}\n{lines}\n{STOOD_CLOSING}"
    return text


def earlier_turns_turn(board, plan_length, left_out=0, surface=None):
    """The request of step 3 of the compaction: the board, from which the caller has left out the
    oldest replaceable entries when it had to, and the ask for a summary of the earlier turns; left_out says how many it left out."""
    return (f"{system_prompt(plan_length, surface, board.server, board.read_advice, board.parallel, board.recall_offered)}\n\n{EARLIER_TURNS_HEADING}\n\n"
            f"Question of the episode: {board.question}\n\n{board.render()}\n\n{EARLIER_TURNS_ASK}" + (f" {LEFT_OUT}" if left_out else ""))


def summary_turn(board, plan_length, note=None, surface=None):
    """The closing summary: the writes and the verdicts in, a few sentences of prose out. The summary is never scored, but
    the verifier's contradiction heuristic reads it, so the turn asks for claims the verdicts support."""
    writes = []
    if board.diagnosis is not None:
        writes.append("Diagnosis submitted: " + ", ".join(f"{c} {v}" for c, v in board.diagnosis["diagnosis"].items())
                      + f" (evidence {', '.join(board.diagnosis.get('evidence_ids') or []) or 'none'}).")
    if board.declaration is not None:
        test = board.declaration["separating_test"]
        writes.append(f"Declared insufficient evidence; separating test {test['sensor_group']} for {n_cycles(test['cycles'])}.")
    if not writes:
        writes.append("No diagnosis was submitted." if "declare" in board.unserved else "No diagnosis and no declaration was submitted.")
    for o in board.orders.values():
        closed = f"closed at cycle {o['closed_t']}" if o["status"] == "closed" else f"still open, expected to close at cycle {o['expected_close_t']}"
        verdict = board.verifications.get(o["prediction_id"])
        if verdict is None:
            verified = "not verified"
        elif not board.server:
            verified = f"verification {verdict['status']}, consistent {verdict['consistent']}, {verdict['message']}"
        else:
            verified = (f"verification {verdict['status']}, consistent {verdict['consistent']}, recovery predicted {fmt(verdict['recovery_predicted'])}, "
                        f"posterior of the healthy combination after the repair {fmt(verdict['recovery_posterior'])}{scored_combination_text(verdict)}"
                        f"{confirmation_text(verdict)}")
        writes.append(f"Work order {o['id']} on the {o['component']} created at cycle {o['created_t']}, {closed}, prediction {o['prediction_id'] or 'none'}, {verified}.")
    if not board.orders:
        writes.append("No work order was created.")
    if board.alarms:
        writes.append("Alarms: " + ", ".join(f"{a['channel']} from cycle {a['first_t']}" for a in board.alarms) + ".")
    status = board.limits_line()  # the board's words for each limit, also for one the manifest leaves out
    surface = surface or RigSurface()
    text = (f"{system_prompt(plan_length, surface, board.server, board.read_advice, board.parallel, board.recall_offered)}\n\n{SUMMARY_HEADING}\n\n"
            f"Question of the episode: {board.question}\n\n" + "\n".join(writes) + "\n" + status
            + ("\n" + note if note else "")
            + "\n\nWrite the closing summary of the episode in a few sentences of plain prose: the diagnosis"
            + ("" if "declare" in board.unserved else " or the declaration") + " and the evidence "
              f"behind it, each work order and whether it closed, what the verification of each prediction said, and the state of the {surface.noun} "
              f"as the verdicts show it. Claim a replacement only for an order that closed, and claim a restored {surface.noun} only when the "
              "verification of the last repair supports it. Return plain text, no JSON.")
    return text
