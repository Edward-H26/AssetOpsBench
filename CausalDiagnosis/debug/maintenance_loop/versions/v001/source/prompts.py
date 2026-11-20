"""The prompts of the CausalDiagnosis agent: one system prompt per mode and the builders of the three user turns, orientation,
planning and summary (CausalDiagnosis/docs/experiments.md, section The driver, agent causal).

Opening turns request only the public queries legal at that phase. After the opening, one system prompt serves every
guidance level and every model, since the design (preet2026simplicity, reddy2026hidden)
fixes one interface and one parser, and the prompt without the diagnostics server shares its role, its opening and the loop
of the controller and differs where the server's ranking, advice and predictions are missing. The system prompt
states the role, the step kinds with their arguments and expectations, the opening the model plans, the plan
schema, the rule that every step cites evidence identifiers and every claim names a failure mode or a timeline fact, and what
the controller refuses whatever the model writes. The evidence guard and the cost rule
are advice the board states: under the cost choice the prompt says the advice weighs the act now cost of a branch, and a
diagnosis expects the candidate on top rather than the one it submits, since a plan may act below rank 1; the rank choice
keeps the prompts whose advice names rank 1. The user turns carry the data: the question, the profile and the failure modes for the
orientation, the rendered board for a plan, the writes and the verdicts for the closing summary. The backend takes one string,
so a turn is the system prompt followed by the user turn.
"""
import json
import re

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import durations_of, fmt, n_cycles, order_prices, read_prices, scored_combination_text, table
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import ARMS, plan_schema
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import RigSurface

PLAN_LENGTH_MARK = "<PLAN_LENGTH>"
DIAGNOSE_TOP_RANK_MARK = "<DIAGNOSE_TOP_RANK>"
ADVICE_RULES_MARK = "<ADVICE_RULES>"  # what the advice names under the read choice and the repair choice of the run
ARM_MARK = "<ARM>"  # inside READ_ARM_RULE, the read of the arm
OBS_FORM_MARK, TOOLS_MARK, CHANGES_MARK = "<OBS_FORM>", "<TOOLS>", "<CHANGES>"  # what the surface the agent runs on fills in (tools/)
NOUN_MARK, EXAMPLE_TAG_MARK = "<NOUN>", "<EXAMPLE_TAG>"  # the noun of the asset and an example channel tag, from the surface
STATISTIC_MARK = "<STATISTIC>"  # the plain statistic of the served values, from the surface
ORIENTATION_HEADING = "## Orientation"
PLANNING_HEADING = "## Planning turn"
SUMMARY_HEADING = "## Closing summary"
EARLIER_TURNS_HEADING = "## Summary of your earlier turns"  # the request of step 3 of the compaction
EARLIER_TURNS_ASK = ("You are writing a summary of your own earlier planning turns, which will leave the board. The board keeps the task, "
                     "the terms, the fix history, the failure modes, the open orders, the predictions, the refuted plans, the latest reading "
                     "of every group, the ranking and the status. Write what those do not say: the progress so far and the decisions you made "
                     "with their reasons, the evidence identifiers that moved you and why, the candidates you set aside and why, and what "
                     "remains to be done. Write no probability of your own. Return the summary as plain text, with no JSON and no plan.")
LEFT_OUT = "The oldest entries of the board were left out of this request to make it fit, and they will not return."
RETRY_HEADING = "### Your previous reply was not usable"
PARALLEL_MARK = "<PARALLEL>"  # at the end of the read line: the batch of read steps and the schedule, empty with the switch off
# the size of a plan in both system prompts: the first sentence with a count of steps (--causal-plan-length), the second without one,
# where the budget, the horizon and the step cap bound a plan (plan.check_plan)
COUNTED_STEPS = 'with at most <PLAN_LENGTH> entries under steps, finish included, and candidates best first; a longer plan is refused and you are asked again.'
PAID_STEPS = ("with as many entries under steps as the budget, the horizon and the step cap pay, finish included, and candidates "
              "best first; a plan they cannot pay is refused and you are asked again.")

ROLE = """You plan the diagnosis and repair of an industrial asset, the one the profile names, for a controller that keeps a situation board and calls the tools your plan names, the free reads of the opening included, and no other. The asset runs under the rules the board states: a reading has the price and the duration of its row, downtime accrues while a component is degraded where the row charges it, and a technician executes work orders after the lead time of its row, where the row sets one. Exactly one diagnosis or one declaration of insufficient evidence is allowed per episode, a work order needs a prior diagnosis or declaration, and the episode ends with finish or when a limit of its row ends it: the horizon, the step cap, the budget, the limit of wall time or a component failure, each where the row sets it. Each turn you receive the board and return one JSON object, with nothing after it."""
OPENING_STEPS = """Step kinds. Three kinds read the opening, each free. profile, no args, reads the asset profile and comes first, since the board holds nothing about the asset before it. failure_modes, args {"component": <a component, or "" for every component>}, reads the failure modes the asset serves. fix_history, no args, reads the repairs recorded before the episode. Five kinds act on the <NOUN>: read, diagnose, declare, order and wait. A sixth, finish, ends the episode, and a seventh, recall, brings collapsed readings back to the board."""
# the recall step in the step kinds and its line, which the switch recall off takes out (without_recall)
RECALL_CLAUSE = ", and a seventh, recall, brings collapsed readings back to the board"
RECALL_LINE = ('- recall: args {"sensor_group": <a group>, "observations": [<first>, <last>]}; no expectation. It calls no tool and costs '
               "nothing: the controller puts the full rows of that group's observations first to last, which the board collapsed into a "
               "summary row, back on the next board, once. A plan of recalls alone makes no call and still takes one step.")
FINISH_STEP = "- finish: no args and no expectation. The controller writes the closing summary and ends the episode.\n" + RECALL_LINE
LOOP = """The controller executes your steps while every expectation holds and calls you again only when one fails, with the surprise written on the board, so plan several steps and attach an expectation you are willing to be stopped by. A plan the controller cannot parse or refuses runs nothing, and you are asked again with its reasons; so is a differential the controller cannot parse. A turn that calls no tool of the <NOUN> still takes one step, and as many requests in a row without a call as the step cap allows end the episode."""

SYSTEM_PROMPT = ROLE + """

The board. Before the profile is read it holds nothing about the <NOUN> and names the reads of the opening. Evidence rows carry the identifiers <OBS_FORM>; a channel <STATISTIC> outside its public normal range is marked NO in the column in normal range, and rows a ranking has absorbed collapse to one summary row per sensor group with the identifiers outside their normal range. The fix history lists the repairs recorded before the episode. The failure modes table lists the failure modes the asset serves with their identifiers fmea/<source>/<code>, and their level, severity, recommended action and indicating channels where this row gives them. Channels outside their normal range shows, for every sensor group read, its latest cycle with the channel <STATISTIC>s outside their normal ranges, below or above. The ranking lists the candidate combinations now with their failure mode codes and posterior mass, the effective number of hypotheses of the belief, and the decision fields of the rule: read_first, the separating test with its price and expected loss, the expected loss of acting now. The advice of the CausalDiagnosis MCP gives, per sensor group, the information gain of one cycle of it and that gain per unit cost, which ranks the reads, with the expected number of hypotheses the cycle eliminates beside them, the server's read batch, the reads best first by gain per unit cost whose choice no outcome can change, and its lines of advice: the reads to take while the ranking says read first, the repair plan to act on and, after a refuted plan, the other plan. It is advice: you decide every read, diagnosis and order, and the controller makes only those your plan names. A troubleshooting plan table, when present, walks each candidate forward: its confirm test (the sensor group that separates it best, with the expected mass of the candidate if it is true and if it is false), its repair plan (one work order per faulty component, created one after another as one technician visit with one visit charge, and the one prediction identifier that every order of the plan cites, the state after all of its replacements), the cycles to a restored <NOUN>, the predicted recovery if acted, the expected loss if acted now and if tested first, and the cost to a restored <NOUN> both ways. Refuted repair plans list the component sets whose verification failed, and the advice keeps the last of them out until a later ranking names it first. Your candidates lists the combinations of your latest plan that listed them, in your order, with the ranks after the ranked candidates. The timeline lists alarms (alarm/<t>/<channel>) and closed work orders (wo/<id>).

""" + OPENING_STEPS + """
- read: args {"sensor_group": <group>, "cycles": <n>}; expect {"top_rank": <rank in the current ranking>, "top_mass_at_least": <mass>}. After the reading, the faulty set of the candidate at that rank should be the set on top of the new ranking, with at least that mass summed over every combination of the set. A read of a group whose recorded cycles are all read already is not made and comes back as a surprise, since it would show nothing new; on a row whose clock runs, a wait lets the record grow.<PARALLEL>
- diagnose: args {"rank": <rank>}; expect {"top_rank": <DIAGNOSE_TOP_RANK>}. The controller submits the current combination of that candidate, scored at the cycle of the write, with the evidence you cite and the evidence the ranking names for it. A rank of your candidates submits that combination, the components it leaves out at their healthy level, with the evidence you cite and the channel <STATISTIC>s outside their normal ranges.
- declare: args {"sensor_group": <group>, "cycles": <n>}; no expectation. The advice names the separating test of the ranking or the confirm test of the branch you follow. It must follow a reading, which the controller cites with the evidence you cite and the evidence the ranking names, since a declaration that cites none is scored as a failure.
- order: args {"component": <one component>, "prediction_id": <a prediction identifier from the branch table or the prediction list, or "estimate" to have the controller call estimate_intervention_effect for replace:<component> first>}; expect {"prediction_id": <the same>, "recovery_at_least": <probability>}. The controller places the order after a diagnosis or a declaration only, and an identifier the board does not know gets an estimate first. One component per order step: a step that joins several names is split into one order per component, which count against the step limit.
- wait: args {"cycles": <n, at least 1; 0 is read as 1>}; expect {"no_new_alarm": true}. The controller waits event by event (wait_for_notification with the cycles as the timeout, so the first notification ends each wait) and records every notification<CHANGES>. Once the last order citing a prediction has closed, plan a read, best of a group whose channels indicate the repaired components: on the first reading at or after the close the controller verifies that prediction with verify_outcome. A plan whose verification finds the reading inconsistent with its prediction is refuted: the controller records its component set and marks it on the board, and the advice names the cheapest other previewed branch. A consistent verified verdict stands whatever the ranking after that reading shows.
""" + FINISH_STEP + """
Every step carries "evidence": a list of the identifiers from the board that justify it, the few that moved you; never copy the ranking's evidence list, which the controller cites for you, and cite only identifiers that appear on the board.<TOOLS>

The plan. Return
{"branch": <rank of the candidate you follow>, "mode": "test_first" | "act_now" | "repair_as_test" | "declare", "why": "<one line>", "candidates": [{"levels": {"<component>": <level>, ...}, "failure_mode_ids": ["fmea/<source>/<code>", ...]}, ...], "steps": [{"kind": ..., "args": {...}, "expect": {...}, "evidence": [...]}, ...]}
with at most <PLAN_LENGTH> entries under steps, finish included, and candidates best first; a longer plan is refused and you are asked again. candidates is optional and lists combinations of your own: a candidate names the faulty components at levels of the profile, every other component staying at its healthy level, and your candidates take the ranks after the ranked candidates, in your order, so that a diagnosis may name one of them. test_first reads the confirm test or the separating test and commits when the expected mass arrives; act_now diagnoses and orders the branch's repair plan, then waits to the close and reads so that the controller verifies; repair_as_test orders the cheapest repair of a high mass branch when the confirm test is unaffordable or slower than the downtime it would save; declare names the separating test when no branch can be separated within the budget. The why line names the failure mode identifiers or the timeline facts that moved you beyond the numbers, and says whether the faulty set of the branch reads as the cause or as a symptom of what the evidence shows. You never write a probability of your own: masses, intervals and recoveries come from the board.

The controller refuses a plan whatever you write only where the environment or the plan itself would fail: the opening comes first, the profile before the rest; no work order without a prior diagnosis or declaration; no declaration before a reading it can cite; no finish before the reading that verifies the last order, unless the budget, the horizon or the step cap leaves no room for it; no plan that the budget, the horizon or the step cap cannot pay. Before a diagnosis or a declaration it ranks again on the current evidence. The advice asks you to read the advised reads, the separating test or the followed branch's confirm test before a diagnosis while the ranking says read first and such a read fits.<ADVICE_RULES> """ + LOOP

SYSTEM_PROMPT_WITHOUT_SERVER = ROLE + """

The board. Before the profile is read it holds nothing about the <NOUN> and names the reads of the opening. Evidence rows carry the identifiers <OBS_FORM>; a channel <STATISTIC> outside its public normal range is marked NO in the column in normal range, and rows you have planned on collapse to one summary row per sensor group with the identifiers outside their normal range. A normal range spans the smallest to the largest <STATISTIC> of the channel over the stable recordings of the healthy <NOUN>, so a healthy <NOUN> shows no channel <STATISTIC> outside its normal range, while a fault usually shows at least one. The fix history lists the repairs recorded before the episode. The CausalDiagnosis MCP does not run on this row, so no ranking, advice or prediction exists: you form the candidates yourself. The failure modes table lists the failure modes the asset serves with their identifiers fmea/<source>/<code>, and their level, severity, recommended action and indicating channels where this row gives them. Channels outside their normal range shows, for every sensor group read, its latest cycle with the channel <STATISTIC>s outside their normal ranges, below or above. Your candidates lists the combinations of your latest plan that listed them, in your order. Refuted repair plans list the component sets whose verification on the normal ranges failed. The timeline lists alarms (alarm/<t>/<channel>) and closed work orders (wo/<id>).

""" + OPENING_STEPS + """
- read: args {"sensor_group": <group>, "cycles": <n>}; expect {"outside": [<the channel <STATISTIC>s your followed candidate pushes outside their normal ranges, each by its field name or its tag<EXAMPLE_TAG>>]}. After the reading, a listed channel that the reading served within its normal range is a surprise. A read of a group whose recorded cycles are all read already is not made and comes back as a surprise, since it would show nothing new; on a row whose clock runs, a wait lets the record grow.<PARALLEL>
- diagnose: args {"rank": <rank in your candidates>}; no expectation. The controller submits the full combination of that candidate, the components it leaves out at their healthy level, with the evidence you cite and the channel <STATISTIC>s outside their normal ranges.
- declare: args {"sensor_group": <the group you would read next to separate your candidates>, "cycles": <n>}; no expectation. It must follow a reading, which the controller cites, since a declaration that cites none is scored as a failure.
- order: args {"component": <one component>}; no expectation, since no prediction exists on this row. The controller places the order after a diagnosis or a declaration only. One component per order step: a step that joins several names is split into one order per component, which count against the step limit.
- wait: args {"cycles": <n, at least 1; 0 is read as 1>}; expect {"no_new_alarm": true}. The controller waits event by event (wait_for_notification with the cycles as the timeout, so the first notification ends each wait) and records every notification<CHANGES>. The orders of one plan form one repair plan: once its last order has closed and no other order of yours is open, plan a read, and on the first reading at or after the close the controller verifies the repair plan on the normal ranges, which judge the whole <NOUN>. It is refuted when any channel <STATISTIC> of that reading lies outside its normal range: the controller records its component set and marks it on the board, and you repair another candidate.
""" + FINISH_STEP + """
Every step carries "evidence": a list of the identifiers from the board that justify it, the few that moved you, and only identifiers that appear on the board.<TOOLS>

The plan. Return
{"branch": <rank of the candidate you follow in your candidates>, "mode": "test_first" | "act_now" | "repair_as_test" | "declare", "why": "<one line>", "candidates": [{"levels": {"<component>": <level>, ...}, "failure_mode_ids": ["fmea/<source>/<code>", ...]}, ...], "steps": [{"kind": ..., "args": {...}, "expect": {...}, "evidence": [...]}, ...]}
with at most <PLAN_LENGTH> entries under steps, finish included, and candidates best first; a longer plan is refused and you are asked again. A candidate names the faulty components at levels of the profile, every other component staying at its healthy level, and a candidate with no faulty component states a healthy unit. candidates is needed before your first diagnosis and may be left out later, when the board keeps your last list. test_first reads to separate your candidates and commits when the readings support one; act_now diagnoses and orders the faulty components of the candidate you follow, then waits to the close and reads so that the controller verifies; repair_as_test orders the cheapest repair of a likely candidate when no affordable reading separates it; declare names a sensor group when no reading you can afford would separate your candidates. The why line names the failure mode identifiers or the readings that moved you. You never write a probability, since no mass exists on this row.

The controller refuses a plan whatever you write only where the environment or the plan itself would fail: the opening comes first, the profile before the rest; no diagnosis without your candidates; no work order without a prior diagnosis or declaration; no declaration before a reading it can cite; no finish before the reading that verifies the last repair plan, unless the budget, the horizon or the step cap leaves no room for it; no plan that the budget, the horizon or the step cap cannot pay. """ + LOOP


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
        ("; on a row whose clock runs, a wait lets the record grow.", "; on a row whose clock runs, the record grows as the clock moves."),
        ("then waits to the close and reads so that", "then reads after the close so that"),
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
    """A prompt without the recall step: its clause in the step kinds and its step line (the switch recall off)."""
    return text.replace(RECALL_CLAUSE, "").replace("\n" + RECALL_LINE, "")


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
    text = without_kinds(text, surface.unserved_kinds())
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


def _opening_planning_turn(board, kinds, retry=None):
    """Describe only the queries legal in this opening phase; the model still submits every action."""
    example = {
        "branch": None,
        "mode": "test_first",
        "why": "Read the opening information before planning an investigation.",
        "steps": [{"kind": kind, "args": {"component": ""} if kind == "failure_modes" else {}}
                  for kind in kinds],
    }
    context = board.render()
    text = (f"You return a JSON plan for a controller. You do not execute tools or write their results.\n\n"
            f"{PLANNING_HEADING}\n\nTask context for later investigation:\nQuestion of the episode: {board.question}\n\n{context}\n\n"
            "CURRENT PHASE: opening queries only. The investigation starts in a later planning turn. "
            "Putting an opening query before a diagnosis in the same plan does not make that diagnosis legal. "
            "This reply must contain only the opening steps listed below, in that order. "
            "Do not add read, diagnose, declare, order, wait, recall or finish. "
            "No diagnostic candidate is selected in this phase: branch must be null, never 0. "
            "Omit candidates and evidence; do not invent component conditions, identifiers, observations or counters.\n\n"
            f"Plan format and allowed steps for this turn:\n{json.dumps(example)}\n")
    if retry:
        text += f"\n{RETRY_HEADING}\n{retry}\n"
    return text + "\nReturn only the JSON opening plan, without explanation or simulated tool output."


def _plan_protocol(board):
    """Keep controller step names and public identifiers separate from their MCP bindings."""
    allowed = [kind for kind in plan_schema.KINDS if kind not in board.unserved
               and (kind != "recall" or board.recall_offered)]
    return ("## Controller plan protocol\n"
            "The kind field contains controller step kinds, not MCP tool names. "
            f"The supported kinds on this row are: {', '.join(allowed)}. "
            "Do not use get_episode_evidence, rank_root_causes, estimate_intervention_effect or verify_outcome as kind. "
            "A read after repair lets the controller verify the outcome; there is no verify_outcome plan step. "
            f"For component fields and candidate levels, use only these profile keys: {', '.join(board.profile['components'])}. "
            "An asset ID or child asset ID is not a component key; the controller binds components to assets. "
            "Use only evidence and prediction identifiers actually present on the board. "
            "For a future observation, leave evidence empty rather than inventing its identifier. "
            "Until a candidate is selected, branch may be null. A selected rank starts at 1, never 0. "
            "Return proposed actions only, not their imagined readings, costs, clock values or repair results.")


def planning_turn(board, plan_length, retry=None, surface=None):
    """The planning turn: the rendered board in, one plan out. retry carries the parse error or the reasons the plan check
    refused the previous plan, quoted so the model can repair it in the next turn."""
    if board.profile is None:
        return _opening_planning_turn(board, ["profile"], retry)
    opened = set(board.opening_done)
    pending = list(dict.fromkeys(kind for kind in ("failure_modes", "fix_history", *board.required_opening)
                                 if kind not in opened and kind not in board.unserved))
    if pending:
        # A bounded plan can take the remaining opening queries over several turns.
        return _opening_planning_turn(board, pending[:plan_length] if plan_length is not None else pending, retry)
    text = (f"{system_prompt(plan_length, surface, board.server, board.read_advice, board.parallel, board.recall_offered)}\n\n{PLANNING_HEADING}\n\n"
            f"Question of the episode: {board.question}\n\n{board.render()}\n\n"
            f"{_plan_protocol(board)}\n\n"
            f"Objective: exactly one diagnosis{'' if 'declare' in board.unserved else ' or one declaration'}, the work orders that restore the unit, none when no component is faulty, "
            f"then finish, within the budget and the horizon. Return one plan as the JSON object described above, "
            + (f"with at most {plan_length} steps and nothing after it." if plan_length is not None else "and nothing after it."))
    if retry:
        text += f"\n\n{RETRY_HEADING}\n{retry}\nReturn a plan that passes, as the JSON object only."
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
                        f"posterior of the healthy combination after the repair {fmt(verdict['recovery_posterior'])}{scored_combination_text(verdict)}")
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
