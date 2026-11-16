import json
from . import formatting, controller_walk, call_placement, tool_answers


# ----- the controller's process in order -----
SOURCE_WORDS = {"model": "the model's plan", "model_retry": "the model's plan on a retry", "model_repaired": "the model's plan after a repair",
                "rule_only": "the rule's plan, the test tool", "no_plan_after_parse_failure": "no plan, the model's reply failed to parse",
                "no_plan_after_check_failure": "no plan, the plan check refused the model's plan"}


def causal_process_lines(ep):
    """Every plan at its first call, the ranking and the candidate plans after each reading, every read with the elimination the
    ranking expected against the one it realized, every rewind with the refuted plan and the plan that followed, every compaction
    with what it collapsed and every verification, by verify_outcome or on the normal ranges, in the order of the calls, so that a
    reader follows the whole episode without the files."""
    labels, _ = ep.walk
    plans, board = ep.plans.get("plans") or [], ep.plans.get("board") or {}
    by_index = {p["index"]: p for p in plans}
    starts = call_placement.plan_starts(labels, plans)
    reads = {(entry.get("plan"), entry.get("step")): entry for entry in board.get("reads") or []}
    verdicts = {tuple(v["reading"]): (key, v) for key, v in (board.get("verifications") or {}).items() if isinstance(v.get("reading"), list)}
    events = []  # (call index the event belongs to, order within the call, text), the index len(calls) for after the last call
    for plan in plans:
        events.append((starts[plan["index"]], 0, plan_text(plan)))
        for k, step in enumerate(plan.get("steps") or [], 1):
            refuted = (step.get("observed") or {}).get("refuted")
            if refuted:
                mine = [i for i, label in enumerate(labels) if label == (plan["index"], k)]
                events.append((mine[-1] if mine else starts[plan["index"]], 1,
                               rewind_text(plan, k, refuted, board, by_index.get(plan["index"] + 1), ep.plans.get("end_reason"))))
    for i, call in enumerate(ep.calls):
        if call.tool == ep.tools["read"] and labels[i] in reads:
            events.append((i, 2, read_text(reads[labels[i]])))
            if labels[i] in verdicts:
                key, verdict = verdicts[labels[i]]
                events.append((i, 2, f"verification of {key} on the normal ranges, consistent {formatting.num(verdict.get('consistent'))}, {verdict.get('message')}"))
        elif call.tool == "rank_root_causes" and (call.answer or {}).get("candidates"):
            events.append((i, 2, ranking_text(call, labels[i], ep.calls[i - 1] if i else None, ep.components, ep.tools)))
        elif call.tool == "verify_outcome" and call.answer is not None:
            events.append((i, 2, f"verification, {tool_answers.diagnostics_answer(call.tool, call.answer, ep.components)}"))
    for entry in board.get("compaction_log") or []:
        after = entry.get("after_call") or 0
        events.append((after - 1, 3, compaction_text(entry)))
    events.sort(key=lambda event: event[:2])
    n = len(ep.calls)
    lines = ["The controller's process in order follows, from `trajectory.json` and `plans.json`, with every plan at its first call, the ranking "
             "and the candidate plans after each reading, every read with the elimination the ranking expected against the one it realized, every "
             "rewind with the refuted plan and the plan that followed, every compaction with what it collapsed, and every verification. The step "
             "numbers are those of the table.", ""]
    for position, _, text in events:
        where = "Before the first call" if position < 0 else "After the last call" if position >= n else f"Step {position + 1}"
        lines.append(f"- {where}, {text}")
    return lines + [""]


def plan_text(plan):
    body, steps = plan.get("plan") or {}, plan.get("steps") or []
    planned = body.get("steps") or []
    text = f"plan {plan['index']}, {SOURCE_WORDS.get(plan.get('source'), str(plan.get('source')).replace('_', ' '))}"
    if plan.get("after_surprise"):
        text += ", a re-plan after the surprise"
    text += ((f", mode {body.get('mode')}" if body.get("mode") else "") + (f", branch {formatting.num(body['branch'])}" if body.get("branch") is not None else "")
             + f", written as {formatting.quoted(body.get('why') or '', 200)}.")
    if steps:
        text += " It ran " + formatting.joined(f"{call_placement.step_args_text(s)} ({call_placement.step_outcome(s)})" for s in steps) + "."
    if len(planned) > len(steps):
        text += f" {formatting.plural(len(planned) - len(steps), 'planned step')} did not run."
    return text


def read_text(entry):
    """A reading a plan named, with the elimination the ranking before it expected of its group against the one it realized
    and how many of its channel means lay outside their normal ranges in every cycle it served."""
    text = f"read of {entry.get('sensor_group')} in plan {formatting.num(entry.get('plan'))} step {formatting.num(entry.get('step'))}, {formatting.plural(entry.get('cycles'), 'cycle')}"
    if entry.get("advised"):
        text += ", a read the advice named"
    before, expected, realized = entry.get("effective_before"), entry.get("expected_after"), entry.get("realized_after")
    batch = entry.get("batch") or [None, None, 1]
    if before is not None and expected is not None and realized is not None and batch[2] > 1:
        # the ranking follows the whole batch, so the value after it belongs to the batch and not to this read alone
        text += (f", was expected to take the effective hypotheses from {formatting.num(before)} to {formatting.num(expected)}, and the batch of "
                 f"{formatting.num(batch[2])} reads it ran in, from step {formatting.num(batch[1])}, took them to {formatting.num(realized)}")
    elif before is not None and expected is not None and realized is not None:
        text += (f", was expected to take the effective hypotheses from {formatting.num(before)} to {formatting.num(expected)} and took them to {formatting.num(realized)}, "
                 f"{formatting.num(before - realized)} eliminated against {formatting.num(entry.get('expected_elimination'))} expected"
                 + (f" with an expected information gain of {formatting.num(entry['expected_information_gain'])} nats" if entry.get("expected_information_gain") is not None else ""))
    cycles = entry.get("cycles_outside") or []
    if cycles:
        text += ", " + formatting.joined(f"{formatting.num(outside)} of {formatting.num(banded)} channel means outside their normal ranges at cycle {formatting.num(t)}" for _n, t, outside, banded in cycles)
    return text + "."


def rewind_text(plan, k, refuted, board, next_plan, end_reason):
    reasons = {tuple(e.get("components") or []): e.get("reason") for e in board.get("refuted") or []}
    text = f"rewind in plan {plan['index']} step {k}, " + formatting.joined(
        f"the plan on the {formatting.joined(components)} was refuted" + (f", {reasons[tuple(components)]}" if tuple(components) in reasons else "")
        for components in refuted) + "."
    if next_plan is None:
        return text + f" No plan followed, the episode ended by {end_reason or 'the environment'}."
    body = next_plan.get("plan") or {}
    ordered = [(s.get("args") or {}).get("component") for s in body.get("steps") or [] if s.get("kind") == "order"]
    return (text + (" The controller ranked again and the" if board.get("server", True) else " The") + f" next plan, plan {next_plan['index']}, "
            f"{SOURCE_WORDS.get(next_plan.get('source'), str(next_plan.get('source')).replace('_', ' '))}, mode {body.get('mode')}"
            + (f", branch {formatting.num(body['branch'])}" if body.get("branch") is not None else "") + f", written as {formatting.quoted(body.get('why') or '', 200)}, "
            + (f"orders the {formatting.joined(ordered)}." if ordered else "orders nothing."))


def ranking_text(call, label, previous, components, tools):
    data = call.answer
    where = ("controller finish" if label == controller_walk.FORCED_FINISH else f"before plan {label[0]}" if label[1] == 0 else f"plan {label[0]} step {label[1]}")
    text = (f"{call_placement.call_role(call, previous, tools)} ({where}). {formatting.num(data.get('n_candidates'))} candidates"
            + (f" over {formatting.num(data['n_states'])} states" if data.get("n_states") is not None else "")
            + (f", {formatting.effective_count(data['effective_hypotheses'])}" if data.get("effective_hypotheses") is not None else "")
            + f", read first {formatting.num(data.get('read_first'))}. Candidates "
            + formatting.joined(candidate_text(c, components) for c in data.get("candidates") or []) + ".")
    if data.get("branches"):
        text += " Candidate plans " + formatting.joined(branch_group_texts(data["branches"], components)) + "."
    if data.get("elimination"):
        text += " Predicted elimination per cost " + formatting.joined(f"{r.get('sensor_group')} {formatting.num(r.get('elimination_per_cost'))}" for r in data["elimination"][:tool_answers.TOP_SHOWN]) + "."
        if any(r.get("ig_per_cost") is not None for r in data["elimination"]):
            text += " Predicted information gain per unit cost " + formatting.joined(f"{r.get('sensor_group')} {formatting.num(r.get('ig_per_cost'))}" for r in data["elimination"][:tool_answers.TOP_SHOWN]) + "."
        if any(r.get("ec2_per_cost") is not None for r in data["elimination"]):
            text += " Predicted edge cut per unit cost " + formatting.joined(f"{r.get('sensor_group')} {formatting.num(r.get('ec2_per_cost'))}" for r in data["elimination"][:tool_answers.TOP_SHOWN]) + "."
    if data.get("read_batch"):
        text += " The server advises the read batch " + formatting.joined(f"{b.get('sensor_group')} for {formatting.plural(b.get('cycles'), 'cycle')}" for b in data["read_batch"]) + "."
    test = data.get("separating_test")
    if test:
        text += f" The separating test is {test.get('sensor_group')} for {formatting.plural(test.get('cycles'), 'cycle')} at price {formatting.num(test.get('price'))}."
    return text


def candidate_text(candidate, components):
    text = f"rank {candidate.get('rank')} {formatting.short_levels(candidate.get('combination'), components)} at {formatting.prob(candidate.get('probability'))}"
    current = candidate.get("current_combination")
    if current and current != candidate.get("combination"):
        text += f", now {formatting.short_levels(current, components)}"
    return text


def branch_group_texts(branches, components):
    """One text per distinct candidate plan, with the ranks that share it, since candidates that differ in a level alone
    share the plan on their faulty set."""
    groups = []
    for branch in branches:
        text = branch_text(branch, components)
        if groups and groups[-1][1] == text:
            groups[-1][0].append(branch.get("rank"))
        else:
            groups.append(([branch.get("rank")], text))
    return [f"rank{'s' if len(ranks) > 1 else ''} {formatting.joined(formatting.num(r) for r in ranks)} on the {text}" for ranks, text in groups]


def branch_text(branch, components):
    plan, cost = branch.get("repair_plan") or {}, branch.get("cost_to_restored") or {}
    confirm, outcome = branch.get("confirm_test") or {}, branch.get("predicted_outcome") or {}
    steps = plan.get("steps") or []
    text = formatting.joined(branch.get("faulty_set") or []) or "healthy rig"
    if steps:
        text += (f", replace the {formatting.joined(s.get('component') for s in steps)} at price {formatting.num(plan.get('price'))} with "
                 f"{formatting.plural(plan.get('cycles_to_restored'), 'cycle')} to restored")
    else:
        text += ", nothing to replace"
    text += (f", predicted cost to restored acting now {formatting.num(cost.get('act_now'))} against reading the confirm test and acting on what "
             f"is first after it {formatting.num(cost.get('test_first'))}")
    if confirm:
        text += f" with the confirm test {confirm.get('sensor_group')} for {formatting.plural(confirm.get('cycles'), 'cycle')}"
    if outcome.get("recovery_if_acted") is not None:
        text += f", predicted recovery if acted {formatting.prob(outcome['recovery_if_acted'])}"
    return text


def compaction_text(entry):
    parts = []
    for item in entry.get("collapsed") or []:
        if item.get("kind") == "evidence":
            first, last = (list(item.get("observations") or []) + [None, None])[:2]
            span = f"observation {formatting.num(first)}" if first == last else f"observations {formatting.num(first)} to {formatting.num(last)}"
            parts.append(f"{formatting.plural(item.get('values'), 'value')} of {item.get('sensor_group')} from {span} collapsed into one summary row")
        elif item.get("kind") == "summary":
            replaced = item.get("replaced") or {}
            lost = sum((item.get("left_out_of_summary_request") or {}).values())
            parts.append(f"one summary of the earlier turns written, replacing {formatting.plural(sum(replaced.values()), 'entry')}"
                         + (f", {formatting.plural(lost, 'entry')} left out of the summary request" if lost else ""))
        elif item.get("kind") == "trim":
            parts.append(f"{formatting.plural(item.get('dropped'), 'older entry')} of the surprises, notes and notifications dropped, the latest "
                         f"{formatting.num(item.get('kept_surprises'))}, {formatting.num(item.get('kept_notes'))} and {formatting.num(item.get('kept_notifications'))} kept")
        else:
            parts.append(formatting.quoted(json.dumps(item), 120))
    return f"compaction after the call, {formatting.joined(parts) or 'nothing collapsed'}, the working memory at {formatting.num(entry.get('rendered_characters'))} characters"
