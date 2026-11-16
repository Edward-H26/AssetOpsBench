from . import formatting, episode_files, call_placement, tool_answers, controller_process


PLAN_COLUMN_WORDS = {"plan_execute": " Plan step names the task the planner wrote for the call.",
                     "causal": (" Plan step names the plan and the step that made the call, with the step's expectation and whether it "
                                "held on the call that carries the step. The controller lets a result miss an expectation by the "
                                "expectation tolerance of the agent options, as a share of a ceiling on the effective hypotheses and as a "
                                "margin below a floor on a mass or a recovery.")}


# ----- part 2, the steps -----
def part_steps(ep):
    out = [f"## {formatting.PARTS[1]}", ""]
    if ep.kind == "plan_execute" and ep.plan_text:
        out += ["The plan the planner wrote before the first call, `plan.txt`, verbatim.", "", "```text", ep.plan_text.rstrip("\n"), "```", ""]
    labels, plan_note = call_placement.plan_column(ep)
    counts, model_note = call_placement.model_call_counts(ep)
    if ep.kind == "causal" and ep.plans:
        out += causal_plan_lines(ep)
    example = next(iter(next(iter((ep.released.get("rules") or {}).get("sensor_groups", {}).values()), [])), None)
    if ep.aob:
        source = ("Clock after and budget left are the receipt line that the metering layer appends to the answers of AssetOpsBench's "
                  "servers and of the task server from the priced row on, so the base row leaves them blank.")
        events = "Notifications are the public events the receipt carried, none when it listed none and blank without a receipt."
        named = f", so {formatting.tag(example)} stands for `{example}`, the channel's mean in a history row" if example else ""
    else:
        source = "Clock after and budget left are the action server's answer to the call."
        events = ("Notifications are the public events the action server returned with the call, none when it returned an empty list "
                  "and blank when the server did not report them.")
        named = f", so {formatting.tag(example)} stands for `{example}_mean`" if example else ""
    out += ["One row per tool call, in the order the agent made them. " + source + " Price is what the call took from the budget, the "
            "budget left before it less the budget left after it, and cycles is the clock's move over the call, the thinking time charged "
            "before it included under a timed preset. " + events + " The CausalDiagnosis MCP reads the evidence released so far, charges "
            "nothing and moves no clock, so its rows show a price and cycles of 0 and leave the clock, the budget and the notifications "
            "empty. Model calls so far counts the language model calls made before the tool call returned, from `llm_calls.json`. A channel "
            "is named by the first part of its field" + named
            + f", and a combination lists the levels of the {formatting.joined(ep.components)} in that order." + PLAN_COLUMN_WORDS.get(ep.kind, ""), ""]
    headers = ["Step", "Clock after", "Budget left", "Price", "Cycles", "Tool and arguments", "Answer", "Notifications", "Model calls so far"]
    with_plan = any(labels)
    rows = []
    time_before, budget_before = 0, ep.config.get("budget")
    for n, call in enumerate(ep.calls, 1):
        envelope = call.answer if call.server in episode_files.ACTION_SERVERS and call.answer is not None else {}
        price = cycles = "" if call.server in episode_files.ACTION_SERVERS else "0"  # an action call without an envelope has no known price or clock move
        if "budget_left" in envelope:
            price = formatting.num(budget_before - envelope["budget_left"]) if budget_before is not None else ""
            budget_before = envelope["budget_left"]
        if "time" in envelope:
            cycles = formatting.num(envelope["time"] - time_before)
            time_before = envelope["time"]
        row = [str(n), formatting.num(envelope["time"]) if "time" in envelope else "", formatting.num(envelope["budget_left"]) if "budget_left" in envelope else "",
               price, cycles, f"{episode_files.SERVER_NAMES.get(call.server, call.server)} {tool_answers.describe_args(call)}", tool_answers.describe_answer(call, ep),
               tool_answers.notifications_text(envelope) if call.server in episode_files.ACTION_SERVERS else "", "n/a" if counts is None else str(counts[n - 1])]
        rows.append(row + ([labels[n - 1]] if with_plan else []))
    out += formatting.table(headers + (["Plan step"] if with_plan else []), rows) if rows else ["The agent made no tool call."]
    out.append("")
    out += [note for note in (plan_note, model_note) if note]
    if counts and len(ep.llm_calls) > counts[-1]:
        out.append(f"After the last tool call the agent made {formatting.plural(len(ep.llm_calls) - counts[-1], 'more model call')}"
                   + (", the runner's closing answer." if ep.kind == "plan_execute" else "."))
    out += [f"Plan step {formatting.num(r.get('step'))}, {formatting.quoted(r.get('task') or '', 120)}, named no tool, so the runner made no call for it."
            for r in ep.toolless]
    if ep.kind == "causal" and ep.plans and ep.walk[1]:
        out += controller_process.causal_process_lines(ep)
    if ep.llm_calls:
        out.append(f"The agent made {formatting.plural(len(ep.llm_calls), 'model call')} in all, with {sum(c.get('input_tokens') or 0 for c in ep.llm_calls)} "
                   f"input and {sum(c.get('output_tokens') or 0 for c in ep.llm_calls)} output tokens.")
    else:
        out.append("This agent called no model.")
    out.append("")
    summary = closing_summary(ep)
    if summary is not None:
        out += ["The closing summary the agent passed to `finish` follows.", ""] + ["> " + line for line in (summary.strip() or "(empty)").splitlines()] + [""]
    return out


def closing_summary(ep):
    for call in reversed(ep.calls):
        if call.tool == "finish" and isinstance(call.args, dict) and isinstance(call.args.get("summary"), str):
            return call.args["summary"]
    answer = ep.folder / "answer.txt"
    return answer.read_text(encoding="utf-8") if answer.exists() else None


def causal_plan_lines(ep):
    plans = ep.plans.get("plans") or []
    differential = [d.get("component") for d in ep.plans.get("differential") or [] if isinstance(d, dict)]
    out = []
    if differential:
        out += [f"At the orientation the model wrote a differential over the {formatting.joined(differential)}, the components it expected could "
                "be faulty, before any reading.", ""]
    out += ["The controller's plans, `plans.json`, follow. A plan made after a surprise is a re-plan. The source says who wrote the "
            "plan, the model (on a run of older code also the model on a retry or after a repair), nobody when the model's plan "
            "failed to parse or to pass the plan check and nothing ran, or the rule only test tool.", ""]
    out += formatting.table(["Plan", "Source", "After a surprise", "Mode", "Branch followed", "Why line of the plan", "Steps run"],
                 [(formatting.num(p.get("index")), str(p.get("source")).replace("_", " "), "yes" if p.get("after_surprise") else "no",
                   (p.get("plan") or {}).get("mode") or "", formatting.num((p.get("plan") or {}).get("branch")),
                   formatting.quoted((p.get("plan") or {}).get("why") or "", 240),
                   f"{len(p.get('steps') or [])} of {len((p.get('plan') or {}).get('steps') or [])}") for p in plans])
    return out + [""]
