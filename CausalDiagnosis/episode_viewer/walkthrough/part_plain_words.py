from . import formatting, controller_walk, episode_files, tool_answers


END_WORDS = {"finish": "the agent's finish", "budget": "running out of budget", "horizon": "reaching the horizon",
             "max_steps": "reaching the step cap", "failure": "a failure under the strict rule"}


# ----- part 5, in plain words -----
def part_plain(ep):
    return [f"## {formatting.PARTS[4]}", "", " ".join(story(ep)), ""]


def story(ep):
    """The episode in plain words from the public files, telling what the agent read and decided, how the environment answered its
    orders, what the agent saw of the change, and how the episode ended. Hidden values stay in parts 3 and 4."""
    components, calls, events = ep.components, ep.calls, ep.released.get("events") or []
    budget = f"a budget of {formatting.num(ep.config['budget'])}" if ep.config.get("budget") is not None else "no budget"
    out = [f"The episode began at cycle 0 with {budget}, and the agent knew only the briefing and the profile of part 1."]
    oracle = ep.run.get("agent") == "oracle"
    if oracle:
        out.append("The oracle policy takes the hidden start from the environment, so it read no sensor.")
    commit = next((i for i, c in enumerate(calls) if c.tool in {ep.tools[k] for k in controller_walk.COMMIT_KINDS} and c.result is not None), None)
    before = calls if commit is None else calls[:commit]
    if ep.aob:
        reads = [c for c in before if c.tool == ep.tools["read"] and (c.result or {}).get("observations")]
        read_words = [f"{ep.group_of(c) or 'sensors as asked'} at {tool_answers.row_span(c.result['observations'])}" for c in reads]
    else:
        reads = [c for c in before if c.tool == "read_sensors" and (c.result or {}).get("served")]
        read_words = [f"{c.result.get('sensor_group')} at cycle {c.result['served'][0]['t']}" for c in reads]
    rankings = [c.answer for c in before if c.tool == "rank_root_causes" and (c.answer or {}).get("candidates")]
    if reads:
        sentence = (("Before it committed" if commit is not None else "In the episode") + ", the agent read sensor group "
                    + formatting.joined(read_words))
        if rankings:
            first, last = rankings[0]["candidates"][0], rankings[-1]["candidates"][0]
            if len(rankings) == 1:
                sentence += f", and the ranking put {formatting.short_levels(first['combination'], components)} on top at {formatting.prob(first.get('probability'))}"
            elif first["combination"] == last["combination"]:
                sentence += (f", and the ranking kept {formatting.short_levels(first['combination'], components)} on top, at "
                             f"{formatting.prob(first.get('probability'))} after the first reading and {formatting.prob(last.get('probability'))} after the last")
            else:
                sentence += (f", and the top of the ranking moved from {formatting.short_levels(first['combination'], components)} at "
                             f"{formatting.prob(first.get('probability'))} to {formatting.short_levels(last['combination'], components)} at {formatting.prob(last.get('probability'))}")
            sentence += ", with the ranking still saying read first" if rankings[-1].get("read_first") else ""
        out.append(sentence + ".")
    elif commit is not None and not oracle:
        out.append("The agent committed without reading any sensor.")
    surprises = (ep.plans or {}).get("surprises") or [] if ep.kind == "causal" else []
    # a surprise observed after the commitment, such as a repair plan that the reading after the repairs refuted, is told after
    # the orders, so the plain words keep the order of the episode
    committed_at = (calls[commit].answer or {}).get("time") if commit is not None else None
    later = [s for s in surprises if committed_at is not None and ((s.get("observed") or {}).get("time") or 0) > committed_at]
    earlier = [s for s in surprises if s not in later]
    if earlier:
        stops = formatting.joined(f"plan {s['plan']} at its {s['kind']} step" for s in earlier)
        out.append(f"The controller stopped {stops} when the result broke the step's expectation, and planned again"
                   + (" after each stop." if len(earlier) > 1 else "."))
    out.append(commitment_sentence(ep, commit))
    out += [f"At cycle {formatting.num(s['observed']['time'])} the controller stopped plan {s['plan']} at its {s['kind']} step and planned again, "
            f"its note reading {formatting.quoted(s.get('reason'), 200)}." for s in later]
    out += change_sentences(ep, events)
    end = next((e for e in events if e["event"] == "end"), None)
    if end:
        paid = prices_paid(ep)
        out.append(f"The episode ended by {END_WORDS.get(end.get('reason'), end.get('reason'))} at cycle {formatting.num(end.get('time'))}"
                   + (f", with {formatting.num(paid)} of the budget of {formatting.num(ep.config['budget'])} paid in prices." if paid is not None else "."))
    else:
        out.append("The log has no end entry, so the episode stood open when the run stopped.")
    out.append("Part 3 shows what changed out of the agent's sight, and part 4 scores the diagnosis and the repairs against the hidden start "
               "and gives the cost, which counts the downtime and the penalties beside the prices.")
    return out


def prices_paid(ep):
    """The budget less the last budget left that the action server reported, which is the prices paid, since the counter falls
    by prices alone. The cost with the downtime and the penalties stays in the hidden log and the verdict. None without a budget."""
    budget = ep.config.get("budget")
    lefts = [c.answer["budget_left"] for c in ep.calls if c.server in episode_files.ACTION_SERVERS and "budget_left" in (c.answer or {})]
    return budget - lefts[-1] if budget is not None and lefts else None


def commitment_sentence(ep, commit):
    """The commitment and the orders. AssetOpsBench's wo answers an order with its document, which names the child asset and
    no closing cycle, so the orders of those rows go without one."""
    if commit is None:
        return "The agent never submitted a diagnosis or a declaration of insufficient evidence."
    calls, components = ep.calls, ep.components
    call = calls[commit]
    when = f"At cycle {formatting.num(call.answer['time'])} the agent" if "time" in call.answer else "The agent"
    if call.tool == "submit_diagnosis":
        sentence = f"{when} submitted the diagnosis {formatting.short_levels(call.result.get('diagnosis'), components)}"
    else:
        test = call.result.get("separating_test") or {}
        sentence = (f"{when} declared the evidence insufficient, naming {test.get('sensor_group')} for "
                    f"{formatting.plural(test.get('cycles'), 'cycle')} as the separating test")
    if ep.aob:
        # an order is created at the clock its receipt reports, and the base row, without a receipt, names no cycle
        orders = [c for c in calls if c.tool == ep.tools["order"] and (c.result or {}).get("wonum")]
        if not orders:
            return sentence + " and created no work order."
        named = []
        for c in orders:
            asset = (c.result.get("work_order") or {}).get("assetnum")
            where = f"on the {ep.children[asset]}" if asset in ep.children else f"on {asset}"
            named.append(f"{c.result['wonum']} {where}" + (f" at cycle {formatting.num(c.answer['time'])}" if "time" in c.answer else ""))
        return sentence + f" and created {formatting.plural(len(orders), 'work order')}, " + formatting.joined(named) + "."
    orders = [c.result["work_order"] for c in calls if c.tool == "create_work_order" and (c.result or {}).get("work_order")]
    if not orders:
        return sentence + " and created no work order."
    closes = sorted({o["expected_close_t"] for o in orders})
    return (sentence + f" and created {formatting.plural(len(orders), 'work order')}, "
            + formatting.joined(f"{o['id']} on the {o['component']} at cycle {formatting.num(o.get('created_t'))}" for o in orders)
            + f", expected to close at cycle {formatting.joined(closes)}.")


def change_sentences(ep, events):
    """How the environment answered the orders, as the public events tell it, and what the agent saw of the change."""
    out = []
    executed = sorted({e["t_start"] for e in events if e["event"] == "work_order_executed"})
    closed = [e for e in events if e["event"] == "work_order_closed"]
    mine = [e for e in closed if e.get("source") == "agent"]
    by_t = {}
    for e in mine:
        by_t.setdefault(e["t_start"], []).append(e["id"])
    closes = formatting.joined(f"{formatting.joined(ids)} at cycle {t}" for t, ids in by_t.items())
    server = "AssetOpsBench's iot server" if ep.aob else "the action server"
    restored = f". From each closure on, {server} served recordings of the rig with that component restored."
    if executed:
        out.append(f"The technician arrived at cycle {formatting.joined(executed)} and stopped the rig for the repairs, "
                   + (f"then closed {closes}" if by_t else "and no order closed before the end") + restored)
    elif by_t:
        # a repair without downtime closes as the technician reaches it, so no executed event marks a stop of the rig
        out.append(f"The repairs took no downtime, so the rig never stopped, and the technician closed {closes}" + restored)
    failures = [e for e in events if e["event"] == "failure"]
    if failures:
        out.append(f"{'A failure' if len(failures) == 1 else 'Failures'} fired at cycle {formatting.joined(e['t_start'] for e in failures)}, adding "
                   f"{formatting.joined(formatting.num(e.get('cost')) for e in failures)} to the cost, without telling the agent which component failed.")
    if not mine:
        return out
    first_close, last_close = min(e["t_start"] for e in mine), max(e["t_start"] for e in mine)
    observations = ep.released.get("observations") or []
    seen_before = [o for o in observations if o["t"] < first_close]
    seen_after = [o for o in observations if o["t"] >= last_close]
    if seen_after:
        out.append(seen_change(seen_before[-1] if seen_before else None, seen_after[0], ep.bands))
    else:
        out.append("The agent took no reading after the repairs, so the new recordings never reached it.")
        summaries = [c for c in ep.calls if c.tool == "summarize_changes_since" and c.result and c.answer.get("time", -1) >= last_close]
        if summaries:
            result = summaries[-1].result
            out.append(f"The agent's change summary at cycle {formatting.num(summaries[-1].answer.get('time'))} listed "
                       f"{formatting.plural(len(result.get('alarms') or []), 'alarm')} since cycle {formatting.num(result.get('since'))} and "
                       f"{formatting.plural(len(result.get('work_orders_closed') or []), 'closed order')}.")
    bands = [(key, v) for key, v in (((ep.plans or {}).get("board") or {}).get("verifications") or {}).items()
             if str(v.get("status", "")).endswith("the normal ranges")]
    if bands:
        out.append("The controller verified " + formatting.joined(f"{key} on the normal ranges, consistent {formatting.num(v.get('consistent'))}" for key, v in bands)
                   + ", on the reading after the close, since no CausalDiagnosis MCP ran.")
    checks = [c.answer for c in ep.calls if c.tool == "verify_outcome" and (c.answer or {}).get("status")]
    if checks:
        out.append("The CausalDiagnosis MCP's `verify_outcome` then returned " + formatting.joined(f"the status {formatting.quoted(a.get('status'))} for {a.get('prediction_id')} "
                                                                                    f"with consistent {formatting.num(a.get('consistent'))}" for a in checks) + ".")
    return out


def seen_change(before, after, bands):
    if before is None:
        return f"After the repairs the agent read {after['sensor_group']} at cycle {after['t']}, its first reading of the episode."
    common = [f for f in before["values"] if f.endswith("_mean") and f in after["values"]]
    if not common:
        return f"After the repairs the agent read {after['sensor_group']} at cycle {after['t']}, a group it had not read before them."
    status = {f: (formatting.outside_band(f, before["values"][f], bands), formatting.outside_band(f, after["values"][f], bands)) for f in common}
    field = (next((f for f in common if status[f] == (True, False)), None) or next((f for f in common if status[f] == (False, True)), None)
             or common[0])
    v1, v2 = formatting.num(before["values"][field]), formatting.num(after["values"][field])
    sentence = f"After the repairs the agent read {after['sensor_group']} again at cycle {after['t']}, and the {formatting.tag(field)} mean "
    if status[field] == (True, False):
        return sentence + f"went from {v1} at cycle {before['t']}, outside its normal range, to {v2}, inside it."
    if status[field] == (False, True):
        return sentence + f"went from {v1} at cycle {before['t']}, inside its normal range, to {v2}, outside it."
    return sentence + f"read {v2} against {v1} at cycle {before['t']}."
