import json
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import TOKEN_MODE
from . import formatting


# ----- part 3, out of the agent's sight -----
def part_hidden(ep):
    out = [f"## {formatting.PARTS[2]}", "", "These events come from the hidden log, `hidden/log.json`. The table leaves out the events that "
           "`released.json` holds in full, such as the alarms of the sensor groups the agent read, and keeps the hidden side of the "
           "others, such as what a closed order did to the hidden combination.", ""]
    log = ep.hidden("log.json")
    if not log:
        return out + ["The hidden log is not in this episode folder, so this part is empty.", ""]
    end = next((e for e in log if e["event"] == "end"), None)
    played = (end or {}).get("played") or []
    created = {e["id"]: e for e in log if e["event"] == "work_order_created"}
    rows = [start_row(log[0], played, ep.components)]
    # an entry the agent was told of in full is no hidden event, so it is matched against the released events field by field
    told = {json.dumps(e, sort_keys=True) for e in ep.released.get("events") or []}
    rows += hidden_rows(log, told, created, ep.components)
    out += formatting.table(["Cycle", "Hidden event", "What changed"], rows)
    if played:
        out += ["", "The rig played these recordings, from the `played` field of the end entry. A declared rule gives the levels that a "
                "repair or a drop leaves, and the change of the hidden combination switches the served recording to the stable block of "
                "the new combination, so every reading after it is a recorded cycle and no reading is simulated.", ""]
        out += formatting.table(["Cycles", "Combination", "Recorded cycles served"], played_rows(played, ep.components))
    thinking = sum(e.get("think_cycles") or 0 for e in log)
    mode = ep.config["clock_mode"]
    if mode == "generation_time":
        seconds = sum(ep.state.get("think_seconds") or [])
        charge = (f", one cycle for every {formatting.num(ep.config['think_seconds_per_cycle'])} seconds of the {formatting.num(seconds)} seconds between the "
                  f"calls, which add up from call to call, with {formatting.num(ep.state.get('think_seconds_carried'))} seconds carried past the last call.")
    elif mode == TOKEN_MODE:
        tokens = sum(ep.state.get("think_tokens") or [])
        seconds = sum(ep.state.get("think_seconds") or [])
        charge = (f", one cycle for every {formatting.num(ep.config['think_seconds_per_cycle'])} seconds of the {formatting.num(seconds)} seconds that the "
                  f"{formatting.num(tokens)} tokens the agent's model generated between the calls stand for at {formatting.num(ep.config['seconds_per_token'])} "
                  f"seconds per token, which add up from call to call, with {formatting.num(ep.state.get('think_seconds_carried'))} seconds carried past "
                  "the last call.")
    else:
        charge = f", since the {mode} clock mode charges no thinking time."
    out += ["", f"The log charges {formatting.plural(thinking, 'thinking cycle')} to the agent's actions, its `think_cycles`" + charge]
    return out + [""]


def start_row(reset, played, components):
    schedule = reset.get("schedule") or []
    detail = [f"start combination {formatting.long_levels(reset.get('hidden'), components)}"]
    if played and played[0].get("cycle") is not None:
        detail.append(f"first recorded cycle served {played[0]['cycle']}")
    detail.append("scheduled drops " + formatting.joined(f"of the {d['component']} to level {d['to_level']} at cycle {d['t']}" for d in schedule)
                  if schedule else "no scheduled drop")
    return ("0", "start", ", ".join(detail))


def hidden_rows(log, told, created, components):
    """The rows of part 3 after the start, one for each entry the agent was not told of in full. The hidden combination is
    followed from the start through every close, expiry and drop, so a close that left it as it was, such as an unnecessary
    order on a healthy component, says so instead of naming a switch of recordings."""
    rows, combination = [], list(log[0].get("hidden") or [])
    for entry in log[1:]:
        before, combination = combination, combination_after(entry, combination, components)
        row = None if json.dumps(entry, sort_keys=True) in told else hidden_row(entry, created, components, before)
        if row:
            rows.append(row)
    return rows


def combination_after(entry, combination, components):
    """The hidden combination after a log entry, the one a close or an expiry records or the one before with a drop's new level."""
    if entry.get("combination_after") is not None:
        return list(entry["combination_after"])
    if entry["event"] in ("scheduled_drop", "hazard_drop"):
        return [entry["to_level"] if c == entry["component"] else level for c, level in zip(components, combination)]
    return combination


def hidden_row(entry, created, components, before):
    """One row of part 3 for a log entry that holds something the agent never saw, else None. before is the hidden
    combination before the entry."""
    event, t = entry["event"], entry.get("t_end", entry.get("t_start"))
    if entry.get("step") is not None:
        if event == "work_order_created" and entry.get("unnecessary"):
            return (formatting.num(t), f"{entry['id']} created on the {entry['component']}",
                    "the component was at its healthy level, so the order is unnecessary")
        return None
    if event == "scheduled_drop":
        return (formatting.num(t), f"scheduled drop of the {entry['component']}",
                f"level {formatting.num(entry.get('from_level'))} to {formatting.num(entry.get('to_level'))}, the rig switched to the recordings of the new combination")
    if event == "work_order_closed":
        order = created.get(entry["id"])
        necessity = "" if order is None else f"the order was {'unnecessary' if order.get('unnecessary') else 'necessary'}, and "
        after = entry.get("combination_after")
        change = (f"the hidden combination stayed {formatting.long_levels(after, components)}, so the rig went on with the same recordings"
                  if after is not None and after == before else f"the rig switched to the recordings of {formatting.long_levels(after, components)}")
        return (formatting.num(t), f"{entry['id']} closed on the {entry['component']}", necessity + change)
    if event == "failure":
        return (formatting.num(t), f"failure of the {entry.get('component')}", f"penalty {formatting.num(entry.get('cost'))}, the agent learned the cycle but not the component")
    if event == "end":
        return (formatting.num(t), f"end by {entry.get('reason')}", f"final combination {formatting.long_levels(entry.get('final'), components)}")
    if event in ("reset", "work_order_executed"):
        return None
    fields = {k: v for k, v in entry.items() if k not in ("event", "step", "t_start", "t_end", "class")}
    return (formatting.num(t), event, ", ".join(f"{k} {json.dumps(v)}" for k, v in fields.items()))


def played_rows(played, components):
    segments = []
    for p in played:
        key = (tuple(p.get("combination") or []), p.get("cycle") is None)
        if segments and segments[-1]["key"] == key and p["t"] == segments[-1]["last"] + 1:
            segments[-1]["last"] = p["t"]
            segments[-1]["cycles"].append(p.get("cycle"))
        else:
            segments.append({"key": key, "first": p["t"], "last": p["t"], "combination": p.get("combination"), "cycles": [p.get("cycle")]})
    rows = []
    for s in segments:
        span = str(s["first"]) if s["first"] == s["last"] else f"{s['first']} to {s['last']}"
        cycles = s["cycles"]
        if s["key"][1]:
            served = "none, the rig was stopped for a repair"
        elif cycles == list(range(cycles[0], cycles[0] + len(cycles))):
            served = str(cycles[0]) if len(cycles) == 1 else f"{cycles[0]} to {cycles[-1]}"
        else:
            served = ", ".join(map(str, cycles[:6])) + (f" and {len(cycles) - 6} more" if len(cycles) > 6 else "")
        rows.append((span, formatting.long_levels(s["combination"], components), served))
    return rows
