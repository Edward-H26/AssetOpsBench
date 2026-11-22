"""The levels of the answers of the diagnostics tools, the parameter response_format (main.py). raw is the answer of CausalTools
as FastMCP writes it, unchanged. detailed, concise and headline are written from the full answer and the public view of the
asset alone, so every value they hold is that of the full answer: the ranking, the advice, the plans, the predictions and the
verdicts. detailed keeps every value but the anomaly score of each value of the evidence and the channels of the attribution
after the first TOP_CHANNELS of each component, which raw alone holds, and shortens what every call repeats. An assumption text
becomes its key (tool_texts.assumption_key), a list of evidence identifiers becomes the few that a reader cites, with the
readings and the rule that name the rest, and the items of the evidence become one long table with the unit, the rate and the
normal range of each field once. concise keeps what an agent acts on and headline one line. Their text is JSON without
indentation, since indentation adds characters that carry no value, and a repeated answer opens with the key unchanged, which
names the call that first returned it."""
from CausalDiagnosis.mcp_servers.utils.response_levels import DEFAULT_LEVEL, LEVELS, first_step, level_error, level_of, marked, text_of  # noqa: F401  the levels every server shares
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import tool_texts

TOP_CHANNELS = 3  # the channels of each component that the attribution of a shorter ranking keeps, in the order of the attribution
# the one line on the levels in every tool description, which names what detailed leaves to raw
LEVELS_LINE = (" response_format is headline (one line), concise (the default: what an agent acts on), detailed (every value in compact "
               f"form but the anomaly score of each evidence value and the attribution channels after the first {TOP_CHANNELS} of each "
               "component) or raw (the whole answer, with those values, every evidence ID and the sentence of every assumption, which "
               "the other levels name by key).")
# what a detailed ranking holds and concise leaves out: the bookkeeping of the filter, the level distributions, the stability,
# the expected losses behind the advice, the Monte Carlo settings and the earlier rule
RANKING_DETAIL_ONLY = ("n_candidates", "n_states", "resampled_fits", "level_probability", "stability", "expected_saving", "expected_loss_act",
                       "expected_loss_test", "monte_carlo", "e10_rule")
# the name of the combination that the readers of version 6 use; evaluation/metrics/mcp_metrics.py reads the combination of a
# branch, so the branch keeps its combination and its faults
CANDIDATE_REPEATS = BRANCH_REPEATS = ("current_combination",)


# ----- the evidence identifiers -----
def _reading_fields(ids):
    """{n: the fields that the obs identifiers name for reading n}, in the order of the identifiers."""
    fields = {}
    for i in ids:
        if i.startswith("obs/"):
            _, n, field = i.split("/", 2)
            fields.setdefault(n, []).append(field)
    return fields


def readings(ids, view):
    """{n: the sensor group whose summary fields are those that the identifiers name for reading n, or those fields where no
    group's are}, as a released reading holds the fields of its group but a read of some sensors holds only theirs."""
    groups = {}
    for group in sorted(view.sensor_groups):
        groups.setdefault(frozenset(view.group_fields(group)), group)
    return {n: groups.get(frozenset(fields), fields) for n, fields in _reading_fields(ids).items()}


def kept_ids(ids, top_fields=(), preferred=()):
    """The identifiers a shorter level keeps of ids, in their order: each field of top_fields at the newest reading that holds
    it, of every reading that holds none of those its first identifier among preferred or else its first, and every identifier
    that names no reading. Every reading and every closed work order keeps an identifier, so a reader that takes the readings
    and the orders an answer saw from its evidence_ids (evaluation/metrics/realized_read_elimination.forecasts) reads the same
    from every level."""
    fields = _reading_fields(ids)
    newest = {}
    for n, names in fields.items():
        for field in names:
            newest[field] = max(newest.get(field, n), n, key=int)
    kept = {f"obs/{newest[f]}/{f}" for f in top_fields if f in newest}
    covered = {i.split("/", 2)[1] for i in kept}
    for n, names in fields.items():
        if n not in covered:
            named = [f"obs/{n}/{f}" for f in names]
            kept.add(next((i for i in named if i in preferred), named[0]))
    return [i for i in ids if i in kept or not i.startswith("obs/")]


def _shortened(full, kept, rule, named=None, replaced=None):
    """The full answer in its order of keys with the kept identifiers, then the readings that name the others where given and
    the rule, the keys of its assumptions, and in place of each key of replaced the keys and values it maps to."""
    out = {}
    for key, value in full.items():
        if key == "evidence_ids":
            out[key] = kept
            if named is not None:
                out["readings"] = named
            out["evidence_rule"] = rule
        elif key == "assumptions":
            out[key] = [tool_texts.assumption_key(text) for text in value]
        elif replaced and key in replaced:
            out.update(replaced[key])
        else:
            out[key] = value
    return out


# ----- rank_root_causes -----
def _top_fields(ranking):
    """The first channels of the attribution of each faulty component of the top candidate, whose attribution it is."""
    faulty = {fault["component"] for fault in ranking["candidates"][0]["faults"]}
    return [channel["field"] for component in ranking.get("attribution") or [] if component["component"] in faulty
            for channel in component["channels"][:TOP_CHANNELS]]


def ranking(full, view, level):
    if level == "headline":
        top = full["candidates"][0]
        combination = "(" + ", ".join(f"{c} {v}" for c, v in top["combination"].items()) + ")"
        advice = full["advice"]["text"] if full.get("advice") else "none after a diagnosis or a declaration"
        # evaluation/metrics/mcp_metrics.py reads whether the last ranking before a decision said read first, and when its belief stood
        return {"headline": f"first candidate {combination} at {top['probability']:.3f}; advice: {advice}", "read_first": full["read_first"],
                "separating_test": full["separating_test"], "belief_time": full["belief_time"]}
    # a reading without a top channel keeps an identifier that bears on the top candidate where it has one, so the first
    # candidate cites a reading under branch_detail compact too, which computes no attribution
    kept = kept_ids(full["evidence_ids"], _top_fields(full), set(full["candidates"][0]["evidence_ids"]))
    keep = set(kept)
    candidates = [dict(c, evidence_ids=[i for i in c["evidence_ids"] if i in keep]) for c in full["candidates"]]
    if level == "concise":
        candidates = [{k: v for k, v in c.items() if k not in CANDIDATE_REPEATS} for c in candidates]
    replaced = {"candidates": {"candidates": candidates}}
    if full.get("attribution") is not None:
        replaced["attribution"] = {"attribution": [dict(a, channels=a["channels"][:TOP_CHANNELS]) for a in full["attribution"]]}
    if level == "concise":
        replaced.update({key: {} for key in RANKING_DETAIL_ONLY})
        replaced["branches"] = {"branches": [{k: v for k, v in b.items() if k not in BRANCH_REPEATS} for b in full["branches"]]}
    elif full.get("e10_rule") is not None:
        replaced["e10_rule"] = {"e10_rule": dict(full["e10_rule"], text=tool_texts.assumption_key(full["e10_rule"]["text"]))}
    return _shortened(full, kept, tool_texts.RANKING_EVIDENCE_RULE, readings(full["evidence_ids"], view), replaced)


# ----- get_episode_evidence -----
def _cell(item):
    """A value as the table shows it: the released value, ending in low or high where it lies outside its normal range."""
    if item["in_normal_range"] is False:
        return f"{item['value']!r} {'low' if item['value'] < item['normal_low'] else 'high'}"
    return repr(item["value"])


def _outside(cell):
    return cell is not None and cell.endswith((" low", " high"))


def _long_table(items):
    """(the readings in release order, one entry per field by name with its unit, rate and normal range once and its values
    across the readings, None where a reading holds no such field)."""
    columns, place = [], {}
    for item in items:
        if item["n"] not in place:
            place[item["n"]] = len(columns)
            columns.append({"n": item["n"], "t": item["t"], "sensor_group": item["sensor_group"]})
    fields = {}
    for item in items:
        entry = fields.setdefault(item["field"], {**{key: item[key] for key in ("field", "unit", "rate", "normal_low", "normal_high")
                                                     if item[key] is not None}, "values": [None] * len(columns)})
        entry["values"][place[item["n"]]] = _cell(item)
    return columns, [fields[f] for f in sorted(fields)]


def _by_first_group(columns, fields):
    """{sensor group: the count of the fields whose first reading is of that group}, by group name."""
    counts = {}
    for entry in fields:
        group = next(columns[k]["sensor_group"] for k, cell in enumerate(entry["values"]) if cell is not None)
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def evidence(full, view, level):
    columns, fields = _long_table(full["items"])
    outside = [entry for entry in fields if any(_outside(cell) for cell in entry["values"])]
    if level == "headline":
        groups = ", ".join(f"{g} {k}" for g, k in _by_first_group(columns, outside).items())
        return {"headline": f"{len(columns)} readings, {len(outside)} fields outside their normal range" + (f" ({groups})" if groups else "")
                            + f", {len(full['work_orders_closed'])} work orders closed"}
    cited = {f"obs/{column['n']}/{entry['field']}" for entry in outside for column, cell in zip(columns, entry["values"]) if _outside(cell)}
    kept = kept_ids(full["evidence_ids"], preferred=cited)
    if level == "detailed":
        return _shortened(full, kept, tool_texts.EVIDENCE_TABLE_RULE, replaced={"items": {"readings": columns, "fields": fields}})
    # a call for one channel or one field asks for its values, so concise narrows only all and a sensor group
    listed = outside if full["sensor_group"] == "all" or full["sensor_group"] in view.sensor_groups else fields
    table = {"readings": columns, "fields": listed, "not_listed": _by_first_group(columns, [entry for entry in fields if entry not in listed])}
    return _shortened(full, kept, tool_texts.CONCISE_TABLE_RULE, replaced={"items": table})


# ----- estimate_intervention_effect and verify_outcome -----
def effect(full, view, level):
    if level == "headline":
        return {"headline": full["message"]}
    replaced = {"predicted": {}} if level == "concise" else None
    return _shortened(full, kept_ids(full["evidence_ids"]), tool_texts.EFFECT_EVIDENCE_RULE, readings(full["evidence_ids"], view), replaced)


def at_level(tool, full, view, level):
    """The answer of a tool of the CausalDiagnosis MCP at a shorter level, from the model of its raw answer and the view of the
    asset, as main.answered writes it before the key unchanged: for a reader that shows a recorded raw answer at every level."""
    if tool == "verify_outcome":
        return verification(full, level)
    return {"get_episode_evidence": evidence, "rank_root_causes": ranking, "estimate_intervention_effect": effect}[tool](full, view, level)


def verification(full, level):
    if level == "headline":
        return {"headline": full["message"]}
    if level == "concise":
        return _shortened(full, kept_ids(full["evidence_ids"]), tool_texts.VERIFICATION_CONCISE_RULE, replaced={"channels": {}})
    return _shortened(full, kept_ids(full["evidence_ids"]), tool_texts.VERIFICATION_EVIDENCE_RULE)
