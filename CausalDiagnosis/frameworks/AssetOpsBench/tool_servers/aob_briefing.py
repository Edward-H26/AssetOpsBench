"""The question that opens an episode on the AssetOpsBench rows.

The agents of these rows read AssetOpsBench's own tool descriptions, whose examples name a chiller, so the question
names what those descriptions cannot: the site, the asset and the child asset of every component, the record fields of
the telemetry, the sensor list a history should carry, the citation forms of the evidence and the two time forms of the
documents. It keeps the levels of the rig's briefing. Level 1 states the asset and the objective with the scoring
sentence, the rig's sentence on the loop of the replayed cycles, which says where a cycle of the episode can run that the
cycles recorded before the episode are other recordings, the time sentence on the clock rows, the rig's sentence of every deterioration card the episode switches on
and, from the repair rules row on, the rig's sentences of the repair rules the episode switches on, with the manual and
the fix history named by the tools that carry them here. Level 2 adds the tools of every server on
the row with their prices and durations, the rig's misdiagnosis sentence in the submit_diagnosis entry where the price
table declares that price, the emergency clause with the scoring of a hold in force at the end where the
emergency mode is on, the receipt line from the priced row on, with the rule that an alarm is raised once per channel
where the receipt lists the notifications, and the limits. Level 3 adds a
procedure. No level states the fault rule of the starts. A sensor group that holds the fields of a smaller one is named by that
group and its other fields, so the question lists no field twice. The wording is part of the environment, so BRIEFING_VERSION
changes with it.
"""
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import (DECLARATION_RULE, DECLARE_STEP, DIAGNOSTICS_SERVER_SENTENCE, HOLD_SCORING, _finish_line, _limits_paragraph, _lookup_line, _notified_events,
                                            _repair_sentences, alarm_once_sentence, asset_text, deterioration_sentences, level_line,
                                            misdiagnosis_sentence, n_cycles, opening_sentence, question_sentence, readings_sentence,
                                            repair_cards, repair_sentences, repair_step, scoring_sentence, time_sentence, time_values)
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import listed as _listed

BRIEFING_VERSION = "aob-v1.19"
EVIDENCE_FORMS = ("obs/<timestamp>/<field> for a value of a telemetry row, with the timestamp and the field as the row shows them",
                  "wo/<wonum> for a work order", "fmea/<source>/<code> for a failure mode")
ALARM_FORM = "alarm/<cycle>/<channel> for the alarm of a notification"
PREDICTION_FORM = "pred/<n> for a prediction of the diagnostics server"
ALARM_ONCE = alarm_once_sentence("only a new reading of the telemetry")
WRITTEN = {"std": "standard deviation"}  # the written form of a statistic in the sentence on the telemetry fields, else its name


def fix_history_sentence(noun):
    """The sentence of the fix history card on these rows, where the wo server carries it, for an asset whose short noun is
    noun."""
    return f"The fix history, list_workorders of the wo server, lists the work orders closed on this {noun} before the episode."


def _serves(config, tool):
    """Whether the scenario layers list this environment tool, independently of the separate diagnostic advice server."""
    return tool in layer_contract.tools_for(TASK_SERVER, config)


def _repair_rules_on(config):
    """Whether the repair rules play on the row: the layer rule of the manual's row in the contract (layers.lists_tools), which
    holds on a row with a clock where a repair rule is on, whatever tools the row serves."""
    return layers.lists_tools(layer_contract.entries(TASK_SERVER)["lookup_fix"]["row"], config)


def briefing(profile, config):
    """Compose the question from the asset profile of the engine of the rows and the configuration, at
    config.guidance_level."""
    row = layer_contract.row_of(config)
    parts = [_level1(profile, config)]
    if config.guidance_level >= 2:
        parts.append(_level2(profile, config, row))
    if config.guidance_level >= 3:
        parts.append(_level3(profile, config))
    return "\n\n".join(parts)


def _children(profile):
    return [layer_contract.child_asset(profile["asset_id"], c) for c in profile["components"]]


def _level1(profile, config):
    asset = profile.get("asset") or {}
    site = asset.get("site", "")
    text = asset_text(profile)
    unit, noun = text["unit_noun"], text["short_noun"]
    return (f"Asset {profile['asset_id']} ({asset.get('asset_type')}, {asset.get('description')}, at site {site}), with one child "
            f"asset per component: {_listed(_children(profile))}. {opening_sentence(profile, unit)} {question_sentence(profile, config, unit)} Answer "
            f"through the tools: read the telemetry of {profile['asset_id']} with the iot server, submit exactly one diagnosis, or "
            f"one declaration of insufficient evidence, with the task server, create the work orders that restore the {unit} with "
            "generate_work_order of the wo server, and finish with a summary. "
            + " ".join((scoring_sentence(profile), readings_sentence(profile, unit, _cycles_run(config)), _time_sentence(config, noun),
                        *deterioration_sentences(config), *_repair_rule_sentences(config, noun))))


def _cycles_run(config):
    """Whether a cycle of the episode can run, so that the telemetry holds live cycles beside the cycles recorded before the
    episode: on a row with a clock, and on a row without one whose task server lists the wait."""
    return config.has_clock() or "wait_for_notification" in layer_contract.tools_for(TASK_SERVER, config)


def _time_sentence(config, noun):
    """The clock of the episode, with the two time tools where the configuration lists them."""
    waits = "get_current_time" in layer_contract.tools_for(TASK_SERVER, config)
    if config.has_clock():
        return time_sentence(noun) if waits else "Every answer lists the notifications of the public events since the previous call."
    closes = ("the work orders created one after another close together before the next call that creates none"
              if config.repair_card_on() else "a work order closes as soon as it is created")
    if waits:
        return (f"No cycle runs in this episode except during wait_for_notification, so the telemetry holds the cycles recorded "
                f"before it started until a wait lets the {noun} run, and {closes}.")
    return f"No cycle runs in this episode, so the telemetry holds the cycles recorded before it started, and {closes}."


def _repair_rule_sentences(config, noun):
    """The rig's sentences of the repair rules the episode switches on, wherever they play, with the fix history named by
    the tool that carries it on these rows; the sentence of the manual only where the task server lists it."""
    if not _repair_rules_on(config):
        return []
    sentences = _repair_sentences(config, noun)
    if not _serves(config, "lookup_fix"):
        sentences = sentences[1:]  # the first is the manual's
    return [fix_history_sentence(noun) if s == repair_sentences(noun)["fix_history"] else s.replace("lookup_fix,", "lookup_fix of the task server,")
            for s in sentences]


def _level2(profile, config, row):
    table = {t["name"]: t for t in profile["tools"]}
    rates = "; ".join(f"{c}: " + ", ".join(f"level {lv} {r}" for lv, r in profile["downtime_rate"][c].items()) for c in profile["components"])
    text = f"""{level_line(profile)}

{_telemetry_paragraph(profile, config)}

Tools by server (class, duration in cycles, price):
{_iot_lines(profile, table)}
{_fmsr_line(profile)}
{_wo_lines(profile, config, table)}
{_task_lines(profile, config)}
evidence_ids of the task tools is a comma separated list of identifiers, possibly empty, and aob_source.evidence_ids a list of them: {_evidence_forms(config)}. {_prediction_sentence(config)}{_receipt_sentence(config, row)}

{_limits_paragraph(profile, config, rates)}"""
    if config.diagnostics_server:
        text += "\n\n" + DIAGNOSTICS_SERVER_SENTENCE
    return text


def _fields_sentence(statistics):
    """Where a telemetry row keeps each served statistic of a channel: the plain one, listed first, under the installed
    sensor name, and every further one under that name with the statistic appended."""
    plain, *rest = statistics
    places = [f"the {WRITTEN.get(plain, plain)} under the installed sensor name",
              *(f"the {WRITTEN.get(s, s)} under that name with _{s} appended" for s in rest)]
    return f"for every channel, {_listed(places)}"


def _telemetry_paragraph(profile, config):
    prices = profile["prices"]["read"]
    noun = asset_text(profile)["short_noun"]
    priced = any(p > 0 for p in prices.values())
    if priced:
        intro = ("history and latest_reading charge every cycle they return at the price per cycle of the cheapest sensor group whose "
                 "fields cover the fields returned, unless a group paid earlier for that cycle covers them, so reading a paid row again is free:")
        groups = [f"  - {g} costs {prices[g]} per cycle and returns {returned}" for g, returned in _group_fields(profile["sensor_groups"]).items()]
    else:
        intro = "Every read is free in this episode. The sensor groups, which a declaration of insufficient evidence names, are:"
        groups = [f"  - {g} returns {returned}" for g, returned in _group_fields(profile["sensor_groups"]).items()]
    if not config.has_clock():
        window = ""
    elif profile["durations"]["read_per_cycle"] > 0:
        window = (f" A history window that reaches cycles not yet recorded first lets the {noun} run them, "
                  f"{n_cycles(profile['durations']['read_per_cycle'])} each, and returns and charges them.")
    else:
        window = " A window past the present returns the rows recorded so far, and new rows appear as the clock runs."
    return (f"Telemetry. A row of {profile['asset_id']} holds one recorded cycle: its timestamp and, {_fields_sentence(profile['statistics'])}. "
            "Name the sensors you need in history, "
            "since a history without a sensor list returns every field of every row. Row timestamps carry no time zone and history "
            "bounds take the same form, while work order dates carry the offset +00:00." + window + "\n" + intro + "\n" + "\n".join(groups))


def _group_fields(sensor_groups):
    """{group: the fields it returns in words}, a group that holds every field of an earlier one named by that group and its other
    fields, the largest such group, so that a nested group does not list the fields of the one inside it again."""
    out, earlier = {}, []
    for group, fields in sensor_groups.items():
        inner = max((g for g in earlier if set(sensor_groups[g]) < set(fields)), key=lambda g: len(sensor_groups[g]), default=None)
        rest = [f for f in fields if inner is None or f not in sensor_groups[inner]]
        out[group] = ", ".join(rest) if inner is None else f"the fields of {inner} and {', '.join(rest)}"
        earlier.append(group)
    return out


def _iot_lines(profile, table):
    free = [n for n, e in layer_contract.entries("iot").items() if e["action"] == "aob_read"]
    site, asset = (profile.get("asset") or {}).get("site", ""), profile["asset_id"]
    return (f"- iot, with site_name {site} and asset_id {asset}: {_listed(free)} (read, 0, free) describe the registry and the recorded "
            f"stream. history(site_name, asset_id, start, end, sensors, limit, cursor) (read, {_read_duration(table)}, "
            f"{_read_price(table)}) returns the recorded rows of a window, and latest_reading(site_name, asset_id, sensor) (read, 0, "
            f"{_read_price(table)}) returns the newest recorded row.")


def _read_duration(table):
    return "the cycles it plays" if table["history"]["duration_value"] else "0"


def _read_price(table):
    return "price per cycle by group" if any(v > 0 for v in table["history"]["price_value"].values()) else "free"


def _fmsr_line(profile):
    return (f"- fmsr: get_failure_modes(asset_class) (read, 0, free) returns the failure mode codes of a component class, "
            f"{_listed(profile['components'], 'or')}, with their source.")


def _wo_lines(profile, config, table):
    reads = [n for n, e in layer_contract.entries("wo").items() if e["action"] == "aob_read"]
    site = (profile.get("asset") or {}).get("site", "")
    order = table["generate_work_order"]["price_value"]
    priced = any(v > 0 for v in order["standard"].values()) or order["visit"] > 0
    prices = profile["prices"]
    price = (f"part price plus the repair charge of {prices['repair_charge']}" + (f", plus a visit charge of {order['visit']} for the "
             "order that opens a visit" if order["visit"] > 0 else "") if priced else "free")
    parts = f" Parts: {', '.join(f'{c} {p}' for c, p in prices['parts'].items())}." if priced else ""
    return (f"- wo, with site_id {site}: {_listed(reads)} (read, 0, free) read the work orders of this episode and the failure code "
            f"catalog. generate_work_order(description, asset_num, site_id, priority, work_type, reported_by, location, notes, wonum, "
            f"aob_source) (write, 0, {price}) asks a technician to restore the component of a child asset to its healthy level, with "
            "work_type CM and an empty wonum, since the environment numbers the orders wo-1 and up, and aob_source an object with the "
            f"prediction_id and the evidence_ids behind the order.{_emergency_clause(config)}{parts} {_order_close(profile, config)} A work order is refused "
            "before a diagnosis or a declaration, and while an order on the same component is open. An order on a healthy component is "
            "executed and counted as unnecessary.")


def _emergency_clause(config):
    """work_type EM, the emergency mode of the repair rules, while its card is on and the repair rules play."""
    if not _repair_rules_on(config) or "emergency" not in repair_cards(config):
        return ""
    e = config.emergency
    return (f" With work_type EM the order is an emergency order, which forms a visit of its own, arrives {n_cycles(e['lead'])} after "
            f"its creation at {e['surcharge']} times the part and repair price, and holds the component at its healthy level for "
            f"{n_cycles(e['hold'])} after its close. {HOLD_SCORING}")


def _order_close(profile, config):
    noun = asset_text(profile)["short_noun"]
    if not config.has_clock():
        return ("The work orders created one after another close together before the next call that creates none."
                if config.repair_card_on() else "It closes as soon as it is created.")
    durations = profile["durations"]
    if not any(durations["lead_time"].values()) and not any(durations["repair_downtime"].values()):
        return f"It closes at the next cycle the {noun} plays."
    lead = ", ".join(f"{c} {durations['lead_time'][c]}" for c in profile["components"])
    down = ", ".join(f"{c} {durations['repair_downtime'][c]}" for c in profile["components"])
    return (f"The technician arrives after the lead time and repairs while the {noun} stands still, then the order closes. Lead times: "
            f"{lead}. Repair downtimes: {down}.")


def _task_lines(profile, config):
    components, noun = ", ".join(profile["components"]), asset_text(profile)["short_noun"]
    lines = [f"- task: get_asset_profile() (read, 0, free) returns these rules, the child assets, the normal ranges of the channel {profile['statistics'][0]}s "
             "under the record fields and the tool table."]
    listed = layer_contract.tools_for(TASK_SERVER, config)
    if "get_current_time" in listed:
        budget = ", the budget left" if config.budget is not None else ""
        lines.append(f"  get_current_time() (read, 0, free) returns the cycle, its datetime in the form of the row timestamps, the horizon"
                     f"{budget}, {time_values(config)} without moving the clock. wait_for_notification(timeout) (read, "
                     "the cycles waited, "
                     f"free) lets the {noun} run until the next notification or for timeout cycles, the cycles left before the horizon when it "
                     "is empty, and returns what ended the wait.")
    if "lookup_fix" in listed:
        lines.append(f"  {_lookup_line(profile['prices'], config)}")
    if "summarize_changes_since" in listed:
        lines.append(f"  summarize_changes_since(cycle) (read, 0, free) returns the cycles elapsed since a past cycle, the channel "
                     f"{profile['statistics'][0]}s of the sensor groups already read in this episode that left their normal range, "
                     "with the first such cycle, and the work orders closed since then.")
    diagnosis = " ".join(part for part in (f"submit_diagnosis({components}, evidence_ids) (write, 0, free) records one condition level "
                                           "per component, accepted once and never after a declaration.",
                                           misdiagnosis_sentence(profile["prices"])) if part)
    lines.append(f"  {diagnosis} declare_insufficient_evidence(sensor_group, cycles, evidence_ids) (write, 0, free) records that the "
                 "evidence cannot separate the remaining candidates and names the reading that would, accepted once and never after a "
                 f"diagnosis, and work orders stay allowed afterward. {DECLARATION_RULE} {_finish_line(config, noun)}")
    return "\n".join(lines)


def _evidence_forms(config):
    forms = [*EVIDENCE_FORMS, *([ALARM_FORM] if config.has_clock() else []),
             *([PREDICTION_FORM] if config.diagnostics_server else [])]
    return _listed(forms)


def _prediction_sentence(config):
    if config.diagnostics_server:
        return "aob_source.prediction_id is the identifier of a prediction of the diagnostics server, or empty."
    return "aob_source.prediction_id is empty in this episode."


def _receipt_sentence(config, row):
    if row == "base":
        return ""
    if row == "priced":
        return "\nEvery answer ends with one JSON line with the price of the call, the budget left and the steps left."
    wait = ", then the timeout of a wait that reached it and the end" if _serves(config, "wait_for_notification") else ", then the end"
    return ("\nEvery answer ends with one JSON line with the price of the call, the budget left, the steps left, the cycle and the "
            f"notifications since the previous call of {_notified_events(config)}{wait} of the episode. {ALARM_ONCE}")


def _level3(profile, config):
    steps = [f"read the telemetry of {profile['asset_id']} with history and the sensors of one sensor group"]
    if config.diagnostics_server:
        steps += ["rank the candidates with rank_root_causes after each reading",
                  "while the ranking says read first, take the reading it names before you diagnose"]
    steps += [DECLARE_STEP,
              repair_step(config, " on the child asset of its component", manual=_serves(config, "lookup_fix"),
                          waits=_serves(config, "wait_for_notification")),
              "cite the evidence identifiers behind every diagnosis, declaration and work order",
              "finish with a summary of what was found and done"]
    return "Procedure: " + "; ".join(f"{i}. {s}" for i, s in enumerate(steps, start=1)) + "."
