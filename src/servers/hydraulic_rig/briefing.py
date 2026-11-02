"""The question that opens an episode (docs/hydraulic_rig/action-server.md, section The briefing).

The planner, the argument filler and the summarizer of AssetOpsBench's Plan-Execute runner all read the question, so at
guidance level 2 it carries the legal value of every tool parameter. Level 1 states only the asset, the date when the
episode is anchored and the objective, in the form of an IndustryAssetEQA question, and leaves the rules to
get_asset_profile; level 3 adds a procedure. The wording is part of the environment definition: change BRIEFING_VERSION
with it, since every run records that version.
"""
BRIEFING_VERSION = "v2.0"
ACTION_EVIDENCE_FORMS = "obs/<n>/<field> for a served value, wo/<id> for a work order, alarm/<t>/<channel> for an alarm of the change summary, fmea/<source>/<code> for a failure mode"
PREDICTION_EVIDENCE_FORM = "pred/<n> for a prediction of the diagnostics server"
EVIDENCE_FORMS = f"{ACTION_EVIDENCE_FORMS}, {PREDICTION_EVIDENCE_FORM}"


def briefing(profile, config):
    """Compose the question from the asset profile and the episode configuration at config.guidance_level."""
    parts = [_level1(profile, config)]
    if config.guidance_level >= 2:
        parts.append(_level2(profile, config))
    if config.guidance_level >= 3:
        parts.append(_level3(config))
    return "\n\n".join(parts)


def _level1(profile, config):
    asset = profile.get("asset") or {}
    described = f"{asset['asset_type']}, {asset['description']}, at site {asset['site']}" if asset else "an industrial asset"
    when = f" On {config.date_of(0)}," if config.anchors is not None else ""
    question = config.question or (
        f"Which condition level is each of its components ({', '.join(profile['components'])}) at, and which work orders "
        f"restore the unit, within a budget of {profile['budget']} and a horizon of {profile['horizon']} cycles?")
    return (f"Asset {profile['asset_id']} ({described}).{when} The performance monitor flagged the unit. {question} "
            "Answer through the tools: submit exactly one diagnosis, or one declaration of insufficient evidence, "
            "create the work orders that restore the unit, and finish with a summary.")


def _level2(profile, config):
    prices, durations, rate = profile["prices"], profile["durations"], profile["downtime_rate"]
    levels = "; ".join(f"{c}: {', '.join(map(str, profile['levels'][c]))} (healthy {profile['healthy'][c]})" for c in profile["components"])
    groups = "\n".join(f"  - {g} costs {prices['read'][g]} per cycle and returns {', '.join(fields)}" for g, fields in profile["sensor_groups"].items())
    parts = ", ".join(f"{c} {p}" for c, p in prices["parts"].items())
    lead = ", ".join(f"{c} {durations['lead_time'][c]}" for c in profile["components"])
    down = ", ".join(f"{c} {durations['repair_downtime'][c]}" for c in profile["components"])
    rates = "; ".join(f"{c}: " + ", ".join(f"level {lv} {r}" for lv, r in rate[c].items()) for c in profile["components"])
    statistics = " and ".join(profile["statistics"])
    text = f"""Condition levels: {levels}. Rule: {_fault_count_rule(profile['fault_rule'])}.

Tools of the action server (class, duration in cycles, price):
- get_asset_profile() (read, 0, free) returns these rules, the healthy bands of the channel means and the tool table.
- read_sensors(sensor_group, cycles) (read, {durations['read_per_cycle']} cycle per cycle read, price per cycle by group) returns the {statistics} of every channel of the group for each recorded cycle, each value named <channel>_<statistic>:
{groups}
- advance_time(cycles) (read, the cycles advanced, free) lets the rig run; downtime accrues meanwhile.
- summarize_changes_since(cycle) (read, 0, free) returns the cycles elapsed, the channel means that left their healthy band since that cycle with the first such cycle, and the work orders closed since then.
- get_failure_modes(component) (read, 0, free) returns the failure mode codes of a component, or of all components for an empty string, with the channels that indicate the component.
- submit_diagnosis(cooler, valve, pump, accumulator, evidence_ids) (write, 0, free) records one condition level per component; accepted once, and never after a declaration.
- declare_insufficient_evidence(sensor_group, cycles, evidence_ids) (write, 0, free) records that the evidence cannot separate the remaining candidates and names the reading that would; accepted once, and never after a diagnosis. Reads and work orders stay allowed afterward, a diagnosis does not.
- create_work_order(component, prediction_id, evidence_ids) (write, closes after the lead time and the repair downtime, part price plus the repair charge of {prices['repair_charge']}) asks a technician to restore the component to its healthy level. Parts: {parts}. Lead times: {lead}. Repair downtimes: {down}. The rig is stopped during the repair downtime at a downtime rate of {rate['stopped']} per cycle, and the order closes at the creation time plus the lead time plus the repair downtime. A work order is refused before a diagnosis or a declaration, and while an order on the same component is open. An order on a healthy component is executed and counted as unnecessary.
- finish(summary) (write, 0, free) ends the episode; open work orders stay open.
evidence_ids is a comma separated list of identifiers, possibly empty: {_evidence_forms(config)}. {_prediction_sentence(config)}

Downtime accrues per cycle while a component is below its healthy level ({rates}) and counts against the budget with the prices. Budget: {profile['budget']}. Horizon: {profile['horizon']} cycles. Step cap: {profile['max_steps']} tool calls, refused ones included. {_clock_sentence(profile, config)} The episode ends on finish, at the horizon, at the step cap, when an action would exceed the budget{_failure_sentence(config)}. A refused call costs nothing but counts as a step, and a call with a missing argument or an argument of the wrong type is refused."""
    if config.diagnostics_server:
        text += ("\n\nA diagnostics server offers get_episode_evidence(sensor_group), rank_root_causes(top_k), "
                 "estimate_intervention_effect(action) and verify_outcome(prediction_id). They read the evidence released so far and change nothing.")
    return text


def _level3(config):
    steps = ["read sensors before diagnosing", ]
    if config.diagnostics_server:
        steps += ["rank the candidates with rank_root_causes after each reading",
                  "declare insufficient evidence with the separating test the ranking names when it reports the evidence as insufficient"]
    steps += ["create one work order per diagnosed fault and advance time until the orders close",
              "cite the evidence identifiers behind every diagnosis and work order",
              "finish with a summary of what was found and done"]
    return "Procedure: " + "; ".join(f"{i}. {s}" for i, s in enumerate(steps, start=1)) + "."


def _evidence_forms(config):
    return EVIDENCE_FORMS if config.diagnostics_server else ACTION_EVIDENCE_FORMS


def _prediction_sentence(config):
    if config.diagnostics_server:
        return "prediction_id is the identifier of a prediction of the diagnostics server, or empty."
    return "prediction_id is empty in this episode."


def _clock_sentence(profile, config):
    if config.clock_mode == "instant":
        return "Every tool call advances the clock by one cycle."
    sentence = "Time advances only inside tool calls, by their declared durations"
    if config.clock_mode == "generation_time":
        sentence += f", and by your response time at {profile['seconds_per_cycle']} seconds per cycle"
    return sentence + "."


def _failure_sentence(config):
    rule = config.failure
    text = (f"; a component at its worst level for {rule['patience']} cycles without an open work order is a failure "
            f"that adds {rule['penalty']} to the cost")
    return text + (" and ends the episode" if rule["strict"] else "")


def _fault_count_rule(rule):
    """The public fault-count rule in words."""
    low, high = rule["min_faulty"], rule["max_faulty"]
    if low < high:
        return f"between {low} and {high} components start off their healthy level"
    if low == 1:
        return "exactly 1 component starts off its healthy level"
    return f"exactly {low} components start off their healthy level"
