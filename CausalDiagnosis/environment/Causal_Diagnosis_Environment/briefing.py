"""The question that opens an episode (CausalDiagnosis/docs/environment-actions-mcp.md, section The briefing).

The planner, the argument filler and the summarizer of AssetOpsBench's Plan-Execute runner all read the question, so at
guidance level 2 it carries the legal value of every tool parameter. Level 1 states the asset, the date when the episode
is anchored and the objective, in the form of an IndustryAssetEQA question, states that a diagnosis is scored at the
cycle of its write, states that the readings of unchanged condition levels repeat with the length of the live window,
names the two time tools and the notifications in one sentence, names the scheduled and the random
deterioration transitions in one sentence each when the episode switches their card on, names the repair manual, the fix
history, the emergency mode, the dependency rule and the visit charge in one sentence each when the episode switches that
rule on, and leaves the rules to get_asset_profile; level 2 states the rules of the cards the episode switches on and no
rule of a card that is off, such as the downtime of the base row, the rule that an alarm is raised once per channel
and, where the emergency mode is on, the scoring of a hold in force at the end; level 3 adds a procedure. The wording is part of the environment
definition, so BRIEFING_VERSION changes with it, since every run records that version. The component names and the nouns
come from the asset profile, the hydraulic rig's for hyd_1, and every helper that the briefing of the AssetOpsBench rows
quotes takes the noun of the asset. An asset file may state its own question, opening, scoring sentence, meaning of the
levels and sentence on its readings, and the helpers below give each of them in place of the sentence of the default
form, which the rig keeps.
"""
from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET, STOPPED
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_5_deterioration_and_failure as deterioration, layer_2_real_time as real_time, layer_4_repair_rules as repair_rules

BRIEFING_VERSION = "v2.20"
# the wall limit of the agent's part of an episode, which the driver enforces for every agent and both briefings state among the
# ends of an episode (CausalDiagnosis/experiments/table_definitions/experiment_tables.py reads it from here, so the rule has one value)
EPISODE_WALL_LIMIT_S = 1800
# the same sentence in the tool description and both briefings, so that every row of the environment states one rule
DECLARATION_RULE = "It must cite at least one reading already read, and a declaration that cites none is scored as a failure."
ACTION_EVIDENCE_FORMS = ("obs/<n>/<field> for a served value, wo/<id> for a work order, alarm/<t>/<channel> for an alarm of the change summary, "
                         "fmea/<source>/<code> for a failure mode, hist/<n> for a past work order of the fix history")
PREDICTION_EVIDENCE_FORM = "pred/<n> for a prediction of the diagnostics server"
EVIDENCE_FORMS = f"{ACTION_EVIDENCE_FORMS}, {PREDICTION_EVIDENCE_FORM}"
SCORING_SENTENCE = ("The diagnosis states the condition level of each component at the cycle it is submitted, and it is scored "
                    "against the hidden condition at that cycle.")
# the scoring of a hold in force at the end, in both briefings where the emergency mode is on and in the description of finish
HOLD_SCORING = "A component that an emergency order still holds when the episode ends is scored at the level it had before that order."
# the sentence of level 2 on the diagnostics server, in both briefings where the episode registers it
DIAGNOSTICS_SERVER_SENTENCE = ("A diagnostics server offers get_episode_evidence(sensor_group), rank_root_causes(top_k), "
                               "estimate_intervention_effect(action) and verify_outcome(prediction_id). They read the evidence released "
                               "so far and change nothing.")
# the scoring of an end at the wall limit while the agent's own work runs, in both briefings after the ends of an episode
WALL_END_SENTENCE = ("If the wall limit ends the episode while every call is answered or while your reply is being written, the "
                     "episode ends there and is scored.")
# the declaration step of the level 3 procedure in both briefings, in the words of the entry of declare_insufficient_evidence
DECLARE_STEP = ("when the evidence cannot separate the remaining candidates, declare insufficient evidence with "
                "declare_insufficient_evidence, naming the reading that would separate them, instead of diagnosing")


def time_sentence(noun):
    """The sentence on the two time tools and the notifications, for an asset whose short noun is noun."""
    return (f"get_current_time reads the clock without moving it, wait_for_notification lets the {noun} run until the next "
            "notification or a timeout, and every answer lists the notifications of the public events since the previous call.")


def idle_turn_sentence(noun, diagnostics_server=False):
    """The step that a turn without a call takes and the end such turns bring, in both briefings beside the step cap and in the
    words of the system prompt of the CausalDiagnosis agent, for an asset whose short noun is noun. Where the episode registers
    the diagnostics server the sentence opens with its rule: a call of that server reaches no tool of the asset, so it takes
    no step, and a turn that calls it alone still takes one."""
    turn = (f"a turn that calls no tool of the {noun} still takes one step, and as many requests in a row without a call as the "
            "step cap allows end the episode.")
    if diagnostics_server:
        return f"A call of the diagnostics server is no call of the {noun} and takes no step, but {turn}"
    return f"A{turn[1:]}"


def outlasts_its_call(config):
    """Whether a work order outlasts the call that creates it, so that the orders created one after another form one visit and
    their closes come in later answers: on a row with a clock or with the repair card, since the engine closes an order inside
    its call where neither holds."""
    return config.has_clock() or config.repair_card_on()


def repair_step(config, target="", manual=True, waits=True):
    """The repair step of the level 3 procedure in both briefings, true on the row of config: the repair manual of every
    diagnosed fault where the episode serves it, one work order per diagnosed fault on target, created one after another, and
    only where an order outlasts its call one visit for them, after the components each depends on with the dependency card
    alone, and the wait for the notifications of their closes where the episode serves the wait."""
    look = "look up the repair manual of every diagnosed fault, " if manual else ""
    visit = wait = ""
    if outlasts_its_call(config):
        visit = (", so that one visit repairs every component after the components it depends on" if "dependencies" in repair_cards(config)
                 else ", so that one visit repairs them")
        wait = ", and wait for the notifications of their closes" if waits else ""
    return f"{look}create one work order per diagnosed fault{target}, one after another, with no other call between them{visit}{wait}"


def time_values(config):
    """The values that get_current_time returns beside the cycle and the horizon, in the words of both briefings: seconds_per_token
    joins them in the generated_tokens mode alone."""
    if config.clock_mode == real_time.TOKEN_MODE:
        return "seconds_per_cycle, think_seconds_per_cycle and seconds_per_token"
    return "seconds_per_cycle and think_seconds_per_cycle"


def n_cycles(n):
    """A count of cycles in its number, the singular only for one: 1 cycle, 0 cycles, 10 cycles."""
    return f"{n} cycle" if n == 1 else f"{n} cycles"


def replay_sentence(noun, live, recorded_before=None):
    """The rule of the replay, which every briefing of an asset whose readings loop states at every guidance level: while
    the condition levels stay the same, the asset whose noun is noun plays the live recorded cycles of its window again in
    the same order. A surface that also serves recorded_before cycles from before the episode, beside the cycles the
    episode runs, says that those are other recordings."""
    before = ("" if recorded_before is None else
              f", and the {n_cycles(recorded_before)} recorded before the episode started are other recordings outside that loop")
    return (f"Every reading is a recorded cycle, and while the condition levels stay the same the {noun} replays the same "
            f"{n_cycles(live)} in a loop, so its readings repeat every {n_cycles(live)} it runs under unchanged levels{before}.")


def listed(names, word="and"):
    """The names as one phrase with the word before the last, such as mean and std."""
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {word} {names[-1]}"


def _asset(profile):
    """The asset profile of the asset that a public profile describes."""
    return get_profile(profile["asset_id"])


def _limits(profile):
    return (f"a budget of {profile['budget']} and " if profile["budget"] is not None else "") + f"a horizon of {n_cycles(profile['horizon'])}"


def question_sentence(profile, config, unit):
    """The question of the episode: the question of the configuration when it states one, else the question of the asset
    file with its marks {components} and {limits} filled, else the question of the default form."""
    components, limits = ", ".join(profile["components"]), _limits(profile)
    stated = _asset(profile).text.get("question")
    if config.question:
        return config.question
    if stated is not None:
        return stated.replace("{components}", components).replace("{limits}", limits)
    return f"Which condition level is each of its components ({components}) at, and which work orders restore the {unit}, within {limits}?"


def opening_sentence(profile, unit, date=None):
    """The sentence that opens the episode, the opening of the asset file when it states one, else the flag of the
    performance monitor. After the date of an anchored start the sentence goes on in lower case."""
    stated = _asset(profile).text.get("opening")
    if stated is None:
        return f"{f'On {date}, the' if date else 'The'} performance monitor flagged the {unit}."
    return f"On {date}, {stated[:1].lower()}{stated[1:]}" if date else stated


def scoring_sentence(profile):
    """The rule of the scoring of a diagnosis, the scoring sentence of the asset file when it states one."""
    return _asset(profile).text.get("scoring", SCORING_SENTENCE)


def level_line(profile):
    """The condition levels of every component with its healthy level, then the meaning of the levels that the asset file
    states, if any."""
    levels = "; ".join(f"{c}: {', '.join(map(str, profile['levels'][c]))} (healthy {profile['healthy'][c]})" for c in profile["components"])
    meaning = _asset(profile).text.get("level_meaning")
    return f"Condition levels: {levels}." + (f" {meaning}" if meaning else "")


def readings_sentence(profile, noun, with_history=False):
    """What a reading of the asset is: for an evidence kind whose readings loop the rule of the replay with the two lengths
    of the kind, with the cycles recorded before the episode where the surface serves them, else the sentence of the asset
    file. For a kind that replays no repair the rule follows that no reading is recorded after the first close."""
    asset = _asset(profile)
    kind = kinds.kind_of(asset.evidence_kind)
    if kind.loop is None:
        sentence = asset.text["readings"]
    else:
        live, before = kind.part("loop")(asset)
        sentence = replay_sentence(noun, live, before if with_history else None)
    if not kind.replays_repairs:
        sentence += (f" The records of this {asset.text['unit_noun']} end at its first repair: from the close of the first work order "
                     "on, no new reading is recorded.")
    return sentence


def alarm_once_sentence(shown_by):
    """The rule of the alarm notifications, stated where a briefing lists them: the environment raises one alarm per
    channel in an episode, and shown_by names what shows a later excursion of that channel on the surface of the briefing."""
    return ("An alarm is raised once per channel in an episode, so a channel that leaves its normal range again after a "
            f"repair raises no second alarm, and {shown_by} shows it.")


def repair_sentences(noun):
    """One sentence per repair rule card, in the order of the cards, for an asset whose short noun is noun."""
    return {
        "fix_history": f"The fix history, get_fix_history, lists the work orders on this {noun} before the episode, each citable as hist/<n>.",
        "emergency": ("An emergency work order arrives sooner at a surcharge but holds the component at its healthy level only for a "
                      "limited number of cycles, after which the component returns to its earlier level unless a standard order on it "
                      "closes first."),
        "dependencies": "A repair takes only when every component it depends on is at its healthy level at its close.",
        "visits": ("The work orders created one after another, with no other call between them, form one technician visit with one visit charge that repairs them in "
                   "priority order."),
    }


# the sentences of the default asset, which the tests of its briefings read
_DEFAULT_NOUN = get_profile(DEFAULT_ASSET, engine=False).text["short_noun"]
TIME_SENTENCE = time_sentence(_DEFAULT_NOUN)
REPAIR_SENTENCES = repair_sentences(_DEFAULT_NOUN)
ALARM_ONCE = alarm_once_sentence("summarize_changes_since or a new reading")
DETERIORATION_SENTENCES = {
    "schedule": ("Scheduled deterioration transitions lower components one condition level at a time during the episode, at the "
                 "parameters published in the processes block of get_asset_profile."),
    "hazard": ("Random deterioration transitions may lower components one condition level at a time during the episode, at the "
               "rates published in the processes block of get_asset_profile."),
}


def repair_cards(config):
    """The repair rules the episode switches on, in the order of their sentences: none in the base, priced and real time
    presets, all four in the static and timed presets."""
    return repair_rules.cards(config)


def deterioration_sentences(config):
    """One sentence per deterioration card the episode switches on, whose parameters the processes block publishes:
    the schedule card while it draws transitions, the hazard card while it draws random drops or collateral
    damage. None where both cards are off, as on every row of the environment table but the stochastic one."""
    return [DETERIORATION_SENTENCES[card] for card in deterioration.cards(config)]


def asset_text(profile):
    """The text block of the asset that a public profile describes, with its nouns."""
    return get_profile(profile["asset_id"]).text


def _repair_sentences(config, noun):
    """The manual sentence and one sentence per active repair rule, or nothing when no repair rule is on."""
    cards = repair_cards(config)
    if not cards:
        return []
    covers = ["its charges and times", *(["its priority", "the components it depends on"] if "dependencies" in cards else []),
              *(["the emergency option"] if "emergency" in cards else [])]
    manual = f"The repair manual, lookup_fix, gives the standard fix of each component with {listed(covers)}."
    sentences = repair_sentences(noun)
    return [manual] + [sentences[card] for card in cards]


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
    text = asset_text(profile)
    unit, noun = text["unit_noun"], text["short_noun"]
    # an anchored start opens with its date, and the opening sentence then goes on in lower case
    opening = opening_sentence(profile, unit, config.date_of(0) if config.anchors is not None else None)
    return (f"Asset {profile['asset_id']} ({described}). {opening} {question_sentence(profile, config, unit)} "
            "Answer through the tools: submit exactly one diagnosis, or one declaration of insufficient evidence, "
            f"create the work orders that restore the {unit}, and finish with a summary. "
            + " ".join((scoring_sentence(profile), readings_sentence(profile, unit), time_sentence(noun), *deterioration_sentences(config),
                        *_repair_sentences(config, noun))))


def _level2(profile, config):
    prices, durations, rate = profile["prices"], profile["durations"], profile["downtime_rate"]
    noun = asset_text(profile)["short_noun"]
    groups = "\n".join(f"  - {g} costs {prices['read'][g]} per cycle and returns {', '.join(fields)}" for g, fields in profile["sensor_groups"].items())
    rates = "; ".join(f"{c}: " + ", ".join(f"level {lv} {r}" for lv, r in rate[c].items()) for c in profile["components"])
    statistics, plain = listed(profile["statistics"]), profile["statistics"][0]
    text = f"""{level_line(profile)}

Tools of the action server (class, duration in cycles, price):
- get_asset_profile() (read, 0, free) returns these rules, the normal ranges of the channel {plain}s and the tool table.
- read_sensors(sensor_group, cycles) (read, {n_cycles(durations['read_per_cycle'])} per cycle read, price per cycle by group) returns the {statistics} of every channel of the group for each recorded cycle, each value named <channel>_<statistic>:
{groups}
- get_current_time() (read, 0, free) returns the cycle, the horizon{', the budget left' if config.budget is not None else ''}, {time_values(config)} without moving the clock.
- wait_for_notification(timeout) (read, the cycles waited, free) lets the {noun} run until the next notification or for timeout cycles, the cycles left before the horizon when it is empty, and returns what ended the wait{', while the downtime accrues' if config.downtime_on() else ''}.
- summarize_changes_since(cycle) (read, 0, free) returns the cycles elapsed, the channel {plain}s of the sensor groups already read in this episode that left their normal range since that cycle with the first such cycle, and the work orders closed since then.
- get_failure_modes(component) (read, 0, free) returns the failure mode codes of a component, or of all components for an empty string, with the channels that indicate the component.
- get_fix_history() (read, 0, free) returns the work orders on this {noun} before the episode started, oldest first, each with its identifier hist/<n>, component, closing cycle (negative, before the start), mode and outcome.
- {_lookup_line(prices, config)}
- {_diagnosis_line(prices, profile['components'])}
- declare_insufficient_evidence(sensor_group, cycles, evidence_ids) (write, 0, free) records that the evidence cannot separate the remaining candidates and names the reading that would; accepted once, and never after a diagnosis. {DECLARATION_RULE} Reads and work orders stay allowed afterward, a diagnosis does not.
- {_work_order_line(prices, durations, rate, config, noun)}
- {_finish_line(config, noun)}
evidence_ids is a comma separated list of identifiers, possibly empty: {_evidence_forms(config)}. {_prediction_sentence(config)}
Every answer carries the notifications of {_notified_events(config)} since the previous call, then the timeout of a wait that reached it and the end of the episode. {ALARM_ONCE}

{_limits_paragraph(profile, config, rates)}"""
    if config.diagnostics_server:
        text += "\n\n" + DIAGNOSTICS_SERVER_SENTENCE
    return text


def misdiagnosis_sentence(prices):
    """The misdiagnosis price of the price table while it is above zero, which the environment adds to the cost of the
    episode at a wrong diagnosis without taking it from the budget, or an empty string."""
    if prices["misdiagnosis"] <= 0:
        return ""
    return (f"A wrong diagnosis costs the misdiagnosis price of {prices['misdiagnosis']}, the call-out that it wastes. The "
            "price is added to the cost of the episode when the episode is scored and does not fall from the budget.")


def _diagnosis_line(prices, components):
    """The submit_diagnosis entry of the tool list, one parameter per component, with the misdiagnosis sentence while the
    price is above zero."""
    text = (f"submit_diagnosis({', '.join(components)}, evidence_ids) (write, 0, free) records one condition level per "
            "component; accepted once, and never after a declaration.")
    return " ".join(part for part in (text, misdiagnosis_sentence(prices)) if part)


def _finish_line(config, noun):
    """The finish entry of the tool list. On a row with a clock the open work orders run to their close before the episode
    is scored, with the downtime of that span while the downtime card is on."""
    text = "finish(summary) (write, 0, free) ends the episode."
    if config.has_clock():
        downtime = " while the downtime accrues" if config.downtime_on() else ""
        text += (f" With no further call the {noun} then runs until the open work orders close{downtime}, and the episode is "
                 "scored after the last close.")
    return text


def _lookup_line(prices, config):
    """The lookup_fix entry of the tool list, naming the visit charge, the priority with the dependencies and the emergency
    option only when the episode switches their card on."""
    cards = repair_cards(config)
    charges = "the part, repair and visit charges" if "visits" in cards else "the part and repair charges"
    items = [f"its standard fix ({' or '.join(sorted(set(prices['fix_kind'].values())))}) with {charges}", "the lead time and the repair downtime"]
    if "dependencies" in cards:
        items += ["its priority", "the components that must be at their healthy level for the fix to take"]
    items += ["the expected effect"] + (["the emergency option"] if "emergency" in cards else [])
    return f"lookup_fix(component) (read, 0, free) returns the repair manual of a component: {', '.join(items[:-1])}, and {items[-1]}."


def _notified_events(config):
    """The public events that notify in this episode: the work orders and the alarms of the paid sensor groups always,
    the failure only when its card is on."""
    events = ["the work orders executed and closed", "the alarms of the sensor groups already read"] + \
             (["the failures"] if config.failure_on() else [])
    return listed(events)


def _work_order_line(prices, durations, rate, config, noun):
    """The create_work_order entry of the tool list, with the clauses of the visit charge, the dependency rule and the
    emergency mode, and the mode parameter itself, only when the episode switches them on, and the close and the visit of an
    order as the row makes them: inside its call without a clock or the repair card, with its visit before the next call that
    creates none with the card alone, after the lead time and the repairs of its visit with a clock."""
    cards = repair_cards(config)
    parts = ", ".join(f"{c} {p}" for c, p in prices["parts"].items())
    lead = ", ".join(f"{c} {durations['lead_time'][c]}" for c in config.components)
    down = ", ".join(f"{c} {durations['repair_downtime'][c]}" for c in config.components)
    visit = f", plus a visit charge of {prices['visit']} for the order that opens a visit" if "visits" in cards else ""
    # equal priorities, the off value of the dependency card, repair a visit in creation order, since the sort is stable
    order = "in the order they were created" if len(set(config.priority.values())) == 1 else "in priority order, smaller number first"
    stopped = f" at a downtime rate of {rate[STOPPED]} per cycle" if config.downtime_on() else ""
    mode = ", in the mode standard or emergency (standard when empty or left out)" if "emergency" in cards else ""
    parameters = "component, prediction_id, evidence_ids, mode" if "emergency" in cards else "component, prediction_id, evidence_ids"
    if config.has_clock():
        closes = "closes after the lead time and the repairs of its visit"
        visits = (" The standard orders created one after another, with no other call between them, form one visit: the technician "
                  "arrives at the largest creation cycle plus lead time among the orders not yet begun, or at the close of the repair "
                  f"begun before them, and repairs them back to back {order}, while a repair that has begun keeps its cycles, and the "
                  f"{noun} is stopped during the repairs{stopped}.")
    elif config.repair_card_on():
        closes = "closes with its visit"
        visits = (" The standard orders created one after another, with no other call between them, form one visit, whose orders close "
                  f"together, {order}, before the next call that creates none.")
    else:
        closes, visits = "closes inside its call", ""
    text = (f"create_work_order({parameters}) (write, {closes}, part price plus the repair charge of {prices['repair_charge']}{visit}) "
            f"asks a technician to restore the component to its healthy level{mode}. Parts: {parts}. Lead times: {lead}. Repair "
            f"downtimes: {down}.{visits}")
    if "dependencies" in cards:
        text += (" A repair takes only when the components it depends on are at their healthy level at its close, else the order "
                 "closes without effect; lookup_fix names the priority and the dependencies of every component.")
    if "emergency" in cards:
        e = config.emergency
        text += (f" An emergency order forms its own visit, arrives {n_cycles(e['lead'])} after its creation at {e['surcharge']} times "
                 f"the part and repair price, and holds the component at its healthy level for {n_cycles(e['hold'])} after its close, "
                 f"after which the component returns to its level before the repair unless a standard order on it closes first. {HOLD_SCORING}")
    return text + (" A work order is refused before a diagnosis or a declaration, and while an order on the same component is open. "
                   "An order on a healthy component is executed and counted as unnecessary.")


def call_rule_sentences(profile, config):
    """The sentences of the limits paragraph on steps and wall time, as both briefings state them: the step cap, the step of a
    turn without a call and the scored end at the wall limit. A prompt that lets the model call no tool leaves them out."""
    return (f"Step cap: {profile['max_steps']} tool calls, refused ones included.",
            idle_turn_sentence(asset_text(profile)["short_noun"], config.diagnostics_server), WALL_END_SENTENCE)


def _limits_paragraph(profile, config, rates):
    """The downtime, the budget, the horizon, the step cap with the step of a turn without a call, the clock and the end rules
    with the scoring of an end at the wall limit, without a downtime sentence when the downtime card is off and without a
    budget sentence when the episode has no budget stop."""
    if not config.downtime_on():
        downtime = ""
    else:
        limit = "" if profile["budget"] is None else ", while the budget limits the prices alone"
        downtime = f"Downtime accrues per cycle while a component is below its healthy level ({rates}) and counts toward the cost of the episode{limit}. "
    budget = "" if profile["budget"] is None else f"Budget: {profile['budget']}. "
    ends = (["on finish", "at the horizon", "at the step cap", f"after {EPISODE_WALL_LIMIT_S} seconds of wall time"]
            + ([] if profile["budget"] is None else ["when an action would exceed the budget"]))
    step_cap, idle, wall = call_rule_sentences(profile, config)
    return (f"{downtime}{budget}Horizon: {n_cycles(profile['horizon'])}. {step_cap} {idle} {_clock_sentence(profile, config)} "
            f"The episode ends {', '.join(ends[:-1])}, or {ends[-1]}{_failure_sentence(config)}. {wall} "
            "A refused call costs nothing but counts as a step, and a call with a missing argument or an argument of the wrong type is refused.")


def _level3(config):
    steps = ["read sensors before diagnosing", ]
    if config.diagnostics_server:
        steps += ["rank the candidates with rank_root_causes after each reading",
                  "while the ranking says read first, take the reading it names before you diagnose"]
    steps += [DECLARE_STEP, repair_step(config),
              "cite the evidence identifiers behind every diagnosis, declaration and work order",
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
        return ("Every tool call but get_current_time advances the clock by one cycle, and so does the step of a turn that calls no "
                f"tool of the {asset_text(profile)['short_noun']}.")
    sentence = "Time advances only inside tool calls, by their declared durations"
    if config.clock_mode == "generation_time":
        sentence += (f", and by your response time, whose seconds add up from call to call so that every {profile['think_seconds_per_cycle']} "
                     "seconds move the clock one cycle")
    elif config.clock_mode == real_time.TOKEN_MODE:
        sentence += (f", and by the tokens you generate between two calls, each counted as {profile['clock']['seconds_per_token']} seconds, whose "
                     f"seconds add up from call to call so that every {profile['think_seconds_per_cycle']} seconds move the clock one cycle")
    return sentence + "."


def _failure_sentence(config):
    rule = config.failure
    if not config.failure_on():
        return ""
    text = (f"; a component at its worst level for {n_cycles(rule['patience'])} is a failure that adds {rule['penalty']} to "
            "the cost")
    # the count runs while a work order is open, so only a repair that has closed averts the failure
    return text + (" and ends the episode" if rule["strict"] else "") + ", and only a repair on it that has closed by then averts it"
