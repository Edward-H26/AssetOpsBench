import json
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import THINKING_MODES, TOKEN_MODE
from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, RIG_SERVER
from . import formatting


NOT_RECORDED = "not recorded, since the run folder holds no run.json and no scores.csv row for this episode"
RUNNER_OF_KIND = {"causal": "plans.json marks the CausalDiagnosis controller",
                  "plan_execute": "the tasks of trajectory.json mark the Plan-Execute runner",
                  "policy": "the folder holds no model call, as a reference policy leaves it"}


# ----- part 1, the start -----
def part_start(ep):
    profile, source = ep.profile()
    out = [f"## {formatting.PARTS[0]}", "", start_text(ep), ""]
    out += formatting.table(["Setting", "Value"], settings_rows(ep))
    out += ["", "The agent never sees this table. It sees the briefing and the profile below.", "",
            "### What the agent saw at step one", "",
            "The briefing and the asset profile below are the only information the agent has at step one. Everything else it "
            "learns from the answers to its own tool calls, which part 2 lists.", "",
            "#### The briefing", "", "The question the runner received, `question.txt`, verbatim.", "",
            "```text", ep.question.rstrip("\n"), "```", "",
            "#### The asset profile", "", source, ""]
    return out + profile_lines(profile, ep.aob)


def gateway_folder(ep):
    """The folder of the metering layer of the declared surface whose servers the episode called, from the module its gateway
    names."""
    called = {call.server for call in ep.calls}
    entry = next(entry for entry in frameworks.surfaces().values() if "gateway" in entry and called & set(entry["servers"]))
    return entry["gateway"].partition(":")[0].rpartition(".")[0].replace(".", "/") + "/"


def start_text(ep):
    if ep.aob:
        store = (ep.run.get("backend") or {}).get("kind")
        text = ("The driver `CausalDiagnosis/experiments/episode_runners/run_episodes.py` created this episode from the seed and the configuration in the "
                "table below. It wrote `episode.json`, the state the servers replay (the seed, the configuration and then every action), "
                "`released.json`, the record of everything the agent has been told, `question.txt`, the briefing, and one launcher per "
                "registered server. The episode ran on AssetOpsBench's iot, fmsr and wo servers and the task server, each behind the "
                f"metering layer of `{gateway_folder(ep)}`. For every call the layer rebuilds the episode by `reset(seed)` "
                "and the replay of the stored actions, applies the call as one action, writes what the action recorded into the "
                "episode's databases" + (f" on the {store} store" if store else "") + ", lets AssetOpsBench's own tool answer over them "
                "and writes both files back.")
    else:
        text = ("The driver `CausalDiagnosis/experiments/episode_runners/run_episodes.py` created this episode from the seed and the configuration in the "
                "table below. It wrote `episode.json`, the state the action server replays (the seed, the configuration and then every "
                "action), `released.json`, the record of everything the agent has been told, `question.txt`, the briefing, and one "
                "launcher per registered server. Every tool call starts a fresh server process from its launcher, which rebuilds the "
                "episode by `reset(seed)` and the replay of the stored actions, applies the new action, writes both files back and answers.")
    if ep.kind == "causal":
        rule_only = ep.plans and all(p.get("source") == "rule_only" for p in ep.plans.get("plans") or [])
        text += (" The CausalDiagnosis controller of `CausalDiagnosis/agents/Causal_Diagnosis_Agent/` received the briefing as its question and keeps the model's "
                 "working memory, which the code calls the board. It calls the model "
                 "for every plan, the reads of the opening included, for an orientation once the profile is read and for the closing "
                 "summary, and executes the plans itself, checking the expectation of every step.")
        text += (" The CausalDiagnosis MCP only advised the model in its working memory, and the controller called it to refresh that advice and to "
                 "verify a repair on a reading a plan named." if (ep.plans or {}).get("server", True) else
                 " This run had no CausalDiagnosis MCP, so the model ranked its own candidates and the controller verified every repair on "
                 "the normal ranges.")
        if rule_only:
            text += " In this run the rule only test tool wrote every plan, so the closing summary was its only model call."
    elif ep.kind == "plan_execute":
        text += (" The Plan-Execute runner of `src/agent/plan_execute/` received the briefing as its question, asked the model once "
                 "for a plan, filled the arguments of each step with one more model call, called the tools in order, and asked the "
                 "model for a closing answer at the end.")
    elif ep.kind == "policy":
        text += (" The reference policy of `CausalDiagnosis/environment/reference_policies/policies.py` played through the same action server, one tool call "
                 "per action, while the driver stepped a shadow environment with the same seed beside it, from which the oracle "
                 "policy reads the hidden state.")
    return text


def settings_rows(ep):
    c, prices, durations = ep.config, ep.config["prices"], ep.config["durations"]
    diagnostics = bool(c.get("diagnostics_server"))
    runner = RUNNER_OF_KIND.get(ep.kind)
    rows = [("Run", f"`{ep.folder.parent.name}`"), ("Episode folder", f"`{ep.folder.name}`"),
            ("Agent", ep.setting("agent_label", "agent") or (NOT_RECORDED + (f", and {runner}" if runner else ""))),
            ("Model", ep.setting("model", "model") or ("none" if ep.kind == "policy" else NOT_RECORDED))]
    options = ep.run.get("causal_options")
    if options:
        rows.append(("Agent options", ", ".join(f"{k} {formatting.num(v)}" for k, v in options.items())))
    servers = ep.run.get("servers") or [RIG_SERVER] + ([CAUSAL_SERVER] if diagnostics else [])
    clock = c["clock_mode"] + (f", {formatting.num(c['think_seconds_per_cycle'])} seconds of thinking per cycle" if c["clock_mode"] in THINKING_MODES else "")
    if c["clock_mode"] == TOKEN_MODE:
        clock += f" and {formatting.num(c['seconds_per_token'])} seconds per generated token"
    groups = ", ".join(f"{g} {formatting.num(p)}" for g, p in prices["read"].items())
    parts = ", ".join(f"{k} {formatting.num(p)}" for k, p in prices["parts"].items())
    rows += [("Seed", formatting.num(ep.state.get("seed"))),
             ("Configuration preset", ep.setting("config_name", "config") or NOT_RECORDED),
             ("Servers registered", formatting.joined(servers)),
             ("Clock mode", clock),
             ("Budget", formatting.num(c["budget"])),
             ("Horizon", f"{formatting.num(c['horizon'])} cycles"),
             ("Step cap", f"{formatting.num(c['max_steps'])} tool calls, refused ones included"),
             ("Price table", f"{ep.setting('prices', 'prices') or 'name not recorded'}, read per cycle {groups}, parts {parts}, "
                             f"repair charge {formatting.num(prices['repair_charge'])}"),
             ("Durations in cycles", f"{formatting.num(durations['read_per_cycle'])} per cycle read, {formatting.num(durations['advance_per_cycle'])} per cycle "
                                     f"waited, lead time {', '.join(f'{k} {formatting.num(v)}' for k, v in durations['lead_time'].items())}, "
                                     f"repair downtime {', '.join(f'{k} {formatting.num(v)}' for k, v in durations['repair_downtime'].items())}"),
             ("Guidance level", formatting.num(c["guidance_level"])),
             ("Degradation schedule", schedule_setting(c["schedule"])),
             ("Failure rule", failure_setting(c["failure"])),
             ("Verifier scope", c["verifier_scope"]),
             ("Ranking model of the CausalDiagnosis MCP", ranking_setting(ep) if diagnostics else "none"),
             ("Briefing version", ep.run.get("briefing") or "not recorded")]
    if ep.run.get("backend"):
        rows.append(("Store of AssetOpsBench's databases", ep.run["backend"].get("kind")))
    if c.get("start") is not None:
        rows.append(("Start", "anchored by the configuration, named in part 3"))
    if c.get("allowed") is not None:
        rows.append(("Allowed starts", "restricted by the configuration to a list of combinations"))
    return rows


def ranking_setting(ep):
    """The ranking model of the run, else the one the diagnostics server reports in its first ranking answer."""
    recorded = ep.setting("ranking_model", "ranking_model")
    if recorded:
        return recorded
    reported = next((c.answer["ranking_model"] for c in ep.calls if c.tool == "rank_root_causes" and (c.answer or {}).get("ranking_model")), None)
    return f"{reported}, as the ranking answers report it" if reported else "not recorded"


def schedule_setting(schedule):
    if schedule.get("mode") != "seeded":
        return "none"
    low, high = schedule["first_drop"]
    return (f"seeded, {formatting.plural(schedule['components'], 'component')}, first drop at a cycle drawn from {low} to {high}, "
            f"then one level every {schedule['interval']} cycles")


def failure_setting(failure):
    """The failure rule card, whose off value is a patience of 0 cycles (EpisodeConfig.failure_on)."""
    if not failure["patience"]:
        return "off, since a patience of 0 cycles switches the rule off"
    return (f"{formatting.num(failure['patience'])} cycles at the worst level is a failure unless a repair restores the component by then, "
            f"penalty {formatting.num(failure['penalty'])}, "
            + ("which ends the episode" if failure["strict"] else "the episode goes on"))


def profile_lines(profile, aob=False):
    asset = profile.get("asset") or {}
    prices, durations, rate = profile["prices"], profile["durations"], profile["downtime_rate"]
    components = profile["components"]
    children = (asset.get("children") or {}) if aob else {}
    out = [f"Asset `{profile.get('asset_id')}`" + (f", {asset.get('asset_type')}, {asset.get('description')}, at site {asset.get('site')}." if asset else ".")
           + (" Its child assets, which the work orders name, are " + formatting.joined(f"`{child}` for the {component}" for component, child in children.items()) + "."
              if children else ""), ""]
    out += formatting.table(["Component", "Levels", "Healthy level", "Downtime per cycle by level", "Part price", "Lead time", "Repair downtime"],
                 [(c, ", ".join(formatting.num(lv) for lv in profile["levels"][c]), formatting.num(profile["healthy"][c]),
                   ", ".join(f"level {lv} {formatting.num(r)}" for lv, r in rate[c].items()), formatting.num(prices["parts"][c]),
                   formatting.num(durations["lead_time"][c]), formatting.num(durations["repair_downtime"][c])) for c in components])
    out += ["", f"A work order costs the part price plus a repair charge of {formatting.num(prices['repair_charge'])}, and the rig stops at a downtime "
            f"rate of {formatting.num(rate['stopped'])} per cycle while a repair runs. A reading takes {formatting.plural(durations['read_per_cycle'], 'cycle')} for "
            f"each cycle read and a wait {formatting.plural(durations['advance_per_cycle'], 'cycle')} for each cycle waited. Budget "
            f"{formatting.num(profile['budget'])}, horizon {formatting.num(profile['horizon'])} cycles, step cap {formatting.num(profile['max_steps'])} tool calls, clock "
            f"mode {profile['clock_mode']}, {formatting.num(profile.get('seconds_per_cycle'))} seconds per recorded cycle and "
            f"{formatting.num(profile.get('think_seconds_per_cycle'))} seconds of thinking per cycle.", ""]
    processes = profile.get("processes") or {}
    if processes:
        out += ["The parameters of the rules follow, as the `processes` block of the profile states them.", ""]
        out += formatting.table(["Rule", "Parameters"], [(f"`{name}`", f"`{json.dumps(value)}`") for name, value in processes.items()]) + [""]
    out += formatting.table(["Sensor group", "Price per cycle", "Channels"],
                 [(g, formatting.num(prices["read"][g]), ", ".join(f"`{f}`" for f in fields)) for g, fields in profile["sensor_groups"].items()])
    if aob:
        out += ["", "A history row holds one recorded cycle, the mean of every channel under the channel's name and its standard "
                "deviation under that name with `_std` appended.", ""]
    else:
        out += ["", f"A reading returns the {formatting.joined(profile.get('statistics') or [])} of every channel of the group, named "
                "`<channel>_<statistic>`.", ""]
    out += formatting.table(["Tool", "Server", "Class", "Parameters", "Duration", "Price", "Scored"],
                 [(f"`{t['name']}`", t["server"], t["class"], ", ".join(f"{k} {v}" for k, v in t["parameters"].items()) or "none",
                   formatting.quoted(t["duration"]), formatting.quoted(t["price"]), formatting.num(t["scored"])) for t in profile.get("tools") or []])
    bands = profile.get("normal_ranges") or {}
    if bands:
        out += ["", "The public normal range of every channel mean follows. " + (
            "A mean outside its normal range is an alarm, and on the clock rows the receipt notifies the alarms of the sensor groups already read."
            if aob else "A mean outside its normal range is an alarm, which `summarize_changes_since` lists, and on the clock rows the answers "
            "notify the alarms of the sensor groups already read."), ""]
        out += formatting.table(["Channel mean", "Low", "High"], [(f"`{f}`", f"{b['low']:.8g}", f"{b['high']:.8g}") for f, b in bands.items()])
    return out + [""]
