"""The action server of the hydraulic rig: twelve MCP tools over one episode state file (CausalDiagnosis/docs/environment-actions-mcp.md).

Every tool builds the action, passes it to env.step through episode.call, and returns the JSON text of the envelope,
refusals included. Each tool is registered with the readOnlyHint of the tool contract (CausalDiagnosis/environment/Causal_Diagnosis_Environment/tool_contract.json), so
the class travels with the tool. The advertised parameters are flat, typed and required, since the Plan-Execute argument
filler sees only their names, JSON types and required flags; the legal values are in the question that opens the episode
(briefing.py) and in the asset profile. The server itself checks nothing: an argument of the wrong type reaches env.step as
sent and a missing one as None, so env.step refuses, logs and counts such a call like any other bad action. The one
exception is the mode of create_work_order, whose default is the standard mode: a missing mode stays out of the action, so
an order without it is the action it was before the emergency mode existed. The two time tools carry the descriptions of
ARE's System app as CausalDiagnosis/environment/Causal_Diagnosis_Environment/clock/time_tools.py adapts them. The asset of the episode's state file gives
the parameters of submit_diagnosis and the component names, their number and the nouns of the descriptions, so a listing
before any episode and every hyd_1 episode show the hydraulic rig's texts.
"""
import inspect
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from mcp.types import ToolAnnotations
from pydantic import PlainValidator, WithJsonSchema

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import time_tools
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET, EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_4_repair_rules as repair_rules
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.mcp_servers.utils import ready_line as ready
from CausalDiagnosis.mcp_servers.utils.transcript_hooks import transcribed

from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import DECLARATION_RULE, HOLD_SCORING, listed


def _as_sent(json_type):
    """A parameter that reaches the tool exactly as the client sent it, advertised with json_type. FastMCP decodes a
    JSON-looking string for every annotation but str (mcp/server/fastmcp/utilities/func_metadata.py), so the
    annotation is str and the plain validator replaces str's own check."""
    return Annotated[str, PlainValidator(lambda value: value), WithJsonSchema({"type": json_type})]


INTEGER = _as_sent("integer")
STRING = _as_sent("string")


def _advertised(fn):
    """The tool of fn with the readOnlyHint of the contract, every parameter advertised as required and without its None
    default, which exists only to let an incomplete call through to env.step; the planner would otherwise see each
    parameter marked optional."""
    tool = Tool.from_function(fn, structured_output=False, annotations=ToolAnnotations(readOnlyHint=contract.read_only_hint(fn.__name__)))
    properties = tool.parameters["properties"]
    for schema in properties.values():
        schema.pop("default", None)
    if properties:
        tool.parameters["required"] = list(properties)
    return tool


def evidence_list(text):
    """A comma separated evidence_ids argument as the list the environment stores; anything but a string passes as sent."""
    if not isinstance(text, str):
        return text
    return [item.strip() for item in text.split(",") if item.strip()]


def action_of(tool, arguments, components):
    """The engine action of a tool of the action server from its arguments as the client sent them, a missing one as None, so
    that env.step refuses a malformed call and counts it as a step. The server's own tools and the task path of a framework's
    gateway build their actions here, so an action is the same whichever entry reached it. A missing mode stays out of an
    order, which is then the order of the standard mode."""
    action = {"type": tool}
    if tool == "read_sensors":
        action.update(sensor_group=arguments.get("sensor_group"), cycles=arguments.get("cycles"))
    elif tool == "wait_for_notification":
        action["timeout"] = arguments.get("timeout")
    elif tool == "summarize_changes_since":
        action["cycle"] = arguments.get("cycle")
    elif tool in ("get_failure_modes", "lookup_fix"):
        action["component"] = arguments.get("component")
    elif tool == "submit_diagnosis":
        action.update(diagnosis={c: arguments.get(c) for c in components}, evidence_ids=evidence_list(arguments.get("evidence_ids")))
    elif tool == "declare_insufficient_evidence":
        action.update(separating_test={"sensor_group": arguments.get("sensor_group"), "cycles": arguments.get("cycles")},
                      evidence_ids=evidence_list(arguments.get("evidence_ids")))
    elif tool == "create_work_order":
        action.update(component=arguments.get("component"), prediction_id=arguments.get("prediction_id"),
                      evidence_ids=evidence_list(arguments.get("evidence_ids")))
        if arguments.get("mode") is not None:
            action["mode"] = arguments["mode"]
    elif tool == "finish":
        action["summary"] = arguments.get("summary")
    return action


COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def asset_of(state_path):
    """The asset of the episode in the state file, and hyd_1 while there is no readable episode, as for a tool listing
    before any episode. A broken state file still fails at the call that reads it."""
    try:
        return episode.load_state(state_path)["config"].get("asset_id", DEFAULT_ASSET)
    except (OSError, ValueError):
        return DEFAULT_ASSET


def _either(names):
    """The names as one choice, such as cooler, valve, pump or accumulator."""
    return listed(names, "or")


# The parts of the asset profile with the words that name them in the description of get_asset_profile, one pair per part
# and every key of the profile in exactly one pair, so a key that enters the profile enters here with its words, and the
# word of every field of the processes block. test_server.py fails while a key of either has no words.
PROCESS_WORDS = {"schedule": "schedule", "hazard": "hazard", "emergency": "emergency", "dependencies": "dependency",
                 "priority": "priority", "fix_history": "fix history", "failure": "failure"}


def profile_parts(plain):
    """The parts of the asset profile with their words, the normal ranges named by the plain statistic of the asset."""
    return ((("asset_id", "asset"), "the asset with its type, description and site"),
            (("components", "levels", "healthy"), "components with their condition levels and healthy level"),
            (("sensor_groups", "statistics"), "sensor groups with their channels and the served statistics"),
            (("prices",), "prices"), (("durations",), "durations"), (("downtime_rate",), "downtime rates"),
            (("budget",), "budget"), (("horizon",), "horizon"), (("max_steps",), "step cap"),
            (("processes",), f"the parameters of the {listed(PROCESS_WORDS.values())} rules"),
            (("clock_mode", "seconds_per_cycle", "think_seconds_per_cycle", "clock"),
             "the clock mode with the seconds per cycle, the seconds of thinking per cycle and the clock rules"),
            (("tools",), "the tool table"), (("normal_ranges",), f"the normal ranges of the channel {plain}s"))


PROFILE_PARTS = profile_parts(get_profile(DEFAULT_ASSET, engine=False).plain_statistic())  # the parts of the default asset


def config_of(state_path):
    """The configuration of the episode in the state file, None while there is no readable episode, as for a listing before
    any episode."""
    try:
        return EpisodeConfig.from_json(episode.load_state(state_path)["config"])
    except (OSError, ValueError, KeyError):
        return None


def _takes(config, cycles="no time"):
    """What a free tool takes in the episode: its declared time, or one cycle in the instant clock mode, where every call
    but get_current_time takes its cycle."""
    if config is not None and config.clock_mode == "instant":
        return "takes one cycle, as every call but get_current_time does in this episode"
    return f"takes {cycles}"


def _read_time(config):
    """The time read_sensors takes: the cycles it reads, none where a read per cycle takes no declared time, and the cycle of
    every call in the instant mode."""
    if config is None or (config.clock_mode != "instant" and config.durations["read_per_cycle"] > 0):
        return "takes the cycles read"
    return _takes(config)


def _manual_line(either, cards):
    """The description of lookup_fix: the parts of the manual the episode serves, the visit charge, the priority with the
    dependencies and the emergency option only where their card is on, as the briefing states them."""
    charges = "the part, repair and visit charges" if "visits" in cards else "the part and repair charges"
    items = [f"its standard fix with {charges}", "the lead time and the repair downtime"]
    if "dependencies" in cards:
        items += ["its priority within a visit", "the components that must be at their healthy level for the fix to take"]
    items += ["the channels that indicate it", "the expected effect"]
    if "emergency" in cards:
        items.append("the emergency option with its surcharge, lead time and hold")
    return f"The repair manual of one component ({either}): {', '.join(items[:-1])}, and {items[-1]}."


def _order_line(either, cards, config):
    """The description of create_work_order: the mode, the visit charge, the dependency rule and the emergency order only
    where their card is on, and the order of the repairs of a visit as the briefing states it."""
    order = "in the order they were created" if config is not None and len(set(config.priority.values())) == 1 else "in priority order"
    mode = ", in the mode standard (the default) or emergency" if "emergency" in cards else ""
    visit = " with one visit charge" if "visits" in cards else ""
    needs = (", and a repair takes only when the components it depends on are at their healthy level at its close"
             if "dependencies" in cards else "")
    emergency = (" An emergency order is its own visit, comes sooner at a surcharge and restores only for a limited hold."
                 if "emergency" in cards else "")
    opens = ", and the visit charge for the order that opens a visit," if "visits" in cards else ""
    return (f"Ask a technician to restore one component ({either}) to its healthy level, citing a prediction identifier of the "
            f"diagnostics server (or an empty string) and the comma separated evidence identifiers behind the order{mode}. The "
            f"standard orders created one after another with no other call between them form one visit{visit}, repaired back to back {order}{needs}.{emergency} Refused "
            "before a diagnosis or a declaration and while an order on the component is open. Costs the part plus the repair "
            f"charge{opens} at creation.")


def descriptions(profile, config=None):
    """The descriptions of the tools that name the components, their number or the asset, from the asset profile: the
    hydraulic rig's texts for hyd_1. With the configuration of an episode the manual and the work order name the visit
    charge, the priority with the dependencies and the emergency mode only where the episode switches their card on
    (repair_rules.cards, the rule of the briefing), and the time of a tool follows the clock; without one, as for a listing
    before any episode, every clause stands."""
    components, noun, plain = list(profile.components), profile.text["short_noun"], profile.plain_statistic()
    cards = set(repair_rules.CARDS) if config is None else set(repair_rules.cards(config))
    free = _takes(config)
    either = _either(components)
    count = COUNT_WORDS[len(components)] if len(components) < len(COUNT_WORDS) else str(len(components))
    statistics = listed(profile.statistics["served"])
    return {
        "get_asset_profile": f"The asset profile: {listed(words for _keys, words in profile_parts(plain))}. Free, {free}.",
        "read_sensors": f"Run the {noun} for the given number of cycles and return the recorded {statistics} of every channel of the sensor group for each cycle, each with its observation number and cycle time. Costs the group's price per cycle that returns a reading and {_read_time(config)}.",
        "summarize_changes_since": f"The changes since a past cycle: the cycles elapsed, the channel {plain}s of the sensor groups already read in this episode that left their normal range, with the first such cycle, and the work orders closed since then. Free, {free}.",
        "get_failure_modes": f"The failure mode codes of one component ({either}), or of all components for an empty string, each with its severity, recommended action, indicating channels and evidence identifier. Free, {free}.",
        "get_fix_history": f"The work orders on this {noun} before the episode started, oldest first, each with its evidence identifier hist/<n>, the component, the closing cycle (negative, before the start), the mode and the outcome. Free, {free}.",
        "lookup_fix": f"{_manual_line(either, cards)} Free, {free}.",
        "submit_diagnosis": f"Submit the diagnosis: one condition level for each of the {count} components, with the comma separated evidence identifiers behind it (possibly empty). Accepted once per episode and never after a declaration of insufficient evidence. Free.",
        "declare_insufficient_evidence": ("Declare that the evidence cannot separate the remaining candidates and name the reading that would (a sensor group and a number of cycles), "
                                          f"with the comma separated evidence identifiers behind it. {DECLARATION_RULE} Accepted once per episode and never after a diagnosis; work orders stay allowed afterward. Free."),
        "create_work_order": _order_line(either, cards, config),
        "finish": f"End the episode with a summary of what was found and done. On a row with a clock the {noun} then runs until the open work orders close, and the episode is scored after the last close. Free. {HOLD_SCORING}",
    }


def levels_signature(components):
    """The signature of submit_diagnosis: one condition level per component of the asset, then the evidence identifiers,
    every parameter typed as the planner sees it and defaulting to None, as in the other tools."""
    parameter = inspect.Parameter.POSITIONAL_OR_KEYWORD
    names = [*((component, INTEGER) for component in components), ("evidence_ids", STRING)]
    return inspect.Signature([inspect.Parameter(name, parameter, default=None, annotation=kind) for name, kind in names],
                             return_annotation=str)


def build_server(state_path, asset_id=None, record=True, config=None):
    """The action server of the episode whose state file is state_path, for the asset asset_id when given and else that of the
    state file. record=False builds it without the transcript, for a caller that needs its tool definitions and serves no
    episode through it. config is the configuration whose cards and clock the descriptions follow, that of the state file
    when None, and none without a readable episode."""
    state_path = Path(state_path)
    if config is None:
        config = config_of(state_path)
    profile = get_profile(asset_id or asset_of(state_path))
    components = list(profile.components)

    def call(tool, arguments):
        return episode.call(state_path, tool, arguments, action_of(tool, arguments, components))

    def get_asset_profile() -> str:
        return call("get_asset_profile", {})

    def read_sensors(sensor_group: STRING = None, cycles: INTEGER = None) -> str:
        return call("read_sensors", {"sensor_group": sensor_group, "cycles": cycles})

    def get_current_time() -> str:
        return call("get_current_time", {})

    def wait_for_notification(timeout: INTEGER = None) -> str:
        return call("wait_for_notification", {"timeout": timeout})

    get_current_time.__doc__ = time_tools.GET_CURRENT_TIME
    wait_for_notification.__doc__ = time_tools.wait_for_notification_description(profile.text["short_noun"])

    def summarize_changes_since(cycle: INTEGER = None) -> str:
        return call("summarize_changes_since", {"cycle": cycle})

    def get_failure_modes(component: STRING = None) -> str:
        return call("get_failure_modes", {"component": component})

    def get_fix_history() -> str:
        return call("get_fix_history", {})

    def lookup_fix(component: STRING = None) -> str:
        return call("lookup_fix", {"component": component})

    def submit_diagnosis(**levels) -> str:
        return call("submit_diagnosis", {**{component: levels.get(component) for component in components}, "evidence_ids": levels.get("evidence_ids")})

    def declare_insufficient_evidence(sensor_group: STRING = None, cycles: INTEGER = None, evidence_ids: STRING = None) -> str:
        return call("declare_insufficient_evidence", {"sensor_group": sensor_group, "cycles": cycles, "evidence_ids": evidence_ids})

    def create_work_order(component: STRING = None, prediction_id: STRING = None, evidence_ids: STRING = None, mode: STRING = None) -> str:
        arguments = {"component": component, "prediction_id": prediction_id, "evidence_ids": evidence_ids}
        if mode is not None:
            arguments["mode"] = mode
        return call("create_work_order", arguments)

    def finish(summary: STRING = None) -> str:
        return call("finish", {"summary": summary})

    submit_diagnosis.__signature__ = levels_signature(components)
    worded = descriptions(profile, config)
    if config is not None and "emergency" not in repair_rules.cards(config):
        # an episode without the emergency mode advertises no mode, as its texts name none and the engine refuses one
        signature = inspect.signature(create_work_order)
        create_work_order.__signature__ = signature.replace(parameters=[p for p in signature.parameters.values() if p.name != "mode"])
    for fn in (get_asset_profile, read_sensors, summarize_changes_since, get_failure_modes, get_fix_history, lookup_fix, submit_diagnosis,
               declare_insufficient_evidence, create_work_order, finish):
        fn.__doc__ = worded[fn.__name__]

    tools = (get_asset_profile, read_sensors, get_current_time, wait_for_notification, summarize_changes_since, get_failure_modes,
             get_fix_history, lookup_fix, submit_diagnosis, declare_insufficient_evidence, create_work_order, finish)
    app = FastMCP(
        contract.ACTION_SERVER,
        instructions=f"Diagnose and repair a {profile.text['noun']} on a clock: priced and timed readings, a fix history and a repair manual, one diagnosis or one declaration, work orders executed by a technician in visits, then finish.",
        log_level="WARNING",
        tools=[_advertised(fn) for fn in tools],
    )
    return transcribed(app, state_path.parent, contract.ACTION_SERVER) if record else app


def main(state_path, started=None):
    """Serve the tools over stdio, the ready line of the process immediately before the stdio loop (CausalDiagnosis/mcp_servers/utils/ready_line.py).
    started is the launcher's stamp."""
    app = build_server(state_path)
    ready.announce(Path(state_path).parent, contract.ACTION_SERVER, started)
    app.run(transport="stdio")
