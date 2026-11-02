"""The action server of the hydraulic rig: nine MCP tools over one episode state file (docs/hydraulic_rig/action-server.md).

Every tool builds the action, passes it to env.step through episode.call, and returns the JSON text of the envelope,
refusals included. Each tool is registered with the readOnlyHint of the tool contract (servers/tool_contract_v2.json), so
the class travels with the tool. The advertised parameters are flat, typed and required, since the Plan-Execute argument
filler sees only their names, JSON types and required flags; the legal values are in the question that opens the episode
(briefing.py) and in the asset profile. The server itself checks nothing: an argument of the wrong type reaches env.step as
sent and a missing one as None, so env.step refuses, logs and counts such a call like any other bad action.
"""
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from mcp.types import ToolAnnotations
from pydantic import PlainValidator, WithJsonSchema

from servers import contract

from . import episode


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


def build_server(state_path):
    state_path = Path(state_path)

    def get_asset_profile() -> str:
        """The asset profile: components with their condition levels and healthy level, sensor groups with their channels, prices, durations, downtime rates, budget, horizon, step cap, the rule on the number of faulty components, the clock mode, the tool table and the healthy bands of the channel means. Free, takes no time."""
        return episode.call(state_path, "get_asset_profile", {}, {"type": "get_asset_profile"})

    def read_sensors(sensor_group: STRING = None, cycles: INTEGER = None) -> str:
        """Run the rig for the given number of cycles and return the recorded mean and std of every channel of the sensor group for each cycle, each with its observation number and cycle time. Costs the group's price per cycle and takes the cycles read."""
        return episode.call(state_path, "read_sensors", {"sensor_group": sensor_group, "cycles": cycles},
                            {"type": "read_sensors", "sensor_group": sensor_group, "cycles": cycles})

    def advance_time(cycles: INTEGER = None) -> str:
        """Let the rig run for the given number of cycles without reading it and return the new time. Free, but downtime accrues while time passes."""
        return episode.call(state_path, "advance_time", {"cycles": cycles}, {"type": "advance_time", "cycles": cycles})

    def summarize_changes_since(cycle: INTEGER = None) -> str:
        """The changes since a past cycle: the cycles elapsed, the channel means that left their healthy band with the first such cycle, and the work orders closed since then. Free, takes no time."""
        return episode.call(state_path, "summarize_changes_since", {"cycle": cycle}, {"type": "summarize_changes_since", "cycle": cycle})

    def get_failure_modes(component: STRING = None) -> str:
        """The failure mode codes of one component (cooler, valve, pump or accumulator), or of all components for an empty string, each with its severity, recommended action, indicating channels and evidence identifier. Free, takes no time."""
        return episode.call(state_path, "get_failure_modes", {"component": component}, {"type": "get_failure_modes", "component": component})

    def submit_diagnosis(cooler: INTEGER = None, valve: INTEGER = None, pump: INTEGER = None, accumulator: INTEGER = None, evidence_ids: STRING = None) -> str:
        """Submit the diagnosis: one condition level for each of the four components, with the comma separated evidence identifiers behind it (possibly empty). Accepted once per episode and never after a declaration of insufficient evidence. Free."""
        diagnosis = {"cooler": cooler, "valve": valve, "pump": pump, "accumulator": accumulator}
        arguments = {**diagnosis, "evidence_ids": evidence_ids}
        return episode.call(state_path, "submit_diagnosis", arguments,
                            {"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": evidence_list(evidence_ids)})

    def declare_insufficient_evidence(sensor_group: STRING = None, cycles: INTEGER = None, evidence_ids: STRING = None) -> str:
        """Declare that the evidence cannot separate the remaining candidates and name the reading that would (a sensor group and a number of cycles), with the comma separated evidence identifiers behind it. Accepted once per episode and never after a diagnosis; work orders stay allowed afterward. Free."""
        arguments = {"sensor_group": sensor_group, "cycles": cycles, "evidence_ids": evidence_ids}
        return episode.call(state_path, "declare_insufficient_evidence", arguments,
                            {"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": sensor_group, "cycles": cycles},
                             "evidence_ids": evidence_list(evidence_ids)})

    def create_work_order(component: STRING = None, prediction_id: STRING = None, evidence_ids: STRING = None) -> str:
        """Ask a technician to restore one component (cooler, valve, pump or accumulator) to its healthy level, citing a prediction identifier of the diagnostics server (or an empty string) and the comma separated evidence identifiers behind the order. Refused before a diagnosis or a declaration and while an order on the component is open. Costs the part plus the repair charge at creation; the order closes after the lead time and the repair downtime."""
        arguments = {"component": component, "prediction_id": prediction_id, "evidence_ids": evidence_ids}
        return episode.call(state_path, "create_work_order", arguments,
                            {"type": "create_work_order", "component": component, "prediction_id": prediction_id, "evidence_ids": evidence_list(evidence_ids)})

    def finish(summary: STRING = None) -> str:
        """End the episode with a summary of what was found and done. Open work orders stay open. Free."""
        return episode.call(state_path, "finish", {"summary": summary}, {"type": "finish", "summary": summary})

    tools = (get_asset_profile, read_sensors, advance_time, summarize_changes_since, get_failure_modes,
             submit_diagnosis, declare_insufficient_evidence, create_work_order, finish)
    return FastMCP(
        contract.ACTION_SERVER,
        instructions="Diagnose and repair a hydraulic test rig on a clock: priced and timed readings, one diagnosis or one declaration, work orders executed by a technician, then finish.",
        log_level="WARNING",
        tools=[_advertised(fn) for fn in tools],
    )


def main(state_path):
    build_server(state_path).run(transport="stdio")
