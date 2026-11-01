"""The tiny world MCP server: five tools over one episode state file (docs/hydraulic_rig/action-server.md).

Every tool builds the action, passes it to env.step through episode.call, and returns the JSON text of the outcome,
refusals included. The advertised parameters are flat, typed and required, since the Plan-Execute argument filler sees
only their names, JSON types and required flags; the legal values are in the question that opens the episode
(briefing.py). The server itself checks nothing: an argument of the wrong type reaches env.step as sent and a missing one
as None, so env.step refuses, logs and counts such a call like any other bad action.
"""
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from pydantic import PlainValidator, WithJsonSchema

from . import episode


def _as_sent(json_type):
    """A parameter that reaches the tool exactly as the client sent it, advertised with json_type. FastMCP decodes a
    JSON-looking string for every annotation but str (mcp/server/fastmcp/utilities/func_metadata.py:159), so the
    annotation is str and the plain validator replaces str's own check."""
    return Annotated[str, PlainValidator(lambda value: value), WithJsonSchema({"type": json_type})]


INTEGER = _as_sent("integer")
STRING = _as_sent("string")


def _advertised_as_required(fn):
    """The tool of fn with every parameter advertised as required and without its None default, which exists only to
    let an incomplete call through to env.step; the planner would otherwise see each parameter marked optional."""
    tool = Tool.from_function(fn, structured_output=False)
    properties = tool.parameters["properties"]
    for schema in properties.values():
        schema.pop("default", None)
    if properties:
        tool.parameters["required"] = list(properties)
    return tool


def build_server(state_path):
    state_path = Path(state_path)

    def measure(tier: INTEGER = None) -> str:
        """Read the next 60-second operating cycle of the rig at sensor tier 1, 2 or 3 and pay the tier's price. Returns the mean, std, min and max of every channel of the tier."""
        return episode.call(state_path, "measure", {"tier": tier}, {"type": "measure", "tier": tier})

    def submit(cooler: INTEGER = None, valve: INTEGER = None, pump: INTEGER = None, accumulator: INTEGER = None) -> str:
        """Submit the diagnosis: one condition level for each of the four components. Ends the investigate phase; free."""
        diagnosis = {"cooler": cooler, "valve": valve, "pump": pump, "accumulator": accumulator}
        return episode.call(state_path, "submit", diagnosis, {"type": "submit", "diagnosis": diagnosis})

    def submit_insufficient_evidence(next_action: STRING = None) -> str:
        """Submit that the evidence cannot separate the remaining candidates, naming the one action that would. Ends the investigate phase; free."""
        action = {"type": "submit", "insufficient_evidence": True, "next_action": next_action}
        return episode.call(state_path, "submit_insufficient_evidence", {"next_action": next_action}, action)

    def replace(component: STRING = None) -> str:
        """Restore one component (cooler, valve, pump or accumulator) to its healthy level, after the diagnosis. Costs the part plus downtime."""
        return episode.call(state_path, "replace", {"component": component}, {"type": "replace", "component": component})

    def finish() -> str:
        """End the episode after the diagnosis. Free."""
        return episode.call(state_path, "finish", {}, {"type": "finish"})

    def advance_time(cycles: INTEGER = None) -> str:
        """Let the rig run for a number of 60-second cycles without doing anything else. Only in episodes with a clock."""
        return episode.call(state_path, "advance_time", {"cycles": cycles}, {"type": "advance_time", "cycles": cycles})

    tools = [measure, submit, submit_insufficient_evidence, replace, finish]
    if state_path.exists() and episode.has_clock(state_path):
        tools.append(advance_time)

    return FastMCP(
        "hydraulic_rig",
        instructions="Diagnose and repair a hydraulic test rig: priced measurements, one diagnosis, then repairs.",
        log_level="WARNING",
        tools=[_advertised_as_required(fn) for fn in tools],
    )


def main(state_path):
    build_server(state_path).run(transport="stdio")
