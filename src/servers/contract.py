"""The tool contract of the two hydraulic rig servers (docs/hydraulic_rig/action-server.md, section Tools).

tool_contract_v2.json declares every tool of the action server and of the diagnostics server with its class (read or
write), its duration and price rules, whether the verifier scores it, its parameters and the names the proof of concept
used for it. The class travels with each MCP tool as its readOnlyHint annotation, the environment logs it on every call,
and each server has a test that compares what it advertises with this file, so that the declaration and the code cannot
drift apart.
"""
import json
from functools import lru_cache
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().with_name("tool_contract_v2.json")
CONTRACT_FORMAT = 2
ACTION_SERVER = "hydraulic_rig"
DIAGNOSTICS_SERVER = "causal_diagnosis"
CLASSES = ("read", "write")


@lru_cache(maxsize=1)
def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("format") != CONTRACT_FORMAT:
        raise ValueError(f"unknown tool contract format in {CONTRACT_PATH}")
    return contract


def tools_of(server):
    """{tool name: contract entry} of one server, in the order of the file."""
    return {name: entry for name, entry in load_contract()["tools"].items() if entry["server"] == server}


def tool_class(name):
    return load_contract()["tools"][name]["class"]


def read_only_hint(name):
    """The readOnlyHint a server registers for the tool: true for a read, false for a write."""
    return tool_class(name) == "read"


def check_advertised(server, hints, parameters=None):
    """The differences between what a server advertises and the contract, as a list of problems; empty when they match.

    hints maps every advertised tool name to its readOnlyHint. parameters, when given, maps every advertised tool name
    to {parameter name: JSON type}, as the input schema shows it."""
    expected = tools_of(server)
    problems = [f"{name} is in the contract but not advertised" for name in expected if name not in hints]
    problems += [f"{name} is advertised but not in the contract" for name in hints if name not in expected]
    for name in expected:
        if name in hints and hints[name] != (expected[name]["class"] == "read"):
            problems.append(f"{name} advertises readOnlyHint {hints[name]} but is a {expected[name]['class']} in the contract")
        if parameters is not None and name in parameters and parameters[name] != expected[name]["parameters"]:
            problems.append(f"{name} advertises parameters {parameters[name]} but the contract has {expected[name]['parameters']}")
    return problems


def public_table(servers=None):
    """The rows an agent reads in the asset profile: name, server, class, parameters, the duration and price rules and
    whether the tool is scored; restricted to `servers` when given."""
    rows = []
    for name, entry in load_contract()["tools"].items():
        if servers is None or entry["server"] in servers:
            rows.append({"name": name, "server": entry["server"], "class": entry["class"], "parameters": dict(entry["parameters"]),
                         "duration": entry["duration"]["rule"], "price": entry["price"]["rule"], "scored": entry["scored"]})
    return rows
