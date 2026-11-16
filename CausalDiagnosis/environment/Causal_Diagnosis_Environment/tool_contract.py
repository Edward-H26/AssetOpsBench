"""The tool contract of the two hydraulic rig servers (CausalDiagnosis/docs/environment-actions-mcp.md, section Tools).

tool_contract.json declares every tool of the action server and of the diagnostics server with its class (read or
write), its duration and price rules, whether the verifier scores it, its parameters and the names the proof of concept
used for it, and the clock: its modes, seconds_per_cycle, think_seconds_per_cycle, the order of the processes inside one cycle
and the events that notify. The class travels with each MCP tool as its readOnlyHint annotation, the environment logs it on every call,
and each server has a test that compares what it advertises with this file, so that the declaration and the code cannot
drift apart.

The file is the contract of hyd_1. An asset of the recorded levels kind has the same tools, and load_contract(asset_id) gives
it the file's contract with the parameters of submit_diagnosis taken from its components, one condition level each. The
file names the word it uses for its asset under asset_words, and for an asset whose short noun differs every text of the
clock block and every duration, price and card clause of a tool reads the short noun of that asset in place of the
file's word. The card_clauses of a tool name the parameters and the clauses of its duration and price rules that hold
only while a rule card is on, and the public table of an episode with that card off leaves them out, so the agent is not
offered what the episode refuses.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

CONTRACT_PATH = Path(__file__).resolve().parent / "tool_contract.json"
CONTRACT_FORMAT = 2
ACTION_SERVER = "hydraulic_rig"
DIAGNOSTICS_SERVER = "causal_diagnosis"
DIAGNOSIS_TOOL = "submit_diagnosis"
REWORDED = ("duration", "price", "card_clauses")  # the parts of a tool entry that the public table copies into its rules


@lru_cache(maxsize=1)
def _file_contract():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("format") != CONTRACT_FORMAT:
        raise ValueError(f"unknown tool contract format in {CONTRACT_PATH}")
    return contract


def reword(text, old, new):
    """Every string of a text, a list or a mapping with each whole word old replaced by new."""
    if isinstance(text, str):
        return re.sub(rf"\b{re.escape(old)}\b", new, text)
    if isinstance(text, list):
        return [reword(item, old, new) for item in text]
    if isinstance(text, dict):
        return {key: reword(value, old, new) for key, value in text.items()}
    return text


def load_contract(asset_id=None):
    """The contract of an asset, the file itself for asset_id None: the file's tools with one condition level per component
    of the asset as the parameters of submit_diagnosis, ahead of its evidence identifiers, and for an asset whose short noun
    differs from the word of the file the texts of the clock block and the duration, price and card clauses of every tool
    in that noun, so that a card clause still stands once in its rule. The profile is read at every call, since a test or a
    draft may register another profile under the same asset id."""
    contract = _file_contract()
    if asset_id is None:
        return contract
    profile = get_profile(asset_id)
    levels = {component: "integer" for component in profile.components}
    diagnosis = {**contract["tools"][DIAGNOSIS_TOOL], "parameters": {**levels, "evidence_ids": "string"}}
    tools = {**contract["tools"], DIAGNOSIS_TOOL: diagnosis}
    old, new = contract["asset_words"]["short_noun"], profile.text["short_noun"]
    if old == new:
        return {**contract, "tools": tools}
    tools = {name: {**entry, **{part: reword(entry[part], old, new) for part in REWORDED if part in entry}} for name, entry in tools.items()}
    return {**contract, "clock": reword(contract["clock"], old, new), "token_clock": reword(contract["token_clock"], old, new), "tools": tools}


def tools_of(server, asset_id=None):
    """{tool name: contract entry} of one server in the contract of an asset, in the order of the file."""
    return {name: entry for name, entry in load_contract(asset_id)["tools"].items() if entry["server"] == server}


def tool_class(name):
    return load_contract()["tools"][name]["class"]


def clock(asset_id=None):
    """The clock block of the contract of an asset, which the asset profile publishes beside the tool table."""
    return load_contract(asset_id)["clock"]


def token_clock(asset_id=None):
    """The texts of the generated_tokens mode in the contract of an asset, which the clock block of an episode adds in that mode
    alone, so that the clock block of every other mode keeps its bytes."""
    return load_contract(asset_id)["token_clock"]


def read_only_hint(name):
    """The readOnlyHint a server registers for the tool: true for a read, false for a write."""
    return tool_class(name) == "read"


def check_advertised(server, hints, parameters=None, asset_id=None):
    """The differences between what a server advertises and the contract of an asset, the file's for asset_id None, as a
    list of problems, empty when they match.

    hints maps every advertised tool name to its readOnlyHint. parameters, when given, maps every advertised tool name
    to {parameter name: JSON type}, as the input schema shows it."""
    expected = tools_of(server, asset_id)
    problems = [f"{name} is in the contract but not advertised" for name in expected if name not in hints]
    problems += [f"{name} is advertised but not in the contract" for name in hints if name not in expected]
    for name in expected:
        if name in hints and hints[name] != (expected[name]["class"] == "read"):
            problems.append(f"{name} advertises readOnlyHint {hints[name]} but is a {expected[name]['class']} in the contract")
        if parameters is not None and name in parameters and parameters[name] != expected[name]["parameters"]:
            problems.append(f"{name} advertises parameters {parameters[name]} but the contract has {expected[name]['parameters']}")
    return problems


def public_table(servers=None, asset_id=None, cards_off=()):
    """The rows an agent reads in the asset profile: name, server, class, parameters, the duration and price rules and
    whether the tool is scored, restricted to `servers` when given. The parameters and rule clauses of a card in cards_off
    are left out."""
    rows = []
    for name, entry in load_contract(asset_id)["tools"].items():
        if servers is None or entry["server"] in servers:
            parameters, duration, price = dict(entry["parameters"]), entry["duration"]["rule"], entry["price"]["rule"]
            for card in cards_off:
                clauses = entry.get("card_clauses", {}).get(card, {})
                parameters = {k: v for k, v in parameters.items() if k not in clauses.get("parameters", ())}
                duration, price = _without(name, duration, clauses.get("duration", ())), _without(name, price, clauses.get("price", ()))
            rows.append({"name": name, "server": entry["server"], "class": entry["class"], "parameters": parameters,
                         "duration": duration, "price": price, "scored": entry["scored"]})
    return rows


def _without(name, rule, clauses):
    """A rule text without the clauses of a card that is off. Each clause occurs exactly once, else the contract and its
    card_clauses have drifted apart."""
    for clause in clauses:
        if rule.count(clause) != 1:
            raise ValueError(f"the card clause {clause!r} of {name} occurs {rule.count(clause)} times in its rule")
        rule = rule.replace(clause, "")
    return rule
