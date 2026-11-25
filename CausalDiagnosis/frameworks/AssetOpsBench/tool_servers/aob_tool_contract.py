"""The tool contract of the AssetOpsBench rows.

aob_tool_contract.json holds one table per server keyed by tool name, since a lookup by name alone would confuse fmsr's
get_failure_modes with the rig's tool of that name. Every tool of AssetOpsBench's iot, fmsr and wo servers is either
listed, with its class, the engine action it becomes, the row that adds it, its parameters, its price and duration rules
with the configuration rule that gives their values, and whether the verifier scores it, or hidden with its reason. The
task tools of the rig's action server take their entries from the rig's contract. All six rows of the environment table
run on these servers, and the layers block names, for each row, the tools that carry the reads and
writes its layer adds. The engine of the rows publishes the table of its row in the asset profile, and a test compares
the table with what AssetOpsBench's servers advertise, so the declaration and the code cannot drift apart.
"""
import json
from functools import lru_cache
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import HISTORY_PREFIX
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract

CONTRACT_PATH = Path(__file__).resolve().parent / "aob_tool_contract.json"
CONTRACT_FORMAT = 1
AOB_SERVERS = ("iot", "fmsr", "wo")
TASK_SERVER = "task"
ROWS = layers.LADDER  # one row of the environment table per big layer
# Environment capabilities depend on the scenario layers, never on whether diagnostic advice is registered.
# the name under which these rows serve the actions of this project's action server, and the actions their own tools carry to the engine
# (the declaration of the framework holds both after the move)
ACTION_SERVER_NAME = TASK_SERVER
COVERS = {"read_sensors": ("iot.history", "iot.latest_reading"), "get_failure_modes": ("fmsr.get_failure_modes",),
          "create_work_order": ("wo.generate_work_order",), "get_fix_history": ("wo.list_workorders",)}


@lru_cache(maxsize=1)
def load_contract():
    """The contract with the entries of the task tools filled in from the rig's contract, whose rules they keep."""
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if data.get("format") != CONTRACT_FORMAT or tuple(data.get("rows", ())) != ROWS or tuple(data.get("layers", ())) != ROWS:
        raise ValueError(f"unknown tool contract format in {CONTRACT_PATH}")
    rig = contract.tools_of(contract.ACTION_SERVER)
    for name, entry in data["tools"][TASK_SERVER].items():
        source = rig[name]
        # the layer adds the level of its own profile answer (mcp_servers/utils/profile_levels.py) to the rig's parameters
        entry.update({"class": source["class"], "action": name, "parameters": {**source["parameters"], **entry.get("added_parameters", {})},
                      "duration": {"rule": source["duration"]["rule"], "value_of": name},
                      "price": {"rule": source["price"]["rule"], "value_of": name}, "scored": source["scored"]})
    return data


def entries(server):
    """{tool name: entry} of the tools a server lists on some row, in the order of the file."""
    return {name: entry for name, entry in load_contract()["tools"][server].items() if "hidden" not in entry}


def hidden(server):
    """{tool name: reason} of the tools of an AssetOpsBench server that no row lists."""
    return {name: entry["hidden"] for name, entry in load_contract()["tools"][server].items() if "hidden" in entry}


def tools_of(server, row=ROWS[-1]):
    """The names a server lists on a row: the tools of that row and of the rows before it, in the order of the file."""
    last = ROWS.index(row)
    return tuple(name for name, entry in entries(server).items() if ROWS.index(entry["row"]) <= last)


def tools_for(server, config):
    """The names a server lists for a configuration: the tools of the base and those of every big layer whose tools
    CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.lists_tools lists, in the order of the file, which on every ladder preset are the tools of
    its row. CausalDiagnosis MCP only adds its own server; it removes no environment tool."""
    return tuple(name for name, entry in entries(server).items() if layers.lists_tools(entry["row"], config))


def child_asset(asset_id, component):
    """The registry asset of a component, on which a work order names it, as the producer writes the registry:
    hyd_1_cooler for the cooler of hyd_1."""
    return f"{asset_id}_{component}"


WORK_TYPE_MODES = {"CM": "standard", "EM": "emergency"}  # wo's work types that are the engine's order modes
HISTORY_WONUM_PREFIX = "hist-"


def history_wonum(order_id):
    """The wonum of a past order of the fix history in the work order database, hist-1 for the engine's hist/1, since the
    wonum is part of the document id, where a slash would split the path of a CouchDB request."""
    return HISTORY_WONUM_PREFIX + order_id.removeprefix(HISTORY_PREFIX)


def history_order_id(wonum):
    """The engine's identifier of the past order with this wonum, or None for any other wonum."""
    number = wonum.removeprefix(HISTORY_WONUM_PREFIX)
    return HISTORY_PREFIX + number if wonum.startswith(HISTORY_WONUM_PREFIX) and number.isdigit() else None


def action_class(kind):
    """The class of an engine action the layer adds to the rig's actions."""
    return load_contract()["actions"][kind]["class"]


def row_of(config):
    """The row of the environment table a configuration plays on these servers, the rule of the big layers
    (CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.row_of)."""
    return layers.row_of(config)


def public_table(config):
    """The tool table of the asset profile on the configuration's row: the row of every tool an agent can call, with its
    price and declared duration under this configuration, zero where the tool is free or takes no time,
    and the rows of the diagnostics server when the configuration lists it, as the rig's profile has them."""
    rows = [_public_row(server, name, entries(server)[name], config) for server in (*AOB_SERVERS, TASK_SERVER)
            for name in tools_for(server, config)]
    if config.diagnostics_server:
        rows += [{**r, "price_value": config.tool_price(r["name"]), "duration_value": config.tool_duration(r["name"])}
                 for r in contract.public_table([contract.DIAGNOSTICS_SERVER])]
    return rows


def clock_durations(table):
    """The declared cycles per unit of every tool of the row's AssetOpsBench and task servers, as the clock block of the
    asset profile states them, from the public table of the configuration (public_table)."""
    return {r["name"]: r["duration_value"] for r in table if r["server"] != contract.DIAGNOSTICS_SERVER}


def _public_row(server, name, entry, config):
    """The row of one tool for a configuration: a task tool takes its parameters from the rig's contract of the configured
    asset, one condition level per component in submit_diagnosis, and every rule names the asset by its short noun."""
    parameters = ({**contract.tools_of(contract.ACTION_SERVER, config.asset_id)[name]["parameters"], **entry.get("added_parameters", {})}
                  if server == TASK_SERVER else entry["parameters"])
    old, new = load_contract()["asset_words"]["short_noun"], get_profile(config.asset_id).text["short_noun"]
    return {"name": name, "server": server, "class": entry["class"], "parameters": dict(parameters),
            "duration": contract.reword(entry["duration"]["rule"], old, new), "price": contract.reword(entry["price"]["rule"], old, new),
            "scored": entry["scored"], "price_value": _value(config.tool_price, entry["price"]),
            "duration_value": _value(config.tool_duration, entry["duration"])}


def _value(rule, part):
    """The value of a price or duration under the configuration: the rule of the rig tool the entry names, else zero."""
    return rule(part["value_of"]) if "value_of" in part else 0
