"""The tool contract of the AssetOpsBench rows (CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_tool_contract.json): every tool of AssetOpsBench's
iot, fmsr and wo servers is listed with the parameters AssetOpsBench advertises or hidden with a reason, the task tools keep
the rig's entries, the six presets play their rows, every row names the tools that carry its layer,
the repair cards change no row without a clock, and the engine of the rows publishes the table of its row with the
price and duration of every tool under the preset."""
import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import AOB_SERVERS, ROWS, TASK_SERVER
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import READS, SERVERS, TOOLS

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config

ENGINE_ACTIONS = {"aob_read": "read", "read_recorded": "read", "get_failure_modes": "read", "create_work_order": "write"}
# the rig's action tools the AssetOpsBench rows replace; fmsr's get_failure_modes only shares a name with the rig's
RIG_ONLY = {"read_sensors", "summarize_changes_since", "get_fix_history", "lookup_fix", "create_work_order"}


def json_type(schema):
    """The JSON type a FastMCP input schema gives a parameter, an optional one through its non-null option."""
    if "type" in schema:
        return schema["type"]
    options = [s for s in schema.get("anyOf", []) if s.get("type") != "null"]
    if len(options) == 1:
        return json_type(options[0])
    return "object" if "$ref" in schema else None


async def advertised(server):
    return {t.name: {p: json_type(s) for p, s in t.inputSchema.get("properties", {}).items()} for t in await SERVERS[server].mcp.list_tools()}


@pytest.mark.parametrize("server", AOB_SERVERS)
async def test_every_assetopsbench_tool_is_listed_with_its_parameters_or_hidden_with_a_reason(server):
    tools = await advertised(server)
    listed, hidden = layer_contract.entries(server), layer_contract.hidden(server)
    assert set(tools) == set(listed) | set(hidden) and not set(listed) & set(hidden)
    assert all(isinstance(reason, str) and reason for reason in hidden.values())
    assert {name: entry["parameters"] for name, entry in listed.items()} == {name: tools[name] for name in listed}
    assert TOOLS[server] == layer_contract.tools_of(server) == tuple(listed)


def test_the_hidden_tools_and_servers_are_those_of_the_base_note():
    hidden = {server: set(layer_contract.hidden(server)) for server in AOB_SERVERS}
    assert hidden == {"iot": {"sensor_stats"}, "fmsr": {"generate_failure_modes", "add_failure_modes"},
                      "wo": {"update_workorder", "approve_workorder", "assign_technician", "close_workorder", "cancel_workorder"}}
    assert set(layer_contract.load_contract()["hidden_servers"]) == {"utilities", "tsfm", "vibration"}  # each with its reason


def test_every_listed_tool_becomes_one_engine_action_of_its_class():
    for server in AOB_SERVERS:
        for name, entry in layer_contract.entries(server).items():
            assert ENGINE_ACTIONS[entry["action"]] == entry["class"], (server, name)
            assert entry["scored"] == (entry["action"] == "create_work_order") and entry["row"] == "base"
    assert {name for name, e in layer_contract.entries("iot").items() if e["action"] == "read_recorded"} == set(READS)
    assert {k: layer_contract.action_class(k) for k in layer_contract.load_contract()["actions"]} == {"read_recorded": "read", "aob_read": "read"}


def test_the_task_tools_keep_the_rig_entries_and_the_time_tools_join_on_the_clock_rows():
    """The task tools keep the rig's entries, with the one parameter the layer adds, the level of its profile answer."""
    rig = contract.tools_of(contract.ACTION_SERVER)
    for name, entry in layer_contract.entries(TASK_SERVER).items():
        source = rig[name]
        added = {"response_format": "string"} if name == "get_asset_profile" else {}
        assert (entry["class"], entry["parameters"], entry["scored"]) == (source["class"], {**source["parameters"], **added}, source["scored"])
        assert (entry["price"]["rule"], entry["duration"]["rule"]) == (source["price"]["rule"], source["duration"]["rule"])
    base = ("get_asset_profile", "summarize_changes_since", "submit_diagnosis", "declare_insufficient_evidence", "finish")
    assert layer_contract.tools_of(TASK_SERVER, "base") == layer_contract.tools_of(TASK_SERVER, "priced") == base
    for row in ("real_time", "advanced_real_time"):
        assert set(layer_contract.tools_of(TASK_SERVER, row)) == {*base, "get_current_time", "wait_for_notification"}
    for row in ROWS[4:]:
        assert set(layer_contract.tools_of(TASK_SERVER, row)) == {*base, "get_current_time", "wait_for_notification", "lookup_fix"}
    names = [n for s in (*AOB_SERVERS, TASK_SERVER) for n in layer_contract.entries(s)]
    assert len(names) == len(set(names))  # the clock block keys its durations by name alone


def test_every_row_names_the_tools_that_carry_the_reads_and_writes_of_its_layer():
    layers = layer_contract.load_contract()["layers"]
    assert tuple(layers) == ROWS and all(isinstance(text, str) and text for text in layers.values())
    assert all(name in layers["repair_rules"] for name in ("lookup_fix", "work_type EM", "list_workorders"))
    assert "get_current_time" in layers["real_time"] and layers["deterioration"].startswith("no new tool")


def test_the_repair_cards_change_no_row_without_a_clock():
    """The repair cards of repair_rules.json over the priced preset change when an order closes, not
    the row, since the repair rules row sits on the clock rows."""
    repair = preset_config("repair_rules")
    cards = {k: getattr(repair, k) for k in ("dependencies", "priority", "emergency", "fix_history")}
    config = preset_config("priced", **cards, prices={**preset_config("priced").prices, "visit": repair.prices["visit"]})
    assert config.repair_card_on() and layer_contract.row_of(config) == "priced"


@pytest.mark.parametrize("row", ROWS)
def test_each_preset_plays_its_row_and_the_table_holds_its_tools_with_their_values(row):
    config = preset_config(row)
    assert layer_contract.row_of(config) == row
    rows = layer_contract.public_table(config)
    table = {r["name"]: r for r in rows}
    assert len(table) == len(rows) and not any(r["server"] == contract.ACTION_SERVER for r in rows)
    assert table["get_failure_modes"]["server"] == "fmsr"
    expected = [n for s in AOB_SERVERS for n in TOOLS[s]] + list(layer_contract.tools_of(TASK_SERVER, row))
    expected += list(contract.tools_of(contract.DIAGNOSTICS_SERVER)) if config.diagnostics_server else []
    rig_only = RIG_ONLY - set(layer_contract.tools_of(TASK_SERVER, row))  # lookup_fix joins the task server on the repair rules row
    assert list(table) == expected and not rig_only & set(table)
    assert all(table[name]["server"] == TASK_SERVER for name in layer_contract.tools_of(TASK_SERVER, row))
    assert table["history"]["price_value"] == table["latest_reading"]["price_value"] == config.prices["read"]
    assert table["history"]["duration_value"] == config.durations["read_per_cycle"] and table["latest_reading"]["duration_value"] == 0
    assert table["generate_work_order"]["price_value"] == config.tool_price("create_work_order")
    if "wait_for_notification" in table:
        assert table["wait_for_notification"]["duration_value"] == 1
    priced, timed = {"history", "latest_reading", "generate_work_order"}, {"history", "wait_for_notification"}
    assert all(r["price_value"] == 0 for n, r in table.items() if n not in priced)
    assert all(r["duration_value"] == 0 for n, r in table.items() if n not in timed)
    assert layer_contract.clock_durations(layer_contract.public_table(config)) == {n: r["duration_value"] for n, r in table.items() if r["server"] != contract.DIAGNOSTICS_SERVER}


@pytest.mark.parametrize("row", ROWS)
def test_the_engine_of_the_rows_publishes_the_table_and_the_clock_of_its_row(world, row):
    cycles, blocks = world
    config = preset_config(row)
    env = RecordedRigEnv(cycles, blocks, config)
    profile = env.reset(7)["result"]
    assert profile["tools"] == layer_contract.public_table(config)
    assert profile["clock"]["durations"] == layer_contract.clock_durations(layer_contract.public_table(config))
    assert profile["clock"]["notifications"] == layer_contract.load_contract()["clock"]["notifications"]
    assert profile["clock"]["mode"] == config.clock_mode and "read_sensors" not in profile["clock"]["durations"]



def test_the_order_rule_states_the_visit_from_the_advanced_real_time_row_and_its_charge_from_the_repair_rules_row():
    """The duration rule of generate_work_order says what the engine does by row: an order closes inside its call without a
    clock or the repair card, the visit forms from the advanced real time clock row on with the schedule of visits.py, and the
    visit charge and the priority order come from the repair rules row."""
    rule = layer_contract.entries("wo")["generate_work_order"]["duration"]["rule"]
    assert "From the advanced real time clock row on, the standard orders created one after another" in rule
    assert "the largest creation cycle plus lead time among the orders not yet begun" in rule and "a repair that has begun keeps its cycles" in rule
    assert "From the repair rules row the visit pays one visit charge and repairs in priority order" in rule
    assert "from the repair rules row the standard orders" not in rule and "latest creation cycle" not in rule
