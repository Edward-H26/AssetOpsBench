"""What the designs of the AssetOpsBench rows read when they order again on a component whose order is open: the engine's
refusal with the close of the open order, as the error of generate_work_order, followed by the receipt of a clocked row, and the
same reason in the entry of the call. On a row without a clock the order closes inside its call, so a second order is no
refusal."""
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import NAMES, body
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import memory_engine_episode as make


def ordered_twice(world, row):
    meter, engine, store, sessions = make(world, row)
    component = next((c for c, level in zip(COMPONENTS, engine.initial) if level != HEALTHY[c]), "cooler")
    sessions.step({"type": "submit_diagnosis", "diagnosis": dict(zip(COMPONENTS, engine.initial)), "evidence_ids": []})
    arguments = {"description": "repair", "asset_num": f"hyd_1_{component}", "site_id": "MAIN"}
    return meter, engine, store, sessions, component, arguments


async def test_a_second_order_through_the_meter_returns_the_engines_refusal_with_the_close(world):
    meter, engine, store, sessions, component, arguments = ordered_twice(world, "repair_rules")
    with bound(store, NAMES):
        first = await meter.call("wo", "generate_work_order", arguments)
        close = next(o["expected_close_t"] for o in engine.public_work_orders() if o["component"] == component)
        documents = len(store.all_docs(NAMES.workorder))
        second = await meter.call("wo", "generate_work_order", arguments)
    reason = f"a work order on the {component} is open; it closes at cycle {close}"
    assert not first.is_error and second.is_error and body(second) == {"error": reason}
    assert sessions.tool_calls[-1]["refused"] == reason and len(store.all_docs(NAMES.workorder)) == documents  # the refusal writes no order


async def test_on_the_base_row_a_second_order_through_the_meter_is_no_refusal(world):
    meter, engine, store, sessions, component, arguments = ordered_twice(world, "base")
    with bound(store, NAMES):
        await meter.call("wo", "generate_work_order", arguments)
        documents = len(store.all_docs(NAMES.workorder))
        second = await meter.call("wo", "generate_work_order", arguments)
    assert not second.is_error and "refused" not in sessions.tool_calls[-1] and len(store.all_docs(NAMES.workorder)) == documents + 1
