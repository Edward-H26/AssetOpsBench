"""Actual shared-tool path for strict static repairs, with both advice settings."""
import json
import pytest
from .harness import memory_engine_episode, NAMES, body
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound


@pytest.mark.parametrize("advice", [False, True])
async def test_static_work_order_feedback_and_database_settle_without_waiting(world, advice):
    meter, engine, store, sessions = memory_engine_episode(
        world, "environment_test_scn1", diagnostics_server=advice,
        anchors=None, start={"combination": [100, 100, 2, 130], "entry_offset": 5})
    with bound(store, NAMES):
        tools = [t.name for t in await meter.listed_tools("task")]
        assert "wait_for_notification" not in tools
        diagnosis = await meter.call("task", "submit_diagnosis", {
            "cooler": 100, "valve": 100, "pump": 2, "accumulator": 130, "evidence_ids": ""})
        assert not diagnosis.is_error, diagnosis.text
        reply = await meter.call("wo", "generate_work_order", {
            "description": "repair", "asset_num": "hyd_1_pump", "site_id": "MAIN"})
        assert not reply.is_error, reply.text
        assert "no waiting time is required" in body(reply)["message"]
        profile = await meter.call("task", "get_asset_profile")
        assert not profile.is_error, profile.text
        assert not engine.public_work_orders() and engine.time == 0
        assert list(engine.hidden) == [100, 100, 0, 130]
        stored = next(o for o in store.all_docs(NAMES.workorder) if o["wonum"] == body(reply)["wonum"])
        assert stored["status"] == "COMP"
