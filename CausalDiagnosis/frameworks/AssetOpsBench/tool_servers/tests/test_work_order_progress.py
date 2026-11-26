"""Paired environment checks, not claims about real LLM performance.

The scripted reference uses the test fixture's truth to isolate the environment path.
Both configurations see identical public operations, scheduling, notifications and costs.
"""
from dataclasses import replace
import json

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import work_order_progress
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import (
    NAMES, answer, memory_engine_episode, preset_config, receipt,
)


def test_completion_feedback_does_not_reveal_hidden_repair_effect():
    envelope = {"time": 8, "open_work_orders": [], "notifications": [
        {"event": "work_order_closed", "id": "wo-1", "component": "pump", "t": 8,
         "effective": False, "hidden_condition": [3, 73, 2, 90]},
    ]}
    feedback = work_order_progress.progress(envelope, wait_available=True)
    assert feedback["completed_in_this_reply"] == [{"id": "wo-1", "component": "pump", "t": 8}]
    assert "does not establish equipment recovery" in feedback["message"]
    assert "effective" not in json.dumps(feedback) and "hidden_condition" not in json.dumps(feedback)




async def scripted_repair(world, advice):
    meter, engine, store, sessions = memory_engine_episode(
        world, "advanced_real_time", diagnostics_server=advice, clock_mode="declared",
    )
    transcript = []

    async def call(server, tool, args=None, *, refused=False):
        reply = await meter.call(server, tool, args or {})
        assert reply.is_error == refused, reply.text
        transcript.append((server, tool, reply.text))
        return reply

    with bound(store, NAMES):
        # List the actual FastMCP definitions, not just the declaration table.
        listed = {server: [t.name for t in await meter.listed_tools(server)]
                  for server in (*contract.AOB_SERVERS, contract.TASK_SERVER)}
        assert "wait_for_notification" in listed["task"]
        read = await call("iot", "history", {"site_name": "MAIN", "asset_id": "hyd_1", "limit": 1})
        row = answer(read, "advanced_real_time")["observations"][-1]
        field = next(k for k, v in row.items() if isinstance(v, (float, int)))
        evidence = f"obs/{row['timestamp']}/{field}"
        await call("task", "submit_diagnosis", {**dict(zip(COMPONENTS, engine.initial)), "evidence_ids": evidence})
        faulty = [c for c, level in zip(COMPONENTS, engine.initial) if level != HEALTHY[c]]
        assert faulty
        for component in faulty:
            reply = await call("wo", "generate_work_order", {
                "description": "repair", "asset_num": f"hyd_1_{component}", "site_id": "MAIN",
            })
            accepted = answer(reply, "advanced_real_time")
            assert accepted["operation_status"] == "accepted"
            assert accepted["status"] == "APPR"
            assert accepted["maintenance"]["work_order"]["expected_close_t"] > engine.time
            assert "pending, not failed" in accepted["message"]
        first = faulty[0]
        before_orders = len(store.all_docs(NAMES.workorder))
        duplicate = await call("wo", "generate_work_order", {
            "description": "duplicate", "asset_num": f"hyd_1_{first}", "site_id": "MAIN",
        }, refused=True)
        assert "closes at cycle" in answer(duplicate, "advanced_real_time")["error"]
        assert "remain accepted" in receipt(duplicate)["work_order_progress"]["message"]
        assert len(store.all_docs(NAMES.workorder)) == before_orders
        # A free query does not magically complete repairs on a declared clock.
        t = engine.time
        await call("task", "get_current_time")
        assert engine.time == t and engine.public_work_orders()
        completed = []
        for _ in range(20):
            if not engine.public_work_orders():
                break
            waited = await call("task", "wait_for_notification", {"timeout": 20})
            completed += [n["id"] for n in receipt(waited)["notifications"] if n.get("event") == "work_order_closed"]
        assert not engine.public_work_orders()
        assert len(completed) == len(faulty)
        # Repair downtime may leave the most recent row from before completion.
        # Let a running cycle be recorded, then retrieve that real measurement.
        await call("task", "wait_for_notification", {"timeout": 1})
        post_start = meter.config.timeline.timestamp(engine.time - 1)
        after = await call("iot", "history", {"site_name": "MAIN", "asset_id": "hyd_1", "limit": 1,
                                               "start": post_start, "end": meter.config.timeline.timestamp(engine.time)})
        post_rows = answer(after, "advanced_real_time")["observations"]
        assert post_rows and all(r["timestamp"] >= post_start for r in post_rows)
        await call("task", "finish", {"summary": "reference repair and post-repair reading completed"})
    assert engine.done and engine.log[-1]["reason"] == "finish"
    assert tuple(engine.hidden) == tuple(HEALTHY[c] for c in COMPONENTS)
    return listed, transcript, sessions.tool_calls


async def test_same_actions_produce_identical_feedback_events_and_costs_with_or_without_advice(world):
    assert await scripted_repair(world, True) == await scripted_repair(world, False)


async def test_thinking_without_an_action_completes_repairs_and_preserves_the_notification_for_both_settings(world):
    replies = []
    for advice in (True, False):
        meter, engine, store, _sessions = memory_engine_episode(
            world, "advanced_real_time", diagnostics_server=advice, clock_mode="generation_time",
            thinks=[0, 0, 10, 0],
        )
        component = next(c for c, level in zip(COMPONENTS, engine.initial) if level != HEALTHY[c])
        with bound(store, NAMES):
            await meter.call("task", "submit_diagnosis", {**dict(zip(COMPONENTS, engine.initial)), "evidence_ids": ""})
            accepted = await meter.call("wo", "generate_work_order", {
                "description": "repair", "asset_num": f"hyd_1_{component}", "site_id": "MAIN",
            })
            order_id = answer(accepted, "advanced_real_time")["wonum"]
            assert engine.public_work_orders()
            # The runner reports an idle turn, not another maintenance action or wait.
            assert not await meter.no_call()
            assert engine.time == 10 and not engine.public_work_orders()
            update = await meter.call("task", "get_current_time")
            assert any(n.get("event") == "work_order_closed" and n.get("id") == order_id
                       for n in receipt(update)["notifications"])
            assert receipt(update)["work_order_progress"]["completed_in_this_reply"]
            replies.append(update.text)
    assert replies[0] == replies[1]


@pytest.mark.parametrize("preset", ["base", "real_time", "advanced_real_time"])
@pytest.mark.parametrize("advice", [False, True])
async def test_creation_reply_reports_the_synced_database_status(world, preset, advice):
    meter, engine, store, _sessions = memory_engine_episode(
        world, preset, diagnostics_server=advice,
    )
    with bound(store, NAMES):
        await meter.call("task", "submit_diagnosis", {**dict(zip(COMPONENTS, engine.initial)), "evidence_ids": ""})
        component = next(c for c, level in zip(COMPONENTS, engine.initial) if level != HEALTHY[c])
        reply = await meter.call("wo", "generate_work_order", {
            "description": "repair", "asset_num": f"hyd_1_{component}", "site_id": "MAIN",
        })
        assert not reply.is_error, reply.text
        data = answer(reply, preset)
        stored = next(o for o in store.all_docs(NAMES.workorder) if o["wonum"] == data["wonum"])
        assert data["status"] == data["work_order"]["status"] == stored["status"]
        assert data["status"] != "WAPPR"
        assert data["operation_status"] == "accepted"
        public = data["maintenance"]["work_order"]
        assert public["component"] == component
        if preset == "base":
            assert data["status"] == "COMP" and public["status"] == "closed"
            assert public["closed_t"] == engine.time == 0
        else:
            assert data["status"] == "APPR" and public["status"] == "open"
            assert public["expected_close_t"] >= engine.time


@pytest.mark.parametrize("advice", [False, True])
async def test_completion_of_a_dependency_blocked_repair_is_not_reported_as_recovery(world, advice):
    meter, engine, store, _sessions = memory_engine_episode(
        world, "repair_rules", diagnostics_server=advice, clock_mode="declared",
        start={"combination": [100, 80, 1, 130], "entry_offset": 5},
    )
    with bound(store, NAMES):
        await meter.call("task", "submit_diagnosis", {**dict(zip(COMPONENTS, engine.initial)), "evidence_ids": ""})
        accepted = await meter.call("wo", "generate_work_order", {
            "description": "repair pump before its valve dependency", "asset_num": "hyd_1_pump", "site_id": "MAIN",
        })
        assert not accepted.is_error
        completed = []
        for _ in range(20):
            if not engine.public_work_orders():
                break
            reply = await meter.call("task", "wait_for_notification", {"timeout": 20})
            assert not reply.is_error
            feedback = receipt(reply).get("work_order_progress")
            if feedback:
                completed += feedback["completed_in_this_reply"]
                assert '"effective"' not in json.dumps(feedback) and '"unnecessary"' not in json.dumps(feedback)
        assert not engine.public_work_orders() and completed
        assert engine.hidden[COMPONENTS.index("pump")] == 1
        assert "does not establish equipment recovery" in feedback["message"]
