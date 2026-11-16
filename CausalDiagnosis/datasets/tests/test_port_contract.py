"""The contract between the engine and a port, held on the made-up port demo_1 of the folder ports, which is mounted from
its folder alone and edits no other file of the tree: its kind records a history before the start, plays no live record
and replays no repair, it serves one statistic, has one component, a healthy start, a label and a table of its own, and
a site written with a lower case letter."""
import asyncio
import dataclasses
import hashlib
import json
import re
from datetime import datetime

import pytest

from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, dataset_packages as ports
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import environment_factory as world
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

from CausalDiagnosis.datasets.tests.conftest import PORT_ID, mount_port

OTHER_ASSETS = r"\b(rig|cooler|valve|pump|accumulator|bench|fan|belt|bearing)\b"

FIELDS = ("F1_Force_kN_1Hz_value", "F2_Stroke_mm_1Hz_value")
FAULTY, HEALTHY = {"combination": [1], "entry_offset": 0}, {"combination": [0], "entry_offset": 0}


def _config(start=FAULTY, **changes):
    """The default configuration of the port from a fixed start."""
    return dataclasses.replace(get_profile(PORT_ID).default_config(), start=start, **changes)


def _verdict(env, grade=None):
    return verify(env.log, released_from_env(env), env.oracle(), None, grade=grade)


def test_the_port_is_found_by_its_folder_alone():
    with mount_port():
        profile = get_profile(PORT_ID)
        assert profile.evidence_kind == "demo_history" and kinds.kind_of("demo_history") is not None
        assert validate(profile.to_json()) == []
    assert kinds.kind_of("demo_history") is None
    with pytest.raises(KeyError, match=f"no asset profile '{PORT_ID}'"):
        get_profile(PORT_ID)


def test_an_episode_of_the_port_plays_on_the_engine_and_is_scored(port_records):
    env = world.make_env(_config())
    assert env.facts_sha256 == hashlib.sha256(port_records.read_bytes()).hexdigest()
    log = run_episode(env, OraclePolicy(env), 0)
    best = verify(log, released_from_env(env), env.oracle(), None)
    assert best["verified"] and best["pass"] and best["failure_class"] == "none", best
    env.reset(1)
    envelope, cost, done, info = env.step({"type": "read_sensors", "sensor_group": "no_such_group", "cycles": 1})
    assert info["refused"] == "unknown sensor group" and not done and cost == 0
    envelope, cost, done, _info = env.step({"type": "read_sensors", "sensor_group": "all", "cycles": 2})
    assert envelope["result"]["served"] == [] and envelope["result"]["stopped"] == 0 and cost == 0 and not done
    assert env.budget_left() == env.config.budget


def test_the_healthy_start_of_the_port_is_a_valid_episode(port_records):
    env = world.make_env(_config(HEALTHY))
    env.reset(0)
    # the one faulty combination would be the development start and leave no fault to score, so no split moves it
    assert env.hidden == [0] and env.candidates == [(0,), (1,)]
    env.step({"type": "submit_diagnosis", "diagnosis": {"press": 0}, "evidence_ids": []})
    env.step({"type": "finish", "summary": "The press is aligned, so no work order was needed."})
    verdict = _verdict(env)
    assert verdict["verified"] and verdict["pass"] and verdict["failure_class"] == "none", verdict


def _recorded_env(config):
    return RecordedRigEnv(*world.load_world(PORT_ID, str(world.default_facts_path(config))), config)


def test_the_engine_of_the_rows_records_the_whole_history_of_the_port(port_records):
    env = _recorded_env(_config())
    env.reset(0)
    assert env.recorded_cycles() == list(range(-7, 0))
    read = {"type": "read_recorded", "tool": "history", "sensor_group": "all", "fields": list(FIELDS), "start": -7, "end": 0,
            "offset": 0, "limit": 10}
    envelope, cost, _done, _info = env.step(read)
    assert envelope["result"]["cycles"] == list(range(-7, 0)) and cost == 7
    observations = [o for o in env.log[-1]["served"]]
    assert len(observations) == 7 and all(set(o["values"]) == set(FIELDS) for o in observations)
    envelope, cost, _done, _info = env.step(read)
    assert envelope["result"]["price"] == 0 and cost == 0


class _LiveSource:
    """A source of the port's kind that does play live records: the records of its start in turn."""

    opened = []

    def __init__(self, profile, cycles, blocks):
        from CausalDiagnosis.datasets.demo_1.source import HistorySource

        self._history = HistorySource(profile, cycles, blocks)
        self.normal_ranges, self.window, self._played = self._history.normal_ranges, None, []
        for name in ("start_candidates", "starts", "prehistory", "served_values", "alarms", "fix_history"):
            setattr(self, name, getattr(self._history, name))

    def forget(self):
        self.window, self._played = None, []

    def open(self, state, t, rng, offset=None):
        _LiveSource.opened.append(tuple(state))
        self.window, self._played = {"start": 0, "offset": 0, "state": tuple(state)}, list(self._history.blocks[tuple(state)])

    def play(self):
        index = self._played.pop(0)
        self._played.append(index)
        return index, False


@pytest.fixture
def live_port(mounted_port):
    """The port with the live source above, under a kind of its own that replays no repair either."""
    kind = kinds.register_kind(kinds.EvidenceKind("demo_live", replays_repairs=False, record_keys=("level_key",),
                                                  check=kinds.kind_of("demo_history").check, loader=kinds.kind_of("demo_history").loader,
                                                  source=_LiveSource, truth=kinds.kind_of("demo_history").truth))
    data = json.loads((mounted_port / ports.ASSET_FILE).read_text(encoding="utf-8"))
    profile = register_profile(AssetProfile.from_json(dict(data, evidence_kind=kind.name)))
    _LiveSource.opened.clear()
    yield profile
    unregister_profile(PORT_ID)
    kinds.unregister_kind("demo_live")


def test_no_recording_is_played_after_the_first_close_of_a_work_order(live_port, port_records):
    for start, diagnosis in ((FAULTY, {"press": 1}), (HEALTHY, {"press": 0})):
        env = world.make_env(_config(start))
        env.reset(0)
        opened = len(_LiveSource.opened)
        env.step({"type": "read_sensors", "sensor_group": "all", "cycles": 2})
        before = next(e for e in reversed(env.log) if e["event"] == "read")["served"]  # the log keeps the hidden combination
        assert [o["combination"] for o in before] == [start["combination"]] * 2
        env.step({"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": []})
        env.step({"type": "create_work_order", "component": "press", "prediction_id": "", "evidence_ids": []})
        while env.clock.open_orders():
            env.step({"type": "wait_for_notification", "timeout": 1})
        envelope, _cost, _done, _info = env.step({"type": "read_sensors", "sensor_group": "all", "cycles": 3})
        assert envelope["result"]["served"] == [] and envelope["result"]["stopped"] == 0
        assert env.hidden == [0] and len(_LiveSource.opened) == opened  # reset opened the start window, and no window opened after it
        assert next(e for e in env.log if e["event"] == "work_order_created")["unnecessary"] == (start is HEALTHY)


def test_a_cycle_without_a_recording_is_neither_charged_nor_counted_as_stopped(port_records):
    timed, untimed = _config(), compose({"asset_id": PORT_ID, "layers": ["priced"], "start": FAULTY})
    assert timed.durations["read_per_cycle"] > 0 and untimed.durations["read_per_cycle"] == 0 and untimed.read_price("all", 1) > 0
    for config in (timed, untimed):
        env = world.make_env(config)
        env.reset(0)
        envelope, cost, _done, _info = env.step({"type": "read_sensors", "sensor_group": "all", "cycles": 3})
        assert envelope["result"] == {"sensor_group": "all", "served": [], "stopped": 0} and cost == 0
        assert env.log[-1]["cost"] == 0 and env.clock.stopped_cycles == set() and env.budget_left() == config.budget
    rows = _recorded_env(_config())
    rows.reset(0)
    read = {"type": "read_recorded", "tool": "history", "sensor_group": "all", "fields": list(FIELDS), "start": 0, "end": 3,
            "offset": 0, "limit": 3}
    envelope, cost, _done, _info = rows.step(read)
    assert envelope["result"]["cycles"] == [] and cost == 0 and rows.log[-1]["stopped"] == 0 and rows.time == 3


def test_the_recorded_work_orders_of_the_source_are_the_fix_history_of_the_episode(port_records):
    from CausalDiagnosis.datasets.demo_1.source import HISTORY

    env = world.make_env(_config())
    env.reset(0)
    envelope, _cost, _done, _info = env.step({"type": "get_fix_history"})
    assert envelope["result"]["history"] == list(HISTORY) == env.fix_history
    off = _config(fix_history={"healthy_probability": 0.0, "healthy_ago": [0, 0], "degraded_probability": 0.0, "degraded_ago": [0, 0]})
    env = world.make_env(off)
    env.reset(0)
    assert env.fix_history == []


def test_an_episode_of_the_port_runs_on_a_clock(port_records):
    for layers in (["priced", "real_time"], ["priced", "real_time", "advanced_real_time"]):
        config = compose({"asset_id": PORT_ID, "layers": layers, "start": FAULTY})
        env = _recorded_env(config)
        env.reset(0)
        env.step({"type": "get_current_time"}, think_cycles=2)
        assert env.time == 2  # the thinking time passes on a row with the real time clock
        log = run_episode(env, OraclePolicy(env), 0)
        order = next(e for e in log if e["event"] == "work_order_created")
        closed = next(e for e in log if e["event"] == "work_order_closed")
        took = config.lead_time("press") + config.durations["repair_downtime"]["press"]
        assert (took > 0) == ("advanced_real_time" in layers) and closed["t_end"] == order["t_start"] + took
        assert env.recorded_cycles() == list(range(-7, 0))  # nothing joins the record, before or after the close
        verdict = verify(log, released_from_env(env), env.oracle(), None)
        assert verdict["verified"] and verdict["pass"], verdict


def test_the_verifier_scores_the_port_with_the_truth_of_its_kind(port_records):
    from CausalDiagnosis.datasets.demo_1.source import MarkedTruth

    env = world.make_env(_config())
    run_episode(env, OraclePolicy(env), 0)
    MarkedTruth.calls.clear()
    assert _verdict(env)["pass"]
    assert MarkedTruth.calls and all(final == {"press": 0} for final in MarkedTruth.calls)


def test_both_task_servers_serve_the_port(port_records, tmp_path):
    """The action server and the task server behind the metering layer serve the port from its asset file alone: the
    parameter, the nouns and the statistic of the port, its site written in either case, its failure mode class, the bare
    fields of its one statistic, and both briefings with the sentences that its asset file states and the sentence of a
    kind that replays no repair."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents import ProfileDocuments
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract, episode_start_and_replay as rows
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.bind_episode_databases import bound, database_names
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing as rows_briefing
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import TASK_SERVER
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.in_memory_couchdb import MemoryServer
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import Meter, StateFileSessions, meter_config
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import Producer
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.cycle_timestamps import Timeline
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing as rig_briefing
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import PROFILE_PARTS, build_server

    profile, config = get_profile(PORT_ID), _config()
    state = tmp_path / "action" / "episode.json"
    state.parent.mkdir()
    episode.start(state, 0, config, port_records)
    episode.hand_over(state)
    tools = {tool.name: tool for tool in asyncio.run(build_server(state).list_tools())}
    assert list(tools["submit_diagnosis"].inputSchema["properties"]) == ["press", "evidence_ids"]
    texts = " ".join(tool.description for tool in tools.values())
    assert "the recorded value of every channel" in texts and "(press)" in texts and not re.search(OTHER_ASSETS, texts)

    timeline, names, site = Timeline(datetime(2026, 1, 1), config.seconds_per_cycle), database_names("eport0000000"), "Plant_7"
    path = tmp_path / "rows" / "state.json"
    path.parent.mkdir()
    started = rows.start(path, 0, config, port_records)
    episode.hand_over(path)
    store = MemoryServer()
    producer = Producer(store, names, ProfileDocuments(profile, timeline), timeline, PORT_ID, site)
    producer.reset(rows.replay(started))
    meter = Meter(meter_config(layer_contract.row_of(config), config, timeline), StateFileSessions(path), producer)
    fields = [profile.field_name(c) for c in profile.channels]
    with bound(store, names):
        listed = {tool.name: tool for tool in asyncio.run(meter.listed_tools(TASK_SERVER))}
        registry = store.all_docs(names.asset)
        history = asyncio.run(meter.call("iot", "history", {"site_name": site, "asset_id": PORT_ID, "sensors": fields}))
        asyncio.run(meter.call("fmsr", "get_failure_modes", {"asset_class": "press"}))
        asyncio.run(meter.call(TASK_SERVER, "submit_diagnosis", {"press": 1, "evidence_ids": ""}))
        order = asyncio.run(meter.call("wo", "generate_work_order", {"description": "realign", "asset_num": "demo_1_press", "site_id": site.lower()}))
    assert list(listed["submit_diagnosis"].inputSchema["properties"]) == ["press", "evidence_ids"]
    assert sorted((d["assetnum"], d["siteid"]) for d in registry) == [("demo_1", site), ("demo_1_press", site)]
    rows_read = json.loads(history.text.rsplit("\n", 1)[0])["observations"]
    assert len(rows_read) == 7 and all(set(row) - {"asset_id", "timestamp"} == set(fields) for row in rows_read)
    engine = rows.replay(episode.load_state(path))
    events = {e["event"]: e for e in engine.log}
    assert events["failure_modes"]["component"] == "press" and events["work_order_created"]["component"] == "press" and not order.is_error

    no_replay = "The records of this unit end at its first repair: from the close of the first work order on, no new reading is recorded."
    text = profile.text
    stated = [text["opening"], text["scoring"], text["level_meaning"], text["readings"], no_replay,
              "Which level does each part of the press of the stamping press stand at, within a budget of "]
    rig_env, rows_env = world.make_env(config), _recorded_env(config)
    rig_env.reset(0)
    rows_env.reset(0)
    leveled = dataclasses.replace(config, guidance_level=2)
    rig_text, rows_text = rig_briefing(rig_env.public_profile(), leveled), rows_briefing(rows_env.public_profile(), leveled)
    for brief in (rig_text, rows_text):
        assert all(brief.count(sentence) == 1 for sentence in stated), brief
        assert not re.search(OTHER_ASSETS, brief), brief
    assert "for every channel, the value under the installed sensor name." in rows_text
    assert set(rig_env.public_profile()) == {key for keys, _words in PROFILE_PARTS for key in keys}


def test_the_port_declares_its_label_and_its_table(port_records):
    env = _recorded_env(_config())
    env.reset(0)
    label = ports.start_label(env)
    assert label["separable"] is False and label["e18"]["separable"] is False
    env.step({"type": "read_recorded", "tool": "history", "sensor_group": "all", "fields": list(FIELDS), "start": -7, "end": 0,
              "offset": 0, "limit": 10})
    first = env.log[-1]["served"][0]
    env.step({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "all", "cycles": 1},
              "evidence_ids": [f"obs/{first['n']}/{FIELDS[0]}"]})
    env.step({"type": "finish", "summary": "The records do not tell the level of the press."})
    verdict = _verdict(env, grade=label)
    assert verdict["abstention_cites_reading"] and verdict["success"], verdict
    assert (f"CausalDiagnosis.datasets.{PORT_ID}.labels", "TABLE_PATH") in ports.generated_tables()
