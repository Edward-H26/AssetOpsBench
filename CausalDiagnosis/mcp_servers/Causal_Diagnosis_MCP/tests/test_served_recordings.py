"""What a row can still serve to a reading (served_recordings.py): the count of the recordings a sensor group has not been shown in
full, from the readings of the episode and the length of the window alone."""
import json
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import LIVE_WINDOW, profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Reading
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import Left, left_to_serve
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests import advice_walk as aw
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as server

VIEW = profile_view()
GROUPS = sorted(VIEW.sensor_groups, key=lambda g: len(VIEW.group_fields(g)))  # the smallest group first, the covering group last
SMALLEST, COVERING = GROUPS[0], GROUPS[-1]


def values(fields, base):
    return {f: float(base + i) for i, f in enumerate(sorted(fields))}


def test_before_any_reading_every_group_has_the_whole_window_left():
    left = left_to_serve(VIEW, [], LIVE_WINDOW)
    assert set(left) == set(VIEW.sensor_groups)
    assert all(left[g] == Left(LIVE_WINDOW, LIVE_WINDOW * len(VIEW.group_fields(g))) for g in VIEW.sensor_groups)


def test_a_second_group_reading_one_cycle_joins_its_recording():
    """After the smallest group has read one cycle it has one cycle less, and the covering group still has every cycle with
    fewer summaries left. A reading of that cycle that brings only the summaries the covering group adds, as the engine of
    the AssetOpsBench rows releases a second group on a recorded cycle, makes one recording of the two, since a record
    names both readings by their cycle."""
    small, large = set(VIEW.group_fields(SMALLEST)), set(VIEW.group_fields(COVERING))
    assert small < large
    first = Reading(1, -1, SMALLEST, values(small, 0), recorded=True)
    left = left_to_serve(VIEW, [first], LIVE_WINDOW)
    assert left[SMALLEST] == Left(LIVE_WINDOW - 1, (LIVE_WINDOW - 1) * len(small))
    assert left[COVERING] == Left(LIVE_WINDOW, LIVE_WINDOW * len(large) - len(small))
    second = Reading(2, -1, COVERING, values(large - small, 100), recorded=True)
    left = left_to_serve(VIEW, [first, second], LIVE_WINDOW)
    assert left[COVERING] == Left(LIVE_WINDOW - 1, (LIVE_WINDOW - 1) * len(large))
    assert left[SMALLEST] == Left(LIVE_WINDOW - 1, (LIVE_WINDOW - 1) * len(small))


def test_a_reading_of_a_covering_group_leaves_nothing_for_the_groups_it_covers():
    fields = VIEW.group_fields(COVERING)
    readings = [Reading(n, t, COVERING, values(fields, 10 * n)) for n, t in enumerate(range(-LIVE_WINDOW, 0), start=1)]
    left = left_to_serve(VIEW, readings, LIVE_WINDOW)
    assert all(left[g] == Left(0, 0) for g in VIEW.sensor_groups)


def test_a_recording_served_again_counts_once_and_the_count_never_falls_below_zero():
    """Readings at one cycle with different values on a shared summary are distinct recordings, as on a row that plays a
    recording per read. A reading that agrees with an earlier one on every summary they share is that recording again."""
    small = set(VIEW.group_fields(SMALLEST))
    one, other = Reading(1, 0, SMALLEST, values(small, 0)), Reading(2, 0, SMALLEST, values(small, 50))
    assert left_to_serve(VIEW, [one, other], LIVE_WINDOW)[SMALLEST].cycles == LIVE_WINDOW - 2
    again = Reading(3, 0, SMALLEST, dict(one.values))
    assert left_to_serve(VIEW, [one, other, again], LIVE_WINDOW)[SMALLEST].cycles == LIVE_WINDOW - 2
    many = [Reading(n, 0, SMALLEST, values(small, 1000 * n)) for n in range(1, LIVE_WINDOW + 3)]
    assert left_to_serve(VIEW, many, LIVE_WINDOW)[SMALLEST] == Left(0, 0)


# ----- one recording, one rule, and what every row can still serve -----

def released_of(reads, rules=None, status=None, events=()):
    """A released document with one read event per reading, (sensor group, cycle, values, tool, cycle of the read's end),
    where the event names the tool that served it as the engine of the AssetOpsBench rows writes it, or no tool, as the
    rig's own engine writes it, after the given events."""
    log, observations = list(events), []
    for n, (group, t, vals, tool, end) in enumerate(reads, start=1):
        log.append({"event": "read", "t_end": end, "sensor_group": group, "served": [{"n": n}], **({"tool": tool} if tool else {})})
        observations.append({"n": n, "t": t, "sensor_group": group, "values": dict(vals)})
    return {"events": log, "observations": observations, "status": status or {"time": 0}, "rules": dict(rules or VIEW.rules)}


def record_view():
    """The view of a row that reads a record: the default rules with the history tool of the AssetOpsBench rows listed."""
    return profile_view({**VIEW.rules, "tools": [{"name": "history", "server": "iot"}]})


def space():
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import EffectSpace
    return EffectSpace.from_view(VIEW)


def test_a_second_read_of_one_recorded_cycle_joins_the_recording_of_that_cycle():
    """One recorded cycle read through the smallest group and then through the covering group, which releases the other
    summaries alone, as the engine of the AssetOpsBench rows does: one recording with the union and both numbers, and one
    observation of the filter."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import merge_observations, parse_released
    small, large = set(VIEW.group_fields(SMALLEST)), set(VIEW.group_fields(COVERING))
    episode = parse_released(released_of([(SMALLEST, -1, values(small, 0), "history", 0), (COVERING, -1, values(large - small, 100), "history", 0)]))
    observations = merge_observations(episode.readings, space())
    assert [o.numbers for o in observations] == [(1, 2)] and set(observations[0].fields) == large
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import group_recordings
    (recording,) = group_recordings(episode.readings)
    assert recording.values == {**values(small, 0), **values(large - small, 100)} and recording.numbers == [1, 2]


def test_two_reads_at_one_cycle_of_a_row_that_plays_a_recording_per_read_stay_two_recordings():
    """The same two readings released by an engine that plays a recording per read, whose read events name no tool, are two
    recordings: one cycle index there names no one recording."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import merge_observations, parse_released
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import recordings
    small, large = set(VIEW.group_fields(SMALLEST)), set(VIEW.group_fields(COVERING))
    episode = parse_released(released_of([(SMALLEST, -1, values(small, 0), None, 0), (COVERING, -1, values(large - small, 100), None, 0)]))
    assert len(recordings(episode.readings)) == 2 and len(merge_observations(episode.readings, space())) == 2


def test_a_wrapped_repeat_joins_its_recording_and_a_later_read_of_the_repeat_joins_it_by_cycle():
    """Cycle t and cycle t + W with equal values are one recording, which keeps the cycle and the standing mark of its first
    reading. A read of cycle t + W through summaries that the recording does not hold yet joins it by its cycle."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import merge_observations, parse_released
    small, large = set(VIEW.group_fields(SMALLEST)), set(VIEW.group_fields(COVERING))
    t = -2
    reads = [(SMALLEST, t, values(small, 0), "history", t), (SMALLEST, t + LIVE_WINDOW, values(small, 0), "history", 0),
             (COVERING, t + LIVE_WINDOW, values(large - small, 100), "history", 0)]
    episode = parse_released(released_of(reads))
    assert [r.standing for r in episode.readings] == [True, False, False]
    observations = merge_observations(episode.readings, space())
    assert [(o.numbers, o.t, o.standing) for o in observations] == [((1, 2, 3), t, True)]
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import group_recordings
    (recording,) = group_recordings(episode.readings)
    assert recording.cycles == {t, t + LIVE_WINDOW} and set(recording.values) == large


def test_the_filter_and_the_serving_group_the_readings_alike():
    """For a list that mixes the reads of a record and the reads of an engine that plays a recording per read, the filter
    holds one observation per recording of the serving, with the same summaries and the same numbers."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import merge_observations, parse_released
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import recordings
    small, large = set(VIEW.group_fields(SMALLEST)), set(VIEW.group_fields(COVERING))
    reads = [(SMALLEST, -1, values(small, 0), None, 0), (COVERING, -1, values(large - small, 100), None, 0),
             (SMALLEST, -3, values(small, 200), "history", 0), (COVERING, -3, values(large - small, 300), "history", 0),
             (SMALLEST, 2, values(small, 200), "history", 0), (SMALLEST, 0, values(small, 400), None, 0)]
    readings = parse_released(released_of(reads)).readings
    observations = merge_observations(readings, space())
    shown = recordings(readings)
    assert len(shown) == len(observations) and [dict(zip(o.fields, o.x)) for o in observations] == shown
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import group_recordings
    assert [tuple(item.numbers) for item in group_recordings(readings)] == [o.numbers for o in observations]


@pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
@pytest.mark.parametrize("row", layers.LADDER)
def test_what_the_server_counts_as_the_record_is_what_the_engine_holds(row, tmp_path):
    """An episode on the engine of the AssetOpsBench rows on every row of the environment table: thinking time, a read, a
    diagnosis, a work order with its repair, waits until it closes and a read after the close. After every step the
    recorded cycles that the server takes from the released evidence are those the engine holds."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import parse_released
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import recorded_cycles
    cycles, blocks = aw.world()
    start = next(s for s in aw.development_combinations() if VIEW.faulty(s))  # a development start with a fault
    env = RecordedRigEnv(cycles, blocks, aw.started(row, start, 0))
    env.reset(0)
    state = tmp_path / "state.json"

    def same():
        server.write_released(state, env)
        episode = parse_released(json.loads(server.released_path(state).read_text(encoding="utf-8")))
        assert recorded_cycles(episode, LIVE_WINDOW) == env.recorded_cycles()

    def read(think):
        newest = max(env.recorded_cycles())
        env.step({"type": "read_recorded", "tool": "history", "sensor_group": COVERING, "fields": sorted(env._group_fields(COVERING)),
                  "start": newest - 1, "end": newest + 1, "offset": 0, "limit": 1000}, think_cycles=think)

    same()
    read(think=2)
    same()
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(VIEW.components, start)), "evidence_ids": []}, think_cycles=1)
    same()
    env.step({"type": "create_work_order", "component": VIEW.faulty(start)[0], "prediction_id": "", "evidence_ids": []})
    same()
    for _ in range(12):
        if not env.public_work_orders() or env.done:
            break
        env.step({"type": "wait_for_notification", "timeout": 3})
        same()
    assert not env.public_work_orders()
    env.step({"type": "wait_for_notification", "timeout": 2})
    same()
    read(think=1)
    same()


@pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
@pytest.mark.parametrize("row", layers.LADDER)
def test_a_row_reads_a_record_exactly_when_its_engine_keeps_one(row, tmp_path):
    """On every row of the environment table the view of the released rules reads a record on the engine of the
    AssetOpsBench rows and none on the rig's own engine, and every reading of the first carries the mark of a record and
    none of the second does."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import parse_released
    cycles, blocks = aw.world()
    start = next(s for s in aw.development_combinations() if VIEW.faulty(s))  # a development start with a fault
    config = aw.started(row, start, 0)
    marks = {}
    for name, env_class, action in (
            ("record", RecordedRigEnv, {"type": "read_recorded", "tool": "history", "sensor_group": SMALLEST, "fields": sorted(VIEW.group_fields(SMALLEST)),
                                        "start": -2, "end": 0, "offset": 0, "limit": 1000}),
            ("rig", HydraulicRigEnv, {"type": "read_sensors", "sensor_group": SMALLEST, "cycles": 2})):
        env = env_class(cycles, blocks, config)
        env.reset(0)
        env.step(action)
        state = tmp_path / name / "state.json"
        state.parent.mkdir()
        server.write_released(state, env)
        raw = json.loads(server.released_path(state).read_text(encoding="utf-8"))
        readings = parse_released(raw).readings
        marks[name] = (profile_view(dict(raw["rules"], asset_id=raw["asset_id"])).reads_record, {r.recorded for r in readings}, len(readings))
    assert marks["record"][:2] == (True, {True}) and marks["rig"][:2] == (False, {False}) and marks["record"][2] and marks["rig"][2]


def test_the_first_read_of_a_row_with_a_clock_takes_every_recorded_cycle():
    """After k played cycles and no reading every group can be served the W recorded cycles before the episode and the k
    played ones. One read of them all through the covering group leaves every group nothing, also once further cycles are
    recorded, since every group has then been shown a window of recordings."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Episode
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import left_in_episode
    view, k = record_view(), 4
    assert view.reads_record
    before = left_in_episode(view, Episode(k, [], [], None, [], True), LIVE_WINDOW)
    assert before == {g: Left(LIVE_WINDOW + k, (LIVE_WINDOW + k) * len(VIEW.group_fields(g))) for g in VIEW.sensor_groups}
    large = VIEW.group_fields(COVERING)
    readings = [Reading(n, t, COVERING, values(large, 100 * n), recorded=True) for n, t in enumerate(range(-LIVE_WINDOW, k), start=1)]
    for time in (k, k + 3):
        assert left_in_episode(view, Episode(time, readings, [], None, [], True), LIVE_WINDOW) == {g: Left(0, 0) for g in VIEW.sensor_groups}


def test_a_stopped_cycle_is_no_cycle_of_the_record_and_the_count_starts_again_at_a_closure():
    """An order that stops the asset for two cycles: the stopped cycles are no cycles of the record, while the repair runs
    every cycle recorded before it counts, and after the closure only the cycles played after it count."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import parse_released
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import left_in_episode, recorded_cycles, since_closure
    component, a = VIEW.components[0], 6
    rules = {**record_view().rules, "clock_mode": "generation_time"}
    created = {"event": "work_order_created", "id": "wo/1", "component": component, "mode": "standard", "t_end": a - 2}
    executed = {"event": "work_order_executed", "id": "wo/1", "t_start": a, "t_end": a}
    closed = {"event": "work_order_closed", "id": "wo/1", "component": component, "t_end": a + 2}
    during = parse_released(released_of([], rules, {"time": a + 1}, [created, executed]))
    assert during.stops == ((a, a + 1),) and recorded_cycles(during, LIVE_WINDOW) == [*range(-LIVE_WINDOW, a)]
    after = parse_released(released_of([], rules, {"time": a + 5}, [created, executed, closed]))
    assert after.stops == ((a, a + 2),) and recorded_cycles(after, LIVE_WINDOW) == [*range(-LIVE_WINDOW, a), a + 2, a + 3, a + 4]
    assert since_closure(after, LIVE_WINDOW) == ([], a + 2)
    view = record_view()
    assert left_in_episode(view, during, LIVE_WINDOW) == {g: Left(LIVE_WINDOW + a, (LIVE_WINDOW + a) * len(VIEW.group_fields(g))) for g in VIEW.sensor_groups}
    assert left_in_episode(view, after, LIVE_WINDOW) == {g: Left(3, 3 * len(VIEW.group_fields(g))) for g in VIEW.sensor_groups}


def test_a_group_is_closed_after_one_window_of_recordings():
    """A group that has been shown one window of recordings in full has nothing left, on a row that reads a record while
    the record holds more cycles, and on a row that plays a recording per read, while a group that lacks summaries of
    those recordings stays open."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Episode
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import left_in_episode
    small, large = VIEW.group_fields(SMALLEST), VIEW.group_fields(COVERING)
    marked = [Reading(n, t, SMALLEST, values(small, 100 * n), recorded=True) for n, t in enumerate(range(-LIVE_WINDOW, 0), start=1)]
    on_record = left_in_episode(record_view(), Episode(3, marked, [], None, [], True), LIVE_WINDOW)
    assert on_record[SMALLEST] == Left(0, 0)
    assert on_record[COVERING] == Left(LIVE_WINDOW + 3, LIVE_WINDOW * (len(large) - len(small)) + 3 * len(large))
    played = [Reading(n, 0, SMALLEST, values(small, 100 * n)) for n in range(1, LIVE_WINDOW + 1)]
    per_read = left_in_episode(VIEW, Episode(0, played, [], None, [], False), LIVE_WINDOW)
    assert per_read[SMALLEST] == Left(0, 0) and per_read[COVERING].cycles == LIVE_WINDOW
