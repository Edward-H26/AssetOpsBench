"""The three lines of the board's status on the work orders where a clock runs: the open orders on schedule with their closes and
how time passes on the row, a closed repair that waits for its verifying read with that read, and every repair verified, after which
a finish step ends the episode. Each line rests on public facts of the board alone and is printed only while its facts hold, with a
control for each fact; the finish guard of the controller reads the same tail, so a verified board lets a finish through and a board
that names the verifying read holds one back. Every test builds its board by hand, so no server runs."""
from types import SimpleNamespace

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import CausalLoopRunner
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import system_prompt
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY

CE, PS1, FS1 = "CE_CoolingEfficiency_percent_1Hz", "PS1_Pressure_bar_100Hz", "FS1_VolumeFlow_l_per_min_10Hz"
PROFILE = {
    "asset_id": "hyd_1", "components": list(COMPONENTS), "healthy": dict(HEALTHY),
    "levels": {"cooler": [3, 20, 100], "valve": [73, 80, 90, 100], "pump": [0, 1, 2], "accumulator": [90, 100, 115, 130]},
    "sensor_groups": {"tier1": [CE], "tier2": [PS1], "tier3": [FS1]},
    "prices": {"read": {"tier1": 1, "tier2": 3, "tier3": 10}, "parts": {"cooler": 20, "valve": 20, "pump": 30, "accumulator": 20}, "repair_charge": 20,
               "visit": 15},
    "durations": {"read_per_cycle": 1, "advance_per_cycle": 1, "lead_time": {c: 3 for c in COMPONENTS},
                  "repair_downtime": {"cooler": 1, "valve": 1, "pump": 2, "accumulator": 1}},
    "downtime_rate": {}, "budget": 300, "horizon": 60, "max_steps": 60, "statistics": ["mean", "std"]}
UNHEALTHY = {**HEALTHY, "pump": 1}
STATUS_LINES = ("Open work orders, on schedule", "and the repair is not verified yet", "Every work order has closed")


def _board(mode="declared", server=True, clock_runs=True, unserved=(), reads_take_time=True, recorded=False, time=1):
    """A committed board on a clocked row with the given clock mode; recorded gives it the recorded window of a row that records
    ahead of its reads, whose read serves the cycles before the clock."""
    board = Board(server=server, clock_runs=clock_runs, unserved=frozenset(unserved), reads_take_time=reads_take_time)
    board.set_profile({**PROFILE, "clock_mode": mode})
    board.manual_channels = {c: {f for fields in PROFILE["sensor_groups"].values() for f in fields} for c in COMPONENTS}
    board.status = {"time": time, "budget_left": 299, "steps_left": 57, "open_work_orders": [], "episode_over": False}
    if recorded:
        board.status.update(recorded_from=-10, recorded_until=time - 1)
    board.diagnosis = {"diagnosis": UNHEALTHY, "evidence_ids": []}
    return board


def _order(board, order_id, component, close, key="pred/b1", closed_t=None):
    board.add_order({"work_order": {"id": order_id, "component": component, "created_t": 1, "expected_close_t": close, "status": "open",
                                    "closed_t": None}}, key)
    if closed_t is not None:
        board.add_changes({"alarms": [], "work_orders_closed": [{"id": order_id, "component": component, "source": "agent", "closed_t": closed_t}]})


def _verdict(board, key, status="verified", consistent=True, predicted=HEALTHY, confirmed=None):
    """A verdict as the board keeps it; with the server it says whether the belief after the reading confirms the prediction, by
    default as consistent says, since an inconsistent verdict is never confirmed."""
    board.verifications[key] = {"status": status, "consistent": consistent, "predicted_combination": dict(predicted),
                                **({"confirmed": consistent if confirmed is None else confirmed} if board.server else {})}


def _status(board):
    return board._render_status()


def _order_lines(board):
    return [line for line in _status(board).split("\n") if line.startswith(STATUS_LINES) or " closed by cycle " in line]


def _guard(board):
    return CausalLoopRunner._unverified_reason(SimpleNamespace(board=board))


# ----- an open order on schedule -----
@pytest.mark.parametrize("mode, moves", [
    ("declared", "time passes only inside calls that take time"),
    ("generation_time", "time passes inside calls and with your response time"),
    ("generated_tokens", "time passes inside calls and with the tokens you generate"),
    ("instant", "every call and every turn without a call moves the clock one cycle")])
def test_an_open_order_on_schedule_is_named_with_its_close_and_how_time_passes_on_the_row(mode, moves):
    board = _board(mode=mode)
    _order(board, "wo-1", "pump", 6)
    _order(board, "wo-2", "valve", 8, key="pred/b2")
    assert _order_lines(board) == [
        "Open work orders, on schedule and not failed: wo-1 on the pump closes at cycle 6, wo-2 on the valve closes at cycle 8; the latest answer "
        f"was at cycle 1. On this row (clock mode {mode}) {moves}, and a wait step lets the asset run to the close."]
    assert _guard(board) is not None  # the finish guard holds a finish back while the order is open


def test_without_a_wait_the_line_names_the_read_whose_window_plays_to_the_close_or_no_means_where_no_read_does():
    forward = _board(unserved={"wait"}, reads_take_time=False, recorded=True)
    _order(forward, "wo-1", "pump", 6)
    assert _order_lines(forward) == [
        "Open work orders, on schedule and not failed: wo-1 on the pump closes at cycle 6; the latest answer was at cycle 1. On this row (clock "
        "mode declared) time passes only inside calls that take time, and a read of tier1 whose window reaches past the record plays the cycles to the close."]
    neither = _board(unserved={"wait"})
    _order(neither, "wo-1", "pump", 6)
    assert _order_lines(neither) == [
        "Open work orders, on schedule and not failed: wo-1 on the pump closes at cycle 6; the latest answer was at cycle 1. On this row (clock "
        "mode declared) time passes only inside calls that take time."]


def test_the_open_order_line_is_absent_where_its_facts_do_not_hold():
    no_clock = _board(clock_runs=False)
    _order(no_clock, "wo-1", "pump", 1)
    past = _board(time=7)  # the board's time lies past the close it holds, so it does not call the order on schedule
    _order(past, "wo-1", "pump", 6)
    horizon = _board()
    _order(horizon, "wo-1", "pump", 60)  # a close at the horizon comes after the episode
    nothing = _board()
    for board in (no_clock, past, horizon, nothing):
        assert _order_lines(board) == []
    partly = _board()
    _order(partly, "wo-1", "pump", 6)
    _order(partly, "wo-2", "valve", 61, key="pred/b2")
    assert _order_lines(partly)[0].startswith("Open work orders, on schedule and not failed: wo-1 on the pump closes at cycle 6; the latest")


# ----- a closed repair whose verification is pending -----
def test_a_closed_repair_names_the_read_that_verifies_it_now_when_its_first_cycle_has_come():
    board = _board(time=6)
    _order(board, "wo-1", "pump", 6, closed_t=6)
    _order(board, "wo-2", "valve", 5, closed_t=5)
    assert _order_lines(board) == ["Work orders wo-1 on the pump and wo-2 on the valve closed by cycle 6 and the repair is not verified yet: "
                                   "read tier1 now, and that reading verifies it."]
    assert _guard(board) is not None  # the finish guard refuses a finish before that reading


def test_on_a_row_that_records_ahead_of_its_reads_the_line_names_the_wait_to_the_first_verifying_cycle():
    board = _board(time=6, recorded=True)
    _order(board, "wo-1", "pump", 6, closed_t=6)
    assert _order_lines(board) == ["Work order wo-1 on the pump closed by cycle 6 and the repair is not verified yet: wait to cycle 7, then read "
                                   "tier1, and that reading verifies it."]
    assert _guard(board) is not None


def test_without_a_wait_the_line_names_the_read_whose_window_plays_past_the_close():
    board = _board(time=6, recorded=True, unserved={"wait"}, reads_take_time=False)
    _order(board, "wo-1", "pump", 6, closed_t=6)
    assert _order_lines(board) == ["Work order wo-1 on the pump closed by cycle 6 and the repair is not verified yet: read tier1 for 1 cycle, "
                                   "whose window plays past the close, and that reading verifies it."]
    assert _guard(board) is not None


def test_the_closed_repair_line_is_absent_where_its_facts_do_not_hold():
    still_open = _board(time=6)
    _order(still_open, "wo-1", "pump", 6, closed_t=6)
    _order(still_open, "wo-2", "valve", 9, key="pred/b2")  # another order is open, so the line of the open order stands alone
    assert [line for line in _order_lines(still_open) if "not verified yet" in line] == []
    verified = _board(time=7)
    _order(verified, "wo-1", "pump", 6, closed_t=6)
    _verdict(verified, "pred/b1")
    no_clock = _board(time=6, clock_runs=False)
    _order(no_clock, "wo-1", "pump", 6, closed_t=6)
    unpaid = _board(time=6)  # the tail does not fit the steps left, so the guard lets a finish through and no read is named
    unpaid.status["steps_left"] = 1
    _order(unpaid, "wo-1", "pump", 6, closed_t=6)
    for board in (verified, no_clock, unpaid):
        assert [line for line in _order_lines(board) if "not verified yet" in line] == []
    assert _guard(unpaid) is None


# ----- every repair verified -----
@pytest.mark.parametrize("server, status", [(True, "verified"), (False, "verified by the normal ranges")])
def test_every_repair_verified_with_the_latest_consistent_says_that_a_finish_step_ends_the_episode(server, status):
    board = _board(server=server, time=8)
    _order(board, "wo-1", "pump", 6, closed_t=6)
    _verdict(board, "pred/b1", status)
    latest = "confirmed" if server else "consistent"  # with the server the latest verdicts are also confirmed by the belief
    line = f"Every work order has closed and every repair is verified, the latest {latest}: no wait or read is pending, and a finish step ends the episode."
    assert _order_lines(board) == [line]
    assert _guard(board) is None  # the finish guard lets the finish through


def test_an_earlier_refuted_repair_does_not_hold_the_line_back_when_the_latest_is_consistent():
    board = _board(time=12)
    _order(board, "wo-1", "valve", 6, key="pred/b2", closed_t=6)
    _verdict(board, "pred/b2", consistent=False, predicted={**HEALTHY, "valve": 90})
    _order(board, "wo-2", "pump", 10, closed_t=10)
    _verdict(board, "pred/b1")
    assert _order_lines(board) == ["Every work order has closed and every repair is verified, the latest confirmed: no wait or read is pending, "
                                   "and a finish step ends the episode."]


@pytest.mark.parametrize("change", ["inconsistent", "partial", "not comparable", "error", "action refused", "open", "no clock"])
def test_the_verified_line_is_absent_where_its_facts_do_not_hold(change):
    board = _board(time=8, clock_runs=change != "no clock")
    _order(board, "wo-1", "pump", 6, closed_t=6)
    _verdict(board, "pred/b1", consistent=change != "inconsistent", predicted=UNHEALTHY if change == "partial" else HEALTHY)
    if change in ("not comparable", "error", "action refused"):
        board.verifications["pred/b1"]["status"] = f"{change}: a reason"
    if change == "open":
        _order(board, "wo-2", "valve", 9, key="pred/b2")
    assert [line for line in _order_lines(board) if line.startswith("Every work order")] == []


@pytest.mark.parametrize("server", [True, False])
def test_a_partial_repair_verified_consistent_leaves_the_verified_line_out_until_every_diagnosed_component_holds_a_closed_order(server):
    """The diagnosis names the pump and the valve faulty; the pump alone is ordered, closed and verified consistent (with the server
    predicting the healthy combination, as a prediction can when the belief holds the valve healthy), so the line stays out. The
    control repairs both in one plan, verified the same way, and the line stands."""
    status = "verified" if server else "verified by the normal ranges"
    partial = _board(server=server, time=8)
    partial.diagnosis = {"diagnosis": {**HEALTHY, "pump": 1, "valve": 90}, "evidence_ids": []}
    _order(partial, "wo-1", "pump", 6, closed_t=6)
    _verdict(partial, "pred/b1", status)
    assert [line for line in _order_lines(partial) if line.startswith("Every work order")] == []
    both = _board(server=server, time=9)
    both.diagnosis = dict(partial.diagnosis)
    _order(both, "wo-1", "pump", 7, closed_t=7)
    _order(both, "wo-2", "valve", 6, closed_t=6)
    _verdict(both, "pred/b1", status)
    assert [line for line in _order_lines(both) if line.startswith("Every work order")]


def test_with_the_server_a_consistent_verdict_that_is_not_confirmed_leaves_the_verified_line_out_and_lets_the_finish_through():
    """verify_outcome confirms a consistent verdict only when the belief after the reading puts its confirmation mass on the predicted
    combination and the verifying readings do not favor its strongest rival. With the server the verified line waits for a confirmed
    verdict, while the finish guard, which reads the pending verifications alone, lets the finish through either way; the control
    confirms the same verdict and the line stands."""
    unconfirmed = _board(time=8)
    _order(unconfirmed, "wo-1", "pump", 6, closed_t=6)
    _verdict(unconfirmed, "pred/b1", confirmed=False)
    assert [line for line in _order_lines(unconfirmed) if line.startswith("Every work order")] == []
    assert _guard(unconfirmed) is None
    confirmed = _board(time=8)
    _order(confirmed, "wo-1", "pump", 6, closed_t=6)
    _verdict(confirmed, "pred/b1", confirmed=True)
    assert [line for line in _order_lines(confirmed) if line.startswith("Every work order")]
    assert _guard(confirmed) is None


def test_without_the_server_a_consistent_verdict_on_the_normal_ranges_needs_no_predicted_combination():
    board = _board(server=False, time=8)
    _order(board, "wo-1", "pump", 6, key="plan 3", closed_t=6)
    board.verifications["plan 3"] = {"status": "verified by the normal ranges", "consistent": True}
    assert [line for line in _order_lines(board) if line.startswith("Every work order")]


# ----- the lines on the path that ships: an episode through AssetOpsBench's servers -----
def _progress_plans(ordered):
    """The scripted model of the visit tests (_visit_plans) until its orders stand and while one is open; once every order closed it
    waits one cycle and reads without finishing while the board says the repair is not verified yet, and finishes once the board
    states no such line, so that the verified line reaches a planning turn."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (ORDER_ROW, VERIFYING_READ, _section, _visit_plans,
                                                                                               plan_json)
    visit = _visit_plans(ordered)

    def plans(prompt, call):
        if "Commitment: none" in prompt or any(row[4] == "open" for row in ORDER_ROW.findall(_section(prompt, "## Work orders"))):
            return visit(prompt, call)
        if "the repair is not verified yet" in prompt:
            return plan_json([{"kind": "wait", "args": {"cycles": 1}, "expect": {}, "evidence": []}, dict(VERIFYING_READ)])
        return plan_json([{"kind": "finish", "args": {}, "expect": {}, "evidence": []}])
    return plans


def test_an_episode_on_the_repair_rules_row_shows_each_line_in_its_turn(tmp_path):
    """Two faulty components ordered in one visit through AssetOpsBench's servers on the memory store, with a thinking rate that no call
    reaches: while the orders are open every planning turn names them on schedule with the closes at which the environment closed them
    and the clock mode of the row; the first turn after the last close names the wait to the first cycle that verifies, the cycle after
    the close on a row whose read serves the cycles before the clock; and once the reading verified the repair a turn states that a
    finish step ends the episode, and that finish ends it."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (ScriptedPlanner, UNREACHED_THINKING, _aob_run,
                                                                                               _planning_prompts, _visit_start)
    from CausalDiagnosis.utils.json_files import read_json
    start, faulty = _visit_start("repair_rules")
    planner = ScriptedPlanner(_progress_plans(faulty))
    ep_dir, row, _backend = _aob_run(tmp_path, "repair_rules", planner, {}, start=start, think_seconds=UNREACHED_THINKING)
    assert row["error"] == "" and row["end_reason"] == "finish", row
    orders = read_json(ep_dir / "plans.json")["board"]["orders"]
    last = max(o["closed_t"] for o in orders.values())
    prompts = _planning_prompts(planner)
    on_schedule = [p for p in prompts if "Open work orders, on schedule and not failed: " in p]
    assert on_schedule and all("(clock mode generation_time)" in p for p in on_schedule)
    for o in orders.values():
        assert any(f"{o['id']} on the {o['component']} closes at cycle {o['closed_t']}" in p for p in on_schedule), o
    unverified = [p for p in prompts if "and the repair is not verified yet: " in p]
    assert unverified and f"closed by cycle {last} and the repair is not verified yet: wait to cycle {last + 1}, then read " in unverified[0]
    verified = [i for i, p in enumerate(prompts) if "Every work order has closed and every repair is verified, the latest confirmed" in p]
    assert verified == [len(prompts) - 1]


# ----- the system prompt -----
def test_the_system_prompt_with_the_server_says_when_a_verdict_is_confirmed_and_the_prompt_without_it_does_not():
    sentence = ("The verdict also says whether the belief after that reading confirms the prediction, and names the belief's most probable "
                "combination with its mass when it does not; after a later reading the controller asks again, and the latest verdict counts. "
                "The status says that a finish step ends the episode only once the latest repair is confirmed, and a finish is never held "
                "back for it.")
    assert sentence in system_prompt(8, server=True) and sentence not in system_prompt(8, server=False)


@pytest.mark.parametrize("server", [True, False])
def test_both_system_prompts_introduce_the_three_lines(server):
    sentence = ("Where a clock runs, the status also states the open work orders on schedule with their closes and how time passes on the row, the read "
                "that verifies a repair whose orders have closed, and, once every repair is verified, that a finish step ends the episode.")
    assert sentence in system_prompt(8, server=server)
