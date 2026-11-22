"""Standing steps (plan/standing.py): a step whose effect already stands on the board makes no call and does not stop the plan.
The five clauses on the board and through the driver, each beside the one fact that falsifies it; what the check still refuses;
the walk of the plan check; the turn of standing steps alone; the records of an episode; compaction and recall; the option of the
reply schema; and the walk of the episode viewer, which places every recorded call on the step that made it."""
import asyncio
import contextlib
import json
import re
from unittest import mock

import pytest

from CausalDiagnosis.agents.utils import core_texts
from CausalDiagnosis.agents.Causal_Diagnosis_Agent import controller, episode_result
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import OMITTED_AT_DEFAULT, CausalLoopOptions
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import Plan, Step, check_plan, plan_check, standing, step_problems, submitted_levels
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.standing import (EPISODE_ANSWER, NOTHING_NEW, SAME_COMMITMENT, STANDS_STATE, STATE_HOLDS,
                                                                         SURFACE_ANSWER, Standing, Walked)
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board.board import KIND_WORDS, call_text, commitment_text
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (
    ScriptedPlanner, _covering_cap, _open_order_board, _play, _synthetic_board, act_now, plan_json)
from CausalDiagnosis.episode_viewer.walkthrough import controller_walk
from CausalDiagnosis.episode_viewer.walkthrough.episode_files import RIG_TOOLS, Call
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, RIG_SERVER
from CausalDiagnosis.utils.json_files import read_json


TIER1 = {"sensor_group": "tier1", "cycles": 3}


def _answered(board, *requests):
    """The board with these requests of the opening answered."""
    for kind, request in requests:
        board.note_answered(kind, request)
    return board


def _plan(*steps, mode="test_first"):
    return Plan(branch=1, mode=mode, why="", steps=list(steps))


# ----- clause episode_answer -----

def test_a_read_of_the_opening_whose_answer_the_board_holds_stands_for_the_whole_episode():
    board = _synthetic_board()
    steps = {"profile": Step("profile"), "fix_history": Step("fix_history"), "failure_modes {'component': 'cooler'}": Step("failure_modes", {"component": "cooler"})}
    assert all(standing(step, board) is None for step in steps.values())  # control: nothing answered
    _answered(board, ("profile", {}), ("fix_history", {}), ("failure_modes", {"component": "cooler"}))
    for call, step in steps.items():
        assert standing(step, board) == Standing(EPISODE_ANSWER, core_texts.STANDS_EPISODE.format(call=call)), call
    assert standing(Step("failure_modes", {"component": "valve"}), board) is None
    assert board.clock_runs  # an answer that holds for the whole episode stands where the clock runs as well


def test_a_failure_modes_step_for_every_component_stands_when_every_request_is_answered_and_calls_only_for_the_others():
    board = _answered(_synthetic_board(), ("failure_modes", {"component": "cooler"}), ("failure_modes", {"component": "valve"}))
    every = Step("failure_modes", {"component": ""})
    assert standing(every, board) is None
    board.status["steps_left"] = 2
    plan = _plan(every, Step("read", {"sensor_group": "tier1", "cycles": 1}))
    assert any("steps are left" in r for r in check_plan(plan, board))  # two components unanswered and the read: three calls
    _answered(board, ("failure_modes", {"component": "pump"}))
    assert check_plan(plan, board) == []  # one component unanswered and the read: two calls
    _answered(board, ("failure_modes", {"component": "accumulator"}))
    assert standing(every, board).reason == core_texts.STANDS_EPISODE.format(call=call_text("failure_modes", {"component": ""}))


# ----- clause surface_answer -----

def test_a_read_that_the_surface_needs_first_stands_once_answered_where_no_cycle_passes_and_is_called_where_one_does():
    board = _synthetic_board()
    board.required_opening, board.opening_done = ["extent"], ["extent"]
    _answered(board, ("extent", {}))
    board.clock_runs = False
    assert standing(Step("extent"), board) == Standing(SURFACE_ANSWER, core_texts.STANDS_SURFACE.format(call="extent"))
    assert standing(Step("time"), board) is None  # not answered
    board.status["steps_left"] = 1
    assert check_plan(_plan(Step("extent"), Step("read", {"sensor_group": "tier1", "cycles": 1})), board) == []
    board.clock_runs = True  # the clock moves the answer, so the call is made
    assert standing(Step("extent"), board) is None
    assert any("steps are left" in r for r in check_plan(_plan(Step("extent"), Step("read", {"sensor_group": "tier1", "cycles": 1})), board))


# ----- clause nothing_new -----

def _recorded(board, first, last, clock=False):
    """The board with a recorded window from first to last, the clock running or not."""
    board.status.update(recorded_from=first, recorded_until=last)
    board.clock_runs = clock
    return board


def _clocked(board, mode):
    """The board with the clock mode of its profile, which the synthetic profile declares."""
    board.profile["clock_mode"] = mode
    return board


def test_a_read_of_a_group_whose_every_recorded_cycle_the_board_holds_stands_where_no_cycle_passes():
    board = _recorded(_synthetic_board(), 0, 0)  # the board holds cycle 0 of tier1
    read = Step("read", TIER1)
    assert standing(read, board) == Standing(NOTHING_NEW, core_texts.STANDS_NOTHING_NEW.format(**KIND_WORDS, name="tier1"))
    assert standing(read, _recorded(_synthetic_board(), -1, 0)) is None  # one recorded cycle unread
    assert standing(read, _clocked(_recorded(_synthetic_board(), 0, 0, clock=True), "generation_time")) is None  # thinking grows the record
    unrecorded = _synthetic_board()
    unrecorded.clock_runs = False
    assert standing(read, unrecorded) is None  # a surface that plays the cycles it serves: every cycle of a read is new


def test_a_read_of_a_group_whose_every_recorded_cycle_the_board_holds_stands_on_a_declared_clock_until_a_step_moves_the_walk():
    """Where the clock runs by declaration the record grows only inside a call that takes time, so a read of a group whose every
    recorded cycle the board holds stands with the reason of that clock and the walk counts no call for it, while after a wait of
    the same plan, which records the cycles it plays, the walk counts the read as a call, since the board does not hold the cycles
    that the wait records (a read that stood there would fit the plan into a step cap one call too small). Thinking and the instant
    mode record cycles outside such calls, so there the call is made."""
    board = _recorded(_synthetic_board(), 0, 0, clock=True)
    assert board.profile["clock_mode"] == "declared" and "wait" not in board.unserved
    read = Step("read", TIER1)
    assert standing(read, board) == Standing(NOTHING_NEW, core_texts.STANDS_NOTHING_RECORDED.format(**KIND_WORDS, name="tier1"))
    for mode in ("generation_time", "generated_tokens", "instant"):
        assert standing(read, _clocked(_recorded(_synthetic_board(), 0, 0, clock=True), mode)) is None, mode
    board.status["steps_left"] = 3
    assert step_problems(_plan(read, Step("finish")), board) == []  # the read stands: the finish is the one call
    waited = step_problems(_plan(Step("wait", {"cycles": 1}), read, Step("finish")), board)  # the wait and its change summary, the read, the finish
    assert [step for step, _reason in waited] == [None] and "needs 4 tool calls" in waited[0][1], waited


def test_a_read_stands_where_a_group_that_covers_its_fields_was_read():
    board = _recorded(_synthetic_board(), -1, 0)
    board.profile["sensor_groups"]["full"] = [f for fields in list(board.profile["sensor_groups"].values()) for f in fields]
    assert standing(Step("read", TIER1), board) is None
    board.add_reading({"sensor_group": "full", "served": [{"n": 2, "t": -1, "values": {"CE_CoolingEfficiency_percent_1Hz_mean": 30.0}}]})
    assert standing(Step("read", TIER1), board).clause == NOTHING_NEW


def test_a_standing_read_after_a_compaction_still_stands_and_names_the_recall_step():
    board = _recorded(_synthetic_board(), 0, 0)
    board.compact(fits=lambda b: False)  # past the trigger, every absorbed row collapses into a summary row
    assert board.collapsed_rows and not board.evidence
    stands = standing(Step("read", TIER1), board)
    assert stands.reason == (core_texts.STANDS_NOTHING_NEW.format(**KIND_WORDS, name="tier1")
                             + core_texts.STANDS_COLLAPSED.format(first=1, last=1, recall_step="recall"))
    board.recall_offered = False
    assert standing(Step("read", TIER1), board).reason == core_texts.STANDS_NOTHING_NEW.format(**KIND_WORDS, name="tier1")


def test_after_a_summary_of_earlier_turns_every_clause_decides_as_before():
    board = _answered(_recorded(_synthetic_board(), 0, 0), ("profile", {}))
    board.diagnosis = {"diagnosis": dict(submitted_levels(board, board.candidates()[0])), "evidence_ids": []}
    steps = [Step("profile"), Step("read", TIER1), Step("diagnose", {"rank": 1}), Step("diagnose", {"rank": 2})]
    before = [standing(s, board, walked=Walked(candidates=board.candidates())) for s in steps]
    board.add_note("the ranking failed: made up", tie="ranking")
    board.apply_summary("The earlier turns read tier1.", board.replaceable(), 0)
    assert [standing(s, board, walked=Walked(candidates=board.candidates())) for s in steps] == before
    assert [b is None for b in before] == [False, False, False, True]


def test_two_reads_in_one_plan_the_second_stands_once_the_first_took_every_unread_cycle():
    board = _recorded(_synthetic_board(), -2, 0)
    board.status["steps_left"] = 2
    plan = _plan(Step("read", TIER1), Step("read", TIER1), Step("read", {"sensor_group": "tier2", "cycles": 1}))
    assert check_plan(plan, board) == []  # the second read of tier1 stands in the walk and costs no call
    with mock.patch.object(plan_check, "standing", lambda *a, **k: None):
        assert any("steps are left" in r for r in check_plan(plan, board))  # control: one call counted for each read


# ----- clause same_commitment, and what is still refused -----

def _committed(board, rank=1):
    """The board with the diagnosis of its candidate at rank, as the environment echoes it."""
    board.diagnosis = {"diagnosis": dict(submitted_levels(board, board.candidates()[rank - 1])), "evidence_ids": []}
    return board


def _second(board, name):
    stands = ("diagnose", board.diagnosis["diagnosis"]) if board.diagnosis else ("declare", (board.declaration["separating_test"]["sensor_group"],
                                                                                          board.declaration["separating_test"]["cycles"]))
    return f"{name}: {core_texts.SECOND_COMMITMENT.format(commitment=commitment_text(*stands))}"


def test_a_diagnosis_equal_to_the_one_that_stands_stands_and_one_with_other_levels_is_refused():
    board = _synthetic_board()
    walked = Walked(candidates=board.candidates())
    assert standing(Step("diagnose", {"rank": 1}), board, walked=walked) is None  # control: nothing committed
    _committed(board)
    assert standing(Step("diagnose", {"rank": 1}), board, walked=walked) == Standing(
        SAME_COMMITMENT, core_texts.STANDS_COMMITMENT.format(commitment=commitment_text("diagnose", board.diagnosis["diagnosis"])))
    assert check_plan(_plan(Step("diagnose", {"rank": 1}), Step("read", {"sensor_group": "tier1", "cycles": 1})), board) == []
    assert standing(Step("diagnose", {"rank": 2}), board, walked=walked) is None  # other levels
    assert check_plan(_plan(Step("diagnose", {"rank": 2})), board) == [_second(board, "step 1 (diagnose)")]
    assert check_plan(_plan(Step("declare", {"sensor_group": "tier1", "cycles": 1})), board) == [_second(board, "step 1 (declare)")]


def test_a_declaration_equal_to_the_one_that_stands_stands_and_a_diagnosis_is_refused():
    board = _synthetic_board()
    board.declaration = {"separating_test": {"sensor_group": "tier2", "cycles": 1}, "evidence_ids": []}
    assert standing(Step("declare", {"sensor_group": "tier2", "cycles": 1}), board).clause == SAME_COMMITMENT
    assert standing(Step("declare", {"sensor_group": "tier2", "cycles": 2}), board) is None
    assert check_plan(_plan(Step("declare", {"sensor_group": "tier2", "cycles": 1})), board) == []
    assert check_plan(_plan(Step("declare", {"sensor_group": "tier2", "cycles": 2})), board) == [_second(board, "step 1 (declare)")]
    assert check_plan(_plan(Step("diagnose", {"rank": 1})), board) == [_second(board, "step 1 (diagnose)")]


def test_inside_one_plan_a_second_equal_commit_step_stands_and_a_different_one_is_refused():
    board = _synthetic_board()
    board.status["steps_left"] = 2
    assert check_plan(_plan(Step("diagnose", {"rank": 1}), Step("diagnose", {"rank": 1}), Step("finish")), board) == []
    planned = commitment_text("diagnose", submitted_levels(board, board.candidates()[0]))
    assert check_plan(_plan(Step("diagnose", {"rank": 1}), Step("diagnose", {"rank": 2})), board) == [
        f"step 2 (diagnose): {core_texts.SECOND_COMMITMENT.format(commitment=planned)}"]
    with mock.patch.object(plan_check, "standing", lambda *a, **k: None):  # control: every restated commit step counted and refused
        assert len(check_plan(_plan(Step("diagnose", {"rank": 1}), Step("diagnose", {"rank": 1}), Step("finish")), board)) == 2


def test_a_work_order_on_a_component_whose_order_is_open_is_refused_naming_its_close_and_passes_after_it():
    board = _open_order_board()  # the pump's order closes at cycle 6, the board at cycle 1
    order = Step("order", {"component": "pump", "prediction_id": "pred/b1"}, {})
    reason = (f"step 1 (order): {core_texts.OPEN_REPAIR.format(**KIND_WORDS, name='pump')}"
              f"{core_texts.OPEN_REPAIR_UNTIL.format(**KIND_WORDS, close=6)}")
    assert reason in check_plan(_plan(order, mode="act_now"), board)
    board.status["time"] = 7
    board.orders["wo-1"]["status"], board.status["open_work_orders"] = "closed", []
    assert not any("is still open" in r for r in check_plan(_plan(order, mode="act_now"), board))


def test_a_commitment_that_the_walk_took_as_standing_and_that_differs_when_reached_makes_no_call_and_is_a_surprise():
    """Case of a ranking that moved between the plan and the step: the plan's candidates hold rank 1 as it was, the board's
    commitment is another combination, so the step is refused at the call, where the environment would refuse it."""
    board = _committed(_synthetic_board(), rank=2)
    calls = []

    class Sessions:
        async def call(self, server, tool, arguments):
            calls.append(tool)
            return "{}", False
    loop_runner = controller.CausalLoopRunner(llm=ScriptedPlanner(act_now), server_paths={RIG_SERVER: "unused", CAUSAL_SERVER: "unused"})
    loop_runner.board, loop_runner.trajectory, loop_runner._sessions = board, [], Sessions()
    record = episode_result.PlanRecord(index=1, source="model", plan={}, candidates=loop_runner._snapshot())
    outcome = asyncio.run(loop_runner._diagnose(Step("diagnose", {"rank": 1}), record, 1))
    assert calls == [] and outcome.status == "surprise"
    assert outcome.note == core_texts.SECOND_COMMITMENT.format(commitment=commitment_text("diagnose", board.diagnosis["diagnosis"]))
    equal = asyncio.run(loop_runner._diagnose(Step("diagnose", {"rank": 2}, {"top_rank": 1}), record, 1))
    assert calls == [] and equal.status == "standing" and equal.observed["stands"] == SAME_COMMITMENT


# ----- clause state_holds -----

def _state_board(states, steps_left):
    """The synthetic board declaring two writes of one public state, which no surface of this agent declares, with the states and
    the steps left given."""
    board = _synthetic_board()
    board.state_writes = {"lock_out": ("lockout", "on"), "restore": ("lockout", "off")}
    board.states = dict(states)
    board.status["steps_left"] = steps_left
    return board


def test_a_state_write_whose_value_the_state_already_has_stands_and_the_walk_counts_no_call_for_it():
    """Clause state_holds on a board that declares a state write: a write that sets its state to the value the state has stands
    with the reason of the plan page and the walk counts no call for it, and so does a write after an earlier write of the plan
    that set the same value. The control: a write of another value, and a write of a state the board does not know, is a call."""
    lock, restore = Step("lock_out"), Step("restore")
    board = _state_board({"lockout": "on"}, 1)
    assert standing(lock, board) == Standing(STATE_HOLDS, STANDS_STATE.format(state="lockout", value="on"))
    assert step_problems(_plan(lock, lock, Step("finish")), board) == []  # both writes stand: the finish is the one call
    assert standing(restore, board) is None and standing(lock, _state_board({}, 1)) is None
    assert step_problems(_plan(restore, restore, Step("finish")), _state_board({"lockout": "on"}, 2)) == []  # the second holds what the first set
    called = step_problems(_plan(restore, Step("finish")), _state_board({}, 1))
    assert [step for step, _reason in called] == [None] and "needs 2 tool calls" in called[0][1], called
    called = step_problems(_plan(restore, lock, Step("finish")), _state_board({"lockout": "on"}, 2))
    assert [step for step, _reason in called] == [None] and "needs 3 tool calls" in called[0][1], called


# ----- the walk -----

def test_a_plan_that_fits_the_step_cap_only_because_its_standing_steps_cost_no_call_is_accepted():
    plan = _plan(Step("profile"), Step("fix_history"), Step("read", {"sensor_group": "tier1", "cycles": 1}))
    for answered in (True, False):  # control: the reads not answered, one call counted for each step
        board = _answered(_synthetic_board(), ("profile", {}), ("fix_history", {})) if answered else _synthetic_board()
        board.status["steps_left"] = 1
        assert (check_plan(plan, board) == []) == answered


def test_a_standing_read_gives_a_declaration_nothing_to_cite():
    board = _recorded(_synthetic_board(), 0, 0)
    board.evidence, board.absorbed_ids = [], set()  # no reading to cite on the board
    for candidate in board.ranking["candidates"]:
        candidate["evidence_ids"] = []
    board.summaries = [{"sensor_group": "tier1", "cycles": [0], "n_first": 1, "n_last": 1, "t_first": 0, "t_last": 0, "values": 1, "outside": [], "sides": {}}]
    declare = Step("declare", {"sensor_group": "tier1", "cycles": 1})
    reasons = check_plan(_plan(Step("read", TIER1), declare, mode="declare"), board)
    assert any("a declaration counts only when it cites a reading" in r for r in reasons), reasons
    assert check_plan(_plan(Step("read", {"sensor_group": "tier2", "cycles": 1}), declare, mode="declare"), board) == []


def test_a_standing_read_is_not_the_reading_that_verifies_a_repair():
    """On a row whose clock runs a read after a wait does not stand, since the wait records cycles that the board does not hold;
    were one to stand, it would buy no verification, so the finish after it is still refused for the reading that verifies the
    repair (control: the read that is made verifies it and the plan passes)."""
    board = _open_order_board()
    plan = _plan(Step("wait", {"cycles": 5}), Step("read", {"sensor_group": "tier3", "cycles": 1}), Step("finish"), mode="act_now")
    assert check_plan(plan, board) == []
    stood = Standing(NOTHING_NEW, "made up")
    with mock.patch.object(plan_check, "standing", lambda step, *a, **k: stood if step.kind == "read" else None):
        assert any("verifies the repair" in r for r in check_plan(plan, board))


# ----- through the driver: the turn of standing steps alone and the records -----

STOOD_SECTION = re.compile(re.escape(core_texts.STOOD_HEADING) + r"\n((?:- step .*\n)+)" + re.escape(core_texts.STOOD_CLOSING))
PLANS_KEYS = {"mode_of_first_plan", "differential", "plans", "surprises", "revisions", "fallbacks", "model_calls", "end_reason", "error", "server",
              "reader_faults", "replaced_characters", "board"}


def _restating(first):
    """A planner whose first plan after the opening is first and whose later plans act now, keeping every planning prompt."""
    def plans(prompt, call):
        return plan_json(first, mode="test_first") if call == 1 else act_now(prompt, call)
    return ScriptedPlanner(plans)


def _stood_lines(prompt):
    found = STOOD_SECTION.search(prompt)
    return None if found is None else found.group(1).splitlines()


def _no_calls(ep_dir):
    return [e for e in read_json(ep_dir / "hidden" / "log.json") if e.get("event") == "no_call"]


@pytest.mark.parametrize("stood", [True, False], ids=["stands", "control"])
def test_a_turn_of_standing_steps_alone_takes_one_step_and_the_next_request_names_them(tmp_path, stood):
    planner = _restating([{"kind": "profile"}, {"kind": "fix_history"}])
    with contextlib.ExitStack() as stack:
        if not stood:  # control: no step stands, so both reads are called again
            for module in (controller, plan_check):
                stack.enter_context(mock.patch.object(module, "standing", lambda *a, **k: None))
        ep_dir, row, _backend = _play(tmp_path, planner, {})
    plans = read_json(ep_dir / "plans.json")
    restated = next(p for p in plans["plans"] if [s["kind"] for s in p["plan"]["steps"]] == ["profile", "fix_history"])
    tools = [t["tool"] for t in read_json(ep_dir / "trajectory.json")]
    if not stood:
        assert tools.count("get_asset_profile") == 2 and not any(_stood_lines(p) for p in planner.prompts)
        return
    assert tools.count("get_asset_profile") == 1 and tools.count("get_fix_history") == 1
    assert [(s["status"], s["observed"], s["note"]) for s in restated["steps"]] == [
        ("standing", {"stands": EPISODE_ANSWER}, core_texts.STANDS_EPISODE.format(call="profile")),
        ("standing", {"stands": EPISODE_ANSWER}, core_texts.STANDS_EPISODE.format(call="fix_history"))]
    assert restated["source"] == "model" and restated["surprise"] is None and row["fallbacks"] == 0 and len(_no_calls(ep_dir)) == 1
    named = [i for i, p in enumerate(planner.prompts) if _stood_lines(p)]
    assert len(named) == 1 and _stood_lines(planner.prompts[named[0]]) == [
        f"- step 1 (profile): {core_texts.STANDS_EPISODE.format(call='profile')}",
        f"- step 2 (fix_history): {core_texts.STANDS_EPISODE.format(call='fix_history')}"]
    assert named[0] + 1 < len(planner.prompts)  # a request followed it, without the section
    # the records: the status, the clause and the reason in plans.json, its keys, the four keys of the board, and the column
    assert set(plans) == PLANS_KEYS
    assert {"answered", "calls", "retired_notes", "reply_schema"} <= set(plans["board"])
    assert ["profile", {}] in plans["board"]["answered"] and ["fix_history", {}] in plans["board"]["answered"]
    assert plans["board"]["reply_schema"]["carried"] is False and plans["board"]["reply_schema"]["bound"] == 0
    assert row["standing_steps"] == 2


def test_a_cut_plan_whose_steps_before_the_cut_stand_names_both_sections_the_steps_that_did_not_run_first(tmp_path):
    planner = _restating([{"kind": "profile"}, {"kind": "read", "args": {"sensor_group": "tier9", "cycles": 1}, "expect": {}}])
    ep_dir, row, _backend = _play(tmp_path, planner, {})
    plans = read_json(ep_dir / "plans.json")
    cut = next(p for p in plans["plans"] if p["source"] == "model_cut")
    assert [s["status"] for s in cut["steps"]] == ["standing"] and cut["cut"]["step"] == 2 and row["fallbacks"] == 0
    request = next(p for p in planner.prompts if _stood_lines(p))
    assert request.index(core_texts.NOT_RUN_HEADING) < request.index(core_texts.STOOD_HEADING)
    assert len(_no_calls(ep_dir)) == 1


# ----- the option of the reply schema -----

def test_the_option_of_the_reply_schema_names_its_label_and_stands_in_run_json_only_where_a_run_turns_it_off(monkeypatch):
    assert CausalLoopOptions().label() == "causal" and "structured_output" not in CausalLoopOptions().to_json()
    off = CausalLoopOptions(structured_output=False)
    assert off.label() == "causal_noschema" and off.to_json()["structured_output"] is False
    monkeypatch.delitem(OMITTED_AT_DEFAULT, "structured_output")  # control: the name taken out of the options omitted at their default
    assert "structured_output" in CausalLoopOptions().to_json()


def test_a_run_without_the_diagnostics_server_takes_the_option():
    runner = controller.CausalLoopRunner(llm=ScriptedPlanner(act_now), server_paths={RIG_SERVER: "unused"},
                                         options=CausalLoopOptions(structured_output=False))
    assert runner.server is False and runner.options.structured_output is False


# ----- the walk of the viewer -----

def _call(tool, text="{}"):
    return Call(server=RIG_SERVER, tool=tool, args={}, text=text)


DONE = json.dumps({"result": {}})


def _labels(plans, calls, drop_stands=False):
    """The step every call lands on; drop_stands hands the walk the records without observed.stands, as a walk that does not
    read it would see them."""
    if drop_stands:
        plans = json.loads(json.dumps(plans))
        for plan in plans:
            for step in plan["steps"]:
                step.get("observed", {}).pop("stands", None)
    labels, _consistent = controller_walk.causal_walk(calls, plans, RIG_TOOLS, {})
    return labels


def test_the_viewer_places_every_call_on_the_step_that_made_it_and_none_on_a_standing_step():
    opening = [{"kind": "failure_modes", "args": {"component": "cooler"}, "status": "done", "observed": {}},
               {"kind": "failure_modes", "args": {"component": "cooler"}, "status": "standing", "observed": {"stands": EPISODE_ANSWER}},
               {"kind": "failure_modes", "args": {"component": "valve"}, "status": "done", "observed": {}}]
    calls = [_call(RIG_TOOLS["failure_modes"], DONE), _call(RIG_TOOLS["failure_modes"], DONE)]
    assert _labels([{"index": 1, "steps": opening}], calls) == [(1, 1), (1, 3)]
    assert _labels([{"index": 1, "steps": opening}], calls, drop_stands=True)[1] == (1, 2)  # control
    reads = [{"kind": "read", "args": TIER1, "status": "done", "observed": {}},
             {"kind": "read", "args": TIER1, "status": "standing", "observed": {"stands": NOTHING_NEW}},
             {"kind": "read", "args": {"sensor_group": "tier2", "cycles": 1}, "status": "done", "observed": {}}]
    calls = [_call(RIG_TOOLS["read"], DONE), _call(RIG_TOOLS["read"], DONE)]
    assert _labels([{"index": 1, "steps": reads}], calls) == [(1, 1), (1, 3)]
    assert _labels([{"index": 1, "steps": reads}], calls, drop_stands=True)[1] == (1, 2)  # control


# ----- a restated diagnosis by rank after the ranking moved -----

def _restating_diagnosis():
    """A planner on a clocked row that reads as advised and diagnoses rank 1, then restates that diagnosis ahead of its orders and
    ahead of the wait to their close, after which the ranking moves, then ahead of the read that verifies the repair, and once the
    request quotes a refusal, leaves it out."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (LONGEST_REPAIR, ORDER_ROW, ORDERED, RETRY_HEADING,
                                                                                               VERIFYING_READ, advised_read, top_faulty)
    diagnose = {"kind": "diagnose", "args": {"rank": 1}, "expect": {}, "evidence": []}
    wait = {"kind": "wait", "args": {"cycles": LONGEST_REPAIR}, "expect": {"no_new_alarm": True}}

    def plans(prompt, call):
        if "Commitment: none" in prompt:
            return advised_read(prompt) or plan_json([diagnose])
        if not ORDERED.findall(prompt):
            orders = [{"kind": "order", "args": {"component": c, "prediction_id": "estimate"}, "expect": {"prediction_id": "estimate"}}
                      for c in top_faulty(prompt)]
            return plan_json([diagnose, *orders])
        if any(row[4] != "closed" for row in ORDER_ROW.findall(prompt)):
            return plan_json([diagnose, wait])
        verify = [dict(VERIFYING_READ), {"kind": "finish"}]
        return plan_json(verify if RETRY_HEADING in prompt else [diagnose, *verify])
    return ScriptedPlanner(plans)


@pytest.mark.parametrize("compared", [True, False], ids=["compared", "control"])
def test_a_restated_diagnosis_stands_while_its_rank_names_the_commitment_and_is_refused_once_the_ranking_moved(tmp_path, compared):
    with contextlib.ExitStack() as stack:
        if not compared:  # control: the levels not compared, so every restated commit step stands
            stack.enter_context(mock.patch("CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.standing.same_commitment",
                                           lambda made, stands, board: stands is not None))
        ep_dir, row, _backend = _play(tmp_path, _restating_diagnosis(), {}, promote_after_order=True)
    plans = read_json(ep_dir / "plans.json")["plans"]
    tools = [t["tool"] for t in read_json(ep_dir / "trajectory.json")]
    refused = [p for p in plans if any("since another commitment already stands" in r for reasons in p["check_problems"] for r in reasons)]
    assert tools.count("submit_diagnosis") == 1
    ordering = next(p for p in plans if any(s["kind"] == "order" for s in p["plan"]["steps"]))
    assert ordering["steps"][0]["status"] == "standing" and ordering["steps"][0]["observed"]["stands"] == SAME_COMMITMENT
    assert tools.count("create_work_order") >= 1
    if not compared:
        assert not refused
        return
    assert len(refused) == 1 and refused[0]["source"] == "no_plan_after_check_failure"
    after = plans[plans.index(refused[0]) + 1]
    assert [s["kind"] for s in after["plan"]["steps"]] == ["read", "finish"] and row["end_reason"] == "finish"


# ----- episodes whose plans restate a read, the extent or a diagnosis, through the driver's command on AssetOpsBench's servers -----

PRICES = re.compile(r"^Read prices per cycle: ([^;\n]*)", re.MULTILINE)
LEVELS = re.compile(r"Condition levels: ([^\n]*)")


def _last_group(prompt):
    return [part.strip().rsplit(" ", 1)[0] for part in PRICES.search(prompt).group(1).split(",")][-1]


def _own_candidate(prompt):
    """One candidate of the model's own, the first listed level of every component, and its faulty components."""
    levels, faulty = {}, []
    for part in LEVELS.search(prompt).group(1).rstrip(".").split(";"):
        name, rest = part.split(":", 1) if ":" in part else part.split(" ", 1)
        first, healthy = int(rest.split(",")[0].split()[-1]), int(re.search(r"healthy (\d+)", rest).group(1))
        levels[name.strip()] = first
        if first != healthy:
            faulty.append(name.strip())
    return {"levels": levels, "failure_mode_ids": []}, faulty


def _acts(prompt, server):
    """Diagnose rank 1, one order per faulty component of it and finish, with the planner's own candidate without the server."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ESTIMATE, top_faulty
    candidate, faulty = (None, top_faulty(prompt)) if server else _own_candidate(prompt)
    orders = [{"kind": "order", "args": {"component": c, "prediction_id": ESTIMATE}, "expect": {"prediction_id": ESTIMATE}} for c in faulty]
    return [{"kind": "diagnose", "args": {"rank": 1}}, *orders, {"kind": "finish"}], candidate


def _read(prompt, expect=None):
    return {"kind": "read", "args": {"sensor_group": _last_group(prompt), "cycles": 5}, "expect": expect or {}}


class EpisodePlanner(ScriptedPlanner):
    """A scripted planner whose count of planning calls starts again with each episode."""

    def generate(self, prompt, temperature=0.0):
        if "No profile read yet" in prompt and "Your plan of turn" not in prompt:
            self.plan_calls = 0
        return super().generate(prompt, temperature)


def _shape(name):
    """The planner of a shape: reread (with the server) reads the last group, then every plan reads it again ahead of the
    commitment, the repairs and the end; restate and restate_first (without the server) read and diagnose, then every plan
    restates the extent, the read and the diagnosis, or the diagnosis alone, ahead of the repairs and the end."""
    server = name == "reread"

    def plans(prompt, call):
        acts, candidate = _acts(prompt, server)
        first = [_read(prompt)] + ([] if server else acts[:1])
        steps = (first if call == 1 else [_read(prompt), *acts] if name == "reread"
                 else [{"kind": "extent"}, _read(prompt), *acts] if name == "restate" else acts)
        plan = json.loads(plan_json(steps))
        if candidate is not None:
            plan["candidates"] = [candidate]
        return json.dumps(plan)
    return EpisodePlanner(plans), server


def _shape_run(tmp_path, monkeypatch, planner, server):
    """The driver's command on the base preset with AssetOpsBench's servers and the memory store, on the development starts that
    the covering start cap keeps (_covering_cap); returns the episode folders."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import _planner_endpoint
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    endpoint, _requests = _planner_endpoint(planner)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    try:
        run_episodes.main(["--config", "base", "--servers", "aob", "--store", "memory", "--agent", "causal", "--model", "vendor/stand-in-1.0",
                           "--base-url", f"http://127.0.0.1:{endpoint.server_address[1]}/v1/", "--seeds", "1",
                           "--starts", "development", "--start-cap", str(_covering_cap()),
                           "--out", str(tmp_path / "run"), *([] if server else ["--no-diagnostics"])])
    finally:
        endpoint.shutdown()
    run_dir = next(p for p in (tmp_path / "run").iterdir() if p.is_dir())
    return sorted(run_dir.glob("seed*"))


def _ends_by_its_own_last_step(ep_dir):
    """The episode ended by the agent's finish, with a commitment, one order per faulty component of its last plan, at least one
    standing step and no surprise from one."""
    plans = read_json(ep_dir / "plans.json")
    tools = [t["tool"] for t in read_json(ep_dir / "trajectory.json")]
    steps = [s for p in plans["plans"] for s in p["steps"]]
    last = next((p for p in reversed(plans["plans"]) if p["plan"]["steps"] and p["plan"]["steps"][-1]["kind"] == "finish"), None)
    if last is None:
        return False
    ordered = sum(1 for s in last["plan"]["steps"] if s["kind"] == "order")
    return (plans["end_reason"] == "finish" and plans["board"]["diagnosis"] is not None and tools.count("generate_work_order") == ordered
            and any(s["status"] == "standing" for s in steps)
            and not any(s["status"] == "surprise" and (s.get("observed") or {}).get("stands") for s in steps))


# the clauses that each shape's restated steps stand by, over its episodes: the restated read of a group read whole, the restated
# extent where no cycle passes, and the restated diagnosis
SHAPE_CLAUSES = {"reread": {NOTHING_NEW}, "restate": {SURFACE_ANSWER, SAME_COMMITMENT}, "restate_first": {SAME_COMMITMENT}}


def _clauses(ep_dir):
    return {s["observed"]["stands"] for p in read_json(ep_dir / "plans.json")["plans"] for s in p["steps"] if s["status"] == "standing"}


@pytest.mark.parametrize("stood", [True, False], ids=["stands", "control"])
@pytest.mark.parametrize("shape", ["reread", "restate", "restate_first"])
def test_episodes_whose_plans_restate_standing_steps_end_by_the_agents_own_last_step(tmp_path, monkeypatch, shape, stood):
    planner, server = _shape(shape)
    with contextlib.ExitStack() as stack:
        if not stood:  # control: no step stands, and every episode ends at the step cap
            for module in (controller, plan_check):
                stack.enter_context(mock.patch.object(module, "standing", lambda *a, **k: None))
        episodes = _shape_run(tmp_path, monkeypatch, planner, server)
    assert episodes and all(_ends_by_its_own_last_step(ep) for ep in episodes) == stood
    if stood:
        assert SHAPE_CLAUSES[shape] <= set().union(*map(_clauses, episodes)), {ep.name: _clauses(ep) for ep in episodes}


def _reread_expect(leave_out):
    """With the server: the first plan reads the last group; every later plan restates that read with an expectation that the board
    as it stands fails, ahead of the commitment, the repairs and the end, and with leave_out leaves the read out once a request shows
    the surprise of a standing step."""
    surprised = re.escape(core_texts.STANDS_EXPECTATION.split("{stands}", 1)[1].split("{detail}", 1)[0])

    def plans(prompt, call):
        acts, _candidate = _acts(prompt, True)
        if call == 1:
            return plan_json([_read(prompt)])
        if leave_out and re.search(surprised, prompt):
            return plan_json(acts)
        return plan_json([_read(prompt, {"effective_at_most": 0.5}), *acts])
    return EpisodePlanner(plans)


@pytest.mark.parametrize("judged", [True, False], ids=["judged", "control"])
def test_an_expectation_on_a_standing_read_that_the_board_fails_stops_the_plan_until_the_planner_leaves_the_read_out(tmp_path, monkeypatch, judged):
    with contextlib.ExitStack() as stack:
        if not judged:  # control: the expectation of a standing step not judged, so the commitment runs in the first such turn
            stack.enter_context(mock.patch.object(controller.CausalLoopRunner, "_stand_read", lambda self, step, record, stands: _awaitable(self._stand(step, stands))))
        episodes = _shape_run(tmp_path, monkeypatch, _reread_expect(leave_out=True), True)
    assert len(episodes) == _covering_cap()
    for ep_dir in episodes:
        plans = read_json(ep_dir / "plans.json")
        stopped = [p for p in plans["plans"] if p["surprise"] and p["surprise"]["observed"].get("stands") == NOTHING_NEW]
        if not judged:
            assert not stopped, ep_dir.name
            continue
        assert len(stopped) == 1, ep_dir.name
        reason = stopped[0]["surprise"]["reason"]
        assert reason.startswith(core_texts.STANDS_NOTHING_NEW.format(**KIND_WORDS, name=_group_of(stopped[0]))) and "its expectation" in reason
        assert len(stopped[0]["steps"]) == 1  # the plan stopped at the read: no commitment and no repair ran in that turn
        assert plans["end_reason"] == "finish" and plans["board"]["diagnosis"] is not None, ep_dir.name


def test_a_planner_that_never_leaves_the_failed_standing_read_out_ends_at_the_step_cap_without_a_write(tmp_path, monkeypatch):
    """The fixed point of a standing step whose expectation the board fails: every turn stops on it, and nothing is written."""
    episodes = _shape_run(tmp_path, monkeypatch, _reread_expect(leave_out=False), True)
    assert len(episodes) == _covering_cap()
    for ep_dir in episodes:
        plans = read_json(ep_dir / "plans.json")
        tools = [t["tool"] for t in read_json(ep_dir / "trajectory.json")]
        assert plans["end_reason"] != "finish" and "submit_diagnosis" not in tools and "generate_work_order" not in tools, ep_dir.name


def _group_of(plan):
    return plan["plan"]["steps"][0]["args"]["sensor_group"]


async def _coroutine(value):
    return value


def _awaitable(value):
    return _coroutine(value)


def test_the_column_of_standing_steps_is_empty_for_every_agent_but_the_causal_one(tmp_path):
    """scores.csv counts the standing steps of the CausalDiagnosis agent and leaves the cell empty for any other agent, here a
    reference policy of the environment (the causal row's count: test_a_turn_of_standing_steps_alone_takes_one_step_and_the_next_request_names_them)."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, read_rows
    run_episodes.main(["--config", "static", "--agent", "oracle", "--seeds", "1", "--out", str(tmp_path / "run")])
    rows = read_rows(next(p for p in (tmp_path / "run").iterdir() if p.is_dir()))
    assert "standing_steps" in COLUMNS and rows and all(row["standing_steps"] == "" for row in rows)
