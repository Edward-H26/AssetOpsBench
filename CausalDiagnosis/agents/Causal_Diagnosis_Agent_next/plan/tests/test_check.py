"""A copied public ledger checks prefixes, defaults and every standing clause."""

import copy
import json
from dataclasses import replace

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.working_memory.board import Board
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests.fixture_declarations import declarations
from ..plan_schema import Step, Plan
from ..plan_check import inspect_plan, check_plan
from ..reply_parser import parse_usable_plan
from .fakes import Ledger
from .test_reply_schema import values


def setup(name="rig", **limits):
    declaration, listing = declarations()[name]
    table = declaration.table(listing)
    ledger = Ledger(**limits)
    return table, ledger, Board(table, ledger, "Inspect the machine.", Options())


def plan(*steps, candidates=(), follow=None):
    return Plan(follow, "Inspect the evidence.", list(steps), list(candidates), "{}")


def step(tool, **args):
    return Step(tool, args, {}, [])


def record(board, item, value=None):
    board.record_call(item, ("server", item.tool, item.args),
                      {"result": value if value is not None else {}, "status": {}, "notifications": []}, [])


@pytest.mark.parametrize("name", ("rig", "aob", "aob_real_time", "port"))
def test_an_answer_of_the_episode_stands_only_for_its_exact_arguments(name):
    table, ledger, board = setup(name, steps_left=0)
    row = next((row for row in table.tools.values() if row.facts or row.role == "knowledge"), None)
    if row is None:
        row = table.tools["read_gauge"]
        table = replace(table, tools={**table.tools, row.name: replace(row, role="knowledge")})
        board.table = table
    item = step(row.name, **values(row.schema)[0])
    record(board, item)
    result = inspect_plan(plan(item), board, ledger, table, ())
    assert not result.problems and result.steps[0].stood.clause == "episode_answer"
    board.calls.clear()
    result = inspect_plan(plan(item), board, ledger, table, ())
    assert result.problems and result.steps[0].stood is None


def test_surface_answers_stand_only_without_a_clock():
    table, ledger, board = setup("aob")
    item = step("stream_extent")
    record(board, item)
    result = inspect_plan(plan(item), board, ledger, table, ())
    assert result.steps[0].stood.clause == "surface_answer"
    ledger.state["time"] = 1
    assert inspect_plan(plan(item), board, ledger, table, ()).steps[0].stood is None


def test_nothing_new_and_state_holds_follow_the_kind_walk_and_their_controls():
    table, ledger, board = setup()
    board._facts = {"sensor_groups": ["all"]}
    item = step("read_sensors", sensor_group="all", cycles=1)
    ledger.stands[item.tool] = "nothing_new"
    result = inspect_plan(plan(item), board, ledger, table, ())
    assert result.steps[0].stood.clause == "nothing_new" and not result.problems
    ledger.stands.clear()
    assert inspect_plan(plan(item), board, ledger, table, ()).steps[0].stood is None
    table, ledger, board = setup("port")
    item = step("lock_out")
    ledger.stands[item.tool] = "state_holds"
    assert inspect_plan(plan(item), board, ledger, table, ()).steps[0].stood.clause == "state_holds"
    ledger.stands.clear()
    assert inspect_plan(plan(item), board, ledger, table, ()).steps[0].stood is None


def test_a_commitment_stands_and_a_changed_commitment_is_refused_with_the_change():
    table, ledger, board = setup("single")
    # The ending write never stands, even when it also commits.
    row = replace(table.tools["answer"], ends=False)
    table = replace(table, tools={row.name: row})
    board.table = table
    record(board, step("answer", text="ready"))
    result = inspect_plan(plan(step("answer", text="ready")), board, ledger, table, ())
    assert result.steps[0].stood.clause == "same_commitment"
    problems = check_plan(plan(step("answer", text="changed")), board, ledger, table, ())
    assert "leave this step out" in problems[0] and "another commitment" in problems[0]
    table = replace(table, tools={row.name: replace(row, ends=True)})
    assert inspect_plan(plan(step("answer", text="ready")), board, ledger, table, ()).steps[0].stood is None


def test_standing_steps_do_not_spend_the_call_limit_and_checking_changes_no_live_state():
    table, ledger, board = setup(steps_left=1)
    item = step("get_asset_profile")
    record(board, item)
    before = copy.deepcopy(ledger.__dict__)
    result = inspect_plan(plan(item, step("get_fix_history")), board, ledger, table, ())
    assert not result.problems and result.steps[0].stood
    assert ledger.__dict__ == before
    board.calls.clear()
    assert any(index is None and "counted calls" in reason
               for index, reason in inspect_plan(plan(item, step("get_fix_history")), board, ledger, table, ()).problems)


def test_a_later_unusable_step_cuts_the_walk_and_whole_plan_limits_run_nothing():
    table, ledger, board = setup()
    result = inspect_plan(plan(step("get_fix_history"), step("unknown"), step("get_asset_profile")), board, ledger, table, ())
    assert len(result.executable()) == 1 and result.problems[0][0] == 2
    ledger.state["budget_left"] = 0
    result = inspect_plan(plan(step("get_fix_history")), board, ledger, table, ())
    assert result.problems[0][0] is None and result.executable() == []


def test_omitted_candidate_values_resolve_before_check_and_non_candidate_commits_require_arguments():
    table, ledger, board = setup()
    row = table.tools["submit_diagnosis"]
    board._facts = {"healthy": {key: 1 for key in row.candidate}, "levels": {key: [1, 2] for key in row.candidate}}
    raw = {"why": "Inspect.", "candidates": [{}], "steps": [{"tool": row.name}]}
    parsed = parse_usable_plan(json.dumps(raw), table, {}, None)
    result = inspect_plan(parsed.plan, board, ledger, table, ())
    assert not result.problems and result.steps[0].step.args == {key: 1 for key in row.candidate}
    plain = replace(row, candidate=(), omitted={})
    table = replace(table, tools={**table.tools, row.name: plain})
    refusal = parse_usable_plan(json.dumps(raw | {"candidates": []}), table, {}, None).refusal
    assert refusal and "required" in refusal.reason


def test_an_open_repair_is_refused_without_hiding_it_as_standing():
    table, ledger, board = setup("port")
    ledger.open["NZ4"] = 5
    item = step("swap_part", part="NZ4")
    result = inspect_plan(plan(item), board, ledger, table, ())
    assert result.steps[0].stood is None
    assert any("leave this step out" in reason and "still open" in reason for _, reason in result.problems)
    ledger.open.clear()
    assert not inspect_plan(plan(item), board, ledger, table, ()).problems


def test_facts_dependent_actions_wait_for_an_answer_but_knowledge_can_follow_the_opening():
    table, ledger, board = setup()
    read = step("read_sensors", sensor_group="all", cycles=1)
    result = inspect_plan(plan(step("get_asset_profile"), read), board, ledger, table, ())
    assert result.problems[0][0] == 2 and "once get_asset_profile is answered" in result.problems[0][1]
    knowledge = step("get_failure_modes", component="supply")
    assert not inspect_plan(plan(step("get_asset_profile"), knowledge), board, ledger, table, ()).problems
    board._facts = {"sensor_groups": ["all"]}
    assert not inspect_plan(plan(read), board, ledger, table, ()).problems
    read.args["sensor_group"] = "unknown"
    assert any("must be one of" in reason for _, reason in inspect_plan(plan(read), board, ledger, table, ()).problems)


def test_step_order_and_end_last_are_checked_at_their_own_step():
    table, ledger, board = setup()
    row = table.tools["create_work_order"]
    table = replace(table, tools={**table.tools, row.name: replace(row, values={})})
    repair = step(row.name, component="supply")
    result = inspect_plan(plan(repair), board, ledger, table, ())
    assert result.problems[0][0] == 1 and "put submit_diagnosis or declare_insufficient_evidence before" in result.problems[0][1]
    table = replace(table, tools={**table.tools, row.name: replace(table.tools[row.name], after=())})
    assert not inspect_plan(plan(repair), board, ledger, table, ()).problems
    result = inspect_plan(plan(step("finish"), step("get_fix_history")), board, ledger, table, ())
    assert result.problems[0][0] == 1 and "last step" in result.problems[0][1]
    assert not inspect_plan(plan(step("finish")), board, ledger, table, ()).problems


def test_candidate_rank_scope_and_candidate_values_have_whole_plan_controls():
    table, ledger, board = setup("single")
    valid = step("answer", text="ready")
    result = inspect_plan(plan(valid, follow=1), board, ledger, table, ())
    assert result.problems[0][0] is None
    assert not inspect_plan(plan(valid, candidates=[{"text": "ready"}], follow=1), board, ledger, table, ()).problems
    table = replace(table, tools={"answer": replace(table.tools["answer"], ends=False)})
    board.table = table
    ranked = Step("answer", {}, {}, [], candidate=2)
    result = inspect_plan(plan(valid, ranked, candidates=[{"text": "ready"}]), board, ledger, table, ())
    assert result.problems[0][0] == 2
    table, ledger, board = setup()
    row = table.tools["submit_diagnosis"]
    board._facts = {"levels": {key: [1, 2] for key in row.candidate}}
    candidates = [{key: 99 for key in row.candidate}]
    result = inspect_plan(plan(step("get_fix_history"), candidates=candidates), board, ledger, table, ())
    assert result.problems and all(index is None for index, _ in result.problems)
    candidates = [{key: 1 for key in row.candidate}]
    assert not inspect_plan(plan(step("get_fix_history"), candidates=candidates), board, ledger, table, ()).problems


def test_evidence_requirements_recall_and_expectations_have_negative_controls():
    table, ledger, board = setup()
    board._facts = {"sensor_groups": ["all"]}
    declare = step("declare_insufficient_evidence", sensor_group="all", cycles=1)
    # Use the row's listed arguments rather than inventing a declaration shape.
    declare.args.update(values(table.tools[declare.tool].schema)[0])
    declare.args.update(sensor_group="all", cycles=1)
    declare.args.pop("evidence_ids", None)
    result = inspect_plan(plan(declare), board, ledger, table, ())
    assert any("none is on the board" in reason for _, reason in result.problems)
    board.observations.append({"id": "obs/1/x", "n": 1})
    assert not inspect_plan(plan(declare), board, ledger, table, ()).problems
    recall = step("recall", observations=[1, 1])
    assert inspect_plan(plan(recall), board, ledger, table, ()).problems
    board.collapsed_rows.append({"id": "obs/1/x", "n": 1})
    assert not inspect_plan(plan(recall), board, ledger, table, ()).problems
    read = step("read_sensors", sensor_group="all", cycles=1)
    read.expect = {"unknown": True}
    assert any("no expectation" in reason for _, reason in inspect_plan(plan(read), board, ledger, table, ()).problems)
    read.expect = {"outside": ["pressure"]}
    assert not inspect_plan(plan(read), board, ledger, table, ()).problems


def test_preconditions_and_ending_guards_come_from_the_kind_and_limits_are_plan_reasons():
    table, ledger, board = setup("port")
    lock = step("lock_out")
    ledger.reasons[lock.tool] = ("set the supply first",)
    result = inspect_plan(plan(lock), board, ledger, table, ())
    assert result.problems == [(1, "set the supply first")]
    ledger.reasons.clear()
    assert not inspect_plan(plan(lock), board, ledger, table, ()).problems
    table, ledger, board = setup("single")
    item = step("answer", text="ready")
    ledger.end_reasons = ("verify the repair first",)
    assert inspect_plan(plan(item), board, ledger, table, ()).problems == [(1, "verify the repair first")]
    ledger.end_reasons = ()
    assert not inspect_plan(plan(item), board, ledger, table, ()).problems
    ledger.state.update(time=0, time_limit=1)
    ledger.durations[item.tool] = 1
    result = inspect_plan(plan(item), board, ledger, table, ())
    assert result.problems[0][0] is None and "before answer can run" in result.problems[0][1]
    ledger.durations[item.tool] = 0
    assert not inspect_plan(plan(item), board, ledger, table, ()).problems
