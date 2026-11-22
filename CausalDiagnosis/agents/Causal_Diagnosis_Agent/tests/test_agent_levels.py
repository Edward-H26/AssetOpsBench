"""The agent follows the tools the row's level serves, as the opening listing names them. A step kind
is offered only where the row serves its tool (tools.OPTIONAL_KIND_TOOLS), and the change summary is a counted call where the row serves
summarize_changes_since. The repair tail on a row that serves no wait follows the one rule the agent and the CausalDiagnosis MCP share."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import _synthetic_board  # noqa: E402

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import Plan, Step, check_plan, rule_plan  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import orientation_turn, planning_turn, summary_turn, system_prompt  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import OPTIONAL_KIND_TOOLS, RigSurface  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents.causal_diagnosis_agent_surface import AobSurface  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import world  # noqa: E402,F401  the recorded cycles, a fixture
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import completion_rules  # noqa: E402

# the task tools of the level without the CausalDiagnosis MCP (the answer tools are on every row at both levels) and of the level with them
WITHOUT_OUR_TOOLS = ("get_asset_profile", "submit_diagnosis", "declare_insufficient_evidence", "finish")
WITH_OUR_TOOLS = (*WITHOUT_OUR_TOOLS, "summarize_changes_since", "get_current_time", "wait_for_notification")
# what would offer a kind on a row that does not serve it: its step line, its tool, and the sentences that promise it
PROMISES = {
    "declare": ("- declare:", "declare_insufficient_evidence", "declar", '"declare"'),
    "wait": ("- wait:", "wait_for_notification", "waits to the close", "a wait lets", ", order and wait"),
    "time": ("get_current_time", "time, no args"),
}
CLOCKED = ("real_time", "advanced_real_time", "repair_rules", "deterioration")


def _surface(tools):
    surface = AobSurface()
    surface.listed({"task": [SimpleNamespace(name=name) for name in tools], "iot": [SimpleNamespace(name="history")]})
    return surface


def _profiled(surface):
    """The surface after the profile of a clocked row with a read duration, so that its texts name the clock and the opening."""
    surface.site, surface.clock, surface.read_per_cycle = "MAIN", True, 1
    return surface


def _texts(surface, server):
    board = _synthetic_board()
    board.unserved, board.server = surface.unserved_kinds(), server
    return (system_prompt(6, surface=surface, server=server), planning_turn(board, 6, surface=surface).split("## Planning turn")[1],
            summary_turn(board, 6, surface=surface).split("## Closing summary")[1])


def test_the_table_names_every_kind_a_level_may_leave_out_and_the_listing_decides_it():
    assert OPTIONAL_KIND_TOOLS == {"declare": "declare_insufficient_evidence", "wait": "wait_for_notification", "time": "get_current_time"}
    assert _surface(WITHOUT_OUR_TOOLS).unserved_kinds() == {"wait", "time"}
    assert _surface([t for t in WITHOUT_OUR_TOOLS if t != "declare_insufficient_evidence"]).unserved_kinds() == {"declare", "wait", "time"}  # a stand-in listing
    assert _surface(WITH_OUR_TOOLS).unserved_kinds() == frozenset()
    assert _surface([t for t in WITH_OUR_TOOLS if t not in ("get_current_time", "wait_for_notification")]).unserved_kinds() == {"wait", "time"}
    assert AobSurface().unserved_kinds() == frozenset() and RigSurface().unserved_kinds() == frozenset()  # before a listing every tool is served


@pytest.mark.parametrize("kind", sorted(OPTIONAL_KIND_TOOLS))
@pytest.mark.parametrize("server", (False, True))
def test_a_kind_the_row_does_not_serve_is_named_nowhere_and_refused(kind, server):
    tools = [t for t in WITH_OUR_TOOLS if t != OPTIONAL_KIND_TOOLS[kind]]
    surface = _profiled(_surface(tools))
    assert surface.unserved_kinds() == {kind}
    for text in _texts(surface, server):
        for promise in PROMISES[kind]:
            assert promise not in text, (kind, promise)
    board = _synthetic_board()
    board.unserved = surface.unserved_kinds()
    args = {"declare": {"sensor_group": "tier2", "cycles": 1}, "wait": {"cycles": 2}, "time": {}}[kind]
    assert f"step 1 ({kind}): leave this step out, since this row serves no {kind} step" in check_plan(
        Plan(branch=1, mode="test_first", why="", steps=[Step(kind, args)]), board)


@pytest.mark.parametrize("server", (False, True))
def test_the_texts_of_a_row_that_serves_every_kind_are_those_of_today(server):
    for listed, today in ((_surface(WITH_OUR_TOOLS), AobSurface()), (_profiled(_surface(WITH_OUR_TOOLS)), _profiled(AobSurface()))):
        assert _texts(listed, server) == _texts(today, server)
    board = _synthetic_board()
    assert orientation_turn("q", board.profile, [], 6, surface=_surface(WITH_OUR_TOOLS), server=server) == orientation_turn(
        "q", board.profile, [], 6, surface=AobSurface(), server=server)


def test_the_mode_declare_is_refused_where_no_declaration_is_served():
    board = _synthetic_board()
    plan = Plan(branch=1, mode="declare", why="", steps=[Step("declare", {"sensor_group": "tier2", "cycles": 1})])
    assert check_plan(plan, board) == []
    board.unserved = frozenset({"declare"})
    assert "mode declare: choose a mode among test_first, act_now and repair_as_test, since this row serves no declare step" in check_plan(plan, board)


def test_the_rule_writes_no_kind_the_row_does_not_serve():
    board = _synthetic_board()
    board.status["budget_left"] = 40  # no separating read fits, so the rule declares where it may
    assert rule_plan(board, 6).mode == "declare"
    board.unserved = frozenset({"declare", "wait", "time"})
    plan = rule_plan(board, 6)
    assert plan.mode != "declare" and not {"declare", "wait", "time"} & {s.kind for s in plan.steps}
    board = _synthetic_board()
    board.ranking = None
    assert "declare" not in [s.kind for s in rule_plan(board, 6).steps]


def test_the_change_summary_is_a_counted_call_where_served_and_a_local_answer_elsewhere():
    served, without = _surface(WITH_OUR_TOOLS), _surface(WITHOUT_OUR_TOOLS)
    assert served.summary_is_call and served.bind("changes", None, since=3) == ("task", "summarize_changes_since", {"cycle": 3})
    assert served.prompt_changes == RigSurface.prompt_changes
    assert not without.summary_is_call and without.bind("changes", None, since=3) is None and without.prompt_changes == ""
    reply = '{"since": 3, "time": 5, "cycles_elapsed": 2, "alarms": [], "work_orders_closed": []}\n{"budget_left": 10, "steps_left": 7, "cycle": 5}'
    data = served.parse("changes", "summarize_changes_since", reply, None, since=3)
    assert data["result"]["since"] == 3 and data["steps_left"] == 7 and data["time"] == 5 and served.calls == 1


# ----- the repair tail -----
def _row_board(world, preset, ours):  # noqa: F811
    """The board of an AssetOpsBench row of the preset at a level, as the runner sets it after the opening listing and the profile."""
    config = preset_config(preset, diagnostics_server=ours)
    engine = RecordedRigEnv(*world, config)
    engine.reset(7)
    surface = AobSurface()
    surface.listed({s: [SimpleNamespace(name=n) for n in layer_contract.tools_for(s, config)]
                    for s in (*layer_contract.AOB_SERVERS, layer_contract.TASK_SERVER)})
    board = Board(reads_take_time=False, server=ours, summary_call=surface.summary_is_call)
    board.unserved = surface.unserved_kinds()
    board.set_profile(engine.public_profile())
    board.clock_runs = real_time.has_clock(config.clock_mode, config.durations)
    board.status = {"time": 10, "budget_left": 10 ** 6, "steps_left": 100, "recorded_until": 9, "recorded_from": 0}
    return board


@pytest.mark.parametrize("preset", CLOCKED)
@pytest.mark.parametrize("ours", (False, True))
def test_the_tail_reads_the_verifying_group_forward_where_no_wait_is_served(world, preset, ours):  # noqa: F811
    board = _row_board(world, preset, ours)
    board.diagnosis = {"diagnosis": dict(board.profile["healthy"]), "evidence_ids": []}
    tail = board.completion(["pump"], commit=False)
    group = board.verification_group(board.repaired(["pump"]))
    assert board.verifies(group, ["pump"])
    order, finish = Step("order", {"component": "pump", "prediction_id": "estimate"}), Step("finish")

    def problems(*steps):
        return check_plan(Plan(branch=1, mode="act_now", why="", steps=[order, *steps, finish]), board)
    if ours:  # the wait is served: wait to the close, read one cycle, finish, as before
        assert "wait" not in board.unserved and board.forward_read == 0 and tail.reads == 1
        assert problems(Step("wait", {"cycles": tail.first - board.time}), Step("read", {"sensor_group": group, "cycles": 1})) == []
    elif preset == "real_time":  # no wait and no read that plays cycles: finish without a read is allowed
        assert board.forward_read == 0 and tail.reads == 0 and tail.first is None
        assert problems() == []
    else:  # no wait, and a window past the present plays its cycles: one forward read over the cycles to the close, then finish
        played = tail.to_close + 1
        assert board.forward_read > 0 and tail.reads == 1 and tail.cycles == played * board.forward_read
        assert tail.read_price == played * board.read_price(group)
        assert problems(Step("read", {"sensor_group": group, "cycles": played})) == []
        assert any(f"read {group} over the cycles to the close" in p for p in problems())


@pytest.mark.parametrize("preset", ("base", *CLOCKED))
def test_the_agent_and_the_server_finish_a_repair_by_the_same_rule(world, preset):  # noqa: F811
    board = _row_board(world, preset, True)
    assert completion_rules(profile_view(rules=board.profile)) == board.rules()


# ----- the TaskAgent of the ports -----
def _task_tools(names):
    schema = SimpleNamespace(model_fields={"alpha": None, "beta": None, "evidence_ids": None})
    return [SimpleNamespace(name=n, args_schema=schema) for n in names]


@pytest.mark.parametrize("names, count", (
    (("get_asset_profile", "submit_diagnosis", "end_episode"), 3),  # a stand-in listing without the declaration, which keeps the rule covered
    (("get_asset_profile", "submit_diagnosis", "declare_insufficient_evidence", "end_episode"), 4),  # the level without the CausalDiagnosis MCP, and with them on a row without a clock
    (("get_asset_profile", "get_current_time", "wait_for_notification", "lookup_fix", "submit_diagnosis", "declare_insufficient_evidence",
      "end_episode"), 7),  # with the CausalDiagnosis MCP on the repair rules row
))
def test_the_task_agent_shows_and_describes_only_the_tools_it_holds(names, count):
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import in_context_examples as examples, sub_agent_registry
    spec = next(s for s in sub_agent_registry.SURFACES["rig"] if s.name == "TaskAgent")
    tools = _task_tools(names)
    shown = spec.own_examples(tools, {"alpha": 4, "beta": 7})
    assert shown.count("Question:") == count
    description = spec.description_for(tools)
    for kind, promises in PROMISES.items():
        if OPTIONAL_KIND_TOOLS[kind] not in names:
            assert not any(p in shown for p in promises if p.startswith(("declare_", "wait_", "get_current"))), kind
    assert ("declaration" in description) == ("declare_insufficient_evidence" in names)
    assert ("clock" in description) == ("get_current_time" in names)
    if count == 7:
        assert description == sub_agent_registry.OURS["TaskAgent"][0] and shown.count("Question:") == len(examples.TASK_TOOLS)


def test_no_listing_serves_every_tool_and_an_empty_listing_stops_the_episode():
    surface = AobSurface()
    surface.listed(None)  # no listing given, as the in-process stand-in sessions of the tests
    assert surface.served is None and surface.unserved_kinds() == frozenset()
    for empty in ({}, {"task": [], "iot": []}):
        with pytest.raises(ValueError, match="the opening listing of /runs/real_time/seed000003 names no tool"):
            AobSurface().listed(empty, where="/runs/real_time/seed000003")
