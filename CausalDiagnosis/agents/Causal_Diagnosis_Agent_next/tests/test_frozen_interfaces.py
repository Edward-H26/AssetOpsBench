"""The freeze of the interfaces (CausalDiagnosis/docs/general-agent-tasks.md, step 0): the four fixture declarations build and join
with their listings, the defaults that need the listing hold where the plan page says, every name of the plan page that the freeze
writes imports with the parameters that it freezes, and the files written complete keep their fields and rules."""
import dataclasses
import importlib
import inspect
import json
from types import SimpleNamespace

import pytest

from CausalDiagnosis.agents.utils.answer import answer
from CausalDiagnosis.agents.utils.declaration import DeclarationError
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next import agent_options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.plan_schema import Parsed, Plan, Refusal, Step, Unusable
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests import fixture_declarations as fixtures

ROLES = ("observe", "knowledge", "act", "task", "advice")
WRITES = ("repair", "probe", "state", "answer")
NEXT = "CausalDiagnosis.agents.Causal_Diagnosis_Agent_next"
# the rule functions, one per row of the table of the check of the plan page, in its order; a call with nothing new is a clause of
# standing and no rule
RULES = ("known_tool", "arguments", "facts_first", "order", "single_commitment", "end_last", "cites_observation", "candidate_rank",
         "candidate_values", "recall_rows", "expectation", "open_repair", "precondition", "end_guard", "budget", "time_limit",
         "step_limit")
# the twelve switches of the plan page, in the order of the fields of Options
SWITCH_NAMES = ("branches", "single_hypothesis", "plan_length", "expectations", "reads", "backtracking", "expectation_tolerance",
                "parallel", "recall", "summary_turn", "structured_output", "binding")
# every name that the plan page gives a file of the freeze, with the parameters that the freeze fixes for it; a class by the
# parameters of its constructor, a data class by its fields in order
FROZEN = {
    "CausalDiagnosis.agents.utils.declaration": {
        "ToolRow": ("name", "server", "role", "description", "schema", "write", "commits", "ends", "sets", "target", "words", "facts",
                    "orients", "each", "withheld", "minimum", "after", "values", "fills", "candidate", "omitted", "waits", "answer",
                    "series", "requires_observation", "advice", "call"),
        "Vocabulary": ("noun", "short_noun", "example_field", "identifiers", "terms"),
        "Declaration": ("asset_id", "kind", "surface", "vocabulary", "rows", "listed_rows", "server_roles", "cards", "advice", "events",
                        "rules", "binding"),
        "Declaration.table": ("self", "listing"), "Declaration.listed": ("self", "listing"), "Declaration.to_json": ("self",),
        "Table": ("declaration", "tools"), "Table.of_role": ("self", "role"), "Table.of_write": ("self", "write"),
        "Table.commit_tools": ("self",), "Table.end_tool": ("self",), "Table.facts_tool": ("self",), "Table.advice_kinds": ("self",),
        "DeclarationError": None,
    },
    "CausalDiagnosis.agents.utils.answer": {"answer": ("result", "refused", "status", "notifications", "observations")},
    "CausalDiagnosis.agents.utils.answer_forms": {"observations": ("form", "result", "ranges")},
    "CausalDiagnosis.utils.public_rules.plan_rules": {
        "KindRules.ledger": ("self", "facts", "flags", "cards", "task"),
        "Ledger.record": ("self", "row", "args", "answer"), "Ledger.status": ("self",), "Ledger.limits": ("self",),
        "Ledger.walk": ("self",), "Ledger.open_targets": ("self",), "Ledger.paid": ("self", "series"), "Ledger.pending": ("self",),
        "Ledger.verifies": ("self", "row", "args", "targets"), "Ledger.verdict": ("self", "pending", "observations"),
        "Ledger.targets": ("self", "candidate"), "Ledger.verifying_call": ("self", "targets"), "Ledger.ranges": ("self",),
        "Ledger.sections": ("self", "vocabulary"),
        "Walk.step": ("self", "row", "args"), "Walk.end": ("self",), "Walk.needs": ("self", "repairs"), "Walk.totals": ("self",),
        "StepCost": ("price", "time", "calls", "needs_facts", "reasons", "stands"), "Needs": ("price", "time", "calls"),
    },
    f"{NEXT}.plan.plan_schema": {
        "Step": ("tool", "args", "expect", "evidence", "candidate"), "Plan": ("follow", "why", "steps", "candidates", "raw"),
        "Unusable": ("step", "reasons", "kinds"), "Refusal": ("reason", "fault"), "Parsed": ("plan", "unusable", "refusal"),
        "Standing": ("clause", "reason"),
        "PlanError": None, "OWN_STEPS": None,
    },
    f"{NEXT}.plan.reply_parser": {
        "last_json_object": ("text",), "parse_usable_plan": ("text", "table", "forms", "plan_length"),
        "parse_candidates": ("raw", "table"), "parse_differential": ("text",),
    },
    f"{NEXT}.plan.readers": {
        "read_name": ("value", "names", "name"), "read_count": ("value", "minimum", "name"),
        "read_number": ("value", "low", "high", "name"), "read_text": ("value", "name"), "read_list": ("value", "name"),
        "read_object": ("value", "name"),
    },
    f"{NEXT}.plan.plan_check": {
        "step_problems": ("plan", "board", "ledger", "table", "modules"), "check_plan": ("plan", "board", "ledger", "table", "modules"),
        "WalkState": ("table", "board", "walk", "earlier", "plan"),
    },
    f"{NEXT}.plan.rules": {name: ("step", "index", "state") for name in (*RULES, "standing")},
    f"{NEXT}.plan.reply_schema": {"plan_schema_of": ("table", "forms", "options"), "differential_schema": (),
                                  "response_format": ("name", "schema"), "digest": ("schema",)},
    f"{NEXT}.schedule": {"schedule": ("board", "ledger", "plans", "named", "reserve", "refuted"),
                         "Schedule": ("reads", "cut", "left_out", "remaining", "differs")},
    f"{NEXT}.working_memory.board": {
        "Board": ("table", "ledger", "question", "options"),
        "Board.record_call": ("self", "step", "call", "answer", "observations"), "Board.candidates": ("self",),
        "Board.known_ids": ("self",), "Board.render": ("self", "modules"), "Board.to_json": ("self", "modules"),
        "Board.compact": ("self", "fits", "after_call", "summarize", "fits_context"), "Board.mark_turn": ("self", "turn"),
        "Board.replaceable": ("self",), "Board.without": ("self", "entries"), "Board.apply_summary": ("self", "text", "entries", "left_out"),
        "Board.recall": ("self", "series", "first", "last"), "Board.recallable": ("self", "series", "first", "last"),
        "Board.drop_unfit_recall": ("self", "fits_context"), "Board.holds_answer": ("self", "tool", "args"),
        "Board.repeated": ("self",), "Board.add_note": ("self", "text", "tie", "turn"), "Board.retire_notes": ("self", "passed"),
        "Board.reply_schema": None,
    },
    f"{NEXT}.prompt.core_protocol": {},
    f"{NEXT}.prompt.assembler": {
        "system_prompt": ("table", "board", "options", "modules"),
        "orientation_turn": ("board", "table", "options", "modules", "retry"),
        "planning_turn": ("board", "table", "options", "modules", "retry", "not_run", "stood"),
        "earlier_turns_turn": ("board", "table", "options", "modules", "left_out"),
        "summary_turn": ("board", "table", "options", "modules", "note"),
        "identifier_forms": ("table", "board", "modules"),
    },
    f"{NEXT}.advice": {
        "modules_for": ("table", "options"), "expectation_forms": ("table", "modules"),
        "Module.blocks": ("self",), "Module.forms": ("self", "row"), "Module.step_problems": ("self", "step", "index", "state"),
        "Module.candidates": ("self",), "Module.sections": ("self", "board"), "Module.known_ids": ("self",),
        "Module.evidence_for": ("self", "step", "row"), "Module.before_turn": ("self", "ctx"),
        "Module.before_step": ("self", "step", "row", "ctx"), "Module.after_batch": ("self", "steps", "answers", "ctx"),
        "Module.on_pending": ("self", "pending", "observation", "ctx"), "Module.to_json": ("self",),
        "ModuleContext.ask": ("self", "kind", "inputs"),
    },
    f"{NEXT}.advice.ranking": {"RankingModule": ("tools", "table", "options")},
    f"{NEXT}.advice.warrants": {"WarrantModule": ("tools", "table", "options")},
    f"{NEXT}.agent_options": {"Options": SWITCH_NAMES, "Options.label": ("self",),
                              "Options.to_json": ("self",), "SWITCHES": None},
    f"{NEXT}.episode_result": {
        "ToolRecord": ("step", "server", "tool", "tool_args", "response", "error", "duration_ms", "guard"),
        "StepRecord": ("tool", "args", "expect", "status", "observed", "note"),
        "PlanRecord": ("index", "source", "plan", "candidates", "after_surprise", "steps", "surprise", "parse_errors", "check_problems",
                       "schedule", "cut", "guard"),
        "EpisodeResult": ("question", "answer", "plans", "trajectory", "model_calls", "surprises", "revisions", "fallbacks",
                          "mode_of_first_plan", "differential", "board", "end_reason", "error", "server", "advice", "end"),
    },
    f"{NEXT}.controller": {"AgentRunner": ("llm", "server_paths", "declaration", "options"), "AgentRunner.run": ("self", "question"),
                           "request_settings": ("llm", "temperature"), "BindingView.paid": ("self", "series")},
}


def _parameters(thing):
    if dataclasses.is_dataclass(thing):
        return tuple(f.name for f in dataclasses.fields(thing))
    return tuple(inspect.signature(thing).parameters)


@pytest.mark.parametrize("module", sorted(FROZEN))
def test_every_name_of_the_plan_page_that_the_freeze_writes_imports_with_its_parameters(module):
    """A changed parameter of a frozen function, a field of a frozen data class added, dropped or moved, or a name that is missing
    fails here: the signatures change only through the owner of step 0."""
    loaded = importlib.import_module(module)
    for dotted, expected in FROZEN[module].items():
        thing = loaded
        for part in dotted.split("."):
            thing = getattr(thing, part)
        if expected is not None:
            assert _parameters(thing) == expected, dotted


def _tables(name):
    declaration, listing = fixtures.declarations()[name]
    return declaration, listing, declaration.table(listing), declaration.listed(listing)


@pytest.mark.parametrize("name", ("rig", "aob", "aob_real_time", "port", "single"))
def test_the_fixture_declarations_build_and_join_with_their_listings(name):
    """Each fixture builds and both joins answer: the agent's table in listing order with the description and the schema that the
    listing or the binding gives each tool, the listed table with every listed tool under its listed name, and every lookup of a
    table on both; the declaration is recorded as JSON."""
    declaration, listing, table, listed = _tables(name)
    listed_names = [tool.name for tools in listing.values() for tool in tools]
    assert list(listed.tools) == listed_names
    carried = {carrier: tool for tool, carrier in (getattr(declaration.binding, "carried", None) or {}).items()}
    served = [carried.get((server, tool.name), tool.name) for server, tools in listing.items() for tool in tools]
    assert list(table.tools) == [tool for tool in served if tool in table.tools]
    assert all(row.description and row.schema for row in table.tools.values()), name
    for joined in (table, listed):
        assert set(sum((joined.of_role(role) for role in ROLES), ())) == set(joined.tools)
        assert all(joined.tools[tool].write == write for write in WRITES for tool in joined.of_write(write))
        assert joined.facts_tool() is None or joined.tools[joined.facts_tool()].facts
    json.dumps(declaration.to_json())
    expected = {"rig": (14, 16), "aob": (14, 31), "aob_real_time": (16, 33), "port": (9, 9), "single": (1, 1)}[name]
    assert (len(table.tools), len(listed.tools)) == expected


def test_a_listed_tool_without_a_row_raises_by_name_in_both_joins():
    """The control of the join: a tool that the rig's action server lists and that no row, no carried tool and no role of its
    server covers stops both joins with its name."""
    declaration, listing = fixtures.declarations()["rig"]
    stray = {**listing, "hydraulic_rig": [*listing["hydraulic_rig"], SimpleNamespace(name="xray", description="", inputSchema={})]}
    for join in (declaration.table, declaration.listed):
        with pytest.raises(DeclarationError, match="xray"):
            join(stray)


def test_a_carried_tool_stands_in_the_table_where_its_carrier_is_listed():
    declaration, listing, table, _listed = _tables("aob")
    row = table.tools["read_sensors"]
    assert (row.server, row.answer, row.omitted) == ("iot", "records", {"cycles": "fixed:1"})
    assert row.description == declaration.rows["read_sensors"].description != ""
    without = {**listing, "iot": [tool for tool in listing["iot"] if tool.name != "history"]}
    assert "read_sensors" not in declaration.table(without).tools and "stream_extent" in declaration.table(without).tools


def test_a_withheld_tool_stays_out_of_the_agents_table_and_in_the_listed_one():
    declaration, listing, table, listed = _tables("rig")
    withheld = [name for name, row in declaration.rows.items() if row.withheld]
    assert withheld == ["get_current_time", "lookup_fix"]
    assert not set(withheld) & set(table.tools) and set(withheld) <= set(listed.tools)
    shown = dataclasses.replace(declaration, rows={n: dataclasses.replace(r, withheld=False) for n, r in declaration.rows.items()})
    assert set(withheld) <= set(shown.table(listing).tools)
    _d, _l, aob_table, _a = _tables("aob_real_time")
    assert "get_current_time" in aob_table.tools  # a read needs the clock on the rows of AssetOpsBench's servers


def test_the_tools_of_the_role_advice_stand_in_the_table():
    expected = {"rig": ("evidence", "ranking", "plan_assessment", "outcome_check"),
                "aob": ("evidence", "ranking", "plan_assessment", "outcome_check"), "port": ("evidence", "plan_assessment"),
                "single": ()}
    for name, kinds in expected.items():
        _declaration, _listing, table, listed = _tables(name)
        assert table.advice_kinds() == listed.advice_kinds() == kinds, name
        assert len(table.of_role("advice")) == len(kinds), name


def test_the_lookups_name_the_commitment_the_end_and_the_facts_of_each_fixture():
    expected = {"rig": (("submit_diagnosis", "declare_insufficient_evidence"), "finish", "get_asset_profile"),
                "aob": (("submit_diagnosis", "declare_insufficient_evidence"), "finish", "get_asset_profile"),
                "port": (("file_report",), "file_report", None), "single": (("answer",), "answer", None)}
    for name, (commits, end, facts) in expected.items():
        _declaration, _listing, table, _listed = _tables(name)
        assert (table.commit_tools(), table.end_tool(), table.facts_tool()) == (commits, end, facts), name


def test_the_rig_gives_its_declaration_no_candidate_arguments_and_the_ports_answer_takes_its_required_ones():
    """The default of candidate holds only where no row of the contract states the key: both declarations of the rig state it on
    submit_diagnosis, so declare_insufficient_evidence holds none, while the answer write of the made-up port and of the single
    tool takes its required arguments that no fill writes. The default applied whatever the contract states would give the
    declaration of the rig its sensor group and its count."""
    for name in ("rig", "aob", "aob_real_time"):
        _declaration, _listing, table, listed = _tables(name)
        components = fixtures.rig_rows()["submit_diagnosis"].candidate
        for joined in (table, listed):
            assert joined.tools["declare_insufficient_evidence"].candidate == (), name
            assert joined.tools["submit_diagnosis"].candidate == components, name
        unconditional = tuple(a for a in table.tools["declare_insufficient_evidence"].schema["required"]
                              if a not in table.tools["declare_insufficient_evidence"].fills)
        assert unconditional == ("sensor_group", "cycles")
    assert _tables("port")[2].tools["file_report"].candidate == _tables("port")[3].tools["file_report"].candidate == ("cause",)
    assert _tables("single")[2].tools["answer"].candidate == ("text",)
    assert fixtures.port_declaration().rows["file_report"].candidate == ()  # the declaration holds no default that needs the listing


def test_the_rig_holds_omitted_on_the_rows_that_the_plan_page_gives_it():
    """Both declarations of the rig: a count left out reads as 1, a component that a candidate leaves out stands at its healthy
    level. The base row of AssetOpsBench's servers lists no wait, so its binding states no row of it, and the row real_time does."""
    expected = {"read_sensors", "wait_for_notification", "submit_diagnosis", "declare_insufficient_evidence"}
    assert {n for n, r in fixtures.rig_declaration().rows.items() if r.omitted} == expected
    assert {n for n, r in fixtures.aob_declaration("real_time").rows.items() if r.omitted} == expected
    assert {n for n, r in fixtures.aob_declaration("base").rows.items() if r.omitted} == expected - {"wait_for_notification"}


def test_two_servers_that_list_one_tool_name_raise_by_name():
    declaration = fixtures.port_declaration()
    listing = fixtures.port_listing()
    second = {**listing, "other": [SimpleNamespace(name="read_gauge", description="", inputSchema={})]}
    second_roles = dataclasses.replace(declaration, server_roles={"other": "observe"})
    with pytest.raises(DeclarationError, match="read_gauge.*bottler_1.*other"):
        second_roles.listed(second)
    assert "read_gauge" in second_roles.listed(listing).tools  # the control: without the second server the table builds


def test_the_table_names_every_kind_a_level_may_leave_out_and_the_listing_decides_it():
    """Re-expressed through the declaration: a row whose tool the listing of the row does not serve is not in the table. On the
    rows of AssetOpsBench's servers the task tools of the table are those that the task server lists, with the clock and the wait
    where the row real_time lists them, and a stand-in listing without the declaration leaves the declaration out; on the rig's own
    action server every listed tool that is not withheld stands in the table."""
    task = ("get_asset_profile", "get_current_time", "wait_for_notification", "summarize_changes_since", "submit_diagnosis",
            "declare_insufficient_evidence", "finish")
    base, base_listing, base_table, _l = _tables("aob")
    real, real_listing, real_table, _r = _tables("aob_real_time")
    assert tuple(t for t in task if t in base_table.tools) == tuple(t.name for t in base_listing["task"])
    assert tuple(t for t in task if t in real_table.tools) == tuple(t.name for t in real_listing["task"])
    assert {"get_current_time", "wait_for_notification"} <= set(real_table.tools) - set(base_table.tools)
    stand_in = {**base_listing, "task": [t for t in base_listing["task"] if t.name != "declare_insufficient_evidence"]}
    assert "declare_insufficient_evidence" not in base.table(stand_in).tools
    no_wait = {**real_listing, "task": [t for t in real_listing["task"] if t.name != "wait_for_notification"]}
    assert "wait_for_notification" not in real.table(no_wait).tools
    rig, rig_listing, rig_table, _g = _tables("rig")
    assert set(rig_table.tools) == {t.name for tools in rig_listing.values() for t in tools} - {"get_current_time", "lookup_fix"}


def test_a_listed_tool_reads_as_an_mcp_tool_or_as_a_mapping():
    declaration, listing = fixtures.declarations()["port"]
    as_mappings = {server: [dict(vars(tool)) for tool in tools] for server, tools in listing.items()}
    assert declaration.table(as_mappings) == declaration.table(listing)
    assert declaration.listed(as_mappings) == declaration.listed(listing)


def test_the_json_of_a_declaration_writes_the_keys_of_a_row_that_differ_from_their_default():
    data = fixtures.rig_declaration().to_json()
    assert data["rows"]["get_current_time"] == {"name": "get_current_time", "server": "hydraulic_rig", "role": "task", "withheld": True}
    assert data["rows"]["create_work_order"]["after"] == [["submit_diagnosis", "declare_insufficient_evidence"]]
    assert (data["surface"], data["binding"], data["events"], data["cards"][0]) == (None, False, list(fixtures.RIG_EVENTS), "fix_history")
    assert data["advice"] == {"get_episode_evidence": "evidence", "rank_root_causes": "ranking",
                              "estimate_intervention_effect": "plan_assessment", "verify_outcome": "outcome_check"}
    assert fixtures.aob_declaration().to_json()["binding"] is True


def test_the_rules_module_holds_one_function_per_row_of_the_check_and_the_standing_clauses():
    """One rule function per row of the table of the check, and standing beside them; a call with nothing new is a clause of
    standing, so no rule function stands for it."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan import rules

    functions = {name for name, value in vars(rules).items() if inspect.isfunction(value) and not name.startswith("_")}
    assert functions == {*RULES, "standing"} and len(RULES) == 17


def test_the_binding_stub_needs_the_facts_tool_before_the_tools_it_writes_from_the_facts():
    """The stub of the binding of the rows of AssetOpsBench's servers holds the facts tool among the needs of the tools whose
    arguments it writes from the facts, the extent of the record and, where the row lists it, the clock before a read, and nothing
    before get_failure_modes, which takes the component of the step."""
    for row, clock in (("base", ()), ("real_time", ("get_current_time",))):
        binding = fixtures.aob_declaration(row).binding
        for tool in ("stream_extent", "read_sensors", "create_work_order"):
            assert "get_asset_profile" in binding.needs(tool), (row, tool)
        assert binding.needs("read_sensors") == ("get_asset_profile", *clock, "stream_extent")
        assert binding.needs("get_failure_modes") == () and binding.needs("get_asset_profile") == ()


def test_a_standing_step_has_its_clause_and_reason_and_a_step_cost_states_none_by_default():
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.plan_schema import Standing
    from CausalDiagnosis.utils.public_rules.plan_rules import StepCost

    standing = Standing("same_commitment", "leave this step out")
    with pytest.raises(dataclasses.FrozenInstanceError):
        standing.clause = "nothing_new"
    assert StepCost(price=0.0, time=0, calls=0, needs_facts=False, reasons=()).stands is None
    assert StepCost(0.0, 0, 0, False, (), "state_holds").stands == "state_holds"


def test_a_parsed_reply_holds_a_plan_or_a_refusal_and_never_both():
    plan = Plan(follow=None, why="", steps=[Step("finish", {}, {}, [])], candidates=[], raw="{}")
    assert Parsed(plan, None, None).plan is plan and Parsed(plan, Unusable(1, ("r",), ("finish",)), None).unusable.step == 1
    assert Parsed(None, None, Refusal("the reply could not be read", "ValueError")).refusal.fault == "ValueError"
    for both_or_none in ((plan, None, Refusal("r")), (None, None, None)):
        with pytest.raises(ValueError):
            Parsed(*both_or_none)


def test_a_step_and_a_plan_record_follow_and_tool_in_place_of_branch_and_kind():
    step = Step("submit_diagnosis", {}, {}, ["obs/1/x"], candidate=2)
    assert step.to_json() == {"tool": "submit_diagnosis", "args": {}, "expect": {}, "evidence": ["obs/1/x"], "candidate": 2}
    assert Step("finish", {}, {}, []).to_json() == {"tool": "finish", "args": {}, "expect": {}, "evidence": []}
    plan = Plan(follow=1, why="w", steps=[step], candidates=[{"part": "NZ4", "cites": []}], raw="{}")
    assert plan.to_json() == {"follow": 1, "why": "w", "steps": [step.to_json()], "candidates": [{"part": "NZ4", "cites": []}]}


def test_an_answer_holds_a_result_or_a_refusal_with_its_status_notifications_and_observations():
    assert answer() == {"result": None, "status": {}, "notifications": [], "observations": []}
    refused = answer(result={"x": 1}, refused="the answer of read_gauge could not be read", status={"steps_left": 3})
    assert list(refused) == ["refused", "status", "notifications", "observations"] and "result" not in refused
    assert answer(result=[1], notifications=({"event": "alarm"},))["notifications"] == [{"event": "alarm"}]


def test_the_options_hold_one_field_per_switch_and_label_and_record_as_the_present_options_do():
    """The twelve switches of the plan page, recall and summary_turn among them with no flag; the label and the run.json of a run
    that sets no option are those of the present options' form, and a moved option is recorded and labeled."""
    options = agent_options.Options()
    assert all(getattr(options, s.name) == s.default for s in agent_options.SWITCHES)
    assert {s.owner for s in agent_options.SWITCHES} == {"core", "ranking"}
    assert options.label() == "causal_next"
    moved = agent_options.Options(branches=False, single_hypothesis=True, plan_length=4, expectations=False, reads="separating",
                                  backtracking=False, expectation_tolerance=0.1, parallel=False, recall=False, summary_turn=False,
                                  structured_output=False, binding=False)
    assert moved.label() == "causal_next_nobranch_single_len4_noexp_e10_noback_seq_norecall_nosumturn_noschema_raw"
    assert agent_options.Options(reads="ec2").label() == "causal_next_ec2"
    assert list(options.to_json()) == ["branches", "single_hypothesis", "plan_length", "expectations", "reads", "backtracking",
                                       "expectation_tolerance", "parallel"]
    assert {"recall", "summary_turn", "structured_output", "binding"} <= set(moved.to_json())
    assert agent_options.Options(structured_output=False).to_json()["structured_output"] is False
    assert [s.name for s in agent_options.SWITCHES] == list(SWITCH_NAMES)


def test_every_flag_of_a_switch_but_the_new_one_is_a_flag_of_the_driver_today():
    """The registry names the flags that set each switch; every one is an option of the driver's command line today, but
    --causal-raw-calls, which the driver of the new agent adds."""
    from CausalDiagnosis.experiments.episode_runners.run_episodes import command_line

    known = set(command_line.build_parser()._option_string_actions)
    flags = [flag for switch in agent_options.SWITCHES for flag in switch.flags]
    assert len(flags) == len(set(flags)) == 11 and [flag for flag in flags if flag not in known] == ["--causal-raw-calls"]
