"""The tests of the ranking module (CausalDiagnosis/docs/general-agent-tasks.md, Task 8): the tests of the map under ranking module,
each on the rig's table and on the tiny asset of CausalDiagnosis/datasets/tests/, re-expressed through the interfaces of the plan
page, the block of the rig as the plan page writes it, the switches of the module, the forms and their judgment, the rules of the
module, and the stand-in model of the rule. The board, the ledger and the loop are read through the doubles of
ranking_doubles.py. A test of the map that needs an episode, the schedule or the check waits, marked skip with the task it needs.

Each test names its control in its docstring: the part of the module that it holds, switched off or changed, makes it fail."""
import asyncio
import json
import re
from pathlib import Path

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import SWITCHES, Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.advice import ranking as ranking_module
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.advice.tests import ranking_doubles as d
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.plan_schema import Step

PLAN_PAGE = d.ROOT / "CausalDiagnosis" / "docs" / "general-agent-plan.md"
TABLES = ("rig", "tiny")


def run(coroutine):
    return asyncio.run(coroutine)


def setup(name, options=None):
    """(module, table, facts, the first two targets, the first two series) of a table."""
    table, facts = d.tables()[name]
    m = d.module(table, options)
    targets = list(facts["healthy"])
    return m, table, facts, targets, list(facts["sensor_groups"])


def ranked(m, facts, *args, **kwargs):
    m.ranking = d.ranking(m, facts, *args, **kwargs)
    m.planned = m.ranking
    return m


def observe(m, expect, ranked_before=None):
    return Step(m.observe_tool, {m.series_arg: "g", m.count_arg: 1}, expect, [])


# ----- the block -----

def test_the_block_of_the_rig_is_the_text_of_the_plan_page():
    """For the rig the block reads as the section Advice modules of the plan page writes it. Control: a word of a sentence of the
    block changed, or the tool name typed into it."""
    page = PLAN_PAGE.read_text(encoding="utf-8").split("## Advice modules", 1)[1]
    expected = re.search(r"For the rig it reads:\n\n```text\n(.*?)\n```", page, re.DOTALL).group(1)
    m, *_ = setup("rig")
    assert m.blocks()[0] == expected


@pytest.mark.parametrize("name", TABLES)
def test_the_block_names_the_tool_of_the_kind_and_the_words_of_the_vocabulary(name):
    """The block names the ranking tool of the table and the terms of its vocabulary, and no tool or term of another table.
    Control: the tool name of the rig typed into a sentence of the block."""
    m, table, *_ = setup(name)
    text = m.blocks()[0]
    words = table.declaration.vocabulary.terms
    assert f"asks {m.tool} and shows" in text and f"A step of {m.commit_tool} can expect" in text
    assert f"faulty {words['target']}s" in text and f"Per {words['series']}" in text
    other = "tiny" if name == "rig" else "rig"
    other_module = setup(other)[0]
    assert other_module.tool not in text and other_module.commit_tool not in text
    assert m.blocks()[1][0] == "board"


@pytest.mark.parametrize("name", TABLES)
def test_the_batch_rule_the_declaration_and_the_switches(name):
    """The module owns the switches branches, single_hypothesis and reads, as the registry of switches records them; branches off
    leaves out the sentence of the candidate plans, a value of reads other than gain replaces the sentence of the advised reads,
    and parallel off leaves out the sentence of the schedule, every other sentence staying. Control: a switch that changes another
    sentence."""
    assert set(ranking_module.RankingModule.switches) == {s.name for s in SWITCHES if s.owner == "ranking"}
    on = setup(name)[0].blocks()[0]
    words = setup(name)[0]._words()
    plans = ranking_module.fill(ranking_module.BLOCK[ranking_module.BRANCHES_SENTENCE], **words)
    schedule = ranking_module.BLOCK[ranking_module.SCHEDULE_SENTENCE]
    assert setup(name, Options(branches=False))[0].blocks()[0] == on.replace(" " + plans, "")
    assert setup(name, Options(parallel=False))[0].blocks()[0] == on.replace(" " + schedule, "")
    arm = setup(name, Options(reads="ec2"))[0].blocks()[0]
    assert arm == on.replace(ranking_module.BLOCK[ranking_module.READS_SENTENCE], ranking_module.fill(ranking_module.READS_ARM, arm=ranking_module.READ_ARMS["ec2"]))


# ----- the forms and their judgment -----

@pytest.mark.parametrize("name", TABLES)
def test_the_forms_are_those_of_an_observe_step_and_of_a_commit_write_with_candidate_arguments(name):
    """An observe step can state top_rank, top_mass_at_least and effective_at_most, a commit write with candidate arguments
    top_rank, any other row none; each form reads its value. Control: the forms of an observe step given to every row."""
    m, table, *_ = setup(name)
    forms = {tool: m.forms(row) for tool, row in table.tools.items()}
    assert set(forms[m.observe_tool]) == {"top_rank", "top_mass_at_least", "effective_at_most"}
    assert set(forms[m.commit_tool]) == {"top_rank"}
    assert all(not f for tool, f in forms.items() if tool not in (m.observe_tool, m.commit_tool))
    observe_forms = forms[m.observe_tool]
    assert observe_forms["top_rank"](2) == (2, None) and observe_forms["top_rank"](0)[0] is None and observe_forms["top_rank"](True)[0] is None
    assert observe_forms["top_mass_at_least"](0.5) == (0.5, None) and observe_forms["top_mass_at_least"](1.5)[0] is None
    assert observe_forms["effective_at_most"](float("nan"))[0] is None and isinstance(observe_forms["effective_at_most"], str)


@pytest.mark.parametrize("name", TABLES)
def test_a_read_is_judged_on_the_faulty_set_on_top_and_its_mass_summed_over_the_set(name):
    """An observe step that expects top_rank r and top_mass_at_least x holds when the faulty set of the candidate at rank r before
    the batch is the set on top after it with at least x, the mass summed over every combination of the set. Control: the mass of
    the top candidate alone in place of the mass of the set."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    before = d.ranking(m, facts, [((a,), 0.5), ((b,), 0.3), ((a, b), 0.2)])
    m.ranking = d.ranking(m, facts, [((b,), 0.35), ((a,), 0.3), ((a, b), 0.05)])
    m.ranking["set_masses"] = [{"faulty_set": [b], "mass": 0.6}, {"faulty_set": [a], "mass": 0.3}]
    assert m.judge(observe(m, {"top_rank": 2, "top_mass_at_least": 0.55}), before)["holds"]
    assert not m.judge(observe(m, {"top_rank": 1}), before)["holds"]
    assert not m.judge(observe(m, {"top_rank": 2, "top_mass_at_least": 0.9}), before)["holds"]
    m.ranking["effective_hypotheses"] = 2.0
    assert m.judge(observe(m, {"effective_at_most": 2.0}), before)["holds"] and not m.judge(observe(m, {"effective_at_most": 1.5}), before)["holds"]
    assert m.judge(observe(m, {}), before) is None


@pytest.mark.parametrize("name", TABLES)
def test_a_diagnosis_below_rank_one_without_an_expectation_expects_the_candidate_on_top(name):
    """A commit write by a ranked candidate that states no expectation expects the candidate on top when the plan was made to stay
    on top: its judgment fails when an observation before it moved another set up. Control: the default of top_rank left out."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)])
    commit = Step(m.commit_tool, {}, {}, [], candidate=2)
    assert m.judge(commit, m.ranking)["holds"]
    m.ranking = d.ranking(m, facts, [((b,), 0.7), ((a,), 0.3)])
    assert not m.judge(commit, m.planned)["holds"]
    assert m.judge(Step(m.commit_tool, {}, {}, [], candidate=None), m.planned) is None


# ----- the rules of the module -----

@pytest.mark.parametrize("name", TABLES)
def test_the_rules_refuse_a_top_rank_outside_the_ranked_candidates_and_a_ranked_commitment_before_any_ranking(name):
    """A top_rank outside the ranked candidates is refused on an observe step, and a commit write that expects top_rank before any
    ranking is refused; a step within the ranks passes. Control: the rule of the rank switched off."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    commit = Step(m.commit_tool, {}, {"top_rank": 1}, [], candidate=1)
    assert m.step_problems(commit, 0, None) == [ranking_module.fill(ranking_module.REASON_NO_RANKING, tool=m.commit_tool)]
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)])
    assert m.step_problems(commit, 0, None) == []
    assert m.step_problems(observe(m, {"top_rank": 3}), 0, None) == [ranking_module.fill(ranking_module.REASON_RANK, n=2, rank=3)]
    assert m.step_problems(observe(m, {"top_rank": 2}), 0, None) == []


@pytest.mark.parametrize("name", TABLES)
def test_a_diagnosis_while_the_ranking_says_read_first_runs_with_the_advice_on_the_board(name):
    """While the ranking says read first, a commit write gets no reason of the module, and the board shows the advised reads.
    Control: a rule that refuses a commitment while the ranking says read first."""
    m, _table, facts, (a, b, *_), groups = setup(name)
    ranked(m, facts, [((a,), 0.5), ((b,), 0.5)], read_first=True, advice={"action": "read", m.series_arg: groups[0], m.count_arg: 2, "text": "read it"})
    assert m.step_problems(Step(m.commit_tool, {}, {"top_rank": 1}, [], candidate=1), 0, None) == []
    lines = [line for _place, _heading, rows in m.sections(d.Board()) for line in rows]
    assert ranking_module.fill(ranking_module.LINE_SERVER_READ, reads=f"{groups[0]} x2") in lines


# ----- the loop -----

@pytest.mark.parametrize("name", TABLES)
def test_the_controller_ranks_before_the_first_planning_turn_and_the_model_reads_the_advised_head(name):
    """Before a turn the module asks the ranking once the facts are on the board and whenever the observations, the open repairs
    or the clock moved since it, and never twice on one state; its candidates take the first ranks. Control: a ranking asked
    before every turn."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    board = d.Board(facts=None)
    ctx = d.Context(board, [d.ranking(m, facts, [((a,), 0.6), ((b,), 0.4)])] * 3)
    run(m.before_turn(ctx))
    assert ctx.asked == []
    board.facts = facts
    run(m.before_turn(ctx))
    run(m.before_turn(ctx))
    assert len(ctx.asked) == 1 and [c["rank"] for c in m.candidates()] == [1, 2]
    board.observations.append({"id": "obs/1"})
    run(m.before_turn(ctx))
    board.ledger.time = 4
    run(m.before_turn(ctx))
    assert len(ctx.asked) == 3


@pytest.mark.parametrize("name", TABLES)
def test_consecutive_read_steps_run_with_one_ranking_after_the_last(name):
    """After a batch of observe steps that made calls the module asks the ranking once and judges every step against the ranking
    before the batch; a standing step, with None as its answer, is judged on the ranking that the board holds, asked again only
    when it is stale. Control: a ranking asked after every step, or for a standing step on a current ranking."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    board = d.Board(facts=facts)
    ctx = d.Context(board, [d.ranking(m, facts, [((a,), 0.6), ((b,), 0.4)]), d.ranking(m, facts, [((b,), 0.8), ((a,), 0.2)])])
    run(m.before_turn(ctx))
    board.observations += [{"id": "obs/1"}, {"id": "obs/2"}]
    judged = run(m.after_batch([observe(m, {"top_rank": 2}), observe(m, {"top_rank": 1})], [{"result": 1}, {"result": 2}], ctx))
    assert len(ctx.asked) == 2 and [j["holds"] for j in judged] == [True, False]
    standing = run(m.after_batch([observe(m, {"top_rank": 1})], [None], ctx))
    assert len(ctx.asked) == 2 and standing[0]["holds"]


@pytest.mark.parametrize("name", TABLES)
def test_the_board_shows_the_advice_of_the_server_and_the_rank_call_asks_for_its_form(name):
    """The ranking call asks the compact form under the default of reads and the long form under an arm, and the board shows the
    advice of the server. Control: the form of the call fixed."""
    for reads, form in (("gain", "compact"), ("ec2", "summary"), ("separating", "summary")):
        m, _table, facts, (a, *_), _groups = setup(name, Options(reads=reads))
        ctx = d.Context(d.Board(facts=facts), [d.ranking(m, facts, [((a,), 1.0)], advice={"action": "diagnose", "rank": 1, "text": "act on rank 1"})])
        run(m.before_turn(ctx))
        assert ctx.asked == [("ranking", {"detail": form})]
        lines = [line for _p, _h, rows in m.sections(ctx.board) for line in rows]
        assert "Advice of the server: act on rank 1" in lines


@pytest.mark.parametrize("name", TABLES)
def test_a_diagnosis_is_possible_when_every_ranking_failed(name):
    """When every ranking call fails the module keeps no candidate, refuses no commit write that states no top_rank, prints that
    the ranking could not be read, and raises nowhere. Control: a failed call read as an empty ranking."""
    m, _table, facts, *_ = setup(name)
    ctx = d.Context(d.Board(facts=facts), [None, None])
    run(m.before_turn(ctx))
    run(m.before_step(Step(m.commit_tool, {}, {}, [], candidate=None), m.table.tools[m.commit_tool], ctx))
    assert m.ranking is None and m.failures == 2 and m.candidates() == []
    assert m.step_problems(Step(m.commit_tool, {next(iter(facts["healthy"])): 0}, {}, []), 0, None) == []
    assert m.sections(ctx.board) == [("advice", ranking_module.HEADING_RANKING, [ranking_module.fill(ranking_module.FAILED_RANKING, count=2)])]
    assert m.advice_lines(ctx.board) == [] and json.dumps(m.to_json())


# ----- the advice lines -----

@pytest.mark.parametrize("name", TABLES)
def test_a_ranking_without_advice_is_planned_as_before_under_every_setting(name):
    """Without advice of the server the lines advise rank 1, the top candidate, before a commitment, under every value of reads.
    Control: a line of the server printed without its advice."""
    for reads in ("gain", "ec2", "separating"):
        m, _table, facts, (a, *_), _groups = setup(name, Options(reads=reads))
        ranked(m, facts, [((a,), 1.0)])
        m.ranking.pop("message")
        assert m.advice_lines(d.Board()) == [ranking_module.LINE_RANK_ONE]


@pytest.mark.parametrize("name", TABLES)
def test_an_arm_keeps_its_own_choice_and_the_registered_agent_follows_the_advice(name):
    """Under the default of reads the advised read is the server's; under an arm it is the read batch, the one step edge cutting
    read or the separating test of that arm, whatever the server advises. Control: an arm that follows the server."""
    m, _table, facts, (a, b, *_), groups = setup(name)
    args = dict(read_first=True, advice={"action": "read", m.series_arg: groups[0], m.count_arg: 1}, read_batch=[(groups[-1], 2)],
                separating=(groups[-1], 3), elimination=[(groups[0], 0.1, 0.0), (groups[-1], 0.2, 0.4)])
    assert ranked(m, facts, [((a,), 0.5), ((b,), 0.5)], **args).advised_reads() == [(groups[0], 1)]
    for reads, expected in (("ec2", [(groups[-1], 1)]), ("separating", [(groups[-1], 3)])):
        arm = ranked(setup(name, Options(reads=reads))[0], facts, [((a,), 0.5), ((b,), 0.5)], **args)
        assert arm.advised_reads() == expected
        assert any(line.startswith(f"Advised reads, {ranking_module.READ_ARMS[reads]}") for line in arm.advice_lines(d.Board()))


@pytest.mark.parametrize("name", TABLES)
def test_the_agent_reads_the_flag_under_its_new_name_alone(name):
    """The module reads the read choice from the option reads, the name of the registry, and from no other name. Control: the
    module reads an attribute read_advice of the options."""
    options = Options(reads="ec2")
    m = setup(name, options)[0]
    assert m.options.reads == "ec2" and not hasattr(options, "read_advice")
    assert ranking_module.READ_ARMS["ec2"] in m.blocks()[0]


@pytest.mark.parametrize("name", TABLES)
def test_the_reserve_is_the_plan_of_the_first_candidate(name):
    """The reserve repairs the repair plan of the first candidate plan, else the faulty set of the top candidate, without a target
    whose repair is open. Control: the reserve of the cheapest plan in place of the first."""
    m, _table, facts, (a, b, c, *_), _groups = setup(name)
    ranked(m, facts, [((a, b), 0.6), ((c,), 0.4)], branches=[(1, (a, b), 9.0), (2, (c,), 1.0)])
    assert m.reserve_targets() == (a, b)
    assert m.reserve_targets(d.Ledger(open_targets={a: 5})) == (b,)
    ranked(m, facts, [((c,), 0.6), ((a,), 0.4)])
    assert m.reserve_targets() == (c,)


@pytest.mark.parametrize("name", TABLES)
def test_the_reserve_holds_the_top_plan_and_a_visit_charge_without_the_fault_rule(name):
    """The reserve is the completion that the walk of the kind prices for the repair of the top plan, with its visit charge, and
    reads no rule of faults of its own. Control: the reserve priced on the targets alone, without the walk."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((a, b), 1.0)], branches=[(1, (a, b), 3.0)])
    needs = m.reserve(d.Ledger(prices={a: 10, b: 20}, visit=5))
    assert (needs.price, needs.calls) == (35, 3)


@pytest.mark.parametrize("name", TABLES)
def test_the_reserve_of_the_base_row_keeps_the_calls_the_environment_charges(name):
    """On a row that prices nothing the reserve still keeps the calls of the repair and of the observation that verifies it.
    Control: a reserve of no call where the price is zero."""
    m, _table, facts, (a, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 1.0)], branches=[(1, (a,), 0.0)])
    needs = m.reserve(d.Ledger())
    assert (needs.price, needs.calls) == (0, 2)


@pytest.mark.parametrize("name", TABLES)
def test_a_candidate_without_a_fault_is_an_option_of_the_rule(name):
    """Before a commitment a candidate plan with no faulty target is an option of the cost choice with its own cost, and the rule
    acts on it where it is cheapest. Control: plans without a faulty target left out of the options."""
    m, table, facts, (a, *_), _groups = setup(name)
    ranked(m, facts, [((), 0.5), ((a,), 0.5)], branches=[(1, (), 1.0), (2, (a,), 5.0)])
    assert m.options_of_repair(d.Ledger()) == [(1.0, 1), (5.0, 2)]
    plan = json.loads(d.RuleModel(m, table).complete(d.Board(facts=facts)))
    assert plan["follow"] == 1 and [s["tool"] for s in plan["steps"]] == [m.commit_tool, table.end_tool()]


@pytest.mark.parametrize("name", TABLES)
def test_after_a_refuted_plan_a_candidate_without_a_fault_is_no_option_of_backtracking(name):
    """After a commitment a plan with no faulty target is no option, since the refuted verdict says the repair did not restore
    the asset. Control: such a plan kept among the options after a commitment."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((), 0.5), ((a,), 0.3), ((b,), 0.2)], branches=[(1, (), 1.0), (2, (a,), 5.0), (3, (b,), 7.0)])
    m.record_refuted([a])
    assert m.options_of_repair(d.Ledger(), after_commitment=True) == [(7.0, 3)]


@pytest.mark.parametrize("name", TABLES)
def test_a_refuted_verdict_leads_to_the_cheapest_other_plan_and_the_ablation_keeps_the_surprise_only(name):
    """After a refuted plan the lines advise the cheapest other plan; with the switch backtracking off no line advises one.
    Control: the refuted set kept among the options."""
    m, _table, facts, (a, b, c, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 0.5), ((b,), 0.3), ((c,), 0.2)], branches=[(1, (a,), 1.0), (2, (b,), 3.0), (3, (c,), 2.0)])
    m.ranking.pop("message")
    m.record_refuted([a])
    assert m.advice_lines(d.Board(commitment={"tool": m.commit_tool})) == [ranking_module.fill(ranking_module.LINE_BACKTRACK, rank=3)]
    off = ranked(setup(name, Options(backtracking=False))[0], facts, [((a,), 0.5), ((b,), 0.3), ((c,), 0.2)], branches=[(1, (a,), 1.0), (2, (b,), 3.0), (3, (c,), 2.0)])
    off.ranking.pop("message")
    off.record_refuted([a])
    assert not any("refuted" in line for line in off.advice_lines(d.Board()))


@pytest.mark.parametrize("name", TABLES)
def test_the_advice_keeps_the_refuted_set_out_until_a_later_ranking_names_it_first(name):
    """The refuted set stays out of the advice until a ranking after a further observation names it first; a ranking on the same
    observations does not release it. Control: the set released by any ranking."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    board = d.Board(facts=facts, observations=[{"id": "obs/1"}])
    ctx = d.Context(board, [d.ranking(m, facts, [((a,), 0.9), ((b,), 0.1)])] * 2)
    m.record_refuted([a], observations=1)
    run(m.before_turn(ctx))
    assert m.refuted_last == (a,)
    board.observations.append({"id": "obs/2"})
    run(m.before_turn(ctx))
    assert m.refuted_last is None


@pytest.mark.parametrize("name", TABLES)
def test_after_a_refuted_plan_with_no_other_plan_left_the_advice_names_what_the_ranking_still_holds_as_faulty(name):
    """With no other plan left after a refuted one, the line names the targets of the set with the most mass outside the refuted
    set and outside an open repair. Control: the refuted set named again."""
    m, _table, facts, (a, b, c, *_), _groups = setup(name)
    ranked(m, facts, [((a, b, c), 1.0)], branches=[(1, (a,), 1.0)])
    m.ranking.pop("message")
    m.record_refuted([a])
    lines = m.advice_lines(d.Board(ledger=d.Ledger(open_targets={c: 9}), commitment={"tool": m.commit_tool}))
    words = m._words()
    assert lines == [ranking_module.fill(ranking_module.LINE_STILL_FAULTY, head=ranking_module.HEAD_REFUTED, outside=ranking_module.OUTSIDE_REFUTED,
                                         targets=b, **words)]


@pytest.mark.parametrize("name", TABLES)
def test_after_a_commitment_with_no_plan_offered_the_advice_names_what_the_ranking_still_holds_as_faulty(name):
    """After a commitment with no candidate plan offered, the line names the targets of the set with the most mass. Control: no
    line without candidate plans."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((a, b), 1.0)])
    m.ranking.pop("message")
    lines = m.advice_lines(d.Board(commitment={"tool": m.commit_tool}))
    assert lines == [ranking_module.fill(ranking_module.LINE_STILL_FAULTY, head=ranking_module.HEAD_NONE_LEFT, outside="", targets=", ".join(sorted((a, b))), **m._words())]


@pytest.mark.parametrize("name", TABLES)
def test_the_line_names_a_repair_only_when_the_budget_left_pays_it(name):
    """The line of what is still faulty stands only where the budget left pays the completion of its repair. Control: the line
    printed whatever the budget."""
    m, _table, facts, (a, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 1.0)])
    m.ranking.pop("message")
    committed = {"tool": m.commit_tool}
    assert m.advice_lines(d.Board(ledger=d.Ledger(budget_left=20, prices={a: 15}, visit=5), commitment=committed))
    assert m.advice_lines(d.Board(ledger=d.Ledger(budget_left=19, prices={a: 15}, visit=5), commitment=committed)) == []


@pytest.mark.parametrize("name", TABLES)
def test_while_a_candidate_plan_is_left_the_advice_names_the_plan_and_no_component(name):
    """While another candidate plan is left after a refuted one, the line names its rank and no target. Control: the line of what
    is still faulty printed beside it."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)], branches=[(1, (a,), 1.0), (2, (b,), 2.0)])
    m.ranking.pop("message")
    m.record_refuted([a])
    lines = m.advice_lines(d.Board(commitment={"tool": m.commit_tool}))
    assert lines == [ranking_module.fill(ranking_module.LINE_BACKTRACK, rank=2)] and b not in re.findall(r"\w+", lines[0])


@pytest.mark.parametrize("name", TABLES)
def test_after_a_refuted_plan_the_line_leaves_out_the_refuted_set_that_the_board_and_the_prompt_keep_out(name):
    """The board marks the refuted plan as the prompt says, and the advice line names no plan of the refuted set. Control: the
    mark of the refuted plan left out."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)], branches=[(1, (a,), 1.0), (2, (b,), 2.0)])
    m.record_refuted([a])
    plans = next(rows for _p, heading, rows in m.sections(d.Board()) if heading == ranking_module.HEADING_PLANS)
    assert plans[0].endswith(ranking_module.PLAN_REFUTED + ".") and not plans[1].endswith(ranking_module.PLAN_REFUTED + ".")
    assert "until a later ranking names it first" in m.blocks()[0] or "refuted plan" in m.blocks()[0]
    assert ranking_module.fill(ranking_module.LINE_BACKTRACK, rank=1) not in m.advice_lines(d.Board(commitment={"tool": m.commit_tool}))


# ----- the board -----

@pytest.mark.parametrize("name", TABLES)
def test_the_prompt_says_of_the_advice_what_the_board_shows(name):
    """What the block says the board shows stands on the board: the candidates with their mass, the effective hypotheses,
    read_first, the separating test, the gain per series, the lines of advice and the candidate plans with their warrant, recovery
    and cost both ways. Control: a part of the ranking left off the board."""
    m, _table, facts, (a, b, *_), groups = setup(name)
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)], read_first=True, effective=1.8, separating=(groups[0], 2),
           elimination=[(groups[0], 0.3, 0.1)], branches=[(1, (a,), 1.0)])
    board = [line for _p, _h, rows in m.sections(d.Board()) for line in rows]
    text = "\n".join(board)
    for part in ("mass 0.6", "Effective hypotheses: 1.8.", "read_first True", "Separating test:", f"per {m._words()['series']}",
                 "warrant pred/b1", "predicted recovery if acted 0.9", "act now 1, test first 2"):
        assert part in text, part
    block = m.blocks()[0]
    for said in ("posterior mass", "effective number of hypotheses", "read_first", "separating test", "information gain", "lines of advice",
                 "warrant", "predicted recovery if acted", "the cost both ways"):
        assert said in block, said


@pytest.mark.parametrize("name", TABLES)
def test_for_every_asset_with_nothing_for_the_schedule_to_show_the_board_with_the_switch_parallel_is_the_board_without_it(name):
    """The sections of the module are the same with the switch parallel on and off where no schedule stands. Control: a section
    that reads the switch parallel."""
    m_on, _table, facts, (a, b, *_), _groups = setup(name, Options(parallel=True))
    m_off = setup(name, Options(parallel=False))[0]
    for m in (m_on, m_off):
        ranked(m, facts, [((a,), 0.6), ((b,), 0.4)], branches=[(1, (a,), 1.0)])
    assert m_on.sections(d.Board()) == m_off.sections(d.Board())


@pytest.mark.parametrize("name", TABLES)
def test_the_branches_and_the_single_hypothesis_switch_set_the_candidate_plans_shown(name):
    """With branches off no candidate plan is shown or offered, with single_hypothesis the first alone. Control: the switches not
    read by branches()."""
    m, _table, facts, (a, b, *_), _groups = setup(name)
    args = ([((a,), 0.6), ((b,), 0.4)],)
    kwargs = {"branches": [(1, (a,), 1.0), (2, (b,), 2.0)]}
    assert len(ranked(m, facts, *args, **kwargs).branches()) == 2
    assert ranked(setup(name, Options(branches=False))[0], facts, *args, **kwargs).branches() == []
    assert [b_["rank"] for b_ in ranked(setup(name, Options(single_hypothesis=True))[0], facts, *args, **kwargs).branches()] == [1]


@pytest.mark.parametrize("name", TABLES)
def test_the_known_identifiers_and_the_evidence_of_a_commitment_come_from_the_ranking(name):
    """The module knows the identifiers that its candidates cite and the warrants of its candidate plans, and adds to a commitment
    by a ranked candidate the identifiers that candidate cites. Control: the warrants of the plans left out."""
    m, table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)], branches=[(1, (a,), 1.0)])
    assert {"obs/1/x", "obs/2/x", "pred/b1", f"pred/{a}"} <= m.known_ids()
    assert m.evidence_for(Step(m.commit_tool, {}, {}, [], candidate=2), table.tools[m.commit_tool]) == ["obs/2/x"]
    assert m.evidence_for(Step(m.repair_tool, {}, {}, []), table.tools[m.repair_tool]) == []


# ----- the stand-in model of the rule -----

@pytest.mark.parametrize("name", TABLES)
def test_every_read_is_a_step_of_a_model_plan_and_the_board_shows_the_read_batch_as_advice(name):
    """The rule plays no read of its own: while the ranking says read first, it writes each advised read as a step of a plan in
    the JSON form of a model, with the expectation the ranking supports, and the board shows the reads as advice. Control: a rule
    that reads outside its plan."""
    m, table, facts, (a, b, *_), groups = setup(name, Options(reads="gain"))
    ranked(m, facts, [((a,), 0.5), ((b,), 0.5)], read_first=True, effective=2.0,
           advice={"action": "read", m.series_arg: groups[0], m.count_arg: 2})
    plan = json.loads(d.RuleModel(m, table).complete(d.Board(facts=facts)))
    assert plan["steps"] == [{"tool": m.observe_tool, "args": {m.series_arg: groups[0], m.count_arg: 2}, "expect": {"effective_at_most": 2.0}}]
    assert any(groups[0] in line for line in m.advice_lines(d.Board()))


@pytest.mark.parametrize("name", TABLES)
def test_under_the_advice_of_the_server_the_reads_of_its_rule_run_as_one_batch_where_it_names_more_than_one(name):
    """Under an arm whose rule names several reads the rule writes them as consecutive observe steps of one plan, one batch.
    Control: one read per plan."""
    m, table, facts, (a, b, *_), groups = setup(name, Options(reads="ec2"))
    ranked(m, facts, [((a,), 0.5), ((b,), 0.5)], read_first=True, separating=(groups[0], 1))
    m.ranking["read_batch"] = []
    m.options = Options(reads="gain")
    m.ranking.pop("advice", None)
    m.ranking["read_batch"] = [{m.series_arg: g, m.count_arg: 1} for g in groups[:2]]
    plan = json.loads(d.RuleModel(m, table).complete(d.Board(facts=facts)))
    assert [s["tool"] for s in plan["steps"]] == [m.observe_tool] * 2


@pytest.mark.parametrize("name", TABLES)
def test_the_rule_acts_on_the_cheapest_branch_and_the_advice_without_the_server_names_rank_one(name):
    """Where the ranking does not say read first the rule commits by the rank of the cheapest candidate plan, repairs its targets,
    reads to verify and ends; without candidate plans the advice names rank 1. Control: the rule acts on rank 1 whatever the
    costs."""
    m, table, facts, (a, b, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)], branches=[(1, (a,), 5.0), (2, (b,), 2.0)])
    plan = json.loads(d.RuleModel(m, table).complete(d.Board(facts=facts)))
    assert plan["follow"] == 2 and plan["steps"][0] == {"tool": m.commit_tool, "args": {"candidate": 2}, "expect": {"top_rank": 1}}
    assert [s["tool"] for s in plan["steps"][1:]] == [m.repair_tool, m.observe_tool, table.end_tool()]
    bare = ranked(setup(name)[0], facts, [((a,), 1.0)])
    bare.ranking.pop("message")
    assert bare.advice_lines(d.Board()) == [ranking_module.LINE_RANK_ONE]


@pytest.mark.parametrize("name", TABLES)
def test_the_rule_writes_no_kind_the_row_does_not_serve(name):
    """On a row whose table does not list a tool the rule writes no step of it. Control: a rule that writes every step of the
    rig."""
    m, table, facts, (a, *_), _groups = setup(name)
    ranked(m, facts, [((a,), 1.0)], branches=[(1, (a,), 1.0)])
    for tool in (m.repair_tool, m.observe_tool):
        narrow = d.without(table, tool)
        plan = json.loads(d.RuleModel(m, narrow).complete(d.Board(facts=facts)))
        assert all(step["tool"] in narrow.tools for step in plan["steps"]) and tool not in [s["tool"] for s in plan["steps"]]


@pytest.mark.parametrize("name", TABLES)
def test_the_plans_of_the_rule_are_replies_that_the_reader_takes(name):
    """Every plan of the stand-in model is one JSON object with the head of a plan, every step naming a tool of the table; where
    the reader of replies is in the tree, it reads the plan with no refusal. Control: a plan written with a key of the present
    plan form."""
    m, table, facts, (a, b, *_), groups = setup(name)
    replies = [d.RuleModel(m, table).complete(d.Board(facts=None))]
    ranked(m, facts, [((a,), 0.6), ((b,), 0.4)], branches=[(1, (a,), 1.0)])
    replies.append(d.RuleModel(m, table).complete(d.Board(facts=facts)))
    for reply in replies:
        plan = json.loads(reply)
        assert set(plan) <= {"follow", "why", "candidates", "steps"} and plan["steps"]
        assert all(set(step) <= {"tool", "args", "expect", "evidence"} and step["tool"] in table.tools for step in plan["steps"])
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan import reply_parser
    try:
        parsed = [reply_parser.parse_usable_plan(reply, table, {}, None) for reply in replies]
    except NotImplementedError:
        pytest.skip("needs task 4: the reader of replies")
    assert all(p.refusal is None for p in parsed)


# ----- the tests of the map that wait for other tasks -----

WAITING = {
    "the_batch_reserves_the_verifying_read_the_agent_reserves_whatever_it_was_told": "needs task 7: the schedule and the loop",
    "the_batch_reserves_the_verifying_read_of_every_repair_the_agent_waits_on": "needs task 7: the schedule and the loop",
    "the_rule_that_follows_the_reads_of_the_server_passes_the_plan_check": "needs task 4: the plan check",
    "a_read_that_several_candidate_plans_share_is_scheduled_once_with_the_plans_it_serves": "needs task 7: the schedule",
    "the_reads_of_a_batch_are_cut_where_they_no_longer_fit_together": "needs task 7: the schedule",
    "a_plan_that_could_not_be_completed_after_the_named_reads_is_left_out_and_named": "needs task 7: the schedule",
    "a_read_the_server_does_not_name_is_added_only_beside_the_repair_of_every_component": "needs task 7: the schedule",
    "under_one_read_through_a_group_that_covers_every_test_the_board_is_the_board_without_the_switch": "needs task 7: the schedule",
    "after_a_refuted_plan_the_schedule_lists_what_each_remaining_plan_needs_and_repeats_no_executed_step": "needs task 7: the schedule",
    "the_rule_plans_the_scheduled_reads_as_one_run_and_follows_the_board_otherwise": "needs task 7: the schedule",
    "the_ranking_after_a_batch_is_the_ranking_after_the_same_readings_one_by_one": "needs tasks 2 and 7: an episode on the rig with the server",
    "after_a_refuted_plan_the_agent_reads_what_the_remaining_plans_share_before_the_next_repair": "needs task 7: the schedule and an episode",
}


@pytest.mark.parametrize("name", sorted(WAITING))
def test_a_test_of_the_map_that_waits_for_another_task(name):
    """A test of the map under ranking module that needs an episode, the schedule or the check, kept by its name until the task it
    needs is in the tree."""
    pytest.skip(WAITING[name])
