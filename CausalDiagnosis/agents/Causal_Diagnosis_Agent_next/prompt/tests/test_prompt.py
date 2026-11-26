"""The tests of the prompt (CausalDiagnosis/docs/general-agent-tasks.md, Task 6): the core texts as the plan page writes them, the
rendered clauses, the four turns, the variants of the switches, the sections of the steps that did not run and of the standing
steps, identifier_forms, the prompts of the fixture declarations kept as files, and the tests of the map under prompt,
re-expressed through the interfaces of the plan page. The board and the ledger are read through the doubles of doubles.py.

Each test names its control in its docstring: the change that makes it fail. A control that a test can run itself is run by a
test of its own whose name says what it plants."""
import dataclasses
import os
import re
from pathlib import Path

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.prompt import assembler
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.prompt import core_protocol as core
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.prompt.tests import doubles
from CausalDiagnosis.agents.utils import core_texts

DATA = Path(__file__).resolve().parent / "data"
PLAN_PAGE = doubles.ROOT / "CausalDiagnosis" / "docs" / "general-agent-plan.md"
NAMES = ("rig", "aob", "aob_real_time", "port", "single", "tiny")
WRITE_FILES = os.environ.get("CAUSAL_NEXT_WRITE_PROMPTS") == "1"  # set to write the kept prompt files again after a reviewed change


def _table(name, binding=True):
    declaration, listing, facts = doubles.declarations()[name]
    return (declaration.table(listing) if binding else declaration.listed(listing)), facts


def _turns(name, options=None, modules=(), facts=True, board=None):
    """name of turn -> text: the system prompt before and after the facts, the orientation with a retry, the planning turn before
    the facts, after them, and with a retry, a cut and standing steps, the summary of earlier turns with entries left out, and the
    closing summary with a write, a verdict and a note."""
    table, asset_facts = _table(name)
    board = board or doubles.Board(facts=asset_facts if facts else None)
    before = doubles.Board(facts=None)
    tool = next(iter(table.tools))
    written = doubles.Board(facts=asset_facts, writes=[{"tool": table.end_tool(), "args": {}, "answer": {"result": "done"}}],
                            verdicts=["the repair is consistent with its warrant"])
    return {
        "system_before_facts": assembler.system_prompt(table, before, options, modules),
        "system": assembler.system_prompt(table, board, options, modules),
        "orientation": assembler.orientation_turn(board, table, options, modules, retry="The reply could not be read."),
        "planning_before_facts": assembler.planning_turn(before, table, options, modules),
        "planning": assembler.planning_turn(board, table, options, modules),
        "planning_after_a_cut": assembler.planning_turn(
            board, table, options, modules, retry=["step 2 (x): the reason of the check"],
            not_run={"step": 2, "steps": [{"tool": tool}], "reasons": [f"step 2 ({tool}): the reason of the check"]},
            stood=[(1, tool, "the reason it stood")]),
        "earlier_turns": assembler.earlier_turns_turn(board, table, options, modules, left_out=2),
        "summary": assembler.summary_turn(written, table, options, modules, note="The episode ended."),
    }


def _pattern(words):
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")(?!\w)", re.IGNORECASE)


def _own_words(name):
    """The words that a declaration holds itself: its asset's words where it is a registered or the tiny asset, its vocabulary,
    the names of its tools and arguments, the words of its rows, and the texts of its listing."""
    declaration, listing, _facts = doubles.declarations()[name]
    words = set()
    for data in [*doubles.registered_assets(), doubles.asset_data(doubles.TINY_FILE)]:
        if data["asset_id"] == declaration.asset_id:
            words |= doubles.asset_words(data)
    vocabulary = declaration.vocabulary
    held = [vocabulary.noun, vocabulary.short_noun, vocabulary.example_field or "", *map(str, vocabulary.terms.values())]
    for tools in listing.values():
        for tool in tools:
            held += [tool.name, tool.description, *((tool.inputSchema or {}).get("properties") or {})]
    for row in declaration.rows.values():
        held += [*row.words, str(row.sets or ""), *map(str, row.values.values())]
    text = " ".join(held)
    return {word for word in _banned_of(name, everyone=True) if _pattern([word]).search(text)} | words


def _banned_of(name, everyone=False):
    declaration = doubles.declarations()[name][0]
    return set().union(*(doubles.asset_words(data) for data in doubles.registered_assets()
                         if everyone or data["asset_id"] != declaration.asset_id))


def _banned(name):
    """The words of every registered asset but the declaration's own that the declaration does not hold itself."""
    return _banned_of(name) - _own_words(name)


def _hits(name, texts):
    banned = _banned(name)
    if not banned:
        return []
    pattern = _pattern(banned)
    return sorted({m.group(0) for text in texts for m in pattern.finditer(text)})


# ----- the core texts -----

def _plan_page_blocks():
    """name -> text of every block of the section The prompt of the plan page that stands under a name of its own."""
    text = PLAN_PAGE.read_text(encoding="utf-8")
    section = text.split("## The prompt", 1)[1].split("\n## ", 1)[0]
    return {m.group(1): m.group(2) for m in re.finditer(r"\n([A-Z]+(?: \(head\))?)\n```text\n(.*?)\n```", section, re.DOTALL)}


def test_the_core_constants_hold_the_words_of_the_plan_page():
    """Every core text that the plan page writes under a name of its own is a constant of core_protocol word for word, and the
    shared ones are the constants of agents/utils/core_texts.py. Control: one word of a constant changed."""
    blocks = _plan_page_blocks()
    constants = {"ROLE": core.ROLE, "TOOLS (head)": core.TOOLS_HEAD, "EXPECTATIONS": core.EXPECTATIONS, "EVIDENCE": core.EVIDENCE,
                 "CANDIDATES": core.CANDIDATES, "BOARD": core.BOARD, "PLAN": core.PLAN, "RECALL": core.RECALL, "BATCH": core.BATCH,
                 "RULES (head)": core.RULES_HEAD, "LOOP": core.LOOP}
    assert set(constants) <= set(blocks), sorted(set(constants) - set(blocks))
    for name, constant in constants.items():
        assert constant == blocks[name], name
    for name in ("ROLE", "EXPECTATIONS", "EVIDENCE", "BOARD", "RULES_HEAD", "LOOP", "OBJECTIVE", "LOOP_STANDING", "READ_STANDS"):
        assert getattr(core, name) is getattr(core_texts, name), name
    page = PLAN_PAGE.read_text(encoding="utf-8")
    for constant in (core.ORIENTATION_ASK, core.BEFORE_FACTS_TURN, core.BEFORE_FACTS_BOARD, core.EVENTS, core.EVIDENCE_CARRIED,
                     core.BOARD_HELD, core.BOARD_HELD_RECORD, core.LOOP_STANDING, core.CLOSING_ASK, core.EARLIER_TURNS_KEEPS,
                     core.COUNT_SET, core.RECALL_SERIES):
        assert constant and (constant in page or constant.replace('"', "") in page), constant


def test_a_changed_word_of_a_core_constant_is_found(monkeypatch):
    """The control of the test of the core constants, run: a word of BATCH changed."""
    monkeypatch.setattr(core, "BATCH", core.BATCH.replace("one batch", "a batch"))
    with pytest.raises(AssertionError):
        test_the_core_constants_hold_the_words_of_the_plan_page()


def test_no_core_text_and_no_text_of_the_assembler_holds_an_em_or_en_dash():
    """The texts of the repository use neither dash. Control: an em dash planted in a constant."""
    texts = [value for value in vars(core).values() if isinstance(value, str)]
    texts += [text for name in NAMES for text in _turns(name).values()]
    assert not [t for t in texts if "\u2014" in t or "\u2013" in t]


# ----- words of an asset -----

@pytest.mark.parametrize("name", NAMES)
def test_for_every_registered_asset_the_prompts_hold_no_word_of_another_asset(name):
    """The system prompt before and after the facts and the four turns, with a module and without, hold no whole word of a
    registered asset that the declaration does not hold itself. Control: a noun of another asset planted in a core constant
    (test_a_noun_of_another_asset_planted_in_a_core_constant_is_found)."""
    texts = [*_turns(name).values(), *_turns(name, modules=[doubles.Module()]).values()]
    assert not _hits(name, texts)
    if doubles.declarations()[name][0].asset_id not in {data["asset_id"] for data in doubles.registered_assets()}:
        assert _banned(name)


def test_a_noun_of_another_asset_planted_in_a_core_constant_is_found(monkeypatch):
    """The control of the test of the words of an asset, run: the short noun of the rig planted in BATCH shows in every prompt
    of the made-up port and of the tiny asset."""
    rig = next(data for data in doubles.registered_assets() if data["asset_id"] == "hyd_1")
    monkeypatch.setattr(core, "BATCH", core.BATCH + f" The {rig['text']['short_noun']} waits.")
    for name in ("port", "tiny"):
        assert _hits(name, _turns(name).values()) == [rig["text"]["short_noun"]]


@pytest.mark.parametrize("name", NAMES)
def test_the_prompt_of_an_asset_speaks_of_no_write_class_role_or_event_that_its_table_lacks(name):
    """A role, a write class or the events stand in the system prompt only where the table holds them. Control: every clause of
    the roles and the write classes printed whatever the table holds."""
    table, facts = _table(name)
    text = assembler.system_prompt(table, doubles.Board(facts=facts), None, [])
    rows = [row for row in table.tools.values() if row.role != "advice"]
    slots = {"short_noun": table.declaration.vocabulary.short_noun}
    for role, clause in core.ROLE_CLAUSES:
        assert (assembler.fill(clause, **slots) in text) == any(row.role == role for row in rows), (name, role)
    for write, clause in core.WRITE_CLAUSES:
        assert (assembler.fill(clause, **slots) in text) == any(row.write == write for row in rows), (name, write)
    assert ("no_event" in text) == bool(table.declaration.events)
    assert not any(f"- {row.name} (" in text for row in table.tools.values() if row.role == "advice")


# ----- the switches -----

def _remove(text, part):
    assert part in text
    return text.replace(part, "")


@pytest.mark.parametrize("name", NAMES)
def test_with_a_switch_off_its_block_or_sentence_is_absent_and_no_other_text_changes(name):
    """recall off takes out RECALL, parallel off takes out BATCH, expectations off changes the first sentence of LOOP alone, and
    a plan length changes the count of PLAN alone; every other text stays. Control: a switch that changes another block."""
    table, facts = _table(name)
    board = doubles.Board(facts=facts)
    on = assembler.system_prompt(table, board, Options(), [])
    assert assembler.system_prompt(table, board, Options(recall=False), []) == _remove(on, "\n\n" + assembler.recall_block(table))
    assert assembler.system_prompt(table, board, Options(parallel=False), []) == _remove(on, "\n\n" + core.BATCH)
    first = assembler.fill(core.LOOP, plan_head=core.PLAN_HEAD, limits=assembler.limits(table, board), short_noun=table.declaration.vocabulary.short_noun).split(". ", 1)[0] + "."
    assert assembler.system_prompt(table, board, Options(expectations=False), []) == on.replace(first, core.LOOP_WITHOUT_EXPECTATIONS)
    counted = assembler.system_prompt(table, board, Options(plan_length=6), [])
    paid = assembler.fill(core.COUNT_PAID, limits=assembler.limits(table, board, "and"))
    assert counted == on.replace(paid, assembler.fill(core.COUNT_SET, n=6))


@pytest.mark.parametrize("name", NAMES)
def test_with_the_switch_structured_output_off_no_text_changes(name):
    """The schema of a reply travels in a field of the request, never in its text, and the switches of the loop alone change no
    text either. Control: a sentence printed under structured_output."""
    on = _turns(name, Options())
    for switch in ("structured_output", "summary_turn", "backtracking", "binding"):
        assert _turns(name, Options(**{switch: False})) == on, switch


# ----- the rendered texts -----

def test_evidence_names_the_tools_that_fill_an_argument_from_evidence_and_leaves_that_sentence_out_without_one():
    """EVIDENCE carries the sentence of the tools whose rows fill an argument from evidence, and leaves it out on a table without
    one. Control: the sentence printed on every table."""
    rig, _facts = _table("rig")
    text = assembler.evidence_block(rig, doubles.Board(), [])
    assert assembler.fill(core.EVIDENCE_CARRIED, evidence_steps="submit_diagnosis, declare_insufficient_evidence or create_work_order") in text
    for name in ("port", "single"):
        table, _facts = _table(name)
        assert '"evidence"' not in assembler.evidence_block(table, doubles.Board(), [])


def test_the_sentence_of_a_call_with_nothing_new_stands_only_where_the_vocabulary_holds_its_terms():
    """The sentence stands at the end of the line of an observe tool with a series, and only where the vocabulary holds the terms
    time_unit and series. Control: the sentence printed without the terms."""
    declaration, listing, _facts = doubles.declarations()["rig"]
    line = next(l for l in assembler.tools_block(declaration.table(listing), doubles.Board(), []).split("\n") if l.startswith("- read_sensors"))
    assert line.endswith(assembler.fill(core.READ_STANDS, time_unit="cycle", series="sensor group", step_key="tool"))
    bare = dataclasses.replace(declaration, vocabulary=dataclasses.replace(declaration.vocabulary, terms={"target": "component"}))
    assert "makes no call and stands" not in assembler.system_prompt(bare.table(listing), doubles.Board(), None, [])
    for name in ("port", "single"):
        table, _facts = _table(name)
        assert "makes no call and stands" not in assembler.tools_block(table, doubles.Board(), [])


def test_the_values_of_an_argument_come_from_the_facts_once_they_are_on_the_board():
    """Before the facts a line names the path of the facts tool; once they are on the board, the values themselves. Control: the
    path printed after the facts."""
    table, facts = _table("rig")
    before = assembler.system_prompt(table, doubles.Board(facts=None), None, [])
    after = assembler.system_prompt(table, doubles.Board(facts=facts), None, [])
    assert "one of the sensor_groups that get_asset_profile lists" in before
    assert "one of " + ", ".join(list(facts["sensor_groups"])[:-1]) + " or " + list(facts["sensor_groups"])[-1] in after
    assert "one of the sensor_groups that get_asset_profile lists" not in after


def test_the_requirement_of_a_tool_follows_the_precondition_card():
    """A tool that the card requires names the state it needs, and an argument value that needs another state names it with the
    value. Control: the card left out of the declaration."""
    table, _facts = _table("port")
    lines = {l.split(" ", 2)[1]: l for l in assembler.tools_block(table, doubles.Board(), []).split("\n") if l.startswith("- ")}
    assert "Needs supply live." in lines["read_gauge"] and "Needs supply isolated." in lines["swap_part"]
    assert "With mode open: needs supply isolated." in lines["inspect_nozzle"]
    assert "Needs" not in lines["file_report"]


def test_candidates_and_the_commitment_render_from_the_commit_writes():
    """{candidate_arguments} names the candidate arguments with their values and their omitted path, and {commitment} names the
    commit writes and the write that ends, or one write that does both. Control: the commitment of the rig printed for the port."""
    rig, _facts = _table("rig")
    port, _facts = _table("port")
    text = assembler.candidates_block(rig)
    assert "cooler, valve, pump and accumulator, each one of its values above; one you leave out takes the value that get_asset_profile lists for it under healthy." in text
    assert text.endswith("You commit your answer with submit_diagnosis or declare_insufficient_evidence, once in an episode. finish ends the episode.")
    assert assembler.candidates_block(port).endswith("You commit your answer with file_report, which also ends the episode.")


def test_the_limits_are_those_that_the_ledger_knows():
    """{limits} names the limits of the ledger in the words of the kind, and the limits of the episode while the ledger knows
    none. Control: the limits printed whatever the ledger knows."""
    table, _facts = _table("rig")
    assert assembler.limits(table, doubles.Board(ledger=doubles.Ledger(limits=("budget", "steps")))) == "the budget or the step limit"
    assert assembler.limits(table, doubles.Board(ledger=doubles.Ledger(limits=("budget", "time", "steps"))), "and") == "the budget, the horizon and the step limit"
    assert assembler.limits(table, doubles.Board(ledger=doubles.Ledger(limits=()))) == core.NO_LIMITS


def test_the_blocks_sentences_forms_and_identifiers_of_the_modules_stand_in_the_prompt():
    """A module adds its block after BOARD, its sentence at the end of BOARD, its forms in the lines of the tools it judges and
    its identifier forms in EVIDENCE; without it none stands. Control: the blocks of a module left out."""
    table, facts = _table("rig")
    module = doubles.Module()
    with_module = assembler.system_prompt(table, doubles.Board(facts=facts), None, [module])
    without = assembler.system_prompt(table, doubles.Board(facts=facts), None, [])
    board_end = with_module.index(core_texts.BOARD_HELD)
    assert with_module.index("The board also shows the advice.") > board_end
    assert with_module.index(module.blocks()[0]) < with_module.index("The plan. Return")
    line = next(l for l in with_module.split("\n") if l.startswith("- read_sensors"))
    assert 'expect {"top_rank": <a rank of at least 1>}.' in line
    assert "adv/<n> for an answer of the module" in with_module
    for text in (module.blocks()[0], "The board also shows the advice.", "top_rank", "adv/<n>"):
        assert text not in without


def test_identifier_forms_join_the_answer_forms_the_kind_the_modules_and_the_knowledge_tools():
    """identifier_forms returns each form once with what it names: obs forms by the answer forms of the table, the forms of the
    modules, and one form per knowledge tool. Control: a form of a module left out."""
    rig, _facts = _table("rig")
    port, _facts = _table("port")
    forms = assembler.identifier_forms(rig, doubles.Board(), [doubles.Module()])
    assert forms[0] == core.FORM_RECORDS and ("adv/<n>", "an answer of the module") in forms
    assert ("<as an answer of get_failure_modes gives it>", "an entry of that answer") in forms
    assert assembler.identifier_forms(port, doubles.Board(), []) == (core.FORM_TEXT,)
    assert len(set(forms)) == len(forms)


def test_the_sentences_of_the_rules_follow_the_head_of_rules_joined_by_semicolons(monkeypatch):
    """RULES prints its head and the sentences of the rules in force in the order that plan/rules.py gives, joined by semicolons,
    and is left out while the rules state none. Control: the sentences printed in another order."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan import rules
    table, facts = _table("port")
    assert core.RULES_HEAD.split(" {")[0] not in assembler.system_prompt(table, doubles.Board(facts=facts), None, [])
    monkeypatch.setattr(rules, "sentences", lambda table, board, modules: ("no first rule", "no second rule"), raising=False)
    text = assembler.system_prompt(table, doubles.Board(facts=facts), None, [])
    assert assembler.fill(core.RULES_HEAD, short_noun="filler") + " no first rule; no second rule." in text
    assert text.index("no second rule.") < text.index("A turn that calls no tool of the filler still takes one step.")


# ----- the four turns -----

def test_planning_turn_prints_the_standing_steps_after_the_steps_that_did_not_run_and_none_without_them():
    """With standing steps the planning turn prints their section after the section of the steps that did not run, each step in
    the form of STOOD_LINE; without them it prints none. Control: the keyword stood left out of the call."""
    table, facts = _table("rig")
    board = doubles.Board(facts=facts)
    not_run = {"step": 3, "steps": [{"tool": "read_sensors"}], "reasons": ["step 3 (read_sensors): a reason"]}
    stood = [(1, "get_asset_profile", "leave this step out, since it stood")]
    text = assembler.planning_turn(board, table, None, [], not_run=not_run, stood=stood)
    assert text.index(core.NOT_RUN_HEADING) < text.index(core.NOT_RUN_CLOSING) < text.index(core.STOOD_HEADING) < text.index(core.STOOD_CLOSING)
    assert "- step 1 (get_asset_profile): leave this step out, since it stood" in text
    assert "- step 3 (read_sensors)\n- step 3 (read_sensors): a reason" in text
    assert core.STOOD_HEADING not in assembler.planning_turn(board, table, None, [], not_run=not_run)
    assert core.NOT_RUN_HEADING not in assembler.planning_turn(board, table, None, [], stood=stood)


def test_the_four_turns_hold_their_heading_their_ask_and_their_retry():
    """The orientation ends with the ask of the differential and a retry, the planning turn with the objective, the summary of
    earlier turns with its ask and the note of entries left out, the closing summary with the writes, the verdicts, the limits
    and its ask. Control: the ask of a turn left out."""
    turns = _turns("rig")
    assert core.ORIENTATION_HEADING in turns["orientation"] and turns["orientation"].endswith(f"{core.RETRY_HEADING}\nThe reply could not be read.\n{core.RETRY_DIFFERENTIAL}")
    assert core.ORIENTATION_ASK in turns["orientation"] and "Write the differential:" in turns["orientation"]
    assert turns["planning"].endswith(core.OBJECTIVE)
    assert turns["earlier_turns"].endswith(core.EARLIER_TURNS_ASK + " " + core.LEFT_OUT)
    summary = turns["summary"]
    assert '- finish: done' in summary and "- the repair is consistent with its warrant" in summary
    assert "Limits: cycle 3, horizon 100, budget left 50, steps left 9." in summary and summary.endswith(assembler.fill(core.CLOSING_ASK, short_noun="rig"))


# ----- the tests of the map under prompt -----

@pytest.mark.parametrize("name", NAMES)
def test_the_system_prompt_tells_that_a_turn_without_a_call_takes_a_step(name):
    """Told equals done: the system prompt states that a turn without a call of a tool still takes one step, and that a reply
    that runs nothing is asked again with the reasons. Control: LOOP left out of the system prompt."""
    table, facts = _table(name)
    text = assembler.system_prompt(table, doubles.Board(facts=facts), None, [])
    assert f"A turn that calls no tool of the {table.declaration.vocabulary.short_noun} still takes one step." in text
    assert "asked again with the reasons" in text and "refuses twice" not in text


def test_the_planning_turn_names_a_count_only_when_one_is_set():
    """The ask of the planning turn is the objective, followed before the facts by the sentence of the opening, and a count
    stands in the plan block of the system prompt alone. Control: the count printed without plan_length."""
    table, facts = _table("rig")
    for length in (None, 6):
        options = Options(plan_length=length)
        assert assembler.planning_turn(doubles.Board(facts=facts), table, options, []).endswith(core.OBJECTIVE)
        assert assembler.planning_turn(doubles.Board(facts=None), table, options, []).endswith(
            core.OBJECTIVE + " " + assembler.fill(core.BEFORE_FACTS_TURN, facts_tool="get_asset_profile"))
    counted = assembler.fill(core.COUNT_SET, n=6)
    assert counted in assembler.planning_turn(doubles.Board(facts=facts), table, Options(plan_length=6), [])
    assert counted not in assembler.planning_turn(doubles.Board(facts=facts), table, Options(), [])
    assert assembler.planning_turn(doubles.Board(facts=facts), table, Options(plan_length=6), []).count(counted) == 1


def test_the_board_names_the_time_tools_on_a_clock_row_only():
    """The clock read stands in the prompt only on a row whose table lists it: on the row real_time of AssetOpsBench's servers and
    not on the base row, nor on the rig's own server, where the row withholds it. Control: a line printed for a tool the table
    does not hold."""
    real_time, facts = _table("aob_real_time")
    assert "- get_current_time (task)" in assembler.planning_turn(doubles.Board(facts=facts), real_time, None, [])
    for name in ("aob", "rig"):
        table, facts = _table(name)
        assert "get_current_time" not in assembler.planning_turn(doubles.Board(facts=facts), table, None, [])


@pytest.mark.parametrize("name", NAMES)
def test_every_prompt_of_an_episode_holds_the_task_message_with_its_block(name):
    """The task message of the episode, the briefing with a block appended, stands once in every prompt: the orientation, every
    planning turn, the summary of earlier turns and the closing summary. Control: the task left out of a turn, or printed twice."""
    for turn, text in _turns(name).items():
        if turn.startswith("system"):
            assert doubles.TASK not in text
        else:
            assert text.count(doubles.TASK) == 1, turn


def test_the_texts_of_the_registered_asset_keep_their_words():
    """The texts of the registered asset name it by its noun and its short noun and give its example field, from the
    vocabulary; a vocabulary without an example gives none. Control: the noun typed into a core constant in place of its slot."""
    declaration, listing, facts = doubles.declarations()["rig"]
    data = next(d for d in doubles.registered_assets() if d["asset_id"] == declaration.asset_id)
    text = assembler.system_prompt(declaration.table(listing), doubles.Board(facts=facts), None, [])
    assert f"You plan the work on one {data['text']['noun']} for a controller." in text
    assert f"A turn that calls no tool of the {data['text']['short_noun']} still takes one step." in text
    assert f", such as {data['text']['example_channel']}," in text
    plain = assembler.system_prompt(doubles.without_example(declaration).table(listing), doubles.Board(facts=facts), None, [])
    assert ", such as" not in plain and "and an underscore, and a field" in plain


def test_the_texts_the_agent_builds_for_the_tiny_asset_hold_no_word_of_another_asset():
    """The texts that the prompt builds for the tiny asset, read from its file, before and after the facts, with a retry, a cut,
    standing steps, a note and a module, on its own table and on the listed one, hold no whole word of a registered asset that the
    tiny asset does not hold itself; the words are read from the asset files, so a further registered asset needs no change.
    Control: a noun of the rig planted in a core constant (test_a_noun_of_another_asset_planted_in_a_core_constant_is_found)."""
    texts = [*_turns("tiny").values(), *_turns("tiny", modules=[doubles.Module()]).values()]
    declaration, listing, facts = doubles.declarations()["tiny"]
    texts.append(assembler.system_prompt(declaration.listed(listing), doubles.Board(facts=facts), None, []))
    assert _banned("tiny") and not _hits("tiny", texts)


def test_the_texts_of_the_agent_name_the_plain_statistic_the_asset_serves():
    """The core names the values of an asset by no statistic of its own: a statistic reaches a text only through the vocabulary or
    the listing, so the texts of the tiny asset whose plain statistic is value hold no word of another statistic. Control: the
    word mean planted in a core constant."""
    other = re.compile(r"(?<![A-Za-z0-9])(?:means?|std|standard deviations?)(?![A-Za-z0-9])", re.IGNORECASE)
    declaration, listing, facts = doubles.tiny_declaration(value="value")
    table = declaration.table(listing)
    board = doubles.Board(facts=facts)
    texts = [assembler.system_prompt(table, board, None, []), assembler.planning_turn(board, table, None, []),
             assembler.orientation_turn(board, table, None, []), assembler.summary_turn(board, table, None, [])]
    assert not [m.group(0) for text in texts for m in other.finditer(text)]
    assert not [value for value in vars(core).values() if isinstance(value, str) and other.search(value)]


def test_the_texts_of_the_schedule_on_the_tiny_asset_hold_no_word_of_another_asset():
    """The texts of the batch, the block BATCH and the sentences of a module beside it, on the tiny asset hold no whole word of
    another registered asset; the lines of the schedule itself are the module's and the loop's. Control: a noun of the rig
    planted in BATCH."""
    table, facts = _table("tiny")
    module = doubles.Module(block="Advice: the reads that several plans share.", sentence="The schedule shows the shared reads.")
    text = assembler.system_prompt(table, doubles.Board(facts=facts), Options(parallel=True), [module])
    assert core.BATCH in text
    assert not _hits("tiny", [core.BATCH, *module.blocks()[:1], text])


# ----- the prompts kept as files -----

def _kept(name):
    out = {}
    for turn, text in _turns(name).items():
        out[f"{name}_{turn}.txt"] = text + "\n"
    out[f"{name}_system_with_module.txt"] = assembler.system_prompt(_table(name)[0], doubles.Board(facts=_table(name)[1]), None, [doubles.Module()]) + "\n"
    return out


@pytest.mark.parametrize("name", NAMES)
def test_the_prompts_of_the_fixture_declarations_are_the_files_kept_for_review(name):
    """The prompts of every fixture declaration are kept as files under data/, so that a change of a text shows in review; with
    CAUSAL_NEXT_WRITE_PROMPTS=1 the test writes them again. Control: one word of a core constant changed."""
    for file, text in _kept(name).items():
        path = DATA / file
        if WRITE_FILES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        assert path.read_text(encoding="utf-8") == text, file
