"""The check's public rule sentences survive the prompt owner's assembler."""

from dataclasses import replace

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.prompt import assembler, core_protocol
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests.fixture_declarations import declarations
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.working_memory.board import Board
from .. import messages, rules
from .fakes import Ledger


class PublicLedger(Ledger):
    def limits(self):
        return tuple(key for key in ("budget_left", "time_limit", "steps_left") if self.state.get(key) is not None)


def fixture(name, facts, options):
    declaration, listing = declarations()[name]
    table = declaration.table(listing)
    ledger = PublicLedger(time=0, time_limit=20, budget_left=20, steps_left=20)
    board = Board(table, ledger, "Inspect the public evidence.", options)
    board._facts = {} if facts else None
    return table, board


def assert_rule_sentences(table, board, options):
    active = assembler.active_rules(table, board, options, ())
    blocks = dict(assembler.rendered_blocks(table, board, options, ()))
    request = assembler.planning_turn(board, table, options, ())
    for name, sentence in messages.RULE_SENTENCES:
        assert callable(getattr(rules, name)), name
        assert (sentence in blocks["RULES"]) == (name in active), name
        if name in active:
            assert sentence in request, name
    return set(active)


@pytest.mark.parametrize("facts", [False, True])
def test_each_declared_rule_sentence_is_printed_before_and_after_facts(facts):
    seen = set()
    for name in declarations():
        options = Options()
        table, board = fixture(name, facts, options)
        seen.update(assert_rule_sentences(table, board, options))
    assert seen == {name for name, _sentence in messages.RULE_SENTENCES}


def test_inactive_limits_and_disabled_features_print_no_rule_sentence():
    options = replace(Options(), recall=False, expectations=False)
    table, board = fixture("single", False, options)
    board.ledger.state = {"time": None}
    active = assert_rule_sentences(table, board, options)
    assert not active & {"budget", "time_limit", "step_limit", "recall_rows", "expectation", "facts_first", "order"}


def test_removing_the_prompt_rule_sentences_is_detected(monkeypatch):
    options = Options()
    table, board = fixture("rig", True, options)
    assert_rule_sentences(table, board, options)
    monkeypatch.setattr(core_protocol, "RULES", "The rule sentences were removed.")
    with pytest.raises(AssertionError):
        assert_rule_sentences(table, board, options)
