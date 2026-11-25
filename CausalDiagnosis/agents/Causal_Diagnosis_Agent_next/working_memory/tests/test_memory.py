"""Retained calls, note lifetimes and lossless compaction over public fixtures."""

import copy
import itertools
import json

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests.fixture_declarations import declarations
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.plan_schema import Step
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.tests.fakes import Ledger
from CausalDiagnosis.agents.utils import answer_forms, tool_answers
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import metering_receipts
from ..board import Board


def board(name="rig"):
    declaration, listing = declarations()[name]
    table = declaration.table(listing)
    return Board(table, Ledger(), "Keep the exact task.", Options())


def record(memory, tool, args=None, result=None, observations=None, refused=None):
    args = args or {}
    item = Step(tool, args, {}, [])
    answer = {"refused": refused} if refused else {"result": result if result is not None else {}}
    memory.record_call(item, ("server", tool, args), answer, observations)


def test_facts_knowledge_repeated_calls_and_refusals_have_distinct_status():
    memory = board()
    assert memory.reply_schema is None
    record(memory, "get_asset_profile", result={"facts": "public"})
    record(memory, "get_fix_history", {"component": "a"})
    record(memory, "get_fix_history", {"component": "a"})
    record(memory, "get_fix_history", {"component": "b"})
    record(memory, "get_failure_modes", {"component": "a"}, refused="not carried out")
    assert memory.facts == {"facts": "public"}
    assert memory.holds_answer("get_fix_history", {"component": "a"})
    assert not memory.holds_answer("get_failure_modes", {"component": "a"})
    assert memory.repeated() == [{"tool": "get_fix_history", "args": {"component": "a"}, "count": 2}]
    status = memory.to_json(())["status"]
    assert len(status["retained_answers"]) == 3
    assert status["repeated_answers"] == memory.repeated()
    assert memory.to_json(())["sections"] == list(memory._parts(()))
    assert memory.render(()).endswith(json.dumps(status, ensure_ascii=False))


def test_repeated_plans_and_clock_ticks_are_not_progress_but_new_observations_are():
    memory = board()
    for turn in range(1, 4):
        memory.plan = {"why": "Restate the plan.", "turn": turn}
        memory.ledger.state["steps_left"] = 20 - turn
        memory.ledger.state["time"] = turn
        memory.mark_turn(turn)
    assert memory.to_json(())["status"]["turns_without_progress"] == 2
    for turn in range(4, 8):
        record(memory, "read_sensors", {"sensor_group": "all"}, observations=[
            {"id": f"obs/{turn}/pressure", "n": turn, "field": "pressure", "value": 2}])
        memory.mark_turn(turn)
        assert "turns_without_progress" not in memory.to_json(())["status"]


@pytest.mark.parametrize("tie", ["facts", "candidates", "differential", ("advice", "ranking")])
def test_a_note_retires_only_when_its_own_state_passes(tie):
    memory = board()
    memory.add_note("A state has not arrived.", tie=tie)
    assert memory.retire_notes("another state") == 0 and len(memory.notes) == 1
    assert memory.retire_notes(tie) == 1 and not memory.notes


def test_equal_plan_notes_move_once_with_their_counts_and_retirement_keeps_chronology():
    memory = board()
    memory.mark_turn(1)
    memory.add_note("plan 1: the value is required", tie="facts", turn=1)
    memory.notifications.append({"event": "first"})
    memory.mark_turn(2)
    memory.add_note("plan 2: the value is required", tie="facts", turn=2)
    assert len(memory.notes) == 1 and memory.notes[0]["count"] == 2
    assert "earlier turns with the same reasons: 1" in memory.render(())
    memory.add_note("Keep a later note.")
    memory.retire_notes("facts")
    entries = memory.replaceable()
    assert [(entry["kind"], entry["turn"]) for entry in entries] == [("notifications", 1), ("notes", 2)]


def test_compaction_retains_evidence_refutations_and_scoped_one_turn_recall():
    memory = board()
    rows = [{"id": f"obs/{n}/pressure", "n": n, "series": "all", "field": "pressure", "value": n,
             "inside_band": n <= 3} for n in range(1, 5)]
    record(memory, "read_sensors", {"sensor_group": "all"}, observations=rows)
    memory.refuted = [{"answer": "earlier", "cites": [rows[0]["id"]]}]
    memory.mark_absorbed(4)
    before = copy.deepcopy(memory.ledger.__dict__)
    assert memory.compact(fits=lambda current: len(current.observations) <= 1)
    assert memory.ledger.__dict__ == before
    assert memory.known_ids() == {row["id"] for row in rows}
    assert memory.refuted == [{"answer": "earlier", "cites": [rows[0]["id"]]}]
    assert memory.recall("all", 1, 2) == 2
    assert not dict(memory._parts(()))["Recalled observations"]
    memory.mark_turn(1)
    assert len(dict(memory._parts(()))["Recalled observations"]) == 2
    memory.mark_turn(2)
    assert not dict(memory._parts(()))["Recalled observations"]


def test_a_summary_cited_identifier_pins_its_entry_and_without_is_a_copy():
    memory = board()
    memory._ids.add("obs/1/pressure")
    memory.add_note("The reading obs/1/pressure is outside.", tie="facts")
    memory.add_note("An expendable note.")
    entries = memory.replaceable()
    empty = memory.without(entries)
    assert not empty.notes and len(memory.notes) == 2
    memory.apply_summary("Keep obs/1/pressure.", entries, 0)
    assert len(memory.notes) == 1 and memory.retire_notes("facts") == 0
    memory.apply_summary("A later summary.", memory.replaceable(), 0)
    assert len(memory.notes) == 1 and memory.known_ids() == {"obs/1/pressure"}


def test_the_metering_layer_imports_the_one_receipt_reader_and_keeps_legacy_behavior():
    assert metering_receipts.split_receipt is tool_answers.split_receipt
    for text in ("plain", "body\n{\"steps_left\": 2}", "body\n{\"other\": 1}", "{\"steps_left\": 1}"):
        assert metering_receipts.split_receipt(text) == tool_answers.split_receipt(text)
    result = tool_answers.read('high pressure\n{"steps_left": 2, "cycle": 4}', False)
    assert result["result"] == "high pressure" and result["status"] == {"steps_left": 2, "time": 4}
    assert tool_answers.read("operation not carried out", False, ("not carried out",))["refused"]


def test_generated_answer_shapes_return_without_reaching_a_guard(capsys):
    odd = [None, True, False, [], {}, "", "text", -1, 1e300, "NaN", "１２", "--1", 10 ** 1000]
    base = {"result": {}, "status": {}, "notifications": [], "observations": []}
    count = 0
    for key, value in itertools.product(base, odd):
        result = tool_answers.read(json.dumps({**base, key: value}), False)
        assert set(result) >= {"status", "notifications", "observations"}
        assert ("result" in result) != ("refused" in result)
        assert not result.get("guard")
        count += 1
    for first, second in itertools.combinations(base, 2):
        for value in odd:
            result = tool_answers.read(json.dumps({**base, first: value, second: value}), False)
            assert ("result" in result) != ("refused" in result) and not result.get("guard")
            count += 1
    for value in odd:
        result = tool_answers.read(value, False)
        if not isinstance(value, str):
            assert "refused" in result
        assert not result.get("guard")
    with capsys.disabled():
        print(f"Generated tool answers checked: {count}")


def test_answer_forms_keep_records_ranges_and_whole_words_and_refuse_partial_records():
    rows, note = answer_forms.observations("records", {"served": [
        {"n": 1, "t": 2, "values": {"pressure": 4}}]}, {"pressure": (1, 3)})
    assert note is None and rows[0]["id"] == "obs/1/pressure" and rows[0]["inside_band"] is False
    assert answer_forms.observations("records", [{"n": True, "t": 1, "values": {}}], {})[1]
    memory = board("port")
    record(memory, "read_gauge", result="normal, not abnormally high")
    assert memory.observations[0]["words"] == ["normal", "high"]
    other = board("port")
    record(other, "read_gauge", result="abnormally higher")
    assert not other.observations[0]["words"]


def test_only_an_unexpected_answer_reader_failure_reaches_the_guard(monkeypatch):
    def broken(text):
        raise RuntimeError("reader failure")
    monkeypatch.setattr(tool_answers, "split_receipt", broken)
    result = tool_answers.read("body", False)
    assert result["refused"] == "the answer could not be read" and result["guard"] == "RuntimeError"


def test_successful_state_updates_retire_their_notes_and_refused_calls_are_counted():
    memory = board()
    memory.add_note("Await facts.", tie="facts")
    record(memory, "get_asset_profile", refused="not carried out")
    assert memory.notes
    record(memory, "get_asset_profile")
    assert not memory.notes and memory.repeated()[0]["count"] == 2
    memory.add_note("Await candidates.", tie="candidates")
    memory.model_candidates = [{"answer": "ready"}]
    assert not memory.notes
    memory.add_note("Await differential.", tie="differential")
    memory.differential = [{"about": "supply", "expected": "low"}]
    assert not memory.notes


def test_module_identifiers_join_retained_evidence_without_changing_its_ownership():
    class Module:
        def known_ids(self):
            return {"advice/read"}

    memory = board()
    memory._ids.add("obs/1/pressure")
    memory.set_modules((Module(),))
    assert memory.known_ids() == {"obs/1/pressure", "advice/read"}
    memory.set_modules(())
    assert memory.known_ids() == {"obs/1/pressure"}


def test_generated_observation_metadata_is_refused_before_it_changes_memory(capsys):
    sample = board("port")
    valid = {"id": "obs/reading", "n": 1, "t": 2, "text": "normal", "field": "pressure",
             "value": 2, "words": ["normal"], "inside_band": True}
    invalid = {
        "id": [None, True, 1, [], {}, "", " "],
        "n": [None, True, [], {}, "1", float("inf"), 10 ** 1000],
        "t": [None, True, [], {}, "1", float("nan"), 10 ** 1000],
        "text": [None, True, 1, [], {}],
        "field": [None, True, 1, [], {}, ""],
        "value": [None, True, [], {}, "1", float("inf"), 10 ** 1000],
        "words": [None, "normal", [1], [""], [{}]],
        "inside_band": [None, 1, "yes", [], {}],
    }
    mutations = [{key: value} for key, values in invalid.items() for value in values]
    mutations.extend({first: left, second: right}
                     for first, second in itertools.combinations(invalid, 2)
                     for left, right in zip(invalid[first], invalid[second]))
    for change in mutations:
        memory = Board(sample.table, Ledger(), "Inspect the public answer.", Options())
        rows = [{**valid, **change}]
        original = copy.deepcopy(rows)
        record(memory, "read_gauge", result="normal", observations=rows)
        assert not memory.holds_answer("read_gauge", {}) and not memory.observations, change
        assert memory.observation_count == 0 and not memory.known_ids(), change
        assert memory.calls[-1]["answer"]["refused"].startswith("read_gauge:"), change
        assert memory.notes[-1]["text"].startswith("read_gauge:"), change
        memory.mark_absorbed(1)
        memory.compact(fits=lambda current: not current.observations)
        memory.mark_turn(1)
        memory.render(())
        assert rows == original
    with capsys.disabled():
        print(f"Generated observation metadata checked: {len(mutations)}")
    record(sample, "read_gauge", result="normal", observations=[valid])
    assert sample.holds_answer("read_gauge", {}) and sample.observations[0]["id"] == valid["id"]


def test_missing_text_metadata_is_assigned_and_partial_numeric_rows_are_refused():
    memory = board("port")
    record(memory, "read_gauge", result="normal", observations=[{"text": "normal"}])
    assert memory.observations == [{"text": "normal", "n": 1, "id": "obs/1", "words": ["normal"]}]
    for rows in ([{}], [{"id": "obs/a", "n": 1, "field": "pressure"}],
                 [{"n": 1, "field": "pressure", "value": 2}], {"text": "normal"}):
        other = board("port")
        record(other, "read_gauge", result="normal", observations=rows)
        assert not other.holds_answer("read_gauge", {}) and not other.observations


def test_normalized_wire_rows_are_checked_and_removing_the_check_admits_bad_metadata(monkeypatch):
    wire = json.dumps({"result": "normal", "observations": [{"id": "obs/a", "n": [], "text": "normal"}]})
    normalized = tool_answers.read(wire, False)
    assert not normalized.get("guard") and "result" in normalized
    memory = board("port")
    item = Step("read_gauge", {}, {}, [])
    memory.record_call(item, ("server", item.tool, {}), normalized, None)
    assert not memory.observations and not memory.holds_answer(item.tool, {})
    assert "n must be a finite number" in memory.notes[-1]["text"]
    from .. import board as board_module
    monkeypatch.setattr(board_module, "validate_observations", lambda rows: None)
    control = board("port")
    control.record_call(item, ("server", item.tool, {}), normalized, None)
    assert control.holds_answer(item.tool, {}) and control.observations[0]["n"] == []


@pytest.mark.parametrize("place", ["result", "status", "notifications", "observations", "key", "receipt"])
def test_escaped_lone_surrogates_never_reach_the_board_or_record_encoder(place):
    broken = "\ud800"
    envelope = {"result": {}, "status": {}, "notifications": [], "observations": []}
    if place == "result":
        envelope["result"] = {"nested": [broken]}
    elif place == "status":
        envelope["status"] = {"detail": broken}
    elif place == "notifications":
        envelope["notifications"] = [{"event": broken}]
    elif place == "observations":
        envelope["observations"] = [{"id": "obs/a", "n": 1, "text": broken}]
    elif place == "key":
        envelope["result"] = {broken: "value"}
    wire = json.dumps(envelope)
    if place == "receipt":
        wire = "plain answer\n" + json.dumps({"steps_left": 2, "notifications": [{"detail": broken}]})
    normalized = tool_answers.read(wire, False)
    assert "refused" in normalized and not normalized.get("guard")
    json.dumps(normalized, ensure_ascii=False, allow_nan=False).encode("utf-8")
    memory = board()
    item = Step("get_asset_profile", {}, {}, [])
    memory.record_call(item, ("server", item.tool, {}), normalized, None)
    assert memory.facts is None and not memory.holds_answer(item.tool, {})
    memory.render(()).encode("utf-8")


def test_complete_unicode_pairs_are_kept_and_legacy_object_decoding_is_unchanged():
    wire = r'{"result": {"text": "\ud83d\ude00"}}'
    normalized = tool_answers.read(wire, False)
    assert normalized["result"]["text"] == "\U0001f600"
    json.dumps(normalized, ensure_ascii=False).encode("utf-8")
    legacy = r'{"result": {"text": "\ud800"}}'
    assert tool_answers.json_object(legacy)["result"]["text"] == "\ud800"
