"""Typed replies over the official frozen declarations, without tool execution."""

import copy
import itertools
import json
import sys
from dataclasses import replace
from functools import partial

import pytest

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan import readers
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.plan_schema import PlanError
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.reply_parser import (
    last_json_object, parse_candidates, parse_differential, parse_usable_plan,
)
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests.fixture_declarations import declarations


def table(name="single"):
    declaration, listing = declarations()[name]
    return declaration.table(listing)


def read(data, tools=None, forms=None, limit=None):
    return parse_usable_plan(json.dumps(data), tools or table(), forms or {}, limit)


def reply(*steps, **head):
    return {"why": "Inspect the public evidence.", "steps": list(steps), **head}


def answer(value="supported"):
    return {"tool": "answer", "args": {"text": value}}


@pytest.mark.parametrize("value,expected", [(1, 1), (1.0, 1), ("12", 12), ("0" * 5000 + "1", 1)])
def test_counts_accept_integral_values_without_changing_the_frozen_signature(value, expected):
    assert readers.read_count(value, 1, "count") == (expected, None)


@pytest.mark.parametrize("value", [True, False, None, [], {}, "", " 1 ", "+1", "--1", "１２", "1.0",
                                   float("nan"), float("inf"), -1, 1.5, sys.maxsize + 1, "9" * 5000])
def test_counts_refuse_bad_values_by_argument_name(value):
    parsed, reason = readers.read_count(value, 0, "count")
    assert parsed is None and reason.startswith("count:")


@pytest.mark.parametrize("value", [True, "1", None, [], {}, float("inf"), float("nan"), 10 ** 1000])
def test_numbers_refuse_nonfinite_or_coerced_values(value):
    assert readers.read_number(value, 0, 1, "confidence")[1]
    assert readers.read_number(0.5, 0, 1, "confidence") == (0.5, None)
    assert readers.read_number(1.5, 0, 1, "confidence")[1]


def test_text_names_lists_and_objects_use_named_type_checks_and_copied_containers():
    assert readers.read_name([], ("allowed",), "tool")[1]
    assert readers.read_name("allowed", ("allowed",), "tool") == ("allowed", None)
    for value in (None, True, [], {}, "", " ", "x\ud800", "x\x01", "x\t"):
        assert readers.read_text(value, "text")[1]
    assert readers.read_text(" line\nnext ", "text") == ("line\nnext", None)
    source = {"nested": [1]}
    copied, reason = readers.read_object(source, "args")
    assert reason is None
    copied["nested"].append(2)
    assert source == {"nested": [1]}
    assert readers.read_list({}, "steps")[1] and readers.read_object([], "args")[1]


def test_json_extraction_handles_fences_quoted_braces_duplicates_and_truncation():
    assert last_json_object('Example {"a":1}\n```json\n{"why":"a } brace","steps":[]}\n```')["steps"] == []
    for text in ('{"a":1,"a":2}', '{"x":{"a":1,"a":2}}'):
        with pytest.raises(PlanError, match="duplicate"):
            last_json_object(text)
    with pytest.raises(PlanError, match="incomplete"):
        last_json_object('{"steps":[]} {"steps":[')
    with pytest.raises(PlanError):
        last_json_object('[{"steps":[]}]')


def test_required_why_and_invalid_heads_refuse_the_whole_reply():
    assert read({"steps": [answer()]}).refusal.reason.startswith("why:")
    assert read(reply(answer(), follow=True)).refusal
    assert read(reply(answer(), candidates=[{}])).refusal
    assert read(reply(answer(), answer()), limit=1).refusal
    assert read(reply(answer())).refusal is None


def test_a_bad_later_step_preserves_the_prefix_and_the_written_suffix_tool_names():
    parsed = read(reply(answer(), {"tool": "invented"}, {"tool": []}, answer("later")))
    assert parsed.refusal is None and len(parsed.plan.steps) == 1
    assert parsed.unusable.step == 2 and parsed.unusable.kinds == ("invented", "no tool", "answer")
    assert parsed.unusable.reasons and parsed.plan.raw
    assert read(reply({"tool": "invented"}, answer())).refusal.fault is None


def test_a_step_cannot_name_an_advice_tool_even_when_it_is_in_the_joined_table():
    tools = table("rig")
    advice = next(row.name for row in tools.tools.values() if row.role == "advice")
    parsed = read(reply({"tool": advice}), tools)
    assert parsed.refusal is not None and parsed.refusal.fault is None


def test_candidate_ranks_and_fixed_defaults_follow_the_new_records():
    tools = table()
    row = tools.tools["answer"]
    tools = replace(tools, tools={row.name: replace(row, omitted={"text": "fixed:ready"})})
    parsed = read(reply({"tool": "answer"}, candidates=[{}]), tools)
    assert parsed.plan.candidates == [{"text": "ready", "cites": []}]
    assert parsed.plan.steps[0].candidate == 1 and parsed.plan.steps[0].args == {"text": "ready"}
    assert parsed.plan.steps[0].to_json()["candidate"] == 1
    assert read(reply({"tool": "answer", "args": {"text": "ready", "candidate": 2}}), tools).refusal
    plain = replace(tools, tools={row.name: replace(row, candidate=(), omitted={})})
    assert "text is required" in read(reply({"tool": "answer"}), plain).refusal.reason
    assert read(reply(answer(True))).refusal


def test_facts_dependent_omissions_are_not_replaced_by_invented_values():
    parsed = read(reply({"tool": "submit_diagnosis"}, candidates=[{}]), table("rig"))
    assert parsed.plan.steps[0].candidate == 1 and parsed.plan.candidates == [{"cites": []}]
    assert parsed.plan.steps[0].args == {}
    assert "healthy." not in json.dumps(parsed.plan.to_json())


def test_model_candidates_use_the_declared_argument_types_and_copy_the_input():
    raw = [{"text": "ready", "cites": ["obs/1"]}]
    assert parse_candidates(raw, table()) == raw
    assert raw[0]["cites"] == ["obs/1"]
    for value in ([{}], [{"text": True}], [{"invented": 1}]):
        with pytest.raises(PlanError):
            parse_candidates(value, table())


def evidence_table():
    tools = table()
    row = tools.tools["answer"]
    schema = {"type": "object", "properties": {"text": {"type": "string"}, "receipt": {"type": "array"}},
              "required": ["text", "receipt"]}
    return replace(tools, tools={row.name: replace(row, fills={"receipt": "evidence"}, schema=schema)})


def test_evidence_is_read_only_for_a_declared_fill_and_written_fills_wait_for_the_live_check():
    for value in (None, 9, {}, False, "not a list"):
        assert read(reply({**answer(), "evidence": value})).plan.steps[0].evidence == []
        assert read(reply({**answer(), "evidence": value}), evidence_table()).refusal
    parsed = read(reply({"tool": "answer", "args": {"text": "ready", "receipt": {"invented": 1}},
                         "evidence": ["obs/1"]}), evidence_table())
    assert parsed.plan.steps[0].evidence == ["obs/1"]
    assert parsed.plan.steps[0].args["receipt"] == {"invented": 1}


def test_waits_and_empty_each_arguments_normalize_before_the_live_check():
    tools = table("rig")
    for args in ({}, {"timeout": 0}, {"timeout": "0"}):
        parsed = read(reply({"tool": "wait_for_notification", "args": args}), tools)
        assert parsed.plan.steps[0].args["timeout"] == 1
    assert read(reply({"tool": "wait_for_notification", "args": {"timeout": -1}}), tools).refusal
    for empty in (None, ""):
        parsed = read(reply({"tool": "get_failure_modes", "args": {"component": empty}}), tools)
        assert parsed.plan.steps[0].args == {}
    assert read(reply({"tool": "get_failure_modes", "args": {"component": True}}), tools).refusal


def test_expectation_readers_normalize_values_and_unknown_keys_are_refused():
    forms = {"answer": {"rank": partial(readers.read_count, minimum=1, name="expect.rank")}}
    parsed = read(reply({**answer(), "expect": {"rank": "2"}}), forms=forms)
    assert parsed.plan.steps[0].expect == {"rank": 2}
    assert read(reply({**answer(), "expect": {"rank": True}}), forms=forms).refusal
    assert read(reply({**answer(), "expect": {"other": 1}}), forms=forms).refusal


def test_recall_and_orientation_use_the_current_generic_shapes():
    parsed = read(reply({"tool": "recall", "args": {"observations": ["1", 2.0], "series": "a"}}))
    assert parsed.plan.steps[0].args == {"observations": [1, 2], "series": "a"}
    assert read(reply({"tool": "recall", "args": {"observations": [2, 1]}})).refusal
    body = {"differential": [{"about": "supply", "expected": "low output", "cites": ["manual/1"]}]}
    entries, refusal = parse_differential(json.dumps(body))
    assert refusal is None and entries == body["differential"]
    body["differential"][0]["probability"] = 0.5
    assert parse_differential(json.dumps(body))[1].fault is None


def test_an_unexpected_form_failure_is_recorded_as_a_reader_fault():
    def broken(value):
        raise RuntimeError("broken form")
    result = read(reply({**answer(), "expect": {"custom": 1}}), forms={"answer": {"custom": broken}})
    assert result.refusal.reason == "the reply could not be read" and result.refusal.fault == "RuntimeError"


def sample(schema):
    if not isinstance(schema, dict):
        return None
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema:
        return sample(schema["anyOf"][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        return sample({**schema, "type": kind[0]})
    if kind == "string":
        return "value"
    if kind in ("integer", "number"):
        return max(1, schema.get("minimum", 1))
    if kind == "array":
        return [sample(schema.get("items", {})) for _ in range(schema.get("minItems", 0))]
    if kind == "object":
        return {key: sample(schema["properties"][key]) for key in schema.get("required", ())}
    return True if kind == "boolean" else None


def positions(value, prefix=()):
    yield prefix
    if isinstance(value, dict):
        for key, item in value.items():
            yield from positions(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from positions(item, (*prefix, index))


def replace_at(value, path, replacement):
    if not path:
        return replacement
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    return value


def test_generated_replies_at_each_position_and_in_pairs_never_reach_the_fault_guard(capsys):
    odd = (None, True, False, 0, -1, 1.25, [], {}, "", " ", "--1", "１２", "9" * 100,
           10 ** 1000, 1e300, float("nan"), float("inf"), "\ud800", "text\x01")
    count = 0
    for name in declarations():
        tools = table(name)
        for row in tools.tools.values():
            args = {key: sample(shape) for key, shape in row.schema.get("properties", {}).items()
                    if key not in row.fills and key not in row.omitted and key != row.each}
            base = reply({"tool": row.name, "args": args, "expect": {}, "evidence": ["obs/1"]}, follow=None, candidates=[])
            if row.role == "advice":
                assert read(base, tools).refusal
                continue
            assert read(base, tools).plan, (name, row.name, read(base, tools).refusal)
            paths = tuple(positions(base))
            for path, value in itertools.product(paths, odd):
                parsed = read(replace_at(copy.deepcopy(base), path, value), tools)
                assert parsed.refusal is None or parsed.refusal.fault is None, (name, row.name, path, parsed)
                count += 1
            for index, pair in enumerate(itertools.combinations(paths, 2)):
                if pair[0] == pair[1][:len(pair[0])] or pair[1] == pair[0][:len(pair[1])]:
                    continue
                value = replace_at(copy.deepcopy(base), pair[0], odd[index % len(odd)])
                value = replace_at(value, pair[1], odd[(index + 1) % len(odd)])
                parsed = read(value, tools)
                assert parsed.refusal is None or parsed.refusal.fault is None
                count += 1
    with capsys.disabled():
        print(f"Generated declared replies checked: {count}")


def test_skipping_a_type_check_makes_a_generated_reply_reach_the_guard(monkeypatch):
    generated = [reply(value) for value in ([], None, True, "wrong type")]
    assert all(read(value).refusal.fault is None for value in generated)
    monkeypatch.setattr(readers, "read_object", lambda value, name: (value, None))
    assert any(read(value).refusal.fault is not None for value in generated)


def test_decoder_limits_nonfinite_numbers_and_surrogates_have_named_refusals():
    depth = sys.getrecursionlimit() + 1
    inputs = [None, [], 1, '{"steps":' + '[' * depth + '0' + ']' * depth + '}',
              '{"steps":[],"follow":' + '9' * (sys.get_int_max_str_digits() + 1) + '}',
              '{"why":1e9999}', '{"why":NaN}', '{"why":"\\ud800"}', '{"why":"x\x01"}']
    for text in inputs:
        parsed = parse_usable_plan(text, table(), {}, None)
        assert parsed.refusal and parsed.refusal.fault is None
        entries, refusal = parse_differential(text)
        assert not entries and refusal and refusal.fault is None
