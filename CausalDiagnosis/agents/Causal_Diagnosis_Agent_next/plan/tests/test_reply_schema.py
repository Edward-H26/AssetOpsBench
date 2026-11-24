"""Generate replies from every supported schema shape and its declared bounds."""

import copy
import json
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.tests.fixture_declarations import declarations
from ..expectations import expectation_forms, Form
from ..readers import read_number
from ..reply_parser import parse_usable_plan, parse_differential
from ..reply_schema import KEYWORDS, plan_schema_of, differential_schema, response_format, digest


def values(schema):
    """Walk every branch, optional property and bound without a product explosion."""
    if "const" in schema:
        return [schema["const"]]
    if "enum" in schema:
        return schema["enum"]
    if "anyOf" in schema:
        return [value for branch in schema["anyOf"] for value in values(branch)]
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        samples = {key: values(child) for key, child in properties.items()}
        required = {key: samples[key][0] for key in schema.get("required", ())}
        full = {key: sample[0] for key, sample in samples.items()}
        result = [required, full]
        for key, sample in samples.items():
            result.extend({**full, key: value} for value in sample)
            if key not in schema.get("required", ()):
                result.append({name: value for name, value in full.items() if name != key})
        return result
    if kind == "array":
        counts = {schema.get("minItems", 0), max(1, schema.get("minItems", 0))}
        if "maxItems" in schema:
            counts.add(schema["maxItems"])
            counts = {count for count in counts if count <= schema["maxItems"]}
        return [[copy.deepcopy(value) for _ in range(count)] for count in sorted(counts)
                for value in values(schema.get("items", {}))]
    if kind == "string":
        return ["v" * max(1, schema.get("minLength", 1))]
    if kind in ("integer", "number"):
        low = schema.get("minimum", 0)
        high = schema.get("maximum", max(1, low))
        return sorted({low, high})
    if kind == "boolean":
        return [False, True]
    return [None]


def check_keywords(schema):
    assert set(schema) <= KEYWORDS
    for child in schema.get("properties", {}).values():
        check_keywords(child)
    for child in schema.get("anyOf", ()):
        check_keywords(child)
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            check_keywords(schema[key])


class Module:
    def forms(self, row):
        return {"mass": Form(lambda value: read_number(value, 0, 1, "mass"),
                             {"type": "number", "minimum": 0, "maximum": 1})}


@pytest.mark.parametrize("modules", [(), (Module(),)])
def test_every_generated_reply_matches_its_schema_and_reads_without_a_reason(modules, capsys):
    count = 0
    for declaration, listing in declarations().values():
        table = declaration.table(listing)
        forms = expectation_forms(table, modules)
        for options in (Options(), Options(plan_length=2, expectations=False, recall=False)):
            schema = plan_schema_of(table, forms, options)
            check_keywords(schema)
            validator = Draft202012Validator(schema)
            for raw in values(schema):
                validator.validate(raw)
                parsed = parse_usable_plan(json.dumps(raw), table, forms, options.plan_length)
                assert parsed.refusal is None and parsed.unusable is None, (raw, parsed)
                count += 1
    for raw in values(differential_schema()):
        assert parse_differential(json.dumps(raw))[1] is None
        count += 1
    with capsys.disabled():
        print(f"Generated schema replies checked: {count}")


def test_a_schema_bound_below_the_reader_and_an_unknown_keyword_are_detected():
    declaration, listing = declarations()["single"]
    table = declaration.table(listing)
    schema = plan_schema_of(table, {}, Options())
    broken = copy.deepcopy(schema)
    broken["properties"]["why"]["minLength"] = 0
    raw = values(broken)[0]
    raw["why"] = ""
    Draft202012Validator(broken).validate(raw)
    assert parse_usable_plan(json.dumps(raw), table, {}, None).refusal
    with pytest.raises(AssertionError):
        check_keywords({**schema, "pattern": "unknown"})


def test_unsupported_argument_shapes_are_left_to_the_live_check_and_schema_records_are_copied():
    declaration, listing = declarations()["single"]
    table = declaration.table(listing)
    row = table.tools["answer"]
    shape = copy.deepcopy(row.schema)
    shape["properties"]["text"]["pattern"] = "^value$"
    changed = replace(table, tools={row.name: replace(row, schema=shape)})
    schema = plan_schema_of(changed, {}, Options())
    check_keywords(schema)
    wrapped = response_format("plan", schema)
    assert wrapped["json_schema"]["strict"] is True
    before = digest(schema)
    wrapped["json_schema"]["schema"].clear()
    assert digest(schema) == before
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.working_memory.board import Board
    from ..plan_check import check_plan
    from .fakes import Ledger

    ledger = Ledger()
    memory = Board(changed, ledger, "Read the text.", Options())
    for text, refused in (("wrong", True), ("value", False)):
        raw = {"why": "Read.", "steps": [{"tool": "answer", "args": {"text": text}}]}
        parsed = parse_usable_plan(json.dumps(raw), changed, {}, None)
        assert parsed.refusal is None
        assert bool(check_plan(parsed.plan, memory, ledger, changed, ())) is refused
