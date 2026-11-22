"""The schema of a reply (plan/reply_schema.py): per row the variants and the modes it serves, the keywords and the bounds it
uses, its property against the reader (every reply that the schema takes, within the reader's domain, is read with no reason),
the values outside that domain that the reader refuses, and the request field in the forms that SGLang and vLLM document."""
import json
import re

import jsonschema
import pytest
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import plan_schema, reply_schema
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.reply_parser import FINISH_LAST, parse_differential, parse_usable_plan


# the keywords that the xgrammar compiler of both engines handles; it drops not, if, then, else, dependentRequired, multipleOf,
# uniqueItems and contains without a message and does not enforce them, and vLLM sends a schema with multipleOf, uniqueItems,
# contains, format, pattern, propertyNames, patternProperties or allOf away from xgrammar
HANDLED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "const", "enum", "anyOf", "items", "minItems", "maxItems",
                    "minLength", "minimum", "maximum"}


def _row_schema(unserved=frozenset(), opening=plan_schema.SURFACE_KINDS, recall=True, plan_length=None):
    return reply_schema.plan_schema_of(reply_schema.served_kinds(unserved, opening, recall), reply_schema.served_modes(unserved), plan_length)


def _variants(schema):
    return [v["properties"]["kind"]["const"] for v in schema["properties"]["steps"]["items"]["anyOf"]]


def _nodes(schema, path=()):
    """Every schema node with its path, the properties and the variants walked as schemas."""
    yield path, schema
    for key, value in schema.items():
        if key == "properties":
            for name, sub in value.items():
                yield from _nodes(sub, (*path, name))
        elif key == "anyOf":
            for i, sub in enumerate(value):
                yield from _nodes(sub, (*path, i))
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            yield from _nodes(value, (*path, key))


# ----- per row -----

@pytest.mark.parametrize("unserved", [frozenset(), frozenset({"declare"}), frozenset({"wait", "time"})])
def test_a_row_schema_holds_a_variant_and_a_mode_only_for_what_the_row_serves(unserved):
    schema = _row_schema(unserved, opening=tuple(k for k in plan_schema.SURFACE_KINDS if k not in unserved))
    assert set(_variants(schema)) == set(plan_schema.KINDS) - unserved, _variants(schema)
    assert _variants(schema) == [k for k in plan_schema.KINDS if k in _variants(schema)]  # in the order of KINDS
    assert schema["properties"]["mode"]["enum"] == [m for m in plan_schema.MODES if m not in unserved]


def test_a_row_schema_leaves_out_the_reads_the_surface_does_not_need_and_the_recall_with_the_switch_off():
    rig = _row_schema(opening=())
    assert not {"time", "extent"} & set(_variants(rig)) and "recall" in _variants(rig)
    assert "recall" not in _variants(_row_schema(recall=False))
    assert "maxItems" not in _row_schema()["properties"]["steps"]
    assert _row_schema(plan_length=6)["properties"]["steps"]["maxItems"] == 6


def test_the_top_of_a_plan_puts_the_free_text_first_and_a_step_its_kind_first():
    schema = _row_schema()
    assert list(schema["properties"]) == ["why", "branch", "mode", "candidates", "steps"]
    assert schema["required"] == ["why", "mode", "steps"] and schema["additionalProperties"] is False
    for variant in schema["properties"]["steps"]["items"]["anyOf"]:
        assert list(variant["properties"])[0] == "kind"
        assert ("evidence" in variant["properties"]) == (variant["properties"]["kind"]["const"] in plan_schema.EVIDENCE_KINDS)


# ----- what the schema holds and what it never holds -----

def _keys_of_the_form():
    keys = (set(plan_schema.HEAD_FORM) | set(plan_schema.CANDIDATE_FORM) | set(plan_schema.DIFFERENTIAL_FORM)
            | {"kind", "args", "expect", "evidence", "differential"})
    for form in plan_schema.STEP_FORM.values():
        keys |= set(form["args"]) | set(form["expect"])
    return keys


def _bounds_and_names(schema):
    """The keywords, the property names, the enums and constants, and the maxima of a schema."""
    keywords, names, fixed, maxima = set(), set(), [], []
    for path, node in _nodes(schema):
        keywords |= set(node)
        names |= set(node.get("properties") or {})
        fixed += list(node.get("enum") or []) + ([node["const"]] if "const" in node else [])
        maxima += [(path, key, node[key]) for key in ("maximum", "maxItems") if key in node]
    return keywords, names, fixed, maxima


def _probability_span_or_count(path, key, value, plan_length):
    """The only maxima the reader sets: the ends of a probability, the two items of a span and the count of steps of the run."""
    return ((key, value) == ("maximum", 1) or (key, value) == ("maxItems", 2) and path[-1] == "observations"
            or (key, value) == ("maxItems", plan_length) and path == ("steps",))


@pytest.mark.parametrize("plan_length", [None, 6])
def test_the_schema_names_no_word_of_the_asset_and_no_bound_that_the_reader_does_not_set(plan_length):
    for schema in (_row_schema(plan_length=plan_length), reply_schema.differential_schema()):
        keywords, names, fixed, maxima = _bounds_and_names(schema)
        assert names <= _keys_of_the_form(), names - _keys_of_the_form()
        assert set(fixed) <= set(plan_schema.KINDS) | set(plan_schema.MODES), fixed
        assert all(_probability_span_or_count(path, key, value, plan_length) for path, key, value in maxima), maxima
        assert keywords <= HANDLED_KEYWORDS, keywords - HANDLED_KEYWORDS


def test_the_walk_of_the_schema_finds_a_planted_bound_and_a_planted_keyword():
    """The control of the test above: a maximum that no reader sets, and a keyword that the compiler drops without a message."""
    schema = _row_schema()
    schema["properties"]["steps"]["items"]["anyOf"][0]["properties"]["args"]["properties"]["cycles"]["maximum"] = 100
    _keywords, _names, _fixed, maxima = _bounds_and_names(schema)
    assert not all(_probability_span_or_count(path, key, value, None) for path, key, value in maxima)
    schema["properties"]["why"]["not"] = {"const": ""}
    assert not _bounds_and_names(schema)[0] <= HANDLED_KEYWORDS


def test_the_schema_requires_why_and_the_reader_still_reads_a_plan_without_it():
    """Under a grammar an optional first key may be left out, so the schema requires why; the reader stays the one way in and reads a
    plan without why as before, as it reads every reply of a run with the schema off."""
    plan = {"mode": "test_first", "steps": [{"kind": "read", "args": {"sensor_group": "g", "cycles": 1}}]}
    with pytest.raises(jsonschema.ValidationError, match="'why' is a required property"):
        jsonschema.validate(plan, _row_schema())
    jsonschema.validate({"why": "", **plan}, _row_schema())
    parsed, unusable = parse_usable_plan(json.dumps(plan), None)
    assert unusable is None and parsed.why == "" and [s.kind for s in parsed.steps] == ["read"]


def test_a_row_has_one_digest_that_the_order_of_its_properties_moves():
    assert reply_schema.digest(_row_schema()) == reply_schema.digest(_row_schema())
    assert re.fullmatch(r"[0-9a-f]{16}", reply_schema.digest(_row_schema()))
    shuffled = {**_row_schema(), "properties": dict(sorted(_row_schema()["properties"].items()))}
    assert reply_schema.digest(shuffled) != reply_schema.digest(_row_schema())


# ----- the property: what the schema takes, the reader reads -----

LARGEST = plan_schema.LARGEST_INTEGER
TEXTS = ("x", " step a ", "fmea/source/code", "\xfc")  # texts that hold a visible character


def _samples(schema):
    """Values that the schema takes and that together cover every variant, every optional key on and off, the bounds and the
    values next to them, with texts that hold a visible character and integers up to the largest that the reader takes."""
    if "anyOf" in schema:
        return [value for variant in schema["anyOf"] for value in _samples(variant)]
    if "const" in schema:
        return [schema["const"]]
    if "enum" in schema:
        return list(schema["enum"])
    kind = schema["type"]
    if kind == "string":
        return list(TEXTS) + ([""] if not schema.get("minLength") else [])
    if kind == "integer":
        least = schema.get("minimum", -LARGEST)
        return [least, least + 1, LARGEST]
    if kind == "number":
        if "maximum" in schema:
            return [schema["minimum"], (schema["minimum"] + schema["maximum"]) / 2, schema["maximum"]]
        return [-1.5, 0, 7, 1e300, -1e300]
    if kind == "boolean":
        return [True, False]
    if kind == "null":
        return [None]
    if kind == "array":
        items = _samples(schema["items"])
        least, most = schema.get("minItems", 0), schema.get("maxItems")
        lengths = sorted({n for n in (least, least + 1, len(items), most) if n is not None and n >= least and (most is None or n <= most)})
        return [[items[(start + i) % len(items)] for i in range(n)] for n in lengths for start in range(0, len(items), max(n, 1))]
    properties, required = schema.get("properties") or {}, set(schema.get("required") or ())
    if not properties:  # the levels of a candidate: keys of its own, each a level
        return [{}, {f"c{i}": value for i, value in enumerate(_samples(schema["additionalProperties"]))}]
    each = {name: _samples(sub) for name, sub in properties.items()}
    whole = [{name: values[j % len(values)] for name, values in each.items()} for j in range(max(len(v) for v in each.values()))]
    return whole + [{name: values[0] for name, values in each.items() if name in required}]


def _in_the_readers_domain(plan):
    """A plan of the schema with its finish moved to the end and its spans in order, the two forms that the schema cannot state."""
    steps = [s for s in plan["steps"] if s["kind"] != "finish"]
    if len(steps) < len(plan["steps"]):
        steps.append({"kind": "finish"})
    for step in steps:
        if "observations" in (step.get("args") or {}):
            step["args"]["observations"] = sorted(step["args"]["observations"])
    return {**plan, "steps": steps}


def _plans(schema):
    """The plans of the schema's own samples, and every sample of every step variant alone in a plan."""
    alone = [{"why": TEXTS[0], "mode": schema["properties"]["mode"]["enum"][0], "steps": [step]} for step in _samples(schema["properties"]["steps"]["items"])]
    return [_in_the_readers_domain(plan) for plan in _samples(schema) + alone]


@pytest.mark.parametrize("plan_length", [None, 3])
def test_every_plan_that_the_schema_takes_in_the_readers_domain_is_read_with_no_reason(plan_length):
    schema = _row_schema(plan_length=plan_length)
    plans = _plans(schema)
    print(f"{len(plans)} generated plans, plan_length {plan_length}")
    assert len(plans) > 50
    kinds = set()
    for plan in plans:
        jsonschema.validate(plan, schema)  # the generator stays inside the schema
        parsed, unusable = parse_usable_plan(json.dumps(plan), plan_length)
        assert unusable is None, (plan, unusable)
        assert len(parsed.steps) == len(plan["steps"])
        kinds |= {s.kind for s in parsed.steps}
    assert kinds == set(_variants(schema))


def test_a_lower_bound_of_the_form_below_the_readers_gives_a_generated_plan_a_reason(monkeypatch):
    """The control of the property: a form that takes what the reader refuses fails it."""
    monkeypatch.setitem(plan_schema.STEP_FORM["read"]["args"], "cycles", (plan_schema.count(0), False))
    assert any(parse_usable_plan(json.dumps(plan), None)[1] is not None for plan in _plans(_row_schema()))


READ = {"kind": "read", "args": {"sensor_group": "g", "cycles": 1}}
HUGE = "@huge@"  # where a number past the largest float stands in the text of a reply, which no float writes
# values that the schema takes and the reader refuses, each with a fragment of the reason of the reader, which cuts the plan there
OUTSIDE_THE_DOMAIN = {
    "a text of blanks alone": ({**READ, "args": {"sensor_group": "   ", "cycles": 1}}, "must be a non empty string"),
    "an integer above the largest": ({**READ, "args": {"sensor_group": "g", "cycles": LARGEST + 1}}, "must be an integer of at most"),
    "a number past the largest float": ({**READ, "expect": {"effective_at_most": HUGE}}, "must be a finite number"),
    "a span whose first comes after its last": ({"kind": "recall", "args": {"sensor_group": "g", "observations": [5, 2]}},
                                                "observations must be [first, last]"),
    "a finish before the last step": ({"kind": "finish"}, FINISH_LAST),
}


@pytest.mark.parametrize("row", sorted(OUTSIDE_THE_DOMAIN))
def test_a_value_that_the_schema_takes_outside_the_readers_domain_is_refused_with_its_reason_and_cuts_the_plan(row):
    step, reason = OUTSIDE_THE_DOMAIN[row]
    plan = {"why": "x", "mode": "test_first", "steps": [READ, step, READ]}
    text = json.dumps(plan).replace(json.dumps(HUGE), "1e400")
    # the schema takes the reply: a number of any size is a number of the schema, which a float of the test stands for here
    jsonschema.validate(json.loads(json.dumps(plan).replace(json.dumps(HUGE), "1e300")), _row_schema())
    parsed, unusable = parse_usable_plan(text, None)
    assert [s.kind for s in parsed.steps] == ["read"] and unusable is not None and unusable.step == 2, (parsed, unusable)
    assert any(reason in r for r in unusable.reasons), unusable.reasons


def test_every_differential_that_the_schema_takes_is_read():
    schema = reply_schema.differential_schema()
    replies = _samples(schema)
    print(f"{len(replies)} generated differentials")
    assert len(replies) > 3
    for reply in replies:
        jsonschema.validate(reply, schema)
        assert len(parse_differential(json.dumps(reply))) == len(reply["differential"])


# ----- the request field, in the forms that the two engines document -----

class SGLangJsonSchema(BaseModel):
    """JsonSchemaResponseFormat of SGLang v0.5.19 (python/sglang/srt/entrypoints/openai/protocol.py), as released."""
    name: str
    description: str | None = None
    schema_: dict[str, object] | None = Field(alias="schema", default=None)
    strict: StrictBool | None = None


class SGLangResponseFormat(BaseModel):
    type: str
    json_schema: SGLangJsonSchema | None = None


class VLLMJsonSchema(BaseModel):
    """JsonSchemaResponseFormat of vLLM v0.34.0 (vllm/entrypoints/generate/base/protocol.py), as released; its base model takes
    extra fields."""
    model_config = ConfigDict(extra="allow")
    name: str
    description: str | None = None
    json_schema: dict | None = Field(default=None, alias="schema")
    strict: bool | None = None


class VLLMResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    json_schema: VLLMJsonSchema | None = None


@pytest.mark.parametrize("engine", [(SGLangResponseFormat, "schema_"), (VLLMResponseFormat, "json_schema")], ids=["sglang", "vllm"])
@pytest.mark.parametrize("name", [reply_schema.PLAN, reply_schema.DIFFERENTIAL])
def test_the_request_field_is_the_form_that_both_engines_document(engine, name):
    """The documented form only: whether a served engine compiles the schema, decodes the keys in its order and keeps the prefix
    cache is a probe on a served model."""
    model, field = engine
    schema = _row_schema() if name == reply_schema.PLAN else reply_schema.differential_schema()
    sent = json.loads(json.dumps(reply_schema.response_format(name, schema)))
    parsed = model.model_validate(sent)
    assert parsed.type == "json_schema" and parsed.json_schema.name == name and parsed.json_schema.strict is True
    assert getattr(parsed.json_schema, field) == schema
    assert list(sent["json_schema"]["schema"]["properties"]) == list(schema["properties"])


# ----- the request field through the driver -----

def _driver(tmp_path, monkeypatch, planner, *flags, endpoint=None):
    """One episode of the driver's command on the hydraulic rig with the scripted planner behind a local endpoint; returns the run
    folder, the episode folder and the bodies of the chat requests."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import _planner_endpoint
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    server, requests = endpoint or _planner_endpoint(planner)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    try:
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url",
                           f"http://127.0.0.1:{server.server_address[1]}/v1/", "--seeds", "1", "--out", str(tmp_path / "run"), *flags])
    finally:
        server.shutdown()
    run_dir = next(p for p in (tmp_path / "run").iterdir() if p.is_dir())
    return run_dir, next(run_dir.glob("seed*")), [body for path, body in requests if path.rstrip("/").endswith("chat/completions")]


def _turn(body):
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import ORIENTATION_HEADING, PLANNING_HEADING, SUMMARY_HEADING
    prompt = body["messages"][-1]["content"]
    return next((name for heading, name in ((ORIENTATION_HEADING, "orientation"), (SUMMARY_HEADING, "summary"), (PLANNING_HEADING, "planning"))
                 if heading in prompt), "earlier turns")


def _planner():
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ScriptedPlanner, act_now
    return ScriptedPlanner(act_now)


def test_the_planning_and_the_orientation_requests_carry_the_schema_of_the_row_and_the_summaries_none(tmp_path, monkeypatch):
    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS
    run_dir, ep_dir, bodies = _driver(tmp_path, monkeypatch, _planner())
    row = _row_schema(opening=())  # the rig's action server serves every kind and needs no read before a reading
    carried = {}
    for body in bodies:
        carried.setdefault(_turn(body), []).append(body.get("response_format"))
    assert carried["planning"] and all(f == reply_schema.response_format(reply_schema.PLAN, row) for f in carried["planning"])
    assert carried["orientation"] == [reply_schema.response_format(reply_schema.DIFFERENTIAL, reply_schema.differential_schema())]
    assert set(carried["summary"]) == {None} and set(carried.get("earlier turns", [None])) == {None}
    first = next(b for b in bodies if b.get("response_format"))["response_format"]["json_schema"]["schema"]
    assert list(first["properties"]) == ["why", "branch", "mode", "candidates", "steps"]  # the order arrives as the schema states it
    plans = json.loads((ep_dir / "plans.json").read_text(encoding="utf-8"))
    assert plans["board"]["reply_schema"] == {"plan": reply_schema.digest(row), "differential": reply_schema.digest(reply_schema.differential_schema()),
                                              "bound": len(carried["planning"]) + len(carried["orientation"]), "carried": True}
    assert (run_dir / "scores.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == list(COLUMNS)
    assert "structured_output" not in json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["causal_options"]


def test_without_the_pass_through_of_the_recording_backend_no_request_carries_the_schema(tmp_path, monkeypatch):
    """The control of the test above: the driver's wrapper that does not hand the field to its inner backend sends none."""
    from CausalDiagnosis.experiments.episode_runners.run_episodes import model_backends
    forward = model_backends.RecordingBackend.generate_with_usage
    monkeypatch.setattr(model_backends.RecordingBackend, "generate_with_usage",
                        lambda self, prompt, temperature=0.0, response_format=None: forward(self, prompt, temperature))
    _run_dir, _ep_dir, bodies = _driver(tmp_path, monkeypatch, _planner())
    assert bodies and not any("response_format" in body for body in bodies)


def test_the_driver_flag_plays_the_agent_with_the_schema_off(tmp_path, monkeypatch):
    run_dir, ep_dir, bodies = _driver(tmp_path, monkeypatch, _planner(), "--causal-no-structured-output")
    assert bodies and not any("response_format" in body for body in bodies)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["causal_options"]["structured_output"] is False and run_dir.name.startswith("static_causal_noschema_")
    plans = json.loads((ep_dir / "plans.json").read_text(encoding="utf-8"))
    assert plans["board"]["reply_schema"]["bound"] == 0 and plans["board"]["reply_schema"]["carried"] is True


def _refusing_endpoint(message):
    """A stand-in endpoint that refuses with 400 every chat request that carries a response format, as an engine without a grammar
    backend or with a schema it cannot compile does, and answers any other as a scripted planner."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import answer_context_route
    planner, requests = _planner(), []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the http.server hook name
            if not answer_context_route(self, None):
                self.send_error(404)

        def do_POST(self):  # noqa: N802 - the http.server hook name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if answer_context_route(self, body):
                return
            requests.append((self.path, body))
            refused = "response_format" in body
            reply = ({"object": "error", "message": message, "type": "BadRequestError", "code": 400} if refused else
                     {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                      "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": planner.generate(body["messages"][-1]["content"])}}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
            data = json.dumps(reply).encode()
            self.send_response(400 if refused else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def test_an_engine_that_refuses_the_field_ends_the_episode_as_an_error_with_its_message_and_nothing_is_sent_again_without_it(tmp_path, monkeypatch):
    message = "Grammar-based generation is not supported when the server is launched with --grammar-backend none"
    _run_dir, ep_dir, bodies = _driver(tmp_path, monkeypatch, None, endpoint=_refusing_endpoint(message))
    plans = json.loads((ep_dir / "plans.json").read_text(encoding="utf-8"))
    assert plans["error"].startswith("BadRequestError") and message in plans["error"]
    assert bodies and all("response_format" in body for body in bodies)  # no request was sent again without the field


# ----- the evidence of a step whose call records none -----

ODD_EVIDENCE = ('"x"', "5", "0.5", "true", "null", "[]", "[1]", "{}", '{"x": 1}', '[" "]')


@pytest.mark.parametrize("value", ODD_EVIDENCE)
def test_any_value_under_evidence_of_a_step_whose_call_records_none_cuts_no_plan(value):
    """The reader reads the key only where the call records evidence; the control is the same value on a declaration, whose call
    records evidence, which the reader still refuses where it is no list of identifier strings."""
    read = '{"kind": "read", "args": {"sensor_group": "g", "cycles": 1}, "evidence": VALUE}'.replace("VALUE", value)
    parsed, unusable = parse_usable_plan('{"mode": "test_first", "steps": [READ, {"kind": "finish"}]}'.replace("READ", read), None)
    assert unusable is None and parsed.steps[0].evidence == []
    declare = '{"kind": "declare", "args": {"sensor_group": "g", "cycles": 1}, "evidence": VALUE}'.replace("VALUE", value)
    parsed, unusable = parse_usable_plan('{"mode": "declare", "steps": [DECLARE]}'.replace("DECLARE", declare), None)
    # a value that reads as no evidence (null, an empty list or object) or as blank identifiers only, which are dropped, passes
    assert (unusable is None) == (value in ("null", "[]", "{}", '[" "]')), (value, unusable)
    if unusable is not None:
        assert unusable.reasons == ["step 1: evidence must be a list of identifier strings"]


# ----- the static prefix of the requests, which the prefix cache of the engine computes once -----

BOARD_HEAD = "# Situation board"


def _prefixes(prompts):
    """The system text with the tool texts, every request's head before its turn heading, and the static prefix of every planning
    request, everything before the board."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import PLANNING_HEADING
    system = {p.split("\n\n## ", 1)[0] for p in prompts}
    planning = {p[:p.index(BOARD_HEAD)] for p in prompts if PLANNING_HEADING in p}
    return system, planning


def _episode_prompts(tmp_path, surface):
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (ScriptedPlanner, _aob_run, _play, act_now,
                                                                                               act_now_on_assetopsbench)
    if surface == "rig":
        planner = ScriptedPlanner(act_now)
        _play(tmp_path, planner, {})
    else:
        planner = ScriptedPlanner(act_now_on_assetopsbench)
        _aob_run(tmp_path, "base", planner, {})
    return planner.prompts


@pytest.mark.parametrize("surface", ["rig", "aob"])
def test_the_system_text_and_the_tool_texts_are_byte_identical_in_every_request_of_an_episode(tmp_path, surface):
    prompts = _episode_prompts(tmp_path, surface)
    system, planning = _prefixes(prompts)
    assert len(prompts) > 3 and len(system) == 1 and len(planning) == 1, (len(system), len(planning))


def test_a_tool_text_that_changes_with_the_profile_breaks_the_prefix(tmp_path, monkeypatch):
    """The control of the test above: the clock sentence of the AssetOpsBench surface written only once the profile is read."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.causal_diagnosis_agent_surface import AobSurface
    stated = AobSurface.prompt_tools
    monkeypatch.setattr(AobSurface, "prompt_tools", property(lambda self: "" if self.site is None else stated.fget(self)))
    system, _planning = _prefixes(_episode_prompts(tmp_path, "aob"))
    assert len(system) == 2


class _WithoutWhy:
    """A scripted planner whose plans hold no why, as a model that decodes without the schema may write them."""

    def __init__(self, planner):
        self.planner = planner

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def generate(self, prompt, temperature=0.0):
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import PLANNING_HEADING
        reply = self.planner.generate(prompt, temperature)
        if PLANNING_HEADING not in prompt:
            return reply
        plan = json.loads(reply)
        plan.pop("why", None)
        return json.dumps(plan)


def _plays_without_why(tmp_path, monkeypatch):
    import csv
    run_dir, ep_dir, bodies = _driver(tmp_path, monkeypatch, _WithoutWhy(_planner()), "--causal-no-structured-output")
    plans = json.loads((ep_dir / "plans.json").read_text(encoding="utf-8"))["plans"]
    with open(run_dir / "scores.csv", encoding="utf-8", newline="") as f:
        return bodies, plans, next(csv.DictReader(f))


def test_with_the_schema_off_a_plan_without_why_is_read_and_the_episode_plays_to_its_end(tmp_path, monkeypatch):
    """Ag10: the schema requires why, while the reader reads a reply without it, so a run with the schema off plays as before."""
    bodies, plans, row = _plays_without_why(tmp_path, monkeypatch)
    assert bodies and not any("response_format" in body for body in bodies)
    assert plans and all(p["source"] == "model" and p["plan"]["why"] == "" for p in plans) and row["end_reason"] == "finish"


def test_a_reader_that_requires_why_fails_the_test_above(tmp_path, monkeypatch):
    """The control of the test above: a reader that refuses a plan without why leaves the run with the schema off no usable plan."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import reply_parser
    head = reply_parser._plan_head

    def requiring(text, plan_length):
        data = reply_parser.last_json_object(text or "")
        if isinstance(data, dict) and "why" not in data:
            raise plan_schema.PlanError("why is required")
        return head(text, plan_length)
    monkeypatch.setattr(reply_parser, "_plan_head", requiring)
    _bodies, plans, _row = _plays_without_why(tmp_path, monkeypatch)
    assert plans and not any(p["source"] == "model" for p in plans)
