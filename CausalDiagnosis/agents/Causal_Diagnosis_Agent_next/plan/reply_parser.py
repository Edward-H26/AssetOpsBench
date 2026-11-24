"""Read tool plans without dataset names or episode facts.

The live checker resolves facts-dependent defaults, drops written fills with a
board note and judges legality. A parsed plan is never dispatched directly.
"""

import json
import math
from . import readers
from .plan_schema import OWN_STEPS, Parsed, Plan, PlanError, Refusal, Step, Unusable


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise PlanError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise PlanError(f"{value} is not a finite JSON number")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise PlanError("the JSON number exceeds the finite number range")
    return number


def _encodable(value):
    """Refuse surrogate halves before a value can reach requests or run files."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                raise PlanError("the reply must contain complete UTF-8 characters") from None
        elif isinstance(item, dict):
            pending.extend(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


def _written_tool(raw):
    tool = raw.get("tool") if isinstance(raw, dict) else None
    return tool if isinstance(tool, str) and tool else "no tool"


def last_json_object(text):
    """Read the last root object, refusing duplicate keys and incomplete JSON."""
    if not isinstance(text, str):
        raise PlanError("the reply must be text")
    _encodable(text)
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    start, found, failure = None, None, None
    stack = []
    quoted = escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char not in "{[":
                continue
            start = index
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or (stack[-1], char) not in (("{", "}"), ("[", "]")):
                raise PlanError("the JSON delimiters do not match")
            stack.pop()
            if stack:
                continue
            try:
                value = decoder.decode(text[start:index + 1])
            except PlanError as error:
                failure, found = error, None
            except RecursionError:
                failure, found = PlanError("the JSON nesting exceeds the decoder's recursion limit"), None
            except ValueError:
                failure, found = PlanError("the reply contains an invalid JSON value"), None
            else:
                if isinstance(value, dict):
                    found, failure = _encodable(value), None
                else:
                    found, failure = None, PlanError("the reply needs a JSON object")
            start = None
    if start is not None:
        raise PlanError("the JSON object of the reply is incomplete")
    if failure is not None:
        raise failure
    return found


def _need(read, name):
    value, reason = read
    if reason is not None:
        raise PlanError(reason if reason.startswith(f"{name}:") else f"{name}: {reason}")
    return value


def _texts(value, name):
    return [_need(readers.read_text(item, f"{name}[{index}]"), f"{name}[{index}]")
            for index, item in enumerate(_need(readers.read_list(value, name), name), 1)]


def _unknown(value, allowed, name):
    extra = set(value) - set(allowed)
    if extra:
        raise PlanError(f"{name}: unknown keys {', '.join(sorted(extra))}")


def _value(value, schema, name):
    """Read supported shapes; the live checker judges unsupported schemas."""
    supported = {"type", "properties", "required", "additionalProperties", "const", "enum", "anyOf",
                 "items", "minItems", "maxItems", "minLength", "minimum", "maximum", "description", "title", "default"}
    if not isinstance(schema, dict) or set(schema) - supported:
        return readers.copy_json(value)
    if "anyOf" in schema:
        for form in schema["anyOf"]:
            try:
                return _value(value, form, name)
            except PlanError:
                pass
        raise PlanError(f"{name}: no allowed argument form matches")
    kind = schema.get("type")
    if isinstance(kind, list):
        return _value(value, {"anyOf": [{**schema, "type": part} for part in kind]}, name)
    if kind == "integer":
        value = _need(readers.read_count(value, schema.get("minimum", -readers.MAX_COUNT - 1), name), name)
        if "maximum" in schema and value > schema["maximum"]:
            raise PlanError(f"{name}: must be at most {schema['maximum']}")
    elif kind == "number":
        value = _need(readers.read_number(value, schema.get("minimum"), schema.get("maximum"), name), name)
    elif kind == "string":
        value = _need(readers.read_text(value, name), name)
        if len(value) < schema.get("minLength", 0):
            raise PlanError(f"{name}: text is shorter than its declared minimum")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise PlanError(f"{name}: must be a boolean")
    elif kind == "null":
        if value is not None:
            raise PlanError(f"{name}: must be null")
    elif kind == "array":
        value = _need(readers.read_list(value, name), name)
        if len(value) < schema.get("minItems", 0) or "maxItems" in schema and len(value) > schema["maxItems"]:
            raise PlanError(f"{name}: list length is outside its declared bounds")
        value = [_value(item, schema.get("items", {}), f"{name}[{index}]") for index, item in enumerate(value, 1)]
    elif kind == "object":
        value = _need(readers.read_object(value, name), name)
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            _unknown(value, properties, name)
        for key in schema.get("required", ()):
            if key not in value:
                raise PlanError(f"{name}: {key} is required")
        value = {key: _value(item, properties.get(key, {}), f"{name}.{key}") for key, item in value.items()}
    if "enum" in schema and not any(value == item and (isinstance(value, bool) == isinstance(item, bool)) for item in schema["enum"]):
        raise PlanError(f"{name}: must be one of its declared values")
    if "const" in schema and (value != schema["const"] or isinstance(value, bool) != isinstance(schema["const"], bool)):
        raise PlanError(f"{name}: differs from its declared constant")
    return value


def _fixed(row, args):
    for name, source in row.omitted.items():
        if name not in args and source.startswith("fixed:"):
            written = source[len("fixed:"):]
            try:
                args[name] = json.loads(written)
            except ValueError:
                args[name] = written


def parse_candidates(raw, table):
    """Read candidate values and citations from the commitment rows."""
    raw = _need(readers.read_list(raw, "candidates"), "candidates")
    rows = [row for row in table.tools.values() if row.commits and row.candidate]
    result = []
    for index, item in enumerate(raw, 1):
        name = f"candidate {index}"
        item = _need(readers.read_object(item, name), name)
        cites = _texts(item.pop("cites", []), f"{name}.cites")
        matched, errors = None, []
        for row in rows:
            try:
                _unknown(item, row.candidate, name)
                args = dict(item)
                _fixed(row, args)
                properties = row.schema.get("properties", {})
                for key in row.candidate:
                    if key not in args:
                        if key in row.schema.get("required", ()) and key not in row.omitted:
                            raise PlanError(f"{name}: {key} is required")
                    else:
                        args[key] = _value(args[key], properties.get(key, {}), f"{name}.{key}")
                matched = {key: value for key, value in args.items() if key in row.candidate}
                break
            except PlanError as error:
                errors.append(str(error))
        if matched is None:
            raise PlanError(errors[0] if errors else f"{name}: no tool declares candidate arguments")
        result.append({**matched, "cites": cites})
    return result


def _recall(args):
    _unknown(args, ("observations", "series"), "recall.args")
    span = _need(readers.read_list(args.get("observations"), "recall.observations"), "recall.observations")
    if len(span) != 2:
        raise PlanError("recall.observations: must contain first and last")
    span = [_need(readers.read_count(value, 1, "recall.observations"), "recall.observations") for value in span]
    if span[0] > span[1]:
        raise PlanError("recall.observations: first must not follow last")
    result = {"observations": span}
    if "series" in args:
        result["series"] = _need(readers.read_text(args["series"], "recall.series"), "recall.series")
    return result


def _step(raw, table, forms):
    raw = _need(readers.read_object(raw, "step"), "step")
    _unknown(raw, ("tool", "args", "expect", "evidence"), "step")
    tool = _need(readers.read_name(raw.get("tool"), (*[row.name for row in table.tools.values() if row.role != "advice"], *OWN_STEPS), "step.tool"), "step.tool")
    args = _need(readers.read_object(raw.get("args", {}), f"{tool}.args"), f"{tool}.args")
    expect = _need(readers.read_object(raw.get("expect", {}), f"{tool}.expect"), f"{tool}.expect")
    _unknown(expect, forms.get(tool, {}), f"{tool}.expect")
    expect = {key: _need(forms[tool][key](value), f"{tool}.expect.{key}") for key, value in expect.items()}
    if tool == "recall":
        return Step(tool, _recall(args), expect, [])
    row = table.tools[tool]
    properties = row.schema.get("properties", {})
    allowed = set(properties) | set(row.fills) | ({"candidate"} if row.candidate else set())
    _unknown(args, allowed, f"{tool}.args")
    explicit = set(args) & set(row.candidate)
    candidate = None
    if row.candidate:
        if "candidate" in args:
            candidate = _need(readers.read_count(args.pop("candidate"), 1, f"{tool}.candidate"), f"{tool}.candidate")
            if explicit:
                raise PlanError(f"{tool}: give candidate arguments or a rank, not both")
        elif not explicit:
            candidate = 1
    if row.each and args.get(row.each) in (None, ""):
        args.pop(row.each, None)
    _fixed(row, args)
    if row.waits and row.waits in args:
        args[row.waits] = max(1, _need(readers.read_count(args[row.waits], 0, f"{tool}.{row.waits}"), f"{tool}.{row.waits}"))
    for key in row.schema.get("required", ()):
        if key not in args and key not in row.fills and key not in row.omitted and key != row.each and not (candidate and key in row.candidate):
            raise PlanError(f"{tool}: {key} is required")
    # The checker drops written fills with a board note and resolves facts paths.
    args = {key: value if key in row.fills else _value(value, properties.get(key, {}), f"{tool}.{key}")
            for key, value in args.items()}
    for key, minimum in row.minimum.items():
        if key in args and key not in row.fills:
            args[key] = _need(readers.read_count(args[key], minimum, f"{tool}.{key}"), f"{tool}.{key}")
    evidence = _texts(raw.get("evidence", []), f"{tool}.evidence") if "evidence" in row.fills.values() else []
    return Step(tool, args, expect, evidence, candidate)


def parse_usable_plan(text, table, forms, plan_length):
    """Return a usable prefix, a named refusal, or an unexpected-error guard."""
    try:
        data = last_json_object(text)
        if data is None:
            raise PlanError("no JSON object found in the reply")
        _unknown(data, ("follow", "why", "candidates", "steps"), "plan")
        steps = _need(readers.read_list(data.get("steps"), "steps"), "steps")
        if not steps:
            raise PlanError("steps must contain at least one entry")
        if plan_length is not None and len(steps) > plan_length:
            raise PlanError(f"the plan has {len(steps)} steps, at most {plan_length} are allowed")
        follow = None if data.get("follow") is None else _need(readers.read_count(data["follow"], 1, "follow"), "follow")
        why = _need(readers.read_text(data.get("why"), "why"), "why")
        candidates = parse_candidates(data.get("candidates", []), table)
        parsed, unusable = [], None
        for index, entry in enumerate(steps, 1):
            try:
                parsed.append(_step(entry, table, forms))
            except PlanError as error:
                if not parsed:
                    raise PlanError(f"step {index}: {error}") from None
                unusable = Unusable(index, (str(error),), tuple(_written_tool(item) for item in steps[index - 1:]))
                break
        return Parsed(Plan(follow, why, parsed, candidates, text), unusable, None)
    except PlanError as error:
        return Parsed(None, None, Refusal(str(error)))
    except Exception as error:
        return Parsed(None, None, Refusal("the reply could not be read", type(error).__name__))


def parse_differential(text):
    """Read hypotheses expressed with the vocabulary of the current task."""
    try:
        data = last_json_object(text)
        if data is None:
            raise PlanError("no JSON object found in the reply")
        _unknown(data, ("differential",), "orientation")
        entries = _need(readers.read_list(data.get("differential"), "differential"), "differential")
        result = []
        for index, entry in enumerate(entries, 1):
            name = f"differential entry {index}"
            entry = _need(readers.read_object(entry, name), name)
            _unknown(entry, ("about", "expected", "cites"), name)
            result.append({"about": _need(readers.read_text(entry.get("about"), f"{name}.about"), f"{name}.about"),
                           "expected": _need(readers.read_text(entry.get("expected"), f"{name}.expected"), f"{name}.expected"),
                           "cites": _texts(entry.get("cites", []), f"{name}.cites")})
        return result, None
    except PlanError as error:
        return [], Refusal(str(error))
    except Exception as error:
        return [], Refusal("the reply could not be read", type(error).__name__)
