"""Schemas derived from the declaration and options alone."""

import hashlib
import json

from .readers import MAX_COUNT, copy_json


KEYWORDS = frozenset(("type", "properties", "required", "additionalProperties", "const", "enum", "anyOf",
                      "items", "minItems", "maxItems", "minLength", "minimum", "maximum"))
ANNOTATIONS = frozenset(("description", "title", "default"))


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def _text():
    return {"type": "string", "minLength": 1}


def _texts():
    return {"type": "array", "items": _text()}


def _shape(schema):
    """Project supported argument shapes; leave other shapes to the live check."""
    if not isinstance(schema, dict) or set(schema) - KEYWORDS - ANNOTATIONS:
        return {}
    result = {key: copy_json(value) for key, value in schema.items() if key in KEYWORDS}
    kind = result.get("type")
    if isinstance(kind, list):
        return {"anyOf": [_shape({**schema, "type": item}) for item in kind]}
    if "anyOf" in result:
        result["anyOf"] = [_shape(item) for item in result["anyOf"]]
    if kind == "string":
        result["minLength"] = max(1, result.get("minLength", 0))
    if kind == "integer":
        result.setdefault("minimum", -MAX_COUNT - 1)
    if "properties" in result:
        result["properties"] = {name: _shape(value) for name, value in result["properties"].items()}
    for name in ("items", "additionalProperties"):
        if isinstance(result.get(name), dict):
            result[name] = _shape(result[name])
    return result


def _arguments(row, *, candidate=False):
    properties = {name: _shape(value) for name, value in row.schema.get("properties", {}).items()
                  if name not in row.fills and not (candidate and name in row.candidate)}
    for name, minimum in row.minimum.items():
        if name in properties:
            properties[name] = {**properties[name], "type": "integer",
                                "minimum": max(minimum, properties[name].get("minimum", minimum))}
    required = [name for name in row.schema.get("required", ())
                if name in properties and name not in row.omitted and name != row.each]
    if candidate:
        properties["candidate"] = {"type": "integer", "minimum": 1}
    return _object(properties, required)


def _step(row, forms, options):
    args = _arguments(row)
    args_required = bool(args["required"])
    if row.candidate:
        ranked = _arguments(row, candidate=True)
        args_required = bool(ranked["required"])
        args = {"anyOf": [args, ranked]}
    properties = {"tool": {"const": row.name}, "args": args}
    if options.expectations:
        expected = {}
        for name, form in forms.get(row.name, {}).items():
            schema = getattr(form, "schema", None)
            if schema is None:
                raise ValueError(f"{row.name}.expect.{name}: the reader declares no schema")
            expected[name] = _shape(schema)
        properties["expect"] = _object(expected)
    if "evidence" in row.fills.values():
        properties["evidence"] = _texts()
    return _object(properties, ("tool", "args") if args_required else ("tool",))


def plan_schema_of(table, forms, options):
    """Build the schema of a plan reply."""
    variants = [_step(row, forms, options) for row in table.tools.values() if row.role != "advice"]
    if options.recall:
        args = _object({"observations": {"type": "array", "items": {"type": "integer", "minimum": 1},
                                          "minItems": 2, "maxItems": 2}, "series": _text()}, ("observations",))
        properties = {"tool": {"const": "recall"}, "args": args}
        if options.expectations:
            properties["expect"] = _object({})
        variants.append(_object(properties, ("tool", "args")))
    steps = {"type": "array", "minItems": 1, "items": {"anyOf": variants}}
    if options.plan_length is not None:
        steps["maxItems"] = options.plan_length
    candidates = []
    for row in table.tools.values():
        if row.commits and row.candidate:
            args = _arguments(row)
            properties = {name: value for name, value in args["properties"].items() if name in row.candidate}
            properties["cites"] = _texts()
            candidates.append(_object(properties, [name for name in row.schema.get("required", ())
                                                   if name in row.candidate and name not in row.omitted]))
    items = {"anyOf": candidates} if candidates else {}
    candidate_list = {"type": "array", "items": items}
    if not candidates:
        candidate_list["maxItems"] = 0
    return _object({"follow": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
                    "why": _text(), "candidates": candidate_list, "steps": steps}, ("why", "steps"))


def differential_schema():
    """Build the schema of a differential reply."""
    entry = _object({"about": _text(), "expected": _text(), "cites": _texts()}, ("about", "expected"))
    return _object({"differential": {"type": "array", "items": entry}}, ("differential",))


def response_format(name, schema):
    """Wrap a reply schema for structured output."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": copy_json(schema), "strict": True}}


def digest(schema):
    """Hash a canonical reply schema."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
