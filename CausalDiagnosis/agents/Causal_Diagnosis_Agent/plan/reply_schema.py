"""The form of a reply as a JSON schema, which the planning request and the orientation request carry in the request field
response_format, so that a serving engine with a grammar backend decodes only replies of that form. SGLang and vLLM take the same
field, {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": true}}, and both hand the schema to xgrammar.

The schema is written from the declared form of the reader (plan_schema.STEP_FORM, HEAD_FORM, CANDIDATE_FORM and
DIFFERENTIAL_FORM) for the kinds and the modes that a row serves and the count of steps of the run, and from nothing else: no name
of the asset and no bound that the reader does not set, so a row has one schema and one digest, and the engine compiles it once.
It is the canonical form of a reply and narrower than the reader on purpose: the reader goes on taking a count written as a string
of digits, the old name of the wait step, keys that it does not know and a plan without why. Some values pass the schema and not the reader, since a
schema of these keywords cannot state them: a text of blanks alone, an integer above plan_schema.LARGEST_INTEGER, a number past the
largest float, a finish before the last step and a span whose first observation comes after its last. Two more pass the grammar
that the engine compiles from the schema and not the reader: a probability above 1 and below 2, such as 1.5, which the grammar admits
past the maximum of 1 and the reader refuses with "must lie between 0 and 1", and a raw control character other than CR and LF inside
a text with a minimum length, which the grammar admits and the reader refuses with "the JSON object of the plan is not complete
(Invalid control character ...)" in a plan, and with "the reply needs a list under differential" or "no JSON object was found in the
reply" in a differential, the second when no other object of the reply decodes. The reader refuses each with its reason, so it stays
the one way in for every reply, with the schema or without it."""
from CausalDiagnosis.utils.json_digest import digest as json_digest
from . import plan_schema


PLAN, DIFFERENTIAL = "plan", "differential"  # the names of the two schemas in the request
DIGEST_DIGITS = 16  # the hex digits of the SHA-256 of the canonical JSON that name a schema in the records
LEVEL = {"anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}]}  # a level of the profile, of whatever type it gives


def _value(reader, modes=()):
    """The schema of one value of the declared form."""
    if isinstance(reader, tuple):
        kind, least = reader
        number = {"type": "integer", "minimum": least}
        return {"anyOf": [number, {"type": "null"}]} if kind == "count or null" else number
    return {plan_schema.TEXT: {"type": "string", "minLength": 1},
            plan_schema.ANY_TEXT: {"type": "string"},
            plan_schema.PROBABILITY: {"type": "number", "minimum": 0, "maximum": 1},
            plan_schema.NUMBER: {"type": "number"},
            plan_schema.FLAG: {"type": "boolean"},
            plan_schema.NAMES: {"type": "array", "items": {"type": "string", "minLength": 1}},
            plan_schema.SPAN: {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2},
            # the engine's compiler allows no property that a schema does not state, so the levels state theirs
            plan_schema.LEVELS: {"type": "object", "additionalProperties": LEVEL},
            plan_schema.MODE: {"enum": list(modes)}}[reader]


def _object(fields, values):
    """An object of the declared fields, values the schema of each, with its required fields and no other key."""
    required = [name for name, (_reader, needed) in fields.items() if needed]
    return {"type": "object", "properties": values, **({"required": required} if required else {}), "additionalProperties": False}


def _fields(fields):
    return _object(fields, {name: _value(reader) for name, (reader, _needed) in fields.items()})


def _step(kind):
    """One step of the kind: its name as a constant, its arguments, required where one of them is, its expectation, never
    required, and the evidence on the kinds whose call records it."""
    form = plan_schema.STEP_FORM[kind]
    values, required = {"kind": {"const": kind}}, ["kind"]
    if form["args"]:
        values["args"] = _fields(form["args"])
        if any(needed for _reader, needed in form["args"].values()):
            required.append("args")
    if form["expect"]:
        values["expect"] = _fields(form["expect"])
    if kind in plan_schema.EVIDENCE_KINDS:
        values["evidence"] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": values, "required": required, "additionalProperties": False}


def served_kinds(unserved, opening, recall):
    """The kinds of a plan that a row serves, in the order of KINDS: none whose tool the row does not serve, no read of the opening
    that the surface does not need before a reading (opening), and no recall with the switch off."""
    return tuple(kind for kind in plan_schema.KINDS if kind not in unserved
                 and (kind not in plan_schema.SURFACE_KINDS or kind in opening) and (recall or kind != "recall"))


def served_modes(unserved):
    """The modes that a row serves: a mode named after a kind that the row does not serve leaves."""
    return tuple(mode for mode in plan_schema.MODES if mode not in unserved)


def plan_schema_of(kinds, modes, plan_length=None):
    """The schema of a plan reply: the head in the order of HEAD_FORM, one variant of a step per kind of kinds in the order of
    KINDS, and at most plan_length steps when the run sets a count."""
    steps = {"type": "array", "items": {"anyOf": [_step(kind) for kind in plan_schema.KINDS if kind in kinds]}, "minItems": 1,
             **({"maxItems": plan_length} if plan_length is not None else {})}
    composite = {plan_schema.CANDIDATES: {"type": "array", "items": _fields(plan_schema.CANDIDATE_FORM)}, plan_schema.STEPS: steps}
    return _object(plan_schema.HEAD_FORM, {name: composite[reader] if reader in composite else _value(reader, modes)
                                           for name, (reader, _needed) in plan_schema.HEAD_FORM.items()})


def differential_schema():
    """The schema of the reply of the orientation turn: the list of the differential, each entry of DIFFERENTIAL_FORM."""
    return {"type": "object", "properties": {"differential": {"type": "array", "items": _fields(plan_schema.DIFFERENTIAL_FORM)}},
            "required": ["differential"], "additionalProperties": False}


def response_format(name, schema):
    """The request field that binds a reply to the schema, in the form that both engines document."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


def digest(schema):
    """The name of a schema in the records: the first hex digits of the SHA-256 of its canonical JSON, which keeps the order of its
    properties, since the engine decodes the keys in that order."""
    return json_digest(schema)[:DIGEST_DIGITS]
