import json
import re
import sys
from . import plan_schema


JOINED_NAMES = re.compile(r"\s*(?:[+,&/]|\band\b)\s*")  # the joiners a model writes between several components of one order step
COUNT_TEXT = re.compile(r"(-?)0*([0-9]+)")  # a count written as a string: the digits 0 to 9 after an optional minus sign
NUMBER_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")  # a number written as a string, in the digits 0 to 9
# the reason for a reply whose lists and objects nest deeper than the JSON decoder follows
TOO_DEEP = "the reply nests its lists and objects deeper than the controller can read; return one JSON object"
UNREADABLE = "the reply could not be read"  # the refusal of the last guard of read_reply, a fault of the reader and not of the model
# the reasons for a reply without a whole object and for a step that is no object, each naming what to write instead
INCOMPLETE_OBJECT = "the JSON object of the plan is not complete ({detail}); return the whole object again, and nothing else"
NO_OBJECT = "no JSON object was found in the reply; return one JSON object, and nothing else"
STEP_NOT_OBJECT = "step {number} is not an object; write every step as an object"


def read_reply(parse, text, *args):
    """The one way in for a model's reply: what parse reads from the text (parse_usable_plan, parse_differential), or a
    plan_schema.Refusal that names why it reads nothing, and nothing else. parse refuses a reply with a PlanError that names its
    reason. Any other exception is a fault of the reader and never of the model: the last guard turns it into the refusal
    UNREADABLE with the class of the exception, which the controller keeps in the records of the episode (reader_faults), so the
    planning call cannot raise and the fault stays visible."""
    try:
        return parse(text, *args)
    except plan_schema.PlanError as exc:
        return plan_schema.Refusal(reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - the last guard of the boundary, RecursionError included
        return plan_schema.Refusal(reason=UNREADABLE, fault=type(exc).__name__)


# ----- parsing -----
def _json_integer(text):
    """An integer of the reply's JSON as the decoder hands it over (its parse_int): the integer, or an infinity of its sign when it
    has more digits than int() converts (sys.get_int_max_str_digits), which every reader refuses by its range, where the decoder
    would refuse the whole reply as an object that does not decode."""
    limit = sys.get_int_max_str_digits()
    if limit and len(text.lstrip("-")) > limit:
        return float("-inf") if text.startswith("-") else float("inf")
    return int(text)


def _decoder():
    return json.JSONDecoder(parse_int=_json_integer)


def last_json_object(text):
    """The last complete top level JSON object of a text, or None. Objects are found by decoding at every opening brace that
    is not inside an earlier object, so prose before, between or after them does not matter. A text whose lists and objects nest
    deeper than the decoder follows is refused whole (TOO_DEEP), since decoding again at every brace inside such an object would
    take a time that grows with the square of its depth, and so is an object that holds a lone surrogate (_encodable)."""
    decoder = _decoder()
    found, position = None, 0
    while True:
        start = text.find("{", position)
        if start < 0:
            return _encodable(found)
        try:
            value, end = decoder.raw_decode(text, start)
        except RecursionError:
            raise plan_schema.PlanError(TOO_DEEP) from None
        except ValueError:
            position = start + 1
            continue
        if isinstance(value, dict):
            found = value
        position = start + max(end - start, 1)


def _encodable(value):
    """value, a decoded JSON value, when every key and every string that it holds is text that UTF-8 encodes. A lone surrogate, which
    a JSON escape or the client of the endpoint can put into a string and which no UTF-8 text holds (RFC 8259, section 8.2), refuses
    the reply whole, since the requests, the tool calls and the files of the episode encode every string that the controller keeps.
    Walked with a stack, since the value can nest as deep as the decoder follows."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise plan_schema.PlanError(f"a string of the reply holds U+{ord(item[exc.start]):04X}, half of a surrogate pair, which no text "
                                            "holds alone; write every character as itself") from None
        elif isinstance(item, dict):
            stack += [*item, *item.values()]
        elif isinstance(item, list):
            stack += item
    return value


def _integer_text(text):
    """The integer that a string of the digits 0 to 9 after an optional minus sign writes, or None for any other string, a
    superscript, a circled digit or a doubled sign among them. A string with more digits than LARGEST_INTEGER has reads as one past
    it, so no conversion meets the limit that int() sets on the digits of a string."""
    written = COUNT_TEXT.fullmatch(text.strip())
    if written is None:
        return None
    sign, digits = written.groups()
    magnitude = int(digits) if len(digits) <= len(str(plan_schema.LARGEST_INTEGER)) else plan_schema.LARGEST_INTEGER + 1
    return -magnitude if sign else magnitude


def _count(value, minimum, name):
    """An integer argument from minimum to LARGEST_INTEGER, written as a JSON integer, as a JSON number with an integral value or as
    a string of digits (_integer_text), and the parser accepts all three forms."""
    if isinstance(value, bool):
        raise plan_schema.PlanError(f"{name} must be an integer, got {value!r}")
    number = value
    if isinstance(value, str):
        number = _integer_text(value)
    elif isinstance(value, float):  # an infinity, which no integer reaches, is past the largest
        number = int(value) if value.is_integer() else plan_schema.LARGEST_INTEGER + 1 if value > plan_schema.LARGEST_INTEGER else None
    if not isinstance(number, int) or number < minimum:
        raise plan_schema.PlanError(f"{name} must be an integer of at least {minimum}, got {value!r}")
    if number > plan_schema.LARGEST_INTEGER:
        raise plan_schema.PlanError(f"{name} must be an integer of at most {plan_schema.LARGEST_INTEGER}, got {value!r}")
    return number


def _number(value):
    """The number that a JSON number, or a string of one in the digits 0 to 9, writes, or None; a boolean is no number."""
    if isinstance(value, str):
        return float(value.strip()) if NUMBER_TEXT.fullmatch(value.strip()) else None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _probability(value, name):
    number = _number(value)
    if number is None:
        raise plan_schema.PlanError(f"{name} must be a number between 0 and 1, got {value!r}")
    if not 0 <= number <= 1:  # NaN fails the test, and an integer is held against the range before any conversion to a float
        raise plan_schema.PlanError(f"{name} must lie between 0 and 1, got {value!r}")
    return float(number)


def _finite(value, name):
    """A number that a float holds, NaN and the infinities excluded."""
    number = _number(value)
    if number is None or not -sys.float_info.max <= number <= sys.float_info.max:  # as in _probability, NaN fails the test
        raise plan_schema.PlanError(f"{name} must be a finite number, got {value!r}")
    return float(number)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise plan_schema.PlanError(f"{name} must be a non empty string, got {value!r}")
    return value.strip()


def parse_step(raw, index, components=()):
    """The steps one raw step of a reply stands for: one, or one order per component when an order step joins several names of
    the profile's components, as a model writes two components joined by a plus sign or a comma, since every order names one
    component. A name the profile holds is never split, whatever characters it holds, and before the profile is read no name is."""
    if not isinstance(raw, dict):
        raise plan_schema.PlanError(STEP_NOT_OBJECT.format(number=index))
    written = raw.get("kind")
    kind = plan_schema.KIND_ALIASES.get(written, written) if isinstance(written, str) else written  # a kind is a string before any lookup
    if kind not in plan_schema.KINDS:
        raise plan_schema.PlanError(f"step {index}: kind must be one of {', '.join(plan_schema.KINDS)}, got {written!r}")
    args = raw.get("args") or {}
    expect = raw.get("expect") or {}
    # only a step whose call records evidence carries it, so the key of any other step is not read, whatever it holds
    evidence = (raw.get("evidence") or []) if kind in plan_schema.EVIDENCE_KINDS else []
    if not isinstance(args, dict) or not isinstance(expect, dict):
        raise plan_schema.PlanError(f"step {index}: args and expect must be objects")
    if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
        raise plan_schema.PlanError(f"step {index}: evidence must be a list of identifier strings")
    name = f"step {index} ({kind})"
    cited = [e.strip() for e in evidence if e.strip()]
    clean_args, clean_expect = {}, {}
    if kind == "read":
        clean_args = {"sensor_group": _text(args.get("sensor_group"), f"{name}: sensor_group"), "cycles": _count(args.get("cycles", 1), 1, f"{name}: cycles")}
        if "top_rank" in expect:
            clean_expect["top_rank"] = _count(expect["top_rank"], 1, f"{name}: expect.top_rank")
        if "top_mass_at_least" in expect:
            clean_expect["top_mass_at_least"] = _probability(expect["top_mass_at_least"], f"{name}: expect.top_mass_at_least")
        if "effective_at_most" in expect:
            clean_expect["effective_at_most"] = _finite(expect["effective_at_most"], f"{name}: expect.effective_at_most")
        if "outside" in expect:
            outside = expect["outside"]
            if not isinstance(outside, list) or not all(isinstance(c, str) and c.strip() for c in outside):
                raise plan_schema.PlanError(f"{name}: expect.outside must be a list of channel names")
            clean_expect["outside"] = [c.strip() for c in outside]
    elif kind == "diagnose":
        clean_args = {"rank": _count(args.get("rank", 1), 1, f"{name}: rank")}
        if "top_rank" in expect:  # without one the controller expects what the advice of the run's repair choice expects (runner._diagnose)
            clean_expect["top_rank"] = _count(expect["top_rank"], 1, f"{name}: expect.top_rank")
    elif kind == "declare":
        clean_args = {"sensor_group": _text(args.get("sensor_group"), f"{name}: sensor_group"), "cycles": _count(args.get("cycles", 1), 1, f"{name}: cycles")}
    elif kind == "order":
        prediction = _text(args.get("prediction_id") or plan_schema.ESTIMATE, f"{name}: prediction_id")
        written = _text(args.get("component"), f"{name}: component")
        parts = [c for c in JOINED_NAMES.split(written) if c]
        named = parts if written not in components and len(parts) > 1 and all(p in components for p in parts) else [written]
        clean_expect["prediction_id"] = _text(expect.get("prediction_id") or prediction, f"{name}: expect.prediction_id")
        if "recovery_at_least" in expect:
            clean_expect["recovery_at_least"] = _probability(expect["recovery_at_least"], f"{name}: expect.recovery_at_least")
        return [plan_schema.Step(kind=kind, args={"component": c, "prediction_id": prediction}, expect=dict(clean_expect), evidence=list(cited)) for c in named]
    elif kind == "wait":
        # a wait of 0 cycles asks for the next event, which the first cycle brings, so it reads as 1
        clean_args = {"cycles": max(1, _count(args.get("cycles", 1), 0, f"{name}: cycles"))}
        clean_expect = {"no_new_alarm": bool(expect.get("no_new_alarm", True))}
    elif kind == "recall":  # the full rows of collapsed observations back on the board
        span = args.get("observations")
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in span)
                and span[0] <= span[1]):
            raise plan_schema.PlanError(f"{name}: observations must be [first, last], two observation numbers with first not after last")
        clean_args = {"sensor_group": _text(args.get("sensor_group"), f"{name}: sensor_group"), "observations": list(span)}
    elif kind == "failure_modes":
        component = args.get("component") or ""
        if not isinstance(component, str):
            raise plan_schema.PlanError(f"{name}: component must be a component name, or empty for every component")
        clean_args = {"component": component.strip()}
    return [plan_schema.Step(kind=kind, args=clean_args, expect=clean_expect, evidence=cited)]


def _undecoded_object(text):
    """The decoder's message for the first opening brace of a text, outside the objects before it, whose object does not decode,
    or None when every such object decodes."""
    decoder = _decoder()
    start = text.find("{")
    while start >= 0:
        try:
            _value, end = decoder.raw_decode(text, start)
        except ValueError as exc:
            return str(exc)
        start = text.find("{", end)
    return None


FINISH_LAST = "finish must be the last step of the plan"


def _plan_head(text, plan_length):
    """The plan object of the model's reply, its steps as written and its head, the branch, the mode and the why, or a PlanError
    that names what is wrong. When the last complete object holds no steps and an object of the reply does not decode, the plan
    object is the incomplete one, and the error is the decoder's for it."""
    data = last_json_object(text or "")
    steps = None if data is None else data.get("steps")
    if not isinstance(steps, list) or not steps:
        broken = _undecoded_object(text or "")
        if broken is not None:
            raise plan_schema.PlanError(INCOMPLETE_OBJECT.format(detail=broken))
    if data is None:
        raise plan_schema.PlanError(NO_OBJECT)
    if not isinstance(steps, list) or not steps:
        raise plan_schema.PlanError("the plan needs a non empty list under steps")
    if plan_length is not None and len(steps) > plan_length:
        raise plan_schema.PlanError(f"the plan has {len(steps)} steps, at most {plan_length} are allowed")
    mode = data.get("mode")
    if mode not in plan_schema.MODES:
        raise plan_schema.PlanError(f"mode must be one of {', '.join(plan_schema.MODES)}, got {mode!r}")
    branch = data.get("branch")
    branch = _count(branch, 1, "branch") if branch is not None else None
    why = data.get("why", "")
    if not isinstance(why, str):
        raise plan_schema.PlanError("why must be a string")
    return data, steps, branch, mode, why.strip()


def _split_count(parsed, plan_length):
    """The count of steps once the joined components of the order steps are split, which bears on the plan as a whole."""
    if plan_length is not None and len(parsed) > plan_length:
        raise plan_schema.PlanError(f"the plan has {len(parsed)} steps once the joined components of its order steps are split into one order each, "
                        f"at most {plan_length} are allowed")


def parse_plan(text, plan_length, components=()):
    """The plan in the model's reply, or a PlanError that names what is wrong; components are those of the profile, which an
    order step that joins several names is split into."""
    data, steps, branch, mode, why = _plan_head(text, plan_length)
    parsed = [step for i, raw in enumerate(steps) for step in parse_step(raw, i + 1, components)]
    _split_count(parsed, plan_length)
    if any(s.kind == "finish" for s in parsed[:-1]):
        raise plan_schema.PlanError(FINISH_LAST)
    return plan_schema.Plan(branch=branch, mode=mode, why=why, steps=parsed, raw=text, candidates=parse_candidates(data.get("candidates")))


def _written_kind(raw):
    """The kind of a step as the reply wrote it, for a step that does not run."""
    kind = raw.get("kind") if isinstance(raw, dict) else None
    return kind if isinstance(kind, str) and kind else "no kind"


def parse_usable_plan(text, plan_length, components=()):
    """The plan in the model's reply up to its first step that the parser cannot use, with that step (plan_schema.Unusable) or
    None: a step that does not parse, or a finish with steps after it, since a finish ends the episode. The steps before it are
    those the reply wrote. A reply whose JSON object, head or candidates do not parse, and a plan with more steps than plan_length,
    raise a PlanError, since they bear on the plan as a whole."""
    data, steps, branch, mode, why = _plan_head(text, plan_length)
    candidates = parse_candidates(data.get("candidates"))
    parsed, unusable = [], None
    for i, raw in enumerate(steps, 1):
        try:
            taken = parse_step(raw, i, components)
        except plan_schema.PlanError as exc:
            unusable = plan_schema.Unusable(step=len(parsed) + 1, reasons=[str(exc)], kinds=[_written_kind(r) for r in steps[i - 1:]])
            break
        parsed += taken
    _split_count(parsed, plan_length)
    early = next((k for k, s in enumerate(parsed, 1) if s.kind == "finish" and (k < len(parsed) or unusable is not None)), None)
    if early is not None:  # a finish with steps after it, the parsed ones and those of a step that does not parse
        later = unusable or plan_schema.Unusable(step=len(parsed) + 1, reasons=[], kinds=[])
        unusable = plan_schema.Unusable(step=early, reasons=[FINISH_LAST, *later.reasons], kinds=[s.kind for s in parsed[early - 1:]] + later.kinds)
        parsed = parsed[:early - 1]
    return plan_schema.Plan(branch=branch, mode=mode, why=why, steps=parsed, raw=text, candidates=candidates), unusable


def parse_candidates(raw):
    """The model's own candidates of a plan in its order, each {"levels": {component: level}, "failure_mode_ids": [...]}, a
    component left out being at its healthy level; the check holds the components and the levels, of whatever type the
    profile gives them, against the profile."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise plan_schema.PlanError("candidates must be a list")
    out = []
    for i, item in enumerate(raw, 1):
        levels = item.get("levels") if isinstance(item, dict) else None
        if not isinstance(levels, dict):
            raise plan_schema.PlanError(f"candidate {i} needs levels, an object from component to level")
        ids = item.get("failure_mode_ids") or []
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise plan_schema.PlanError(f"candidate {i}: failure_mode_ids must be a list of identifier strings")
        out.append({"levels": {str(c): v for c, v in levels.items()},
                    "failure_mode_ids": [x.strip() for x in ids if x.strip()]})
    return out


def parse_differential(text):
    """The differential of the orientation turn: a list of {component, expected_symptoms, failure_mode_ids}."""
    data = last_json_object(text or "")
    if data is None:
        raise plan_schema.PlanError(NO_OBJECT)
    items = data.get("differential")
    if not isinstance(items, list):
        raise plan_schema.PlanError("the reply needs a list under differential")
    out = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict) or not isinstance(item.get("component"), str):
            raise plan_schema.PlanError(f"differential entry {i} needs a component")
        ids = item.get("failure_mode_ids") or []
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise plan_schema.PlanError(f"differential entry {i}: failure_mode_ids must be a list of strings")
        out.append({"component": item["component"], "expected_symptoms": str(item.get("expected_symptoms", "")), "failure_mode_ids": ids})
    return out
