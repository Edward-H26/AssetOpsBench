import json
import re
from . import plan_schema


JOINED_NAMES = re.compile(r"\s*(?:[+,&/]|\band\b)\s*")  # the joiners a model writes between several components of one order step


# ----- parsing -----
def last_json_object(text):
    """The last complete top level JSON object of a text, or None. Objects are found by decoding at every opening brace that
    is not inside an earlier object, so prose before, between or after them does not matter."""
    decoder = json.JSONDecoder()
    found, position = None, 0
    while True:
        start = text.find("{", position)
        if start < 0:
            return found
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 1
            continue
        if isinstance(value, dict):
            found = value
        position = start + max(end - start, 1)


def _count(value, minimum, name):
    """An integer argument at or above minimum; digit strings and integral floats are accepted, since models write both."""
    if isinstance(value, bool):
        raise plan_schema.PlanError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or value < minimum:
        raise plan_schema.PlanError(f"{name} must be an integer of at least {minimum}, got {value!r}")
    return value


def _probability(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise plan_schema.PlanError(f"{name} must be a number between 0 and 1, got {value!r}") from None
    if not 0 <= number <= 1:
        raise plan_schema.PlanError(f"{name} must lie between 0 and 1, got {value!r}")
    return number


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise plan_schema.PlanError(f"{name} must be a non empty string, got {value!r}")
    return value.strip()


def parse_step(raw, index, components=()):
    """The steps one raw step of a reply stands for: one, or one order per component when an order step joins several names of
    the profile's components, as a model writes two components joined by a plus sign or a comma, since every order names one
    component. A name the profile holds is never split, whatever characters it holds, and before the profile is read no name is."""
    if not isinstance(raw, dict):
        raise plan_schema.PlanError(f"step {index} is not an object")
    kind = plan_schema.KIND_ALIASES.get(raw.get("kind"), raw.get("kind"))
    if kind not in plan_schema.KINDS:
        raise plan_schema.PlanError(f"step {index}: kind must be one of {', '.join(plan_schema.KINDS)}, got {raw.get('kind')!r}")
    args = raw.get("args") or {}
    expect = raw.get("expect") or {}
    evidence = raw.get("evidence") or []
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
            try:
                clean_expect["effective_at_most"] = float(expect["effective_at_most"])
            except (TypeError, ValueError):
                raise plan_schema.PlanError(f"{name}: expect.effective_at_most must be a number") from None
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
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            _value, end = decoder.raw_decode(text, start)
        except ValueError as exc:
            return str(exc)
        start = text.find("{", end)
    return None


def parse_plan(text, plan_length, components=()):
    """The plan in the model's reply, or a PlanError that names what is wrong; components are those of the profile, which an
    order step that joins several names is split into. When the last complete object holds no steps and an object of the reply
    does not decode, the plan object is the incomplete one, and the error is the decoder's for it."""
    data = last_json_object(text or "")
    steps = None if data is None else data.get("steps")
    if not isinstance(steps, list) or not steps:
        broken = _undecoded_object(text or "")
        if broken is not None:
            raise plan_schema.PlanError(f"the JSON object of the plan is not complete: {broken}")
    if data is None:
        raise plan_schema.PlanError("no JSON object found in the reply")
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
    parsed = [step for i, raw in enumerate(steps) for step in parse_step(raw, i + 1, components)]
    if plan_length is not None and len(parsed) > plan_length:
        raise plan_schema.PlanError(f"the plan has {len(parsed)} steps once the joined components of its order steps are split into one order each, "
                        f"at most {plan_length} are allowed")
    if any(s.kind == "finish" for s in parsed[:-1]):
        raise plan_schema.PlanError("finish must be the last step of the plan")
    return plan_schema.Plan(branch=branch, mode=mode, why=why.strip(), steps=parsed, raw=text, candidates=parse_candidates(data.get("candidates")))


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
        raise plan_schema.PlanError("no JSON object found in the reply")
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
