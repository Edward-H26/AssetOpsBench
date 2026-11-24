"""Resolve a step's defaults and candidate values on a copy before checking."""

from .plan_schema import PlanError, Step
from .readers import copy_json
from .reply_parser import _fixed, _value
from .readers import read_count


MISSING = object()


def fact_value(facts, path):
    if not isinstance(path, str):
        return copy_json(path)
    value = facts
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return copy_json(value)


def candidates_of(plan, board, modules):
    ranked = [copy_json(candidate) for module in modules for candidate in module.candidates()]
    return ranked + copy_json(plan.candidates) if plan.candidates else copy_json(board.candidates())


def prepare_step(step, table, facts, candidates):
    """Return a normalized copy and notes; never change the model plan or facts."""
    prepared = Step(step.tool, copy_json(step.args), copy_json(step.expect), list(step.evidence), step.candidate)
    if step.tool not in table.tools:
        return prepared, ()
    row = table.tools[step.tool]
    notes = []
    if prepared.candidate is not None:
        prepared.candidate, reason = read_count(prepared.candidate, 1, f"{row.name}.candidate")
        if reason:
            raise PlanError(f"{row.name}.candidate: {reason}")
    for name in row.fills:
        if name in prepared.args:
            del prepared.args[name]
            notes.append(f"{row.name}.{name} is supplied by the controller; its written value was dropped")
    if prepared.candidate is not None and 1 <= prepared.candidate <= len(candidates):
        candidate = candidates[prepared.candidate - 1]
        for name in row.candidate:
            if name in candidate:
                prepared.args[name] = copy_json(candidate[name])
        if "evidence" in row.fills.values():
            prepared.evidence = list(dict.fromkeys([*prepared.evidence, *candidate.get("cites", ())]))
    if row.each and prepared.args.get(row.each) in (None, ""):
        prepared.args.pop(row.each, None)
    _fixed(row, prepared.args)
    for name, path in row.omitted.items():
        if name not in prepared.args and not path.startswith("fixed:") and facts is not None:
            value = fact_value(facts, path)
            if value is MISSING:
                raise PlanError(f"public facts do not define {path} for {row.name}.{name}")
            prepared.args[name] = value
    properties = row.schema.get("properties", {})
    for name, value in list(prepared.args.items()):
        prepared.args[name] = _value(value, properties.get(name, {}), f"{row.name}.{name}")
    return prepared, tuple(notes)


def expand_step(step, table, facts):
    """Expand a declared each argument only when its legal values are known."""
    row = table.tools.get(step.tool)
    if row is None or not row.each or step.args.get(row.each) not in (None, ""):
        return [step]
    if row.each in row.values:
        if facts is None and isinstance(row.values[row.each], str):
            return [step]
        choices = fact_value(facts, row.values[row.each])
    else:
        choices = row.schema.get("properties", {}).get(row.each, {}).get("enum", MISSING)
    if choices is MISSING:
        raise PlanError(f"public facts do not define the values of {row.name}.{row.each}")
    if not isinstance(choices, (dict, list, tuple)):
        raise PlanError(f"the values of {row.name}.{row.each} must be a list or mapping")
    return [Step(step.tool, {**copy_json(step.args), row.each: choice}, copy_json(step.expect), list(step.evidence), step.candidate)
            for choice in choices]
