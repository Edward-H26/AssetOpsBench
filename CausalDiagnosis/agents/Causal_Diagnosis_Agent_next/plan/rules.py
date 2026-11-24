"""Plan clauses over declarations and public state, with no dataset field names."""

import copy

from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import NoSuchResource

from . import messages
from .plan_schema import OWN_STEPS, PlanError, Standing
from .preparation import MISSING, expand_step as _expand_step, fact_value as _fact_value
from .readers import copy_json as _copy_json, read_count as _read_count


def _no_remote_resource(uri):
    raise NoSuchResource(ref=uri)


def _row(step, state):
    return state.table.tools.get(step.tool)


def _seen(tool, state):
    return (any(previous.tool == tool for previous in state.earlier)
            or bool(state.board.holds_answer(tool, None))
            or tool == state.table.facts_tool() and state.board.facts is not None
            or state.walk.commitment is not None and state.walk.commitment.get("tool") == tool)


def _choices(source, state):
    return _fact_value(state.board.facts, source)


def _contains(values, value):
    if not isinstance(values, (dict, list, tuple)):
        return False
    return any(value == choice and isinstance(value, bool) == isinstance(choice, bool) for choice in values)


def _preview(step, state):
    if state.walk.cost is None:
        state.walk.preview = copy.deepcopy(state.walk.base)
        state.walk.cost = state.walk.preview.step(state.table.tools[step.tool], step.args)
    return state.walk.cost


def standing(step, index, state):
    """Return a standing clause or no standing clause."""
    row = _row(step, state)
    if row is None or row.write == "repair" or row.waits or row.ends:
        return None
    if known_tool(step, index, state) or arguments(step, index, state) or expectation(step, index, state) or candidate_rank(step, index, state):
        return None
    call = f"{step.tool}({step.args})"
    if row.facts or row.role == "knowledge":
        calls = _expand_step(step, state.table, state.board.facts)
        if all(state.board.holds_answer(item.tool, item.args)
               or any(old.tool == item.tool and old.args == item.args for old in state.earlier) for item in calls):
            return Standing("episode_answer", messages.STANDS_EPISODE.format(call=call))
    binding = state.table.declaration.binding
    if binding is not None and state.walk.ledger.status().get("time") is None and state.board.holds_answer(step.tool, step.args):
        needed = {name for tool in state.table.tools for name in binding.needs(tool)}
        if step.tool in needed and not row.facts:
            return Standing("surface_answer", messages.STANDS_SURFACE.format(call=call))
    if row.commits and state.walk.commitment is not None:
        held = state.walk.commitment
        own = {name: value for name, value in step.args.items() if name not in row.fills}
        previous = {name: value for name, value in held.get("args", {}).items() if name not in row.fills}
        if held.get("tool") == step.tool and previous == own:
            return Standing("same_commitment", messages.STANDS_COMMITMENT.format(commitment=held))
    cost = _preview(step, state)
    if cost.stands == "nothing_new" and row.role == "observe" and row.series:
        series = step.args.get(row.series)
        terms = state.table.declaration.vocabulary.terms
        template = (messages.STANDS_NOTHING_NEW if state.walk.ledger.status().get("time") is None
                    else messages.STANDS_NOTHING_RECORDED)
        reason = template.format(name=series, time_unit=terms.get("time_unit", "unit"),
                                 series=terms.get("series", row.series))
        collapsed = [item["n"] for item in getattr(state.board, "collapsed_rows", ()) if item.get("series") == series]
        if collapsed and state.board.options.recall:
            reason += messages.STANDS_COLLAPSED.format(first=min(collapsed), last=max(collapsed), recall_step="recall")
        return Standing("nothing_new", reason)
    if cost.stands == "state_holds" and row.write == "state":
        return Standing("state_holds", messages.STANDS_STATE.format(
            state=state.table.declaration.vocabulary.terms.get("state", "state"), value=row.sets))
    return None


def known_tool(step, index, state):
    """Read the declared clause over the current walk."""
    return [] if (step.tool in state.table.tools and state.table.tools[step.tool].role != "advice") or step.tool in OWN_STEPS else [messages.KNOWN_TOOL.format(
        tool=step.tool, names=", ".join((*state.table.tools, *OWN_STEPS)))]


def arguments(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    if row is None:
        return []
    schema = _copy_json(row.schema)
    properties = schema.get("properties", {})
    reasons = [f"{name} is no argument of {step.tool}" for name in step.args if name not in properties]
    deferred = state.board.facts is None
    skip = set(row.fills)
    if deferred:
        skip.update(row.omitted)
    if row.each and row.each not in step.args:
        skip.add(row.each)
    schema["required"] = [name for name in schema.get("required", ()) if name not in skip]
    try:
        validator = Draft202012Validator(schema, registry=Registry(retrieve=_no_remote_resource))
        for error in validator.iter_errors(step.args):
            name = ".".join(str(part) for part in error.path) or step.tool
            if error.validator == "required":
                missing = [key for key in error.validator_value if key not in error.instance]
                reasons.extend(f"{key} is a required argument of {step.tool}" for key in missing)
            elif error.validator == "type":
                reasons.append(f"{name} must be of type {error.validator_value}")
            else:
                reasons.append(f"{name} must satisfy the tool's {error.validator} constraint")
    except RecursionError:
        reasons.append(f"{step.tool} arguments exceed the schema validator's recursion limit")
    except Exception as error:
        reasons.append(f"{step.tool} schema could not be evaluated locally ({type(error).__name__})")
    for name, minimum in row.minimum.items():
        if name in step.args:
            _value, reason = _read_count(step.args[name], minimum, name)
            if reason:
                reasons.append(f"{name} {reason}")
    for name, source in row.values.items():
        if name not in step.args or state.board.facts is None and isinstance(source, str):
            continue
        values = _choices(source, state)
        if values is MISSING:
            reasons.append(f"public facts do not define the values of {step.tool}.{name}")
        elif not _contains(values, step.args[name]):
            reasons.append(f"{name} must be one of {values}")
    return list(dict.fromkeys(reasons))


def facts_first(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    if row is None or row.facts or state.board.facts is not None:
        return []
    facts = state.table.facts_tool()
    if row.role == "knowledge" and facts and _seen(facts, state):
        return []
    paths = [name for name, source in row.values.items() if isinstance(source, str)]
    paths += [name for name, source in row.omitted.items() if not source.startswith("fixed:")]
    binding = state.table.declaration.binding
    needs_facts = bool(binding is not None and facts in binding.needs(row.name))
    if paths or needs_facts or _preview(step, state).needs_facts:
        if facts is None:
            return [f"no declared facts tool supplies the information required by {step.tool}"]
        return [messages.FACTS_FIRST.format(facts=facts, detail=", ".join(paths) if paths else "its price or time")]
    return []


def order(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    if row is None:
        return []
    reasons = []
    for required in row.after:
        choices = (required,) if isinstance(required, str) else required
        if not any(_seen(tool, state) for tool in choices):
            reasons.append(messages.ORDER.format(needed=" or ".join(choices), tool=row.name))
    binding = state.table.declaration.binding
    if binding is not None:
        for tool in binding.needs(row.name):
            if tool != state.table.facts_tool() and not _seen(tool, state):
                reasons.append(messages.ORDER.format(needed=tool, tool=row.name))
    if row.each and row.each not in step.args and isinstance(row.values.get(row.each), str):
        tool = state.table.facts_tool()
        if tool and not _seen(tool, state):
            reasons.append(messages.ORDER.format(needed=tool, tool=row.name))
    return reasons


def single_commitment(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    if row is not None and row.commits and state.walk.commitment is not None:
        previous = state.walk.commitment
        own = {name: value for name, value in step.args.items() if name not in row.fills}
        held = {name: value for name, value in previous.get("args", {}).items() if name not in row.fills}
        if previous.get("tool") == step.tool and own == held:
            return []
        return [messages.SINGLE_COMMITMENT.format(commitment=state.walk.commitment)]
    return []


def end_last(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    return [messages.END_LAST.format(tool=step.tool)] if row is not None and row.ends and index != len(state.plan.steps) else []


def open_repair(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    if row is None or row.write != "repair" or not row.target or row.target not in step.args:
        return []
    targets = state.walk.open_targets()
    target = step.args[row.target]
    if target not in targets:
        return []
    terms = state.table.declaration.vocabulary.terms
    reason = messages.OPEN_REPAIR.format(repair=terms.get("repair", "repair"),
                                         target=terms.get("target", row.target), name=target)
    if targets[target] is not None:
        reason += messages.OPEN_REPAIR_UNTIL.format(time_unit=terms.get("time_unit", "time"), close=targets[target])
    return [reason]


def cites_observation(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    return [messages.CITES_OBSERVATION.format(tool=step.tool)] if row is not None and row.requires_observation and not state.walk.observed else []


def budget(step, index, state):
    """Read the declared clause over the current walk."""
    left = state.walk.ledger.status().get("budget_left")
    price = state.walk.preview.totals()[0]
    return [f"end the plan before this step or make the steps up to it cheaper, since the plan's prices ({price:g}) exceed the budget left ({left:g}) and pass it at this step"] if left is not None and price > left else []


def candidate_rank(step, index, state):
    """Read the declared clause over the current walk."""
    rank = state.plan.follow if step is None else step.candidate
    if rank is None:
        return []
    value, reason = _read_count(rank, 1, "rank")
    label = "follow" if step is None else "candidate"
    if reason:
        return [f"{label}: {reason}"]
    count = len(state.walk.candidates)
    if value > count:
        return [f"{label} {value} is not among the {count} candidates shown"]
    return []

def candidate_values(step, index, state):
    """Read the declared clause over the current walk."""
    rows = [row for row in state.table.tools.values() if row.commits and row.candidate]
    reasons = []
    for rank, candidate in enumerate(state.plan.candidates, 1):
        keys = set(candidate) - {"cites"}
        choices = [row for row in rows if keys <= set(row.candidate)]
        if not choices:
            reasons.append(f"candidate {rank}: an argument is no argument that states the answer")
            continue
        issues = []
        for row in choices:
            current = []
            for name in keys:
                if name in row.values:
                    source = row.values[name]
                    if state.board.facts is None and isinstance(source, str):
                        continue
                    values = _choices(source, state)
                    if values is MISSING or not _contains(values, candidate[name]):
                        current.append(f"candidate {rank}: {name} must be one of its declared values")
            if not current:
                issues = []
                break
            issues = current
        reasons.extend(issues)
    return reasons

def recall_rows(step, index, state):
    """Read the declared clause over the current walk."""
    if step.tool != "recall":
        return []
    if not state.board.options.recall:
        return ["this run offers no recall step"]
    first, last = step.args["observations"]
    return [] if state.board.recallable(step.args.get("series"), first, last) else [
        f"no row of observations {first} to {last} was collapsed, so nothing comes back"]

def expectation(step, index, state):
    """Read the declared clause over the current walk."""
    forms = state.walk.forms.get(step.tool, {})
    reasons = []
    for key, value in step.expect.items():
        if key not in forms:
            reasons.append(f"expect.{key} is no expectation of {step.tool}, and its expectations are {', '.join(forms)}")
        else:
            _value, reason = forms[key](value)
            if reason:
                reasons.append(f"expect.{key} {reason}")
    return reasons

def precondition(step, index, state):
    """Read the declared clause over the current walk."""
    return list(_preview(step, state).reasons)

def end_guard(step, index, state):
    """Read the declared clause over the current walk."""
    row = _row(step, state)
    return list(state.walk.end()) if row is not None and row.ends else []

def time_limit(step, index, state):
    """Read the declared clause over the current walk."""
    status = state.walk.ledger.status()
    initial, limit = status.get("time"), status.get("time_limit")
    if initial is None or limit is None:
        return []
    reached = initial + state.walk.preview.totals()[1]
    end = state.table.end_tool() or "the ending write"
    unit = state.table.declaration.vocabulary.terms.get("time_unit", "time")
    return [f"end the plan before this step or shorten the steps up to it, since the plan reaches {unit} {reached} but the episode ends at {limit} before {end} can run, and this step reaches it"] if reached >= limit else []

def step_limit(step, index, state):
    """Read the declared clause over the current walk."""
    left = state.walk.ledger.status().get("steps_left")
    calls = state.walk.preview.totals()[2]
    return [f"end the plan before this step, since the plan needs {calls} counted calls but {left} steps are left, and this step passes them"] if left is not None and calls > left else []
