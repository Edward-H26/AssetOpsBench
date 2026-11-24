"""Build prompt text from declarations, public memory and module-owned blocks."""

import json
from dataclasses import asdict, is_dataclass

from ..plan import messages
from ..plan.expectations import expectation_forms
from ..plan.preparation import MISSING, fact_value
from . import core_protocol as core


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      default=lambda item: asdict(item) if is_dataclass(item) else str(item))


def _rows(table):
    return tuple(row for row in table.tools.values() if row.role != "advice")


def _limits(board, table):
    terms = table.declaration.vocabulary.terms
    names = {"budget_left": "the budget", "time_limit": "the " + terms.get("time_limit", "time limit"),
             "steps_left": "the step limit"}
    return ", ".join(names.get(name, name) for name in board.ledger.limits()) or "the limits of the episode"


def active_rules(table, board, options, modules):
    """Return rule names whose declaration keys, cards or public limits apply."""
    rows = _rows(table)
    binding = table.declaration.binding
    has_candidates = any(row.candidate for row in rows)
    forms = expectation_forms(table, tuple(modules)) if options.expectations else {}
    limits = board.ledger.limits()
    applicable = {
        "known_tool": True, "arguments": True,
        "facts_first": bool(table.facts_tool() and any(not row.facts for row in rows)),
        "order": any(row.after or binding is not None and binding.needs(row.name) for row in rows),
        "single_commitment": any(row.commits for row in rows), "end_last": any(row.ends for row in rows),
        "cites_observation": any(row.requires_observation for row in rows),
        "candidate_rank": has_candidates, "candidate_values": any(row.candidate and row.values for row in rows),
        "recall_rows": options.recall, "expectation": any(forms.values()),
        "open_repair": any(row.write == "repair" for row in rows),
        "precondition": any(card.get("kind") == "precondition" for card in table.declaration.cards),
        "end_guard": any(row.ends for row in rows) and any(row.write == "repair" for row in rows),
        "budget": "budget_left" in limits, "time_limit": "time_limit" in limits, "step_limit": "steps_left" in limits,
    }
    return tuple(name for name, _sentence in messages.RULE_SENTENCES if applicable[name])


def _argument(name, schema, row, table, board):
    details = [name + ": " + _json(schema.get("type", "JSON value"))]
    if "enum" in schema:
        details.append("one of " + _json(schema["enum"]))
    constraints = {key: value for key, value in schema.items() if key not in {"type", "enum", "description", "title", "default"}}
    if constraints:
        details.append("listed constraints " + _json(constraints))
    if name in row.fills:
        details.append("written by the controller")
    if name in row.values:
        source = row.values[name]
        values = fact_value(board.facts, source)
        if isinstance(source, str) and board.facts is None:
            details.append(f"one of the {source} that {table.facts_tool()} lists")
        elif values is not MISSING:
            details.append("one of " + _json(list(values) if isinstance(values, dict) else values))
    if name == row.each:
        details.append("or empty, for one call per value")
    if name in row.minimum:
        details.append(f"at least {row.minimum[name]}")
    if name == row.waits:
        details.append("0 is read as 1")
    if name in row.omitted:
        source = row.omitted[name]
        if source.startswith("fixed:"):
            details.append("left out, it is " + source[len("fixed:"):])
        elif board.facts is not None and (value := fact_value(board.facts, source)) is not MISSING:
            details.append("left out, it is " + _json(value))
        else:
            details.append(f"left out, it takes the value that {table.facts_tool()} lists for it under {source}")
    elif name in row.schema.get("required", ()) and name not in row.fills and name != row.each:
        details.append("required")
    return "; ".join(details)


def _tool_line(row, table, board, options, forms):
    vocabulary = table.declaration.vocabulary
    terms = vocabulary.terms
    arguments = [_argument(name, schema, row, table, board) for name, schema in row.schema.get("properties", {}).items()]
    role = row.role + (f", {row.write} write" if row.write else "")
    pieces = [f"- {row.name} ({role}): args {{{' | '.join(arguments)}}}.", row.description]
    for enabled, sentence in ((row.facts, "Its answer holds the public facts of the episode."),
                              (row.commits, "It commits your answer."), (row.ends, "It ends the episode."),
                              (row.requires_observation, "It must cite an observation."),
                              (row.waits, "The controller repeats the call until the time asked has passed.")):
        if enabled:
            pieces.append(sentence)
    if row.sets is not None:
        state = next((card.get("parameters", {}).get("state") for card in table.declaration.cards
                      if card.get("kind") == "precondition"), None) or terms.get("state", "public state")
        pieces.append(f"It sets {state} to {row.sets}.")
    if row.target:
        pieces.append(f"It repairs what {row.target} names.")
    if row.after:
        groups = [item if isinstance(item, str) else " or ".join(item) for item in row.after]
        pieces.append("Only after " + ", ".join(groups) + ".")
    binding = table.declaration.binding
    if binding is not None and (needed := binding.needs(row.name)):
        pieces.append("Needs the answer of " + ", ".join(needed) + ".")
    if row.words:
        pieces.append("Its answer holds one of the words " + ", ".join(row.words) + ".")
    if row.candidate:
        pieces.append('Give these arguments, or "candidate": <rank> to take them from that candidate.')
        pieces.append("Leaving out all candidate arguments and the rank takes candidate 1.")
    for card in table.declaration.cards:
        if card.get("kind") == "precondition":
            parameters = card.get("parameters", {})
            required = parameters.get("requires", {}).get(row.name)
            if required:
                pieces.append(f"Needs {parameters['state']} " + (" or ".join(required) if isinstance(required, (list, tuple)) else _json(required)) + ".")
    if options.expectations:
        own = forms.get(row.name, {})
        if "outside" in own:
            example = ", such as " + vocabulary.example_field if vocabulary.example_field else ""
            pieces.append(core.OUTSIDE.format(example=example) + ".")
        if "word" in own:
            pieces.append(core.WORD.format() + ".")
        extra = {key: getattr(form, "schema", None) for key, form in own.items() if key not in {"outside", "word", "no_event"}}
        if extra:
            for key, shape in extra.items():
                if shape is None:
                    raise ValueError(f"{row.name}.expect.{key}: the reader declares no schema")
            pieces.append("Other declared expectations: " + _json(extra) + ".")
    if row.role == "observe" and row.series and terms.get("time_unit") and terms.get("series"):
        pieces.append(core.NOTHING_NEW.format(time_unit=terms["time_unit"], series=terms["series"], step_key="tool"))
    return " ".join(piece for piece in pieces if piece)


def _commitment(table):
    commits, end = table.commit_tools(), table.end_tool()
    if len(commits) == 1 and commits[0] == end:
        return f"You commit your answer with {end}, which also ends the episode."
    parts = []
    if commits:
        parts.append("You commit your answer with " + " or ".join(commits) + ", once in an episode.")
    if end:
        parts.append(f"{end} ends the episode.")
    return " ".join(parts)


def rendered_blocks(table, board, options, modules):
    """Return named blocks without modifying a rendered sentence afterward."""
    modules = tuple(modules)
    rows, vocabulary = _rows(table), table.declaration.vocabulary
    forms = expectation_forms(table, modules) if options.expectations else {}
    roles = {"observe": f"observe for the tools that read the {vocabulary.short_noun}",
             "knowledge": "knowledge for what is known about it", "act": "act for the work on it",
             "task": "task for the tools that read or answer the task"}
    writes = {"repair": f"a repair restores a part of the {vocabulary.short_noun}", "probe": "a probe changes it in order to test it",
              "state": "a state write sets a public state", "answer": "an answer records your answer to the task"}
    present_writes = [sentence for name, sentence in writes.items() if any(row.write == name for row in rows)]
    tool_head = core.TOOLS.format(roles=", ".join(sentence for name, sentence in roles.items() if any(row.role == name for row in rows)),
                                 writes="A tool that changes something is a write, and its line names its class: " + "; ".join(present_writes) + ". " if present_writes else "")
    facts_tool = table.facts_tool()
    parts = []
    if facts_tool:
        parts.append(f"the public facts once {facts_tool} is answered")
    if table.of_role("knowledge"):
        parts.append("what the knowledge tools returned")
    if options.summary_turn:
        parts.append("your summary of earlier turns, when one was written")
    if any(row.answer in ("records", "text") for row in rows):
        parts.append("the observations with their identifiers")
    if any(row.candidate for row in rows):
        parts.append("your candidates")
    if any(row.write in ("repair", "state") for row in rows):
        parts.append("the public state of the " + vocabulary.short_noun)
    parts.extend(("the surprises", "your last plan with the state of each step"))
    candidate = next((row for row in rows if row.candidate), None)
    candidate_arguments = ", ".join(_argument(name, candidate.schema.get("properties", {}).get(name, {}), candidate, table, board)
                                    for name in candidate.candidate) if candidate else ""
    carried = [row.name for row in rows if "evidence" in row.fills.values()]
    identifiers = identifier_forms(table, board, modules)
    active = active_rules(table, board, options, modules)
    slots = {"noun": vocabulary.noun, "short_noun": vocabulary.short_noun, "step_key": "tool",
             "roles": "", "writes": "", "parts": "; ".join(parts), "candidate_arguments": candidate_arguments,
             "commitment": _commitment(table), "events": core.EVENTS.format(names=", ".join(table.declaration.events)) if table.declaration.events else "",
             "carried": core.EVIDENCE_CARRIED.format(tools=", ".join(carried)) if carried else "",
             "identifiers": " Identifier forms: " + "; ".join(f"{form}: {meaning}" for form, meaning in identifiers) + "." if identifiers else "",
             "expect": ', "expect": {...}' if options.expectations else "",
             "follow": "<rank of the candidate you follow, or null>" if candidate else "null",
             "candidate_list": "[{...}, ...]" if candidate else "[]",
             "count": f" and at most {options.plan_length} entries; a longer plan is refused and you are asked again" if options.plan_length is not None else f", as many as {_limits(board, table)} pay; a plan they cannot pay is refused and you are asked again",
             "series": ', "series": <one ' + vocabulary.terms["series"] + ">" if vocabulary.terms.get("series") and any(row.series for row in rows) else "",
             "series_note": " Name one " + vocabulary.terms["series"] + " to bring back its rows alone." if vocabulary.terms.get("series") and any(row.series for row in rows) else "",
             "rules": ";\n".join(sentence for name, sentence in messages.RULE_SENTENCES if name in active),
             "plan_head": "follow or candidates", "limits": _limits(board, table)}
    output = []
    for name, template in core.blocks(table, options):
        if name == "PLAN":
            for index, module in enumerate(modules):
                output.extend((f"MODULE_{index}_{part}", text) for part, text in enumerate(module.blocks()))
        output.append((name, template.format(**slots)))
        if name == "ROLE":
            output.append(("TOOLS", tool_head + "\n" + "\n".join(_tool_line(row, table, board, options, forms) for row in rows)))
    return output


def system_prompt(table, board, options, modules):
    """Assemble the system prompt from public sections."""
    return "\n\n".join(text for _name, text in rendered_blocks(table, board, options, modules))


def _request(heading, board, table, options, modules, body):
    return f"{system_prompt(table, board, options, modules)}\n\n# {heading}\n\nTask of the episode: {board.question}\n\n{body}"


def orientation_turn(board, table, options, modules, retry=None):
    """Assemble the orientation request."""
    modules = tuple(modules)
    public = "\n\n".join(("## Public facts\n" + _json(board.facts), "## Knowledge\n" + _json(board.knowledge)))
    body = public + "\n\n" + core.ORIENTATION.format()
    if retry:
        body += "\n\n" + core.RETRY.format(reason=_json(retry) if not isinstance(retry, str) else retry)
    return _request("Orientation", board, table, options, modules, body)


def planning_turn(board, table, options, modules, retry=None, not_run=None, stood=None):
    """Assemble a planning request and unused or standing steps."""
    modules = tuple(modules)
    facts_tool = table.facts_tool()
    body = board.render(modules)
    if facts_tool and board.facts is None:
        knowledge = ", ".join(table.of_role("knowledge"))
        opening = facts_tool + " first" + (", then " + knowledge if knowledge else "")
        body += "\n\n" + core.OPENING_BOARD.format(facts_tool=facts_tool, short_noun=table.declaration.vocabulary.short_noun, opening=opening)
    body += "\n\n" + core.PLANNING
    if facts_tool and board.facts is None:
        body += "\n" + core.OPENING.format(facts_tool=facts_tool)
    if board.facts is not None and getattr(board, "opening_candidates_dropped", False):
        body += "\n" + core.OPENING_CANDIDATES
    if retry:
        body += "\n\n" + core.RETRY.format(reason=_json(retry) if not isinstance(retry, str) else retry)
    if not_run:
        body += "\n\n" + core.NOT_RUN_HEADING + "\n" + _json(not_run) + "\n" + core.NOT_RUN_END
    if stood:
        lines = []
        for ordinal, entry in enumerate(stood, 1):
            if is_dataclass(entry):
                number, name, reason = entry.index, entry.step.tool, entry.stood.reason
            elif isinstance(entry, tuple):
                number, name, reason = entry
            else:
                number, name, reason = entry.get("number", entry.get("index", ordinal)), entry.get("step_name", entry.get("tool", "")), entry["reason"]
            lines.append(core.STOOD_LINE.format(number=number, step_name=name, reason=reason))
        body += "\n\n" + core.STOOD_HEADING + "\n" + "\n".join(lines) + "\n" + core.STOOD_END
    return _request("Planning", board, table, options, modules, body)


def earlier_turns_turn(board, table, options, modules, left_out=0):
    """Assemble the earlier-turn compaction input."""
    modules = tuple(modules)
    keeps = ["the task"]
    if table.facts_tool():
        keeps.append("the public facts")
    if table.of_role("knowledge"):
        keeps.append("what the knowledge tools returned")
    if any(row.write in ("repair", "state") for row in _rows(table)):
        keeps.append("the public state")
    if modules:
        keeps.append("the advice")
    if board.refuted:
        keeps.append("the refuted answers")
    if any(row.answer in ("records", "text") for row in _rows(table)):
        keeps.append("the latest observation of every series")
    keeps.append("the status")
    body = board.render(modules) + "\n\n" + core.EARLIER.format(keeps=", ".join(keeps))
    if left_out:
        body += " " + core.LEFT_OUT
    return _request("Summary of earlier turns", board, table, options, modules, body)


def summary_turn(board, table, options, modules, note=None):
    """Assemble the closing summary request."""
    modules = tuple(modules)
    writes = [{"tool": call["tool"], "args": call["args"], "answer": call["answer"]}
              for call in board.calls if table.tools[call["tool"]].write]
    body = "## Writes and answers\n" + _json(writes) + "\n\n" + board.render(modules)
    if note:
        body += "\n\n" + str(note)
    body += "\n\n" + core.CLOSING.format(short_noun=table.declaration.vocabulary.short_noun)
    return _request("Closing summary", board, table, options, modules, body)


def identifier_forms(table, board, modules):
    """Return the citation forms of the current declaration and modules."""
    forms = []
    for row in _rows(table):
        if row.answer == "records":
            forms.append(("obs/<label or n>/<field>", "a served numeric value"))
        elif row.answer == "text":
            forms.append(("obs/<n>", "a text answer"))
    for item in table.declaration.vocabulary.identifiers:
        forms.append((item, "an identifier declared by the vocabulary") if isinstance(item, str) else tuple(item))
    for module in modules:
        provider = getattr(module, "identifier_forms", None)
        if provider is not None:
            forms.extend(tuple(item) for item in provider())
    return tuple(dict.fromkeys(forms))
