"""The assembler of the prompt (CausalDiagnosis/docs/general-agent-plan.md, section The prompt). A block is core (a constant with
slots), rendered (built from declared fields) or given by a module; no sentence is edited after it is written, and a block or
sentence of a part that is switched off is left out.

The assembler reads the table and its declaration, the options, the modules and, through the documented interface of the board,
the task message, the facts, the commitment, the ledger with its limits, the status and the rendered board. It reads every
attribute of the board through one guard, so that a part that the board does not serve yet prints as absent and never raises.

A module gives its blocks as a list from blocks(): an item that is a text is its block of the system prompt, and a pair
("board", text) is its sentence of BOARD. forms(row) gives the expectation keys that it judges for a step of the row, each with its
shape as a text, which the line of the tool names.

The sentences of the rules of the check are constants of the plan package: the assembler prints those that the function
sentences(table, board, modules) of plan/rules.py returns, in its order, and the block RULES is left out while that function is
not in the tree."""
import json
import re

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.agent_options import Options
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.prompt import core_protocol as core

SLOT = re.compile(r"\{([a-z_]+(?: [a-z_]+)*)\}")
STEP_ROLES = ("observe", "knowledge", "act", "task")  # the roles that a step can name; a tool of the role advice has no line
FIXED = "fixed:"


def fill(template, /, **slots):
    """The template with every slot that slots names filled; a slot that it does not name, and every other brace, stay as they
    are. A slot whose name holds a space is given with an underscore in its place."""
    named = {**{key.replace("_", " "): value for key, value in slots.items()}, **slots}
    return SLOT.sub(lambda m: str(named[m.group(1)]) if m.group(1) in named else m.group(0), template)


def _read(thing, name, default=None):
    """An attribute of the board, the ledger or a module, called where it is a method without arguments; the default where it is
    missing or not built yet."""
    try:
        value = getattr(thing, name)
        return value() if callable(value) else value
    except Exception:  # noqa: BLE001 - a part that the board or the ledger does not serve yet prints as absent
        return default


def _either(words, last="or"):
    """Words joined with commas and a last or, or and."""
    words = [str(w) for w in words]
    if len(words) < 2:
        return "".join(words)
    return ", ".join(words[:-1]) + f" {last} " + words[-1]


def _terms(table):
    return dict(table.declaration.vocabulary.terms or {})


def _slots(table):
    """The slots that the vocabulary and the plan form fill."""
    vocabulary, terms = table.declaration.vocabulary, _terms(table)
    return {"noun": vocabulary.noun, "short_noun": vocabulary.short_noun, "step_key": core.STEP_KEY, "plan_head": core.PLAN_HEAD,
            "time_unit": terms.get("time_unit", core.TIME_WORD), "time_limit": terms.get("time_limit", core.TIME_LIMIT_WORD)}


def _planned(table):
    """The rows that a step can name, in table order."""
    return {name: row for name, row in table.tools.items() if row.role in STEP_ROLES}


def _facts(board):
    facts = _read(board, "facts")
    return facts if isinstance(facts, dict) else None


def before_facts(table, board):
    """Whether the table has a facts tool whose answer the board does not hold yet."""
    return table.facts_tool() is not None and _facts(board) is None


def _at_path(facts, path):
    """The keys or the items at a dotted path of the facts, or None where the path holds neither."""
    value = facts
    for part in str(path).split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return list(value) if isinstance(value, (dict, list, tuple)) else None


def _module_blocks(modules):
    """(the blocks of the system prompt, the sentences of BOARD) of the modules, in their order."""
    blocks, sentences = [], []
    for module in modules or ():
        for item in _read(module, "blocks", ()) or ():
            if isinstance(item, (tuple, list)) and len(item) == 2:
                (sentences if item[0] == "board" else blocks).append(str(item[1]))
            elif item:
                blocks.append(str(item))
    return blocks, sentences


def _module_forms(row, modules):
    forms = {}
    for module in modules or ():
        try:
            forms.update(module.forms(row) or {})
        except Exception:  # noqa: BLE001 - a module that judges nothing for this row prints no form
            continue
    return forms


def limits(table, board, last="or"):
    """{limits}: the names of the limits that the episode has, joined, or the limits of the episode while the ledger knows none."""
    names = _read(_read(board, "ledger"), "limits", ()) or ()
    if not names:
        return core.NO_LIMITS
    words = {name: fill(text, time_limit=_slots(table)["time_limit"]) for name, text in core.LIMIT_WORDS.items()}
    return _either([words.get(name, "the " + str(name).replace("_", " ")) for name in names], last)


# ----- the rendered texts of a tool line -----

def argument_text(row, name, spec, table, facts):
    """The text of one argument of a tool line: its type, its legal values, and what the keys of its row say of it."""
    facts_tool = table.facts_tool()
    if name in row.fills:
        return core.ARG_FILL
    types = spec.get("type")
    if types is None and isinstance(spec.get("anyOf"), list):
        types = [part.get("type") for part in spec["anyOf"] if isinstance(part, dict) and part.get("type")]
    parts = [_either(types) if isinstance(types, list) and types else str(types) if types else core.ARG_ANY]
    if isinstance(spec.get("enum"), list):
        parts.append(fill(core.ARG_ENUM, values=_either(spec["enum"])))
    if name in row.values:
        listed = _at_path(facts, row.values[name]) if facts is not None else None
        if listed is not None:
            parts.append(fill(core.ARG_ENUM, values=_either(listed)))
        elif facts_tool is not None:
            parts.append(fill(core.ARG_VALUES, path=row.values[name], facts_tool=facts_tool))
    if row.each == name:
        parts.append(core.ARG_EACH)
    if name in row.minimum:
        parts.append(fill(core.ARG_MINIMUM, n=row.minimum[name]))
    if row.waits == name:
        parts.append(core.ARG_WAITS)
    if name in row.omitted:
        omitted = str(row.omitted[name])
        if omitted.startswith(FIXED):
            parts.append(fill(core.ARG_OMITTED_FIXED, value=omitted[len(FIXED):]))
        elif facts_tool is not None:
            parts.append(fill(core.ARG_OMITTED_PATH, facts_tool=facts_tool, path=omitted))
    return ", ".join(parts)


def arguments_text(row, table, facts):
    properties = (row.schema or {}).get("properties") or {}
    items = [fill(core.ARGUMENT, name=name, text=argument_text(row, name, spec if isinstance(spec, dict) else {}, table, facts))
             for name, spec in properties.items()]
    return "{" + ", ".join(items) + "}"


def key_sentences(row, table):
    """The sentences of the row keys, in the order of the plan page."""
    terms, out = _terms(table), []
    for key in core.KEY_ORDER:
        text = core.KEY_SENTENCES[key]
        if key in ("facts", "commits", "ends", "waits", "requires_observation") and getattr(row, key):
            out.append(text)
        elif key == "sets" and row.sets:
            out.append(fill(text, state=terms.get("state", core.STATE_WORD), value=row.sets))
        elif key == "target" and row.target:
            out.append(fill(text, argument=row.target))
        elif key == "after" and row.after:
            out.append(fill(text, x=_either([_either(item) if isinstance(item, (tuple, list)) else item for item in row.after], "and")))
        elif key == "words" and row.words:
            out.append(fill(text, words=_either(row.words)))
        elif key == "candidate" and row.commits and row.candidate:
            out.append(text)
    return out


def requirement(row, table):
    """The requirement of a tool from the card precondition: Needs ..., or With ...: needs ... for each state its argument needs."""
    out = []
    for card in table.declaration.cards:
        if card.get("kind") != "precondition":
            continue
        parameters = card.get("parameters") or {}
        state = _terms(table).get("state") or parameters.get("state") or core.STATE_WORD
        need = (parameters.get("requirements") or {}).get(row.name)
        if isinstance(need, str):
            out.append(fill(core.NEEDS, state=state, value=need))
        elif isinstance(need, dict):
            grouped = {}
            for value, required in (need.get("values") or {}).items():
                grouped.setdefault(required, []).append(value)
            out += [fill(core.NEEDS_WITH, argument=need.get("argument"), values=_either(values), state=state, value=required)
                    for required, values in grouped.items()]
    return out


def expectations_of(row, table, modules):
    """The expectations that a step of the tool can state: the form of its answer, then the forms of the modules."""
    vocabulary, out = table.declaration.vocabulary, []
    if row.role in ("observe", "knowledge", "act"):
        if row.answer == "records":
            example = fill(core.EXPECT_EXAMPLE, example_field=vocabulary.example_field) if vocabulary.example_field else ""
            out.append(fill(core.EXPECT_RECORDS, example=example))
        elif row.answer == "text":
            out.append(core.EXPECT_TEXT)
    for key, shape in _module_forms(row, modules).items():
        out.append(fill(core.EXPECT_MODULE, key=key, shape=shape if isinstance(shape, str) else key))
    return out


def read_stands(row, table):
    """The sentence of a call with nothing new, on an observe tool with a series where the vocabulary holds its terms."""
    terms = _terms(table)
    if row.role != "observe" or not row.series or "time_unit" not in terms or "series" not in terms:
        return None
    return fill(core.READ_STANDS, time_unit=terms["time_unit"], series=terms["series"], step_key=core.STEP_KEY)


def tool_line(row, table, facts, modules):
    head = fill(core.TOOL_LINE_HEAD, name=row.name, role=row.role, write=fill(core.TOOL_WRITE, write=row.write) if row.write else "",
                args=arguments_text(row, table, facts))
    parts = [head, " ".join((row.description or "").split()), *key_sentences(row, table), *requirement(row, table),
             *expectations_of(row, table, modules), read_stands(row, table)]
    return " ".join(part for part in parts if part)


# ----- the blocks of the system prompt -----

def tools_block(table, board, modules):
    slots, rows = _slots(table), _planned(table)
    roles = {row.role for row in rows.values()}
    writes = {row.write for row in rows.values() if row.write}
    role_text = _either([fill(text, **slots) for role, text in core.ROLE_CLAUSES if role in roles], "and")
    write_text = (core.WRITES_HEAD + _either([fill(text, **slots) for write, text in core.WRITE_CLAUSES if write in writes], "and") + "."
                  if writes else "")
    head = fill(core.TOOLS_HEAD, roles=role_text, writes=write_text).replace("  ", " ")
    facts = _facts(board)
    return "\n".join([head, *(tool_line(row, table, facts, modules) for row in rows.values())])


def expectations_block(table):
    events = table.declaration.events
    sentence = " " + fill(core.EVENTS, event_names=_either(events)) if events else ""
    return fill(core.EXPECTATIONS, short_noun=table.declaration.vocabulary.short_noun, step_key=core.STEP_KEY, events=sentence)


def evidence_block(table, board, modules):
    steps = [name for name, row in _planned(table).items() if "evidence" in row.fills.values()]
    carried = " " + fill(core.EVIDENCE_CARRIED, evidence_steps=_either(steps)) if steps else ""
    text = fill(core.EVIDENCE, carried=carried)
    forms = identifier_forms(table, board, modules)
    if forms:
        text += " " + fill(core.IDENTIFIERS, forms="; ".join(fill(core.IDENTIFIER_FORM, form=form, names=names) for form, names in forms))
    return text


def candidate_arguments(table):
    """{candidate_arguments}: the candidate arguments of the first commit write that holds any, with the sentences of their
    values and of an argument left out."""
    commit = next((row for row in _planned(table).values() if row.commits and row.candidate), None)
    if commit is None:
        return _either([f"the arguments of {name}" for name in table.commit_tools()])
    text = _either(commit.candidate, "and")
    if any(name in commit.values for name in commit.candidate):
        text += ", " + core.CANDIDATE_VALUES
    left = {name: str(commit.omitted[name]) for name in commit.candidate if name in commit.omitted}
    facts_tool = table.facts_tool()
    if left and facts_tool is not None and not any(path.startswith(FIXED) for path in left.values()):
        heads = {path[:-len(name) - 1] for name, path in left.items() if path.endswith("." + name)}
        path = heads.pop() if len(heads) == 1 and all(path.endswith("." + name) for name, path in left.items()) else _either(left.values(), "and")
        text += fill(core.CANDIDATE_OMITTED, facts_tool=facts_tool, path=path)
    return text


def commitment_text(table):
    """{commitment}: the commit writes, once in an episode, and the write that ends it."""
    commits, end = list(table.commit_tools()), table.end_tool()
    if end is not None and commits == [end]:
        return fill(core.COMMIT_ENDS, tool=end)
    out = [fill(core.COMMIT_ONCE, commit_tools=_either(commits))] if commits else []
    if end is not None:
        out.append(fill(core.END_TOOL, end_tool=end))
    return " ".join(out)


def candidates_block(table):
    return fill(core.CANDIDATES, candidate_arguments=candidate_arguments(table), commitment=commitment_text(table)).rstrip()


def has_open_work(table):
    """Whether the ledger of the kind holds open work: a repair write on a target where time passes, so that a repair closes after
    the call that began it."""
    return "time_unit" in _terms(table) and any(row.target for row in _planned(table).values() if row.write == "repair")


def board_block(table, modules):
    rows, facts_tool = _planned(table), table.facts_tool()
    knowledge = any(row.role == "knowledge" for row in rows.values())
    parts = []
    if facts_tool is not None:
        parts.append(fill(core.PART_FACTS, facts_tool=facts_tool) + (core.PART_FACTS_KNOWLEDGE if knowledge else ""))
    elif knowledge:
        parts.append(core.PART_KNOWLEDGE)
    parts += [core.PART_SUMMARY, core.PART_OBSERVATIONS, core.PART_CANDIDATES]
    if has_open_work(table):
        parts.append(fill(core.PART_OPEN_WORK, **_slots(table)))
    parts += [core.PART_SURPRISES, core.PART_LAST_PLAN]
    text = [fill(core.BOARD, parts="; ".join(parts)), core.BOARD_HELD]
    series = _terms(table).get("series")
    if series and any(row.series for row in rows.values()):
        text.append(fill(core.BOARD_HELD_RECORD, series=series))
    return " ".join(text + _module_blocks(modules)[1])


def plan_block(table, board, options):
    if options.plan_length is not None:
        count = fill(core.COUNT_SET, n=options.plan_length)
    else:
        count = fill(core.COUNT_PAID, limits=limits(table, board, "and"))
    return fill(core.PLAN, count=count)


def recall_block(table):
    series = _terms(table).get("series")
    with_series = bool(series) and any(row.series for row in _planned(table).values())
    text = fill(core.RECALL, series=fill(core.RECALL_SERIES_ARGUMENT, series_term=series) if with_series else "")
    return text + (" " + fill(core.RECALL_SERIES, series_term=series) if with_series else "")


def rule_sentences(table, board, modules):
    """The sentences of the rules in force, in the order of the table of the check, from plan/rules.py; () while it states none."""
    try:
        from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan import rules
        return tuple(rules.sentences(table, board, modules))
    except Exception:  # noqa: BLE001 - the sentences of the rules come with the check
        return ()


def rules_block(table, board, modules):
    sentences = rule_sentences(table, board, modules)
    if not sentences:
        return None
    return fill(core.RULES_HEAD, short_noun=table.declaration.vocabulary.short_noun) + " " + "; ".join(sentences) + "."


def loop_block(table, board, options):
    text = fill(core.LOOP, plan_head=core.PLAN_HEAD, limits=limits(table, board, "or"), short_noun=table.declaration.vocabulary.short_noun)
    if not options.expectations:
        text = core.LOOP_WITHOUT_EXPECTATIONS + " " + text.split(". ", 1)[1]
    return text + " " + core.LOOP_STANDING


# ----- the identifier forms -----

def identifier_forms(table, board, modules):
    """The forms of the identifiers that a step may cite, each (form, what it names): those of the answer forms of the table,
    those that the vocabulary states, those of the ledger of the kind and of the modules where they state identifier_forms, and
    those of the knowledge tools."""
    rows = _planned(table)
    answers = {row.answer for row in rows.values() if row.role in ("observe", "knowledge", "act")}
    forms = [form for answer, form in (("records", core.FORM_RECORDS), ("text", core.FORM_TEXT)) if answer in answers]
    forms += [tuple(form) for form in table.declaration.vocabulary.identifiers]
    for source in (table.declaration.rules, _read(board, "ledger"), *(modules or ())):
        stated = _read(source, "identifier_forms", ()) or ()
        forms += [tuple(form) for form in stated if isinstance(form, (tuple, list)) and len(form) == 2]
    forms += [(fill(core.FORM_KNOWLEDGE[0], tool=name), core.FORM_KNOWLEDGE[1]) for name, row in rows.items() if row.role == "knowledge"]
    return tuple(dict.fromkeys(forms))


# ----- the system prompt and the four turns -----

def system_prompt(table, board, options, modules):
    """The system prompt of a turn, which reads the facts and the limits of the ledger from the board."""
    options = options if options is not None else Options()
    blocks = [fill(core.ROLE, **_slots(table)), tools_block(table, board, modules), expectations_block(table),
              evidence_block(table, board, modules), candidates_block(table), board_block(table, modules),
              *_module_blocks(modules)[0], plan_block(table, board, options),
              recall_block(table) if options.recall else None, core.BATCH if options.parallel else None,
              rules_block(table, board, modules), loop_block(table, board, options)]
    return "\n\n".join(block for block in blocks if block)


def task_text(board):
    question = _read(board, "question")
    return core.TASK + ("" if question is None else str(question))


def _render(board, modules):
    try:
        return str(board.render(modules) or "")
    except Exception:  # noqa: BLE001 - a board that cannot render yet prints nothing
        return ""


def _head(board, table, options, modules, heading):
    parts = (system_prompt(table, board, options, modules), heading, task_text(board), _render(board, modules))
    return "\n\n".join(part for part in parts if part)


def _retry_text(retry):
    return "\n".join(map(str, retry)) if isinstance(retry, (list, tuple)) else str(retry)


def orientation_turn(board, table, options, modules, retry=None):
    """The request for the differential: the system prompt, the task, the facts and the knowledge answers as the board prints
    them, then the ask; retry the reason why the last reply was not usable, or None."""
    text = _head(board, table, options, modules, core.ORIENTATION_HEADING) + "\n\n" + core.ORIENTATION_ASK
    if retry:
        text += f"\n\n{core.RETRY_HEADING}\n{_retry_text(retry)}\n{core.RETRY_DIFFERENTIAL}"
    return text


def _step_name(step):
    return step.get("tool") if isinstance(step, dict) else getattr(step, "tool", step)


def not_run_section(not_run):
    """The section of the steps of the last plan that did not run: one line per step from the first that could not be used, then
    the reasons, closed by NOT_RUN_CLOSING."""
    first = not_run.get("step") or 1
    lines = [fill(core.NOT_RUN_LINE, number=first + offset, step_name=_step_name(step)) for offset, step in enumerate(not_run.get("steps") or ())]
    lines += [fill(core.NOT_RUN_REASON, reason=reason) for reason in not_run.get("reasons") or ()]
    return "\n".join([core.NOT_RUN_HEADING, *lines, core.NOT_RUN_CLOSING])


def stood_section(stood):
    """The section of the standing steps of the last plan, one line per step, closed by STOOD_CLOSING."""
    lines = [fill(core.STOOD_LINE, number=number, step_name=name, reason=reason) for number, name, reason in stood]
    return "\n".join([core.STOOD_HEADING, *lines, core.STOOD_CLOSING])


def planning_turn(board, table, options, modules, retry=None, not_run=None, stood=None):
    """The planning request: the system prompt, the task, the board and the objective; retry the reasons of the last reply or
    plan that ran nothing, or None; not_run the cut of the last plan, {"step", "steps", "reasons"}, or None; stood the standing
    steps of the last plan, each (number, tool, reason), or None."""
    text = _head(board, table, options, modules, core.PLANNING_HEADING) + "\n\n" + core.OBJECTIVE
    if before_facts(table, board):
        text += " " + fill(core.BEFORE_FACTS_TURN, facts_tool=table.facts_tool())
    if retry:
        text += f"\n\n{core.RETRY_HEADING}\n{_retry_text(retry)}\n{core.RETRY_PLAN}"
    if not_run:
        text += "\n\n" + not_run_section(not_run)
    if stood:
        text += "\n\n" + stood_section(stood)
    return text


def earlier_turns_turn(board, table, options, modules, left_out=0):
    """The request for the summary of earlier turns at the compaction; left_out the count of entries left out of it."""
    return (_head(board, table, options, modules, core.EARLIER_TURNS_HEADING) + "\n\n" + core.EARLIER_TURNS_ASK
            + (" " + core.LEFT_OUT if left_out else ""))


def _call_text(tool, args):
    return f"{tool} {json.dumps(args, sort_keys=True, default=str)}" if args else str(tool)


def _answer_text(write):
    answer = write.get("answer")
    if isinstance(answer, dict):
        if isinstance(answer.get("refused"), str):
            return "refused: " + answer["refused"]
        answer = answer.get("result")
    return answer if isinstance(answer, str) else json.dumps(answer, sort_keys=True, default=str)


def status_line(table, board):
    """The limits line of the closing summary, from the status of the board."""
    status = _read(board, "status") or {}
    status = status if isinstance(status, dict) else {}
    parts = [fill(core.STATUS_WORDS[key], value=status[key], **_slots(table)) for key in core.STATUS_WORDS if status.get(key) is not None]
    if status.get("over"):
        parts.append(core.STATUS_OVER)
    return fill(core.STATUS_LINE, limits=", ".join(parts) or core.NO_LIMITS)


def summary_turn(board, table, options, modules, note=None):
    """The request for the closing summary: every write of the episode with what it returned, every verdict and the limits."""
    writes = _read(board, "writes")
    if writes is None:
        commitment = _read(board, "commitment")
        writes = [commitment] if isinstance(commitment, dict) else []
    lines = [fill(core.WRITE_LINE, call=_call_text(w.get("tool"), w.get("args")), answer=_answer_text(w)) for w in writes if isinstance(w, dict)]
    parts = [system_prompt(table, board, options, modules), core.SUMMARY_HEADING, task_text(board),
             "\n".join([core.WRITTEN_HEAD, *lines]) if lines else core.NO_WRITE]
    verdicts = _read(board, "verdicts") or ()
    if verdicts:
        parts.append("\n".join([core.VERDICTS_HEAD, *(fill(core.VERDICT_LINE, verdict=v) for v in verdicts)]))
    parts += [status_line(table, board), str(note) if note else None,
              fill(core.CLOSING_ASK, short_noun=table.declaration.vocabulary.short_noun)]
    return "\n\n".join(part for part in parts if part)
