from __future__ import annotations
import json
import re
from typing import Annotated, Any, Callable, Optional
from langchain_core.tools import StructuredTool
from pydantic import BeforeValidator, ConfigDict, Field, ValidationError, create_model


# ----- the MCP tools of a sub-agent -----
TYPES = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
# ReActXen reads an action named Finish, in any case, as the end of the agent's own task (agents.py step, the Finish branch),
# so the rig's task tool finish could never be called under its name, and the agent sees it under this one
RESERVED = {"finish": "end_episode"}
# the words the engine reads as its own inside an action, in any case (agents/react/utils.py, the readers of an action), so
# that a malformed action which holds one of them ends the agent's task or asks itself instead of calling the tool
ENGINE_WORDS = ("finish", "final answer", "self-ask", "agent-ask")


def agent_name(tool):
    """The name under which an agent sees an MCP tool. A name that would hold a word of ENGINE_WORDS is refused, so a tool
    that needs another name cannot be listed without its entry in RESERVED."""
    shown = RESERVED.get(tool.lower(), tool)
    held = [word for word in ENGINE_WORDS if word in shown.lower()]
    if held:
        raise ValueError(f"the tool {tool} would be shown to an agent as {shown}, which holds {held[0]!r}, a word that the engine "
                         "reads as its own inside an action: give the tool another shown name in RESERVED")
    return shown


# ----- the answers as the engine shows them -----
_SPACE = re.compile(r"\s*")


def one_line(text):
    """An answer as one line, the form in which every answer of a tool of the port reaches a ReActXen agent: each JSON value of
    the answer, the receipt line of the metering layer included, in JSON without indentation and one space apart, and any
    other answer with its line breaks replaced by spaces. The engine shortens a scratchpad that outgrows its limit only by the
    lines that open with Observation (agents/react/utils.py, truncate_scratchpad), so an answer over many lines cannot be
    shortened, and the truncation raises once it has cut every such line."""
    decoder, values, body = json.JSONDecoder(), [], text.strip()
    at = 0
    try:
        while at < len(body):
            value, end = decoder.raw_decode(body, at)
            values.append(value)
            at = _SPACE.match(body, end).end()
    except ValueError:
        return " ".join(text.splitlines())
    return " ".join(json.dumps(value, ensure_ascii=False) for value in values)


# ----- the engine's forms of an argument -----
def _object_from_text(value):
    """An object argument that the engine's key=value form passes as text: a text that is a JSON object is read as that
    object, so that the model checks the object. Any other value passes unchanged and the model refuses it as before."""
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        return parsed if isinstance(parsed, dict) else value
    return value


def _no_value(value):
    """Whether a value is one of the engine's forms for no value: None, a list with no item that holds text, or nothing but a
    pair of quotes."""
    if isinstance(value, list):
        return all(_no_value(item) or (isinstance(item, str) and not item.strip()) for item in value)
    return value is None or (isinstance(value, str) and value.strip() in ('""', "''"))


def _paired(text):
    return len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'"


def _peeled(text):
    """A text without its outer quotes, peeled layer by layer, its spaces kept. The layers are the pairs of the same quote, of
    either kind, that wrap the text one inside the other, and as many go as leave none of their quotes in what is left, so
    ""CM"" gives CM, while a pair whose quote also stands inside the text in another way stays, as in "pump" is "bad"."""
    core = text.strip()
    layers = [core]
    while _paired(layers[-1]):
        layers.append(layers[-1][1:-1])
    # the deepest layer first: in ""CM"" one layer off leaves the quote of the second, two leave none
    for depth in range(len(layers) - 1, 0, -1):
        if not any(layer[0] in layers[depth] for layer in layers[:depth]):
            return text.replace(core, layers[depth], 1)
    return text


def _unquoted(value):
    """A text without its outer quotes. A model writes a value in quotes in the engine's key=value form, as in work_type="CM",
    and the engine keeps them (agents/react/agents.py, the call of a tool), so the server would read the quotes as part of the
    value. The text is peeled layer by layer (_peeled), and then each comma item of what is left by the same rule, so that
    evidence_ids="obs/1/a", "obs/2/b", whose outer quotes belong to different items, gives obs/1/a, obs/2/b. A value that
    loses no quote passes unchanged."""
    if isinstance(value, str):
        text = value.strip()
        peeled = ",".join(_peeled(item) for item in _peeled(text).split(","))
        return peeled if peeled != text else value
    return value


def _joined(items):
    return ",".join(_unquoted(str(item).strip()) for item in items)


def _text_from_engine(value):
    """A required text argument: the engine's forms for no value are the empty text, a list is its items joined by commas,
    and the text and every item lose their outer quotes (_unquoted). Any other value passes unchanged."""
    if _no_value(value):
        return ""
    if isinstance(value, list):
        return _joined(value)
    return _unquoted(value)


def _optional_text_from_engine(value):
    """An optional text argument: the engine's forms for no value are no value, so that the server's default holds, a list
    is its items joined by commas, and the text and every item lose their outer quotes (_unquoted). Any other value passes
    unchanged."""
    if _no_value(value):
        return None
    if isinstance(value, list):
        return _joined(value)
    return _unquoted(value)


def _kind(prop):
    if isinstance(prop.get("type"), str):
        return TYPES.get(prop["type"], Any)
    kinds = [p.get("type") for p in prop.get("anyOf", []) if p.get("type") != "null"]
    return TYPES.get(kinds[0], Any) if len(kinds) == 1 else Any


def _field(prop, required):
    kind = _kind(prop)
    description = prop.get("description", "")
    # the reading of the engine's forms follows the type of the listed schema, for any tool of any server
    if kind is dict:
        kind = Annotated[dict, BeforeValidator(_object_from_text)]
    if required:
        if kind is str:
            kind = Annotated[str, BeforeValidator(_text_from_engine)]
        return kind, Field(..., description=description)
    if kind is str:
        return Annotated[Optional[str], BeforeValidator(_optional_text_from_engine)], Field(default=prop.get("default"), description=description)
    return Optional[kind], Field(default=prop.get("default"), description=description)


def _failure_on_one_line(failure):
    """The exception of a failed call with its text on one line (one_line), since the engine writes that text into the
    scratchpad as the observation of the call (agents/react/agents.py, the call of a tool), as it writes an answer. The
    exception keeps its type, which the engine also reads, and one whose text is already one line is left as raised. A class
    that writes its text from fields of its own rather than from its arguments, pydantic's ValidationError for one, gives a
    ValueError, as a refusal of the argument model does (_PortTool)."""
    text = str(failure)
    line = one_line(text)
    if line == text:
        return failure
    raised, failure.args = failure.args, (line,)
    if str(failure) == line:
        return failure
    failure.args = raised
    return ValueError(line)


class OneLineRefusal:
    """A LangChain tool whose refusal of its arguments reaches the agent on one line, as its answers do, since the engine
    writes the text of the refusal into the scratchpad as the observation of the call. The port's MCP tools (_PortTool) and
    the supervisor's sub-agent tools (agent_builders.SubAgentTool) are such tools."""

    def _parse_input(self, tool_input, tool_call_id):
        try:
            return super()._parse_input(tool_input, tool_call_id)
        except ValidationError as refusal:
            raise ValueError(one_line(str(refusal))) from None


class _PortTool(OneLineRefusal, StructuredTool):
    """The LangChain tool over one MCP tool (mcp_tool)."""


def mcp_tool(name, description, schema, call: Callable[[dict], str]):
    """A LangChain tool over one MCP tool: its name and description, and an argument model with the names and types of
    the tool's input schema, so that ReActXen lists it as name[arguments] and checks the arguments as it checked the paper's
    tools before the call. The answer, and the text of an exception that the call raises, reach the agent on one line
    (one_line, _failure_on_one_line)."""
    required = set(schema.get("required", []))
    properties = schema.get("properties") or {}
    fields = {key: _field(prop, key in required) for key, prop in properties.items()}
    model = create_model(f"{name}_inputs", __config__=ConfigDict(protected_namespaces=()), **fields)
    first = next(iter(fields), None)

    def run(*text, **arguments):
        # a plain string input goes to the first field, the way the paper's one argument tools took it. LangChain validates it
        # there but hands on the raw text, so a text field reads it here as its argument model reads it
        try:
            if text:
                value = text[0]
                if first is not None and _kind(properties[first]) is str:
                    value = getattr(model.model_validate({first: value}), first)
                arguments = {first: value} if fields else {}
            return one_line(call(arguments))
        except Exception as failure:
            raise _failure_on_one_line(failure)

    return _PortTool(name=agent_name(name), description=description or "", args_schema=model, func=run)
