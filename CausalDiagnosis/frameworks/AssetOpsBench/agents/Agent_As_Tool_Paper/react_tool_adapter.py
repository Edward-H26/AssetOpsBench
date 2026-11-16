from __future__ import annotations
import json
from typing import Annotated, Any, Callable, Optional
from langchain_core.tools import StructuredTool
from pydantic import BeforeValidator, ConfigDict, Field, create_model


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


def _text_from_engine(value):
    """A required text argument: the engine's forms for no value are the empty text, and a list is its items joined by
    commas. Any other value passes unchanged."""
    if _no_value(value):
        return ""
    if isinstance(value, list):
        return ",".join(str(item).strip() for item in value)
    return value


def _optional_text_from_engine(value):
    """An optional text argument: the engine's forms for no value are no value, so that the server's default holds, and a
    list is its items joined by commas. Any other value passes unchanged."""
    if _no_value(value):
        return None
    if isinstance(value, list):
        return ",".join(str(item).strip() for item in value)
    return value


def _field(prop, required):
    if isinstance(prop.get("type"), str):
        kind = TYPES.get(prop["type"], Any)
    else:
        kinds = [p.get("type") for p in prop.get("anyOf", []) if p.get("type") != "null"]
        kind = TYPES.get(kinds[0], Any) if len(kinds) == 1 else Any
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


def mcp_tool(name, description, schema, call: Callable[[dict], str]):
    """A LangChain tool over one MCP tool: its name and description, and an argument model with the names and types of
    the tool's input schema, so that ReActXen lists it as name[arguments] and checks the arguments as it checked the paper's
    tools before the call."""
    required = set(schema.get("required", []))
    fields = {key: _field(prop, key in required) for key, prop in (schema.get("properties") or {}).items()}
    model = create_model(f"{name}_inputs", __config__=ConfigDict(protected_namespaces=()), **fields)

    def run(*text, **arguments):
        # a plain string input goes to the first field, as LangChain validated it, the way the paper's one argument tools took it
        if text:
            arguments = {next(iter(fields)): text[0]} if fields else {}
        return call(arguments)

    return StructuredTool(name=agent_name(name), description=description or "", args_schema=model, func=run)
