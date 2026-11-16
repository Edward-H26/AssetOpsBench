"""The engine's reading of an Action Input, copied for the tests of the two ports: the pattern with which the vendored ReActXen
splits a key=value input into arguments, and the steps it takes on each pair (agents/react/agents.py, the call of a tool). A test
asserts that the pattern still stands in the engine, so the copy cannot drift. A JSON object as the whole input reaches the tool as
it stands."""
import json
import re

ENGINE_PAIRS = r"(\w+)=\[(.*?)\]|(\w+)=(.+?)(?=, \w+=|\Z)"


def engine_split(argument):
    """The arguments of an Action Input as the engine hands them to a tool. A key=value pair keeps its text and the text None is
    None. A pair written as key=[...] raises as it does in the engine, which strips the list it made, so that form never reaches
    a tool. A JSON object is taken as it stands, and any other text is the one input."""
    argument = argument.strip()
    if "=" in argument:
        out = {}
        for match in re.findall(ENGINE_PAIRS, argument):
            if match[0]:
                key, value = match[0], match[1].split(", ")
            else:
                key, value = match[2], match[3].strip()
            key = key.strip()
            value = value.strip() if value != "None" else None  # a list raises here, as in the engine
            out[key] = value
        return out
    if argument.startswith("{") and argument.endswith("}"):
        return json.loads(argument)
    return argument
