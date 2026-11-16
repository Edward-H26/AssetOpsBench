"""Adapters of a served model to the requests of the benchmark, one entry per model.

An entry is recognized by the chat template of the model's tokenizer config, since the serving job may serve a model from
a folder whose name says nothing of the model. The first entry is granite-3.3-8b-instruct: its own template prints every
message under the role it carries, and SGLang's OpenAI endpoint accepts no role but ``tool`` for a tool result, so the
model would read ``<|start_of_role|>tool<|end_of_role|>``. The vLLM reference template for granite and IBM's Granite
documentation show a tool result as a tool response, so the serving job writes granite's own template with that one
expression changed and serves the model with it. A template that no entry recognizes is kept, and the job then serves the
model's own template. A further model is one more entry of MODEL_ADAPTERS.

usage: python3 model_adapters.py MODEL_FOLDER/tokenizer_config.json OUT.jinja
exits 0 when it wrote OUT.jinja and 3 when the model keeps its own template.
"""

import json
import sys
from dataclasses import dataclass
from typing import Callable

ROLE = "'<|start_of_role|>' + message['role'] + '<|end_of_role|>'"
TOOL_RESPONSE = "'<|start_of_role|>' + ('tool_response' if message['role'] == 'tool' else message['role']) + '<|end_of_role|>'"


def tool_response_template(chat_template: str) -> str:
    """granite's chat template with a message of role ``tool`` printed under ``tool_response``."""
    if chat_template.count(ROLE) != 1:
        raise ValueError(f"the template holds the role expression {chat_template.count(ROLE)} times, expected once")
    if "\\n" in chat_template:
        # SGLang's loader turns a literal backslash n of a template file into a newline
        raise ValueError("the template holds a literal backslash n")
    return chat_template.replace(ROLE, TOOL_RESPONSE)


@dataclass(frozen=True)
class ModelAdapter:
    """One served model: its name and the rewrite of its chat template, which raises ValueError on a template it does not
    recognize."""

    model: str
    chat_template: Callable[[str], str]


MODEL_ADAPTERS = (ModelAdapter("granite-3.3-8b-instruct", tool_response_template),)


def served_template(chat_template: str) -> str:
    """The template the model is served with: the rewrite of the first entry that recognizes the template. ValueError with
    the reasons of every entry when none does."""
    reasons = []
    for adapter in MODEL_ADAPTERS:
        try:
            return adapter.chat_template(chat_template)
        except ValueError as exc:
            reasons.append(str(exc))
    raise ValueError("; ".join(reasons))


def main(argv: list[str]) -> int:
    config_path, out_path = argv
    with open(config_path, encoding="utf-8") as handle:
        chat_template = json.load(handle).get("chat_template") or ""
    try:
        template = served_template(chat_template)
    except ValueError as exc:
        print(f"{config_path}: {exc}, so the model's own template stays", file=sys.stderr)
        return 3
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(template)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
