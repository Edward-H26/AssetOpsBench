"""The serving template of granite with a tool result shown under the role tool_response: granite's own template from
the model's tokenizer config, changed in one place, rendered as transformers renders a chat template, and the table of
model adapters that the serving job reads."""
import json
from pathlib import Path

import pytest
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: F401
from CausalDiagnosis.agents.utils import model_adapters  # noqa: E402

CONFIG = Path(__file__).parent / "data" / "granite-3.3-8b-instruct_tokenizer_config.json"
TOOLS = [{"type": "function", "function": {"name": "sites", "description": "List the sites.", "parameters": {"type": "object", "properties": {}}}}]
CALL = '<|tool_call|>[{"name": "sites", "arguments": {}}]'


def render(template, messages, tools=None):
    """The chat template rendered with the options and filters that transformers gives a chat template."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda value, indent=None: json.dumps(value, ensure_ascii=False, indent=indent)
    env.globals["strftime_now"] = lambda fmt: "November 07, 2026"
    return env.from_string(template).render(messages=messages, tools=tools, add_generation_prompt=True)


def granite():
    return json.loads(CONFIG.read_text())["chat_template"]


def test_a_tool_result_is_shown_as_tool_response():
    history = [{"role": "system", "content": "Prompt."}, {"role": "user", "content": "Which sites?"},
               {"role": "assistant", "content": CALL}, {"role": "tool", "content": '{"sites": ["MAIN"]}'}]
    mapped = render(model_adapters.tool_response_template(granite()), history, TOOLS)
    assert '<|start_of_role|>tool_response<|end_of_role|>{"sites": ["MAIN"]}<|end_of_text|>' in mapped
    assert "<|start_of_role|>tool<|end_of_role|>" not in mapped
    assert "<|start_of_role|>tool<|end_of_role|>" in render(granite(), history, TOOLS)


def test_everything_but_the_tool_role_renders_as_granite_does():
    template = model_adapters.tool_response_template(granite())
    histories = [
        [{"role": "user", "content": "Which sites?"}],
        [{"role": "system", "content": "Prompt."}, {"role": "user", "content": "Which sites?"}, {"role": "assistant", "content": CALL}],
    ]
    for history in histories:
        for tools in (None, TOOLS):
            assert render(template, history, tools) == render(granite(), history, tools)
    with_tool = histories[1] + [{"role": "tool", "content": "{}"}]
    assert render(template, with_tool, TOOLS).replace("tool_response<|end_of_role|>", "tool<|end_of_role|>") == render(granite(), with_tool, TOOLS)


def test_a_template_without_the_role_expression_is_refused():
    with pytest.raises(ValueError):
        model_adapters.tool_response_template("{% for message in messages %}{{ message['content'] }}{% endfor %}")
    with pytest.raises(ValueError):
        model_adapters.served_template("{% for message in messages %}{{ message['content'] }}{% endfor %}")


def test_the_table_serves_granite_with_the_rewrite_of_its_entry():
    assert [adapter.model for adapter in model_adapters.MODEL_ADAPTERS][0] == "granite-3.3-8b-instruct"
    assert model_adapters.served_template(granite()) == model_adapters.tool_response_template(granite())


def test_the_command_writes_the_template_and_refuses_another_model(tmp_path):
    out = tmp_path / "chat_template.jinja"
    assert model_adapters.main([str(CONFIG), str(out)]) == 0
    assert out.read_text(encoding="utf-8") == model_adapters.tool_response_template(granite())
    other = tmp_path / "tokenizer_config.json"
    other.write_text(json.dumps({"chat_template": "{{ messages[0]['content'] }}"}))
    assert model_adapters.main([str(other), str(tmp_path / "other.jinja")]) == 3
    assert not (tmp_path / "other.jinja").exists()
