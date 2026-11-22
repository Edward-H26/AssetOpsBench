from __future__ import annotations
import json
import logging
from typing import Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.react.agents import ReactReflectAgent
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.tools.jsonreader.jsonreader import JSONReader
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.tools.jsonreader.jsonwrapper import JSONWrapperFunctions
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.utils.tool_description import get_tool_description
from . import sub_agent_registry
from .react_tool_adapter import OneLineRefusal, one_line


_log = logging.getLogger(__name__)
FAILURE_MESSAGE = "An error occurred while processing the request."  # the paper's tools on an exception


# ----- the tools of the supervisor -----
class _SubAgentInputs(BaseModel):
    request: str = Field(description="")


def _inputs(spec):
    return create_model(f"{spec.name}Inputs", request=(str, Field(description=spec.request)))


class SubAgentTool(OneLineRefusal, BaseTool):
    """A paper-era agent tool: one request in, the wrapper's JSON reply out, JSON encoded once more. A refusal of its argument
    model reaches the supervisor on one line (OneLineRefusal)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    args_schema: Any = _SubAgentInputs
    response_format: str = "text"
    spec: Any = None
    runtime: Any = None  # the runner's Runtime, or None for a supervisor built for inspection
    parent_agent: Optional[Any] = None
    parent_model_id: Optional[Any] = None

    def _run(self, request: str) -> str:
        try:
            if self.spec.kind == "stub":
                raise RuntimeError("the RuleLogicAgent's package nl2anomaliesrules is not public, so the port has no rule agent")
            return json.dumps(self.runtime.run_sub_agent(self.spec, request, self.parent_agent, self.parent_model_id))
        except Exception as exc:  # the paper's tools answer a failure instead of raising (IoTAgent._run in IoTTool.py)
            _log.warning("%s failed: %s", self.name, exc)
            # the paper's text of the failure, on one line as every answer of a tool of the port (react_tool_adapter.one_line)
            return one_line(json.dumps({"status": "failure", "error": str(exc), "message": FAILURE_MESSAGE}, indent=2))


def supervisor_tools(specs, runtime):
    tools = []
    for spec in specs:
        if spec.kind == "jsonreader":
            tools.append(JSONReader(functions=JSONWrapperFunctions()))
        else:
            # with a runtime the description follows the tools the agent holds; without one, as for an inspection, the fixed text
            description = spec.description_for(runtime.tools_of(spec)) if runtime is not None else spec.description
            tools.append(SubAgentTool(name=spec.name, description=description, args_schema=_inputs(spec), spec=spec, runtime=runtime))
    return tools


def build_supervisor(question, specs, runtime, model, stop_when=None):
    """getAgent of src/tmp/meta_agent/utils.py with the arguments MetaAgent.run passes: AgentHub's agents and tools, the
    merged examples, 10 steps, one round, Agent-Ask off and the debug printout on. stop_when stops the supervisor at its
    first step after the environment ended the episode. On a run with the CausalDiagnosis MCP the merged examples show the asset of
    the run (Runtime.mcp_row) and the supervisor has no step limit of its own (sub_agent_registry.step_cap)."""
    tools = supervisor_tools(specs, runtime)
    servers = tuple(runtime.definitions) if runtime is not None else ()
    return ReactReflectAgent(question=question, key="", cbm_tools=tools, tool_desc=get_tool_description(tools),
                             max_steps=sub_agent_registry.step_cap(sub_agent_registry.SUPERVISOR_STEPS, servers),
                             react_llm_model_id=model, reflect_llm_model_id=model, react_example=sub_agent_registry.merged_examples(specs, runtime.mcp_row() if runtime is not None else None),
                             num_reflect_iteration=sub_agent_registry.REFLECTIONS, handle_context_length_overflow=True, apply_loop_detection_check=True,
                             log_structured_messages=True, early_stop=True, enable_agent_ask=False, debug=True, stop_when=stop_when)


def build_sub_agent(spec, request, tools, model, own_examples, stop_when=None, max_steps=None):
    """The paper's wrappers call their agent function with the request, the parent's model, reflect_step 1 and Agent-Ask on,
    and the agent functions are getAgent with the agent's tools, examples and step limit. own_examples is the text of the
    agent's examples for these tools (Runtime.examples_of), and max_steps the step limit of the run when it is not the agent's
    own (sub_agent_registry.step_cap). The engine renders its prompt template once per process, for the
    supervisor, so the line that Agent-Ask adds to a prompt reaches no sub-agent, as in the source."""
    return ReactReflectAgent(question=request, key="", cbm_tools=tools, tool_desc=get_tool_description(tools),
                             max_steps=spec.max_steps if max_steps is None else max_steps,
                             react_llm_model_id=model, reflect_llm_model_id=model, react_example=own_examples,
                             num_reflect_iteration=1, handle_context_length_overflow=True, apply_loop_detection_check=True,
                             log_structured_messages=True, early_stop=True, enable_agent_ask=True, debug=False, stop_when=stop_when)
