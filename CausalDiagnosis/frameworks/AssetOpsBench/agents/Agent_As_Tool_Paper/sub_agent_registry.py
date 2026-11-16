from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import in_context_examples as examples


SUPERVISOR_STEPS = 10  # meta_agent.py run(inner_loop_step=10)
REFLECTIONS = 1  # tests/test_singleagent.py NUM_REFLECT, passed as getAgent's reflect_step
# the CausalDiagnosis MCP advises and is no server of the environment, so a call of it is no counted call (Runtime.answered)
CAUSAL_SERVER = "causal_diagnosis"


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    request: str  # the description of the request field
    servers: tuple = ()  # the MCP servers whose tools the agent uses, the first of which decides whether it is registered
    max_steps: int = 10
    examples: Any = ""  # the agent's own examples, or a function of the names of its tools
    supervisor_examples: str = ""
    speaker: str = ""  # the name the agent gives itself in its summary
    kind: str = "react"  # react, stub (RuleLogicAgent) or jsonreader
    describe: Any = None  # the description as a function of the names of the agent's tools, else the fixed description

    def description_for(self, tools):
        """The description the supervisor reads for the tools the agent holds, the LangChain tools that Runtime.tools_of builds."""
        return self.describe({tool.name for tool in tools}) if callable(self.describe) else self.description

    def own_examples(self, tools, example_start=None):
        """The agent's own examples for the tools it has, the LangChain tools that Runtime.tools_of builds, and the example
        start of the run, the levels that the diagnosis example shows."""
        return self.examples(tools, example_start) if callable(self.examples) else self.examples


def _per_tool(table):
    """The examples of a table for the tools an agent has, in the order of the table. The diagnosis example is written for the
    parameters of the tool's own model, in the order of its listed schema, and the example start of the run."""
    def own(tools, example_start):
        shown = {tool.name: tool for tool in tools}
        texts = []
        for name, text in table.items():
            if name not in shown:
                continue
            if name == examples.DIAGNOSIS_TOOL:
                text = examples.diagnosis_example(list(shown[name].args_schema.model_fields), example_start)
            texts.append(text)
        return "\n\n".join(texts)
    return own


PAPER = {  # the descriptions and request fields of the paper's tools, IoTTool.py, TSFMTool.py, FMSRTool.py, WorkOrderTool.py, RuleLogicTool.py
    "IoTAgent": ("The IoTAgent provides information about IoT sites, asset details, sensor data, and retrieves historical data and "
                 "metadata for various assets and equipment.", "An Internet of Things (IoT) request for information."),
    "TSFMAgent": ("The Time Series Agent can assist with time series analysis, forecasting, anomaly detection, and model selection, and "
                  "supports pretrained models, context length specifications, and regression tasks for various time series data.",
                  "a Time Series Foundation Model request for information"),
    "FMSRAgent": ("The FMSRAgent provides information about failure modes, the mapping between failure modes and sensors, detection "
                  "strategies for assets like chillers and wind turbines, and can generate detection recipes, anomaly detection plans, "
                  "and machine learning recipes for specific failures.", "A Failure Mode Sensor Relevance (FMSR) request for information."),
    "WOAgent": ("The Work Order (WO) agent can retrieve, analyze, and generate work orders for equipment based on historical data, "
                "anomalies, alerts, and performance metrics, offering recommendations for preventive and corrective actions, including "
                "bundling, prioritization, and predictive maintenance.", "A Work Order (WO) request for asset-related information."),
    "RuleLogicAgent": ("The RuleAgent can generate executable code from a natural language description of a monitoring rule. It expects "
                       "knowledge of available sensors for the asset, and can execute rules on sensor data stored in a file.",
                       "A Rule Logic Agent request describing the logic of a monitoring rule."),
}
OURS = {
    "TaskAgent": ("The TaskAgent works on the task itself. It reads the asset profile with the rules of the task, records a diagnosis of "
                  "the asset or a declaration that the evidence is insufficient, and ends the task. Where the task has a clock it also "
                  "reads the clock and waits for notifications, and where repair rules hold it looks up the repair manual.",
                  "A request to the task tools of the asset."),
    "CausalDiagnosisAgent": ("The CausalDiagnosisAgent gives the advice of the CausalDiagnosis MCP. It lists the evidence released so far, "
                             "ranks the candidate root causes of the readings, estimates the effect of an intervention and verifies the "
                             "outcome of a repair against its prediction.", "A request for the advice of the CausalDiagnosis MCP."),
}
DECLARATION_TOOL, CLOCK_TOOLS, MANUAL_TOOL = "declare_insufficient_evidence", ("get_current_time", "wait_for_notification"), "lookup_fix"


def _task_description(names):
    """TaskAgent's description from the names of the tools it holds (a text names only what the row serves): the
    declaration, the clock and the repair manual each where its tool is held, so that with every task tool it is the text of OURS
    and on the level without the CausalDiagnosis MCP it names the profile, the diagnosis and the end of the task alone."""
    text = ("The TaskAgent works on the task itself. It reads the asset profile with the rules of the task, records a diagnosis of the asset"
            + (" or a declaration that the evidence is insufficient" if DECLARATION_TOOL in names else "") + ", and ends the task.")
    also = (["Where the task has a clock it also reads the clock and waits for notifications"] if all(t in names for t in CLOCK_TOOLS) else [])         + (["where repair rules hold it looks up the repair manual"] if MANUAL_TOOL in names else [])
    if also:
        sentence = ", and ".join(also)
        text += " " + sentence[0].upper() + sentence[1:] + "."
    return text


def _spec(name, servers, max_steps, own, supervisor, speaker, **kwargs):
    description, request = (PAPER.get(name) or OURS[name])
    return SubAgentSpec(name, description, request, servers, max_steps, own, supervisor, speaker, **kwargs)


SURFACES = {
    "rig": (
        _spec("IoTAgent", ("iot",), 15, "\n\n".join(examples.IOT_RIG), examples.PAPER_SUPERVISOR["IoTAgent"], "IoT Agent"),
        _spec("TaskAgent", ("task",), SUPERVISOR_STEPS, _per_tool(examples.TASK_TOOLS), examples.TASK_SUPERVISOR, "Task Agent",
              describe=_task_description),
        _spec("FMSRAgent", ("fmsr",), 10, "\n\n".join(examples.FMSR_RIG), examples.PAPER_SUPERVISOR["FMSRAgent"], "FMSR Agent"),
        _spec("WOAgent", ("wo",), 20, "\n\n".join(examples.WO_RIG), examples.PAPER_SUPERVISOR["WOAgent"], "Work Order Agent"),
        _spec("CausalDiagnosisAgent", (CAUSAL_SERVER,), SUPERVISOR_STEPS, _per_tool(examples.CAUSAL_TOOLS), examples.CAUSAL_SUPERVISOR,
              "Causal Diagnosis Agent"),
    ),
    "repro": (
        _spec("IoTAgent", ("iot", "utilities"), 15, "\n\n".join(examples.IOT_REPRO), examples.PAPER_SUPERVISOR["IoTAgent"], "IoT Agent"),
        _spec("TSFMAgent", ("tsfm", "utilities"), 6, "\n\n".join(examples.TSFM_REPRO), examples.PAPER_SUPERVISOR["TSFMAgent"], "TSFM Agent"),
        _spec("FMSRAgent", ("fmsr",), 10, "\n\n".join(examples.FMSR_REPRO), examples.PAPER_SUPERVISOR["FMSRAgent"], "FMSR Agent"),
        _spec("WOAgent", ("wo",), 20, "\n\n".join(examples.WO_REPRO), examples.PAPER_SUPERVISOR["WOAgent"], "Work Order Agent"),
        _spec("RuleLogicAgent", (), 0, "", examples.PAPER_SUPERVISOR["RuleLogicAgent"], "Rule Logic Agent", kind="stub"),
        SubAgentSpec("jsonreader", "", "", kind="jsonreader"),
    ),
}
# servers of main-0.x that no paper-era agent used, which the reproduction may list and which stay unregistered
UNUSED = {"repro": ("vibration",), "rig": ()}


def register_surface(name, specs, unused=()):
    """Add a surface of sub-agents, as a port that brings agents of its own does: its specs in their order and the servers it
    may list without an agent. A name that a surface holds already is refused."""
    if name in SURFACES:
        raise ValueError(f"the surface {name!r} is already registered")
    SURFACES[name], UNUSED[name] = tuple(specs), tuple(unused)
    return SURFACES[name]


def registration(servers, surface):
    """The supervisor's agents and tools for the servers of a run, in the paper's order."""
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}: {' or '.join(SURFACES)}")
    specs = [s for s in SURFACES[surface] if s.kind != "react" or s.servers[0] in servers]
    covered = {name for s in specs for name in s.servers} | set(UNUSED[surface])
    unknown = [name for name in servers if name not in covered]
    if unknown:
        raise ValueError(f"no paper-era sub-agent works over the server(s) {unknown} on the {surface} surface, since the port "
                         "plays AssetOpsBench's servers")
    return specs


def merged_examples(specs):
    """AgentHub.merge_all_examples: every agent's supervisor examples in registration order, one newline apart."""
    return "\n".join(s.supervisor_examples for s in specs if s.supervisor_examples)
