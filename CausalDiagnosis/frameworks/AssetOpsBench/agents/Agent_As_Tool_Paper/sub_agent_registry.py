from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from typing import Any
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import in_context_examples as examples
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import manual as repair_manual
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import child_asset
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents import ProfileDocuments, class_keys


SUPERVISOR_STEPS = 10  # meta_agent.py run(inner_loop_step=10)
REFLECTIONS = 1  # tests/test_singleagent.py NUM_REFLECT, passed as getAgent's reflect_step
# the CausalDiagnosis MCP advises and is no server of the environment, so a call of it is no counted call (Runtime.answered)
CAUSAL_SERVER = "causal_diagnosis"
# On a run whose servers list the CausalDiagnosis MCP both ports run with the limits of the CausalDiagnosis agent (with_mcp): every
# request asks for the room the served context leaves its prompt (text_completion_route.CompletionRoute, reply_room), no agent
# has a step limit of its own (NO_STEP_CAP), and every ReAct step without a counted call takes a no_call step of the episode's
# step cap, as a planning turn of the CausalDiagnosis agent without one does (sub_agent_runtime.ReactStepCharge). Every other
# run keeps the paper's form.
PAPER_FORM, MCP_FORM = "paper_era", "ibm_with_causal_mcp"
NO_STEP_CAP = sys.maxsize  # the step limit of an agent that has none of its own, so the step cap of the episode governs
# the least served context of a run with the CausalDiagnosis MCP, 48k tokens: the driver refuses a smaller one, or an endpoint that
# names none, before the run folder (agent_runs.PaperAgent.served_refusal), since on such a run no reply is capped and no agent has
# a step limit of its own, and a prompt that leaves no room ends the run with the end context (CompletionRoute.room_of)
SERVED_CONTEXT_MIN = 49152
# the rule of the no_call steps of a run with the CausalDiagnosis MCP as run.json records it
NO_CALL_RULE = ("one for each ReAct step of a sub-agent or a step agent without a counted call, one for each planning round of "
                "Plan-Execute, and one for a supervisor step or the closing pass of Plan-Execute only when nothing in it took a step of "
                "the step cap")


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    request: str  # the description of the request field
    servers: tuple = ()  # the MCP servers whose tools the agent uses, the first of which decides whether it is registered
    max_steps: int = 10
    examples: Any = ""  # the agent's own examples, or a function of its tools, the example start and the McpRow of the run
    supervisor_examples: Any = ""  # the supervisor's examples of the agent, or a function of the McpRow of the run
    speaker: str = ""  # the name the agent gives itself in its summary
    kind: str = "react"  # react, stub (RuleLogicAgent) or jsonreader
    describe: Any = None  # the description as a function of the names of the agent's tools, else the fixed description

    def description_for(self, tools):
        """The description the supervisor reads for the tools the agent holds, the LangChain tools that Runtime.tools_of builds."""
        return self.describe({tool.name for tool in tools}) if callable(self.describe) else self.description

    def own_examples(self, tools, example_start=None, row=None):
        """The agent's own examples for the tools it has, the LangChain tools that Runtime.tools_of builds, the example start
        of the run, the levels that the diagnosis example shows, and row, the McpRow of a run that lists the CausalDiagnosis MCP
        (mcp_row), None on every other run."""
        return self.examples(tools, example_start, row) if callable(self.examples) else self.examples

    def supervisor_examples_for(self, row=None):
        """The supervisor's examples of the agent on a run whose McpRow is row, None on a run without the CausalDiagnosis MCP."""
        return self.supervisor_examples(row) if callable(self.supervisor_examples) else self.supervisor_examples


def _per_tool(table):
    """The examples of a table for the tools an agent has, in the order of the table. The diagnosis example is written for the
    parameters of the tool's own model, in the order of its listed schema, and the example start of the run. On a run with the
    CausalDiagnosis MCP every example shows the asset of the run (examples.on_mcp_row)."""
    def own(tools, example_start, row=None):
        shown = {tool.name: tool for tool in tools}
        texts = []
        for name, text in table.items():
            if name not in shown:
                continue
            if name == examples.DIAGNOSIS_TOOL:
                text = examples.diagnosis_example(list(shown[name].args_schema.model_fields), example_start)
            texts.append(text if row is None else examples.on_mcp_row(text, row))
        return "\n\n".join(texts)
    return own


def _own_written_here(table, shown=examples.on_mcp_row):
    """The own examples of an agent that this port wrote, as a function of the tools, the example start and the McpRow of the
    run: as they stand on a run without the CausalDiagnosis MCP and as shown writes them for the asset of the run on a run with it."""
    text = "\n\n".join(table)
    return lambda tools, example_start, row=None: text if row is None else shown(text, row)


def _supervisor_written_here(text):
    """The supervisor's examples of an agent that this port wrote, as a function of the McpRow of the run, as _own_written_here."""
    return lambda row=None: text if row is None else examples.on_mcp_row(text, row)


def with_mcp(servers):
    """Whether a run on these servers lists the CausalDiagnosis MCP, and so runs with the limits of the CausalDiagnosis agent."""
    return CAUSAL_SERVER in servers


def step_cap(paper_cap, servers):
    """The step limit of an agent on a run on these servers: the paper's, or none of its own on a run with the CausalDiagnosis MCP."""
    return NO_STEP_CAP if with_mcp(servers) else paper_cap


def recorded_cap(paper_cap, servers):
    """The step limit of an agent as run.json records it: the paper's, or null for none of its own."""
    return None if with_mcp(servers) else paper_cap


def mcp_row(servers, asset_id, example_start):
    """The McpRow of a run on these servers where they list the CausalDiagnosis MCP, None on every other run: the site of the asset
    of the run, the site of its episodes on the AssetOpsBench rows as the metering layer reads it from the profile
    (metering_layer.meter_config), the field of the first channel of its first sensor group, the first component that the
    example start of the run holds off its healthy level, the start whose levels the diagnosis example shows, with the child asset
    on which a work order names it (aob_tool_contract.child_asset), what the profile examples show of the asset under its default
    configuration, the levels of the example start and the healthy levels, a second field and the plain statistic, the repair manual
    of the component under the default configuration (lookup_fix's answer without its emergency option), and a class and the
    failure modes that fmsr files under it, the asset's own class where it files codes (profile_documents)."""
    if not with_mcp(servers):
        return None
    if asset_id is None:
        raise ValueError("the examples of a run with the CausalDiagnosis MCP show the asset of the run, which the driver names")
    profile = get_profile(asset_id)
    first_group = next(iter(profile.observation_kinds.values()))
    component = next((c for c in profile.components if c in (example_start or {}) and example_start[c] != profile.healthy[c]), None)
    if component is None:
        raise ValueError(f"the work order examples of a run with the CausalDiagnosis MCP name a faulty component of the example start of "
                         f"the run, and the start {example_start} holds none of {asset_id}")
    config = profile.default_config()
    second = first_group["channels"][1] if len(first_group["channels"]) > 1 else first_group["channels"][0]
    indicating = [profile.field_name(s) for s in profile.channels if component in profile.knowledge["indicates"].get(s, ())]
    manual = {key: value for key, value in repair_manual(config, component, indicating).items() if key != "emergency"}
    # the FMSR example asks for the asset's own class, or, where fmsr files no code under it, for the first class with codes among the
    # component's and the other components'
    filed, classes = {d["asset_class"]: d["failure_modes"] for d in ProfileDocuments(profile, None).failure_modes()}, class_keys(profile)
    keys = [None, component, *(c for c in profile.components if c != component)]
    whole = next((classes[k] for k in keys if filed.get(classes[k])), classes[None])
    modes = filed.get(whole, [])
    return examples.McpRow(site=profile.text["fallback"]["site"], field=profile.field_name(first_group["channels"][0]), asset=asset_id,
                           component=component, child=child_asset(asset_id, component), asset_type=profile.registry_entry()["asset_type"],
                           components=tuple(profile.components), children=tuple(child_asset(asset_id, c) for c in profile.components),
                           sensor_groups=tuple((group, tuple(profile.field_name(s) for s in sensors)) for group, sensors in config.sensor_groups.items()),
                           budget=config.budget, horizon=config.horizon,
                           levels=tuple((c, example_start[c]) for c in profile.components if c in example_start),
                           healthy=tuple((c, profile.healthy[c]) for c in profile.components), second_field=profile.field_name(second),
                           statistic=profile.plain_statistic(), manual=json.dumps(manual), asset_class=whole, failure_modes=tuple(modes))


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
        _spec("TaskAgent", ("task",), SUPERVISOR_STEPS, _per_tool(examples.TASK_TOOLS), _supervisor_written_here(examples.TASK_SUPERVISOR),
              "Task Agent", describe=_task_description),
        _spec("FMSRAgent", ("fmsr",), 10, _own_written_here(examples.FMSR_RIG), examples.PAPER_SUPERVISOR["FMSRAgent"], "FMSR Agent"),
        _spec("WOAgent", ("wo",), 20, _own_written_here(examples.WO_RIG, examples.orders_on_mcp_row), examples.PAPER_SUPERVISOR["WOAgent"], "Work Order Agent"),
        _spec("CausalDiagnosisAgent", (CAUSAL_SERVER,), SUPERVISOR_STEPS, _per_tool(examples.CAUSAL_TOOLS),
              _supervisor_written_here(examples.CAUSAL_SUPERVISOR), "Causal Diagnosis Agent"),
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


def merged_examples(specs, row=None):
    """AgentHub.merge_all_examples: every agent's supervisor examples in registration order, one newline apart, on a run whose
    McpRow is row, None on a run without the CausalDiagnosis MCP."""
    shown = (s.supervisor_examples_for(row) for s in specs)
    return "\n".join(text for text in shown if text)
