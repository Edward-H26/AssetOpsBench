"""Plan-Execute runner over IBM's paper-era AgentHive, the Plan-Execute of the AssetOpsBench paper as a port.

The released Plan-Execute runner of AssetOpsBench plans once and executes one tool call per step without examples or a
reviewer, which is not the Plan-Execute the paper measured. This runner plays the paper's: AgentHive's planner with its
reviewer, the plan's steps in order on ReActXen ReactReflectAgents with their few-shot examples, and a summarization step
(workflow.py), on the completion route of the Agent-As-Tool port (CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route).

The agents of a step are the sub-agents of the Agent-As-Tool port for the servers of the row (CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.supervisor_and_sub_agents). The
two reference designs share the tools of each server and the in-context examples of each sub-agent, and each keeps the
limits of its own paper-era runner: the step limit of an agent, Agent-Ask, which is off here, and the reply, which here is
the agent's answer. The planner knows an agent by AgentHive's name and description where the agent carries the paper's
description (tests/multi_agent/run.py and tools/*.py of the archive), and by the agent's own otherwise, and sees its task
examples, the Question lines of its few-shots, as ReactReflectAgent derives them. Each step runs ReactReflectAgent with
the keywords of AgentHive's wrapper (STEP_AGENT_OPTIONS).

Three departures from the archive, each on purpose. One reflection round per step agent (REFLECTIONS). A run that raises
is not retried: the archived runner retries a whole scenario up to seven times, which would replay an episode of the
environment on a changed state, so a raised run is recorded as the episode's error, as for the Agent-As-Tool port. The run
ends when the environment ends the episode, as the Agent-As-Tool port's runtime detects it, and a step agent stops at its
first step after that end (stop_when). Each planning round calls no tool of the rig, so it goes to the driver's no_call
entry (on_idle), as a turn without a call does for every design, and so does the pass of the step agents with the summary
when that pass made no counted call.

Reconstructed by this port, where the archive cannot run as written or holds nothing. The archived runner imports two names
that its tool file does not define. The executor and the two agent wrappers follow the text of the public branch. A step
agent has the tools and the examples of the Agent-As-Tool port, so the examples of the agents that the archive lacks are this
port's, and the planner lists for an agent the Question lines of those examples. The planner's prompt lists the agents in
the order of the hub. A sampling run replaces the three temperatures of the summarization by the one temperature of the run.

Step counts: the steps of a run are the plan steps it started (workflow.Workflow.reached), the step in which a run ends
included and the summarization step not. plan_steps is the length of the plan that ran, and ``trajectory`` holds the
attempted MCP calls. The history the archived runner logged, the executed steps that returned and the summarization
step, stays in the record.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from observability import persist_trajectory

from CausalDiagnosis.agents.utils.mcp_sessions import open_sessions
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import sub_agent_registry
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import PAPER_DECODING, CompletionRoute, RunStopped, activated
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.sub_agent_runtime import Runtime
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper import workflow
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.react.agents import ReactReflectAgent

RUNNER_NAME = "plan-execute-paper"
STEP_AGENT_STEPS = 6  # AgentHive ReactReflectAgent.execute_task max_steps=6, WorderOrderAgent the same
# One reflection round for every step agent. AgentHive's runner asks for it, passing reflect_step=1 to every agent
# (tests/multi_agent/run.py), but its ReactReflectAgent takes no such argument and fixes num_reflect_iteration=5
# (agents/react_reflect_agent.py), so the archived runner raises a TypeError as written. Only its WorderOrderAgent takes
# reflect_step (agents/wo_agent.py). The port keeps the runner's stated value, which is also the Agent-As-Tool port's.
REFLECTIONS = 1
# The keywords of AgentHive's wrapper for a step agent (agents/react_reflect_agent.py). Agent-Ask is not among them, so the
# engine's default, off, holds.
STEP_AGENT_OPTIONS = {"max_steps": STEP_AGENT_STEPS, "num_reflect_iteration": REFLECTIONS, "handle_context_length_overflow": True,
                      "apply_loop_detection_check": True, "log_structured_messages": True, "early_stop": True}
# AgentHive's names and descriptions of the agents it has (tests/multi_agent/run.py with tools/skyspark.py, fmsr.py, tsfm.py, wo.py)
AGENT_HIVE = {
    "IoTAgent": ("IoT Data Download", "Can provide information about IoT sites, asset details, sensor data, and retrieve historical "
                                      "data and metadata for various assets and equipment"),
    "FMSRAgent": ("Failure Mode and Sensor Relevancy Expert for Industrial Asset",
                  "Can provide information about failure modes, mapping between failure modes and sensors, and can generate machine "
                  "learning recipes for specific failures"),
    "TSFMAgent": ("Time Series Analytics and Forecasting",
                  "Can assist with time series analysis, forecasting, anomaly detection, and model selection, and supports pretrained "
                  "models, context length specifications, and regression tasks for various time series data"),
    "WOAgent": ("Work Order Management",
                "Can retrieve, analyze, and generate work orders for equipment based on historical data, anomalies, alerts, and "
                "performance metrics, offering recommendations for preventive and corrective actions, including bundling, "
                "prioritization, and predictive maintenance"),
}


@dataclass
class PlanExecutePaperResult:
    question: str
    answer: str
    end: str  # finish, episode_over, cancelled or error
    error: str | None
    plan_steps: int | None
    executed_steps: int = 0  # the plan steps the run started, its step count
    history: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    plan: str = ""
    trajectory: list = field(default_factory=list)
    step_agents: list = field(default_factory=list)


def plan_agent(spec):
    """(name, description) under which the planner knows a sub-agent: AgentHive's for an agent that carries the paper's
    description, and the agent's own for every other one, so that an agent with another role is shown with that role."""
    if spec.name in AGENT_HIVE and spec.description == sub_agent_registry.PAPER[spec.name][0]:
        return AGENT_HIVE[spec.name]
    return spec.name, spec.description


def run_settings(server_names, surface="rig", temperature=None):
    """The settings a run folder records for the port on these servers, known before any episode starts."""
    specs = [s for s in sub_agent_registry.registration(tuple(server_names), surface) if s.kind == "react"]
    return {"form": "paper_era", "engine": "AgentHive 27552e9 on ReActXen da40e4b, vendored", "plans": workflow.PLANS,
            "step_agent_max_steps": STEP_AGENT_STEPS, "reflections": REFLECTIONS, "surface": surface,
            "agents": [plan_agent(s)[0] for s in specs], "decoding": dict(PAPER_DECODING, temperature=0.0 if temperature is None else temperature)}


class PlanExecutePaperRunner:
    """The paper-era Plan-Execute on the launchers of a run; the arguments are those of CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.MetaAgentRunner.

    ``calls`` holds one record per model call and ``trajectory`` one per MCP call, both filled while the run goes."""

    def __init__(self, server_paths, base_url, api_key, model, surface="rig", temperature=None, log_path=None, example_start=None,
                 on_idle=None):
        self._server_paths, self._model, self._surface = dict(server_paths), model, surface
        self._log_path = Path(log_path) if log_path else None
        self.on_idle = on_idle  # the driver's no_call entry, called after each planning round, which calls no tool of the rig
        self._example_start = example_start  # the registered example start, which the diagnosis example of the task tools shows
        self.specs = [s for s in sub_agent_registry.registration(tuple(self._server_paths), surface) if s.kind == "react"]
        self.route = CompletionRoute(base_url, api_key, model, temperature=temperature)
        self.calls = self.route.calls
        self.trajectory: list[dict] = []
        self.flow = None  # the workflow of the run, kept so that a run without a result leaves its plans, history and count
        self.step_agents: list[dict] = []

    def settings(self):
        return run_settings(tuple(self._server_paths), self._surface, self.route.temperature)

    def steps_reached(self):
        """The plan steps the run started, its step count, also for a run that returned no result. None before a workflow
        exists."""
        return self.flow.reached if self.flow is not None else None

    def record(self, result=None):
        """What plan_execute_trajectory.json holds of the run as it stands: every plan with its review, the plan that ran,
        the history, the requests and answers of the step agents and the two counts. A run without a result ends as
        no_result. None before a workflow exists."""
        flow = self.flow
        if flow is None:
            return None
        return {"end": result.end if result is not None else "no_result", "plan": flow.plan,
                "plan_steps": len(flow.steps) if flow.plan else None, "plans": flow.plans, "history": flow.history,
                "executed_steps": flow.reached, "step_agents": self.step_agents}

    async def run(self, question):
        async with open_sessions(self._server_paths) as sessions:
            definitions = sessions.tools
            runtime = Runtime(self.route, sessions, definitions, asyncio.get_running_loop(), self._model,
                              example_start=self._example_start, on_idle=self.on_idle)
            refusal = runtime.prepare(self.specs)
            if refusal is None:
                self.trajectory = runtime.trajectory
                log = open(self._log_path, "w", encoding="utf-8") if self._log_path else open(os.devnull, "w")
                try:
                    with log, contextlib.redirect_stdout(log):
                        result = await asyncio.to_thread(self._play_then_close, question, runtime)
                finally:
                    self.route.stop("cancelled")  # a run cancelled from outside stops its thread at the next call
        if refusal is not None:
            self.route.close()  # no model call was made
            raise refusal
        persist_trajectory(runner_name=RUNNER_NAME, model=self._model, question=question, answer=result.answer, trajectory=result.history)
        return result

    def _play_then_close(self, question, runtime):
        """_play on the agents' thread, then the client of the route closed. Every model call of the run is made on this
        thread, so no call is in flight when the client closes, also when the run was cut from outside."""
        try:
            return self._play(question, runtime)
        finally:
            self.route.close()

    def _agents(self, runtime, step_agents):
        def executor(spec):
            def execute(user_input):
                tools = runtime.tools_of(spec)
                outer, runtime._request = runtime._request, user_input
                try:
                    with self.route.agent(spec.name):
                        agent = ReactReflectAgent(question=user_input, key="", cbm_tools=tools,
                                                  react_llm_model_id=self._model, reflect_llm_model_id=self._model,
                                                  react_example=runtime.examples_of(spec), stop_when=runtime.episode_ended,
                                                  **STEP_AGENT_OPTIONS)
                        agent.run()
                finally:
                    runtime._request = outer
                step_agents.append({"agent": spec.name, "request": user_input, "answer": agent.answer,
                                    "react_steps": agent.per_round_info[-1]["step"] if agent.per_round_info else agent.step_n,
                                    "metric": agent.metric})
                return agent.answer
            return execute

        agents = []
        for spec in self.specs:
            name, description = plan_agent(spec)
            few_shots = runtime.examples_of(spec)
            agents.append(workflow.PlanAgent(name, description, workflow.task_examples(few_shots), executor(spec)))
        return agents

    def _play(self, question, runtime):
        step_agents, error = self.step_agents, None
        with activated(self.route):
            flow = self.flow = workflow.Workflow(question, self._agents(runtime, step_agents), self._model, self.route,
                                                 round_ended=lambda: runtime.iteration_end(False))
            try:
                flow.make_plan()
                planned = runtime.answered()
                flow.execute()
                # the pass of the step agents with the summary is one iteration of its own, charged when it made no counted call
                runtime.iteration_end(runtime.answered() > planned)
                end = self.route.stop_reason or "finish"
            except RunStopped as stop:
                end = str(stop.args[0]) if stop.args else "cancelled"
            except Exception as exc:  # noqa: BLE001 - a failed run is a result of the benchmark
                end, error = "error", f"{type(exc).__name__}: {exc}"
        return PlanExecutePaperResult(question=question, answer=flow.answer(), end=end, error=error,
                                      plan_steps=len(flow.steps) if flow.plan else None, executed_steps=flow.reached,
                                      history=flow.history, plans=flow.plans, plan=flow.plan, trajectory=runtime.trajectory,
                                      step_agents=step_agents)
