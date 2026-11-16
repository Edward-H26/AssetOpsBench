from __future__ import annotations
import asyncio
import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from observability import persist_trajectory
from CausalDiagnosis.agents.utils.mcp_sessions import open_sessions
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import agent_builders, sub_agent_registry
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import CONTEXT_END, PAPER_DECODING, CompletionRoute, RunStopped, activated
from CausalDiagnosis.frameworks.AssetOpsBench.agents.ReActXen.agents.react.agents import ReflexionStrategy
from . import sub_agent_runtime


RUNNER_NAME = "agent-as-tool-paper"


@dataclass
class MetaAgentResult:
    question: str
    answer: str
    end: str  # finish, max_steps, loop, context, episode_over, cancelled or error
    error: str | None
    react_steps: int | None
    trajectory: list = field(default_factory=list)
    sub_agents: list = field(default_factory=list)
    react_trajectory: dict = field(default_factory=dict)


def run_settings(server_names, surface="rig", temperature=None):
    """The settings a run folder records for the port on these servers, known before any episode starts."""
    specs = sub_agent_registry.registration(tuple(server_names), surface)
    return {"form": "paper_era", "engine": "ReActXen da40e4b, vendored", "supervisor_max_steps": sub_agent_registry.SUPERVISOR_STEPS,
            "reflections": sub_agent_registry.REFLECTIONS, "surface": surface, "sub_agents": [s.name for s in specs],
            "sub_agent_max_steps": {s.name: s.max_steps for s in specs if s.kind == "react"},
            "decoding": dict(PAPER_DECODING, temperature=0.0 if temperature is None else temperature)}


class MetaAgentRunner:
    """The paper-era Agent-As-Tool on the launchers of a run.

    Args:
        server_paths: ``{server name: launcher path}``.
        base_url: the OpenAI compatible endpoint of SGLang, for example ``http://node:port/v1``.
        api_key: the key the endpoint expects.
        model: the served model name.
        surface: ``rig`` for the rows of the hydraulic rig, ``repro`` for AssetOpsBench's own scenarios (agent_builders.py, react_tool_adapter.py and sub_agent_registry.py).
        temperature: None for the paper's 0.0 on every call, or the temperature of a sampling run.
        log_path: where ReActXen's printout goes, and nothing is kept without it.
        on_idle: the driver's no_call entry for a supervisor step whose subtree made no counted call (Runtime.iteration_end).
        example_start: ``{component: level}``, the registered example start that the diagnosis example of the task tools
            shows, which a run whose servers list the diagnosis tool needs (examples.diagnosis_example).

    ``calls`` holds one record per model call and ``trajectory`` one per MCP call, both filled while the run goes, so a
    run that raised leaves them for the driver.
    """

    def __init__(self, server_paths, base_url, api_key, model, surface="rig", temperature=None, log_path=None, example_start=None,
                 on_idle=None):
        self._server_paths, self._model, self._surface = dict(server_paths), model, surface
        self._log_path = Path(log_path) if log_path else None
        self._example_start = example_start
        self.on_idle = on_idle
        self.specs = sub_agent_registry.registration(tuple(self._server_paths), surface)
        self.route = CompletionRoute(base_url, api_key, model, temperature=temperature)
        self.calls = self.route.calls
        self.trajectory: list[dict] = []
        self.supervisor = None  # kept with its counter, so that a run without a result still has its step count
        self.react_steps = None

    def settings(self):
        return run_settings(tuple(self._server_paths), self._surface, self.route.temperature)

    def steps_reached(self):
        """The supervisor's ReAct counter of the run, its step count, also for a run that returned no result: the counter the
        workflow ended with, else the counter of the supervisor in flight plus one, the rule of a run that an error ends.
        None before a supervisor exists."""
        if self.react_steps is not None:
            return self.react_steps
        return self.supervisor.step_n + 1 if self.supervisor is not None else None

    async def run(self, question):
        async with open_sessions(self._server_paths) as sessions:
            definitions = sessions.tools
            runtime = sub_agent_runtime.Runtime(self.route, sessions, definitions, asyncio.get_running_loop(), self._model,
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
        persist_trajectory(runner_name=RUNNER_NAME, model=self._model, question=question, answer=result.answer,
                           trajectory=result.react_trajectory.get("trajectory"))
        return result

    def _play_then_close(self, question, runtime):
        """_play on the agents' thread, then the client of the route closed. Every model call of the run is made on this
        thread, so no call is in flight when the client closes, also when the run was cut from outside."""
        try:
            return self._play(question, runtime)
        finally:
            self.route.close()

    def _play(self, question, runtime):
        error, metric = None, None
        with activated(self.route):
            supervisor = self.supervisor = agent_builders.build_supervisor(question, self.specs, runtime, self._model, stop_when=runtime.supervisor_turn)
            runtime.supervisor = supervisor
            try:
                supervisor.run(reflect_strategy=ReflexionStrategy.REFLEXION)
                metric = supervisor.metric
                react_steps = supervisor.per_round_info[-1]["step"]
                looped = bool(supervisor.json_log) and supervisor.json_log[-1].get("is_loop_detected") == "early_loop"
                # an unfinished supervisor below its step limit was halted because its prompt passed the served context
                end = ("finish" if supervisor.is_finished() else self.route.stop_reason if self.route.stop_reason
                       else "loop" if looped else "max_steps" if supervisor.step_n > supervisor.max_steps else CONTEXT_END)
            except RunStopped as stop:
                end, react_steps = str(stop.args[0]) if stop.args else "cancelled", supervisor.step_n + 1
            except Exception as exc:  # noqa: BLE001 - a failed run is a result of the benchmark
                end, react_steps, error = "error", supervisor.step_n + 1, f"{type(exc).__name__}: {exc}"
            self.react_steps = react_steps
            react_trajectory = supervisor.export_trajectory()
        react_trajectory["overall_metric"] = metric
        react_trajectory["end"] = end
        return MetaAgentResult(question=question, answer=supervisor.answer or "", end=end, error=error, react_steps=react_steps,
                               trajectory=runtime.trajectory, sub_agents=runtime.sub_agents, react_trajectory=react_trajectory)
