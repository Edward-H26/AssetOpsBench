from __future__ import annotations
import asyncio
import json
import time
import concurrent.futures
from CausalDiagnosis.utils import model_request_retry as retry
from CausalDiagnosis.utils.model_request_retry import DeadlineReached
from CausalDiagnosis.agents.utils.mcp_sessions import SESSIONS
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper import text_completion_route as completion, agent_builders, react_tool_adapter, sub_agent_registry
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import RunStopped
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts import ended, split_receipt


def episode_over(tool, text, is_error):
    """Whether a tool's answer shows that the environment ended the episode, read by CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_receipts as the controller of the
    CausalDiagnosis agent reads it: the refusal after the end, an accepted finish, or the receipt line of the priced rows with
    no step left or the end event among its notifications."""
    body, receipt = split_receipt(text)
    return ended(tool, body, receipt, is_error)


def request_settings(temperature=None, servers=()):
    """What a run folder records of the requests of a port, the same for both ports, since they share the route and the
    runtime: the block decoding, what the completion route sends, and the block transport, how its client and the tool
    sessions carry a request. temperature is None for the paper's value, or the temperature of a sampling run, and servers
    are those of the run, whose max_tokens is the room of each request where they list the CausalDiagnosis MCP."""
    paper = completion.PAPER_DECODING
    max_tokens = completion.REPLY_ROOM if sub_agent_registry.with_mcp(servers) else paper["max_tokens"]
    return {"decoding": {"route": "/v1/completions", "max_tokens": max_tokens, "min_tokens": paper["min_tokens"],
                         "temperature": paper["temperature"] if temperature is None else temperature,
                         "stop": {kind: list(stop) for stop, kind in completion.KINDS.items()}, "no_stop_trim": paper["no_stop_trim"],
                         "extra_body": None},
            "transport": {"retries": None, "retry_status": "no answer, " + ", ".join(str(s) for s in completion.RETRY_STATUS),
                          "retry_delay_s": completion.RETRY_DELAY_S, "timeout_s": None, "connect_timeout_s": None,
                          # no client and no session has a time limit of its own: every call is bounded by the row's deadline
                          "deadline": "the wall limit of the row (CausalDiagnosis.utils.model_request_retry.deadline_left)",
                          "sessions": SESSIONS, "tool_timeout_s": None}}


class Runtime:
    """What the agents' thread reaches: the MCP sessions on the run's event loop, the route, the records."""

    def __init__(self, route, sessions, definitions, loop, model, example_start=None, on_idle=None, example_asset=None):
        self.route, self.sessions, self.definitions, self.loop = route, sessions, definitions, loop
        self.model = model
        # the driver's no_call entry for an iteration whose subtree made no counted call; it returns whether
        # the environment ended the episode with it, and None runs no such entry
        self.on_idle = on_idle
        self._no_calls = 0  # the no_call steps the run took through on_idle
        self._supervisor_mark = (0, 0)  # mark() at the supervisor's last step
        # on a run with the CausalDiagnosis MCP every ReAct step of a sub-agent or a step agent that makes no counted call takes a
        # no_call step (ReactStepCharge), and no agent has a step limit of its own (sub_agent_registry.step_cap)
        self.charges_steps = sub_agent_registry.with_mcp(tuple(definitions or ()))
        self.example_start = example_start  # {component: level} of the registered example start, which the diagnosis example shows
        self.example_asset = example_asset  # the asset of the run, whose site, field and profile the examples of a run with the MCP show
        self.trajectory: list[dict] = []
        self.sub_agents: list[dict] = []
        self.supervisor = None
        self._tools = {}
        self._examples = {}
        self._row = None

    def turn(self):
        return self.supervisor.step_n if self.supervisor is not None else None

    def episode_ended(self):
        """Whether the run must stop: the environment ended the episode or the run was cut. A ReActXen agent reads it before
        every step (stop_when), so a sub-agent stops inside the supervisor's call."""
        return self.route.stopped()

    def answered(self):
        """The counted calls of the run: the calls of a server of the environment that it answered, a refusal included. A call
        of the CausalDiagnosis MCP is no counted call, as in the controller of the causal agent."""
        return sum(1 for record in self.trajectory if record["response"] is not None and record["server"] != sub_agent_registry.CAUSAL_SERVER)

    def mark(self):
        """(counted calls, no_call steps) of the run so far, against which acted_since reads an iteration."""
        return self.answered(), self._no_calls

    def acted_since(self, mark):
        """Whether the run took a step of the episode's step cap since mark: a counted call, or a no_call step that a ReAct step
        of a sub-agent or a step agent took on a run with the CausalDiagnosis MCP, so that no iteration is charged twice."""
        answered, no_calls = mark
        return self.answered() > answered or self._no_calls > no_calls

    def iteration_end(self, acted):
        """The end of an outer iteration: one without a counted call goes to the driver's no_call entry, which takes the step
        of the row's step cap and plays the thinking it charges; when that ends the episode the run stops."""
        if not acted and self.on_idle is not None:
            self._no_calls += 1
            if self.on_idle():
                self.route.stop("episode_over")

    def supervisor_turn(self):
        """stop_when of the supervisor, read before each of its steps: the step before it ends an iteration, acted when its
        sub-agent runs took a step of the episode's step cap (acted_since), and the supervisor stops once the episode is over."""
        if self.supervisor is not None and self.supervisor.step_n > 1:
            self.iteration_end(self.acted_since(self._supervisor_mark))
            self._supervisor_mark = self.mark()
        return self.episode_ended()

    def step_charge(self):
        """The stop_when of a sub-agent or a step agent with the charge of its ReAct steps on a run with the CausalDiagnosis MCP
        (ReactStepCharge, whose close the caller calls after the agent's run), and None on every other run."""
        return ReactStepCharge(self) if self.charges_steps else None

    def tools_of(self, spec):
        """The LangChain tools of a sub-agent, one per MCP tool of its servers, built once per run."""
        if spec.name not in self._tools:
            tools = []
            for server in spec.servers:
                for definition in self.definitions.get(server, []):
                    tools.append(react_tool_adapter.mcp_tool(definition.name, definition.description, definition.inputSchema or {},
                                              self._caller(spec.name, server, definition.name)))
            self._tools[spec.name] = tools
        return self._tools[spec.name]

    def mcp_row(self):
        """The McpRow of the run, built once: what the examples written for the port show of the asset of a run whose servers list
        the CausalDiagnosis MCP, None on every other run (sub_agent_registry.mcp_row)."""
        if self._row is None:
            self._row = (sub_agent_registry.mcp_row(tuple(self.definitions), self.example_asset, self.example_start),)
        return self._row[0]

    def examples_of(self, spec):
        """The agent's own examples for the tools it has, written once per run: the diagnosis example from the tool's listed
        schema and the example start of the run (hub._per_tool), and on a run with the CausalDiagnosis MCP the asset of the run
        in every example written for the port (mcp_row)."""
        if spec.name not in self._examples:
            self._examples[spec.name] = spec.own_examples(self.tools_of(spec), self.example_start, self.mcp_row())
        return self._examples[spec.name]

    def prepare(self, specs):
        """The tools and the examples of every sub-agent of kind react, built before the first model call, so that a tool name
        that the engine would misread, an example start that does not fit the tool contract or a run with the CausalDiagnosis MCP
        that names no asset stops the run with its reason
        and is never answered to a supervisor as a failed delegation. Returns that refusal, a ValueError, or None. A runner
        raises it once its session block has ended: raised inside, where the MCP clients run in task groups, it would leave
        run() inside an exception group of the client, without its reason."""
        try:
            for spec in specs:
                if spec.kind == "react":
                    self.examples_of(spec)
        except ValueError as refusal:
            return refusal
        return None

    def _caller(self, agent, server, tool):
        def call(arguments):
            return self.call_tool(agent, server, tool, arguments)
        return call

    def call_tool(self, agent, server, tool, arguments):
        self.route.check()
        record = {"step": len(self.trajectory) + 1, "turn": self.turn(), "agent": agent, "task": self._request, "server": server,
                  "tool": tool, "tool_args": arguments, "response": None, "error": None, "duration_ms": None}
        self.trajectory.append(record)
        started = time.perf_counter()
        try:
            # the call has no time limit of its own: the thread waits until the row's deadline and gives the call up there, the
            # deadline read again at each expiry, since a credit can move it while the call is in flight (retry.waits_to_deadline)
            future = asyncio.run_coroutine_threadsafe(self.sessions.call(server, tool, arguments), self.loop)
            for left in retry.waits_to_deadline():
                try:
                    text, is_error = future.result(timeout=left)
                    break
                except concurrent.futures.TimeoutError:
                    continue
            else:
                future.cancel()
                raise DeadlineReached(f"{tool} of {server} was open when the deadline of the episode passed")
        except Exception as exc:
            if self.route.stop_reason:
                # the run ended while the call was in flight, so the call was given up with the run and not by the transport
                record["error"] = f"RunStopped: {self.route.stop_reason}"
                raise RunStopped(self.route.stop_reason) from exc
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        record["response"] = text
        if is_error:
            record["error"] = text
        if episode_over(tool, text, is_error):
            self.route.stop("episode_over")
            raise RunStopped("episode_over")
        return text

    _request = None

    def run_sub_agent(self, spec, request, parent_agent, parent_model_id):
        """The paper's wrapper (IoTWrapper.request and its siblings): the agent on the request, then the JSON reply."""
        tools = self.tools_of(spec)
        outer, self._request = self._request, request
        try:
            with self.route.agent(spec.name):
                charge = self.step_charge()
                agent = agent_builders.build_sub_agent(spec, request, tools, self.model, self.examples_of(spec),
                                                       stop_when=charge if charge is not None else self.episode_ended,
                                                       max_steps=sub_agent_registry.step_cap(spec.max_steps, tuple(self.definitions)))
                if parent_agent:
                    agent.parent_agent = parent_agent
                if charge is not None:
                    charge.agent = agent
                reaction = agent.run()
                if charge is not None:
                    charge.close()
        finally:
            self._request = outer
        answer = agent.answer if agent.answer.strip() else "Agent failed to generate final answer"
        review = ({"status": reaction["status"], "reasoning": reaction["reasoning"], "suggestions": reaction["suggestions"]}
                  if reaction else {"status": "None", "reasoning": "None", "suggestions": "None"})
        reply = {"answer": answer, "review": review, "final_result": answer,
                 "summary": (f"I am {spec.speaker}, and I completed my task. The status field denotes the status of execution. I also "
                             "received feedback from the review agent, whose suggestions are included in the review field for further insights.")}
        if parent_agent:
            parent_agent.add_step_trajectory_and_metric(json.dumps(agent.trajectory), json.dumps(agent.metric))
        self.sub_agents.append({"agent": spec.name, "turn": self.turn(), "request": request, "answer": answer, "review": review,
                                "react_steps": agent.per_round_info[-1]["step"] if agent.per_round_info else agent.step_n,
                                "metric": agent.metric})
        return json.dumps(reply, indent=2)


class ReactStepCharge:
    """The stop_when of a sub-agent or a step agent on a run with the CausalDiagnosis MCP: one no_call step for each of its ReAct
    steps that made no counted call (a call of the MCP, Finish, an invalid action), as a planning turn of the CausalDiagnosis agent
    without one takes it. ReActXen asks stop_when before every step and once after its round, and the caller calls close after
    the agent's run, so each step is charged once, read from the agent's step counter, which every step advances by one."""

    def __init__(self, runtime):
        self.runtime, self.agent = runtime, None
        self._mark, self._steps = runtime.mark(), 1  # ReActXen counts steps from 1

    def __call__(self):
        self.close()
        return self.runtime.episode_ended()

    def close(self):
        """Charges each step of the agent since the last charge that made no counted call, until the episode ends."""
        while self.agent is not None and self._steps < self.agent.step_n and not self.runtime.episode_ended():
            self._steps += 1
            self.runtime.iteration_end(self.runtime.acted_since(self._mark))
            self._mark = self.runtime.mark()
