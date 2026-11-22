"""How the driver of CausalDiagnosis runs the two paper-era agents of AssetOpsBench, Plan-Execute
(Plan_Execute_Paper) and Agent-As-Tool (Agent_As_Tool_Paper): for each, its runner class, the block of run.json that records
its form, limits and decoding, one episode on the launchers of the run with its record files, and the example start that both
designs show. PLAN_EXECUTE and AGENT_AS_TOOL are the entries of the declaration (frameworks/AssetOpsBench/__init__.py, AGENTS)
that experiments/episode_runners/run_episodes/ resolves through utils/frameworks.py when a run names the agent.
"""
import os

from CausalDiagnosis.frameworks.AssetOpsBench.agents import upstream_agent_package  # noqa: F401  registers upstream's agent package first
from CausalDiagnosis.agents.utils import idle_hook as idle
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.splits.development_split import EXAMPLE_SEED, example_start
from CausalDiagnosis.experiments.episode_runners.run_episodes import diagnostics_calls, run_limited
from CausalDiagnosis.utils.json_files import write_json
from observability import set_run_context

API_KEY_REFUSAL = "API_KEY is not set; the paper-era runners send it to the endpoint of --base-url"  # pragma: allowlist secret


def diagnosis_example(config, blocks):
    """The example start of a run of a paper-era port, whose levels the one diagnosis example of both reference designs shows:
    the development start of the asset that the registered seed draws among those with the fewest faults
    (CausalDiagnosis.datasets.splits.development_split.example_start), from the profile and the recorded combinations, as run.json records it. An asset whose
    recordings give no development start refuses the run by name, since every example would then show a start of its test set."""
    profile = get_profile(config.asset_id)
    try:
        combination = example_start(profile, set(blocks))
    except ValueError as e:  # development_split.SplitRefused, or no development start at all
        raise SystemExit(f"the two paper-era designs show one diagnosis example, drawn from the development starts of the asset so that "
                         f"it shows no start of its test set, and the asset {config.asset_id} has none: {e}") from None
    healthy = [profile.healthy[c] for c in profile.components]
    return {"seed": EXAMPLE_SEED, "levels": dict(zip(profile.components, combination)),
            "n_faulty": sum(level != ok for level, ok in zip(combination, healthy))}


def agent_as_tool_runner_class():
    """The paper-era Agent-As-Tool runner, imported on first use, since it imports the vendored ReActXen."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.supervisor_runner import MetaAgentRunner

    return MetaAgentRunner


def agent_as_tool_settings(servers, temperature=None):
    """What run.json records of the port on the servers of a run: its form, limits, sub-agents and decoding, at the temperature
    of the run (None for the paper's)."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.supervisor_runner import run_settings

    return run_settings(servers, "rig", temperature)


def plan_execute_runner_class():
    """The paper-era Plan-Execute runner, imported on first use, since it imports the vendored ReActXen."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.runner import PlanExecutePaperRunner

    return PlanExecutePaperRunner


def plan_execute_settings(servers, temperature=None):
    """What run.json records of the Plan-Execute port on the servers of a run: its form, limits, agents and decoding, at the
    temperature of the run (None for the paper's)."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.runner import run_settings

    return run_settings(servers, "rig", temperature)


def run_agent_as_tool(launchers, question, run_id, ep_dir, base_url, api_key, model, temperature=None, example_start=None):
    """One run of the paper-era Agent-As-Tool over the driver's launchers, ReActXen's printout in react_log.txt. A run that raised, or
    that ended on an error inside the agents (an API error that outlasts the client's retries), records the error as
    Plan-Execute does, and the model calls and tool calls made before it stay with the runner. A sampling run passes its
    temperature, every other run none, which keeps the paper's 0.0. example_start is the run's example start, {component: level}.
    A supervisor step whose subtree made no counted call takes the episode's no_call step through CausalDiagnosis.agents.utils.idle_hook, as a turn without a
    call does for every design."""
    set_run_context(run_id=run_id, scenario_id=ep_dir.name)
    os.environ["AGENT_TRAJECTORY_DIR"] = str(ep_dir)
    runner = agent_as_tool_runner_class()(server_paths=dict(launchers), base_url=base_url, api_key=api_key, model=model,
                                          temperature=temperature, log_path=ep_dir / "react_log.txt", example_start=example_start,
                                          on_idle=idle.no_call)
    return (*run_limited(lambda: runner.run(question)), runner)


def record_agent_as_tool(ep_dir, result, calls, trajectory):
    """What a failure analysis of an Agent-As-Tool run needs besides the log: every model call with its raw prompt and reply,
    the trajectory of every MCP call of the sub-agents in the flat shape of Plan-Execute (step, turn, the sub-agent and the
    request it was given, server, tool, arguments, response, error), written also when the run raised, and the supervisor's
    ReActXen trajectory with its metric and every sub-agent's reply. There is no plan, so plan.txt is not written.
    causal_calls counts the attempted calls of the diagnostics tools whatever their error (diagnostics_calls), and
    react_steps is the supervisor's ReAct counter (CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/sub_agent_runtime.py and CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/supervisor_runner.py)."""
    write_json(ep_dir / "llm_calls.json", calls)
    write_json(ep_dir / "trajectory.json", trajectory)
    if result is not None:
        (ep_dir / "answer.txt").write_text(result.answer or "", encoding="utf-8")
        write_json(ep_dir / "react_trajectory.json", {**result.react_trajectory, "sub_agents": result.sub_agents})
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
        "plan_steps": "",
        "causal_calls": diagnostics_calls((r["server"], r["tool"]) for r in trajectory),
        "react_steps": result.react_steps if result is not None else "",
    }


def run_plan_execute_paper(launchers, question, run_id, ep_dir, base_url, api_key, model, temperature=None, example_start=None):
    """One run of the paper-era Plan-Execute over the driver's launchers, ReActXen's printout of the step agents in react_log.txt. A run
    that raised, or that ended on an error inside the workflow, records the error as Agent-As-Tool does, and is not retried
    (CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper/runner.py). A sampling run passes its temperature, every other run none, which keeps 0.0.
    example_start is the run's example start, {component: level}. A planning round, which calls no tool of the row, and a pass of the
    step agents without a counted call take the episode's no_call step through CausalDiagnosis.agents.utils.idle_hook, as a turn without a call does for every
    design."""
    set_run_context(run_id=run_id, scenario_id=ep_dir.name)
    os.environ["AGENT_TRAJECTORY_DIR"] = str(ep_dir)
    runner = plan_execute_runner_class()(server_paths=dict(launchers), base_url=base_url, api_key=api_key, model=model,
                                         temperature=temperature, log_path=ep_dir / "react_log.txt", example_start=example_start,
                                         on_idle=idle.no_call)
    return (*run_limited(lambda: runner.run(question)), runner)


def record_plan_execute_paper(ep_dir, result, runner):
    """What a failure analysis of a Plan-Execute run needs besides the log: every model call with its raw prompt and reply, the
    trajectory of every MCP call of the step agents in the flat shape of Agent-As-Tool, the plan that ran in plan.txt, the
    closing answer, and plan_execute_trajectory.json with the question, every plan and its review, the history the runner
    logged, every step agent's request and answer, how the run ended and its two counts, the steps of the plan that ran and
    the plan steps the run started. The calls and the trajectory are written for every run and the record for every run that
    built its workflow, where a run that returned no result ends as no_result. plan_steps counts the steps of the plan that
    ran, and causal_calls the attempted calls of the diagnostics tools whatever their error (diagnostics_calls)."""
    calls, trajectory, flow = runner.calls, runner.trajectory, runner.record(result)
    write_json(ep_dir / "llm_calls.json", calls)
    write_json(ep_dir / "trajectory.json", trajectory)
    if result is not None:
        (ep_dir / "plan.txt").write_text(result.plan or "", encoding="utf-8")
        (ep_dir / "answer.txt").write_text(result.answer or "", encoding="utf-8")
    if flow is not None:
        write_json(ep_dir / "plan_execute_trajectory.json", {"question": result.question if result is not None else None, **flow})
    plan_steps = result.plan_steps if result is not None else (flow or {}).get("plan_steps")
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
        "plan_steps": "" if plan_steps is None else plan_steps,
        "causal_calls": diagnostics_calls((r["server"], r["tool"]) for r in trajectory),
    }


class PaperAgent:
    """A paper-era agent of AssetOpsBench as the driver asks for it: play one episode and keep its records, the run.json block
    of its form, the request settings of its route, the example start it shows, and the refusal of a run on servers it has no
    agents for."""

    def __init__(self, name, play, settings, servers_refusal):
        self.name, self._play, self._settings, self.servers_refusal = name, play, settings, servers_refusal
        self.api_key_refusal = API_KEY_REFUSAL

    def play(self, runner, launchers, task, run_id, ep_dir):
        """(result, error, extra, calls, tool calls) of one episode, as EpisodeRunner.run_agent returns them."""
        return self._play(runner, launchers, task, run_id, ep_dir)

    def settings(self, servers, temperature=None):
        return self._settings(servers, temperature)

    def request_settings(self, temperature=None):
        """The decoding and transport blocks of run.json, the one completion route both paper-era agents take."""
        from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.sub_agent_runtime import request_settings
        return request_settings(temperature=temperature)

    def example_start(self, config, blocks):
        return diagnosis_example(config, blocks)


def play_agent_as_tool(runner, launchers, task, run_id, ep_dir):
    result, error, _, agent = run_agent_as_tool(launchers, task, run_id, ep_dir, runner.base_url, os.environ.get("API_KEY", ""),
                                                runner.served_model, runner.temperature, runner.diagnosis_example["levels"])
    return result, error, record_agent_as_tool(ep_dir, result, agent.calls, agent.trajectory), agent.calls, agent.trajectory


def play_plan_execute(runner, launchers, task, run_id, ep_dir):
    result, error, _, agent = run_plan_execute_paper(launchers, task, run_id, ep_dir, runner.base_url, os.environ.get("API_KEY", ""),
                                                     runner.served_model, runner.temperature, runner.diagnosis_example["levels"])
    return result, error, record_plan_execute_paper(ep_dir, result, agent), agent.calls, agent.trajectory


AGENT_AS_TOOL = PaperAgent(
    "agent_as_tool", play_agent_as_tool, agent_as_tool_settings,
    "--agent agent_as_tool runs the paper-era supervisor over AssetOpsBench's servers, whose "
    "sub-agents have no counterpart on the rig's action server. Drop --servers rig or use a row of the environment table")
PLAN_EXECUTE = PaperAgent(
    "plan_execute", play_plan_execute, plan_execute_settings,
    "--agent plan_execute runs the paper-era Plan-Execute over AssetOpsBench's servers, whose "
    "step agents have no counterpart on the rig's action server. Drop --servers rig or use a row of the environment table")
