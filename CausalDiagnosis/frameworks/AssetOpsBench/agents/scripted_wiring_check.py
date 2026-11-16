"""The scripted wiring check of the driver: AssetOpsBench's released Plan-Execute runner (src/agent/plan_execute), unchanged,
with a fixed plan over the servers of the run and no model (ScriptedBackend), its one episode and its record files. SCRIPTED is
the entry of the declaration (frameworks/AssetOpsBench/__init__.py, AGENTS["scripted"]) that
experiments/episode_runners/run_episodes/ resolves through utils/frameworks.py when a run names the agent.
"""
import json
import os

from CausalDiagnosis.frameworks.AssetOpsBench.agents import upstream_agent_package  # noqa: F401  registers upstream's agent package first
from agent.plan_execute.executor import _ARG_RESOLUTION_PROMPT
from agent.plan_execute.runner import PlanExecuteRunner
from llm import LLMBackend
from observability import set_run_context
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
from CausalDiagnosis.experiments.episode_runners.run_episodes import diagnostics_calls, full_group_of, run_limited
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, FMSR_SERVER, IOT_SERVER, RIG_SERVER, TASK_SERVER, WO_SERVER
from CausalDiagnosis.utils.json_files import write_json


SCRIPTED_SUMMARY = "scripted run complete"


def scripted_steps(profile, aob=False):
    """The plan of the scripted wiring check on an asset, as (task, server, tool, arguments) in call order, over the rig's action
    server or, with aob, over the servers of the AssetOpsBench rows, each with the diagnostics server. One rule builds it from the
    asset profile: the component of the plan is the one whose work order takes longest to close under the default durations,
    its lead time plus its repair downtime, the first on a tie, so that the two waits of the plan see that repair start and close;
    the diagnosis holds every other component at its healthy level and this one at its first faulty level in degradation order;
    the cheapest sensor group is the one of the lowest read price and the full sensor group the one with the most channels, the
    first on a tie of either. Over the AssetOpsBench servers the plan reads the profile from the task server, the registry of the
    site that the asset file names, the failure modes of the asset type of the component, a history of the fields of the cheapest
    group and the newest row of every field, then the diagnostics server and the diagnosis, an order on the child asset of the
    component as the task server names it, the work order reads, and the finish."""
    config = profile.default_config()
    durations, groups, prices = config.durations, config.sensor_groups, config.prices["read"]
    component = max(profile.components, key=lambda c: durations["lead_time"][c] + durations["repair_downtime"][c])
    cheapest = min(groups, key=lambda group: prices[group])
    full = full_group_of(groups)
    levels = {c: profile.healthy[c] for c in profile.components}
    levels[component] = next(level for level in profile.degradation[component] if level != profile.healthy[component])
    diagnosis = json.dumps({**levels, "evidence_ids": ""})
    evidence = ("List the released evidence", CAUSAL_SERVER, "get_episode_evidence", '{"sensor_group": "all"}')
    ranking = ("Rank the candidate fault combinations", CAUSAL_SERVER, "rank_root_causes", '{"top_k": 3}')
    finish = json.dumps({"summary": SCRIPTED_SUMMARY})
    if aob:
        site = profile.text["fallback"]["site"]
        asset = {"site_name": site, "asset_id": profile.asset_id}
        fields = [profile.field_name(sensor) for sensor in groups[cheapest]]
        return [
            ("Read the asset profile", TASK_SERVER, "get_asset_profile", "{}"),
            ("Read the asset registry of the site", IOT_SERVER, "assets", json.dumps({"site_name": site})),
            # the asset type that the registry of the documents gives the child asset of the component (CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.profile_documents)
            (f"Read the failure modes of the {component}", FMSR_SERVER, "get_failure_modes", json.dumps({"asset_class": component.upper()})),
            ("Read the recorded history of the cheapest sensor group", IOT_SERVER, "history", json.dumps({**asset, "sensors": fields, "limit": 5})),
            ("Read the newest recorded row of every field", IOT_SERVER, "latest_reading", json.dumps(asset)),
            evidence,
            ranking,
            ("Submit a diagnosis", TASK_SERVER, "submit_diagnosis", diagnosis),
            (f"Create a work order for the {component}", WO_SERVER, "generate_work_order",
             json.dumps({"description": f"replace the {component}", "asset_num": f"{profile.asset_id}_{component}", "site_id": site,
                         "priority": 2, "work_type": "CM"})),
            ("List the work orders of the site", WO_SERVER, "list_workorders", json.dumps({"site_id": site})),
            ("Read the failure code catalog", WO_SERVER, "get_failure_codes", "{}"),
            ("End the episode", TASK_SERVER, "finish", finish),
        ]
    return [
        ("Read the asset profile", RIG_SERVER, "get_asset_profile", "{}"),
        (f"Read the failure modes of the {component}", RIG_SERVER, "get_failure_modes", json.dumps({"component": component})),
        (f"Read the fix history of the {profile.text['short_noun']}", RIG_SERVER, "get_fix_history", "{}"),
        (f"Look up the repair manual of the {component}", RIG_SERVER, "lookup_fix", json.dumps({"component": component})),
        ("Read one cycle of the cheapest sensor group", RIG_SERVER, "read_sensors", json.dumps({"sensor_group": cheapest, "cycles": 1})),
        ("Read one cycle of the full sensor group", RIG_SERVER, "read_sensors", json.dumps({"sensor_group": full, "cycles": 1})),
        evidence,
        ranking,
        ("Summarize the changes since the start", RIG_SERVER, "summarize_changes_since", '{"cycle": 0}'),
        ("Submit a diagnosis", RIG_SERVER, "submit_diagnosis", diagnosis),
        # replace:<component> is the action of the diagnostics server for any component, whatever its standard fix
        (f"Predict the effect of replacing the {component}", CAUSAL_SERVER, "estimate_intervention_effect", json.dumps({"action": f"replace:{component}"})),
        (f"Create a work order for the {component}", RIG_SERVER, "create_work_order",
         json.dumps({"component": component, "prediction_id": "pred/1", "evidence_ids": ""})),
        ("Wait until the technician starts the repair", RIG_SERVER, "wait_for_notification", '{"timeout": 5}'),
        ("Wait until the work order closes", RIG_SERVER, "wait_for_notification", '{"timeout": 5}'),
        ("Read the clock", RIG_SERVER, "get_current_time", "{}"),
        ("Read one cycle of the cheapest sensor group after the repair", RIG_SERVER, "read_sensors", json.dumps({"sensor_group": cheapest, "cycles": 1})),
        ("Verify the prediction against the reading", CAUSAL_SERVER, "verify_outcome", '{"prediction_id": "pred/1"}'),
        ("Try to declare the evidence insufficient after the diagnosis", RIG_SERVER, "declare_insufficient_evidence",
         json.dumps({"sensor_group": full, "cycles": 1, "evidence_ids": ""})),
        ("End the episode", RIG_SERVER, "finish", finish),
    ]


class ScriptedBackend(LLMBackend):
    """Wiring check without a model: a fixed plan over both servers, fixed arguments and a fixed closing summary,
    returned in the order the runner asks for them (the plan, one argument reply per tool step, the summary). The
    diagnosis and the work order are fixed too, so the episode's verdict says nothing about diagnosis; the run checks
    that every tool of both servers answers through the runner, that the prediction stored by one call is read back by
    the next, that the work order closes after the declared lead time and downtime of the component of the plan, and that a
    declaration after the diagnosis is refused, which is the answer the wiring check expects. The plan is the one of the asset of
    the run (scripted_steps); STEPS and AOB_STEPS are those of the engine's default asset."""

    STEPS = scripted_steps(get_profile(DEFAULT_ASSET))
    AOB_STEPS = scripted_steps(get_profile(DEFAULT_ASSET), aob=True)
    SUMMARY = SCRIPTED_SUMMARY

    def __init__(self, servers=(RIG_SERVER, CAUSAL_SERVER), profile=None):
        self._model_id = "scripted"
        # the plan of the asset of the run, which build_runner hands in, and of the engine's default asset for a caller that names none
        steps = scripted_steps(profile or get_profile(DEFAULT_ASSET), aob=IOT_SERVER in servers)
        self.steps = [s for s in steps if s[1] in servers]
        self._replies = iter(())

    def start_episode(self):
        plan = "\n\n".join(
            f"#Task{n}: {task}\n#Server{n}: {server}\n#Tool{n}: {tool}\n"
            f"#Dependency{n}: {'None' if n == 1 else f'#S{n - 1}'}\n#ExpectedOutput{n}: the tool's answer"
            for n, (task, server, tool, _) in enumerate(self.steps, 1)
        )
        self._replies = iter([plan, *(arguments for _, _, _, arguments in self.steps), self.SUMMARY])

    def generate(self, prompt, temperature=0.0):
        return next(self._replies)

    @property
    def model_id(self):
        return self._model_id


def run_plan_execute(backend, launchers, question, run_id, ep_dir):
    """One run of AssetOpsBench's released runner, which only the scripted agent plays.
    A plan with a malformed dependency raises out of run(), and so does an API error
    that outlasts the client's retries; either is recorded as the episode's error."""
    set_run_context(run_id=run_id, scenario_id=ep_dir.name)
    os.environ["AGENT_TRAJECTORY_DIR"] = str(ep_dir)
    runner = PlanExecuteRunner(llm=backend, server_paths=dict(launchers))
    return run_limited(lambda: runner.run(question))


def record_plan_execute(ep_dir, result, calls):
    """What a failure analysis of a Plan-Execute run needs besides the log: every model call, the plan, the closing
    answer and the trajectory of both servers."""
    write_json(ep_dir / "llm_calls.json", calls)
    if result is not None:
        (ep_dir / "plan.txt").write_text(result.plan.raw, encoding="utf-8")
        (ep_dir / "answer.txt").write_text(result.answer or "", encoding="utf-8")
        write_json(ep_dir / "trajectory.json", [
            {"step": r.step_number, "task": r.task, "server": r.server, "tool": r.tool, "tool_args": r.tool_args,
             "response": r.response, "error": r.error, "duration_ms": r.duration_ms}
            for r in result.trajectory
        ])
    trajectory = result.trajectory if result is not None else []
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
        "plan_steps": len(result.plan.steps) if result is not None else "",
        "causal_calls": diagnostics_calls((r.server, r.tool) for r in trajectory),
    }


class ScriptedCheck:
    """The scripted check as the driver asks for it: the backend of a run on its servers, one episode with its record files, and
    the first line of every argument call of the released executor, by which the walk-through tells those calls apart."""

    argument_marker = _ARG_RESOLUTION_PROMPT.splitlines()[0]

    def backend(self, servers, profile):
        return ScriptedBackend(servers, profile)

    def play(self, runner, launchers, task, run_id, ep_dir):
        """(result, error, extra, calls, tool calls) of one episode, as EpisodeRunner.run_agent returns them."""
        result, error, _ = run_plan_execute(runner.backend, launchers, task, run_id, ep_dir)
        tools = [{"server": r.server, "tool": r.tool, "error": r.error} for r in result.trajectory] if result is not None else []
        return result, error, record_plan_execute(ep_dir, result, runner.backend.calls), runner.backend.calls, tools


SCRIPTED = ScriptedCheck()
