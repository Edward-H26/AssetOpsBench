import json
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, tool_surface  # noqa: E402
from CausalDiagnosis.utils.json_files import write_json  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import CausalLoopOptions
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import CausalLoopRunner  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tool_surfaces import RigSurface  # noqa: E402
from . import play_one_episode, harness_rule


def diagnostics_calls(records):
    """The attempted calls of the diagnostics server among the (server, tool) pairs of a trajectory, whatever their error, the one
    rule of causal_calls for the three runners: a failed call is an invocation, as the state file counts a refused
    one, and the steps of the execution efficiency add this count to the calls of the action or metered servers. A step that names the
    server and no tool, a reasoning step of Plan-Execute, is no call."""
    return sum(1 for server, tool in records if server == CAUSAL_SERVER and tool and str(tool).lower() not in ("none", "null"))


def causal_surface(asset_id, store=None):
    """The surface of the CausalDiagnosis agent on the servers of a row, the AssetOpsBench servers when the row has a store and
    else the action server of the asset: named by the short noun of the asset's text with its example channel and by the
    statistics the asset serves, the plain one first, so that the texts of the agent hold no word of another asset. The driver
    and the digest script build the agent's texts with it."""
    profile = play_one_episode.get_profile(asset_id)
    kind = (store or {}).get("kind")
    surface = next((tool_surface(name).agent_surface for name, entry in frameworks.surfaces().items() if kind in entry["stores"]), RigSurface)
    return surface(profile.text["short_noun"], profile.text.get("example_channel"), profile.statistics["served"])


def run_causal(backend, launchers, question, options, surface=None):
    """One run of the CausalDiagnosis agent with the ablation switches of the run, on the hydraulic rig's action server or
    on the surface given (the AssetOpsBench servers of --servers aob). The runner keeps what it recorded before a failure
    and reports the failure in its result, so a partial episode is still written and verified."""
    runner = CausalLoopRunner(llm=backend, server_paths=dict(launchers), options=CausalLoopOptions(**options), surface=surface)
    return harness_rule.run_limited(lambda: runner.run(question))


def render_plans(result):
    """The plans of a CausalDiagnosis run as text, for plan.txt and the audit report: every plan with its source, its steps
    with their status, and the surprise that ended it."""
    lines = [f"Differential: {json.dumps(result.differential)}", ""]
    for record in result.plans:
        plan = record["plan"]
        lines.append(f"Plan {record['index']} ({record['source']}, branch {plan.get('branch')}, mode {plan.get('mode')}"
                     + (", after a surprise" if record["after_surprise"] else "") + f"): {plan.get('why', '')}")
        for n, step in enumerate(plan["steps"], 1):
            done = record["steps"][n - 1] if n <= len(record["steps"]) else None
            status = done["status"] if done else "not reached"
            note = f" {done['note']}" if done and done.get("note") else ""
            lines.append(f"  {n}. {step['kind']} {json.dumps(step['args'])} expect {json.dumps(step['expect'])} -> {status}{note}")
        for problems in record["check_problems"]:
            lines.append("  refused by the plan check: " + "; ".join(problems))
        for error in record["parse_errors"]:
            lines.append(f"  parse error: {error}")
        if record["surprise"]:
            lines.append(f"  surprise at step {record['surprise']['step']}: {record['surprise']['reason']}")
        lines.append("")
    return "\n".join(lines)


def record_causal(ep_dir, result, calls):
    """What a failure analysis of a CausalDiagnosis run needs besides the log: every model call, every plan with its
    expectations, its executed steps and the surprise that ended it, the differential, the closing answer and the trajectory
    of both servers, plus the row extras of scores.csv."""
    write_json(ep_dir / "llm_calls.json", calls)
    if result is not None:
        write_json(ep_dir / "plans.json", {"mode_of_first_plan": result.mode_of_first_plan, "differential": result.differential, "plans": result.plans,
                                           "surprises": result.surprises, "revisions": result.revisions, "fallbacks": result.fallbacks,
                                           "model_calls": result.model_calls, "end_reason": result.end_reason, "error": result.error,
                                           "server": result.server, "board": result.board})
        (ep_dir / "plan.txt").write_text(render_plans(result), encoding="utf-8")
        (ep_dir / "answer.txt").write_text(result.answer or "", encoding="utf-8")
        write_json(ep_dir / "trajectory.json", result.trajectory)
    trajectory = result.trajectory if result is not None else []
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
        "plans": len(result.plans) if result is not None else "",
        "surprises": len(result.surprises) if result is not None else "",
        "revisions": result.revisions if result is not None else "",
        "fallbacks": result.fallbacks if result is not None else "",
        "first_plan_mode": (result.mode_of_first_plan or "") if result is not None else "",
        "causal_calls": diagnostics_calls((r["server"], r.get("tool")) for r in trajectory),
    }
