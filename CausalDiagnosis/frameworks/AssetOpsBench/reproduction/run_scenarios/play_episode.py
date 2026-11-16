import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_sets import family_of, task_kind
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_folder_names import CALLS_FILE, EPISODE_FILE, PE_PORT, PORT, PORTS, episode_name
from CausalDiagnosis.frameworks.AssetOpsBench.agents import agent_runs  # noqa: E402
from CausalDiagnosis.utils.json_files import write_json  # noqa: E402  the rig's atomic JSON writer
from observability import set_run_context  # noqa: E402
from CausalDiagnosis.utils import model_request_retry as retry  # noqa: E402  the episode deadline every model request and tool wait of the ports reads
from . import server_launchers, step_counts


@dataclass(frozen=True)
class Settings:
    launchers: dict
    model: str
    base_url: str
    api_key: str
    reload: bool
    timeout_s: float
    layer: Path | None = None


def timed_run(runner, question, timeout_s):
    """runner.run(question) with its wall clock. An exception (an API error that outlasts the client's retries, a server
    that fails) is the episode's error, since a failed run is a result of the benchmark. The time limit keeps one stuck
    tool server from holding the job."""
    started = time.perf_counter()
    try:
        # the deadline reaches every model request and tool wait of the paper-era ports (CausalDiagnosis.utils.model_request_retry), so on a dead service
        # their run ends at the cut, its thread returns and its client closes, where wait_for alone leaves the thread running
        with retry.episode_deadline(time.monotonic() + timeout_s):
            result, error = asyncio.run(asyncio.wait_for(runner.run(question), timeout_s)), None
    except TimeoutError:
        result, error = None, f"TimeoutError: the run exceeded {timeout_s:g} s"
    except Exception as exc:  # noqa: BLE001
        result, error = None, f"{type(exc).__name__}: {exc}"
    return result, error, time.perf_counter() - started


def pe_port_runner():
    """The paper-era Plan-Execute runner, imported on first use."""
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.runner import PlanExecutePaperRunner

    return PlanExecutePaperRunner


def play(agent, row, repeat, run_dir, settings):
    """One episode of one agent on one scenario. Returns the record written to episode.json."""
    sid = int(row["id"])
    name = episode_name(agent, sid, repeat)
    ep_dir = Path(run_dir) / name
    ep_dir.mkdir()
    reload_s = server_launchers.reload_couchdb(settings.layer) if settings.reload else None
    set_run_context(run_id=name, scenario_id=str(sid))
    os.environ["AGENT_TRAJECTORY_DIR"] = str(ep_dir)  # the runners persist their trajectory record here
    started = datetime.now().isoformat(timespec="seconds")
    if agent == PORT:
        runner = agent_runs.agent_as_tool_runner_class()(
            server_paths=dict(settings.launchers), base_url=settings.base_url, api_key=settings.api_key, model=settings.model,
            surface="repro", log_path=ep_dir / "react_log.txt")
        result, error, seconds = timed_run(runner, row["text"], settings.timeout_s)
        error = error or (result.error if result is not None else None)
        calls = runner.calls
        counts = step_counts.port_counts(result, runner)
        write_json(ep_dir / "trajectory.json", runner.trajectory)
        if result is not None:
            write_json(ep_dir / "react_trajectory.json", {**result.react_trajectory, "sub_agents": result.sub_agents})
    elif agent == PE_PORT:
        runner = pe_port_runner()(server_paths=dict(settings.launchers), base_url=settings.base_url, api_key=settings.api_key,
                                  model=settings.model, surface="repro", log_path=ep_dir / "react_log.txt")
        result, error, seconds = timed_run(runner, row["text"], settings.timeout_s)
        error = error or (result.error if result is not None else None)
        calls = runner.calls
        counts = step_counts.pe_port_counts(result, runner)
        write_json(ep_dir / "trajectory.json", runner.trajectory)
        flow = runner.record(result)
        if flow is not None:  # every run that built its workflow leaves its plans and its history, a run without a result too
            write_json(ep_dir / "plan_execute_trajectory.json", flow)
    else:
        raise ValueError(f"{agent} is no agent of the reproduction, which plays {', '.join(PORTS)}")
    write_json(ep_dir / CALLS_FILE, calls)
    record_file = ep_dir / f"{name}.json"
    record = {
        "format": 1, "episode": name, "agent": agent, "scenario_id": sid, "family": family_of(sid), "task_kind": task_kind(sid),
        "repeat": repeat, "question": row["text"], "answer": result.answer if result is not None else "", "error": error,
        "runtime_s": round(seconds, 3), "reload_s": None if reload_s is None else round(reload_s, 3), **counts,
        "model_calls": len(calls), "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
        "trajectory_record": record_file.name if record_file.exists() else None,
        "started": started, "ended": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(ep_dir / EPISODE_FILE, record)
    return record
