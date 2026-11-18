"""Made-up run folders for the tests of the performance package. Every value here is made up for a test and is no measurement and
no setting of any asset. The episode of known_episode() is built so that each stage of the profile has one answer that a reader
works out by hand from the planted times below, and its total lies on the 0.1 s grid on which the driver rounds wall_s."""
import copy
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification.run_ledger.serving_endpoints import CARD_KEYS, SERVING_KEYS

ZONE = timezone(timedelta(hours=-5))  # the made-up UTC offset of the made-up node
STARTED = "2026-01-05T10:05:00-05:00"  # the made-up started line of the endpoint file
W0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=ZONE).timestamp()  # the made-up epoch of the start of an episode
RIG, MCP = "hydraulic_rig", "causal_diagnosis"


def wall(t):
    return W0 + t


def known_episode():
    """The parts of one made-up episode, relative seconds on the wall clock, and the answers that its profile must give."""
    lines = [
        {"kind": "start", "wall": wall(0.0), "mono": 0.0},
        {"kind": "handover", "wall": wall(2.0), "mono": 2.0},
        {"kind": "ready", "server": RIG, "s_wall": wall(2.25), "s_mono": 2.25, "s_thread_cpu": 0.0, "r_wall": wall(3.0), "r_mono": 3.0, "r_thread_cpu": 0.5},
        {"kind": "list", "server": RIG},
        {"kind": "ready", "server": MCP, "s_wall": wall(3.0), "s_mono": 3.0, "s_thread_cpu": 0.0, "r_wall": wall(3.5), "r_mono": 3.5, "r_thread_cpu": 0.25},
        {"kind": "list", "server": MCP},
        {"kind": "enter", "server": RIG, "tool": "read_sensors", "wall": wall(7.5), "mono": 7.5, "thread_cpu": 0.25},
        {"kind": "call", "server": RIG, "tool": "read_sensors", "stepped": True, "enter_n": 6, "end_wall": wall(9.0), "end_mono": 9.0, "end_thread_cpu": 1.25},
        {"kind": "exit", "server": RIG, "enter_n": 6, "outcome": "answered"},
        {"kind": "enter", "server": RIG, "tool": "no_call", "wall": wall(12.25), "mono": 12.25, "thread_cpu": 1.25},
        {"kind": "call", "server": RIG, "tool": "no_call", "stepped": True, "enter_n": 9, "end_wall": wall(12.5), "end_mono": 12.5, "end_thread_cpu": 1.375},
        {"kind": "exit", "server": RIG, "enter_n": 9, "outcome": "answered"},
        {"kind": "call", "server": MCP, "tool": "rank_root_causes", "stepped": False, "answer": "made up"},
        {"kind": "enter", "server": RIG, "tool": "finish", "wall": wall(15.375), "mono": 15.375, "thread_cpu": 1.375},
        {"kind": "call", "server": RIG, "tool": "finish", "stepped": True, "enter_n": 13, "end_wall": wall(15.625), "end_mono": 15.625, "end_thread_cpu": 1.5},
        {"kind": "exit", "server": RIG, "enter_n": 13, "outcome": "answered"},
    ]
    llm_calls = [{"started_at": wall(4.0), "ended_at": wall(7.0), "seconds": 3.0, "input_tokens": 100, "output_tokens": 30},
                 {"started_at": wall(9.5), "ended_at": wall(12.0), "seconds": 2.5, "input_tokens": 200, "output_tokens": 50},
                 {"started_at": wall(14.0), "ended_at": wall(15.0), "seconds": 1.0, "input_tokens": 300, "output_tokens": 20}]
    trajectory = [{"step": 1, "server": RIG, "tool": "read_sensors", "duration_ms": 2000.0},
                  {"step": 2, "server": MCP, "tool": "rank_root_causes", "duration_ms": 1000.0},
                  {"step": 3, "server": RIG, "tool": "finish", "duration_ms": 500.0}]
    # the state as the servers leave it: its credits and failed seconds lag the clock report that scores.csv holds
    state = {"episode_started_at": wall(0.0), "handover_wall": wall(2.0), "handover_mono": 2.0, "episode_ended_at": wall(16.0),
             "cache_flush": {"flush_seconds": 1.5, "status": 200, "started_at": wall(0.25)}, "think_seconds": [3.25, 4.5],
             "think_cycles": [0, 1], "config": {"horizon": 60, "think_seconds_per_cycle": 10, "clock_mode": "generation_time",
                                                 "asset_id": "made_up_asset"}, "failed_model_seconds": 0.25,
             "credited_seconds": {"starts": 1.0, "calls": 1.0}}
    row = {"seed": "7", "scenario_id": "", "wall_s": "16.5", "pass": "True", "failed_model_s": "0.5", "failed_model_attempts": "1",
           "credited_s": '{"starts":1.0,"calls":1.25}', "uncredited_s": '{"starts":0.25,"calls":0.5}', "uncharged_after_last_s": "0.375",
           "clock_skew": "0"}
    answer = {"total_s": 16.5, "creation_s": 2.0, "cache_flush_s": 1.5, "agent_part_s": 14.0, "end_s": 0.5, "model_s": 6.5,
              "failed_model_s": 0.5, "tool_service_s": 1.75, "tool_service_cpu_s": 1.125, "no_call_service_s": 0.25,
              "server_starts_s": 1.25, "env_tools_client_s": 2.5, "diagnostics_client_s": 1.0, "transport_hold_s": 0.75, "rest_s": 2.75}
    return {"lines": lines, "llm_calls": llm_calls, "trajectory": trajectory, "state": state, "row": row, "answer": answer}


def numbered(lines):
    """The lines of a transcript with their numbers, each line's number its place, as the servers write them."""
    return [{"n": n, "pid": 1, **line} for n, line in enumerate(lines)]


def write_episode(ep_dir, parts):
    ep_dir = Path(ep_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "episode.json").write_text(json.dumps(parts["state"]), encoding="utf-8")
    (ep_dir / "transcript.jsonl").write_text("".join(json.dumps(line) + "\n" for line in numbered(parts["lines"])), encoding="utf-8")
    (ep_dir / "llm_calls.json").write_text(json.dumps(parts["llm_calls"]), encoding="utf-8")
    (ep_dir / "trajectory.json").write_text(json.dumps(parts["trajectory"]), encoding="utf-8")
    return ep_dir


def model_meta(**fields):
    """A made-up run.json of a model agent on one made-up card, with fields over it."""
    serving = {"weights_commit": "made_up", "engine": "made_up", "dtype": "bfloat16", "launch_args_sha256": "made_up",
               "context_length": 1000, "gpu_name": "MADE_UP_CARD", "gpu_mib": "1000", "sampler": {"sha256": None, "bytes": None}}
    return {"asset_id": "made_up_asset", "agent": "causal", "agent_label": "causal", "config_name": "made_up_preset", "clock_mode": "generation_time",
            "model": "made-up-model", "runs_in_flight": 1, "serving": {"agent": serving, "judge": None}, **fields}


def ledger_of(meta, process=None, started=STARTED):
    """A made-up ledger.json of a run with the start record that the made-up run.json meta implies: the agent, its label and the
    preset, the model with its endpoint record, which gives back the serving of meta and holds the started line when started is
    given, and the command of the process (line, replicate) of the grid runner, None for a run that no grid line played, with
    GRID_PARALLEL of the runs in flight."""
    serving = (meta.get("serving") or {}).get("agent") or {}
    endpoint = {key: serving[key] for key in (*SERVING_KEYS, *CARD_KEYS) if serving.get(key) is not None}
    endpoint.update({"sglang": serving.get("engine")} if serving.get("engine") is not None else {})
    endpoint.update({"started": started} if started is not None else {})
    line, replicate = process or (None, "none")
    return {"agent": meta.get("agent"), "agent_label": meta.get("agent_label"), "configuration": {"config_name": meta.get("config_name")},
            "model": {"name": meta.get("model"), "endpoint_file": endpoint,
                      "endpoint_models": [{"id": meta.get("model"), "max_model_len": serving.get("context_length")}]},
            "command": {"argv": ["--agent", str(meta.get("agent"))] + ([] if replicate == "none" else ["--replicate", replicate]),
                        "grid_line": None if line is None else str(line),
                        "env": {} if meta.get("runs_in_flight") is None else {"GRID_PARALLEL": str(meta["runs_in_flight"])}}}


def write_run(run_dir, episodes, meta=None, started=STARTED, process=None, unfinished=False):
    """A run folder with scores.csv, its ledger.json (ledger_of) and one episode folder per (seed, parts), each part's row of
    scores.csv written with its seed, and with run.json unless the run did not finish."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = model_meta() if meta is None else meta
    if not unfinished:
        (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / ledger.LEDGER).write_text(json.dumps(ledger_of(meta, process, started)), encoding="utf-8")
    rows = []
    for seed, parts in episodes:
        write_episode(run_dir / f"seed{seed:06d}", parts)
        rows.append({**parts["row"], "seed": str(seed)})
    fields = sorted({key for row in rows for key in row})
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def shifted(parts, seconds):
    """A copy of the parts of an episode with every wall stamp moved by seconds, as a later episode of the same run."""
    out = copy.deepcopy(parts)
    for line in out["lines"]:
        for key in ("wall", "s_wall", "r_wall", "end_wall"):
            if line.get(key) is not None:
                line[key] += seconds
    for call in out["llm_calls"]:
        call["started_at"] += seconds
        call["ended_at"] += seconds
    for key in ("episode_started_at", "handover_wall", "episode_ended_at"):
        out["state"][key] += seconds
    return out


def local(t):
    """The made-up local time of the made-up node, as sacct and the log of a job print it, of relative second t."""
    return datetime.fromtimestamp(wall(t), ZONE).strftime("%Y-%m-%dT%H:%M:%S")


def sampler_text(rows):
    """A made-up sampler file with the header that nvidia-smi --format=csv prints and one row per (relative second, percent, watts)."""
    head = "timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]\n"
    return head + "".join(f"{datetime.fromtimestamp(wall(t), ZONE).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}, {busy} %, 0 %, 1 MiB, "
                          f"{'[N/A]' if watts is None else f'{watts:.2f} W'}\n" for t, busy, watts in rows)


def write_grid(grid, job, record=None, log=None, in_flight=None, sampler=None):
    """A made-up grid folder grid_<job id> of a cluster job: its job record lines, the lines of its Slurm log in job_logs, its
    record of the processes in flight as (epoch, event, loop, kind, name, pid, wall) tuples and the text of its sampler copy."""
    grid = Path(grid)
    logs = grid / "job_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if record is not None:
        (grid / "job_record.txt").write_text("".join(line + "\n" for line in record), encoding="utf-8")
    if log is not None:
        (logs / f"cd-serve-{job}.out").write_text("".join(line + "\n" for line in log), encoding="utf-8")
    if in_flight is not None:
        (grid / "in_flight.tsv").write_text("".join("\t".join(map(str, event)) + "\n" for event in in_flight), encoding="utf-8")
    if sampler is not None:
        (logs / f"gpu-{job}.csv").write_text(sampler, encoding="utf-8")
    return grid


def recorded_sampler(path):
    """The record of a sampler file that run.json keeps at the end of a run, its SHA-256 and its length as they stand."""
    sha256, length = ledger.prefix_hash(path)
    return {"sha256": sha256, "bytes": length}


def runner_header(t):
    """The made-up first line of the grid runner in the log of a serving job at relative second t."""
    return (f"### node7 {local(t).replace('T', ' ')} EST fork /made/up cluster_jobs /made/up/cj tree made_up (clean) "
            "grid local_made_up.txt, runs in flight 1")


PROMPT = "MADE_UP_PROMPT_TEXT"  # the made-up prompt in every made-up line of an engine export, which no output may hold


def export_line(received, prompt_tokens, completion_tokens, queue=0.25, prefill=0.5, decode=2.0, cached=0):
    """One made-up line of the engine's export, with every key that the engine writes, of a request received at the epoch received,
    queued for queue seconds, in its prefill for prefill seconds and in its decode for decode seconds."""
    forward = received + queue
    finished = forward + prefill + decode
    return {"request_received_ts": received, "api_server_dispatch_finish_ts": received, "queue_time": queue,
            "forward_entry_time": forward, "prefill_finished_time": forward + prefill, "request_finished_ts": finished,
            "response_sent_to_client_ts": finished, "e2e_latency": finished - received, "decode_throughput": 0.0,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cached_tokens": cached,
            "cached_tokens_details": None, "reasoning_tokens": 0, "finish_reason": {"type": "stop"}, "num_retractions": 0,
            "id": "made_up", "dp_rank": None, "weight_version": "default", "weight_versions": None,
            "request_parameters": json.dumps({"text": PROMPT, "stream": False})}


def write_export(folder, hours):
    """A made-up export folder of the engine with one file per (hour, lines), named as the engine names its files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for hour, lines in hours:
        (folder / f"sglang-request-metrics-20260105_{hour}.log").write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return folder
