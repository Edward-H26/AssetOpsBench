"""Audit reports rendered from the public files of an episode folder and from the run folder, without a model and
without a hidden value (CausalDiagnosis/docs/experiments.md).

episode_report shows what a reader needs to audit an episode as the agent saw it: the question, the plan, every tool
call of both servers with its arguments and a one line result, the predictions the diagnostics tools stored, the public
end of the episode and the closing summary, and the model calls. The oracle, the verdict and the scores live under
hidden/ and are never read here, so a report can sit beside the agent's files. run_summary counts the episodes and the
files of a run and states no score.
"""
import json
from pathlib import Path

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, RIG_SERVER
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, HIDDEN_FILES, PUBLIC_FILES, STATE_FILE, context_or_api_error, episode_dirs, episode_folder, input_tokens_per_call, read_rows
from CausalDiagnosis.utils.public_rules.visits import is_no_call

MAX_RESULT = 220


def _read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def _short(text, limit=MAX_RESULT):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _status(data):
    return (f"time {data.get('time')}, budget left {data.get('budget_left')}, steps left {data.get('steps_left')}, "
            f"open work orders {len(data.get('open_work_orders') or [])}, over {data.get('episode_over')}")


def summarize_response(server, tool, response, error=None):
    """One line per tool answer: the action server's result and counters, the diagnostics server's message, or the
    error."""
    if error:
        return f"error: {_short(error)}"
    try:
        data = json.loads(response) if isinstance(response, str) else response
    except (TypeError, ValueError):
        return _short(response)
    if not isinstance(data, dict):
        return _short(response)
    if "error" in data and len(data) == 1:
        return f"error: {_short(data['error'])}"
    if server == CAUSAL_SERVER or "message" in data:
        return _short(data.get("message", response))
    if "refused" in data:
        return f"refused: {_short(data['refused'])}; {_status(data)}"
    result = data.get("result")
    if not isinstance(result, dict):
        return _short(f"{result}; {_status(data)}" if result is not None else _status(data))
    if tool == "read_sensors":
        served = result.get("served") or []
        return f"served {len(served)} cycle(s) of {result.get('sensor_group')}; {_status(data)}"
    if tool == "wait_for_notification":
        return f"waited {result.get('waited')} cycle(s) to cycle {result.get('time')}, ended by {result.get('ended_by')}; {_status(data)}"
    if tool == "get_current_time":
        return f"cycle {result.get('time')} of {result.get('horizon')}; {_status(data)}"
    if tool == "summarize_changes_since":
        return (f"{len(result.get('alarms') or [])} alarm(s) and {len(result.get('work_orders_closed') or [])} work order(s) closed "
                f"since cycle {result.get('since')}; {_status(data)}")
    if tool == "get_asset_profile":
        return f"profile of {len(result.get('components') or [])} components; {_status(data)}"
    if tool == "submit_diagnosis":
        return f"diagnosis {result.get('diagnosis')}; {_status(data)}"
    if tool == "declare_insufficient_evidence":
        return f"declared insufficient evidence, separating test {result.get('separating_test')}; {_status(data)}"
    if tool == "create_work_order":
        order = result.get("work_order") or {}
        return f"work order {order.get('id')} on {order.get('component')}, expected to close at cycle {order.get('expected_close_t')}; {_status(data)}"
    if tool == "finish":
        return f"finished; {_status(data)}"
    return _short(f"{json.dumps(result)}; {_status(data)}")


def _tool_rows(ep_dir):
    """The tool calls in order: the trajectory of both servers when the runner wrote one, else the action server's own
    record of its calls (the policies play through the action server only)."""
    trajectory = _read(ep_dir / "trajectory.json")
    if trajectory:
        return [(r["step"], f"{r['server']}.{r['tool']}" if r.get("tool") else r["server"], json.dumps(r.get("tool_args", {}), separators=(",", ":")),
                 summarize_response(r["server"], r.get("tool"), r.get("response", ""), r.get("error"))) for r in trajectory]
    state = _read(ep_dir / STATE_FILE) or {}
    return [(n, f"{RIG_SERVER}.{c['tool']}", json.dumps(c["arguments"], separators=(",", ":")), summarize_response(RIG_SERVER, c["tool"], c["response"]))
            for n, c in enumerate((c for c in state.get("tool_calls", []) if not is_no_call(c)), 1)]


def _public_events(released):
    return [e for e in (released or {}).get("events", []) if isinstance(e, dict)]


def episode_report(ep_dir, meta=None):
    ep_dir = Path(ep_dir)
    meta = meta or _read(ep_dir.parent / "run.json") or {}
    released = _read(ep_dir / "released.json") or {}
    predictions = _read(ep_dir / "predictions.json") or []
    calls = _read(ep_dir / "llm_calls.json")
    state = _read(ep_dir / STATE_FILE) or {}
    out = [f"# Episode {ep_dir.name}", ""]
    out.append(f"Config {meta.get('config_name', '?')}, agent {meta.get('agent', '?')}" + (f" ({meta['model']})" if meta.get("model") else "")
               + f", servers {', '.join(meta.get('servers', []) or ['none'])}, seed {state.get('seed', '?')}, clock mode {meta.get('clock_mode', '?')}, "
               f"guidance level {released.get('guidance_level', meta.get('guidance_level', '?'))}.")
    if released.get("task_text"):
        out.append(f"Task text: {released['task_text']}")
    out.append("")
    anchored = meta.get("anchored")
    if anchored:
        out += ["## Anchored scenario assumptions", ""]
        out += [f"- {a}" for a in anchored.get("assumptions", [])]
        anchor = (anchored.get("anchors") or {}).get(ep_dir.name)
        if anchor:
            out.append(f"- This episode is anchored at {anchor.get('date')} (cycle {anchor.get('cycle_index')} from {anchored.get('date0')}"
                       + ("" if anchor.get("stable") else f", an unstable cycle, so the nearest stable cycle {anchor.get('anchored_cycle_index')} of the same combination serves")
                       + ").")
        out.append("")

    if meta.get("mode") == "static":
        out += ["## Static mode", "",
                "This episode was answered without the gym (ablation A3): the agent saw the question, the features of the recorded cycle "
                "the gym would have served first, which the gym charges for, and the failure modes, and answered in one model call; no tool "
                "was called, no work order was executed, and the recordings were never consulted. The answer is judged against the "
                "oracle's reference answer and parsed for the components it calls faulty and repaired; both scorings sit under hidden/. "
                "The prompt is prompt.txt.", ""]

    plan = _read(ep_dir / "plan.txt")
    if plan:
        out += ["## Plan", "", "```", plan.strip(), "```", ""]

    rows = _tool_rows(ep_dir)
    out += ["## Tool calls", ""]
    if rows:
        out += ["| n | tool | arguments | result |", "|---|---|---|---|"]
        out += [f"| {n} | {tool} | `{args}` | {result} |" for n, tool, args, result in rows]
    else:
        out.append("No tool call was made.")
    out.append("")

    if predictions:
        out += ["## Predictions stored by the diagnostics tools", "", "| id | action | fields |", "|---|---|---|"]
        out += [f"| {p.get('id')} | {p.get('action')} | {', '.join(sorted(k for k in p if k not in ('id', 'action')))} |" for p in predictions]
        out.append("")

    events = _public_events(released)
    status = released.get("status", {})
    end = next((e for e in events if e.get("event") == "end"), None)
    finish = next((e for e in events if e.get("event") == "finish"), None)
    refused = [e for e in events if e.get("event") == "refused"]
    created = [e for e in events if e.get("event") == "work_order_created"]
    closed = [e for e in events if e.get("event") == "work_order_closed"]
    out += ["## Public end of the episode", ""]
    out.append(f"Time {status.get('time')}, budget left {status.get('budget_left')}, steps left {status.get('steps_left')}, "
               f"episode over {status.get('episode_over')}; {len(created)} work order(s) created, {len(closed)} closed; {len(refused)} refused call(s).")
    if end:
        out.append(f"End reason {end.get('reason')!r} at cycle {end.get('time')}.")  # the cost and the downtime stay in the hidden verdict
    if refused:
        out.append("Refused calls: " + "; ".join(f"{json.dumps(e.get('action'), separators=(',', ':'))} ({e.get('reason')})" for e in refused))
    out.append("")

    answer = (finish or {}).get("summary") if finish else _read(ep_dir / "answer.txt")
    if answer is not None:
        out += ["## Static answer" if meta.get("mode") == "static" else "## Closing summary", ""]
        out += ["> " + line for line in (str(answer).strip() or "(empty)").splitlines()]
        out.append("")

    if calls:
        out += ["## Model calls", "", f"{len(calls)} calls, {sum(c.get('input_tokens', 0) for c in calls)} input tokens, "
                f"{sum(c.get('output_tokens', 0) for c in calls)} output tokens, {sum(c.get('seconds', 0) for c in calls):.1f} s in the model.", ""]
    harness = _read(ep_dir / "harness.json")
    if harness:
        standing = harness.get("standing") or {}
        cut = sum(1 for c in calls or [] if c.get("finish_reason") == "length")
        failed = sum(1 for c in calls or [] if c.get("error"))
        out += ["## Harness", "",
                f"Attempts {harness.get('attempts')}, class of the attempt that stands {standing.get('class') or 'none'}, finished by "
                f"{standing.get('finish_by') or 'the environment'}, runner end {standing.get('runner_end') or 'none'}"
                + (", cut at the wall limit" if standing.get("wall_limit") else "") + f"; {cut} cut repl{'y' if cut == 1 else 'ies'}, {failed} failed model call(s)."]
        if harness.get("set_aside"):
            aside = harness["set_aside"]
            out.append(f"The attempt set aside in {aside.get('folder')}/ was of the class {aside.get('class')}: {_short(aside.get('error') or '', 160)}")
        out.append("")
    out += [f"The oracle, the verdict and the scores of this episode are under {HIDDEN}/ and are not shown here.", ""]
    return "\n".join(out)


def run_summary(run_dir):
    """Counts of the episodes and of the files a run wrote, the episodes whose error names a context overflow or an error of the
    endpoint with the input tokens per model call, and a link to every episode's report; no score."""
    run_dir = Path(run_dir)
    meta = _read(run_dir / "run.json") or {}
    rows = read_rows(run_dir)
    folders = episode_dirs(run_dir)
    tokens, calls = input_tokens_per_call(folders)
    per_call = f"{tokens:.1f} over {calls} call{'s' if calls != 1 else ''}" if calls else "none (no call recorded)"
    out = [f"# Run {run_dir.name}", "",
           f"Config {meta.get('config_name')}, agent {meta.get('agent')}" + (f" ({meta['model']})" if meta.get("model") else "")
           + f", servers {', '.join(meta.get('servers', []) or ['none'])}, clock mode {meta.get('clock_mode')}, guidance level {meta.get('guidance_level')}, "
           f"price table {meta.get('prices')}, verifier scope {meta.get('verifier_scope')}, judge {meta.get('judge')}.",
           f"Episodes planned {meta.get('episodes_planned')}, episode folders {len(folders)}, rows in scores.csv {len(rows)}, "
           f"rows with an error {sum(1 for r in rows if r.get('error'))}. Started {meta.get('started')}, ended {meta.get('ended')}.",
           f"Episodes whose error names a context overflow or an API error {sum(1 for r in rows if context_or_api_error(r.get('error')))} of {len(rows)}, "
           f"input tokens per model call {per_call}.",
           f"Harness: episode folders holding harness.json {sum(1 for f in folders if (f / 'harness.json').exists())}, rows finished by the harness "
           f"{sum(1 for r in rows if r.get('finish_by') == 'harness')}, rows with a replay {sum(1 for r in rows if str(r.get('replays') or '0') not in ('', '0'))}, "
           f"rows of the class harness_error {sum(1 for r in rows if r.get('error_class') == 'harness_error')}, cut replies "
           f"{sum(int(r.get('cut_replies') or 0) for r in rows)}, failed calls {sum(int(r.get('failed_calls') or 0) for r in rows)}.", ""]
    if meta.get("anchored"):
        out += ["## Anchored scenario assumptions", ""] + [f"- {a}" for a in meta["anchored"].get("assumptions", [])] + [""]
    out += ["## Files per episode", "", "| file | episodes holding it |", "|---|---|"]
    for name in PUBLIC_FILES:
        out.append(f"| {name} | {sum(1 for f in folders if (f / name).exists())} |")
    for name in HIDDEN_FILES:
        out.append(f"| {HIDDEN}/{name} | {sum(1 for f in folders if (f / HIDDEN / name).exists())} |")
    out += ["", "Scores are in scores.csv and under each episode's hidden/ folder; this summary states none.", ""]
    out += ["## Episodes", "", "| episode | error | report |", "|---|---|---|"]
    for r in rows:
        name = episode_folder(run_dir, r).name
        out.append(f"| {name} | {_short(r.get('error', ''), 80)} | [{name}]({name}/report.md) |")
    out.append("")
    return "\n".join(out)
