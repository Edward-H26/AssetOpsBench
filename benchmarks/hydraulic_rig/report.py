"""Audit reports rendered from the files of an episode folder and the score table of a run, without a model.

episode_report writes what a reader needs to check an episode: the plan, every tool call of both servers with its
arguments and a one-line result, the predictions the causal tools stored, the diagnosis against the hidden start
combination (which the scorer reveals after the episode), each replacement with the recorded outcome, and the closing
answer beside what the log says. run_summary aggregates scores.csv per agent and links every episode's report.
"""
import csv
import json
import statistics
from pathlib import Path

from common import CAUSAL_SERVER, RIG_SERVER, answer_claims_recovery
from envs.hydraulic_rig.data import COMPONENTS

MAX_RESULT = 220


def _read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def _combo(values):
    return "(" + ", ".join(f"{c} {v}" for c, v in zip(COMPONENTS, values)) + ")"


def _short(text, limit=MAX_RESULT):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def summarise_response(server, tool, response, error=None):
    """One line per tool answer: the rig server's event and counters, the causal server's message, or the error."""
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
    if "note" in data:
        return "the episode was over; the call changed nothing"
    event = data.get("event")
    status = f"budget left {data.get('budget_left')}, repairs left {data.get('repairs_left')}, over {data.get('episode_over')}"
    if not event:
        return _short(f"{data}")
    kind = event.get("event")
    if kind == "measure":
        return f"measured tier {event['tier']} for {event['cost']} ({len(event.get('result', {}))} values{', repeated cycle' if event.get('repeated') else ''}); {status}"
    if kind == "submit":
        if event.get("insufficient_evidence"):
            return f"submitted insufficient evidence, next action {event.get('next_action')!r}; {status}"
        return f"submitted {event.get('diagnosis')}; {status}"
    if kind == "replace":
        return f"replaced {event['component']} for {event['cost']}; {status}"
    if kind == "refused":
        return f"refused: {event.get('reason')}; {status}"
    return _short(f"{kind}; {status}")


def _tool_rows(ep_dir):
    """The tool calls in order: the trajectory of both servers when the runner wrote one, else the rig server's own
    record of its calls (the policies play through the rig server only)."""
    trajectory = _read(ep_dir / "trajectory.json")
    if trajectory:
        return [(r["step"], f"{r['server']}.{r['tool']}" if r.get("tool") else r["server"], json.dumps(r.get("tool_args", {}), separators=(",", ":")),
                 summarise_response(r["server"], r.get("tool"), r.get("response", ""), r.get("error"))) for r in trajectory]
    state = _read(ep_dir / "episode.json") or {}
    return [(n, f"{RIG_SERVER}.{c['tool']}", json.dumps(c["arguments"], separators=(",", ":")), summarise_response(RIG_SERVER, c["tool"], c["response"]))
            for n, c in enumerate(state.get("tool_calls", []), 1)]


def episode_report(ep_dir, meta=None):
    ep_dir = Path(ep_dir)
    meta = meta or _read(ep_dir.parent / "run.json") or {}
    log = _read(ep_dir / "log.json") or []
    scores = _read(ep_dir / "score.json") or {}
    released = _read(ep_dir / "released.json") or {}
    predictions = _read(ep_dir / "predictions.json") or []
    answer = _read(ep_dir / "answer.txt")
    calls = _read(ep_dir / "llm_calls.json")
    hidden = tuple(log[0]["hidden"]) if log else None
    seed = log[0]["seed"] if log else ep_dir.name
    out = [f"# Episode {ep_dir.name}", ""]
    out.append(f"Scenario {meta.get('scenario', '?')}, agent {meta.get('agent', '?')}"
               + (f" ({meta['model']})" if meta.get("model") else "") + f", servers {', '.join(meta.get('servers', []) or ['none'])}, seed {seed}.")
    if released.get("task_text"):
        out.append(f"Task text: {released['task_text']}")
    if hidden:
        out.append(f"Hidden start combination, revealed by the scorer after the episode: {_combo(hidden)}.")
    out.append("")

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
        out += ["## Predictions stored by the causal tools", "", "| id | action | after measurement | addresses a fault | recovery |", "|---|---|---|---|---|"]
        out += [f"| {p['id']} | {p['action']} | {p['measurements']} | {p['addresses']:.3f} | {p['recovery']:.3f} |" for p in predictions]
        out.append("")

    out += ["## Outcome from the log", ""]
    if not log:
        out.append("No log.")
    else:
        submitted = next((e for e in log if e["event"] == "submit"), None)
        if submitted is None:
            out.append("No diagnosis was submitted.")
        elif submitted.get("insufficient_evidence"):
            out.append(f"The agent submitted insufficient evidence, naming the next action {submitted.get('next_action')!r}.")
        else:
            diagnosis = submitted["diagnosis"]
            out += ["| component | truth | diagnosed | right |", "|---|---|---|---|"]
            out += [f"| {c} | {hidden[i]} | {diagnosis.get(c)} | {'yes' if diagnosis.get(c) == hidden[i] else 'no'} |" for i, c in enumerate(COMPONENTS)]
            out.append("")
            out.append(f"Whole combination right: {'yes' if scores.get('diagnosis_all_correct') else 'no'}.")
        replacements = [e for e in log if e["event"] == "replace"]
        if replacements:
            out += ["", "| replacement | component | was faulty | combination after | cost |", "|---|---|---|---|---|"]
            out += [f"| {n} | {e['component']} | {'no (unnecessary)' if e['unnecessary'] else 'yes'} | {_combo(e['combo_after'])} | {e['cost']} |"
                    for n, e in enumerate(replacements, 1)]
        refused = [e for e in log if e["event"] == "refused"]
        if refused:
            out += ["", "Refused calls: " + "; ".join(f"{e.get('action')} ({e.get('reason')})" for e in refused)]
        out += ["", f"Recovered: {'yes' if scores.get('recovered') else 'no'}; end reason {scores.get('end_reason')!r}; cost {scores.get('cost')} "
                f"(before the submission {scores.get('cost_before_submit')}); steps {scores.get('steps')}; measurements per tier "
                f"{scores.get('measures_tier1')}/{scores.get('measures_tier2')}/{scores.get('measures_tier3')}; unnecessary replacements "
                f"{scores.get('unnecessary_replacements')}; repairs match the diagnosis: {'yes' if scores.get('repairs_match_diagnosis') else 'no'}."]
    out.append("")

    if answer is not None:
        claims = answer_claims_recovery(answer)
        recovered = bool(scores.get("recovered"))
        out += ["## Closing answer", ""]
        out += ["> " + line for line in (answer.strip() or "(empty)").splitlines()]
        out += ["", f"Heuristic reading of the answer: {'claims a recovery' if claims else 'claims no recovery'}; the log says "
                f"{'recovered' if recovered else 'not recovered'}: {'agreement' if claims == recovered else 'DISAGREEMENT, read the answer'}.", ""]

    if calls:
        out += ["## Model calls", "", f"{len(calls)} calls, {sum(c.get('input_tokens', 0) for c in calls)} input tokens, "
                f"{sum(c.get('output_tokens', 0) for c in calls)} output tokens, {sum(c.get('seconds', 0) for c in calls):.1f} s in the model.", ""]
    return "\n".join(out)


def _as_bool(value):
    return str(value).strip().lower() == "true"


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(rows, key):
    values = [_as_bool(r.get(key)) for r in rows]
    return f"{sum(values) / len(values):.2f}" if values else ""


def _mean(rows, key):
    values = [v for v in (_as_float(r.get(key)) for r in rows) if v is not None]
    return f"{statistics.mean(values):.1f}" if values else ""


def run_summary(run_dir):
    run_dir = Path(run_dir)
    meta = _read(run_dir / "run.json") or {}
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = [f"# Run {run_dir.name}", "",
           f"Scenario {meta.get('scenario')}, agent {meta.get('agent')}" + (f" ({meta['model']})" if meta.get("model") else "")
           + f", servers {', '.join(meta.get('servers', []) or ['none'])}, {len(rows)} episodes, started {meta.get('started')}, ended {meta.get('ended')}.", ""]
    out += ["## Per agent", "", "| agent | episodes | submitted | whole combination right | recovered | mean cost | unnecessary replacements | insufficient evidence | mean steps | errors |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    agents = list(dict.fromkeys(r["agent"] for r in rows))
    for agent in agents:
        rs = [r for r in rows if r["agent"] == agent]
        unnecessary = sum(int(_as_float(r.get("unnecessary_replacements")) or 0) for r in rs)
        errors = sum(1 for r in rs if r.get("error"))
        out.append(f"| {agent} | {len(rs)} | {_rate(rs, 'submitted')} | {_rate(rs, 'diagnosis_all_correct')} | {_rate(rs, 'recovered')} | {_mean(rs, 'cost')} | "
                   f"{unnecessary} | {_rate(rs, 'insufficient_evidence')} | {_mean(rs, 'steps')} | {errors} |")
    out.append("")
    if any("answer_claims_recovery" in r for r in rows):
        claims = [r for r in rows if _as_bool(r.get("answer_claims_recovery"))]
        false_claims = [r for r in claims if not _as_bool(r.get("recovered"))]
        missed = [r for r in rows if _as_bool(r.get("recovered")) and not _as_bool(r.get("answer_claims_recovery"))]
        out += ["## Closing answers against the log (heuristic reading, see each report)", "",
                f"{len(claims)} of {len(rows)} answers claim a recovery; {len(false_claims)} of those episodes did not recover; "
                f"{len(missed)} recovered episodes have an answer that claims none.",
                f"Causal tool calls per episode: {_mean(rows, 'causal_calls')}; model calls per episode: {_mean(rows, 'llm_calls')}; "
                f"input tokens per episode: {_mean(rows, 'input_tokens')}.", ""]
    out += ["## Episodes", "", "| seed | record | ended | whole combination right | recovered | unnecessary | cost | steps | error | report |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = f"seed{int(r['seed']):06d}"
        out.append(f"| {r['seed']} | {r.get('record', '')} | {r.get('end_reason', '')} | {r.get('diagnosis_all_correct')} | {r.get('recovered')} | "
                   f"{r.get('unnecessary_replacements')} | {r.get('cost')} | {r.get('steps')} | {_short(r.get('error', ''), 60)} | [{name}]({name}/report.md) |")
    out.append("")
    return "\n".join(out)
