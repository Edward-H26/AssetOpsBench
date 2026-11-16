"""A step-through page of one episode, one slide per tool call, for a person who diagnoses what happened. It is the
companion of walkthrough/, whose parsing of the episode folder it reuses: walkthrough/ tells the whole episode as one
document, while this page shows one call at a time with who acted, the tool and its arguments, what the agent was told,
the clock, the budget and the steps left after the call, the notifications, the open work orders and the thinking time
charged before the call.

    PATH=$PWD/.venv/bin:$PATH python -m CausalDiagnosis.episode_viewer.step_through_page <episode folder> [--out file.html]

The page is one HTML file with its style and script inline and no network reference, written to --out or to standard
output, never into the episode folder, whose files the run ledger hashes, and never inside the fork. A toggle shows the
hidden side of every step, the hidden combination after the call and the entries of hidden/log.json the call caused. The
toggle is off when the page opens and every hidden value sits inside an element of the class hidden-side, since the page is
for people and never for an agent. The transcript of the servers places the thinking charge and the hidden entries of a
call: a call of the runner's record is joined to its line by the hash of its answer, and a call that the transcript does
not hold is not placed, alone.
"""
import argparse
import json
import sys
from html import escape
from pathlib import Path

from CausalDiagnosis.episode_viewer import walkthrough
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import refuse_page_output
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import wall_clock
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time
from CausalDiagnosis.mcp_servers.utils import transcript

HIDDEN_CLASS = "hidden-side"
NO_STEP = 0  # the place of a call that took no step of the environment
NOT_PLACED = "cannot be placed from the record of the servers"
PEOPLE_ONLY = ("The page shows the hidden state of a finished run: its source holds every hidden value of the episode, also while "
               "the toggle is off, so the page is for people and is never handed to an agent or published before the results of its run.")
BEFORE_TRANSCRIPT = ("The episode was recorded before the transcript of the servers existed, so its thinking charges and hidden "
                     "entries are placed by the count of the calls of the action servers.")

STYLE = """
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1d232b; background: #f4f5f7; }
header { padding: 14px 24px; background: #fff; border-bottom: 1px solid #d9dde3; }
header h1 { margin: 0 0 4px; font-size: 19px; }
header .facts { color: #4a5563; font-size: 14px; }
.layout { display: grid; grid-template-columns: 250px minmax(0, 1fr); min-height: calc(100vh - 120px); }
nav { background: #fff; border-right: 1px solid #d9dde3; overflow-y: auto; max-height: calc(100vh - 120px); position: sticky; top: 0; }
nav ol { list-style: none; margin: 0; padding: 6px 0; }
nav .legend { padding: 8px 14px 0; font-size: 12px; color: #6b7684; }
nav a { display: block; padding: 4px 14px; color: #1d232b; text-decoration: none; font-size: 13px; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; }
nav a .n { display: inline-block; width: 2.2em; color: #6b7684; }
nav a.diagnostics { color: #4b5f8a; }
nav a[aria-current="true"] { background: #e3ecfa; font-weight: 600; }
main { padding: 16px 24px 40px; max-width: 980px; }
.controls { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.controls button { font: inherit; padding: 6px 14px; border: 1px solid #b8c0cc; border-radius: 6px; background: #fff; cursor: pointer; }
.controls button:disabled { opacity: 0.4; cursor: default; }
.controls .counter { min-width: 7em; color: #4a5563; }
.controls .keys { color: #6b7684; font-size: 13px; }
.toggle { margin-left: auto; padding: 5px 10px; border: 2px solid #b04a00; border-radius: 6px; color: #8a3a00; background: #fff6ee; }
section.step { background: #fff; border: 1px solid #d9dde3; border-radius: 8px; padding: 16px 20px; }
section.step h2 { margin: 0 0 12px; font-size: 18px; }
section.step h2 .server { font-weight: 400; color: #6b7684; font-size: 15px; }
dl { display: grid; grid-template-columns: 190px minmax(0, 1fr); gap: 6px 16px; margin: 0; }
dt { color: #4a5563; }
dd { margin: 0; }
.numbers { display: flex; gap: 12px; margin: 0 0 14px; flex-wrap: wrap; }
.numbers div { padding: 6px 12px; background: #f4f5f7; border-radius: 6px; }
.numbers b { display: block; font-size: 18px; }
pre { margin: 4px 0 0; padding: 8px 10px; background: #f6f8fa; border-radius: 6px; white-space: pre-wrap; word-break: break-word;
      font: 12.5px/1.45 Menlo, Consolas, monospace; max-height: 340px; overflow-y: auto; }
details summary { cursor: pointer; color: #2d5fa8; }
ul.plain { margin: 0; padding-left: 18px; }
.hidden-side { display: none; margin-top: 16px; padding: 12px 16px; border: 2px dashed #b04a00; border-radius: 8px; background: #fff6ee; }
body.show-hidden .hidden-side { display: block; }
.hidden-side h3 { margin: 0 0 8px; font-size: 15px; color: #8a3a00; }
header .hidden-side { margin-top: 10px; }
header .note { margin: 6px 0 0; color: #8a3a00; font-size: 13px; }
@media (max-width: 720px) {
  .layout { grid-template-columns: minmax(0, 1fr); min-height: 0; }
  nav { position: static; max-height: 30vh; border-right: none; border-bottom: 1px solid #d9dde3; }
  main { padding: 12px 16px 32px; }
  dl { grid-template-columns: minmax(0, 1fr); }
  .toggle { margin-left: 0; }
}
"""

SCRIPT = """
const steps = Array.from(document.querySelectorAll("section.step"));
const links = Array.from(document.querySelectorAll("nav a"));
const prev = document.getElementById("prev"), next = document.getElementById("next"), counter = document.getElementById("counter");
let current = 0;
function show(n) {
  current = Math.max(0, Math.min(steps.length - 1, n));
  steps.forEach((s, k) => { s.hidden = k !== current; });
  links.forEach((a, k) => { a.setAttribute("aria-current", k === current ? "true" : "false"); });
  links[current].scrollIntoView({block: "nearest"});
  counter.textContent = "Step " + (current + 1) + " of " + steps.length;
  prev.disabled = current === 0;
  next.disabled = current === steps.length - 1;
  history.replaceState(null, "", "#" + steps[current].id);
}
prev.addEventListener("click", () => show(current - 1));
next.addEventListener("click", () => show(current + 1));
links.forEach((a, k) => a.addEventListener("click", (e) => { e.preventDefault(); show(k); }));
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); show(current + 1); }
  if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); show(current - 1); }
});
const toggle = document.getElementById("show-hidden");
toggle.addEventListener("change", () => document.body.classList.toggle("show-hidden", toggle.checked));
const start = steps.findIndex((s) => "#" + s.id === location.hash);
if (steps.length) show(start < 0 ? 0 : start);
"""


def placed_steps(ep):
    """(the place of every tool call, whether the episode has a transcript). The place is the step of the environment that
    the call took, 1 for the first, NO_STEP for a call that took none, and None for a call that the record cannot place. A
    call of the runner's record is joined to the next line of its server in the transcript whose answer hash is the hash of
    the text the runner received, and the k-th line that took a step took step k. An episode recorded before the transcript
    existed places the calls of the action servers by their count, and places none when that count is not the number of
    steps of the state file."""
    lines = [line for line in transcript.read(ep.folder) if line.get("kind") == "call"]
    if not lines:
        steps = [None if call.server in walkthrough.ACTION_SERVERS else NO_STEP for call in ep.calls]
        actions = [i for i, step in enumerate(steps) if step is None]
        if len(actions) == len(ep.state.get("think_cycles") or []):
            for k, i in enumerate(actions, 1):
                steps[i] = k
        return steps, False
    step_of, by_server = {}, {}
    for line in lines:
        if line.get("stepped"):
            step_of[line["n"]] = len(step_of) + 1
        by_server.setdefault(line.get("server"), []).append(line)
    taken, steps = {server: 0 for server in by_server}, []
    for call in ep.calls:
        queue, start, digest = by_server.get(call.server, []), taken.get(call.server, 0), transcript.text_sha256(call.text)
        found = next((i for i in range(start, len(queue)) if queue[i].get("answer_sha256") == digest), None)
        if found is None:
            steps.append(None)
            continue
        taken[call.server] = found + 1
        steps.append(step_of.get(queue[found]["n"], NO_STEP))
    return steps, True


def charge_sequence(config, recorded, seconds):
    """The thinking charge of every action call by the engine's rule (wall_clock.think_cycles): the call's seconds,
    the seconds carried in from the calls before, the cycles charged and the seconds carried on, beside the cycles the
    state file recorded, so a reader can add the seconds up."""
    out, carried = [], 0.0
    for cycles_recorded, call_seconds in zip(recorded, seconds):
        cycles, _charged, after = wall_clock.think_cycles(config, carried, call_seconds)
        out.append({"recorded": cycles_recorded, "cycles": cycles, "seconds": call_seconds, "carried_in": carried, "carried_out": after})
        carried = after
    return out


def think_charges(ep, steps, config):
    """The charge of charge_sequence before each tool call that took a step, by the step that placed_steps gives it. The
    state file holds one entry per step, so a call that took no step or that the record cannot place gets None."""
    recorded, seconds = ep.state.get("think_cycles") or [], ep.state.get("think_seconds") or []
    sequence = charge_sequence(config, recorded, seconds) if len(recorded) == len(seconds) else []
    return [sequence[step - 1] if step and step <= len(sequence) else None for step in steps]


def thinking_text(charge, config):
    """The seconds behind a charge with two decimals, so that the sum is checkable against the seconds per cycle of the
    episode, which one decimal could round up to the seconds per cycle. Whether the clock mode charges thinking time is the
    engine's rule (real_time.thinking), asked here for a thinking time of one full cycle."""
    if real_time.thinking(config, 1) == 0:
        return f"none, the {config.clock_mode} clock mode charges no thinking time"
    per_cycle = walkthrough.num(config.think_seconds_per_cycle)
    text = (f"{charge['seconds']:.2f} s of this call plus {charge['carried_in']:.2f} s carried in, at {per_cycle} s per cycle, make "
            f"{walkthrough.plural(charge['cycles'], 'cycle')} with {charge['carried_out']:.2f} s carried on")
    if charge["cycles"] != charge["recorded"]:
        text += f", while the state file records {walkthrough.plural(charge['recorded'], 'cycle')}"
    return text


def hidden_steps(ep, log, steps):
    """For each tool call, the hidden combination after it and the entries of the hidden log it caused, each with the
    combination before it: the entries of its step, from the call's own entry to the system entries before the entry of the
    next step. A log entry names its step, a call that took no step causes no entry, and a call that the record cannot place
    gets None in place of its entries."""
    entries, after = {}, {}
    combination, action = list(log[0].get("hidden") or []), 0
    for entry in log[1:]:
        action = entry["step"] if entry.get("step") is not None else action
        before, combination = combination, walkthrough.combination_after(entry, combination, ep.components)
        entries.setdefault(action, []).append((entry, before))
        after[action] = combination
    out, combination = [], list(log[0].get("hidden") or [])
    for step in steps:
        if step:
            combination = after.get(step, combination)
        out.append((combination, entries.get(step, []) if step else [] if step == NO_STEP else None))
    return out


def entry_text(entry, created, components, before):
    """One line for a hidden log entry, the walkthrough's sentence for the entries it tells and the event with its cycle
    for the rest."""
    row = walkthrough.hidden_row(entry, created, components, before)
    cycle = walkthrough.num(entry.get("t_end", entry.get("t_start")))
    return f"cycle {row[0]}, {row[1]}, {row[2]}" if row else f"cycle {cycle}, {str(entry['event']).replace('_', ' ')}"


def pre(value):
    return f"<pre>{escape(json.dumps(value, indent=2) if not isinstance(value, str) else value)}</pre>"


def answer_text(call):
    """The answer as the agent received it, its JSON indented when it holds one."""
    try:
        return json.dumps(json.loads(call.text), indent=2)
    except ValueError:
        return call.text


def server_name(server):
    """The name of a server on the page, the one walkthrough/ gives it."""
    return walkthrough.SERVER_NAMES.get(server, server)


def charge_cell(step, charge):
    """The thinking cycles charged before a call of an action server, as the cell of its numbers."""
    if step == NO_STEP:
        return "none, the call took no step"
    return "not placed" if charge is None else walkthrough.plural(charge["recorded"], "cycle")


def numbers_block(call, step, charge):
    if call.server not in walkthrough.ACTION_SERVERS:
        return (f"<p>The {escape(server_name(call.server))} server reads the evidence released so far, moves no clock and charges "
                "nothing, so the clock, the budget and the steps left are those of the call before.</p>")
    envelope = call.answer or {}
    cells = [(label, walkthrough.num(envelope[key]) if key in envelope else "not reported")
             for label, key in (("Clock after", "time"), ("Budget left", "budget_left"), ("Steps left", "steps_left"))]
    cells.append(("Thinking charged before the call", charge_cell(step, charge)))
    return '<div class="numbers">' + "".join(f"<div>{escape(label)}<b>{escape(value)}</b></div>" for label, value in cells) + "</div>"


def told_text(call, ep):
    """The walkthrough's one line summary of the answer, apart from the profile, which the walkthrough points to its part 1."""
    if call.tool == "get_asset_profile" and call.result is not None:
        return "the asset profile, in the full answer below"
    return walkthrough.describe_answer(call, ep)


def orders_text(envelope):
    if "open_work_orders" not in envelope:
        return "not reported by this server"
    orders = envelope["open_work_orders"] or []
    return walkthrough.joined(f"{o.get('id')} on the {o.get('component')}, {str(o.get('status')).replace('_', ' ')}, expected to close at "
                              f"cycle {walkthrough.num(o.get('expected_close_t'))}" for o in orders) or "none"


def thinking_row(call, step, charge, config):
    """The row of the thinking time of a call of an action server, none for a call that took no step."""
    if call.server not in walkthrough.ACTION_SERVERS or step == NO_STEP:
        return []
    if step is None or charge is None:
        return [("Thinking time", NOT_PLACED)]
    return [("Thinking time", thinking_text(charge, config))]


def public_rows(ep, call, plan_label, step, charge, config):
    """The rows of a step that the agent could know, the answer and the full answer included."""
    acting = call.server in walkthrough.ACTION_SERVERS
    envelope = call.answer if acting and call.answer is not None else {}
    server = server_name(call.server)
    notifications = (walkthrough.notifications_text(envelope) or "not reported by this server") if acting else f"none, the {server} server sends none"
    rows = [("Who acted", f"{ep.label()}, through the {server} server"),
            *([("Plan step", plan_label.replace("**", ""))] if plan_label else []),
            ("What the agent was told", told_text(call, ep)),
            ("Notifications", notifications),
            ("Open work orders", orders_text(envelope) if acting else "unchanged"),
            *thinking_row(call, step, charge, config)]
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    body += f"<dt>Arguments</dt><dd>{pre(call.args if call.args is not None else {})}</dd>"
    return body + (f"<dt>Full answer</dt><dd><details><summary>Show the answer as the agent received it</summary>"
                   f"{pre(answer_text(call))}</details></dd>")


def hidden_block(ep, hidden, created):
    combination, entries = hidden
    if entries is None:
        after = listed = escape(NOT_PLACED)
    else:
        after = escape(walkthrough.long_levels(combination, ep.components))
        lines = "".join(f"<li>{escape(entry_text(e, created, ep.components, before))}<details><summary>log entry</summary>{pre(e)}</details></li>"
                        for e, before in entries)
        listed = f'<ul class="plain">{lines}</ul>' if lines else "none"
    return (f'<div class="{HIDDEN_CLASS}"><h3>Hidden side, never shown to the agent</h3><dl>'
            f"<dt>Hidden combination after the call</dt><dd>{after}</dd>"
            f"<dt>Hidden log entries of the call</dt><dd>{listed}</dd></dl></div>")


def step_section(ep, n, call, plan_label, step, charge, config, hidden, created):
    return (f'<section class="step" id="step-{n}"><h2>Step {n}. <code>{escape(call.tool)}</code> '
            f'<span class="server">on the {escape(server_name(call.server))} server</span></h2>'
            + numbers_block(call, step, charge) + f"<dl>{public_rows(ep, call, plan_label, step, charge, config)}</dl>"
            + (hidden_block(ep, hidden, created) if hidden is not None else "") + "</section>")


def header(ep, log, transcribed):
    c = ep.config
    facts = (f"Run {ep.folder.parent.name}, episode {ep.folder.name}, preset {ep.setting('config_name', 'config') or 'not recorded'}, "
             f"clock mode {c['clock_mode']}, budget {walkthrough.num(c['budget'])}, horizon {walkthrough.num(c['horizon'])} cycles, "
             f"step cap {walkthrough.num(c['max_steps'])}, {walkthrough.plural(len(ep.calls), 'tool call')}")
    return (f"<header><h1>Seed {escape(walkthrough.num(ep.state.get('seed')))}, agent {escape(ep.label())}</h1>"
            f'<div class="facts">{escape(facts)}</div>'
            + "".join(f'<p class="note">{escape(note)}</p>' for note in (PEOPLE_ONLY, *(() if transcribed else (BEFORE_TRANSCRIPT,))))
            + f"<details><summary>The briefing the agent received at step one</summary>{pre(ep.question)}</details>"
            f'<div class="{HIDDEN_CLASS}"><h3>Hidden side, never shown to the agent</h3>{hidden_start(ep, log)}</div></header>')


def hidden_start(ep, log):
    """The start the reset drew with its scheduled drops, and the verdict."""
    if not log:
        return "<p>This episode folder holds no hidden log, so the page has no hidden side.</p>"
    end = next((e for e in log if e["event"] == "end"), {})
    verdict = ep.hidden("verdict.json") or {}
    start = walkthrough.start_row(log[0], end.get("played") or [], ep.components)[2]
    verdict_text = (f"failure class {verdict.get('failure_class')}, pass {walkthrough.num(verdict.get('pass'))}"
                    if verdict.get("verified") else "no verdict in this episode folder")
    return f"<dl><dt>Start</dt><dd>{escape(start)}</dd><dt>Verdict</dt><dd>{escape(verdict_text)}</dd></dl>"


def render(folder):
    ep = walkthrough.Episode(folder)
    log = ep.hidden("log.json")
    labels, _note = walkthrough.plan_column(ep)
    steps, transcribed = placed_steps(ep)
    config = EpisodeConfig.from_json(ep.state["config"])
    charges = think_charges(ep, steps, config)
    hidden = hidden_steps(ep, log, steps) if log else [None] * len(ep.calls)
    created = {e["id"]: e for e in log or [] if e["event"] == "work_order_created"}
    nav = "".join(f'<li><a href="#step-{n}" class="{"diagnostics" if call.server not in walkthrough.ACTION_SERVERS else "action"}">'
                  f'<span class="n">{n}</span>{escape(call.tool)}</a></li>' for n, call in enumerate(ep.calls, 1))
    slides = "".join(step_section(ep, n, call, labels[n - 1], steps[n - 1], charges[n - 1], config, hidden[n - 1], created)
                     for n, call in enumerate(ep.calls, 1)) or "<p>The agent made no tool call.</p>"
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Steps of {escape(ep.folder.name)}</title>"
            f"<style>{STYLE}</style></head><body>{header(ep, log, transcribed)}<div class=\"layout\"><nav>"
            f"<div class=\"legend\">Blue names are calls of the {escape(server_name(CAUSAL_SERVER))} server</div><ol>{nav}</ol></nav><main>"
            '<div class="controls"><button id="prev" type="button">Previous</button><button id="next" type="button">Next</button>'
            '<span class="counter" id="counter"></span><span class="keys">arrow keys move too</span>'
            '<label class="toggle"><input type="checkbox" id="show-hidden"> Show the hidden side (for people only)</label></div>'
            f"{slides}</main></div><script>{SCRIPT}</script></body></html>\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the step-through page of one episode folder as one HTML file.")
    ap.add_argument("episode", help="an episode folder of a run, such as <run folder>/seed000015")
    ap.add_argument("--out", help="the HTML file to write, standard output when absent")
    args = ap.parse_args(argv)
    if args.out:
        refuse_page_output(args.out, args.episode)
    text = render(args.episode)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
