"""The data builder of the episode explorer: what the page of one episode shows, read from its run folder and nothing else.

build(episode folder, out folder, hidden=False) writes into out, never into the run folder: data.json holds what an agent of
the episode could see plus the public facts of every run under the runs folder, files/ holds the allowed public files of the
episode and run folders with every machine path shortened, and only with hidden=True hidden.json and hidden_files/ hold the
hidden side (hidden/*.json, the ledger, the score columns). A launcher (*_server.py) and a lock file are never read: a
launcher names the hidden start. The public side reads scores.csv and ledger.json through allow lists, never deny lists,
so a column or key a later version adds stays out until someone puts it on a list. Whether a call followed the advice is
decided by the functions the measures of the CausalDiagnosis MCP use (mcp_metrics.py), never by a rule of its own.

Every answer keeps the text the agent received. An answer of the CausalDiagnosis MCP, and the profile of the task server of the
AssetOpsBench rows, carries the level of response_format it was written at (mcp_servers/utils/response_levels.py); a profile at a
shorter level, which writes the tool table and the normal ranges as long tables, also carries its raw form, rebuilt by
profile_levels.raw_of. Every answer of the CausalDiagnosis MCP is written at all four levels into levels/<n>.json: the level the
agent received as it received it, and the others from its raw answer, the served text at raw or the record the server kept beside
the line of the call (transcript.RAW_NAME), by the level code of the server (answer_levels.at_level). Only with hidden=True, the
raw answers of a run that kept none are recomputed (recompute.py) into hidden_levels/<n>.json. An answer of AssetOpsBench's servers reads as the envelope of the rig (walkthrough/episode_files.py). For
an episode of the CausalDiagnosis agent, every planning request of llm_calls.json gives the board the model read, cut into its
sections, and every call names the board of the plan that made it (walkthrough/call_placement.py).
"""
import csv
import itertools
import io
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import OBJECTIVE, PLANNING_HEADING
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import cost_unit
from CausalDiagnosis.episode_viewer.walkthrough import call_placement, controller_walk, episode_files
from CausalDiagnosis.evaluation.metrics import mcp_metrics
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, METERED_SERVERS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.answer_levels import at_level
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.mcp_servers.utils.profile_levels import raw_of
from CausalDiagnosis.mcp_servers.utils.response_levels import LEVELS, level_of, text_of
from CausalDiagnosis.utils.public_rules.visits import is_no_call

# Kinds of call for the lanes of the timeline, by the tools of the engine's action surface. A tool missing here falls back on
# the class that the released rules give it, so a ported asset with tools of its own shows without an edit.
KIND_OF_TOOL = {
    "get_current_time": "time", "wait_for_notification": "time",
    "create_work_order": "orders",
    "submit_diagnosis": "decide", "declare_insufficient_evidence": "decide", "finish": "decide",
}
# Score columns of scores.csv that the verifier computes from the hidden state; they go to hidden.json.
PUBLIC_SCORE_COLUMNS = {"run", "seed", "scenario_id", "date", "config", "agent", "model", "clock_mode", "guidance_level",
                        "diagnostics_server", "prices", "verifier_scope", "ranking_model", "wall_s", "error",
                        "end_reason", "time_used", "steps", "tool_calls", "refusals", "causal_calls", "llm_calls",
                        "input_tokens", "output_tokens", "plan_steps", "plans"}
# Allow lists of the files the page may show. scores.csv and ledger.json hold hidden columns and blocks, so they sit on the
# hidden side whole and reach the public side only through the allow lists below.
PUBLIC_EPISODE_FILES = {"trajectory.json", "transcript.jsonl", "released.json", "predictions.json", "episode.json",
                        "question.txt", "answer.txt", "plan.txt", "report.md", "harness.json", "llm_calls.json",
                        "plans.json", "belief.json"}
PUBLIC_RUN_FILES = {"run.json", "summary.md"}
HIDDEN_RUN_FILES = {"scores.csv", "ledger.json"}
LEDGER_PUBLIC_KEYS = ("format", "run", "label", "agent", "agent_label", "status", "started", "updated", "ended", "judge",
                      "configuration", "seeds", "run_files", "events")
LEDGER_EPISODE_KEYS = ("seed", "files")
ESTIMATE = "estimate_intervention_effect"  # the tool of the CausalDiagnosis MCP whose answer is a stored prediction
PROFILE = "get_asset_profile"
ABS_PATH = re.compile(r"/(?:private|Users|tmp|var|home|scratch|opt)/[^\s\"',;)\]]*")


def shorten_paths(text, run_dir):
    """A path under the run folder becomes relative to it; any other absolute path keeps its last component, marked."""
    root = str(run_dir) + "/"
    text = text.replace(root, "")
    text = ABS_PATH.sub(lambda m: "\u2026/" + m.group(0).rstrip("/").rsplit("/", 1)[-1], text)
    # the name of the machine that played the run identifies it as much as a path does
    return re.sub(r'("host"\s*:\s*)"[^"]*"', lambda m: m.group(1) + '"\u2026 (machine name left out)"', text)


def load(path, default=None):
    """The content of a regular file, default for a missing file or a symbolic link, which may point outside the run folder."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return default
    return json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()


def parse_answer(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {"text": text}


def answer_of(server, text):
    """The answer of a call as the page reads it: the envelope of the metering layer on AssetOpsBench's servers, else the JSON
    the text holds."""
    return episode_files.aob_envelope(text) if server in METERED_SERVERS else parse_answer(text)


def answer_level(server, tool, args):
    """The level of response_format an answer was written at: every answer of the CausalDiagnosis MCP and the profile of the
    task server of the AssetOpsBench rows take the level the call names, or the default; None for a tool without levels."""
    if server == CAUSAL_SERVER or (server in METERED_SERVERS and tool == PROFILE):
        return level_of((args.get("response_format") if isinstance(args, dict) else None) or "")
    return None


def plan_lines(text):
    """The plan file of the plan-and-execute runner: #TaskN, #ServerN, #ToolN, #DependencyN, #ExpectedOutputN."""
    tasks = {}
    for line in (text or "").splitlines():
        if not line.startswith("#") or ":" not in line:
            continue
        key, value = line[1:].split(":", 1)
        name = key.rstrip("0123456789")
        number = key[len(name):]
        if number:
            tasks.setdefault(int(number), {})[name] = value.strip()
    return [tasks[k] for k in sorted(tasks)]


def model_messages(calls, llm):
    """Index of the model call whose reply is the call as sent: the first later entry of llm_calls.json whose reply parses
    to the arguments of the call. A call no entry matches keeps None, which the page shows as not recorded."""
    at = 0
    for c in calls:
        c["llm_index"] = None
        for i in range(at, len(llm)):
            try:
                reply = json.loads(llm[i].get("reply") or "")
            except ValueError:
                continue
            if reply == c["args"]:
                c["llm_index"], at = i, i + 1
                break


def copy_files(folder, run_dir, out, with_hidden):
    """Copies of the allowed files: byte copies on the hidden side; on the public side the same bytes with every machine
    path shortened, plus scores.csv and ledger.json reduced to their allow-listed columns and keys."""
    def stat(dest, data, note=None):
        row = {"name": dest.name, "bytes": len(data), "lines": data.count(b"\n") + (0 if data.endswith(b"\n") else 1)}
        return row | ({"note": note} if note else {})

    def inside(p, root):
        # a symbolic link may point anywhere on the machine, so only a regular file that stays in its folder is read
        return p.is_file() and not p.is_symlink() and p.resolve().parent == root.resolve()

    def put(src, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return stat(dest, src.read_bytes())

    def put_public(src, dest, text=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_text() if text is None else text
        short = shorten_paths(raw, run_dir)
        data = short.encode()
        dest.write_bytes(data)
        return stat(dest, data, "Paths on the computer that ran this are shortened, to the run folder or to their last part after \u2026/." if short != raw else None)

    def public_extras():
        rows = []
        scores = run_dir / "scores.csv"
        if inside(scores, run_dir):
            reader = list(csv.DictReader(open(scores)))
            cols = [c for c in (reader[0].keys() if reader else []) if c in PUBLIC_SCORE_COLUMNS]
            left = [c for c in (reader[0].keys() if reader else []) if c not in PUBLIC_SCORE_COLUMNS]
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader(); w.writerows(reader)
            r = put_public(scores, out / "files" / "run" / "scores.public.csv", buf.getvalue())
            rows.append(r | {"note": f"Only the score columns that hide no ground truth are shown. {len(left)} columns are left out ({', '.join(left)})."})
        ledger = run_dir / "ledger.json"
        if inside(ledger, run_dir):
            full = json.loads(ledger.read_text())
            pub = {k: full[k] for k in LEDGER_PUBLIC_KEYS if k in full}
            pub["episodes"] = {e: {k: v[k] for k in LEDGER_EPISODE_KEYS if k in v} for e, v in (full.get("episodes") or {}).items()}
            left = [k for k in full if k not in LEDGER_PUBLIC_KEYS and k != "episodes"]
            r = put_public(ledger, out / "files" / "run" / "ledger.public.json", json.dumps(pub, indent=1))
            rows.append(r | {"note": f"Only the parts of the ledger that hide no ground truth are shown. Left out are {', '.join(left)}, and of each episode everything but {', '.join(LEDGER_EPISODE_KEYS)}."})
        return rows
    for sub in ("files", "hidden_files"):
        shutil.rmtree(out / sub, ignore_errors=True)
    aob = f"{run_dir.name}-{folder.name}.json"  # the trajectory in the form of the AssetOpsBench runners
    public = {"ep": [put_public(p, out / "files" / "ep" / p.name) for p in sorted(folder.iterdir())
                     if inside(p, folder) and (p.name in PUBLIC_EPISODE_FILES or p.name == aob)],
              "run": [put_public(p, out / "files" / "run" / p.name) for p in sorted(run_dir.iterdir())
                      if inside(p, run_dir) and p.name in PUBLIC_RUN_FILES]
              + public_extras()}
    if not with_hidden:
        return public, None
    hidden = {"ep": [put(p, out / "hidden_files" / "ep" / "hidden" / p.name) | {"name": f"hidden/{p.name}"}
                     for p in sorted((folder / "hidden").iterdir()) if inside(p, folder / "hidden") and p.suffix == ".json"]
              if (folder / "hidden").is_dir() and not (folder / "hidden").is_symlink() else [],
              "run": [put(p, out / "hidden_files" / "run" / p.name) for p in sorted(run_dir.iterdir())
                      if inside(p, run_dir) and p.name in HIDDEN_RUN_FILES]}
    return public, hidden


def kind_of(tool, row, stepped):
    if stepped is False:
        return "advice"
    if tool in KIND_OF_TOOL:
        return KIND_OF_TOOL[tool]
    return "decide" if (row or {}).get("class") == "write" else "reads"


def start_states(rules, fault_rule):
    """Every combination with between min_faulty and max_faulty components off their healthy level."""
    comps, levels, healthy = rules["components"], rules["levels"], rules["healthy"]
    lo, hi = fault_rule.get("min_faulty", 1), fault_rule.get("max_faulty", len(comps))
    out = []
    for combo in itertools.product(*(levels[c] for c in comps)):
        faulty = sum(1 for c, v in zip(comps, combo) if v != healthy[c])
        if lo <= faulty <= hi:
            out.append(list(combo))
    return out


def episode_calls(folder, rules):
    trajectory = load(folder / "trajectory.json", [])
    transcript = [json.loads(l) for l in (load(folder / "transcript.jsonl", "") or "").splitlines() if l.strip()]
    lines = [l for l in transcript if l.get("kind") == "call"]
    tool_rows = {t["name"]: t for t in rules.get("tools", [])}
    state = load(folder / "episode.json", {}) or {}
    think_s, think_c = state.get("think_seconds") or [], state.get("think_cycles") or []
    think_t = state.get("think_tokens") or []  # the token clock alone keeps the tokens it charged, one entry per stepped line
    calls, idle, env_step, at, counted = [], [], 0, 0, 0
    time_before, budget_before, steps_before = 0, rules.get("budget"), rules.get("max_steps")

    def thinking(line):
        # the state keeps the thinking charged before each call that stepped the environment, in the order of the stepped lines
        # of the transcript, so every stepped line takes the next entry, also one the cursor passes without a match
        nonlocal counted
        if not (line and line.get("stepped")):
            return None, None, None
        k, counted = counted, counted + 1
        return tuple(entries[k] if k < len(entries) else None for entries in (think_s, think_c, think_t))

    def set_apart_no_calls(after):
        nonlocal at, env_step
        while at < len(lines) and is_no_call(lines[at]):
            if lines[at].get("stepped"):
                env_step += 1
            seconds, cycles, tokens = thinking(lines[at])
            idle.append({"after": after, "env_step": env_step, "time": time_before, "think_seconds": seconds, "think_cycles": cycles,
                         "think_tokens": tokens})
            at += 1

    for i, c in enumerate(trajectory):
        set_apart_no_calls(i)
        taken = lines[at] if at < len(lines) else None
        at += 1
        line = taken if taken and taken.get("tool") == c["tool"] else None
        stepped = line.get("stepped") if line else None
        if stepped:
            env_step += 1
        seconds, cycles, tokens = thinking(taken)
        if line is None:
            seconds, cycles, tokens = None, None, None
        answer = answer_of(c["server"], c.get("response"))
        # a row without a budget (the base row) leaves budget_left out of the envelope, so time and steps_left mark it
        status = answer if isinstance(answer, dict) and "time" in answer and ("budget_left" in answer or "steps_left" in answer) else None
        call = {
            "n": i + 1, "tool": c["tool"], "server": c["server"], "args": c.get("tool_args") or {},
            "task": c.get("task"), "error": c.get("error"), "duration_ms": c.get("duration_ms"),
            "stepped": stepped, "env_step": env_step if stepped else None,
            "kind": kind_of(c["tool"], tool_rows.get(c["tool"]), stepped),
            "think_seconds": seconds, "think_cycles": cycles, "think_tokens": tokens,
            "tool_class": (tool_rows.get(c["tool"]) or {}).get("class"),
            "answer": answer, "answer_text": c.get("response"),
            "answer_sha256": line.get("answer_sha256") if line else None, "line": line.get("n") if line else None,
            "refused": (answer or {}).get("refused") if isinstance(answer, dict) else None,
            "level": answer_level(c["server"], c["tool"], c.get("tool_args")),
            "time_before": time_before, "budget_before": budget_before, "steps_left_before": steps_before,
        }
        if status:
            call.update(time_after=status["time"], budget_after=status.get("budget_left"),
                        steps_left_after=status.get("steps_left"), open_work_orders=status.get("open_work_orders", []),
                        notifications=status.get("notifications", []), episode_over=status.get("episode_over"))
            time_before, budget_before, steps_before = status["time"], status.get("budget_left"), status.get("steps_left")
        else:
            call.update(time_after=time_before, budget_after=budget_before, steps_left_after=steps_before,
                        open_work_orders=None, notifications=None, episode_over=None)
        result = call["answer"].get("result") if isinstance(call["answer"], dict) else None
        if c["tool"] == PROFILE and isinstance(result, dict) and isinstance(result.get("tools"), dict):
            call["profile_raw"] = raw_of(result)  # a shorter level writes the tool table and the normal ranges as long tables
        call["price"] = (call["budget_before"] - call["budget_after"]) if call["budget_before"] is not None else None
        call["cycles"] = call["time_after"] - call["time_before"]
        calls.append(call)
    set_apart_no_calls(len(trajectory))
    return calls, idle, bool(lines)


def row_unit(rules):
    """The cost unit of the row from its released prices (cost_unit): money where something has a price, calls on the free
    base row; money when the rules carry no price table."""
    return cost_unit(rules["prices"]) if rules.get("prices") else "money"


def clock_of(folder):
    """The clock of the thinking time as episode.json holds it: the seconds charged in all (the sum of think_seconds, as
    episode.clock_report sums them), on the token clock the tokens charged in all (the sum of think_tokens, None on every other
    clock), the credited seconds of the starts and the counted calls, which the wall clock left out, the uncredited ones,
    which it charged at the next call, and the failed model requests."""
    state = load(folder / "episode.json", {}) or {}
    clock = {k: state.get(k) for k in ("credited_seconds", "uncredited_seconds", "failed_model_seconds", "failed_model_attempts")}
    tokens = state.get("think_tokens")
    return clock | {"charged_seconds": sum(state.get("think_seconds") or []), "charged_tokens": None if tokens is None else sum(tokens)}


def served_of(folder):
    """How the episode was served: the cut at the wall limit with its end and probe and the cache flush (harness.json, the
    attempt that stands), and the instant the agent's part ended (episode.json)."""
    standing = (load(folder / "harness.json", {}) or {}).get("standing") or {}
    state = load(folder / "episode.json", {}) or {}
    return {"wall_limit": bool(standing.get("wall_limit")), "cut_end": standing.get("cut_end") or "",
            "cut_probe": standing.get("cut_probe"), "cache_flush": standing.get("cache_flush"),
            "episode_ended_at": state.get("episode_ended_at")}


def advice_of(calls):
    """What the CausalDiagnosis MCP answered, per call, in the fields the views use."""
    rankings, effects, checks = [], [], []
    for c in calls:
        a = c["answer"] if isinstance(c["answer"], dict) else {}
        if c["kind"] != "advice" or c.get("error") or "error" in a:
            continue
        # by the tool, since a shorter level leaves out fields a raw answer holds; a ranking at the level headline lists no
        # candidate and stays a call of its own
        if c["tool"] == mcp_metrics.RANK and isinstance(a.get("candidates"), list):
            top = (a.get("branches") or [{}])[0]
            rankings.append({
                "n": c["n"], "cycle": a.get("belief_time"), "message": a.get("message"),
                "candidates": [{k: v for k, v in x.items() if k != "evidence_ids"} | {"cites": x.get("evidence_ids", [])}
                               for x in a["candidates"]],
                "n_candidates": a.get("n_candidates"), "more_candidates": a.get("more_candidates"),
                "perplexity": a.get("effective_hypotheses"),
                "fault_probability": a.get("current_fault_probability"), "level_probability": a.get("level_probability"),
                "read_first": a.get(mcp_metrics.READ_FIRST), "discriminating_test": a.get("separating_test"),
                "expected_loss_act": a.get("expected_loss_act"), "expected_loss_test": a.get("expected_loss_test"),
                "elimination": a.get("elimination"), "cites": a.get("evidence_ids", []),
                "top_branch": {k: top.get(k) for k in ("faulty_set", "repair_plan", "predicted_outcome",
                                                       "expected_loss_if_acted", "expected_loss_if_tested",
                                                       "cost_to_restored", "confirm_test")},
                "branches": [{"rank": b.get("rank"), "faulty_set": b.get("faulty_set"),
                              "plan_prediction_id": (b.get("repair_plan") or {}).get("plan_prediction_id"),
                              "steps": [s.get("prediction_id") for s in (b.get("repair_plan") or {}).get("steps", [])]}
                             for b in a.get("branches", [])],
            })
        elif c["tool"] == ESTIMATE and "prediction_id" in a:
            effects.append({k: a.get(k) for k in ("prediction_id", "action", "component", "addresses_fault", "recovery",
                                                  "recovery_interval_low", "recovery_interval_high", "price",
                                                  "duration", "predicted", "alpha", "direction", "message",
                                                  "time_window")} | {"n": c["n"], "cites": a.get("evidence_ids", [])})
        elif c["tool"] == mcp_metrics.VERIFY and "status" in a:
            checks.append({k: a.get(k) for k in ("prediction_id", "action", "status", "observation", "observed_t",
                                                 "channels", "consistent", "p_value", "recovery_status",
                                                 "predicted_combination", "predicted_mass", "message")} | {"n": c["n"]})
    return rankings, effects, checks


def follow_marks(calls, rules):
    """Each call's advice in force (the last ranking before it, as the measures read it) and whether it followed that
    advice, by the functions of mcp_metrics.py: a run of work orders follows when its components equal the repair plan of
    the first branch, a paid read when its sensor group is the read the ranking advised while the ranking says to read
    first, a diagnosis when it equals the first candidate. A paid read is one of read_elimination.is_paid under the cost unit of
    the row, the rule the read measure uses. A call no ranking precedes, or one the functions give no verdict, is marked no
    advice."""
    from CausalDiagnosis.evaluation.metrics.realized_read_elimination import is_paid  # loaded here, as mcp_metrics.read_uptake loads it

    comps = rules.get("components") or []
    unit = row_unit(rules)
    orders = mcp_metrics.order_tools()
    word = {True: "followed", False: "deviated", None: "no advice"}
    last, run, visits = None, [], 0

    def close_run():
        nonlocal visits
        if run:
            visits += 1
            mark = word[mcp_metrics.plan_followed({c["args"].get("component") for c in run}, run[0]["_answer"])]
            for c in run:
                c["follow"], c["visit"] = mark, visits
                del c["_answer"]
            run.clear()

    for c in calls:
        a = c["answer"] if isinstance(c["answer"], dict) else {}
        c["advice_in_force"], c["follow"] = (last or {}).get("_n"), None
        # every answer of a ranking without an error, as the measures read the answers of the transcript (mcp_metrics._ranking)
        if c["tool"] == mcp_metrics.RANK and a and "error" not in a and not c.get("error"):
            close_run()
            last = dict(a, _n=c["n"])
            continue
        if c["kind"] == "advice":
            continue
        if c["tool"] in orders and not c.get("refused"):
            c["_answer"] = last
            run.append(c)
            continue
        if c["stepped"]:
            close_run()
        if last is None:
            c["follow"] = "no advice"
        elif c["tool"] == mcp_metrics.SUBMIT and comps and all(k in c["args"] for k in comps):
            def to_levels(combination):
                return tuple(int(combination[k]) for k in comps)
            c["follow"] = word[mcp_metrics.diagnosis_followed(to_levels(c["args"]), last, to_levels)]
        elif "sensor_group" in c["args"] and is_paid({"event": "read", "cost": c["price"]}, unit):
            c["follow"] = word[mcp_metrics.read_followed(c["args"]["sensor_group"], last)]
        else:
            c["follow"] = "no advice"
    close_run()


def run_row(run_dir):
    run = load(run_dir / "run.json", {})
    scores = load(run_dir / "scores.csv", "")
    rows = list(csv.DictReader(io.StringIO(scores))) if scores else []
    played = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and (p / "trajectory.json").exists())
    public_rows = [{k: v for k, v in r.items() if k in PUBLIC_SCORE_COLUMNS} for r in rows]
    keep = ("label", "config_name", "mode", "prices", "agent", "agent_label", "servers", "model", "model_id", "judge",
            "ranking_model", "clock_mode", "guidance_level", "diagnostics_server", "verifier_scope", "briefing",
            "starts", "seeds", "episodes_planned", "training_variant", "started", "ended")
    # the page keys a run by its folder, which the server routes by: a copied or renamed folder keeps the label of its run.json
    return ({k: run.get(k) for k in keep} | {"folder": run_dir.name, "played": played, "rows": public_rows,
                                            "summary": load(run_dir / "summary.md", "")}, rows, run)


def channel_table(rules, asset_id):
    """{field: {id, quantity, unit}} for every sensor field the rules list, so the page can name a sensor in plain words. The
    asset's own channel table gives the id and the quantity when its profile is registered or bundled; otherwise both come
    from the field, read the way the asset profile reads it (<id>_<quantity>_<unit>_<rate>). A field of fewer parts keeps
    only its id."""
    try:
        from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
        profile = get_profile(asset_id, engine=False)
        known = {spec["field"]: (cid, spec["quantity"]) for cid, spec in profile.channels.items()}
    except (KeyError, ValueError):  # an asset that is neither registered nor bundled, a port of the tests among them
        known = {}
    table = {}
    for field in sorted({f for fields in (rules.get("sensor_groups") or {}).values() for f in fields}):
        parts = field.split("_")
        full = len(parts) >= 4
        cid, quantity = known.get(field, (parts[0], parts[1] if full else None))
        table[field] = {"id": cid, "quantity": quantity, "unit": "_".join(parts[2:-1]) if full else None}
    return table


def level_view(released):
    """The view of the asset of the released evidence from which the level code of the CausalDiagnosis MCP writes a shorter level
    (CausalTools.released_view), or None where the asset is not registered on this machine, as a made-up asset of the tests is
    outside their mount."""
    rules = released.get("rules") or {}
    try:
        return profile_view(dict(rules, asset_id=released["asset_id"]) if "asset_id" in released else rules)
    except (KeyError, ValueError):
        return None


def write_levels(out, sub, calls, raws, view, source):
    """Every answer of the CausalDiagnosis MCP among calls at the four levels, one file per call, out/<sub>/<n>.json: the level the
    agent received as the text it received, the raw level, and the other two written from the raw answer by the level code of the
    server. raws gives the raw text of a call by its line of the transcript. Returns {n: source} for every call written, else
    "not recorded" where no raw answer is at hand and "no profile" where the view of the asset is missing."""
    shutil.rmtree(out / sub, ignore_errors=True)
    status = {}
    for c in calls:
        answer = c["answer"] if isinstance(c["answer"], dict) else {}
        if c["server"] != CAUSAL_SERVER or c["level"] is None or c.get("error") or not answer or "error" in answer:
            continue  # an error answer is a result of its own, which the server sends as it is at every level
        raw = c["answer_text"] if c["level"] == "raw" else raws.get(c.get("line"))
        if raw is None or view is None:
            status[c["n"]] = "not recorded" if raw is None else "no profile"
            continue
        full = json.loads(raw)
        texts = {level: c["answer_text"] if level == c["level"] else raw if level == "raw" else text_of(at_level(c["tool"], full, view, level))
                 for level in LEVELS}
        (out / sub).mkdir(parents=True, exist_ok=True)
        (out / sub / f"{c['n']}.json").write_text(json.dumps({"seen": c["level"], "source": source, "texts": texts}, indent=1))
        status[c["n"]] = source
    return status


def recomputed_levels(out, folder, calls, run, start, view):
    """{n: recomputed or not recomputed} for every call whose raw answer the run did not keep, the recomputed ones written at the
    four levels into hidden_levels/ (recompute.py), with the start of the hidden log, whose block the fits of the server leave
    out, in the order of the components."""
    missing = [c for c in calls if c.get("levels") == "not recorded"]
    if not missing or start is None or view is None:
        return {c["n"]: "not recomputed" for c in missing}
    from . import recompute  # the server package of the replay loads only for a run that kept no raw answer

    order = view.components
    levels = [start[k] for k in order] if isinstance(start, dict) else list(start)
    written = write_levels(out, "hidden_levels", missing, recompute.recomputed(folder, calls, run, levels, view), view, "recomputed")
    return {c["n"]: "recomputed" if written.get(c["n"]) == "recomputed" else "not recomputed" for c in missing}


def board_sections(text):
    """The board as its sections, in the order of the text: [{"heading", "blocks"}], every block a table {"table": {"header",
    "rows"}}, a list {"items"} or a line {"text"}. A table is a run of lines between bars, as the board writes it
    (situation_board/text_formatting.table), its second line the rule under the header."""
    sections, blocks = [], None
    for line in text.split("\n"):
        if line.startswith("#"):
            blocks = []
            sections.append({"heading": line.lstrip("#").strip(), "level": len(line) - len(line.lstrip("#")), "blocks": blocks})
        elif blocks is None or not line.strip():
            continue
        elif line.startswith("|"):
            table = blocks[-1].get("table") if blocks else None
            if not set(line.strip()) - set("|-: "):  # the rule under a header; after rows, the row before it heads a table of its own
                if table is not None and table["rows"]:
                    blocks.append({"table": {"header": table["rows"].pop(), "rows": []}})
                continue
            cells = line.strip().removeprefix("| ").removesuffix(" |").split(" | ")
            if table is None:
                blocks.append({"table": {"header": cells, "rows": []}})
            else:
                table["rows"].append(cells)
        elif line.startswith("- "):
            if blocks and "items" in blocks[-1]:
                blocks[-1]["items"].append(line[2:])
            else:
                blocks.append({"items": [line[2:]]})
        else:
            blocks.append({"text": line})
    return sections


def board_of(prompt):
    """The board of a planning request: from its first heading of the first level after the heading of the planning turn to
    the objective that closes it (agents/Causal_Diagnosis_Agent/prompts.planning_turn), or None for any other request."""
    turn = prompt.find(PLANNING_HEADING)
    start = prompt.find("\n# ", turn) + 1 if turn >= 0 else 0
    end = prompt.rfind("\n\n" + OBJECTIVE)
    return prompt[start:end] if turn >= 0 and 0 < start < end else None


def walkthrough_episode(folder):
    """The episode as the walkthrough reads it (walkthrough/episode_files.Episode), or None for a folder it does not read."""
    try:
        return episode_files.Episode(folder)
    except SystemExit:
        return None


def asset_children(ep):
    """{child asset: component} of the profile the agent read, the assets that the work orders of AssetOpsBench's servers name
    (episode_files.Episode.children); empty where the profile names none."""
    return dict(ep.children) if ep is not None else {}


def boards_of(ep, calls, llm):
    """For an episode of the CausalDiagnosis agent: the boards the model read, one per planning request of a plan it wrote,
    and, on every call, the board of the plan that made it (call["board"], an index into the boards), or of the plan whose
    fresh ranking it is. Empty for every other agent and where the walk over the plans cannot place every call."""
    if ep is None or ep.kind != "causal" or not ep.plans or len(ep.calls) != len(calls):
        return []
    requests = call_placement.planning_requests(ep)
    labels, placed = ep.walk
    if requests is None:
        return []
    boards, at_plan = [], {}
    for index, group in requests.items():
        for i in group:
            text = board_of(llm[i].get("prompt") or "")
            if text is not None:
                at_plan[index] = len(boards)
                boards.append({"plan": index, "llm_index": i, "retry": i != group[0], "sections": board_sections(text)})
    for c, label in zip(calls, labels if placed else []):
        if label != controller_walk.FORCED_FINISH and label[0] in at_plan:
            c["board"] = at_plan[label[0]]
            c["board_before_plan"] = label[1] == 0
    return boards


def build(episode, out, hidden=False):
    """Write the data of one episode folder into out (section text of the module); return the number of calls."""
    folder = Path(episode).resolve()
    run_dir = folder.parent
    out = Path(out).resolve()
    if out.is_relative_to(run_dir):
        raise SystemExit(f"{out} lies inside the run folder, whose files the run ledger hashes")
    released = load(folder / "released.json", {})
    rules = released.get("rules", {})
    calls, idle_steps, transcribed = episode_calls(folder, rules)
    rankings, effects, checks = advice_of(calls)
    follow_marks(calls, rules)
    llm = load(folder / "llm_calls.json", []) or []
    model_messages(calls, llm)
    walked = walkthrough_episode(folder)
    boards = boards_of(walked, calls, llm)
    out.mkdir(parents=True, exist_ok=True)
    public_files, hidden_files = copy_files(folder, run_dir, out, hidden)
    view = level_view(released)
    for n, source in write_levels(out, "levels", calls, transcript.raw_answers(folder), view, "recorded").items():
        calls[n - 1]["levels"] = source
    shutil.rmtree(out / "hidden_levels", ignore_errors=True)
    run_public, score_rows, run = run_row(run_dir)
    # the tool answers stay whole for the Raw tab; the long assumptions lists are kept, nothing is cut
    data = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs": [run_public] + [run_row(r)[0] for r in sorted(run_dir.parent.iterdir())
                                if r != run_dir and r.is_dir() and not r.name.startswith(".") and (r / "run.json").exists()],
        "start_states": start_states(rules, (run.get("config") or {}).get("fault_rule", {})),
        "episode": {
            "id": folder.name, "run": run_dir.name, "seed": (load(folder / "episode.json", {}) or {}).get("seed"),
            "task_description": load(folder / "question.txt", None) or released.get("task_text"),
            "answer": load(folder / "answer.txt", None),
            "plan": plan_lines(load(folder / "plan.txt", "")),
            "transcribed": transcribed,
            "rules": rules, "channels": channel_table(rules, run.get("asset_id")), "status": released.get("status"),
            # the combinations the CausalDiagnosis MCP ranks, every level combination (Causal_Diagnosis_MCP/asset_view.py), which a
            # ranking at a shorter level does not repeat in n_candidates
            "n_combinations": math.prod(len(levels) for levels in (rules.get("levels") or {}).values()) if rules.get("levels") else None,
            "events": released.get("events", []), "observations": released.get("observations", []),
            "predictions": [{k: v for k, v in p.items() if k != "after"} for p in load(folder / "predictions.json", [])],
            "calls": calls, "rankings": rankings, "effects": effects, "checks": checks, "boards": boards,
            "asset_children": asset_children(walked),
            "idle_steps": idle_steps, "clock": clock_of(folder), "served": served_of(folder), "cost_unit": row_unit(rules),
            "files": public_files,
        },
    }
    (out / "data.json").write_text(json.dumps(data, indent=1))
    if not hidden:
        return len(calls)
    log = load(folder / "hidden" / "log.json", None)
    verdict = load(folder / "hidden" / "verdict.json", None)
    score = load(folder / "hidden" / "score.json", None)
    oracle = load(folder / "hidden" / "oracle.json", None)
    ledger = load(run_dir / "ledger.json", {})
    reset = next((e for e in log or [] if e["event"] == "reset"), {})
    end = next((e for e in log or [] if e["event"] == "end"), {})
    hidden_side = {
        "episode": folder.name,
        "start_state": reset.get("hidden"), "entry_offset": reset.get("entry_offset"),
        "final": end.get("final"), "end": {k: end.get(k) for k in ("reason", "time", "cost", "ledger",
                                                                     "downtime_total", "downtime_by_cycle", "played")},
        "log": log, "verdict": verdict, "score": score, "oracle": oracle,
        "ledger_blocks": {k: v.get("block") for k, v in (ledger.get("episodes") or {}).items()},
        "score_rows": [{k: v for k, v in r.items() if k not in PUBLIC_SCORE_COLUMNS} | {"seed": r.get("seed")}
                       for r in score_rows],
        "files": hidden_files,
        "levels": recomputed_levels(out, folder, calls, run, reset.get("hidden"), view),
    }
    (out / "hidden.json").write_text(json.dumps(hidden_side, indent=1))
    return len(calls)
