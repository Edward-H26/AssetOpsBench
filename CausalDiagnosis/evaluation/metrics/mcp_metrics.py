"""The measures of the CausalDiagnosis MCP in played episodes (CausalDiagnosis/docs/metrics.md): the uptake of its advice and
the harm of the advice an agent followed, the calibration of the recovery rate prediction at the reading that verifies a plan,
the error of the cost prediction of every plan acted on beside a baseline that predicts the plan's price and downtime alone,
the coverage of the predicted intervals and the error of the predicted readings beside a baseline that holds the current
readings, the verdict of the post repair check recomputed offline with its false alarms and misses, the production lost and
the failures in the cycles the clock charged to thinking, and the ranking metrics at the hidden combination of the moment of
each answer.

Every measure reads files that the servers and the verifier write: the transcript of the episode (servers/transcript.py), the
hidden log, released.json, predictions.json and the verdict. None reads trajectory.json or any other record of a runner, so a
measure does not depend on what the measured agent's runner chose to record. The calls of an episode are the call lines of the
transcript in the order the servers answered them. A call took a step of the environment exactly when its line says so, and
the k-th such call belongs to the one entry of the hidden log with step k, so what a call did is read from its log entry and
never from a reply text. An episode whose transcript and log do not agree leaves the measures that need the match empty and
says so in `unaligned`. The verdicts of the post repair check replay the MCP's own verify_outcome over the episode's released
evidence and stored predictions, with the fits that leave out the block of the start, as the MCP of the episode served them.
"""
import json
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER
from CausalDiagnosis.utils.csv_rows import as_bool
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig, cost_unit
# the verifier's own rules for the asset of an episode, the combination after a part of the log, the configuration of the
# reset entry and the cycle of an entry, so every measure reads the truth the verdict reads
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import _asset_truth as asset_truth
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import _combination_after as combination_after
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import _reset_config as reset_config
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import _time as time_of
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.utils.public_rules.visits import keeps_run
from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.evaluation.metrics.table_statistics import cluster_bootstrap, ratio, rounded

RANK, VERIFY, SUBMIT = "rank_root_causes", "verify_outcome", "submit_diagnosis"
ORDER_ACTION = "create_work_order"
READ_FIRST = "read_first"  # the key under which a ranking tells the agent to read before it acts
# the kind of decision that each log entry of a step stands for; a refused step is no decision
DECISION_OF_EVENT = {"work_order_created": "order", "read": "read", SUBMIT: "diagnosis"}

Call = namedtuple("Call", "n server tool index answer")  # a call line of the transcript, the log index of its step, its ranking
Decision = namedtuple("Decision", "kind n index answer")  # a call that ordered, read or diagnosed, with the last ranking before it
Visit = namedtuple("Visit", "t components predictions first answer orders")  # a run of orders, the log index of its first order


def cost_unit_of(log, released):
    """The cost unit of the row of an episode by the rule of the environment (config.cost_unit), over the prices of the
    configuration of the reset entry that the verifier reads, else over the prices of the released rules, and money when
    neither holds prices, as in a log written before the free base row."""
    prices = reset_config(log).get("prices") or ((released or {}).get("rules") or {}).get("prices")
    return cost_unit(prices) if prices else "money"


def _json(text):
    try:
        return json.loads(text or "")
    except (TypeError, ValueError):
        return None


def _ranking(line):
    """The parsed answer of a ranking that the diagnostics server recorded in its line, None for every other line, for a
    failed call and for an answer that holds an error."""
    if line.get("server") != CAUSAL_SERVER or line.get("tool") != RANK or line.get("is_error"):
        return None
    value = _json(line.get("answer"))
    return value if isinstance(value, dict) and "error" not in value else None


def calls(ep_dir, log):
    """(the calls of an episode in transcript order, whether its transcript and its hidden log agree). They agree when the
    calls that took a step are as many as the steps of the log and every step 1 to m has exactly one entry, so a folder with
    steps in its log and no transcript does not agree. Only an agreeing episode gives a call the log index of its step."""
    lines = [line for line in transcript.read(ep_dir) if line.get("kind") == "call"]
    entries = {}
    for i, e in enumerate(log):
        if e.get("step") is not None:
            entries.setdefault(e["step"], []).append(i)
    steps = sum(len(found) for found in entries.values())
    aligned = (sum(1 for line in lines if line.get("stepped")) == steps
               and all(len(entries.get(k, ())) == 1 for k in range(1, steps + 1)))
    out, k = [], 0
    for line in lines:
        index = None
        if line.get("stepped"):
            k += 1
            index = entries[k][0] if aligned else None
        out.append(Call(line["n"], line.get("server"), line.get("tool"), index, _ranking(line)))
    return out, aligned


@lru_cache(maxsize=None)
def order_tools():
    """The tools that create a work order on either tool surface: the tool of the rig's contract that is the engine action
    itself, and every tool that the contract of a declared tool surface binds to that action."""
    rig = {name for name in contract.load_contract()["tools"] if name == ORDER_ACTION}
    declared = {name for surface in frameworks.surfaces() for name in frameworks.code("SURFACES", surface).tools_of_action(ORDER_ACTION)}
    return frozenset(rig | declared)


def read_first(answer):
    """Whether an answer of rank_root_causes tells the agent to read before it acts."""
    return bool((answer or {}).get(READ_FIRST))


def decisions(call_list, log):
    """The calls whose log entry is an order, a reading or the diagnosis, in transcript order, each with the last answer of
    rank_root_causes that the diagnostics server recorded before it, or None before the first."""
    last, out = None, []
    for call in call_list:
        if call.answer is not None:
            last = call.answer
            continue
        if call.index is None:
            continue
        kind = DECISION_OF_EVENT.get(log[call.index].get("event"))
        if kind:
            out.append(Decision(kind, call.n, call.index, last))
    return out


def visits(made, call_list, log):
    """The visits of an episode: a run of orders that the agent creates one after another, with no call between them that
    took a step of the environment and whose tool creates no order, so an order the engine refused lies between two orders
    without splitting them, a call that took no step never splits a run, and a no_call step keeps the run open as the
    environment books it (visits.keeps_run). Each visit holds its components, the prediction
    identifiers its orders cite, the log index of its first order, the last ranking before that order and the identifiers of
    its orders. Runs and not the visits that the environment books, since on a row whose orders close in their own call every
    order of a plan books a visit of its own."""
    tools = order_tools()
    breaks = [call.n for call in call_list if call.index is not None and call.tool not in tools and not keeps_run(call.tool)]
    runs = []
    for d in (d for d in made if d.kind == "order"):
        if runs and not any(runs[-1][-1].n < b < d.n for b in breaks):
            runs[-1].append(d)
        else:
            runs.append([d])
    out = []
    for members in runs:
        entries = [log[d.index] for d in members]
        out.append(Visit(time_of(entries[0]), frozenset(e["component"] for e in entries),
                         tuple(dict.fromkeys(e["prediction_id"] for e in entries if e.get("prediction_id"))), members[0].index,
                         members[0].answer, tuple(e.get("id") for e in entries)))
    return out


def plan_components(branch):
    return frozenset(step["component"] for step in (branch.get("repair_plan") or {}).get("steps") or [])


def advice_measures(visit_list, log, asset):
    """The uptake of the advice over the visits of an episode and its harm: a visit follows the first plan when its components
    equal the repair plan of the first branch of the last answer before it, and any plan when they equal the repair plan of
    some branch; the plan it follows is wrong when its components differ from the faulty set of the hidden combination before
    the visit's first order, the oracle's repair set, and a deviation is right when the visit's components equal that set."""
    hidden = asset.levels(log[0]["hidden"])
    out = {"visits": len(visit_list), "advised": 0, "follow_first": 0, "follow_any": 0, "harm": 0, "deviate": 0, "good_deviation": 0,
           "first_visit_follows": None}
    for v in visit_list:
        branches = (v.answer or {}).get("branches") or []
        if not branches:
            continue
        out["advised"] += 1
        needed = frozenset(asset.faulty(combination_after(log[:v.first], hidden, asset)))
        first = v.components == plan_components(branches[0])
        out["follow_first"] += first
        out["follow_any"] += any(v.components == plan_components(b) for b in branches)
        out["harm"] += first and v.components != needed
        out["deviate"] += not first
        out["good_deviation"] += (not first) and v.components == needed
        if out["first_visit_follows"] is None:
            out["first_visit_follows"] = first
    return out


def read_uptake(reads, log, unit):
    """The paid reads taken while the last answer told the agent to read first, and those of them whose sensor group is the
    separating test or the head of the read batch of that answer. A paid read is one of read_elimination.is_paid, every read
    on a row that prices nothing and a read with a charged price on the others, so a free read of cycles paid before is no
    decision to read."""
    from CausalDiagnosis.evaluation.metrics.realized_read_elimination import is_paid

    flagged = taken = 0
    for d in reads:
        answer, entry = d.answer or {}, log[d.index]
        if not read_first(answer) or not is_paid(entry, unit):
            continue
        advised = {(answer.get("separating_test") or {}).get("sensor_group")} | {((answer.get("read_batch") or [{}])[0] or {}).get("sensor_group")}
        flagged += 1
        taken += entry.get("sensor_group") in advised - {None}
    return {"reads_read_first": flagged, "reads_advised": taken}


def diagnosis_uptake(writes, log, asset):
    """Whether the diagnosis an agent wrote is the first candidate of the last answer before it, None without either."""
    if len(writes) != 1:
        return None
    d = writes[0]
    candidates = (d.answer or {}).get("candidates") or []
    entry = log[d.index]
    if not candidates or entry.get("diagnosis") is None:
        return None
    return asset.levels(entry["diagnosis"]) == asset.levels(candidates[0]["combination"])


def combination_at(log, t, asset):
    """The hidden combination the rig played in cycle t: the combination the end of the log records for that cycle, else the
    combination after every log entry of a cycle no later than t."""
    end = log[-1] if log and log[-1].get("event") == "end" else {}
    played = {p["t"]: p["combination"] for p in end.get("played") or [] if p.get("combination") is not None}
    if t in played:
        return asset.levels(played[t])
    return combination_after([e for e in log if (time_of(e) or 0) <= t], asset.levels(log[0]["hidden"]), asset)


def _row_rules(log):
    """(whether the row has a clock, whether an order closes inside its own call) from the configuration of the reset entry, a
    clock when the entry holds none or one that no configuration accepts, as in the synthetic logs of the tests."""
    try:
        config = EpisodeConfig.from_json(log[0].get("config") or {})
    except (TypeError, ValueError):
        return True, False
    return config.has_clock(), not config.has_clock() and not config.repair_card_on()


def answer_truths(call_list, log, asset, aligned=True):
    """{line number of an answer of rank_root_causes: the hidden combination at the moment of the answer, or None where the log
    cannot place it}. On a row with a clock the combination holds still between the action that brings the clock to the cycle
    of an answer and the action that moves it on, so the truth follows every log entry up to the last stepped entry that ended
    at or before the `belief_time` of the answer. On a row without a clock the answer follows the entry of the last call
    before it that took a step, and where an order closes inside its own call, the closes logged before the next stepped entry.
    An episode whose transcript and log do not agree places no answer."""
    hidden = asset.levels(log[0]["hidden"])
    clocked, in_call = _row_rules(log)
    stepped = [i for i, e in enumerate(log) if e.get("step") is not None]
    out, last = {}, None
    for call in call_list:
        if call.index is not None:
            last = call.index
        if call.answer is None:
            continue
        if not aligned:
            out[call.n] = None
            continue
        if clocked and call.answer.get("belief_time") is not None:
            cut = max((j for j in stepped if (time_of(log[j]) or 0) <= int(call.answer["belief_time"])), default=0)
        else:
            cut = 0 if last is None else last
            if in_call and last is not None:
                cut = next((j for j in stepped if j > cut), len(log)) - 1
        out[call.n] = combination_after(log[:cut + 1], hidden, asset)
    return out


def ranked_answers(ep_dir):
    """For the calibration of the ranking in played episodes (table A12 of tables_from_runs/): per answer of rank_root_causes in
    transcript order, (the accepted readings before it, its first candidate in component order, that candidate's probability,
    the hidden combination at the moment of the answer in component order or None where the log cannot place it). A reading
    is a call whose log entry is a read, so a refused read and a call that never reached the engine count nothing, for every
    runner alike. [] for a folder without a hidden log or without its hidden start."""
    ep_dir = Path(ep_dir)
    log = read_json(ep_dir / "hidden" / "log.json")
    if not log or "hidden" not in log[0]:
        return []
    asset = asset_truth(log)
    call_list, aligned = calls(ep_dir, log)
    truths = answer_truths(call_list, log, asset, aligned)
    out, readings = [], 0
    for call in call_list:
        if call.index is not None and log[call.index].get("event") == "read":
            readings += 1
        candidates = (call.answer or {}).get("candidates") or []
        if not candidates:
            continue
        top, truth = asset.levels(candidates[0]["combination"]), truths.get(call.n)
        out.append((readings, [top[c] for c in asset.components], float(candidates[0]["probability"]),
                    None if truth is None else [truth[c] for c in asset.components]))
    return out


def ranking_measures(answers, truths, asset):
    """For every answer of rank_root_causes that the log places, the metrics on the ranking at the hidden combination of the
    moment of the answer: AC@1 and AC@k of the faulty set, the reciprocal rank of the first candidate with the true faulty
    set, zero when no candidate has it, and the precision and recall of the first candidate's faulty set, where true
    negatives earn nothing, so a healthy first candidate has no precision and an answer whose truth is healthy has no recall."""
    out = []
    for n, answer in answers:
        candidates = answer.get("candidates") or []
        if not candidates or truths.get(n) is None:
            continue
        truth = frozenset(asset.faulty(truths[n]))
        faulty = [frozenset(asset.faulty(asset.levels(c["combination"]))) for c in candidates]
        rank = next((k + 1 for k, f in enumerate(faulty) if f == truth), None)
        top = faulty[0]
        out.append({"ac1": float(top == truth), "ack": float(rank is not None), "rr": 0.0 if rank is None else 1.0 / rank,
                    "precision": len(top & truth) / len(top) if top else None, "recall": len(top & truth) / len(truth) if truth else None})
    return out


def thinking_loss(log):
    """The production lost that the ledger charged in the cycles the clock spent on thinking, its share of the episode's
    production lost, and the failures in those cycles. The thinking cycles of an action are the cycles the clock advanced
    before it, those just before the cycle its entry starts at (env.step charges them first), and never before the end of the
    action before it, since at the horizon an entry records the cycles asked for and not those the clock ran."""
    end = next((e for e in reversed(log) if e.get("event") == "end"), {})
    charges = {d["t"]: d["charge"] for d in end.get("downtime_by_cycle") or []}
    cycles, last_end = set(), 0
    for e in log:
        if e.get("step") is None:
            continue
        k = int(e.get("think_cycles") or 0)
        if k > 0 and e.get("t_start") is not None:
            cycles.update(range(max(e["t_start"] - k, last_end), e["t_start"]))
        last_end = time_of(e) or last_end
    lost = float(sum(charges.get(t, 0) for t in cycles))
    total = float(end.get("downtime_total") or 0)
    return {"thinking_cycles": len(cycles), "think_loss": lost, "downtime_total": total, "think_loss_share": lost / total if total else None,
            "think_failures": sum(1 for e in log if e.get("event") == "failure" and time_of(e) in cycles)}


def verdicts(ep_dir, left_out, ranking_model, prediction_ids):
    """({prediction identifier: the answer of verify_outcome}, the digest of models.json), replayed offline over the episode's
    released evidence and stored predictions by the MCP over the fits that leave out the block of the start, which
    read_elimination's yardstick loads, ({}, "") without the models file."""
    from CausalDiagnosis.evaluation.metrics import realized_read_elimination as read_elimination  # the server package loads only when a table reads the verdicts
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools

    try:
        stick = read_elimination.yardstick(left_out=tuple(left_out))
    except FileNotFoundError:
        return {}, ""
    tools = CausalTools(stick.models, released_path=Path(ep_dir) / "released.json", ranking_model=ranking_model)
    out = {}
    for pid in prediction_ids:
        result = tools.verify_outcome(pid)
        out[pid] = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return out, stick.digest


def _close_of(log, start, component):
    """The log index of the first close by the agent of an order on the component after the log index start, or None."""
    return next((i for i in range(start, len(log)) if log[i].get("event") == "work_order_closed" and log[i].get("component") == component
                 and log[i].get("source") == "agent"), None)


def _branch_acted(answer, plan_id, diagnosis, asset):
    """The branch of an answer whose repair plan carries the plan identifier the orders cite, the one whose combination is the
    diagnosis the agent wrote when several share the plan, else the first of them."""
    branches = [b for b in (answer or {}).get("branches") or [] if (b.get("repair_plan") or {}).get("plan_prediction_id") == plan_id]
    if diagnosis is not None:
        named = [b for b in branches if asset.levels(b["combination"]) == diagnosis]
        branches = named or branches
    return branches[0] if branches else None


def restored_at(log, visit, asset):
    """The log index that closes the cost window of a visit's plan: the first entry from the visit's first order on after
    which every component is at its healthy level and no order of the visit is still open, None when no entry of the log
    restores the asset so, and the window runs to the end of the episode. The combination is followed as the verifier
    follows it, so a drop, a close and the expiry of an emergency repair move it, and an emergency repair that holds a
    component at its healthy level counts as a restore while the hold lasts. The second condition keeps the plan's own later
    repairs inside the window when the plan orders a component beside the faulty one, or orders on a healthy asset."""
    healthy = dict(asset.healthy)
    current = combination_after(log[:visit.first], asset.levels(log[0]["hidden"]), asset)
    still_open = set(visit.orders)
    for i in range(visit.first, len(log)):
        current = combination_after(log[i:i + 1], current, asset)
        if log[i].get("event") == "work_order_closed":
            still_open.discard(log[i].get("id"))
        if not still_open and current == healthy:
            return i
    return None


def cost_to_restore(log, visit, asset):
    """In money, what an episode cost from the first order of a visit to the entry that closes its window (restored_at): the
    price or penalty of every log entry in the window, the end entry aside, and the production lost in every cycle from the
    cycle of the first order up to, and not including, the cycle of the entry that closes the window, since the clock
    charges a cycle before it moves on and a close is logged at the cycle from which the repaired level holds. A window that
    runs to the end of the episode takes every later entry and cycle."""
    close = restored_at(log, visit, asset)
    window = log[visit.first:] if close is None else log[visit.first:close + 1]
    until = None if close is None else time_of(log[close])
    end = next((e for e in reversed(log) if e.get("event") == "end"), {})
    lost = sum(d["charge"] for d in end.get("downtime_by_cycle") or [] if d["t"] >= visit.t and (until is None or d["t"] < until))
    return float(sum(e.get("cost") or 0 for e in window if e.get("event") != "end") + lost)


def prediction_measures(visit_list, log, released, predictions, verified, verdict, asset):
    """Per plan an agent acted on, counted once at the first visit that cites it, the recovery rate prediction against the
    recovery at its verifying reading, the cost prediction and the price-only baseline against the cost to the restore of the
    asset, the coverage of the predicted intervals and the error of the predicted readings beside a baseline that holds the
    current readings, and the verdict of the post repair check against the truth at the reading. The verifying reading is the
    observation of the replayed answer of verify_outcome, else the cycle of the last close by the agent on a component of the
    plan. verified maps a prediction identifier to that answer. The cost prediction is measured on a row whose rules hold a
    price, so a row whose cost unit is the call gets no cost error and no price error."""
    by_id = {p["id"]: p for p in predictions or [] if isinstance(p, dict) and "id" in p}
    observations = {o["n"]: o for o in released.get("observations") or []}
    priced = cost_unit_of(log, released) != "calls"
    prices = (released.get("rules") or {}).get("prices") or {}
    misdiagnosis = float(prices.get("misdiagnosis") or 0) if priced else 0.0
    written = next((e for e in log if e.get("event") == SUBMIT), None)
    diagnosis = asset.levels(written["diagnosis"]) if written and written.get("diagnosis") is not None else None
    wrong = written is not None and verdict.get("diagnosis_all_correct") is False
    oracle_cost = verdict.get("cost_oracle")
    healthy = dict(asset.healthy)
    plans, seen = [], set()
    for v in visit_list:
        for pid in v.predictions:
            record = by_id.get(pid)
            if record is None or pid in seen:
                continue
            seen.add(pid)
            components = record.get("components") or ([record["component"]] if record.get("component") else [])
            closes = [_close_of(log, v.first, c) for c in components]
            row = {"id": pid, "components": list(components), "recovery": record.get("recovery"), "target": None}
            answer = verified.get(pid) or {}
            observed = answer.get("observation") if answer.get("status") == "verified" else None
            if observed is not None:
                at = combination_at(log, int(answer["observed_t"]), asset)
            elif closes and None not in closes:
                at = combination_at(log, max(time_of(log[i]) for i in closes), asset)
            else:
                at = None
            if at is not None:
                row["target"] = float(at == healthy)
            if observed is not None:
                predicted = asset.levels(answer["predicted_combination"])
                row.update(verdict_consistent=bool(answer.get("consistent")), predicted_right=predicted == at)
                values = (observations.get(observed) or {}).get("values") or {}
                if predicted == at:
                    inside = [low <= values[f] <= high for f, (_m, low, high, _q) in (record.get("predicted") or {}).items() if f in values]
                    row.update(covered=sum(inside), summaries=len(inside))
                held = {}
                for n in sorted(n for n in observations if n <= int(record.get("observations") or 0)):
                    held.update(observations[n].get("values") or {})
                errors = [(abs(values[f] - mean) / ((high - mean) / q), abs(values[f] - held[f]) / ((high - mean) / q))
                          for f, (mean, _low, high, q) in (record.get("predicted") or {}).items()
                          if f in values and f in held and q and high > mean]
                row.update(error_mcp=sum(e for e, _ in errors), error_hold=sum(h for _, h in errors), errors=len(errors))
            branch = _branch_acted(v.answer, pid, diagnosis, asset)
            if priced and branch is not None and oracle_cost:
                realized = cost_to_restore(log, v, asset) + misdiagnosis * wrong
                plan = branch.get("repair_plan") or {}
                row.update(cost_error=(float(branch["cost_to_restored"]["act_now"]) - realized) / float(oracle_cost),
                           price_error=(float(plan.get("price") or 0) + float(plan.get("downtime_cost") or 0) - realized) / float(oracle_cost))
            plans.append(row)
    return plans


def start_of(cell):
    """The start combination of a block cell of scores.csv, a JSON list, or None for a cell that holds none."""
    try:
        start = json.loads(cell or "")
    except (TypeError, ValueError):
        return None
    return start if isinstance(start, list) else None


def episode_measures(ep_dir, left_out, ranking_model, verdict, with_verdicts=True):
    """Every measure of this module for one episode folder, {} for a folder without a hidden log, and the thinking measures alone
    for a log without its hidden start. left_out is the start combination whose block the fits of the episode's MCP leave out,
    None when unknown, which skips the replay of the verdicts."""
    ep_dir = Path(ep_dir)
    log = read_json(ep_dir / "hidden" / "log.json")
    if not log:
        return {}
    out = thinking_loss(log)
    if "hidden" not in log[0]:
        return out
    asset = asset_truth(log)
    call_list, aligned = calls(ep_dir, log)
    out["unaligned"] = not aligned
    released = read_json(ep_dir / "released.json", {})
    answers = [(call.n, call.answer) for call in call_list if call.answer is not None]
    truths = answer_truths(call_list, log, asset, aligned)
    out["ranking"] = ranking_measures(answers, truths, asset)
    out["unplaced"] = sum(1 for n, _ in answers if truths.get(n) is None)
    if not aligned:
        return out
    made = decisions(call_list, log)
    visit_list = visits(made, call_list, log)
    out.update(advice_measures(visit_list, log, asset))
    predictions = read_json(ep_dir / "predictions.json", [])
    cited = list(dict.fromkeys(p for v in visit_list for p in v.predictions))
    verified, digest = verdicts(ep_dir, left_out, ranking_model, cited) if with_verdicts and left_out and cited and predictions else ({}, "")
    out["yardstick"] = digest
    out["plans"] = prediction_measures(visit_list, log, released, predictions, verified, verdict, asset)
    out.update(read_uptake([d for d in made if d.kind == "read"], log, cost_unit_of(log, released)))
    out["diagnosis_follows"] = diagnosis_uptake([d for d in made if d.kind == "diagnosis"], log, asset)
    return out


# ----- the cells of a row of the table MCP of tables_from_runs/ -----

def _brier_skill(sums):
    """1 minus the Brier score over the Brier score of the base rate, from the sums of (r - y) squared, of y and of the count of
    every row of sums, no value where the base rate is 0 or 1."""
    sums = np.asarray(sums, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = sums[:, 1] / sums[:, 2]
        spread = base * (1 - base)
        return np.where((sums[:, 2] > 0) & (spread > 0), 1 - (sums[:, 0] / sums[:, 2]) / spread, np.nan)


def _skill(sums):
    """1 minus the summed absolute error over the summed absolute error of the baseline, of every row of sums, no value where the
    baseline has none."""
    sums = np.asarray(sums, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sums[:, 1] != 0, 1 - sums[:, 0] / sums[:, 1], np.nan)


# the statistics over the items of a row, each as the summed columns of an item and a statistic of the sums, which the
# bootstrap of stats.cluster_bootstrap evaluates for every resample of the starts at once
RATIO = (lambda item: (item[0], item[1]), ratio)  # a share or a ratio of two totals
MEAN = (lambda item: (item[0], 1.0), ratio)
ABS_MEAN = (lambda item: (abs(item[0]), 1.0), ratio)
BRIER_SKILL = (lambda item: ((item[0] - item[1]) ** 2, item[1], 1.0), _brier_skill)
SKILL = (lambda item: (abs(item[0]), abs(item[1])), _skill)

STATISTICS = (  # (cell, the per episode items of the row, the statistic over the items of the resampled episodes)
    ("uptake", lambda m: [(m["follow_first"], m["advised"])] if "advised" in m else [], RATIO),
    ("uptake_any", lambda m: [(m["follow_any"], m["advised"])] if "advised" in m else [], RATIO),
    ("harm", lambda m: [(m["harm"], m["follow_first"])] if "advised" in m else [], RATIO),
    ("good_deviation", lambda m: [(m["good_deviation"], m["deviate"])] if "advised" in m else [], RATIO),
    ("pass_follow", lambda m: [(m["pass"], 1)] if m.get("first_visit_follows") is True else [], RATIO),
    ("pass_deviate", lambda m: [(m["pass"], 1)] if m.get("first_visit_follows") is False else [], RATIO),
    ("read_uptake", lambda m: [(m["reads_advised"], m["reads_read_first"])] if "reads_read_first" in m else [], RATIO),
    ("diagnosis_uptake", lambda m: [(float(m["diagnosis_follows"]), 1)] if m.get("diagnosis_follows") is not None else [], RATIO),
    ("recovery_bss", lambda m: [(p["recovery"], p["target"]) for p in m.get("plans", []) if p["target"] is not None and p["recovery"] is not None],
     BRIER_SKILL),
    ("recovery_bias", lambda m: [(p["recovery"] - p["target"],) for p in m.get("plans", []) if p["target"] is not None and p["recovery"] is not None],
     MEAN),
    ("cost_bias", lambda m: [(p["cost_error"],) for p in m.get("plans", []) if "cost_error" in p], MEAN),
    ("cost_mae", lambda m: [(p["cost_error"],) for p in m.get("plans", []) if "cost_error" in p], ABS_MEAN),
    ("price_mae", lambda m: [(p["price_error"],) for p in m.get("plans", []) if "price_error" in p], ABS_MEAN),
    ("cost_skill", lambda m: [(p["cost_error"], p["price_error"]) for p in m.get("plans", []) if "cost_error" in p], SKILL),
    ("coverage", lambda m: [(p["covered"], p["summaries"]) for p in m.get("plans", []) if "summaries" in p], RATIO),
    # the share of the error of holding the current readings that the predicted means remove, a ratio of summed errors whose
    # count is the plans and not its denominator
    ("readings_skill", lambda m: [(p["error_hold"] - p["error_mcp"], p["error_hold"]) for p in m.get("plans", []) if p.get("errors")], RATIO),
    ("false_alarm", lambda m: [(float(not p["verdict_consistent"]), 1) for p in m.get("plans", []) if p.get("predicted_right") is True], RATIO),
    ("miss", lambda m: [(float(p["verdict_consistent"]), 1) for p in m.get("plans", []) if p.get("predicted_right") is False], RATIO),
    ("ac1", lambda m: [(a["ac1"],) for a in m.get("ranking", [])], MEAN),
    ("ack", lambda m: [(a["ack"],) for a in m.get("ranking", [])], MEAN),
    ("mrr", lambda m: [(a["rr"],) for a in m.get("ranking", [])], MEAN),
    ("precision", lambda m: [(a["precision"],) for a in m.get("ranking", []) if a["precision"] is not None], MEAN),
    ("recall", lambda m: [(a["recall"],) for a in m.get("ranking", []) if a["recall"] is not None], MEAN),
    ("think_loss_share", lambda m: [(m["think_loss"], m["downtime_total"])] if m.get("downtime_total") else [], RATIO),
    ("think_failures", lambda m: [(m["think_failures"],)] if "think_failures" in m else [], MEAN),
)
CELLS = tuple(name for name, _, _ in STATISTICS)
# the counted cells whose count is the summed denominator of their items; every other cell counts its items
DENOMINATOR_COUNTED = frozenset(name for name, _, kind in STATISTICS if kind is RATIO and name != "readings_skill")


def row_cells(episodes, resamples, seed, cluster_of=None, min_clusters=None):
    """The cells of one row: every statistic of STATISTICS over the episodes with its 95% interval and its standard error from a
    bootstrap that resamples the clusters of the episodes, cluster_of(episode) as every table of tables_from_runs/ resamples them, the block
    of the start when it is None, and the count behind it, the summed denominator of a share, the visits, the reads or the plans it
    counts, and the items of any other statistic, the plans, the answers or the episodes. Below min_clusters clusters a cell prints its
    estimate alone, without interval and standard error, the registered intervals.min_clusters_rule."""
    unit = cluster_of or (lambda e: e.get("block", ""))
    out = {}
    for name, items_of, (columns_of, statistic) in STATISTICS:
        lists = [items_of({**e.get("mcp", {}), "pass": float(as_bool(e.get("pass")) is True)}) for e in episodes]
        items = [(item, unit(e)) for found, e in zip(lists, episodes) for item in found]
        boot = cluster_bootstrap([columns_of(item) for item, _ in items], [cluster for _, cluster in items], statistic, resamples, seed) if items else {}
        if boot and min_clusters is not None and boot["clusters"] < min_clusters:
            boot = {**boot, "low": None, "high": None, "se": None}
        pooled = [item for item, _ in items]
        out.update({name: rounded(boot.get("estimate")), f"{name}_low": rounded(boot.get("low")), f"{name}_high": rounded(boot.get("high")),
                    f"{name}_se": rounded(boot.get("se")),
                    f"{name}_n": sum(d for _, d in pooled) if name in DENOMINATOR_COUNTED else len(pooled)})
    out["unaligned"] = sum(1 for e in episodes if e.get("mcp", {}).get("unaligned"))
    out["unplaced"] = sum(e.get("mcp", {}).get("unplaced") or 0 for e in episodes)
    return out

