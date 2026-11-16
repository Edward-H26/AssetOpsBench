from itertools import combinations
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import STATE_FILE  # noqa: E402
from CausalDiagnosis.utils.csv_rows import as_bool, as_float  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402  the verifier's reading of the log, which the time to repair in declared cycles shares
from CausalDiagnosis.evaluation.metrics import judge_records  # noqa: E402  the one rule of a parsed review
from CausalDiagnosis.evaluation.metrics import mcp_metrics  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import OURS  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import episode_files, cell_statistics


def scope_pass(episode):
    """The pass under the verifier scope of an episode: the verified pass under the scope writes, and under the scope all the
    pass with the read gate, since the scope leaves the outcome as it is and A7 would otherwise compare one column with itself.
    None for an episode the verifier did not score."""
    passed = as_bool(episode.get("pass"))
    if passed is None or episode.get("verifier_scope") != "all":
        return passed
    return passed and as_bool(episode.get("read_gate")) is True


DETERIORATION_EVENTS = ("scheduled_drop", "hazard_drop")  # the deterioration transitions of the clock (episode_clock.py)


def thinking_cycles(episode, log=None):
    """The cycles the episode charged for thinking: the sum of think_cycles over the entries of hidden/log.json. The
    environment writes the field on every accepted action and every refusal (env._entry and env._log_refusal) with the
    value the action server measured from the wall time between calls, nonzero in the generation_time mode only, and the
    system events carry no field and count zero. None without a hidden log (a published run keeps none). The derived
    fields take the log read once by add_derived_fields, or read it themselves."""
    log = episode_files._hidden_log(episode) if log is None else log
    if log is None:
        return None
    return sum(int(e.get("think_cycles") or 0) for e in log)


# the seconds charged for thinking and their split, the columns of THINKING beside the cycles charged
THINKING_SPLIT = ("seconds_charged", "generation_seconds", "wall_minus_monotonic_seconds", "rest_seconds")
GENERATION_TIME = "generation_time"


def merged_spans(spans):
    """The union of (start, end) intervals as disjoint intervals in order, so that model calls that overlap count once."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def thinking_split(episode):
    """The seconds an episode in the generation_time mode was charged for thinking, the sum of think_seconds of the state
    file, which the chess clock charges on the monotonic clock (servers/thinking_clock.py), and of these: generation_seconds,
    the wall seconds of its model calls (started_at and ended_at of llm_calls.json, overlapping calls counted once) inside
    the span from the handover to its last call, less the failed model attempts with their waits (failed_model_seconds of the
    state file), which a call's span holds and the clock credits, and never below zero; wall_minus_monotonic_seconds, the wall gap between the entries of two calls
    minus their monotonic gap summed over the calls, which the monotonic clock leaves out when the machine sleeps or the wall
    clock is set and which the charge therefore never holds; and rest_seconds, what remains, serving, tools and the harness.
    A model call across a sleep counts its wall seconds as generation, so the rest of such an episode can fall below zero,
    which the table shows as it is. Every field is None outside the generation_time mode and for an episode whose files lack
    a field it needs (one played before the chess clock); generation_seconds and rest_seconds are None when a model call
    lacks its span while the other two stand."""
    split = dict.fromkeys(THINKING_SPLIT)
    folder = episode.get("episode_dir")
    if folder is None or episode.get("clock_mode") != GENERATION_TIME:
        return split
    state = read_json(Path(folder) / STATE_FILE, None) or {}
    charged, wall, monotonic, called = (state.get(key) for key in ("think_seconds", "think_seconds_wall", "think_seconds_monotonic", "called_at"))
    if charged is None or wall is None or monotonic is None or called is None or not len(charged) == len(wall) == len(monotonic) == len(called) \
            or None in monotonic or None in wall:
        return split
    split["seconds_charged"] = _r(sum(float(c) for c in charged))
    split["wall_minus_monotonic_seconds"] = _r(sum(max(0.0, float(w) - float(m)) for w, m in zip(wall, monotonic)))
    calls = read_json(Path(folder) / "llm_calls.json", []) or []
    spans = [(c.get("started_at"), c.get("ended_at")) for c in calls]
    if any(start is None or end is None for start, end in spans) or state.get("handover_wall") is None:
        return split
    first, last = float(state["handover_wall"]), max((float(t) for t in called), default=float(state["handover_wall"]))
    if state.get("failed_model_seconds") is None:
        return split
    # a model call's span holds its failed attempts and their waits, which the thinking clock credits, so they leave generation
    spanned = sum(max(0.0, min(end, last) - max(start, first)) for start, end in merged_spans((float(s), float(t)) for s, t in spans))
    generation = max(0.0, spanned - float(state["failed_model_seconds"]))
    split["generation_seconds"] = _r(generation)
    split["rest_seconds"] = _r(split["seconds_charged"] - generation)
    return split


def deterioration(episode, log=None):
    """Whether the asset deteriorated under the agent: the hidden log holds a deterioration transition, a scheduled drop
    or a drop the hazard drew, which a static episode never has. None without a hidden log."""
    log = episode_files._hidden_log(episode) if log is None else log
    if log is None:
        return None
    return any(e.get("event") in DETERIORATION_EVENTS for e in log)


def read_price(episode, log=None):
    """The test cost of an episode: the sum of the prices the hidden log charged for its read entries. None without a
    hidden log."""
    log = episode_files._hidden_log(episode) if log is None else log
    if log is None:
        return None
    return sum(float(e.get("cost") or 0) for e in log if e.get("event") == "read")


def repeated_repair_sets(episode, log=None):
    """The repair sets an episode ordered again, the measure of Ag8: the components ordered on one visit form a repair
    set, an order without a visit forms its own, and every set equal to the set of an earlier visit counts once. None
    without a hidden log."""
    log = episode_files._hidden_log(episode) if log is None else log
    if log is None:
        return None
    sets = {}
    for e in log:
        if e.get("event") == "work_order_created":
            visit = e.get("visit") if e.get("visit") is not None else ("order", e.get("id"))
            sets.setdefault(visit, set()).add(e.get("component"))
    seen, repeated = [], 0
    for components in sets.values():
        repeated += components in seen
        seen.append(components)
    return repeated


def separating_test_unaffordable(episode, trajectory=None):
    """Whether the agent faced a separating test it could not pay for: at some rank_root_causes call of trajectory.json
    that returned read_first, the cost of the separating test (its price plus the downtime of its duration, the
    field the adapter fills from the decision rule) exceeded the budget_left of the latest envelope of a tool that is not
    the diagnostics server's before the call, or the episode's budget (rules.budget of released.json) when no such call
    preceded it. None without a trajectory (the policies write none)."""
    folder = episode.get("episode_dir")
    trajectory = episode_files._trajectory(episode) if trajectory is None else trajectory
    if trajectory is None:
        return None
    budget_left = as_float((read_json(Path(folder) / "released.json", {}).get("rules") or {}).get("budget"))
    for step in trajectory:
        answer = episode_files._answer(step)
        if step.get("server") != CAUSAL_SERVER and "budget_left" in answer:
            budget_left = as_float(answer["budget_left"])
        elif step.get("tool") == "rank_root_causes" and answer.get("read_first") and isinstance(answer.get("separating_test"), dict):
            cost = as_float(answer["separating_test"].get("cost"))
            if budget_left is not None and cost is not None and cost > budget_left:
                return True
    return False


def first_ranking_shares_faulty_set(episode, trajectory=None):
    """Whether the first rank_root_causes answer of the trajectory puts on top two candidates that degrade the same
    components at different levels, the starts of the Ag7 comparison on which the separating test sees no difference.
    None without such an answer."""
    trajectory = episode_files._trajectory(episode) if trajectory is None else trajectory
    components, healthy = episode_files.run_tables(episode)
    for step in trajectory or []:
        candidates = episode_files._answer(step).get("candidates") if step.get("tool") == "rank_root_causes" else None
        if not candidates:
            continue
        try:
            top = [episode_files._levels(c["combination"], components) for c in candidates[:2]]
        except (KeyError, TypeError):
            return None
        faulty = [{c for c, level in zip(components, levels) if level != healthy[c]} for levels in top]
        return len(top) == 2 and faulty[0] == faulty[1] and top[0] != top[1]
    return None


def pairwise_agreement(answers):
    """The share of the pairs of answers that agree, None below two answers, the agreement over the five judge calls of
    Table 14 of AssetOpsBench, whose formula that paper does not print."""
    pairs = list(combinations(answers, 2))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def judge_fields(episode):
    """The judge's record of an episode, hidden/judge.json, per criterion of JUDGE_CRITERIA: the share of true answers over the
    planned calls, where a call without a parsed review counts as false with the denominator kept, as the reproduction scores
    them (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/score_runs.py), the majority of the planned calls, which is what AssetOpsBench prints, and the agreement of the
    parsed calls, beside the protocol and the judge model of the record and the failed calls. Empty without a record, which
    table_judge counts as false."""
    folder = episode.get("episode_dir")
    record = read_json(Path(folder) / "hidden" / "judge.json") if folder is not None else None
    if not record:
        return {}
    calls, planned = record["calls"], record["calls_planned"]
    reviewed = [call["details"] for call in calls if judge_records.reviewed(call)]
    cut = record.get("cut") or record.get("truncation") or {}  # format 3 names the cut to the served context, format 2 the registered pair
    endpoint = record.get("endpoint_identity") or {}
    fields = {"judge_record_protocol": record["protocol"], "judge_record_model": record["judge_model"], "judge_failed_calls": len(calls) - len(reviewed),
              "judge_record_cut": bool(cut.get("records_left_out") or cut.get("texts_headed") or cut.get("answer_headed") or cut.get("texts_cut"))}
    fields["judge_record_weights_commit"] = endpoint.get("weights_commit")
    # the judge that wrote the record: its protocol, its endpoint's bound serving and the code of the judge, so that a later clone
    # whose judging code differs is another judge and never pools with the first
    fields["judge_record_identity"] = pooling_rules.ledger.canonical_sha256(
        {"protocol": record["protocol"], "serving": pooling_rules.serving_key(endpoint), "code": record.get("judge_code_sha256")})
    for criterion in cell_statistics.JUDGE_CRITERIA:
        answers = [details.get(criterion) is True for details in reviewed]
        fields.update({criterion: sum(answers) / planned, f"{criterion}_majority": sum(answers) > planned / 2, f"{criterion}_agreement": pairwise_agreement(answers)})
    return fields


def judge_of_record():
    """The judge block of the tracked serving file (ledger.serving_of_record): its served name and weights commit, which a judge
    record of a run of record must carry."""
    return pooling_rules.ledger.serving_endpoints.serving_of_record()["judge"]


def foreign_judges(episodes, kind):
    """In a root of runs of record a judge record counts only when its served name and weights commit are those of the judge of
    record, so that no row pools two judges: another judge's record leaves its episode unjudged, which the tables count as false,
    and each such episode is named. The records that count must come from one judge (judge_record_identity: protocol, bound serving,
    code of the judge), else the tool stops and names both. Every other root keeps its records."""
    if kind != "record":
        return []
    judge, named, identities = None, [], {}
    for e in episodes:
        if not e.get("judge_record_model"):
            continue
        judge = judge or judge_of_record()  # read only when a record is to be compared, so a root without one needs no serving file
        if (e["judge_record_model"], e.get("judge_record_weights_commit")) != (judge.get("served_name"), judge.get("weights_commit")):
            named.append(f"run {e['run']}, seed {e.get('seed')}: judged by {e['judge_record_model']} at the weights commit "
                         f"{str(e.get('judge_record_weights_commit'))[:12]}, not by the judge of record {judge.get('served_name')} at "
                         f"{str(judge.get('weights_commit'))[:12]}")
            for key in [k for k in e if k.startswith("judge_record") or k == "judge_failed_calls" or k.split("_majority")[0].split("_agreement")[0] in cell_statistics.JUDGE_CRITERIA]:
                e.pop(key)
            e["judge_pass"] = ""
            continue
        identities.setdefault(e.get("judge_record_identity"), []).append(f"run {e['run']}, seed {e.get('seed')}")
    if len(identities) > 1:
        raise SystemExit("the judge records of this root come from more than one judge (protocol, serving or judging code), and no table "
                         "pools two: " + "; ".join(f"{str(identity)[:12]}: {', '.join(where[:2])}" for identity, where in identities.items()))
    return named


def tool_invocations(episode):
    """The steps of Table 3 of AssetOpsBench, which counts tool invocations: the calls that the state files of the action or
    metered servers record and the calls to the diagnostics server together. The verifier's steps cell counts the former
    alone."""
    calls = as_float(episode.get("tool_calls"))
    return None if calls is None else calls + (as_float(episode.get("causal_calls")) or 0)


def planned_steps(episode):
    """The planned steps of Table 12 of AssetOpsBench: the plan_steps cell of Plan-Execute and, for the CausalDiagnosis
    agent, which plans more than once in an episode, the steps of all its plans in plans.json. None for an agent without a plan."""
    label, folder = str(episode.get("agent", "")), episode.get("episode_dir")
    if label.startswith(OURS):
        plans = read_json(Path(folder) / "plans.json") if folder is not None else None
        return None if plans is None else sum(len(record["plan"].get("steps") or []) for record in plans["plans"])
    if not label.startswith("plan_execute"):
        return None
    return as_float(episode.get("plan_steps"))


def tokens_used(episode):
    """The tokens of an episode, input and output together, the average of Table 13 of AssetOpsBench, None without a record."""
    cells = [as_float(episode.get(column)) for column in ("input_tokens", "output_tokens")]
    return None if None in cells else sum(cells)


def wrong_diagnosis(episode, log):
    """Whether the diagnosis an episode submitted is wrong, read from the top-1 cell and the write in the hidden log, the
    one the verifier scores, so it asks for no label and no cell that a run has not written. An episode that declared or wrote no
    diagnosis is not wrong, and the field stays empty without a log, or without a top-1 cell on an episode that the counting rule
    leaves out (counts_as_failure)."""
    correct = cell_statistics.value(episode, "diagnosis_all_correct")
    if correct is None or log is None:
        return ""
    return correct == 0 and any(event.get("event") == "submit_diagnosis" for event in log)


def add_derived_fields(episodes):
    """The per episode fields that scores.csv does not hold, read once from the episode files into the episode dicts for
    the agent tables: from the hidden log thinking_cycles, deterioration, test_cost and
    repeated_repair_sets and with the top-1 cell diagnosis_wrong, from the judge record the fields of judge_fields, from the cells of
    scores.csv tool_invocations,
    planned_steps and tokens, from the trajectory separating_test_unaffordable and level_accuracy (the diagnosis_all_correct
    cell on the starts whose first ranking puts two levels of one faulty set on top, empty elsewhere), test_first, whether
    the first plan of the causal agent was a test first plan (empty without a first plan), deviations, the surprises cell
    of the causal agent under the paper's name, and turns_capped, the turns to a correct diagnosis capped at the step limit
    of the run (capped_turns). The realized elimination comes from add_elimination_fields."""
    for e in episodes:
        log, trajectory = episode_files._hidden_log(e), episode_files._trajectory(e)
        e["thinking_cycles"] = thinking_cycles(e, log)
        e.update(thinking_split(e))
        e["deterioration"] = deterioration(e, log)
        e["test_cost"] = read_price(e, log)
        e["repeated_repair_sets"] = repeated_repair_sets(e, log)
        e.update(judge_fields(e))
        e["tool_invocations"], e["planned_steps"], e["tokens"] = tool_invocations(e), planned_steps(e), tokens_used(e)
        e["scope_pass"] = scope_pass(e)
        e["separating_test_unaffordable"] = separating_test_unaffordable(e, trajectory)
        e["level_accuracy"] = cell_statistics.value(e, "diagnosis_all_correct") if first_ranking_shares_faulty_set(e, trajectory) else ""
        e["diagnosis_wrong"] = wrong_diagnosis(e, log)
        e["unmatched_evidence"] = unmatched_evidence(e)
        e["test_first"] = (e.get("first_plan_mode") == "test_first") if e.get("first_plan_mode") else None
        e.setdefault("deviations", e.get("surprises", ""))
        e["turns_capped"] = capped_turns(e)
        e["restored_clean"] = restored_clean(e)
        e["repair_declared"] = repair_declared(e, log)
        e["refuted_repair"] = refuted_repair(e)
        e["work_orders"] = None if log is None else sum(1 for entry in log if entry.get("event") == "work_order_created")
        e["ranking_calls"] = ranking_calls(e, log)
    return episodes


def ranking_calls(episode, log=None):
    """The calls of the ranking tool of the MCP in an episode, read through mcp_metrics.calls from the transcript of the servers, so
    that a failed or a refused call counts as a call; None without a transcript."""
    folder = episode.get("episode_dir")
    if folder is None or not (Path(folder) / mcp_metrics.transcript.NAME).is_file():
        return None
    found, _ = mcp_metrics.calls(folder, (episode_files._hidden_log(episode) if log is None else log) or [])
    return sum(1 for call in found if call.server == CAUSAL_SERVER and call.tool == mcp_metrics.RANK)


def restored_clean(episode):
    """Restored with no unnecessary order, for any asset: the asset ends with every component at its healthy level and no work order
    was placed on a healthy component (the verdict keys recovered and unnecessary). Empty without a verdict, which the counting rule
    reads as 0. Pass implies it."""
    recovered, unnecessary = as_bool(episode.get("recovered")), as_float(episode.get("unnecessary"))
    return "" if recovered is None or unnecessary is None else bool(recovered and unnecessary == 0)


def repair_declared(episode, log=None):
    """The time to repair of an episode in declared cycles, the measure of the family parallel_time, for any asset: per fault
    occurrence of the episode (verifier.fault_occurrences), the cycles from its onset to the agent's restoring close that ended it
    less the cycles charged for thinking from the onset to the creation of the order whose close that is (an entry with t_start tau
    and think_cycles theta charges the cycles tau - theta to tau - 1, wherever it stands in the log), and the
    horizon of the preset for an occurrence the agent did not restore. The onset is verifier.fault_onset, the cycle of the hidden
    log entry that began the occurrence, cycle zero for a fault present at the start (registered definitions.repair_declared), the
    one onset of the verdict's delays as well; the largest over every occurrence of the
    components, 0 when none was off its level, and the horizon for an episode without a verdict or without a log. Empty
    for an episode whose run records no horizon."""
    horizon = as_float(episode.get("horizon"))
    if horizon is None:
        return ""
    verdict, log = episode_files._verdict(episode), episode_files._hidden_log(episode) if log is None else log
    if not verdict or log is None or not isinstance(verdict.get("hidden_start"), dict):
        return int(horizon)
    asset = verifier._asset_truth(log)  # the truth model that scored the episode: fault_occurrences reads combinations with its levels()
    values = []
    for component, began, ended in verifier.fault_occurrences(log, verdict["hidden_start"], asset):
        if ended is None or not verifier.restored_by_agent(log, ended, component):
            values.append(int(horizon))
            continue
        onset = verifier.fault_onset(log, began)
        delay = verifier._time(log[ended]) - onset
        ordered = next(e["t_start"] for e in log if e["event"] == "work_order_created" and e["id"] == log[ended]["id"])
        thinking = sum(max(0, min(e["t_start"], ordered) - max(e["t_start"] - int(e.get("think_cycles") or 0), onset))
                       for e in log if e.get("t_start") is not None)
        values.append(int(delay - thinking))
    return max(values, default=0)


def refuted_repair(episode):
    """Whether the episode met a refuted repair: a work order whose prediction the MCP verified as inconsistent with the readings
    after it (the verification stored for the order, verdict key prediction_verdict). Empty without a verdict."""
    verdict = episode_files._verdict(episode)
    if not verdict:
        return ""
    return any(isinstance(row.get("verification"), dict) and row["verification"].get("consistent") is False
               for row in verdict.get("prediction_verdict") or [])


def unmatched_evidence(e):
    """Whether the episode cited an evidence identifier that the released evidence and the stored predictions do not hold, the
    complement of the verifier's provenance check; empty where the row has no provenance cell."""
    ok = as_bool(e.get("provenance_ok"))
    return "" if ok is None else not ok


def turn_record(episode):
    """The turns to a correct diagnosis of a scored episode as (turn, observed, step limit): the turn of the correct
    diagnosis, or the step limit of the run, where the episode is censored, when it reached none. None for an episode the
    verifier did not score or whose run.json names no step limit."""
    cap = as_float(episode.get("max_steps"))
    if cap is None or cell_statistics.value(episode, "diagnosis_all_correct") is None:
        return None
    turn = as_float(episode.get("turns_to_correct_diagnosis"))
    return (cap, False, cap) if turn is None else (min(turn, cap), True, cap)


def capped_turns(episode):
    """The turns of turn_record capped at the step limit, the per episode term whose mean over the episodes of one step
    limit is the restricted mean survival time over turns, since the only censoring is at the step limit."""
    record = turn_record(episode)
    return None if record is None else record[0]
