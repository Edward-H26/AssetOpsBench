import json
from collections import Counter
from statistics import median
from CausalDiagnosis.utils.csv_rows import as_bool, as_float  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402  the verifier's reading of the log, which the time to repair in declared cycles shares
from CausalDiagnosis.evaluation.metrics import mcp_metrics  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AS_RELEASED, HEALTHY_CONTROL_PRESET, HEALTHY_CONTROL_ROWS, HYPOTHESIS_MEASURES, TRAINING_PRESET, row_label  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import episode_files, cell_statistics, family_correction, episode_selection, episode_fields, training_tables


# the strata of STRATA.csv: the separable starts under the reference policy's separability and under E18 beside the inseparable
# ones, and the starts both mark inseparable, where a declaration earns its credit
STRATA = ("deterioration", "separable", "inseparable", "separable_e18", "inseparable_e18", "inseparable_both", "test_unaffordable")
EFFICIENCY_METRICS = ("tool_invocations", "wall_s")
# the ratios of sums of EFFICIENCY as (numerator, denominator): the tokens and the runtime of every episode of a row, the failed ones
# included, per episode that passes, and the share of the input tokens that the server reported as cached
EFFICIENCY_RATIOS = {"tokens_per_solved": ("tokens", "pass"), "seconds_per_solved": ("wall_s", "pass"),
                     "cached_share": ("cached_tokens", "input_tokens")}
FAILURE_END = "failure"  # the end reason of an episode that a failure of the failure card ended
# the steps of FUNNEL, each the share of the episodes of a row that reached it, and the counters of the harness from the columns it writes
FUNNEL_STEPS = ("diagnosis_accepted", "faulty_set_right", "diagnosis_all_correct", "orders_on_every_faulty", "no_order_on_healthy", "finished")
FUNNEL_COUNTED = {"finished_by_harness": ("finish_by", "harness"), "runner_limit": ("runner_end", "own_limit")}
FUNNEL_SUMMED = ("cut_replies", "failed_calls", "replays", "wall_limit")
FUNNEL_COLUMNS = (*(f"{step}{part}" for step in FUNNEL_STEPS for part in ("", "_k")), "named_faulty_mean", "faulty_at_write_mean", "named_more",
                  "named_as_many", "named_fewer", "orders_on_healthy_per_healthy_component", "failure_before_order", "at_cost_cap",
                  "refused_diagnoses", *FUNNEL_COUNTED, *FUNNEL_SUMMED)


def funnel_facts(episode):
    """The steps and the counts of FUNNEL for one episode, read from its hidden log, its oracle and its row of scores.csv and never
    from a record of a runner: whether the server accepted a diagnosis, the faulty set and Top-1 under the counting rule, whether an
    accepted order stands on every component the oracle repairs and on no other, whether the log holds a finish, the components the
    first accepted diagnosis names as faulty and those faulty at its cycle, the orders on healthy components and the healthy
    components of the start, whether a failure ended the episode before any order, the orders the oracle holds due and whether the
    first repair of an accepted order closed no later than the first failure, the deadline of the failure card, which an order
    still open at the failure no longer meets. A count the files cannot give is None."""
    log, oracle = episode_files._hidden_log(episode) or [], episode_files._oracle(episode)
    due = None if oracle is None else {o["component"] for o in oracle.get("work_orders") or []}
    ordered = [e.get("component") for e in log if e.get("event") == "work_order_created"]
    diagnoses = [e for e in log if e.get("event") == "submit_diagnosis" and e.get("diagnosis") is not None]
    healthy_orders = None if due is None else sum(1 for c in ordered if c not in due)
    close_times = [e.get("t_start") or 0 for e in log if e.get("event") == "work_order_closed"]
    failure_times = [e.get("t_start") or 0 for e in log if e.get("event") == "failure"]
    facts = {"diagnosis_accepted": bool(diagnoses), "faulty_set_right": cell_statistics.value(episode, "faulty_set_right") == 1.0,
             "diagnosis_all_correct": cell_statistics.value(episode, "diagnosis_all_correct") == 1.0,
             "orders_on_every_faulty": due is not None and due <= set(ordered), "no_order_on_healthy": healthy_orders == 0,
             "finished": any(e.get("event") == "finish" for e in log),
             "failure_before_order": episode.get("end_reason") == FAILURE_END and not ordered,
             "orders_due": None if due is None else len(due),
             "order_in_time": bool(close_times) and (not failure_times or min(close_times) <= min(failure_times)),
             "healthy_orders": healthy_orders, "healthy_components": None, "named": None, "faulty_at_write": None}
    if log and log[0].get("hidden") is not None:
        truth = verifier._asset_truth(log)
        facts["healthy_components"] = len(truth.components) - len(truth.faulty(truth.levels(log[0]["hidden"])))
        if diagnoses:
            first = diagnoses[0]
            facts["named"] = len(truth.faulty(truth.levels(first["diagnosis"])))
            facts["faulty_at_write"] = len(truth.faulty(mcp_metrics.combination_at(log, first.get("t_start") or 0, truth)))
    return facts


def funnel_row(episodes):
    """The cells of one row of FUNNEL over its episodes: each step as a share of the episodes played with its count, the mean of the
    components named as faulty by the first accepted diagnosis beside the mean of those faulty at its cycle with the shares of the
    diagnoses that name more, as many and fewer, the orders on healthy components per healthy component of the start, the shares of
    the episodes that a failure ended before any order and of those whose cost ratio stands at the cap of the reward, the refused
    diagnoses by reason, and the counters of the harness."""
    facts = [funnel_facts(e) for e in episodes]
    row = {}
    for step in FUNNEL_STEPS:
        k = sum(1 for f in facts if f[step])
        row[step], row[f"{step}_k"] = cell_statistics._share(k, len(facts)), k
    named = [f for f in facts if f["named"] is not None]
    row["named_faulty_mean"] = cell_statistics._share(sum(f["named"] for f in named), len(named))
    row["faulty_at_write_mean"] = cell_statistics._share(sum(f["faulty_at_write"] for f in named), len(named))
    for column, keep in (("named_more", lambda a, b: a > b), ("named_as_many", lambda a, b: a == b), ("named_fewer", lambda a, b: a < b)):
        row[column] = cell_statistics._share(sum(1 for f in named if keep(f["named"], f["faulty_at_write"])), len(named))
    counted = [f for f in facts if f["healthy_orders"] is not None and f["healthy_components"] is not None]
    row["orders_on_healthy_per_healthy_component"] = cell_statistics._share(sum(f["healthy_orders"] for f in counted), sum(f["healthy_components"] for f in counted))
    row["failure_before_order"] = cell_statistics._share(sum(1 for f in facts if f["failure_before_order"]), len(facts))
    ratios = [as_float(e.get("cost")) / as_float(e.get("cost_oracle")) for e in episodes
              if as_float(e.get("cost")) is not None and (as_float(e.get("cost_oracle")) or 0) > 0]
    row["at_cost_cap"] = cell_statistics._share(sum(1 for r in ratios if r >= verifier.COST_CAP), len(ratios))
    refused = Counter()
    for e in episodes:
        refused.update(json.loads(e.get("diagnosis_refusals") or "{}"))
    row["refused_diagnoses"] = json.dumps(dict(sorted(refused.items()))) if refused else ""
    for column, (source, word) in FUNNEL_COUNTED.items():
        row[column] = sum(1 for e in episodes if e.get(source) == word)
    for column in FUNNEL_SUMMED:
        row[column] = int(sum(as_float(e.get(column)) or 0 for e in episodes))
    return row


def table_funnel(episodes, ctx):
    """FUNNEL: one row per condition of the agents tables, the steps of funnel_row over the episodes played, with the counts of the
    counting rule. A row with a missing episode prints incomplete, and its shares are over the episodes played, since a missing
    episode is never a failure."""
    return [cell_statistics.trace(ctx, {**dict(zip(episode_selection.CONDITION, condition)), "n": len(eps), "status": cell_statistics.row_status(eps, ctx), **cell_statistics.completeness(eps, ctx), **funnel_row(eps)}, eps)
            for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items()]


# the classes of FAILURES: the taxonomy of the verifier in its order, then none, the class of a success and of a failure that no class describes
FAILURE_CLASS_ORDER = (*verifier.FAILURE_CLASSES, "none")
FAILURE_COLUMNS = (*(f"{name}{part}" for name in FAILURE_CLASS_ORDER for part in ("", "_k")), "recovery_claims",
                   *(f"recovery_contradicted{part}" for part in ("", "_low", "_high", "_n")))


def recovery_claim(episode):
    """(claims, contradicted) of the closing summary of an episode, from the summary check of its verdict: whether the summary claims that
    the asset recovered, and whether that claim stands while the asset ended off its healthy state. An episode without a verdict has no
    checked summary and claims nothing."""
    verdict = episode_files._verdict(episode) or {}
    claims = bool((verdict.get("summary_check") or {}).get("claims_recovery"))
    return claims, claims and not verdict.get("recovered")


def table_failures(episodes, ctx):
    """FAILURES: one row per condition of the agents tables, the failure class of every episode played (the column failure_class of
    scores.csv) as a share of the episodes played and a count, an episode without a verdict in the unscored count alone, and the share of
    the closing claims of recovery that the hidden state contradicts, counted over the claims, with its interval and standard error from a
    bootstrap clustered by recorded block. No test."""
    rows = []
    for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items():
        row = {**dict(zip(episode_selection.CONDITION, condition)), "n": len(eps), "status": cell_statistics.row_status(eps, ctx), **cell_statistics.completeness(eps, ctx)}
        classes = Counter(e.get("failure_class") for e in eps)
        for name in FAILURE_CLASS_ORDER:
            row[name], row[f"{name}_k"] = cell_statistics._share(classes[name], len(eps)), classes[name]
        claims = [(cell_statistics.cluster_of(e), contradicted) for e in eps for claimed, contradicted in [recovery_claim(e)] if claimed]
        row["recovery_claims"] = row["recovery_contradicted_n"] = len(claims)
        if claims:
            boot = cell_statistics.proportion_cells([1.0 if contradicted else 0.0 for _, contradicted in claims], [cluster for cluster, _ in claims], ctx)
            row.update(cell_statistics.interval_cells("recovery_contradicted", boot, ctx, estimate="proportion"))
        else:
            row.update(dict.fromkeys(("recovery_contradicted", "recovery_contradicted_low", "recovery_contradicted_high", "recovery_contradicted_se"), ""))
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


def first_drop(episode):
    """The first cycle at which the hidden log of an episode records a deterioration transition, a component leaving its level with
    no action (DETERIORATION_EVENTS), None for an episode without one or without a log."""
    times = [e.get("t_start") or 0 for e in episode_files._hidden_log(episode) or [] if e.get("event") in episode_fields.DETERIORATION_EVENTS]
    return min(times) if times else None


def drop_groups(episodes):
    """[(group, median, episodes)]: the episodes split at the median of their first drop, early at or before it and late after it,
    and the episodes without a drop as a group of their own, each group only when it holds an episode."""
    drops = [(e, first_drop(e)) for e in episodes]
    times = [t for _e, t in drops if t is not None]
    cut = median(times) if times else ""
    parts = (("early", [e for e, t in drops if t is not None and t <= cut]), ("late", [e for e, t in drops if t is not None and t > cut]),
             ("no_drop", [e for e, t in drops if t is None]))
    return [(group, cut, part) for group, part in parts if part]


def table_first_drop(episodes, ctx):
    """FIRST_DROP: on the top layer, each condition of the agents tables split at the median of the first drop over the starts it
    played (drop_groups), with Top-1 and pass per group. Descriptive, no test."""
    top = [e for e in episode_selection.agent_episodes(episodes) if e.get("config") == TRAINING_PRESET and episode_selection.at_main_levels(e)]
    rows = []
    for condition, eps in cell_statistics.groups(top, *episode_selection.CONDITION).items():
        for group, cut, part in drop_groups(eps):
            row = {**dict(zip(episode_selection.CONDITION, condition)), "group": group, "median_first_drop": cut, "n": len(part)}
            for m in HYPOTHESIS_MEASURES:
                row.update(cell_statistics.summary(part, m, ctx))
            rows.append(cell_statistics.trace(ctx, row, part))
    return rows


def table_healthy_control(episodes, ctx):
    """HEALTHY_CONTROL: one row per row of the main table on the preset of the healthy control, the start with no faulty component,
    counts with no interval and no test: of the registered episodes, how many accepted a diagnosis, how many raised a false alarm with
    the components falsely named, how many placed a work order with the number of orders. Its episodes enter no other table."""
    control = [e for e in episodes if e.get("mode") == "gym" and e.get("model") and e.get("config") == HEALTHY_CONTROL_PRESET]
    rows = []
    for (model, surface), eps in cell_statistics.groups(control, "model", "surface").items():
        for row in HEALTHY_CONTROL_ROWS:
            arm = episode_selection._arm(eps, row_label(row, model), row[1])
            if not arm:
                continue
            logs = [episode_files._hidden_log(e) or [] for e in arm]
            alarms, orders = [as_float(e.get("false_alarms")) or 0 for e in arm], [as_float(e.get("unnecessary")) or 0 for e in arm]
            rows.append(cell_statistics.trace(ctx, {"agent": row_label(row, model), "agent_server": row[1], "model": model, "surface": surface,
                                    "config": HEALTHY_CONTROL_PRESET, "n": len(arm), **cell_statistics.completeness(arm, ctx),
                                    "diagnosed": sum(1 for log in logs if any(entry.get("event") == "submit_diagnosis" for entry in log)),
                                    "false_alarm_episodes": sum(1 for a in alarms if a > 0), "components_falsely_named": int(sum(alarms)),
                                    "work_order_episodes": sum(1 for o in orders if o > 0), "work_orders": int(sum(orders))}, arm))
    return rows


def table_restored(episodes, ctx):
    """RESTORED: per condition of the agents tables, the share of episodes restored with no unnecessary order (restored_clean) as the
    mean with its interval and standard error and k of n, the share of pass beside it, and the share of the episodes restored with no
    unnecessary order whose diagnosis did not match the faulty set exactly. No test."""
    rows = []
    for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items():
        row = {**dict(zip(episode_selection.CONDITION, condition)), "n": len(eps), **cell_statistics.summary(eps, "restored_clean", ctx), **cell_statistics.summary(eps, "pass", ctx)}
        clean = [e for e in eps if cell_statistics.value(e, "restored_clean") == 1.0]
        row["restored_clean_k"] = len(clean)
        row["restored_clean_not_exact"] = _r(sum(1 for e in clean if cell_statistics.value(e, "faulty_set_right") != 1.0) / len(clean)) if clean else ""
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


def table_thinking(episodes, ctx):
    """One row per CONDITION of the agents of the six rows played in the generation_time mode, trained variants among them as rows
    of their own, read through the episodes of the training tables so that a fold without an adapter reads the runs of no training
    and its episodes count as aliased: the cycles charged for thinking and the seconds charged with their split (thinking_split), each the mean over the
    episodes with a bootstrap interval clustered by block, and the episodes whose files lack the split (unsplit), those played
    before the episodes recorded the spans of their model calls. The cycles are the seconds charged over think_seconds_per_cycle with the rest carried, so they read
    against the seconds charged."""
    rows = []
    untrained = [e for e in episode_selection.agent_episodes(episodes, every_clock_mode=True) if e.get("training_variant", AS_RELEASED) == AS_RELEASED]
    # the trained variants through the episodes of the training tables, so that a fold without an adapter reads the runs of no
    # training by the rule of the training table and its rows count them as aliased
    trained = [e for e in training_tables.training_episodes(episodes, ctx) if e.get("training_variant") != AS_RELEASED]
    timed = [e for e in untrained + trained if e.get("clock_mode") == episode_fields.GENERATION_TIME]
    for condition, eps in cell_statistics.groups(timed, *episode_selection.CONDITION).items():
        row = {**dict(zip(episode_selection.CONDITION, condition)), "n": len(eps), "unsplit": sum(1 for e in eps if e.get("seconds_charged") is None),
               "aliased": sum(1 for e in eps if e.get("aliased"))}
        for metric in ("thinking_cycles", *episode_fields.THINKING_SPLIT):
            row.update(cell_statistics.summary(eps, metric, ctx))
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


def table_calls(episodes, ctx):
    """One row per CONDITION of the five agents: model calls, thinking cycles and tokens with bootstrap intervals clustered by
    block (none is a share), the paired differences of the first two against Agents-as-Tools with the server in the same clock mode,
    surface and SETTINGS, the agent that decides every step, and under the prefix vs_bare_ against Agents-as-Tools
    without the server, with agent and diagnostics_server left out of the pairing key and empty on the reference row, and
    Holm over the model call differences against the row with the server."""
    rows = []
    for (model, config, clock, surface, *settings), eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), "model", "config", "clock_mode",
                                                                  "surface", *episode_selection.SETTINGS).items():
        reference = episode_selection._arm(eps, f"agent_as_tool_{model}", True)
        bare = episode_selection._arm(eps, f"agent_as_tool_{model}", False)
        for (agent, server), level in cell_statistics.groups(eps, "agent", "diagnostics_server").items():
            row = {"agent": agent, "model": model, "config": config, "diagnostics_server": server, "clock_mode": clock, "surface": surface,
                   **dict(zip(episode_selection.SETTINGS, settings)), "n": len(level)}
            for m in ("llm_calls", "thinking_cycles", "tokens"):
                row.update(cell_statistics.summary(level, m, ctx))
            is_reference = agent == f"agent_as_tool_{model}" and as_bool(server) is True
            is_bare = agent == f"agent_as_tool_{model}" and as_bool(server) is False
            # a pair whose rows miss a planned episode or are refused is not computed (row_status), and its test enters the Holm at one
            row["status"] = cell_statistics.row_status(level, ctx) or (cell_statistics.row_status(reference, ctx) if reference and not is_reference else "") or (
                cell_statistics.row_status(bare, ctx) if bare and not is_bare else "") if (reference and not is_reference) or (bare and not is_bare) else ""
            if reference and not is_reference:
                row.update(cell_statistics._paired_cells([] if row["status"] else level, [] if row["status"] else reference, episode_selection.AGENT_KEY, ("llm_calls", "thinking_cycles"), ctx))
            else:
                row.update(cell_statistics._no_pairing(("llm_calls", "thinking_cycles")))
            if bare and not is_bare:
                row.update(cell_statistics._prefixed(cell_statistics._paired_cells([] if row["status"] else level, [] if row["status"] else bare, episode_selection.AGENT_KEY, ("llm_calls", "thinking_cycles"), ctx),
                                     "vs_bare_"))
            else:
                row.update(cell_statistics._prefixed(cell_statistics._no_pairing(("llm_calls", "thinking_cycles")), "vs_bare_"))
            against = (reference if reference and not is_reference else []) + (bare if bare and not is_bare else [])
            rows.append(cell_statistics.trace(ctx, row, level, against or None))
    return family_correction.add_holm(rows, "llm_calls", ctx["alpha"])


def in_stratum(episode, stratum):
    """Whether an episode belongs to a stratum of STRATA.csv: deterioration and test_unaffordable read the derived flags,
    separable and inseparable the separable cell of scores.csv (the reference policy's separability, which the verifier reads),
    separable_e18 and inseparable_e18 the server's E18 grade inside the grade cell, and inseparable_both a start that both mark
    inseparable, where a declaration earns its credit. An unknown flag or a start without an E18 grade leaves the episode
    outside."""
    if stratum in ("separable", "inseparable"):
        return as_bool(episode.get("separable")) is (stratum == "separable")
    if stratum in ("separable_e18", "inseparable_e18"):
        return episode_files.e18_separable(episode) is (stratum == "separable_e18")
    if stratum == "inseparable_both":
        return in_stratum(episode, "inseparable") and in_stratum(episode, "inseparable_e18")
    flag = "separating_test_unaffordable" if stratum == "test_unaffordable" else stratum
    return as_bool(episode.get(flag)) is True


def table_strata(episodes, ctx):
    """One row per CONDITION and stratum of the six rows, every clock mode of a preset in rows of its own, with the intervals of
    pass, success and the share of episodes that declared insufficient evidence (abstained) and the number of episodes in the
    stratum. A stratum without episodes keeps its row with empty cells, so the strata of an agent read side by side."""
    rows = []
    for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items():
        for stratum in STRATA:
            inside = [e for e in eps if in_stratum(e, stratum)]
            row = {**dict(zip(episode_selection.CONDITION, condition)), "stratum": stratum, "n": len(inside)}
            for m in ("pass", "success", "abstained"):
                row.update(cell_statistics.summary(inside, m, ctx))
            rows.append(cell_statistics.trace(ctx, row, inside))
    return rows


def table_efficiency(episodes, ctx):
    """One row per CONDITION of the six rows and fault count of the start, the metrics of Table 3 of AssetOpsBench with the
    single-fault and two-fault starts in the place of its single-agent and multi-agent tasks: the steps (tool_invocations) and
    the runtime per episode (wall_s, the wall clock around the agent, which stops before the verifier and the judge), each
    with a bootstrap interval clustered by block and the standard deviation over episodes, and the ratios of EFFICIENCY_RATIOS,
    each with such an interval and its number of episodes. A count without episodes keeps its row with empty cells, as a
    stratum does."""
    rows = []
    for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items():
        for faults, count in episode_selection.FAULT_COUNTS.items():
            inside = [e for e in eps if as_float(e.get("n_faulty")) == count]
            row = {**dict(zip(episode_selection.CONDITION, condition)), "faults": faults, "n": len(inside)}
            for m in EFFICIENCY_METRICS:
                row.update({**cell_statistics.summary(inside, m, ctx), **cell_statistics.spread(inside, m)})
            for name, (numerator, denominator) in EFFICIENCY_RATIOS.items():
                row.update(cell_statistics.ratio_of_sums(inside, numerator, denominator, name, ctx))
            rows.append(cell_statistics.trace(ctx, row, inside))
    return rows


def table_plans(episodes, ctx):
    """One row per CONDITION of the six rows with a plan, the metrics of Table 12 of AssetOpsBench: the planned steps
    per episode (planned_steps) with a bootstrap interval clustered by block, their standard deviation, smallest and largest
    value, and the episodes that planned no step."""
    rows = []
    for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items():
        planned = cell_statistics.with_metric(eps, "planned_steps")
        if planned:
            steps = [int(cell_statistics.value(e, "planned_steps")) for e in planned]
            rows.append(cell_statistics.trace(ctx, {**dict(zip(episode_selection.CONDITION, condition)), "n": len(eps), **cell_statistics.summary(planned, "planned_steps", ctx), **cell_statistics.spread(planned, "planned_steps"),
                                    "planned_steps_min": min(steps), "planned_steps_max": max(steps), "planned_steps_zero": steps.count(0)}, planned))
    return rows
