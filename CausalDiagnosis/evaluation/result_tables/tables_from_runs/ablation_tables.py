from CausalDiagnosis.utils.csv_rows import as_bool  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402  the verifier's reading of the log, which the time to repair in declared cycles shares
from CausalDiagnosis.evaluation.metrics import mcp_metrics  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AGENTS_PRESET, TRAINING_PRESET  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import cluster_bootstrap, ratio, spearman  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import episode_files, cell_statistics, family_correction, episode_selection


BINS = 10  # calibration bins of A12 over the posterior of the top candidate


# ----- factor tables (A9, A10, SWITCHES) -----
def factor_table(factor, reference, metrics, paired_metrics, by=("agent", "model"), select=None, unmatched=()):
    """Rows per value of the by fields, the agent and the model unless a table names more, and per level of the factor,
    with the mean of every metric and the paired difference to the reference level for the paired metrics, of which the
    first is the table's primary test. A table that reads part of the episodes names them with select, and the fields of
    unmatched leave the pairing key beside the factor, as the preset does when the levels of a factor play on two
    presets. A level paired against an incomplete or refused row, or incomplete or refused itself, prints the status and no paired
    cells, and its test enters the Holm at p = 1."""

    def build(episodes, ctx):
        rows = []
        for keys, eps in cell_statistics.groups(select(episodes) if select else episodes, *by).items():
            levels = {}
            for e in eps:
                levels.setdefault(e.get(factor, ""), []).append(e)
            for level, level_eps in sorted(levels.items(), key=lambda kv: str(kv[0])):
                row = {**dict(zip(by, keys)), factor: level, "n": len(level_eps)}
                for m in metrics:
                    row.update(cell_statistics.summary(level_eps, m, ctx))
                pairs = paired_metrics and str(level) != str(reference) and reference in levels
                if paired_metrics:
                    row["status"] = (cell_statistics.row_status(level_eps, ctx) or cell_statistics.row_status(levels[reference], ctx)) if pairs else ""
                for m in paired_metrics:
                    if pairs:
                        row.update(cell_statistics.paired([] if row["status"] else level_eps, [] if row["status"] else levels[reference], (factor, *unmatched), m, ctx))
                    else:
                        row.update(cell_statistics._no_pairing((m,)))
                against = levels[reference] if paired_metrics and str(level) != str(reference) and reference in levels else None
                rows.append(cell_statistics.trace(ctx, row, level_eps, against))
        return family_correction.add_holm(rows, paired_metrics[0], ctx["alpha"]) if paired_metrics else rows

    columns = [*by, factor, "n"]
    for m in metrics:
        columns += [m, f"{m}_low", f"{m}_high", f"{m}_n"]
    for m in paired_metrics:
        columns += [f"{m}_diff", f"{m}_diff_low", f"{m}_diff_high", f"{m}_p"]
    if paired_metrics:
        columns += ["status", "pairs", "clusters", "holm_p", "reject"]
    return columns, build


# ----- A3, A7, A8, A12 -----
def table_a3(episodes, ctx):
    """Sc3 on layer 0: the static answers of a model against the runs of the agents table, the six rows on its preset at the
    main level of every setting."""
    rows = []
    static = cell_statistics.groups([e for e in episodes if e.get("mode") == "static"], "model")
    gym = cell_statistics.groups([e for e in episode_selection.agent_episodes(episodes) if e.get("config") == AGENTS_PRESET and episode_selection.at_main_levels(e)], "model")
    for (model,) in sorted(set(static) | set(gym)):
        row = {"model": model, "static_n": len(static.get((model,), [])), "gym_n": len(gym.get((model,), []))}
        for metric, prefix in (("judge_pass", "static_judge_pass"), ("static_faulty_set_correct", "static_faulty_set_correct")):
            s = cell_statistics.summary(static.get((model,), []), metric, ctx)
            row.update({prefix: s[metric], **{f"{prefix}{part}": s[f"{metric}{part}"] for part in ("_low", "_high", "_se")}})
        recorded = cell_statistics.summary(gym.get((model,), []), "pass", ctx)
        row.update({"gym_recorded_pass": recorded["pass"], **{f"gym_recorded_pass{part}": recorded[f"pass{part}"] for part in ("_low", "_high", "_se")}})
        row.update({"gym_cost": cell_statistics.summary(gym.get((model,), []), "cost", ctx)["cost"]})
        rows.append(cell_statistics.trace(ctx, row, static.get((model,), []) + gym.get((model,), [])))
    return rows


A7_KEY = ("agent", "model", "diagnostics_server")  # the rows with and without the server share an agent and a model


def table_a7(episodes, ctx):
    """En4 per row of the main comparison: the pass under each verifier scope (scope_pass) on the episodes of en4_episodes,
    and per model Spearman's rank correlation of the rows between the scopes, since across models the differences between the
    models would raise the correlation whatever the read gate does."""
    rows, by_scope, scoped = [], {"writes": {}, "all": {}}, []
    for key, eps in cell_statistics.groups(episode_selection.en4_episodes(episodes), *A7_KEY).items():
        row = dict(zip(A7_KEY, key))
        for scope in ("writes", "all"):
            s = cell_statistics.summary([e for e in eps if e.get("verifier_scope") == scope], "scope_pass", ctx)
            row.update({f"pass_{scope}": s["scope_pass"], f"pass_{scope}_low": s["scope_pass_low"], f"pass_{scope}_high": s["scope_pass_high"], f"n_{scope}": s["scope_pass_n"]})
            if s["scope_pass"] != "":
                by_scope[scope][key] = s["scope_pass"]
        mine = [e for e in eps if e.get("verifier_scope") in ("writes", "all") and cell_statistics.value(e, "pass") is not None]
        scoped += mine
        rows.append(cell_statistics.trace(ctx, row, mine))
    shared = sorted(set(by_scope["writes"]) & set(by_scope["all"]))
    for model in sorted({row["model"] for row in rows}):
        of_model = [key for key in shared if dict(zip(A7_KEY, key))["model"] == model]
        rho = spearman([by_scope["writes"][k] for k in of_model], [by_scope["all"][k] for k in of_model]) if len(of_model) > 1 else None
        rows.append(cell_statistics.trace(ctx, {"agent": "all agents", "model": model, "diagnostics_server": "", "rank_correlation": _r(rho), "agents_in_both_scopes": len(of_model)},
                          [e for e in scoped if tuple(e.get(field, "") for field in A7_KEY) in of_model]))
    return rows


A8_KEY = ("agent", "model", "config", "diagnostics_server", "abstention_rule")
SEPARABILITY_LABELS = ("separability", "e18")  # the independent label the verifier reads, and the server's E18 grade


def table_a8(episodes, ctx):
    """One row per agent, model, preset, server level, abstention rule and label, every other setting at its main level
    (at_main_levels_but): abstention precision (correct
    declarations among declarations) and recall (correct declarations among the starts the label marks inseparable) with
    intervals from a bootstrap clustered by block. Under the reference policy's separability a declaration is correct when the
    verifier says so (the abstention_correct cell) and the starts are those of the separable cell. Under E18 the starts are those
    the server's grade covers, and a declaration that cites a reading released before it is correct on a start it marks
    inseparable."""
    rows = []
    for (*key, surface), eps in cell_statistics.groups([e for e in episodes if episode_selection.at_main_levels_but(e, "abstention_rule")], *A8_KEY, "surface").items():
        scored = [e for e in eps if cell_statistics.value(e, "abstained") is not None]
        for label in SEPARABILITY_LABELS:
            if label == "separability":
                graded = scored
                marked = [as_bool(e.get("separable")) is False for e in graded]
                right = [cell_statistics.value(e, "abstained") == 1 and cell_statistics.value(e, "abstention_correct") == 1 for e in graded]
            else:
                graded = [e for e in scored if episode_files.e18_separable(e) is not None]
                marked = [episode_files.e18_separable(e) is False for e in graded]
                right = [inside and cell_statistics.value(e, "abstained") == 1 and cell_statistics.value(e, "abstention_cites_reading") == 1 for e, inside in zip(graded, marked)]
            if not graded:
                continue
            declared = [cell_statistics.value(e, "abstained") == 1 for e in graded]
            blocks = [cell_statistics.cluster_of(e) for e in graded]
            abstentions, inseparable = sum(declared), sum(marked)
            precision, recall = cell_statistics.proportion_cells(right, blocks, ctx, declared), cell_statistics.proportion_cells(right, blocks, ctx, marked)
            # the declarations and the label are cells of the verdict, so an unscored episode enters neither denominator; it is counted
            rows.append(cell_statistics.trace(ctx, {**dict(zip(A8_KEY, key)), "label": label, "n": len(graded), "unscored": len(eps) - len(scored), "abstentions": abstentions,
                                    "inseparable_starts": inseparable, **cell_statistics.interval_cells("precision", precision, ctx, "proportion"),
                                    **cell_statistics.interval_cells("recall", recall, ctx, "proportion"), "surface": surface}, graded))
    return rows


def rankings(episode):
    """(readings so far, top candidate as a list, its posterior, the hidden combination at the moment of the answer as a list
    or None) of every rank_root_causes answer of an episode, from the transcript that the servers wrote and the hidden log
    (mcp_metrics.ranked_answers): a reading is a call whose log entry is a read, and the truth follows the log to the moment of
    the answer. The policies call no diagnostics server and contribute nothing."""
    folder = episode.get("episode_dir")
    return [] if folder is None else mcp_metrics.ranked_answers(folder)


def table_a12(episodes, ctx):
    """One row per agent, model, preset, ranking model and number of readings (and one over all readings), every other setting at its
    main level (at_main_levels_but): the Brier
    score of the top candidate's posterior and the top candidate rate, each with a bootstrap interval clustered by the episode's
    block. The calibration bins go to A12_curves.csv."""
    rows = []
    ctx["a12_curves"] = []
    for (agent, model, config, ranking), eps in cell_statistics.groups([e for e in episodes if episode_selection.at_main_levels_but(e, "ranking_model")], "agent", "model", "config",
                                                       "ranking_model").items():
        points, owners = [], []  # (block, readings, hit, posterior), and the episode of each point for the results index
        for e in eps:
            # each answer against the hidden combination at its own moment, which a repair or a drop moves away from the start,
            # and no point for an answer that the log cannot place
            mine = [(cell_statistics.cluster_of(e), readings, float(top == truth), p) for readings, top, p, truth in rankings(e) if truth is not None]
            points += mine
            owners += [e] * len(mine)
        if not points:
            continue
        for readings in [*sorted({r for _, r, _, _ in points}), "all"]:
            subset = [pt for pt in points if readings == "all" or pt[1] == readings]
            behind = list({id(e): e for pt, e in zip(points, owners) if readings == "all" or pt[1] == readings}.values())
            brier = cluster_bootstrap([[(p - hit) ** 2, 1.0] for _, _, hit, p in subset], [pt[0] for pt in subset], ratio, ctx["resamples"], ctx["seed"])
            top1 = cell_statistics.proportion_cells([hit for _, _, hit, _ in subset], [pt[0] for pt in subset], ctx)
            top1_cells = cell_statistics.interval_cells("top1", top1, ctx, "proportion")
            rows.append(cell_statistics.trace(ctx, {"agent": agent, "model": model, "config": config, "ranking_model": ranking, "readings": readings, "calls": len(subset),
                                    **cell_statistics.interval_cells("brier", brier, ctx), "top_candidate_correct": top1_cells.pop("top1"), **top1_cells}, behind))
            for b in range(BINS):
                lo, hi = b / BINS, (b + 1) / BINS
                inside = [(hit, p) for _, _, hit, p in subset if lo <= p < hi or (b == BINS - 1 and p == 1.0)]
                if inside:
                    ctx["a12_curves"].append({"agent": agent, "model": model, "config": config, "ranking_model": ranking, "readings": readings,
                                              "bin_lo": lo, "bin_hi": hi, "predicted_mean": _r(sum(p for _, p in inside) / len(inside)),
                                              "observed_frequency": _r(sum(h for h, _ in inside) / len(inside)), "n": len(inside)})
    return rows


# rows Ag1 to Ag7 of the paper in order without the rule only switch and the rank choice, which left the experiments, and without
# the ablations of the expectation checks and of replanning, whose rows the experiments no longer play, the suffixes e10 and ec2 of CausalLoopOptions.label() keeping their names
SWITCH_ORDER = ("full", "branches_off", "single_hypothesis", "plan_length_1", "e10", "ec2")
# the per episode measures of A13_A17, deviations being the surprises cell of the causal agent under the paper's name
A13_METRICS = ("pass", "unnecessary", "llm_calls", "thinking_cycles", "deviations", "test_first", "cost", "test_cost", "level_accuracy",
               "repeated_repair_sets", "realized_elimination_per_read")
MISDIAGNOSIS_ABLATION = "L1"
# the measures of MISDIAGNOSIS_PRICE: Top-1 and pass, the shares of wrong diagnoses and of declarations, the cycles read before the
# first diagnosis, the cost without the misdiagnosis charge, on one scale for both presets, and the charge
MISDIAGNOSIS_MEASURES = ("diagnosis_all_correct", "pass", "diagnosis_wrong", "abstained", "cycles_read_before_diagnosis", "cost_without_charge",
                         "misdiagnosis_charge")


def cycles_read_before_diagnosis(log):
    """The cycles that the reads of an episode served before its first diagnosis, the sum of cycles over the read entries of the
    hidden log before the first submit_diagnosis entry, and None for an episode without a log or without a diagnosis."""
    first = next((i for i, e in enumerate(log or []) if e.get("event") == "submit_diagnosis"), None)
    return None if first is None else sum(int(e.get("cycles") or 0) for e in log[:first] if e.get("event") == "read")


def table_misdiagnosis_price(episodes, ctx):
    """One row per agent, model and preset of the ablation L1, the misdiagnosis price at the budget against the preset of training,
    with the summary of every measure of MISDIAGNOSIS_MEASURES. The charge is the one of the verdict of the episode
    (verifier.misdiagnosis_charge_of), and the cost without it puts the episodes of both presets on one scale. Descriptive: no
    pair, no test and no family."""
    rows = []
    for (agent, model, config, surface, server), eps in cell_statistics.groups(episode_selection.arm_episodes(episodes, (MISDIAGNOSIS_ABLATION,)), "agent", "model", "config", "surface",
                                                                "diagnostics_server").items():
        scored = []
        for e in eps:
            verdict, cost = episode_files._verdict(e), cell_statistics.value(e, "cost")
            charge = None if verdict is None else verifier.misdiagnosis_charge_of(verdict)
            scored.append({**e, "misdiagnosis_charge": charge, "cost_without_charge": None if None in (cost, charge) else cost - charge,
                           "cycles_read_before_diagnosis": cycles_read_before_diagnosis(episode_files._hidden_log(e))})
        row = {"agent": agent, "model": model, "config": config, "n": len(scored)}
        for m in MISDIAGNOSIS_MEASURES:
            row.update(cell_statistics.summary(scored, m, ctx))
        row.update({"surface": surface, "diagnostics_server": server})
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


def table_a13_a17(episodes, ctx):
    """One row per switch, model, config and tool surface of the causal agent with the CausalDiagnosis MCP over the episodes at the
    main level of every setting (at_main_levels), so that a run of an Environment ablation never pools into the full row and the
    agent without the MCP, whose label is that of the full agent, stays out of it: the summary intervals of A13_METRICS, the paired
    differences of pass and model calls against the full agent with the agent label left out of the pairing key (the labels differ
    between rows), empty on the full row, and Holm over the pass differences."""
    rows = []
    main = [e for e in episode_selection.causal_episodes(episodes) if episode_selection.at_main_levels(e) and as_bool(e.get("diagnostics_server")) is True]
    for (model, config, surface), eps in cell_statistics.groups(main, "model", "config", "surface").items():
        levels = {}
        for e in eps:
            levels.setdefault(episode_selection.switch_of(e), []).append(e)
        full = levels.get("full", [])
        for switch in sorted(levels, key=lambda s: (SWITCH_ORDER.index(s) if s in SWITCH_ORDER else len(SWITCH_ORDER), s)):
            level = levels[switch]
            row = {"agent": level[0].get("agent", ""), "switch": switch, "model": model, "config": config, "surface": surface,
                   "n": len(level)}
            for m in A13_METRICS:
                row.update(cell_statistics.summary(level, m, ctx))
            row["status"] = (cell_statistics.row_status(level, ctx) or cell_statistics.row_status(full, ctx)) if switch != "full" and full else ""
            if switch != "full" and full:
                row.update(cell_statistics._paired_cells([] if row["status"] else level, [] if row["status"] else full, "agent", ("pass", "llm_calls"), ctx))
            else:
                row.update(cell_statistics._no_pairing(("pass", "llm_calls")))
            rows.append(cell_statistics.trace(ctx, row, level, full if switch != "full" and full else None))
    return family_correction.add_holm(rows, "pass", ctx["alpha"])


# the three tables of one factor against its reference level: the price table (En6), the guidance level (En7) and the options of
# layers 2, 3 and 5 against the preset of training
A9_COLUMNS, table_a9 = factor_table("prices", "default", ["cost", "cost_normalized", "pass"], ["cost_normalized", "pass"],
                                    by=("agent", "model", "diagnostics_server"), select=lambda episodes: episode_selection.placed_episodes(episodes, "En6", "prices"))
A10_COLUMNS, table_a10 = factor_table("guidance_level", "2", ["pass"], ["pass"], by=("agent", "model", "diagnostics_server"),
                                      select=lambda episodes: episode_selection.placed_episodes(episodes, "En7", "guidance_level"))
SWITCHES_COLUMNS, table_switches = factor_table("config", TRAINING_PRESET, ["pass", "diagnosis_all_correct", "cost_normalized", "detection_delay", "repair_delay"],
                                                ["pass", "diagnosis_all_correct"], by=("agent", "model", "diagnostics_server"), select=episode_selection.switch_episodes,
                                                unmatched=("clock_mode",))
