from CausalDiagnosis.utils.csv_rows import as_bool  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AGENTS_PRESET, AGENT_ROWS  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import cluster_bootstrap, kappa_of_counts, paired_cluster_bootstrap, sign_flip_p  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import cell_statistics, family_correction, episode_selection, hypothesis_tables


ALL_FALSE = {**{criterion: 0.0 for criterion in cell_statistics.JUDGE_CRITERIA}, **{f"{criterion}_majority": False for criterion in cell_statistics.JUDGE_CRITERIA}}  # an episode without a judge record
# the constructs of the verifier that the majority verdict of the judge is set against in the table of the judge
JUDGE_CONSTRUCTS = ("pass", "diagnosis_all_correct", "faulty_set_right", "restored_clean")


def judge_agreements(episodes, ctx):
    """Per construct of JUDGE_CONSTRUCTS Cohen's kappa between the judge's majority pass under the protocol of the paper and the
    construct, with its interval from a bootstrap clustered by start, and the four counts of the 2 by 2 table, over the episodes
    that carry both. No test reads them: the tests of the judge are those of h0."""
    cells = {}
    for construct in JUDGE_CONSTRUCTS:
        both = [e for e in episodes if e.get("judge_record_protocol") == cell_statistics.PAPER_PROTOCOL and as_bool(e.get("judge_pass")) is not None
                and cell_statistics.value(e, construct) is not None]
        ratings = [(bool(as_bool(e["judge_pass"])), cell_statistics.value(e, construct) == 1.0) for e in both]
        counts = [[j and v, j and not v, v and not j, not j and not v] for j, v in ratings]
        boot = cluster_bootstrap(counts, [cell_statistics.cluster_of(e) for e in both], kappa_of_counts, ctx["resamples"], ctx["seed"])
        cells.update({**cell_statistics.interval_cells(f"kappa_{construct}", boot, ctx),
                      **{f"{construct}_{name}": sum(1 for c in counts if c[i]) for i, name in enumerate(("both", "judge_only", "verifier_only", "neither"))}})
    return cells


def judge_kappa(episodes, ctx):
    """Cohen's kappa between the judge's majority pass under the five call protocol and the verified pass, over the episodes that
    carry both, with an interval from a bootstrap clustered by block, so that the scores of the judge print beside how far the judge
    agrees with the verifier. Empty when either rule puts every such episode on one side, as label_kappa is; judge_kappa_n counts them."""
    both = [e for e in episodes if e.get("judge_record_protocol") == cell_statistics.PAPER_PROTOCOL and as_bool(e.get("judge_pass")) is not None
            and as_bool(e.get("pass")) is not None]
    ratings = [(as_bool(e["judge_pass"]), as_bool(e["pass"])) for e in both]
    counts = [[j and v, j and not v, v and not j, not j and not v] for j, v in ratings]  # the columns of stats.kappa_of_counts
    boot = cluster_bootstrap(counts, [cell_statistics.cluster_of(e) for e in both], kappa_of_counts, ctx["resamples"], ctx["seed"])
    return {**cell_statistics.interval_cells("judge_kappa", boot, ctx), "judge_kappa_n": len(both)}


def _judged_scores(episodes):
    """The episodes that an H0 test reads, each a copy with J, the judge's majority under the registered protocol, in h0_judge and V,
    the verified pass, in h0_verified, as numbers. An episode without a judge record reads J = 0, a failure of the judge under the
    registered counting.unjudged, as the tables of the judge count it (without_record). An episode judged under another protocol
    has no J and stays out, and h0_status leaves its test unrun."""
    scored = []
    for e in episodes:
        protocol = e.get("judge_record_protocol")
        judged = cell_statistics.value(e, "judge_pass") if protocol == cell_statistics.PAPER_PROTOCOL else None if protocol else 0.0
        verified = cell_statistics.value(e, "pass")
        if judged is not None and verified is not None:
            scored.append({**e, "h0_judge": judged, "h0_verified": verified})
    return scored


def h0_status(rows, scored, ctx):
    """Why an H0 test over the episodes of its rows is not run, empty when it runs: the status of a row that misses a planned episode
    or that the unscored limit refuses (row_status), as for the tests of the agents, an episode judged under another protocol, which
    the registered protocol gives no J, an episode without the pass of the judge or of the verifier, or no judge record of the
    registered protocol in any episode, so that no judge job has scored the rows. scored are the episodes that _judged_scores reads."""
    for eps in rows:
        status = cell_statistics.row_status(eps, ctx)
        if status:
            return status
    episodes = [e for eps in rows for e in eps]
    if any(e.get("judge_record_protocol") not in (None, "", cell_statistics.PAPER_PROTOCOL) for e in episodes):
        return "unrun, judged under another protocol"
    if len(scored) < len(episodes):
        return "unrun, an episode without the pass of the judge or of the verifier"
    if not any(e.get("judge_record_protocol") == cell_statistics.PAPER_PROTOCOL for e in episodes):
        return "unrun, no judge record"
    return ""


def _nonzero_starts(values, blocks):
    totals = {}
    for v, block in zip(values, blocks):
        totals[block] = totals.get(block, 0.0) + v
    return sum(1 for total in totals.values() if total != 0)


def _h0_statistic(values, blocks, ctx):
    """The mean of the per episode (or per pair) values of an H0 test with its block bootstrap interval, the sign flip p value, exact
    since the values are differences of zero or one outcomes, and the number of starts whose sum is not zero."""
    boot = paired_cluster_bootstrap(values, [0.0] * len(values), blocks, ctx["resamples"], ctx["seed"])
    return {**cell_statistics.interval_cells("statistic", boot, ctx, "difference"),
            "h0_p": _r(sign_flip_p(values, blocks, ctx["resamples"], ctx["seed"], whole=True), 6), "n": len(values), "clusters": boot["clusters"],
            "nonzero_starts": _nonzero_starts(values, blocks)}


def h0_row_cells(eps, ctx):
    """The row test of H0 on the episodes of one row: D = mean(J - V), and beside it kappa over the episodes with a judge record of the
    registered protocol (judge_kappa), the discordance (a + b) / n, the false pass rate b / #(V = 0) and the false fail rate a / #(V = 1),
    with b the episodes the judge passes and the verifier fails and a the reverse, J and V as _judged_scores reads them. status says
    why the test is not run (h0_status), empty when it runs."""
    scored = _judged_scores(eps)
    cells = {"status": h0_status([eps], scored, ctx), **judge_kappa(eps, ctx)}
    if cells["status"]:
        return {**cells, "n": 0, "h0_p": ""}
    j, v = [e["h0_judge"] for e in scored], [e["h0_verified"] for e in scored]
    b = sum(1 for x, y in zip(j, v) if x > y)
    a = sum(1 for x, y in zip(j, v) if x < y)
    failed, passed = v.count(0.0), v.count(1.0)
    return {**cells, **_h0_statistic([x - y for x, y in zip(j, v)], [cell_statistics.cluster_of(e) for e in scored], ctx),
            "discordance": _r((a + b) / len(j)), "false_pass": _r(b / failed) if failed else "", "false_fail": _r(a / passed) if passed else ""}


def h0_comparison_cells(level, against, ctx):
    """The comparison test of H0 on two rows: theta = mean[(J_a - J_b) - (V_a - V_b)] over the episodes matched as in AGENTS_pairs, with
    the judge's and the verified paired differences beside it and reversed, the two of opposite sign, J and V as _judged_scores reads
    them. status says why the test is not run (h0_status), empty when it runs."""
    scored_level, scored_against = _judged_scores(level), _judged_scores(against)
    status = h0_status([level, against], scored_level + scored_against, ctx)
    pairs = [] if status else cell_statistics.matched(scored_level, scored_against, episode_selection.AGENT_KEY, "h0_judge")
    if not pairs:
        return {"status": status, "n": 0, "h0_p": ""}
    judged = [x["h0_judge"] - y["h0_judge"] for x, y in pairs]
    verified = [x["h0_verified"] - y["h0_verified"] for x, y in pairs]
    judge_diff, verified_diff = sum(judged) / len(pairs), sum(verified) / len(pairs)
    return {"status": status, **_h0_statistic([jd - vd for jd, vd in zip(judged, verified)], [cell_statistics.cluster_of(x) for x, _ in pairs], ctx),
            "judge_diff": _r(judge_diff), "verified_diff": _r(verified_diff), "reversed": judge_diff * verified_diff < 0}


def table_h0(episodes, ctx):
    """H0: per model and settings of PAIR_SETTINGS on the preset of the main table (design.AGENTS_PRESET), a row test for each of
    the six rows and a comparison test for each comparison of design.HYPOTHESES, in one family. An episode without a judge record
    counts as a failure of the judge (_judged_scores). A test that h0_status leaves unrun, on a row that misses a planned episode or is
    refused, on an episode judged under another protocol or without any judge record, counts with p = 1 in the family, so the family
    keeps its size; every test that runs enters with its computed p value, as counting.holm of the registered file states, so Holm's
    step-down over the family decides each test (h0_reject), and the sign of the statistic says whether the judge overstates or
    understates the verified pass."""
    rows = []
    for (model, config, *setting), eps in episode_selection.main_level_groups(episodes).items():
        if config != AGENTS_PRESET:
            continue
        family = []
        for (label, server), (_, _, name) in zip(hypothesis_tables.agent_rows(model), AGENT_ROWS):
            arm = episode_selection._arm(eps, label, server)
            if arm:
                row = {"test": "row", "name": name, **hypothesis_tables._comparison_row("", label, server, "", "", model, config, setting)}
                family.append(cell_statistics.trace(ctx, {**row, **h0_row_cells(arm, ctx)}, arm))
        for row, level, against in hypothesis_tables.comparison_arms(eps, hypothesis_tables.hypothesis_comparisons(model), model, config, setting):
            family.append(cell_statistics.trace(ctx, {"test": "comparison", "name": row["hypothesis"], **row, **h0_comparison_cells(level, against, ctx)}, level, against))
        # the registered family, unrun tests included, each run test with its computed p value
        size = family_correction.registered_size(ctx, "h0")
        for row in family:
            row.pop("hypothesis", None)
            status = row.pop("status")
            row["status"] = (status or "unrun, no episode") if row["h0_p"] == "" else "tested"
            if row["status"] != "tested":
                row["h0_p"] = 1.0
        rows += family_correction.holm_over(family, ("h0",), ctx["alpha"], size)
    return rows


# ----- the metrics of the AssetOpsBench paper (JUDGE, JUDGE_AGREEMENT, EFFICIENCY, PLANS) -----
def judged_by_protocol(episodes):
    """{protocol: the episodes with a judge record under it} in sorted order."""
    return {protocol: judged for (protocol,), judged in cell_statistics.groups([e for e in episodes if e.get("judge_record_protocol")], "judge_record_protocol").items()}


def judged_by_judge(episodes):
    """{(protocol, judge model): the episodes with a judge record of that protocol and model} in sorted order."""
    return cell_statistics.groups([e for e in episodes if e.get("judge_record_protocol")], "judge_record_protocol", "judge_record_model")


def passed_unrecovered(episode):
    """Whether the judge's majority passes a judged episode while the asset is not recovered at its end, an episode without a
    verdict counting as not recovered, as the counting rule reads it; None without the judge's pass."""
    judged = as_bool(episode.get("judge_pass"))
    return None if judged is None else judged and cell_statistics.value(episode, "recovered") == 0.0


def without_record(episodes):
    """The episodes without a judge record, false on every criterion, as the reproduction counts a run without a record."""
    return [{**e, **ALL_FALSE} for e in episodes if not e.get("judge_record_protocol")]


def table_judge(episodes, ctx):
    """One row per CONDITION of the six rows, protocol and judge model of the judge records of its episodes, so that no row
    pools two judges, the metrics of Figure 5 of
    AssetOpsBench: per criterion of JUDGE_CRITERIA the mean over the episodes of the share of the planned calls that answered
    true (judge_fields), with a bootstrap interval clustered by block and the standard deviation over episodes, and the share
    of the episodes whose majority of the calls answered true, with such an interval, which is what that paper prints. An
    episode without a record counts as false on every criterion, as the reproduction counts a run without a record, and the
    row names how many, while a condition without a record under any protocol has no row. An episode judged under another
    protocol stays out, which n, the episodes of the condition, shows against the _n cells. The row counts the declaring
    episodes of the condition (declared), which the judge scores as failures, so that the pages tell them apart. Two shares over the
    judged episodes describe the judge: the records cut to the context of the served judge (judge_record_cut), and the episodes that
    the judge passes while the asset is not recovered (judge_passed_unrecovered)."""
    rows = []
    for condition, eps in cell_statistics.groups(episode_selection.agent_episodes(episodes, every_clock_mode=True), *episode_selection.CONDITION).items():
        unjudged = without_record(eps)
        for (protocol, model), judged in judged_by_judge(eps).items():
            scored = judged + unjudged
            row = {**dict(zip(episode_selection.CONDITION, condition)), "judge_protocol": protocol, "judge_model": model,
                   "n": len(eps), "failed_calls": sum(e["judge_failed_calls"] for e in judged), "episodes_without_record": len(unjudged),
                   "declared": sum(1 for e in eps if as_bool(e.get("abstained")))}
            for criterion in cell_statistics.JUDGE_CRITERIA:
                row.update({**cell_statistics.summary(scored, criterion, ctx), **cell_statistics.spread(scored, criterion), **cell_statistics.summary(scored, f"{criterion}_majority", ctx)})
            row.update({**cell_statistics.summary(judged, "judge_record_cut", ctx),
                        **cell_statistics.summary([{**e, "judge_passed_unrecovered": passed_unrecovered(e)} for e in judged], "judge_passed_unrecovered", ctx)})
            if protocol == cell_statistics.PAPER_PROTOCOL:
                row.update(judge_agreements(scored, ctx))
            rows.append(cell_statistics.trace(ctx, row, scored))
    return rows


def table_judge_agreement(episodes, ctx):
    """The metrics of Table 14 of AssetOpsBench over the six rows together, one row per model, config, settings of
    PAIR_SETTINGS and protocol with more than one call: per criterion the mean pairwise agreement of the parsed calls of an
    episode over the episodes, with a bootstrap interval clustered by block, and the uncertainty, one minus it."""
    rows = []
    for (model, config, *setting), eps in episode_selection.main_level_groups(episodes).items():
        for protocol, judged in judged_by_protocol(cell_statistics.with_metric(eps, f"{cell_statistics.JUDGE_CRITERIA[0]}_agreement")).items():
            row = {"model": model, "config": config, **dict(zip(episode_selection.PAIR_SETTINGS, setting)), "judge_protocol": protocol, "n": len(judged)}
            for criterion in cell_statistics.JUDGE_CRITERIA:
                row.update(cell_statistics.summary(judged, f"{criterion}_agreement", ctx))
                row[f"{criterion}_uncertainty"] = _r(1 - row[f"{criterion}_agreement"])
            rows.append(cell_statistics.trace(ctx, row, judged))
    return rows
