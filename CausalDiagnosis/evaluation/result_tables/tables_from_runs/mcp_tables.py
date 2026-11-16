from collections import Counter
from CausalDiagnosis.utils.csv_rows import as_bool  # noqa: E402
from CausalDiagnosis.evaluation.metrics import mcp_metrics  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AGENTS_PRESET  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import cohens_kappa, spearman  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import episode_files, cell_statistics, episode_selection, hypothesis_tables


ELIM_METRICS = ("realized_elimination_per_read", "realized_elimination_per_cost", "expected_elimination", "realized_elimination_calibrated",
                "calibration_gap")
ELIM_PAIRED = ("realized_elimination_per_cost", "realized_elimination_per_read")  # the first gives the pairs and clusters


def table_elim(episodes, ctx):
    """One row per CONDITION of the six rows and of the reference policies: the episodes, their paid reads and the
    reads with a forecast of the belief they met, the digest of the models.json the replay read, and the summary
    intervals of ELIM_METRICS over the episodes with a paid read, or with such a forecast for the calibration columns."""
    rows = []
    for condition, eps in cell_statistics.groups(episode_selection.elimination_episodes(episodes), *episode_selection.CONDITION).items():
        row = {**dict(zip(episode_selection.CONDITION, condition)), "n": len(eps), "paid_reads": sum(int(e.get("paid_reads") or 0) for e in eps),
               "calibrated_reads": sum(int(e.get("calibrated_reads") or 0) for e in eps),
               "yardstick": " ".join(sorted({e["yardstick"] for e in eps if e.get("yardstick")}))}
        for m in ELIM_METRICS:
            row.update(cell_statistics.summary(eps, m, ctx))
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


def table_elim_pairs(episodes, ctx):
    """The comparisons of the hypotheses of design.HYPOTHESES, paired as in AGENTS_pairs on its preset over the episodes at the main
    level of every setting, on the realized elimination per unit of cost and per read, which no Holm decision covers since the paper
    reports the measure beside the hypotheses. A comparison of an incomplete or refused row prints the status and no cells."""
    rows = []
    for (model, config, *setting), eps in episode_selection.main_level_groups(episodes).items():
        if config != AGENTS_PRESET:
            continue
        for row, level, against in hypothesis_tables.comparison_arms(eps, hypothesis_tables.hypothesis_comparisons(model), model, config, setting):
            row["status"] = cell_statistics.row_status(level, ctx) or cell_statistics.row_status(against, ctx)
            row.update(cell_statistics._paired_cells([] if row["status"] else level, [] if row["status"] else against, episode_selection.AGENT_KEY, ELIM_PAIRED, ctx))
            rows.append(cell_statistics.trace(ctx, row, level, against))
    return rows


# ----- the measures of the CausalDiagnosis MCP (MCP) -----
MCP_KEY = ("model", "config", "clock_mode", "surface", *episode_selection.SETTINGS)  # within a group, an agent with the MCP pairs against itself without it
MCP_COLUMNS = ([*episode_selection.CONDITION, "n", "unaligned", "unplaced", "yardstick"] + [f"{c}{part}" for c in mcp_metrics.CELLS for part in ("", "_low", "_high", "_n")]
               + ["think_loss", "think_loss_low", "think_loss_high", "think_loss_n", "think_loss_diff", "think_loss_diff_low",
                  "think_loss_diff_high", "pairs", "clusters"])


def table_mcp(episodes, ctx):
    """One row per CONDITION of an agent with the CausalDiagnosis MCP: the cells of mcp_metrics.row_cells with intervals from a
    bootstrap that resamples the starts, and the production lost while thinking with its paired difference against the same
    agent without the MCP in the same condition, an interval and no test, since the measures are descriptive."""
    rows = []
    for (model, config, clock, surface, *settings), eps in cell_statistics.groups([e for e in episodes if "mcp" in e], *MCP_KEY).items():
        for (agent, server), level in cell_statistics.groups(eps, "agent", "diagnostics_server").items():
            if as_bool(server) is not True:
                continue
            row = {"agent": agent, "model": model, "config": config, "diagnostics_server": server, "clock_mode": clock, "surface": surface,
                   **dict(zip(episode_selection.SETTINGS, settings)), "n": len(level),
                   **mcp_metrics.row_cells(level, ctx["resamples"], ctx["seed"], cell_statistics.cluster_of, cell_statistics.cluster_floor(ctx)),
                   **cell_statistics.summary(level, "think_loss", ctx), "yardstick": " ".join(sorted({e["mcp"]["yardstick"] for e in level if e["mcp"].get("yardstick")}))}
            without = episode_selection._arm(eps, agent, False)
            cells = cell_statistics.paired(level, without, "diagnostics_server", "think_loss", ctx) if without else cell_statistics._no_pairing(("think_loss",))
            cells.pop("think_loss_p", None)
            row.update(cells)
            rows.append(cell_statistics.trace(ctx, row, level, without or None))
    return rows


# ----- the two separability labels (LABELS) -----
LABEL_COLUMNS = ("rules_key", "ranking_model", "starts", "both_separable", "separable_label_only", "separable_e18_only", "both_inseparable",
                 "kappa", "share_spearman", "commit_share_spearman")


def label_kappa(ours, theirs):
    """Cohen's kappa of the two labels, None when one label puts every start on one side: a margin is empty there, and the
    formula reads 0 whatever the other label says, which would print as an agreement no better than chance."""
    return cohens_kappa(ours, theirs) if len(set(ours)) > 1 and len(set(theirs)) > 1 else None


def table_labels(episodes, ctx):
    """One row per label table (the first twelve hex digits of its rules key) and ranking model over the starts whose
    grade cell holds both labels, each start once however many episodes played it: the two by two table of the independent
    label against E18, both under the majority rule, Cohen's kappa where both labels use both values, and Spearman's correlation
    between the share of greedy rollouts that submit the start and the share of E18 rollouts whose first commitment names it,
    with the correlation against the share of E18 rollouts that ever commit in a column of its own."""
    starts = {}
    for e in episodes:
        record = episode_files.grade_record(e)
        if record and record.get("e18"):
            starts.setdefault((record["rules_key"][:12], e.get("ranking_model", "")), {}).setdefault(e.get("block", ""), (record, e))
    rows = []
    for (key, model), by_start in sorted(starts.items()):
        records = [record for record, _ in by_start.values()]
        ours, theirs = [r["separable"] for r in records], [r["e18"]["separable"] for r in records]
        cells = Counter(zip(ours, theirs))
        rows.append({"rules_key": key, "ranking_model": model, "starts": len(records), "both_separable": cells[(True, True)],
                     "separable_label_only": cells[(True, False)], "separable_e18_only": cells[(False, True)],
                     "both_inseparable": cells[(False, False)], "kappa": _r(label_kappa(ours, theirs)),
                     "share_spearman": _r(spearman([r["share"] for r in records], [r["e18"]["first_share"] for r in records])),
                     "commit_share_spearman": _r(spearman([r["share"] for r in records], [r["e18"]["commit_share"] for r in records]))})
        cell_statistics.trace(ctx, rows[-1], [e for _, e in by_start.values()])
    return rows
