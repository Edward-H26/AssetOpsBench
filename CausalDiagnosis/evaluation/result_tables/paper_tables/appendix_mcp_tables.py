from CausalDiagnosis.utils.csv_rows import as_float
from CausalDiagnosis.experiments.table_definitions.experiment_tables import MCP, AGENTS_PRESET, TRAINING_PRESET, TRAINING_ROWS, row_label
from . import table_cells, thinking_tables


OFFLINE_PREHISTORY = 5
OFFLINE_READINGS = (0, 1, 2, 3, 5, 10)
# the columns of the offline table, each head as the lines the LaTeX form stacks, a prediction of the MCP named as one
OFFLINE_CELLS = ((("Predicted top-1", "probability"), "top_mass"), (("Top-1 rate",), "top1_rate"), (("Exact wrong",), "exact_wrong"),
                 (("E14 prediction interval", "covers the truth"), "e14_covers_truth"), (("E17 rejects", "a wrong repair"), "e17_rejects_wrong_repair"))
OFFLINE_COUNTS = ("exact_wrong",)


def share(row, field):
    """A share of offline_mcp.csv to two digits or a count of it as an integer, or the word for no run or no value."""
    if row is None:
        return table_cells.plain(table_cells.NO_RUN)
    value = as_float(row.get(field))
    return table_cells.plain(table_cells.NO_VALUE if value is None else f"{value:.0f}" if field in OFFLINE_COUNTS else f"{value:.2f}")


def offline_table(sheet):
    def cells(readings):
        row = sheet.cite("mcpoffline", str(readings), "offline_mcp", sheet.first("OFFLINE", prehistory=OFFLINE_PREHISTORY, readings=readings))
        out = [share(row, field) for _, field in OFFLINE_CELLS]
        if row:  # the page names the episodes behind the rates and the wrong repairs behind the last one, which has none without them
            out[1] = table_cells.Cell(out[1].short, f"{out[1].full} of {row['episodes']} episodes")
            out[4] = out[4] if out[4].full == table_cells.NO_VALUE else table_cells.Cell(out[4].short, f"{out[4].full} of {row['wrong_repairs']} wrong repairs")
        return out

    return table_cells.MainTable("mcpoffline", f"The {MCP} offline",
                     "No model and no agent, on the episodes of the test set with the five recordings that precede the initial state served before the live readings. "
                     "Every "
                     "initial state is scored by fits that leave out its block, the predicted top-1 probability is the mean probability that the CausalDiagnosis "
                     "MCP gives its first "
                     "candidate, printed beside the top-1 rate, "
                     "exact wrong counts the episodes whose first candidate is not the true combination, the fifth column is the share of the released summaries that "
                     "lie inside the E14 prediction interval of the true combination, and the last column is the share of the wrong repairs whose first reading after "
                     "the repair E17 finds inconsistent.", ["Live readings", *(" ".join(head) for head, _ in OFFLINE_CELLS)], "lccccc",
                     [[sheet.counted("mcpoffline", str(readings), [str(readings)], cells(readings)) for readings in OFFLINE_READINGS]],
                     tex_header=["Live readings & " + " & ".join(table_cells.stacked(*head) if len(head) > 1 else head[0] for head, _ in OFFLINE_CELLS) + r" \\"])


# the appendix table of the CausalDiagnosis MCP: the measures of metrics.md for the agents with the MCP, with all six layers on and
# on layer 0, each cell a cell of the aggregate table MCP, as (head, field)
MCP_SETTINGS = ((TRAINING_PRESET, "All six big layers"), (AGENTS_PRESET, "Layer 0"))
MCP_CELLS = ((("Uptake",), "uptake"), (("Harm",), "harm"), (("Recovery rate prediction", "Brier skill"), "recovery_bss"),
             (("Cost prediction", "bias"), "cost_bias"), (("Cost prediction", "absolute error"), "cost_mae"), (("Cost prediction", "skill over price"), "cost_skill"))
MCP_THINKING = (f"{table_cells.DELTA}Production loss", "while thinking")


def mcp_table(sheet):
    def cells(line, label, preset):
        row = sheet.read("mcpvalue", line, "MCP", label, True, preset)
        if row is None:
            return [table_cells.plain(table_cells.NO_RUN)] * (len(MCP_CELLS) + 1)
        # a cost cell of a row that prices nothing is empty, so estimate prints the word for no value, as the thinking column does
        # on a preset whose clock charges no thinking
        thinking = table_cells.difference(row, "think_loss") if thinking_tables.charges_thinking(preset) else table_cells.plain(table_cells.NO_VALUE)
        return [*(table_cells.estimate(row, field, ".2f") for _, field in MCP_CELLS), thinking]

    rows = [(name, row_label((prefix, server), sheet.model)) for prefix, server, name in TRAINING_ROWS]
    groups = [[sheet.counted("mcpvalue", f"{setting}, {name}", [setting, name], cells(f"{setting}, {name}", label, preset)) for name, label in rows]
              for preset, setting in MCP_SETTINGS]
    heads = [*(" ".join(head) for head, _ in MCP_CELLS), " ".join(MCP_THINKING)]
    return table_cells.MainTable("mcpvalue", f"What the {MCP} adds and risks",
                     f"Descriptive measures of the {MCP} for every agent that has it, on the preset with all six big layers and on layer 0. Each cell "
                     "pools the visits or the plans of the episodes of its row, with intervals from a bootstrap that resamples the test combinations. "
                     "Uptake is the share of the visits after a ranking of the CausalDiagnosis MCP that order the candidate plan of the first candidate "
                     "of the last ranking "
                     "before them. Harm is the share of those visits whose plan differs from the faulty set of the hidden combination at the visit. The "
                     "Brier skill of the recovery rate prediction is taken against a predictor at the base rate and scored at the reading that verifies "
                     "the plan. The error of the cost prediction is the predicted cost of acting now minus the cost of the episode from the first order of "
                     "the plan until the asset is restored, divided by the cost of the oracle on the same seed. Its skill is taken against a prediction of "
                     "the price and the downtime of the plan alone. The last column is the paired difference of the production loss in the cycles charged "
                     "to thinking, the agent with the CausalDiagnosis MCP minus the same agent without it on the same episodes. Layer 0 prices nothing and charges no "
                     "thinking time, so its cost columns and its last column print none.",
                     ["Setting", "Agent", *heads], "ll" + "c" * len(heads), groups,
                     tex_header=["Setting & Agent & " + " & ".join(table_cells.stacked(*head) if len(head) > 1 else head[0] for head in (*(h for h, _ in MCP_CELLS), MCP_THINKING))
                                 + r" \\"], first_only=True)
