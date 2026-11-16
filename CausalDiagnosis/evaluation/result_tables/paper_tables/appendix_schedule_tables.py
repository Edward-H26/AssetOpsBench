from CausalDiagnosis.utils.csv_rows import as_bool, as_float
from CausalDiagnosis.experiments.table_definitions.experiment_tables import MCP, PARALLEL_CONTRAST_ROWS, row_label
from . import table_cells


# the columns of the two rows of the schedule of candidate plans as (head lines, column of PARALLEL, format), and the names of the two rows
PARALLEL_CELLS = ((("Repair,", "declared"), "repair_declared", ".1f"), (("Repair with", "thinking"), "repair_delay", ".1f"),
                  (("Work", "orders"), "work_orders", ".2f"), (("Downtime",), "downtime_total", ".1f"), (("Pass",), "pass", ".2f"),
                  (("Cost /", "oracle"), "cost_normalized", ".2f"), (("Model", "calls"), "llm_calls", ".1f"), (("Ranking", "calls"), "ranking_calls", ".1f"))
SWITCH_NAMES = {"on": "Schedule of candidate plans on", "off": "Schedule of candidate plans off"}
SCHEDULE = "the schedule of candidate plans"  # the switch of ablation Ag9, named by what it does and by no coined name


def parallel_table(sheet):
    def cells(row):
        if not row:
            return [table_cells.plain(table_cells.NO_RUN)] * (len(PARALLEL_CELLS) + 2)
        advised = int(as_float(row.get("advised_batches")) or 0)
        kept = table_cells.plain(f"{int(as_float(row['kept_batches']))} of {advised}" if advised else table_cells.NO_VALUE)
        gap = table_cells.number(row, "gap_explained") if row.get("switch") == "" else table_cells.plain("")
        return [*(table_cells.level_over_se(row, metric, spec) for _, metric, spec in PARALLEL_CELLS), kept, gap]

    pair = [sheet.counted("parallel", name, [name], cells(sheet.cite("parallel", name, "PARALLEL",
                                                                    sheet.first("PARALLEL", model=sheet.model, switch=switch))))
            for switch, name in SWITCH_NAMES.items()]
    designs = [sheet.counted("parallel", row[2], [row[2]], cells(sheet.cite("parallel", row[2], "PARALLEL", sheet.first("PARALLEL", model=sheet.model, switch="",
                                                                                                                                   agent=row_label(row, sheet.model)))))
               for row in PARALLEL_CONTRAST_ROWS]
    return table_cells.MainTable("parallel", "The schedule of candidate plans",
                     f"The CausalDiagnosis agent with the {MCP} and all six big layers on, with {SCHEDULE} on, the agent as released, and off, and "
                     "under them the other designs with the CausalDiagnosis MCP on the same layer, each over its registered episodes. With the schedule on, "
                     "the agent carries out its candidate plans as one schedule, in which a read that several plans name runs once and consecutive reads "
                     "run as one batch with one ranking after the last, and with it off every read step runs alone with its own ranking. The columns are "
                     "the time to "
                     "repair in declared cycles, which leaves out the cycles charged for thinking, the time to repair with them, the work orders, the "
                     "downtime, Pass as a percentage, the cost over the cost of the oracle, the model calls and the calls of the ranking tool of the CausalDiagnosis "
                     "MCP, each the mean "
                     f"over the episodes and, with the sign {table_cells.PLUS_MINUS}, its standard error. Batches kept counts the batches of reads that the board advised to a plan the model "
                     "wrote and that the plan kept unchanged, out of all such batches. Gap explained is the share of the gap between a design and the agent "
                     "with the schedule on that the switch explains, computed on the measure that the next table tests as the difference off less "
                     "on over the difference of the design less on, and printed where the design is behind. Neither is a test, and the next table tests "
                     "the pair.",
                     ["", *(table_cells.level_head(" ".join(head), metric) for head, metric, _ in PARALLEL_CELLS), "Batches kept", "Gap explained"],
                     "l" + "c" * (len(PARALLEL_CELLS) + 2),
                     [pair, designs],
                     tex_header=[" & " + " & ".join([*(table_cells.stacked(*head) if len(head) > 1 else head[0] for head, _, _ in PARALLEL_CELLS),
                                                     table_cells.stacked("Batches", "kept"), table_cells.stacked("Gap", "explained")]) + r" \\"])


def parallel_pair_table(sheet):
    row = sheet.cite("parallelpair", "pair", "PARALLEL_pairs", sheet.first("PARALLEL_pairs", model=sheet.model))

    def margin(metric):
        if row is None:
            return table_cells.plain(table_cells.NO_RUN)
        bound, met = as_float(row.get(f"margin_{metric}")), as_bool(row.get(f"margin_{metric}_met"))
        return table_cells.plain(table_cells.NO_VALUE if bound is None else f"{bound:.2f}, {table_cells.NO_VALUE if met is None else 'met' if met else 'not met'}")

    def split(kind):
        if row is None:
            return table_cells.plain(table_cells.NO_RUN)
        mean = as_float(row.get(f"{kind}_diff"))
        return table_cells.plain(f"{row[f'{kind}_pairs']} pairs" + ("" if mean is None else f", mean {table_cells.DELTA} {mean:+.2f}"))

    pairs = table_cells.plain(table_cells.NO_RUN) if row is None else table_cells.plain(f"{row['pairs']} in {row['clusters']} test combinations")
    lines = [(["Pairs"], [pairs]), ([f"{table_cells.DELTA}Repair, declared cycles"], [table_cells.difference(row, "repair_declared")]), (["Exact p"], [table_cells.holm_cell(row, "repair_declared_p")]),
             (["Holm p"], [table_cells.holm_cell(row, "repair_declared_holm_p")]), (["Reading"], [table_cells.word(row, "reading")]), ([f"{table_cells.DELTA}Pass"], [table_cells.difference(row, "pass")]),
             (["Margin on Pass"], [margin("pass")]), ([f"{table_cells.DELTA}Cost / oracle"], [table_cells.difference(row, "cost_normalized")]), (["Margin on cost"], [margin("cost")]),
             (["Pairs without a cost"], [table_cells.number(row, "cost_pairs_without_cost", ".0f")]), (["Holds"], [table_cells.flag(row, "holds")]),
             (["With a refuted repair"], [split("refuted")]), (["Without a refuted repair"], [split("unrefuted")])]
    lines = [sheet.counted("parallelpair", "pair", labels, cells) for labels, cells in lines]
    return table_cells.MainTable("parallelpair", "The schedule of candidate plans, on against off",
                     f"{SCHEDULE[0].upper()}{SCHEDULE[1:]} on less off on the same seeds: the paired difference of the time to repair in declared cycles "
                     "with its 95 percent interval, its exact p value and its Holm decision over the registered size of its family, the paired differences "
                     "of Pass and of the cost over the oracle with their 95 percent intervals beside their registered margins, the pairs left out of the "
                     "second for an episode without a cost, and whether the comparison holds. The last two lines describe and test nothing: the mean "
                     "paired difference of the time to repair over the pairs in which an episode met a refuted repair, the only place the mechanism acts, "
                     "and over the pairs in which neither did.",
                     ["", "On less off"], "lc", [lines])
