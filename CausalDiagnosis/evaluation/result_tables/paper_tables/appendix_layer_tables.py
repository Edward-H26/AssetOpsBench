from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.utils.csv_rows import as_bool
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ABLATION_PLACES, ARM_PRESETS, FULL_ENVIRONMENT, LADDER, LADDER_NAMES, LADDER_ROWS, MCP, REFERENCE_POLICIES, row_label
from . import table_cells


POLICY_CELLS = (("Oracle", "oracle", "pass"), *((name, label, "reward") for label, name in REFERENCE_POLICIES))  # (policy name, label, metric)


def layers_table(sheet):
    pooled = []

    def cells(preset, setting):
        found = [sheet.cite("layers", setting, "LAYERS_pairs", sheet.first("LAYERS_pairs", model=sheet.model, config=preset, agent=agent))
                 for agent in [row_label(row, sheet.model) for row in LADDER_ROWS] + ["pooled"]]
        pooled.append(found[-1])
        return [*(table_cells.difference(row, "pass") for row in found), table_cells.holm_cell(found[-1], "holm_p")]

    lines = []
    for preset, setting in zip(LADDER[1:], LADDER_NAMES[1:]):
        values = cells(preset, setting)
        floor = " (layer below at floor)" if pooled[-1] and as_bool(pooled[-1].get("at_floor")) else ""
        lines.append(sheet.counted("layers", setting, [f"{setting}{floor}"], values))
    return table_cells.MainTable("layers", "What each big layer changes",
                     f"The paired difference of Pass between a layer and the layer below it for the three agents with the {MCP}, by test combination "
                     f"and entry offset{table_cells.pairs_clause(pooled)} for the pooled difference, with its 95 percent interval. A layer has an effect when the "
                     "difference pooled over the three agents differs from zero, Holm over the five layers. A test whose layer below has a pooled Pass that "
                     "lies below the floor of Pass is uninformative.",
                     ["Layer added", *(f"{table_cells.DELTA}Pass {name}" for _, _, name in LADDER_ROWS), f"pooled {table_cells.DELTA}Pass", "Holm p"], "lccccc", [lines],
                     tex_header=["Layer added & " + " & ".join([*(table_cells.stacked(f"{table_cells.DELTA}Pass", name) for _, _, name in LADDER_ROWS), table_cells.stacked("pooled", f"{table_cells.DELTA}Pass"), r"Holm p \\"])])


# the columns of the table of the ablation L1, the misdiagnosis price at the budget, in the form of the measures of the agents
MISDIAGNOSIS_COLUMNS = (
    table_cells.Column(("Top-1",), "diagnosis_all_correct", ".1f", "the share of episodes whose diagnosis is right for every component", 100),
    table_cells.Column(("Pass",), "pass", ".1f", "the share of episodes that pass", 100),
    table_cells.Column(("Wrong",), "diagnosis_wrong", ".1f", "the share of episodes whose submitted diagnosis is wrong", 100),
    table_cells.Column(("Declared",), "abstained", ".1f", "the share of episodes that declared insufficient evidence", 100),
    table_cells.Column(("Cycles read",), "cycles_read_before_diagnosis", ".1f", "the number of cycles the reads served before the first diagnosis"),
    table_cells.Column(("Cost without", "the charge"), "cost_without_charge", ".1f", "the cost of the episode without the misdiagnosis charge, so that both "
           "prices stand on one scale"),
    table_cells.Column(("Charge",), "misdiagnosis_charge", ".1f", "the misdiagnosis charge of the episode"),
)


def misdiagnosis_table(sheet):
    read = []
    place = ABLATION_PLACES[aggregate.MISDIAGNOSIS_ABLATION][0]
    # the price of the asset file on the place of L1 and the price at the budget on its one arm, each named by where the price comes from
    settings = ((place, "Asset file"), *((preset, "Budget") for preset, (ablation, _) in ARM_PRESETS.items() if ablation == aggregate.MISDIAGNOSIS_ABLATION))

    def cells(line, label, preset):
        found = sheet.first("MISDIAGNOSIS_PRICE", agent=label, model=sheet.model, config=preset)
        read.append(sheet.cite("misdiagnosis", line, "MISDIAGNOSIS_PRICE", found))
        if found is None:
            return [table_cells.plain(table_cells.NO_RUN)] * len(MISDIAGNOSIS_COLUMNS)
        return [table_cells.over_se(found, column.metric, column.spec, column.scale) for column in MISDIAGNOSIS_COLUMNS]

    groups = [[sheet.counted("misdiagnosis", f"{name}, {setting}", [name, setting], cells(f"{name}, {setting}", row_label((prefix, server), sheet.model), preset))
               for preset, setting in settings] for prefix, server, name in ABLATION_PLACES[aggregate.MISDIAGNOSIS_ABLATION][1]]
    names = {column.metric: " ".join(column.head) for column in MISDIAGNOSIS_COLUMNS}
    sentences = [f"Each design without the {MCP} on all six big layers, at the misdiagnosis price of the asset file and at a misdiagnosis price "
                 f"equal to the budget of the asset{sheet.played(read)}.",
                 *(f"{names[column.metric]} is {column.meaning}." for column in MISDIAGNOSIS_COLUMNS),
                 f"{table_cells.listed([names[column.metric] for column in MISDIAGNOSIS_COLUMNS if column.scale == 100])} are percentages.",
                 f"Every cell is the mean over the episodes of the line and, after the sign {table_cells.PLUS_MINUS}, its standard error from a bootstrap that "
                 "resamples test combinations. The table describes the ablation and runs no test."]
    return table_cells.MainTable("misdiagnosis", "The misdiagnosis price at the budget", " ".join(sentences),
                     ["Agent", "Misdiagnosis price", *(f"{names[column.metric]}{table_cells.PERCENT_UNIT if column.scale == 100 else ''}" for column in MISDIAGNOSIS_COLUMNS)],
                     "ll" + "c" * len(MISDIAGNOSIS_COLUMNS), groups,
                     tex_header=["Agent & Misdiagnosis price & " + " & ".join(table_cells.stacked(*column.head) if len(column.head) > 1 else column.head[0]
                                                                            for column in MISDIAGNOSIS_COLUMNS) + r" \\"],
                     first_only=True)


def full_environment_table(sheet):
    found = []

    def line(name, row):
        found.append(sheet.cite("fullenv", name, "FULL_ENV_pairs", sheet.first("FULL_ENV_pairs", test=name, model=sheet.model)))
        floor = " (base at floor)" if found[-1] and as_bool(found[-1].get("at_floor")) else ""
        return sheet.counted("fullenv", name, [name, f"{row[2]}{floor}"], [table_cells.difference(found[-1], "pass"), table_cells.holm_cell(found[-1], "pass_p"),
                                                                           table_cells.holm_cell(found[-1], "pass_holm_p"), table_cells.word(found[-1], "reading")])

    lines = [line(name, row) for name, row in FULL_ENVIRONMENT]
    return table_cells.MainTable("fullenv", "The full environment against the base layer",
                     f"For each agent with the {MCP}, the paired difference of Pass between its line without training in the training table, all six big "
                     "layers on, and its line of the agents table on layer 0, on the same test combinations, entry offsets and seeds"
                     f"{table_cells.pairs_clause(found)}, with its 95 percent interval, the exact two-sided sign-flip p value over test combinations, Holm over the "
                     "tests of this table and the reading, which gives the sign of a difference that holds. A test whose base line lies below the "
                     "floor of Pass is uninformative.",
                     ["", "Agent", f"{table_cells.DELTA}Pass", "p", "Holm p", "Reading"], r"l>{\raggedright\arraybackslash}p{6.4cm}cccc", [lines])


def policies_table(sheet):
    def cells(preset, setting):
        mode = aggregate.config_clock_mode(preset)
        out = []
        for name, label, metric in POLICY_CELLS:
            row = sheet.cite("policies", f"{setting}, {name}", "POLICIES", sheet.first("POLICIES", agent=label, config=preset, clock_mode=mode))
            out.append(table_cells.estimate(row, metric, *table_cells.level_form(metric)) if row else table_cells.plain(table_cells.NO_RUN))
        return out

    lines = [sheet.counted("policies", setting, [setting], cells(preset, setting)) for preset, setting in zip(LADDER, LADDER_NAMES)]
    return table_cells.MainTable("policies", "The reference policies on the layers",
                     "The policies need no model. The oracle knows the hidden state and must remain the best of the five reference policies, and doing nothing must score below every "
                     "policy that passes, on every initial state of every row. The oracle's pass is a percentage.",
                     ["Setting", *(table_cells.level_head(f"{name} {metric}", metric) for name, _, metric in POLICY_CELLS)],
                     "lcccccc", [lines], tex_header=["Setting & " + " & ".join(table_cells.stacked(name, metric) for name, _, metric in POLICY_CELLS) + r" \\"])


def policy_layers_table(sheet):
    def cells(label, name, preset, setting):
        row = sheet.cite("policylayers", f"{name}, {setting}", "POLICY_LAYERS", sheet.first("POLICY_LAYERS", agent=label, config=preset))
        return [table_cells.difference(row, metric) for metric in aggregate.POLICY_METRICS]

    groups = [[sheet.counted("policylayers", f"{name}, {setting}", [name, setting], cells(label, name, preset, setting))
               for preset, setting in zip(LADDER[1:], LADDER_NAMES[1:])] for label, name in REFERENCE_POLICIES]
    return table_cells.MainTable("policylayers", "What each big layer changes for the reference policies",
                     "The paired difference of Pass, success and reward between a layer and the layer below it for each reference policy, "
                     "matched on seed and every factor but the preset and the clock mode, with its 95 percent interval. The policies need no model, so "
                     "the effect of each single layer reads without the noise of an agent. The table tests nothing.",
                     ["Policy", "Layer added", *(f"{table_cells.DELTA}{metric.capitalize()}" for metric in aggregate.POLICY_METRICS)],
                     "ll" + "c" * len(aggregate.POLICY_METRICS), groups, first_only=True)
