from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.utils.csv_rows import as_bool, as_float
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AS_RELEASED, HYPOTHESES, ENVIRONMENT, MCP, AGENTS_PRESET, TRAINING_BASELINES, TRAINING_ROWS, TRAINING_VARIANTS, VARIANT_NAMES, row_label
from . import table_cells, latex_page


# the verified outcome of the agents table after its three judge columns, each a percentage: Top-1 and Pass, which decide H1 and H2,
# and pass^k at the k of PASS_K_SHOWN, every start state of a start combination until the metrics review names another k
PASS_K_SHOWN = aggregate.PASS_KS[-1]
AGENT_OUTCOME = (("Top-1", "diagnosis_all_correct"), ("Pass", "pass"), (f"Pass^{PASS_K_SHOWN}", f"pass_k{PASS_K_SHOWN}"))
JUDGE_KAPPA = f"Judge {table_cells.KAPPA}"  # the head of the agreement of the judge with the verifier, beside the metrics of the judge
LADDER_OUTCOME = (("Top-1", "diagnosis_all_correct"), ("Pass", "pass"), ("Cost / oracle", "cost_normalized"))
# Top-1 before Pass in every table, since a pass needs a correct diagnosis
MEASURES = (("Top-1", "diagnosis_all_correct"), ("Pass", "pass"))
SUBSETS = (("all", "all"), ("single", "single-fault"), ("two", "two-fault"))  # the subsets of TRAINING and TRAINING_pairs with their heads
# the columns of the table of the agents' measures, the measures beside the agents table, which the metrics review may change
AGENT_MEASURES = (
    table_cells.Column(("Exact", "match"), "faulty_set_right", ".1f", "the share of episodes whose diagnosis names exactly the faulty components", 100),
    table_cells.Column(("Success",), "success", ".1f", "the share of episodes that pass or declare insufficient evidence where the declaration earns its credit", 100),
    table_cells.Column(("Wrong",), "diagnosis_wrong", ".1f", "the share of episodes whose submitted diagnosis is wrong, so an episode that declared or wrote no "
           "diagnosis is not wrong", 100),
    table_cells.Column(("Declared",), "abstained", ".1f", "the share of episodes that declared insufficient evidence", 100),
    table_cells.Column(("Model", "calls"), "llm_calls", ".1f", "the number of calls of the model per episode"),
    table_cells.Column(("Cost /", "oracle"), "cost_normalized", ".2f", "the cost of the episode over the cost of the oracle policy on the same seed, in steps on "
           "the base layer and in money with the layers on"),
    table_cells.Column(("Cost",), "cost", ".1f", "the cost of the episode in money, the prices paid, the misdiagnosis price, the production loss and the failure penalties",
           layers=True, group="With the layers on"),
    table_cells.Column(("Production", "loss"), "downtime_total", ".1f", "the part of that cost that the asset loses in the cycles it runs degraded or stands stopped for a repair", layers=True, group="With the layers on"),
    table_cells.Column(("Time to", "repair"), "repair_delay", ".1f", "the largest number of cycles from the onset of a fault to the close of the work order "
           "that restored its component", layers=True, group="With the layers on"),
)


# ----- the body of the paper, one table per experiment -----
def agents_table(sheet):
    read = []

    def cells(line, label, server, preset):
        judged = sheet.judge_cells("agents", line, table_cells.percent_one_line, label, server, preset)
        read.append(sheet.read("agents", line, "AGENTS", label, server, preset))
        if read[-1] is None:
            return [*judged, *[table_cells.plain(table_cells.NO_RUN)] * len(AGENT_OUTCOME)]
        return [*judged, *(table_cells.percent_one_line(read[-1], metric) for _, metric in AGENT_OUTCOME)]

    groups = sheet.agent_lines("agents", cells)
    columns = len(table_cells.JUDGED) + len(AGENT_OUTCOME)
    return table_cells.MainTable("agents", f"Agents, the {MCP} and the {ENVIRONMENT}",
                     f"Each design without and with the {MCP}, on the base layer and in the {ENVIRONMENT} with all six big layers on"
                     f"{sheet.played(read)}. The first three columns are the metrics of Figure 5 of AssetOpsBench, task completion (TC), data "
                     "retrieval accuracy (DRA) and result verification (RV), each per episode the mean of five judge calls at temperature 0.3. "
                     f"{table_cells.UNJUDGED_RULE} The "
                     "verified outcome follows. Top-1 and Pass decide the hypotheses H1 and H2 on the base layer, and Pass needs the right diagnosis "
                     f"and the restored asset. Pass^{PASS_K_SHOWN} is the probability that {table_cells.in_words(PASS_K_SHOWN)} episodes of one test combination "
                     "from different initial states all pass, by the estimator of tau-bench, averaged over the test combinations. Every cell is a "
                     f"percentage, the mean over the episodes of the line and, after the sign {table_cells.PLUS_MINUS}, its standard error from a bootstrap "
                     "that resamples test combinations. The other measures of the same lines follow in the next table.",
                     ["Agent", *table_cells.judge_heads(), *(f"{name}{table_cells.PERCENT_UNIT}" for name, _ in AGENT_OUTCOME)], "l" + "c" * columns, groups,
                     tex_header=[rf" & \multicolumn{{{len(table_cells.JUDGED)}}}{{c}}{{{table_cells.JUDGE_GROUP}}} & \multicolumn{{{len(AGENT_OUTCOME)}}}{{c}}{{Verified outcome}} \\",
                                 rf"\cmidrule(lr){{2-{1 + len(table_cells.JUDGED)}}}\cmidrule(l){{{2 + len(table_cells.JUDGED)}-{1 + columns}}}",
                                 "Agent & " + " & ".join(name for name, _ in (*table_cells.JUDGED, *AGENT_OUTCOME)) + r" \\"],
                     indented=True, spaced=True)


def agent_measures_table(sheet):
    read = []

    def cells(line, label, server, preset):
        read.append(sheet.read("agentmeasures", line, "AGENTS", label, server, preset))
        if read[-1] is None:
            return [table_cells.plain(table_cells.NO_RUN)] * len(AGENT_MEASURES)
        return [table_cells.plain(table_cells.DASH) if column.layers and preset == AGENTS_PRESET else table_cells.one_line_se(read[-1], column.metric, column.spec, column.scale)
                for column in AGENT_MEASURES]

    groups = sheet.agent_lines("agentmeasures", cells)
    names = {column.metric: " ".join(column.head) for column in AGENT_MEASURES}
    shares = [names[column.metric] for column in AGENT_MEASURES if column.scale == 100]
    layered = [names[column.metric] for column in AGENT_MEASURES if column.layers]
    sentences = [f"The lines of the agents table{sheet.played(read)}.", *(f"{names[column.metric]} is {column.meaning}." for column in AGENT_MEASURES)]
    if "faulty_set_right" in names:
        sentences.append(f"{names['faulty_set_right']} stands against {sheet.exact_baseline()}.")
    if shares:
        sentences.append(f"{table_cells.listed(shares)} {'is a percentage' if len(shares) == 1 else 'are percentages'}.")
    if layered:
        sentences.append(f"{table_cells.listed(layered)} {'exists' if len(layered) == 1 else 'exist'} only with the layers on, so a line on the base layer "
                         f"prints {table_cells.DASH} there.")
    sentences.append(f"Every cell is the mean over the episodes of the line and, after the sign {table_cells.PLUS_MINUS}, its standard error from a bootstrap "
                     "that resamples test combinations.")
    runs = table_cells.column_groups(AGENT_MEASURES)
    return table_cells.MainTable("agentmeasures", "The measures of the agents", " ".join(sentences),
                     ["Agent", *(f"{names[column.metric]}{table_cells.PERCENT_UNIT if column.scale == 100 else ''}" for column in AGENT_MEASURES)],
                     "l" + "c" * len(AGENT_MEASURES), groups,
                     tex_header=[" & " + " & ".join(rf"\multicolumn{{{last - first + 1}}}{{c}}{{{group}}}" for group, first, last in runs) + r" \\",
                                 "".join(rf"\cmidrule({'l' if i == len(runs) - 1 else 'lr'}){{{first}-{last}}}" for i, (_, first, last) in enumerate(runs)),
                                 "Agent & " + " & ".join(table_cells.stacked(*column.head, vertical="b") if len(column.head) > 1 else column.head[0]
                                                         for column in AGENT_MEASURES) + r" \\"],
                     indented=True, spaced=True)


def hypotheses_table(sheet):
    found = []

    def line(name, row, reference):
        found.append(sheet.hypothesis("hyp", name, name))
        floor = " (baseline at floor)" if found[-1] and as_bool(found[-1].get("at_floor")) else ""
        comparison = f"{row[2]} against {reference[2]}{floor}"
        return sheet.counted("hyp", name, [name, comparison], [table_cells.difference(found[-1], "diagnosis_all_correct"), table_cells.holm_cell(found[-1], "diagnosis_all_correct_holm_p"),
                                                               table_cells.difference(found[-1], "pass"), table_cells.holm_cell(found[-1], "pass_holm_p")])

    lines = [line(*hypothesis) for hypothesis in HYPOTHESES]
    groups = [[entry for entry in lines if entry[0][0].startswith(prefix)] for prefix in ("H1", "H2")]
    return table_cells.MainTable("hyp", "The hypotheses H1 and H2",
                     f"The hypotheses H1 on the agent and H2 on the {MCP} as paired differences of the agents table by test combination and entry offset"
                     f"{table_cells.pairs_clause(found)}, each with its 95 percent interval from a bootstrap that resamples test combinations. H1, the "
                     f"CausalDiagnosis agent does better than each AssetOpsBench agent. H2, the {MCP} helps every agent. A comparison holds when both "
                     "differences lie above zero and both Holm decisions reject, Holm over the 14 tests of this table. A baseline whose verified pass rate on "
                     f"the test runs lies below {float(aggregate.registered_floor(None)):.0%} is at floor, and H1 against it is uninformative, its tests kept in the family.",
                     ["", "Comparison", f"{table_cells.DELTA}Top-1", "Holm p", f"{table_cells.DELTA}Pass", "Holm p"], r"l>{\raggedright\arraybackslash}p{8.8cm}cccc", groups)


def ladder_table(sheet):
    def cells(line, label, server, preset):
        judged = sheet.judge_cells("ladder", line, table_cells.judge_over_se, label, server, preset)
        row = sheet.read("ladder", line, "AGENTS", label, server, preset)
        if row is None:
            return [*judged, *[table_cells.plain(table_cells.NO_RUN)] * len(LADDER_OUTCOME)]
        return [*judged, *(table_cells.level_over_se(row, metric) for _, metric in LADDER_OUTCOME)]

    return table_cells.MainTable("ladder", "The cumulative layer study",
                     f"Each row adds one big layer to the row above it within one agent with the {MCP}. The first three columns are the metrics of "
                     "Table 16 of AssetOpsBench, TC, DRA and RV, and the verified outcome follows, where cost over the cost of the oracle counts calls on "
                     f"Base and prices from the next row on. Each cell is the mean over the episodes of the line, a percentage in every column but the cost "
                     f"over the oracle, and, with the sign {table_cells.PLUS_MINUS}, its "
                     f"standard error from a bootstrap that resamples test combinations. {table_cells.UNJUDGED_RULE} The Base rows are the runs of the agents table.",
                     ["Agent", "Setting", *table_cells.judge_heads(), *(table_cells.level_head(name, metric) for name, metric in LADDER_OUTCOME)],
                     "ll" + "c" * (len(table_cells.JUDGED) + len(LADDER_OUTCOME)),
                     sheet.ladder("ladder", cells),
                     tex_header=[rf" & & \multicolumn{{3}}{{c}}{{{table_cells.JUDGE_GROUP}}} & \multicolumn{{3}}{{c}}{{Verified outcome}} \\", r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
                                 "Agent & Setting & " + " & ".join(latex_page.tex_text(name) for name in [*(name for name, _ in table_cells.JUDGED), *(name for name, _ in LADDER_OUTCOME)]) + r" \\"],
                     first_only=True)


def training_table(sheet):
    def row_of(line, label, server, variant, subset):
        return sheet.cite("training", f"{line}, {subset}", "TRAINING", sheet.first("TRAINING", model=sheet.model, agent=label, diagnostics_server=server,
                                                                                 training_variant=variant, subset=subset))

    def cells(line, label, server, variant):
        # the judge metrics over every episode of the line, computed in TRAINING on the episodes of its pass and top-1, so a fold
        # without an adapter carries them through the runs of no training
        overall = row_of(line, label, server, variant, "all")
        if overall is None:
            out = [table_cells.plain(table_cells.NO_RUN)] * len(table_cells.JUDGED)
        elif not as_float(overall["judged"]):
            out = [table_cells.plain(table_cells.NO_JUDGE)] * len(table_cells.JUDGED)
        else:
            # an episode without a judge record counts as false on every criterion (counting.unjudged) and one judged under another
            # protocol fills no cell, as in the agents table, and a line with its cells enters the sentence above the tables
            out = [table_cells.judge_over_se(overall, metric) if overall[f"{metric}_n"] == overall["n"] else table_cells.plain(table_cells.MIXED) for _, metric in table_cells.JUDGED]
            if table_cells.plain(table_cells.MIXED) not in out:
                sheet.judged.setdefault("training", []).append((line, overall))
        out.append(table_cells.marked(table_cells.kappa_cell(overall, table_cells.over_se), overall) if overall else table_cells.plain(table_cells.NO_RUN))
        for _, metric in MEASURES:
            for subset, _ in SUBSETS:
                row = row_of(line, label, server, variant, subset)
                out.append(table_cells.marked(table_cells.level_over_se(row, metric), row) if row else table_cells.plain(table_cells.NO_RUN))
        out.append(table_cells.marked(table_cells.over_se(overall, "unmatched_evidence", ".2f"), overall) if overall else table_cells.plain(table_cells.NO_RUN))
        return out

    # each agent's group opens with its baseline without the MCP and its row with the MCP as released, and the trained variants
    # follow indented, as the tables of the paper print them
    def group(baseline, row):
        first = f"{baseline[2]}, {VARIANT_NAMES[AS_RELEASED]}"
        lines = [sheet.counted("training", first, [baseline[2]], cells(first, row_label(baseline, sheet.model), baseline[1], AS_RELEASED))]
        return lines + [sheet.counted("training", f"{row[2]}, {VARIANT_NAMES[variant]}", [row[2] if variant == AS_RELEASED else f"+ {VARIANT_NAMES[variant]}"],
                                      cells(f"{row[2]}, {VARIANT_NAMES[variant]}", row_label(row, sheet.model), row[1], variant)) for variant in TRAINING_VARIANTS]

    groups = [group(baseline, row) for baseline, row in zip(TRAINING_BASELINES, TRAINING_ROWS)]
    return table_cells.MainTable("training", "Training in the environment",
                     "All six big layers on, the metrics of AssetOpsBench over every test episode and the verified outcome over every episode and on the "
                     "test combinations with a single fault and with two faults, under cross-fitting, where every test episode is played by the model that holds out the fold "
                     "of its fault combination. GEPA on the score alone runs the loop of GEPA on the base weights with feedback that carries the score "
                     "alone. Fine-tuning is LoRA rejection sampling fine-tuning of the model with the filter on the verified pass, and the control, "
                     "fine-tuning on unfiltered trajectories, trains on as many episodes. A dagger marks a cell whose episodes include a fold on which the filter kept fewer "
                     "episodes than the model has training initial states, after one more round of sampling, so that no adapter was trained and the variant "
                     "reads the untrained runs of that fold, with paired differences of zero there. The first row of each agent plays it without the CausalDiagnosis MCP and without training, its reference line on all six big "
                     "layers on the same episodes, the second row with the CausalDiagnosis MCP without training on the server that holds the LoRA adapters "
                     f"of its trained variants, and the rows under it its trained variants. {table_cells.UNJUDGED_RULE} Judge kappa and Unmatched "
                     f"read every test episode of the row. Each cell is the mean over the episodes, a percentage in the columns of the judge, of Top-1 and "
                     f"of Pass, and, with the sign {table_cells.PLUS_MINUS}, its standard error "
                     "from a bootstrap that resamples test combinations.",
                     ["Agent", *table_cells.judge_heads(), JUDGE_KAPPA, *(table_cells.level_head(f"{name} {head}", metric) for name, metric in MEASURES for _, head in SUBSETS),
                      "Unmatched"],
                     "lccccccccccc", groups,
                     tex_header=[rf" & \multicolumn{{3}}{{c}}{{{table_cells.JUDGE_GROUP}}} & & \multicolumn{{7}}{{c}}{{Verified outcome}} \\", r"\cmidrule(lr){2-4}\cmidrule(l){6-12}",
                                 " & & & & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in MEASURES) + r" & \\", r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
                                 "Agent & " + " & ".join([*(latex_page.tex_text(name) for name, _ in table_cells.JUDGED), latex_page.tex_text(JUDGE_KAPPA), *(head for _ in MEASURES for _, head in SUBSETS),
                                                          "Unmatched"]) + r" \\"],
                     indented=True)
