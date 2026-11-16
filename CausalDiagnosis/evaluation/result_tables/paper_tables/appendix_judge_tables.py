from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.table_definitions.experiment_tables import HYPOTHESES, AGENTS_PRESET, AGENT_ROWS, row_label
from . import table_cells, result_sheet


# the constructs of the verifier that the appendix table of the judge sets its majority against, in the order of aggregate.JUDGE_CONSTRUCTS
CONSTRUCT_NAMES = {"pass": "Pass", "diagnosis_all_correct": "Top-1", "faulty_set_right": "Exact match", "restored_clean": "Restored"}


def h0_row(sheet, table_label, line, test, name):
    return sheet.cite(table_label, line, "H0", sheet.first("H0", test=test, name=name, model=sheet.model, config=AGENTS_PRESET))


def h0_rows_table(sheet):
    def cells(name):
        found = h0_row(sheet, "judgerows", name, "row", name)
        return [table_cells.signed(found, "statistic"), table_cells.holm_cell(found, "h0_holm_p"), table_cells.kappa_cell(found) if found else table_cells.plain(table_cells.NO_RUN), table_cells.number(found, "discordance"),
                table_cells.number(found, "false_pass"), table_cells.number(found, "false_fail")]

    return table_cells.MainTable("judgerows", "The judge against the verified pass, per row",
                     f"Layer 0, the judge's majority under the protocol of AssetOpsBench against the verified pass of the same episodes. {table_cells.UNJUDGED_RULE} "
                     "The difference is the share the judge passes minus the share the verifier passes, Holm over the row and comparison tests of the "
                     "judge together. A test of a row that misses a planned episode, with an episode judged under another protocol or without any judge "
                     "record, or with too few initial states on which the two disagree counts with p equal to one. Kappa, the discordance and the false "
                     "pass and false fail rates of the judge decide nothing.",
                     ["Agent", f"{table_cells.DELTA}Judge - verified", "Holm p", f"Judge {table_cells.KAPPA}", "Discordance", "False pass", "False fail"], "lcccccc",
                     [[sheet.counted("judgerows", name, [name], cells(name)) for _, _, name in AGENT_ROWS]])


def h0_comparisons_table(sheet):
    def line(name, row, reference):
        found = h0_row(sheet, "judgecmp", name, "comparison", name)
        return sheet.counted("judgecmp", name, [name, f"{row[2]} against {reference[2]}"],
                             [table_cells.signed(found, "statistic"), table_cells.holm_cell(found, "h0_holm_p"), table_cells.number(found, "judge_diff", "+.2f"),
                              table_cells.number(found, "verified_diff", "+.2f"), table_cells.flag(found, "reversed")])

    return table_cells.MainTable("judgecmp", "The judge against the verified pass, per comparison",
                     f"Layer 0, the comparisons of H1 and H2 scored twice on the same matched episodes. {table_cells.UNJUDGED_RULE} The difference is the judge's "
                     "paired difference "
                     "minus the verified one, in the same Holm family as the row tests, and the judge reverses a comparison when the two paired differences "
                     "have opposite signs.",
                     ["", "Comparison", f"{table_cells.DELTA}Judge - verified", "Holm p", f"{table_cells.DELTA}Pass judge", f"{table_cells.DELTA}Pass verified", "Reversed"],
                     r"l>{\raggedright\arraybackslash}p{6.4cm}ccccc", [[line(*hypothesis) for hypothesis in HYPOTHESES]])


def judge_agreement_table(sheet):
    def lines(prefix, server, name):
        label = row_label((prefix, server), sheet.model)
        agents = sheet.read("judgeagree", name, "AGENTS", label, server)
        restored = sheet.read("judgeagree", name, "RESTORED", label, server)
        verified = [table_cells.over_se(agents, "unmatched_evidence", ".2f") if agents else table_cells.plain(table_cells.NO_RUN),
                    table_cells.over_se(restored, "restored_clean", ".2f") if restored else table_cells.plain(table_cells.NO_RUN)]
        judged = sheet.find_all("JUDGE", label, server, judge_protocol=result_sheet.PAPER_PROTOCOL)
        if not judged:
            return [sheet.counted("judgeagree", name, [name], [*[table_cells.plain(table_cells.NO_JUDGE if agents else table_cells.NO_RUN)] * (len(aggregate.JUDGE_CONSTRUCTS) + 2), *verified])]
        out = []
        for row in judged:  # one line per judge model, which only a development root can hold more than one of
            line = name if len(judged) == 1 else f"{name}, judge {row['judge_model']}"
            sheet.cite("judgeagree", line, "JUDGE", row)
            out.append(sheet.counted("judgeagree", name, [line], [*(table_cells.over_se(row, f"kappa_{construct}", ".2f") for construct in aggregate.JUDGE_CONSTRUCTS),
                                                                  table_cells.over_se(row, "judge_record_cut", ".2f"), table_cells.over_se(row, "judge_passed_unrecovered", ".2f"), *verified]))
        return out

    groups = [[entry for prefix, server, name in AGENT_ROWS for entry in lines(prefix, server, name)]]
    return table_cells.MainTable("judgeagree", "The judge against four constructs of the verifier",
                     "Layer 0, the rows of the agents table, the majority of the judge under the protocol of AssetOpsBench against four constructs of the "
                     "verifier: Cohen's kappa against Pass, Top-1, exact match and restored with no unnecessary order, each over the judged episodes and, "
                     f"with the sign {table_cells.PLUS_MINUS}, its standard error, none when either rule puts every episode on one side. Cut is the share of the judged "
                     "episodes whose judge input was cut to the context of the served judge, and Passed unrecovered the share of the judged episodes that the judge "
                     "passes while a component ends off its healthy level. Unmatched is the share of episodes that cited an evidence identifier the "
                     "released evidence does not hold, and Restored the share of episodes that end with every component at its healthy level and no order "
                     "on a healthy component. A row that two judge models scored prints one line per judge.",
                     ["Agent", *(f"{table_cells.KAPPA} {CONSTRUCT_NAMES[construct]}" for construct in aggregate.JUDGE_CONSTRUCTS), "Cut", "Passed unrecovered",
                      "Unmatched", "Restored"], "l" + "c" * (len(aggregate.JUDGE_CONSTRUCTS) + 4), groups,
                     tex_header=["Agent & " + " & ".join([*(table_cells.stacked(table_cells.KAPPA, CONSTRUCT_NAMES[construct]) for construct in aggregate.JUDGE_CONSTRUCTS), "Cut",
                                                         table_cells.stacked("Passed", "unrecovered"), "Unmatched", "Restored"]) + r" \\"])
