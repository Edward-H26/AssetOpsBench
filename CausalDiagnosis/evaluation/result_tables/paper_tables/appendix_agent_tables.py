from CausalDiagnosis.experiments.table_definitions.experiment_tables import H2_PRIME, H2_PRIME_MEASURES, LADDER, MCP
from . import table_cells


H2_PRIME_HEADS = {"pass": ("Pass", "pass"), "reward": ("Reward", "the verifier's reward")}  # head and caption words of each H2_PRIME_MEASURES
FAULTS = {"single": "Single fault", "two": "Two faults"}
STRATA_SHOWN = (("Label separable", "separable"), ("E18 separable", "separable_e18"), ("Both inseparable", "inseparable_both"))


# ----- the appendix of the paper -----
def efficiency_table(sheet):
    def cells(line, label, server):
        out = []
        for faults in FAULTS:
            row = sheet.read("efficiency", line, "EFFICIENCY", label, server, faults=faults)
            out += [table_cells.mean_std(row, "tool_invocations", ".1f"), table_cells.mean_std(row, "wall_s", ".0f")] if row else [table_cells.plain(table_cells.NO_RUN)] * 2
        return out

    return table_cells.MainTable("efficiency", "Execution efficiency",
                     f"Agents and the {MCP} on layer 0 in the metrics of Table 3 of AssetOpsBench, execution efficiency as mean {table_cells.PLUS_MINUS} standard "
                     "deviation over episodes. Steps count every attempted tool call, one rule for every agent, of the action or metered servers and of the "
                     "CausalDiagnosis MCP, failed and refused calls included, and the test combinations with a single fault and with two faults take the place of its "
                     "single-agent and multi-agent tasks.",
                     ["Agent", *(f"{name} {part}" for name in FAULTS.values() for part in ("steps", "runtime (s)"))], "lcccc", sheet.six("efficiency", cells),
                     tex_header=[rf" & \multicolumn{{2}}{{c}}{{{FAULTS['single']}}} & \multicolumn{{2}}{{c}}{{{FAULTS['two']}}} \\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
                                 r"Agent & Steps & Runtime (s) & Steps & Runtime (s) \\"])


def h2_prime_table(sheet):
    heads = [H2_PRIME_HEADS[measure] for measure in H2_PRIME_MEASURES]

    def line(name, row, reference):
        found = sheet.cite("mcptop", name, "H2PRIME_pairs", sheet.first("H2PRIME_pairs", hypothesis=name, model=sheet.model, config=LADDER[-1]))
        return sheet.counted("mcptop", name, [name, f"{row[2]} against {reference[2]}"],
                             [cell for measure in H2_PRIME_MEASURES for cell in (table_cells.difference(found, measure), table_cells.holm_cell(found, f"{measure}_holm_p"))])

    return table_cells.MainTable("mcptop", "The hypotheses H2' on the top layer",
                     f"All six big layers on, each agent with the {MCP} as released against itself without it, as paired differences of "
                     f"{' and of '.join(words for _, words in heads)} by test combination and entry offset. A comparison holds when every difference "
                     f"favors the {MCP} and every Holm decision rejects, with Holm run once over the registered size of the group that joins these tests "
                     "to those of H1 on the top layer.",
                     ["", "Comparison", *(part for head, _ in heads for part in (f"{table_cells.DELTA}{head}", "Holm p"))],
                     r"l>{\raggedright\arraybackslash}p{8.8cm}" + "cc" * len(heads), [[line(*comparison) for comparison in H2_PRIME]])


def strata_table(sheet):
    def cells(line, label, server):
        out = []
        for _, stratum in STRATA_SHOWN:
            row = sheet.read("strata", f"{line}, {stratum}", "STRATA", label, server, stratum=stratum)
            out += [table_cells.estimate(row, "success", ".2f"), table_cells.estimate(row, "abstained", ".2f")] if row else [table_cells.plain(table_cells.NO_RUN)] * 2
        return out

    return table_cells.MainTable("strata", "Success and declarations by stratum",
                     f"Layer 0, a secondary table. The label is the separability of the reference policy, E18 the first commitment majority of the {MCP}, "
                     "and a declaration earns its credit only where both mark the initial state inseparable.",
                     ["Agent", *(f"{name} {part}" for name, _ in STRATA_SHOWN for part in ("success", "declared"))], "lcccccc", sheet.six("strata", cells),
                     tex_header=[" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for name, _ in STRATA_SHOWN) + r" \\",
                                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", "Agent & " + " & ".join("Success & Declared" for _ in STRATA_SHOWN) + r" \\"])


def exact_match_table(sheet):
    def cells(line, label, server):
        row = sheet.read("exactmatch", line, "AGENTS", label, server)
        if row is None:
            return [table_cells.plain(table_cells.NO_RUN)] * 4
        baseline = table_cells.number(row, "faulty_set_right_baseline")
        if str(row.get("faulty_set_right_baseline_n", "")) != "":  # the page names the episodes behind the baseline, the LaTeX form its share
            baseline = table_cells.Cell(baseline.short, f"{baseline.full} ({row['faulty_set_right_baseline_k']} of {row['faulty_set_right_baseline_n']})")
        return [table_cells.over_se(row, "faulty_set_right", ".2f"), baseline, table_cells.number(row, "faulty_set_right_floor"), table_cells.flag(row, "faulty_set_right_at_floor")]

    return table_cells.MainTable("exactmatch", "Exact match and its majority-class baseline",
                     "Layer 0, the rows of the agents table. Exact match is the share of the episodes of a row whose diagnosis names exactly the faulty "
                     f"components, the mean and, with the sign {table_cells.PLUS_MINUS}, its standard error. The majority-class baseline is the share of the episodes "
                     "of the row that fall on the most frequent faulty set of the test combinations it played, the rate of an agent that always names that set. The "
                     "floor of a row is the larger of that baseline and the floor of Pass, and a row at or below its floor is marked.",
                     ["Agent", "Exact match", "Majority-class baseline", "Floor", "At floor"], "lcccc", sheet.six("exactmatch", cells))
