from pathlib import Path
from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.evaluation.result_tables import index_file, pooling_rules
from . import read_sources, table_cells, result_sheet, appendix_layer_tables, thinking_tables, markdown_page, latex_page, body_tables, appendix_judge_tables, appendix_schedule_tables, appendix_agent_tables, appendix_training_tables, appendix_mcp_tables


# one list per heading, in the order of the paper's tables
LAYOUT = ((body_tables.agents_table, body_tables.agent_measures_table, body_tables.hypotheses_table, body_tables.ladder_table, body_tables.training_table),
          (appendix_agent_tables.efficiency_table, appendix_agent_tables.exact_match_table, thinking_tables.thinking_table, appendix_layer_tables.layers_table, appendix_layer_tables.misdiagnosis_table, appendix_layer_tables.full_environment_table, appendix_layer_tables.policies_table, appendix_layer_tables.policy_layers_table,
           appendix_training_tables.training_pairs_table, appendix_training_tables.training_folds_table, appendix_training_tables.budget_table, appendix_training_tables.hazard_table, appendix_agent_tables.h2_prime_table, appendix_judge_tables.h0_rows_table, appendix_judge_tables.h0_comparisons_table,
           appendix_judge_tables.judge_agreement_table, appendix_agent_tables.strata_table, appendix_mcp_tables.offline_table, appendix_mcp_tables.mcp_table, appendix_schedule_tables.parallel_table, appendix_schedule_tables.parallel_pair_table))


def partial_sentence(read):
    """The sentence under a table read below the fraction of record: the fraction it reads, the fraction of record, no registered
    test, and for a cut look below one replicate the estimates without intervals (counting.fraction_of_record)."""
    plan = aggregate.registered_of({})
    record = plan["counting"]["fraction_of_record"]
    sentence = (f"Every line reads the fraction {read} of the test set, below the fraction of record {record}. No test is run "
                "at such a read, so the table reports no p value and no Holm decision.")
    if aggregate.read_of(read, plan)["kind"] == "cut":
        sentence += " A read below one replicate takes its start combinations for coverage, so its cells print estimates without intervals."
    return sentence


def counting_sentences(sheet, table):
    """The sentences that state the registered counting rule under a table that holds a refused or an incomplete line or reads
    below the fraction of record, as (page, LaTeX), the LaTeX one already in LaTeX, or () for a table without one: one to three
    incomplete lines are named with their episodes, more point to the table of counts. The wording of the refused and incomplete
    sentences is the paper's."""
    marked = sheet.marked.get(table.label, [])
    read = sheet.partial_read
    if not marked and not read:
        return ()
    page, tex = [], []
    if read:
        page.append(partial_sentence(read))
        tex.append(latex_page.tex_text(partial_sentence(read)))
    if any(state == table_cells.REFUSED for _, state, _, _, _ in marked):
        limit = aggregate.registered_of({})["counting"]["max_unscored_share"]
        sentence = (f"A line whose cells read refused has more than {100 * limit:g} percent of its episodes unscored. The counting "
                    "rule refuses such a line, so its cells and its tests are not reported.")
        page.append(sentence)
        tex.append(latex_page.tex_text(sentence))
    short = [(line, counts, held) for line, state, counts, _, held in marked if state == table_cells.INCOMPLETE]
    # a read below the fraction of record runs no test, so its incomplete lines enter no Holm procedure
    holm = "" if read else (" Under the counting rule, the tests of such a line are not run and are assigned a p value of one "
                            "in the Holm procedure.")
    holm_many = "" if read else " Under the counting rule, their tests are not run and are assigned a p value of one in the Holm procedure."
    holm_tex, holm_many_tex = (latex_page.tex_text(text).replace(" p value", " $p$ value") for text in (holm, holm_many))
    form = "(incomplete, k of n, replicates H)" if any(held for _, _, held in short) else "(incomplete, k of n)"
    unknown = ("" if all(planned is not None for _, (_, planned), _ in short) else
               " A line marked (incomplete, k played) misses a run whose planned episodes are not known, so it prints the episodes it played.")
    held_words = " in runs of the replicates H of the test set" if form.endswith("H)") else ""
    if 0 < len(short) <= 3:
        named = [f"{played} {'played' if planned is None else f'of {planned}'}{f' with the replicates {held}' if held else ''} for {line}"
                 for line, (played, planned), held in short]
        named = named[0] if len(named) == 1 else " and ".join(named) if len(named) == 2 else f"{named[0]}, {named[1]}, and {named[2]}"
        page.append(f"A line marked {form} ran k of its n planned episodes{held_words}.{unknown}{holm}")
        tex.append(f"A line marked {latex_page.tex_signs(table_cells.SHORT)} ran fewer episodes than planned, {latex_page.tex_text(named)}.{latex_page.tex_text(unknown)}{holm_tex}")
    elif short:
        page.append(f"Lines marked {form} ran fewer episodes than planned{held_words}. Table `tab:counts` gives the episodes run and planned for "
                    f"each.{unknown}{holm_many}")
        tex.append(f"Lines marked {latex_page.tex_signs(table_cells.SHORT)} ran fewer episodes than planned. Table~\\ref{{tab:counts}} gives the episodes run and planned "
                   f"for each.{latex_page.tex_text(unknown)}{holm_many_tex}")
    return " ".join(page), " ".join(tex)


def counts_table(sheet):
    """The table of counts at the end of the appendix: every refused or incomplete line of the tables of a sheet, with the table it
    stands in, the episodes it ran and planned and the unscored share of its episodes, which the caption of a table with more than
    three incomplete lines points to."""
    def count(counts, i):
        return table_cells.plain(table_cells.NO_VALUE if counts is None or counts[i] is None else str(counts[i]))

    lines = [([f"tab:{label}", line], [count(counts, 0), count(counts, 1), table_cells.plain(table_cells.NO_VALUE if share is None else f"{share:.1%}"),
                                         table_cells.plain(f"{state}, replicates {held}" if held else state)])
             for label, marked in sheet.marked.items() for line, state, counts, share, held in marked]
    return table_cells.MainTable("counts", "Lines that ran fewer episodes than planned or are refused",
                     "Every line of the tables above that ran fewer episodes than planned or that the counting rule refuses, with "
                     "the table it stands in, the episodes it ran and planned, and the share of its episodes that are unscored. The tests that read "
                     "these lines are not run and are assigned a p value of one in the Holm procedure.",
                     ["Table", "Line", "Run", "Planned", "Unscored", "State"], "llcccc", [lines])


def built_layout(sheet):
    """The tables of every section of LAYOUT, each with the sentences of the counting rule its lines need, and the table of counts
    after the appendix when a line carries a state, so that tables without one print as before."""
    built = [[build(sheet) for build in section] for section in LAYOUT]
    built = [[table._replace(counting=counting_sentences(sheet, table)) for table in section] for section in built]
    if sheet.marked:
        built[-1].append(counts_table(sheet))
    return built


def write(aggregate_dir, folder, index_path):
    """The two forms in the folder of the results README and their section in the results index. The tables are for
    granite first, the model the paper tests its hypotheses on, and repeat for every other model."""
    tables = read_sources.read_tables(aggregate_dir)
    models = sorted({row["model"] for row in tables["AGENTS"]}, key=lambda model: (not model.startswith("granite"), model)) or [""]
    built = []
    for model in models:
        sheet = result_sheet.Sheet(tables, model)
        built.append((sheet, built_layout(sheet)))
    folder = Path(folder)
    # the kind of the root behind the tables, as the index of tables_from_runs/ records it: a development root opens every page with the banner
    index = pooling_rules.read_json(index_path, {})
    head = pooling_rules.head_lines(index.get("root_kind"), index.get("record"))
    (folder / read_sources.PAGE).write_text("".join(f"{line}\n" for line in head) + markdown_page.page(built, tables), encoding="utf-8")
    (folder / read_sources.LATEX).write_text("".join(f"% {line}\n" for line in head if line) + latex_page.latex(built, tables), encoding="utf-8")
    index_file.add_section(index_path, "main_tables", {"files": [read_sources.PAGE, read_sources.LATEX], "lines": [entry for sheet, _ in built for entry in sheet.cited],
                                                          "thinking_outside_generation": {sheet.model: thinking_tables.thinking_outside(sheet) for sheet, _ in built}})
