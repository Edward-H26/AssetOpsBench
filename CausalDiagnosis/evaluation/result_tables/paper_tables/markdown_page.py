from CausalDiagnosis.evaluation.result_tables import leaderboard
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AGENT_ROWS, row_label
from . import read_sources, table_cells, thinking_tables


SENTENCE_NO_RUN = "No run has been played on the current code, so the tables below hold no number."
HEADINGS = ("Tables of the body of the paper, one per experiment, the metrics of the AssetOpsBench paper first and the verified outcome after them",
            "Tables of the appendix of the paper")


# ----- the page -----
def heading(table):
    return f"#### {table.title} (`tab:{table.label}`)"


def page(built, tables):
    """MAIN_TABLES.md from the tables of every model."""
    # RUNS feeds the footnotes on the seeds and no cell, so the sentence leaves it out by name, wherever it stands in SOURCES
    trace = ("a cell of " + ", ".join(f"`tables/{name}.csv`" for name in read_sources.SOURCES if name != "RUNS") + f" or of `{read_sources.OFFLINE}`, and `INDEX.json` names its row")
    out = ["# Main tables of the hydraulic rig gym", ""]
    if tables["AGENTS"] or tables["POLICIES"] or tables["OFFLINE"]:
        out += ["The result tables of the paper in its order and layout, each under its label in the paper, in the body one table per "
                "experiment with the metrics of the AssetOpsBench paper first and the verified outcome after them, then the tables of the appendix. The "
                f"reproduction of AssetOpsBench answers the scenarios of that paper and stands apart. Every number is {trace}. A mean with the sign "
                f"{table_cells.PLUS_MINUS} gives its standard error, except in the execution efficiency, which gives the standard deviation over episodes as "
                "AssetOpsBench does, and brackets give the estimate with its 95 percent interval. Every interval and standard error comes from a "
                "bootstrap that resamples test combinations. `main_tables.tex` holds the same tables in LaTeX, with the standard error under the mean, "
                "beside it in the two tables of the agents, and intervals only where the tables of the paper print them.", ""]
    else:
        out += [f"{SENTENCE_NO_RUN} They stay empty until the test set is played on Columbia's cluster, and every number then traces to {trace}. The "
                "reproduction of AssetOpsBench answers the scenarios of that paper and stands apart. `main_tables.tex` holds the same tables in LaTeX.", ""]
    for sheet, sections in built:
        out += [f"## {sheet.model or 'No model yet'}", ""]
        if sheet.model:
            sets = episode_sets(sheet)
            out += [f"The episode {'set' if len(sets) == 1 else 'sets'} of the runs of the six rows {'is' if len(sets) == 1 else 'are'} {' and '.join(sets)}.", ""]
            out += [line for sentence in sheet.judge_counts() for line in (sentence, "")]
            outside = thinking_tables.thinking_outside(sheet)
            out += [outside["sentence"], ""] if outside else []
        for title, section in zip(HEADINGS, sections):
            out += [f"### {title}", ""]
            for table in section:
                caption = " ".join([table.caption, table.counting[0]]) if table.counting else table.caption
                out += [heading(table), "", caption, "", "| " + " | ".join(table.header) + " |", "|" + "---|" * len(table.header)]
                out += ["| " + " | ".join([*map(table_cells.page_label, labels), *(cell.full for cell in cells)]) + " |" for group in table.groups for labels, cells in group]
                out.append("")
    return "\n".join(out)


def episode_sets(sheet):
    """The sets of episodes that the runs of the six rows of a model played, by the names of the leaderboard, so that a page
    whose rows pool two sets says so, since the aggregate tables pool every run of a condition."""
    labels = {row_label(row, sheet.model) for row in AGENT_ROWS}
    return sorted({leaderboard.episode_set(run) or "seeds of their own" for run in sheet.tables["RUNS"] if run["agent"] in labels})
