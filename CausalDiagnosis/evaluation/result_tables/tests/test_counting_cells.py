"""The registered counting rule in the paper tables (registered_analysis.json, counting.max_unscored_share_rule,
counting.not_played and counting.fraction_of_record): a refused line prints refused in every cell, a line that misses planned
episodes keeps its cells and is marked with its played and planned episodes and the replicates it holds, every line of a read
below the fraction of record is marked partial with its fraction, the caption of its table states the rule, and the table of
counts at the end of the appendix lists every refused or incomplete line. A build without such a line holds no mark, no sentence
and no table of counts, and every row a table cites reaches the rule through the line it fills."""
import csv
import json

import pytest

from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AGENT_ROWS, PE, PE_MCP
from CausalDiagnosis.evaluation.result_tables.tests.test_paper_tables import FOLDERS, _aggregated, _caption, _every_table, _six_rows, _table, _two_folds  # noqa: F401  _two_folds, the autouse fixture of the folds

REFUSED_SENTENCE = ("A line whose cells read refused has more than 2 percent of its episodes unscored. The counting rule refuses "
                    "such a line, so its cells and its tests are not reported.")
SHORT_TEX = r"$^\ddagger$"


def _unscored(folder, seed):
    """The verifier scores no episode of the given seed in a run folder, as an episode that the infrastructure broke."""
    path = folder / "scores.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(row["seed"]) == seed:
            row["pass"] = ""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _planned(folder, planned):
    path = folder / "run.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "episodes_planned": planned}), encoding="utf-8")


def _sheets(results):
    """The sheets of the page of a results folder, built again from its tables, with the tables of each."""
    tables = main_tables.read_tables(results / "tables")
    models = sorted({row["model"] for row in tables["AGENTS"]}) or [""]
    out = []
    for model in models:
        sheet = main_tables.Sheet(tables, model)
        out.append((sheet, main_tables.built_layout(sheet)))
    return out


def _tex(results):
    return (results / main_tables.LATEX).read_text(encoding="utf-8")


def _rewritten(results, table, change):
    """The page of a results folder written again by main_tables after change has set the cells of every row of one aggregate table,
    as aggregate writes them for a row of the test set or for a read below the fraction of record, which these fixtures cannot play."""
    path = results / "tables" / f"{table}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        change(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    main_tables.write(results / "tables", results, results / "INDEX.json")
    return (results / main_tables.PAGE).read_text(encoding="utf-8")


def _lines_by_label(page, label):
    return {line[0]: line[1:] for line in _table(page, label)}


@pytest.mark.registered_file  # the registered largest unscored share, 2 percent, which the test copy leaves open
def test_a_refused_row_prints_refused_in_every_cell_and_its_caption_states_the_rule(tmp_path):
    """One episode of five of Plan-and-Execute on layer 0 unscored, a share of 20% above the registered 2%: its line of the agents table
    reads refused in every cell and keeps its label, the caption names the rule with its registered share, the table of counts lists
    the line with its unscored share, and every other line prints as in the build without the unscored episode."""
    plain_runs, runs = tmp_path / "plain" / "runs", tmp_path / "refused" / "runs"
    _six_rows(plain_runs)
    folders = _six_rows(runs)
    _unscored(folders[PE], 0)
    _, before, _ = _aggregated(tmp_path / "plain", plain_runs)
    _, page, results = _aggregated(tmp_path / "refused", runs)
    lines, plain_lines = _lines_by_label(page, "agents"), _lines_by_label(before, "agents")
    assert set(lines[PE[2]]) == {"refused"}
    assert {name: cells for name, cells in lines.items() if name != PE[2]} == {name: cells for name, cells in plain_lines.items() if name != PE[2]}
    assert _caption(page, "agents").endswith(REFUSED_SENTENCE) and REFUSED_SENTENCE not in _caption(before, "agents")
    counts = _table(page, "counts")
    assert ["tab:agents", PE[2], "5", "none", "20.0%", "refused"] in counts
    tex = _tex(results)
    assert REFUSED_SENTENCE in tex and r"\label{tab:counts}" in tex
    assert all(sheet.uncounted() == [] for sheet, _ in _sheets(results))


def test_an_incomplete_row_keeps_its_cells_and_carries_its_played_and_planned_episodes(tmp_path):
    """Plan-and-Execute with the MCP plans six episodes and plays five: its line keeps the cells of the build that planned five,
    its label carries (incomplete, 5 of 6) on the page and the double dagger in LaTeX, the caption names the line with its counts,
    and the hypotheses that read the row are marked as well, their tests not run."""
    plain_runs, runs = tmp_path / "plain" / "runs", tmp_path / "short" / "runs"
    _six_rows(plain_runs)
    folders = _six_rows(runs)
    _planned(folders[PE_MCP], 6)
    _, before, _ = _aggregated(tmp_path / "plain", plain_runs)
    _, page, results = _aggregated(tmp_path / "short", runs)
    marked = f"+ {main_tables.MCP} (incomplete, 5 of 6)"
    rows, plain_rows = _table(page, "agents"), _table(before, "agents")
    assert [row[1:] for row in rows if row[0] == marked] == [row[1:] for row in plain_rows if row[0] == f"+ {main_tables.MCP}"][:1]
    assert sum(row[0] == marked for row in rows) == 1
    caption = _caption(page, "agents")
    assert caption.endswith("A line marked (incomplete, k of n) ran k of its n planned episodes. Under the counting rule, the tests "
                            "of such a line are not run and are assigned a p value of one in the Holm procedure.")
    tex = _tex(results)
    assert rf"\quad + {main_tables.MCP}{SHORT_TEX}" in tex
    assert f"ran fewer episodes than planned, 5 of 6 for {PE_MCP[2]}. Under" in tex
    hypotheses = {row[0]: row for row in _table(page, "hyp")}
    short = [name for name, row in hypotheses.items() if "(incomplete, 5 of 6)" in row[1]]
    assert short and all(hypotheses[name][3] == "none" and hypotheses[name][5] == "none" for name in short)
    assert all(sheet.uncounted() == [] for sheet, _ in _sheets(results))


def test_four_incomplete_lines_point_to_the_table_of_counts(tmp_path):
    """Four of the six rows each play one episode fewer than planned: the caption of the agents table points to the table of counts
    instead of naming the lines, and the table of counts lists the four lines of the agents table with their counts."""
    runs = tmp_path / "runs"
    folders = _six_rows(runs)
    for row in AGENT_ROWS[:4]:
        _planned(folders[row], 6)
    _, page, results = _aggregated(tmp_path, runs)
    assert "Table `tab:counts` gives the episodes run and planned for each." in _caption(page, "agents")
    assert r"Table~\ref{tab:counts} gives the episodes run and planned for each." in _tex(results)
    listed = [row for row in _table(page, "counts") if row[0] == "tab:agents"]
    assert len(listed) == 4 and {tuple(row[2:4]) for row in listed} == {("5", "6")} and {row[5] for row in listed} == {"incomplete"}


def test_a_build_without_a_refused_or_incomplete_row_holds_no_mark_and_every_cited_row_is_counted(tmp_path):
    """The runs behind every table, none refused and none short of its plan: no line is marked or refused, no caption states the
    rule, no table of counts stands in either form, and every row that a table cites falls under exactly one line it fills."""
    runs = tmp_path / "runs"
    _every_table(runs)
    _, page, results = _aggregated(tmp_path, runs)
    tex = _tex(results)
    assert "(incomplete," not in page and "tab:counts" not in page and "| refused |" not in page
    assert SHORT_TEX not in tex and "tab:counts" not in tex
    for stem in ("A line whose cells read refused has more than", "ran fewer episodes than planned", "assigned a $p$ value of one",
                 "assigned a p value of one"):
        assert stem not in page and stem not in tex
    for sheet, sections in _sheets(results):
        assert sheet.marked == {} and sheet.uncounted() == []
        assert all(table.counting == () for section in sections for table in section)


def test_a_line_of_the_test_set_short_of_a_replicate_carries_the_replicates_it_holds(tmp_path):
    """Plan-and-Execute with the MCP holds the run of replicate 1 alone, 5 of its 25 planned episodes, as aggregate counts a row of
    the test set by its replicates: its label carries (incomplete, 5 of 25, replicates 1), the caption names the form with the
    replicates, the LaTeX names the line with the replicates it holds, and the table of counts states them."""
    runs = tmp_path / "runs"
    _six_rows(runs)
    _, _, results = _aggregated(tmp_path, runs)

    def short_of_a_replicate(row):
        if row["agent"].startswith("plan_execute") and row["diagnostics_server"] == "True":
            row.update(incomplete="True", planned="25", replicates="1")

    page = _rewritten(results, "AGENTS", short_of_a_replicate)
    marked = f"+ {main_tables.MCP} (incomplete, 5 of 25, replicates 1)"
    assert sum(row[0] == marked for row in _table(page, "agents")) == 1
    assert ("A line marked (incomplete, k of n, replicates H) ran k of its n planned episodes in runs of the replicates H of the test set. "
            "Under the counting rule") in _caption(page, "agents")
    assert f"5 of 25 with the replicates 1 for {PE_MCP[2]}." in _tex(results)
    assert ["tab:agents", PE_MCP[2], "5", "25", "0.0%", "incomplete, replicates 1"] in _table(page, "counts")


def test_every_line_of_a_partial_read_is_marked_and_runs_no_test(tmp_path):
    """A read at 0.4 of the test set, below the fraction of record 1.0, as aggregate writes it in every row: every line of the agents
    table carries (partial 0.4), the caption states the read and that no registered test runs, and with no refused or short line
    the page holds no table of counts and the LaTeX no double dagger. A cut look at 0.05 adds that its cells print no interval."""
    runs = tmp_path / "runs"
    _six_rows(runs)
    _, _, results = _aggregated(tmp_path, runs)
    page = _rewritten(results, "AGENTS", lambda row: row.update(read="partial 0.4"))
    assert all(row[0].endswith(" (partial 0.4)") for row in _table(page, "agents"))
    caption = _caption(page, "agents")
    assert ("Every line reads the fraction 0.4 of the test set, below the fraction of record 1.0. No test is run at such a "
            "read, so the table reports no p value and no Holm decision.") in caption and "without intervals" not in caption
    assert "tab:counts" not in page and SHORT_TEX not in _tex(results) and "Holm procedure" not in caption
    page = _rewritten(results, "AGENTS", lambda row: row.update(read="partial 0.05"))
    assert all(row[0].endswith(" (partial 0.05)") for row in _table(page, "agents"))
    assert "A read below one replicate takes its start combinations for coverage, so its cells print estimates without intervals." in _caption(page, "agents")


def test_a_short_line_of_a_partial_read_carries_both_and_enters_no_holm_procedure(tmp_path):
    """A line short of a replicate in a read at 0.4 carries both parts in the form of aggregate.counting_text, and its caption names
    no Holm procedure, since a partial read runs no test."""
    runs = tmp_path / "runs"
    _six_rows(runs)
    _, _, results = _aggregated(tmp_path, runs)

    def partial_and_short(row):
        row.update(read="partial 0.4")
        if row["agent"].startswith("plan_execute") and row["diagnostics_server"] == "True":
            row.update(incomplete="True", planned="10", replicates="1")

    page = _rewritten(results, "AGENTS", partial_and_short)
    assert f"+ {main_tables.MCP} (partial 0.4; incomplete, 5 of 10, replicates 1)" in [row[0] for row in _table(page, "agents")]
    assert "Holm procedure" not in _caption(page, "agents")


def test_the_state_of_a_row_reads_its_counts_and_else_its_status():
    """row_state reads the counts of a table that carries them and the status of a paired table in the form of
    aggregate.counting_text, refused before incomplete before partial, and no other status."""
    assert main_tables.row_state({"refused": "True", "incomplete": "True", "played": "4", "planned": "5"}) == ("refused", None)
    assert main_tables.row_state({"refused": "False", "incomplete": "True", "played": "4", "planned": "5"}) == ("incomplete", (4, 5, ""))
    assert main_tables.row_state({"refused": "False", "incomplete": "True", "played": "17", "planned": "", "replicates": "1",
                                  "read": "partial 0.4"}) == ("incomplete", (17, None, "1"))
    assert main_tables.row_state({"refused": "False", "incomplete": "False", "played": "5", "planned": "5", "read": "partial 0.4"}) == ("partial", "0.4")
    assert main_tables.row_state({"refused": "False", "incomplete": "False", "played": "5", "planned": "5"}) is None
    assert main_tables.row_state({"status": "incomplete, 12 of 13"}) == ("incomplete", (12, 13, ""))
    assert main_tables.row_state({"status": "incomplete, 17 of 85, replicates 1"}) == ("incomplete", (17, 85, "1"))
    assert main_tables.row_state({"status": "partial 0.4; incomplete, 17 played, replicates 1-2"}) == ("incomplete", (17, None, "1-2"))
    assert main_tables.row_state({"status": "partial 0.05"}) == ("partial", "0.05")
    assert main_tables.row_state({"status": "refused"}) == ("refused", None)
    assert main_tables.row_state({"status": "unrun, no judge record"}) is None and main_tables.row_state({"status": ""}) is None
    # the status text that aggregate writes is the text that row_state reads
    for parts in (("partial 0.4",), ("", 17, 85, "1"), ("partial 0.4", 17, None, "1-2")):
        assert main_tables.row_state({"status": main_tables.aggregate.counting_text(*parts)}) is not None
