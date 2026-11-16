"""A8 and MISDIAGNOSIS_PRICE carry the tool level of each row, its surface and its diagnostics server, after their existing
columns, so a row of the level without the CausalDiagnosis tools is told apart from one with them in every table of the agents."""
from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.table_definitions.experiment_tables import PE, PE_MCP
from CausalDiagnosis.evaluation.result_tables.tests.test_paper_tables import _episode, _meta, _run


def test_the_two_tables_append_the_level_columns_after_their_existing_ones():
    columns = {table.id: table.columns for table in aggregate.tables()}
    assert columns["A8"][-1] == "surface" and "diagnostics_server" in columns["A8"]
    assert columns["MISDIAGNOSIS_PRICE"][-2:] == ["surface", "diagnostics_server"]


def test_a8_splits_the_rows_of_the_two_levels(tmp_path):
    runs = tmp_path / "runs"
    for row, folder in ((PE, "pe_bare"), (PE_MCP, "pe_mcp")):
        _run(runs, folder, [_episode(s, row, abstained=s == 4, **{"pass": s < 2}) for s in range(5)], _meta())
    tables = {t.id: rows for t, rows in aggregate.aggregate(runs, tmp_path / "out", resamples=50, fraction="1")}
    levels = {(r["surface"], str(r["diagnostics_server"])) for r in tables["A8"]}
    assert levels == {("aob", "False"), ("aob", "True")}
