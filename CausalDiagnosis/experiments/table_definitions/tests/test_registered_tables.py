"""Every table that tables_from_runs/ prints has a registered place, a family of confirmatory tests or a descriptive entry, through the map
tables of the registered file, and the map names exactly the printed tables. A registered key that no table prints stands on the
explicit list below with its reason, so no registered entry and no printed table goes unmatched."""
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate

# registered keys that name no printed table of their own, each with where its number appears instead
NOT_A_TABLE = {
    "descriptive.exact_match": "a column of the main table, with its constant-answer baseline",
    "descriptive.floor_of_pass": "a registered value, the floor of the pass rate, not a table",
    "descriptive.offline_accuracy": "a number quoted beside the main table, no row and no episode",
    "companions.En9": "the plain classifier in a table beside the main table, which aggregate.py names under no table id",
    "companions.healthy_control": "the companion statement of the rows printed by the table HEALTHY_CONTROL (descriptive.healthy_control)",
}


def _node(data, path):
    for key in path.split("."):
        data = data[key]
    return data


def test_every_printed_table_has_its_registered_entry_and_no_other_is_registered():
    data = registered.load()
    printed = [table.id for table in aggregate.tables()]
    assert len(printed) == len(set(printed))
    assert set(data["tables"]) == set(printed)
    for name, path in data["tables"].items():
        section = path.split(".")[0]
        assert section in ("families", "descriptive"), (name, path)
        entry = _node(data, path)
        assert isinstance(entry, (str, dict)) and entry, (name, path)


def test_every_descriptive_and_companion_key_is_a_table_or_listed():
    data = registered.load()
    targets = set(data["tables"].values())
    keys = {f"{section}.{key}" for section in ("descriptive", "companions") for key in data[section]}
    assert not targets & set(NOT_A_TABLE)
    assert keys - targets == set(NOT_A_TABLE)
