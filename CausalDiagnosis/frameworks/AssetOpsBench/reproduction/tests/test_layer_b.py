"""Layer B without a network: the merged scenario file, the Chiller 9 timestamps, the dataset folder and the launcher, and the
refusals of the driver. With AOB_LAYER_B naming a built folder, also the real servers of main-0.x over the real data on a
CouchDB in one process, the loaders, every tool list and a tool call per family."""
import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import fetch_public_servers_and_data as layer_b
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import scenario_file, scenario_sets
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_scenarios
from agent.plan_execute.executor import _call_tool, _list_tools
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.tests.fake_couchdb import start as start_couchdb

BUILT = os.environ.get("AOB_LAYER_B")
real = pytest.mark.skipif(not BUILT, reason="AOB_LAYER_B names no built folder")


def test_only_the_two_rows_take_a_later_form_and_every_other_line_keeps_its_bytes():
    base = '{"id": 1, "text": "a \\u2028 b",  "characteristic_form": "x"}\n{"id": 621, "text": "t", "characteristic_form": null}\n'
    merged = layer_b.with_forms(base, {621: {"characteristic_form": "later"}, 1: {"characteristic_form": "not taken"}})
    first, second, end = merged.split("\n")
    assert first == base.split("\n")[0] and end == ""
    assert json.loads(second) == {"id": 621, "text": "t", "characteristic_form": "later"}


def test_chiller_9_timestamps_take_the_form_of_chiller_6(tmp_path):
    six, nine = tmp_path / "six.json", tmp_path / "nine.json"
    six.write_text(json.dumps([{"asset_id": "Chiller 6", "timestamp": "2020-06-01T00:00:00"}]))
    nine.write_text(json.dumps([{"asset_id": "Chiller 9", "timestamp": "2020-04-23 07:30:00-05:00"},
                                {"asset_id": "Chiller 9", "timestamp": "2016-09-21 12:30:00", "Date": "2016-09-21 12:30:00"}]))
    stamps = [r["timestamp"] for r in layer_b.merged_telemetry(six, [nine])]
    assert stamps == ["2020-06-01T00:00:00", "2020-04-23T07:30:00", "2016-09-21T12:30:00"]


def test_the_dataset_folder_serves_scenario_223_from_chiller9_tsad(tmp_path):
    for name in layer_b.CHILLER_9:
        (tmp_path / f"{name}.csv").write_text(name)
    layer_b.build_datasets(tmp_path, tmp_path / "datasets")
    assert (tmp_path / "datasets" / layer_b.ALIAS[0]).read_text() == "chiller9_tsad"


def test_the_servers_start_from_the_layer_tree_with_its_settings(tmp_path):
    settings = layer_b.server_settings(tmp_path, tmp_path / "out")
    assert settings["IOT_DBNAME"] == "chiller" and settings["WO_DBNAME"] == "workorder"
    assert settings["PATH_TO_DATASETS_DIR"] == str(tmp_path / "datasets")
    _, fmsr = run_scenarios.server_env("granite-test", "http://127.0.0.1:9/v1", "key")
    assert fmsr == {"FMSR_MODEL_ID": "litellm_proxy/granite-test", "LITELLM_BASE_URL": "http://127.0.0.1:9/v1", "LITELLM_API_KEY": "key"}
    paths = run_scenarios.write_launchers(tmp_path, {"wo": None}, settings, fmsr, src=tmp_path / "servers/src")
    text = paths["wo"].read_text(encoding="utf-8")
    assert repr(str(tmp_path / "servers/src")) in text and "from servers.wo.main import main" in text


def test_a_run_needs_the_layer_unless_a_test_names_its_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.delenv("AOB_LAYER_B", raising=False)
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(json.dumps({"id": 1, "type": "IoT", "text": "t", "characteristic_form": "f"}) + "\n")
    common = ["--scenarios", str(scenarios), "--allow-unpinned", "--ids", "1", "--model", "m", "--base-url", "http://127.0.0.1:9/v1",
              "--out", str(tmp_path / "runs")]
    with pytest.raises(SystemExit, match="--layer-b"):
        run_scenarios.main(common)
    with pytest.raises(SystemExit, match="--layer-b"):
        run_scenarios.main([*common, "--server", "iot=/absent.py"])
    assert not (tmp_path / "runs").exists()


PUBLIC = [{"asset_id": "Chiller 6", "timestamp": "2020-06-01T00:00:00", "Chiller 6 Tonnage": "1.0"},
          {"asset_id": "Chiller 9", "timestamp": "2020-06-01T00:00:00", "Chiller 9 Tonnage": "2.0"}]
# Made up rows in the format of the optional file.
MADE_UP = [{"_id": "x", "_rev": "1-y", "asset_id": "Chiller 6", "timestamp": "2020-06-01 00:00:00-05:00", "Chiller 6 Tonnage": "9.0"},
           {"asset_id": "Chiller 6", "timestamp": "2020-04-27T00:15:00", "Chiller 6 Tonnage": "8.0", "Chiller 6 Supply Temperature": "4.0", "Chiller 6 Run Status": True},
           {"asset_id": "Chiller 3", "timestamp": "2021-01-01T00:00:00", "Chiller 3 Tonnage": "7.0"}]


def made_up_file(folder):
    path = Path(folder) / "made_up_rows.json"
    path.write_text(json.dumps(MADE_UP), encoding="utf-8")
    return path


def test_rows_win_on_the_same_asset_and_timestamp_and_only_hash_and_counts_are_recorded(tmp_path):
    records, recovered = layer_b.write_telemetry(tmp_path, list(PUBLIC), made_up_file(tmp_path))
    assert json.loads((tmp_path / "iot.json").read_text()) == records and len(records) == 4
    june = [r for r in records if r["asset_id"] == "Chiller 6" and r["timestamp"] == "2020-06-01T00:00:00"]
    assert len(june) == 1 and june[0]["Chiller 6 Tonnage"] == "9.0" and june[0]["_id"] != "x"
    # the fullest row of an asset comes first under the lowest id, and a status stays a boolean
    assert [r["_id"] for r in records] == sorted(r["_id"] for r in records)
    first = next(r for r in records if r["asset_id"] == "Chiller 6")
    assert first["timestamp"] == "2020-04-27T00:15:00" and first["Chiller 6 Run Status"] is True
    assert recovered["sha256"] == scenario_file.sha256_of(tmp_path / "made_up_rows.json")
    assert recovered["rows"] == 3 and recovered["public_dropped"] == {"Chiller 6": 1}
    assert recovered["assets"] == ["Chiller 3", "Chiller 6"]
    # the April week, the first week of June with both chillers, and the asset alone for Chiller 3
    assert recovered["unlocked_ids"] == [7, 10, 11, 46, 501, 503, 504, 505, 508, 509, 512, 513, 516, 520]
    assert "9.0" not in json.dumps(recovered)


def test_in_a_month_the_rows_touch_no_public_row_stays_and_the_rest_is_counted():
    public = [{"asset_id": "Chiller 6", "timestamp": stamp, "source": "public"} for stamp in
              ("2020-05-31T23:45:00", "2020-06-01T00:07:00", "2020-06-02T11:30:00", "2020-06-03T00:07:00", "2020-06-08T00:00:00", "2020-06-09T00:00:00")]
    public.append({"asset_id": "Chiller 9", "timestamp": "2020-06-01T00:07:00", "source": "public"})
    # rows off the public clock, with a hole of a few hours and a recording days later, all in June
    rows = [{"asset_id": "Chiller 6", "timestamp": stamp, "source": "rows"} for stamp in
            ("2020-06-01T00:00:00", "2020-06-01T20:00:00", "2020-06-02T12:00:00", "2020-06-03T00:15:00", "2020-06-20T00:00:00")]
    merged, counts = layer_b.over_public(public, rows)
    left = sorted(r["timestamp"] for r in merged if r["source"] == "public" and r["asset_id"] == "Chiller 6")
    # every June row goes, also those days after the rows end, and May stays
    assert left == ["2020-05-31T23:45:00"]
    assert counts["public_dropped"] == {"Chiller 6": 5}
    assert counts["spans"]["Chiller 6"] == [["2020-06-01T00:00:00", "2020-06-03T00:15:00", 4], ["2020-06-20T00:00:00", "2020-06-20T00:00:00", 1]]
    assert counts["public_left"]["Chiller 6"] == [["2020-05-31T23:45:00", "2020-05-31T23:45:00", 1]]
    assert counts["public_left"]["Chiller 9"] == [["2020-06-01T00:07:00", "2020-06-01T00:07:00", 1]]  # another asset is untouched


def test_without_the_file_the_telemetry_is_the_public_data(tmp_path):
    records, recovered = layer_b.write_telemetry(tmp_path, list(PUBLIC))
    assert records == PUBLIC and recovered is None


def test_a_rows_file_with_a_repeated_row_is_refused(tmp_path):
    path = tmp_path / "twice.json"
    path.write_text(json.dumps([MADE_UP[1], MADE_UP[1]]), encoding="utf-8")
    with pytest.raises(SystemExit, match="twice"):
        layer_b.write_telemetry(tmp_path, list(PUBLIC), path)


def test_the_named_set_needs_a_folder_with_rows_and_holds_what_they_unlock(tmp_path):
    (tmp_path / layer_b.LAYER_FILE).write_text(json.dumps({"recovered_rows": None}))
    with pytest.raises(SystemExit, match="--recovered-rows"):
        layer_b.recovered_ids(tmp_path)
    (tmp_path / layer_b.LAYER_FILE).write_text(json.dumps({"recovered_rows": {"sha256": "a" * 64, "unlocked_ids": [7, 46]}}))
    ids = layer_b.recovered_ids(tmp_path)
    assert len(ids) == 122 and ids[-2:] == [7, 46] and 43 not in scenario_sets.SETS["recovered"] and len(scenario_sets.SETS["recovered"]) == 139


def test_a_folder_built_with_other_rows_is_refused(tmp_path):
    (tmp_path / layer_b.LAYER_FILE).write_text(json.dumps({"recovered_rows": None}))
    with pytest.raises(SystemExit, match="other rows"):
        layer_b.prepare(tmp_path / "unused", tmp_path, rows_file=made_up_file(tmp_path))
    assert layer_b.prepare(tmp_path / "unused", tmp_path) == tmp_path


def test_prepare_refuses_another_scenario_file(tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n")
    with pytest.raises(SystemExit, match="SHA-256"):
        layer_b.prepare(other, tmp_path / "layer")


@pytest.fixture(scope="module")
def layer_on_couchdb(tmp_path_factory):
    """The built layer loaded by its own loaders into a CouchDB in one process, and the launchers of its six servers."""
    server, url = start_couchdb()
    folder = tmp_path_factory.mktemp("launchers")
    with pytest.MonkeyPatch.context() as patch:
        for key, value in {"COUCHDB_URL": url, "COUCHDB_USERNAME": "admin", "COUCHDB_PASSWORD": "password", **layer_b.server_settings(BUILT, folder)}.items():
            patch.setenv(key, value)
        run_scenarios.reload_couchdb(Path(BUILT))
        shared, fmsr = run_scenarios.server_env("granite-test", "http://127.0.0.1:9/v1", "key")
        yield server, run_scenarios.write_launchers(folder, run_scenarios.parse_servers([]), {**shared, **layer_b.server_settings(BUILT, folder)}, fmsr,
                                                    src=Path(BUILT) / "servers/src")
    server.shutdown()


@real
def test_the_real_build_has_the_merged_scenarios_and_every_csv_the_texts_name():
    built = Path(BUILT)
    assert scenario_file.sha256_of(built / "scenarios.jsonl") == scenario_file.LAYER_SCENARIO_SHA256
    assert layer_b.provenance(built)["servers_commit"] == layer_b.SERVERS_COMMIT
    rows = scenario_file.load_scenarios(built / "scenarios.jsonl", scenario_sets.SETS["runnable"])
    assert all(rows[i].get("characteristic_form") for i in scenario_sets.SETS["runnable"])
    names = {word.strip("'\",.") for i in range(216, 224) for word in rows[i]["text"].split() if ".csv" in word}
    assert names and all((built / "datasets" / name).exists() for name in names)


@real
def test_the_loaders_fill_the_three_databases(layer_on_couchdb):
    server, _ = layer_on_couchdb
    assert len(server.dbs["chiller"]) == layer_b.provenance(BUILT)["iot_records"] and server.dbs["vibration"]
    assert {doc.get("dataset") for doc in server.dbs["workorder"]} >= {"wo_events", "events", "alert_events", "failure_codes"}
    assert sum(doc.get("dataset") == "wo_events" for doc in server.dbs["workorder"]) == 4249


@real
def test_every_server_of_main_0x_lists_its_tools(layer_on_couchdb):
    _, launchers = layer_on_couchdb
    found = {name: {tool["name"] for tool in asyncio.run(_list_tools(path))} for name, path in launchers.items()}
    assert {"get_failure_modes", "get_failure_mode_sensor_mapping"} <= found["fmsr"]
    assert {"get_work_orders", "get_events", "predict_next_work_order"} <= found["wo"]
    assert {"run_tsad", "run_integrated_tsad"} <= found["tsfm"] and found["iot"] and found["utilities"] and found["vibration"]


def call(launchers, server, tool, arguments):
    return json.loads(asyncio.run(_call_tool(launchers[server], tool, arguments)))


@real
def test_a_tool_of_each_family_reads_the_layer_data(layer_on_couchdb):
    _, launchers = layer_on_couchdb
    # the form of scenario 400 expects 33 records of CWC04013 in 2017
    orders = call(launchers, "wo", "get_work_orders", {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2017-12-31"})
    assert orders["total"] == 33
    assert "Chiller 9" in call(launchers, "iot", "assets", {"site_name": "MAIN"})["assets"]
    week = call(launchers, "iot", "history", {"site_name": "MAIN", "asset_id": "Chiller 9", "start": "2020-04-27T00:00:00", "final": "2020-04-28T00:00:00"})
    assert week["total_observations"] and all(r["timestamp"].startswith("2020-04-27") for r in week["observations"])
    assert len(call(launchers, "fmsr", "get_failure_modes", {"asset_name": "Chiller"})["failure_modes"]) == 7


@real
def test_a_tsfm_tool_runs_a_checkpoint_of_main_0x_on_chiller_9(layer_on_couchdb):
    _, launchers = layer_on_couchdb
    result = call(launchers, "tsfm", "run_tsfm_forecasting", {"dataset_path": "chiller9_tsad.csv", "timestamp_column": "Timestamp",
                                                              "target_columns": ["Chiller 9 Condenser Water Flow"]})
    assert Path(result["results_file"]).exists()


@real
def test_made_up_rows_reach_the_real_iot_server_and_win_over_the_public_row(tmp_path, monkeypatch):
    built = tmp_path / "layer"
    shutil.copytree(BUILT, built)
    public = json.loads((built / "iot.json").read_text(encoding="utf-8"))
    public = [r for r in public if r["asset_id"] != "Chiller 3"]
    first = next(r for r in public if r["asset_id"] == "Chiller 6")
    rows = [{"asset_id": "Chiller 3", "timestamp": "2021-01-01T00:00:00", "Chiller 3 Tonnage": "7.0"},
            {**first, "Chiller 6 Tonnage": "123456.0"}]
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    _, recovered = layer_b.write_telemetry(built, public, tmp_path / "rows.json")
    june = sum(r["asset_id"] == "Chiller 6" and r["timestamp"].startswith(first["timestamp"][:7]) for r in public)
    assert recovered["public_dropped"] == {"Chiller 6": june} and recovered["rows"] == 2
    server, url = start_couchdb()
    for key, value in {"COUCHDB_URL": url, "COUCHDB_USERNAME": "admin", "COUCHDB_PASSWORD": "password", **layer_b.server_settings(built, tmp_path)}.items():
        monkeypatch.setenv(key, value)
    run_scenarios.reload_couchdb(built)
    shared, fmsr = run_scenarios.server_env("granite-test", "http://127.0.0.1:9/v1", "key")
    launchers = run_scenarios.write_launchers(tmp_path, {"iot": None}, {**shared, **layer_b.server_settings(built, tmp_path)}, fmsr, src=built / "servers/src")
    assert "Chiller 3" in call(launchers, "iot", "assets", {"site_name": "MAIN"})["assets"]
    start = first["timestamp"]
    found = call(launchers, "iot", "history", {"site_name": "MAIN", "asset_id": "Chiller 6", "start": start, "final": start[:18] + "9"})
    assert [r["Chiller 6 Tonnage"] for r in found["observations"] if r["timestamp"] == start] == ["123456.0"]
    server.shutdown()
