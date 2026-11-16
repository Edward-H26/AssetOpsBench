"""The configuration presets and price tables of CausalDiagnosis/configs/, and the launcher writer."""
import importlib.util
import json

import pytest

from CausalDiagnosis.utils.paths import preset_files, preset_path  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, CAUSAL_TOOLS, METERED_SERVERS, RIG_SERVER, RIG_TOOLS, TASK_SERVER, contract_tools, registered_servers
from CausalDiagnosis.experiments.episode_runners.run_configuration import CONFIGS, PRICE_TABLES, apply_price_table, load_config, load_price_table
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL, write_launchers

needs_config = pytest.mark.skipif(importlib.util.find_spec("CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config") is None, reason="CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config is not importable")
PRESETS = sorted(p.stem for p in preset_files(CONFIGS).values())


def test_the_presets_and_tables_of_the_plan_exist():
    assert {"static", "timed", "anchored"} <= set(PRESETS)
    assert {p.stem for p in PRICE_TABLES.glob("*.json")} == {"default", "full_tier_high", "sme"}
    for path in sorted({*CONFIGS.rglob("*.json"), *preset_files(CONFIGS).values()}):  # the presets a framework brings too
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == path.stem, path


def test_price_tables_share_the_fields_and_the_template_is_refused():
    default = load_price_table("default")
    high = load_price_table("full_tier_high")
    assert set(default["prices"]) == set(high["prices"]) == {"read", "parts", "repair_charge", "visit", "misdiagnosis", "fix_kind"}
    assert set(default["prices"]["read"]) == {"tier1", "tier2", "tier3"} and set(default["prices"]["parts"]) == {"cooler", "valve", "pump", "accumulator"}
    assert high["prices"]["read"]["tier3"] > default["prices"]["read"]["tier3"]
    assert high["prices"]["parts"] == default["prices"]["parts"]
    with pytest.raises(SystemExit, match="empty values"):
        load_price_table("sme")
    with pytest.raises(SystemExit, match="unknown price table"):
        load_price_table("no_such_table")


def test_the_sme_template_lists_every_field_with_an_empty_value():
    sme = json.loads((PRICE_TABLES / "sme.json").read_text(encoding="utf-8"))
    default = json.loads((PRICE_TABLES / "default.json").read_text(encoding="utf-8"))

    def shape(value):
        return {k: shape(v) for k, v in value.items()} if isinstance(value, dict) else None

    assert shape(sme["prices"]) == shape(default["prices"])
    assert set(sme) >= {"prices", "durations", "downtime_rate"}
    assert json.dumps(sme["prices"]).count("null") == 14


@needs_config
@pytest.mark.parametrize("name", PRESETS)
def test_every_preset_loads_over_the_defaults_and_validates(name):
    preset_name, config = load_config(name)
    assert preset_name == name
    assert config.to_json() == type(config).from_json(config.to_json()).to_json()
    preset = json.loads(preset_path(name).read_text(encoding="utf-8"))
    for key, value in preset.items():
        if key not in ("name", "description", "layers", "switches"):  # the base fields a composition sets
            assert config.to_json()[key] == value, key


@needs_config
def test_presets_differ_only_where_the_plan_says():
    static, timed, anchored = (load_config(n)[1].to_json() for n in ("static", "timed", "anchored"))
    assert static["schedule"]["mode"] == "none" and static["anchors"] is None
    assert timed["schedule"]["mode"] == "seeded"
    assert {k for k in static if static[k] != timed[k]} == {"schedule"}
    assert anchored["anchors"] == {"date0": "2022-07-01"} and anchored["start"] is None and anchored["question"] is None


@needs_config
def test_a_price_table_replaces_the_prices_and_keeps_the_rest():
    _, config = load_config("static")
    high = apply_price_table(config, load_price_table("full_tier_high"))
    assert high.to_json()["prices"] == load_price_table("full_tier_high")["prices"]
    assert {k for k in config.to_json() if config.to_json()[k] != high.to_json()[k]} == {"prices"}
    with pytest.raises(SystemExit, match="unknown config preset"):
        load_config("no_such_preset")


def test_contract_tools_fall_back_to_the_plan_without_the_contract_file():
    assert set(contract_tools(RIG_SERVER)) >= RIG_TOOLS or contract_tools(RIG_SERVER) == RIG_TOOLS
    assert set(contract_tools(CAUSAL_SERVER)) >= CAUSAL_TOOLS or contract_tools(CAUSAL_SERVER) == CAUSAL_TOOLS


def test_launchers_bind_each_server_to_its_file_and_always_name_the_ranking_model(tmp_path):
    paths = write_launchers(tmp_path, left_out=(3, 100, 0, 130))
    assert set(paths) == {RIG_SERVER, CAUSAL_SERVER}
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths.values())  # a metered launcher holds a password
    rig = (tmp_path / "hydraulic_rig_server.py").read_text(encoding="utf-8")
    causal = (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    # every launcher passes the readings of its own start last, the start of the process for the thinking clock
    assert "with_name('episode.json'), started=_STARTED)" in rig and "with_name('released.json'), ranking_model=" in causal
    assert f"ranking_model={DEFAULT_RANKING_MODEL!r}" in causal and "abstention" not in causal  # the recorded model is the served one
    # the start's block, which every fit of the diagnostics server leaves out, and no other hidden value
    assert "left_out=(3, 100, 0, 130), started=_STARTED)" in causal and "left_out" not in rig
    assert "entry_offset" not in causal + rig and "start_combination" not in causal + rig
    write_launchers(tmp_path, ranking_model="discriminant_urn")
    causal = (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert "ranking_model='discriminant_urn'," in causal and "abstention" not in causal
    write_launchers(tmp_path, abstention=("fixed_cut", 0.9))
    causal = (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert "abstention_rule='fixed_cut', abstention_cut=0.9, started=_STARTED)" in causal and f"ranking_model={DEFAULT_RANKING_MODEL!r}," in causal
    write_launchers(tmp_path, abstention=("decision", 0.95))
    assert "abstention" not in (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert write_launchers(tmp_path / "rig_only", servers=(RIG_SERVER,)) .keys() == {RIG_SERVER} if (tmp_path / "rig_only").mkdir() is None else True


def test_launchers_of_the_metered_servers_name_the_server_and_the_store(tmp_path):
    assert registered_servers("rig", False) == (RIG_SERVER,) and registered_servers("rig", True) == (RIG_SERVER, CAUSAL_SERVER)
    assert registered_servers("aob", False) == METERED_SERVERS == ("iot", "fmsr", "wo", TASK_SERVER)
    assert registered_servers("aob", True) == (*METERED_SERVERS, CAUSAL_SERVER)
    paths = write_launchers(tmp_path, registered_servers("aob", True), store={"kind": "memory"})
    assert set(paths) == {*METERED_SERVERS, CAUSAL_SERVER}
    for server in METERED_SERVERS:
        launcher = (tmp_path / f"{server}_server.py").read_text(encoding="utf-8")
        assert "from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import main" in launcher
        assert f"with_name('episode.json'), server={server!r}, store={{'kind': 'memory'}}, started=_STARTED)" in launcher
    assert "from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import main" in (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    # the contract of the AssetOpsBench rows lists the metered servers' tools, hidden tools left out, the task server's by row
    assert {"history", "latest_reading", "sites"} <= contract_tools("iot") and "sensor_stats" not in contract_tools("iot")
    assert contract_tools("fmsr") == {"get_failure_modes"} and "generate_work_order" in contract_tools("wo")
    assert contract_tools(TASK_SERVER, "base") == {"get_asset_profile", "summarize_changes_since", "submit_diagnosis",
                                                   "declare_insufficient_evidence", "finish"}
    assert contract_tools(TASK_SERVER, "real_time") == contract_tools(TASK_SERVER, "base") | {"get_current_time", "wait_for_notification"}
    assert contract_tools(TASK_SERVER) >= contract_tools(TASK_SERVER, "real_time")  # without a row, every row's tools
    # a configuration lists the tools its big layers list, on a ladder preset those of its row
    for row in ("base", "real_time", "deterioration"):
        assert contract_tools(TASK_SERVER, config=load_config(row)[1]) == contract_tools(TASK_SERVER, row)


def test_a_preset_of_a_dataset_folder_is_found_by_name_and_a_name_in_two_places_stops(tmp_path, monkeypatch):
    """preset_files under CONFIGS reads datasets/<asset_id>/presets/ of every asset beside configs/ and the folders a framework
    declares, so a port brings its own presets; a name that two files share stops the lookup."""
    from CausalDiagnosis.utils import paths

    configs, datasets = tmp_path / "configs", tmp_path / "datasets"
    (configs / "controls").mkdir(parents=True)
    (datasets / "made_up" / "presets").mkdir(parents=True)
    (configs / "controls" / "plain.json").write_text(json.dumps({"name": "plain"}), encoding="utf-8")
    planted = datasets / "made_up" / "presets" / "planted.json"
    planted.write_text(json.dumps({"name": "planted"}), encoding="utf-8")
    monkeypatch.setattr(paths, "CONFIGS", configs)
    monkeypatch.setattr(paths, "DATASETS", datasets)
    assert paths.preset_path("planted") == planted and paths.preset_path("plain") == configs / "controls" / "plain.json"
    assert "anchored" in paths.preset_files()  # the framework's preset beside them
    assert "planted" not in paths.preset_files(configs / "controls")  # a folder other than CONFIGS reads that folder alone
    (configs / "controls" / "planted.json").write_text(json.dumps({"name": "planted"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="two presets share the name planted"):
        paths.preset_files()
