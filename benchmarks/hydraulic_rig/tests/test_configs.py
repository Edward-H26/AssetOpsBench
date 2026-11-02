"""The configuration presets and price tables of benchmarks/hydraulic_rig/configs/, and the launcher writer."""
import importlib.util
import json

import pytest

from common import (CAUSAL_SERVER, CAUSAL_TOOLS, CONFIGS, DEFAULT_RANKING_MODEL, PRICE_TABLES, RIG_SERVER, RIG_TOOLS, apply_price_table, contract_tools,
                    load_config, load_price_table, write_launchers)

needs_config = pytest.mark.skipif(importlib.util.find_spec("envs.hydraulic_rig.config") is None, reason="waiting for env-core's config.py")
PRESETS = sorted(p.stem for p in CONFIGS.glob("*.json"))


def test_the_presets_and_tables_of_the_plan_exist():
    assert {"static", "timed", "pilot"} <= set(PRESETS)
    assert {p.stem for p in PRICE_TABLES.glob("*.json")} == {"placeholder", "full_tier_high", "sme"}
    for path in CONFIGS.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == path.stem, path


def test_price_tables_share_the_fields_and_the_template_is_refused():
    placeholder = load_price_table("placeholder")
    high = load_price_table("full_tier_high")
    assert set(placeholder["prices"]) == set(high["prices"]) == {"read", "parts", "repair_charge"}
    assert set(placeholder["prices"]["read"]) == {"tier1", "tier2", "tier3"} and set(placeholder["prices"]["parts"]) == {"cooler", "valve", "pump", "accumulator"}
    assert high["prices"]["read"]["tier3"] > placeholder["prices"]["read"]["tier3"]
    assert high["prices"]["parts"] == placeholder["prices"]["parts"]
    with pytest.raises(SystemExit, match="empty values"):
        load_price_table("sme")
    with pytest.raises(SystemExit, match="unknown price table"):
        load_price_table("no_such_table")


def test_the_sme_template_lists_every_field_with_an_empty_value():
    sme = json.loads((PRICE_TABLES / "sme.json").read_text(encoding="utf-8"))
    placeholder = json.loads((PRICE_TABLES / "placeholder.json").read_text(encoding="utf-8"))

    def shape(value):
        return {k: shape(v) for k, v in value.items()} if isinstance(value, dict) else None

    assert shape(sme["prices"]) == shape(placeholder["prices"])
    assert set(sme) >= {"prices", "durations", "downtime_rate"}
    assert json.dumps(sme["prices"]).count("null") == 8


@needs_config
@pytest.mark.parametrize("name", PRESETS)
def test_every_preset_loads_over_the_defaults_and_validates(name):
    preset_name, config = load_config(name)
    assert preset_name == name
    assert config.to_json() == type(config).from_json(config.to_json()).to_json()
    preset = json.loads((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))
    for key, value in preset.items():
        if key not in ("name", "description"):
            assert config.to_json()[key] == value, key


@needs_config
def test_presets_differ_only_where_the_plan_says():
    static, timed, pilot = (load_config(n)[1].to_json() for n in ("static", "timed", "pilot"))
    assert static["schedule"]["mode"] == "none" and static["external_events"]["mode"] == "none" and static["anchors"] is None
    assert timed["schedule"]["mode"] == "seeded" and timed["external_events"]["mode"] == "seeded"
    assert {k for k in static if static[k] != timed[k]} == {"schedule", "external_events"}
    assert pilot["anchors"] == {"date0": "2022-07-01"} and pilot["start"] is None and pilot["question"] is None


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
    paths = write_launchers(tmp_path)
    assert set(paths) == {RIG_SERVER, CAUSAL_SERVER}
    rig = (tmp_path / "hydraulic_rig_server.py").read_text(encoding="utf-8")
    causal = (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert "with_name('episode.json'))" in rig and "with_name('released.json'), ranking_model=" in causal
    assert f"ranking_model={DEFAULT_RANKING_MODEL!r}" in causal and "abstention" not in causal  # the recorded model is the served one
    write_launchers(tmp_path, ranking_model="discriminant_urn")
    causal = (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert "ranking_model='discriminant_urn')" in causal and "abstention" not in causal
    write_launchers(tmp_path, abstention=("fixed_cut", 0.9))
    causal = (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert "abstention_rule='fixed_cut', abstention_cut=0.9)" in causal and f"ranking_model={DEFAULT_RANKING_MODEL!r}" in causal
    write_launchers(tmp_path, abstention=("decision", 0.95))
    assert "abstention" not in (tmp_path / "causal_diagnosis_server.py").read_text(encoding="utf-8")
    assert write_launchers(tmp_path / "rig_only", servers=(RIG_SERVER,)) .keys() == {RIG_SERVER} if (tmp_path / "rig_only").mkdir() is None else True
