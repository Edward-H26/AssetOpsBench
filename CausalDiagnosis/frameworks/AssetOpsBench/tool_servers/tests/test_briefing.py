"""The briefing of the AssetOpsBench rows (CausalDiagnosis/frameworks/AssetOpsBench/tool_servers/aob_briefing.py) on the six presets: it names what
AssetOpsBench's own tool descriptions cannot, every tool its row lists, the receipt line from the priced row on and the
time tools on the clock rows only, and no rig tool and no fault rule of the starts. From the repair rules
row on it names the manual, the fix history and the emergency order by the tools that carry them, on
every clock row finish lets the open orders close, and it carries the rig's deterioration sentences
and misdiagnosis sentence of briefing v2.6 on the rows that switch them on."""
import copy
import dataclasses
import json
import re

import pytest

from CausalDiagnosis.utils.paths import preset_path  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader import evidence_kinds as kinds, recordings as records
from CausalDiagnosis.datasets.dataset_loader.evidence_source import RecordingSource
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.profile_registry import register_profile, unregister_profile
from CausalDiagnosis.datasets.tests.conftest import tiny, tiny_data, tiny_recordings, write_recordings  # noqa: F401
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import ALARM_FORM, ALARM_ONCE, EVIDENCE_FORMS, PREDICTION_FORM, briefing, fix_history_sentence
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import AOB_SERVERS, ROWS, TASK_SERVER
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import (DECLARATION_RULE, DETERIORATION_SENTENCES, HOLD_SCORING, REPAIR_SENTENCES, SCORING_SENTENCE,
                                            TIME_SENTENCE, deterioration_sentences, misdiagnosis_sentence)

from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import PRESETS, preset_config

RIG_ONLY = ("read_sensors", "create_work_order", "summarize_changes_since", "get_fix_history", "lookup_fix", "advance_time")
RECEIPT = "Every answer ends with one JSON line"


def text_of(world, row, level=2, **overrides):
    cycles, blocks = world
    config = dataclasses.replace(preset_config(row, **overrides), guidance_level=level)
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(7)
    return briefing(env.public_profile(), config), config


@pytest.mark.parametrize("row", ROWS)
def test_the_briefing_names_the_asset_its_children_the_fields_and_the_citation_and_time_forms(world, row):
    text, config = text_of(world, row)
    for needed in ("site MAIN", "asset_id hyd_1", "hyd_1_cooler, hyd_1_valve, hyd_1_pump and hyd_1_accumulator", "_std appended",
                   "without a sensor list", "+00:00", "work_type CM", "empty wonum", SCORING_SENTENCE, *EVIDENCE_FORMS):
        assert needed in text, needed
    assert (PREDICTION_FORM in text) == config.diagnostics_server
    assert not [name for name in RIG_ONLY if name in text and name not in layer_contract.tools_of(TASK_SERVER, row)]
    assert "off their healthy level" not in text and "off its healthy level" not in text  # the fault rule of the starts stays private


@pytest.mark.parametrize("row", ROWS)
def test_the_briefing_lists_every_tool_of_its_row_and_the_receipt_and_clock_of_the_row(world, row):
    text, config = text_of(world, row)
    for server in (*AOB_SERVERS, TASK_SERVER):
        missing = [name for name in layer_contract.tools_of(server, row) if name not in text]
        assert not missing, (server, missing)
    clock = config.has_clock()
    assert (TIME_SENTENCE in text) == clock and ("get_current_time" in text) == clock and (ALARM_FORM in text) == clock
    assert ("No cycle runs in this episode" in text) == (not clock)
    assert (RECEIPT in text) == (row != "base") and ("costs 1 per cycle" in text) == (row != "base")
    assert ("Lead times" in text) == (row in ROWS[3:]) and ("is a failure" in text) == config.failure_on()


def test_the_levels_add_the_tools_then_a_procedure(world):
    level1, _ = text_of(world, "priced", level=1)
    level2, _ = text_of(world, "priced", level=2)
    level3, _ = text_of(world, "priced", level=3)
    assert "Tools by server" not in level1 and level2.startswith(level1) and level3.startswith(level2)
    assert "Procedure:" in level3 and "Procedure:" not in level2


@pytest.mark.parametrize("row", ROWS)
def test_finish_lets_the_open_orders_close_on_the_clock_rows(world, row):
    text, config = text_of(world, row)
    assert ("the episode is scored after the last close" in text) == config.has_clock()
    assert "open work orders stay open" not in text


def test_the_repair_rules_row_names_the_manual_the_fix_history_and_the_emergency_order_by_their_tools(world):
    text, _ = text_of(world, "repair_rules")
    for needed in ("The repair manual, lookup_fix of the task server,", fix_history_sentence("rig"), REPAIR_SENTENCES["dependencies"],
                   REPAIR_SENTENCES["visits"], REPAIR_SENTENCES["emergency"], "With work_type EM", "lookup_fix(component) (read, 0, free)"):
        assert needed in text, needed
    for server in (*AOB_SERVERS, TASK_SERVER):
        assert not [name for name in layer_contract.tools_of(server, "repair_rules") if name not in text], server
    assert not any(name in text for name in RIG_ONLY if name not in layer_contract.tools_of(TASK_SERVER, "repair_rules"))


def test_the_deterioration_row_adds_the_rigs_hazard_sentence_the_failure_card_and_no_tool_to_the_repair_rules_row(world):
    """The hazard card's sentence of briefing v2.6 follows the time sentence, as on the rig's rows, and points to the
    processes block of get_asset_profile, which carries the rates on this surface too. The failure card joins with big
    layer 5, so its sentence and its notification stand on the deterioration row alone."""
    repair, _ = text_of(world, "repair_rules")
    deterioration, config = text_of(world, "deterioration")
    hazard = DETERIORATION_SENTENCES["hazard"]
    assert layer_contract.tools_of(TASK_SERVER, "deterioration") == layer_contract.tools_of(TASK_SERVER, "repair_rules")
    assert hazard not in repair and f"{TIME_SENTENCE} {hazard}" in deterioration
    assert "is a failure" not in repair and "is a failure" in deterioration and config.failure_on() and config.failure["strict"]


@pytest.mark.parametrize("row", ROWS)
def test_the_rigs_deterioration_and_misdiagnosis_sentences_stand_where_the_rig_puts_them(world, row):
    """Level 1 holds the deterioration sentence of every card the episode switches on, only the hazard card on the six
    presets and only on the deterioration row, and the submit_diagnosis entry of level 2 states the misdiagnosis
    price on every row whose price table declares it, all but the base row."""
    level1, config = text_of(world, row, level=1)
    level2, _ = text_of(world, row)
    assert [s for s in DETERIORATION_SENTENCES.values() if s in level1] == deterioration_sentences(config)
    assert deterioration_sentences(config) == ([DETERIORATION_SENTENCES["hazard"]] if row == "deterioration" else [])
    declared = misdiagnosis_sentence(config.prices)
    assert bool(declared) == (row != "base")
    if declared:
        assert declared not in level1 and f"never after a declaration. {declared} declare_insufficient_evidence" in level2
        assert DECLARATION_RULE not in level1 and DECLARATION_RULE in level2


def test_a_row_without_a_clock_with_a_repair_card_says_when_its_orders_close(world):
    text, config = text_of(world, "priced", prices={**preset_config("priced").prices, "visit": 15})
    assert config.repair_card_on() and "close together before the next call that creates none" in text
    assert "It closes as soon as it is created" not in text


def test_every_count_of_cycles_reads_one_cycle_or_n_cycles(world):
    """A count of cycles takes the rig's n_cycles, so one reads "1 cycle" and two "2 cycles", on the horizon, the history
    window, the lead time and the hold of an emergency order, and none reads "cycle(s)"."""
    rules = preset_config("repair_rules")
    text, _ = text_of(world, "repair_rules", horizon=1, durations={**rules.durations, "read_per_cycle": 2},
                      emergency={**rules.emergency, "lead": 1, "hold": 1})
    for needed in ("a horizon of 1 cycle?", "2 cycles each", "arrives 1 cycle after", "for 1 cycle after its close"):
        assert needed in text, needed
    for old in ("a horizon of 1 cycles", "2 cycle each", "1 cycle(s) after", "for 1 cycles after its close"):
        assert old not in text, old


@pytest.mark.parametrize("row", ROWS)
def test_every_row_keeps_the_opening_and_the_question(world, row):
    """Every row opens with the sentence on the performance monitor and asks the one question at every guidance level, so
    the text of a row never tells whether its start holds a fault."""
    for level in (1, 2, 3):
        text, _config = text_of(world, row, level=level)
        assert text.count("The performance monitor flagged the unit.") == 1, level
        assert text.count("Which condition level is each of its components (") == 1, level
        assert text.count("and which work orders restore the unit, within ") == 1, level
        assert text.count("create the work orders that restore the unit with generate_work_order of the wo server") == 1, level


@pytest.mark.parametrize("row", ROWS)
def test_the_alarm_rule_stands_on_the_rows_that_list_notifications(world, row):
    """Where the receipt of a row lists the notifications, level 2 says once that an alarm is raised once per channel and
    that only a new reading shows a later excursion. No other row and no level 1 text says it."""
    level1, _ = text_of(world, row, level=1)
    level2, _ = text_of(world, row, level=2)
    listed = "notifications since the previous call of" in level2
    assert listed == (row not in ("base", "priced"))
    assert (level1.count(ALARM_ONCE), level2.count(ALARM_ONCE)) == (0, 1 if listed else 0)
    assert "once per channel" in ALARM_ONCE and "summarize_changes_since" not in ALARM_ONCE


@pytest.mark.parametrize("row", ROWS)
def test_the_scoring_of_a_hold_follows_the_emergency_clause_on_the_rows_that_offer_it(world, row):
    """Where a row offers the emergency order (work_type EM), level 2 says once, right after the emergency clause, that a
    component an emergency order still holds at the end is scored at its level before that order. No other row and no level
    1 text says it."""
    level1, _ = text_of(world, row, level=1)
    level2, config = text_of(world, row, level=2)
    offered = "With work_type EM the order is an emergency order" in level2
    assert offered == ("lookup_fix" in layer_contract.tools_for(TASK_SERVER, config) and config.emergency_on())
    assert (level1.count(HOLD_SCORING), level2.count(HOLD_SCORING)) == (0, 1 if offered else 0)
    assert (f"after its close. {HOLD_SCORING}" in level2) == offered  # the sentence closes the emergency clause


def test_the_rows_offer_the_emergency_order_on_some_rows_and_not_on_others(world):
    """The ladder holds rows with and without the emergency clause, so the test above reads both cases."""
    offered = {row: "With work_type EM" in text_of(world, row, level=2)[0] for row in ROWS}
    assert set(offered.values()) == {True, False}, offered


# the five sentences that an asset file may state, for a tiny variant whose kind states no loop
STATED = {"question": "Which level does each of the {components} of the bench stand at, within {limits}?",
          "opening": "The bench monitor flagged a deviation.",
          "scoring": "The diagnosis is scored at the cycle it is submitted.",
          "level_meaning": "A larger belt level is a tighter belt.",
          "readings": "A reading is one summary of a recorded cycle of the bench."}


def tiny_row(tiny, row):
    """The configuration of a row of the environment table on the tiny asset, its preset composed as the drivers compose it."""
    return layers.compose({**json.loads(preset_path(row).read_text(encoding="utf-8")), "asset_id": tiny.asset_id})


@pytest.mark.parametrize("row", ROWS)
def test_the_briefing_of_a_tiny_episode_names_the_bench_and_never_the_rig_at_every_level(tiny, tiny_recordings, row):
    cycles = records.load_cycles(tiny_recordings, tiny)
    config = tiny_row(tiny, row)
    env = RecordedRigEnv(cycles, records.build_blocks(cycles), config)
    env.reset(0)
    for level in (1, 2, 3):
        text = briefing(env.public_profile(), dataclasses.replace(config, guidance_level=level))
        assert not re.search(r"\b(rig|cooler|valve|pump|accumulator)\b", text), (row, level)
        assert "tiny_1_belt" in text and "bench" in text


def test_a_tiny_asset_that_states_its_five_sentences_shows_each_once(tiny_data, tmp_path):
    kind = kinds.register_kind(kinds.EvidenceKind("tiny_stated", record_keys=("label_keys", "stable_flag_stable"), check=records.record_problems,
                                                  loader=records.load_recordings, source=RecordingSource))
    data = copy.deepcopy(tiny_data)
    data["evidence_kind"] = kind.name
    data["text"].update(STATED)
    profile = register_profile(AssetProfile.from_json(data))
    try:
        cycles = records.load_cycles(write_recordings(tmp_path / "stated.jsonl", profile), profile)
        config = tiny_row(profile, "real_time")
        env = RecordedRigEnv(cycles, records.build_blocks(cycles), config)
        env.reset(0)
        text = briefing(env.public_profile(), dataclasses.replace(config, guidance_level=2))
        for key in ("opening", "scoring", "level_meaning", "readings"):
            assert text.count(STATED[key]) == 1, key
        assert text.count("Which level does each of the fan, belt, bearing of the bench stand at, within a budget of ") == 1
        assert "{components}" not in text and "{limits}" not in text and "Every reading is a recorded cycle" not in text
    finally:
        unregister_profile(profile.asset_id)
        kinds.unregister_kind(kind.name)

@pytest.mark.parametrize("row", ["base", "priced"])
def test_a_nested_sensor_group_names_the_group_inside_it_and_lists_no_field_twice(world, row):
    """Every field of the telemetry stands once among the sensor groups, and a group that holds every field of an earlier one
    is named by that group and its other fields, from which each group's fields follow. The control: listing every field of
    every group again fails on the nested groups of the rig."""
    text, _config = text_of(world, row)
    cycles, blocks = world
    env = RecordedRigEnv(cycles, blocks, preset_config(row))
    env.reset(7)
    groups = env.public_profile()["sensor_groups"]
    lines = {line.split(" ")[3]: line for line in text.splitlines() if line.startswith("  - ") and line.split(" ")[3] in groups}
    assert set(lines) == set(groups)
    fields = {}
    for group, line in lines.items():
        returned = line.split("returns ", 1)[1]
        inner = re.match(r"the fields of (\S+) and (.*)", returned)
        fields[group] = (set(fields[inner.group(1)]) | set(inner.group(2).split(", "))) if inner else set(returned.split(", "))
        assert fields[group] == set(groups[group]), group
    listed = [f for line in lines.values() for f in line.split("returns ", 1)[1].removeprefix("the fields of ").split(", ")]
    assert sorted(f.split(" and ")[-1] for f in listed if f not in groups) == sorted({f for g in groups.values() for f in g})
