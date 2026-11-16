"""paper_tables/ keeps two rules apart: plays_generation_time, the mode whose episodes tab:thinking reads, chooses the groups of
that table, and charges_thinking, the engine's rule, chooses where the MCP table prints the production loss while thinking. No
token-clock twin stands in a group of the thinking table, and on every twin the two rules part."""
import types

from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ARM_PRESETS, LADDER, TOKEN_PRESETS, TRAINING_PRESET


def test_no_token_clock_twin_stands_in_a_group_of_the_thinking_table_and_the_two_rules_part_on_every_twin():
    assert main_tables.thinking_tables.plays_generation_time is not main_tables.charges_thinking
    grouped = [preset for preset in (*LADDER, TRAINING_PRESET, *ARM_PRESETS) if main_tables.thinking_tables.plays_generation_time(preset)]
    assert grouped and not set(grouped) & set(TOKEN_PRESETS)
    # the control: every twin charges thinking by the engine's rule, so the wider rule would have let it into the table
    assert TOKEN_PRESETS and all(main_tables.charges_thinking(twin) and not main_tables.thinking_tables.plays_generation_time(twin) for twin in TOKEN_PRESETS)


def _presets_of_thinking_table(monkeypatch, planted):
    """The presets that thinking_groups puts into tab:thinking when a token-clock twin is planted among the arm presets of the
    place of the first rung that charges thinking time, read through a cells_of that notes the preset of every line."""
    first = next(preset for preset in LADDER if main_tables.thinking_tables.plays_generation_time(preset))
    monkeypatch.setattr(main_tables.appendix_layer_tables, "ARM_PRESETS", {**ARM_PRESETS, planted: (f"L{LADDER.index(first)}", "a planted twin")})
    seen = []
    main_tables.thinking_groups(types.SimpleNamespace(model="m"), lambda labels, agent, server, preset, variant: seen.append(preset))
    return seen


def test_a_planted_token_clock_twin_does_not_reach_the_thinking_table(monkeypatch):
    twin = sorted(TOKEN_PRESETS)[0]
    assert twin not in _presets_of_thinking_table(monkeypatch, twin)
    # the control: with the engine's wider rule in the place of the clock mode's, the same twin reaches the table
    monkeypatch.setattr(main_tables.thinking_tables, "plays_generation_time", main_tables.charges_thinking)
    assert twin in _presets_of_thinking_table(monkeypatch, twin)
