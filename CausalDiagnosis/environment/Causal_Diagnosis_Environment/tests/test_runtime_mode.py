"""Behavioral contracts for explicit modes; no LLM or hidden-truth agent policy."""
import dataclasses
import json

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.runtime_mode import apply
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
from CausalDiagnosis.experiments.episode_runners.run_episodes import build_parser, configure
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_tool_contract import tools_for


@pytest.mark.parametrize("number", range(1, 9))
def test_eight_scenarios_have_explicit_modes_and_identical_public_tools_with_or_without_advice(number):
    _, config = load_config(f"environment_test_scn{number}")
    expected = "static" if number in (1, 2, 7, 8) else "dynamic"
    assert config.environment_mode == expected
    other = dataclasses.replace(config, diagnostics_server=False)
    for server in ("iot", "fmsr", "wo", "task"):
        assert tools_for(server, config) == tools_for(server, other)
    assert ("wait_for_notification" in tools_for("task", config)) == (expected == "dynamic")
    if expected == "static":
        assert config.clock_mode == "declared" and not config.has_clock()
        assert config.failure["patience"] == 0 and config.schedule["mode"] == "none"
    else:
        assert config.clock_mode == "generation_time" and config.has_clock()


def test_legacy_config_keeps_serialization_and_wait_semantics(world):
    _, legacy = load_config("base")
    assert legacy.environment_mode is None
    assert "environment_mode" not in legacy.to_json()
    assert EpisodeConfig.from_json(legacy.to_json()).to_json() == legacy.to_json()
    env = HydraulicRigEnv(*world, legacy);env.reset(7)
    env.step({"type": "wait_for_notification", "timeout": 1})
    assert env.time == 1


@pytest.mark.parametrize("agent", ["causal", "plan_execute", "agent_as_tool"])
@pytest.mark.parametrize("advice", [True, False])
def test_one_cli_parameter_sets_the_same_environment_for_every_agent(agent, advice):
    args = ["--config", "environment_test_scn6", "--agent", agent, "--environment-mode", "static"]
    if not advice:
        args += ["--no-diagnostics"]
    _, config, _ = configure(build_parser().parse_args(args))
    assert config.environment_mode == "static" and not config.has_clock()
    assert config.start["combination"] == [100, 90, 1, 130]
    assert config.schedule["mode"] == "none"
    assert config.diagnostics_server == advice


@pytest.mark.parametrize("mode,clock", [("static", "generation_time"), ("dynamic", "declared")])
def test_conflicting_clock_override_is_rejected(mode, clock):
    args = build_parser().parse_args(["--environment-mode", mode, "--clock-mode", clock])
    with pytest.raises(SystemExit, match="requires --clock-mode"):
        configure(args)


def test_inconsistent_serialized_static_mode_is_rejected():
    _, base = load_config("static")
    static = apply(base, "static")
    data = static.to_json();data["durations"]["lead_time"]["pump"] = 3
    with pytest.raises(ValueError, match="environment_mode static conflicts"):
        EpisodeConfig.from_json(data)


def test_static_freezes_idle_time_rejects_wait_and_settles_repairs_without_advancing(world):
    _, config = load_config("environment_test_scn6")
    config = apply(config, "static")
    config = dataclasses.replace(config, start={"combination": [100, 100, 2, 130], "entry_offset": 5})
    env = HydraulicRigEnv(*world, config);env.reset(1)
    initial = list(env.hidden)
    env.step({"type": "no_call"}, think_cycles=30)
    assert env.time == 0 and list(env.hidden) == initial
    refused = env.step({"type": "wait_for_notification", "timeout": 10})
    assert "static environment" in json.dumps(refused) and env.time == 0
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(config.components, initial)), "evidence_ids": []})
    env.step({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []})
    # Same-cycle consecutive orders form a visit; the next non-order settles it.
    env.step({"type": "get_asset_profile"})
    assert env.time == 0 and not env.public_work_orders()
    assert list(env.hidden) == [100, 100, 0, 130]
    assert not any(e["event"] == "scheduled_drop" for e in env.log)


def test_static_still_enforces_repair_dependencies(world):
    _, config = load_config("environment_test_scn3")
    config = apply(config, "static")
    env = HydraulicRigEnv(*world, config);env.reset(0)
    truth = dict(zip(config.components, env.initial))
    env.step({"type": "submit_diagnosis", "diagnosis": truth, "evidence_ids": []})
    env.step({"type": "create_work_order", "component": "pump", "prediction_id": "", "evidence_ids": []})
    env.step({"type": "get_asset_profile"})
    assert not env.public_work_orders() and env.time == 0
    assert env.hidden[config.components.index("pump")] == 1
    assert any(e["event"] == "order_ineffective" for e in env.log)


def test_dynamic_idle_turns_advance_time_and_execute_the_scenario_schedule(world):
    _, config = load_config("environment_test_scn6")
    env = HydraulicRigEnv(*world, config);env.reset(1)
    env.step({"type": "no_call"}, think_cycles=4)
    assert env.time == 4 and list(env.hidden) == [20, 90, 1, 130]
    env.step({"type": "no_call"}, think_cycles=10)
    assert env.time == 14 and list(env.hidden) == [3, 90, 1, 130]
    assert env.clock.downtime_total > 0


def test_static_mode_wins_over_duration_values_in_a_price_table(tmp_path):
    _, legacy = load_config("static")
    table = tmp_path / "priced_durations.json"
    table.write_text(json.dumps({"prices": legacy.prices, "durations": legacy.durations}))
    args = build_parser().parse_args(["--config", "environment_test_scn1", "--environment-mode", "static", "--prices", str(table)])
    _, config, _ = configure(args)
    assert config.environment_mode == "static" and not config.has_clock()
    assert all(value == 0 for value in config.durations["lead_time"].values())


def test_dynamic_override_of_static_scenario_preserves_initial_state():
    args = build_parser().parse_args(["--config", "environment_test_scn7", "--environment-mode", "dynamic"])
    _, config, _ = configure(args)
    assert config.environment_mode == "dynamic" and config.has_clock()
    assert config.start["combination"] == [100, 100, 1, 130]
    assert config.schedule["mode"] == "none"  # Changing time semantics does not invent a fault event.


def test_grid_generator_keeps_three_paired_modes_and_selectors(tmp_path):
    from CausalDiagnosis.cluster_jobs.scenario_tests.write_grids import write
    write(tmp_path)
    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 8
    for number in range(1, 9):
        lines = (tmp_path / f"local_environment_scn{number}.txt").read_text().splitlines()
        assert len(lines) == 3
        configs = [configure(build_parser().parse_args(line.split()))[1] for line in lines]
        assert [c.diagnostics_server for c in configs] == [True, False, True]
        a, b, c = [d.to_json() for d in configs]
        b["diagnostics_server"] = True
        assert a == b == c
        if number in (1, 2):
            assert all(f"--scenario-ids {2007 if number == 1 else 2004}" in line for line in lines)


@pytest.mark.parametrize("number", [1, 6])
def test_reference_label_configuration_can_be_built_for_explicit_modes(number):
    from CausalDiagnosis.environment.reference_policies.diagnosability_label import static_config, rules_of
    _, config = load_config(f"environment_test_scn{number}")
    reference = static_config(config)
    assert reference.environment_mode is None and reference.clock_mode == "declared"
    assert rules_of(config)["environment_mode"] == config.environment_mode
    assert "environment_mode" not in rules_of(reference)


def test_cli_override_preserves_the_scenario_events_before_applying_the_mode(tmp_path):
    from CausalDiagnosis.utils.paths import preset_path
    raw = json.loads(preset_path("environment_test_scn6").read_text())
    raw["environment_mode"] = "static"
    path = tmp_path / "static_with_declared_events.json"
    path.write_text(json.dumps(raw))
    _, static = load_config(str(path))
    assert static.schedule["mode"] == "none"
    args = build_parser().parse_args(["--config", str(path), "--environment-mode", "dynamic"])
    _, dynamic, _ = configure(args)
    assert dynamic.schedule["mode"] == "seeded" and dynamic.schedule["first_drop"] == [3, 3]
