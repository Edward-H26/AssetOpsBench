"""The tiny synthetic asset of three components with two, three and two levels, whose recordings a seeded Gaussian
generator writes, so that the generic code of the asset layer runs on an asset that shares no name, group or statistic
with hyd_1: validation, the round trip, the configuration its cards build, the registry, the record loader, the data file and
recordings that a configuration of the asset names, episodes of the rig engine served from those recordings and scored
by the verifier on the tiny tables, the tool table, the action server and the briefing of a tiny episode, the diagnostics
model fitted on the tiny recordings, and one tiny episode that the rule only agent plays over both servers."""
from pathlib import Path
import asyncio
import copy
import dataclasses
import json
import re

import pytest

from CausalDiagnosis.datasets.dataset_loader import recordings as records, rule_cards as rules
from CausalDiagnosis.utils.json_digest import canonical
from CausalDiagnosis.datasets.dataset_loader.evidence_source import EvidenceSource
from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, register_profile, unregister_profile
from CausalDiagnosis.datasets.dataset_loader.profile_validation import validate
from CausalDiagnosis.datasets.dataset_loader.ground_truth import LevelsTruth, TruthModel
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import episode_config as engine_config
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import environment_factory as world
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify
from llm import LLMResult
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import tool_contract as contract
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import build_server

from CausalDiagnosis.datasets.tests.conftest import ROOT, TINY_FILE


def test_a_tiny_profile_validates_round_trips_and_lists_every_problem(tiny_data):
    assert validate(tiny_data) == []
    profile = AssetProfile.from_json(tiny_data)
    assert canonical(profile.to_json()) == canonical(tiny_data)
    broken = copy.deepcopy(tiny_data)
    broken["healthy"]["belt"] = 60
    broken["observation_kinds"]["full"]["channels"].append("X9")
    broken["failure_modes"][1]["level"] = 3
    broken["knowledge"]["indicates"]["X9"] = ["fan"]
    broken["text"]["example_channel"] = "X9"
    assert validate(broken) == [
        "the healthy level of belt must be one of its levels",
        "observation kind full names channels ['X9'] that channels does not declare",
        "failure_modes[1] gives the level 3, not a level of fan",
        "knowledge.indicates must map declared channels to lists of components",
        "text.example_channel must be a declared channel",
    ]
    late = copy.deepcopy(tiny_data)
    next(c for c in late["rules"] if c["kind"] == "dependencies")["parameters"]["priority"] = {"belt": 3, "fan": 2, "bearing": 1}
    assert validate(late) == ["the configuration built from the profile: "
                              "dependencies ['belt'] of bearing need a smaller priority number than bearing"]


def test_a_tiny_profile_without_cards_validates_with_every_rule_off(tiny_data):
    tiny_data["rules"] = []
    assert validate(tiny_data) == []
    config = AssetProfile.from_json(tiny_data).default_config()
    assert config.dependencies == {} and config.priority == {"fan": 1, "belt": 1, "bearing": 1} and config.prices["visit"] == 0
    assert not config.emergency_on() and not config.failure_on()
    assert config.downtime_rate == {"fan": {"0": 0, "1": 0}, "belt": {"50": 0, "75": 0, "100": 0}, "bearing": {"1": 0, "2": 0},
                                    engine_config.STOPPED: 0}


def test_the_engine_checks_sensor_groups_against_the_channels_of_the_profile_of_the_asset_id(tiny_data):
    """The tiny profile validates and builds its configuration without being registered, since both serve it
    for their length and leave the registry as it was. A configuration of a registered tiny profile refuses a hyd_1 channel by the tiny channels, and a
    configuration of an asset id without a profile is refused by that id."""
    assert validate(tiny_data) == []
    with pytest.raises(KeyError):
        get_profile(tiny_data["asset_id"])
    profile = register_profile(AssetProfile.from_json(tiny_data))
    try:
        fields = profile.config_fields()
        fields["sensor_groups"]["basic"] = ["PS1"]
        with pytest.raises(ValueError) as refused:
            engine_config.EpisodeConfig(**fields)
        assert f"sensor group basic must be a non-empty list of distinct channels among {list(tiny_data['channels'])}" in str(refused.value)
    finally:
        unregister_profile(profile.asset_id)
    with pytest.raises(ValueError, match="asset_id 'no_such_asset' names no registered or bundled asset profile"):
        engine_config.EpisodeConfig(asset_id="no_such_asset")


def test_the_tiny_configuration_follows_its_tables_and_its_cards(tiny, tiny_data):
    config = tiny.default_config()
    assert (config.asset_id, config.components, config.healthy) == ("tiny_1", ["fan", "belt", "bearing"], tiny_data["healthy"])
    assert config.sensor_groups == {"basic": ["T1"], "full": ["T1", "V1", "I1"]}
    assert list(config.prices) == ["read", "parts", "repair_charge", "visit", "fix_kind", "misdiagnosis"]
    assert config.prices["visit"] == config.prices["misdiagnosis"] == 4
    assert list(config.downtime_rate) == ["fan", "belt", "bearing", engine_config.STOPPED] and config.downtime_rate[engine_config.STOPPED] == 3
    assert config.depends_on("bearing") == ["belt"] and config.fault_rule == {"min_faulty": 1, "max_faulty": 1}
    assert [c for c in tiny.combinations() if config.within_fault_rule(c)] == [(0, 50, 1), (0, 75, 1), (0, 100, 2), (1, 100, 1)]
    assert config.order_price("bearing", "emergency") == (15 + 5) * 1.5 and config.level_rate("belt", 50) == 3
    preset_defaults = {f.name: f.default for f in dataclasses.fields(engine_config.EpisodeConfig)
                       if f.name in ("guidance_level", "verifier_scope", "allowed")}
    assert {name: getattr(config, name) for name in preset_defaults} == preset_defaults
    cards = tiny_data["rules"]
    assert [rules.read_card(config.to_json(), rules.kind_of(c["kind"])) for c in cards] == [c["parameters"] for c in cards]


def test_the_registry_serves_the_tiny_profile_only_while_it_is_registered(tiny):
    assert get_profile("tiny_1") is tiny
    unregister_profile("tiny_1")
    with pytest.raises(KeyError, match="no asset profile 'tiny_1'"):
        get_profile("tiny_1")
    register_profile(tiny)
    assert get_profile("tiny_1") is tiny


def test_the_tiny_lookups_and_failure_modes_come_from_its_own_tables(tiny, tiny_recordings):
    """The channel, code and registry lookups that the engine, the servers and the drivers read from a profile answer from
    the tiny tables, and a tiny episode serves the tiny failure modes and the tiny registry entry."""
    asked = ("T1", tiny.field_name("V1"), f"{tiny.field_name('I1')}_mean", "XX_mean")
    assert [tiny.channel_of(n) for n in asked] == ["T1", "V1", "I1", None]
    faults = ["fan_stalled", "belt_worn", "belt_slipping", "bearing_dry"]
    assert [m["code"] for m in tiny.fault_modes()] == faults
    assert (tiny.level_code("belt", 75), tiny.level_code("belt", 60), tiny.level_code("fan", None)) == ("belt_worn", None, None)
    assert tiny.registry_entry() == {"asset_id": "tiny_1", **tiny.text["fallback"]}
    env = _tiny_env(tiny, tiny_recordings)
    env.reset(0)
    assert env.public_profile()["asset"] == tiny.registry_entry()
    envelope, _cost, _done, _info = env.step({"type": "get_failure_modes", "component": ""})
    assert [m["code"] for m in envelope["result"]["failure_modes"]] == faults
    assert {m["evidence_id"] for m in envelope["result"]["failure_modes"]} == {f"fmea/tiny_generator/{code}" for code in faults}


def test_the_tiny_recordings_load_into_one_contiguous_block_per_combination(tiny, tiny_recordings):
    cycles = records.load_cycles(tiny_recordings, tiny)
    assert len(cycles) == 12 * 11 and sum(c.stable for c in cycles) == 12 * 10
    blocks = records.build_blocks(cycles)
    assert list(blocks) == list(tiny.combinations()) and all(len(block) == 10 for block in blocks.values())
    assert list(records.group_features(cycles[0], ["T1"], tiny)) == ["Temperature_T1_mean", "Temperature_T1_std"]
    served = tuple(f"_{s}" for s in tiny.statistics["served"])
    assert list(records.group_features(cycles[0], ["T1", "V1", "I1"], tiny, served)) == \
        ["Temperature_T1_mean", "Temperature_T1_std", "Vibration_V1_mean", "Vibration_V1_std", "Current_I1_mean", "Current_I1_std"]
    assert list(records.group_features(cycles[0], ["V1"], tiny, ("_std",))) == ["Vibration_V1_std"]
    assert records.public_name("Vibration_V1_mean", tiny) == "V1_Vibration_mm_per_s_10Hz_mean"
    broken = list(cycles)
    broken[1] = dataclasses.replace(broken[1], stable=False)
    with pytest.raises(ValueError, match="not contiguous"):
        records.build_blocks(broken)


def test_a_tiny_configuration_names_its_own_data_file_and_recordings(tiny, tiny_recordings, monkeypatch):
    config = tiny.default_config()
    monkeypatch.delenv(tiny.records["data_env"], raising=False)
    assert world.default_facts_path(config) == records.release_path(tiny) == Path(__file__).resolve().parent / "data" / "tiny_cycles.jsonl"
    monkeypatch.setenv(tiny.records["data_env"], str(tiny_recordings))
    assert world.default_facts_path(config) == tiny_recordings
    cycles, blocks = world.load_world(config.asset_id, str(tiny_recordings))
    assert list(blocks) == list(tiny.combinations()) and world.load_world(config.asset_id, str(tiny_recordings))[1] is blocks


def _tiny_env(tiny, recordings):
    cycles = records.load_cycles(recordings, tiny)
    return HydraulicRigEnv(cycles, records.build_blocks(cycles), tiny.default_config())


def _first_faulty_seed(env, healthy):
    for seed in range(100):
        env.reset(seed)
        if tuple(env.hidden) != healthy:
            return seed
    raise AssertionError("no seed below 100 starts with a faulty component")


def _repaired_episode(tiny, recordings, seed):
    """One read, the true diagnosis, an order on a faulty component, waits until it closes, and a second read."""
    env = _tiny_env(tiny, recordings)
    env.reset(seed)
    group, start = next(iter(env.config.sensor_groups)), tuple(env.hidden)
    faulty = next(c for c, level in zip(tiny.components, start) if level != tiny.healthy[c])
    env.step({"type": "read_sensors", "sensor_group": group, "cycles": 1})
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(tiny.components, start)), "evidence_ids": []})
    env.step({"type": "create_work_order", "component": faulty, "prediction_id": "", "evidence_ids": []})
    while env.clock.open_orders():
        env.step({"type": "wait_for_notification", "timeout": 5})
    env.step({"type": "read_sensors", "sensor_group": group, "cycles": 1})
    return env, start


def test_a_tiny_repair_switches_the_served_block_and_a_seed_replays_the_reads(tiny, tiny_recordings):
    seed = _first_faulty_seed(_tiny_env(tiny, tiny_recordings), tiny.healthy_state())
    env, start = _repaired_episode(tiny, tiny_recordings, seed)
    assert isinstance(env.source, EvidenceSource)
    first, second = (e for e in env.log if e["event"] == "read")
    assert tuple(env.hidden) != start
    assert first["served"][0]["cycle"] in env.blocks[start] and second["served"][0]["cycle"] in env.blocks[tuple(env.hidden)]
    channels = env.config.sensor_groups[first["sensor_group"]]
    assert set(first["served"][0]["values"]) == {f"{tiny.field_name(s)}_{stat}" for s in channels for stat in tiny.statistics["served"]}
    assert _repaired_episode(tiny, tiny_recordings, seed)[0].log == env.log


def test_the_public_profile_of_a_tiny_episode_lists_the_served_statistics_of_its_asset_file(tiny, tiny_recordings):
    env = _tiny_env(tiny, tiny_recordings)
    assert env.reset(0)["result"]["statistics"] == tiny.statistics["served"] == ["mean", "std"]


def test_the_profile_of_a_tiny_episode_holds_no_word_of_the_rig_on_either_engine(tiny, tiny_recordings):
    """The rule texts of the tool table and the clock block that the profile of a tiny episode publishes name the bench and
    never the rig, on the engine of the rig's own rows and on the engine of the AssetOpsBench rows, whose task server takes
    the parameters of submit_diagnosis from the tiny components."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv

    cycles = records.load_cycles(tiny_recordings, tiny)
    for engine in (HydraulicRigEnv, RecordedRigEnv):
        profile = engine(cycles, records.build_blocks(cycles), tiny.default_config()).reset(0)["result"]
        texts = json.dumps([[row["duration"], row["price"]] for row in profile["tools"]] + [profile["clock"]])
        assert not re.search(r"\b(rig|cooler|valve|pump|accumulator)\b", texts), engine.__name__
        assert re.search(r"\bbench\b", texts), engine.__name__
        diagnosis = next(row for row in profile["tools"] if row["name"] == "submit_diagnosis")
        assert list(diagnosis["parameters"]) == [*tiny.components, "evidence_ids"], engine.__name__


def test_a_tiny_configuration_whose_fault_rule_starts_at_zero_plays_the_healthy_start_to_a_pass(tiny, tiny_recordings):
    config = dataclasses.replace(tiny.default_config(), fault_rule={"min_faulty": 0, "max_faulty": 1},
                                 start={"combination": list(tiny.healthy_state()), "entry_offset": 0})
    cycles = records.load_cycles(tiny_recordings, tiny)
    env = HydraulicRigEnv(cycles, records.build_blocks(cycles), config)
    env.reset(0)
    assert tuple(env.hidden) == tiny.healthy_state()
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(tiny.components, tiny.healthy_state())), "evidence_ids": []})
    env.step({"type": "finish", "summary": "Every component is at its healthy level, so no work order was needed."})
    verdict = verify(env.log, released_from_env(env), env.oracle(), None)
    assert verdict["verified"] and verdict["pass"] and verdict["failure_class"] == "none", verdict


def _released_after(tiny, recordings, seed, actions):
    """The released evidence of a tiny episode from the seed after the actions, each a step of the engine."""
    env = _tiny_env(tiny, recordings)
    env.reset(seed)
    for action in actions(env):
        env.step(action)
    return env, released_from_env(env)


def _top_combination(models, released, start):
    """The combination the fits of an episode of the start rank first on the released evidence."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.diagnostics_model import EffectDiagnostics
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view

    posterior = EffectDiagnostics(models.without(start)).replay(profile_view(released["rules"]), released).posterior()
    return posterior.candidates[max(range(len(posterior.candidates)), key=lambda i: posterior.probabilities[i])]


def test_the_tiny_fit_ranks_the_true_start_first_and_verifies_its_repair(tiny, tiny_recordings):
    """The diagnostics model fitted on the tiny recordings ranks the start of a tiny episode first after one read of the
    widest sensor group, and once the order on the faulty component has closed and one more read follows, it ranks the
    repaired combination first. The test checks an ordering of the belief and no value of the fit."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import Models
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import fit

    models = Models(fit(tiny_recordings, asset_id=tiny.asset_id, resamples=20))
    seed = _first_faulty_seed(_tiny_env(tiny, tiny_recordings), tiny.healthy_state())

    def widest(env):
        return max(env.config.sensor_groups, key=lambda g: len(env.config.sensor_groups[g]))

    def read(env):
        return {"type": "read_sensors", "sensor_group": widest(env), "cycles": 1}

    env, released = _released_after(tiny, tiny_recordings, seed, lambda env: [read(env)])
    start = tuple(env.hidden)
    assert _top_combination(models, released, start) == start

    def repair(env):
        faulty = next(c for c, level in zip(tiny.components, env.hidden) if level != tiny.healthy[c])
        yield read(env)
        yield {"type": "submit_diagnosis", "diagnosis": dict(zip(tiny.components, env.hidden)), "evidence_ids": []}
        yield {"type": "create_work_order", "component": faulty, "prediction_id": "", "evidence_ids": []}
        while env.clock.open_orders():
            yield {"type": "wait_for_notification", "timeout": 5}
        yield read(env)

    env, released = _released_after(tiny, tiny_recordings, seed, repair)
    assert tuple(env.hidden) != start
    assert _top_combination(models, released, start) == tuple(env.hidden)


def _idle_verdict(tiny, recordings, seed, summary):
    """The verdict of an episode that does nothing and finishes at once with the summary."""
    env = _tiny_env(tiny, recordings)
    env.reset(seed)
    env.step({"type": "finish", "summary": summary})
    return verify(env.log, released_from_env(env), env.oracle(), None)


def test_the_tiny_oracle_passes_and_do_nothing_fails_with_its_class(tiny, tiny_recordings):
    """The verifier scores a tiny episode on the tables of the asset that its reset entry names: the oracle policy passes
    on the tiny components, and an episode that does nothing and claims that the bench recovered fails as a summary that
    contradicts the log, since bench is a word of the tiny asset for the whole asset while rig is not."""
    env = _tiny_env(tiny, tiny_recordings)
    seed = _first_faulty_seed(env, tiny.healthy_state())
    log = run_episode(env, OraclePolicy(env), seed)
    best = verify(log, released_from_env(env), env.oracle(), None)
    assert best["verified"] and best["pass"] and best["all_checks_ok"] and best["failure_class"] == "none", best
    assert list(best["diagnosis_per_component"]) == list(tiny.components)
    assert isinstance(LevelsTruth(tiny), TruthModel)
    idle = _idle_verdict(tiny, tiny_recordings, seed, "The bench is back to normal.")
    assert idle["verified"] and not idle["pass"] and not idle["recovered"]
    assert idle["failure_class"] == "summary_contradicts_log" and idle["summary_check"]["claims_recovery"]
    assert _idle_verdict(tiny, tiny_recordings, seed, "The rig is back to normal.")["failure_class"] == "none"


def test_the_tool_table_and_the_briefing_offer_the_emergency_mode_only_while_its_card_is_on(tiny, tiny_recordings):
    """The tool row of create_work_order in the asset profile and its entry in the briefing name the mode, and the row
    the emergency clauses of its rules, only while the emergency card is on, its hold above 0, and the card changes no
    other row."""
    cycles = records.load_cycles(tiny_recordings, tiny)
    config = tiny.default_config()
    off_config = dataclasses.replace(config, emergency={**config.emergency, "hold": 0})
    profiles = [HydraulicRigEnv(cycles, records.build_blocks(cycles), c).reset(0)["result"] for c in (config, off_config)]
    on, off = ({row["name"]: row for row in profile["tools"]} for profile in profiles)
    assert config.emergency_on() and not off_config.emergency_on()
    assert "mode" in on["create_work_order"]["parameters"] and "mode" not in off["create_work_order"]["parameters"]
    assert "emergency" not in off["create_work_order"]["duration"] + off["create_work_order"]["price"]
    assert [name for name in on if on[name] != off[name]] == ["create_work_order"]
    on_text, off_text = (briefing(profile, dataclasses.replace(c, guidance_level=2)) for profile, c in zip(profiles, (config, off_config)))
    assert "create_work_order(component, prediction_id, evidence_ids, mode)" in on_text
    assert "create_work_order(component, prediction_id, evidence_ids)" in off_text and "evidence_ids, mode)" not in off_text


def test_the_action_server_of_a_tiny_episode_takes_its_components_and_nouns_from_the_asset(tiny, tiny_recordings, tmp_path, monkeypatch):
    """The action server of a tiny episode advertises one condition level per tiny component in submit_diagnosis, names
    the components, their number and the bench in its descriptions and its instructions, matches the tiny asset's
    contract rather than the file's, and passes the levels to the environment under the tiny names."""
    path = tmp_path / "episode.json"
    # the state file names the data file by its basename, outside the tree, and the server finds it through the variable of the asset
    monkeypatch.setenv(tiny.records["data_env"], str(tiny_recordings))
    episode.start(path, 0, tiny.default_config(), tiny_recordings)
    episode.hand_over(path)
    server = build_server(path)
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    diagnosis = tools["submit_diagnosis"].parameters
    assert list(diagnosis["properties"]) == diagnosis["required"] == [*tiny.components, "evidence_ids"]
    assert all("(fan, belt or bearing)" in tools[name].description for name in ("get_failure_modes", "lookup_fix", "create_work_order"))
    assert "each of the three components" in tools["submit_diagnosis"].description
    assert tools["read_sensors"].description.startswith("Run the bench ")
    assert "The bench runs while it waits" in tools["wait_for_notification"].description
    assert "tiny test bench" in server.instructions
    assert not [name for name, t in tools.items() if re.search(r"\b(rig|cooler|valve|pump|accumulator)\b", t.description)]
    hints = {name: t.annotations.readOnlyHint for name, t in tools.items()}
    parameters = {name: {k: v["type"] for k, v in t.parameters["properties"].items()} for name, t in tools.items()}
    assert contract.check_advertised(contract.ACTION_SERVER, hints, parameters, asset_id=tiny.asset_id) == []
    assert contract.check_advertised(contract.ACTION_SERVER, hints, parameters) != []
    levels = dict(zip(tiny.components, episode.replay(episode.load_state(path)).hidden))
    asyncio.run(server.call_tool("submit_diagnosis", {**levels, "evidence_ids": ""}))
    action = episode.load_state(path)["actions"][-1]
    assert action == {"type": "submit_diagnosis", "diagnosis": levels, "evidence_ids": []}


def test_the_briefing_of_a_tiny_episode_names_its_components_and_never_the_rig(tiny, tiny_recordings):
    env = _tiny_env(tiny, tiny_recordings)
    profile = env.reset(0)["result"]
    texts = [briefing(profile, dataclasses.replace(env.config, guidance_level=level)) for level in (1, 2, 3)]
    assert all("submit_diagnosis(fan, belt, bearing, evidence_ids)" in text for text in texts[1:])
    assert not [text for text in texts if re.search(r"\b(rig|cooler|valve|pump|accumulator)\b", text)]


class _SilentModel:
    """The language model of a rule only run, asked only for the closing summary, which it leaves to the runner."""

    def generate(self, prompt, temperature=0.0):
        return ""

    def generate_with_usage(self, prompt, temperature=0.0):
        return LLMResult(text="")


def _launcher(path, module, served, extra="", data=None):
    """The launcher of one server process of a tiny episode. A server process gets no environment of its own, so the
    launcher puts the root and src of the fork on the import path and registers the tiny profile from its file before the server reads
    the episode, as no bundled asset file holds it."""
    path.write_text(f"""import sys
from pathlib import Path

sys.path[:0] = {[str(ROOT), str(ROOT / "src")]!r}
from CausalDiagnosis.datasets.dataset_loader.profile_registry import load_profile, register_profile

register_profile(load_profile({str(TINY_FILE)!r}))
{f"import os{chr(10)}os.environ[{data[0]!r}] = {str(data[1])!r}" if data else ""}
from {module} import main

main(Path({str(served)!r}){extra})
""", encoding="utf-8")
    return path


def test_the_tiny_asset_runs_both_servers_and_the_rule_only_agent(tiny, tiny_recordings, tmp_path, monkeypatch):
    """One tiny episode end to end over stdio: every call starts the action server or the diagnostics server as its own
    process on the tiny profile, the diagnostics server on models fitted on the tiny recordings, and the rule only
    CausalDiagnosis agent plays the episode through both. The verifier scores the log against the released evidence and the
predictions the diagnostics server wrote, as the drivers do, and passes it."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import CausalLoopOptions
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import CausalLoopRunner
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import fit

    models = tmp_path / "models.json"
    models.write_text(json.dumps(fit(tiny_recordings, asset_id=tiny.asset_id, resamples=20)), encoding="utf-8")
    seed = _first_faulty_seed(_tiny_env(tiny, tiny_recordings), tiny.healthy_state())
    state = tmp_path / "episode.json"
    monkeypatch.setenv(tiny.records["data_env"], str(tiny_recordings))  # the data file outside the tree, found through the variable of the asset
    started = episode.start(state, seed, tiny.default_config(), tiny_recordings)
    episode.hand_over(state)
    released = episode.released_path(state)
    # the data file of the episode reaches every server through the variable of the asset, as the drivers' launchers set it
    data = (tiny.records["data_env"], tiny_recordings)
    launchers = {"hydraulic_rig": _launcher(tmp_path / "action_server.py", "CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main", state, data=data),
                 "causal_diagnosis": _launcher(tmp_path / "diagnostics_server.py", "CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main", released,
                                               f", models_path={str(models)!r}, left_out={tuple(episode.replay(started).initial)!r}", data=data)}
    runner = CausalLoopRunner(llm=_SilentModel(), server_paths=launchers, options=CausalLoopOptions(rule_only=True))
    result = asyncio.run(runner.run(json.loads(released.read_text(encoding="utf-8"))["task_text"]))
    assert result.error is None, result.error
    assert {call["server"] for call in result.trajectory} == {"hydraulic_rig", "causal_diagnosis"}
    refused = [call["response"][:200] for call in result.trajectory if call["server"] == "causal_diagnosis" and not call["response"].startswith("{")]
    assert not refused, refused[0]  # every answer of the diagnostics server is a JSON object, none a tool error
    env = episode.replay(episode.load_state(state))
    predictions = json.loads(released.with_name("predictions.json").read_text(encoding="utf-8"))
    verdict = verify(env.log, json.loads(released.read_text(encoding="utf-8")), env.oracle(), None, predictions=predictions)
    assert verdict["verified"] and verdict["pass"] and verdict["failure_class"] == "none", verdict


def test_the_repair_notification_names_the_asset_by_its_short_noun(tiny, tiny_recordings):
    """The notification that a repair began says that the asset stands still in the short noun of its profile, so an episode
    of another asset never reads the rig's word."""
    seed = _first_faulty_seed(_tiny_env(tiny, tiny_recordings), tiny.healthy_state())
    env = _tiny_env(tiny, tiny_recordings)
    env.reset(seed)
    start = tuple(env.hidden)
    faulty = next(c for c, level in zip(tiny.components, start) if level != tiny.healthy[c])
    env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(tiny.components, start)), "evidence_ids": []})
    notes = env.step({"type": "create_work_order", "component": faulty, "prediction_id": "", "evidence_ids": []})[0]["notifications"]
    while env.clock.open_orders() and not env.done:
        notes += env.step({"type": "wait_for_notification", "timeout": 5})[0]["notifications"]
    executed = [n["message"] for n in notes if n["event"] == "work_order_executed"]
    noun = tiny.text["short_noun"]
    assert noun != "rig" and executed and all(f"the {noun} is stopped until it closes" in m for m in executed)
    assert not [m for m in executed if "rig" in m.split()]
