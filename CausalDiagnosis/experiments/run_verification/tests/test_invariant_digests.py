"""The digest tools of the asset layer refactor: digests that see key order, the comparison of two digest maps, the
digest file that holds hashes only, and the recomputation of a stored verdict from an episode folder."""
from CausalDiagnosis.utils.paths import CONFIGS, PACKAGE, preset_files
import json
import re
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.run_verification import invariant_digests as asset_digests
from CausalDiagnosis.experiments.run_verification.invariant_digests import compare, digest
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import digest_families as families
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.experiments.run_verification.recompute_verdicts import check, reverify, run

DIGESTS = PACKAGE / "experiments" / "run_verification" / "invariant_digests.json"
HEX = re.compile(r"[0-9a-f]{64}")
needs_data = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")


def test_a_digest_sees_key_order_and_equal_values_give_equal_digests():
    assert digest({"a": 1, "b": [2, 3]}) == digest({"a": 1, "b": [2, 3]})
    assert digest({"a": 1, "b": 2}) != digest({"b": 2, "a": 1})
    assert digest([1, 2]) != digest([2, 1])


def test_compare_names_changed_missing_and_new_digests_and_accepts_expected_prefixes():
    stored = {"starts.static.start": "1", "starts.static.fix_history": "2", "logs.static.oracle": "3"}
    computed = {"starts.static.start": "1", "starts.static.fix_history": "9", "texts.agent.system": "4"}
    same, changed, missing, new, unexpected = compare(stored, computed)
    assert (same, changed, missing, new) == (["starts.static.start"], ["starts.static.fix_history"], ["logs.static.oracle"], ["texts.agent.system"])
    assert unexpected == ["starts.static.fix_history", "logs.static.oracle", "texts.agent.system"]
    assert compare(stored, computed, ("starts.static.fix_history", "logs.", "texts."))[4] == []


def test_the_digest_file_holds_hashes_only_keyed_by_the_facts_file_hash():
    data = json.loads(DIGESTS.read_text(encoding="utf-8"))
    assert data["format"] == 1 and data["baselines"]
    for facts_hash, baselines in data["baselines"].items():
        assert HEX.fullmatch(facts_hash)
        for baseline in baselines.values():
            assert baseline["digests"] and all(HEX.fullmatch(v) for v in baseline["digests"].values())


def test_the_digest_presets_are_the_preset_files_of_the_configs_folder(tmp_path, monkeypatch):
    from CausalDiagnosis.experiments.episode_runners import run_configuration

    files = sorted(path.stem for path in preset_files(run_configuration.CONFIGS).values())
    assert files and asset_digests.presets() == files
    for path in preset_files(run_configuration.CONFIGS).values():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    base = json.loads((CONFIGS / "layer_presets" / "base.json").read_text(encoding="utf-8"))
    (tmp_path / "added_preset.json").write_text(json.dumps({**base, "name": "added_preset"}), encoding="utf-8")
    monkeypatch.setattr(run_configuration, "CONFIGS", tmp_path)
    assert asset_digests.presets() == sorted([*files, "added_preset"])
    assert "config.added_preset" in asset_digests.config_digests(None, {})


@pytest.mark.parametrize("name", ("default", "rows"))
def test_a_preset_named_like_a_key_of_a_family_is_refused(name, tmp_path, monkeypatch):
    from CausalDiagnosis.experiments.episode_runners import run_configuration

    (tmp_path / f"{name}.json").write_bytes((CONFIGS / "layer_presets" / "base.json").read_bytes())
    monkeypatch.setattr(run_configuration, "CONFIGS", tmp_path)
    with pytest.raises(SystemExit, match=name):
        asset_digests.presets()


def test_the_config_family_holds_every_digest_preset():
    keys = asset_digests.config_digests(None, {})
    assert {key for key in keys if key.count(".") == 1} == {"config.default", *(f"config.{name}" for name in asset_digests.presets())}


@needs_data
def test_the_process_family_leaves_out_the_presets_that_do_not_replay_through_processes(tmp_path):
    """A preset whose clock charges the ten cycles of wall seconds before a call does not replay through processes; the clock of
    generated tokens charges the tokens of its ledger, none here, and replays."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    names = asset_digests.presets()
    charged = {}
    for name in names:
        config = asset_digests.preset(name)
        state_path = tmp_path / name / "episode.json"
        state_path.parent.mkdir()
        if config.clock_mode == real_time.TOKEN_MODE:
            token_clock.open_ledger(state_path.parent)
        state = episode.start(state_path, 0, config, default_facts_path())
        episode.hand_over(state_path)
        later = 10 * config.think_seconds_per_cycle  # ten cycles of thinking on a clock that charges the wall seconds
        episode.call(state_path, "get_current_time", {}, {"type": "get_current_time"}, now=state["last_call_wall_time"] + later,
                     monotonic=state["last_call_monotonic_time"] + later)
        charged[name] = episode.load_state(state_path)["think_cycles"][-1]
    replays = [name for name in names if charged[name] == 0 and asset_digests.preset(name).anchors is None]
    assert asset_digests.process_presets() == replays
    assert {"static", "timed"} <= set(replays) and len(replays) < len(names)


@needs_data
def test_the_scripted_episode_of_a_token_preset_enters_the_process_presets_of_the_digests(tmp_path):
    """Every token twin of a clock preset replays through processes and its episode of the rows charges one cycle per call by
    the tokens of one think_seconds_per_cycle, byte for byte on a second play, as its wall-clock preset under the given clock."""
    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design

    assert set(design.TOKEN_PRESETS) <= set(asset_digests.process_presets())
    config = asset_digests.preset(sorted(design.TOKEN_PRESETS)[0])
    first = families.play_rows(config, 0, default_facts_path(), tmp_path / "a" / "seed000000")
    second = families.play_rows(config, 0, default_facts_path(), tmp_path / "b" / "seed000000")
    assert {rel: digest(value) for rel, value in first.items()} == {rel: digest(value) for rel, value in second.items()}
    state = first["episode.json"]
    assert state["think_cycles"] and set(state["think_cycles"]) == {1} and len(set(state["think_tokens"])) == 1


@needs_data
def test_two_runs_of_the_scripted_family_give_equal_digests_and_a_value_of_the_state_file_moves_its_digest(monkeypatch):
    """The scripted family leaves out every clock that the driver and the servers write into the files it digests, the end of the
    episode that the driver writes after the agent ends included, so two runs of one tree give equal digests. A value of the state
    file that is no clock moves the digest of that file and of no other."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    monkeypatch.setattr(asset_digests, "process_presets", lambda: ["static"])
    monkeypatch.setattr(asset_digests, "SCRIPTED_SEEDS", 1)
    monkeypatch.setattr(asset_digests, "rows_digests", lambda facts: {})
    first, second = (asset_digests.scripted_digests(None, {}, default_facts_path()) for _ in range(2))
    assert first == second and "scripted.static.seed000000.episode.json" in first
    written = run_episodes.play_one_episode.update_state

    def with_a_value(state_path, **fields):
        written(state_path, **fields, **({"control_value": 1} if "episode_ended_at" in fields else {}))

    monkeypatch.setattr(run_episodes.play_one_episode, "update_state", with_a_value)
    changed = asset_digests.scripted_digests(None, {}, default_facts_path())
    assert [key for key in first if changed[key] != first[key]] == ["scripted.static.seed000000.episode.json"]


@needs_data
def test_an_episode_of_the_rows_replays_byte_for_byte_under_the_given_clock(tmp_path):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time

    name = next(name for name in asset_digests.presets() if real_time.thinking(asset_digests.preset(name), 1))
    config = asset_digests.preset(name)
    first = families.play_rows(config, 0, default_facts_path(), tmp_path / "a" / "seed000000")
    second = families.play_rows(config, 0, default_facts_path(), tmp_path / "b" / "c" / "seed000000")
    assert list(first) == list(families.ROWS_FILES)
    assert {rel: digest(value) for rel, value in first.items()} == {rel: digest(value) for rel, value in second.items()}
    assert first["replies"][-1][:2] == ["task", "finish"]
    assert first["episode.json"]["think_cycles"] and set(first["episode.json"]["think_cycles"]) == {1}


@needs_data
def test_the_text_family_holds_every_preset_level_and_server():
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    _facts, cycles, blocks = load_world(None)
    keys = set(asset_digests.text_digests((cycles, blocks), {}, default_facts_path()))
    names = asset_digests.presets()
    expected = {*(f"texts.profile.{name}" for name in names), *(f"texts.rows.profile.{name}" for name in names),
                *(f"texts.briefing.{name}.{level}" for name in names for level in (1, 2, 3)),
                *(f"texts.rows.briefing.{name}.{level}" for name in names for level in (1, 2, 3)),
                *(f"texts.rows.tools.{name}.{server}" for name in names for server in METERED_SERVERS)}
    assert {key for key in keys if key.startswith(("texts.profile.", "texts.briefing.", "texts.rows."))} == expected
    assert {"texts.tools.hydraulic_rig", "texts.tools.causal_diagnosis", "texts.agent.system", "texts.agent.orientation",
            "texts.agent.planning", "texts.agent.summary"} <= keys


def aob_texts(monkeypatch=None):
    """The .aob keys of the text family and their detail, on the inputs text_digests gives them."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing

    _facts, cycles, blocks = load_world(None)
    config = asset_digests.preset("static")
    env = HydraulicRigEnv(cycles, blocks, config)
    profile = env.reset(0)["result"]
    envelope = env.step({"type": "get_failure_modes", "component": ""})[0]
    detail = {}
    out = families.aob_agent_texts(briefing(profile, config), profile, envelope, default_facts_path(), detail)
    return out, detail


@needs_data
def test_the_text_family_holds_the_agent_texts_on_the_surface_of_the_rows_for_every_preset():
    out, detail = aob_texts()
    names = families.aob_presets()
    parts = ("orientation", "planning", "summary", "bindings")
    assert set(out) == {"texts.agent.system.aob", *(f"texts.agent.{part}.aob" for part in parts)}
    assert all(set(detail[f"texts.agent.{part}.aob"]) == set(names) for part in parts)
    assert len(detail["texts.agent.system.aob"]) == len(names) + 1  # before the profile is read, then after each opening


@needs_data
def test_a_changed_word_of_the_surface_of_the_rows_changes_its_digest_keys(monkeypatch):
    from CausalDiagnosis.frameworks.AssetOpsBench.agents import causal_diagnosis_agent_surface as tools

    before, _ = aob_texts()
    monkeypatch.setattr(tools.AobSurface, "prompt_changes", " ")
    order = tools.AobSurface._bind_order
    monkeypatch.setattr(tools.AobSurface, "_bind_order", lambda self, *a, **k: (*order(self, *a, **k)[:2], {}))
    after, _ = aob_texts()
    assert {key for key in before if before[key] != after[key]} == set(before)


@needs_data
def test_the_start_and_log_families_run_on_every_preset(monkeypatch):
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.experiments.table_definitions.experiment_tables import REFERENCE_POLICIES
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states

    _facts, cycles, blocks = load_world(None)
    monkeypatch.setattr(asset_digests, "START_SEEDS", (0, 1))
    monkeypatch.setattr(start_states, "combination_seeds", lambda env, per_combination=1, after=0: [0])
    keys = {**asset_digests.start_digests((cycles, blocks), {}), **asset_digests.log_digests((cycles, blocks), {})}
    names = asset_digests.presets()
    policies = [label for label, _name in REFERENCE_POLICIES]
    assert set(keys) == {*(f"starts.{name}.{part}" for name in names for part in ("start", "schedule", "hazard", "fix_history")),
                         *(f"logs.{name}.seeds" for name in names),
                         *(f"{family}.{name}.{policy}" for family in ("logs", "verdicts") for name in names for policy in policies)}


def test_a_label_of_record_names_the_commit_of_its_digests(tmp_path):
    facts = "f" * 64
    path = tmp_path / "digests.json"
    labels = {"record": {"commit": "0123abc", "digests": {"config.x": "0" * 64}}, "working": {"commit": "uncommitted", "digests": {}},
              "unnamed": {"commit": None, "digests": {}}}
    path.write_text(json.dumps({"format": 1, "baselines": {facts: labels}}), encoding="utf-8")
    assert asset_digests.record_digests(path, facts, "record") == ({"config.x": "0" * 64}, None)
    for label in ("working", "unnamed", "absent"):
        digests, reason = asset_digests.record_digests(path, facts, label)
        assert digests is None and f"label {label}" in reason and facts in reason and "--commit HASH" in reason
    digests, reason = asset_digests.record_digests(path, "e" * 64, "record")
    assert digests is None and "e" * 64 in reason
    values = ("0123abc", "0" * 40, "012345", "0" * 41, "0123ABC", "none", "", None)
    assert [asset_digests.is_commit_hash(value) for value in values] == [True, True, False, False, False, False, False, False]


def test_the_digest_file_of_the_checkout_takes_only_a_label_that_names_its_commit(tmp_path, monkeypatch):
    def refused(*_arguments):
        raise AssertionError("the digests were computed before the label was refused")

    facts = tmp_path / "data.jsonl"
    facts.write_text("x", encoding="utf-8")
    before = asset_digests.DIGESTS_FILE.read_bytes()
    monkeypatch.setattr(asset_digests, "compute", refused)
    monkeypatch.setattr(asset_digests, "checkout_commit", lambda: None)
    for given in ([], ["--commit", "none"], ["--commit", ""]):
        with pytest.raises(SystemExit) as stop:
            asset_digests.main(["--facts", str(facts), "--label", "current", "--write", str(asset_digests.DIGESTS_FILE), *given])
        assert stop.value.code == 2
    assert asset_digests.DIGESTS_FILE.read_bytes() == before
    monkeypatch.setattr(asset_digests, "compute", lambda _facts, _families: ({"config.x": "0" * 64}, {}))
    scratch = tmp_path / "map.json"
    asset_digests.main(["--facts", str(facts), "--label", "lane", "--write", str(scratch), "--commit", "none"])
    asset_digests.main(["--facts", str(facts), "--label", "current", "--write", str(scratch), "--commit", "0123abc"])
    facts_hash = asset_digests.file_sha256(facts)
    assert asset_digests.record_digests(scratch, facts_hash, "lane")[0] is None
    assert asset_digests.record_digests(scratch, facts_hash, "current") == ({"config.x": "0" * 64}, None)


def test_the_rows_plan_lists_the_sites_diagnoses_the_start_and_orders_each_faulty_component():
    from types import SimpleNamespace

    config = SimpleNamespace(components=("a", "b", "c"), healthy={"a": 1, "b": 1, "c": 1}, asset_id="unit")
    plan = families.rows_plan(config, (1, 2, 3))
    tools = [tool for _server, tool, _arguments in plan]
    assert tools.index("sites") == tools.index("assets") - 1
    diagnosis = next(arguments for _server, tool, arguments in plan if tool == "submit_diagnosis")
    assert {key: value for key, value in diagnosis.items() if key != "evidence_ids"} == {"a": 1, "b": 2, "c": 3}
    orders = [arguments for _server, tool, arguments in plan if tool == "generate_work_order"]
    assert [order["asset_num"] for order in orders] == ["unit_b", "unit_c"]
    assert [order["description"] for order in orders] == ["replace the b", "replace the c"]
    assert tools[-3:] == ["wait_for_notification", "history", "finish"] and tools.count("history") == 2
    assert "generate_work_order" not in [tool for _server, tool, _arguments in families.rows_plan(config, (1, 1, 1))]


def test_a_baseline_of_another_data_file_is_refused_by_name():
    """A digest file that holds no baseline of the label for the data file in use is refused with the command that writes it,
    not with a KeyError."""
    from CausalDiagnosis.experiments.run_verification import invariant_digests as asset_digests
    from CausalDiagnosis.datasets.dataset_loader import recordings as records

    with pytest.raises(records.DataFileError) as refused:
        asset_digests.load_baseline(DIGESTS, "0" * 64, "current")
    assert "0" * 64 in str(refused.value) and "python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts FILE --label current --write" in str(refused.value)


def test_the_scripted_digests_fit_and_compare_on_the_file_they_are_given(tmp_path, monkeypatch):
    """The scripted family fits the models on the data file that the digest tool was given and compares them with it, as the
    driver does with the data file of its run."""
    from CausalDiagnosis.experiments.run_verification import invariant_digests as asset_digests
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit

    class Stop(Exception):
        pass

    asked = []

    def stopped(**keywords):
        asked.append(keywords)
        raise Stop

    monkeypatch.setattr(fit, "ensure_models", stopped)
    given = tmp_path / "recordings.jsonl"
    with pytest.raises(Stop):
        asset_digests.scripted_digests(None, {}, given)
    assert asked == [{"facts_path": given}]


@needs_data
def test_reverify_recomputes_a_stored_verdict_and_names_the_keys_that_differ(tmp_path, world_and_config):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode

    cycles, blocks, config = world_and_config
    env = HydraulicRigEnv(cycles, blocks, config)
    log = run_episode(env, OraclePolicy(env), 3)
    released = verifier.released_from_env(env)
    verdict = verifier.verify(log, released, env.oracle(), env.cost, grade=None, scope=config.verifier_scope)
    folder = tmp_path / "runs" / "run_a" / "seed000003"
    (folder / "hidden").mkdir(parents=True)
    for name, data in (("hidden/log.json", log), ("hidden/oracle.json", env.oracle()), ("hidden/verdict.json", verdict), ("released.json", released)):
        (folder / name).write_text(json.dumps(data), encoding="utf-8")
    first = run(tmp_path / "runs")
    assert first["summary"] == {"episodes": 1, "recomputed": 1, "stored_equal": 1, "skipped": 0}
    assert first["episodes"]["run_a/seed000003"]["digest"] == digest(verdict)
    (folder / "hidden/verdict.json").write_text(json.dumps({**verdict, "pass": not verdict["pass"]}), encoding="utf-8")
    assert reverify(folder)["differs"] == ["pass"]
    assert check(run(tmp_path / "runs"), first) == ([], [])


@needs_data
def test_no_text_a_model_reads_names_a_band_and_the_profile_serves_normal_ranges(monkeypatch):
    """The texts the digests cover, which are what an agent reads before its first call on every preset (both briefings at every
    level, the profiles, the tool listings of the action, task, metered and diagnostics servers, the agent's system and turn
    prompts), say "normal range", the term of the paper, and never the term it replaced in any spelling. The profile serves the key
    normal_ranges, the evidence answer of the diagnostics server names its keys normal_low, normal_high and in_normal_range,
    and the repair manual says that a repair returns the indicating channels to their normal ranges."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import EvidenceItem

    texts = []
    monkeypatch.setattr(asset_digests, "digest", lambda value: texts.append(json.dumps(value, ensure_ascii=False)) or "")
    _facts, cycles, blocks = load_world(None)
    asset_digests.text_digests((cycles, blocks), {}, default_facts_path())
    # the dead band of the direction of E15 (estimate_intervention_effect, the field dead_band) is another notion and stays
    band = re.compile(r"healthy[\s_\-]*band|(?<!dead )(?<![\w\-])bands?(?![\w\-])", re.IGNORECASE)
    assert texts and not [text[max(0, m.start() - 60):m.end() + 60] for text in texts for m in band.finditer(text)]
    env = HydraulicRigEnv(cycles, blocks, asset_digests.preset("deterioration"))
    profile = env.reset(0)["result"]
    assert profile["normal_ranges"] and not [key for key in profile if "band" in key]
    assert {"normal_low", "normal_high", "in_normal_range"} <= set(EvidenceItem.model_fields)
    manual = env.step({"type": "lookup_fix", "component": env.config.components[0]})[0]["result"]
    assert "normal ranges" in manual["effect"] and not band.search(json.dumps(manual))


@pytest.fixture
def world_and_config():
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig

    _facts, cycles, blocks = load_world(None)
    return cycles, blocks, EpisodeConfig()
