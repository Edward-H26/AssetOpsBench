"""The Gymnasium wrapper (CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper): determinism, the reward order of the oracle and the do
nothing policy, truncation at the horizon and at the step cap, the information barrier of the observation text,
the development starts it plays by default and the guard of the test set, gymnasium's checker, and the import without
gymnasium."""
from CausalDiagnosis.utils.paths import CONFIGS, preset_path
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HIDDEN_FIELDS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import GYMNASIUM_AVAILABLE, HydraulicRigGym, oracle_cost, parse_action
from CausalDiagnosis.environment.reference_policies.policies import DeclarePolicy, DoNothingPolicy, OraclePolicy
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import ABSTENTION_CREDIT, released_from_env
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory import make_env

from CausalDiagnosis.datasets.tests.conftest import session_tables, tiny, tiny_data, tiny_recordings  # noqa: F401

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import keys_of

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

FACTS = default_facts_path()
from CausalDiagnosis.utils.paths import SRC
PRESETS = CONFIGS
LABELS = ("cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure")  # the label keys of the facts file
# the seeded schedule gives the hidden processes something to leak
TIMED = replace(
    EpisodeConfig(),
    schedule=dict(EpisodeConfig().schedule, mode="seeded"),
)
SCRIPTED = [
    {"type": "get_asset_profile"},
    {"type": "read_sensors", "sensor_group": "tier3", "cycles": 2},
    {"type": "wait_for_notification", "timeout": 12},
    {"type": "summarize_changes_since", "cycle": 0},
    {"type": "get_failure_modes", "component": ""},
    {"type": "read_sensors", "sensor_group": "tier1", "cycles": 1},
    {"type": "finish", "summary": "scripted"},
]


def play(gym, policy, seed):
    """One episode through the wrapper: the observation texts, the rewards, the final flags and the final info."""
    text, _info = gym.reset(seed)
    texts, rewards = [text], []
    while True:
        text, reward, terminated, truncated, info = gym.step(policy.act(gym.observation))
        texts.append(text)
        rewards.append(reward)
        if terminated or truncated:
            return texts, rewards, (terminated, truncated), info


def preset(name):
    return compose(json.loads(preset_path(name).read_text(encoding="utf-8")))


def observation_json(gym, text):
    """The JSON part of an observation text: a step text is the envelope, the reset text is the briefing followed by
    the profile. The hidden field names are checked as keys, since the briefing prose may use the words in another
    sense and the contract's parameter table (skipped by keys_of) names an argument cycle."""
    if text.startswith(gym.task_text + "\n\n"):
        return json.loads(text[len(gym.task_text) + 2:])
    return json.loads(text)


def assert_no_hidden_field(gym, text):
    """No hidden field and no label of the facts file as a key; a failure mode code may contain a label's words."""
    assert not keys_of(observation_json(gym, text)) & set(HIDDEN_FIELDS), text
    assert '"seed":' not in text and not any(f'"{label}":' in text for label in LABELS)


def test_same_seed_gives_the_same_texts_and_the_same_terminal_reward():
    a = HydraulicRigGym(EpisodeConfig(), FACTS)
    b = HydraulicRigGym(EpisodeConfig(), FACTS)
    texts_a, rewards_a, flags_a, info_a = play(a, OraclePolicy(a.env), 5)
    texts_b, rewards_b, flags_b, info_b = play(b, OraclePolicy(b.env), 5)
    assert texts_a == texts_b and rewards_a == rewards_b and flags_a == flags_b
    assert info_a["end_reason"] == info_b["end_reason"] and info_a["cost_oracle"] == info_b["cost_oracle"]
    # the same instance reproduces the episode after another episode ran on it
    play(a, DoNothingPolicy(), 6)
    texts_c, rewards_c, _flags, _info = play(a, OraclePolicy(a.env), 5)
    assert texts_c == texts_a and rewards_c == rewards_a


def test_seed_offset_shifts_the_seed_and_a_missing_seed_counts_episodes():
    gym = HydraulicRigGym(EpisodeConfig(), FACTS, seed_offset=100)
    _text, info = gym.reset(3)
    assert info["seed"] == 103
    _text, first = gym.reset()
    _text, second = gym.reset()
    assert (first["seed"], second["seed"]) == (101, 102)


def test_reward_is_zero_before_the_end_and_the_oracle_beats_do_nothing():
    gym = HydraulicRigGym(EpisodeConfig(), FACTS)
    for seed in range(3):
        _texts, oracle_rewards, _flags, oracle_info = play(gym, OraclePolicy(gym.env), seed)
        _texts, idle_rewards, _flags, idle_info = play(gym, DoNothingPolicy(), seed)
        assert all(r == 0 for r in oracle_rewards[:-1]) and all(r == 0 for r in idle_rewards[:-1])
        assert oracle_rewards[-1] > idle_rewards[-1]
        assert oracle_info["reward"] == oracle_rewards[-1] and idle_info["reward"] == idle_rewards[-1]
        assert "verdict" in oracle_info and "verdict" in idle_info


def test_the_gym_of_an_asset_that_declares_no_label_plays_to_a_verdict_without_one(tiny, tiny_recordings):
    """The default label of the gym is the declaration of the episode's asset, so an asset that declares none, as the tiny
    one, plays its episodes with no label, and every start counts as separable, while the rig keeps its own label. The
    recordings of the tiny asset admit no development split, so its gym plays the test starts, as a registered run would."""
    gym = HydraulicRigGym(tiny.default_config(), tiny_recordings, starts="test", unregistered=[])
    for seed in range(3):
        _texts, rewards, _flags, info = play(gym, OraclePolicy(gym.env), seed)
        assert gym.label is None and info["verdict"]["grade"] is None and info["verdict"]["separable"] is True
        assert info["verdict"]["pass"] and rewards[-1] > 0


def test_a_declaration_after_reading_on_a_start_the_label_marks_inseparable_earns_the_credit_below_the_oracle():
    inseparable = {"separable": False, "cost": None, "e18": {"separable": False}}
    labelled = HydraulicRigGym(EpisodeConfig(), FACTS, label_of=lambda env: inseparable)
    unlabelled = HydraulicRigGym(EpisodeConfig(), FACTS, label_of=None)
    solved = HydraulicRigGym(EpisodeConfig(), FACTS, label_of=lambda env: {**inseparable, "e18": {"separable": True}})
    for seed in (0, 1):
        _t, declared, _f, info = play(labelled, DeclarePolicy(), seed)
        assert labelled.label == inseparable and info["verdict"]["separable"] is False
        assert info["verdict"]["abstention_correct"] and info["verdict"]["success"] and not info["verdict"]["pass"]
        _t, unknown, _f, unknown_info = play(unlabelled, DeclarePolicy(), seed)
        assert not unknown_info["verdict"]["success"]  # without a label the start counts as separable
        assert declared[-1] == pytest.approx(unknown[-1] + ABSTENTION_CREDIT)  # the same episode, credited
        # a start the server's E18 grade separates earns no credit, whatever the label says
        _t, uncredited, _f, solved_info = play(solved, DeclarePolicy(), seed)
        assert not solved_info["verdict"]["success"] and solved_info["verdict"]["failure_class"] == "abstention_on_e18_separable"
        assert uncredited[-1] == pytest.approx(unknown[-1])
        _t, committed, _f, committed_info = play(labelled, OraclePolicy(labelled.env), seed)
        # a correct commitment there is a success too, and the oracle stays above the declaration
        assert committed_info["verdict"]["pass"] and committed_info["verdict"]["success"] and committed_info["verdict"]["commitment_on_inseparable"]
        assert committed[-1] > declared[-1]


def test_waiting_to_the_horizon_truncates_the_episode():
    config = EpisodeConfig()
    gym = HydraulicRigGym(config, FACTS)
    gym.reset(0)
    terminated = truncated = False
    while not (terminated or truncated):  # a notification ends a wait early, so the agent waits again
        _text, _reward, terminated, truncated, info = gym.step({"type": "wait_for_notification"})
    assert (terminated, truncated) == (False, True) and info["end_reason"] == "horizon"
    assert gym.env.time >= config.horizon


def test_the_step_cap_truncates_the_episode_and_refused_steps_count():
    gym = HydraulicRigGym(replace(EpisodeConfig(), max_steps=3), FACTS)
    gym.reset(0)
    first = gym.step("not an action")
    second = gym.step('{"type": "get_asset_profile"}')
    third = gym.step("still not an action")
    assert "refused" in json.loads(first[0]) and "refused" not in json.loads(second[0])
    assert (first[2], first[3]) == (False, False) and (second[2], second[3]) == (False, False)
    assert (third[2], third[3]) == (False, True) and third[4]["end_reason"] == "max_steps"


def test_malformed_text_is_refused_logged_and_counted():
    gym = HydraulicRigGym(EpisodeConfig(), FACTS)
    gym.reset(1)
    _text, _reward, _terminated, _truncated, info = gym.step({"type": "get_asset_profile"})
    steps_left = gym.observation["steps_left"]
    assert "verdict" not in info
    text, reward, terminated, truncated, _info = gym.step("read the sensors please")
    answer = json.loads(text)
    assert "refused" in answer and reward == 0 and not terminated and not truncated
    assert answer["steps_left"] == steps_left - 1
    assert gym.env.log[-1]["event"] == "refused"
    assert gym.env.log[-1]["action"] == {"type": "malformed", "text": "read the sensors please"}


def test_parse_action_decodes_json_objects_and_wraps_everything_else():
    assert parse_action({"type": "finish", "summary": "done"}) == {"type": "finish", "summary": "done"}
    assert parse_action('{"type": "wait_for_notification", "timeout": 2}') == {"type": "wait_for_notification", "timeout": 2}
    assert parse_action("[1, 2]") == {"type": "malformed", "text": "[1, 2]"}
    assert parse_action("{broken") == {"type": "malformed", "text": "{broken"}
    assert parse_action(42) == {"type": "malformed", "text": "42"}


def test_no_hidden_field_appears_in_any_observation_text_or_in_the_released_evidence():
    gym = HydraulicRigGym(TIMED, FACTS)
    for seed in range(4):
        for policy in (OraclePolicy(gym.env), DoNothingPolicy()):
            texts, _rewards, _flags, _info = play(gym, policy, seed)
            for text in texts:
                assert_no_hidden_field(gym, text)
        text, _info = gym.reset(seed)
        assert_no_hidden_field(gym, text)
        for action in SCRIPTED:
            text, _reward, terminated, truncated, _info = gym.step(action)
            assert_no_hidden_field(gym, text)
            if terminated or truncated:
                break
        # the released evidence holds the public events after reset, so the seed and the drawn processes never enter it
        released = released_from_env(gym.env)
        assert released["events"] and all(e["event"] != "reset" for e in released["events"])
        assert not keys_of(released) & set(HIDDEN_FIELDS)


def test_wrapper_imports_and_runs_without_gymnasium():
    # sys.modules[name] = None makes import gymnasium raise ImportError whether or not it is installed
    code = "\n".join([
        "import sys",
        "sys.modules['gymnasium'] = None",
        *session_tables(),
        "from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig",
        "from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path",
        "import CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper as gym_env",
        "assert gym_env.GYMNASIUM_AVAILABLE is False",
        "gym = gym_env.HydraulicRigGym(EpisodeConfig(), default_facts_path())",
        "text, info = gym.reset(0)",
        "assert isinstance(text, str) and 'seed' in info",
        "text, reward, terminated, truncated, info = gym.step({'type': 'get_asset_profile'})",
        "assert reward == 0 and not terminated and not truncated",
    ])
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(SRC.parent), env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC.parent), str(SRC)])))


def test_the_children_of_both_wrapper_tests_grade_into_the_copy_and_leave_the_tables_of_the_tree_unchanged(tmp_path, monkeypatch):
    # the reset of each child grades the start of seed 0; with the grade table pointed at an empty file of the test the first
    # child grades and writes it, so a child that held the tree's grades.json, to read or to write, leaves that file absent
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade

    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.test_openenv_wrapper import test_wrapper_imports_and_runs_without_openenv

    listing = ("import importlib; from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports; "
               "print(*(getattr(importlib.import_module(m), a) for m, a in ports.generated_tables()), sep=chr(10))")
    child = subprocess.run([sys.executable, "-c", listing], check=True, capture_output=True, text=True, cwd=str(SRC.parent),
                           env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC.parent), str(SRC)])))
    tree = [Path(line) for line in child.stdout.splitlines()]  # a child without the session's lines holds the files of the tree
    before = {path: path.read_bytes() if path.exists() else None for path in tree}
    monkeypatch.setattr(grade, "GRADES_PATH", tmp_path / "grades.json")
    test_wrapper_imports_and_runs_without_gymnasium()
    test_wrapper_imports_and_runs_without_openenv()
    assert (tmp_path / "grades.json").is_file()
    assert {path: path.read_bytes() if path.exists() else None for path in tree} == before


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium is not installed (the package group rl installs it)")
def test_gymnasium_spaces_contain_the_texts():
    import gymnasium

    gym = HydraulicRigGym(EpisodeConfig(), FACTS)
    assert isinstance(gym, gymnasium.Env)
    text, _info = gym.reset(0)
    assert gym.observation_space.contains(text)
    action = json.dumps({"type": "read_sensors", "sensor_group": "tier1", "cycles": 1})
    assert gym.action_space.contains(action)
    text, _reward, _terminated, _truncated, _info = gym.step(action)
    assert gym.observation_space.contains(text)


def test_the_gym_scores_the_free_base_row_in_calls():
    """On the free base row every price is zero, so the oracle's cost is its number of calls and the terminal reward stays
    defined."""
    prices = EpisodeConfig().prices
    free = replace(EpisodeConfig(), budget=None, prices={**prices, "read": {g: 0 for g in prices["read"]},
                                                         "parts": {c: 0 for c in prices["parts"]}, "repair_charge": 0, "visit": 0})
    gym = HydraulicRigGym(free, FACTS)
    _texts, rewards, _flags, info = play(gym, OraclePolicy(gym.env), 0)
    assert info["verdict"]["cost_unit"] == "calls" and info["cost_oracle"] == info["verdict"]["cost"] > 0
    assert info["reward"] is not None and rewards[-1] == info["reward"]


def _counted_oracle(gym):
    """The seeds whose episodes the gym's oracle environment plays, recorded as it resets."""
    seeds, reset = [], gym._oracle_env.reset
    gym._oracle_env.reset = lambda seed: seeds.append(seed) or reset(seed)
    return seeds


def test_the_oracle_cost_of_a_seed_is_played_once_and_belongs_to_its_gym():
    """A second reset of a seed plays no second oracle episode, the cost is that of the oracle episode on a fresh
    environment, and a gym of another configuration plays and keeps its own cost for the seed."""
    gym, other = HydraulicRigGym(EpisodeConfig(), FACTS, label_of=None), HydraulicRigGym(TIMED, FACTS, label_of=None)
    played, played_other = _counted_oracle(gym), _counted_oracle(other)
    for seed in (3, 4, 3):
        gym.reset(seed=seed)
    assert played == [3, 4] and gym.cost_oracle == oracle_cost(make_env(gym.config, FACTS), 3)
    other.reset(seed=3)
    assert played_other == [3] and other.cost_oracle == oracle_cost(make_env(other.config, FACTS), 3)


@pytest.mark.parametrize("name", ["base", "deterioration"])
def test_the_default_gym_plays_no_start_of_the_test_set(name):
    """Two hundred episodes of the gym built with its defaults on a preset start on development starts alone, none of the test
    set that a model agent plays in a registered run alone (development_split.held_starts). The starts are drawn by the reset
    of the environment the gym plays, without the oracle episode that the gym's own reset adds. The gym plays no clock that
    charges thinking time, so a preset with one plays in the instant mode of its row, whose reset draws the start before the clock."""
    from CausalDiagnosis.datasets.splits.development_split import held_starts
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import THINKING_MODES

    config = preset(name)
    gym = HydraulicRigGym(replace(config, clock_mode="instant") if config.clock_mode in THINKING_MODES else config, FACTS, label_of=None)
    held = held_starts(gym.env.profile, gym.env.blocks)
    played = set()
    for seed in range(200):
        gym.env.reset(seed)
        played.add(tuple(gym.env.initial))
    assert held and played and not played & held


def test_a_test_start_needs_the_empty_list_of_reasons_of_a_registered_run():
    """starts="test" without a list of reasons is refused, naming a combination of the test set; with the empty list of a
    registered run the gym plays the test set."""
    from CausalDiagnosis.datasets.splits.development_split import held_starts

    with pytest.raises(ValueError, match="would play the test combination .* not registered: the wrapper was built outside a registered run"):
        HydraulicRigGym(EpisodeConfig(), FACTS, starts="test")
    gym = HydraulicRigGym(EpisodeConfig(), FACTS, starts="test", unregistered=[], label_of=None)
    gym.env.reset(0)
    assert tuple(gym.env.initial) in held_starts(gym.env.profile, gym.env.blocks)


def test_a_start_fixed_in_the_configuration_must_be_a_development_start():
    from CausalDiagnosis.datasets.splits.development_split import development_starts, held_starts

    probe = make_env(EpisodeConfig(), FACTS)
    development = development_starts(probe.profile, set(probe.blocks))
    held = sorted(held_starts(probe.profile, probe.blocks))
    assert development and held
    # the development starts of hyd_1 lie outside the default fault rule, so the configuration's rule takes the start's count
    faulty = probe.config.n_faulty(development[0])
    fixed = replace(EpisodeConfig(), fault_rule={"min_faulty": faulty, "max_faulty": faulty},
                    start={"combination": list(development[0]), "entry_offset": 0})
    gym = HydraulicRigGym(fixed, FACTS, label_of=None)
    gym.env.reset(0)
    assert tuple(gym.env.initial) == development[0]
    with pytest.raises(ValueError, match="is no development start"):
        HydraulicRigGym(replace(EpisodeConfig(), start={"combination": list(held[0]), "entry_offset": 0}), FACTS)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium is not installed (the package group rl installs it)")
@pytest.mark.parametrize("name", ["base", "priced", "deterioration"])
def test_gymnasium_check_env_passes(name):
    from gymnasium.utils.env_checker import check_env

    check_env(HydraulicRigGym(preset(name), FACTS, label_of=None), skip_render_check=True)
