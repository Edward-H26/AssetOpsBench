from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from scenario_anchors import CLASSES, anchor_checks, class_env, class_numbers, start_state_table
from envs.hydraulic_rig.classifiers import FOLD_SEEDS, TierClassifiers
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import DEFAULT_PRICES
from envs.hydraulic_rig.start_states import start_states

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)
START_COMBINATIONS = {"S1": 10, "S2": 37}


@pytest.fixture(scope="module")
def fold_classifiers():
    return [TierClassifiers(CYCLES, BLOCKS, seed=seed) for seed in FOLD_SEEDS]


@pytest.fixture(scope="module")
def numbers(fold_classifiers):
    out = {}
    for name in CLASSES:
        env = class_env(name, CYCLES, BLOCKS)
        assert env.prices == DEFAULT_PRICES and len(env.candidates) == START_COMBINATIONS[name]
        assert env.reset(0)["min_faulty"] == env.reset(0)["max_faulty"] == CLASSES[name]
        out[name] = class_numbers(name, CYCLES, BLOCKS, fold_classifiers)
    return out


@pytest.mark.parametrize("name", list(CLASSES))
@pytest.mark.parametrize("anchor", [1, 2, 3, 4, 5, 6])
def test_the_admission_anchors_hold_with_the_default_prices(numbers, name, anchor):
    rows = [c for c in anchor_checks(numbers[name]) if c[0] == anchor]
    assert len(rows) == (len(FOLD_SEEDS) if anchor == 5 else 1)
    failed = [condition for _, _, _, _, holds, condition in rows if not holds]
    assert not failed, failed


@pytest.mark.parametrize("name", list(CLASSES))
def test_every_start_state_is_played_and_every_record_weighs_the_same(numbers, name):
    table = start_state_table(name, CYCLES, BLOCKS)
    per_record = Counter(combo for combo, _, _ in start_states(class_env(name, CYCLES, BLOCKS)))
    assert {combo: len(seeds) for combo, seeds in table.items()} == dict(per_record)
    for policy in ("oracle", "do_nothing", "replace_everything", "random", "symptom_only"):
        n = numbers[name][policy]
        assert n["episodes"] == sum(per_record.values()) and len(n["records"]) == START_COMBINATIONS[name]
    # symptom-only resolves exactly the record it guesses, which weighs 1/R whatever its number of start states
    assert numbers[name]["symptom_only"]["resolved"] == Fraction(1, START_COMBINATIONS[name])
    assert numbers[name]["oracle"]["cost_per_resolved_episode"] == 1
