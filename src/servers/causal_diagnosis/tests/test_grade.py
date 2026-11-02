"""E18 on the fitted models with a tiny configuration: rollouts stop within the budget, a budget too small for any reading
makes every start inseparable, and grades.json caches one table per rules key."""
import json
from pathlib import Path

import numpy as np
import pytest

from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from servers.causal_diagnosis import grade as gr
from servers.causal_diagnosis.adapter import load_models
from servers.causal_diagnosis.effect_model import GROUP_FIELDS, PoolData
from servers.causal_diagnosis.posterior import Posterior
from servers.causal_diagnosis.tests.fixture_episode import PLACEHOLDER_RULES

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

RULES = {**PLACEHOLDER_RULES, "sensor_groups": {g: list(f) for g, f in GROUP_FIELDS.items()}}
START = (100, 80, 0, 90)


@pytest.fixture(scope="module")
def block():
    cycles = load_cycles(default_facts_path())
    return PoolData(cycles, build_blocks(cycles), [START]).arrays[0]


def test_information_gain_is_nonnegative_and_largest_for_the_richest_group():
    models = load_models()
    post = Posterior(models.effect, RULES, [], [], time=0)
    rng = np.random.default_rng(0)
    gains = {g: gr.expected_information_gain(post, GROUP_FIELDS[g], 256, rng) for g in GROUP_FIELDS}
    assert all(v >= -1e-9 for v in gains.values()) and gains["tier3"] >= gains["tier1"] - 0.05


def test_rollouts_respect_the_budget_and_a_budget_below_any_price_is_inseparable(block):
    models = load_models()
    rng = np.random.default_rng(0)
    committed, spent, n = gr.rollout(START, 40, models.effect, RULES, block, rng, samples=32)
    assert spent <= 40 and n >= 0 and isinstance(committed, bool)
    assert gr.grade(START, 0, models.effect, RULES, block, rng, rollouts=2, samples=32) is None
    result = gr.grade(START, 300, models.effect, RULES, block, rng, rollouts=2, samples=32)
    assert result is None or (0 < result["commit_share"] <= 1 and result["cost"] <= 300 and result["rollouts"] == 2)


def test_grades_json_caches_one_table_per_rules_key(tmp_path, monkeypatch):
    models = load_models()
    with pytest.raises(ValueError):
        gr.compute_grades(models, RULES, 300, ranking_model="discriminant_urn")
    calls = []

    def fake_compute(models_, rules, budget, ranking_model="joint", **options):
        calls.append(budget)
        return {"rules_key": gr.rules_key(rules, budget, ranking_model), "budget": budget, "grades": {"100,80,0,90": None}}

    monkeypatch.setattr(gr, "compute_grades", fake_compute)
    path = tmp_path / "grades.json"
    first = gr.ensure_grades(models, RULES, path=path)
    again = gr.ensure_grades(models, RULES, path=path)
    other = gr.ensure_grades(models, RULES, budget=10, path=path)
    literal = gr.ensure_grades(models, RULES, ranking_model="joint", path=path)
    assert first == again and calls == [RULES["budget"], 10, RULES["budget"]] and other["budget"] == 10
    assert first["rules_key"] == gr.rules_key(RULES, RULES["budget"], "joint_block") and literal["rules_key"] == gr.rules_key(RULES, RULES["budget"], "joint")
    stored = json.loads(path.read_text())
    assert stored["format"] == gr.GRADES_FORMAT and len(stored["tables"]) == 3
    assert gr.grade_of(START, models, RULES, path=path) is None


def test_the_command_line_regenerates_a_table_for_the_default_configuration(tmp_path):
    path = tmp_path / "grades.json"
    gr.main(["--budget", "2", "--rollouts", "1", "--samples", "8", "--out", str(path)])
    stored = json.loads(path.read_text())
    table = next(iter(stored["tables"].values()))
    assert table["ranking_model"] == "joint_block" and table["budget"] == 2 and len(table["grades"]) == len(load_models().pool["evaluation"])
    assert all(v is None for v in table["grades"].values())  # no reading is affordable under a budget below every price plus downtime
    gr.main(["--budget", "2", "--rollouts", "1", "--samples", "8", "--out", str(path), "--force", "--ranking-model", "joint"])
    assert len(json.loads(path.read_text())["tables"]) == 2
