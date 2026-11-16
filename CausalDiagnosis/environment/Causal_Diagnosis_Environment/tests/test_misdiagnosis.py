"""The misdiagnosis price that the environment charges: an accepted diagnosis that misses the condition at its write in
the level of any component adds the price of the rules to the hidden ledger once, as a system entry of the log that no
reply, notification, release or counter shows, on both engines of the recorded levels kind. The verdict names the charge
beside the penalties only when the log holds one, and a table reads a verdict without it as no charge. A true diagnosis,
a declaration, an episode without a write and the oracle pay nothing, the base row charges nothing, and a cost in calls
refuses a misdiagnosis price. No test states a value computed from the rig data."""
import json
from pathlib import Path

import pytest

from CausalDiagnosis.utils.paths import preset_files, preset_path  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import env as engine_module
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import END_LEDGER_FIELDS, WRITE_LOCK_REASON, HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose, unit_problems
from CausalDiagnosis.environment.reference_policies.policies import DeclarePolicy, OraclePolicy, run_episode
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import misdiagnosis_charge_of, released_from_env, verify
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing as rows_briefing
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing as rig_briefing, misdiagnosis_sentence

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, finish, truth

ROOT = Path(__file__).resolve().parents[4]
PRESETS = ROOT / "CausalDiagnosis" / "configs"
SEED = 3
TIMED = {"schedule": {"mode": "seeded", "components": 1, "first_drop": [5, 30], "interval": 10}}
TEXT_SUFFIXES = (".py", ".md", ".json", ".txt", ".toml", ".yaml", ".yml", ".sh", ".sbatch")


def preset(name):
    """A preset of the environment table, composed as the drivers compose it."""
    return compose(json.loads(preset_path(name).read_text(encoding="utf-8")))


def engine(world, config, kind=HydraulicRigEnv):
    cycles, blocks = world
    return kind(cycles, blocks, config)


def missed(env):
    """The hidden start with the first component at another of its levels, a diagnosis that misses the start in one level."""
    levels = truth(env)
    first = COMPONENTS[0]
    levels[first] = next(level for level in env.config.levels[first] if level != levels[first])
    return levels


def diagnose(env, levels):
    return env.step({"type": "submit_diagnosis", "diagnosis": dict(levels), "evidence_ids": []})


def verdict_of(env):
    return verify(env.log, released_from_env(env), env.oracle(), None)


def charges(env):
    return [e["cost"] for e in events(env.log, "misdiagnosis")]


def test_a_wrong_diagnosis_on_a_priced_preset_is_charged_once_on_the_hidden_ledger(world):
    """The write of a diagnosis that misses the start is followed by one hidden system entry with the misdiagnosis price at
    the cycle of the write, and a second write is refused and charges nothing. The cost of the episode, in the engine, in
    the end entry and in the verdict, exceeds that of the same steps with the true diagnosis by the price, through the
    penalties, and the cost less the charge that the verdict names puts both on one scale."""
    config = preset("priced")
    price = config.prices["misdiagnosis"]
    assert price > 0
    played = {}
    for name in ("true", "missed"):
        env = engine(world, config)
        env.reset(SEED)
        diagnose(env, truth(env) if name == "true" else missed(env))
        assert diagnose(env, missed(env))[0]["refused"] == WRITE_LOCK_REASON
        finish(env)
        played[name] = (env, verdict_of(env))
    (right, right_verdict), (wrong, verdict) = played["true"], played["missed"]
    assert charges(right) == [] and charges(wrong) == [price]
    write, entry = events(wrong.log, "submit_diagnosis")[0], events(wrong.log, "misdiagnosis")[0]
    assert wrong.log.index(entry) == wrong.log.index(write) + 1
    assert (entry["step"], entry["class"], entry["t_start"], entry["t_end"]) == (None, "system", write["t_start"], write["t_start"])
    assert not verdict["diagnosis_all_correct"] and verdict["misdiagnosis_charge"] == misdiagnosis_charge_of(verdict) == price
    assert wrong.cost == wrong.log[-1]["cost"] == verdict["cost"] == right_verdict["cost"] + price
    assert verdict["penalties"] == right_verdict["penalties"] + price
    assert wrong.ledger["penalties"] == right.ledger["penalties"] + price
    assert verdict["cost"] - misdiagnosis_charge_of(verdict) == right_verdict["cost"]


def test_the_charge_leaves_every_reply_and_the_release_unchanged(world, monkeypatch):
    """The same steps with the charge and with the charge taken out of the engine: every answer, the info of every step
    without the hidden log of the last, the counters, the public events and the release are equal, and both servers
    answer a call with the envelope alone, from which the receipt line of the AssetOpsBench rows is read. Only the hidden
    log, the step cost of the write and the cost of the episode differ, by the misdiagnosis entry and its price."""
    config = preset("priced")
    price = config.prices["misdiagnosis"]

    def play():
        env = engine(world, config)
        answers, step_costs = [env.reset(SEED)], []
        faulty = next(c for c in COMPONENTS if truth(env)[c] != HEALTHY[c])
        for action in ({"type": "read_sensors", "sensor_group": "tier1", "cycles": 1},
                       {"type": "submit_diagnosis", "diagnosis": missed(env), "evidence_ids": []},
                       {"type": "create_work_order", "component": faulty, "prediction_id": "", "evidence_ids": []},
                       {"type": "finish", "summary": "done"}):
            envelope, cost, done, info = env.step(action)
            answers.append((envelope, done, {k: v for k, v in info.items() if k != "log"}))
            step_costs.append(cost)
        return env, answers, step_costs

    charged, charged_answers, charged_costs = play()
    monkeypatch.setattr(engine_module, "misdiagnosis_charge", lambda log, diagnosis: 0)
    plain, plain_answers, plain_costs = play()
    assert charged_answers == plain_answers
    assert [a - b for a, b in zip(charged_costs, plain_costs)] == [0, price, 0, 0]
    assert (charged.budget_left(), charged.prices_paid) == (plain.budget_left(), plain.prices_paid)
    assert charged.public_events() == plain.public_events()
    assert released_from_env(charged) == released_from_env(plain)

    def hidden_without_the_charge(env):
        log = [dict(e) for e in env.log if e["event"] != "misdiagnosis"]
        for key in END_LEDGER_FIELDS:
            log[-1].pop(key, None)
        return log

    assert charges(charged) == [price] and charges(plain) == []
    assert hidden_without_the_charge(charged) == hidden_without_the_charge(plain)
    assert charged.cost == plain.cost + price


def test_a_true_diagnosis_a_declaration_no_write_and_the_oracle_are_not_charged(world):
    """No diagnosis, no charge, and a true diagnosis pays nothing: the log holds no misdiagnosis entry, the verdict no
    field, and the reader of the field gives no charge."""
    config = preset("priced")

    def true_write(env):
        env.reset(SEED)
        diagnose(env, truth(env))
        finish(env)

    def no_write(env):
        env.reset(SEED)
        finish(env)

    def declaration(env):
        run_episode(env, DeclarePolicy(), SEED)

    def oracle(env):
        env.reset(SEED)
        run_episode(env, OraclePolicy(env), SEED)

    for play in (true_write, no_write, declaration, oracle):
        env = engine(world, config)
        play(env)
        verdict = verdict_of(env)
        assert charges(env) == [] and "misdiagnosis_charge" not in verdict, play.__name__
        assert misdiagnosis_charge_of(verdict) == 0, play.__name__


def test_a_fault_diagnosed_on_a_healthy_start_is_charged_on_a_priced_row(world):
    """The fault rule of the healthy control draws the start with every component at its healthy level. On the priced row
    a diagnosis that names a fault there misses it and pays the price, the healthy diagnosis pays nothing, and the control
    preset itself, on the base row, prices no misdiagnosis and charges nothing."""
    spec = json.loads((PRESETS / "healthy_control.json").read_text(encoding="utf-8"))
    on_priced, control = compose({**spec, "layers": ["priced"]}), compose(spec)
    price = on_priced.prices["misdiagnosis"]
    first = COMPONENTS[0]
    fault = {**HEALTHY, first: next(level for level in on_priced.levels[first] if level != HEALTHY[first])}
    for config, diagnosis, expected in ((on_priced, fault, [price]), (on_priced, dict(HEALTHY), []), (control, fault, [])):
        env = engine(world, config)
        env.reset(SEED)
        assert truth(env) == HEALTHY
        diagnose(env, diagnosis)
        finish(env)
        verdict = verdict_of(env)
        assert charges(env) == expected and misdiagnosis_charge_of(verdict) == sum(expected)
        assert verdict["diagnosis_all_correct"] == (diagnosis == HEALTHY)


def test_the_base_row_charges_nothing_and_keeps_its_verdict_keys(world):
    """On the base row every price is zero, the misdiagnosis price included, so a wrong diagnosis logs no charge and its
    verdict has the keys of the verdict of the true diagnosis, without misdiagnosis_charge."""
    base = preset("base")
    assert base.prices["misdiagnosis"] == 0 and base.cost_unit() == "calls"
    verdicts = []
    for levels in (truth, missed):
        env = engine(world, base)
        env.reset(SEED)
        diagnose(env, levels(env))
        finish(env)
        assert charges(env) == []
        verdicts.append(verdict_of(env))
    assert verdicts[0]["diagnosis_all_correct"] and not verdicts[1]["diagnosis_all_correct"]
    assert set(verdicts[0]) == set(verdicts[1]) and "misdiagnosis_charge" not in verdicts[1]


def test_both_engines_charge_the_same(world):
    """The engine of the rig and the engine of the AssetOpsBench rows log the same charge for the same wrong diagnosis, and
    their costs and verdicts agree, as their oracle costs do."""
    config = preset("priced")
    seen = []
    for kind in (HydraulicRigEnv, RecordedRigEnv):
        env = engine(world, config, kind)
        env.reset(SEED)
        diagnose(env, missed(env))
        finish(env)
        verdict = verdict_of(env)
        assert env.cost == verdict["cost"], kind.__name__
        seen.append((events(env.log, "misdiagnosis"), env.cost, verdict["cost"], verdict["misdiagnosis_charge"]))
    assert seen[0] == seen[1] and seen[0][3] == config.prices["misdiagnosis"]
    assert oracle_cost(engine(world, config), SEED) == oracle_cost(engine(world, config, RecordedRigEnv), SEED)


def test_a_misdiagnosis_price_under_a_cost_in_calls_is_refused():
    """A misdiagnosis price above zero charges a hidden ledger that a cost in calls leaves out, so the unit rule names it in a
    sentence of its own and the misdiagnosis switch alone over the base is refused, while every preset of the environment
    table composes and prices no misdiagnosis wherever its cost unit is calls."""
    base = compose({}).to_json()
    problems = unit_problems({**base, "prices": {**base["prices"], "misdiagnosis": 1}})
    assert len(problems) == 1 and problems[0].startswith("the cost unit is calls, since every price of a read and of a work")
    assert problems[0].endswith("or switch misdiagnosis off")
    with pytest.raises(ValueError, match="while the misdiagnosis price charges a hidden ledger"):
        compose({"switches": {"misdiagnosis": "on"}})
    assert compose({"switches": {"misdiagnosis": "on", "part_prices": "on"}}).cost_unit() == "money"
    for path in sorted(preset_files(PRESETS).values()):
        config = compose(json.loads(path.read_text(encoding="utf-8")))
        assert config.cost_unit() == "money" or config.prices["misdiagnosis"] == 0, path.stem


def _first_drop(make_env):
    """The first seed whose seeded schedule changes the hidden condition before the horizon, with the cycle after the change
    and the condition before and after it."""
    for seed in range(20):
        env = make_env(**TIMED)
        env.reset(seed)
        start = list(env.hidden)
        while not env.done and list(env.hidden) == start:
            advance(env, 1)
        if not env.done:
            return seed, env.time, start, list(env.hidden)
    raise AssertionError("no seed of the first twenty changes the hidden condition before the horizon")


def test_the_charge_reads_the_truth_at_the_write(make_env):
    """On a schedule that lowers a level during the episode, the start written before the drop pays nothing though the
    level drops after the write, the start written after the drop pays the price, and the condition after the drop
    written before the drop pays the price: the charge and diagnosis_all_correct read the condition at the write."""
    seed, dropped_at, start, dropped = _first_drop(make_env)
    price = EpisodeConfig().prices["misdiagnosis"]
    cases = {"start before the drop": (0, start, []), "start after the drop": (dropped_at, start, [price]),
             "drop before the drop": (0, dropped, [price])}
    for name, (write_at, levels, expected) in cases.items():
        env = make_env(**TIMED)
        env.reset(seed)
        if write_at:
            advance(env, write_at)
        assert list(env.hidden) == (dropped if write_at else start), name
        diagnose(env, dict(zip(COMPONENTS, levels)))
        advance(env, dropped_at + 1 - env.time)
        finish(env)
        verdict = verdict_of(env)
        assert charges(env) == expected and verdict["diagnosis_all_correct"] == (not expected), name


def _text_files():
    """The text files of the fork's own folders, without caches and without the data files and generated tables above a
    megabyte, which hold no prose."""
    for top in ("src", "benchmarks", "docs", "examples"):
        for path in sorted((ROOT / top).rglob("*")):
            if (path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts
                    and path.stat().st_size < 1 << 20):
                yield path


def test_the_briefing_states_the_charge_and_no_text_says_no_tool_charges(world):
    """Both briefings state in the submit_diagnosis entry of the priced preset that a wrong diagnosis costs the misdiagnosis
    price, added to the cost of the episode and not taken from the budget, and say nothing of a misdiagnosis on the base
    row. No text of the fork keeps the claim of the old briefing that the price is declared and never charged."""
    priced = preset("priced")
    sentence = misdiagnosis_sentence(priced.prices)
    assert sentence.startswith(f"A wrong diagnosis costs the misdiagnosis price of {priced.prices['misdiagnosis']}, the")
    assert sentence.endswith("does not fall from the budget.")
    for name, stated in (("priced", True), ("base", False)):
        config = preset(name)
        rig, rows = engine(world, config), engine(world, config, RecordedRigEnv)
        rig.reset(SEED)
        rows.reset(SEED)
        for text in (rig_briefing(rig.public_profile(), config), rows_briefing(rows.public_profile(), config)):
            assert (sentence in text) == stated and ("misdiagnosis" in text) == stated, name
    phrase = " ".join(("no", "tool", "charges"))
    stale = [str(p.relative_to(ROOT)) for p in _text_files() if phrase in p.read_text(encoding="utf-8", errors="replace")]
    assert stale == []
