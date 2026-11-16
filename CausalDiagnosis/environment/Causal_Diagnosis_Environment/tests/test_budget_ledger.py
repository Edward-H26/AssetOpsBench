"""The budget counter remedy: the budget counter falls by the
prices paid alone, the downtime of the hidden condition and the failure penalties go to a hidden ledger that the end entry
of the log keeps, the verifier's cost is the prices plus that ledger, and no agent facing view carries a number that the
ledger drives. No test states a value computed from the rig data."""
import json

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import END_LEDGER_FIELDS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import released_from_env, verify

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, events, faulty_components, finish, keys_of, order, read, submit_truth, wait

HAND_START = (3, 90, 0, 130)  # the cooler at its worst level, the valve one level below healthy
MILDER_START = (20, 100, 0, 130)  # one level of the cooler down, a smaller downtime rate than HAND_START
TIMED = {"schedule": {"mode": "seeded", "components": 1, "first_drop": [5, 30], "interval": 10}}
LEDGER_KEYS = {"ledger", "downtime_total", "downtime_by_cycle"}


def anchored(make_env, combination=HAND_START, **overrides):
    return make_env(start={"combination": list(combination), "entry_offset": 0}, **overrides)


def rate_of(config, combination):
    return sum(config.level_rate(c, level) for c, level in zip(COMPONENTS, combination))


def test_waiting_without_a_purchase_leaves_the_budget_counter_where_it_was(make_env):
    env = make_env(**TIMED)
    env.reset(0)
    obs, _cost, done, _info = advance(env, 20)
    assert not done and obs["budget_left"] == env.config.budget == env.budget_left()
    assert env.prices_paid == 0 and env.ledger["downtime"] > 0  # every start is degraded, so the ledger grew meanwhile


def test_two_hidden_conditions_answer_the_same_calls_with_the_same_counter(make_env):
    counters, ledgers = [], []
    for start in (HAND_START, MILDER_START):
        env = anchored(make_env, start)
        env.reset(0)
        answers = [read(env, "tier1", 2)[0], wait(env, 5)[0], read(env, "tier2", 1)[0], env.step({"type": "get_current_time"})[0]]
        counters.append([a["budget_left"] for a in answers] + [answers[-1]["result"]["budget_left"]])
        ledgers.append(env.ledger["downtime"])
    assert counters[0] == counters[1]  # the counter tells nothing about the hidden condition
    assert ledgers[0] > ledgers[1] > 0


def test_the_ledger_grows_by_the_rates_of_the_hidden_condition_and_holds_the_penalties(make_env):
    patience, penalty = 4, 50
    env = anchored(make_env, failure={"patience": patience, "strict": False, "penalty": penalty})
    env.reset(0)
    obs, _cost, done, _info = advance(env, 9)
    assert not done and len(events(env.log, "failure")) == 1
    assert env.ledger == {"downtime": 9 * rate_of(env.config, HAND_START), "penalties": penalty}
    assert obs["budget_left"] == env.config.budget  # neither the downtime nor the penalty moves the counter
    finish(env)
    end = env.log[-1]
    assert end["ledger"] == env.ledger and end["downtime_total"] == env.ledger["downtime"]
    assert end["cost"] == env.cost == env.prices_paid + env.ledger["downtime"] + env.ledger["penalties"]


def test_the_verifier_cost_is_the_prices_plus_the_ledger(make_env):
    env = anchored(make_env)
    env.reset(0)
    read(env, "tier1", 1)
    submit_truth(env)
    for component in faulty_components(HAND_START):
        order(env, component)
    advance(env, 10)
    finish(env)
    verdict = verify(env.log, released_from_env(env), env.oracle(), None)
    assert verdict["cost_prices"] == env.prices_paid > 0 and env.budget_left() == env.config.budget - env.prices_paid
    assert (verdict["downtime_total"], verdict["penalties"]) == (env.ledger["downtime"], env.ledger["penalties"])
    assert verdict["cost"] == env.prices_paid + env.ledger["downtime"] + env.ledger["penalties"] == env.log[-1]["cost"]


def test_the_budget_stops_on_prices_alone_however_large_the_ledger(make_env):
    rates = {c: {str(level): 9 for level in EpisodeConfig().levels[c]} for c in COMPONENTS}
    env = anchored(make_env, budget=5, downtime_rate={**rates, "stopped": 9})
    env.reset(0)
    obs, _cost, done, _info = advance(env, 20)
    assert not done and env.ledger["downtime"] > env.config.budget and obs["budget_left"] == env.config.budget
    _obs, _cost, done, _info = read(env, "tier3", 1)  # the price alone exceeds what the budget leaves
    assert done and env.log[-1]["reason"] == "budget" and env.prices_paid == 0


def test_no_agent_facing_view_carries_a_number_of_the_ledger(make_env):
    env = make_env(**TIMED)
    env.reset(1)
    answers = [read(env, "tier1", 2)[0], wait(env, 6)[0], submit_truth(env)[0]]
    answers += [order(env, faulty_components(env.initial)[0])[0], advance(env, 15)[0], finish(env)[0]]
    for answer in answers:
        assert not keys_of(answer) & LEDGER_KEYS
        assert all("cost" not in note for note in answer["notifications"])
    end = env.public_event(env.log[-1])
    assert end["event"] == "end" and not set(END_LEDGER_FIELDS) & set(end)
    released = released_from_env(env)
    assert not keys_of(released) & LEDGER_KEYS and "ledger" not in json.dumps(released["status"])
    assert all("cost" not in e for e in released["events"] if e["event"] == "end")
