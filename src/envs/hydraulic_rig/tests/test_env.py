import itertools
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, TIER_SENSORS, build_blocks, default_facts_path, load_cycles, tier_features
from envs.hydraulic_rig.env import DEFAULT_PRICES, HEALTHY_COMBO, TinyHydraulicEnv, closed_under_repair, n_faulty
from envs.hydraulic_rig.policies import OraclePolicy, RandomPolicy, run_episode
from envs.hydraulic_rig.scoring import score

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)


def test_blocks_cover_all_combinations_with_one_contiguous_stable_block_each():
    assert [c.index for c in CYCLES] == list(range(len(CYCLES)))
    assert set(BLOCKS) == set(itertools.product(*(LEVELS[c] for c in COMPONENTS)))
    window = TinyHydraulicEnv(CYCLES, BLOCKS).block_cap
    for combo, idx in BLOCKS.items():
        assert len(idx) >= window
        assert idx == list(range(idx[0], idx[0] + len(idx)))
        assert all(CYCLES[i].stable and CYCLES[i].combo == combo for i in idx)
    assert tuple(HEALTHY[c] for c in COMPONENTS) in BLOCKS
    for c in COMPONENTS:
        assert set(LEVELS[c]) == {k[COMPONENTS.index(c)] for k in BLOCKS}


def test_tier_features_select_the_right_sensors():
    cyc = CYCLES[0]
    f1, f2, f3 = (tier_features(cyc, t) for t in (1, 2, 3))
    assert len(f1) == 4 * len(TIER_SENSORS[1]) and len(f2) == 4 * len(TIER_SENSORS[2]) and len(f3) == 4 * 17
    assert all(("PS" not in n) and ("FS" not in n) for n in f1)
    assert any("_FS1_" in n for n in f2) and any("_PS6_" in n for n in f3)


def test_reset_is_reproducible_and_hidden_state_is_not_observable():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    o1 = env.reset(7); h1 = env.initial; m1 = env.step({"type": "measure", "tier": 3})[0]["history"][-1]
    env.reset(7); h2 = env.initial; m2 = env.step({"type": "measure", "tier": 3})[0]["history"][-1]
    assert h1 == h2 and m1 == m2
    assert "hidden" not in o1 and all(k not in str(o1) for k in ("cooler_condition", "valve_condition"))
    assert h1 != tuple(HEALTHY[c] for c in COMPONENTS)
    assert env.min_faulty <= sum(1 for i, c in enumerate(COMPONENTS) if h1[i] != HEALTHY[c]) <= env.max_faulty


def test_replace_is_refused_before_the_diagnosis_then_moves_to_the_repaired_block():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_repairs=4)
    env.reset(3)
    faulty = next(c for i, c in enumerate(COMPONENTS) if env.initial[i] != HEALTHY[c])
    obs, cost, done, info = env.step({"type": "replace", "component": faulty})
    assert info.get("refused") and cost == 0 and not done and tuple(env.hidden) == env.initial
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})
    expected = list(env.initial); expected[COMPONENTS.index(faulty)] = HEALTHY[faulty]
    obs, cost, done, _ = env.step({"type": "replace", "component": faulty})
    assert cost == env.prices["parts"][faulty] + env.prices["downtime"]
    assert tuple(env.hidden) == tuple(expected)
    # the served block belongs to the repaired combination whether or not the episode ended on this repair
    assert all(CYCLES[i].combo == tuple(expected) and CYCLES[i].stable for i in env._block)


def test_budget_refusal_ends_the_episode_without_a_submission():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, budget=5)
    env.reset(1)
    obs, cost, done, info = env.step({"type": "measure", "tier": 3})
    assert done and cost == 0 and env.log[-2]["event"] == "refused" and env.log[-1]["reason"] == "budget"
    assert score(env.log)["submitted"] is False


def test_scorer_is_a_function_of_the_log_and_the_oracle_is_the_upper_bound():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    for seed in range(20):
        log = run_episode(env, OraclePolicy(env), seed)
        s = score(log)
        assert s["recovered"] and s["diagnosis_all_correct"] and s["unnecessary_replacements"] == 0 and s["repairs_match_diagnosis"]
        assert s["end_reason"] == "recovered"
        saved = json.loads(json.dumps(log))  # the scorer must reproduce the numbers from a log saved to disk, as is
        assert s == score(saved)
        log_r = run_episode(env, RandomPolicy(seed), seed)
        assert score(log_r)["cost"] <= env.budget


def test_observation_never_leaks_hidden_state_cycle_numbers_or_repair_flags():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_repairs=4)
    env.reset(3)
    env.step({"type": "measure", "tier": 3})
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})
    healthy_comp = next((c for i, c in enumerate(COMPONENTS) if env.initial[i] == HEALTHY[c]), None)
    if healthy_comp:
        env.step({"type": "replace", "component": healthy_comp})
    obs, *_ = env.step({"type": "measure", "tier": 1}) if not env.done else (env.observation(),)
    s = repr(obs)
    for forbidden in ("combo_after", "unnecessary", "'cycle'", "hidden", "cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", repr(env.initial), repr(tuple(env.hidden))):
        assert forbidden not in s, forbidden


def test_step_limit_ends_an_episode_that_never_submits():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_steps=5)
    env.reset(2)
    for _ in range(4):
        obs, cost, done, info = env.step({"type": "measure", "tier": 1})
        assert not done and obs["steps_left"] == env.max_steps - env.steps
    obs, cost, done, info = env.step({"type": "measure", "tier": 1})
    assert done and cost == 1 and info["reason"] == "step limit"
    s = score(env.log)
    assert s["submitted"] is False and s["end_reason"] == "step limit" and s["cost"] == 5 and s["cost_before_submit"] is None


def test_malformed_and_out_of_order_actions_are_refused_without_charge_or_state_change():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    env.reset(5)
    before = (tuple(env.hidden), env._pos, env._seen, env.cost)
    bad = [
        {"type": "teleport"},
        {"type": "measure", "tier": 4},
        {"type": "measure"},
        {"type": "submit", "diagnosis": {"cooler": 100}},
        {"type": "submit", "diagnosis": {c: 999 for c in COMPONENTS}},
        {"type": "submit", "diagnosis": {**dict(zip(COMPONENTS, env.initial)), "pump": True}},
        {"type": "submit", "diagnosis": {**dict(zip(COMPONENTS, env.initial)), "cooler": float(env.initial[0])}},
        {"type": "submit", "diagnosis": {**dict(zip(COMPONENTS, env.initial)), "valve": str(env.initial[1])}},
        {"type": "submit", "insufficient_evidence": True, "diagnosis": dict(zip(COMPONENTS, env.initial))},
        {"type": "submit", "insufficient_evidence": True},
        {"type": "finish"},
    ]
    for action in bad:
        obs, cost, done, info = env.step(action)
        assert info.get("refused") and cost == 0 and not done, action
        assert (tuple(env.hidden), env._pos, env._seen, env.cost) == before
    assert sum(1 for e in env.log if e["event"] == "refused") == len(bad)
    assert env.phase == "investigate"
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})
    obs, cost, done, info = env.step({"type": "replace", "component": "turbine"})
    assert info.get("refused") == "unknown component" and env.cost == before[3]


def test_insufficient_evidence_is_a_valid_submission_that_is_not_scored_as_a_diagnosis():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    env.reset(9)
    env.step({"type": "measure", "tier": 2})
    obs, cost, done, info = env.step({"type": "submit", "insufficient_evidence": True, "next_action": "measure(tier=3)"})
    assert not info.get("refused") and env.phase == "repair"
    env.step({"type": "finish"})
    s = score(env.log)
    assert s["submitted"] and s["insufficient_evidence"] and s["diagnosis_all_correct"] is False
    assert all(v is None for v in s["diagnosis_per_component"].values()) and s["cost_before_submit"] == 3


def test_every_served_block_has_the_same_length_and_a_healthy_replacement_does_not_move_it():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_repairs=4)
    for seed in range(60):
        env.reset(seed)
        assert len(env._block) == env.block_cap == 10
        assert all(CYCLES[i].combo == env.initial for i in env._block)
    env.reset(3)
    env.step({"type": "measure", "tier": 1})
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})
    healthy_comp = next(c for i, c in enumerate(COMPONENTS) if env.initial[i] == HEALTHY[c])
    block, pos, seen = list(env._block), env._pos, env._seen
    env.step({"type": "replace", "component": healthy_comp})
    assert (list(env._block), env._pos, env._seen) == (block, pos, seen)
    assert env.log[-1]["unnecessary"] is True


def test_allowed_split_restricts_the_starting_combinations():
    train = {k for k in BLOCKS if 1 <= sum(1 for i, c in enumerate(COMPONENTS) if k[i] != HEALTHY[c]) <= 2 and k[0] != 3}
    env = TinyHydraulicEnv(CYCLES, BLOCKS, allowed=train)
    assert set(env.candidates) == train and len(env.candidates) < 47
    for seed in range(40):
        env.reset(seed)
        assert env.initial in train


def test_holding_out_the_two_fault_combinations_is_closed_under_repair_and_a_random_split_is_not():
    candidates = [k for k in BLOCKS if 1 <= n_faulty(k) <= 2]
    train = {k for k in candidates if n_faulty(k) == 1}
    held_out = {k for k in candidates if n_faulty(k) == 2}
    assert len(train) == 10 and len(held_out) == 37 and closed_under_repair(train, held_out)
    ordered = sorted(candidates)
    assert not closed_under_repair(set(ordered[::2]), set(ordered[1::2]))
    env = TinyHydraulicEnv(CYCLES, BLOCKS, allowed=train)
    for seed in range(20):
        env.reset(seed)
        env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})
        while not env.done:
            faulty = [c for i, c in enumerate(COMPONENTS) if env.hidden[i] != HEALTHY[c]]
            env.step({"type": "replace", "component": faulty[0]} if faulty else {"type": "finish"})
            assert tuple(env.hidden) not in held_out


def test_tier_must_be_an_integer_and_the_refused_action_is_visible_to_the_agent():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    obs = env.reset(4)
    assert obs["max_faulty"] == env.max_faulty
    for bad in ("1", 1.0, True, None):
        obs, cost, done, info = env.step({"type": "measure", "tier": bad})
        assert info.get("refused") == "unknown tier" and cost == 0
    assert obs["history"][-1] == {"event": "refused", "action": {"type": "measure", "tier": None}, "reason": "unknown tier"}
    obs, cost, done, info = env.step({"type": "measure", "tier": 1})
    assert cost == 1 and not info.get("refused")


def test_the_log_keeps_copies_so_a_policy_cannot_rewrite_its_diagnosis_afterwards():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    env.reset(6)
    wrong = {c: LEVELS[c][0] for c in COMPONENTS}
    obs = env.step({"type": "submit", "diagnosis": wrong})[0]
    before = score(env.log + [{"event": "end", "cost": 0, "reason": "probe"}])
    wrong["cooler"] = env.initial[0]  # the policy edits the dict it handed in
    obs["history"][-1]["diagnosis"] = dict(zip(COMPONENTS, env.initial))  # and the observation it was shown
    after = score(env.log + [{"event": "end", "cost": 0, "reason": "probe"}])
    assert before == after and before["diagnosis_all_correct"] is False


def test_steps_equal_the_environment_counter_including_an_accepted_finish():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    env.reset(8)
    env.step({"type": "measure", "tier": 2})
    env.step({"type": "finish"})  # refused in the investigate phase
    env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})
    env.step({"type": "finish"})
    s = score(env.log)
    assert env.steps == 4 and s["steps"] == 4 and s["steps_before_submit"] == 3 and s["end_reason"] == "finish"


def test_random_policy_stream_is_separate_from_the_environment_stream():
    assert RandomPolicy(5).rng.getstate() != random.Random(5).getstate()


# Values written out from docs/hydraulic_rig/environment.md, so that the tests below check the code against the document, not against its own constants.
SPEC_TIER_CHANNELS = {
    1: {"TS1", "TS2", "TS3", "TS4", "VS1", "CE"},
    2: {"TS1", "TS2", "TS3", "TS4", "VS1", "CE", "FS1", "FS2", "CP"},
    3: {"PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"},
}
SPEC_LEVELS = {"cooler": [3, 20, 100], "valve": [73, 80, 90, 100], "pump": [0, 1, 2], "accumulator": [90, 100, 115, 130]}
SPEC_HEALTHY = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
SPEC_END_REASONS = {"finish", "recovered", "repairs used up", "budget", "step limit", "no repair phase"}
HAND_START = (3, 90, 0, 130)  # cooler and valve faulty


def faulty_components(combo):
    return [c for i, c in enumerate(COMPONENTS) if combo[i] != HEALTHY[c]]


def repaired(combo, component):
    fixed = list(combo)
    fixed[COMPONENTS.index(component)] = HEALTHY[component]
    return tuple(fixed)


def env_from(start, **kwargs):
    """An environment whose episodes all start from one combination (an `allowed` split with a single member)."""
    return TinyHydraulicEnv(CYCLES, BLOCKS, allowed={start}, **kwargs)


def submit_truth(env):
    return env.step({"type": "submit", "diagnosis": dict(zip(COMPONENTS, env.initial))})


def measured_cycles(log):
    return [e["cycle"] for e in log if e["event"] == "measure"]


def is_window(cycles, block):
    """Ten contiguous stable cycles inside the block."""
    return sorted(cycles) == list(range(min(cycles), min(cycles) + 10)) and set(cycles) <= set(block)


def state(env):
    return (tuple(env.hidden), env.phase, env.cost, env.repairs, list(env._block), env._pos, env._seen, env.done)


def hand_scored_episode():
    """A refusal, one measurement at each tier, a diagnosis right on the cooler and wrong on the valve, an unnecessary
    and a necessary replacement, and finish; the test below works out every score of docs/hydraulic_rig/environment.md by hand."""
    env = env_from(HAND_START, max_repairs=3)
    env.reset(0)
    for action in ({"type": "teleport"},
                   {"type": "measure", "tier": 1}, {"type": "measure", "tier": 2}, {"type": "measure", "tier": 3},
                   {"type": "submit", "diagnosis": {"cooler": 3, "valve": 80, "pump": 0, "accumulator": 130}},
                   {"type": "replace", "component": "accumulator"},
                   {"type": "replace", "component": "cooler"},
                   {"type": "finish"}):
        env.step(action)
    return env


def budget_ended_episode():
    """The budget is one below the price of a pump replacement, so the repair of the faulty pump is refused."""
    env = env_from((100, 100, 1, 130), budget=DEFAULT_PRICES["parts"]["pump"] + DEFAULT_PRICES["downtime"] - 1)
    env.reset(0)
    submit_truth(env)
    env.step({"type": "replace", "component": "pump"})
    return env


class AlwaysFinish:
    """Sends finish during the investigate phase, which is refused, so only the step cap can end the episode."""

    def act(self, obs):
        return {"type": "finish"}


def step_capped_episode():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    obs = env.reset(11)
    for _ in range(env.max_steps + 1):  # bounded, so a cap that never fires fails the test instead of hanging it
        if env.done:
            break
        obs, *_ = env.step(AlwaysFinish().act(obs))
    return env


def no_repair_phase_episode():
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_repairs=0)
    env.reset(0)
    env.step({"type": "measure", "tier": 2})
    submit_truth(env)
    return env


def insufficient_evidence_episode():
    env = env_from(HAND_START)
    env.reset(0)
    env.step({"type": "measure", "tier": 1})
    env.step({"type": "submit", "insufficient_evidence": True, "next_action": "measure(tier=3)"})
    env.step({"type": "replace", "component": "cooler"})
    env.step({"type": "finish"})
    return env


def scripted_log(seed):
    """A fixed action sequence for the episode's start: a refusal, eleven tier-1 measurements (one past the window),
    the true diagnosis, a repair that changes the combination (a new window and entry offset), two measurements in
    the new window, an unnecessary replacement and finish."""
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_repairs=3)
    env.reset(seed)
    start = env.initial
    healthy_component = next(c for c in COMPONENTS if c not in faulty_components(start))
    actions = [{"type": "teleport"}] + [{"type": "measure", "tier": 1}] * 11 + [
        {"type": "submit", "diagnosis": dict(zip(COMPONENTS, start))},
        {"type": "replace", "component": faulty_components(start)[0]},
        {"type": "measure", "tier": 2}, {"type": "measure", "tier": 3},
        {"type": "replace", "component": healthy_component},
        {"type": "finish"},
    ]
    for action in actions:
        if env.done:
            break
        env.step(action)
    return env.log


def test_default_episodes_start_from_the_47_combinations_with_one_or_two_faulty_components():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    assert (env.max_faulty, env.max_repairs, env.budget, env.max_steps, env.block_cap) == (2, 2, 150, 200, 10)
    one_or_two = {k for k in itertools.product(*(LEVELS[c] for c in COMPONENTS)) if 1 <= len(faulty_components(k)) <= 2}
    assert len(one_or_two) == 47 and set(env.candidates) == one_or_two
    starts = set()
    for seed in range(400):
        env.reset(seed)
        starts.add(env.initial)
    assert starts == one_or_two


def test_the_observation_holds_the_agent_facing_fields_of_the_spec_and_nothing_else():
    env = env_from(HAND_START, max_repairs=3)
    obs = env.reset(0)
    assert set(obs) == {"phase", "components", "levels", "healthy", "tiers", "prices", "budget_left", "repairs_left", "min_faulty", "max_faulty", "steps_left", "history"}
    assert obs["levels"] == SPEC_LEVELS and obs["healthy"] == SPEC_HEALTHY and obs["max_faulty"] == 2 and obs["history"] == []
    assert {t: set(channels) for t, channels in obs["tiers"].items()} == SPEC_TIER_CHANNELS
    for action in ({"type": "measure", "tier": 1}, {"type": "measure", "tier": 2}, {"type": "measure", "tier": 3}, {"type": "teleport"},
                   {"type": "submit", "diagnosis": dict(zip(COMPONENTS, HAND_START))},
                   {"type": "replace", "component": "pump"}, {"type": "replace", "component": "cooler"}):
        obs, *_ = env.step(action)
    history = obs["history"]
    assert [e["event"] for e in history] == ["measure"] * 3 + ["refused", "submit", "replace", "replace"]
    assert all(not {"cycle", "combo_after", "unnecessary", "hidden"} & set(e) for e in history)
    for tier, (shown, logged) in enumerate(zip(history[:3], env.log[1:4]), start=1):
        assert shown["cost"] == obs["prices"]["measure"][tier] and shown["result"] == tier_features(CYCLES[logged["cycle"]], tier)
        assert {n.split("_")[1] for n in shown["result"]} == SPEC_TIER_CHANNELS[tier] and len(shown["result"]) == 4 * len(SPEC_TIER_CHANNELS[tier])
        assert {n.rsplit("_", 1)[1] for n in shown["result"]} == {"mean", "std", "min", "max"}
    assert [e["cost"] for e in history[5:]] == [obs["prices"]["parts"][c] + obs["prices"]["downtime"] for c in ("pump", "cooler")]


def test_a_seed_and_an_action_sequence_reproduce_the_log_exactly_including_the_draws_after_a_repair():
    for seed in range(30):
        random.seed(1)
        first = scripted_log(seed)
        random.seed(2)  # the environment must not draw from the global generator
        random.random()
        assert scripted_log(seed) == first
        # a seeded agent reproduces too, since its own generator is the only other source of randomness
        assert run_episode(TinyHydraulicEnv(CYCLES, BLOCKS), RandomPolicy(seed), seed) == run_episode(TinyHydraulicEnv(CYCLES, BLOCKS), RandomPolicy(seed), seed)


def test_measurements_walk_the_window_in_cycle_order_from_the_entry_offset_and_flag_every_repeat():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    for seed in range(40):
        env.reset(seed)
        for _ in range(25):
            obs, *_ = env.step({"type": "measure", "tier": 1})
        cycles = measured_cycles(env.log)
        first_pass = cycles[:10]
        assert is_window(first_pass, BLOCKS[env.initial])
        breaks = [(a, b) for a, b in zip(first_pass, first_pass[1:]) if b != a + 1]
        assert breaks in ([], [(max(first_pass), min(first_pass))])  # the next cycle in recording order, wrapping once inside the window
        assert cycles[10:] == (first_pass * 2)[:15]
        shown = [e for e in obs["history"] if e["event"] == "measure"]
        assert [e["repeated"] for e in shown] == [False] * 10 + [True] * 15
        assert [e["result"] for e in shown[10:]] == [e["result"] for e in (shown[:10] * 2)[:15]]


def test_a_replace_lands_in_the_block_of_the_new_combination_whether_or_not_the_episode_ends_on_it():
    ends = Counter()
    for start in TinyHydraulicEnv(CYCLES, BLOCKS).candidates:
        for max_repairs in (1, 2):
            env = env_from(start, max_repairs=max_repairs)
            env.reset(0)
            submit_truth(env)
            component = faulty_components(start)[0]
            after = repaired(start, component)
            obs, cost, done, info = env.step({"type": "replace", "component": component})
            replace = next(e for e in reversed(env.log) if e["event"] == "replace")
            assert replace["combo_after"] == after == tuple(env.hidden)
            assert len(env._block) == env.block_cap and is_window(env._block, BLOCKS[after])
            if done:
                ends[info["reason"]] += 1
                assert info["reason"] == ("recovered" if after == HEALTHY_COMBO else "repairs used up")
            else:
                ends["continues"] += 1
                env.step({"type": "measure", "tier": 1})
                assert CYCLES[env.log[-1]["cycle"]].combo == after and env.log[-1]["repeated"] is False
    assert ends == {"recovered": 2 * 10, "repairs used up": 37, "continues": 37}


def test_candidate_blocks_longer_than_a_window_are_served_through_seeded_windows_of_contiguous_cycles():
    cap = TinyHydraulicEnv(CYCLES, BLOCKS).block_cap
    long_blocks = [k for k in TinyHydraulicEnv(CYCLES, BLOCKS).candidates if len(BLOCKS[k]) > cap]
    assert long_blocks
    for long_block in long_blocks:
        env = env_from(long_block)
        starts = set()
        for seed in range(50):
            env.reset(seed)
            for _ in range(cap):
                env.step({"type": "measure", "tier": 1})
            window = measured_cycles(env.log)
            assert is_window(window, BLOCKS[long_block])
            starts.add(min(window))
        assert len(starts) > 1


def test_refusals_in_both_phases_are_logged_with_their_reason_cost_nothing_and_change_no_state():
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    env.reset(6)
    env.step({"type": "measure", "tier": 1})
    truth = dict(zip(COMPONENTS, env.initial))
    investigate = [
        {"type": "submit", "insufficient_evidence": True, "next_action": ""},
        {"type": "submit"},
        {"type": "submit", "diagnosis": {**truth, "pump": None}},
        {"type": "replace", "component": "pump"},
        {"type": "finish"},
        {},
    ]
    repair = [
        {"type": "submit", "diagnosis": truth},
        {"type": "submit", "insufficient_evidence": True, "next_action": "measure(tier=3)"},
        {"type": "measure", "tier": 0},
        {"type": "replace", "component": "turbine"},
        {"type": "replace"},
        {"type": "teleport"},
    ]
    for phase, actions in (("investigate", investigate), ("repair", repair)):
        if phase == "repair":
            submit_truth(env)
        for action in actions:
            before = state(env)
            obs, cost, done, info = env.step(action)
            assert info.get("refused") and cost == 0 and not done and state(env) == before and env.phase == phase, action
            logged = env.log[-1]
            assert logged["event"] == "refused" and logged["action"] == action and logged["reason"] == info["refused"]
            assert obs["history"][-1] == {"event": "refused", "action": action, "reason": logged["reason"]}


def test_a_replacement_the_budget_cannot_cover_is_refused_and_ends_the_episode():
    env = budget_ended_episode()
    refused, end = env.log[-2:]
    assert refused["event"] == "refused" and refused["reason"] == "budget" and refused["action"] == {"type": "replace", "component": "pump"}
    assert env.done and end["event"] == "end" and end["reason"] == "budget"
    assert tuple(env.hidden) == env.initial and env.cost == 0 and env.repairs == 0
    s = score(env.log)
    assert s["submitted"] and s["diagnosis_all_correct"] and not s["recovered"] and s["replacements"] == 0 and s["end_reason"] == "budget"
    assert s["steps"] == env.steps == 2


def test_the_step_cap_counts_refusals_and_ends_an_episode_that_never_submits_without_writing_a_submission():
    env = step_capped_episode()
    assert env.done and env.steps == env.max_steps == 200
    assert [e["event"] for e in env.log[1:]] == ["refused"] * 200 + ["end"]
    assert run_episode(TinyHydraulicEnv(CYCLES, BLOCKS), AlwaysFinish(), 11) == env.log  # the driver adds no submission
    s = score(env.log)
    assert s["end_reason"] == "step limit" and s["submitted"] is False and s["refusals"] == s["steps"] == 200 and s["steps_before_submit"] is None
    assert s["cost"] == 0 and s["cost_before_submit"] is None and s["diagnosis_all_correct"] is False
    # the cap ends the repair phase as well; the action that reaches it is still carried out and charged
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_steps=3)
    env.reset(0)
    submit_truth(env)
    env.step({"type": "measure", "tier": 1})
    obs, cost, done, info = env.step({"type": "measure", "tier": 1})
    assert done and info["reason"] == "step limit" and cost == env.prices["measure"][1]
    s = score(env.log)
    assert s["submitted"] and s["diagnosis_all_correct"] and s["end_reason"] == "step limit" and s["measures_tier1"] == 2


def test_a_step_that_ends_the_episode_for_another_reason_keeps_that_reason_on_the_capped_step():
    single, pump_only = (20, 100, 0, 130), (100, 100, 1, 130)
    unaffordable_pump = DEFAULT_PRICES["parts"]["pump"] + DEFAULT_PRICES["downtime"] - 1
    cases = {  # reason: (environment, the action taken on step 2 = max_steps, after the submission on step 1)
        "recovered": (env_from(single, max_steps=2), {"type": "replace", "component": "cooler"}),
        "repairs used up": (env_from(HAND_START, max_steps=2, max_repairs=1), {"type": "replace", "component": "cooler"}),
        "budget": (env_from(pump_only, max_steps=2, budget=unaffordable_pump), {"type": "replace", "component": "pump"}),
        "finish": (env_from(single, max_steps=2), {"type": "finish"}),
    }
    for reason, (env, last) in cases.items():
        env.reset(0)
        submit_truth(env)
        obs, cost, done, info = env.step(last)
        assert done and info["reason"] == reason and env.steps == env.max_steps, reason
        assert [e["reason"] for e in env.log if e["event"] == "end"] == [reason] and score(env.log)["end_reason"] == reason
    env = TinyHydraulicEnv(CYCLES, BLOCKS, max_repairs=0, max_steps=1)  # here the submission itself is the capped step
    env.reset(0)
    obs, cost, done, info = submit_truth(env)
    assert done and info["reason"] == "no repair phase" and env.steps == env.max_steps
    assert [e["reason"] for e in env.log if e["event"] == "end"] == ["no repair phase"]


def test_with_max_repairs_zero_the_episode_ends_at_the_submission():
    env = no_repair_phase_episode()
    assert env.done and env.log[-2]["event"] == "submit" and env.log[-1]["reason"] == "no repair phase"
    s = score(env.log)
    assert s["submitted"] and s["diagnosis_all_correct"] and s["end_reason"] == "no repair phase"
    assert s["replacements"] == 0 and s["cost"] == s["cost_before_submit"] == env.prices["measure"][2] and s["steps"] == env.steps == 2


def test_an_insufficient_evidence_submission_keeps_its_next_action_and_no_repair_agrees_with_it():
    env = insufficient_evidence_episode()
    submitted = next(e for e in env.log if e["event"] == "submit")
    assert submitted["insufficient_evidence"] is True and submitted["next_action"] == "measure(tier=3)" and submitted["diagnosis"] is None
    shown = next(e for e in env.observation()["history"] if e["event"] == "submit")
    assert shown["insufficient_evidence"] is True and shown["next_action"] == "measure(tier=3)"
    s = score(env.log)
    assert s["insufficient_evidence"] and s["diagnosis_all_correct"] is False and set(s["diagnosis_per_component"].values()) == {None}
    assert s["replacements"] == 1 and s["repairs_match_diagnosis"] is False and s["end_reason"] == "finish"


def test_hand_computed_scores_of_an_episode_with_a_refusal_a_wrong_valve_and_an_unnecessary_replacement():
    env = hand_scored_episode()
    p = env.prices
    expected = {
        "submitted": True,
        "insufficient_evidence": False,
        # graded against the initial combination, so the cooler counts as right although it is replaced afterwards
        "diagnosis_per_component": {"cooler": True, "valve": False, "pump": True, "accumulator": True},
        "diagnosis_all_correct": False,
        "recovered": False,  # the valve is still at 90
        "repairs_match_diagnosis": False,  # the accumulator was replaced although the diagnosis called it healthy
        "cost": sum(p["measure"].values()) + p["parts"]["accumulator"] + p["parts"]["cooler"] + 2 * p["downtime"],
        "cost_before_submit": sum(p["measure"].values()),
        "steps": 8,  # every step call: the refusal, three measurements, the submission, two replacements and the finish
        "steps_before_submit": 5,  # up to and including the submission
        "measures_tier1": 1, "measures_tier2": 1, "measures_tier3": 1,
        "replacements": 2,
        "unnecessary_replacements": 1,
        "refusals": 1,
        "end_reason": "finish",
        "initial": (3, 90, 0, 130),
        "final": (100, 90, 0, 130),
    }
    s = score(env.log)
    assert {k: s[k] for k in expected} == expected
    replaced = [e for e in env.log if e["event"] == "replace"]
    assert [e["unnecessary"] for e in replaced] == [True, False]
    assert [e["cost"] for e in replaced] == [p["parts"][c] + p["downtime"] for c in ("accumulator", "cooler")]


def test_scores_reproduce_from_logs_saved_to_json_files(tmp_path):
    env = TinyHydraulicEnv(CYCLES, BLOCKS)
    logs = []
    for seed in range(50):
        logs.append(run_episode(env, RandomPolicy(seed), seed))
        logs.append(run_episode(env, OraclePolicy(env), seed))
    logs += [make().log for make in (hand_scored_episode, budget_ended_episode, step_capped_episode, no_repair_phase_episode, insufficient_evidence_episode)]
    assert {score(log)["end_reason"] for log in logs} == SPEC_END_REASONS
    for i, log in enumerate(logs):
        path = tmp_path / f"episode_{i}.json"
        path.write_text(json.dumps(log))
        assert score(json.loads(path.read_text())) == score(log), i


def test_the_oracle_recovers_every_candidate_start_at_the_lowest_possible_cost():
    for start in TinyHydraulicEnv(CYCLES, BLOCKS).candidates:
        env = env_from(start)
        s = score(run_episode(env, OraclePolicy(env), 0))
        # recovery needs one replacement of every faulty component, and no action has a negative price
        lowest = sum(env.prices["parts"][c] + env.prices["downtime"] for c in faulty_components(start))
        assert s["recovered"] and s["diagnosis_all_correct"] and s["repairs_match_diagnosis"] and s["end_reason"] == "recovered"
        assert s["cost"] == lowest and s["unnecessary_replacements"] == s["refusals"] == 0
