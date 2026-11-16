"""The training data of rejection sampling fine tuning (rejection_sampling_examples.py): the filtered arm keeps the episodes
the verifier passed, the control draws as many episodes without reading a verdict, both arms drop the same calls, every call
becomes the prompt and the reply the server saw on its agent's route, the data belongs to one fold, the episodes that reached a
single fault of the held-out fold leave before the filter and the control and are counted, and no test start state, reserved seed,
label rollout seed or start of
the held-out fold enters training. The run folders are synthetic and built under tmp_path, except where the real start states and a
real episode log are checked."""
from CausalDiagnosis.utils.paths import CONFIGS
import json
from pathlib import Path

import pytest

from CausalDiagnosis.training import rejection_sampling_examples as rft_data
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, episode_name
from CausalDiagnosis.utils.json_files import write_json
from CausalDiagnosis.experiments.table_definitions.training_variants import (DATA_MANIFEST, DEFAULT_ASSET, FILTERED, FOLD_COMPONENTS, FOLDS, TRAINING_MODE, UNFILTERED, VISITS, asset_training,
                      combinations_played, sha256_file, visits_test_combination)

# a single fault combination that the model of fold 1 trains on, and a combination of fold 1, which that model holds out and a drop
# of the hazard can reach (a model is named by the fold it holds out)
START = next(c for c, (_fold, models) in sorted(FOLDS.items()) if 1 in models and sum(1 for x, h in zip(c, (100, 100, 0, 130)) if x != h) == 1)
OTHER = next(c for c, (fold, _models) in sorted(FOLDS.items()) if fold == 1)
STAYS = [{"event": "reset", "hidden": list(START)}, {"event": "read"}]
VISITS_OTHER = [{"event": "reset", "hidden": list(START)}, {"event": "work_order_closed", "combination_after": list(OTHER)}]

CHAT_CALLS = [
    {"prompt": "plan the repair", "reply": "#Task1: read", "input_tokens": 40, "output_tokens": 6, "finish_reason": "stop", "seconds": 1.0},
    {"prompt": "arguments of read", "reply": '{"sensor_group": "tier1"}', "input_tokens": 30, "output_tokens": 8, "finish_reason": "stop",
     "date": "2026-11-22"},
    {"prompt": "a call that raised", "error": "APIConnectionError: down", "seconds": 1.0},
    {"prompt": "a call without a reply", "reply": "", "input_tokens": 12, "output_tokens": 0, "finish_reason": "stop"},
    {"prompt": "a call cut at the cap", "reply": "#Task1: re", "input_tokens": 12, "output_tokens": 2048, "finish_reason": "length"},
]
# the record of the paper-era Agent-As-Tool port: raw text, the reply untrimmed, the stop string in it
AAT_CALLS = [
    {"prompt": "Question: q\nThought:", "reply": " read the pump\nAction: read\nObservation:", "matched_stop": "\nObservation:",
     "finish_reason": "stop", "stop": ["\nObservation:"], "temperature": 0.7, "agent": "supervisor", "kind": "thought",
     "input_tokens": 9, "output_tokens": 8, "max_tokens": 500, "model": "granite", "seconds": 0.4},
    {"prompt": "Final:", "reply": " done", "matched_stop": None, "finish_reason": "stop", "kind": "review"},
    {"prompt": "Final again:", "reply": " done", "matched_stop": 0, "finish_reason": "stop", "kind": "probe"},
    {"prompt": "Trimmed:", "reply": " read", "matched_stop": "\nObservation:", "finish_reason": "stop", "kind": "action"},
]


def _run(root, sample, passes, agent="plan_execute", calls=CHAT_CALLS, mode=TRAINING_MODE, variant="as_released", first_seed=None, fold=1,
         logs=None):
    """A finished sampling run of one agent and fold with one episode per entry of passes, each holding the calls, a verdict and
    a hidden log, the log of logs[i] for the i-th episode or one that never leaves its start."""
    seeds = [1000 * sample + i for i in range(len(passes))] if first_seed is None else [first_seed + i for i in range(len(passes))]
    run = root / f"deterioration_{agent}_sample{sample}"
    write_json(run / "run.json", {"mode": mode, "agent": agent, "agent_label": f"{agent}_granite", "model": "granite", "config_name": "base",
                                  "starts": "training", "training_sample": sample, "training_variant": variant, "seeds": seeds,
                                  "training_fold": fold,
                                  "master_seed": 42, "seed_pool": 8,
                                  "started": f"2026-11-2{sample}T09:00:00", "ended": f"2026-11-2{sample}T10:00:00"})
    with open(run / "scores.csv", "w", encoding="utf-8") as f:
        f.write(",".join(COLUMNS) + "\n")
        for seed in seeds:
            row = {k: "" for k in COLUMNS}
            row.update(seed=str(seed), agent=f"{agent}_granite")
            f.write(",".join(row[k] for k in COLUMNS) + "\n")
    for i, (seed, passed) in enumerate(zip(seeds, passes)):
        episode = run / episode_name(seed)
        write_json(episode / "llm_calls.json", calls)
        write_json(episode / "hidden" / "verdict.json", {"verified": True, "pass": passed, "checks": {"provenance": {"ok": True}}})
        write_json(episode / "hidden" / "log.json", (logs or {}).get(i, STAYS))
        (episode / "episode.json").write_text("{}", encoding="utf-8")
    return run


def _starts(training_states=1, state=lambda episode: episode["seed"]):
    """A stand-in for check_training_starts: the count of training start states of the fold's model and each episode's state."""
    return lambda episodes: {"training_states": training_states, "states": {(e["run"], e["episode"]): state(e) for e in episodes}}


def _build(tmp_path, runs, samples=2, out=None, starts=None):
    out = out or tmp_path / "data"
    manifest = rft_data.build(runs, out, samples=samples, check_starts=starts or _starts())
    examples = {arm: [json.loads(line) for line in (out / rft_data.EXAMPLES[arm]).read_text(encoding="utf-8").splitlines()]
                for arm in (FILTERED, UNFILTERED)}
    return manifest, examples, out


def test_the_filtered_arm_keeps_the_passed_episodes_with_every_usable_call(tmp_path):
    # the CausalDiagnosis agent, the one agent on the chat route since both ports call raw completions
    runs = [_run(tmp_path, 1, [True, False, False], agent="causal"), _run(tmp_path, 2, [False, True, False], agent="causal")]
    manifest, examples, out = _build(tmp_path, runs)
    filtered = examples[FILTERED]
    assert {e["episode"] for e in filtered} == {"seed001000", "seed002001"} and all(e["arm"] == FILTERED for e in filtered)
    # two usable calls per episode, the call that raised, the empty one and the cut one dropped
    assert [(e["episode"], e["call"]) for e in filtered] == [("seed001000", 0), ("seed001000", 1), ("seed002001", 0), ("seed002001", 1)]
    assert filtered[0]["route"] == "chat" and filtered[0]["prompt"] == "plan the repair" and filtered[0]["completion"] == "#Task1: read"
    assert filtered[0]["agent"] == "causal" and filtered[0]["date"] == "2026-11-21" and filtered[0]["sample"] == 1
    # a call keeps the day it was made on, past midnight of the run's start, and the prompt tokens the server counted
    assert (filtered[1]["date"], filtered[1]["input_tokens"], filtered[0]["input_tokens"]) == ("2026-11-22", 30, 40)
    assert manifest["agent"] == "causal" and manifest["label"] == "causal_granite" and manifest["model"] == "granite"
    assert manifest["fold"] == 1 and filtered[0]["fold"] == 1 and manifest["arms"][FILTERED][VISITS] == manifest["arms"][UNFILTERED][VISITS] == 0
    assert manifest["episodes_sampled"] == 6 and manifest["arms"][FILTERED]["episodes"] == 2
    assert manifest["arms"][FILTERED]["examples"] == 4 and manifest["arms"][FILTERED]["dropped"] == {"error": 2, "empty": 2, "cut": 2}
    assert manifest["runs"][0]["training_sample"] == 1 and manifest["runs"][1]["training_sample"] == 2
    assert manifest["arms"][FILTERED]["sha256"] == sha256_file(out / rft_data.EXAMPLES[FILTERED])
    assert json.loads((out / DATA_MANIFEST).read_text(encoding="utf-8")) == manifest


def test_the_control_draws_as_many_episodes_without_reading_a_verdict(tmp_path):
    """Two sets of runs on the same seeds with as many passes on other episodes: the filtered arms differ, and the control,
    which sees the episodes and its seed alone, draws the same episodes in both."""
    runs_a = [_run(tmp_path / "a", 1, [True, False, False, False]), _run(tmp_path / "a", 2, [False, True, False, False])]
    runs_b = [_run(tmp_path / "b", 1, [False, False, True, False]), _run(tmp_path / "b", 2, [False, False, False, True])]
    a, b = _build(tmp_path / "a", runs_a), _build(tmp_path / "b", runs_b)
    filtered = [sorted({e["episode"] for e in built[1][FILTERED]}) for built in (a, b)]
    control = [sorted({e["episode"] for e in built[1][UNFILTERED]}) for built in (a, b)]
    assert filtered[0] != filtered[1] and control[0] == control[1] and len(control[0]) == 2
    assert a[0]["arms"][UNFILTERED]["episodes"] == a[0]["arms"][FILTERED]["episodes"] == 2
    assert a[0]["control_seed"] == asset_training(DEFAULT_ASSET).config["control_seed"] and all(e["arm"] == UNFILTERED for e in a[1][UNFILTERED])
    pool = [f"seed{n:06d}" for n in range(8)]
    assert rft_data.control_draw(pool, 3, 0) == rft_data.control_draw(list(reversed(pool)), 3, 0) != rft_data.control_draw(pool, 3, 1)


def test_the_control_pool_is_every_episode_with_a_usable_call(tmp_path, monkeypatch):
    seen = {}

    def draw(pool, count, seed):
        seen.update(pool=list(pool), count=count, seed=seed)
        return sorted(pool)[:count]

    monkeypatch.setattr(rft_data, "control_draw", draw)
    runs = [_run(tmp_path, 1, [True, False]), _run(tmp_path, 2, [False], calls=[{"prompt": "x", "error": "boom"}])]
    manifest, _examples, _out = _build(tmp_path, runs)
    assert seen["count"] == 1 and len(seen["pool"]) == 2 and not any("seed002" in name for name in seen["pool"])
    assert manifest["episodes_with_a_usable_call"] == 2 and manifest["episodes_sampled"] == 3


@pytest.mark.parametrize("agent", ["agent_as_tool", "plan_execute"])  # the two paper-era ports record raw completions alike
def test_a_call_of_a_port_keeps_its_raw_prompt_and_the_stop_string_that_ended_it(tmp_path, agent):
    runs = [_run(tmp_path, k, [True], agent=agent, calls=AAT_CALLS) for k in (1, 2)]
    _manifest, examples, _out = _build(tmp_path, runs)
    first, second, third, fourth = examples[FILTERED][:4]
    assert first["route"] == "completion" and first["prompt"] == "Question: q\nThought:"
    assert first["completion"] == " read the pump\nAction: read\nObservation:" and first["ends_at_stop"] is True
    # a reply that ended at the end token, reported as no stop or as the token's id, gets the end token from the trainer
    assert (second["completion"], second["ends_at_stop"]) == (third["completion"], third["ends_at_stop"]) == (" done", False)
    # a trimmed reply gets its stop string back, so both ways of serving give the tokens the model wrote
    assert (fourth["completion"], fourth["ends_at_stop"]) == (" read\nObservation:", True)


def test_a_call_without_its_prompt_is_refused(tmp_path):
    """A chat record of the released Agent-As-Tool, which the reproduction plays alone, holds messages and no prompt, and
    would be encoded as a request the server never received."""
    call = {"messages": [{"role": "user", "content": "q"}], "tools": ["sites"], "reply": "x", "finish_reason": "stop"}
    runs = [_run(tmp_path, k, [True], agent="agent_as_tool", calls=[call]) for k in (1, 2)]
    with pytest.raises(SystemExit, match="cannot be rebuilt"):
        _build(tmp_path, runs)


@pytest.mark.parametrize("problem", ["mode", "agents", "variant", "missing", "repeated", "labels", "folds"])
def test_runs_other_than_the_complete_samples_of_one_agent_are_refused(tmp_path, problem):
    runs = [_run(tmp_path, 1, [True]), _run(tmp_path, 2, [True])]
    if problem == "mode":
        runs[1] = _run(tmp_path / "gym", 2, [True], mode="gym")
    elif problem == "agents":
        runs[1] = _run(tmp_path / "cd", 2, [True], agent="causal")
    elif problem == "variant":
        runs[1] = _run(tmp_path / "ft", 2, [True], variant="ft")
    elif problem == "missing":
        runs = runs[:1]
    elif problem == "repeated":
        runs[1] = _run(tmp_path / "again", 1, [True])
    elif problem == "folds":
        runs[1] = _run(tmp_path / "f2", 2, [True], fold=2)
    elif problem == "labels":
        scores = runs[1] / "scores.csv"
        scores.write_text(scores.read_text(encoding="utf-8").replace("plan_execute_granite", "plan_execute_other"), encoding="utf-8")
    with pytest.raises(SystemExit):
        _build(tmp_path, runs)


def test_an_episode_that_met_a_held_out_fault_leaves_the_kept_set_and_the_pool(tmp_path):
    """The split is closed under one repair, so only a drop of the layer 5 hazard leads a training episode into a combination its
    model is judged on. Such an episode leaves before the filter and before the draw of the control, whatever its outcome: of two
    passed episodes of which one met a held-out fault one is kept, the control draws from the other episodes alone, the manifest
    counts one left out, and neither example file holds a line of it."""
    runs = [_run(tmp_path, 1, [True, True], logs={0: VISITS_OTHER}), _run(tmp_path, 2, [False, False])]
    manifest, examples, _out = _build(tmp_path, runs)
    assert {e["episode"] for e in examples[FILTERED]} == {"seed001001"}
    assert manifest["arms"][FILTERED][VISITS] == manifest["arms"][UNFILTERED][VISITS] == 1
    assert not [e for arm in (FILTERED, UNFILTERED) for e in examples[arm] if e["episode"] == "seed001000"]
    assert manifest["episodes_sampled"] == 4 and manifest["episodes_with_a_usable_call"] == 3


def test_the_gate_counts_what_is_kept_after_the_filter(tmp_path):
    """A round that keeps the minimum before an episode is left out falls below it after, so the gate asks for one more round."""
    left = [_run(tmp_path / "left", 1, [True, True], logs={1: VISITS_OTHER}), _run(tmp_path / "left", 2, [False, False])]
    manifest, _examples, _out = _build(tmp_path / "left", left, starts=_starts(training_states=2))
    assert (manifest["gate"]["kept"], manifest["gate"]["minimum"], manifest["gate"]["second_round_due"]) == (1, 2, True)
    kept = [_run(tmp_path / "kept", 1, [True, True]), _run(tmp_path / "kept", 2, [False, False])]
    manifest, _examples, _out = _build(tmp_path / "kept", kept, starts=_starts(training_states=2))
    assert manifest["gate"]["kept"] == 2 and manifest["gate"]["trained"]


# one value per field of rft_data.IDENTITY that the second sampling run of a pair takes where the first holds none of it
OTHER_IDENTITY = {"asset": {"config": {"asset_id": "tiny"}}, "split": {"training_split": "development"}, "configuration": {"config_name": "priced"},
                  "code identity": {"code": {"identity": "f" * 64}}, "model": {"model": "other"},
                  "tool servers": {"servers": ["iot"]}, "MCP switch": {"diagnostics_server": True}, "store": {"backend": {"kind": "couchdb"}},
                  "sampling temperature": {"sampling_temperature": 0.9}, "stamp": {"stamp": "development"}}


@pytest.mark.parametrize("field", sorted(OTHER_IDENTITY))
def test_runs_that_differ_in_a_field_of_their_identity_are_refused(tmp_path, field):
    """The data of one adapter comes from one condition, so two sampling runs that differ in one field of their identity stop the
    build, and the refusal names the field and both runs."""
    assert sorted(OTHER_IDENTITY) == sorted(rft_data.IDENTITY)
    runs = [_run(tmp_path, 1, [True]), _run(tmp_path, 2, [True])]
    meta = json.loads((runs[1] / "run.json").read_text(encoding="utf-8"))
    write_json(runs[1] / "run.json", {**meta, **OTHER_IDENTITY[field]})
    with pytest.raises(SystemExit, match=f"differ in their {field}: {runs[0].name} has .* and {runs[1].name} has"):
        _build(tmp_path, runs)


def test_data_from_stamped_runs_carries_the_stamp(tmp_path):
    """Sampling runs that the start check stamped give data that says development, which the adapter and the blocks carry on, and
    the manifest lists every sampling run with the SHA-256 of its run.json, the runs an adapter is built from."""
    runs = [_run(tmp_path, 1, [True]), _run(tmp_path, 2, [True])]
    for run_dir in runs:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        write_json(run_dir / "run.json", {**meta, "stamp": "development", "code": {"identity": "a" * 64}})
    manifest, _examples, _out = _build(tmp_path, runs)
    assert (manifest["stamp"], manifest["code_identity"]) == ("development", "a" * 64)
    assert [(r["run"], r["run_json_sha256"]) for r in manifest["runs"]] == [(r.name, sha256_file(r / "run.json")) for r in runs]


def test_the_data_of_a_trial_on_the_development_split_carries_the_split(tmp_path):
    """Sampling runs of --training-split development name the split in run.json, and their data names it as well, which the
    adapter carries on and a run of the registered split refuses (training.check_split). Runs that name no split are registered."""
    runs = [_run(tmp_path / "trial", 1, [True]), _run(tmp_path / "trial", 2, [True])]
    for run_dir in runs:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        write_json(run_dir / "run.json", {**meta, "training_split": "development"})
    assert _build(tmp_path / "trial", runs)[0]["training_split"] == "development"
    registered = [_run(tmp_path / "registered", 1, [True]), _run(tmp_path / "registered", 2, [True])]
    assert _build(tmp_path / "registered", registered)[0]["training_split"] == "registered"


def test_the_combinations_an_episode_played_come_from_its_hidden_log():
    log = [{"event": "reset", "hidden": [20, 100, 0, 130]}, {"event": "read"}, {"event": "hazard_drop", "component": "valve", "to_level": 90},
           {"event": "work_order_closed", "combination_after": [100, 90, 0, 130]}, {"event": "emergency_expired", "component": "valve",
           "to_level": 90, "combination_after": [100, 90, 0, 130]}, {"event": "scheduled_drop", "component": "pump", "to_level": 1}]
    assert combinations_played(log, FOLD_COMPONENTS) == [(20, 100, 0, 130), (20, 90, 0, 130), (100, 90, 0, 130), (100, 90, 1, 130)]
    assert not visits_test_combination(STAYS, 1, DEFAULT_ASSET) and visits_test_combination(VISITS_OTHER, 1, DEFAULT_ASSET)


def test_the_played_combinations_of_a_real_episode_are_those_of_its_clock():
    """A real episode under a high hazard rate with the oracle's repairs: the combinations read from the log are the start and
    every combination the clock's history holds, in order."""
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy, run_episode

    _, cycles, blocks = load_world()
    env = HydraulicRigEnv(cycles, blocks, EpisodeConfig(hazard={"mode": "stochastic", "base_rate": 0.05, "depth_factor": 1.0, "collateral": 0.0}))
    seen_drop = False
    for seed in range(20):
        log = run_episode(env, OraclePolicy(env), seed)
        expected = []
        for combination in [env.initial, *(c for _t, c in env.clock.history)]:
            if tuple(combination) not in expected:
                expected.append(tuple(combination))
        assert combinations_played(log, FOLD_COMPONENTS) == expected
        seen_drop = seen_drop or any(e.get("event") == "hazard_drop" for e in log)
    assert seen_drop


def test_a_seed_outside_the_training_states_stops_the_build(tmp_path, monkeypatch):
    """The real start states: a seed of a test replicate, a reserved seed and a training seed of the other model are refused,
    and the training seeds of the fold pass."""
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    _, cycles, blocks = load_world()
    env = HydraulicRigEnv(cycles, blocks, load_config("base")[1])
    good = start_states.training_seeds(env, FOLDS, 1, 1)[:2]
    test_seed = start_states.replicate_seeds(env, 3)[0]
    development = start_states.combination_seeds(env, 1)[0]
    other_fold = start_states.training_seeds(env, FOLDS, 1, 2)[0]
    asked, states_of = [], start_states.training_start_states
    monkeypatch.setattr(start_states, "training_start_states", lambda env, split, fold, master_seed, pool: asked.append((fold, master_seed, pool))
                        or states_of(env, split, fold, master_seed, pool))
    rft_data.check_training_starts([{"seed": seed, "config": "base", "test_set": (42, 8), "fold": 1} for seed in good])
    for seed in (test_seed, development, other_fold):
        with pytest.raises(SystemExit, match="training start state of fold 1"):
            rft_data.check_training_starts([{"seed": seed, "config": "base", "test_set": (42, 8), "fold": 1}])
    assert asked == [(1, 42, 8)] * 4  # the fold and the test set the run recorded decide the states it is checked against


def test_the_gate_counts_kept_episodes_against_the_training_states_and_asks_for_one_more_round(tmp_path):
    """The minimum is one kept episode per training start state of the model, the count the start states give,
    and a kept episode is one the verifier passed that holds a usable call; below it after the first round a second round is due,
    after the second no adapter trains, and the distinct training start states among the kept episodes are counted."""
    calls_none = [{"prompt": "x", "error": "boom"}]
    runs = [_run(tmp_path / "r1", 1, [True, True, False]), _run(tmp_path / "r1", 2, [True, False, False], calls=calls_none)]
    by_state = _starts(training_states=3, state=lambda e: e["seed"] % 1000)
    first, _examples, _out = _build(tmp_path / "a", runs, starts=by_state)
    assert first["gate"] == {"minimum": 3, "kept_per_training_state": 1, "training_states": 3, "rounds_sampled": 1,
                             "kept": 2, "distinct_training_states": 2, "second_round_due": True, "trained": False, "episodes_sampled": 6}
    assert first["arms"][FILTERED]["episodes"] == 2  # the passed episode without a usable call is not kept
    second = runs + [_run(tmp_path / "r2", k, [False, False, False]) for k in (3, 4)]
    both, _examples, _out = _build(tmp_path / "b", second, starts=by_state)
    assert (both["gate"]["rounds_sampled"], both["gate"]["kept"], both["gate"]["second_round_due"], both["gate"]["trained"]) == (2, 2, False, False)
    assert both["episodes_with_a_usable_call"] == 9 and both["gate"]["episodes_sampled"] == 12  # the control's pool spans both rounds
    met, _examples, _out = _build(tmp_path / "c", runs, starts=_starts(training_states=2))
    assert met["gate"]["trained"] and not met["gate"]["second_round_due"]


def test_a_second_round_after_a_first_round_that_met_the_minimum_is_refused(tmp_path):
    runs = [_run(tmp_path, 1, [True]), _run(tmp_path, 2, [True]), _run(tmp_path, 3, [False]), _run(tmp_path, 4, [False])]
    with pytest.raises(SystemExit, match="no second round was due"):
        _build(tmp_path, runs, starts=_starts(training_states=2))
    with pytest.raises(SystemExit, match="whole rounds"):
        _build(tmp_path / "three", runs[:3], starts=_starts(training_states=2))


def test_the_count_of_training_states_comes_from_the_split_of_the_fold(tmp_path):
    """The real start states: the gate's count is the number of training start states of the fold's model, never a number of the
    rig written into the rule."""
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    _, cycles, blocks = load_world()
    env = HydraulicRigEnv(cycles, blocks, load_config("base")[1])
    for fold in (1, 2):
        seeds = start_states.training_seeds(env, FOLDS, 1, fold)[:2]
        info = rft_data.check_training_starts([{"seed": s, "config": "base", "test_set": (42, 8), "fold": fold, "run": "r", "episode": str(s)}
                                               for s in seeds])
        assert info["training_states"] == len(start_states.training_start_states(env, FOLDS, fold))
        assert set(info["states"]) == {("r", str(s)) for s in seeds}


def test_the_search_finds_the_sampling_runs_of_one_agent_and_fold_by_their_run_json(tmp_path):
    """The run order names one command, rejection_sampling_examples.py --search <grid folders> --agent --training-fold, and the
    builder picks the finished sampling runs of that agent and fold from their run.json, whatever the folders are called."""
    grid = tmp_path / "grid"
    mine = [_run(grid, k, [True]) for k in (1, 2)]
    _run(grid / "x", 1, [True], agent="causal")  # another agent, in a folder below the search depth as well
    other = _run(tmp_path / "grid2", 1, [True], fold=2)
    assert rft_data.found_runs([grid, tmp_path / "grid2"], "plan_execute", 1) == mine
    assert rft_data.found_runs([tmp_path / "grid2"], "plan_execute", 2) == [other]
    with pytest.raises(SystemExit, match="no finished sampling run"):
        rft_data.found_runs([grid], "causal", 2)
    with pytest.raises(SystemExit, match="needs --agent"):
        rft_data.main(["--search", str(grid)])


def test_a_hazard_drop_to_a_held_out_single_fault_counts_even_in_a_combination_no_test_starts_at():
    """A training episode of the model of fold 1 that reaches a single fault of fold 1, the fold it holds
    out, counts, also when the drop makes a three-fault combination that is no start combination at all."""
    from CausalDiagnosis.experiments.table_definitions.training_variants import FAULT_FOLDS, FOLD_COMPONENTS

    def faulty(c):
        return [i for i, level in enumerate(c) if (FOLD_COMPONENTS[i], level) in FAULT_FOLDS]

    pair = next(c for c, (_fold, models) in sorted(FOLDS.items()) if 1 in models and len(faulty(c)) == 2)
    free = next(i for i in range(len(FOLD_COMPONENTS)) if i not in faulty(pair))
    held = next(level for (component, level), fold in sorted(FAULT_FOLDS.items()) if component == FOLD_COMPONENTS[free] and fold == 1)
    log = [{"event": "reset", "hidden": list(pair)}, {"event": "hazard_drop", "component": FOLD_COMPONENTS[free], "to_level": held}]
    reached = combinations_played(log, FOLD_COMPONENTS)[-1]
    assert len(faulty(reached)) == 3 and reached not in FOLDS
    assert visits_test_combination(log, 1, DEFAULT_ASSET) and not visits_test_combination(log[:1], 1, DEFAULT_ASSET)


def test_each_episode_is_checked_on_the_world_and_the_preset_of_its_asset(monkeypatch):
    """The check of the training start states loads the world and composes the preset of each episode's asset, once per asset, so
    that a port's sampling runs are checked on their own records. The loaders are recorded and answer with the rig's, so the same
    seeds pass for both assets."""
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    from CausalDiagnosis.experiments.episode_runners import run_configuration
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    worlds, presets = [], []
    load_world, load_config = run_configuration.load_world, run_configuration.load_config
    monkeypatch.setattr(run_configuration, "load_world", lambda facts=None, asset_id=None: worlds.append(asset_id) or load_world())
    monkeypatch.setattr(run_configuration, "load_config", lambda name, asset_id=None: presets.append((name, asset_id)) or load_config(name))
    _, cycles, blocks = load_world()
    seeds = start_states.training_seeds(HydraulicRigEnv(cycles, blocks, load_config("base")[1]), FOLDS, 1, 1)[:2]
    other = "made_up_asset"
    episodes = [{"seed": seeds[0], "config": "base", "test_set": (42, 8), "fold": 1, "asset_id": DEFAULT_ASSET},
                {"seed": seeds[1], "config": "base", "test_set": (42, 8), "fold": 1, "asset_id": other}]
    rft_data.check_training_starts(episodes)
    assert sorted(worlds) == sorted([DEFAULT_ASSET, other]) and sorted(presets) == sorted([("base", DEFAULT_ASSET), ("base", other)])


def test_the_sampling_runs_of_one_adapter_pool_across_serving_jobs_and_never_across_models():
    """Two serving jobs of the same model serve it at two addresses, so sampling runs that differ in their endpoint alone are of
    one condition, while runs of two models are refused by name."""
    meta = {"config_name": "deterioration", "model": "granite", "endpoint": "http://10.0.0.1:31000/v1", "servers": ["hydraulic_rig"]}
    runs = [(Path("run_a"), meta), (Path("run_b"), {**meta, "endpoint": "http://10.0.0.9:39000/v1"})]
    rft_data.check_identity(runs)
    with pytest.raises(SystemExit, match="differ in their model"):
        rft_data.check_identity([*runs, (Path("run_c"), {**meta, "model": "another"})])


def test_the_filter_keeps_a_good_episode_and_drops_a_pass_with_a_made_up_identifier(tmp_path):
    """The filter reads verifier.good, the predicate of the reward: a pass whose cited identifier the released record does not hold
    is no good episode, and rft.json registers the filter by that name."""
    def episode(name, verdict):
        folder = tmp_path / name
        write_json(folder / "hidden" / "verdict.json", verdict)
        return {"ep_dir": folder}

    assert rft_data.good(episode("good", {"pass": True, "checks": {"provenance": {"ok": True}}})) is True
    assert rft_data.good(episode("made_up", {"pass": True, "checks": {"provenance": {"ok": False}}})) is False
    assert rft_data.FILTER == "good" == json.loads((CONFIGS / "training" / "hyd_1" / "rft.json").read_text())["filter"]
