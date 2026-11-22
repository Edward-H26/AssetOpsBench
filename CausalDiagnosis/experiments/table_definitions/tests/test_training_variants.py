"""The variants of the training table in the driver (run_episodes/ and training_variants/): every run records its
variant and its fold, the block of a GEPA variant reaches the three agents at one position, after the briefing and an empty line,
a fine tuned variant sends the model with its adapter's name while the run keeps the base model and the agent's label, a variant
whose adapter or block is missing or belongs to another agent, arm or fold stops before any episode, a trained run plays the
test episodes of the fold its model holds out alone, and the training starts play the training seeds of that model at the sampling temperature in
the mode training. Stand-in endpoints and fake runners take the place of every model, and the run folders and the training
folders go under tmp_path."""
from CausalDiagnosis.utils.paths import CONFIGS, preset_files
import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, RIG_SERVER
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_name, read_rows
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import agent_runs  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import scripted_wiring_check  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import training_variants as training  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402

BLOCK = "Read the cheapest group first and name the component the readings point to."
MODEL = "vendor/stand-in-1.0"


RIG = training.DEFAULT_ASSET  # the asset of every run of this file
CONVENTION = {training.FOLD_CONVENTION_KEY: training.FOLD_CONVENTION, "asset_id": RIG}  # the fields every record of a training folder carries
GATE = training.gate_of(kept=37, distinct_states=30, training_states=30, rounds=1, asset_id=RIG)  # met, an adapter trains
BELOW = training.gate_of(kept=12, distinct_states=9, training_states=30, rounds=1 + training.numbers(RIG).extension_rounds, asset_id=RIG)  # below after the last round
DATA = {**CONVENTION, "episodes_sampled": 1190, "gate": GATE,
        "arms": {"filtered": {"episodes": 37, training.VISITS: 2}, "unfiltered": {"episodes": 37, training.VISITS: 3}}}
COUNTS = {"filtered": ({"examples": 412, "kept": 409, "dropped_over_max_length": 3}, 52),
          "unfiltered": ({"examples": 455, "kept": 455, "dropped_over_max_length": 0}, 57)}


def _data(root, agent, fold=1, data=DATA):
    """The data manifest rejection_sampling_examples.py writes for an agent and a fold; returns its SHA-256."""
    path = root / "data" / training.data_name(agent, fold, RIG) / training.DATA_MANIFEST
    write_json(path, data)
    return training.sha256_file(path)


ENDED = "2026-11-20T10:00:00-05:00"  # the end of every training job of the fixtures, with its offset from UTC as train_lora.py writes it


def _adapter(root, agent, arm, fold=1, weights=b"stand-in weights", data_sha=None, recorded_fold=None):
    """The folder train_lora.py writes for one arm of an agent and a fold, with its manifest naming the data manifest it trained
    on; returns the SHA-256 of the weights."""
    name = training.adapter_name_of_arm(agent, arm, fold, RIG)
    folder = root / "adapters" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_model.safetensors").write_bytes(weights)
    counts, steps = COUNTS[arm]
    write_json(folder / training.ADAPTER_MANIFEST, {"agent": agent, "arm": arm, "fold": recorded_fold or fold, "adapter": name, "counts": counts,
                                                    "steps": steps, "data_manifest_sha256": data_sha or _data(root, agent, fold), "ended": ENDED,
                                                    **CONVENTION})
    return hashlib.sha256(weights).hexdigest()


def _block(root, agent, base, fold=1, text=BLOCK, adapter_weights=None, recorded=None, recorded_fold=None, card="verifier",
           recorded_card=None):
    """The block and record the export of gepa_prompt_optimization/ writes for the loop on one agent, base variant, fold and card."""
    name = training.block_name(agent, base, fold, RIG, card)
    (root / "blocks").mkdir(parents=True, exist_ok=True)
    (root / "blocks" / f"{name}.txt").write_text(text, encoding="utf-8")
    write_json(root / "blocks" / f"{name}.json", {"agent": agent, "base_variant": base, "fold": recorded_fold or fold,
                                                 "card": recorded_card or card, "run": "loop", **CONVENTION,
                                                 "candidate": "c004", "sha256": recorded or hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                                 "adapter": {"weights_sha256": adapter_weights} if adapter_weights else None,
                                                 "candidates_played": 9, "stream_episodes": 240, training.VISITS: 1})


def _run_dir(out):
    runs = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _first_test_episode(monkeypatch):
    """A trained variant plays the test set (--starts sampled); the endpoint tests play its first episode of the held-out fold alone."""
    every = run_episodes.start_states.fold_replicate_seeds
    monkeypatch.setattr(run_episodes.start_states, "fold_replicate_seeds", lambda *args, **kwargs: every(*args, **kwargs)[:1])


class _Played(Exception):
    """Stops play() once the agent has received its task, before any model or verifier."""


@pytest.mark.parametrize("block", [BLOCK, ""])
def test_the_three_agents_receive_one_task_message_with_the_block_after_the_briefing(tmp_path, monkeypatch, block):
    seen = {}

    def capture(position):
        def agent(*args, **kwargs):
            seen.setdefault("tasks", []).append(args[position])
            raise _Played
        return agent

    monkeypatch.setattr(agent_runs, "run_plan_execute_paper", capture(1))
    monkeypatch.setattr(agent_runs, "run_agent_as_tool", capture(1))
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", capture(2))
    monkeypatch.setattr(run_episodes.play_one_episode, "check_servers", lambda launchers, row=None: None)
    facts_path, cycles, blocks = load_world()
    config_name, config = load_config("static")
    questions = []
    for agent in training.TRAINED_AGENTS:
        backend = None if agent in run_episodes.PAPER_ERA_AGENTS else run_episodes.RecordingBackend(scripted_wiring_check.ScriptedBackend((RIG_SERVER, CAUSAL_SERVER)))
        runner = run_episodes.EpisodeRunner(config, config_name, agent, (RIG_SERVER, CAUSAL_SERVER), cycles, blocks, facts_path, f"run-{agent}",
                                            {"judge": None, "grade": False, "ranking_model": run_episodes.DEFAULT_RANKING_MODEL},
                                            backend=backend, model=MODEL, task_block=block)
        ep = tmp_path / agent / "seed000003"
        with pytest.raises(_Played):
            runner.play(ep, ep.name, 3, None, config, {})
        questions.append((ep / "question.txt").read_text(encoding="utf-8"))
        assert (ep / "task_block.txt").exists() == bool(block)
    # one briefing for the three agents on one seed, and each received it with the same block at the same place
    assert len(set(questions)) == 1 and BLOCK not in questions[0]
    expected = f"{questions[0]}\n\n{block}" if block else questions[0]
    assert seen["tasks"] == [expected] * 3 == [training.with_task_block(questions[0], block)] * 3


class FakePlanExecute:
    """The constructor and run() of the paper-era Plan-Execute runner, recording what the driver gave it."""

    seen = {}

    def __init__(self, server_paths, base_url, api_key, model, temperature=None, log_path=None, example_start=None, on_idle=None):
        FakePlanExecute.seen = {"model": model, "temperature": temperature, "servers": sorted(server_paths)}
        self.calls, self.trajectory = [], []

    async def run(self, question):
        FakePlanExecute.seen["question"] = question
        return types.SimpleNamespace(question=question, answer="No repair was made.", end="finish", error=None, plan_steps=1,
                                     plan="## Step 1", plans=[], history=[], step_agents=[])

    def record(self, result=None):
        return None  # this stand-in builds no workflow, so the driver writes no record of one


def test_plan_execute_gets_the_block_and_run_json_records_the_variant(tmp_path, monkeypatch):
    root = tmp_path / "training"
    _block(root, "plan_execute", training.AS_RELEASED)
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setattr(agent_runs, "plan_execute_runner_class", lambda: FakePlanExecute)
    _first_test_episode(monkeypatch)
    run_episodes.main(["--config", "base", "--agent", "plan_execute", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--seeds", "1",
                       "--starts", "sampled", "--training-variant", "gepa", "--training-fold", "1", "--out", str(tmp_path / "runs")])
    run_dir = _run_dir(tmp_path / "runs")
    meta = read_json(run_dir / "run.json")
    ep = run_dir / episode_name(meta["seeds"][0])
    question = (ep / "question.txt").read_text(encoding="utf-8")
    assert BLOCK not in question and (ep / "task_block.txt").read_text(encoding="utf-8") == BLOCK
    # the port receives the task with the block after the briefing, which its planner prompt carries as the question
    assert FakePlanExecute.seen["question"] == f"{question}\n\n{BLOCK}"
    assert FakePlanExecute.seen["model"] == MODEL and FakePlanExecute.seen["temperature"] is None
    assert meta["training_variant"] == "gepa" and meta["adapter"] is None and meta["mode"] == "gym" and meta["sampling_temperature"] is None
    assert meta[training.TRAINING_FOLD_KEY] == 1
    # what the table of what each agent trained on reads of the loop travels with the run, in paths relative to the training folder
    assert meta["task_block"] == {"name": "gepa_plan_execute_f1", "path": "blocks/gepa_plan_execute_f1.txt", "fold": 1, "card": "verifier",
                                  "sha256": hashlib.sha256(BLOCK.encode("utf-8")).hexdigest(), "characters": len(BLOCK), "run": "loop",
                                  "candidate": "c004", "candidates_played": 9, "stream_episodes": 240, training.VISITS: 1,
                                  "stamp": None, "built_from": None, training.TRAINING_SPLIT_KEY: training.REGISTERED}
    assert {row["agent"] for row in read_rows(run_dir)} == {f"plan_execute_{MODEL}"}  # the variant leaves the label alone


def test_every_call_on_the_chat_route_keeps_the_day_it_was_made_on(tmp_path, monkeypatch):
    """The causal agent, the one agent on the chat route: every model call keeps the day it was made on, which the chat
    template prints and the training data rebuilds."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ScriptedPlanner, _planner_endpoint, act_now

    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server, _requests = _planner_endpoint(ScriptedPlanner(act_now))
    base = f"http://127.0.0.1:{server.server_address[1]}/v1/"
    try:
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", MODEL, "--base-url", base, "--seeds", "1", "--no-grade",
                           "--out", str(tmp_path / "runs")])
    finally:
        server.shutdown()
    run_dir = _run_dir(tmp_path / "runs")
    meta = read_json(run_dir / "run.json")
    calls = read_json(run_dir / episode_name(meta["seeds"][0]) / "llm_calls.json")
    assert calls and {call["date"] for call in calls} <= {meta["started"][:10], meta["ended"][:10]}


class FakeAgentAsTool:
    """The constructor and run() of an Agent-As-Tool runner, recording what the driver gave it."""

    seen = {}

    def __init__(self, server_paths, base_url, api_key, model, temperature=None, log_path=None, example_start=None, on_idle=None):
        FakeAgentAsTool.seen = {"model": model, "temperature": temperature}
        self.calls, self.trajectory = [], []

    async def run(self, question):
        FakeAgentAsTool.seen["question"] = question
        return types.SimpleNamespace(question=question, answer="No repair was made.", error=None, react_steps=1,
                                     react_trajectory={"trajectroy_log": []}, sub_agents=[])


@pytest.mark.parametrize("variant,fold,adapter", [("ft", 1, "rft_agent_as_tool_f1"), ("ft_unfiltered", 2, "rft_agent_as_tool_unfiltered_f2"),
                                                  ("ft_gepa", 1, "rft_agent_as_tool_f1")])
def test_a_fine_tuned_variant_sends_the_adapter_name_and_records_its_weights(tmp_path, monkeypatch, variant, fold, adapter):
    root = tmp_path / "training"
    weights = {arm: _adapter(root, "agent_as_tool", arm, fold, weights=arm.encode()) for arm in (training.FILTERED, training.UNFILTERED)}
    _block(root, "agent_as_tool", "ft", fold, adapter_weights=weights[training.FILTERED])
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setattr(agent_runs, "agent_as_tool_runner_class", lambda: FakeAgentAsTool)
    _first_test_episode(monkeypatch)
    run_episodes.main(["--config", "base", "--agent", "agent_as_tool", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--seeds", "1",
                       "--starts", "sampled", "--training-variant", variant, "--training-fold", str(fold), "--out", str(tmp_path / "runs")])
    run_dir = _run_dir(tmp_path / "runs")
    meta = read_json(run_dir / "run.json")
    ep = run_dir / episode_name(meta["seeds"][0])
    question = (ep / "question.txt").read_text(encoding="utf-8")
    arm = training.ARM_OF[variant]
    assert FakeAgentAsTool.seen["model"] == f"{MODEL}:{adapter}" and FakeAgentAsTool.seen["temperature"] is None
    assert FakeAgentAsTool.seen["question"] == (f"{question}\n\n{BLOCK}" if variant == "ft_gepa" else question)
    assert meta["model"] == MODEL and meta["model_id"] == f"{MODEL}:{adapter}" and meta["training_variant"] == variant
    assert meta[training.TRAINING_FOLD_KEY] == fold
    folder = root / "adapters" / adapter
    counts, steps = COUNTS[arm]
    # what the table of training records reads of the adapter travels with the run: the data it trained on, the examples and steps of its arm and the
    # episodes of its arm that reached a single fault of the held-out fold
    assert meta["adapter"] == {"name": adapter, "path": f"adapters/{adapter}", "arm": arm, "fold": fold, "asset_id": RIG, "weights_sha256": weights[arm],
                               "manifest_sha256": training.sha256_file(folder / training.ADAPTER_MANIFEST),
                               "data_manifest_sha256": training.sha256_file(root / "data" / f"agent_as_tool_f{fold}" / training.DATA_MANIFEST),
                               "training": {"episodes_sampled": 1190, "kept_by_filter": 37, "control_trajectories": 37,
                                            "examples": counts["kept"], "examples_over_max_length": counts["dropped_over_max_length"],
                                            "optimizer_steps": steps, training.VISITS: DATA["arms"][arm][training.VISITS], "gate": GATE},
                               "trained_ended": ENDED, "served_started": None, "served_checked": False, "stamp": None, "built_from": None,
                               training.TRAINING_SPLIT_KEY: training.REGISTERED}
    assert meta["agent_label"] == f"agent_as_tool_{MODEL}" and {row["agent"] for row in read_rows(run_dir)} == {f"agent_as_tool_{MODEL}"}


def _stale(root):
    (root / "blocks" / "gepa_ft_causal_f1.txt").write_text(BLOCK + " Changed after the export.", encoding="utf-8")


@pytest.mark.parametrize("variant,prepare,message", [
    ("ft", lambda root: None, "holds no training_manifest.json"),
    ("ft", lambda root: _adapter(root, "causal", training.FILTERED, data_sha="2" * 64), "trained on the data manifest"),
    ("ft", lambda root: write_json(root / "adapters" / "rft_causal_f1" / training.ADAPTER_MANIFEST,
                                   {"agent": "plan_execute", "arm": "filtered", "fold": 1, **CONVENTION}), "trained for the agent 'plan_execute'"),
    ("ft", lambda root: _adapter(root, "causal", training.FILTERED, recorded_fold=2), "of fold 2, and"),
    ("ft_unfiltered", lambda root: write_json(root / "adapters" / "rft_causal_unfiltered_f1" / training.ADAPTER_MANIFEST,
                                              {"agent": "causal", "arm": "filtered", "fold": 1, **CONVENTION}), "needs the arm 'unfiltered'"),
    ("gepa", lambda root: None, "holds no gepa_causal_f1.txt"),
    ("gepa", lambda root: _block(root, "causal", training.AS_RELEASED, fold=2), "holds no gepa_causal_f1.txt"),
    ("gepa", lambda root: _block(root, "causal", training.AS_RELEASED, recorded="0" * 64), "does not match its record"),
    ("gepa", lambda root: _block(root, "causal", training.AS_RELEASED, recorded_fold=2), "does not match its record"),
    ("ft_gepa", lambda root: _block(root, "causal", "ft", adapter_weights="1" * 64), "written by a loop on the adapter weights"),
    ("ft_gepa", lambda root: (_block(root, "causal", "ft", adapter_weights=hashlib.sha256(b"stand-in weights").hexdigest()), _stale(root)),
     "does not match its record"),
])
def test_a_variant_whose_adapter_or_block_is_missing_or_foreign_stops_before_any_episode(tmp_path, monkeypatch, variant, prepare, message):
    root = tmp_path / "training"
    root.mkdir()
    if variant == "ft_gepa":
        _adapter(root, "causal", training.FILTERED)
    if prepare is not None:
        (root / "adapters" / "rft_causal_f1").mkdir(parents=True, exist_ok=True)
        (root / "adapters" / "rft_causal_f1" / "adapter_model.safetensors").write_bytes(b"stand-in weights")
        (root / "adapters" / "rft_causal_unfiltered_f1").mkdir(parents=True, exist_ok=True)
        (root / "adapters" / "rft_causal_unfiltered_f1" / "adapter_model.safetensors").write_bytes(b"stand-in weights")
        prepare(root)
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    with pytest.raises(SystemExit, match=message):
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--seeds", "1",
                           "--starts", "sampled", "--training-variant", variant, "--training-fold", "1", "--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()  # refused before the run folder, so no episode played


def test_a_trained_variant_needs_the_training_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("HYD_TRAINING", raising=False)
    monkeypatch.setenv("API_KEY", "stand-in-key")
    with pytest.raises(SystemExit, match="HYD_TRAINING"):
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--seeds", "1",
                           "--starts", "sampled", "--training-variant", "gepa", "--training-fold", "1", "--out", str(tmp_path / "runs")])


@pytest.mark.parametrize("replicate", [1, 4])
def test_a_trained_run_plays_the_test_episodes_of_the_fold_its_model_holds_out_and_the_untrained_run_all_of_them(replicate):
    """The model of a fold holds that fold out and never plays a combination it trained on or can
    reach by a repair, so a trained run of a replicate plays the episodes of its fold, 24 for the model of fold 1 and 23 for that of
    fold 2, which together are the replicate, while the agent without training plays all 47 in one run."""
    _, cycles, blocks = load_world()
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    _name, config = load_config("static")
    env = HydraulicRigEnv(cycles, blocks, config)
    line = ["--config", "static", "--agent", "causal", "--starts", "sampled", "--replicate", str(replicate)]
    played = {}
    for fold in training.TRAINING_FOLDS:
        a = run_episodes.build_parser().parse_args([*line, "--training-variant", "ft", "--training-fold", str(fold)])
        played[fold] = run_episodes.planned_episodes.chosen_seeds(a, config, cycles, blocks)
        for seed in played[fold]:
            env.reset(seed)
            assert training.fold_of(start_states.start_state(env)[0]) == fold
    untrained = run_episodes.planned_episodes.chosen_seeds(run_episodes.build_parser().parse_args(line), config, cycles, blocks)
    assert untrained == start_states.replicate_seeds(env, replicate) and len(untrained) == 47
    assert (len(played[1]), len(played[2])) == (24, 23) and sorted(played[1] + played[2]) == sorted(untrained)


def test_every_seeded_preset_deals_the_registered_split():
    """The driver deals the test episodes and the training starts of a fold through the registered split, which the rule deals on
    the start combinations of the asset, so the rule on every seeded preset a trained or a sampling run may name gives that split."""
    _, cycles, blocks = load_world()
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    checked = []
    registered = json.loads(training.FOLDS_FILE.read_text(encoding="utf-8"))
    for path in sorted(preset_files(CONFIGS).values()):
        _name, config = load_config(path.stem)
        # a start without a fault belongs to no fold, so no trained or sampling run names a preset whose fault rule admits one
        if config.anchors is None and config.fault_rule["min_faulty"] > 0:
            assert start_states.combination_folds(HydraulicRigEnv(cycles, blocks, config), registered["k"], registered["seed"]) == training.FOLDS, path.stem
            checked.append(path.stem)
    assert "base" in checked and "static" in checked


def test_the_training_starts_play_the_training_seeds_at_the_sampling_temperature_in_the_mode_training(tmp_path, monkeypatch):
    """A trial of the training path, which plays the development split as every sampling run outside a run of record does: the
    training starts of a fold under one sample, at the sampling temperature, in the mode training."""
    seeds_of = start_states.training_seeds
    monkeypatch.setattr(start_states, "training_seeds", lambda env, split, sample, fold, master_seed, pool: seeds_of(env, split, sample, fold, master_seed, pool)[:2])
    monkeypatch.setenv("API_KEY", "stand-in-key")
    monkeypatch.setattr(agent_runs, "plan_execute_runner_class", lambda: FakePlanExecute)
    run_episodes.main(["--config", "base", "--agent", "plan_execute", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--starts", "training",
                       "--training-sample", "2", "--training-fold", "2", "--training-split", training.DEVELOPMENT, "--out", str(tmp_path / "runs")])
    run_dir = _run_dir(tmp_path / "runs")
    meta = read_json(run_dir / "run.json")
    _, cycles, blocks = load_world()
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    played = run_episodes.planned_episodes.development_config(load_config("base")[1], blocks)
    split = training.asset_training(RIG, cycles.facts_sha256, split=training.DEVELOPMENT).split
    expected = seeds_of(HydraulicRigEnv(cycles, blocks, played), split, 2, 2, 42, 8)[:2]
    assert meta["mode"] == training.TRAINING_MODE and meta["starts"] == "training" and meta["training_sample"] == 2
    assert meta[training.TRAINING_FOLD_KEY] == 2 and meta[training.TRAINING_SPLIT_KEY] == training.DEVELOPMENT
    assert meta["seeds"] == expected and (meta["master_seed"], meta["seed_pool"]) == (42, 8)
    assert meta["sampling_temperature"] == training.numbers(RIG).sampling_temperature == 0.7 and meta["training_variant"] == training.AS_RELEASED
    assert FakePlanExecute.seen["temperature"] == training.numbers(RIG).sampling_temperature


class Reached(Exception):
    """Raised by a stand-in of the step after a check, so that a test sees that a run passed the check."""


def test_a_sampling_run_on_the_registered_split_plays_only_as_a_run_of_record(tmp_path, monkeypatch):
    """The registered split is cross-fitted, so the training starts of the model of a fold lie on test combinations of the other
    folds: a run of --starts training on the registered split whose start check gives a reason is refused by name before its run
    folder exists and pointed to the development split, while the same run on the development split, the same run of record and
    an evaluation run on the test starts with the same reason each reach the building of their runner."""
    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(run_episodes.runner_setup, "build_runner", reached)
    monkeypatch.setenv(READ_ONLY_TABLES, "0")  # a run of record holds the tables read only for its process, so the teardown restores it
    out = tmp_path / "runs"
    model = ["--config", "base", "--agent", "plan_execute", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--out", str(out)]
    sampling = [*model, "--starts", "training", "--training-sample", "1", "--training-fold", "1"]
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args: ["a reason of the start check"])
    with pytest.raises(SystemExit, match="--starts training on the registered split .*a reason of the start check.*--training-split development"):
        run_episodes.main(sampling)
    for args in ([*sampling, "--training-split", training.DEVELOPMENT], [*model, "--starts", "sampled"]):  # a trial, and an evaluation run
        with pytest.raises(Reached):
            run_episodes.main(args)
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args: [])  # a run of record
    with pytest.raises(Reached):
        run_episodes.main(sampling)
    assert not out.exists()


@pytest.mark.parametrize("args,message", [
    (["--agent", "oracle", "--training-variant", "ft"], "trains one of plan_execute, agent_as_tool, causal"),
    (["--agent", "plan_execute", "--training-sample", "2"], "needs --starts training"),
    (["--agent", "plan_execute", "--starts", "training", "--training-variant", "ft"], "samples one of"),
    (["--agent", "scripted", "--starts", "training"], "samples one of"),
    (["--agent", "plan_execute", "--starts", "training", "--config", "anchored"], "samples one of"),
    (["--agent", "plan_execute", "--training-variant", "ft"], "need --training-fold"),
    (["--agent", "plan_execute", "--starts", "training"], "need --training-fold"),
    (["--agent", "plan_execute", "--training-fold", "1"], "plays every test episode in one line"),
    (["--agent", "plan_execute", "--starts", "sampled", "--training-fold", "2"], "plays every test episode in one line"),
    (["--agent", "plan_execute", "--training-variant", "ft", "--training-fold", "1"], "needs --starts sampled"),
    (["--agent", "causal", "--starts", "combinations", "--training-variant", "gepa", "--training-fold", "2"], "needs --starts sampled"),
])
def test_the_training_options_are_refused_where_they_do_not_apply(tmp_path, monkeypatch, args, message):
    monkeypatch.setenv("API_KEY", "stand-in-key")
    with pytest.raises(SystemExit, match=message):
        run_episodes.main(["--config", "static", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", *args, "--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()


def test_the_variants_their_files_and_the_task_message():
    assert training.TRAINING_VARIANTS == ("as_released", "gepa", "gepa_score_only", "ft", "ft_gepa", "ft_unfiltered")
    assert training.TRAINING_VARIANT_KEY == "training_variant"
    assert training.TRAINING_FOLDS == training.held_out_folds.fold_ids() == (1, 2) and training.TRAINING_FOLD_KEY == "training_fold"
    assert run_episodes.build_parser().get_default("training_variant") == training.AS_RELEASED
    assert run_episodes.build_parser().get_default("training_fold") is None
    assert [training.adapter_name("causal", v, 1, RIG) for v in training.TRAINING_VARIANTS] == [
        None, None, None, "rft_causal_f1", "rft_causal_f1", "rft_causal_unfiltered_f1"]
    assert training.block_name("causal", "as_released", 2, RIG) == "gepa_causal_f2" and training.block_name("causal", "ft", 1, RIG) == "gepa_ft_causal_f1"
    assert training.block_name("causal", "as_released", 1, RIG, "score_only") == "gepa_score_only_causal_f1"
    assert training.BLOCK_OF == {"gepa": ("as_released", "verifier"), "gepa_score_only": ("as_released", "score_only"), "ft_gepa": ("ft", "verifier")}
    with pytest.raises(SystemExit, match="base weights alone"):
        training.block_name("causal", "ft", 1, RIG, "score_only")  # the control runs on the base weights only
    assert training.data_name("causal", 2, RIG) == "causal_f2"
    for bad in (None, 0, 3):
        with pytest.raises(SystemExit):
            training.block_name("causal", "ft", bad, RIG)
    assert training.with_task_block("Briefing.", "") == "Briefing." and training.with_task_block("Briefing.", "Block.") == "Briefing.\n\nBlock."


def test_pending_lines_tell_the_variants_the_folds_and_the_sampling_lines_apart():
    """A finished run of one variant, one fold or one sample never marks another line done, and a run.json from before the
    variants or the folds existed reads as the agent without training."""
    from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines

    line = "--config static --agent causal"
    untrained = pending_lines.expected(line, "sampled")
    fold_1, fold_2 = (pending_lines.expected(f"{line} --training-variant ft --training-fold {k}", "sampled") for k in (1, 2))
    assert untrained["training_variant"] == training.AS_RELEASED and untrained["training_fold"] is None
    assert (fold_1["training_variant"], fold_1["training_fold"], fold_2["training_fold"]) == ("ft", 1, 2)
    assert "training_sample" not in untrained
    first, second = (pending_lines.expected(f"{line} --training-sample {k} --training-fold 1", "training") for k in (1, 2))
    other = pending_lines.expected(f"{line} --training-sample 1 --training-fold 2", "training")
    assert (first["training_sample"], second["training_sample"]) == (1, 2) and (first["master_seed"], first["seed_pool"]) == (42, 8)
    assert pending_lines.matches({k: v for k, v in untrained.items() if k not in ("training_variant", "training_fold")}, untrained, {})
    assert not pending_lines.matches({**untrained, "training_variant": "ft"}, untrained, {})
    assert pending_lines.matches(fold_1, fold_1, {}) and not pending_lines.matches(fold_1, fold_2, {})
    assert not pending_lines.matches({k: v for k, v in fold_1.items() if k != "training_fold"}, fold_1, {})
    assert pending_lines.matches(first, first, {}) and not pending_lines.matches(first, second, {}) and not pending_lines.matches(other, first, {})
    assert pending_lines.units([line], "training") == [line]  # a line of the training starts is one run, no replicate


def test_the_score_only_control_plays_its_own_block_and_never_the_verifier_loop(tmp_path):
    """gepa_score_only reads the block that the loop under the score-only card exported on the base weights, under
    a name of its own, with the counts of that loop, and refuses a block whose record names the other card."""
    root = tmp_path / "training"
    _block(root, "causal", training.AS_RELEASED, text="Read the verifier's card.")
    _block(root, "causal", training.AS_RELEASED, text="Read the score alone.", card="score_only")
    text, record = training.block_of("causal", "gepa_score_only", 1, RIG, root)
    assert text == "Read the score alone." and record["name"] == "gepa_score_only_causal_f1" and record["card"] == "score_only"
    assert record[training.VISITS] == 1 and record["stream_episodes"] == 240
    assert training.block_of("causal", "gepa", 1, RIG, root)[0] == "Read the verifier's card."
    assert training.adapter_of("causal", "gepa_score_only", 1, RIG, root) is None
    _block(root, "causal", training.AS_RELEASED, text="Read the score alone.", card="score_only", recorded_card="verifier")
    with pytest.raises(SystemExit, match="card 'verifier'"):
        training.block_of("causal", "gepa_score_only", 1, RIG, root)


def test_a_trained_run_counts_as_done_without_the_judge_since_the_judge_runs_after_the_grid(tmp_path):
    """Nothing co-serves the judge, and judge_finished_runs.py scores the evaluation runs of the training table after the grid
    as for the main table and the ladder, so pending_lines counts a finished run of a trained variant as done with or without a judge record."""
    from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines

    trained = "--config static --agent causal --training-variant gepa --training-fold 1"
    fields = pending_lines.expected(f"{trained} --replicate 1", "sampled")
    folder = tmp_path / "runs" / "run0"
    write_json(folder / "run.json", {**fields, "ended": "2026-11-09T02:00:00", "started": "2026-11-09T01:00:00", "episodes_planned": 1,
                                     "judge": None})
    (folder / "scores.csv").write_text("seed\n0\n", encoding="utf-8")
    assert pending_lines.pending([trained], [tmp_path / "runs"], {}, replicates=(1,)) == []


def test_the_minimum_and_the_extension_are_registered_and_read_from_the_split():
    """The fields of configs/training/hyd_1/rft.json, one kept episode per training start state and one more round of the
    same size, and a gate that reads the count of training start states it is given, no number of the rig."""
    assert training.CONFIG["minimum"] == {"kept_per_training_state": 1, "extension_rounds": 1}
    assert training.numbers(RIG)[1:3] == (1, 1)
    per_round = training.numbers(RIG).samples_per_state
    assert training.round_samples(1, RIG) == list(range(1, per_round + 1)) and training.extension_samples(RIG) == training.round_samples(2, RIG)
    assert training.gate_of(4, 4, 5, 1, RIG) == {"minimum": 5, "kept_per_training_state": 1, "training_states": 5,
                                            "rounds_sampled": 1, "kept": 4, "distinct_training_states": 4, "second_round_due": True,
                                            "trained": False}
    assert not training.gate_of(4, 4, 5, 2, RIG)["second_round_due"] and training.gate_of(5, 3, 5, 1, RIG)["trained"]


@pytest.mark.parametrize("variant", ["ft", "ft_gepa", "ft_unfiltered"])
def test_a_fold_below_the_minimum_writes_a_marker_run_that_plays_nothing_and_names_its_alias(tmp_path, monkeypatch, variant):
    """The same command on a fold without an adapter writes a run folder of the
    replicate with no episode, the gate and the alias in run.json, the runs without training of the fold for all three variants."""
    from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines

    root = tmp_path / "training"
    data_sha = _data(root, "causal", 1, {**DATA, "gate": BELOW})
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    line = f"--config static --agent causal --model {MODEL} --base-url http://127.0.0.1:9/v1 --training-variant {variant} --training-fold 1"
    run_episodes.main([*line.split(), "--starts", "sampled", "--replicate", "2", "--out", str(tmp_path / "runs")])
    run_dir = _run_dir(tmp_path / "runs")
    meta = read_json(run_dir / "run.json")
    assert meta["episodes_planned"] == 0 and meta["seeds"] == [] and read_rows(run_dir) == [] and meta["replicate"] == 2
    assert meta["adapter"] == {"name": None, "trained": False, "reason": "below_minimum", "alias": training.AS_RELEASED,
                               "arm": training.ARM_OF[variant], "fold": 1, "asset_id": RIG, training.TRAINING_SPLIT_KEY: training.REGISTERED,
                               "data_manifest_sha256": data_sha, "gate": BELOW}
    assert meta["task_block"] is None and ":" not in str(meta["model_id"]) and meta["training_variant"] == variant
    # the line of that replicate is done for pending_lines, with or without a judge, since the marker plays no episode
    assert pending_lines.pending([line], [tmp_path / "runs"], {}, replicates=(2,)) == []


def test_a_marker_never_stands_beside_an_adapter_and_an_adapter_never_beside_a_marker(tmp_path, monkeypatch):
    root = tmp_path / "training"
    _data(root, "causal", 1, {**DATA, "gate": BELOW})
    (root / "adapters" / "rft_causal_f1").mkdir(parents=True)
    with pytest.raises(SystemExit, match="holds an adapter anyway"):
        training.adapter_of("causal", "ft", 1, RIG, root)
    other = tmp_path / "other"
    _data(other, "causal", 1, DATA)  # the gate is met, so a missing adapter is an error and never a marker
    with pytest.raises(SystemExit, match="holds no training_manifest.json"):
        training.adapter_of("causal", "ft", 1, RIG, other)


@pytest.mark.parametrize("variant", ["ft", "ft_gepa", "ft_unfiltered"])
def test_a_fine_tuned_line_stops_while_a_further_round_of_sampling_is_due(tmp_path, monkeypatch, variant):
    """A gate below the minimum is final only after the last round of sampling. While a round is still due the fold has neither
    an adapter nor a marker, so a fine tuned line stops before any run folder and its pair stays pending until the data is built
    again."""
    root = tmp_path / "training"
    due = training.gate_of(kept=12, distinct_states=9, training_states=30, rounds=1, asset_id=RIG)
    assert due["second_round_due"] and not due["trained"]
    _data(root, "causal", 1, {**DATA, "gate": due})
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    with pytest.raises(SystemExit, match="waits for a further round of sampling"):
        training.adapter_of("causal", variant, 1, RIG, root)
    with pytest.raises(SystemExit, match="waits for a further round of sampling"):
        run_episodes.main(["--config", "static", "--agent", "causal", "--model", MODEL, "--base-url", "http://127.0.0.1:9/v1",
                           "--starts", "sampled", "--training-variant", variant, "--training-fold", "1", "--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()


def test_the_loop_on_the_adapter_stops_while_a_further_round_of_sampling_is_due(tmp_path, monkeypatch, capsys):
    """The loop of the ft_gepa variant waits for the same gate: while a round is due it stops with the message of the gate, prints no
    SKIPPED and writes no run folder."""
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online

    root = tmp_path / "training"
    _data(root, "causal", 2, {**DATA, "gate": training.gate_of(kept=12, distinct_states=9, training_states=30, rounds=1, asset_id=RIG)})
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    out = tmp_path / "runs"
    with pytest.raises(SystemExit, match="waits for a further round of sampling"):
        gepa_online.main(["--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--reflection-model", "r", "--reflection-base-url",
                          "http://127.0.0.1:9/v1", "--training-fold", "2", "--training-variant", "ft", "--config", "static",
                          "--servers", "rig", "--out", str(out)])
    assert "SKIPPED" not in capsys.readouterr().out and not out.exists()


def test_the_loop_on_the_filtered_adapter_does_not_run_on_a_fold_without_one(tmp_path, monkeypatch, capsys):
    """The ft_gepa variant of a fold without an adapter reads the runs without training, so its loop of GEPA does not
    run; gepa_prompt_optimization/ says so and writes no run folder."""
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online

    root = tmp_path / "training"
    _data(root, "causal", 2, {**DATA, "gate": BELOW})
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    out = tmp_path / "runs"
    assert gepa_online.main(["--model", MODEL, "--base-url", "http://127.0.0.1:9/v1", "--reflection-model", "r", "--reflection-base-url",
                             "http://127.0.0.1:9/v1", "--training-fold", "2", "--training-variant", "ft", "--config", "static",
                             "--servers", "rig", "--out", str(out)]) is None
    assert "SKIPPED causal f2 ft" in capsys.readouterr().out and not out.exists()


def test_a_server_that_started_before_the_adapter_was_trained_is_refused(tmp_path, monkeypatch):
    """A server that started before the training job of the adapter ended serves other weights under the adapter's name, so
    adapter_of stops and names both times, a later start passes and the record keeps both, and without a start time the record says
    that nothing was compared. The driver reads the start from the endpoint file of the grid job, and only for its own endpoint."""
    root = tmp_path / "training"
    _adapter(root, "causal", training.FILTERED)
    with pytest.raises(SystemExit, match=f"started at 2026-11-20T09:59:59-05:00, before the training of the adapter rft_causal_f1 ended at {ENDED}"):
        training.adapter_of("causal", "ft", 1, RIG, root, served_started="2026-11-20T09:59:59-05:00")
    later = training.adapter_of("causal", "ft", 1, RIG, root, served_started="2026-11-20T15:30:00+00:00")
    assert (later["trained_ended"], later["served_started"], later["served_checked"]) == (ENDED, "2026-11-20T15:30:00+00:00", True)
    unknown = training.adapter_of("causal", "ft", 1, RIG, root)
    assert unknown["served_started"] is None and unknown["served_checked"] is False
    with pytest.raises(SystemExit, match="without its offset from UTC"):
        training.adapter_of("causal", "ft", 1, RIG, root, served_started="2026-11-20T15:30:00")
    endpoint = tmp_path / "endpoint.txt"
    endpoint.write_text("http://node:30000/v1\njob 7 node n1 model m served_name m\nstarted 2026-11-20T11:00:00-05:00\n", encoding="utf-8")
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(endpoint))
    assert run_episodes.serving_started("http://node:30000/v1") == "2026-11-20T11:00:00-05:00"
    assert run_episodes.serving_started("http://other:30000/v1") is None


def test_the_adapter_and_the_block_records_name_the_runs_they_were_built_from(tmp_path):
    """The manifest of an adapter repeats the sampling runs of its data with the SHA-256 of each run.json and the stamp of the data,
    the block record names the run of its loop and the runs behind the adapter the loop played on, and the records of run.json carry
    both, which training.built_from returns for the replay check."""
    root = tmp_path / "training"
    _adapter(root, "causal", training.FILTERED)
    sampled = [{"run": "sample1", "run_json_sha256": "a" * 64}, {"run": "sample2", "run_json_sha256": "b" * 64}]
    path = root / "adapters" / "rft_causal_f1" / training.ADAPTER_MANIFEST
    write_json(path, {**read_json(path), "built_from": sampled, "stamp": "development"})
    adapter = training.adapter_of("causal", "ft", 1, RIG, root)
    assert training.built_from(adapter) == sampled and adapter["stamp"] == "development"
    loop = [{"run": "loop_f1", "run_json_sha256": "c" * 64}, *sampled]
    _block(root, "causal", training.AS_RELEASED)
    record_path = root / "blocks" / "gepa_causal_f1.json"
    write_json(record_path, {**read_json(record_path), "built_from": loop, "stamp": "development"})
    _text, block = training.block_of("causal", "gepa", 1, RIG, root)
    assert training.built_from(block) == loop and block["stamp"] == "development" and training.built_from(None) == []
