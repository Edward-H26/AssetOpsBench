"""The folds and the training files of an asset: K and the fold ids come from the asset's folds file
through training.asset_training, the files of every asset live under configs/training/<asset_id>/, the rig's moved files keep
their numbers and their split, the derived numbers of rft.json follow from the split, a made-up asset with K = 3 runs through the
training entry points and through the drivers' code that reads folds, a run that names no fold reads no training file, a split
dealt from another data file is refused to the recordings of a run, and --starts development plays the development starts of
the asset. A trial on the development split plays development combinations alone, and a run of one split refuses the
data, the adapters and the blocks of the other."""
from CausalDiagnosis.utils.paths import ROOT
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.training import rejection_sampling_examples as rft_data
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.table_definitions import training_variants as training
from CausalDiagnosis.datasets.splits import development_split
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

HAS_DATA = Path(default_facts_path()).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
# the rig's two files as they stood before they moved to configs/training/hyd_1/: the md5 of the canonical JSON of each without
# its text fields, the "about" and the labels of the same kind that the old files carried, and without facts_sha256, the hash of
# the data file that the split was dealt from, and starts, the name of the starts it was dealt over, so the numbers and the
# split are pinned
# rft.json without the number of folds and the training start states of each model, which the split gives, and without a cap on the length of a
# task block, with the gepa block of a loop that has no held-out phase, a count of attempts of a proposal and a limit of unanswered proposals, and with the folds file of the development split
MOVED = {"folds.json": "174d14f51d49d64457fb574536d49e91", "rft.json": "60c26d2b9a86737d3893bff7819ecc59"}
UNPINNED = ("about", "decision", "facts_sha256", "starts")
MADE_UP = "made_up_3"
COMPONENTS = ["motor", "bearing", "seal"]
# a split of K = 3 written by hand: fold h holds the starts of its own faults, and a start trains every model whose fold it avoids
SPLIT = [{"combination": [1, 0, 0], "fold": 1, "trains": True, "models": [2, 3]},
         {"combination": [0, 1, 0], "fold": 2, "trains": True, "models": [1, 3]},
         {"combination": [0, 0, 1], "fold": 3, "trains": True, "models": [1, 2]},
         {"combination": [1, 1, 0], "fold": 1, "trains": True, "models": [3]},
         {"combination": [0, 1, 1], "fold": 3, "trains": True, "models": [1]}]
ATOMS = [{"component": "motor", "level": 1, "fold": 1}, {"component": "bearing", "level": 1, "fold": 2},
         {"component": "seal", "level": 1, "fold": 3}]


@pytest.fixture
def made_up(tmp_path, monkeypatch):
    """configs/training/<asset_id>/ of a made-up asset with K = 3, beside the rig's own folder, read through asset_training."""
    folder = tmp_path / "training"
    (folder / MADE_UP).mkdir(parents=True)
    (folder / "hyd_1").symlink_to(training.held_out_folds.TRAINING_DIR / "hyd_1")
    # numbers of its own, so that a call that read the rig's numbers would show
    config = {**training.CONFIG, "samples_per_state": 5, "minimum": {"kept_per_training_state": 2, "extension_rounds": 2},
              "sampling_temperature": 0.3, "folds": {"file": "folds.json"},
              "gepa": {**training.CONFIG["gepa"], "unanswered_proposals": 4, "proposal_attempts": 3}}
    (folder / MADE_UP / "rft.json").write_text(json.dumps(config), encoding="utf-8")
    record = {"name": "folds", "asset_id": MADE_UP, "k": 3, "seed": 0, "components": COMPONENTS,
              "combinations": SPLIT, "atoms": ATOMS}
    (folder / MADE_UP / "folds.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(training.held_out_folds, "TRAINING_DIR", folder)
    training.asset_training.cache_clear()
    yield MADE_UP
    training.asset_training.cache_clear()


def test_a_split_dealt_from_the_recordings_of_another_data_file_is_refused(made_up):
    """The folds file names the SHA-256 of the data file it was dealt from. A caller with recordings in hand reads the split only
    when they come from that file, and is told the command that deals the split again, a caller without recordings compares
    nothing, and a split that names no data file is refused to recordings as well, since nothing says what it was dealt from."""
    folds_file = training.held_out_folds.TRAINING_DIR / made_up / "folds.json"
    record = json.loads(folds_file.read_text(encoding="utf-8"))
    folds_file.write_text(json.dumps({**record, "facts_sha256": "a" * 64}), encoding="utf-8")
    training.asset_training.cache_clear()
    assert training.asset_training(made_up).facts_sha256 == "a" * 64 and training.asset_training(made_up, "a" * 64).fold_ids == (1, 2, 3)
    with pytest.raises(records.DataFileError, match=f"built from the data file with SHA-256 {'a' * 64}.*python -m CausalDiagnosis.experiments.table_definitions.write_training_split --asset {made_up} --k 3 --seed 0"):
        training.asset_training(made_up, "b" * 64)
    with pytest.raises(records.DataFileError, match="only for recordings that were loaded from a data file"):
        training.asset_training(made_up, None)
    folds_file.write_text(json.dumps(record), encoding="utf-8")
    training.asset_training.cache_clear()
    with pytest.raises(records.DataFileError, match="a data file that it does not name"):
        training.asset_training(made_up, "a" * 64)


@needs_data
def test_the_seeds_of_a_trained_and_a_sampling_line_refuse_the_split_under_the_recordings_of_another_data_file():
    """run_episodes.chosen_seeds reads the split with the hash of the recordings it plays: on the recordings of the data file that
    the rig's folds file names a trained line and a sampling line get their seeds, and on the same recordings named as another file
    both stop before any seed is chosen."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world

    _, cycles, blocks = load_world()
    config = load_config("static")[1]
    parser = run_episodes.build_parser()
    lines = [parser.parse_args(["--config", "static", "--agent", "causal", "--starts", "training", "--training-fold", "1"]),
             parser.parse_args(["--config", "static", "--agent", "causal", "--starts", "sampled", "--training-variant", "ft",
                                "--training-fold", "1"])]
    assert cycles.facts_sha256 == training.asset_training(training.DEFAULT_ASSET).facts_sha256
    assert all(run_episodes.planned_episodes.chosen_seeds(line, config, cycles, blocks) for line in lines)
    other = records.Recordings(cycles, "0" * 64)
    for line in lines:
        with pytest.raises(records.DataFileError, match="the data file in use has SHA-256 " + "0" * 64):
            run_episodes.planned_episodes.chosen_seeds(line, config, other, blocks)


def test_the_flat_training_files_of_the_rig_are_gone_and_the_rig_reads_the_folder_of_its_asset():
    assert not any((training.held_out_folds.TRAINING_DIR / name).exists() for name in MOVED)
    rig = training.asset_training()
    assert rig.config_file.parent == rig.folds_file.parent == training.held_out_folds.TRAINING_DIR / training.DEFAULT_ASSET


def _pinned(value):
    """A registered file without the fields that the pins leave out, at any depth."""
    if isinstance(value, dict):
        return {key: _pinned(item) for key, item in value.items() if key not in UNPINNED}
    return [_pinned(item) for item in value] if isinstance(value, list) else value


def _keys(value):
    """Every key of a registered file, at any depth."""
    if isinstance(value, dict):
        return set(value).union(*map(_keys, value.values()))
    return set().union(*map(_keys, value)) if isinstance(value, list) else set()


def test_the_moved_files_of_the_rig_keep_their_numbers_and_their_split_and_hold_no_decision_key():
    folder = training.held_out_folds.TRAINING_DIR / training.DEFAULT_ASSET
    for name, digest in MOVED.items():
        text = (folder / name).read_text(encoding="utf-8")
        canonical = json.dumps(_pinned(json.loads(text)), sort_keys=True).encode("utf-8")
        assert hashlib.md5(canonical).hexdigest() == digest, name
        # the key decision, which named a decision of the project, stands in neither file
        assert "decision" not in _keys(json.loads(text)), name
    assert training.CONFIG_FILE == training.held_out_folds.TRAINING_DIR / training.DEFAULT_ASSET / training.CONFIG_NAME


def test_a_training_folder_of_the_earlier_fold_naming_is_refused_by_every_reader(tmp_path):
    """Before the held-out naming the fold number of a training record named the fold its model trained on, under the same names,
    so a record that does not state the fold convention could put a model on the test starts it trained on. Every reader of a
    training folder refuses such a record by name: the data manifest beside the examples, the training manifest of an adapter
    and the record of a block, here written as the earlier code wrote them."""
    from CausalDiagnosis.training import train_lora

    root, block = tmp_path / "training", "Read the cheapest group first."
    data = root / "data" / training.data_name("causal", 1, training.DEFAULT_ASSET)
    adapter = root / "adapters" / training.adapter_name("causal", "ft", 1, training.DEFAULT_ASSET)
    record = root / "blocks" / "gepa_causal_f1.json"
    for folder in (data, adapter, record.parent):
        folder.mkdir(parents=True)
    (data / "examples_filtered.jsonl").write_text(json.dumps({"agent": "causal", "arm": "filtered", "fold": 1}) + "\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"stand-in weights")
    record.with_suffix(".txt").write_text(block, encoding="utf-8")

    def write(stated):
        """The three records of one agent and fold, with the fields of stated."""
        arms = {arm: {"episodes": 37, training.VISITS: 0} for arm in (training.FILTERED, training.UNFILTERED)}
        manifest = {"agent": "causal", "fold": 1, "episodes_sampled": 1190, "gate": training.gate_of(37, 30, 30, 1, training.DEFAULT_ASSET), "arms": arms, **stated}
        (data / training.DATA_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        (adapter / training.ADAPTER_MANIFEST).write_text(json.dumps({"agent": "causal", "arm": "filtered", "fold": 1, "steps": 52, **stated,
            "counts": {"kept": 409}, "data_manifest_sha256": training.sha256_file(data / training.DATA_MANIFEST)}), encoding="utf-8")
        record.write_text(json.dumps({"agent": "causal", "base_variant": "as_released", "fold": 1, "card": "verifier", "adapter": None,
                                      "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(), **stated}), encoding="utf-8")

    write({})
    earlier = "was written under the earlier fold convention"
    with pytest.raises(SystemExit, match=earlier):
        training.adapter_of("causal", "ft", 1, training.DEFAULT_ASSET, root)
    with pytest.raises(SystemExit, match=earlier):
        training.block_of("causal", "gepa", 1, training.DEFAULT_ASSET, root)
    with pytest.raises(SystemExit, match=earlier):
        train_lora.check_gate(data / "examples_filtered.jsonl")
    # another value is refused the same way, and the records that state the convention are read
    write({training.FOLD_CONVENTION_KEY: "trained_on"})
    for read in (lambda: training.adapter_of("causal", "ft", 1, training.DEFAULT_ASSET, root), lambda: training.block_of("causal", "gepa", 1, training.DEFAULT_ASSET, root)):
        with pytest.raises(SystemExit, match=earlier):
            read()
    write({training.FOLD_CONVENTION_KEY: training.FOLD_CONVENTION, "asset_id": training.DEFAULT_ASSET})
    assert training.adapter_of("causal", "ft", 1, training.DEFAULT_ASSET, root)["name"] == "rft_causal_f1"
    assert training.block_of("causal", "gepa", 1, training.DEFAULT_ASSET, root)[0] == block
    assert train_lora.check_gate(data / "examples_filtered.jsonl")["trained"]
    # the adapter's manifest alone without the convention, beside a data manifest that states it, is refused by its own check
    manifest = json.loads((adapter / training.ADAPTER_MANIFEST).read_text(encoding="utf-8"))
    del manifest[training.FOLD_CONVENTION_KEY]
    (adapter / training.ADAPTER_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match=f"{training.ADAPTER_MANIFEST} was written under the earlier fold convention"):
        training.adapter_of("causal", "ft", 1, training.DEFAULT_ASSET, root)


def test_the_default_asset_of_training_is_the_default_asset_of_the_environment():
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET

    assert training.DEFAULT_ASSET == DEFAULT_ASSET


def test_names_gates_and_rounds_read_the_numbers_of_the_asset_of_the_run(made_up):
    """On the made-up asset with K = 3 and numbers of its own, the names take fold 3 and refuse fold 4, and the rounds and the gate
    read its samples per state and its minimum. The same calls for the rig give the rig's numbers."""
    assert training.numbers(made_up) == (5, 2, 2, 0.3, 4, 3)
    assert training.adapter_name("causal", "ft", 3, made_up) == "rft_causal_f3" and training.data_name("causal", 3, made_up) == "causal_f3"
    assert training.block_name("causal", "as_released", 3, made_up) == "gepa_causal_f3"
    for name in (lambda: training.adapter_name("causal", "ft", 4, made_up), lambda: training.block_name("causal", "ft", 4, made_up),
                 lambda: training.data_name("causal", 4, made_up), lambda: training.adapter_name("causal", "ft", 3, training.DEFAULT_ASSET)):
        with pytest.raises(SystemExit, match="names the folds"):
            name()
    assert training.round_samples(2, made_up) == [6, 7, 8, 9, 10] and training.extension_samples(made_up) == list(range(6, 16))
    gate = training.gate_of(9, 5, 5, 1, made_up)
    assert (gate["minimum"], gate["kept_per_training_state"], gate["trained"], gate["second_round_due"]) == (10, 2, False, True)
    assert training.gate_of(9, 5, 5, 3, made_up)["second_round_due"] is False  # after the last of its two extension rounds
    rig = training.numbers(training.DEFAULT_ASSET)
    assert rig != training.numbers(made_up)
    assert training.round_samples(2, training.DEFAULT_ASSET) == list(range(rig.samples_per_state + 1, 2 * rig.samples_per_state + 1))
    assert training.gate_of(9, 5, 5, 1, training.DEFAULT_ASSET)["minimum"] == rig.kept_per_training_state * 5


def _training_records(root, asset_ids, block, extras=({}, {}, {})):
    """The data manifest, the adapter manifest and the block record of the causal agent on fold 1, each naming its asset, with the
    fields of extras added to each in that order."""
    data = root / "data" / "causal_f1"
    adapter = root / "adapters" / "rft_causal_f1"
    record = root / "blocks" / "gepa_causal_f1.json"
    for folder in (data, adapter, record.parent):
        folder.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"stand-in weights")
    record.with_suffix(".txt").write_text(block, encoding="utf-8")
    convention = {training.FOLD_CONVENTION_KEY: training.FOLD_CONVENTION}
    data_asset, adapter_asset, block_asset = asset_ids
    data_extra, adapter_extra, block_extra = extras
    arms = {arm: {"episodes": 37, training.VISITS: 0} for arm in (training.FILTERED, training.UNFILTERED)}
    gate = {"minimum": 30, "kept_per_training_state": 1, "training_states": 30, "rounds_sampled": 1, "kept": 37,
            "distinct_training_states": 30, "second_round_due": False, "trained": True}
    (data / training.DATA_MANIFEST).write_text(json.dumps({"agent": "causal", "fold": 1, "episodes_sampled": 1190, "arms": arms,
                                                           "gate": gate, **convention, "asset_id": data_asset, **data_extra}),
                                              encoding="utf-8")
    (adapter / training.ADAPTER_MANIFEST).write_text(json.dumps({"agent": "causal", "arm": "filtered", "fold": 1, "steps": 52,
                                                                 "counts": {"kept": 409},
                                                                 "data_manifest_sha256": training.sha256_file(data / training.DATA_MANIFEST),
                                                                 **convention, "asset_id": adapter_asset, **adapter_extra}),
                                                     encoding="utf-8")
    record.write_text(json.dumps({"agent": "causal", "base_variant": "as_released", "fold": 1, "card": "verifier", "adapter": None,
                                  "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(), **convention, "asset_id": block_asset,
                                  **block_extra}),
                      encoding="utf-8")


def test_a_run_refuses_training_files_of_another_asset(tmp_path):
    """Each asset has a training folder of its own: adapter_of and block_of stop on a data manifest, an adapter manifest and a block
    record that name another asset, and read the records of the run's asset."""
    root, block, rig = tmp_path / "training", "Read the cheapest group first.", training.DEFAULT_ASSET
    _training_records(root, ("tiny_1", rig, rig), block)
    with pytest.raises(SystemExit, match="manifest.json belongs to the asset 'tiny_1', and the run plays hyd_1"):
        training.adapter_of("causal", "ft", 1, rig, root)
    _training_records(root, (rig, "tiny_1", rig), block)
    with pytest.raises(SystemExit, match="training_manifest.json belongs to the asset 'tiny_1'"):
        training.adapter_of("causal", "ft", 1, rig, root)
    _training_records(root, (rig, rig, "tiny_1"), block)
    with pytest.raises(SystemExit, match="gepa_causal_f1.json belongs to the asset 'tiny_1'"):
        training.block_of("causal", "gepa", 1, rig, root)
    _training_records(root, (rig, rig, rig), block)
    assert training.adapter_of("causal", "ft", 1, rig, root)["asset_id"] == rig
    assert training.block_of("causal", "gepa", 1, rig, root)[0] == block


def test_no_function_on_the_path_of_a_run_has_a_default_asset():
    """The functions that a run calls take the asset of the run, and none of them falls back to the rig."""
    import inspect

    for function in (training.check_fold, training.adapter_name_of_arm, training.adapter_name, training.block_name, training.data_name,
                     training.adapter_of, training.block_of, training.gate_of, training.round_samples, training.extension_samples,
                     training.visits_test_combination, training.numbers):
        assert inspect.signature(function).parameters["asset_id"].default is inspect.Parameter.empty, function.__name__
    assert inspect.signature(training.combinations_played).parameters["components"].default is inspect.Parameter.empty


class _Rigs:
    """A module constant of the rig, put where a run for another asset must not look: any use of it fails."""

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def _fail(self, *args, **kwargs):
        raise AssertionError(f"a run for another asset read {object.__getattribute__(self, '_name')}, a module constant of the rig")

    __getattr__ = __getitem__ = __iter__ = __len__ = __contains__ = __call__ = __bool__ = __int__ = __float__ = __index__ = _fail
    __str__ = __format__ = __hash__ = __eq__ = __lt__ = __gt__ = __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _fail
    __truediv__ = __rtruediv__ = __fspath__ = _fail

    def __repr__(self):
        return f"<the rig's {object.__getattribute__(self, '_name')}>"


RIG_CONSTANTS = ("_DEFAULT", "CONFIG", "CONFIG_FILE", "FOLDS_FILE", "FOLD_COMPONENTS", "FOLDS", "FAULT_FOLDS", "TRAINING_FOLDS")


def test_a_run_for_another_asset_reads_no_module_constant_of_the_rig(tmp_path, monkeypatch):
    """The tiny asset of the asset layer gets training files of its own, with numbers that differ from the rig's. Every module
    constant of training_variants/ that holds a value of the rig, under every name that a module of the fork binds it to, is replaced by
    an object that fails on any use. Then the paths of a run walk the tiny asset: the checks of a sampling line and of a trained
    line, their seeds, the names, the rounds and the gate, the data builder, the adapter record, the block record, the numbers of
    a loop with its stream, and the export of a loop. None of them fails, and every number they read is the tiny asset's."""
    import sys
    import types

    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online
    from CausalDiagnosis.utils.json_files import read_json, write_json
    from CausalDiagnosis.datasets.dataset_loader import recordings as records
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import load_profile, register_profile, unregister_profile
    from CausalDiagnosis.datasets.tests.conftest import TINY_FILE, write_recordings
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.training.tests.test_rejection_sampling_examples import _run
    from CausalDiagnosis.experiments.table_definitions import write_training_split as write_folds

    tiny = json.loads(TINY_FILE.read_text(encoding="utf-8"))["asset_id"]
    folder = tmp_path / "training_configs"
    (folder / tiny).mkdir(parents=True)
    (folder / "hyd_1").symlink_to(training.held_out_folds.TRAINING_DIR / "hyd_1")
    profile = register_profile(load_profile(str(TINY_FILE)))
    try:
        facts = write_recordings(tmp_path / "cycles.jsonl", profile)
        write_folds.main(["--asset", tiny, "--k", "2", "--facts", str(facts), "--out", str(folder / tiny / "folds.json")])
        own = {**json.loads(json.dumps(training.CONFIG)), "samples_per_state": 3, "sampling_temperature": 0.4,
               "control_seed": 11, "minimum": {"kept_per_training_state": 2, "extension_rounds": 2}, "folds": {"file": "folds.json"}}
        own["gepa"] = {**own["gepa"], "max_episodes": 7, "block": 2, "unanswered_proposals": 3, "proposal_attempts": 4}
        (folder / tiny / "rft.json").write_text(json.dumps(own), encoding="utf-8")
        cycles = records.load_cycles(str(facts), profile)
        blocks = records.build_blocks(cycles)
        config = profile.default_config()
        monkeypatch.setattr(training.held_out_folds, "TRAINING_DIR", folder)
        training.asset_training.cache_clear()
        # every name that a module of the fork binds to a constant of the rig fails from here on
        for module in [m for m in sys.modules.values() if isinstance(m, types.ModuleType)]:
            if not (getattr(module, "__file__", None) or "").startswith(str(ROOT)):
                continue
            if "/tests/" in module.__file__:
                continue
            for name in RIG_CONSTANTS:
                if name in vars(module) and vars(module)[name] is {**vars(training), "_DEFAULT": training.held_out_folds._DEFAULT}[name]:
                    monkeypatch.setattr(module, name, _Rigs(name))

        assert training.numbers(tiny) == (3, 2, 2, 0.4, 3, 4)
        parser = run_episodes.build_parser()
        sampling = parser.parse_args(["--config", "static", "--agent", "causal", "--starts", "training", "--training-sample", "2",
                                      "--training-fold", "1"])
        trained = parser.parse_args(["--config", "static", "--agent", "causal", "--starts", "sampled", "--training-variant", "ft",
                                     "--training-fold", "1"])
        for line in (sampling, trained):
            run_episodes.training_checks(line, config)
        env = HydraulicRigEnv(cycles, blocks, config)
        split = training.asset_training(tiny).split
        states = set(start_states.training_start_states(env, split, 1))

        def state_of(seed):
            env.reset(seed)
            return start_states.start_state(env)

        sampled_seeds = run_episodes.planned_episodes.chosen_seeds(sampling, config, cycles, blocks)
        assert sampled_seeds and all(state_of(seed) in states for seed in sampled_seeds)
        assert run_episodes.planned_episodes.chosen_seeds(trained, config, cycles, blocks)
        assert training.adapter_name("causal", "ft", 2, tiny) == "rft_causal_f2" and training.block_name("causal", "ft", 1, tiny) == "gepa_ft_causal_f1"
        assert training.round_samples(2, tiny) == [4, 5, 6] and training.extension_samples(tiny) == [4, 5, 6, 7, 8, 9]
        assert training.gate_of(3, 3, 2, 1, tiny)["minimum"] == 4

        # the data builder reads the numbers of the asset of its sampling runs
        runs = []
        for sample in (1, 2, 3):
            run = _run(tmp_path / "sampling", sample, [True, True], agent="causal", logs={0: [{"event": "reset", "hidden": [1, 100, 1]}]})
            write_json(run / "run.json", {**read_json(run / "run.json"), "config": {"asset_id": tiny}})
            runs.append(run)

        def counted(episodes):
            return {"training_states": 2, "states": {(e["run"], e["episode"]): e["seed"] % 2 for e in episodes}}

        root = tmp_path / "training"
        manifest = rft_data.build(runs, root / "data" / training.data_name("causal", 1, tiny), check_starts=counted)
        assert manifest["asset_id"] == tiny and manifest["control_seed"] == 11 and manifest["gate"]["minimum"] == 4
        assert manifest["gate"]["kept_per_training_state"] == 2 and manifest["gate"]["rounds_sampled"] == 1 and manifest["gate"]["trained"]
        # the adapter record and the block record of the asset
        adapter = root / "adapters" / training.adapter_name("causal", "ft", 1, tiny)
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"stand-in weights")
        data_manifest = root / "data" / "causal_f1" / training.DATA_MANIFEST
        (adapter / training.ADAPTER_MANIFEST).write_text(json.dumps({
            "agent": "causal", "arm": "filtered", "fold": 1, "steps": 3, "counts": {"kept": 3},
            "data_manifest_sha256": training.sha256_file(data_manifest), training.FOLD_CONVENTION_KEY: training.FOLD_CONVENTION,
            "asset_id": tiny}), encoding="utf-8")
        record = training.adapter_of("causal", "ft", 1, tiny, root)
        assert record["asset_id"] == tiny and record["training"]["gate"]["minimum"] == 4
        # the numbers and the stream of a loop, and its export
        loop = gepa_online.with_registered_numbers(gepa_online.build_parser().parse_args(
            ["--model", "m", "--reflection-model", "r", "--reflection-base-url", "http://x", "--training-fold", "1"]), tiny)
        assert (loop.max_episodes, loop.block) == (7, 2)
        stream = start_states.training_stream(env, split, loop.max_episodes, 1)
        assert 0 < len(stream) <= 7 and all(state_of(seed) in states for seed in stream)
        finished = tmp_path / "loop"
        write_json(finished / "run.json", {"mode": gepa_online.MODE, "config": {"asset_id": tiny}, "training_fold": 1, "adapter": None,
                                           "gepa": {"card": "verifier", "agent": "causal", "training_variant": "as_released",
                                                    "training_fold": 1, training.FOLD_CONVENTION_KEY: training.FOLD_CONVENTION}})
        write_json(finished / gepa_online.RESULT, {"status": "finished", "final_candidate": "c001", "episodes_played": 1,
                                                   "episodes": [{"call": 1, "episode": "e1", "candidate": "c001", "phase": "stream",
                                                                 "left_out": False}], "settings": {}, "proposals": 1,
                                                   "proposals_answered": 1})
        (finished / "candidates").mkdir()
        (finished / "candidates" / f"c001_{gepa_online.BLOCK}.txt").write_text("Read the belt first.", encoding="utf-8")
        target = gepa_online.main(["export", str(finished), "--training-root", str(root)])
        assert read_json(target.with_suffix(".json"))["asset_id"] == tiny
        assert training.block_of("causal", "gepa", 1, tiny, root)[0] == "Read the belt first."
    finally:
        unregister_profile(tiny)
        training.asset_training.cache_clear()


@needs_data
def test_rft_json_holds_no_number_that_the_split_gives():
    """The number of folds and the training start states of each model follow from the split, which start_states computes, so
    rft.json holds neither, and the K of the folds file gives the folds."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    rig = training.asset_training()
    record = json.loads(rig.folds_file.read_text(encoding="utf-8"))
    assert rig.config["folds"] == {"file": "folds.json", "development_file": "folds_development.json"} and record["k"] == len(rig.fold_ids)
    _, cycles, blocks = load_world()
    env = HydraulicRigEnv(cycles, blocks, load_config("base")[1])
    assert all(start_states.training_start_states(env, rig.split, h) for h in rig.fold_ids)  # every model has training start states


def test_a_made_up_asset_with_three_folds_reads_its_own_folds_file(made_up):
    assert training.held_out_folds.fold_ids(made_up) == (1, 2, 3) and training.held_out_folds.fold_ids() == (1, 2)
    assert training.fold_of([0, 0, 1], made_up) == 3
    training.check_fold(3, made_up)
    with pytest.raises(SystemExit, match="names the folds"):
        training.check_fold(4, made_up)
    with pytest.raises(SystemExit, match="names the folds"):
        training.check_fold(3, training.DEFAULT_ASSET)
    # the model of fold 2 holds out the bearing fault, so an episode that reaches it counts and one that stays on the motor does not
    reaches = [{"event": "reset", "hidden": [1, 0, 0]}, {"event": "hazard_drop", "component": "bearing", "to_level": 1}]
    assert training.visits_test_combination(reaches, 2, made_up) and not training.visits_test_combination(reaches[:1], 2, made_up)


def test_the_training_entry_points_check_the_fold_against_the_asset_of_the_run(made_up, tmp_path):
    config = SimpleNamespace(asset_id=made_up, anchors=None)
    parser = run_episodes.build_parser()
    line = ["--config", "static", "--agent", "causal", "--starts", "sampled", "--training-variant", "ft"]
    run_episodes.training_checks(parser.parse_args([*line, "--training-fold", "3"]), config)
    with pytest.raises(SystemExit, match="no fold of the registered split of made_up_3"):
        run_episodes.training_checks(parser.parse_args([*line, "--training-fold", "4"]), config)
    # the sampling runs of rejection_sampling_examples.py name their asset in run.json, whose split decides the fold
    run = tmp_path / "sample"
    run.mkdir()
    meta = {"ended": "x", "mode": training.TRAINING_MODE, "agent": "causal", "config_name": "deterioration", "training_fold": 3,
            "training_sample": 1, "config": {"asset_id": made_up}}
    (run / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    assert rft_data.sampling_runs([run], 1)[1] == 3
    (run / "run.json").write_text(json.dumps({**meta, "training_fold": 4}), encoding="utf-8")
    with pytest.raises(SystemExit, match="names the folds"):
        rft_data.sampling_runs([run], 1)


def test_a_run_that_names_no_fold_reads_no_training_file():
    """The split is read only where a fold is named, so an asset without training files still plays its runs without training,
    and a fold named for it is refused with the name of the missing file."""
    config = SimpleNamespace(asset_id="asset_without_training_files", anchors=None)
    parser = run_episodes.build_parser()
    untrained = parser.parse_args(["--config", "static", "--agent", "static", "--seeds", "2"])
    run_episodes.training_checks(untrained, config)
    assert run_episodes.planned_episodes.chosen_seeds(untrained, config, None, None) == [0, 1]
    trained = parser.parse_args(["--config", "static", "--agent", "causal", "--starts", "sampled", "--training-variant", "ft",
                                 "--training-fold", "1"])
    with pytest.raises(SystemExit, match="asset_without_training_files has no training configuration"):
        run_episodes.training_checks(trained, config)


def test_the_drivers_lay_out_one_column_group_per_fold_and_check_the_held_out_fold(made_up):
    """The drivers' code that reads folds on K = 3: the subsets of the training comparisons, the check that a trained run plays only
    the fold its model holds out, the alias that reads a fold without an adapter through the runs of no training, and the column groups of the table by fold."""
    assert aggregate.training_tables.training_pair_subsets(made_up)[-3:] == ("fold_1", "fold_2", "fold_3")

    def episode(seed, combination, fold=None, variant="as_released"):
        return {"run": f"r{fold}", "block": json.dumps(combination), "asset_id": made_up, "training_variant": variant, "training_fold": fold,
                "model": "m", "agent": "causal_m", "diagnostics_server": True, "seed": seed}

    played = [episode(0, [0, 0, 1], 3, "ft"), episode(1, [0, 1, 1], 3, "ft")]
    assert aggregate.cross_fitted(played) == played
    with pytest.raises(ValueError, match="held-out fold 2 and plays an episode of fold 3"):
        aggregate.cross_fitted([episode(0, [0, 0, 1], 2, "ft")])
    untrained = [episode(0, [1, 0, 0]), episode(1, [0, 1, 0]), episode(2, [0, 0, 1])]
    aliased = aggregate.aliased(untrained, {("m", "causal_m", True, "ft", 2): Path("marker")})
    assert [e["seed"] for e in aliased if e.get("aliased")] == [1]  # the model of fold 2 is tested on the start of fold 2 alone
    assert main_tables.fold_groups(3, len(training.held_out_folds.fold_ids(made_up))) == r"\cmidrule(lr){3-5}\cmidrule(l){6-8}"


@needs_data
def test_starts_development_plays_the_development_starts_and_no_trained_variant(tmp_path):
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    _, cycles, blocks = load_world()
    a = run_episodes.build_parser().parse_args(["--config", "base", "--agent", "static", "--starts", "development", "--seeds", "1"])
    _name, config, _prices = run_episodes.configure(a)
    run_episodes.training_checks(a, config)
    widened = run_episodes.development_config(config, blocks)
    development = development_split.development_starts(get_profile(config.asset_id), set(blocks))
    env = HydraulicRigEnv(cycles, blocks, widened)
    assert sorted(env.candidates) == sorted(development) and not set(development) & set(HydraulicRigEnv(cycles, blocks, config).candidates)
    seeds = run_episodes.planned_episodes.chosen_seeds(a, widened, cycles, blocks)
    assert seeds == start_states.combination_seeds(env, 1) and len(seeds) == len(development)
    trained = run_episodes.build_parser().parse_args(["--config", "base", "--agent", "causal", "--starts", "development",
                                                      "--training-variant", "ft", "--training-fold", "1"])
    anchored = run_episodes.build_parser().parse_args(["--config", "anchored", "--agent", "static", "--starts", "development"])
    for refused, its_config in ((trained, config), (anchored, run_episodes.configure(anchored)[1])):
        with pytest.raises(SystemExit, match="plays the development starts without training on a seeded preset"):
            run_episodes.training_checks(refused, its_config)
    grid = tmp_path / "grid.txt"
    grid.write_text("# no line\n", encoding="utf-8")
    pending_lines.main(["--grid", str(grid), "--runs", str(tmp_path), "--starts", "development"])  # a gate job that stopped is continued


@needs_data
def test_a_run_on_the_development_split_plays_development_combinations_only(tmp_path, monkeypatch):
    """--training-split development widens the preset to the development starts before any seed is chosen and reads the folds of
    the development split: a sampling line and a trained line of a trial start at development combinations alone, the training
    start states of the model of a fold and the starts of its test replicates do not meet, run.json records the split, the start
    check names it, and the tables read the fold of an episode from it."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.utils.json_files import read_json
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    facts, cycles, blocks = load_world()
    parser = run_episodes.build_parser()
    trial = ["--config", "static", "--agent", "causal", "--training-split", "development", "--training-fold", "1"]
    sampling = parser.parse_args([*trial, "--starts", "training"])
    trained = parser.parse_args([*trial, "--starts", "sampled", "--training-variant", "ft"])
    _name, config, _prices = run_episodes.configure(sampling)
    development = set(development_split.development_starts(get_profile(config.asset_id), set(blocks)))
    split = training.asset_training(config.asset_id, cycles.facts_sha256, split=training.DEVELOPMENT).split
    assert set(split) == development
    played = run_episodes.played_config(trained, config, blocks)
    assert {tuple(combination) for combination in played.allowed} == development
    env = HydraulicRigEnv(cycles, blocks, played)

    def states_of(seeds):
        found = set()
        for seed in seeds:
            env.reset(seed)
            found.add(start_states.start_state(env))
        return found

    starts = {}
    for line in (sampling, trained):
        run_episodes.training_checks(line, config)
        starts[line.starts] = states_of(run_episodes.planned_episodes.chosen_seeds(line, run_episodes.played_config(line, config, blocks), cycles, blocks))
        assert starts[line.starts] and {state[0] for state in starts[line.starts]} <= development
    # every replicate of the test episodes of fold 1 enters the combinations of fold 1, on which its model never starts to train
    tested_combinations = {combination for combination, (fold, _models) in split.items() if fold == 1}
    assert {state[0] for state in starts["sampled"]} == tested_combinations
    assert not {state[0] for state in starts["training"]} & tested_combinations and not starts["training"] & starts["sampled"]
    # a trained line of a trial on a fold whose data kept too few episodes is a marker run, which plays nothing and writes run.json
    root = tmp_path / "training"
    data = root / "data" / training.data_name("causal", 1, config.asset_id)
    data.mkdir(parents=True)
    gate = {"minimum": 9, "kept": 0, "second_round_due": False, "trained": False}
    (data / training.DATA_MANIFEST).write_text(json.dumps({"agent": "causal", "fold": 1, "gate": gate, "asset_id": config.asset_id,
                                                           training.FOLD_CONVENTION_KEY: training.FOLD_CONVENTION,
                                                           training.TRAINING_SPLIT_KEY: training.DEVELOPMENT}), encoding="utf-8")
    monkeypatch.setenv("HYD_TRAINING", str(root))
    monkeypatch.setenv("API_KEY", "stand-in-key")
    run_dir = run_episodes.main([*trial, "--starts", "sampled", "--training-variant", "ft", "--model", "granite", "--base-url",
                                 "http://127.0.0.1:9/v1", "--out", str(tmp_path / "runs")])
    meta = read_json(run_dir / "run.json")
    assert meta[training.TRAINING_SPLIT_KEY] == meta["adapter"][training.TRAINING_SPLIT_KEY] == training.DEVELOPMENT
    assert meta["stamp"] == "development" and {tuple(combination) for combination in meta["config"]["allowed"]} == development
    # the start check names the split, in a tree whose other rules all pass
    monkeypatch.setattr(run_episodes.start_check, "tree_reasons", lambda: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "data_reasons", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "open_reasons", lambda: [])
    oracle = ["--config", "static", "--agent", "oracle", "--starts", "sampled"]
    reason = "--training-split development plays the development split of a trial before the runs of record"
    assert run_episodes.start_check.registration(parser.parse_args([*oracle, "--training-split", "development"]), config, "static", facts)[-1] == reason
    assert reason not in run_episodes.start_check.registration(parser.parse_args(oracle), config, "static", facts)
    # the tables read the fold of an episode from the split of its run
    combination = sorted(development)[0]
    episode = {"block": json.dumps(list(combination)), "asset_id": config.asset_id, "training_split": training.DEVELOPMENT}
    assert aggregate.episode_fold(episode) == split[combination][0]
    with pytest.raises(SystemExit, match="is no start combination of the registered split"):
        aggregate.episode_fold({**episode, "training_split": training.REGISTERED})


def test_a_registered_run_refuses_an_adapter_of_the_development_split(tmp_path, monkeypatch):
    """The data manifest, the adapter manifest and the block record of a trial name the development split: adapter_of and
    block_of of the registered split stop on any of them, those of the development split stop on a registered record, and a
    record that names no split is registered. An adapter or a block built from stamped runs is a reason of the start check."""
    root, block, rig = tmp_path / "training", "Read the cheapest group first.", training.DEFAULT_ASSET
    trial = {training.TRAINING_SPLIT_KEY: training.DEVELOPMENT}
    for extras, source in (((trial, {}, {}), training.DATA_MANIFEST), (({}, trial, {}), training.ADAPTER_MANIFEST)):
        _training_records(root, (rig, rig, rig), block, extras)
        with pytest.raises(SystemExit, match=f"{source} belongs to the development split, and the run plays the registered split"):
            training.adapter_of("causal", "ft", 1, rig, root)
    _training_records(root, (rig, rig, rig), block, ({}, {}, trial))
    with pytest.raises(SystemExit, match="gepa_causal_f1.json belongs to the development split, and the run plays the registered split"):
        training.block_of("causal", "gepa", 1, rig, root)
    _training_records(root, (rig, rig, rig), block, (trial, trial, trial))
    assert training.adapter_of("causal", "ft", 1, rig, root, split=training.DEVELOPMENT)[training.TRAINING_SPLIT_KEY] == training.DEVELOPMENT
    assert training.block_of("causal", "gepa", 1, rig, root, split=training.DEVELOPMENT)[1][training.TRAINING_SPLIT_KEY] == training.DEVELOPMENT
    _training_records(root, (rig, rig, rig), block)
    assert training.adapter_of("causal", "ft", 1, rig, root)[training.TRAINING_SPLIT_KEY] == training.REGISTERED
    with pytest.raises(SystemExit, match="belongs to the registered split, and the run plays the development split"):
        training.adapter_of("causal", "ft", 1, rig, root, split=training.DEVELOPMENT)
    monkeypatch.setenv("HYD_TRAINING", str(root))
    config, parser = SimpleNamespace(asset_id=rig), run_episodes.build_parser()

    def line(variant):
        return parser.parse_args(["--config", "static", "--agent", "causal", "--starts", "sampled", "--training-variant", variant,
                                  "--training-fold", "1"])

    assert run_episodes.stamped_inputs(line("ft"), config) == run_episodes.stamped_inputs(line("gepa"), config) == []
    stamped = {"stamp": "development"}
    _training_records(root, (rig, rig, rig), block, ({}, stamped, stamped))
    assert run_episodes.stamped_inputs(line("ft"), config) == ["the adapter"]
    assert run_episodes.stamped_inputs(line("gepa"), config) == ["the block"]
