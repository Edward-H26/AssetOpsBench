"""The independent separability label (CausalDiagnosis/environment/reference_policies/diagnosability_label.py): the table is deterministic and keyed by
the rules, an observation model that reads every level labels every start separable and a blind one labels none, and the
gym and the verifier read the label. No test states a value computed from the rig data."""
import dataclasses
import json
import threading
import time

import pytest

from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports
from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT
from CausalDiagnosis.datasets.tests.conftest import tiny, tiny_data  # noqa: F401
from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
from CausalDiagnosis.environment.reference_policies.tier_classifiers import COMBOS, TIERS, signature
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, LEVELS, TIER_SENSORS, default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv, served_values
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import HydraulicRigGym
from CausalDiagnosis.environment.reference_policies.policies import OraclePolicy
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import EVALUATION_POOL, combination_seeds

ONE_FAULT = {"fault_rule": {"min_faulty": 1, "max_faulty": 1}}  # ten starts, so a table takes a second


class ObservationModel:
    """A stand in for ServedClassifiers: every tier predicts the levels of every component, either the true ones read off
    the served cycle (sharp) or always the first level (blind), with the matching likelihood rows."""

    def __init__(self, cycles, blocks, sharp):
        by_index = {c.index: c for c in cycles}
        self.sharp = sharp
        self.truth = {t: {} for t in TIERS}
        for combo, indices in blocks.items():
            levels = tuple(LEVELS[c].index(combo[j]) for j, c in enumerate(COMPONENTS))
            for i in indices:
                for t in TIERS:
                    self.truth[t][signature(served_values(by_index[i], TIER_SENSORS[t]))] = levels
        self.base = {t: [[self._row(c, h[j]) for h in COMBOS] for j, c in enumerate(COMPONENTS)] for t in TIERS}
        self.alpha = {t: [1e6] * len(COMPONENTS) for t in TIERS}

    def _row(self, component, level):
        k = len(LEVELS[component])
        if not self.sharp:
            return [1 / k] * k
        return [0.98 if i == LEVELS[component].index(level) else 0.02 / (k - 1) for i in range(k)]

    def predict(self, tier, values):
        return self.truth[tier][signature(values)] if self.sharp else (0,) * len(COMPONENTS)


def test_the_label_table_is_deterministic_and_keyed_by_the_rules(world, tmp_path):
    cycles, blocks = world
    config = EpisodeConfig(**ONE_FAULT)
    first = separability.ensure_labels(config, cycles, blocks, tmp_path / "a.json")
    assert first == separability.ensure_labels(config, cycles, blocks, tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    env = HydraulicRigEnv(cycles, blocks, config)
    assert set(first["labels"]) == {separability.combo_key(c) for c in env.candidates}
    for entry in first["labels"].values():
        assert len(entry["seeds"]) == separability.ROLLOUTS and entry["separable"] == (entry["share"] > 0.5)
    # a timed preset is labelled by its static counterpart, a new budget is a new table
    timed = EpisodeConfig(**ONE_FAULT, schedule={"mode": "seeded", "components": 1, "first_drop": [5, 30], "interval": 10})
    assert separability.ensure_labels(timed, cycles, blocks, tmp_path / "a.json") == first
    cheaper = separability.ensure_labels(EpisodeConfig(**ONE_FAULT, budget=150), cycles, blocks, tmp_path / "a.json")
    assert cheaper["rules_key"] != first["rules_key"] and cheaper["rules"]["budget"] == 150
    # the default fault rule draws other starts on other seeds, so its table is another one too
    assert separability.rules_key(separability.rules_of(EpisodeConfig())) != first["rules_key"]


def test_an_observation_model_that_reads_every_level_separates_every_start_and_a_blind_one_none(world):
    cycles, blocks = world
    config = EpisodeConfig(**ONE_FAULT)
    sharp = separability.compute_table(config, cycles, blocks, classifiers=ObservationModel(cycles, blocks, sharp=True))
    blind = separability.compute_table(config, cycles, blocks, classifiers=ObservationModel(cycles, blocks, sharp=False))
    assert all(e["separable"] and e["share"] == 1.0 for e in sharp["labels"].values())
    assert not any(e["separable"] or e["share"] for e in blind["labels"].values())


def test_the_static_configuration_of_an_asset_takes_the_defaults_of_that_asset(tiny):
    """The configuration a label is computed in takes the hidden processes off from the defaults of its own asset, so a
    configuration of another asset keeps the tables of that asset."""
    drawn = dataclasses.replace(tiny.default_config(), schedule={"mode": "seeded", "components": 1, "first_drop": [2, 4], "interval": 2})
    static = separability.static_config(drawn)
    assert (static.schedule, static.hazard, static.clock_mode) == (tiny.default_config().schedule, tiny.default_config().hazard, "declared")
    assert static.asset_id == "tiny_1" and static.components == list(tiny.components)


def test_the_gym_hands_the_independent_label_to_the_verifier():
    """By default the gym hands the verifier start_grade, the independent label with the server's E18 grade nested, so its
    reward follows the one credit rule of the drivers."""
    gym = HydraulicRigGym(EpisodeConfig(), default_facts_path())
    gym.reset(0)
    policy = OraclePolicy(gym.env)
    done, info = False, {}
    while not done:
        _text, _reward, terminated, truncated, info = gym.step(policy.act(gym.observation))
        done = terminated or truncated
    grade = separability.start_grade(gym.env)
    assert gym.label_of is ports.start_label and ports.declared("hyd_1", "LABELS") == "CausalDiagnosis.environment.reference_policies.diagnosability_label:start_grade"
    assert info["verdict"]["grade"] == grade and grade["source"] == "greedy reference"
    assert {k: v for k, v in grade.items() if k != "e18"} == separability.label_of(gym.env)
    assert info["verdict"]["separable"] is grade["separable"] and info["verdict"]["separable_e18"] is grade["e18"]["separable"]

class RecordingModel(ObservationModel):
    """The blind stand in that records every prediction it is asked for, in order."""

    def __init__(self, cycles, blocks):
        super().__init__(cycles, blocks, sharp=False)
        self.asked = []

    def predict(self, tier, values):
        self.asked.append((tier, signature(values)))
        return super().predict(tier, values)


def test_the_seeds_of_the_label_lie_after_the_evaluation_pool_of_every_start(world):
    """No rollout of the label replays an episode of the evaluation set or of the development set: the ten seeds of a start
    are the smallest after the eight of its evaluation pool."""
    cycles, blocks = world
    config = EpisodeConfig(**ONE_FAULT)
    table = separability.compute_table(config, cycles, blocks, classifiers=ObservationModel(cycles, blocks, sharp=False))
    env = HydraulicRigEnv(cycles, blocks, separability.static_config(config))
    pool = combination_seeds(env, EVALUATION_POOL)
    after = combination_seeds(env, separability.ROLLOUTS, after=EVALUATION_POOL)
    for i, start in enumerate(env.candidates):
        seeds = table["labels"][separability.combo_key(start)]["seeds"]
        assert seeds == after[i * separability.ROLLOUTS:(i + 1) * separability.ROLLOUTS]
        assert min(seeds) > max(pool[EVALUATION_POOL * i:EVALUATION_POOL * (i + 1)])


def test_every_rollout_of_the_label_reads_the_prehistory_of_its_seed_before_any_reading_of_its_own(world):
    """The AssetOpsBench rows record the five recordings of the start at the entry offset before cycle 0, so the reference
    policy of the label reads them first, every channel through the full group, as the rows serve them. The blind model
    never buys a reading, so the predictions it is asked for are the prehistory of every rollout and nothing else."""
    cycles, blocks = world
    config = EpisodeConfig(**ONE_FAULT)
    model = RecordingModel(cycles, blocks)
    table = separability.compute_table(config, cycles, blocks, classifiers=model)
    group = table["history"]["sensor_group"]
    env = HydraulicRigEnv(cycles, blocks, separability.static_config(config))
    assert table["history"]["recordings"] == SEGMENT and set(env.config.sensor_groups[group]) == set(env.profile.channels)
    tier = next(t for t in TIERS if set(TIER_SENSORS[t]) == set(env.config.sensor_groups[group]))
    expected = []
    for start in env.candidates:
        for seed in table["labels"][separability.combo_key(start)]["seeds"]:
            env.reset(seed)
            assert env.initial == tuple(start)
            expected += [(tier, signature(env.source.served_values(index, env.config.sensor_groups[group])))
                         for index in env.source.prehistory(env.initial)]
    assert model.asked == expected and len(expected) == SEGMENT * separability.ROLLOUTS * len(env.candidates)


def test_the_table_records_the_prehistory_its_rollouts_read_and_keeps_the_budget_of_the_rules(world):
    """The prehistory is read without charge, so the rules of the table are the static rules of the preset, and the table
    says what its rollouts read before their first action and which seeds they play."""
    cycles, blocks = world
    config = EpisodeConfig(**ONE_FAULT)
    static = separability.static_config(config)
    table = separability.compute_table(config, cycles, blocks, classifiers=ObservationModel(cycles, blocks, sharp=False))
    assert table["rules"] == separability.rules_of(static) and table["rules"]["budget"] == static.budget
    assert table["history"] == {"recordings": SEGMENT, "sensor_group": separability.history_group(static)}
    assert table["seeds_after"] == EVALUATION_POOL and table["rule"] == separability.RULE and "without charge" in table["rule"]


def test_the_label_table_is_the_file_labels_path_names_when_it_is_asked(tmp_path, monkeypatch):
    """ensure_labels and label_of read LABELS_PATH when they are called, not when the module is loaded, so that the tests run on
    a copy of the table (conftest.py of the repository) and never write the file a run in the same tree reads."""
    config = EpisodeConfig(**ONE_FAULT)
    key = separability.rules_key(separability.rules_of(separability.static_config(config)))
    path = tmp_path / "labels.json"
    table = {"rules_key": key, "code": separability.code_digest(config.asset_id), "labels": {}}
    path.write_text(json.dumps({"format": separability.LABELS_FORMAT, "facts_sha256": "a" * 64, "tables": {key: table}}))
    monkeypatch.setattr(separability, "LABELS_PATH", path)
    assert separability.ensure_labels(config, records.Recordings([], "a" * 64), None) == table
    with pytest.raises(records.DataFileError) as refused:
        separability.ensure_labels(config, records.Recordings([], "b" * 64), None)
    assert "a" * 64 in str(refused.value) and "b" * 64 in str(refused.value) and separability.RELABEL in str(refused.value)


def test_a_label_table_names_the_data_file_it_was_computed_from(world, tmp_path, monkeypatch):
    cycles, blocks = world
    config = EpisodeConfig(**ONE_FAULT)
    key = separability.rules_key(separability.rules_of(separability.static_config(config)))
    monkeypatch.setattr(separability, "compute_table", lambda config, cycles, blocks, **kwargs: {
        "rules_key": key, "code": separability.code_digest(config.asset_id), "labels": {}})
    path = tmp_path / "labels.json"
    separability.ensure_labels(config, cycles, blocks, path)
    assert json.loads(path.read_text(encoding="utf-8"))["facts_sha256"] == cycles.facts_sha256 is not None


def test_the_grade_of_a_start_is_refused_when_the_models_were_fitted_on_another_data_file(make_env, monkeypatch):
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import REFIT

    class Stale:
        raw = {"facts_sha256": "0" * 64}

    env = make_env(**ONE_FAULT)
    env.reset(0)
    monkeypatch.setattr(separability, "label_of", lambda env: {"separable": True, "share": 1.0})
    monkeypatch.setattr(adapter.fitted_models, "load_models", lambda *args: Stale())
    monkeypatch.setattr(grade, "ensure_grades", lambda *args, **kwargs: {"grades": {}})
    with pytest.raises(records.DataFileError) as refused:
        separability.start_grade(env)
    assert REFIT in str(refused.value) and "0" * 64 in str(refused.value)


def test_recordings_from_no_file_get_no_label_table(world, tmp_path):
    cycles, blocks = world
    path = tmp_path / "labels.json"
    with pytest.raises(records.DataFileError, match="loaded from a data file"):
        separability.ensure_labels(EpisodeConfig(**ONE_FAULT), records.Recordings(cycles, None), blocks, path)
    assert not path.exists()


def test_the_rules_key_carries_the_rule_of_the_label(monkeypatch):
    """A table computed under another rule, another prehistory or other seeds is never read, as the grade table's key holds its rule."""
    rules = separability.rules_of(EpisodeConfig())
    key = separability.rules_key(rules)
    monkeypatch.setattr(separability, "RULE", "another rule")
    assert separability.rules_key(rules) != key


def test_two_writers_of_the_label_file_at_once_keep_both_tables(world, tmp_path, monkeypatch):
    """Two processes of one clone that fill the file at once, as the lines of a grid did before the precompute step, must
    not drop each other's table: the read, the computation and the write run under a lock on the file."""
    cycles, blocks = world
    path = tmp_path / "labels.json"

    def slow(config, cycles, blocks, rollouts=separability.ROLLOUTS, classifiers=None):
        time.sleep(0.3)  # the window in which the other writer reads the file before this one writes it
        return {"rules_key": separability.rules_key(separability.rules_of(separability.static_config(config))), "labels": {}}

    monkeypatch.setattr(separability, "compute_table", slow)
    configs = [EpisodeConfig(**ONE_FAULT), EpisodeConfig(**ONE_FAULT, budget=150)]
    threads = [threading.Thread(target=separability.ensure_labels, args=(config, cycles, blocks, path)) for config in configs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(stored["tables"]) == {separability.rules_key(separability.rules_of(separability.static_config(c))) for c in configs}


def test_a_label_table_is_computed_again_when_the_code_that_wrote_it_differs(tmp_path, monkeypatch):
    """A stored table carries the digest of the code that wrote it: a table of another digest is computed again, and a table of
    the current digest is read without a computation."""
    config = EpisodeConfig(**ONE_FAULT)
    key = separability.rules_key(separability.rules_of(separability.static_config(config)))
    computed = []

    def compute(config, cycles, blocks, **kwargs):
        computed.append(config)
        return {"rules_key": key, "code": separability.code_digest(config.asset_id), "labels": {"fresh": True}}

    monkeypatch.setattr(separability, "compute_table", compute)
    path = tmp_path / "labels.json"
    stale = {"rules_key": key, "code": "another code", "labels": {}}
    path.write_text(json.dumps({"format": separability.LABELS_FORMAT, "facts_sha256": "a" * 64, "tables": {key: stale}}))
    recordings = records.Recordings([], "a" * 64)
    assert separability.ensure_labels(config, recordings, None, path)["labels"] == {"fresh": True} and len(computed) == 1
    assert separability.ensure_labels(config, recordings, None, path)["labels"] == {"fresh": True} and len(computed) == 1
    assert json.loads(path.read_text())["tables"][key]["code"] == separability.code_digest(config.asset_id)


def test_the_rules_key_reads_the_step_cap_the_failure_card_the_sensor_groups_and_the_asset():
    """Every field that a rollout reads keys its table: two configurations that differ in the step cap, the failure card, the
    sensor groups or the asset have different rules keys."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, register_profile, unregister_profile

    def key(config):
        return separability.rules_key(separability.rules_of(separability.static_config(config)))

    base = EpisodeConfig(**ONE_FAULT)
    groups = {**base.sensor_groups, "tier1": list(base.sensor_groups["tier1"][:1])}
    pairs = [EpisodeConfig(**ONE_FAULT, max_steps=base.max_steps + 1),
             EpisodeConfig(**ONE_FAULT, failure={**base.failure, "penalty": base.failure["penalty"] + 1}),
             EpisodeConfig(**ONE_FAULT, sensor_groups=groups)]
    assert len({key(base), *(key(c) for c in pairs)}) == 1 + len(pairs)
    twin = register_profile(dataclasses.replace(get_profile(base.asset_id), asset_id="hyd_twin_of_the_test"))
    try:
        other = EpisodeConfig.from_json({**base.to_json(), "asset_id": twin.asset_id})
        assert key(other) != key(base)
    finally:
        unregister_profile(twin.asset_id)


def test_the_code_digest_changes_with_any_engine_file_and_with_the_asset_file(tmp_path):
    """The digest of the code that writes a table reads every .py file of the engine package outside the tests folders and
    the asset's profile: a changed byte in an engine file or in the asset changes it, and a changed test file does not."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile, register_profile, unregister_profile

    def tree(name, engine="x = 1\n", test="assert True\n"):
        root = tmp_path / name
        (root / "pkg" / "tests").mkdir(parents=True)
        (root / "pkg" / "engine.py").write_text(engine)
        (root / "pkg" / "tests" / "test_engine.py").write_text(test)
        return root

    asset = EpisodeConfig().asset_id
    plain = separability.code_digest(asset, tree("plain"))
    assert separability.code_digest(asset, tree("engine", engine="x = 2\n")) != plain
    assert separability.code_digest(asset, tree("tests", test="assert 1\n")) == plain
    profile = get_profile(asset)
    register_profile(dataclasses.replace(profile, text={**profile.text, "noun": profile.text["noun"] + " of the test"}))
    try:
        assert separability.code_digest(asset, tmp_path / "plain") != plain
    finally:
        unregister_profile(asset)
    assert separability.code_digest(asset, tmp_path / "plain") == plain


def test_the_code_digest_of_an_asset_skips_the_folders_of_other_assets(tmp_path):
    """A port adds the folder of its asset, which holds an asset.json and writes no table of another asset: a .py file in the
    folder of another asset leaves the digest unchanged, while a change in the asset's own folder, in a folder of the assets
    package without an asset.json or in the engine changes it."""
    asset = EpisodeConfig().asset_id

    def tree(name, own="y = 1\n", other=None, shared="w = 1\n", engine="x = 1\n"):
        root = tmp_path / name
        for folder, text, marked in ((asset, own, True), ("made_up_1", other, True), ("shared", shared, False)):
            if text is not None:
                (root / "assets" / folder).mkdir(parents=True)
                (root / "assets" / folder / "records.py").write_text(text)
                if marked:
                    (root / "assets" / folder / "asset.json").write_text("{}")
        (root / "engine.py").write_text(engine)
        return root

    plain = separability.code_digest(asset, tree("plain"))
    assert separability.code_digest(asset, tree("port", other="z = 1\n")) == plain
    assert separability.code_digest(asset, tree("own", own="y = 2\n")) != plain
    assert separability.code_digest(asset, tree("shared", shared="w = 2\n")) != plain
    assert separability.code_digest(asset, tree("engine", engine="x = 2\n")) != plain


def test_first_diagnosis_finds_the_entry_when_an_event_of_its_cycle_follows_it(world):
    """In the instant mode a submission takes a cycle, and a scheduled drop in that cycle follows the entry of the diagnosis
    in the log. The reader of a rollout finds the entry by its event, so it returns the submitted levels."""
    cycles, blocks = world
    drop = 3
    schedule = {**EpisodeConfig().schedule, "mode": "seeded", "components": 1, "first_drop": [drop, drop]}
    env = HydraulicRigEnv(cycles, blocks, EpisodeConfig(clock_mode="instant", schedule=schedule))

    class SubmitAtTheDrop:
        def start(self, obs):
            self.submitted = None

        def act(self, obs):
            if env.time < drop:
                return {"type": "wait_for_notification", "timeout": 1}
            if self.submitted is None:
                self.submitted = tuple(env.hidden)
                return {"type": "submit_diagnosis", "diagnosis": dict(zip(env.config.components, env.hidden)), "evidence_ids": []}
            return {"type": "finish", "summary": "done"}

    policy = SubmitAtTheDrop()
    policy.start(None)
    got = separability.first_diagnosis(env, policy, 0)
    names = [e["event"] for e in env.log]
    assert names.index("submit_diagnosis") < names.index("scheduled_drop")  # the event of the cycle follows the entry
    assert got == policy.submitted


def test_the_label_table_of_the_control_holds_its_one_start(world):
    """The control preset has one start, the combination with every component at its healthy level, and its label table holds
    that start alone, labelled like any start by the rollouts of the module's observation model."""
    cycles, blocks = world
    config = EpisodeConfig(fault_rule={"min_faulty": 0, "max_faulty": 0})
    table = separability.compute_table(config, cycles, blocks, classifiers=ObservationModel(cycles, blocks, sharp=True))
    healthy = separability.combo_key(tuple(config.healthy[c] for c in config.components))
    assert list(table["labels"]) == [healthy] and table["labels"][healthy]["separable"] is True


def test_a_label_is_read_from_the_file_as_it_stands_now(world, tmp_path, monkeypatch):
    """label_of reads the label file as it stands: a label is read, the file is written again in place with another label
    under the same key, and the next call returns the new one, and a table that other code wrote is computed again, once,
    and then read from the file. A parse kept without the time and the size of the file fails."""
    cycles, blocks = world
    env = HydraulicRigEnv(cycles, blocks, EpisodeConfig(**ONE_FAULT))
    env.reset(0)
    path = tmp_path / "labels.json"
    key = separability._table_key(env.config)  # the one definition of the key, which ensure_labels and label_of share
    code = separability.code_digest(env.config.asset_id)

    def table(share, code=code):
        return {"rules_key": key, "rollouts": separability.ROLLOUTS, "code": code,
                "labels": {separability.combo_key(env.initial): {"separable": share > 0.5, "share": share, "seeds": []}}}

    def write(share, code=code):
        stored = {"format": separability.LABELS_FORMAT, "facts_sha256": env.facts_sha256, "tables": {key: table(share, code)}}
        path.write_text(json.dumps(stored), encoding="utf-8")

    write(0.6)
    assert separability.label_of(env, path)["share"] == 0.6
    write(0.55)
    assert separability.label_of(env, path)["share"] == 0.55
    computed = []
    monkeypatch.setattr(separability, "compute_table", lambda config, cycles, blocks, **kwargs: computed.append(config) or table(0.1))
    write(0.6, code="0" * 64)
    assert separability.label_of(env, path)["share"] == 0.1 and separability.label_of(env, path)["share"] == 0.1 and len(computed) == 1


def zero_durations():
    """The durations of the default configuration with every duration that makes a clock run at zero."""
    durations = EpisodeConfig().durations
    return {**durations, "read_per_cycle": 0, "lead_time": {c: 0 for c in durations["lead_time"]},
            "repair_downtime": {c: 0 for c in durations["repair_downtime"]}}


def test_a_row_without_a_clock_is_labeled_on_the_prehistory_alone(world):
    """A row without a clock serves no live recording, so a rollout of the label reads the record before cycle 0 and takes the
    most probable combination of the reference after it, with no read of its own, and the table of such a row says that its
    rollouts play no live recording. The model that reads every level then labels every start separable from the record."""
    from CausalDiagnosis.environment.reference_policies.greedy_policy import GreedyPolicy

    cycles, blocks = world
    still = EpisodeConfig(**ONE_FAULT, durations=zero_durations())
    assert not still.has_clock() and EpisodeConfig(**ONE_FAULT).has_clock()
    static = separability.static_config(still)
    env = HydraulicRigEnv(cycles, blocks, static)
    model = ObservationModel(cycles, blocks, sharp=True)
    seed = combination_seeds(env, 1, after=EVALUATION_POOL)[0]
    got = separability.first_diagnosis(env, GreedyPolicy(model), seed, separability.history_group(static), live=False)
    assert got == tuple(env.initial) and not any(entry["event"] == "read" for entry in env.log)
    table = separability.compute_table(still, cycles, blocks, classifiers=model)
    assert table["live"] is False and all(entry["separable"] for entry in table["labels"].values())
    assert separability.compute_table(EpisodeConfig(**ONE_FAULT), cycles, blocks, classifiers=model)["live"] is True


def test_the_label_key_tells_a_row_with_a_clock_from_one_without():
    """Two rows that differ only in whether a clock runs, as the clock mode alone can make one run, serve different recordings,
    so their labels never share a table, though the static configuration of both declares the clock."""
    still = EpisodeConfig(durations=zero_durations())
    timed = EpisodeConfig(durations=zero_durations(), clock_mode="generation_time")
    assert not still.has_clock() and timed.has_clock() and not separability.static_config(timed).has_clock()
    assert separability.rules_key(separability.rules_of(still)) != separability.rules_key(separability.rules_of(timed))
    assert separability.LABELS_FORMAT == 6


def test_a_process_that_only_reads_the_tables_never_computes_a_label_table(tmp_path, monkeypatch):
    """With CD_TABLES_READ_ONLY=1, the setting of every cluster job but the tables job and of a registered run, a label table that
    the file lacks stops the process by name and the file stays as it was, so no table changes under a run in flight."""
    from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES, GeneratedTableMissing

    config = EpisodeConfig(**ONE_FAULT)
    monkeypatch.setattr(separability, "compute_table", lambda *args, **kwargs: pytest.fail("a read-only process computed a table"))
    monkeypatch.setenv(READ_ONLY_TABLES, "1")
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"format": separability.LABELS_FORMAT, "facts_sha256": "a" * 64, "tables": {}}))
    before = path.read_bytes()
    with pytest.raises(GeneratedTableMissing, match="precompute_tables.sbatch"):
        separability.ensure_labels(config, records.Recordings([], "a" * 64), None, path)
    assert path.read_bytes() == before and separability.current_label_table(config, "a" * 64, path) is None
