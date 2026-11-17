"""precompute_grid_tables.py: the tables a grid reads are computed once before its lines start, into the files the drivers read,
for every rules key of the lines of the grid file and of the named presets, and a second call computes nothing again. The
label and the grade computations are stood in for, since the test checks which tables are asked for and not their values."""
import json
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables
from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

LINES = ["# the first two rows of a grid, one with the price table of the cost ablation",
         "--config real_time --servers aob --agent plan_execute", "",
         "--config real_time --prices full_tier_high --servers aob --agent causal"]


def test_a_line_that_names_its_replicate_resolves_to_the_configuration_of_its_line(tmp_path):
    """The job that continues a grid plays the pairs that pending_grid_lines.py prints, each a line with its --replicate,
    and run_grid.sbatch computes the tables of that file before its first run, so such a line resolves to the configuration of
    its line."""
    grid = tmp_path / "pending.txt"
    grid.write_text(f"{LINES[1]} --replicate 3\n{LINES[1]}\n", encoding="utf-8")
    (_, with_replicate), (_, bare) = precompute_tables.configurations([grid], [])
    assert with_replicate == bare


def test_the_tables_of_every_rules_key_of_a_grid_and_its_presets_are_computed_once(tmp_path, monkeypatch):
    grid = tmp_path / "grid.txt"
    grid.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    computed = []

    def label_table(config, cycles, blocks, rollouts=separability.ROLLOUTS, classifiers=None):
        rules = separability.rules_of(config)
        computed.append(("labels", separability.rules_key(rules)))
        # the code digest that compute_table records, which ensure_labels compares before it reads a stored table
        return {"rules_key": separability.rules_key(rules), "rules": rules, "labels": {}, "code": separability.code_digest(config.asset_id)}

    def graded(view, serving, start, budget, rng, rollouts, samples):
        computed.append(("grades", grade.combo_key(start)))
        return None

    monkeypatch.setattr(separability, "compute_table", label_table)
    monkeypatch.setattr(grade, "grade", graded)
    labels, grades = tmp_path / "labels.json", tmp_path / "grades.json"
    args = ["--grid", str(grid), "--config", "base", "--labels", str(labels), "--grades", str(grades)]
    report = precompute_tables.main(args)
    label_keys, grade_keys = set(json.loads(labels.read_text())["tables"]), json.loads(grades.read_text())["tables"]
    # real_time, real_time under the high price table, base and the asset defaults are four rules keys of either table
    assert len(label_keys) == 4 and sorted(report["labels"]) == sorted(label_keys)
    assert len(grade_keys) == 4 and sorted(report["grades"]) == sorted(grade_keys)
    assert all(len(table["grades"]) == len(grade.profile_view().combinations) for table in grade_keys.values())
    assert sum(1 for kind, _ in computed if kind == "labels") == 4 and sum(1 for kind, _ in computed if kind == "grades") == 4 * len(grade.profile_view().combinations)
    before = len(computed)
    again = precompute_tables.main(args)
    assert len(computed) == before and again == report


def test_the_asset_default_configuration_is_computed_with_no_grid_and_no_preset_named(tmp_path, monkeypatch):
    """The gym and the OpenEnv wrapper default to EpisodeConfig(), the defaults of the asset, so precompute_grid_tables.py computes its
    tables after those of the grid lines and the presets, here with none named, and an outside user never grades it on first
    use. The label and the grade computations are stood in for."""
    monkeypatch.setattr(separability, "compute_table", lambda config, cycles, blocks, rollouts=separability.ROLLOUTS, classifiers=None: {
        "rules_key": separability.rules_key(separability.rules_of(config)), "labels": {}})
    monkeypatch.setattr(grade, "grade", lambda view, serving, start, budget, rng, rollouts, samples: None)
    report = precompute_tables.main(["--labels", str(tmp_path / "labels.json"), "--grades", str(tmp_path / "grades.json")])
    assert report["labels"] == [separability.rules_key(separability.rules_of(EpisodeConfig()))]
    assert len(report["grades"]) == 1 and precompute_tables.configurations([], []) == []


def test_the_precompute_of_the_development_start_set_leaves_no_table_to_compute_at_first_use(tmp_path, monkeypatch):
    """Under --starts development every configuration is resolved to the development starts, as the driver resolves it, so on
    every preset that a table plays the label table of the rules of a development run exists before a line plays and its first
    episode computes nothing."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    presets = sorted(run_episodes.table_presets())
    grid = tmp_path / "grid.txt"
    grid.write_text("".join(f"--config {preset} --agent causal\n" for preset in presets), encoding="utf-8")
    keys = []

    def label_table(config, cycles, blocks, rollouts=separability.ROLLOUTS, classifiers=None):
        rules = separability.rules_of(config)
        keys.append(separability.rules_key(rules))
        return {"rules_key": separability.rules_key(rules), "rules": rules, "labels": {}}

    monkeypatch.setattr(separability, "compute_table", label_table)
    monkeypatch.setattr(grade, "grade", lambda *args, **kwargs: None)
    labels, grades = tmp_path / "labels.json", tmp_path / "grades.json"
    report = precompute_tables.main(["--grid", str(grid), "--starts", "development", "--labels", str(labels), "--grades", str(grades)])
    _facts, cycles, blocks = load_world()
    written = set(json.loads(labels.read_text())["tables"])
    for preset in presets:
        a = run_episodes.build_parser().parse_args(["--config", preset, "--agent", "causal", "--starts", "development"])
        _name, config, _prices = run_episodes.configure(a)
        key = separability.rules_key(separability.rules_of(run_episodes.development_config(config, blocks)))
        assert key in report["labels"] and key in written, preset
        plain = separability.rules_key(separability.rules_of(config))
        assert plain != key, preset  # the development rules are a table of their own, which the plain precompute would leave to the first episode


def test_the_precompute_builds_every_table_from_the_data_file_it_is_given(tmp_path, monkeypatch):
    """The models and the grade tables are built from the data file that --facts names, so that a table of another data file is
    refused and none is built from the default file behind the argument."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit

    facts = tmp_path / "recordings.jsonl"
    facts.write_bytes(Path(default_facts_path()).read_bytes())
    asked = {"models": [], "grades": []}
    monkeypatch.setattr(fit, "ensure_models", lambda path=fit.MODELS_PATH, asset_id=None, facts_path=None: asked["models"].append(facts_path))
    monkeypatch.setattr(grade, "ensure_grades", lambda models, rules, **keywords: asked["grades"].append(keywords.get("facts_path")) or {
        "rules_key": "key", "grades": {}})
    monkeypatch.setattr(separability, "compute_table", lambda config, cycles, blocks, rollouts=separability.ROLLOUTS, classifiers=None: {
        "rules_key": separability.rules_key(separability.rules_of(config)), "labels": {}})
    precompute_tables.main(["--facts", str(facts), "--config", "base", "--labels", str(tmp_path / "labels.json"), "--grades",
                            str(tmp_path / "grades.json")])
    assert asked["models"] == [facts] and asked["grades"] == [facts, facts]


def test_the_tables_of_a_launch_are_the_tables_of_the_runs_it_plays():
    """precompute_grid_tables.py resolves every line and every preset of a launch by the driver's own functions, with the start set
    and the run options of the launch (GRID_STARTS, the words of GRID_EXTRA), so the rules keys it computes are those of the
    runs: for every grid file of the tree, on the test starts, on the development starts and on capped development starts."""
    from CausalDiagnosis.experiments.cluster_runs import grid_files
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.experiments.cluster_runs.pending_grid_lines import grid_lines as lines_of

    _facts, _cycles, blocks = load_world()

    def key(words):
        a = run_episodes.build_parser().parse_args(words)
        return separability.rules_key(separability.rules_of(run_episodes.played_config(a, run_episodes.configure(a)[1], blocks)))

    grids = sorted(grid_files.HPC.glob("grid_*.txt"))
    assert grids
    for grid in grids:
        presets = grid_files.presets(grid)
        for starts, options in ((None, []), ("development", []), ("development", ["--start-cap", "8"])):
            tail = [*(["--starts", starts] if starts else []), *options]
            expected = {key([*line.split(), *([] if "--agent" in line.split() else ["--agent", "plan_execute"]), *tail]) for line in lines_of(grid)}
            for entry in presets:
                preset, _colon, mode = entry.partition(":")
                expected.add(key(["--config", preset, *(["--clock-mode", mode] if mode else []), "--agent", "plan_execute", *tail]))
            found = precompute_tables.configurations([grid], presets, starts, blocks, run_options=options)
            assert {separability.rules_key(separability.rules_of(config)) for _name, config in found} == expected, (grid.name, starts, options)


def test_a_selector_among_the_run_options_keeps_the_tables_of_the_launch(tmp_path):
    """A grid job checks its tables on its start set and on the development starts with the words of GRID_EXTRA. A start state
    among them resolves, with --starts seeds, to the preset whose field start names it, and on the development starts to those
    starts alone, and scenario ids change no configuration, so every rules key of the launch is the key it has without them."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    _facts, _cycles, blocks = load_world()
    grid = tmp_path / "grid.txt"
    grid.write_text(f"{LINES[1]}\n--config anchored --agent oracle\n", encoding="utf-8")
    selectors = ["--start-state=3/100/0/130@2", "--scenario-ids=2001/2003"]
    for starts in ("seeds", "development"):
        plain = precompute_tables.configurations([grid], ["base"], starts, blocks)
        selected = precompute_tables.configurations([grid], ["base"], starts, blocks, run_options=selectors)
        assert [separability.rules_key(separability.rules_of(c)) for _n, c in selected] == \
               [separability.rules_key(separability.rules_of(c)) for _n, c in plain], starts
        starts_fixed = [config.start for _name, config in selected]
        assert starts_fixed == ([{"combination": [3, 100, 0, 130], "entry_offset": 2}] * 3 if starts == "seeds" else [None] * 3), starts


def test_a_pinned_start_plays_on_the_start_set_of_its_line_and_the_development_tables_drop_it(tmp_path, monkeypatch, capsys):
    """A preset whose field start pins a start plays it on the start set of its line, and a configuration widened to the
    development starts no longer pins it. So the second tables call of a grid job, on the development starts, resolves a line whose
    preset pins a start of the test set, one whose preset pins a development start and one that pins none to one configuration,
    and both calls of the launch end with the tables ready. The label and the grade computations are stood in for."""
    from CausalDiagnosis.datasets.splits.development_split import development_starts, starts_of_test_set
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.utils.paths import preset_path

    _facts, _cycles, blocks = load_world()
    profile = run_episodes.get_profile("hyd_1")
    static = json.loads(preset_path("static").read_text(encoding="utf-8"))
    pins = {"test": starts_of_test_set(profile, blocks)[0], "development": development_starts(profile, set(blocks))[0]}
    lines = ["--config static --agent oracle"]
    for kind, combination in pins.items():
        path = tmp_path / f"pin_{kind}.json"
        # a fault rule that admits the starts of either set, so that a preset may pin a development start
        path.write_text(json.dumps({**static, "name": f"pin_{kind}", "fault_rule": {"min_faulty": 1, "max_faulty": 4},
                                    "start": {"combination": list(combination), "entry_offset": 0}}), encoding="utf-8")
        lines.append(f"--config {path} --servers aob --agent causal")
    grid = tmp_path / "grid.txt"
    grid.write_text("\n".join(lines) + "\n", encoding="utf-8")
    seeded = precompute_tables.configurations([grid], [], "seeds", blocks)
    assert [config.start for _name, config in seeded] == [None, *({"combination": list(c), "entry_offset": 0} for c in pins.values())]
    widened = precompute_tables.configurations([grid], [], "development", blocks)
    assert [config.start for _name, config in widened] == [None] * 3
    assert len({json.dumps(config.to_json(), sort_keys=True) for _name, config in widened}) == 1
    monkeypatch.setattr(separability, "compute_table", lambda config, cycles, blocks, rollouts=separability.ROLLOUTS, classifiers=None: {
        "rules_key": separability.rules_key(separability.rules_of(config)), "labels": {}, "code": separability.code_digest(config.asset_id)})
    monkeypatch.setattr(grade, "grade", lambda *args, **kwargs: None)
    files = ["--labels", str(tmp_path / "labels.json"), "--grades", str(tmp_path / "grades.json")]
    for starts in ("seeds", "development"):
        precompute_tables.main(["--grid-presets", "--grid", str(grid), "--starts", starts, *files])
    capsys.readouterr()
    for starts in ("seeds", "development"):
        precompute_tables.main(["--check", "--grid-presets", "--grid", str(grid), "--starts", starts, *files])
        assert "TABLES_READY" in capsys.readouterr().out, starts


def test_the_check_of_a_launch_writes_nothing_and_names_a_missing_table(tmp_path, monkeypatch, capsys):
    """--check reads the files and writes none: on the tables that the precompute of the launch wrote it prints TABLES_READY, and
    with one label table taken out it prints TABLES_MISSING with the line and exits 3, every file byte-equal."""
    grid = tmp_path / "grid.txt"
    grid.write_text(LINES[1] + "\n", encoding="utf-8")
    monkeypatch.setattr(separability, "compute_table", lambda config, cycles, blocks, rollouts=separability.ROLLOUTS, classifiers=None: {
        "rules_key": separability.rules_key(separability.rules_of(config)), "labels": {}, "code": separability.code_digest(config.asset_id)})
    monkeypatch.setattr(grade, "grade", lambda *args, **kwargs: None)
    labels, grades = tmp_path / "labels.json", tmp_path / "grades.json"
    args = ["--grid", str(grid), "--labels", str(labels), "--grades", str(grades)]
    precompute_tables.main(args)
    capsys.readouterr()
    precompute_tables.main([*args, "--check"])
    assert "TABLES_READY" in capsys.readouterr().out
    stored = json.loads(labels.read_text())
    stored["tables"].pop(next(iter(stored["tables"])))
    labels.write_text(json.dumps(stored))
    before = {path: path.read_bytes() for path in (labels, grades)}
    with pytest.raises(SystemExit) as stopped:
        precompute_tables.main([*args, "--check"])
    assert stopped.value.code == 3 and "TABLES_MISSING labels" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in (labels, grades)} == before
