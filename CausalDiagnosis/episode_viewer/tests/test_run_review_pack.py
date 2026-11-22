"""run_review_pack.py: the choice of the fixed starts, fixed_starts.json against what the code picks, the page check against a
ledger, and the pack of one scripted run with both walkthrough pages, its ledger and its line of VERIFY.md, written only
outside the fork. Run and pack folders go under pytest's tmp_path."""
import json
import re
import shutil
from pathlib import Path

import pytest

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.experiments.cluster_runs import grid_files
from CausalDiagnosis.episode_viewer import run_review_pack as review_pack
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results
from CausalDiagnosis.episode_viewer import walkthrough
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _run
from CausalDiagnosis.evaluation.result_tables.tests.test_leaderboard import DEVELOPMENT, _main_row, _runs
from CausalDiagnosis.evaluation.result_tables.tests.test_leaderboard import _page as _leaderboard_page
from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _driver, scratch_repository


def _page(rows):
    return "\n".join(["# Walkthrough", "", review_pack.FILES_HEADER, "|---|---|", *[f"| `{path}` | {sha} |" for path, sha in rows], "", "## 1. Start", ""])


def test_pick_takes_the_smallest_evaluation_seed_of_each_class_with_its_development_partner():
    labels = {30: True, 12: None, 45: False, 21: True}
    assert review_pack.pick([30, 12, 45, 21], [3, 1, 4, 2], labels.get) == {
        "separable": {"evaluation_seed": 21, "development_seed": 2}, "inseparable": {"evaluation_seed": 45, "development_seed": 4}}
    assert review_pack.pick([5], [0], lambda seed: True)["inseparable"] is None


def _static_fixed_starts(tmp_path):
    """The static preset and the fixed starts that the code picks for it on a copy of the label table, so that no table is
    written. Skips without the data file."""
    if not Path(default_facts_path()).exists():
        pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
    from CausalDiagnosis.environment.reference_policies.diagnosability_label import LABELS_PATH

    labels = tmp_path / "labels.json"
    if LABELS_PATH.exists():
        shutil.copyfile(LABELS_PATH, labels)
    _, cycles, blocks = review_pack.load_world()
    _, config = review_pack.load_config("static")
    return config, review_pack.fixed_starts_of(config, cycles, blocks, labels)


def test_fixed_starts_json_holds_what_the_code_picks_for_the_static_preset(tmp_path):
    config, picked = _static_fixed_starts(tmp_path)
    fixed = json.loads(review_pack.FIXED_STARTS.read_text(encoding="utf-8"))
    key = review_pack.rules_key_of(config)
    assert fixed["presets"]["static"] == key and "anchored" not in fixed["presets"]
    assert fixed["rules"][key] == picked


def test_fixed_starts_json_holds_every_preset_a_grid_file_plays_that_takes_fixed_starts():
    """Every preset that a line of a grid file plays names, in the tracked fixed_starts.json, a rules key that the file holds, so
    the review pack of any run of the grid finds its walkthrough starts. A preset that anchors its own scenarios or that the paper
    reports apart (OWN_TABLES of experiment_tables.py) takes no fixed starts and is absent from the file."""
    fixed = json.loads(review_pack.FIXED_STARTS.read_text(encoding="utf-8"))
    played = {entry.split(":")[0] for name in grid_files.FILES for entry in grid_files.presets(grid_files.HPC / name)}
    assert played
    for preset in sorted(played):
        name, config = review_pack.load_config(preset)
        if config.anchors is not None or name in design.OWN_TABLES:
            assert name not in fixed["presets"], name
        else:
            assert fixed["presets"].get(name) in fixed["rules"], name


def test_fixed_starts_json_names_the_data_file_it_was_built_from():
    """The tracked fixed_starts.json names, as the data file its starts were drawn on, the pinned release file of the asset."""
    from CausalDiagnosis.datasets.dataset_loader import recordings as records
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET

    profile = get_profile(DEFAULT_ASSET)
    fixed = json.loads(review_pack.FIXED_STARTS.read_text(encoding="utf-8"))
    assert fixed["facts_sha256"] == records.checksums(profile)[records.release_path(profile).name]


def test_fixed_starts_of_another_data_file_are_refused(tmp_path):
    """A fixed starts file that names another data file than the one in use, or none, is refused with the command that writes
    it again, and a missing file gives the empty record."""
    from CausalDiagnosis.datasets.dataset_loader import recordings as records

    fixed = json.loads(review_pack.FIXED_STARTS.read_text(encoding="utf-8"))
    unnamed = {key: value for key, value in fixed.items() if key != "facts_sha256"}
    for name, record in (("other.json", {**fixed, "facts_sha256": "0" * 64}), ("unnamed.json", unnamed)):
        path = tmp_path / name
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(records.DataFileError, match="python -m CausalDiagnosis.episode_viewer.run_review_pack starts"):
            review_pack.read_fixed_starts(path)
    assert review_pack.read_fixed_starts(tmp_path / "missing.json") == {"presets": {}, "rules": {}}


def test_every_rules_key_holds_the_inseparable_start_that_moved_with_the_first_replicate(tmp_path):
    """The first replicate of (20, 100, 0, 130) moved from seed 114, which replayed the start
    state of the episode of its reserved seed, seed 3, to seed 4 (MOVED_OFF_DEVELOPMENT of test_start_state_seeds.py), now the smallest
    inseparable seed of the evaluation set, so every rules key holds reserved seed 3 and evaluation seed 4 as its inseparable
    start, and as its separable one the pair that the code picks for the static preset on the label table, the same for every
    rules key since the greedy reference reads only the statistics a reading serves."""
    fixed = json.loads(review_pack.FIXED_STARTS.read_text(encoding="utf-8"))
    separable = next(iter(fixed["rules"].values()), {}).get("separable")
    assert fixed["rules"] and all(entry == {"inseparable": {"development_seed": 3, "evaluation_seed": 4},
                                            "separable": separable} for entry in fixed["rules"].values())
    assert separable == _static_fixed_starts(tmp_path)[1]["separable"]  # skipped here without the data file


def test_the_page_check_names_the_files_whose_hash_differs_from_the_ledger():
    record = {"episodes": {"seed000001": {"files": {"episode.json": {"sha256": "a" * 64}}}}, "run_files": {"run.json": {"sha256": "b" * 64}},
              "events": []}
    page = _page([("seed000001/episode.json", "a" * 64), ("run.json", "b" * 64)])
    assert review_pack.page_hashes(page) == {"seed000001/episode.json": "a" * 64, "run.json": "b" * 64}
    assert review_pack.page_check(page, record) == "All 2 files the page read have the SHA-256 the run's ledger states."
    assert review_pack.page_check(page.replace("b" * 64, "c" * 64), record) == "1 of the 2 files the page read differ from the run's ledger, run.json."
    assert review_pack.page_check("# Walkthrough\n", record).startswith("The page carries no table")


def test_a_folder_the_index_left_out_gets_its_reason_and_the_runs_it_read_keep_theirs():
    index = {"inputs": [{"run": "kept", "ledger_sha256": None}], "tables": {}, "leaderboard": None,
             "excluded": [{"folder": "left", "reason": "no run.json, so the run has not finished"}]}
    assert review_pack.appearances(index, Path("/runs/left")) == (["tables_from_runs.py left the run out, no run.json, so the run has not finished."], [])
    assert review_pack.appearances(index, Path("/runs/kept"))[0] == ["tables_from_runs.py read the run without a ledger, since the run predates the ledger."]


def test_the_pack_of_every_replicate_run_of_a_line_names_the_leaderboard_line_that_pools_it(tmp_path):
    """A line of the test set is five runs, one per replicate, that the leaderboard shows in one line,
    so the pack of each of the five runs names that line."""
    runs = _runs(tmp_path / "runs")
    for r in range(1, 6):
        seeds = list(range(10 * r, 10 * r + 4))
        _run(runs, f"causal_r{r}", [_main_row(s, agent="causal_m") for s in seeds],
             {**DEVELOPMENT, "starts": "sampled", "master_seed": 42, "seed_pool": 8, "seeds": seeds, "replicate": r, "replicate_rule": "rule",
              "episodes_planned": 4, "started": f"2026-11-02T1{r}:00:00"})
    _leaderboard_page(tmp_path, runs)
    index = json.loads((tmp_path / "INDEX.json").read_text(encoding="utf-8"))
    for r in range(1, 6):
        said, _rows = review_pack.appearances(index, runs / f"causal_r{r}")
        shown = [sentence for sentence in said if sentence.startswith("The leaderboard line ")]
        assert len(shown) == 1 and shown[0].startswith("The leaderboard line CausalDiagnosis agent + CausalDiagnosis MCP of m shows this run."), r


def test_the_pack_of_a_scripted_run_holds_both_pages_its_ledger_and_its_verify_line_outside_the_fork(tmp_path, monkeypatch, registered_root_numbers):
    from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES

    monkeypatch.setenv(READ_ONLY_TABLES, "0")  # recorded here, so that teardown removes what the run of record sets
    run_episodes = _driver()
    # a pack is made of a run of record alone: the start check is stubbed to pass, and the tree is read as an upload is, without
    # git, so that its changes since the last commit leave no run out; the files the run loads are asked of git (rule 3), which a
    # copy of the tree cannot answer, so a scratch repository holds the tree
    monkeypatch.setattr(run_episodes.start_check, "registration", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "git_state", lambda paths, root=None: (None, None))
    scratch_repository(monkeypatch, tmp_path)
    run_dir = run_episodes.main(["--config", "static", "--agent", "scripted", "--seeds", "2", "--out", str(tmp_path / "runs")])
    results = tmp_path / "results"
    aggregate.aggregate(tmp_path / "runs", results / "tables", resamples=50, departure=True, fraction="1")  # a few resamples, which INDEX.json states
    assert verify_results.main(["--runs", str(tmp_path / "runs"), "--results", str(results)]) == 0
    key = review_pack.rules_key_of(review_pack.load_config("static")[1])
    fixed = {"presets": {"static": key}, "rules": {key: {"separable": {"evaluation_seed": 900, "development_seed": 0},
                                                          "inseparable": {"evaluation_seed": 901, "development_seed": 1}}}}
    pack = review_pack.build(run_dir, results, tmp_path / "packs", fixed=fixed)
    index = (pack / "INDEX.md").read_text(encoding="utf-8")
    assert f"| `{run_dir.name}` | checked |" in index and "That is the INDEX.json this pack read." in index
    assert re.search(r"^Built on \d{4}-\d{2}-\d{2} by ", index, re.M)  # the date alone, no clock time
    assert "| Status | finished, started " in index and "| RUNS.csv | 1 | " in index and "--replay" in index
    hashed = hasattr(walkthrough, "files_read")  # the table of the files a page read, which walkthrough/ adds
    for name, seed in (("separable", 0), ("inseparable", 1)):
        line = next(line for line in index.splitlines() if line.startswith(f"- The {name} start is episode `seed{seed:06d}` in `walkthrough_{name}.md`."))
        assert line.endswith("have the SHA-256 the run's ledger states.") if hashed else "carries no table" in line
        assert (pack / f"walkthrough_{name}.md").read_text(encoding="utf-8").startswith(f"# Walkthrough of seed {seed}")
    if hashed:
        (run_dir / "seed000000" / "report.md").write_text("edited after the run\n", encoding="utf-8")
        index = (review_pack.build(run_dir, results, tmp_path / "packs_after", fixed=fixed) / "INDEX.md").read_text(encoding="utf-8")
        assert "differ from the run's ledger, seed000000/report.md." in index
    elsewhere = {"presets": {}, "rules": {key: {"separable": {"evaluation_seed": 905, "development_seed": 5}, "inseparable": None}}}
    index = (review_pack.build(run_dir, results, tmp_path / "packs_elsewhere", fixed=elsewhere) / "INDEX.md").read_text(encoding="utf-8")
    assert "- The separable start has no page, the run played neither the reserved seed 5 nor the evaluation seed 905." in index
    assert "- The inseparable start has no page, the rules have no such start." in index
    inside = ROOT / "review_packs_test"
    with pytest.raises(SystemExit, match="inside the fork"):
        review_pack.build(run_dir, results, inside, fixed=fixed)
    assert not inside.exists()
