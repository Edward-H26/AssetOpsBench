"""pending_grid_lines.py: a line is done only by a finished, complete, matching run that started after its agent's cutoff."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS
import json

from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states

GRID = CLUSTER_JOBS / "grids" / "grid_agents.txt"
ONE = (1,)  # the first replicate alone, for the tests of the matching rules that the replicates leave as they were


def first(*lines):
    """The pairs of the first replicate of lines of the test set, as pending() returns them."""
    return [f"{line} --replicate 1" for line in lines]


def _run(root, name, line, started, rows=3, planned=3, ended=True, **change):
    folder = root / name
    folder.mkdir(parents=True)
    meta = {**pending_lines.expected(line), "started": started, "episodes_planned": planned, **change}
    if ended:
        meta["ended"] = started
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "scores.csv").write_text("seed\n" + "".join(f"{i}\n" for i in range(rows)), encoding="utf-8")


def test_no_two_run_lines_of_the_grid_share_every_field_a_run_is_matched_on():
    """A line that shares every field of expected() with another line is marked done by the other's run, and whoever continues the grid
    then skips a run that nobody played."""
    lines = pending_lines.grid_lines(GRID)
    keys = [json.dumps(pending_lines.expected(line), sort_keys=True) for line in lines]
    assert len(lines) > 5 and len(set(keys)) == len(lines)


def test_a_run_of_a_main_line_leaves_its_arm_pending_and_a_run_of_the_arm_leaves_the_main_line_pending(tmp_path):
    """The lines of the fixed cut of En5 carry the preset, the agent and so the label of a main line, so the fields of their switch alone
    keep one run from standing in for the other."""
    lines = pending_lines.grid_lines(GRID)
    for switch in ("--abstention-rule fixed_cut",):
        arms = [line for line in lines if switch in line]
        assert len(arms) == 3
        for arm in arms:
            main = arm.replace(f" {switch}", "")
            assert main in lines
            root = tmp_path / str(lines.index(arm))
            _run(root / "main", "a", main, "2026-11-02T10:40:00")
            _run(root / "arm", "b", arm, "2026-11-02T10:40:00")
            assert pending_lines.pending([main, arm], [root / "main"], {}, replicates=ONE) == first(arm)
            assert pending_lines.pending([main, arm], [root / "arm"], {}, replicates=ONE) == first(main)
            assert pending_lines.pending([main, arm], [root / "main", root / "arm"], {}, replicates=ONE) == []


def test_a_run_of_a_selector_counts_for_the_line_of_its_selection_alone(tmp_path):
    """A line of --start-state or of --scenario-ids carries its selection among the fields a run is matched on, in either spelling
    of the option, so a run of the line without the selection, or of another one, leaves the line pending, and a run of the line
    counts for it alone; a run.json written before the selectors reads as a run without one."""
    plain = "--config static --agent oracle"
    named, other = f"{plain} --start-state 3/100/0/130@2", f"{plain} --start-state=3/100/0/130@3"
    assert pending_lines.expected(named)["start_state"] == {"combination": [3, 100, 0, 130], "entry_offset": 2}
    assert pending_lines.expected(plain)["start_state"] is None and pending_lines.expected(plain)["scenario_ids"] is None
    _run(tmp_path / "plain", "a", plain, "2026-11-02T10:40:00")
    _run(tmp_path / "named", "b", named, "2026-11-02T10:40:00")
    lines = [plain, named, other]
    assert pending_lines.pending(lines, [tmp_path / "plain"], {}, replicates=ONE) == first(named, other)
    assert pending_lines.pending(lines, [tmp_path / "named"], {}, replicates=ONE) == first(plain, other)
    path = tmp_path / "plain" / "a" / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in ("start_state", "scenario_ids"):
        meta.pop(key)
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert pending_lines.pending(lines, [tmp_path / "plain"], {}, replicates=ONE) == first(named, other)
    anchored = "--config anchored --agent oracle"
    both, one = f"{anchored} --scenario-ids 2003/2001", f"{anchored} --scenario-ids=2001"
    assert pending_lines.expected(both)["scenario_ids"] == pending_lines.expected(f"{anchored} --scenario-ids=2001/2003")["scenario_ids"] == [2001, 2003]
    _run(tmp_path / "both", "c", both, "2026-11-02T10:40:00")
    assert pending_lines.pending([anchored, both, one], [tmp_path / "both"], {}, replicates=ONE) == first(anchored, one)


def test_a_run_under_another_cut_or_without_the_key_of_the_rule_is_matched_as_it_played(tmp_path):
    main, arm = "--config real_time --agent causal", "--config real_time --agent causal --abstention-rule fixed_cut"
    _run(tmp_path / "cut", "a", arm, "2026-11-02T10:40:00", abstention_cut=0.5)
    assert pending_lines.pending([arm], [tmp_path / "cut"], {}, replicates=ONE) == first(arm)
    _run(tmp_path / "old", "b", main, "2026-11-02T10:40:00")
    path = tmp_path / "old" / "b" / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.pop("abstention_rule")
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert pending_lines.pending([main, arm], [tmp_path / "old"], {}, replicates=ONE) == first(arm)


def test_only_finished_complete_matching_runs_after_the_cutoff_count(tmp_path):
    lines = ["--config static --agent plan_execute", "--config timed --agent plan_execute --no-diagnostics", "--config static --agent causal",
             "--config static --agent causal --causal-no-branches", "--config static --guidance 1"]
    _run(tmp_path, "a", lines[0], "2026-11-02T10:40:00")                      # done
    _run(tmp_path, "b", lines[1], "2026-11-02T09:00:00")                      # before the cutoff of plan_execute
    _run(tmp_path, "c", lines[2], "2026-11-02T10:50:00", rows=2)              # incomplete
    _run(tmp_path, "d", lines[3], "2026-11-02T10:50:00", ended=False)         # still running or cut short
    _run(tmp_path, "e", lines[4], "2026-11-02T10:50:00", guidance_level=2)    # another guidance level
    since = {"plan_execute": "2026-11-02T10:12"}
    assert pending_lines.pending(lines, [tmp_path], since, replicates=ONE) == first(*lines[1:])
    old = {**pending_lines.expected(lines[2])["causal_options"]}
    old.pop("parallel")
    _run(tmp_path, "f", lines[2], "2026-11-02T10:50:00", causal_options=old)  # a run of the code before the parallel switch carries no parallel option
    assert first(lines[2])[0] in pending_lines.pending(lines, [tmp_path], since, replicates=ONE)
    _run(tmp_path, "g", lines[2], "2026-11-02T10:55:00")
    assert first(lines[2])[0] not in pending_lines.pending(lines, [tmp_path], since, replicates=ONE)


def test_a_finished_run_of_the_test_set_is_done_by_default_and_the_starts_option_names_another_set(tmp_path, capsys):
    """The grid plays the test set, --starts sampled of run_grid.sbatch, so a finished run of a line with starts
    sampled is done under the default of this script, a run of the development set is done under --starts combinations alone, and
    the default here is the default of run_grid.sbatch, so that a job that continues a grid replays no finished line."""
    line = "--config real_time --servers aob --store couchdb --agent causal"
    assert pending_lines.expected(line)["starts"] == "sampled" == pending_lines.DEFAULT_STARTS
    assert "GRID_STARTS:-sampled" in (CLUSTER_JOBS / "3_run" / "run_grid.sbatch").read_text(encoding="utf-8")
    _run(tmp_path / "test_set", "a", line, "2026-11-02T10:40:00")
    assert pending_lines.pending([line], [tmp_path / "test_set"], {}, replicates=ONE) == []
    _run(tmp_path / "development", "b", line, "2026-11-02T10:40:00", starts="combinations")
    assert pending_lines.pending([line], [tmp_path / "development"], {}, replicates=ONE) == first(line)
    assert pending_lines.pending([line], [tmp_path / "development"], {}, starts="combinations") == []
    assert pending_lines.pending([line], [tmp_path / "test_set"], {}, starts="combinations") == [line]
    grid = tmp_path / "grid.txt"
    grid.write_text(f"# the line\n{line}\n", encoding="utf-8")
    pending_lines.main(["--grid", str(grid), "--runs", str(tmp_path / "test_set"), "--replicates", "1"])
    assert capsys.readouterr().out == ""
    pending_lines.main(["--grid", str(grid), "--runs", str(tmp_path / "test_set"), "--starts", "combinations"])
    assert capsys.readouterr().out == line + "\n"


def test_a_line_of_the_test_set_is_one_run_per_replicate_and_the_script_prints_the_missing_pairs_alone(tmp_path, capsys):
    """A line of the grid plays its five replicates as five runs of 47 episodes, so a job that ends
    before every replicate leaves whole runs, and the script prints the (line, replicate) pairs without a finished run, each as
    the line with its --replicate, in replicate order, the first replicate of every line first, and nothing else. A run of
    another replicate rule, master seed or seed pool stands for no pair, a line that names its replicate, as the script prints
    it, is checked for that replicate alone, and the development set has no replicate."""
    line, other = "--config real_time --servers aob --store couchdb --agent causal", "--config real_time --servers aob --store couchdb --agent plan_execute"
    assert pending_lines.REPLICATES == (1, 2, 3, 4, 5) == tuple(range(1, start_states.REPLICATES + 1))
    assert pending_lines.units([line]) == [f"{line} --replicate {r}" for r in pending_lines.REPLICATES]
    assert pending_lines.units([line, other], replicates=(1, 2)) == [f"{line} --replicate 1", f"{other} --replicate 1",
                                                                     f"{line} --replicate 2", f"{other} --replicate 2"]
    assert pending_lines.units([f"{line} --replicate 3", other], replicates=(1, 2)) == [f"{line} --replicate 3", f"{other} --replicate 1", f"{other} --replicate 2"]
    fields = pending_lines.expected(f"{line} --replicate 3")
    assert (fields["replicate"], fields["replicate_rule"], fields["master_seed"], fields["seed_pool"]) == (3, start_states.REPLICATE_RULE, 42, 8)
    assert pending_lines.expected(line)["replicate"] == 1
    for r in (1, 2, 4):
        _run(tmp_path / "grid", f"r{r}", f"{line} --replicate {r}", "2026-11-02T10:40:00", rows=47, planned=47)
    assert pending_lines.pending([line], [tmp_path / "grid"], {}) == [f"{line} --replicate 3", f"{line} --replicate 5"]
    assert pending_lines.pending([f"{line} --replicate 3"], [tmp_path / "grid"], {}) == [f"{line} --replicate 3"]
    assert pending_lines.pending([f"{line} --replicate 2"], [tmp_path / "grid"], {}) == []
    for name, change in (("rule", {"replicate_rule": "another rule"}), ("master", {"master_seed": 7}), ("pool", {"seed_pool": 10})):
        _run(tmp_path / name, "r3", f"{line} --replicate 3", "2026-11-02T10:40:00", rows=47, planned=47, **change)
        assert pending_lines.pending([f"{line} --replicate 3"], [tmp_path / name], {}) == [f"{line} --replicate 3"], name
    assert pending_lines.units([line], starts="combinations") == [line] and "replicate" not in pending_lines.expected(line, starts="combinations")
    grid = tmp_path / "grid.txt"
    grid.write_text(f"# the line\n{line}\n", encoding="utf-8")
    pending_lines.main(["--grid", str(grid), "--runs", str(tmp_path / "grid"), "--fraction", "1"])
    assert capsys.readouterr().out == f"{line} --replicate 3\n{line} --replicate 5\n"
    pending_lines.main(["--grid", str(grid), "--runs", str(tmp_path / "grid"), "--replicates", "1", "2"])
    assert capsys.readouterr().out == ""


def test_the_grid_runner_plays_the_replicates_that_pending_lines_counts_and_names_each_on_the_driver(tmp_path):
    """run_grid.sbatch, behind run_lines.sh, plays every line once per replicate that play_plan of this script gives for
    GRID_FRACTION or GRID_REPLICATES, and hands the replicate to the driver, so a job and the job that continues it count the same pairs."""
    hpc = CLUSTER_JOBS / "3_run"
    sbatch = (hpc / "run_grid.sbatch").read_text(encoding="utf-8")
    assert 'grid_job_lists share "${GRID_FRACTION:-}" "${GRID_REPLICATES:-}"' in sbatch and "--replicate $" in sbatch
    assert "1 2 3 4 5" not in sbatch.split("FRACTION_OPTION=\n", 1)[1].split("replicate_option()", 1)[0]
    assert "GRID_REPLICATES" in (hpc / "run_lines.sh").read_text(encoding="utf-8")


def test_a_line_of_the_wiring_check_plays_the_first_replicate_of_the_test_set_alone(tmp_path):
    """One run per preset proves the wiring, so a line of an agent of WIRING_AGENTS is one run on the test set, its first
    replicate, as run_grid.sbatch plays it, and is done once that run has finished; a line of another agent keeps every
    replicate, and on another start set every line is one run."""
    wiring, line = "--config base --agent scripted", "--config base --agent causal"
    assert pending_lines.WIRING_AGENTS == ("scripted",) and pending_lines.wiring_line(wiring) and not pending_lines.wiring_line(line)
    assert not pending_lines.wiring_line("--config base --agent") and not pending_lines.wiring_line("--config base")
    assert pending_lines.units([wiring, line], replicates=(1, 2)) == [f"{wiring} --replicate 1", f"{line} --replicate 1", f"{line} --replicate 2"]
    assert pending_lines.units([wiring], replicates=(2, 3)) == [] and pending_lines.units([wiring], starts="combinations") == [wiring]
    _run(tmp_path / "grid", "r1", f"{wiring} --replicate 1", "2026-11-02T10:40:00", rows=47, planned=47)
    assert pending_lines.pending([wiring], [tmp_path / "grid"], {}) == []


def test_a_run_of_another_model_surface_store_or_identity_leaves_the_line_pending(tmp_path, monkeypatch):
    """A run written since the identity existed counts for a line only on the model, the tool surface and the store of the line and
    with the identity of its agent in the tree that asks; a run written before it is compared as before."""
    line = pending_lines.grid_lines(GRID)[0]
    fields = pending_lines.expected(line)
    monkeypatch.setattr(pending_lines, "tree_identity", lambda agent: "a" * 64)
    servers = ["iot", "fmsr", "wo", "task"] if fields["surface"] == "aob" else ["hydraulic_rig"]
    backend = {"kind": fields["backend_kind"]} if fields["backend_kind"] else None
    base = {**{k: v for k, v in fields.items() if k not in pending_lines.IDENTITY_ERA_KEYS}, "model": fields["model"]}
    _run(tmp_path, "right", line, "2026-11-02T10:00:00", rows=47, planned=47, **{**base, "servers": servers, "backend": backend,
                                                                             "code": {"identity": "a" * 64}})
    assert pending_lines.pending([line], [tmp_path], {}, replicates=ONE) == []
    other_surface = ["hydraulic_rig"] if fields["surface"] == "aob" else ["iot", "fmsr", "wo", "task"]
    for name, change in (("model", {"model": "another-model"}), ("surface", {"servers": other_surface}), ("store", {"backend": {"kind": "other"}}),
                         ("identity", {"code": {"identity": "b" * 64}})):
        root = tmp_path / name
        root.mkdir()
        _run(root, name, line, "2026-11-02T10:00:00", rows=47, planned=47, **{**base, "servers": servers, "backend": backend,
                                                                           "code": {"identity": "a" * 64}, **change})
        assert pending_lines.pending([line], [root], {}, replicates=ONE) == first(line), name
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _run(legacy, "legacy", line, "2026-11-02T10:00:00", rows=47, planned=47, model="another-model")
    assert pending_lines.pending([line], [legacy], {}, replicates=ONE) == []  # no identity: the fields of its time alone


def test_a_finished_run_of_no_episode_counts_only_as_a_marker(tmp_path):
    line = pending_lines.grid_lines(GRID)[0]
    _run(tmp_path, "empty", line, "2026-11-02T10:00:00", rows=0, planned=0)
    assert list(pending_lines.finished_runs([tmp_path])) == []
    _run(tmp_path, "marker", line, "2026-11-02T10:00:00", rows=0, planned=0, adapter={"trained": False})
    assert [meta["adapter"] for meta, _ in pending_lines.finished_runs([tmp_path])] == [{"trained": False}]


def test_a_finished_run_of_the_served_model_counts_only_under_the_model_of_the_job(tmp_path, monkeypatch):
    """No grid line names a model: the job adds the model it serves after the line, so a finished run of a model agent records the
    served name, which differs from the default model of the driver. The pair is done under --model of that name alone, and a run in
    progress of the served model is taken up under it."""
    line = next(line for line in pending_lines.grid_lines(GRID) if pending_lines.expected(line)["model"])
    served = "served-model-of-the-job"
    assert pending_lines.expected(line)["model"] != served and pending_lines.expected(line, model=served)["model"] == served
    monkeypatch.setattr(pending_lines, "tree_identity", lambda agent: "a" * 64)
    fields = pending_lines.expected(line)
    servers = ["iot", "fmsr", "wo", "task"] if fields["surface"] == "aob" else ["hydraulic_rig"]
    backend = {"kind": fields["backend_kind"]} if fields["backend_kind"] else None
    _run(tmp_path, "done", line, "2026-11-02T10:00:00", rows=47, planned=47, model=served, servers=servers, backend=backend,
         code={"identity": "a" * 64})
    assert pending_lines.pending([line], [tmp_path], {}, replicates=ONE) == first(line)
    assert pending_lines.pending([line], [tmp_path], {}, replicates=ONE, model=served) == []
    assert pending_lines.expected("--config base --agent oracle", model=served)["model"] is None  # a reference policy has no model
