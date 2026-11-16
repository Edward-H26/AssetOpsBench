"""The table side of the fraction argument: a row of the test set is complete only with every replicate the
read asks for, the read takes the cells of one fraction alone, a read below the fraction of record runs no registered test, and a
cut look prints its estimates without spread. The episodes below are made up; no run of record is read."""
from pathlib import Path

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.evaluation.result_tables import pooling_rules
from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _row, _run

PLAN = {"counting": {"fraction_of_record": "1.0", "max_unscored_share": None}, "intervals": {"min_clusters": 30}}


def ctx_at(fraction):
    return {"registered": PLAN, "read": aggregate.read_of(fraction, PLAN)}


def run(name, replicate, n=17, planned=17, agent="causal_granite", fold=None, fraction=""):
    """The episodes of one finished or partly played run of a line of the test set."""
    return [{"run_dir": Path("runs") / name, "replicate": replicate, "episodes_planned": planned, "agent": agent, "mode": "gym", "pass": "True",
             "training_fold": fold, "fraction": fraction, "config": "deterioration", "asset_id": "hydraulic_rig"} for _ in range(n)]


def test_a_line_with_its_first_replicate_alone_reads_incomplete_at_the_whole_set():
    # the hole of today's tree: 17 of 17 played in the one run that exists read as complete
    counts = aggregate.completeness(run("r1", 1), ctx_at("1.0"))
    assert counts["incomplete"] is True and counts["replicates"] == "1" and counts["read"] == ""
    assert aggregate.row_status(run("r1", 1), ctx_at("1.0")) == "incomplete, 17 of 85, replicates 1"


def test_every_replicate_of_the_read_finished_is_complete_and_a_missing_one_is_not():
    rows = run("r1", 1) + run("r2", 2)
    assert aggregate.completeness(rows, ctx_at("0.4"))["incomplete"] is False
    # the control: replicate 2 removed before the read at 0.4
    assert aggregate.completeness(run("r1", 1), ctx_at("0.4"))["incomplete"] is True
    # a run of a replicate that stopped short is no finished run of its unit
    assert aggregate.completeness(run("r1", 1) + run("r2", 2, n=16), ctx_at("0.4"))["incomplete"] is True


def test_a_whole_read_takes_the_untagged_replicates_up_to_its_own_and_names_the_rest():
    rows = run("r1", 1) + run("r2", 2) + run("r3", 3) + run("cut", 1, n=1, planned=1, fraction=0.05)
    kept, left_out = aggregate.at_read(rows, aggregate.read_of("0.4", PLAN))
    assert {e["run_dir"].name for e in kept} == {"r1", "r2"}
    assert sorted(left_out) == ["cut", "r3"] and "0.4" in left_out["r3"]
    # an episode outside the replicates of the test set is read at every fraction
    outside = [{"run_dir": Path("runs") / "dev", "replicate": None, "fraction": ""}]
    assert aggregate.at_read(outside, aggregate.read_of("0.4", PLAN))[0] == outside


def test_a_cut_look_reads_its_tagged_runs_alone_and_prints_estimates_without_spread():
    rows = run("r1", 1) + run("cut", 1, n=1, planned=1, fraction=0.05) + run("other", 1, n=2, planned=2, fraction=0.1)
    kept, _ = aggregate.at_read(rows, aggregate.read_of("0.05", PLAN))
    assert {e["run_dir"].name for e in kept} == {"cut"}
    ctx = ctx_at("0.05")
    assert aggregate.too_few_clusters(1000, ctx) is True and aggregate.cluster_floor(ctx) > 10 ** 6
    # the read of record keeps the registered smallest number of clusters
    assert aggregate.too_few_clusters(1000, ctx_at("1.0")) is False and aggregate.cluster_floor(ctx_at("1.0")) == 30


def test_a_read_below_the_record_marks_partial_and_holds_no_test():
    ctx = ctx_at("0.4")
    rows = run("r1", 1) + run("r2", 2)
    assert aggregate.completeness(rows, ctx)["read"] == "partial 0.4"
    assert aggregate.row_status(rows, ctx) == "partial 0.4"
    row = {"pass_diff": "0.1", "pass_p": "0.01", "pass_holm_p": "0.02", "pass_reject": True, "family_tests": 14, "holds": True,
           "reading": "holds"}
    assert aggregate.untested(dict(row)) == {"pass_diff": "0.1", "pass_p": "", "pass_holm_p": "", "pass_reject": "", "family_tests": "",
                                             "holds": "", "reading": "not run"}
    assert ctx["read"]["partial"] is True and ctx_at("1.0")["read"]["partial"] is False


def test_a_trained_row_needs_a_run_per_fold_and_replicate_but_none_for_an_empty_fold(monkeypatch):
    monkeypatch.setattr(aggregate.training.held_out_folds, "fold_ids", lambda asset, split=None: [1, 2, 3])
    monkeypatch.setattr(aggregate.run_configuration, "fold_line_empty", lambda config, fold, fraction, master_seed=42, training_split=None: fold == 3)
    rows = run("f1r1", 1, n=6, planned=6, fold=1) + run("f2r1", 1, n=5, planned=5, fold=2)
    assert aggregate.completeness(rows, ctx_at("0.2"))["incomplete"] is False
    # fold 2 played at replicate 1 and not at replicate 2: its planned episodes come from its run of replicate 1
    rows += run("f1r2", 2, n=6, planned=6, fold=1)
    counts = aggregate.completeness(rows, ctx_at("0.4"))
    assert counts["incomplete"] is True and counts["planned"] == 22 and counts["replicates"] == "1-2"


def test_an_agent_of_the_wiring_check_is_complete_on_its_first_replicate():
    assert aggregate.completeness(run("w1", 1, agent="scripted"), ctx_at("1.0"))["incomplete"] is False


def test_a_direct_call_without_a_read_keeps_the_rule_of_the_planned_episodes():
    counts = aggregate.completeness(run("r1", 1), {"registered": PLAN})
    assert counts["incomplete"] is False and counts["replicates"] == "" and counts["read"] == ""


def _line_run(root, name, replicate, seeds, fraction=None):
    """A run folder of one line of the test set, its replicate and, for a cut look, its share, as run_episodes/ writes run.json."""
    meta = {"starts": "sampled", "master_seed": 42, "replicate": replicate, "seeds": seeds, "episodes_planned": len(seeds)}
    if fraction is not None:
        meta.update(fraction=fraction, fraction_rule="the rule")
    _run(root, name, [_row(s) for s in seeds], meta)


def test_load_episodes_carries_the_replicate_and_the_share_so_a_cut_run_and_a_whole_run_are_two_rows(tmp_path):
    # without the share on the episode a cut run and a whole run of one line would pool into one row
    _line_run(tmp_path, "whole", 1, [1, 2, 3])
    _line_run(tmp_path, "cut", 1, [1], fraction=0.05)
    episodes = aggregate.load_episodes(tmp_path)
    assert {(e["run"], e["replicate"], e["fraction"]) for e in episodes} == {("whole", 1, ""), ("cut", 1, 0.05)}
    assert len(aggregate.groups(episodes, *pooling_rules.ROW_FIELDS)) == 2


def test_a_replicate_whose_run_left_no_scored_row_is_expected_and_missing(tmp_path):
    # M3: the run of replicate 2 crashed before its first row; the read at 0.4 still expects it and counts its planned episodes
    _line_run(tmp_path, "r1", 1, [1, 2, 3])
    _run(tmp_path, "r2", [], {"starts": "sampled", "master_seed": 42, "replicate": 2, "seeds": [4, 5, 6], "episodes_planned": 3})
    counts = aggregate.completeness(aggregate.load_episodes(tmp_path), ctx_at("0.4"))
    assert (counts["incomplete"], counts["played"], counts["planned"], counts["replicates"]) == (True, 3, 6, "1")


def test_runs_off_the_test_set_are_read_at_every_share(tmp_path):
    # the quick start plays --seeds runs, which carry no replicate: every share, a cut look such as 0.05 or a whole one, reads them as they are
    _run(tmp_path, "seeds", [_row(s) for s in (1, 2, 3)], {"seeds": [1, 2, 3], "episodes_planned": 3})
    episodes = aggregate.load_episodes(tmp_path)
    for share in ("0.05", "0.4", "1"):
        kept, left_out = aggregate.at_read(episodes, aggregate.read_of(share, PLAN))
        assert len(kept) == 3 and left_out == {}
    assert aggregate.completeness(episodes, ctx_at("0.05"))["incomplete"] is False


def test_the_read_keeps_the_written_form_of_its_share():
    # INDEX.json records the share as fraction_text writes it, and a rebuild that passes it back reads the same
    for share, text in (("0.05", "0.05"), ("0.050", "0.05"), ("1", "1"), ("1.0", "1"), ("0.4", "0.4")):
        assert aggregate.read_of(share, PLAN)["text"] == text
        assert aggregate.read_of(aggregate.read_of(share, PLAN)["text"], PLAN) == aggregate.read_of(share, PLAN)
