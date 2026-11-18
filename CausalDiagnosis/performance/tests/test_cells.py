"""cells.py: every figure that a summary of a cell prints, each from made-up episodes in which the nearest wrong forms of its
formula give another number: the numerator set apart from the denominator set, a zero beside a missing value, a passed episode
beside a failed one, a joined call beside an unjoined one and a kept episode beside a left-out one; with the key that keeps the
cells of a comparison apart and the card-seconds held alone at one run in flight."""
import pytest

from CausalDiagnosis.performance import cells, engine_export, profile_table, stage_profile
from CausalDiagnosis.performance.tests import made_up
from CausalDiagnosis.performance.tests.summaries import assert_same


def kept(**fields):
    """A made-up row of a kept episode with every column of the table empty but the fields given."""
    return {**dict.fromkeys(profile_table.COLUMNS), "left_out": "", **fields}


def every_second(value):
    """The columns of every stage seconds of the profile set to one value, the cache flush left empty, as off the wall clock."""
    return {**dict.fromkeys(stage_profile.SECONDS, value), "cache_flush_s": None}


def test_every_stage_sum_share_and_spread_holds_the_episodes_that_hold_the_stage():
    """A holds ten seconds of every stage, B zero, C none, E two and a half: the sum and the spread of a stage take A, B and E, and
    its share divides by their totals alone; the total takes all four, and the cache flush, which no episode holds, has no sum."""
    rows = [kept(total_s=100.0, **every_second(10.0)), kept(total_s=50.0, **every_second(0.0)), kept(total_s=300.0, **every_second(None)),
            kept(total_s=20.0, **every_second(2.5))]
    stage = {"sum": 12.5, "share": 12.5 / 170.0, "median": 2.5, "p10": 0.5, "p90": 8.5, "n": 3}
    total = {"sum": 470.0, "share": 1.0, "median": 75.0, "p10": 29.0, "p90": 240.0, "n": 4}
    empty = {"sum": None, "share": None, "median": None, "p10": None, "p90": None, "n": 0}
    assert_same(cells.seconds_summary(rows), {"total_s": total, **{name: stage for name in stage_profile.SECONDS}, "cache_flush_s": empty})


def test_the_clock_sums_hold_the_episodes_that_hold_each_column_and_the_share_divides_the_cycles_by_their_horizons():
    columns = ("thinking_s", "thinking_cycles", "horizon_cycles", "credited_starts_s", "credited_calls_s", "uncredited_starts_s",
               "uncredited_calls_s", "failed_model_attempts", "uncharged_after_last_s")
    rows = [kept(**dict(zip(columns, values))) for values in ((2.0, 1, 10, 1.0, 0.5, None, 0.25, 1.0, 0.125),
                                                              (0.0, 0, 60, 0.0, 0.0, None, 0.0, 0.0, 0.0),
                                                              (4.0, 5, None, None, None, None, None, None, None),
                                                              (1.0, 2, 20, None, None, None, None, None, None))]
    assert_same(cells.clock_summary(rows), {"thinking_s": 7.0, "thinking_cycles": 8, "horizon_cycles": 90, "credited_starts_s": 1.0,
                                            "credited_calls_s": 0.5, "uncredited_starts_s": None, "uncredited_calls_s": 0.25,
                                            "failed_model_attempts": 1.0, "uncharged_after_last_s": 0.125,
                                            "thinking_share_of_horizon": 3 / 90})


def test_the_call_figures_hold_the_episodes_with_llm_calls_json_and_the_rate_divides_by_the_timed_calls():
    """A and B time their calls; C holds no llm_calls.json; E holds an empty one, a zero beside C's missing value; F holds one failed
    call, timed but without output tokens."""
    columns = ("model_calls", "failed_model_calls", "input_tokens", "output_tokens", "call_seconds_sum", "output_tokens_per_call_second")
    rows = [kept(**dict(zip(columns, values))) for values in ((3, 1, 600, 100, 6.5, 100 / 6.5), (2, 0, 100, 50, 5.0, 10.0),
                                                              (None, None, None, None, None, None), (0, 0, 0, 0, 0.0, None),
                                                              (1, 1, 0, 0, 2.0, None))]
    seconds = [3.0, 2.5, 1.0, 4.0, 1.0, 2.0]
    assert_same(cells.call_summary(rows, seconds), {"episodes": 4, "model_calls": 6, "failed_model_calls": 2, "input_tokens": 700,
                                                    "output_tokens": 150, "output_tokens_per_call_second": 150 / 11.5,
                                                    "call_seconds": {"median": 2.25, "p10": 1.0, "p90": 3.5, "n": 6}})
    empty = cells.call_summary(rows[2:3], [])
    assert (empty["episodes"], empty["model_calls"], empty["output_tokens"]) == (0, None, None)  # no episode holds the file


def engine_row(total, values, rate=None):
    return kept(total_s=total, decode_tokens_per_s=rate, **dict(zip((*engine_export.COUNTS, *engine_export.PARTS, "cached_prompt_tokens",
                                                                      "decode_tokens", "call_seconds_sum"), values)))


def test_the_engine_parts_of_a_cell_add_up_to_its_call_seconds_over_every_episode_that_the_join_reached():
    """A joins two calls and leaves one unjoined, B joins none of its two, the episode of a failed join that must stay in the cell; C
    lies outside the join; E holds no call; F joins two, one of them differing in its tokens. The parts and their shares divide by
    the seconds of the calls of the same episodes, never by their totals, and the decode rate is a ratio of sums."""
    rows = [engine_row(100.0, (2, 1, 0, 0, 1, 0.5, 1.0, 3.0, 1.0, 1.0, 0.0, 100, 300, 6.5), rate=100.0),
            engine_row(50.0, (0, 2, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0, 0, 5.0)),
            engine_row(300.0, (None,) * 14), engine_row(20.0, (0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)),
            engine_row(40.0, (2, 0, 0, 1, 0, 0.25, 0.5, 1.0, 0.25, 0.0, 1.0, 10, 50, 3.0), rate=50.0)]
    figures = [{"prefill_s": 1.0, "decode_s": 4.0, "input_match": 1, "output_match": 1},
               {"prefill_s": 3.0, "decode_s": 6.0, "input_match": 1, "output_match": 1},
               {"prefill_s": 99.0, "decode_s": 99.0, "input_match": 1, "output_match": 0}]
    parts = {"queue_s": 0.75, "prefill_s": 1.5, "decode_s": 4.0, "outside_stages_s": 1.25, "unjoined_calls_s": 6.0, "differing_calls_s": 1.0}
    found = cells.engine_summary(rows, figures)
    assert_same(found, {"episodes": 4, "engine_calls_joined": 4, "engine_calls_unjoined": 3, "engine_input_mismatches": 0,
                        "engine_output_mismatches": 1, "engine_extra_lines": 1, **parts, "cached_prompt_tokens": 110, "decode_tokens": 350,
                        "call_seconds": 14.5, **{share: parts[name] / 14.5 for name, share in zip(engine_export.PARTS, engine_export.CALL_SHARES)},
                        "decode_tokens_per_s": 87.5, "prefill_s_per_call": {"median": 2.0, "p10": 1.2, "p90": 2.8, "n": 2},
                        "decode_s_per_call": {"median": 5.0, "p10": 4.2, "p90": 5.8, "n": 2}})
    assert sum(found[share] for share in engine_export.CALL_SHARES) == pytest.approx(1.0)
    none = cells.engine_summary(rows[2:3], [])
    assert none["episodes"] == 0 and all(none[name] is None for name in (*engine_export.PARTS, *engine_export.CALL_SHARES, "decode_tokens_per_s"))


def test_the_gpu_busy_share_weighs_each_sampler_row_and_the_energy_divides_by_the_episodes_that_have_it():
    """E was read with no row inside its span, a zero beside C's missing value; F's pass is unknown; G's rows hold no utilization."""
    columns = ("sampler_rows", "gpu_busy_share", "gpu_energy_j", "pass")
    rows = [kept(**dict(zip(columns, values))) for values in ((10, 0.9, 1000.0, "True"), (30, 0.1, 2000.0, "False"),
                                                              (None, None, None, "True"), (0, None, None, None), (5, 0.5, 500.0, None),
                                                              (4, None, 400.0, "True"))]
    assert_same(cells.gpu_summary(rows), {"sampler_rows": 49, "gpu_busy_share": 14.5 / 45, "gpu_energy_j_per_episode": 3900.0 / 4,
                                          "gpu_energy_j_per_passed_episode": 1950.0})
    assert cells.gpu_summary(rows[2:3])["sampler_rows"] is None  # no episode was read


def test_the_pass_rate_and_the_card_seconds_held_alone_count_their_own_episodes():
    rows = [kept(held_card_s=100.0, **{"pass": "True"}), kept(held_card_s=50.0, **{"pass": "False"}),
            kept(held_card_s=None, **{"pass": "True"}), kept(held_card_s=20.0, **{"pass": None})]
    assert_same(cells.cost_summary(rows), {"pass_rate": 2 / 3, "held_card_s_per_episode": 170.0 / 3, "held_card_s_per_passed_episode": 170.0})


def view_of(grid, cell, elapsed, episodes, passed, window=None, ending=None, shares=None, undivided=False, without_model=()):
    """A made-up view of a job as the cell reads it."""
    return {"grid": grid, "cell": cell, "elapsed_s": elapsed, "model_episodes": episodes, "passed": passed, "window_s": window,
            "episodes_ending_inside": ending, "episode_shares_inside": shares, "card_s_undivided": undivided,
            "runs_without_model": list(without_model)}


def test_the_cost_and_the_rate_over_the_jobs_of_a_cell_divide_by_the_episodes_and_by_the_windows():
    jobs = [view_of("grid_1", 1, 1000.0, 10, 5, 400.0, 8, 7.5), view_of("grid_2", 1, 2000.0, 10, 0),
            view_of("grid_3", 2, 500.0, 1, 1, 100.0, 1, 1.0), view_of("grid_4", 1, None, 3, 3), view_of("grid_5", 1, 600.0, 4, 4, 200.0, 3, 2.5)]
    assert cells.job_summary(1, jobs) == ({"jobs_alone": ["grid_1", "grid_2", "grid_5"], "card_s_per_episode": 150.0,
                                           "card_s_per_passed_episode": 400.0, "window_s": 600.0,
                                           "ending_per_card_hour": 66.0, "shares_per_card_hour": 60.0}, [])
    assert cells.job_summary(3, jobs)[0]["window_s"] is None and cells.job_summary(3, jobs)[0]["jobs_alone"] == []


def test_a_job_that_leaves_its_card_seconds_undivided_leaves_those_of_its_cell_undivided_and_a_baseline_is_named():
    """grid_2 leaves its own card-seconds undivided, so the cell divides none and names it, while its rate still counts; in the
    control grid_2 divides its own, and the note names the runs without the served model whose time the Elapsed of grid_5 holds."""
    jobs = [view_of("grid_1", 1, 1000.0, 10, 5, 400.0, 8, 7.5), view_of("grid_2", 1, 2000.0, 10, 0, undivided=True),
            view_of("grid_5", 1, 600.0, 4, 4, 200.0, 3, 2.5, without_model=("run_o", "run_g"))]
    summary, notes = cells.job_summary(1, jobs)
    assert (summary["card_s_per_episode"], summary["card_s_per_passed_episode"], summary["window_s"]) == (None, None, 600.0)
    assert notes == ["the job grid_2 leaves its card-seconds undivided, so those of the cell are not divided"]
    jobs[1]["card_s_undivided"] = False
    summary, notes = cells.job_summary(1, jobs)
    assert (summary["card_s_per_episode"], summary["card_s_per_passed_episode"]) == (150.0, 400.0)  # the control
    assert notes == ["the card-seconds of the cell divide the Elapsed of the job grid_5, which holds the time of the runs without the "
                     "served model (run_o, run_g)"]


def test_the_counts_of_a_cell_keep_its_kept_episodes_apart_from_the_left_out_ones_and_its_checks_count_what_they_read():
    rows = [kept(total_s=100.0, lines_in_model_spans=0, rest_negative=1, late_server_starts=0),
            kept(total_s=50.0, lines_in_model_spans=0, rest_negative=0, late_server_starts=None),
            kept(total_s=300.0, lines_in_model_spans=None, rest_negative=None, late_server_starts=None),
            {**kept(total_s=1000.0), "left_out": stage_profile.SKEW}, {**kept(total_s=None), "left_out": stage_profile.TORN}]
    found = cells.cell_summary(1, {}, ["made_up_run"], [{**row, "grid": "grid_9"} for row in rows], [], [], [], ["made up"])
    assert {name: found[name] for name in ("cell", "runs", "jobs", "episodes", "kept", "left_out", "checks", "notes")} == {
        "cell": 1, "runs": ["made_up_run"], "jobs": ["grid_9"], "episodes": 5, "kept": 3,
        "left_out": {stage_profile.SKEW: 1, stage_profile.TORN: 1},
        "checks": {"lines_in_model_spans": {"failing": 0, "checked": 2}, "rest_negative": {"failing": 1, "checked": 2},
                   "late_server_starts": {"failing": 0, "checked": 1}}, "notes": ["made up"]}
    assert found["seconds"]["total_s"]["sum"] == 450.0  # the left-out episodes enter no sum


def test_two_runs_that_differ_in_their_asset_alone_play_two_cells(tmp_path):
    """The asset comes from each run's own record: two runs of two assets and of nothing else give two cells through the command,
    and the control, the same two runs with the asset taken out of the key, gives one."""
    one = made_up.write_run(tmp_path / "one", [(7, made_up.known_episode())])
    other = made_up.write_run(tmp_path / "other", [(7, made_up.known_episode())], meta=made_up.model_meta(asset_id="made_up_other_asset"))
    rows, summaries = profile_table.build([one, other])[:2]
    assert [row["cell"] for row in rows] == [1, 2] and [cell["key"]["asset"] for cell in summaries] == ["made_up_asset", "made_up_other_asset"]
    keys = [cells.cell_key(made_up.model_meta()), cells.cell_key(made_up.model_meta(asset_id="made_up_other_asset"))]
    without_asset = [{name: value for name, value in key.items() if name != "asset"} for key in keys]
    assert keys[0] != keys[1] and without_asset[0] == without_asset[1]  # the control: without the asset the two keys are one


def test_the_cell_key_keeps_apart_the_runs_in_flight_the_launch_flags_and_the_card():
    base = made_up.model_meta()
    assert cells.cell_key(base) == cells.cell_key(made_up.model_meta())
    other = []
    for change in ({"runs_in_flight": 8}, {"model": "made-up-other"}):
        other.append(made_up.model_meta(**change))
    for field, value in (("launch_args_sha256", "made_up_other"), ("gpu_name", "MADE_UP_OTHER_CARD"), ("dtype", "float8")):
        meta = made_up.model_meta()
        meta["serving"]["agent"][field] = value
        other.append(meta)
    assert all(cells.cell_key(meta) != cells.cell_key(base) for meta in other)
    meta = made_up.model_meta()
    meta["serving"]["agent"]["sampler"] = {"sha256": "made_up", "bytes": 1}
    assert cells.cell_key(meta) == cells.cell_key(base)  # the control: the sampler of a job is no part of the serving


def test_a_share_over_a_cell_is_a_ratio_of_sums_through_the_command(tmp_path):
    long = made_up.shifted(made_up.known_episode(), 100.0)
    long["row"]["wall_s"] = "36.5"  # twenty more seconds after the agent's part, all in its end
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode()), (8, long)])
    rows, summaries = profile_table.build([run])[:2]
    end = summaries[0]["seconds"]["end_s"]
    assert end["sum"] == pytest.approx(21.0) and end["share"] == pytest.approx(21.0 / 53.0)
    mean_of_ratios = sum(row["share_end"] for row in rows) / len(rows)
    assert mean_of_ratios == pytest.approx((0.5 / 16.5 + 20.5 / 36.5) / 2) and end["share"] != pytest.approx(mean_of_ratios)


def test_the_card_seconds_held_alone_count_the_runs_at_one_run_in_flight_only(tmp_path):
    alone = made_up.write_run(tmp_path / "alone", [(7, made_up.known_episode())])
    shared = made_up.write_run(tmp_path / "shared", [(7, made_up.known_episode())], meta=made_up.model_meta(runs_in_flight=8))
    rows, summaries = profile_table.build([alone, shared])[:2]
    assert [row["held_card_s"] for row in rows] == [16.5, None]
    assert summaries[0]["cost"]["held_card_s_per_episode"] == 16.5 and summaries[1]["cost"]["held_card_s_per_episode"] is None
