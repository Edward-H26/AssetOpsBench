"""jobs.py: the windows of N runs in flight, toward which a run without the served model does not count, the two rates over them
on planted streams with the count of the episodes whole inside a window as the control, the window that a cut leaves open, the
runs that did not finish, the parts of a job's time from its stamps with the runs that play no model episode named in the notes,
its card-seconds per episode, and every note of a job asserted whole."""
import json
from pathlib import Path

import pytest

from CausalDiagnosis.performance import jobs, profile_table, run_files
from CausalDiagnosis.performance.tests import made_up
from CausalDiagnosis.performance.tests.summaries import assert_same

EPISODE = 16.5  # the made-up total of known_episode(), so that a stream that plays it back to back plays one episode per EPISODE


def streams(n, cycle, starts, episodes):
    """The process events of n made-up streams in the loop of the lines and their episode spans: stream k starts at starts[k] and
    plays episodes back to back, each one cycle long, so the planted rate of the job at n streams is n over the cycle."""
    events, spans = [], []
    for k in range(n):
        first = starts[k]
        spans += [(first + j * cycle, first + (j + 1) * cycle) for j in range(episodes)]
        events += [(first, "start", k), (first + episodes * cycle, "end", k)]
    ordered = sorted(events, key=lambda event: (event[0], event[1] == "start"))
    return [event_of(t, kind, f"line_{k}") for t, kind, k in ordered], spans


def event_of(t, kind, name, loop=jobs.LINES_LOOP):
    return {"epoch": t, "event": kind, "loop": loop, "kind": "run", "name": name, "pid": 1, "wall": "0"}


def whole_inside(spans, windows):
    return sum(1 for start, end in spans if any(a <= start and end <= b for a, b, _names, _cut in windows))


def test_the_windows_hold_exactly_n_processes_of_the_loop_of_the_lines():
    events = [event_of(t, kind, name, loop) for t, kind, loop, name in
              [(0, "start", "reference", "r"), (5, "start", "units", "a"), (10, "start", "units", "b"), (20, "end", "reference", "r"),
               (100, "end", "units", "a"), (100, "start", "units", "c"), (150, "end", "units", "b"), (200, "end", "units", "c")]]
    assert jobs.alive_windows(events, 2) == [(10, 100, {"a", "b"}, False), (100, 150, {"b", "c"}, False)]
    assert jobs.alive_windows(events, 1) == [(5, 10, {"a"}, False), (150, 200, {"c"}, False)]  # the reference loop is no process of the lines
    assert jobs.alive_windows(events[:3], 2) == [(10, None, {"a", "b"}, True)]  # open at the end of the record


def test_planted_streams_give_back_their_rate_and_the_count_whole_inside_is_the_control():
    """Three streams of a cycle of 100 seconds started 30 seconds apart: the shares inside the window give back the planted rate
    exactly, the episodes that end inside it come within one episode per stream, and the count of the episodes that lie whole
    inside it misses the planted shares of the four episodes that straddle its two edges."""
    n, cycle = 3, 100.0
    events, spans = streams(n, cycle, (0.0, 30.0, 60.0), 10)
    windows = jobs.alive_windows(events, n)
    ending, shares, seconds = jobs.window_rates(spans, windows)
    assert [(a, b) for a, b, _names, _cut in windows] == [(60.0, 1000.0)] and seconds == 940.0
    assert shares / seconds == pytest.approx(n / cycle, abs=1e-15)
    assert abs(ending - n / cycle * seconds) <= 2 and ending == 27  # within one episode per edge
    straddling = (100 - 60) / 100 + (130 - 60) / 100 + (1000 - 930) / 100 + (1000 - 960) / 100
    assert whole_inside(spans, windows) == pytest.approx(n / cycle * seconds - straddling) == 26


def test_the_shares_keep_the_speedup_of_n_against_one_that_the_count_whole_inside_understates():
    one_events, one_spans = streams(1, 100.0, (0.0,), 10)
    three_events, three_spans = streams(3, 100.0, (0.0, 30.0, 60.0), 10)
    one, three = jobs.alive_windows(one_events, 1), jobs.alive_windows(three_events, 3)
    rate = {name: jobs.window_rates(spans, windows) for name, spans, windows in (("one", one_spans, one), ("three", three_spans, three))}
    assert (rate["three"][1] / rate["three"][2]) / (rate["one"][1] / rate["one"][2]) == pytest.approx(3.0)
    understated = (whole_inside(three_spans, three) / 940.0) / (whole_inside(one_spans, one) / 1000.0)
    assert understated == pytest.approx(26 / 940 * 100) and understated < 2.8


def test_a_window_that_a_cut_leaves_open_closes_at_the_earliest_last_episode_end_of_its_runs():
    """Two streams cut by the end of the job write no end line: the open window closes at the earlier of their last episode ends,
    and the shares give back the planted rate exactly, since the episodes in flight at the cut hold no span, while the episode
    that sets the close ends inside the window. The control closes the same window at the end of the job, as the reading of the
    end lines alone would, and loses the share of both episodes in flight there."""
    events = [event_of(0.0, "start", "a"), event_of(30.0, "start", "b")]
    spans_a = [(0.0 + 100 * j, 100.0 + 100 * j) for j in range(9)]  # a's tenth episode, from 900 on, is cut in flight
    spans_b = [(30.0 + 100 * j, 130.0 + 100 * j) for j in range(9)]  # b's tenth episode, from 930 on, is cut in flight
    windows, notes = jobs.close_open(jobs.alive_windows(events, 2), {"a": 900.0, "b": 930.0})
    assert [(a, b) for a, b, _names, _cut in windows] == [(30.0, 900.0)]
    assert notes == ["the window open at the end of in_flight.tsv from 30.0 closes at 900.0, the earliest last episode end among its "
                     "runs, since the end of the job cut them"]
    ending, shares, seconds = jobs.window_rates(spans_a + spans_b, windows)
    assert shares / seconds == pytest.approx(2 / 100.0, abs=1e-15) and seconds == 870.0
    assert ending == 17 and jobs.window_rates(spans_a + spans_b, [(30.0, 900.0, frozenset(), False)])[0] == 16  # a's end at 900
    _ending, cut_shares, cut_seconds = jobs.window_rates(spans_a + spans_b, [(30.0, 1000.0, frozenset(), True)])
    assert cut_shares == pytest.approx(shares + 0.3) and cut_seconds == 970.0  # b's last 30 seconds; nothing of the two cut episodes
    assert cut_shares / cut_seconds < 2 / 100.0 - 0.001


def test_an_open_window_with_a_run_that_finished_no_episode_gives_no_rate():
    """The other edge: a run alive in the open window that finished no episode in a run folder leaves no end to close at, so the
    window stays out and the note names the process; the control is the same window with both ends known."""
    events = [event_of(0.0, "start", "a"), event_of(30.0, "start", "b")]
    windows, notes = jobs.close_open(jobs.alive_windows(events, 2), {"a": 900.0, "b": None})
    assert windows == [] and notes == ["the window open at the end of in_flight.tsv from 30.0 gives no rate, since b has no finished "
                                       "episode in a run folder"]
    assert jobs.close_open(jobs.alive_windows(events, 2), {"a": 900.0, "b": 930.0})[0] != []
    early, notes = jobs.close_open(jobs.alive_windows(events, 2), {"a": 20.0, "b": 930.0})  # a's last end lies before the window
    assert early == [] and notes == ["the window open at the end of in_flight.tsv from 30.0 gives no rate, since a run alive in it "
                                     "finished no episode inside it"]


def job_folder(tmp_path, log=None, record=None, cells=1, in_flight=True, skewed=False, mixed=False):
    """A made-up grid folder of one serving job with one run of two episodes at made-up stamps, and the view of its job; mixed adds
    a failed episode, a passed one that the profile leaves out and a folder without a row."""
    grid = made_up.write_grid(
        tmp_path / "grid_77", "77",
        record=["77|COMPLETED|2026-01-05T10:00:00|2026-01-05T11:00:00|01:00:00|00:00:00|node7"] if record is None else record,
        log=[made_up.runner_header(1500.0), "OPENING_CALL answered seconds 2.500 cycles made_up 0.250"] if log is None else log,
        in_flight=[(int(made_up.wall(1550.0)), "start", "units", "run", "line_0_replicate_none", 1, 0),
                   (int(made_up.wall(2030.0)), "end", "units", "run", "line_0_replicate_none", 1, 0)] if in_flight else None)
    second = made_up.shifted(made_up.known_episode(), 2000.0)
    if skewed:
        second["row"]["clock_skew"] = "1"
    episodes = [(7, made_up.shifted(made_up.known_episode(), 1560.0)), (8, second)]
    if mixed:
        second["row"]["pass"] = "False"
        skew = made_up.shifted(made_up.known_episode(), 1700.0)
        skew["row"]["clock_skew"] = "1"
        episodes.append((9, skew))
    made_up.write_run(grid / "run_a", episodes[:1] if cells == 2 else episodes, process=("0", "none"))
    if mixed:
        made_up.write_episode(grid / "run_a" / "seed000010", made_up.shifted(made_up.known_episode(), 2020.0))
    if cells == 2:
        made_up.write_run(grid / "run_b", episodes[1:], meta=made_up.model_meta(runs_in_flight=8), process=("1", "none"))
    return profile_table.build([grid / ("run_b" if cells == 2 else "run_a")])[2][0]


def test_the_parts_of_a_job_follow_its_stamps_and_add_up_to_its_elapsed(tmp_path):
    view = job_folder(tmp_path)
    found = {name: view[name] for name in jobs.PARTS}
    assert found == {"server_start_s": pytest.approx(300.0), "opening_s": pytest.approx(1200.0), "opening_check_s": pytest.approx(1197.5),
                     "opening_call_s": 2.5, "to_first_episode_s": pytest.approx(60.0), "episodes_span_s": pytest.approx(456.5),
                     "after_last_episode_s": pytest.approx(1583.5)}
    parts_of_elapsed = ("server_start_s", "opening_s", "to_first_episode_s", "episodes_span_s", "after_last_episode_s")
    assert sum(view[name] for name in parts_of_elapsed) == pytest.approx(view["elapsed_s"]) == 3600.0
    assert view["share_server_start"] == pytest.approx(300.0 / 3600.0) and view["notes"] == []


def test_a_part_without_its_stamp_stays_empty_and_the_note_says_why(tmp_path):
    view = job_folder(tmp_path, log=[made_up.runner_header(1500.0), made_up.runner_header(1700.0)])
    assert view["opening_s"] is None and view["to_first_episode_s"] is None and view["server_start_s"] == pytest.approx(300.0)
    assert view["share_server_start"] == pytest.approx(300.0 / 3600.0)  # over the Elapsed, not over the parts that it holds
    assert view["opening_call_s"] is None and view["notes"] == ["the Slurm log holds no single first line of the grid runner"]


def test_a_job_that_started_twice_reads_no_parts_and_still_its_card_seconds(tmp_path):
    record = ["77|REQUEUED|2026-01-05T09:00:00|2026-01-05T09:30:00|00:30:00|00:00:00|node7",
              "77|COMPLETED|2026-01-05T10:00:00|2026-01-05T11:00:00|01:00:00|00:00:00|node7"]
    view = job_folder(tmp_path, record=record)
    assert all(view[name] is None for name in ("server_start_s", "after_last_episode_s")) and view["elapsed_s"] == 5400.0
    assert view["card_s_per_episode"] == pytest.approx(2700.0) and view["notes"] == ["the job started 2 times, so its parts are not read"]


def test_a_grid_folder_without_a_job_record_and_without_in_flight_tsv_names_both(tmp_path):
    """The record of the job is empty and the record of the processes in flight is missing: the notes say both, whole; the control
    is the folder with both records, whose notes are empty."""
    assert job_folder(tmp_path / "whole")["notes"] == []
    view = job_folder(tmp_path / "bare", record=[], in_flight=False)
    assert view["elapsed_s"] is None and view["window_s"] is None and view["notes"] == [
        "the grid folder holds no job record; job_record.sh writes it after the job",
        "the rate needs in_flight.tsv and the runs in flight of run.json"]


def test_a_record_written_while_the_job_ran_is_not_read(tmp_path):
    view = job_folder(tmp_path, record=["77|RUNNING|2026-01-05T10:00:00|Unknown|00:10:00|00:00:00|node7"])
    assert view["elapsed_s"] is None and view["card_s_per_episode"] is None
    assert view["notes"] == ["the job record was written while the job ran; job_record.sh writes the final one after the job"]


def test_the_card_seconds_and_the_rate_of_a_job_of_one_cell(tmp_path):
    view = job_folder(tmp_path)
    assert (view["model_episodes"], view["passed"], view["card_s_per_episode"]) == (2, 2, pytest.approx(1800.0))
    assert view["window_s"] == 480 and view["episodes_ending_inside"] == 2 and view["episode_shares_inside"] == pytest.approx(2.0)
    assert view["ending_per_card_hour"] == pytest.approx(15.0) and view["shares_per_card_hour"] == pytest.approx(15.0)


def test_every_figure_of_the_view_of_a_job_with_a_failed_a_left_out_and_an_unscored_episode(tmp_path):
    """The run holds a passed and a failed kept episode, a passed episode that the profile leaves out and a folder without a row:
    the card-seconds per episode divide the Elapsed by the three scored episodes, those per passed episode by the two passed ones,
    the left-out one among them, and the rate reads the kept episodes inside the window."""
    view = job_folder(tmp_path, mixed=True)
    parts = {"server_start_s": 300.0, "opening_s": 1200.0, "opening_call_s": 2.5, "opening_check_s": 1197.5, "to_first_episode_s": 60.0,
             "episodes_span_s": 456.5, "after_last_episode_s": 1583.5}
    assert_same(view, {"grid": "grid_77", "model_runs": 1, "runs_without_model": [], "model_episodes": 3, "model_folders_without_row": 1,
                       "elapsed_s": 3600.0, "card_s_undivided": False,
                       **parts, **{f"share_{name[:-2]}": value / 3600.0 for name, value in parts.items()}, "cells_played": 1, "cell": 1,
                       "passed": 2, "card_s_per_episode": 1200.0, "card_s_per_passed_episode": 1800.0, "runs_in_flight": 1,
                       "window_s": 480, "episodes_ending_inside": 2, "episode_shares_inside": 2.0, "ending_per_s": 2 / 480,
                       "shares_per_s": 2 / 480, "ending_per_card_hour": 15.0, "shares_per_card_hour": 15.0, "rate_left_out": 1, "notes": []})


def test_the_note_of_a_blocked_process_counts_the_windows_that_it_takes_out():
    """Process p is alive in two windows and q in one: each note counts the windows of its own process; the control is that the two
    counts differ, and that the window without either stays."""
    windows = [(0, 10, frozenset({"p"}), False), (10, 20, frozenset({"p", "q"}), False), (20, 30, frozenset({"r"}), False)]
    kept, notes = jobs.unblocked(windows, {"p": "made up reason one", "q": "made up reason two"})
    assert kept == [(20, 30, frozenset({"r"}), False)]
    assert notes == ["no rate from the 2 windows in which the process p was alive, since made up reason one",
                     "no rate from the 1 windows in which the process q was alive, since made up reason two"]


def test_an_episode_ends_inside_a_window_from_its_start_on_and_before_its_end_or_at_the_end_that_a_cut_set():
    """Ends at the start of the first window, at the end of the first, which is the start of the second, at the end of the second,
    which a cut set, and after it: three ends inside, each once."""
    names = frozenset({"a"})
    spans = [(-10.0, 0.0), (90.0, 100.0), (190.0, 200.0), (240.0, 250.0)]
    assert jobs.window_rates(spans, [(0, 100, names, False), (100, 200, names, True)])[0] == 3


def test_the_card_seconds_count_every_scored_episode_the_ones_left_out_among_them(tmp_path):
    """One of the two model episodes is skewed and left out of the spans: the card-seconds per episode still divide the Elapsed by
    both, since both held the card, and the control over the kept episode alone differs."""
    view = job_folder(tmp_path, skewed=True)
    assert view["model_episodes"] == 2 and view["rate_left_out"] == 1
    assert view["card_s_per_episode"] == pytest.approx(3600.0 / 2) and view["card_s_per_episode"] != pytest.approx(3600.0 / 1)
    assert view["card_s_per_passed_episode"] == pytest.approx(3600.0 / 2)


def test_a_job_of_two_cells_gets_no_cost_and_no_rate(tmp_path):
    """Three model runs play two cells: the note counts the cells, which the control, the count of the model runs, does not give."""
    job_folder(tmp_path, cells=2)
    grid = tmp_path / "grid_77"
    made_up.write_run(grid / "run_c", [(9, made_up.shifted(made_up.known_episode(), 2500.0))], process=("2", "none"))
    view = profile_table.build([grid / "run_b"])[2][0]
    assert view["cell"] is None and view["cells_played"] == 2 and "card_s_per_episode" not in view and view["model_runs"] == 3
    assert view["notes"] == ["the model runs of the job play 2 cells, so it gets no cost and no rate"]


def test_a_job_without_a_model_run_says_so_and_a_job_whose_model_runs_record_no_whole_cell_says_that(tmp_path):
    grid = made_up.write_grid(tmp_path / "grid_99", "99", record=["99|COMPLETED|2026-01-05T10:00:00|2026-01-05T11:00:00|01:00:00|00:00:00|node7"])
    oracle = made_up.model_meta(agent="oracle", agent_label="oracle", model=None, serving={"agent": None, "judge": None})
    made_up.write_run(grid / "run_o", [(7, made_up.known_episode())], meta=oracle, process=("0", "none"))
    assert profile_table.build([grid / "run_o"])[2][0]["notes"] == [
        "no model run of the job has the started line of the endpoint file in its ledger, which places the local times of the job",
        "the job holds no run of a model agent, so it gets no cost and no rate"]
    lacking = cut_job(tmp_path / "lacking", clock_mode_of_c=False)
    for name in ("run_a", "run_b"):
        for state in (lacking / name).glob("seed*/episode.json"):
            parts = json.loads(state.read_text(encoding="utf-8"))
            del parts["config"]["clock_mode"]
            state.write_text(json.dumps(parts), encoding="utf-8")
    (lacking / "run_a" / "run.json").unlink()
    view = profile_table.build([lacking / "run_b"])[2][0]
    assert view["notes"][-1] == "no model run of the job records its cell whole, so it gets no cost and no rate"


def test_a_job_whose_record_never_holds_n_runs_alive_names_it(tmp_path):
    view = job_folder(tmp_path, in_flight=True)
    assert view["notes"] == [] and view["window_s"] == 480
    meta_path = tmp_path / "grid_77" / "run_a" / "run.json"
    meta_path.write_text(meta_path.read_text(encoding="utf-8").replace('"runs_in_flight": 1', '"runs_in_flight": 2'), encoding="utf-8")
    view = profile_table.build([tmp_path / "grid_77" / "run_a"])[2][0]
    assert view["window_s"] is None and view["notes"] == ["no window of in_flight.tsv with exactly 2 model runs alive in the loop units gives a rate"]


def cut_job(tmp_path, clock_mode_of_c=True):
    """A made-up job at two runs in flight with a refill and a cut: run_a (line 0) plays 4 episodes back to back from 100 and ends
    with run.json; run_b (line 1) plays 9 from 100 and run_c (line 2), the refill, 5 from the end of run_a, and the end of the job
    cuts both in their next episode, which leaves each a folder without a row and neither a run.json nor an end line."""
    start = made_up.wall(100.0)
    grid = made_up.write_grid(tmp_path / "grid_88", "88",
                              record=["88|CANCELLED by 1|2026-01-05T10:00:00|2026-01-05T10:05:00|00:05:00|00:00:00|node7"],
                              log=[made_up.runner_header(50.0)],
                              in_flight=[(int(start), "start", "units", "run", "line_0_replicate_none", 1, 0),
                                         (int(start), "start", "units", "run", "line_1_replicate_none", 2, 0),
                                         (int(start + 4 * EPISODE), "end", "units", "run", "line_0_replicate_none", 1, 0),
                                         (int(start + 4 * EPISODE), "start", "units", "run", "line_2_replicate_none", 3, 0)])
    meta = made_up.model_meta(runs_in_flight=2)

    def played(first, count):
        return [(7 + j, made_up.shifted(made_up.known_episode(), first + j * EPISODE)) for j in range(count)]
    made_up.write_run(grid / "run_a", played(100.0, 4), meta=meta, process=("0", "none"))
    made_up.write_run(grid / "run_b", played(100.0, 9), meta=meta, process=("1", "none"), unfinished=True)
    made_up.write_episode(grid / "run_b" / "seed000016", made_up.shifted(made_up.known_episode(), 100.0 + 9 * EPISODE))
    made_up.write_run(grid / "run_c", played(100.0 + 4 * EPISODE, 5), meta=meta, process=("2", "none"), unfinished=True)
    made_up.write_episode(grid / "run_c" / "seed000012", made_up.shifted(made_up.known_episode(), 100.0 + 9 * EPISODE))
    if not clock_mode_of_c:
        for state in (grid / "run_c").glob("seed*/episode.json"):
            parts = json.loads(state.read_text(encoding="utf-8"))
            del parts["config"]["clock_mode"]
            state.write_text(json.dumps(parts), encoding="utf-8")
    return grid


def test_a_cut_job_at_two_runs_in_flight_reads_its_cut_runs_and_closes_its_window_at_their_last_end(tmp_path):
    """The cut runs are named in the notes and read from their own folders, every scored episode of the three runs counts, the
    rows of the cut runs are marked, and the window open at the cut closes where run_b and run_c both played their last finished
    episode, so the shares give back the planted rate of two streams of one episode each 16.5 seconds. The control is the reading of
    the runs with run.json alone: one run of three and 4 episodes of 18."""
    grid = cut_job(tmp_path)
    rows, _cells, views, _calls, _engine = profile_table.build([grid / "run_a"])
    view = views[0]
    assert (view["model_runs"], view["model_episodes"], view["model_folders_without_row"]) == (3, 18, 2)
    assert len(profile_table.run_dirs(grid)) == 1 and sum(1 for row in rows if row["left_out"] == "") == 4  # the control
    assert view["notes"] == [cut_note("run_b"), cut_note("run_c"),
                             f"the window open at the end of in_flight.tsv from {int(made_up.wall(100.0 + 4 * EPISODE))} closes at "
                             f"{made_up.wall(100.0 + 9 * EPISODE)}, the earliest last episode end among its runs, since the end of the job cut them"]
    assert view["window_s"] == pytest.approx(9 * EPISODE) and view["episodes_ending_inside"] == 18  # the planted count of two streams
    assert view["episode_shares_inside"] / view["window_s"] == pytest.approx(2 / EPISODE, rel=1e-9)
    assert view["card_s_per_episode"] == pytest.approx(300.0 / 18)
    cut_rows = profile_table.build([grid / "run_b"])[0]
    assert {row["unfinished_run"] for row in cut_rows} == {1} and len(cut_rows) == 10 and cut_rows[-1]["left_out"] == "no row in scores.csv"
    assert cut_rows[0]["gpu_note"] == "the run has no run.json, whose record of the sampler the prefix check needs"
    assert {row["unfinished_run"] for row in rows} == {0}


def test_an_unfinished_run_takes_its_asset_from_its_state_files_and_names_it_when_none_records_it(tmp_path):
    grid = cut_job(tmp_path)
    assert run_files.ledger_meta(grid / "run_b")["asset_id"] == "made_up_asset" and run_files.ledger_meta(grid / "run_b")["unrecorded"] == []
    for state in (grid / "run_c").glob("seed*/episode.json"):
        parts = json.loads(state.read_text(encoding="utf-8"))
        del parts["config"]["asset_id"]
        state.write_text(json.dumps(parts), encoding="utf-8")
    assert run_files.ledger_meta(grid / "run_c")["asset_id"] is None and run_files.ledger_meta(grid / "run_c")["unrecorded"] == ["asset"]


def test_a_cut_run_whose_folder_lacks_a_field_of_its_cell_takes_its_windows_out_of_the_rate(tmp_path):
    """The state files of run_c hold no clock mode, so its cell is not recorded whole: no window in which it was alive gives a rate,
    the note names the field, and the card-seconds stay undivided; the control is the same job with the clock mode recorded."""
    whole = profile_table.build([cut_job(tmp_path / "whole") / "run_a"])[2][0]
    view = profile_table.build([cut_job(tmp_path / "lacking", clock_mode_of_c=False) / "run_a"])[2][0]
    assert whole["window_s"] == pytest.approx(9 * EPISODE) and view["window_s"] == pytest.approx(4 * EPISODE)
    assert "process line_2_replicate_none was alive, since its run run_c records no clock mode" in " ".join(view["notes"])
    assert view["card_s_per_episode"] is None and whole["card_s_per_episode"] is not None
    # the job still counts the passed episodes of every model run, run_c's among them; the control is the 4 + 9 of run_a and run_b
    assert view["passed"] == whole["passed"] == 18 != 4 + 9
    assert view["notes"] == [cut_note("run_b"), cut_note("run_c") + ", which record no clock mode",
                             "a model run of the job records its cell in part, so the card-seconds of the job are not divided",
                             "no rate from the 1 windows in which the process line_2_replicate_none was alive, since its run run_c records no "
                             "clock mode"]


def cut_note(name):
    """The note of a run of the made-up cut job that the end of the job cut."""
    return (f"{name} has no run.json and its process has no end line, so the end of the job cut it; its cell and its finished "
            "episodes are read from its ledger, its endpoint record and its state files")


def test_a_run_without_run_json_is_named_cut_or_stopped_by_the_end_line_of_its_process(tmp_path):
    """A run whose process has no end line was cut by the end of the job; one whose process ended stopped before its end, as the
    driver stops a run whose last episodes all end in a harness error; without the record of the processes the note says neither."""
    run = {"dir": tmp_path / "run_b", "process": ("1", "none"), "missing": []}
    read = "; its cell and its finished episodes are read from its ledger, its endpoint record and its state files"
    assert jobs.unfinished_note(run, set()) == f"run_b has no run.json and its process has no end line, so the end of the job cut it{read}"
    assert jobs.unfinished_note(run, {"line_1_replicate_none"}) == f"run_b has no run.json, though its process ended, so it stopped before its end{read}"
    assert jobs.unfinished_note(run) == f"run_b has no run.json{read}"


def test_a_job_pulled_while_it_runs_names_no_run_cut(tmp_path):
    """Before the final record a run without an end line may still be playing, so the note names no cause; the control is the same
    job with its final record."""
    running = cut_job(tmp_path / "running")
    (running / "job_record.txt").write_text("88|RUNNING|2026-01-05T10:00:00|Unknown|00:05:00|00:00:00|node7\n", encoding="utf-8")
    read = " and its finished episodes are read from its ledger, its endpoint record and its state files"
    assert profile_table.build([running / "run_a"])[2][0]["notes"] == [
        "the job record was written while the job ran; job_record.sh writes the final one after the job",
        f"run_b has no run.json; its cell{read}", f"run_c has no run.json; its cell{read}",
        f"the window open at the end of in_flight.tsv from {int(made_up.wall(100.0 + 4 * EPISODE))} closes at "
        f"{made_up.wall(100.0 + 9 * EPISODE)}, the earliest last episode end among its runs, since the record was read while they played"]
    assert cut_note("run_b") in profile_table.build([cut_job(tmp_path / "final") / "run_a"])[2][0]["notes"]


def test_a_ledger_whose_record_holds_no_agent_records_no_agent(tmp_path):
    """A ledger written by a driver that passed no agent: the run records no agent, whatever else its record holds; the control is
    the same ledger with its agent."""
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode())], unfinished=True)
    assert "agent" not in run_files.ledger_meta(run)["unrecorded"]  # the control
    path = run / "ledger.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"agent": "causal"', '"agent": null'), encoding="utf-8")
    assert run_files.ledger_meta(run)["unrecorded"] == ["agent"]


def test_the_process_of_a_run_is_named_by_the_line_and_the_replicate_of_its_ledger(tmp_path):
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode())], process=("3", "2"))
    assert run_files.process_of(run) == ("3", "2") and jobs.process_name({"process": run_files.process_of(run)}) == "line_3_replicate_2"
    assert run_files.process_of(made_up.write_run(tmp_path / "local", [(7, made_up.known_episode())])) is None  # no grid line


def model_line_then_baseline(tmp_path, baseline):
    """The view of a made-up job at one run in flight whose model line plays four episodes back to back from 100 and ends; with
    baseline, a line of a baseline policy, an agent without the served model, then plays alone for 100 seconds, both in the loop
    of the lines."""
    start = made_up.wall(100.0)
    events = [(int(start), "start", "units", "run", "line_0_replicate_none", 1, 0),
              (int(start + 4 * EPISODE), "end", "units", "run", "line_0_replicate_none", 1, 0)]
    if baseline:
        events += [(int(start + 4 * EPISODE), "start", "units", "run", "line_1_replicate_none", 2, 0),
                   (int(start + 4 * EPISODE) + 100, "end", "units", "run", "line_1_replicate_none", 2, 0)]
    grid = made_up.write_grid(tmp_path / "grid_99", "99", record=["99|COMPLETED|2026-01-05T10:00:00|2026-01-05T11:00:00|01:00:00|00:00:00|node7"],
                              log=[made_up.runner_header(50.0)], in_flight=events)
    made_up.write_run(grid / "run_model", [(7 + j, made_up.shifted(made_up.known_episode(), 100.0 + j * EPISODE)) for j in range(4)],
                      process=("0", "none"))
    if baseline:
        policy = made_up.model_meta(agent="oracle", agent_label="oracle", model=None, serving={"agent": None, "judge": None})
        made_up.write_run(grid / "run_baseline", [(7, made_up.shifted(made_up.known_episode(), 100.0 + 4 * EPISODE))], meta=policy,
                          process=("1", "none"))
    return profile_table.build([grid / "run_model"])[2][0]


def test_a_run_without_the_served_model_counts_toward_no_window_of_the_rate(tmp_path):
    """A job at one run in flight plays a model line and then a line of a baseline policy: only the model runs count toward N, so the
    time in which the baseline alone is alive is no window, and the rate reads the episodes of the model line over its own time.
    The control is the same job with the model line alone."""
    view = model_line_then_baseline(tmp_path / "with_baseline", baseline=True)
    alone = model_line_then_baseline(tmp_path / "model_alone", baseline=False)
    rate = ("window_s", "episodes_ending_inside", "episode_shares_inside", "ending_per_card_hour", "shares_per_card_hour")
    assert {name: view[name] for name in rate} == {name: alone[name] for name in rate}
    assert (view["window_s"], view["episodes_ending_inside"], view["episode_shares_inside"]) == (4 * EPISODE, 3, 4.0) and alone["notes"] == []
    assert view["notes"] == ["run_baseline reaches no served model, and its episodes lie in after_last_episode_s",
                             "the card-seconds of the job divide its whole Elapsed, with the time of the runs without the served model "
                             "(run_baseline)"]


def test_a_job_whose_card_seconds_are_not_read_names_its_baseline_with_its_part_alone(tmp_path):
    """With the job record written while the job ran, the job has no Elapsed and no card-seconds, so the note of the baseline names
    its part alone and no note speaks of card-seconds; the control is the job with its final record."""
    final = model_line_then_baseline(tmp_path / "final", baseline=True)
    grid = tmp_path / "running" / "grid_99"
    model_line_then_baseline(tmp_path / "running", baseline=True)
    (grid / "job_record.txt").write_text("99|RUNNING|2026-01-05T10:00:00|Unknown|00:05:00|00:00:00|node7\n", encoding="utf-8")
    view = profile_table.build([grid / "run_model"])[2][0]
    assert view["card_s_per_episode"] is None and view["notes"] == [
        "the job record was written while the job ran; job_record.sh writes the final one after the job",
        "run_baseline reaches no served model, and its episodes lie in after_last_episode_s"]
    assert final["card_s_per_episode"] is not None and any("card-seconds" in note for note in final["notes"])  # the control


def run_of(line, model_agent=True):
    """A made-up run of a line of the grid file as rate_windows reads it, its cell recorded whole and none of its episodes read."""
    return {"dir": Path(f"run_{line}"), "process": (str(line), "none"), "model_agent": model_agent, "missing": [], "rows": []}


def test_at_two_runs_in_flight_a_baseline_run_neither_makes_a_window_nor_splits_one():
    """Model lines 0 and 1 and a baseline line 2: alive beside both model runs, the baseline does not split their window, and alive
    beside one of them it makes no window of two; the control is line 2 as a model run, which splits the first window and makes the
    second."""
    events = [event_of(t, kind, f"line_{line}_replicate_none") for t, kind, line in
              [(0.0, "start", 0), (0.0, "start", 1), (100.0, "start", 2), (200.0, "end", 0), (300.0, "end", 2), (400.0, "end", 1)]]

    def windows(line_2_model):
        found, _notes = jobs.rate_windows(events, 2, [run_of(0), run_of(1), run_of(2, model_agent=line_2_model)])
        return [(a, b) for a, b, _names, _cut in found]
    assert windows(line_2_model=False) == [(0.0, 200.0)]
    assert windows(line_2_model=True) == [(0.0, 100.0), (200.0, 300.0)]  # the control


def test_a_process_without_a_run_folder_takes_its_windows_out_of_the_rate(tmp_path):
    grid = cut_job(tmp_path)
    for path in sorted((grid / "run_c").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (grid / "run_c").rmdir()
    view = profile_table.build([grid / "run_a"])[2][0]
    assert view["window_s"] == pytest.approx(4 * EPISODE) and view["model_runs"] == 2
    assert view["notes"] == [cut_note("run_b"), "no rate from the 1 windows in which the process line_2_replicate_none was alive, since no "
                                                "run folder of the grid folder records that process"]


def job_with_other_runs(tmp_path, others):
    """A made-up job at one run in flight whose grid runner plays, one after another: with others, a reference agent before the
    lines, then model line 0, a baseline line 1, model line 2, a baseline line 3, and a baseline line 4 whose one episode the profile
    leaves out; without others, the two model lines alone, at the same times. Every baseline is an agent without the served model."""
    lines = [(0, 100.0, 2, True), (1, 140.0, 1, False), (2, 160.0, 2, True), (3, 200.0, 1, False), (4, 230.0, 1, False)]
    events = [(int(made_up.wall(first)) + offset, kind, "units", "run", f"line_{line}_replicate_none", line, 0)
              for line, first, count, model in lines if others or model
              for offset, kind in ((0, "start"), (int(count * EPISODE) + 1, "end"))]
    grid = made_up.write_grid(tmp_path / "grid_99", "99", record=["99|COMPLETED|2026-01-05T10:00:00|2026-01-05T11:00:00|01:00:00|00:00:00|node7"],
                              log=[made_up.runner_header(50.0)], in_flight=sorted(events))
    policy = made_up.model_meta(agent="oracle", agent_label="oracle", model=None, serving={"agent": None, "judge": None})
    names = {0: "run_m1", 1: "run_between", 2: "run_m2", 3: "run_after", 4: "run_skewed"}
    for line, first, count, model in lines:
        if not (others or model):
            continue
        played = [(7 + j, made_up.shifted(made_up.known_episode(), first + j * EPISODE)) for j in range(count)]
        if line == 4:
            played[0][1]["row"]["clock_skew"] = "1"
        made_up.write_run(grid / names[line], played, meta=None if model else policy, process=(str(line), "none"))
    if others:
        made_up.write_run(grid / "run_reference", [(7, made_up.shifted(made_up.known_episode(), 60.0))], meta=policy)
    return profile_table.build([grid / "run_m1"])


def test_a_run_without_the_served_model_is_named_with_the_part_of_the_job_that_holds_its_episodes(tmp_path):
    """The parts around the episodes read the model episodes alone, so a reference run before the lines, a baseline between the two
    model lines and one after them leave every part as it is without them, and only the notes of the job say which part holds the
    time of each, the baseline without a kept episode among them; the job's card-seconds and those of its cell divide an Elapsed
    that holds that time, and the notes say so. The control is the job of the two model lines alone, whose parts are the same and
    whose notes are empty."""
    _rows, cells, views, _calls, _engine = job_with_other_runs(tmp_path / "with", others=True)
    _rows, alone_cells, alone_views, _calls, _engine = job_with_other_runs(tmp_path / "alone", others=False)
    parts = ("to_first_episode_s", "episodes_span_s", "after_last_episode_s", "card_s_per_episode")
    assert {name: views[0][name] for name in parts} == {name: alone_views[0][name] for name in parts}
    assert (views[0]["to_first_episode_s"], views[0]["episodes_span_s"]) == (pytest.approx(50.0), pytest.approx(93.0))
    others = "run_after, run_between, run_reference, run_skewed"
    assert views[0]["runs_without_model"] == others.split(", ") and alone_views[0]["runs_without_model"] == []
    assert views[0]["notes"] == [
        "run_after reaches no served model, and its episodes lie in after_last_episode_s",
        "run_between reaches no served model, and its episodes lie in episodes_span_s",
        "run_reference reaches no served model, and its episodes lie in to_first_episode_s",
        "run_skewed reaches no served model, and it has no episode with a span, so no part is named for its time",
        f"the card-seconds of the job divide its whole Elapsed, with the time of the runs without the served model ({others})"]
    assert alone_views[0]["notes"] == []  # the control
    assert cells[0]["notes"] == [f"the card-seconds of the cell divide the Elapsed of the job grid_99, which holds the time of the runs "
                                 f"without the served model ({others})"] and alone_cells[0]["notes"] == []


def test_a_run_whose_episodes_meet_two_parts_is_named_with_both_and_one_that_records_no_agent_says_so():
    def row(start):
        return {"left_out": "", "started_at": start, "total_s": 20.0}
    spans = [(100.0, 150.0), (160.0, 200.0)]
    straddling = {"dir": Path("run_x"), "missing": [], "rows": [row(90.0)]}
    assert jobs.other_run_note(straddling, spans) == "run_x reaches no served model, and its episodes lie in to_first_episode_s and episodes_span_s"
    late = {"dir": Path("run_y"), "missing": ["agent", "preset"], "rows": [row(150.0), row(190.0)]}
    assert jobs.other_run_note(late, spans) == "run_y records no agent, and its episodes lie in episodes_span_s and after_last_episode_s"
    inside = {"dir": Path("run_z"), "missing": [], "rows": [row(150.0)]}
    assert jobs.other_run_note(inside, spans) == "run_z reaches no served model, and its episodes lie in episodes_span_s"  # the control


def test_a_cut_run_whose_folder_records_no_agent_leaves_the_card_seconds_undivided(tmp_path):
    """run_c of the cut job holds scores.csv without its ledger, so nothing records its agent, which may have played the served
    model: the card-seconds of the job and of its cell are not divided, the notes say so and name the part that holds its
    episodes, and its rows say that the run records no agent rather than that its agent reaches no served model. The control is
    the same job with the ledger of run_c, whose card-seconds are divided."""
    whole = cut_job(tmp_path / "whole")
    grid = cut_job(tmp_path / "no_ledger")
    (grid / "run_c" / "ledger.json").unlink()
    rows, cells, views, _calls, _engine = profile_table.build([grid / "run_a"])
    _rows, whole_cells, whole_views, _calls, _engine = profile_table.build([whole / "run_a"])
    view = views[0]
    assert (view["model_runs"], view["model_episodes"], view["runs_without_model"]) == (2, 13, [])
    assert view["card_s_per_episode"] is None and view["card_s_per_passed_episode"] is None and view["card_s_undivided"] is True
    assert whole_views[0]["card_s_per_episode"] == pytest.approx(300.0 / 18) and whole_views[0]["card_s_undivided"] is False
    assert view["notes"] == [cut_note("run_b"), cut_note("run_c") + ", which record no agent, preset, runs in flight, model, serving identity",
                             "run_c records no agent, and its episodes lie in episodes_span_s",
                             "a run of the job records no agent, so the card-seconds of the job are not divided",
                             "no rate from the 1 windows in which the process line_2_replicate_none was alive, since no run folder of the "
                             "grid folder records that process"]
    assert cells[0]["jobs_view"]["card_s_per_episode"] is None and whole_cells[0]["jobs_view"]["card_s_per_episode"] == pytest.approx(300.0 / 18)
    assert cells[0]["notes"] == ["the job grid_88 leaves its card-seconds undivided, so those of the cell are not divided"]
    unknown = profile_table.build([grid / "run_c"])[0]
    assert {(row["gpu_note"], row["engine_note"]) for row in unknown} == {(profile_table.NO_AGENT, profile_table.NO_AGENT)}
    oracle = made_up.model_meta(agent="oracle", agent_label="oracle", model=None, serving={"agent": None, "judge": None})
    policy = profile_table.build([made_up.write_run(tmp_path / "policy", [(7, made_up.known_episode())], meta=oracle)])[0]
    assert (policy[0]["gpu_note"], policy[0]["engine_note"]) == (profile_table.NO_MODEL, profile_table.NO_MODEL)  # the control


def test_a_cut_run_whose_folder_lacks_a_field_of_its_cell_leaves_the_card_seconds_of_its_cell_undivided(tmp_path):
    """The cell reads the card-seconds of the jobs that played it alone, so a job that leaves its own undivided leaves those of
    the cell undivided too, and the note names it; the control is the cell of the same job with the clock mode of run_c."""
    lacking = profile_table.build([cut_job(tmp_path / "lacking", clock_mode_of_c=False) / "run_a"])[1][0]
    whole = profile_table.build([cut_job(tmp_path / "whole") / "run_a"])[1][0]
    assert lacking["jobs_view"]["card_s_per_episode"] is None and lacking["jobs_view"]["card_s_per_passed_episode"] is None
    assert whole["jobs_view"]["card_s_per_episode"] == pytest.approx(300.0 / 18) and whole["notes"] == []
    assert lacking["notes"] == ["the job grid_88 leaves its card-seconds undivided, so those of the cell are not divided"]


def test_a_note_names_the_lines_of_in_flight_tsv_that_it_cannot_read_and_the_slurm_log_that_it_misses(tmp_path):
    """Three lines of the record, which holds two whole lines, are not its seven fields, and the job's log is missing: the notes
    give both figures; the control is the folder with its log and its record whole."""
    view = job_folder(tmp_path / "whole")
    assert view["notes"] == []
    broken = job_folder(tmp_path / "broken")
    grid = tmp_path / "broken" / "grid_77"
    (grid / "job_logs" / "cd-serve-77.out").unlink()
    with (grid / "in_flight.tsv").open("a", encoding="utf-8") as f:
        f.write("not a line\n1\tstart\nx\tstart\tunits\trun\tline_9_replicate_none\t1\t0\n")
    view = profile_table.build([grid / "run_a"])[2][0]
    assert broken["notes"] == [] and view["notes"] == ["job_logs holds no single Slurm log *-77.out",
                                                       "3 lines of in_flight.tsv are not its seven fields"]
