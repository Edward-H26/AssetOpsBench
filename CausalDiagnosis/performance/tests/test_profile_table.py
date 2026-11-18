"""profile_table.py, the command: the flat table of one row per episode with every column a plain value, the summary per cell and
per job, the printed table, every line of which is asserted whole, the refusal of an output inside the fork and of a folder that
holds no run, and no write into a run."""
import copy
import csv
import json
import re

from CausalDiagnosis.performance import profile_table
from CausalDiagnosis.performance.tests import made_up
from CausalDiagnosis.utils.paths import ROOT


def grid_with_a_run(tmp_path):
    grid = made_up.write_grid(
        tmp_path / "grid_77", "77", record=["77|COMPLETED|2026-01-05T10:00:00|2026-01-05T11:00:00|01:00:00|00:00:00|node7"],
        log=[made_up.runner_header(10.0), "OPENING_CALL answered seconds 2.500 cycles made_up 0.250"],
        in_flight=[(int(made_up.wall(15.0)), "start", "units", "run", "line_0_replicate_none", 1, 0),
                   (int(made_up.wall(600.0)), "end", "units", "run", "line_0_replicate_none", 1, 0)],
        sampler=made_up.sampler_text([(t, 50, 100.0) for t in range(0, 600, 5)]))
    meta = made_up.model_meta()
    meta["serving"]["agent"]["sampler"] = made_up.recorded_sampler(grid / "job_logs" / "gpu-77.csv")
    episodes = [(7, made_up.shifted(made_up.known_episode(), 20.0)), (8, made_up.shifted(made_up.known_episode(), 40.0))]
    return made_up.write_run(grid / "made_up_run", episodes, meta=meta, process=("0", "none"))


def test_the_command_writes_the_flat_table_and_the_summary_and_prints_them(tmp_path, capsys):
    run = grid_with_a_run(tmp_path)
    assert profile_table.main([str(run), "--out", str(tmp_path / "out")]) == 0
    with (tmp_path / "out" / profile_table.EPISODES_FILE).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and list(rows[0]) == list(profile_table.COLUMNS)
    assert all(not value.startswith(("{", "[")) for row in rows for value in row.values())  # no nesting in the flat table
    assert rows[0]["sampler_rows"] == "4" and float(rows[0]["gpu_busy_share"]) == 0.5 and rows[0]["left_out"] == ""
    summary = json.loads((tmp_path / "out" / profile_table.CELLS_FILE).read_text(encoding="utf-8"))
    assert [cell["runs"] for cell in summary["cells"]] == [["made_up_run"]] and summary["jobs"][0]["grid"] == "grid_77"
    assert summary["cells"][0]["jobs_view"]["jobs_alone"] == ["grid_77"] and summary["jobs"][0]["window_s"] is not None
    printed = capsys.readouterr().out
    assert printed == profile_table.render(summary["cells"], summary["jobs"], summary["engine"]) + "\n"  # the blocks of what it wrote
    assert printed.splitlines()[0] == ("cell 1: asset made_up_asset, agent causal, preset made_up_preset, clock generation_time, "
                                       "model made-up-model, runs in flight 1, card MADE_UP_CARD 1000 MiB, engine made_up")


def test_an_output_folder_inside_the_fork_and_a_folder_that_holds_no_run_are_refused(tmp_path, capsys):
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode())])
    inside = ROOT / "CausalDiagnosis" / "performance" / "made_up_out"
    assert profile_table.main([str(run), "--out", str(inside)]) == 2 and not inside.exists()
    assert capsys.readouterr().err == (f"{inside.resolve()} lies inside the fork, and the profile holds measured times of runs, which "
                                       "stay outside it\n")
    assert profile_table.main([str(tmp_path), "--out", str(tmp_path / "out")]) == 2 and not (tmp_path / "out").exists()
    assert capsys.readouterr().err == f"no run.json, ledger or scores.csv in {tmp_path}: the profile reads run folders\n"
    unfinished = made_up.write_run(tmp_path / "unfinished", [(7, made_up.known_episode())], unfinished=True)  # the control
    assert profile_table.main([str(unfinished), "--out", str(tmp_path / "out")]) == 0 and (tmp_path / "out" / profile_table.EPISODES_FILE).is_file()


def test_the_command_writes_nothing_into_a_run_folder_or_its_grid_folder(tmp_path):
    run = grid_with_a_run(tmp_path)
    before = {path: path.read_bytes() for path in run.parent.rglob("*") if path.is_file()}
    assert profile_table.main([str(run), "--out", str(tmp_path / "out")]) == 0
    assert {path: path.read_bytes() for path in run.parent.rglob("*") if path.is_file()} == before


def test_a_grid_folder_reached_through_a_link_keeps_the_name_that_names_its_job(tmp_path):
    """The job of a run is read from the name of its grid folder, grid_<job id>, so a copy of a grid folder kept under another name
    and reached through a link of that name still brings the view of its job; the control is the folder under its own name."""
    run = grid_with_a_run(tmp_path / "copy")
    (tmp_path / "copy" / "grid_77").rename(tmp_path / "copy" / "pulled_77")
    (tmp_path / "grid_77").symlink_to(tmp_path / "copy" / "pulled_77", target_is_directory=True)
    through_link = profile_table.build([tmp_path / "grid_77" / run.name])[2]
    under_its_name = profile_table.build([tmp_path / "copy" / "pulled_77" / run.name])[2]
    assert [view["grid"] for view in through_link] == ["grid_77"] and under_its_name == []


def planted(summary, values):
    """A copy of a summary that the command wrote with the value at each dotted path of values in place of its own, each path one
    that the summary holds, so that the planted summary keeps the shape of the code's own."""
    out = copy.deepcopy(summary)
    for path, value in values.items():
        *head, last = path.split(".")
        place = out
        for key in head:
            place = place[key]
        assert last in place, path
        place[last] = value
    return out


def figures_of(lines):
    """The figures that printed lines show, each with its percent sign, in the order printed."""
    return [token for line in lines for token in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?%?(?![\w.])", line)]


def built(tmp_path):
    """The summary of the cell, the view of the job and the summary of the join of the made-up grid of grid_with_a_run, with an
    export of one line for the first model call of its first episode."""
    run = grid_with_a_run(tmp_path)
    export = made_up.write_export(tmp_path / "export", [("10", [made_up.export_line(made_up.wall(24.0625), 100, 30)])])
    _rows, summaries, views, _calls, engine = profile_table.build([run], [export])
    return summaries[0], views[0], engine


# the rows of the table of a cell in the order printed, each quantity beside its label, written out here so that no label comes
# from the code under test
QUANTITIES = (("total_s", "total T"), ("creation_s", "creation C"), ("cache_flush_s", "cache flush Fl, in C"),
              ("agent_part_s", "agent part P"), ("end_s", "end E"), ("model_s", "model M"), ("failed_model_s", "failed model Mf, in M"),
              ("tool_service_s", "tool service S"), ("tool_service_cpu_s", "processor part of S"),
              ("no_call_service_s", "no_call steps, in rest"), ("server_starts_s", "server starts R"),
              ("env_tools_client_s", "environment tools Tc"), ("diagnostics_client_s", "diagnostics server Tm"),
              ("transport_hold_s", "transport and hold Tc - S"), ("rest_s", "rest"))


def table_line(label, *texts):
    return f"  {label:<28}" + "".join(f"{text:>{width}}" for text, width in zip(texts, (12, 9, 10, 10, 10)))


def test_every_line_of_the_block_of_a_cell_is_printed_whole(tmp_path):
    """Every figure of a made-up cell differs from every other, so that a figure printed under the label of another, or two labels
    swapped, print another line; the control asserts that the printed figures differ pairwise."""
    cell, _view, _engine = built(tmp_path)
    values = {"cell": 3, "key.asset": "asset_x", "key.agent": "agent_y", "key.config": "preset_z", "key.clock_mode": "clock_w",
              "key.model": "model_v", "key.runs_in_flight": 5, "key.serving.gpu_name": "CARD_U", "key.serving.gpu_mib": "4096",
              "key.serving.engine": "engine_t", "runs": ["r1", "r2", "r3", "r4"], "episodes": 21, "kept": 8,
              "left_out": {"clock_skew": 6, "no state file": 7},
              "checks.lines_in_model_spans.failing": 26, "checks.lines_in_model_spans.checked": 29,
              "checks.rest_negative.failing": 27, "checks.rest_negative.checked": 30,
              "checks.late_server_starts.failing": 28, "checks.late_server_starts.checked": 31,
              "clock.thinking_s": 31.875, "clock.thinking_cycles": 13, "clock.thinking_share_of_horizon": 0.417,
              "clock.credited_starts_s": 32.625, "clock.credited_calls_s": 33.375, "clock.uncharged_after_last_s": 34.125,
              "calls.model_calls": 16, "calls.failed_model_calls": 14, "calls.call_seconds.median": 35.5, "calls.call_seconds.p10": 36.25,
              "calls.call_seconds.p90": 37.75, "calls.input_tokens": 4001, "calls.output_tokens": 502,
              "calls.output_tokens_per_call_second": 12.3, "engine.episodes": 18, "engine.engine_calls_joined": 19,
              "engine.engine_calls_unjoined": 20, "engine.engine_input_mismatches": 22, "engine.engine_output_mismatches": 23,
              "engine.engine_extra_lines": 24, "engine.call_share_queue": 0.023, "engine.call_share_prefill": 0.034,
              "engine.call_share_decode": 0.456, "engine.call_share_outside_stages": 0.067, "engine.call_share_unjoined_calls": 0.278,
              "engine.call_share_differing_calls": 0.142, "engine.decode_tokens_per_s": 61.4, "gpu.sampler_rows": 25,
              "gpu.gpu_busy_share": 0.583, "gpu.gpu_energy_j_per_episode": 2601.0, "gpu.gpu_energy_j_per_passed_episode": 3702.0,
              "cost.pass_rate": 0.625, "cost.held_card_s_per_episode": 38.5, "cost.held_card_s_per_passed_episode": 40.5,
              "jobs_view.jobs_alone": ["grid_1", "grid_2"], "jobs_view.card_s_per_episode": 41.5,
              "jobs_view.card_s_per_passed_episode": 42.5, "jobs_view.ending_per_card_hour": 43.21,
              "jobs_view.shares_per_card_hour": 44.32, "notes": ["made up note one", "made up note two"]}
    for k, (name, _label) in enumerate(QUANTITIES):
        values.update({f"seconds.{name}.sum": 100 + k + 0.125, f"seconds.{name}.share": (10 * k + 11) / 1000,
                       f"seconds.{name}.median": k + 0.5, f"seconds.{name}.p10": k + 0.25, f"seconds.{name}.p90": k + 0.75})
    expected = [
        "cell 3: asset asset_x, agent agent_y, preset preset_z, clock clock_w, model model_v, runs in flight 5, card CARD_U 4096 MiB, "
        "engine engine_t",
        "  runs 4, episodes 21, kept 8, left out 6 clock_skew, 7 no state file",
        table_line("quantity", "sum s", "share", "median", "p10", "p90"),
        *(table_line(label, f"{100 + k}.125", f"{k + 1}.1%", f"{k}.500", f"{k}.250", f"{k}.750") for k, (_name, label) in enumerate(QUANTITIES)),
        "  checks, failing of checked: a line inside a model span 26 of 29, rest below zero 27 of 30, server starts outside the opening "
        "28 of 31",
        "  thinking: 31.875 s and 13 cycles charged, 41.7% of the horizon; credited starts 32.625 s, calls 33.375 s; uncharged after the "
        "last call 34.125 s",
        "  model calls 16 (14 failed), seconds per call median 35.500 p10 36.250 p90 37.750, tokens in 4001 out 502, out per call-second "
        "12.3",
        "  engine, 18 episodes: calls joined 19, unjoined 20, token mismatches 22 in and 23 out, lines of retries 24",
        "  of the call seconds: queue 2.3%, prefill 3.4%, decode 45.6%, outside the three stages 6.7%, unjoined calls 27.8%, differing "
        "calls 14.2%; decode tokens per s 61.4",
        "  GPU: 25 sampler rows, busy 58.3%, energy per episode 2601 J, per passed episode 3702 J",
        "  pass rate 62.5%; card-seconds held alone per episode 38.5, per passed episode 40.5",
        "  jobs that played the cell alone 2: card-seconds per episode 41.5, per passed episode 42.5; episodes per card-hour, ending "
        "inside 43.21, shares inside 44.32",
        "  note: made up note one", "  note: made up note two"]
    assert profile_table.render_cell(planted(cell, values)) == expected
    shown = figures_of(expected[:2] + expected[3:-2])
    assert len(shown) == len(set(shown)) == 128  # the control: no two printed figures are equal


def test_a_cell_without_a_served_model_prints_a_dash_for_every_figure_it_lacks_and_a_zero_as_zero(tmp_path):
    """The other branch of every figure of a cell: a cell of an agent without a served model and with no episode left out prints
    no card, and a dash for every figure that it lacks, so that a missing figure never reads as zero; the control is the one
    count planted as zero, which prints as zero."""
    cell, _view, _engine = built(tmp_path)
    values = {"key.serving.gpu_name": None, "left_out": {}, "notes": [], "jobs_view.jobs_alone": [], "engine.episodes": 0}
    for path in ("clock.thinking_s", "clock.thinking_cycles", "clock.thinking_share_of_horizon", "clock.credited_starts_s",
                 "clock.credited_calls_s", "clock.uncharged_after_last_s", "calls.model_calls", "calls.failed_model_calls",
                 "calls.call_seconds.median", "calls.call_seconds.p10", "calls.call_seconds.p90", "calls.input_tokens",
                 "calls.output_tokens", "calls.output_tokens_per_call_second", "engine.engine_calls_joined", "engine.engine_calls_unjoined",
                 "engine.engine_input_mismatches", "engine.engine_output_mismatches", "engine.engine_extra_lines",
                 "engine.call_share_queue", "engine.call_share_prefill", "engine.call_share_decode", "engine.call_share_outside_stages",
                 "engine.call_share_unjoined_calls", "engine.call_share_differing_calls", "engine.decode_tokens_per_s",
                 "gpu.gpu_busy_share", "gpu.gpu_energy_j_per_episode", "gpu.gpu_energy_j_per_passed_episode", "cost.pass_rate",
                 "cost.held_card_s_per_episode", "cost.held_card_s_per_passed_episode", "jobs_view.card_s_per_episode",
                 "jobs_view.card_s_per_passed_episode", "jobs_view.ending_per_card_hour", "jobs_view.shares_per_card_hour",
                 *(f"seconds.{name}.{field}" for name, _label in QUANTITIES for field in ("sum", "share", "median", "p10", "p90"))):
        values[path] = None
    printed = profile_table.render_cell(planted(cell, {**values, "gpu.sampler_rows": 0}))
    assert printed[0].endswith(", runs in flight 1, no served model") and printed[1] == "  runs 1, episodes 2, kept 2, left out none"
    assert printed[3:] == [
        *(table_line(label, "-", "-", "-", "-", "-") for _name, label in QUANTITIES),
        "  checks, failing of checked: a line inside a model span 0 of 2, rest below zero 0 of 2, server starts outside the opening 0 of 2",
        "  thinking: - s and - cycles charged, - of the horizon; credited starts - s, calls - s; uncharged after the last call - s",
        "  model calls - (- failed), seconds per call median - p10 - p90 -, tokens in - out -, out per call-second -",
        "  engine, 0 episodes: calls joined -, unjoined -, token mismatches - in and - out, lines of retries -",
        "  of the call seconds: queue -, prefill -, decode -, outside the three stages -, unjoined calls -, differing calls -; decode "
        "tokens per s -",
        "  GPU: 0 sampler rows, busy -, energy per episode - J, per passed episode - J",
        "  pass rate -; card-seconds held alone per episode -, per passed episode -",
        "  jobs that played the cell alone 0: card-seconds per episode -, per passed episode -; episodes per card-hour, ending inside -, "
        "shares inside -"]


def test_every_line_of_the_block_of_a_job_is_printed_whole(tmp_path):
    """The parts, the cost and the rate of a made-up job differ pairwise, so that a figure under the label of another, or two labels
    swapped, print another line; a job without a cell prints its parts and its notes alone, and a figure that it lacks as a dash.
    The control asserts that the printed figures differ pairwise."""
    _cell, view, _engine = built(tmp_path)
    values = {"elapsed_s": 3600.5, "model_runs": 3, "model_episodes": 17, "cell": 9, "server_start_s": 301.1, "opening_s": 1202.2,
              "opening_check_s": 1199.3, "opening_call_s": 2.4, "to_first_episode_s": 60.5, "episodes_span_s": 456.6,
              "after_last_episode_s": 1583.7, "share_server_start": 0.081, "share_opening": 0.334, "share_opening_check": 0.333,
              "share_opening_call": 0.004, "share_to_first_episode": 0.017, "share_episodes_span": 0.127,
              "share_after_last_episode": 0.44, "card_s_per_episode": 211.5, "card_s_per_passed_episode": 317.5, "runs_in_flight": 6,
              "window_s": 480.5, "episodes_ending_inside": 14, "episode_shares_inside": 13.25, "ending_per_s": 0.029167,
              "shares_per_s": 0.027604, "ending_per_card_hour": 105.01, "shares_per_card_hour": 99.38,
              "notes": ["made up note one", "made up note two"]}
    head = ["job grid_77: elapsed 3600.5 s, model runs 3, model episodes 17, cell 9",
            "  server start 301.1 s (8.1%), opening 1202.2 s (33.4%), opening checks 1199.3 s (33.3%), opening call 2.4 s (0.4%), "
            "to the first episode 60.5 s (1.7%), episodes 456.6 s (12.7%), after the last episode 1583.7 s (44.0%)"]
    cost = ["  card-seconds per episode 211.5, per passed episode 317.5",
            "  at 6 runs in flight: windows 480.5 s, episodes ending inside 14, shares inside 13.25; per second 0.029167 and 0.027604, "
            "per card-hour 105.01 and 99.38"]
    notes = ["  note: made up note one", "  note: made up note two"]
    assert profile_table.render_job(planted(view, values)) == head + cost + notes
    shown = figures_of(head + cost)
    assert len(shown) == len(set(shown)) == 28  # the control: no two printed figures are equal
    assert profile_table.render_job(planted(view, {**values, "cell": None})) == [head[0].replace("cell 9", "cell None"), head[1], *notes]
    kept = ("model_runs", "model_episodes", "cell", "runs_in_flight", "notes")
    missing = planted(view, {**values, **{name: None for name in values if name not in kept}, "notes": []})
    assert profile_table.render_job(missing) == [
        "job grid_77: elapsed - s, model runs 3, model episodes 17, cell 9",
        "  server start - s (-), opening - s (-), opening checks - s (-), opening call - s (-), to the first episode - s (-), "
        "episodes - s (-), after the last episode - s (-)",
        "  card-seconds per episode -, per passed episode -",
        "  at 6 runs in flight: windows - s, episodes ending inside -, shares inside -; per second - and -, per card-hour - and -"]


def test_the_line_of_the_join_to_the_engine_is_printed_whole(tmp_path):
    """Every count of the join of a made-up job differs from every other; a join that was not made prints its reason alone. The
    control asserts that the printed counts differ pairwise."""
    _cell, _view, engine = built(tmp_path)
    values = {"export_files": 2, "lines": 31, "unread_lines": 3, "passed_lines": 4, "ambiguous_lines": 5, "calls": 27,
              "joined_calls": 26, "extra_lines": 6}
    expected = ["engine export: 2 files, 31 lines read, 3 unread, 4 received while no call was open, 5 while two were; 26 of 27 model "
                "calls joined, 6 lines of retried attempts"]
    assert sorted(engine) == sorted(values) and profile_table.render_engine(planted(engine, values)) == expected
    assert len(set(figures_of(expected))) == len(values)  # the control: no two printed counts are equal
    assert profile_table.render_engine({"note": profile_table.NO_EXPORT}) == ["engine export: no engine export is given"]


def test_the_printed_table_is_every_cell_then_every_job_then_the_join_each_closed_by_a_blank_line(monkeypatch):
    monkeypatch.setattr(profile_table, "render_cell", lambda cell: [f"cell {cell}", "  of the cell"])
    monkeypatch.setattr(profile_table, "render_job", lambda view: [f"job {view}"])
    monkeypatch.setattr(profile_table, "render_engine", lambda engine: [f"engine {engine}"])
    assert profile_table.render([1, 2], [3], "e") == "cell 1\n  of the cell\n\ncell 2\n  of the cell\n\njob 3\n\nengine e\n"
