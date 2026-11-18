"""engine_export.py: the join of the engine's export to the model calls of llm_calls.json on a made-up export, every line inside
a call's span taken by that call, the lines that it passes over or cannot place, the calls that it leaves unjoined, the token
checks, the parts of the call seconds that add up to one with the line that prints their shares, and that no prompt of the export
reaches an output."""
import json

import pytest

from CausalDiagnosis.performance import cells, engine_export, profile_table
from CausalDiagnosis.performance.tests import made_up
from CausalDiagnosis.performance.tests.summaries import assert_same

# the three model calls of known_episode() start at 4.0, 9.5 and 14.0, end at 7.0, 12.0 and 15.0 and last 3.0, 2.5 and 1.0 seconds;
# their tokens in and out, and the decode seconds of the line planted for each, which the engine finishes before the call ends
CALLS = ((4.0, 100, 30, 2.0), (9.5, 200, 50, 1.5), (14.0, 300, 20, 0.5))
QUEUE, PREFILL = 0.125, 0.25


def line(start, tokens_in, tokens_out, decode, cached=0):
    return made_up.export_line(made_up.wall(start), tokens_in, tokens_out, queue=QUEUE, prefill=PREFILL, decode=decode, cached=cached)


def export(tmp_path, change=None, drop=None):
    """A made-up export of the job of known_episode(): the engine's warm-up and the opening call before the episode, one line per call
    received just after the call starts, and a probe of the harness between the second and the third call, written over two files
    as the engine splits its export by the hour, each in the order in which the engine finished the requests, since it writes the
    line of a request when the request finishes."""
    lines = [made_up.export_line(made_up.wall(-100.0), 6, 8), made_up.export_line(made_up.wall(-50.0), 10, 1)]
    lines += [line(start + 0.0625, tokens_in, tokens_out, decode, 10 * k) for k, (start, tokens_in, tokens_out, decode) in enumerate(CALLS)
              if k != drop]
    lines.append(made_up.export_line(made_up.wall(13.0), 5, 1))
    if change:
        change(lines)
    return made_up.write_export(tmp_path / "export", [("09", lines[:2]), ("10", sorted(lines[2:], key=lambda l: l["request_finished_ts"]))])


def profile(tmp_path, folder, parts=None, meta=None):
    run = made_up.write_run(tmp_path / "run", [(7, parts or made_up.known_episode())], meta=meta)
    rows, summaries, _views, calls, engine = profile_table.build([run], [folder] if folder else [])
    return rows[0], summaries[0]["engine"], calls, engine


def shares(found):
    return sum(found[share] for share in engine_export.CALL_SHARES)


def test_the_join_passes_over_the_lines_of_no_call_and_splits_every_call_into_parts_that_add_up_to_its_seconds(tmp_path):
    row, cell, calls, engine = profile(tmp_path, export(tmp_path))
    assert engine == {"export_files": 2, "lines": 6, "unread_lines": 0, "passed_lines": 3, "ambiguous_lines": 0, "calls": 3,
                      "joined_calls": 3, "extra_lines": 0}
    parts = {"queue_s": 0.375, "prefill_s": 0.75, "decode_s": 4.0, "outside_stages_s": 1.375, "unjoined_calls_s": 0.0, "differing_calls_s": 0.0}
    assert_same({name: row[name] for name in engine_export.COLUMNS},
                {"engine_calls_joined": 3, "engine_calls_unjoined": 0, "engine_input_mismatches": 0, "engine_output_mismatches": 0,
                 "engine_extra_lines": 0, **parts, **{share: parts[name] / 6.5 for name, share in zip(engine_export.PARTS, engine_export.CALL_SHARES)},
                 "cached_prompt_tokens": 30, "decode_tokens": 97, "decode_tokens_per_s": 97 / 4.0, "engine_note": ""})
    assert shares(row) == pytest.approx(1.0) and shares(cell) == pytest.approx(1.0)
    assert [(call["call"], call["engine_lines"], call["call_s"]) for call in calls] == [(0, 1, 3.0), (1, 1, 2.5), (2, 1, 1.0)]
    assert calls[0]["outside_stages_s"] == pytest.approx(0.625) and calls[0]["decode_tokens_per_s"] == pytest.approx(29 / 2.0)
    assert calls[0]["decode_tokens_per_s"] != pytest.approx(30 / 3.0)  # the control: the rate over the whole call holds the prefill


def test_a_call_without_its_line_stays_unjoined_and_its_seconds_are_a_part_of_their_own(tmp_path):
    """The second call's line is missing: the probe received after the call ended is not taken for it, the third call keeps its
    own line, and the second call's seconds are the part of the unjoined calls; the control is the whole export."""
    row, _cell, calls, engine = profile(tmp_path, export(tmp_path, drop=1))
    assert (row["engine_calls_joined"], row["engine_calls_unjoined"], row["unjoined_calls_s"]) == (2, 1, 2.5)
    assert row["call_share_unjoined_calls"] == pytest.approx(2.5 / 6.5) and shares(row) == pytest.approx(1.0)
    assert [call["call"] for call in calls] == [0, 2] and all(call["output_match"] for call in calls) and engine["passed_lines"] == 3
    assert row["engine_note"] == "1 model calls have no line of the engine"


def test_a_call_whose_line_differs_in_its_tokens_is_counted_and_its_seconds_are_a_part_of_their_own(tmp_path):
    def more_tokens(lines):
        lines[2]["completion_tokens"] = 31
    row = profile(tmp_path, export(tmp_path, change=more_tokens))[0]
    assert (row["engine_output_mismatches"], row["engine_input_mismatches"], row["engine_calls_joined"]) == (1, 0, 3)
    assert row["differing_calls_s"] == 3.0 and row["decode_s"] == pytest.approx(2.0) and shares(row) == pytest.approx(1.0)
    assert row["engine_note"] == "1 joined calls differ in their token counts"


def test_a_retried_call_takes_every_line_of_its_span_and_adds_the_stages_of_its_attempts(tmp_path):
    """The second call is retried inside its span: a first attempt whose answer the client dropped, then the attempt it kept. Both
    lines are the call's, their stages add, the output tokens are checked against the line received last, the extra line is
    counted, and no line of the retry is passed over with the warm-up. The control is the reading that takes the first line alone,
    which passes the kept attempt over and checks the dropped one."""
    def retried(lines):
        kept = next(l for l in lines if l["prompt_tokens"] == 200)
        kept.update(line(10.5, 200, 50, 1.0, 10))
        lines.append(line(9.5625, 200, 12, 0.25, 0))
    row, _cell, calls, engine = profile(tmp_path, export(tmp_path, change=retried))
    second = next(call for call in calls if call["call"] == 1)
    assert (second["engine_lines"], second["output_match"], second["input_match"]) == (2, 1, 1)
    assert (second["queue_s"], second["prefill_s"], second["decode_s"]) == (2 * QUEUE, 2 * PREFILL, 1.25)
    assert second["decode_tokens"] == 11 + 49 and second["outside_stages_s"] == pytest.approx(2.5 - 2.0)
    assert row["engine_extra_lines"] == 1 and engine["extra_lines"] == 1 and engine["passed_lines"] == 3 and shares(row) == pytest.approx(1.0)
    assert row["engine_note"] == "1 lines of retried attempts are added to their calls"
    first_only = engine_export.call_figures({"input_tokens": 200, "output_tokens": 50, "seconds": 2.5}, [line(9.5625, 200, 12, 0.25)])
    assert first_only["output_match"] == 0  # the control: the dropped attempt alone differs from the answer the client kept


def test_a_retried_call_whose_attempts_overlap_in_the_engine_adds_both_and_checks_the_answer_received_last(tmp_path):
    """The second call is retried while the engine still runs its first attempt, which finishes after the kept one, so the export,
    which the engine writes as its requests finish, holds the kept attempt first. Both lines are the call's: their cached tokens
    add for the call and for the episode, its output tokens are checked against the line received last, and the stages of the two
    attempts overlap and pass the call's seconds, so the rest outside the three stages is below zero while the six parts still add
    up to the call seconds. The controls: the order of the file differs from the order of receipt, and either line alone holds
    other cached tokens than their sum."""
    def overlapping(lines):
        kept = next(l for l in lines if l["prompt_tokens"] == 200)
        kept.update(line(10.5, 200, 50, 1.0, 10))
        lines.append(line(9.5625, 200, 12, 2.25, 6))
    folder = export(tmp_path, change=overlapping)
    row, _cell, calls, _engine = profile(tmp_path, folder)
    second = next(call for call in calls if call["call"] == 1)
    assert (second["engine_lines"], second["input_match"], second["output_match"]) == (2, 1, 1)
    assert second["cached_prompt_tokens"] == 6 + 10 and row["cached_prompt_tokens"] == 0 + 16 + 20
    assert (second["queue_s"], second["prefill_s"], second["decode_s"]) == (2 * QUEUE, 2 * PREFILL, 2.25 + 1.0)
    assert second["outside_stages_s"] == pytest.approx(2.5 - 4.0) and row["outside_stages_s"] == pytest.approx(0.625 - 1.5 + 0.125)
    assert sum(row[name] for name in engine_export.PARTS) == pytest.approx(row["call_seconds_sum"]) == 6.5 and shares(row) == pytest.approx(1.0)
    written = [json.loads(text) for text in (folder / "sglang-request-metrics-20260105_10.log").read_text(encoding="utf-8").splitlines()]
    assert [l["completion_tokens"] for l in written if l["prompt_tokens"] == 200] == [50, 12]  # the control: the kept attempt first


def test_a_retried_call_whose_answer_differs_still_counts_its_extra_line(tmp_path):
    """The second call is retried and its line received last differs from the call in its output tokens: the call is a differing
    call whose seconds are the part of the differing calls, and its extra line still counts in the episode, since the count holds
    every joined call; the control is the count over the agreeing calls alone, which holds none."""
    def retried_differing(lines):
        kept = next(l for l in lines if l["prompt_tokens"] == 200)
        kept.update(line(10.5, 200, 51, 1.0, 10))
        lines.append(line(9.5625, 200, 12, 0.25, 6))
    row, _cell, calls, engine = profile(tmp_path, export(tmp_path, change=retried_differing))
    assert (row["engine_output_mismatches"], row["engine_extra_lines"], engine["extra_lines"]) == (1, 1, 1)
    assert row["differing_calls_s"] == 2.5 and shares(row) == pytest.approx(1.0)
    assert row["engine_note"] == "1 joined calls differ in their token counts; 1 lines of retried attempts are added to their calls"
    assert sum(call["engine_lines"] - 1 for call in calls if call["input_match"] and call["output_match"]) == 0  # the control


def test_the_input_tokens_are_checked_against_every_line_and_the_output_tokens_against_the_last():
    call = {"input_tokens": 200, "output_tokens": 50, "seconds": 2.5}
    resent = [line(9.5625, 199, 50, 0.25), line(10.5, 200, 50, 1.0)]
    assert engine_export.call_figures(call, resent)["input_match"] == 0  # an attempt with another prompt
    assert engine_export.call_figures(call, [line(9.5625, 200, 50, 0.25), line(10.5, 200, 50, 1.0)])["input_match"] == 1  # the control


def test_a_line_received_while_two_calls_are_open_is_counted_apart_and_taken_by_neither():
    """Two calls overlap from 6 to 7: the line received there belongs to neither, the lines received while one call alone is open
    belong to it; the control is a line received after the second call ended, which the first still takes."""
    calls = [{"started_at": 4.0, "ended_at": 8.0}, {"started_at": 6.0, "ended_at": 7.0}]
    received = [{"request_received_ts": t} for t in (4.0625, 6.5, 7.5, 9.0)]
    matched, passed, ambiguous = engine_export.join(calls, received)
    assert [[l["request_received_ts"] for l in found] for found in matched] == [[4.0625, 7.5], []]
    assert (passed, ambiguous) == (1, 1)


def mixed_episode(tmp_path):
    """known_episode() with a fourth call that the client retried, and its export, in which the second call has no line and the
    third call's line differs in its output tokens: a joined, an unjoined, a differing and a retried call, whose six parts differ."""
    parts = made_up.known_episode()
    parts["llm_calls"].append({"started_at": made_up.wall(15.25), "ended_at": made_up.wall(15.75), "seconds": 0.5, "input_tokens": 400,
                               "output_tokens": 8})

    def mixed(lines):
        next(l for l in lines if l["prompt_tokens"] == 300)["completion_tokens"] = 21
        lines.extend([made_up.export_line(made_up.wall(15.3125), 400, 3, queue=0.03125, prefill=0.0625, decode=0.0625),
                      made_up.export_line(made_up.wall(15.4375), 400, 8, queue=0.03125, prefill=0.0625, decode=0.125)])
    return parts, export(tmp_path, change=mixed, drop=1)


def test_the_parts_add_up_to_one_for_an_episode_and_its_cell_with_a_joined_an_unjoined_a_differing_and_a_retried_call(tmp_path):
    parts, folder = mixed_episode(tmp_path)
    row, cell, _calls, _engine = profile(tmp_path, folder, parts=parts)
    assert (row["engine_calls_joined"], row["engine_calls_unjoined"], row["engine_output_mismatches"], row["engine_extra_lines"]) == (3, 1, 1, 1)
    assert_same({name: row[name] for name in engine_export.PARTS},
                {"queue_s": 0.1875, "prefill_s": 0.375, "decode_s": 2.1875, "outside_stages_s": 0.75, "unjoined_calls_s": 2.5,
                 "differing_calls_s": 1.0})
    assert sum(row[name] for name in engine_export.PARTS) == pytest.approx(row["call_seconds_sum"]) == 7.0
    assert (row["cached_prompt_tokens"], row["decode_tokens"]) == (0, 29 + 2 + 7)  # the differing call's 20 and 20 stay out
    assert row["engine_note"] == ("1 model calls have no line of the engine; 1 joined calls differ in their token counts; 1 lines of "
                                  "retried attempts are added to their calls")
    assert shares(row) == pytest.approx(1.0) and shares(cell) == pytest.approx(1.0)


def test_the_printed_line_of_a_cell_names_each_part_of_the_call_seconds_beside_its_own_share(tmp_path):
    """The six parts of the mixed episode all differ, so each share of the printed line of its cell stands under its own label; the
    control is that the six shares differ pairwise, so that a swap of two labels prints another line."""
    parts, folder = mixed_episode(tmp_path)
    summary = profile_table.build([made_up.write_run(tmp_path / "run", [(7, parts)])], [folder])[1][0]
    (printed,) = [text for text in profile_table.render_cell(summary) if text.startswith("  of the call seconds:")]
    assert printed == ("  of the call seconds: queue 2.7%, prefill 5.4%, decode 31.2%, outside the three stages 10.7%, "
                       "unjoined calls 35.7%, differing calls 14.3%; decode tokens per s 17.4")
    assert len({f"{summary['engine'][share]:.3f}" for share in engine_export.CALL_SHARES}) == 6  # the control


def test_a_token_count_of_the_client_takes_no_line(tmp_path):
    parts = made_up.known_episode()
    parts["llm_calls"].insert(1, {"kind": "tokenize", "started_at": made_up.wall(7.5), "ended_at": made_up.wall(13.5), "error": "made up"})
    run = made_up.write_run(tmp_path / "run", [(7, parts)])
    row, calls, summary = (lambda built: (built[0][0], built[3], built[4]))(profile_table.build([run], [export(tmp_path)]))
    assert (row["engine_calls_joined"], row["engine_calls_unjoined"], summary["passed_lines"], summary["ambiguous_lines"]) == (3, 0, 3, 0)
    # the column call counts the generation calls of the episode, so the calls at places 0, 2 and 3 of llm_calls.json are 0, 1 and 2
    assert [call["call"] for call in calls] == [0, 1, 2] and [call["started_at"] for call in calls] == [
        parts["llm_calls"][place]["started_at"] for place in (0, 2, 3)]
    lines = engine_export.read_lines(engine_export.export_files([tmp_path / "export"]))[0]
    walked = engine_export.join(parts["llm_calls"], lines)
    assert walked[2] == 1 and walked[0][1] != []  # the control: walked as a call, the token count would take the probe and share a line


def test_no_join_without_a_model_run_and_an_episode_without_its_calls_say_why(tmp_path):
    """An export beside the run of a baseline alone joins nothing and says why; an episode of a model run whose folder lacks
    llm_calls.json gets the engine note that names the file. The control is the model run with its file, which joins."""
    folder = export(tmp_path)
    oracle = made_up.model_meta(agent="oracle", agent_label="oracle", model=None, serving={"agent": None, "judge": None})
    rows, _cells, _views, calls, engine = profile_table.build([made_up.write_run(tmp_path / "oracle", [(7, made_up.known_episode())], meta=oracle)], [folder])
    assert engine == {"note": "no run read reaches the served model"} and calls == [] and rows[0]["engine_note"] == profile_table.NO_MODEL
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode()), (8, made_up.shifted(made_up.known_episode(), 100.0))])
    (run / "seed000008" / "llm_calls.json").unlink()
    rows = profile_table.build([run], [folder])[0]
    assert rows[1]["engine_note"] == "the episode holds no llm_calls.json" and rows[1]["engine_calls_joined"] is None
    assert rows[0]["engine_calls_joined"] == 3  # the control


def test_a_job_at_more_runs_in_flight_than_one_gets_no_join_and_says_why(tmp_path):
    row, cell, calls, engine = profile(tmp_path, export(tmp_path), meta=made_up.model_meta(runs_in_flight=2))
    assert row["engine_calls_joined"] is None and calls == [] and cell["episodes"] == 0
    assert row["engine_note"] == engine["note"] == "the join by time needs one run in flight, and the model runs read play 2"


def test_without_an_export_the_columns_stay_empty_with_the_reason(tmp_path):
    row, cell, calls, engine = profile(tmp_path, None)
    assert all(row[name] is None for name in engine_export.COLUMNS if name != "engine_note") and calls == []
    assert row["engine_note"] == profile_table.NO_EXPORT == engine["note"] and cell["decode_tokens_per_s"] is None


def test_a_line_that_is_no_record_of_the_engine_is_counted_unread(tmp_path):
    folder = export(tmp_path)
    with (folder / "sglang-request-metrics-20260105_10.log").open("a", encoding="utf-8") as f:
        f.write("not json\n" + json.dumps({"request_received_ts": 1.0}) + "\n")
    assert profile(tmp_path, folder)[3]["unread_lines"] == 2


def test_no_prompt_of_the_export_reaches_an_output(tmp_path):
    folder = export(tmp_path)
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode())])
    assert profile_table.main([str(run), "--out", str(tmp_path / "out"), "--engine-export", str(folder)]) == 0
    written = [path.read_text(encoding="utf-8") for path in (tmp_path / "out").iterdir()]
    assert len(written) == 3 and not any(made_up.PROMPT in text for text in written)
    assert all(made_up.PROMPT in path.read_text(encoding="utf-8") for path in folder.iterdir())  # the control: the export holds it


def test_an_export_is_refused_for_run_folders_of_two_places(tmp_path, capsys):
    """Three run folders in two grid folders: the refusal counts the places, which the control, the count of the folders, does not
    give."""
    one = made_up.write_run(tmp_path / "grid_1" / "run", [(7, made_up.known_episode())])
    beside = made_up.write_run(tmp_path / "grid_1" / "run_beside", [(7, made_up.known_episode())])
    two = made_up.write_run(tmp_path / "grid_2" / "run", [(7, made_up.known_episode())])
    folders = [str(one), str(beside), str(two)]
    assert profile_table.main([*folders, "--out", str(tmp_path / "out"), "--engine-export", str(export(tmp_path))]) == 2
    assert capsys.readouterr().err == "the run folders lie in 2 places, and an engine export is of the one job that served them\n"
    assert not (tmp_path / "out").exists()


def test_the_cell_keeps_an_episode_that_the_join_reached_with_no_joined_call(tmp_path):
    """Two episodes, the first with every line of the export and the second with none: the cell's unjoined part holds the second
    one's seconds; the control drops episodes without a joined call and would lose them."""
    second = made_up.shifted(made_up.known_episode(), 100.0)
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode()), (8, second)])
    rows, summaries = profile_table.build([run], [export(tmp_path)])[:2]
    assert [row["engine_calls_joined"] for row in rows] == [3, 0] and summaries[0]["engine"]["unjoined_calls_s"] == 6.5
    assert cells.engine_summary([row for row in rows if row["engine_calls_joined"]], [])["unjoined_calls_s"] == 0.0  # the control


def test_no_join_is_made_while_a_run_read_records_no_agent(tmp_path):
    """A run folder that holds scores.csv without its ledger records no agent, and its calls may have left lines of the export,
    which the join would pass over as lines of no call; so no join is made and every engine column says why. The control is the
    same two runs with the ledger of the second, which the join reads as an agent without the served model."""
    folder = export(tmp_path)
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode())])
    oracle = made_up.model_meta(agent="oracle", agent_label="oracle", model=None, serving={"agent": None, "judge": None})
    other = made_up.write_run(tmp_path / "other", [(7, made_up.shifted(made_up.known_episode(), 100.0))], meta=oracle, unfinished=True)
    joined = profile_table.build([run, other], [folder])
    assert joined[4]["joined_calls"] == 3 and joined[0][1]["engine_note"] == profile_table.NO_MODEL  # the control
    (other / "ledger.json").unlink()
    rows, _cells, _views, calls, engine = profile_table.build([run, other], [folder])
    reason = "the join by time needs the agent of every run read, and no agent is recorded for other"
    assert engine == {"note": reason} and calls == [] and rows[0]["engine_note"] == reason and rows[0]["engine_calls_joined"] is None
    assert rows[1]["engine_note"] == profile_table.NO_AGENT


def test_the_note_of_an_episode_counts_its_unjoined_calls_its_differing_calls_and_its_lines_of_retries_apart():
    """One call without a line, two whose lines differ from them in their tokens, and three lines of retries over two calls, so that
    each count of the note differs from the others; the control asserts that the three differ."""
    def call(tokens_in, tokens_out):
        return {"input_tokens": tokens_in, "output_tokens": tokens_out, "seconds": 2.0}
    entries = [(call(100, 30), None),
               (call(200, 50), engine_export.call_figures(call(200, 50), [line(9.5625, 200, 12, 0.25), line(10.5, 200, 51, 1.0)])),
               (call(300, 20), engine_export.call_figures(call(300, 20), [line(14.0625, 299, 20, 0.5)])),
               (call(400, 8), engine_export.call_figures(call(400, 8), [line(15.0, 400, 3, 0.0625), line(15.25, 400, 5, 0.0625),
                                                                        line(15.5, 400, 8, 0.0625)]))]
    found = engine_export.episode_columns(entries)
    counts = (found["engine_calls_unjoined"], found["engine_input_mismatches"] + found["engine_output_mismatches"], found["engine_extra_lines"])
    assert counts == (1, 2, 3) and len(set(counts)) == 3  # the control
    assert found["engine_note"] == ("1 model calls have no line of the engine; 2 joined calls differ in their token counts; 3 lines of "
                                    "retried attempts are added to their calls")
