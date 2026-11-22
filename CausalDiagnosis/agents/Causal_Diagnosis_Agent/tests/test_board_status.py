"""The board's side of the standing steps and of its notes: the lines of the status on what the board holds and on the calls a plan
repeated, the notes that leave once the state they speak of has passed, with equal notes of plans standing once with their
count, and the line of planning turns in a row that made no progress. Every test builds its board by hand, so no server runs."""
import random
import re

from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.plan_schema import EPISODE_KINDS, SURFACE_KINDS
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board.board import KIND_WORDS, NO_PROGRESS, NO_PROGRESS_TURNS, call_text
from CausalDiagnosis.agents.utils import core_texts

GROUPS = {"g1": ["A1"], "g2": ["A1", "B1"], "g3": ["C1"]}  # g2 covers g1
# a line of HELD_SERIES with its values as groups: the name, the first and the last recorded cycle, the held and the recorded count
HELD_SERIES = re.compile("^" + re.escape(core_texts.HELD_SERIES.format(name="NAME", first="FIRST", last="LAST", held="HELD", recorded="RECORDED", **KIND_WORDS))
                         .replace("NAME", r"(\w+)").replace("FIRST", r"(-?\d+)").replace("LAST", r"(-?\d+)").replace("HELD", r"(\d+)")
                         .replace("RECORDED", r"(\d+)") + "$", re.MULTILINE)


def _board(window=None, clock_runs=False, unrecorded=None):
    """A board with a profile of three sensor groups and, where window is given, the recorded window of a surface."""
    board = Board(clock_runs=clock_runs)
    board.set_profile({"components": ["c1"], "sensor_groups": dict(GROUPS), "healthy": {"c1": 0}})
    board.status = {"time": 0}
    if window is not None:
        board.status.update(recorded_from=window[0], recorded_until=window[1], unrecorded=unrecorded or {})
    return board


def _row(board, group, cycle):
    board.evidence.append({"evidence_id": f"obs/{cycle}/x", "n": cycle, "t": cycle, "sensor_group": group, "field": "x", "value": 0.0,
                           "band_low": None, "band_high": None, "inside_band": None})


def _status(board):
    return board._render_status()


def test_the_status_names_the_answers_the_record_and_the_repeated_calls_where_no_time_passes():
    board = _board(window=(1, 10))
    episode, surface = EPISODE_KINDS[0], SURFACE_KINDS[-1]
    board.note_answered(episode, {})
    board.note_answered("failure_modes", {"component": "c1"})
    board.note_answered(surface, {})
    for cycle in (9, 10):
        _row(board, "g2", cycle)
    board.note_call("read", {"sensor_group": "g2", "cycles": 2})
    board.note_call("read", {"sensor_group": "g2", "cycles": 2})
    board.note_call("read", {"sensor_group": "g3", "cycles": 1})
    status = _status(board)
    words = KIND_WORDS
    assert core_texts.HELD_EPISODE.format(calls=f"{call_text(episode, {})}, {call_text('failure_modes', {'component': 'c1'})}") in status
    assert core_texts.HELD_SURFACE.format(calls=surface) in status
    assert core_texts.HELD_HEAD in status and core_texts.HELD_FIXED.format(**words) in status
    held = {m.group(1): (int(m.group(4)), int(m.group(5))) for m in HELD_SERIES.finditer(status)}
    assert held == {"g1": (2, 10), "g2": (2, 10), "g3": (0, 10)}  # g2 covers g1, so its rows hold g1's cycles too
    assert core_texts.REPEATED.format(calls=core_texts.REPEATED_ITEM.format(call=call_text("read", {"sensor_group": "g2", "cycles": 2}), count=2)) in status
    lines = status.split("\n")
    assert lines.index(core_texts.HELD_HEAD) > next(i for i, line in enumerate(lines) if line.startswith("Evidence so far"))


def test_on_a_clocked_row_the_status_says_neither_that_the_surface_answers_hold_nor_that_the_record_stays():
    board = _board(window=(1, 10), clock_runs=True)
    board.note_answered(SURFACE_KINDS[-1], {})
    status = _status(board)
    assert core_texts.HELD_HEAD in status and len(HELD_SERIES.findall(status)) == len(GROUPS)
    assert core_texts.HELD_FIXED.format(**KIND_WORDS) not in status and "Answered, and unchanged" not in status


def test_a_surface_without_a_recorded_window_prints_no_line_of_the_record():
    board = _board()
    _row(board, "g1", 3)
    status = _status(board)
    assert core_texts.HELD_HEAD not in status and not HELD_SERIES.findall(status)
    assert core_texts.HELD_FIXED.format(**KIND_WORDS) not in status
    assert "On the board for the whole episode" not in status and "Called more than once" not in status  # nothing answered, nothing repeated
    # the same board once the surface states both ends of its recorded window prints them
    board.status.update(recorded_from=1, recorded_until=3)
    assert core_texts.HELD_HEAD in _status(board) and len(HELD_SERIES.findall(_status(board))) == len(GROUPS)


def _generated_boards(count=400, seed=7):
    """Boards with a random recorded window, random cycles that hold no record of a group, and random rows and collapsed rows of
    the three groups, inside the window and around it, on a row without a clock."""
    rng = random.Random(seed)
    for _ in range(count):
        first = rng.randint(-3, 5)
        last = first + rng.randint(0, 8)
        span = range(first - 2, last + 3)
        unrecorded = {g: rng.sample(list(span), rng.randint(0, 3)) for g in GROUPS if rng.random() < 0.5}
        board = _board(window=(first, last), unrecorded=unrecorded)
        for group in GROUPS:
            for cycle in rng.sample(list(span), rng.randint(0, len(span))):
                _row(board, group, cycle)
            if rng.random() < 0.3:
                board.summaries.append({"sensor_group": group, "cycles": rng.sample(list(span), rng.randint(1, 4))})
        yield board


def _property_breaks(boards, held_of):
    """The boards on which held_of(board, group, held, recorded) gives a count that is above recorded, or that equals it when the
    read of the group still has a cycle to pay for, or that falls short of it when the read has none (Board.new_cycles)."""
    broken = 0
    for board in boards:
        for m in HELD_SERIES.finditer(_status(board)):
            group, held, recorded = m.group(1), *held_of(int(m.group(4)), int(m.group(5)))
            nothing_new = board.new_cycles(group, 1) == 0 and not board.clock_runs
            broken += held > recorded or (held == recorded) != nothing_new
    return broken


def test_the_held_count_reaches_the_recorded_count_exactly_when_a_read_of_the_group_would_return_nothing_new():
    boards = list(_generated_boards())
    assert _property_breaks(boards, lambda held, recorded: (held, recorded)) == 0
    # the control: a count off by one breaks the property on some generated board
    assert _property_breaks(boards, lambda held, recorded: (held, recorded - 1)) > 0
    assert _property_breaks(boards, lambda held, recorded: (held + 1, recorded)) > 0


# ----- the notes -----

def test_every_tie_takes_its_note_off_at_its_change_of_state_and_not_before():
    board = Board()
    board.add_note("A plan before the facts was refused.")
    board.set_ranking({"candidates": []})
    assert board.notes == ["A plan before the facts was refused."]  # another state passing leaves it
    board.set_profile({"components": ["c1"], "sensor_groups": dict(GROUPS), "healthy": {"c1": 0}})
    assert board.notes == [] and board.retired_notes == 1
    board.add_note("The orientation reply could not be read.", tie="differential")
    board.add_note("the ranking failed: x", tie="ranking")
    board.add_note("the verification of pred/1 failed: x", tie=("verification", "pred/1"))
    board.add_note("2 cited identifier(s) not on the board were dropped: a, b.")
    board.add_verification({"prediction_id": "pred/2"})
    board.set_differential([])
    assert board.notes == ["the ranking failed: x", "the verification of pred/1 failed: x", "2 cited identifier(s) not on the board were dropped: a, b."]
    board.set_ranking({"candidates": []})
    board.add_verification({"prediction_id": "pred/1"})
    assert board.notes == ["2 cited identifier(s) not on the board were dropped: a, b."] and board.retired_notes == 4
    board.add_note("2 cited identifier(s) not on the board were dropped: a, b.")
    assert len(board.notes) == 1  # a note equal to one on the board is not added twice


def _refused(board, turn, reasons="step 2 (read): no sensor group g9"):
    board.mark_turn(turn)
    board.add_note(f"Your plan of turn {turn} was refused: {reasons}.", turn=turn)


def test_equal_notes_of_plans_stand_once_with_their_count_and_different_ones_stay():
    board = _board()
    for turn in (1, 2, 3):
        _refused(board, turn)
    assert board.notes == ["Your plan of turn 3 was refused: step 2 (read): no sensor group g9."
                           + core_texts.NOTE_REPEATED.format(count=2)] and board.retired_notes == 2
    _refused(board, 4, "step 1 (order): no component c9")
    _refused(board, 5)
    assert board.notes == ["Your plan of turn 4 was refused: step 1 (order): no component c9.",
                           "Your plan of turn 5 was refused: step 2 (read): no sensor group g9." + core_texts.NOTE_REPEATED.format(count=3)]


def _notes_line_without_digits(board):
    return re.sub(r"\d+", "", board._notes_line())


def test_the_notes_line_stops_growing_once_the_reasons_of_the_plans_repeat():
    board = _board()
    lines = []
    for turn in range(1, 13):
        _refused(board, turn)
        lines.append(_notes_line_without_digits(board))
    assert len(board.notes) == 1 and len(set(lines[1:])) == 1
    # the control: notes appended as they come, as before, make the line grow every turn
    plain = _board()
    grown = []
    for turn in range(1, 13):
        plain.mark_turn(turn)
        plain.notes.append(f"Your plan of turn {turn} was refused: step 2 (read): no sensor group g9.")
        grown.append(len(_notes_line_without_digits(plain)))
    assert all(a < b for a, b in zip(grown, grown[1:]))


def test_the_order_of_the_replaceable_entries_stays_exact_after_a_retirement():
    board = _board()
    board.profile = None  # the first note is written before the facts
    board.mark_turn(1)
    board.add_note("A plan before the facts was refused.")
    board.profile = {"components": ["c1"], "sensor_groups": dict(GROUPS), "healthy": {"c1": 0}}
    board.mark_turn(2)
    board.add_note("note of turn 2")
    board.mark_turn(3)
    board.add_note("note of turn 3")
    naive = Board(notes=list(board.notes), turn_marks=[(t, dict(l)) for t, l in board.turn_marks])
    board.retire_notes("facts")
    assert [(board._entry_text(e), e["turn"]) for e in board.replaceable()] == [("note of turn 2", 2), ("note of turn 3", 3)]
    # the control: the same note taken off without lowering the turn marks dates the later notes a turn too early
    naive.notes.pop(0)
    assert [(naive._entry_text(e), e["turn"]) for e in naive.replaceable()] == [("note of turn 2", 1), ("note of turn 3", 2)]


def test_a_note_that_a_summary_pinned_is_not_retired():
    board = Board()
    board.add_note("A plan before the facts named obs/1/x.")
    board.add_note("Another plan before the facts was refused.")
    board.pinned = [["notes", "A plan before the facts named obs/1/x."]]
    board.set_profile({"components": ["c1"], "sensor_groups": dict(GROUPS), "healthy": {"c1": 0}})
    assert board.notes == ["A plan before the facts named obs/1/x."] and board.retired_notes == 1


def test_the_record_of_the_board_holds_its_four_new_keys():
    board = _board()
    board.note_answered("profile", {})
    board.note_call("read", {"sensor_group": "g1", "cycles": 1})
    board.reply_schema = {"plan": "0" * 16, "differential": "1" * 16, "bound": 2, "carried": True}
    record = board.to_json()
    assert {k: record[k] for k in ("answered", "calls", "retired_notes", "reply_schema")} == {
        "answered": [["profile", {}]], "calls": [["read", {"sensor_group": "g1", "cycles": 1}, 1]], "retired_notes": 0,
        "reply_schema": {"plan": "0" * 16, "differential": "1" * 16, "bound": 2, "carried": True}}


# ----- the line of a stall -----

def _turns(board, turns, between=lambda board, turn: None):
    """Start the planning turns one after another, with between(board, turn) done in each turn, and return the status of each."""
    seen = []
    for turn in range(1, turns + 1):
        board.mark_turn(turn)
        seen.append(_status(board) if board.profile is not None else board.render())
        between(board, turn)
    return seen


def _restate(board, turn):
    """A turn that only restates its plan, notes a refusal, adds a surprise and lets the clock run: no progress."""
    board.plan = {"branch": 1, "mode": "test_first", "why": f"again {turn}", "steps": [{"kind": "read", "args": {"sensor_group": "g1", "cycles": 1}}]}
    board.add_note(f"Your plan of turn {turn} was refused: the same reasons.", turn=turn)
    board.surprises.append({"plan": turn, "step": 1, "kind": "read", "reason": "x"})
    board.status["time"] = board.status.get("time", 0) + 1


def _stall(count):
    return NO_PROGRESS.format(count=count)


def test_two_turns_in_a_row_without_progress_print_the_line_with_their_count_and_a_restated_plan_is_no_progress():
    seen = _turns(_board(), 4, _restate)
    assert [_stall(n) in status for status, n in zip(seen, (0, 1, 2, 3))] == [False, False, True, True]
    assert NO_PROGRESS_TURNS == 2 and "No progress" not in seen[1]
    # the control: the same turns, each of which also adds an evidence row, never print it
    def reading(board, turn):
        _restate(board, turn)
        _row(board, "g1", turn)
    assert not any("No progress" in status for status in _turns(_board(), 4, reading))


def test_every_kind_of_progress_ends_the_stall():
    progress = {
        "an evidence row": lambda board: _row(board, "g3", 50),
        "a notification": lambda board: board.add_notifications([{"event": "alarm", "channel": "C1", "t": 2}]),
        "an answer of the opening": lambda board: board.note_answered("fix_history", {}),
        "a commitment": lambda board: setattr(board, "diagnosis", {"diagnosis": {"c1": 1}}),
        "a work order": lambda board: board.orders.update({"wo-1": {"id": "wo-1", "status": "open"}}),
        "a verdict": lambda board: board.verifications.update({"pred/1": {"status": "verified"}}),
    }
    for name, act in progress.items():
        board = _board()
        _turns(board, 3, _restate)
        assert _stall(2) in _status(board), name
        act(board)
        board.mark_turn(4)
        assert "No progress" not in _status(board), name


def test_the_board_before_the_facts_prints_the_line_beside_the_notes():
    board = Board()
    seen = _turns(board, 3, lambda board, turn: board.add_note(f"Your plan of turn {turn} was refused: no profile step.", turn=turn))
    assert "No progress" not in seen[1] and _stall(2) in seen[2]
    lines = seen[2].split("\n")
    assert lines.index(_stall(2)) == len(lines) - 2 and lines[-1].startswith("Controller notes: ")


def test_the_limits_line_says_that_no_clock_runs_on_a_row_with_a_horizon_and_no_clock():
    stopped = _board()
    stopped.profile["horizon"] = 60
    assert stopped.limits_line().startswith("Time 0 of 60 cycles (60 left, no clock runs).")
    running = _board(clock_runs=True)
    running.profile["horizon"] = 60
    assert running.limits_line().startswith("Time 0 of 60 cycles (60 left).")
    assert _board().limits_line().startswith("Time 0, no horizon (no clock runs).")


# ----- the evidence in one row per value, and the note on candidates that were not kept -----

def _ranged_board():
    board = _board()
    board.profile.update(levels={"c1": [0, 1]}, normal_ranges={"A1_mean": {"low": 1.0, "high": 2.0}, "B1_mean": {"low": 0.0, "high": 1.0}})
    return board


def _reading(board, n, group, values):
    board.add_reading({"sensor_group": group, "served": [{"n": n, "t": n, "values": values}]})


def test_the_evidence_holds_one_row_per_value_with_its_mark_and_every_normal_range_once_in_a_legend():
    """A value outside its normal range ends in low or high, the normal range of every channel stands once in a legend above the
    rows, and a later reading only adds rows at the end, so two requests share every row the earlier one held; a recalled row
    takes the same form."""
    board = _ranged_board()
    _reading(board, 1, "g1", {"A1_mean": 0.5, "A1_std": 0.1})
    first = board._render_evidence()
    _reading(board, 2, "g2", {"A1_mean": 2.5, "B1_mean": 0.5})
    second = board._render_evidence()
    assert second.startswith(first)  # the rows grow at the end and the legend stays as it was
    assert second.count("| A1_mean | 1.000 to 2.000 |") == second.count("| B1_mean | 0.000 to 1.000 |") == 1
    rows = ["| obs/1/A1_mean | 1 | g1 | 0.500 low |", "| obs/1/A1_std | 1 | g1 | 0.100 |", "| obs/2/A1_mean | 2 | g2 | 2.500 high |",
            "| obs/2/B1_mean | 2 | g2 | 0.500 |"]
    assert all(row in second for row in rows) and "in normal range |" not in second
    board.mark_turn(1)
    board.recalled, board.recall_turn = [dict(board.evidence[0])], 1
    assert "| obs/1/A1_mean | 1 | g1 | 0.500 low |" in board._render_recalled() and "in normal range |" not in board._render_recalled()


def test_the_note_on_candidates_not_kept_stays_past_the_facts_and_leaves_once_a_plan_lists_candidates():
    board = Board()
    kept = "The candidates of your plan of turn 1 were not kept, since no profile was on the board; list them again in your next plan."
    board.add_note(kept, tie="candidates")
    board.add_note("Your plan of turn 1 was refused: no profile step.", turn=1)
    board.set_profile(_ranged_board().profile)
    assert board.notes == [kept]  # the refusal of the opening spoke of the facts and left with them
    board.set_ranking({"candidates": []})
    assert board.notes == [kept]
    board.set_candidates([{"levels": {"c1": 1}, "failure_mode_ids": []}])
    assert board.notes == [] and board.retired_notes == 2


def test_repeated_lists_the_calls_made_more_than_once_and_the_status_line_reads_it(monkeypatch):
    board = _board()
    for fields in ({"sensor_group": "g2", "cycles": 2}, {"sensor_group": "g2", "cycles": 2}, {"sensor_group": "g2", "cycles": 1},
                   {"sensor_group": "g3", "cycles": 1}):
        board.note_call("read", fields)
    assert board.repeated() == [("read", {"sensor_group": "g2", "cycles": 2}, 2)]  # a call with other arguments is no repeat
    line = core_texts.REPEATED.format(calls=core_texts.REPEATED_ITEM.format(call=call_text("read", {"sensor_group": "g2", "cycles": 2}), count=2))
    assert line in _status(board)
    monkeypatch.setattr(Board, "repeated", lambda self: [])  # the control: the line reads repeated() and no other record
    assert core_texts.REPEATED.split("{")[0] not in _status(board)
