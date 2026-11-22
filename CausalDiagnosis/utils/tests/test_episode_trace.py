"""The trace of an episode (CausalDiagnosis.utils.episode_trace): a record takes a start line and an end line that holds what it
gained or changed, also inside a nested value, the merge of the two is the record that the end file holds, and nothing is written
outside an episode."""
import json

import pytest

from CausalDiagnosis.utils import episode_trace


def test_a_record_takes_a_start_line_and_an_end_line_and_their_merge_is_the_record_of_the_end_file(tmp_path):
    """A call that answers and a call that raises each take a start line with the request and an end line with what the record
    gained, so a prompt is written once, and records() gives back the list that llm_calls.json is written from."""
    calls = []
    with episode_trace.bound(tmp_path):
        with episode_trace.recorded(episode_trace.MODEL_CALL, calls, {"prompt": "a long prompt", "started_at": 1.0, "cached_tokens": None}):
            calls[-1].update(reply="ok", cached_tokens=4, ended_at=2.0)
        with pytest.raises(RuntimeError):
            with episode_trace.recorded(episode_trace.MODEL_CALL, calls, {"prompt": "another prompt", "started_at": 3.0}):
                calls[-1]["error"] = "RuntimeError: down"
                raise RuntimeError("down")
    lines = [json.loads(line) for line in (tmp_path / episode_trace.NAME).read_text(encoding="utf-8").splitlines()]
    assert [(line["record"], line["index"], line["part"]) for line in lines] == [
        ("model_call", 0, "start"), ("model_call", 0, "end"), ("model_call", 1, "start"), ("model_call", 1, "end")]
    assert lines[1]["fields"] == {"reply": "ok", "cached_tokens": 4, "ended_at": 2.0} and lines[3]["fields"] == {"error": "RuntimeError: down"}
    assert episode_trace.records(tmp_path) == {"model_call": calls}


def test_a_nested_value_that_the_record_changes_in_place_enters_the_end_line(tmp_path):
    """A list and a dict inside a record that the block changes in place, as a plan marks and adds its steps, enter the end line,
    while a nested value and a field that stay as they started do not, so records() gives back the record as it ended."""
    plans = []
    start = {"index": 1, "steps": [{"status": "pending"}], "usage": {"tokens": 1}, "plan": {"steps": [{"kind": "read"}]}}
    with episode_trace.bound(tmp_path):
        with episode_trace.recorded(episode_trace.PLAN, plans, start):
            plans[-1]["steps"][0]["status"] = "done"
            plans[-1]["steps"].append({"status": "pending"})
            plans[-1]["usage"]["tokens"] = 2
    lines = [json.loads(line) for line in (tmp_path / episode_trace.NAME).read_text(encoding="utf-8").splitlines()]
    assert lines[1]["part"] == "end" and lines[1]["fields"] == {"steps": [{"status": "done"}, {"status": "pending"}], "usage": {"tokens": 2}}
    assert episode_trace.records(tmp_path) == {"plan": plans}


def test_outside_an_episode_nothing_is_written_and_a_line_that_a_kill_cut_short_is_left_out(tmp_path):
    """The control: a record outside a binding enters its list and no trace. A record with an end line alone, as a failed token
    count is written at once, keeps every field, a plan with its start line alone keeps its decision, and a last line without its
    newline, as a kill in the middle of a write leaves it, is not read."""
    calls = []
    with episode_trace.recorded(episode_trace.MODEL_CALL, calls, {"prompt": "p"}):
        pass
    assert calls == [{"prompt": "p"}] and not (tmp_path / episode_trace.NAME).exists() and episode_trace.records(tmp_path) == {}
    with episode_trace.bound(tmp_path):
        episode_trace.started(episode_trace.PLAN, 0, {"index": 1, "plan": {"steps": [{"kind": "read"}]}, "steps": []})
        episode_trace.ended(episode_trace.MODEL_CALL, 0, {"kind": "tokenize", "prompt": "count me", "error": "HTTPStatusError: 500"})
    with (tmp_path / episode_trace.NAME).open("a", encoding="utf-8") as trace:
        trace.write('{"record": "plan", "index": 0, "part": "end", "fields": {"steps": [{"status": "do')
    assert episode_trace.records(tmp_path) == {"plan": [{"index": 1, "plan": {"steps": [{"kind": "read"}]}, "steps": []}],
                                               "model_call": [{"kind": "tokenize", "prompt": "count me", "error": "HTTPStatusError: 500"}]}
