"""The trace of an episode: every model call and every plan of the CausalDiagnosis controller appended to trace.jsonl in the episode
folder while the episode plays, so that a run killed in the middle of an episode keeps them; llm_calls.json and plans.json stay the
files that the end of the episode writes, unchanged. A record takes a start line when it enters the list that its end file is
written from, the request of a model call or the decision of a plan, and an end line when it is complete, the reply or the error
of a call or the executed plan, however its block exits. A line is {"record": the kind, "index": the place of the record in that
list from 0, "part": "start" or "end", "fields": ...}; an end line holds the fields that the record gained or changed since its
start line, so a prompt is written once, and records() merges the two, which gives the record of the end file, and gives a record
that a kill cut off its start line alone. The driver binds the episode folder around the runner (bound); outside an episode
nothing is written. The binding is module-wide, so that a port that plays its steps in a thread reaches it, as the idle hook
(CausalDiagnosis.agents.utils.idle_hook) does."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path

from CausalDiagnosis.utils.atomic_writes_and_locks import locked

NAME = "trace.jsonl"
MODEL_CALL, PLAN = "model_call", "plan"  # the records of llm_calls.json and of the plans of plans.json

_FOLDER = None
_STARTS = {}  # (kind, index) -> the fields of its start line, in the bound episode


@contextmanager
def bound(folder):
    """Bind the folder of the episode the driver plays, for the time of the episode."""
    global _FOLDER
    before, _FOLDER = _FOLDER, folder
    try:
        yield
    finally:
        _FOLDER = before
        _STARTS.clear()


def _append(kind, index, part, fields):
    path = Path(_FOLDER) / NAME
    line = json.dumps({"record": kind, "index": index, "part": part, "fields": fields}, default=str)  # write_json's rule for values
    with locked(path):
        with path.open("a", encoding="utf-8") as trace:
            trace.write(line + "\n")


def started(kind, index, fields):
    """The start line of a record: its fields as they stand."""
    if _FOLDER is not None:
        # a deep copy, so that a list or a dict that the record changes in place differs from its start value in ended
        _STARTS[(kind, index)] = copy.deepcopy(fields)
        _append(kind, index, "start", fields)


def ended(kind, index, fields):
    """The end line of a record: the fields that it gained or changed since its start line, every field without one."""
    if _FOLDER is not None:
        start = _STARTS.pop((kind, index), {})
        _append(kind, index, "end", {key: value for key, value in fields.items() if key not in start or start[key] != value})


@contextmanager
def recorded(kind, records, record):
    """Append record to records, the list that the end file of its kind is written from, with its start line, and write its end
    line when the block exits, by a return or by an exception."""
    index = len(records)
    records.append(record)
    started(kind, index, record)
    try:
        yield
    finally:
        ended(kind, index, record)


def records(folder):
    """{kind: the merged records in the order of their index} of the trace of an episode folder, {} without one. Only whole lines
    are read, so a line that a kill cut short is left out."""
    path = Path(folder) / NAME
    if not path.is_file():
        return {}
    data = path.read_bytes()
    merged = {}
    for line in data[:data.rfind(b"\n") + 1].decode("utf-8").splitlines():
        entry = json.loads(line)
        merged.setdefault(entry["record"], {}).setdefault(entry["index"], {}).update(entry["fields"])
    return {kind: [by_index[i] for i in sorted(by_index)] for kind, by_index in merged.items()}
