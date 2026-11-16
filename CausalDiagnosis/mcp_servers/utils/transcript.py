"""The transcript of an episode: one JSON line per event that a server of the episode served, in the order the servers wrote them
(CausalDiagnosis/docs/experiments.md).

Every server process of an episode appends to transcript.jsonl in the episode folder under one lock: the start of the episode,
the handover of the task to the agent, the ready line of every server process, each tool listing a client asked for, the entry
and the exit of every served call of a tool that steps the environment, and each tool call, answered, refused or failed. A call line holds the server, the tool,
the arguments as the server received them, whether the call took a step of the environment, and the SHA-256 of the exact text
that went back to the client, so a replay through the servers of another checkout compares every answer while the file holds
none. A server that only reads the released evidence also keeps the text of its answer, which no other file of a server holds.
The file is written by server code alone, never by an agent, a runner or the driver, and the run ledger hashes it with the other
files of the episode folder. The module imports nothing of the MCP package, since the episode module of the action server
imports it and stays importable without that package.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from CausalDiagnosis.utils.atomic_writes_and_locks import locked

NAME = "transcript.jsonl"
FORMAT = 1
# what the call that this task is serving has recorded and noted so far (serving), None outside a served call
_CALL = ContextVar("transcript_call", default=None)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    """The SHA-256 of a file, None for a file that is not there."""
    path = Path(path)
    return bytes_sha256(path.read_bytes()) if path.is_file() else None


def listing_sha256(tools):
    """The SHA-256 of a tool listing: the tool definitions in the form that travels to a client, as canonical JSON, so the
    server that sent a listing and the client that received it compute the same value."""
    listing = [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools]
    return text_sha256(json.dumps(listing, sort_keys=True, separators=(",", ":")))


def _path(folder):
    """The transcript of an episode folder, None without a folder or for a path that is no folder, as for a server that is
    built to list its tools before any episode exists."""
    if folder is None:
        return None
    folder = Path(folder)
    return folder / NAME if folder.is_dir() else None


def _line(n, kind, fields):
    # ASCII only, so that a line break is the only byte that ends a line whatever an answer text holds
    return json.dumps({"n": n, "kind": kind, "pid": os.getpid(), **fields}, separators=(",", ":")) + "\n"


def _count(path):
    if not path.is_file():
        return 0
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))


def begin(folder, **fields):
    """Start the transcript of a new episode in the folder, in place of one that is there: line 0, of the kind start."""
    path = _path(folder)
    if path is None:
        return None
    with locked(path):
        path.write_text(_line(0, "start", {"format": FORMAT, **fields}), encoding="utf-8")
    return 0


def append(folder, kind, **fields):
    """Append one line and return its number, None without an episode folder. The number is the count of the lines before it,
    read under the lock, so the lines of every server process of the episode are numbered in the order they were written."""
    path = _path(folder)
    if path is None:
        return None
    with locked(path):
        n = _count(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(_line(n, kind, fields))
    return n


def read(folder):
    """The lines of the transcript of an episode folder as dicts, an empty list without the file. A line that is no JSON
    object, or whose number is not its place, belongs to a torn or edited file and raises ValueError with its place."""
    path = Path(folder) / NAME
    if not path.is_file():
        return []
    lines = []
    pieces = path.read_text(encoding="utf-8").split("\n")
    if pieces.pop() != "":  # every line ends with a line feed, so text after the last one is an append that was cut off
        raise ValueError(f"{path}: line {len(pieces)} is torn or out of place")
    for place, text in enumerate(pieces):
        try:
            line = json.loads(text)
        except ValueError:
            line = None
        if not isinstance(line, dict) or line.get("n") != place:
            raise ValueError(f"{path}: line {place} is torn or out of place")
        lines.append(line)
    return lines


def read_since(folder, offset):
    """(lines, new offset): the complete lines of the transcript after the byte offset, as dicts, and the offset after the
    last of them, so that a reader that keeps the offset reads each line once. Text after the last line feed belongs to an
    append in progress and is left for the next read. A file that is not there gives ([], offset)."""
    path = Path(folder) / NAME
    if not path.is_file():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1
    return [json.loads(text) for text in data[:end].decode("utf-8").splitlines()], offset + end


def answered(folder, server, tool, arguments, text, is_error, stepped=False, keep_text=False, **fields):
    """Append the line of one tool call: the hash of the text that goes back to the client, whether the call took a step of
    the environment, the text itself for a server that asks for it, and the fields of the caller. Inside a served call the
    arguments are those of the request, as the server received them, and the call is told that its line is written."""
    call = _CALL.get()
    if call is not None:
        arguments = call["arguments"]
    line = {"server": server, "tool": tool, "arguments": arguments, "answer_sha256": text_sha256(text), "is_error": bool(is_error),
            "stepped": bool(stepped), **fields}
    if keep_text:
        line["answer"] = text
    n = append(folder, "call", **line)
    if call is not None:
        call["recorded"] = True
    return n


def note(**fields):
    """Fields for the line of the call being served, for code that learns them inside the call, such as the hash of the
    evidence that a read only server read. Outside a served call it does nothing."""
    call = _CALL.get()
    if call is not None:
        call["fields"].update(fields)


@contextmanager
def serving(arguments):
    """One served call while its handler runs: the arguments of the request, whether a line was written for the call
    (answered) and the fields noted for it (note)."""
    call = {"arguments": arguments, "recorded": False, "fields": {}}
    token = _CALL.set(call)
    try:
        yield call
    finally:
        _CALL.reset(token)
