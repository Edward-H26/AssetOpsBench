"""The ledger of the failed attempts of model requests in one episode, whose spans the thinking clock does not charge. The
benchmark's model client (CausalDiagnosis.utils.model_request_retry) appends one line per failed attempt: the span from the start of the failed attempt to the
start of the next one, the registered wait included, in monotonic seconds, with the status that failed it. The servers read
the spans appended since their last counted call (failed_spans_since) and credit them as no thinking of the agent. No import of
the MCP package, so that the client and the servers import this module alike."""
import json
from pathlib import Path

from CausalDiagnosis.utils.atomic_writes_and_locks import locked

NAME = "model_failures.jsonl"


def add_failed(folder, start_mono, next_start_mono, status):
    """Append one failed attempt to the ledger of an episode folder, under the lock of the file."""
    path = Path(folder) / NAME
    with locked(path):
        with path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps({"start": float(start_mono), "next_start": float(next_start_mono), "status": status}) + "\n")


def failed_spans_since(folder, cursor):
    """(spans, new cursor): the (start, next_start) pairs appended since the byte offset cursor, in file order. Cursor 0 reads
    the ledger from its start; a missing ledger gives ([], 0). Only whole lines are read, so a later call reads a line that was
    being written."""
    path = Path(folder) / NAME
    if not path.is_file():
        return [], 0
    with locked(path):
        with path.open("rb") as ledger:
            ledger.seek(cursor)
            data = ledger.read()
    whole = data[:data.rfind(b"\n") + 1]
    spans = [(entry["start"], entry["next_start"]) for entry in map(json.loads, whole.decode("utf-8").splitlines()) if entry]
    return spans, cursor + len(whole)
