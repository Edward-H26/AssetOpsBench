"""Compaction over generic observation rows and public board entries."""

import copy
import json
import re


PLACE = {"notifications": 0, "surprises": 2, "notes": 3}
NO_SUMMARY = "(no summary available)"


def names(text, identifier):
    """Match an entire identifier rather than a prefix of another identifier."""
    return re.search(rf"(?<![\w/.-]){re.escape(identifier)}(?![\w/-]|\.\w)", text or "") is not None


def entry_text(board, entry):
    value = getattr(board, entry["kind"])[entry["index"]]
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def note_key(text, tie):
    """The turn prefix is metadata; numbers inside a reason remain significant."""
    reason = re.sub(r"^(?:turn|plan) \d+:\s*", "", text, flags=re.IGNORECASE)
    return json.dumps([reason, tie], sort_keys=True, ensure_ascii=False)


def token(board, entry):
    value = getattr(board, entry["kind"])[entry["index"]]
    key = note_key(value["text"], value["tie"]) if entry["kind"] == "notes" else entry_text(board, entry)
    return entry["kind"], key


def replaceable(board):
    entries = []
    for kind in PLACE:
        limit = len(getattr(board, kind)) - (1 if kind == "surprises" else 0)
        for index in range(limit):
            entry = {"kind": kind, "index": index}
            if token(board, entry) in board.pinned:
                continue
            turn = 0
            for marked, lengths in board.turn_marks:
                if lengths[kind] <= index:
                    turn = marked
            entries.append({**entry, "turn": turn})
    return sorted(entries, key=lambda entry: (entry["turn"], PLACE[entry["kind"]], entry["index"]))


def remove(board, entries):
    """Remove list entries and shift every turn boundary by the same removals."""
    removed = {}
    for kind in PLACE:
        gone = {entry["index"] for entry in entries if entry["kind"] == kind}
        old = getattr(board, kind)
        gone = {index for index in gone if 0 <= index < len(old)}
        setattr(board, kind, [item for index, item in enumerate(old) if index not in gone])
        for _turn, lengths in board.turn_marks:
            lengths[kind] -= sum(index < lengths[kind] for index in gone)
        removed[kind] = len(gone)
    return removed


def collapse(board, rows):
    taken = {row["id"] for row in rows}
    board.collapsed_rows.extend(copy.deepcopy(rows))
    board.observations = [row for row in board.observations if row["id"] not in taken]
    changed = []
    for row in sorted(rows, key=lambda item: (item["n"], item["id"])):
        last = board.summaries[-1] if board.summaries else None
        series = row.get("series")
        if last is None or last["series"] != series or row["n"] not in (last["last"], last["last"] + 1):
            last = {"series": series, "first": row["n"], "last": row["n"], "ids": [], "outside": [], "words": {}}
            board.summaries.append(last)
        if not any(summary is last for summary in changed):
            changed.append(last)
        last["last"] = row["n"]
        last["ids"].append(row["id"])
        if row.get("inside_band") is False:
            last["outside"].append(row["id"])
        if row.get("words"):
            last["words"][row["id"]] = list(row["words"])
    return [{"kind": "observations", "series": summary["series"],
             "observations": [summary["first"], summary["last"]], "values": len(summary["ids"])} for summary in changed]


def compact(board, fits, after_call, summarize, fits_context):
    fits = fits or (lambda board: True)
    changes = []
    latest = max((row["n"] for row in board.observations), default=0)
    older = [row for row in board.observations if row["n"] <= board.absorbed and row["n"] < latest]
    if older and (board.absorption_by_advice or not fits(board)):
        changes.extend(collapse(board, older))
    if not fits(board):
        rest = [row for row in board.observations if row["n"] <= board.absorbed]
        if rest:
            changes.extend(collapse(board, rest))
    if not fits(board) and summarize is not None and board.options.summary_turn:
        entries = board.replaceable()
        if entries and (fits(board.without(entries)) or (fits_context is not None and not fits_context(board))):
            written = summarize(board, entries)
            if written is not None:
                text, left_out = written
                changes.append(board.apply_summary(text, entries, left_out))
    if changes:
        board.compactions += 1
        board.compaction_log.append({"after_call": after_call, "collapsed": copy.deepcopy(changes),
                                     "rendered_characters": len(board.render(()))})
    return changes


def apply_summary(board, text, entries, left_out):
    named = {identifier for identifier in board.known_ids() if names(text, identifier)}
    kept = [entry for entry in entries if any(names(entry_text(board, entry), identifier) for identifier in named)]
    board.pinned.update(token(board, entry) for entry in kept)
    removed = remove(board, [entry for entry in entries if entry not in kept])
    board.summary_text = (text or "").strip() or NO_SUMMARY
    board.summary_turn = board.turn
    lost, kept_named = {}, {}
    for entry in entries[:left_out]:
        if entry not in kept:
            lost[entry["kind"]] = lost.get(entry["kind"], 0) + 1
    for entry in kept:
        kept_named[entry["kind"]] = kept_named.get(entry["kind"], 0) + 1
    return {"kind": "summary", "replaced": removed, "left_out_of_summary_request": lost,
            "kept_for_named_identifiers": kept_named, "summary_characters": len(board.summary_text)}
