"""Retained public answers and evidence, independent of asset fields."""

import copy
import json
import re
from dataclasses import asdict, is_dataclass

from CausalDiagnosis.agents.utils.answer_forms import observations as read_observations
from CausalDiagnosis.agents.utils.core_texts import HELD_EPISODE, HELD_SURFACE, REPEATED, REPEATED_ITEM
from ..plan.readers import copy_json
from . import compaction
from ..media import text_view


def _identifiers(value):
    found = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key in ("id", "evidence_id"):
                if isinstance(item.get(key), str) and item[key]:
                    found.add(item[key])
            for key in ("evidence_ids", "cites"):
                if isinstance(item.get(key), list):
                    found.update(identifier for identifier in item[key] if isinstance(identifier, str) and identifier)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return found


class Board:
    """The episode working memory and its public kind ledger."""

    facts = knowledge = observations = commitment = status = None
    reply_schema = None

    def __init__(self, table, ledger, question, options):
        """Create the board and its reply-schema record."""
        self.table, self._ledger, self.question, self.options = table, ledger, question, options
        self._facts = self._commitment = None
        self._knowledge, self._observations, self.calls = {}, [], []
        self._modules = ()
        self.model_candidates, self.notifications, self.notes, self.surprises = [], [], [], []
        self.differential = []
        self.plan = None
        self.turn = self.observation_count = 0
        self.absorbed = 0
        self.absorption_by_advice = False
        self.summaries, self.collapsed_rows, self.recalled, self.refuted = [], [], [], []
        self.recall_turn = self.summary_turn = self.summary_text = None
        self.turn_marks, self.compaction_log = [], []
        self.compactions = 0
        self.pinned = set()
        self._ids = set()
        self._status = copy_json(ledger.status())
        self.reply_schema = None
        self._progress = None
        self._idle_turns = 0


    @property
    def ledger(self):
        return self._ledger

    @property
    def facts(self):
        return self._facts

    @property
    def knowledge(self):
        return self._knowledge

    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, rows):
        self._observations = rows

    @property
    def commitment(self):
        return self._commitment

    @property
    def model_candidates(self):
        return self._model_candidates

    @model_candidates.setter
    def model_candidates(self, value):
        self._model_candidates = copy_json(value)
        if value and hasattr(self, "notes"):
            self.retire_notes("candidates")

    @property
    def differential(self):
        return self._differential

    @differential.setter
    def differential(self, value):
        self._differential = copy_json(value)
        if value and hasattr(self, "notes"):
            self.retire_notes("differential")

    @property
    def status(self):
        return self._ledger.status()

    def set_modules(self, modules):
        """Attach the modules whose candidates and public sections the board shows."""
        self._modules = tuple(modules)

    def record_call(self, step, call, answer, observations):
        """Record an answered step and its observations."""
        row = self.table.tools[step.tool]
        answer = copy_json(dict(answer))
        successful = "result" in answer and "refused" not in answer
        form_note = None
        rows = []
        if successful:
            if observations is None:
                if answer.get("observations"):
                    rows = copy_json(answer["observations"])
                else:
                    rows, form_note = read_observations(row.answer, answer["result"], self._ledger.ranges())
            else:
                rows = copy_json(observations)
            if form_note:
                self.add_note(f"{step.tool}: {form_note}")
            elif row.facts:
                self._facts = copy_json(answer["result"])
                self.retire_notes("facts")
            if row.role == "knowledge" and form_note is None:
                self._knowledge.setdefault(step.tool, []).append({"args": copy_json(step.args), "result": copy_json(answer["result"])})
                self._ids.update(_identifiers(answer["result"]))
            if row.commits and form_note is None:
                self._commitment = {"tool": step.tool, "args": copy_json(step.args), "answer": copy_json(answer["result"])}
            if row.advice and form_note is None:
                self.retire_notes(row.advice)
                self.retire_notes(("advice", row.advice))
        else:
            rows = []
        if rows:
            self.observation_count += 1
        retained = []
        for item in rows:
            item.setdefault("n", self.observation_count)
            if "text" in item:
                item.setdefault("id", f"obs/{item['n']}")
                item["words"] = [word for word in row.words if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", item["text"])]
            if row.series and row.series in step.args:
                item["series"] = copy_json(step.args[row.series])
                item[row.series] = copy_json(step.args[row.series])
            if isinstance(item.get("n"), int):
                self.observation_count = max(self.observation_count, item["n"])
            existing = next((old for old in (*self._observations, *self.collapsed_rows) if old.get("id") == item.get("id")), None)
            if existing is not None:
                if existing != item:
                    self.add_note(f"{step.tool}: observation {item.get('id')} conflicts with its retained value")
                continue
            self._observations.append(copy_json(item))
            retained.append(item)
            if isinstance(item.get("id"), str):
                self._ids.add(item["id"])
        answer["observations"] = retained
        ledger_answer = answer
        if form_note:
            ledger_answer = {key: value for key, value in answer.items() if key != "result"}
            ledger_answer["refused"] = f"{step.tool}: {form_note}"
            answer = ledger_answer
        self._ledger.record(row, copy_json(step.args), ledger_answer)
        self._status = copy_json(self._ledger.status())
        self.notifications.extend(copy_json(answer.get("notifications", [])))
        recorded = asdict(call) if is_dataclass(call) else copy_json(call)
        self.calls.append({"tool": step.tool, "args": copy_json(step.args), "call": recorded,
                           "answer": answer, "answered": successful and form_note is None})


    def candidates(self):
        """Return module candidates followed by model candidates."""
        return [copy_json(candidate) for module in self._modules for candidate in module.candidates()] + copy_json(self.model_candidates)


    def known_ids(self):
        """Return known evidence identifiers."""
        return set(self._ids).union(*(module.known_ids() for module in self._modules))

    def _parts(self, modules):
        for heading, value in self._raw_parts(modules):
            yield heading, text_view(value)

    def _raw_parts(self, modules):
        """Yield the same public parts in order for text and JSON readers."""
        yield "Task", self.question
        yield "Public facts", self._facts
        yield "Knowledge", self._knowledge
        yield "Summary of earlier turns", self.summary_text
        yield "Collapsed observations", self.summaries
        yield "Observations", self._observations
        yield "Recalled observations", self.recalled if self.recall_turn == self.turn else []
        yield "Candidates", self.candidates()
        yield "Differential", self.differential
        yield "Kind state", self._ledger.sections(self.table.declaration.vocabulary)
        yield "Advice", [section for module in modules for section in module.sections(self)]
        yield "Commitment", self._commitment
        yield "Refuted answers", self.refuted
        yield "Notifications", self.notifications
        yield "Surprises", self.surprises
        shown_notes = copy.deepcopy(self.notes)
        for note in shown_notes:
            earlier = len(note.get("turns", [note["turn"]])) - 1
            if earlier:
                note["text"] += f" (seen before; earlier turns with the same reasons: {earlier})"
        yield "Notes", shown_notes
        yield "Last plan", self.plan
        yield "Status", self._public_status()

    def _public_status(self):
        status = copy_json(self._ledger.status())
        retained = []
        for call in self.calls:
            row = self.table.tools[call["tool"]]
            item = {"tool": call["tool"], "args": call["args"]}
            if call["answered"] and (row.facts or row.role == "knowledge") and item not in retained:
                retained.append(copy_json(item))
        if retained:
            status["retained_answers"] = retained
            status["held_episode"] = HELD_EPISODE.format(calls=", ".join(
                f"{item['tool']}({item['args']})" for item in retained))
        repeated = self.repeated()
        if repeated:
            status["repeated_answers"] = repeated
            status["repeated_calls"] = REPEATED.format(calls=", ".join(
                REPEATED_ITEM.format(call=f"{item['tool']}({item['args']})", count=item["count"])
                for item in repeated))
        binding = self.table.declaration.binding
        if binding is not None and status.get("time") is None:
            needed = {name for tool in self.table.tools for name in binding.needs(tool)}
            status["surface_answers"] = [
                {"tool": call["tool"], "args": copy_json(call["args"])}
                for call in self.calls if call["answered"] and call["tool"] in needed
                and not self.table.tools[call["tool"]].facts
            ]
            if status["surface_answers"]:
                status["held_surface"] = HELD_SURFACE.format(calls=", ".join(
                    f"{item['tool']}({item['args']})" for item in status["surface_answers"]))
        if self._idle_turns >= 2:
            status["turns_without_progress"] = self._idle_turns
        return status


    def render(self, modules):
        """Render the board sections."""
        return "\n\n".join(f"## {heading}\n{json.dumps(value, ensure_ascii=False, allow_nan=False)}"
                            for heading, value in self._parts(tuple(modules)) if value is not None and value != [])


    def to_json(self, modules):
        """Serialize the board sections."""
        modules = tuple(modules)
        return copy.deepcopy({"question": self.question, "facts": self._facts, "knowledge": self._knowledge,
                          "differential": self.differential,
                          "observations": self._observations, "candidates": self.candidates(), "commitment": self._commitment,
                          "status": self._public_status(), "calls": self.calls, "notes": self.notes,
                          "notifications": self.notifications, "surprises": self.surprises, "plan": self.plan,
                          "reply_schema": self.reply_schema, "sections": list(self._parts(modules)),
                          "advice": [module.to_json() for module in modules], "summaries": self.summaries,
                          "collapsed_rows": self.collapsed_rows, "recalled": self.recalled, "recall_turn": self.recall_turn,
                          "refuted": self.refuted, "summary_text": self.summary_text, "summary_turn": self.summary_turn,
                          "compactions": self.compactions, "compaction_log": self.compaction_log,
                          "turn_marks": self.turn_marks, "pinned": sorted(self.pinned)})


    def compact(self, fits=None, after_call=None, summarize=None, fits_context=None):
        """Compact earlier observations using the existing policy."""
        return compaction.compact(self, fits, after_call, summarize, fits_context)

    def mark_absorbed(self, number, *, by_advice=False):
        """Mark observations already read by the latest plan or advice request."""
        self.absorbed = number
        self.absorption_by_advice = by_advice


    def mark_turn(self, turn):
        """Record a planning-turn boundary."""
        if turn == self.turn and self.turn_marks:
            return
        progress = self._progress_state()
        if self._progress is not None:
            self._idle_turns = self._idle_turns + 1 if progress == self._progress else 0
        self._progress = progress
        self.turn = turn
        self.turn_marks.append((turn, {kind: len(getattr(self, kind)) for kind in compaction.PLACE}))

    def _progress_state(self):
        """Ignore plans, notes, clocks and remaining limits when judging progress."""
        state = {key: value for key, value in self._ledger.status().items()
                 if key not in {"time", "time_limit", "budget_left", "steps_left"}}
        answered = [(call["tool"], call["args"]) for call in self.calls if call["answered"]]
        unique = []
        for call in answered:
            if call not in unique:
                unique.append(call)
        return copy.deepcopy((self._facts, unique, self._ids, self._commitment, self.refuted, self.notifications,
                              self._ledger.sections(self.table.declaration.vocabulary), state))


    def replaceable(self):
        """Return replaceable entries."""
        return compaction.replaceable(self)


    def without(self, entries):
        """Render the board without selected entries."""
        board = copy.deepcopy(self)
        compaction.remove(board, entries)
        return board


    def apply_summary(self, text, entries, left_out):
        """Replace entries with a summary."""
        return compaction.apply_summary(self, text, entries, left_out)


    def recallable(self, series, first, last):
        """Return whether the observation range can be recalled."""
        return copy.deepcopy([row for row in self.collapsed_rows
                              if (series is None or row.get("series") == series) and first <= row["n"] <= last])

    def latest_observations(self, series=None):
        """Return the latest rows of each selected series, including collapsed rows."""
        latest = {}
        for row in (*self.collapsed_rows, *self._observations):
            if series is not None and row.get("series") != series:
                continue
            key = json.dumps(row.get("series"), sort_keys=True)
            held = latest.get(key, [])
            if not held or row["n"] > held[0]["n"]:
                latest[key] = [row]
            elif row["n"] == held[0]["n"]:
                held.append(row)
        return copy.deepcopy([row for rows in latest.values() for row in rows])


    def recall(self, series, first, last):
        """Recall an observation range."""
        if not self.options.recall:
            return 0
        rows = self.recallable(series, first, last)
        turn = self.turn + 1
        retained = self.recalled if self.recall_turn == turn else []
        identifiers = {row["id"] for row in retained}
        added = [row for row in rows if row["id"] not in identifiers]
        self.recalled, self.recall_turn = retained + added, turn
        return len(added)


    def drop_unfit_recall(self, fits_context):
        """Drop recall rows that exceed the request context."""
        if self.recall_turn == self.turn and self.recalled and not fits_context(self):
            count = len(self.recalled)
            self.recalled = []
            self.compactions += 1
            self.compaction_log.append({"after_call": None, "collapsed": [{"kind": "recall_dropped", "rows": count}],
                                        "rendered_characters": len(self.render(()))})


    def holds_answer(self, tool, args):
        """Return whether an answer is retained, with None matching any arguments."""
        return any(call["answered"] and call["tool"] == tool and (args is None or call["args"] == args) for call in self.calls)


    def repeated(self):
        """Return calls made more than once, including refused calls."""
        repeated = []
        for call in self.calls:
            group = next((item for item in repeated if item["tool"] == call["tool"] and item["args"] == call["args"]), None)
            if group is None:
                repeated.append({"tool": call["tool"], "args": copy_json(call["args"]), "count": 1})
            else:
                group["count"] += 1
        return [item for item in repeated if item["count"] > 1]


    def add_note(self, text, tie=None, turn=None):
        """Add a note tied to public state."""
        note_turn = self.turn if turn is None else turn
        key = compaction.note_key(text, tie)
        note = next((item for item in self.notes if compaction.note_key(item["text"], item["tie"]) == key), None)
        if note is None:
            self.notes.append({"text": text, "tie": copy_json(tie), "count": 1, "turn": note_turn, "turns": [note_turn]})
        else:
            index = self.notes.index(note)
            pinned = ("notes", key) in self.pinned
            compaction.remove(self, [{"kind": "notes", "index": index}])
            turns = list(dict.fromkeys([*note.get("turns", [note["turn"]]), note_turn]))
            self.notes.append({"text": text, "tie": copy_json(tie), "count": note["count"] + 1,
                               "turn": note_turn, "turns": turns})
            if pinned:
                self.pinned.add(("notes", key))


    def retire_notes(self, passed):
        """Retire notes whose public state has passed."""
        gone = [{"kind": "notes", "index": index} for index, note in enumerate(self.notes)
                if note["tie"] is not None and note["tie"] == passed
                and ("notes", compaction.note_key(note["text"], note["tie"])) not in self.pinned]
        return compaction.remove(self, gone)["notes"]
