"""Read local episode facts and questions without publishing reference answers."""

import copy
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path


class DataError(ValueError):
    """An input record that cannot be used by the adapter."""


def read_jsonl(path):
    """Yield objects, refusing malformed input with its source location."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_nonfinite)
            except ValueError as error:
                raise DataError(f"{path}, record {number}: {error}") from error
            if not isinstance(value, dict):
                raise DataError(f"{path}, record {number}: expected an object")
            yield value


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _identifier(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"{key} must be a nonempty string")
    return value


def _index(records, key):
    found = {}
    for record in records:
        name = _identifier(record, key)
        if name in found:
            raise DataError(f"duplicate {key}: {name}")
        found[name] = copy.deepcopy(record)
    if not found:
        raise DataError(f"no records with {key}")
    return found


def sqlite_facts(path):
    """Read the upstream facts table without creating or changing a database."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute("SELECT fact_id, fact_json FROM facts ORDER BY fact_id")
            for fact_id, text in rows:
                fact = json.loads(text, parse_constant=_nonfinite)
                if not isinstance(fact, dict) or fact.get("fact_id") != fact_id:
                    raise DataError(f"database fact_id disagrees with fact_json: {fact_id}")
                yield fact
    except (sqlite3.Error, ValueError) as error:
        raise DataError(f"{path}: {error}") from error


class QADataset:
    """Private records and explicit public projections for one local QA set."""

    def __init__(self, facts, questions):
        self._facts = _index(facts, "fact_id")
        self._questions = _index(questions, "qa_id")
        for fact_id, fact in self._facts.items():
            features = fact.get("features")
            if not isinstance(features, list) or not features:
                raise DataError(f"{fact_id}: features must be a nonempty list")
            names = []
            for feature in features:
                if not isinstance(feature, dict) or "value" not in feature:
                    raise DataError(f"{fact_id}: each feature needs a name and value")
                value = feature["value"]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or isinstance(value, float) and not math.isfinite(value)):
                    raise DataError(f"{fact_id}: feature values must be finite numbers")
                names.append(_identifier(feature, "name"))
            if len(set(names)) != len(names):
                raise DataError(f"{fact_id}: duplicate feature name")
        for qa_id, question in self._questions.items():
            fact_id = _identifier(question, "fact_id")
            if fact_id not in self._facts:
                raise DataError(f"{qa_id}: missing fact {fact_id}")
            _identifier(question, "question")
            if question.get("task_type") != "descriptive":
                raise DataError(f"{qa_id}: only descriptive QA is supported")

    @classmethod
    def load(cls, facts_path, questions_path, *, sqlite=False):
        source = sqlite_facts(facts_path) if sqlite else read_jsonl(facts_path)
        return cls(source, read_jsonl(questions_path))

    def question_ids(self):
        return tuple(self._questions)

    def task(self, qa_id):
        """Publish the question and its lookup key, with no reference fields."""
        question = self._questions[qa_id]
        return copy.deepcopy({key: question[key] for key in ("qa_id", "fact_id", "task_type", "question")})

    def public_fact(self, qa_id):
        """Publish measurements and source metadata through a field allowlist."""
        question = self._questions[qa_id]
        fact = self._facts[question["fact_id"]]
        keys = ("fact_id", "dataset", "asset_id", "source_file", "row_index", "start_time", "end_time")
        result = {key: copy.deepcopy(fact[key]) for key in keys if key in fact}
        result["features"] = [{key: copy.deepcopy(feature[key]) for key in ("name", "value", "unit") if key in feature}
                              for feature in fact["features"]]
        return result

    def reference(self, qa_id):
        """Return a private evaluator record, never a tool response."""
        return copy.deepcopy(self._questions[qa_id])
