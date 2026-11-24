"""Controls for evidence isolation, input integrity and terminal submission."""

import copy
import io
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.cli import evaluate
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.build import build_descriptive
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.data import DataError, QADataset, read_jsonl
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.download import FILES, download_raw
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.episode import QAEpisode
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.evaluation import check_answer, direct_numeric_match


def records():
    fact = {"fact_id": "fact_a", "asset_id": "asset_a", "source_file": "measurements.csv", "row_index": 7,
            "features": [{"name": "reading_mean", "value": 12.5, "unit": "unit", "private_note": "hidden"}],
            "label": "private_label", "failure_component": "private_component",
            "provenance": {"failure_index": 7}, "unknown_private_field": "hidden"}
    question = {"qa_id": "question_a", "fact_id": "fact_a", "task_type": "descriptive",
                "question": "What was the average reading?", "direct_answer": "private_reference",
                "reasoning_answer": "private_reasoning", "label": "private_label",
                "provenance": {"features": ["reading_mean"]}}
    answer = {"direct_answer": "The average was 12.5.", "reasoning_answer": "The released mean is 12.5.",
              "confidence": 0.9, "provenance": {"fact_id": "fact_a", "features": ["reading_mean"],
                                                "file": "measurements.csv", "row": 7}}
    return fact, question, answer


class QATests(unittest.TestCase):
    def setUp(self):
        self.fact, self.question, self.answer = records()
        self.dataset = QADataset([self.fact], [self.question])

    def test_public_projections_exclude_private_and_unknown_fields(self):
        public = json.dumps({"task": self.dataset.task("question_a"), "fact": self.dataset.public_fact("question_a")})
        self.assertNotIn("private", public)
        self.assertNotIn("hidden", public)
        self.assertNotIn("failure_index", public)
        self.assertIn("reading_mean", public)
        self.assertIn("private_reference", json.dumps(self.dataset.reference("question_a")))

    def test_mutation_of_input_and_returned_fact_cannot_change_evidence(self):
        self.fact["features"][0]["value"] = 0
        public = self.dataset.public_fact("question_a")
        public["features"][0]["value"] = 1
        self.assertEqual(self.dataset.public_fact("question_a")["features"][0]["value"], 12.5)

    def test_duplicate_identifiers_and_missing_references_are_refused(self):
        for facts, questions in (([self.fact, self.fact], [self.question]),
                                 ([self.fact], [self.question, self.question])):
            with self.assertRaises(DataError):
                QADataset(facts, questions)
        self.question["fact_id"] = "absent"
        with self.assertRaisesRegex(DataError, "missing fact"):
            QADataset([self.fact], [self.question])

    def test_unsupported_task_is_refused_instead_of_reinterpreted(self):
        self.question["task_type"] = "counterfactual"
        with self.assertRaisesRegex(DataError, "only descriptive"):
            QADataset([self.fact], [self.question])

    def test_duplicate_features_are_refused(self):
        self.fact["features"].append(copy.deepcopy(self.fact["features"][0]))
        with self.assertRaisesRegex(DataError, "duplicate feature"):
            QADataset([self.fact], [self.question])

    def test_nested_private_and_nonfinite_feature_values_are_refused(self):
        for value in ({"label": "private"}, float("inf"), True, None):
            self.fact["features"][0]["value"] = value
            with self.assertRaisesRegex(DataError, "finite numbers"):
                QADataset([self.fact], [self.question])

    def test_submission_requires_read_and_only_accepts_released_citations(self):
        episode = QAEpisode(self.dataset, "question_a")
        with self.assertRaisesRegex(ValueError, "read_fact"):
            episode.submit_answer(self.answer)
        episode.read_fact()
        invalid = copy.deepcopy(self.answer)
        invalid["provenance"]["features"] = ["invented"]
        self.assertFalse(episode.submit_answer(invalid)["accepted"])
        self.assertFalse(episode.over)
        self.assertTrue(episode.submit_answer(self.answer)["accepted"])
        with self.assertRaisesRegex(ValueError, "ended"):
            episode.submit_answer(self.answer)
        with self.assertRaisesRegex(ValueError, "ended"):
            episode.read_fact()

    def test_cross_question_fact_is_not_a_valid_citation(self):
        other = copy.deepcopy(self.fact)
        other["fact_id"] = "fact_b"
        dataset = QADataset([self.fact, other], [self.question])
        self.answer["provenance"]["fact_id"] = "fact_b"
        self.assertFalse(check_answer(self.answer, dataset.public_fact("question_a"))["provenance_ok"])

    def test_nonfinite_boolean_and_out_of_range_confidence_are_refused(self):
        for confidence in (float("nan"), float("inf"), True, -1, 2, "0.9", 10 ** 1000):
            self.answer["confidence"] = confidence
            self.assertFalse(check_answer(self.answer, self.fact)["structure_ok"])
        self.answer["confidence"] = 1
        self.assertTrue(check_answer(self.answer, self.fact)["structure_ok"])

    def test_invented_source_locations_are_refused(self):
        for field, value in (("file", "another.csv"), ("row", 8), ("row", True)):
            answer = copy.deepcopy(self.answer)
            answer["provenance"][field] = value
            self.assertFalse(check_answer(answer, self.fact)["provenance_ok"])

    def test_missing_and_errored_predictions_stay_in_denominator(self):
        report = evaluate(self.dataset, [])
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["structure_ok_rate"], 0)
        self.assertTrue(report["checks"][0]["missing"])
        report = evaluate(self.dataset, [{"qa_id": "question_a", "answer": self.answer}])
        self.assertEqual(report["provenance_ok_rate"], 1)
        report = evaluate(self.dataset, [{"qa_id": "question_a", "answer": self.answer, "error": "failed"}])
        self.assertEqual(report["provenance_ok_rate"], 0)

    def test_prediction_duplicates_and_unknown_ids_are_refused(self):
        prediction = {"qa_id": "question_a", "answer": self.answer}
        for predictions in ([prediction, prediction], [{"qa_id": "absent"}]):
            with self.assertRaises(DataError):
                evaluate(self.dataset, predictions)

    def test_numeric_quality_is_separate_from_structural_acceptance(self):
        reference = {"direct_answer": "The mean was approximately 12.50."}
        self.assertTrue(direct_numeric_match({"direct_answer": "12.5"}, reference))
        self.assertFalse(direct_numeric_match({"direct_answer": "12.51"}, reference))
        self.assertFalse(direct_numeric_match({"direct_answer": "Either 10 or 12.5"}, reference))
        self.assertFalse(direct_numeric_match({"direct_answer": "1e999999999999999999999999"}, reference))
        self.answer["direct_answer"] = "13.5"
        self.assertTrue(check_answer(self.answer, self.fact)["structure_ok"])
        self.assertFalse(direct_numeric_match(self.answer, reference))
        self.assertIsNone(direct_numeric_match(self.answer, {"direct_answer": "not numeric"}))

    def test_jsonl_and_sqlite_publish_the_same_evidence_without_database_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            facts, questions, database = directory / "facts.jsonl", directory / "qa.jsonl", directory / "facts.db"
            facts.write_text(json.dumps(self.fact) + "\n", encoding="utf-8")
            questions.write_text(json.dumps(self.question) + "\n", encoding="utf-8")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE facts (fact_id TEXT, fact_json TEXT)")
                connection.execute("INSERT INTO facts VALUES (?, ?)", ("fact_a", json.dumps(self.fact)))
                connection.commit()
            before = database.read_bytes()
            loaded = QADataset.load(database, questions, sqlite=True)
            self.assertEqual(loaded.public_fact("question_a"), QADataset.load(facts, questions).public_fact("question_a"))
            self.assertEqual(database.read_bytes(), before)
            with self.assertRaises(DataError):
                QADataset.load(directory / "absent.db", questions, sqlite=True)
            self.assertFalse((directory / "absent.db").exists())

    def test_invalid_json_reports_source_and_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.jsonl"
            for text in ('{"value": NaN}', '[]', '{broken'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(DataError, "record 1"):
                    list(read_jsonl(path))

    def test_debug_command_inspects_only_public_content(self):
        with tempfile.TemporaryDirectory() as directory:
            facts, questions = Path(directory) / "facts.jsonl", Path(directory) / "qa.jsonl"
            facts.write_text(json.dumps(self.fact), encoding="utf-8")
            questions.write_text(json.dumps(self.question), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa", "inspect",
                                     "--facts", str(facts), "--qa", str(questions), "--qa-id", "question_a"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("private", result.stdout)
            self.assertIn("reading_mean", result.stdout)

    def test_download_uses_one_resolved_revision_and_records_hashes(self):
        urls = []

        def response(request):
            url = request if isinstance(request, str) else request.full_url
            urls.append(url)
            if "api.github.com" in url:
                return io.BytesIO(json.dumps({"sha": "a" * 40}).encode())
            name = url.rsplit("/", 1)[1]
            headers = sorted(FILES[name])
            return io.BytesIO((",".join(headers) + "\n" + ",".join("1" for _ in headers) + "\n").encode())

        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", side_effect=response):
            manifest = download_raw(directory)
            self.assertEqual(len(manifest["files"]), len(FILES))
            self.assertTrue(all("a" * 40 in url for url in urls[1:]))
            self.assertTrue((Path(directory) / "CHECKSUMS.txt").exists())
            with self.assertRaisesRegex(ValueError, "already exist"):
                download_raw(directory)

    def test_build_preserves_source_identity_and_validates_its_outputs(self):
        scripts = {
            "ts_fact_extractor.py": ("import json, pathlib, sys\n"
                                     f"fact = {self.fact!r}\n"
                                     "pathlib.Path(sys.argv[sys.argv.index('--out') + 1]).write_text(json.dumps(fact))\n").encode(),
            "qa_builder_ts.py": ("import json, pathlib, sys\n"
                                 "assert sys.argv[sys.argv.index('--mode') + 1] == 'descriptive'\n"
                                 f"question = {self.question!r}\n"
                                 "pathlib.Path(sys.argv[sys.argv.index('--out') + 1]).write_text(json.dumps(question))\n").encode(),
        }
        real_run = subprocess.run

        def run(command, **kwargs):
            if command[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(command, 0, "b" * 40 + "\n")
            if command[:2] == ["git", "show"]:
                return subprocess.CompletedProcess(command, 0, scripts[command[-1].rsplit("/", 1)[1]])
            return real_run(command, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            raw, destination = Path(directory) / "raw", Path(directory) / "prepared"
            raw.mkdir()
            manifest = {"files": {}}
            for name in FILES:
                (raw / name).write_bytes(b"artificial input")
                manifest["files"][name] = {"sha256": hashlib.sha256(b"artificial input").hexdigest()}
            (raw / "source.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("subprocess.run", side_effect=run), patch("importlib.util.find_spec", return_value=object()):
                build_descriptive(raw, destination)
                built = json.loads((destination / "build.json").read_text())
                self.assertEqual(built["status"], "built")
                self.assertEqual(built["upstream_revision"], "b" * 40)
                self.assertEqual(built["raw"], manifest)
                self.assertEqual(set(built["outputs"]), {"facts.jsonl", "qa_descriptive.jsonl"})
                with self.assertRaisesRegex(ValueError, "must be new"):
                    build_descriptive(raw, destination)
                (raw / "PdM_machines.csv").write_bytes(b"changed input")
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    build_descriptive(raw, Path(directory) / "changed")


if __name__ == "__main__":
    unittest.main()
