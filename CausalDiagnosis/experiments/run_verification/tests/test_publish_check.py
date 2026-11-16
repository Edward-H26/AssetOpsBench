"""CausalDiagnosis/experiments/run_verification/publish_check.py on made-up run folders: each check fails on a folder that breaks it and passes on a
clean one, and the private files of a run never reach the copy."""
import getpass
import json
import shutil

import pytest

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import public_record
from CausalDiagnosis.experiments.run_verification import publish_check
from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _meta


def _clean_run(root):
    run_dir = root / "static_scripted_20261113-120000"
    (run_dir / "seed000000").mkdir(parents=True)
    facts = root / "facts.jsonl"
    facts.write_text("{}\n", encoding="utf-8")
    ledger.start(run_dir, _meta(facts), ["--seeds", "1", "--out", str(root)])
    (run_dir / "seed000000" / "episode.json").write_text(json.dumps({"seed": 0, "facts_path": "facts.jsonl"}), encoding="utf-8")
    ledger.ledger_file.add_episode(run_dir, "seed000000", {"seed": 0, "block": "[100,100,0,90]"})
    (run_dir / "run.json").write_text(json.dumps({"ended": "2026-11-13T12:05:00", "model": "m"}), encoding="utf-8")
    ledger.finish(run_dir)
    return run_dir


def _check(tmp_path, run_dir, **kwargs):
    into = tmp_path / "branch"
    shutil.rmtree(into, ignore_errors=True)
    into.mkdir()
    return publish_check.check([run_dir], into, **kwargs)


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="needs gitleaks")
def test_a_clean_run_passes_and_its_private_file_stays_behind(tmp_path):
    run_dir = _clean_run(tmp_path / "runs")
    assert (run_dir / public_record.PRIVATE_FILE).is_file()
    problems, warnings = _check(tmp_path, run_dir)
    assert problems == [] and warnings == []
    assert not (tmp_path / "branch" / run_dir.name / public_record.PRIVATE_FILE).exists()


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="needs gitleaks")
@pytest.mark.parametrize("text, words", [("/home/someone/runs/x.json", "absolute path"), ("http://10.1.2.3:31000/v1", "URL"),
                                         ("on 10.1.2.3 today", "IPv4"), (getpass.getuser(), "the name"),
                                         ("token ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8", "gitleaks")])
def test_a_file_that_names_a_path_a_host_a_user_or_a_secret_fails_the_check(tmp_path, text, words):
    run_dir = _clean_run(tmp_path / "runs")
    (run_dir / "seed000000" / "report.md").write_text(f"a line of the report: {text}\n", encoding="utf-8")
    ledger.append_event(run_dir, "test", "report written after the run", {"seed000000/report.md": (None, ledger.sha256_of(run_dir / "seed000000" / "report.md"))})
    problems, _ = _check(tmp_path, run_dir)
    assert any(words in problem for problem in problems), problems


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="needs gitleaks")
def test_a_file_changed_after_the_ledger_and_a_file_too_large_for_github_fail_the_check(tmp_path, monkeypatch):
    run_dir = _clean_run(tmp_path / "runs")
    (run_dir / "seed000000" / "episode.json").write_text('{"changed": true}\n', encoding="utf-8")
    problems, _ = _check(tmp_path, run_dir)
    assert any("ledger check of the published copy" in problem for problem in problems), problems
    run_dir = _clean_run(tmp_path / "again")
    monkeypatch.setattr(publish_check, "MAX_FILE_BYTES", 10)
    monkeypatch.setattr(publish_check, "WARN_FILE_BYTES", 5)
    problems, warnings = _check(tmp_path, run_dir)
    assert any("above GitHub's limit" in problem for problem in problems) and problems


def test_without_gitleaks_the_check_fails_closed(tmp_path, monkeypatch):
    run_dir = tmp_path / "results"
    run_dir.mkdir()
    (run_dir / "INDEX.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(publish_check.shutil, "which", lambda name: None)
    problems, _ = _check(tmp_path, run_dir)
    assert problems == ["gitleaks is not installed, so no secret scan ran"]
