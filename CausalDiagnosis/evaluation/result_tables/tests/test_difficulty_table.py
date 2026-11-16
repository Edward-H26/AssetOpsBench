"""The difficulty table of the datasheet (CausalDiagnosis/evaluation/result_tables/difficulty_table.py): the reader counts the outcomes of the reference
policies on the development set from run folders, and the datasheet prints its table as the reader renders it, with empty cells
until the reference job has run. The rows of the tests are made up here."""
import csv
import json
from pathlib import Path

import pytest

from CausalDiagnosis.evaluation.result_tables import difficulty_table as difficulty
from CausalDiagnosis.experiments.table_definitions.experiment_tables import REFERENCE_POLICIES

from CausalDiagnosis.utils.paths import ROOT
DATASET = ROOT / "CausalDiagnosis" / "docs" / "dataset.md"
COLUMNS = ("seed", "config", "agent", "pass", "diagnosis_all_correct", "graded_pass", "graded_top1", "error")


def run(folder, starts, rows, config="base"):
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"starts": starts, "config_name": config}), encoding="utf-8")
    with open(folder / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, (agent, ok, top1, gpass, gtop1) in enumerate(rows):
            w.writerow({"seed": i, "config": config, "agent": agent, "pass": ok, "diagnosis_all_correct": top1,
                        "graded_pass": gpass, "graded_top1": gtop1, "error": ""})
    return folder


def test_the_reader_counts_each_policy_on_the_development_set(tmp_path):
    first = run(tmp_path / "first", "development", [("oracle", True, True, 1.0, 1.0), ("oracle", False, True, 0.5, 1.0),
                                                      ("greedy", False, False, 0.25, 0.5)])
    second = run(tmp_path / "second", "development", [("greedy", True, False, 1.0, 0.5)])
    rows = {(r["policy"], r["starts"]): r for r in difficulty.difficulty_table([first, second])}
    assert rows[("oracle", "development")] == {"policy": "oracle", "starts": "development", "episodes": 2, "pass": 1, "top1": 2,
                                               "graded_pass": 1.5, "graded_top1": 2.0, "unscored": 0}
    assert rows[("greedy", "development")] == {"policy": "greedy", "starts": "development", "episodes": 2, "pass": 1, "top1": 0,
                                               "graded_pass": 1.25, "graded_top1": 1.0, "unscored": 0}
    assert rows[("random", "development")]["episodes"] == 0  # every reference policy has a row, played or not
    assert set(rows) == {(label, "development") for label, _ in REFERENCE_POLICIES}


def test_the_reader_refuses_runs_of_another_start_set_preset_or_agent(tmp_path):
    with pytest.raises(ValueError, match="start set"):
        difficulty.difficulty_table([run(tmp_path / "a", "sampled", [("oracle", True, True, 1.0, 1.0)])])
    with pytest.raises(ValueError, match="preset"):
        difficulty.difficulty_table([run(tmp_path / "b", "development", [("oracle", True, True, 1.0, 1.0)], config="layered")])
    with pytest.raises(ValueError, match="reference"):
        difficulty.difficulty_table([run(tmp_path / "c", "development", [("plan_execute", True, True, 1.0, 1.0)])])


def test_the_rendered_rates_are_counts_and_means_over_episodes(tmp_path):
    dev = run(tmp_path / "dev", "development", [("oracle", True, True, 1.0, 1.0), ("oracle", False, True, 0.5, 1.0)])
    lines = difficulty.markdown(difficulty.difficulty_table([dev])).splitlines()
    oracle = next(line for line in lines if line.startswith("| Oracle |"))
    assert oracle.split(" | ")[1:6] == ["1 of 2", "2 of 2", "0.75", "1.00", "0 |"]
    assert next(line for line in lines if line.startswith("| Random |")) == "| Random |  |  |  |  |  |"  # no episode, empty cells


def test_the_datasheet_prints_the_table_with_empty_cells_until_the_reference_job():
    table = difficulty.markdown(difficulty.difficulty_table([]))
    assert table in DATASET.read_text(encoding="utf-8")
    assert len([line for line in table.splitlines() if line.startswith("| ") and "---" not in line]) == 1 + len(REFERENCE_POLICIES)


def test_an_unscored_episode_counts_as_a_failure_and_the_row_prints_how_many(tmp_path):
    """The registered counting.unscored: an episode that the verifier did not score, here with an error and no verdict, counts in the
    episodes of its row as a failure on pass and Top-1 and as zero on the graded outcomes, and the last cell counts it."""
    dev = run(tmp_path / "dev", "development", [("oracle", True, True, 1.0, 1.0), ("oracle", "", "", "", "")])
    row = {r["policy"]: r for r in difficulty.difficulty_table([dev])}["oracle"]
    assert (row["episodes"], row["pass"], row["top1"], row["graded_pass"], row["unscored"]) == (2, 1, 1, 1.0, 1)
    oracle = next(line for line in difficulty.markdown([row, *[r for r in difficulty.difficulty_table([]) if r["policy"] != "oracle"]]).splitlines()
                  if line.startswith("| Oracle |"))
    assert oracle.split(" | ")[1:6] == ["1 of 2", "1 of 2", "0.50", "0.50", "1 |"]
