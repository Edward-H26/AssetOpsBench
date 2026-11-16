"""The registered analysis file: one record that the table code reads and that names the same tests as experiment_tables.py."""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402

pytestmark = pytest.mark.registered_file  # the file as it is, every number the user set


def _tests(name):
    return [tuple(test) for test in registered.family(name)["tests"]]


def _copy(tmp_path, change):
    data = registered.load()
    change(data)
    path = tmp_path / "registered_analysis.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_registered_file_and_the_design_name_the_same_tests():
    assert _tests("agents") == [(name, design.row_id(row), design.row_id(ref)) for name, row, ref in design.HYPOTHESES]
    assert _tests("h1_top") == [(name, design.row_id(row), design.row_id(ref)) for name, row, ref in design.H1_TOP]
    assert _tests("h2_prime") == [(name, design.row_id(row), design.row_id(ref)) for name, row, ref in design.H2_PRIME]
    assert _tests("full_environment") == [(name, design.row_id(row)) for name, row in design.FULL_ENVIRONMENT]
    assert registered.measures("agents") == design.HYPOTHESIS_MEASURES
    assert registered.measures("h2_prime") == design.H2_PRIME_MEASURES
    assert registered.size("layers") == len(design.LADDER) - 1
    for name in ("agents", "h1_top", "h2_prime", "full_environment", "parallel_time"):
        assert registered.size(name) == len(_tests(name)) * len(registered.measures(name)), name


def test_three_groups_are_primary_and_each_is_one_family_of_the_correction():
    # superseded rule: one primary family; the registered file now makes three groups primary, each corrected on its own
    assert [(group["name"], group["families"]) for group in registered.groups()] == [
        ("judge", ["h0"]), ("agents", ["agents"]), ("agents_environment", ["h1_top", "h2_prime"])]
    assert {name for name in registered.families() if registered.role(name) == "primary"} == {"h0", "agents", "h1_top", "h2_prime"}
    assert registered.size("agents") == 14 == registered.size("h1_top") + registered.size("h2_prime")
    for change, problem in ((lambda d: d["primary"]["groups"][0]["families"].append("agents"), "family agents: in the groups"),
                            (lambda d: d["primary"]["groups"][1]["families"].append("fleet"), "group agents: no family fleet"),
                            (lambda d: d["families"]["h1_top"].update(role="primary"), "family h1_top: the groups of primary give its role"),
                            (lambda d: d["primary"].update(groups=[]), "primary names no group")):
        broken = registered.load()
        change(broken)
        assert any(problem in found for found in registered.check(broken)), problem


def test_the_file_names_no_component_no_level_and_no_count_of_an_asset():
    text = registered.PATH.read_text(encoding="utf-8")
    profile = get_profile(DEFAULT_ASSET)
    for component in profile.components:
        assert not re.search(rf"\b{re.escape(component)}\b", text), component
    numbers = {int(n) for n in re.findall(r"(?<![\w.])\d+(?![\w.])", text)}
    from CausalDiagnosis.experiments.cluster_runs import run_lines  # the counts of the rig's start set, computed by code from the start set and never written into the file
    counts = {len(profile.components), run_lines.STARTS, run_lines.STARTS * run_lines.REPLICATES}
    assert not numbers & counts


def test_every_family_names_its_question_or_none_and_each_question_names_one_table():
    for question in registered.questions():
        assert isinstance(question["table"], str) and question["table"], question["id"]
    answered = {name for question in registered.questions() for name in question.get("families") or []}
    assert answered <= set(registered.families())


def test_the_file_leaves_no_value_open_and_the_judge_holds_no_number_of_its_own():
    """Every number and choice of the file is set; the judge reads the whole input and cuts only past the context of the served
    judge, so the file names its rule and no length."""
    assert registered.missing() == []
    assert not any(key.startswith("truncation") for key in registered.load()["judge"]) and registered.load()["judge"]["cut_rule"]
    assert (registered.value("intervals.min_clusters"), registered.value("counting.max_unscored_share")) == (30, 0.02)
    assert [registered.value(f"development.{key}") for key in ("valid_share", "exact_match_alpha", "max_harness_share", "patience_factor",
                                                               "patience_steps", "start_share")] == [0.5, 0.05, 0.02, 2, 2, 0.5]
    assert (registered.family("parallel_time")["margin_pass"], registered.family("parallel_time")["margin_cost"]) == (0.05, 0.25)
    assert "slots" not in registered.load()


def test_an_open_value_stops_its_reader_with_its_name(tmp_path):
    path = _copy(tmp_path, lambda d: d["development"].update(valid_share=None))
    assert "development.valid_share" in registered.missing(path)
    with pytest.raises(registered.OpenValue, match="development.valid_share"):
        registered.value("development.valid_share", path)
    assert registered.value("development.patience_share") == 0.9


def test_a_changed_size_floor_or_cluster_minimum_in_a_copy_of_the_file_changes_the_tables(tmp_path):
    rows = [{"pass_p": 0.01}]
    plan = registered.load()
    aggregate.holm_over(rows, ("pass",), 0.05, aggregate.registered_size({"registered": plan}, "layers"))
    assert rows[0]["family_tests"] == 5 and rows[0]["pass_holm_p"] == 0.05
    smaller = registered.load(_copy(tmp_path, lambda d: d["families"]["layers"].update(size=2)))
    rows = [{"pass_p": 0.01}]
    aggregate.holm_over(rows, ("pass",), 0.05, aggregate.registered_size({"registered": smaller}, "layers"))
    assert rows[0]["family_tests"] == 2 and rows[0]["pass_holm_p"] == 0.02

    reference = [{"mode": "gym", "pass": "true", "block": "[1]"}, *({"mode": "gym", "pass": "false", "block": f"[{i}]"} for i in range(2, 13))]
    assert aggregate.floor_cells(reference, {"registered": plan})["at_floor"] is False  # 1 of 12 lies above 5 percent
    raised = registered.load(_copy(tmp_path, lambda d: d["descriptive"].update(floor_of_pass=0.10)))
    assert aggregate.floor_cells(reference, {"registered": raised})["at_floor"] is True

    episodes = [{"mode": "gym", "pass": flag, "block": f"[{i}]"} for i, flag in enumerate(("true", "false", "true"))]
    ctx = {"resamples": 200, "seed": 0, "registered": registered.load(_copy(tmp_path, lambda d: d["intervals"].update(min_clusters=None)))}
    assert aggregate.summary(episodes, "pass", ctx)["pass_low"] != ""
    ctx["registered"] = plan  # three starts lie below the registered smallest number of clusters
    assert aggregate.summary(episodes, "pass", ctx)["pass_low"] == ""
    ctx["registered"] = registered.load(_copy(tmp_path, lambda d: d["intervals"].update(min_clusters=4)))
    cells = aggregate.summary(episodes, "pass", ctx)
    assert cells["pass"] == round(2 / 3, 6) and cells["pass_low"] == cells["pass_high"] == cells["pass_se"] == ""


def test_the_level_of_the_tables_is_the_registered_level():
    assert registered.level() == 0.05
    assert Fraction(str(registered.load()["descriptive"]["floor_of_pass"])) == Fraction(1, 20) and not hasattr(design, "H1_FLOOR")


@pytest.mark.parametrize("name", registered.TABLE_VALUES)
def test_the_tables_of_a_registered_root_need_every_number_they_read(name):
    """A number that the tables read and the file leaves open is a problem for the tables of registered runs alone."""
    data = registered.load()
    node = data
    *parents, leaf = name.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = None
    assert registered.check(data) == [] and registered.check(data, "record") == [f"{name} is not set"]


def test_the_registered_numbers_of_the_reward_are_the_verifier_defaults_and_the_drivers_pass_them(monkeypatch):
    """The five numbers stand in the block reward. The verifier keeps them as defaults for the callers outside benchmarks (the Gym
    wrapper, the MCP's worth of a declaration), so the two homes are held equal, and common.episode_reward passes the registered ones."""
    import inspect

    from CausalDiagnosis.experiments.episode_runners import run_folder_layout
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import decision_rule as decision

    numbers = registered.reward_numbers()
    defaults = inspect.signature(verifier.reward).parameters
    assert (numbers["w_c"], numbers["w_t"], numbers["cost_cap"], numbers["declaration_credit"]) == (
        defaults["w_c"].default, defaults["w_t"].default, verifier.COST_CAP, verifier.ABSTENTION_CREDIT)
    assert decision.DECLARATION_WORTH["reward"] == numbers["declaration_credit"] and numbers["stream_declaration_credit"] == 0.0
    assert all(f"reward.{key}" in registered.TABLE_VALUES for key in registered.REWARD_KEYS)
    verdict = {"verified": True, "pass": True, "abstention_correct": False, "cost_normalized": 1.0, "checks": {"provenance": {"ok": True}},
               "detection_delay": {}, "repair_delay": {}, "unrestored": []}
    load = registered.load
    monkeypatch.setattr(registered, "load", lambda *a, **k: {**load(*a, **k), "reward": {**load(*a, **k)["reward"], "w_c": 0.25}})
    assert run_folder_layout.episode_reward(verdict, 10) == verifier.reward(verdict, 10, w_c=0.25) == 0.75
    assert run_episodes.episode_reward is run_folder_layout.episode_reward


def test_the_constants_of_the_retry_policy_are_the_registered_ones():
    """The retry module keeps its two constants with their sources, the registered file states both under
    model_request, and the two agree."""
    from CausalDiagnosis.utils import model_request_retry as retry

    stated = registered.load()["model_request"]
    assert (stated["retry_delay_s"], stated["read_timeout_s"]) == (retry.RETRY_DELAY_S, retry.READ_TIMEOUT_S)
