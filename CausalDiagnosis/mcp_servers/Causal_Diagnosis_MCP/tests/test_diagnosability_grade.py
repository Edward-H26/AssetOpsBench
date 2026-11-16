"""E18 of version 8 on the fitted models: a rollout starts from the record before cycle 0 that the rule of the server reads,
every summary where nothing is priced and the reads that the budget pays beside the repair of every component where prices
hold, and then commits, at no cost, wherever no clock runs and on every row with prices; only a row that prices nothing and
has a clock reads on over the live window, within the budget. A commitment counts only on the true combination at every
level, the first commitment of a rollout and the majority rule of the binary label stand, a table grades a combination when
it is first asked and keeps it in grades.json under a key with the rule and the clock and without the fault rule and the
ranking model, the command line grades every combination, and what a rollout is served is what each registered row
releases."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from CausalDiagnosis.utils.paths import preset_files  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.recordings import DataFileError
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import PRESETS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as gr
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view, recorded_blocks
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import downtime_rate
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests import advice_walk as aw
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import episode_models

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

VIEW = profile_view()
START = (100, 80, 0, 90)


def zero_durations(rules):
    """The durations of the rules with every duration that makes a clock run at zero."""
    durations = rules["durations"]
    return {**durations, "read_per_cycle": 0, "lead_time": {c: 0 for c in durations["lead_time"]},
            "repair_downtime": {c: 0 for c in durations["repair_downtime"]}}


def unpriced(rules):
    """The rules with no price and no budget, a row that prices nothing."""
    prices = rules["prices"]
    free = {**prices, "read": {g: 0 for g in prices["read"]}, "parts": {c: 0 for c in prices["parts"]}, "repair_charge": 0, "visit": 0}
    return {**rules, "prices": free, "budget": None}


FREE_CLOCK = unpriced(VIEW.rules)  # prices nothing, and the clock of the default rules runs
STILL = {**FREE_CLOCK, "clock_mode": "declared", "durations": zero_durations(VIEW.rules)}  # prices nothing, no clock
PRICED_STILL = {**VIEW.rules, "clock_mode": "declared", "durations": zero_durations(VIEW.rules)}  # prices, no clock


def segments(block, entry=0):
    """The record before cycle 0 and the live window that the evidence source serves at the entry offset."""
    before, live = gr.served_pairs()[entry]
    return block[before], block[live]


def record_belief(effect, record, summaries=None):
    """The belief after every recording of the record on the summaries given, every summary the effect space knows when None."""
    shown = set(effect.space.fields) if summaries is None else summaries
    return gr.static_belief(effect, {p: gr.served_of(effect.space, row, shown) for p, row in enumerate(record)}, True)


@pytest.fixture(scope="module")
def block():
    return recorded_blocks(VIEW)[0][START]


def test_information_gain_is_nonnegative_and_largest_for_the_richest_group():
    post = Posterior.uniform(episode_models(START).effect)
    rng = np.random.default_rng(0)
    gains = {g: gr.expected_information_gain(post, post.space.groups[g], 256, rng) for g in post.space.groups}
    assert all(v >= -1e-9 for v in gains.values()) and gains["tier3"] >= gains["tier1"] - 0.05


def test_rollouts_respect_the_budget_and_a_budget_below_any_price_is_inseparable(block, monkeypatch):
    """On a row that prices nothing and has a clock a rollout reads the live window at one call a read, and under a belief
    that never commits it reads until the budget is spent. On the default rules, which hold prices, a budget below the
    repair of every component buys no read of the record, so the rollout commits on the top of the prior, and a start that
    the record puts on top gets a grade only from a budget that buys the record."""
    wrong = next(tuple(h) for h in VIEW.combinations if tuple(h) != START)
    run = scripted_rollout(monkeypatch, block, [(wrong, False)], budget=3)
    assert (run.committed, run.first_right, run.prices, run.cost, run.readings) == (False, False, 3.0, 3.0, 3)
    record, live = segments(block)
    script = [(wrong, False)] * len(record) + [(START, False)]
    assert scripted_rollout(monkeypatch, block, script, budget=0, view=VIEW, record=True) == gr.Rollout(False, 0.0, 0.0, 0, False)
    assert scripted_rollout(monkeypatch, block, script, budget=300, view=VIEW, record=True) == gr.Rollout(True, 0.0, 0.0, 0, True)
    serving = lambda _rng: (episode_models(START).effect, record, live)  # noqa: E731
    assert gr.grade(VIEW, serving, START, 0, np.random.default_rng(0), rollouts=2, samples=8) is None
    assert gr.grade(VIEW, serving, START, 300, np.random.default_rng(0), rollouts=2, samples=8) == {
        "start": list(START), "budget": 300, "cost": 0.0, "commit_share": 1.0, "first_share": 1.0, "readings": 0.0, "rollouts": 2}


def test_a_row_without_a_clock_is_graded_on_the_recordings_before_cycle_0_alone(block):
    """A row that prices nothing and has no clock serves the record before cycle 0 alone: a rollout reads every summary of it
    without charge and commits to the top of that belief, at no cost and with no reading, and never reads the live window,
    which the test fills with NaN. A row with prices commits on the read of the record that its budget pays, which on the
    default budget is the group with every summary, with a clock or without one. Without prices the rules carry no budget,
    so the budget of the grade is the step cap."""
    assert gr.budget_of(STILL) == STILL["max_steps"] and gr.budget_of(VIEW.rules) == VIEW.rules["budget"]
    effect = episode_models(START).effect
    record, live = segments(block)
    post = record_belief(effect, record)
    right = tuple(post.candidates[post.order()[0]]) == START
    for rules in (STILL, PRICED_STILL, VIEW.rules):
        view = profile_view(rules)
        run = gr.rollout(view, effect, START, record, np.full_like(live, np.nan), gr.budget_of(rules), np.random.default_rng(0), samples=8)
        assert run == gr.Rollout(right, 0.0, 0.0, 0, right)


def test_the_budget_counts_the_read_prices_alone_as_the_environment_does(block, monkeypatch):
    """The environment's budget falls by prices only and its downtime goes to a hidden ledger, so the money of a read of the
    record is the budget less the repair of every component, whatever downtime the start runs: a budget that leaves the
    price of one cycle of the cheapest group buys that one cycle, the newest recording, on a faulty start whose downtime
    runs, which a budget that also counted the downtime would refuse, and the key of a grade table carries the rule, so
    that a table graded under another rule is never read."""
    rules = dict(VIEW.rules)
    cost = gr.costs(VIEW, 0)
    cheapest = min(cost.test_price, key=lambda g: (cost.test_price[g], g))
    assert cost.test_price[cheapest] < min(p for g, p in cost.test_price.items() if g != cheapest)
    record, _live = segments(block)
    budget = gr.plan_terms(VIEW, list(VIEW.components))[0] + cost.test_price[cheapest]
    assert downtime_rate(START, cost) > 0
    assert gr.record_reads(VIEW, record, budget) == {len(record) - 1: set(VIEW.group_fields(cheapest))}
    key = gr.rules_key(rules, rules["budget"])
    monkeypatch.setattr(gr, "RULE_TEXT", "another rule")
    assert gr.rules_key(rules, rules["budget"]) != key


def test_the_grade_commits_only_on_the_true_combination_at_every_level(block):
    """Pass scores every level, so the rule commits on the start that the belief puts on top and not on another start of
    the same faulty set: with a budget that buys no read of the record a rollout commits on the top of the prior alone."""
    effect = episode_models(START).effect
    post = Posterior.uniform(effect, block=True)
    top = tuple(post.candidates[post.order()[0]])
    same_set = next(h for h in VIEW.combinations if tuple(h) != top and VIEW.faulty(h) == VIEW.faulty(top))
    record, live = segments(block)
    assert gr.rollout(VIEW, effect, top, record, live, 0, np.random.default_rng(0), samples=8).committed
    assert not gr.rollout(VIEW, effect, same_set, record, live, 0, np.random.default_rng(0), samples=8).committed


class ScriptedBelief:
    """A belief that walks through a script of (top combination, whether E10 commits), one step per reading, its last step held."""

    def __init__(self, script):
        self.script, self.candidates = script, [script[0][0]]

    def order(self):
        return [0]

    def updated(self, _fields, _values):
        return ScriptedBelief(self.script[1:] or self.script)


def scripted_rollout(monkeypatch, block, script, budget=300, rng=None, view=None, record=False):
    """A rollout under a scripted belief, on the row that prices nothing and has a clock unless a view is given, the one row
    whose rollouts read the live window. Without record the record before cycle 0 is empty, so the script walks the reads of
    the live window alone."""
    effect = episode_models(START).effect
    monkeypatch.setattr(gr.Posterior, "uniform", lambda *args, **kwargs: ScriptedBelief(script))
    monkeypatch.setattr(gr, "decide", lambda post, *args, **kwargs: SimpleNamespace(read_first=not post.script[0][1]))
    monkeypatch.setattr(gr, "expected_information_gain", lambda *args, **kwargs: 1.0)
    before, live = segments(block)
    return gr.rollout(view or profile_view(FREE_CLOCK), effect, START, before if record else before[:0], live, budget,
                      rng or np.random.default_rng(0), samples=8)


def test_a_rollout_records_whether_its_first_commitment_names_the_truth(block, monkeypatch):
    """The search goes on after a wrong commitment, so the rollout can still commit to the truth, and its first commitment stays
    what an agent acting on the rule would have taken. A step on which E10 does not commit is no commitment."""
    wrong = next(tuple(h) for h in VIEW.combinations if tuple(h) != START)
    outcomes = {
        "at once": ([(START, True)], True, True, 0),
        "after a wrong commitment": ([(wrong, True), (START, True)], True, False, 1),
        "after a read with no commitment": ([(wrong, False), (START, True)], True, True, 1),
        "wrong, then a read, then right": ([(wrong, True), (wrong, False), (START, True)], True, False, 2),
    }
    for name, (script, committed, first_right, readings) in outcomes.items():
        run = scripted_rollout(monkeypatch, block, script)
        assert (run.committed, run.first_right, run.readings) == (committed, first_right, readings), name
    # out of budget, on a wrong commitment or with none, neither counts
    for script in ([(wrong, True)], [(START, False)]):
        run = scripted_rollout(monkeypatch, block, script, budget=3)
        assert (run.committed, run.first_right) == (False, False)


def test_a_wrong_first_commitment_changes_neither_the_walk_nor_the_draws_of_the_rollout(block, monkeypatch):
    """A rollout that commits wrongly on the way reads on as one that does not commit, with the same cost and the same generator
    afterwards, so recording the first commitment moves no other number of the grade."""
    wrong = next(tuple(h) for h in VIEW.combinations if tuple(h) != START)
    first_rng, second_rng = np.random.default_rng(5), np.random.default_rng(5)
    committed_wrong = scripted_rollout(monkeypatch, block, [(wrong, True), (wrong, True), (START, True)], rng=first_rng)
    no_commitment = scripted_rollout(monkeypatch, block, [(wrong, False), (wrong, False), (START, True)], rng=second_rng)
    assert committed_wrong._replace(first_right=None) == no_commitment._replace(first_right=None)
    assert (committed_wrong.first_right, no_commitment.first_right) == (False, True)
    assert first_rng.bit_generator.state == second_rng.bit_generator.state


def test_the_grade_shares_count_every_rollout_and_the_binary_label_is_the_majority_of_first_commitments(block, monkeypatch):
    effect = episode_models(START).effect
    runs = iter([gr.Rollout(True, 10.0, 1.0, 1, True), gr.Rollout(True, 30.0, 3.0, 3, False), gr.Rollout(False, 50.0, 5.0, 5, False), gr.Rollout(True, 20.0, 2.0, 2, True)])
    monkeypatch.setattr(gr, "rollout", lambda *args, **kwargs: next(runs))
    entry = gr.grade(VIEW, lambda _rng: (effect, *segments(block)), START, 300, np.random.default_rng(0), rollouts=4, samples=8)
    assert entry["commit_share"] == 0.75 and entry["first_share"] == 0.5 and entry["cost"] == 20.0 and entry["readings"] == 2.0
    # the majority, as the independent label reads it: a tie is no majority, and a share that commits at last does not count
    assert not gr.separable(entry) and not gr.separable(None)
    assert gr.separable({**entry, "first_share": 0.6}) and not gr.separable({**entry, "commit_share": 1.0, "first_share": 0.4})


def test_the_table_grades_on_first_use_keeps_the_grade_and_ignores_the_ranking_model(tmp_path, monkeypatch):
    models, rules = load_models(), gr.default_rules()
    path = tmp_path / "grades.json"
    calls = []

    def counted(view, serving, start, budget, rng, rollouts, samples):
        calls.append((start, rng.integers(1 << 30)))
        return None if start == START else {"start": list(start), "budget": budget, "cost": 1.0, "commit_share": 1.0, "first_share": 1.0, "readings": 1.0, "rollouts": rollouts}

    monkeypatch.setattr(gr, "grade", counted)
    key = gr.combo_key(START)
    table = gr.ensure_grades(models, rules, path=path, rollouts=1, samples=8)
    assert key in table["grades"] and len(table["grades"]) == len(VIEW.combinations) and calls == []
    assert table["grades"][key] is None and len(calls) == 1
    other = gr.ensure_grades(models, rules, ranking_model="discriminant_urn", path=path)
    assert other["rules_key"] == table["rules_key"] == gr.rules_key(rules, rules["budget"]) and other["grades"][key] is None and len(calls) == 1
    healthy = gr.combo_key(VIEW.healthy_combination())
    assert other["grades"][healthy]["cost"] == 1.0 and len(calls) == 2
    stored = json.loads(path.read_text())
    assert stored["format"] == gr.GRADES_FORMAT == 9 and set(stored["tables"][table["rules_key"]]["grades"]) == {key, healthy}
    # the generator of a combination depends on its place in the grid, not on the order of the requests
    first_draws = dict(calls)
    calls.clear()
    reverse = gr.ensure_grades(models, rules, path=tmp_path / "fresh.json", rollouts=1, samples=8)
    _ = reverse["grades"][healthy], reverse["grades"][key]
    assert [start for start, _draw in calls] == [VIEW.healthy_combination(), START] and dict(calls) == first_draws
    assert "fault_rule" not in gr.RULE_KEYS and gr.rules_key(rules, 10) != gr.rules_key(rules, rules["budget"])
    assert gr.grade_of(START, models, rules, path=path) is None


def test_the_command_line_grades_every_combination_of_the_default_rules(tmp_path):
    path = tmp_path / "grades.json"
    gr.main(["--budget", "2", "--rollouts", "1", "--samples", "8", "--out", str(path)])
    stored = json.loads(path.read_text())
    table = next(iter(stored["tables"].values()))
    assert table["budget"] == 2 and len(table["grades"]) == len(VIEW.combinations) and "ranking_model" not in table
    gr.main(["--budget", "2", "--rollouts", "1", "--samples", "8", "--out", str(path), "--force"])
    assert len(json.loads(path.read_text())["tables"]) == 1


def test_a_rollout_reads_the_fits_of_an_episode_of_its_start_and_a_live_segment_of_its_block(block, tmp_path, monkeypatch):
    """The grade of a start rolls out under the fits that leave the start's block out, as the server launched for an episode
    of that start scores it, over the record before cycle 0 and the live window of one drawn entry offset, as the
    environment serves them."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import served_pairs

    models, rules, asked = load_models(), gr.default_rules(), []
    monkeypatch.setattr(gr, "grade", lambda view, serving, start, budget, rng, rollouts, samples: asked.append(serving(rng)) or None)
    table = gr.ensure_grades(models, rules, path=tmp_path / "grades.json", rollouts=1, samples=8)
    monkeypatch.setattr(models, "without", lambda start: episode_models(start) if start == START else pytest.fail(f"fits without {start}"))
    assert table["grades"][gr.combo_key(START)] is None
    (effect, record, live), = asked
    assert effect is episode_models(START).effect
    assert any(np.array_equal(record, block[before]) and np.array_equal(live, block[after]) for before, after in served_pairs())


def test_two_graders_of_one_file_at_once_keep_both_grades(tmp_path, monkeypatch):
    """Two driver processes that grade two starts of one rules key at once, as the lines of a grid did before the precompute
    step, must not drop each other's grade: the read of the file, the grade and the write run under a lock on the file."""
    models, rules = load_models(), gr.default_rules()
    path = tmp_path / "grades.json"
    monkeypatch.setattr(gr, "grade", lambda view, serving, start, budget, rng, rollouts, samples:
                        {"start": list(start), "budget": budget, "cost": 1.0, "commit_share": 1.0, "first_share": 1.0, "readings": 1.0, "rollouts": rollouts})
    read = gr._stored

    def slow(file, facts_sha256):
        stored = read(file, facts_sha256)
        time.sleep(0.3)  # the window in which the other grader reads the file before this one writes it
        return stored

    monkeypatch.setattr(gr, "_stored", slow)
    keys = [gr.combo_key(h) for h in VIEW.combinations[:2]]
    threads = [threading.Thread(target=lambda key: gr.ensure_grades(models, rules, path=path, rollouts=1, samples=8)["grades"][key], args=(key,))
               for key in keys]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert set(written["tables"][gr.rules_key(rules, rules["budget"])]["grades"]) == set(keys)


def test_the_grade_table_is_the_file_grades_path_names_when_it_is_asked(tmp_path, monkeypatch):
    """ensure_grades and grade_of read GRADES_PATH when they are called, not when the module is loaded, so that the tests run on
    a copy of the table (conftest.py of the repository) and never write the file a run in the same tree reads. The file names
    the data file of the models it was graded with, and models of another data file are refused."""
    rules = VIEW.rules
    key = gr.rules_key(rules, gr.budget_of(rules))
    header = {"rules_key": key, "budget": gr.budget_of(rules), "rollouts": 1, "samples": 1, "seed": 0, "rule": gr.RULE_TEXT,
              "code": gr.code_digest()}
    path = tmp_path / "grades.json"
    path.write_text(json.dumps({"format": gr.GRADES_FORMAT, "facts_sha256": "a" * 64,
                                "tables": {key: {**header, "grades": {gr.combo_key(START): {"cost": 1.0}}}}}))
    monkeypatch.setattr(gr, "GRADES_PATH", path)
    assert gr.grade_of(START, SimpleNamespace(raw={"facts_sha256": "a" * 64}), rules) == {"cost": 1.0}
    with pytest.raises(DataFileError) as refused:
        gr.grade_of(START, SimpleNamespace(raw={"facts_sha256": "b" * 64}), rules)
    assert gr.REGRADE in str(refused.value) and "a" * 64 in str(refused.value) and "b" * 64 in str(refused.value)


def test_a_grade_table_names_the_data_file_of_its_models(tmp_path, monkeypatch):
    """A grades file written by a first grade names, at its top, the data file that its models were fitted on."""
    models, rules = load_models(), gr.default_rules()
    path = tmp_path / "grades.json"
    monkeypatch.setattr(gr, "grade", lambda view, serving, start, budget, rng, rollouts, samples: None)
    assert gr.ensure_grades(models, rules, path=path, rollouts=1, samples=8)["grades"][gr.combo_key(START)] is None
    assert json.loads(path.read_text(encoding="utf-8"))["facts_sha256"] == models.raw["facts_sha256"]


def test_the_grade_refuses_models_fitted_on_another_data_file_than_the_one_it_reads(tmp_path):
    """The blocks of a grade table are read only when its models were fitted on the data file it reads."""
    stale = SimpleNamespace(raw={"facts_sha256": "0" * 64})
    table = gr.ensure_grades(stale, gr.default_rules(), path=tmp_path / "grades.json")["grades"]
    with pytest.raises(DataFileError) as refused:
        table.blocks()
    assert gr.REFIT in str(refused.value) and "0" * 64 in str(refused.value)


def test_a_read_without_a_price_is_read_by_the_rule_and_the_rollout_ends_after_it(block):
    """On rules with money prices and one read price of zero the rollout reads what record_read names, every cycle of the
    group without a price where the money left pays no cycle of a priced group, divides by no price, raises nothing and
    commits after that read."""
    rules = dict(VIEW.rules)
    group = min(VIEW.sensor_groups, key=lambda g: (len(VIEW.sensor_groups[g]), g))
    rules["prices"] = {**rules["prices"], "read": {**rules["prices"]["read"], group: 0}}
    view = profile_view(rules)
    assert gr.cost_unit(view.rules["prices"]) == "money"
    paid = min(p for g, p in rules["prices"]["read"].items() if g != group)
    budget = gr.plan_terms(view, list(view.components))[0] + paid / 2
    effect = episode_models(START).effect
    record, live = segments(block)
    assert gr.record_reads(view, record, budget) == {p: set(view.group_fields(group)) for p in range(len(record))}
    post = record_belief(effect, record, set(view.group_fields(group)))
    right = tuple(post.candidates[post.order()[0]]) == START
    run = gr.rollout(view, effect, START, record, np.full_like(live, np.nan), budget, np.random.default_rng(0), samples=8)
    assert run == gr.Rollout(right, 0.0, 0.0, 0, right)


def test_a_forced_table_keeps_the_grades_that_another_process_writes_meanwhile(tmp_path, monkeypatch):
    """--force drops the table of the rules from grades.json by a read, a change and a write of the file. It holds the lock that
    grading processes hold, so a grade that another process writes while the drop stands between its read and its write stays
    in the file."""
    rules = gr.default_rules()
    budget = gr.budget_of(rules)
    key = gr.rules_key(rules, budget)
    header = {"budget": budget, "rollouts": 1, "samples": 1, "seed": 0, "rule": gr.RULE_TEXT}
    path = tmp_path / "grades.json"
    # the file names the data file of the models that grade.main loads, as every grades file does
    path.write_text(json.dumps({"format": gr.GRADES_FORMAT, "facts_sha256": load_models().raw["facts_sha256"],
                                "tables": {key: {"rules_key": key, **header, "grades": {}}, "other": {"rules_key": "other", **header, "grades": {}}}}))
    read_first, written, caller, real_stored = threading.Event(), threading.Event(), threading.current_thread(), gr._stored

    def stored_then_wait(p, *data):
        content = real_stored(p, *data)
        if threading.current_thread() is caller and not read_first.is_set():
            read_first.set()
            written.wait(timeout=2)  # the other process grades now, unless the lock keeps it out until the drop is written
        return content

    def other():
        read_first.wait(timeout=5)
        with gr.locked(path):
            stored = json.loads(path.read_text(encoding="utf-8"))
            stored["tables"]["other"]["grades"]["2,2,2,2"] = {"cost": 1.0}
            gr.write_json_atomically(path, stored, indent=1)
        written.set()

    monkeypatch.setattr(gr, "_stored", stored_then_wait)
    monkeypatch.setattr(gr, "ensure_grades", lambda *args, **kwargs: {"rollouts": 1, "samples": 1, "seed": 0, "grades": {}})
    thread = threading.Thread(target=other)
    thread.start()
    gr.main(["--force", "--out", str(path)])
    thread.join()
    tables = json.loads(path.read_text(encoding="utf-8"))["tables"]
    assert key not in tables and tables["other"]["grades"] == {"2,2,2,2": {"cost": 1.0}}


def test_a_row_with_a_clock_starts_from_the_prehistory_and_reads_the_live_window(block, monkeypatch):
    """On a row that prices nothing and has a clock a rollout starts from the belief after every summary of the record
    before cycle 0, with the whole live window open to every group, and goes on over the live window: its first read takes
    the first recording of the window, and the next decision sees the belief after it."""
    effect = episode_models(START).effect
    record, live = segments(block)
    seen = []

    def deciding(post, cost, samples, seed, serving=None):
        seen.append((post.probabilities.copy(), dict(serving)))
        return SimpleNamespace(read_first=len(seen) == 1)

    monkeypatch.setattr(gr, "decide", deciding)
    monkeypatch.setattr(gr, "expected_information_gain", lambda *args, **kwargs: 1.0)
    run = gr.rollout(profile_view(FREE_CLOCK), effect, START, record, live, gr.budget_of(FREE_CLOCK), np.random.default_rng(0), samples=8)
    before = record_belief(effect, record)
    assert np.array_equal(seen[0][0], before.probabilities)
    assert all(left.cycles == len(live) for left in seen[0][1].values())
    after = before.updated(*gr.served_of(effect.space, live[0], set(effect.space.fields)))
    assert run.readings >= 1 and np.array_equal(seen[1][0], after.probabilities)


def test_the_key_of_a_grade_table_tells_a_row_with_a_clock_from_one_without():
    """Two rows that differ only in whether a clock runs serve different recordings, so their grades never share a table."""
    declared = {**VIEW.rules, "clock_mode": "declared", "durations": zero_durations(VIEW.rules)}
    timed = {**declared, "clock_mode": "generation_time"}
    assert not gr.has_clock(declared["clock_mode"], declared["durations"]) and gr.has_clock(timed["clock_mode"], timed["durations"])
    assert gr.rules_key(declared, 300) != gr.rules_key(timed, 300)


def test_what_a_rollout_is_served_is_what_the_row_releases():
    """For every registered preset, an episode of the engine of the AssetOpsBench rows from a development combination at an
    entry offset, whose group with every channel reads the whole record before cycle 0, releases exactly the recordings
    that serving_of hands a rollout for that offset, every summary at the digits served. A preset whose surface served
    something else would fail here and not in a label."""
    start = aw.development_combinations()[0]
    block = recorded_blocks(VIEW)[0][tuple(start)]
    cycles, blocks = aw.world()
    presets = sorted(path.stem for path in preset_files(PRESETS).values())
    assert {"base", "priced"} <= set(presets)
    for entry in (0, 7):
        _effect, record, _live = gr.serving_of(episode_models(tuple(start)), block)(SimpleNamespace(integers=lambda n: entry))
        for preset in presets:
            config = aw.started(preset, start, entry)
            env = RecordedRigEnv(cycles, blocks, config)
            env.reset(0)
            group = max(config.sensor_groups, key=lambda g: (len(config.sensor_groups[g]), g))
            aw.read(env, group, len(record), {g: set() for g in config.sensor_groups})
            served = sorted(next(e for e in reversed(env.log) if e["event"] == "read")["served"], key=lambda s: s["t"])
            assert len(served) == len(record), preset
            for row, reading in zip(record, served):
                assert reading["values"] == {f: float(f"{row[i]:.6g}") for i, f in enumerate(VIEW.summary_fields)}, preset


def test_a_grade_table_carries_the_code_that_graded_it_and_a_read_only_process_grades_nothing(tmp_path, monkeypatch):
    """Every stored grade table records the digest of the server's code: table_problems names a missing table, a table short of
    a recorded combination and a table of other code; with CD_TABLES_READ_ONLY=1 a missing grade stops the process by name and
    the file stays as it was."""
    from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES, GeneratedTableMissing

    models, rules = load_models(), gr.default_rules()
    path = tmp_path / "grades.json"
    monkeypatch.setattr(gr, "grade", lambda view, serving, start, budget, rng, rollouts, samples: {"start": list(start), "cost": 1.0})
    key = gr.rules_key(rules, gr.budget_of(rules))
    assert gr.table_problems(models, rules, path=path) == [f"grades {key[:12]}: missing"]
    table = gr.ensure_grades(models, rules, path=path, rollouts=1, samples=8)
    _ = table["grades"][gr.combo_key(START)]
    assert json.loads(path.read_text())["tables"][key]["code"] == gr.code_digest()
    assert gr.table_problems(models, rules, path=path)[0].endswith("recorded combinations without a grade")
    for combination in table["grades"]:
        _ = table["grades"][combination]
    assert gr.table_problems(models, rules, path=path) == []
    stored = json.loads(path.read_text())
    stored["tables"][key]["code"] = "other code"
    path.write_text(json.dumps(stored))
    assert gr.table_problems(models, rules, path=path) == [f"grades {key[:12]}: of other code"]
    monkeypatch.setenv(READ_ONLY_TABLES, "1")
    before = path.read_bytes()
    with pytest.raises(GeneratedTableMissing, match="precompute_tables.sbatch"):
        _ = gr.ensure_grades(models, rules, path=path)["grades"][gr.combo_key(START)]
    assert path.read_bytes() == before
