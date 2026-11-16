"""The counting rule, the registered families and the descriptive cells of tables_from_runs/, on episodes made by hand."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402

CTX = {"resamples": 200, "seed": 0, "alpha": 0.05}


def test_a_played_episode_without_a_verdict_counts_as_a_failure_and_the_row_prints_it():
    scored, unscored = {"mode": "gym", "pass": "true", "block": "[1]"}, {"mode": "gym", "pass": "", "block": "[2]"}
    assert aggregate.value(unscored, "pass") == 0.0 and aggregate.value(unscored, "cost") is None
    cells = aggregate.summary([scored, unscored], "pass", CTX)
    assert cells["pass"] == 0.5 and cells["pass_n"] == 2
    assert aggregate.completeness([scored, unscored], CTX)["unscored"] == 1
    assert aggregate.value({"mode": "static", "pass": ""}, "pass") is None


def test_an_episode_without_a_judge_record_counts_as_a_failure_when_its_run_names_a_judge():
    assert aggregate.value({"mode": "gym", "judge_pass": "", "run_judge_model": "judge"}, "judge_pass") == 0.0
    assert aggregate.value({"mode": "gym", "judge_pass": "", "run_judge_model": ""}, "judge_pass") is None
    assert aggregate.completeness([{"mode": "gym", "pass": "true", "judge_pass": "", "run_judge_model": "judge"}], CTX)["unjudged"] == 1


def test_a_row_with_a_missing_episode_is_incomplete_and_a_row_above_the_unscored_limit_is_refused():
    row = [{"mode": "gym", "pass": "true", "run": "r", "episodes_planned": 3}, {"mode": "gym", "pass": "", "run": "r", "episodes_planned": 3}]
    assert aggregate.row_status(row, CTX) == "incomplete, 2 of 3"
    plan = registered.load()
    plan["counting"]["max_unscored_share"] = 0.25
    full = [{**e, "episodes_planned": 2} for e in row]
    assert aggregate.row_status(full, {**CTX, "registered": plan}) == "refused"
    assert aggregate.row_status(full, CTX) == ""


def test_holm_runs_over_the_registered_size_whatever_tests_are_present():
    rows = [{"pass_p": 0.01, "diagnosis_all_correct_p": 0.02}]
    aggregate.holm_over(rows, ("diagnosis_all_correct", "pass"), 0.05, 14)
    assert rows[0]["family_tests"] == 14 and rows[0]["pass_holm_p"] == 0.14 and rows[0]["pass_reject"] is False
    present = [{"pass_p": 0.01, "diagnosis_all_correct_p": 0.02}]
    aggregate.holm_over(present, ("diagnosis_all_correct", "pass"), 0.05)
    assert present[0]["family_tests"] == 2 and present[0]["pass_reject"] is True


def test_a_comparison_that_rejects_in_the_other_direction_prints_reversed_and_a_mixed_one_does_not_hold():
    measures = ("diagnosis_all_correct", "pass")
    both = {"diagnosis_all_correct_reject": True, "pass_reject": True}
    assert aggregate.reading({**both, "diagnosis_all_correct_diff": 0.1, "pass_diff": 0.2}, measures) == "holds"
    assert aggregate.reading({**both, "diagnosis_all_correct_diff": -0.1, "pass_diff": -0.2}, measures) == "reversed"
    assert aggregate.reading({**both, "diagnosis_all_correct_diff": -0.1, "pass_diff": 0.2}, measures) == "does not hold"
    assert aggregate.reading({"pass_diff": 0.2, "pass_reject": True}, measures) == "not run"
    assert aggregate.reading({**both, "diagnosis_all_correct_diff": 0.1, "pass_diff": 0.2, "at_floor": True}, measures) == "uninformative"
    assert aggregate.reading({"repair_declared_diff": -3, "repair_declared_reject": True}, ("repair_declared",), {"repair_declared": -1}) == "holds"


def test_two_rows_of_one_block_and_two_clusters_resample_as_two_clusters():
    a = [{"mode": "gym", "pass": "true", "block": "[1, 0]", "cluster": "s1"}, {"mode": "gym", "pass": "false", "block": "[1, 0]", "cluster": "s2"}]
    b = [{"mode": "gym", "pass": "false", "block": "[1, 0]", "cluster": "s1"}, {"mode": "gym", "pass": "false", "block": "[1, 0]", "cluster": "s2"}]
    assert aggregate.paired_difference(list(zip(a, b)), "pass", CTX)["clusters"] == 2
    assert aggregate.paired_difference(list(zip([{**e, "cluster": ""} for e in a], b)), "pass", CTX)["clusters"] == 1


def test_restored_clean_is_recovered_with_no_unnecessary_order_and_pass_implies_it():
    cells = {(True, 0): True, (True, 1): False, (False, 0): False, (False, 2): False}
    for (recovered, unnecessary), expected in cells.items():
        assert aggregate.restored_clean({"recovered": str(recovered).lower(), "unnecessary": str(unnecessary)}) is expected
    assert aggregate.restored_clean({"recovered": "", "unnecessary": ""}) == ""
    assert aggregate.value({"mode": "gym", "restored_clean": ""}, "restored_clean") == 0.0
    passing = {"recovered": "true", "unnecessary": "0", "pass": "true"}  # pass requires the asset restored and no unnecessary order
    assert aggregate.restored_clean(passing) is True


def _episode(tmp_path, name, verdict, log, horizon=40):
    folder = tmp_path / name
    (folder / "hidden").mkdir(parents=True)
    if verdict is not None:
        (folder / "hidden" / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    (folder / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
    return {"episode_dir": folder, "horizon": horizon, "asset_id": DEFAULT_ASSET}


def test_the_time_to_repair_in_declared_cycles_counts_the_horizon_without_a_verdict_and_for_a_fault_never_restored(tmp_path):
    profile = get_profile(DEFAULT_ASSET)
    healthy = dict(profile.healthy)
    first = profile.components[0]
    faulty = {**healthy, first: next(level for level in range(10) if level != healthy[first])}
    assert aggregate.repair_declared(_episode(tmp_path, "none", None, [])) == 40
    never = _episode(tmp_path, "never", {"hidden_start": faulty, "repair_delay": dict.fromkeys(profile.components)}, [])
    assert aggregate.repair_declared(never) == 40
    clean = _episode(tmp_path, "clean", {"hidden_start": healthy, "repair_delay": dict.fromkeys(profile.components)}, [])
    assert aggregate.repair_declared(clean) == 0
    assert aggregate.repair_declared({"horizon": ""}) == ""


def test_a_fault_that_an_order_of_the_agent_closes_is_timed_from_its_onset_to_the_close_less_the_thinking_before_the_order(tmp_path):
    """A close carries the combination after it, which the verifier reads with the levels of its truth model, so the time comes from
    the same model that scored the episode. The read at 4 charged cycles 1 to 3 to thinking, all before the order at 5, and the
    finish at 12 charged cycles 6 to 11, after the order, which stay in the count."""
    profile = get_profile(DEFAULT_ASSET)
    healthy = dict(profile.healthy)
    first = profile.components[0]
    faulty = {**healthy, first: next(level for level in profile.levels[first] if level != healthy[first])}
    log = [{"event": "read", "t_start": 4, "t_end": 4, "think_cycles": 3},
           {"event": "work_order_created", "id": "wo-1", "component": first, "t_start": 5, "t_end": 5, "think_cycles": 0},
           {"event": "work_order_closed", "id": "wo-1", "source": "agent", "component": first, "t_start": 12, "t_end": 12,
            "combination_after": [healthy[c] for c in profile.components]},
           {"event": "finish", "t_start": 12, "t_end": 12, "think_cycles": 6}]
    delays = {**dict.fromkeys(profile.components), first: 12}
    assert aggregate.repair_declared(_episode(tmp_path, "closed", {"hidden_start": faulty, "repair_delay": delays}, log)) == 12 - 3


def _hand_log_case(case):
    """Five hand-checked logs on the cooler, each order with the id of its close:
    (hidden start, log, the declared time). The thinking of an entry with t_start tau and think_cycles theta is the cycles tau - theta
    to tau - 1."""
    profile = get_profile(DEFAULT_ASSET)
    healthy = dict(profile.healthy)
    faulty = {**healthy, "cooler": 20}

    def action(event, t_start, think, t_end=None, **fields):
        return {"event": event, "step": 1, "t_start": t_start, "t_end": t_start if t_end is None else t_end, "class": "x", "cost": 0,
                "think_cycles": think, **fields}

    def system(event, t, **fields):
        return {"event": event, "step": None, "t_start": t, "t_end": t, "class": "system", "cost": 0, **fields}

    def order(t, think):
        return action("work_order_created", t, think, id="wo-1", component="cooler")

    def close(t):
        return system("work_order_closed", t, id="wo-1", component="cooler", source="agent",
                      combination_after=[healthy[c] for c in profile.components])

    reset = {"event": "reset", "step": None, "t_start": 0, "t_end": 0, "class": "system", "cost": 0}
    return {
        # thinking over cycles 3 to 12 straddles the onset at 11; only 11 and 12 lie before the order at 13: 17 - 11 - 2
        "A": (healthy, [reset, action("read", 0, 0), system("hazard_drop", 11, component="cooler", to_level=20), order(13, 10), close(17),
                        action("wait", 13, 0, 17)], 4),
        # a fault of the start; 2 cycles of thinking before the order at 2, 4 more after it: 9 - 0 - 2
        "B": (faulty, [reset, order(2, 2), close(9), action("wait", 6, 4, 9)], 7),
        # the close at 5 falls into the thinking of the next call, which is after the order: 5 - 0 - 2
        "C": (faulty, [reset, order(2, 2), close(5), action("read", 8, 6)], 3),
        # a call between the order and the close thinks 3 cycles after the order: 9 - 0 - 2
        "D": (faulty, [reset, order(2, 2), action("current_time", 5, 3), close(9), action("wait", 5, 0, 9)], 7),
        # no thinking: the time to repair itself
        "E": (faulty, [reset, order(0, 0), close(7), action("wait", 0, 0, 7)], 7),
    }[case]


@pytest.mark.parametrize("case", ["A", "B", "C", "D", "E"])
def test_the_declared_time_takes_out_only_the_thinking_before_the_restoring_order(tmp_path, case):
    """The five hand-checked logs give 4, 7, 3, 7 and 7. Case A is never negative, and B and D, the same order, close and
    thinking before the order, give the same time whatever is thought after the order."""
    hidden, log, expected = _hand_log_case(case)
    verdict = {"hidden_start": hidden, "repair_delay": dict.fromkeys(get_profile(DEFAULT_ASSET).components)}
    assert aggregate.repair_declared(_episode(tmp_path, case, verdict, log, horizon=60)) == expected


def test_the_declared_time_lies_between_the_lead_and_downtime_of_the_repair_and_the_time_to_repair(tmp_path):
    """On random logs of one occurrence of a start fault, played as the clock plays them (each call thinks over the cycles just before
    its stamp and then takes its duration, so no two calls think over one cycle): t_close - t_order <= declared <= t_close - onset,
    and the order of the entries in the log does not move the value."""
    import random

    profile = get_profile(DEFAULT_ASSET)
    healthy = dict(profile.healthy)
    faulty = {**healthy, "cooler": 20}
    rng = random.Random(0)
    for n in range(60):
        cursor, entries, ordering = 0, [], rng.randint(0, 5)
        for k in range(8):
            think = rng.randint(0, 4)
            stamp = cursor + think
            event = "work_order_created" if k == ordering else "read"
            entries.append({"event": event, "id": "wo-1", "component": "cooler", "t_start": stamp, "t_end": stamp, "think_cycles": think}
                           if event == "work_order_created" else {"event": event, "t_start": stamp, "t_end": stamp, "think_cycles": think})
            cursor = stamp + rng.randint(0, 3)
        t_order = entries[ordering]["t_start"]
        t_close = t_order + rng.randint(1, 15)
        entries.append({"event": "work_order_closed", "id": "wo-1", "source": "agent", "component": "cooler", "t_start": t_close,
                        "t_end": t_close, "combination_after": [healthy[c] for c in profile.components]})
        verdict = {"hidden_start": faulty, "repair_delay": dict.fromkeys(profile.components)}
        closes_last = sorted(entries, key=lambda e: e["event"] == "work_order_closed")
        value = aggregate.repair_declared(_episode(tmp_path, f"r{n}", verdict, closes_last, horizon=60))
        assert t_close - t_order <= value <= t_close, (n, value)
        shuffled = [e for e in closes_last if e["event"] != "work_order_closed"]
        rng.shuffle(shuffled)
        again = aggregate.repair_declared(_episode(tmp_path, f"s{n}", verdict, shuffled + closes_last[-1:], horizon=60))
        assert again == value


def test_the_exact_match_baseline_is_the_share_of_the_most_frequent_faulty_set():
    profile = get_profile(DEFAULT_ASSET)
    healthy = [profile.healthy[c] for c in profile.components]
    one = list(healthy)
    one[0] = next(level for level in range(10) if level != healthy[0])
    blocks = [json.dumps(one)] * 3 + [json.dumps(healthy)]
    cells = aggregate.exact_match_floor([{"block": b} for b in blocks], 0.5)
    assert (cells["faulty_set_right_baseline_k"], cells["faulty_set_right_baseline_n"]) == (3, 4)
    assert cells["faulty_set_right_floor"] == 0.75 and cells["faulty_set_right_at_floor"] is True
    assert aggregate.exact_match_floor([{"block": b} for b in blocks], 0.8)["faulty_set_right_at_floor"] is False


def test_the_healthy_control_enters_no_agents_table():
    row = {"mode": "gym", "model": "m", "agent": "causal_m", "config": aggregate.HEALTHY_CONTROL_PRESET, "clock_mode": "static"}
    assert aggregate.agent_episodes([row], every_clock_mode=True) == []
    assert aggregate.agent_episodes([{**row, "config": aggregate.AGENTS_PRESET}], every_clock_mode=True)


def test_the_healthy_start_control_reports_counts_of_false_alarms_and_orders_and_enters_no_family_and_no_row_of_the_agents_table(tmp_path):
    """Three registered episodes of one row on the start with no faulty component: one writes no diagnosis, one names two components
    falsely and places one order, one diagnoses the healthy combination. The row counts what happened with no rate and no test,
    a row that played nothing is left out, and no family reads the preset."""
    row = aggregate.HEALTHY_CONTROL_ROWS[0]
    label = f"{row[0]}_m"

    def episode(name, diagnosed, alarms, orders):
        folder = tmp_path / name
        (folder / "hidden").mkdir(parents=True)
        log = [{"event": "reset"}] + ([{"event": "submit_diagnosis"}] if diagnosed else [])
        (folder / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
        return {"mode": "gym", "model": "m", "surface": "aob", "config": aggregate.HEALTHY_CONTROL_PRESET, "agent": label,
                "diagnostics_server": str(row[1]), "false_alarms": alarms, "unnecessary": orders, "episode_dir": folder,
                "run": "r1", "episodes_planned": 3, "pass": "false"}

    episodes = [episode("a", False, 0, 0), episode("b", True, 2, 1), episode("c", True, 0, 0)]
    rows = aggregate.table_healthy_control(episodes, CTX)
    assert [(r["agent"], r["agent_server"]) for r in rows] == [(label, row[1])]
    [cells] = rows
    assert (cells["n"], cells["planned"], cells["diagnosed"], cells["false_alarm_episodes"], cells["components_falsely_named"],
            cells["work_order_episodes"], cells["work_orders"]) == (3, 3, 2, 1, 2, 1, 1)
    assert aggregate.agent_episodes(episodes, every_clock_mode=True) == []
    assert all(aggregate.HEALTHY_CONTROL_PRESET not in json.dumps(spec) for spec in registered.load()["families"].values())


def test_every_interval_column_has_its_standard_error():
    for table in aggregate.tables():
        for i, column in enumerate(table.columns):
            if column.endswith("_high"):
                assert table.columns[i + 1] == f"{column[:-5]}_se", (table.id, column)
    assert {"H1_TOP_pairs", "FULL_ENV_pairs", "PARALLEL", "PARALLEL_pairs", "HEALTHY_CONTROL", "RESTORED"} <= {t.id for t in aggregate.tables()}


def test_the_pair_of_the_parallel_switch_holds_only_its_columns_and_aggregate_writes_it(tmp_path):
    """Both arms of the pair of ablation Ag9 on a made-up run folder of four starts each: the row of PARALLEL_pairs holds only the
    columns of the table, since its two margins are read on the intervals of the differences and carry no p value, and aggregate
    writes every table of record with the pair in it."""
    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
    from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import BASE_META, _h_row, _run, _start

    meta = {**BASE_META, "config_name": design.PARALLEL_PRESET, "config": {"durations": {"read_per_cycle": 1}, "horizon": 40}}
    on, off = design.row_label(design.PARALLEL_ROW, "m"), f"{design.PARALLEL_ROW[0]}_{design.PARALLEL_OFF_SUFFIX}_m"
    runs = tmp_path / "runs"
    for label in (on, off):
        _run(runs, label, [_h_row(s, label, True, _start(s), config=design.PARALLEL_PRESET) for s in range(4)], meta)
    [row] = aggregate.table_parallel_pairs(aggregate.add_derived_fields(aggregate.load_episodes(runs)), CTX)
    columns = next(table for table in aggregate.tables() if table.id == "PARALLEL_pairs").columns
    assert set(row) <= set(columns) and row["pairs"] == 4
    aggregate.aggregate(runs, tmp_path / "out", resamples=200, pages=False, fraction="1")
    assert (tmp_path / "out" / "PARALLEL_pairs.csv").is_file()


def test_the_generated_tables_end_their_lines_with_a_line_feed_alone(tmp_path):
    out = tmp_path / "out"
    (tmp_path / "runs").mkdir()
    aggregate.aggregate(tmp_path / "runs", out, resamples=10, fraction="1")
    tables = sorted(out.glob("*.csv"))
    assert tables and not any(b"\r\n" in path.read_bytes() for path in tables)


def _levels(profile, faulty):
    """A combination of the profile in component order with the named components at a level other than healthy."""
    return [next(level for level in profile.levels[c] if level != profile.healthy[c]) if c in faulty else profile.healthy[c]
            for c in profile.components]


def _funnel_episode(folder, start, events, due, **cells):
    """An episode folder with its hidden log, its oracle and a trajectory, and the episode of its row: the log opens with the reset
    entry of the start and ends with the end entry that records the combination of cycle 0, and the oracle repairs the components
    of due."""
    (folder / "hidden").mkdir(parents=True)
    log = [{"event": "reset", "step": None, "t_start": 0, "t_end": 0, "hidden": start, "config": {"asset_id": DEFAULT_ASSET}}, *events,
           {"event": "end", "step": None, "t_start": 1, "t_end": 1, "reason": cells.get("end_reason", "finish"),
            "played": [{"t": 0, "cycle": 0, "combination": start}]}]
    (folder / "hidden" / "log.json").write_text(json.dumps(log), encoding="utf-8")
    (folder / "hidden" / "oracle.json").write_text(json.dumps({"work_orders": [{"component": c, "window_end": 9} for c in due]}), encoding="utf-8")
    (folder / "trajectory.json").write_text(json.dumps([{"server": "rig", "tool": "submit_diagnosis"}]), encoding="utf-8")
    return {"mode": "gym", "episode_dir": folder, "block": json.dumps(start), **cells}


def test_the_funnel_counts_every_step_from_the_hidden_log_and_reads_no_record_of_a_runner(tmp_path):
    """Two episodes on a start with one faulty component: the first has a diagnosis accepted that names two components, an order
    on the faulty one and one on a healthy one, a finish, a cost at the cap of the reward and the counters of the harness; a
    failure ends the second before any order. Each step and count is read from the hidden log, the oracle and the row, and
    removing the trajectory of the runner changes no cell."""
    profile = get_profile(DEFAULT_ASSET)
    faulty, healthy = profile.components[0], profile.components[1]
    start = _levels(profile, {faulty})
    named = dict(zip(profile.components, _levels(profile, {faulty, healthy})))
    first = _funnel_episode(tmp_path / "a", start, [
        {"event": "submit_diagnosis", "step": 1, "t_start": 0, "t_end": 0, "diagnosis": named},
        {"event": "work_order_created", "step": 2, "t_start": 0, "t_end": 0, "component": faulty},
        {"event": "work_order_created", "step": 3, "t_start": 0, "t_end": 0, "component": healthy},
        {"event": "finish", "step": 4, "t_start": 0, "t_end": 0, "summary": "done"}], [faulty],
        faulty_set_right="False", diagnosis_all_correct="False", end_reason="finish", cost=str(3 * aggregate.verifier.COST_CAP), cost_oracle="3",
        finish_by="harness", runner_end="own_limit", cut_replies="1", failed_calls="2", replays="1", wall_limit="0",
        diagnosis_refusals=json.dumps({"format": 1}))
    second = _funnel_episode(tmp_path / "b", start, [], [faulty], faulty_set_right="False", diagnosis_all_correct="False",
                             end_reason=aggregate.FAILURE_END, cost="1", cost_oracle="3", finish_by="", runner_end="finish", diagnosis_refusals="")
    row = aggregate.funnel_row([first, second])
    assert {step: row[f"{step}_k"] for step in aggregate.FUNNEL_STEPS} == {
        "diagnosis_accepted": 1, "faulty_set_right": 0, "diagnosis_all_correct": 0, "orders_on_every_faulty": 1, "no_order_on_healthy": 1, "finished": 1}
    healthy_at_start = len(profile.components) - 1
    assert row["orders_on_healthy_per_healthy_component"] == aggregate._r(aggregate.Fraction(1, 2 * healthy_at_start))
    assert (row["failure_before_order"], row["at_cost_cap"], row["refused_diagnoses"]) == (0.5, 0.5, json.dumps({"format": 1}))
    assert (row["finished_by_harness"], row["runner_limit"], row["cut_replies"], row["failed_calls"], row["replays"], row["wall_limit"]) == (1, 1, 1, 2, 1, 0)
    for folder in (tmp_path / "a", tmp_path / "b"):
        (folder / "trajectory.json").unlink()
    assert aggregate.funnel_row([first, second]) == row


def test_the_number_of_components_named_is_counted_beside_the_number_faulty(tmp_path):
    """Three first diagnoses on a start with one faulty component name two, one and no component as faulty, so the means are one and
    one, and the shares that name more, as many and fewer are a third each; the count is taken at the cycle of the write."""
    profile = get_profile(DEFAULT_ASSET)
    faulty, healthy = profile.components[0], profile.components[1]
    start = _levels(profile, {faulty})
    episodes = []
    for name, named in (("more", {faulty, healthy}), ("same", {faulty}), ("fewer", set())):
        diagnosis = dict(zip(profile.components, _levels(profile, named)))
        episodes.append(_funnel_episode(tmp_path / name, start, [{"event": "submit_diagnosis", "step": 1, "t_start": 0, "t_end": 0,
                                                                 "diagnosis": diagnosis}], [faulty], end_reason="finish"))
    row = aggregate.funnel_row(episodes)
    third = aggregate._r(aggregate.Fraction(1, 3))
    assert (row["named_faulty_mean"], row["faulty_at_write_mean"]) == (1.0, 1.0)
    assert (row["named_more"], row["named_as_many"], row["named_fewer"]) == (third, third, third)


def test_every_row_prints_its_replays_and_each_rate_again_with_the_replayed_episodes_failed():
    """Four passed episodes of a row, one played once more and one played twice: the row prints two replayed episodes and three replays,
    and each verified rate read with the replayed episodes as failures drops from one to one half."""
    episodes = [{"mode": "gym", "pass": "True", "diagnosis_all_correct": "True", "faulty_set_right": "True", "success": "True",
                 "restored_clean": "True", "replays": replays} for replays in ("0", "1", "2", "")]
    cells = aggregate.replay_cells(episodes)
    assert (cells["replayed"], cells["replays"]) == (2, 3)
    assert {m: cells[f"{m}_replays_failed"] for m in aggregate.REPLAY_RATES} == {m: 0.5 for m in aggregate.REPLAY_RATES}
    assert aggregate.replay_cells([])["pass_replays_failed"] == ""


def test_no_table_declares_a_column_twice_and_the_graded_rates_are_columns_of_the_agents_table():
    """tables() adds the standard error after every interval, so a declaration that names it as well writes the column twice."""
    tables = {table.id: table for table in aggregate.tables()}
    assert {name: [c for c in t.columns if t.columns.count(c) > 1] for name, t in tables.items() if len(set(t.columns)) != len(t.columns)} == {}
    assert {"graded_top1", "graded_pass", "graded_top1_se", "normalized_score"} <= set(tables["AGENTS"].columns)


def test_the_normalized_score_reads_zero_for_doing_nothing_and_one_for_the_oracle():
    """Doing nothing costs 10 and the oracle 2 on each start: a row at 10 reads 0, at 2 reads 1 and at 6 reads one half, a row whose
    starts the do-nothing policy did not play is not run, and so is a row whose two references are equal."""
    nothing = {("base", str(s), ""): {"cost": "10"} for s in range(2)}

    def row(cost, oracle="2", seeds=(0, 1)):
        return [{"config": "base", "seed": str(s), "cost": str(cost), "cost_oracle": oracle} for s in seeds]
    assert [aggregate.normalized_score(row(c), nothing)["normalized_score"] for c in (10, 2, 6)] == [0.0, 1.0, 0.5]
    assert aggregate.normalized_score(row(6, seeds=(5,)), nothing) == {"normalized_score": "not run", "normalized_score_n": 0}
    assert aggregate.normalized_score(row(6, oracle="10"), nothing)["normalized_score"] == "not run"


def test_the_rows_of_the_top_layer_split_at_the_median_first_drop_with_the_starts_without_one_apart(tmp_path):
    """First drops at cycles 2, 4, 6 and 8 and one start without a drop: the median 5 splits them into two and two, the start
    without a drop is a group of its own, and an action of the agent is no drop."""
    episodes = []
    for i, cycle in enumerate((2, 4, 6, 8, None)):
        events = [{"event": "work_order_created", "t_start": 1}] + ([] if cycle is None else [{"event": aggregate.DETERIORATION_EVENTS[i % 2], "t_start": cycle}])
        (tmp_path / str(i) / "hidden").mkdir(parents=True)
        (tmp_path / str(i) / "hidden" / "log.json").write_text(json.dumps([{"event": "reset", "t_start": 0}, *events]), encoding="utf-8")
        episodes.append({"episode_dir": tmp_path / str(i), "seed": str(i)})
    groups = {group: (cut, [e["seed"] for e in part]) for group, cut, part in aggregate.drop_groups(episodes)}
    assert groups == {"early": (5.0, ["0", "1"]), "late": (5.0, ["2", "3"]), "no_drop": (5.0, ["4"])}


def test_the_system_contrast_equals_the_sum_of_its_two_parts_and_holds_no_p_value(tmp_path):
    """The CausalDiagnosis agent with the MCP passes four starts, Plan-and-Execute with the MCP two and without it one, matched on seed: on every measure
    the total difference, 0.75, is the sum of the design part, 0.5, and the MCP part, 0.25, and no cell is a p value or a Holm cell."""
    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
    from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import BASE_META, _base_row, _run

    runs = tmp_path / "runs"
    ours = next(row for row in design.AGENT_ROWS if row[0] == design.OURS and row[1])
    cells = {(ours[0], True): (True, True, True, True), (design.PE[0], True): (True, True, False, False), (design.PE[0], False): (True, False, False, False)}
    for (agent, server), passes in cells.items():
        _run(runs, f"{agent}_{server}", [_base_row(s, agent=f"{agent}_m", diagnostics_server=server, **{"pass": passes[s], "diagnosis_all_correct": passes[s],
                                                                                                         "faulty_set_right": passes[s]}) for s in range(4)], BASE_META)
    rows = aggregate.table_system_contrast(aggregate.load_episodes(runs), CTX)
    assert [row["design"] for row in rows] == [f"{design.PE[0]}_m"]
    for m in aggregate.CONTRAST_MEASURES:
        assert (rows[0][f"{m}_total"], rows[0][f"{m}_design"], rows[0][f"{m}_mcp"], rows[0][f"{m}_pairs"]) == (0.75, 0.5, 0.25, 4)
    assert not any(key.endswith(("_p", "_holm_p", "_reject")) for key in rows[0])


def test_the_pairs_by_fault_have_one_row_per_component_and_per_faulty_level_of_any_asset(tmp_path):
    """The subsets come from the profile of the asset, one per component healthy at the start and one per component and faulty level,
    on the rig and on the tiny asset of the asset tests. On the rig, H1a prints one row per subset whose difference is the mean over
    the pairs whose start lies in the subset, computed here from the blocks of the rows, and a subset without a start prints none."""
    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import ASSETS_DIR, register_profile, unregister_profile
    from CausalDiagnosis.datasets.dataset_loader.profile_data import AssetProfile
    from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import BASE_META, BLOCK_A, BLOCK_B, _base_row, _run

    def count(profile):
        return len(profile.components) + sum(1 for c in profile.components for level in profile.levels[c] if level != profile.healthy[c])

    rig = get_profile(DEFAULT_ASSET)
    assert len(aggregate.fault_subsets(rig)) == count(rig)
    tiny_file = ASSETS_DIR / "tests" / "tiny_asset.json"
    tiny = register_profile(AssetProfile.from_json(json.loads(tiny_file.read_text(encoding="utf-8")), str(tiny_file)))
    try:
        subsets = aggregate.fault_subsets(tiny)
        assert len(subsets) == count(tiny) and [s[:2] for s in subsets[:len(tiny.components)]] == [("healthy", c) for c in tiny.components]
    finally:
        unregister_profile(tiny.asset_id)
    runs = tmp_path / "runs"
    for agent, passes in ((design.CD[0], (True, True, True, True)), (design.PE[0], (False, True, True, True))):
        _run(runs, agent, [_base_row(s, agent=f"{agent}_m", diagnostics_server=False, **{"pass": passes[s], "diagnosis_all_correct": passes[s]})
                           for s in range(4)], BASE_META)
    rows = [r for r in aggregate.table_pairs_by_fault(aggregate.load_episodes(runs), CTX) if r["hypothesis"] == "H1a"]
    assert len(rows) == count(rig)
    blocks, differences = [json.loads(BLOCK_A)] * 2 + [json.loads(BLOCK_B)] * 2, [1, 0, 0, 0]
    for row in rows:
        i = rig.components.index(row["component"])
        seeds = [s for s in range(4) if blocks[s][i] == row["level"]]
        assert row["pass_pairs"] == len(seeds), (row["subset"], row["component"], row["level"])
        assert row["pass_diff"] == (aggregate._r(sum(differences[s] for s in seeds) / len(seeds)) if seeds else "")


def test_the_policy_layers_pair_each_rung_with_the_rung_below_for_each_policy_and_hold_no_p_value(tmp_path):
    """The oracle on the first two rungs of the ladder, matched on seed: the differences of pass and of the reward between the rung and
    the rung below, with the pairs, and no p value."""
    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
    from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import BASE_META, _base_row, _run

    runs = tmp_path / "runs"
    below, rung = design.LADDER[:2]
    for preset, passes in ((below, (True, True, True, True)), (rung, (True, False, True, False))):
        rows = [_base_row(s, agent="oracle", config=preset, clock_mode=aggregate.config_clock_mode(preset) or "declared", diagnostics_server=False,
                          **{"pass": passes[s], "success": passes[s], "reward": 1.0 if passes[s] else 0.0}) for s in range(4)]
        _run(runs, f"oracle_{preset}", rows, {**BASE_META, "config_name": preset})
    rows = aggregate.table_policy_layers(aggregate.load_episodes(runs), CTX)
    row = next(r for r in rows if r["config"] == rung and r["reference_config"] == below)
    assert (row["pass_diff"], row["pass_pairs"], row["reward_diff"], row["reward_pairs"]) == (-0.5, 4, -0.5, 4)
    assert not any(key.endswith("_p") for key in row)


def test_the_failure_classes_split_the_scored_episodes_and_the_contradicted_recovery_claims_count_claims(tmp_path):
    """One run of the CausalDiagnosis agent with the MCP in a made-up run folder: a success whose summary claims a recovery that holds,
    a summary whose claim of recovery the hidden state contradicts, a premature diagnosis whose summary makes the same claim, which its
    class does not show, and an episode without a verdict. The classes count the three scored episodes and the share of contradicted
    claims counts the three claims."""
    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
    from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import _row, _run, _start

    runs = tmp_path / "runs"
    # (failure class, pass, the summary claims recovery, the asset recovered); the last episode has no verdict
    cases = (("none", True, True, True), ("summary_contradicts_log", False, True, False), ("premature_diagnosis", False, True, False), ("", "", False, False))
    run_dir = _run(runs, "causal_mcp", [_row(s, agent="causal_m", config=design.AGENTS_PRESET, block=_start(s), failure_class=failure, **{"pass": passed})
                                        for s, (failure, passed, _, _) in enumerate(cases)])
    for s, (failure, _, claims, recovered) in enumerate(cases):
        if failure:
            verdict = {"summary_check": {"claims_recovery": claims}, "recovered": recovered}
            (run_dir / f"seed{s:06d}" / "hidden" / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    [row] = aggregate.table_failures(aggregate.load_episodes(runs), CTX)
    assert (row["n"], row["unscored"]) == (4, 1)
    counted = {name: row[f"{name}_k"] for name in aggregate.FAILURE_CLASS_ORDER if row[f"{name}_k"]}
    assert counted == {"none": 1, "summary_contradicts_log": 1, "premature_diagnosis": 1} and row["none"] == 0.25
    assert (row["recovery_claims"], row["recovery_contradicted_n"], row["recovery_contradicted"]) == (3, 3, aggregate._r(2 / 3))


def test_the_parallel_table_counts_the_advised_batches_the_model_kept_and_the_share_of_each_gap_that_the_switch_explains(tmp_path):
    """The pair of the schedule of candidate plans and the two other designs with the MCP in a made-up run folder on the preset of the pair, with a
    horizon of 40 cycles and plans of a count of six steps. No agent restores a fault here, so a start with a faulty component counts the
    horizon as its time to repair and a healthy start counts 0, and the means are 10 with the switch on, 20 with it off, 30 for the first
    other design and 0 for the second. With the switch on, the model keeps the advised batch on the first start and reverses it on the
    second, the rule plans the third, and on the fourth the model writes a plan without a schedule and then one that keeps the seven
    advised reads cut to six, so it kept two of three batches. The switch explains half of the gap of the first design, and the second
    design is ahead of the row with the switch on, which leaves no gap."""
    from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
    from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables
    from CausalDiagnosis.evaluation.result_tables.tests.test_tables_from_runs import BASE_META, _h_row, _run, _start

    profile = get_profile(DEFAULT_ASSET)
    healthy, first = dict(profile.healthy), profile.components[0]
    faulty = {**healthy, first: next(level for level in profile.levels[first] if level != healthy[first])}
    meta = {**BASE_META, "config_name": design.PARALLEL_PRESET, "config": {"durations": {"read_per_cycle": 1}, "horizon": 40},
            "causal_options": {"plan_length": 6}}
    on, off = design.row_label(design.PARALLEL_ROW, "m"), f"{design.PARALLEL_ROW[0]}_{design.PARALLEL_OFF_SUFFIX}_m"
    first_design, second_design = (design.row_label(row, "m") for row in design.PARALLEL_CONTRAST_ROWS)
    runs, folders = tmp_path / "runs", {}
    for label, faults in ((on, (0, 0, 0, 1)), (off, (0, 0, 1, 1)), (first_design, (0, 1, 1, 1)), (second_design, (0, 0, 0, 0))):
        folders[label] = _run(runs, label, [_h_row(s, label, True, _start(s), config=design.PARALLEL_PRESET) for s in range(4)], meta)
        for s, fault in enumerate(faults):
            verdict = {"hidden_start": faulty if fault else healthy, "repair_delay": dict.fromkeys(profile.components)}
            (folders[label] / f"seed{s:06d}" / "hidden" / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    batch, seven = [("g0", 2), ("g1", 3)], [(f"g{i}", i + 1) for i in range(7)]
    # per start of the row with the switch on: (source, the advised reads of the schedule or None, the reads that begin the plan)
    plans = {0: [("model", batch, batch)], 1: [("model_retry", batch, batch[::-1])], 2: [("rule_only", batch, batch)],
             3: [("model", None, batch), ("model_repaired", seven, seven[:6])]}
    for s, records in plans.items():
        written = [{"source": source, "schedule": None if advised is None else {"reads": [{"sensor_group": g, "cycles": c} for g, c in advised]},
                    "plan": {"steps": [*({"kind": "read", "args": {"sensor_group": g, "cycles": c}} for g, c in steps), {"kind": "submit_diagnosis", "args": {}}]}}
                   for source, advised, steps in records]
        (folders[on] / f"seed{s:06d}" / "plans.json").write_text(json.dumps({"plans": written}), encoding="utf-8")
    rows = {(row["agent"], row["switch"]): row for row in aggregate.table_parallel(aggregate.add_derived_fields(aggregate.load_episodes(runs)), CTX)}
    assert list(rows) == [(on, "on"), (off, "off"), (first_design, ""), (second_design, "")]
    assert [rows[key]["repair_declared"] for key in rows] == [10, 20, 30, 0]
    assert [(r["advised_batches"], r["kept_batches"], r["changed_batches"]) for r in rows.values()] == [(3, 2, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    assert rows[(on, "on")]["kept_share"] == aggregate._r(2 / 3) and rows[(off, "off")]["kept_share"] == ""
    assert [r["gap_explained"] for r in rows.values()] == ["", "", 0.5, ""]
    out = tmp_path / "results" / "aggregate"
    aggregate.aggregate(runs, out, resamples=200, pages=False, fraction="1")
    table = main_tables.parallel_table(main_tables.Sheet(main_tables.read_tables(out), "m"))
    assert [(names[0], [cell.full for cell in cells[-2:]]) for group in table.groups for names, cells in group] == [
        (main_tables.SWITCH_NAMES["on"], ["2 of 3", ""]), (main_tables.SWITCH_NAMES["off"], [main_tables.NO_VALUE, ""]),
        (design.PARALLEL_CONTRAST_ROWS[0][2], [main_tables.NO_VALUE, "0.50"]), (design.PARALLEL_CONTRAST_ROWS[1][2], [main_tables.NO_VALUE, main_tables.NO_VALUE])]


def test_holm_over_parts_of_different_measures_equals_holm_over_their_union():
    """holm_joint corrects the parts of a group of primary as one family: the adjusted p values and the decisions equal those of one
    Holm over the union of the tests, and every row counts the registered size of the group."""
    from CausalDiagnosis.evaluation.metrics.table_statistics import holm
    first = [{"pass_p": 0.01, "diagnosis_all_correct_p": 0.2}, {"pass_p": 0.04, "diagnosis_all_correct_p": 0.5}]
    second = [{"pass_p": 0.03, "reward_p": 0.001}]
    aggregate.holm_joint([(first, ("diagnosis_all_correct", "pass")), (second, ("pass", "reward"))], 0.05, 8)
    union = [0.2, 0.01, 0.5, 0.04, 0.03, 0.001, 1.0, 1.0]
    expected = holm(union, 0.05)
    found = [first[0]["diagnosis_all_correct_holm_p"], first[0]["pass_holm_p"], first[1]["diagnosis_all_correct_holm_p"],
             first[1]["pass_holm_p"], second[0]["pass_holm_p"], second[0]["reward_holm_p"]]
    assert found == [round(value, 6) for value in expected["adjusted"][:6]]
    assert [row["family_tests"] for row in first + second] == [8, 8, 8]


def test_two_runs_that_share_the_run_cell_count_their_planned_episodes_apart(tmp_path):
    """Two replicate runs started in the same second share the run cell and lie in two folders, each planning three episodes: five
    played of six planned is incomplete."""
    row = [{"mode": "gym", "pass": "true", "run": "r", "run_dir": tmp_path / ("r" if i < 3 else "r-2"), "episodes_planned": 3} for i in range(5)]
    assert aggregate.row_status(row, CTX) == "incomplete, 5 of 6"


def test_an_aliased_fold_takes_the_status_of_the_row_as_released():
    """The copies of a fold that trained no adapter count by the run without training they copy: incomplete when that run is."""
    released = [{"mode": "gym", "pass": "true", "run": "r", "episodes_planned": 3, "training_variant": "as_released"} for _ in range(2)]
    alias = [{**e, "training_variant": "ft", "run": "marker", "aliased": True} for e in released]
    trained = [{"mode": "gym", "pass": "true", "run": "t", "episodes_planned": 2, "training_variant": "ft"} for _ in range(2)]
    assert aggregate.training_status(alias, released + alias, CTX) == "incomplete, 2 of 3"
    assert aggregate.training_status(trained, released + trained, CTX) == ""


def test_the_resamples_and_the_seed_of_the_tables_default_to_the_registered_file(tmp_path, monkeypatch):
    load = registered.load
    monkeypatch.setattr(registered, "load", lambda *a, **k: {**load(*a, **k), "intervals": {**load(*a, **k)["intervals"], "resamples": 7, "seed": 3}})
    aggregate.aggregate(tmp_path / "no_runs", tmp_path / "out", pages=False, fraction="1")
    parameters = json.loads((tmp_path / "INDEX.json").read_text(encoding="utf-8"))["parameters"]
    assert (parameters["resamples"], parameters["seed"]) == (7, 3)


def test_the_pass_with_the_read_gate_counts_an_unscored_episode_as_a_failure_as_pass_does():
    """scope_pass, the Pass of A7, of RUNS and of the leaderboard, is a pass, so the registered counting.unscored binds it: an unscored
    played episode reads 0 and stays in the share, as on pass."""
    scored = {"mode": "gym", "pass": "true", "scope_pass": True, "block": "[1]"}
    unscored = {"mode": "gym", "pass": "", "scope_pass": None, "block": "[2]"}
    assert aggregate.value(unscored, "scope_pass") == 0.0 and aggregate.value({**unscored, "mode": "static"}, "scope_pass") is None
    cells = aggregate.summary([scored, unscored], "scope_pass", CTX)
    assert (cells["scope_pass"], cells["scope_pass_n"]) == (0.5, 2) == (aggregate.summary([scored, unscored], "pass", CTX)["pass"], 2)


def test_a_fault_restored_without_a_reading_is_timed_from_its_drop_in_the_hidden_log(tmp_path):
    """The registered onset is the cycle at which the hidden log takes the component off its healthy level: a drop at 5 and the
    agent's restoring close at 12 read 7, where the verdict's repair_delay, which waits for a reading, is empty."""
    profile = get_profile(DEFAULT_ASSET)
    healthy = dict(profile.healthy)
    first = profile.components[0]
    faulty = next(level for level in profile.levels[first] if level != healthy[first])
    log = [{"event": "hazard_drop", "component": first, "to_level": faulty, "t_start": 5, "t_end": 5},
           {"event": "work_order_created", "id": "wo-1", "component": first, "t_start": 8, "t_end": 8},
           {"event": "work_order_closed", "id": "wo-1", "source": "agent", "component": first, "t_start": 10, "t_end": 12,
            "combination_after": [healthy[c] for c in profile.components]}]
    verdict = {"hidden_start": healthy, "repair_delay": dict.fromkeys(profile.components)}
    assert aggregate.repair_declared(_episode(tmp_path, "dropped", verdict, log)) == 12 - 5


def test_every_share_graded_outcome_and_the_reward_has_one_counting_rule():
    """OUTCOME_COUNTING is the one home of the counting rule: every share of the tables, every graded outcome and the reward stand
    in it once, the judge outcomes keep their own rule, and an unscored gym episode reads 0 on a failure or zero entry and None on a
    left out one."""
    rules = aggregate.OUTCOME_COUNTING
    assert set(rules.values()) <= {"failure", "zero", "left out", "derived"}
    assert (aggregate.PROPORTIONS - aggregate.JUDGE_OUTCOMES) <= set(rules) and {"graded_pass", "graded_top1", "reward"} <= set(rules)
    unscored = {"mode": "gym", "pass": "", "error": "verifier: RuntimeError"}
    for metric, rule in rules.items():
        if metric in aggregate.STATIC_OUTCOMES or rule == "derived":
            continue
        assert aggregate.value(unscored, metric) == (0.0 if rule in ("failure", "zero") else None), metric
    scored = {"mode": "gym", "pass": "true", "reward": ""}  # a scored episode whose oracle cost is unknown keeps an empty reward
    assert aggregate.value(scored, "reward") is None
    assert aggregate.value({"mode": "static", "static_faulty_set_correct": ""}, "static_faulty_set_correct") == 0.0
